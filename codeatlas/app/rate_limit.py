"""In-process request limits for the endpoints that clone repositories or call the LLM.

State lives in this process only: it resets on restart and isn't shared between workers,
so with N uvicorn workers the effective limits are N times higher.
"""

import ipaddress
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator

from fastapi import Depends, HTTPException, Request

from codeatlas.utils.config import AppConfig

logger = logging.getLogger(__name__)

GLOBAL_KEY = "*"


def client_address(request: Request, trusted_proxy_hops: int) -> str:
    """The address to rate-limit by.

    With 0 hops, the direct connection address. With N hops, the entry N from the right of
    X-Forwarded-For: each trusted proxy appends the address it received the request from,
    so entries further left are whatever the caller chose to send and can't be trusted.
    Falls back to the direct address if the header is shorter than expected or not an IP.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return peer
    # Several X-Forwarded-For headers are equivalent to one comma-joined header.
    raw = ",".join(request.headers.getlist("x-forwarded-for"))
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    if len(entries) < trusted_proxy_hops:
        return peer
    candidate = entries[-trusted_proxy_hops]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return peer


class SlidingWindowLimiter:
    """At most `limit` events per `window_seconds` per key; tracks at most `max_keys` keys.

    Keys idle for a whole window are dropped. When the cap is reached, the least recently
    seen key is evicted (that client's count restarts, which the global limit backstops).
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self.clock = clock
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._last_sweep = clock()

    def __len__(self) -> int:
        return len(self._events)

    def retry_after(self, key: str, now: float) -> float:
        """Seconds until `key` may make another request; 0 if allowed now. Doesn't record."""
        if self.limit <= 0:
            return self.window_seconds
        events = self._events.get(key)
        if events is None:
            return 0.0
        self._trim(events, now)
        if len(events) < self.limit:
            return 0.0
        return max(events[0] + self.window_seconds - now, 0.0)

    def record(self, key: str, now: float) -> None:
        self._sweep(now)
        events = self._events.get(key)
        if events is None:
            events = self._events[key] = deque()
        self._events.move_to_end(key)
        events.append(now)
        while len(self._events) > self.max_keys:
            self._events.popitem(last=False)

    def _trim(self, events: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()

    def _sweep(self, now: float) -> None:
        # At most once per window, drop keys with no events inside the window.
        if now - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = now
        cutoff = now - self.window_seconds
        for key in [k for k, ev in self._events.items() if not ev or ev[-1] <= cutoff]:
            del self._events[key]


class RateLimitExceeded(HTTPException):
    def __init__(self, message: str, retry_after: float) -> None:
        seconds = max(1, math.ceil(retry_after))
        super().__init__(
            status_code=429,
            detail=f"{message} Please try again in {seconds} second{'s' if seconds != 1 else ''}.",
            headers={"Retry-After": str(seconds)},
        )


class RequestLimit:
    """A per-client limit plus a global limit over the same window, checked together."""

    def __init__(
        self,
        name: str,
        per_client: int,
        global_limit: int,
        window_seconds: float,
        max_clients: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.clock = clock
        self.per_client = SlidingWindowLimiter(per_client, window_seconds, max_clients, clock)
        self.global_ = SlidingWindowLimiter(global_limit, window_seconds, 1, clock)
        self._lock = threading.Lock()

    def check(self, client: str) -> None:
        """Count one request from `client`, or raise RateLimitExceeded without counting it."""
        with self._lock:
            now = self.clock()
            wait = self.per_client.retry_after(client, now)
            if wait > 0:
                raise RateLimitExceeded(f"You've sent too many {self.name} requests.", wait)
            wait = self.global_.retry_after(GLOBAL_KEY, now)
            if wait > 0:
                raise RateLimitExceeded(f"The server is handling too many {self.name} requests right now.", wait)
            self.per_client.record(client, now)
            self.global_.record(GLOBAL_KEY, now)


class ConcurrencyLimit:
    """At most `limit` operations at once, across all clients. Never waits: full means 429."""

    def __init__(self, limit: int, retry_after_seconds: float = 15) -> None:
        self._semaphore = threading.BoundedSemaphore(limit)
        self.retry_after_seconds = retry_after_seconds

    def acquire(self) -> None:
        if not self._semaphore.acquire(blocking=False):
            raise RateLimitExceeded(
                "The server is busy indexing other repositories.",
                self.retry_after_seconds,
            )

    def release(self) -> None:
        self._semaphore.release()


class RateLimits:
    def __init__(self, config: AppConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.trusted_proxy_hops = config.trusted_proxy_hops
        self.clone = RequestLimit(
            "repository indexing",
            per_client=config.clone_per_client_per_minute,
            global_limit=config.clone_global_per_minute,
            window_seconds=60,
            max_clients=config.rate_limit_max_clients,
            clock=clock,
        )
        self.clone_concurrency = ConcurrencyLimit(config.max_concurrent_clones)
        self.llm = RequestLimit(
            "chat",
            per_client=config.llm_per_client_per_minute,
            global_limit=config.llm_global_per_minute,
            window_seconds=60,
            max_clients=config.rate_limit_max_clients,
            clock=clock,
        )

    def log_mode(self) -> None:
        if self.trusted_proxy_hops <= 0:
            logger.info("Rate limiting by direct connection address (CODEATLAS_TRUSTED_PROXY_HOPS=0)")
        else:
            logger.info(
                "Rate limiting by X-Forwarded-For entry %d from the right (CODEATLAS_TRUSTED_PROXY_HOPS=%d)",
                self.trusted_proxy_hops,
                self.trusted_proxy_hops,
            )

    def client(self, request: Request) -> str:
        return client_address(request, self.trusted_proxy_hops)


def get_rate_limits(request: Request) -> RateLimits:
    return request.app.state.rate_limits


def limit_llm(request: Request, limits: RateLimits = Depends(get_rate_limits)) -> None:
    limits.llm.check(limits.client(request))


def limit_clone(request: Request, limits: RateLimits = Depends(get_rate_limits)) -> Iterator[None]:
    # Take a slot first so a "server busy" rejection doesn't use up the client's allowance.
    limits.clone_concurrency.acquire()
    try:
        limits.clone.check(limits.client(request))
        yield
    finally:
        limits.clone_concurrency.release()
