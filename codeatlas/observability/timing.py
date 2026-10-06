"""Per-request stage timing.

A request handler starts a StageTimer; services deeper in the call stack add to it through
``stage(name)`` without the timer being passed down. A stage that runs more than once in a
request (the language model is called up to three times per chat question) adds up, and its
call count is kept. Outside a request ``stage`` does nothing.

Only stage names and durations are recorded, never questions, answers or code.
"""

import contextvars
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar["StageTimer | None"] = contextvars.ContextVar("stage_timer", default=None)


class StageTimer:
    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self._start = clock()
        self._ms: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._marks: dict[str, float] = {}

    def add(self, name: str, ms: float) -> None:
        self._ms[name] = self._ms.get(name, 0.0) + ms
        self._calls[name] = self._calls.get(name, 0) + 1

    def mark(self, name: str) -> None:
        """Record the time since the request started, once (later calls keep the first value)."""
        self._marks.setdefault(name, (self._clock() - self._start) * 1000)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            self.add(name, (self._clock() - start) * 1000)

    def elapsed_ms(self) -> float:
        return (self._clock() - self._start) * 1000

    def as_dict(self) -> dict[str, float]:
        """Stage totals and marks in ms, plus ``total``; repeated stages also get ``<name>_calls``."""
        out = {name: round(ms, 1) for name, ms in self._ms.items()}
        out.update({name: round(ms, 1) for name, ms in self._marks.items()})
        out.update({f"{name}_calls": n for name, n in self._calls.items() if n > 1})
        out["total"] = round(self.elapsed_ms(), 1)
        return out

    def server_timing(self) -> str:
        """Value for the Server-Timing response header (shown in the browser's network panel)."""
        return ", ".join(f"{name};dur={ms}" for name, ms in self.as_dict().items() if not name.endswith("_calls"))

    def log(self, what: str) -> None:
        """Log the request's timings and record them in the Prometheus stage histogram."""
        from codeatlas.observability.metrics import observe_stages

        logger.info("timing %s %s", what, " ".join(f"{k}={v}" for k, v in self.as_dict().items()))
        observe_stages(what, dict(self._ms), self.elapsed_ms())


@contextmanager
def timed_request() -> Iterator[StageTimer]:
    """Make a new StageTimer the current one for the code inside the block."""
    timer = StageTimer()
    token = _current.set(timer)
    try:
        yield timer
    finally:
        _current.reset(token)


def current_timer() -> StageTimer | None:
    return _current.get()


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a block into the current request's timer, if there is one."""
    timer = _current.get()
    if timer is None:
        yield
        return
    with timer.stage(name):
        yield


def run_timed(timer: StageTimer, fn, *args):
    """Run fn with timer as the current one; for work handed to a thread pool, which has its own context."""
    token = _current.set(timer)
    try:
        return fn(*args)
    finally:
        _current.reset(token)
