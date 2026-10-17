"""Per-request stage timing.

A request handler starts a StageTimer; services deeper in the call stack add to it through
``stage(name)`` without the timer being passed down. A stage that runs more than once in a
request (the language model is called up to three times per chat question) adds up, and its
call count is kept. Outside a request ``stage`` does nothing.

Besides durations a timer carries two other kinds of number, both added so that a chat
question's latency can be attributed rather than guessed at:

* **counts** -- a number the request produced once per model call, such as the input tokens
  the provider reported. Each value is kept as well as the total, so two calls in one request
  can be told apart.
* **facts** -- one number about the whole request, such as how many earlier turns went into
  the prompt, or whether the follow-up rewrite call ran at all.

Only stage names, counts and durations are recorded, never questions, answers, prompts or
keys. Every number here is either a duration, a token count the provider reported, or a count
of turns -- there is no path by which text reaches a log line or a response body from here.
"""

import contextvars
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_current: contextvars.ContextVar["StageTimer | None"] = contextvars.ContextVar("stage_timer", default=None)

# The stage a provider-side retry backoff is recorded as. It is *inside* the model call, so the
# `llm` total already contains it; it is recorded separately because a model call that was slow
# because the provider made it wait is a different problem from one that was slow because the
# model thought for a long time, and from outside the two look identical.
RETRY_WAIT = "retry_wait"


class StageTimer:
    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self._start = clock()
        self._ms: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._marks: dict[str, float] = {}
        # Per-occurrence values, for stages and counts alike: `llm_each` is how long each model
        # call took, `in_tokens_each` how large each one's prompt was.
        self._each: dict[str, list[float]] = {}
        self._counts: dict[str, float] = {}
        self._facts: dict[str, float] = {}

    def add(self, name: str, ms: float) -> None:
        self._ms[name] = self._ms.get(name, 0.0) + ms
        self._calls[name] = self._calls.get(name, 0) + 1
        self._each.setdefault(name, []).append(round(ms, 1))

    def count(self, name: str, value: float) -> None:
        """Add one occurrence of a number: the total is kept, and so is this occurrence."""
        self._counts[name] = self._counts.get(name, 0.0) + value
        self._each.setdefault(name, []).append(round(value, 1))

    def fact(self, name: str, value: float) -> None:
        """One number describing the whole request; written once, last writer wins."""
        self._facts[name] = value

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

    def durations(self) -> dict[str, float]:
        """Stage totals and marks in ms, plus ``total``: everything here is a length of time."""
        out = {name: round(ms, 1) for name, ms in self._ms.items()}
        out.update({name: round(ms, 1) for name, ms in self._marks.items()})
        out["total"] = round(self.elapsed_ms(), 1)
        return out

    def as_dict(self) -> dict:
        """Everything the timer holds: durations, counts, facts, call counts and per-call lists.

        Not all floats any more, and deliberately so. ``<name>_calls`` and ``<name>_each`` appear
        only where something happened more than once, which keeps a one-model-call request's line
        as short as it was. A reader tells the kinds apart by name: ``total`` and the stage names
        are ms, ``in_tokens`` are tokens, ``history_turns`` is a count of turns.
        """
        out: dict = {name: round(ms, 1) for name, ms in self._ms.items()}
        out.update({name: round(ms, 1) for name, ms in self._marks.items()})
        out.update({name: round(value, 1) for name, value in self._counts.items()})
        out.update(self._facts)
        out.update({f"{name}_calls": n for name, n in self._calls.items() if n > 1})
        out.update({f"{name}_each": values for name, values in self._each.items() if len(values) > 1})
        out["total"] = round(self.elapsed_ms(), 1)
        return out

    def server_timing(self) -> str:
        """Value for the Server-Timing response header (shown in the browser's network panel).

        Durations only. The header's `dur` is milliseconds by definition, so a token count or a
        number of turns put in it would be read by the browser as a time.
        """
        return ", ".join(f"{name};dur={ms}" for name, ms in self.durations().items())

    def log(self, what: str) -> None:
        """Log the request's timings and record them in the Prometheus stage histogram."""
        from codeatlas.observability.metrics import observe_stages

        logger.info("timing %s %s", what, " ".join(f"{k}={_render(v)}" for k, v in self.as_dict().items()))
        observe_stages(what, dict(self._ms), self.elapsed_ms())


def _render(value) -> str:
    """One field of the log line. A per-call list is joined with `|` so the line stays splittable."""
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value)


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


def record_count(name: str, value: float) -> None:
    """Add one occurrence of a number to the current request's timer, if there is one."""
    timer = _current.get()
    if timer is not None:
        timer.count(name, value)


def record_fact(name: str, value: float) -> None:
    """Record one number about the current request, if there is a timer for it."""
    timer = _current.get()
    if timer is not None:
        timer.fact(name, value)


def run_timed(timer: StageTimer, fn, *args):
    """Run fn with timer as the current one; for work handed to a thread pool, which has its own context."""
    token = _current.set(timer)
    try:
        return fn(*args)
    finally:
        _current.reset(token)


# The provider client's own line, logged just before it sleeps. Only the number is read out of it.
_RETRY_SECONDS = re.compile(r"\bin ([0-9]+(?:\.[0-9]+)?) seconds\b")


class _RetryWaitHandler(logging.Handler):
    """Turns the provider client's "retrying in N seconds" line into a stage on this request.

    The OpenAI-compatible client retries a 429 itself, sleeping inside the call, so from outside
    the wait is invisible: it only makes the model call look slow. The client announces each
    sleep at INFO before taking it, and this reads the number of seconds out of that line and
    nothing else -- not the URL logged beside it, and not the request being retried.

    Attached to the `openai` logger, so it sees records from every module under it. It never
    stops the record reaching the normal handlers.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if "Retrying request" not in message:
                return
            match = _RETRY_SECONDS.search(message)
            timer = _current.get()
            if match is not None and timer is not None:
                timer.add(RETRY_WAIT, float(match.group(1)) * 1000)
        except Exception:  # pragma: no cover - a logging handler must never raise
            pass


def watch_provider_retries() -> None:
    """Count the provider's own retry backoffs into whichever request is waiting through them.

    Idempotent: `configure_logging` runs once per `create_app`, and a test may build several.
    """
    provider_logger = logging.getLogger("openai")
    if any(isinstance(handler, _RetryWaitHandler) for handler in provider_logger.handlers):
        return
    # The record has to exist before a handler can see it, and the line is logged at INFO.
    provider_logger.setLevel(logging.INFO)
    provider_logger.addHandler(_RetryWaitHandler())
