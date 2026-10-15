"""The `retry_wait` timing is read out of another library's log line. This is the canary.

When the provider answers 429, the OpenAI-compatible client retries by itself, sleeping
*inside* the model call. From outside there is nothing to see: the call simply took longer.
`codeatlas/observability/timing.py` recovers that wait by attaching a handler to the `openai`
logger and reading the number of seconds out of the line the client logs just before it
sleeps -- `"Retrying request to %s in %f seconds"`.

**That line is not an API.** It is a log message inside a dependency, and it can be reworded,
moved to another logger or dropped in any patch release. If that happens, `retry_wait` quietly
becomes zero on every request for ever, and the only symptom is model calls that look slow for
no reason -- which is precisely the question `retry_wait` was added to answer. Nothing in the
existing tests would notice, because they log the line themselves and then check the handler
read it: they assert the handler matches a string this repository wrote down, not the string
the installed client actually emits.

So this file drives **the real client**. A local HTTP server answers the first request with a
429 and a short `Retry-After` and the second with an ordinary completion; the chat model is the
one `LlmProvider` builds for the app, pointed at that server through the environment variable
the client itself reads. The retry is real, the sleep is real, and the line being matched is
the line the installed version logged. If the wording changes or the logger moves,
`test_a_real_429_is_recorded_as_a_retry_wait_on_the_waiting_request` and
`test_the_client_still_logs_the_line_the_watcher_reads` fail, which is the entire point.

**No key and no external network.** The key is a fixed fake string, the server is on 127.0.0.1
on a port the OS picks, and the proxy variables are cleared so nothing can be routed elsewhere.
"""

import json
import logging
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from codeatlas.observability.timing import _RETRY_SECONDS as WATCHER_PATTERN
from codeatlas.observability.timing import RETRY_WAIT, StageTimer, run_timed, watch_provider_retries
from codeatlas.services.llm.provider import LlmProvider
from codeatlas.utils.config import load_config

# What the watcher in timing.py is coupled to, written down in one place so a failure can say
# which half moved. Both belong to the `openai` package and neither is promised by it.
WATCHED_LOGGER = "openai"
MATCHED_PHRASE = "Retrying request"

# Seconds the server asks to be waited, as a plain `Retry-After`. The client honours any value
# between 0 and 60 exactly rather than backing off on its own schedule, so the number that ends
# up in the log line -- and therefore in the timing -- is this one and not a jittered guess.
RETRY_AFTER_SECONDS = 1
EXPECTED_RETRY_WAIT_MS = RETRY_AFTER_SECONDS * 1000.0

# Plainly not a credential. The local server never looks at it; it exists because the client
# refuses to be constructed without one.
FAKE_KEY = "canary-not-a-real-key"

COMPLETION = {
    "id": "chatcmpl-canary",
    "object": "chat.completion",
    "created": 0,
    "model": "canary-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "the answer"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}

# A 429 body shaped the way the providers shape theirs. Nothing reads it; the status and the
# header are what matter.
TOO_MANY = {"error": {"message": "rate limit reached", "type": "rate_limit_exceeded"}}


class OneRateLimitThenAnAnswer(BaseHTTPRequestHandler):
    """429 once, with a short Retry-After, then a valid completion. Counts what it was sent."""

    requests: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - the name BaseHTTPRequestHandler dispatches to
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).requests.append(self.path)
        if len(type(self).requests) == 1:
            self._send(429, TOO_MANY, {"Retry-After": str(RETRY_AFTER_SECONDS)})
        else:
            self._send(200, COMPLETION, {})

    def _send(self, status: int, body: dict, headers: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        """Silence: this server's own access log is noise in the test output."""


class Records(logging.Handler):
    """Keeps the records the client emitted, so the real line can be asserted on and reused."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def matching(self, phrase: str) -> list[logging.LogRecord]:
        return [r for r in self.records if phrase in r.getMessage()]


@pytest.fixture
def local_provider(monkeypatch):
    """A real chat model, built the way the app builds one, pointed at a local server.

    `OPENAI_API_BASE` is how the installed `langchain-openai` is told where the API is, and
    `LlmProvider` passes no `base_url` of its own, so this is the shipped code path with the
    endpoint moved -- not a client assembled here with different arguments.
    """
    OneRateLimitThenAnAnswer.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), OneRateLimitThenAnAnswer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]

    # Nothing may be routed anywhere else, whatever this machine has configured.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENAI_API_BASE", f"http://{host}:{port}/v1")

    try:
        yield LlmProvider(replace(load_config(), llm_provider="openai", llm_model="canary-model"))
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def watched():
    """Watch the provider's retries, and keep every record the provider logger saw."""
    watch_provider_retries()
    captured = Records()
    logger = logging.getLogger(WATCHED_LOGGER)
    logger.addHandler(captured)
    try:
        yield captured
    finally:
        logger.removeHandler(captured)


@pytest.fixture
def timing_lines(caplog):
    caplog.set_level(logging.INFO, logger="codeatlas.observability.timing")
    return caplog


def _one_call(provider, timer: StageTimer) -> str:
    """One model call through the app's own client, timed as the app times it."""

    def call() -> str:
        with timer.stage("llm"):
            return provider.get_chat_model().invoke("does this retry?").content

    return run_timed(timer, call)


# ---------- the canary ----------


@pytest.mark.builds_model_client
def test_a_real_429_is_recorded_as_a_retry_wait_on_the_waiting_request(local_provider, watched):
    """The whole coupling, end to end, with nothing about the log line written down by hand."""
    timer = StageTimer()

    answer = _one_call(local_provider, timer)

    assert answer == "the answer"
    # Two requests over a real socket: the one that was refused and the one that was answered.
    assert OneRateLimitThenAnAnswer.requests == ["/v1/chat/completions", "/v1/chat/completions"]

    timings = timer.as_dict()
    assert timings[RETRY_WAIT] == EXPECTED_RETRY_WAIT_MS, (
        "the client's retry wait did not reach the timer. Either it no longer logs its sleep, "
        f"or it no longer logs it under {WATCHED_LOGGER!r} with the phrase {MATCHED_PHRASE!r} "
        "-- see watch_provider_retries in codeatlas/observability/timing.py"
    )
    # The sleep happened inside the call, so the model stage already contains it. That is why
    # it is recorded separately: from the `llm` total alone the two are indistinguishable.
    assert timings["llm"] >= timings[RETRY_WAIT]


@pytest.mark.builds_model_client
def test_the_client_still_logs_the_line_the_watcher_reads(local_provider, watched):
    """Asserts on the record the installed client produced, not on one written here.

    A reword fails the phrase assertion, a dropped number fails the pattern assertion, and a
    logger moved out from under `openai` fails the name assertion -- each with its own message.
    """
    _one_call(local_provider, StageTimer())

    lines = watched.matching(MATCHED_PHRASE)
    assert len(lines) == 1, (
        f"the installed openai client logged {len(lines)} line(s) containing {MATCHED_PHRASE!r} "
        "for one retry; the watcher counts one wait per such line"
    )
    record = lines[0]
    assert record.name == WATCHED_LOGGER or record.name.startswith(f"{WATCHED_LOGGER}."), (
        f"the retry is logged by {record.name!r}, which is not under the {WATCHED_LOGGER!r} "
        "logger the watcher attaches to, so the handler would never see it"
    )
    assert record.levelno >= logging.INFO, "the watcher raises the logger to INFO and no lower"
    found = WATCHER_PATTERN.search(record.getMessage())
    assert found is not None, "the seconds can no longer be read out of the line"
    assert float(found.group(1)) == float(RETRY_AFTER_SECONDS)


@pytest.mark.builds_model_client
def test_the_wait_appears_in_the_timing_line_and_nothing_else_does(local_provider, watched, timing_lines):
    """`retry_wait` has to survive into the line that is actually logged, not just the dict."""
    timer = StageTimer()
    _one_call(local_provider, timer)

    timer.log("ask")

    line = next(r.getMessage() for r in timing_lines.records if r.getMessage().startswith("timing ask "))
    assert f"{RETRY_WAIT}={EXPECTED_RETRY_WAIT_MS}" in line
    # The client's line names the URL it is retrying; only the number may cross over. Every
    # field of a timing line is `name=number`, so asserting the shape rules out any text at
    # all reaching it -- without this test ever holding a value to print.
    fields = line[len("timing ask ") :].split(" ")
    assert fields, line
    for field in fields:
        name, _, value = field.partition("=")
        assert name and value, field
        for part in value.split("|"):
            float(part)


# ---------- what a change upstream would look like ----------
#
# The three tests above fail on their own if the client's line moves. These two say what the
# failure would mean, by taking the line the client really logged and breaking it the ways a
# dependency can break it. Each starts from `record.getMessage()`, so none of them can drift
# away from the real wording the way an invented string would.


@pytest.mark.builds_model_client
def test_a_reworded_line_is_matched_by_nothing_and_the_wait_is_lost(local_provider, watched):
    real = _real_retry_line(local_provider, watched)

    rewordings = {
        "a different verb": real.replace(MATCHED_PHRASE, "Retry scheduled for request"),
        "one extra word": real.replace(MATCHED_PHRASE, "Retrying the request"),
        "a shorter unit": real.replace(" seconds", "s"),
        "no number at all": WATCHER_PATTERN.sub("shortly", real),
    }
    for description, message in rewordings.items():
        timer = StageTimer()
        run_timed(timer, lambda m=message: logging.getLogger(f"{WATCHED_LOGGER}._base_client").info("%s", m))
        assert RETRY_WAIT not in timer.as_dict(), f"{description}: {RETRY_WAIT} was recorded anyway"


@pytest.mark.builds_model_client
def test_the_same_line_under_a_moved_logger_is_matched_by_nothing(local_provider, watched):
    """Same text, different logger: the handler is attached to a name, not to the message."""
    real = _real_retry_line(local_provider, watched)

    for moved in ("openai_client._base_client", "httpx", "langchain_openai.chat_models.base"):
        timer = StageTimer()
        run_timed(timer, lambda name=moved: logging.getLogger(name).info("%s", real))
        assert RETRY_WAIT not in timer.as_dict(), f"{moved} reached the watcher"

    # ...and under the watched one it still is, so the loop above proves the logger and not
    # something incidental about running the line through a second time.
    timer = StageTimer()
    run_timed(timer, lambda: logging.getLogger(f"{WATCHED_LOGGER}._base_client").info("%s", real))
    assert timer.as_dict()[RETRY_WAIT] == EXPECTED_RETRY_WAIT_MS


def _real_retry_line(provider, watched: Records) -> str:
    """The retry line the installed client logged, for a 429 it really received."""
    _one_call(provider, StageTimer())
    lines = watched.matching(MATCHED_PHRASE)
    assert len(lines) == 1, "no retry line to start from; the canary above explains why"
    return lines[0].getMessage()


def test_the_watchers_pattern_is_the_one_this_file_asserts_against():
    """Pins the import. `_RETRY_SECONDS` is private, and renaming it must not skip these tests."""
    assert isinstance(WATCHER_PATTERN, re.Pattern)
    assert WATCHER_PATTERN.search("Retrying request to /v1/chat/completions in 1.000000 seconds")
