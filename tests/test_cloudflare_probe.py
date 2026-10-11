"""The probe's safety properties, checked without the network and without credentials.

What matters about this script is not that it works but that it cannot misbehave: a capped
number of calls, tiny inputs, and output that never contains a token, an account id or any
input text.
"""

import json
import math

import pytest

from codeatlas.utils.config import (
    CLOUDFLARE_ACCOUNT_ID_ENV,
    CLOUDFLARE_API_TOKEN_ENV,
    CloudflareCredentials,
    load_cloudflare_credentials,
)
from scripts import cloudflare_probe as probe


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    """Keep the developer's own .env out of these tests.

    load_cloudflare_credentials() reads .env, which on this machine holds the real credentials.
    Without this, a test that deletes a variable would watch load_dotenv put the real one back.
    """
    monkeypatch.setattr("codeatlas.utils.config.load_dotenv", lambda *a, **k: False)


def _no_calls(monkeypatch) -> None:
    """Any attempt to send anything is a test failure, not a network call."""

    def explode(*args, **kwargs):
        raise AssertionError("the probe made a call it should not have made")

    monkeypatch.setattr(probe, "post_json", explode)


def test_the_plan_stays_within_the_call_cap() -> None:
    assert len(probe.plan()) <= probe.MAX_CALLS


def test_main_refuses_to_run_a_plan_over_the_cap(monkeypatch, capsys) -> None:
    _no_calls(monkeypatch)
    oversized = probe.plan() * 2
    assert len(oversized) > probe.MAX_CALLS
    monkeypatch.setattr(probe, "plan", lambda: oversized)
    monkeypatch.setattr("sys.argv", ["cloudflare_probe.py"])
    assert probe.main() == 1
    assert "refusing to run" in capsys.readouterr().out


def test_a_dry_run_sends_nothing_and_needs_no_credentials(monkeypatch, capsys) -> None:
    _no_calls(monkeypatch)
    monkeypatch.delenv(CLOUDFLARE_ACCOUNT_ID_ENV, raising=False)
    monkeypatch.delenv(CLOUDFLARE_API_TOKEN_ENV, raising=False)
    monkeypatch.setattr("sys.argv", ["cloudflare_probe.py", "--dry-run"])
    assert probe.main() == 0
    assert "dry run: nothing was sent" in capsys.readouterr().out


def test_the_printed_plan_names_no_input_text(monkeypatch, capsys) -> None:
    _no_calls(monkeypatch)
    monkeypatch.setattr("sys.argv", ["cloudflare_probe.py", "--dry-run"])
    probe.main()
    out = capsys.readouterr().out
    assert "probe item" not in out
    assert "The signer serializes" not in out
    assert "Fold the dough" not in out


def test_every_probe_input_is_small() -> None:
    """Tiny inputs are the whole reason this is safe to run against a free allowance."""
    for call in probe.plan():
        assert sum(len(text) for text in call.texts) <= probe.LONG_CHARS


def test_the_whole_run_is_a_rounding_error_against_the_daily_allowance() -> None:
    """Every input counted, including the one that is meant to be refused and so costs nothing."""
    tokens = sum(probe.estimated_tokens(call.texts) for call in probe.plan())
    assert probe.estimated_neurons(tokens) < probe.DAILY_FREE_NEURONS / 100


def test_the_token_estimate_is_three_characters_per_token_capped_at_the_window() -> None:
    assert probe.estimated_tokens(["abcdef"]) == 2
    assert probe.estimated_tokens(["abcd"]) == math.ceil(4 / 3)
    assert probe.estimated_tokens(["x" * 100_000]) == probe.MAX_TOKENS_PER_DOCUMENT
    assert probe.estimated_tokens(["abc", "abc"]) == 2


def test_the_long_text_is_two_subjects_and_the_head_is_its_opening() -> None:
    """The truncation test only means something if the tail really is different text."""
    long_text, head = probe.long_text(), probe.head_text()
    assert len(long_text) == probe.LONG_CHARS
    assert len(head) == probe.HEAD_CHARS
    assert long_text.startswith(head)
    assert probe.tail_sample() not in head


def test_exactly_one_call_is_the_deliberately_invalid_one() -> None:
    assert [call.label for call in probe.plan() if call.invalid_token] == ["invalid-token"]


def test_the_invalid_call_is_left_out_of_the_cost_estimate(monkeypatch, capsys) -> None:
    _no_calls(monkeypatch)
    monkeypatch.setattr("sys.argv", ["cloudflare_probe.py", "--dry-run"])
    probe.main()
    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("estimated cost:")]
    billed = sum(probe.estimated_tokens(c.texts) for c in probe.plan() if not c.invalid_token)
    assert printed == [
        f"estimated cost: {billed} input tokens, {probe.estimated_neurons(billed):.2f} estimated neurons"
    ]


def test_an_outcome_reports_sizes_and_statuses_but_never_the_text() -> None:
    call = probe.plan()[1]
    outcome = probe.Outcome(label=call.label, texts=len(call.texts), chars=sum(len(t) for t in call.texts))
    outcome.status, outcome.success, outcome.shape, outcome.pooling = 200, True, [50, 384], "mean"
    line = outcome.line()
    assert "50 texts" in line and "shape=[50, 384]" in line and "pooling=mean" in line
    assert "probe item" not in line


def test_a_transport_failure_is_reported_as_a_type_name(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise probe.JsonClientError("TimeoutError")

    monkeypatch.setattr(probe, "post_json", explode)
    outcome = probe.run_call(probe.plan()[0], "acct-123", "secret-token", timeout=1)
    assert outcome.transport_error == "TimeoutError"
    assert "acct-123" not in outcome.line() and "secret-token" not in outcome.line()


def test_the_errors_kept_from_a_refusal_are_codes_and_messages_only() -> None:
    outcome = probe.Outcome(label="x", texts=1, chars=1)
    body = {
        "success": False,
        "errors": [{"code": 10000, "message": "Authentication error", "trace": "ignored"}],
        "result": None,
    }
    probe._read(outcome, probe.JsonResponse(status=401, body=body))
    assert outcome.errors == [{"code": 10000, "message": "Authentication error"}]
    assert "trace" not in json.dumps(outcome.errors)


def test_the_truncation_verdict_follows_the_two_cosines() -> None:
    def outcome(label, vector):
        return probe.Outcome(label=label, texts=1, chars=1, status=200, success=True, vectors=[vector])

    head, tail = [1.0, 0.0], [0.0, 1.0]
    dropped = probe.report(
        {
            "long-8000": outcome("long-8000", [0.99, 0.14]),
            "head-2400": outcome("head-2400", head),
            "tail-2400": outcome("tail-2400", tail),
        }
    )
    assert any("the tail was dropped" in line for line in dropped)

    read_whole = probe.report(
        {
            "long-8000": outcome("long-8000", [0.3, 0.95]),
            "head-2400": outcome("head-2400", head),
            "tail-2400": outcome("tail-2400", tail),
        }
    )
    assert any("the whole text was read" in line for line in read_whole)


def test_credentials_are_none_unless_both_are_set(monkeypatch) -> None:
    monkeypatch.setenv(CLOUDFLARE_ACCOUNT_ID_ENV, "acct-123")
    monkeypatch.delenv(CLOUDFLARE_API_TOKEN_ENV, raising=False)
    assert load_cloudflare_credentials() is None
    monkeypatch.setenv(CLOUDFLARE_API_TOKEN_ENV, "   ")
    assert load_cloudflare_credentials() is None
    monkeypatch.setenv(CLOUDFLARE_API_TOKEN_ENV, "secret-token")
    assert load_cloudflare_credentials() == CloudflareCredentials("acct-123", "secret-token")


def test_printing_the_credentials_prints_neither_of_them() -> None:
    """They travel through scripts and tracebacks; the repr is the one place this is cheap to guarantee."""
    credentials = CloudflareCredentials(account_id="acct-123", api_token="secret-token")
    for rendered in (repr(credentials), str(credentials), f"{credentials}"):
        assert "acct-123" not in rendered
        assert "secret-token" not in rendered
    assert repr(credentials) == "CloudflareCredentials(account_id=<set>, api_token=<redacted>)"


@pytest.mark.parametrize("missing", [CLOUDFLARE_ACCOUNT_ID_ENV, CLOUDFLARE_API_TOKEN_ENV])
def test_main_stops_before_calling_when_a_credential_is_missing(monkeypatch, capsys, missing) -> None:
    _no_calls(monkeypatch)
    monkeypatch.setenv(CLOUDFLARE_ACCOUNT_ID_ENV, "acct-123")
    monkeypatch.setenv(CLOUDFLARE_API_TOKEN_ENV, "secret-token")
    monkeypatch.delenv(missing, raising=False)
    monkeypatch.setattr(probe, "load_cloudflare_credentials", lambda: None)
    monkeypatch.setattr("sys.argv", ["cloudflare_probe.py"])
    assert probe.main() == 1
    assert "must both be set" in capsys.readouterr().out
