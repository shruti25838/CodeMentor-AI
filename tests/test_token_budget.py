"""Reading the provider's quota headers, and refusing to invent the one it does not send.

This file used to assert that the daily token figure was estimated from the daily request
counter and labelled an estimate. That estimate has been removed: the counter it was built
on refills every few minutes, so it read 996 of 1000 while 199,400 of 200,000 daily tokens
had been spent, and the probe reported about 199,200 left just before the provider refused
a call. What remains true is tested here; the local count that replaced it is in
`tests/test_token_ledger.py`.

No network: the parsing and the rendering are tested, not the call.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from token_budget import DAILY_UNKNOWN, Budget, parse_tpd  # noqa: E402

TPD_MESSAGE = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`openai/gpt-oss-20b` in organization `org_x` service tier `on_demand` on tokens per day "
    "(TPD): Limit 200000, Used 199248, Requested 1503. Please try again in 5m24.432s.'}}"
)
TPM_MESSAGE = (
    "Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-20b` "
    "on tokens per minute (TPM): Limit 8000, Requested 10391'}}"
)


def budget(**overrides) -> Budget:
    base = {
        "model": "openai/gpt-oss-20b",
        "tokens_per_minute_limit": 8000,
        "tokens_per_minute_remaining": 7927,
        "tokens_reset": "547ms",
        "requests_limit": 1000,
        "requests_remaining": 998,
        "requests_reset": "2m52.8s",
        "spent_today": 0,
        "calls_today": 0,
        "ledger": ".codeatlas/token-spend.json",
    }
    return Budget(**{**base, **overrides})


# ---------- reading the daily figure out of a refusal ----------


def test_the_daily_limit_is_read_from_the_providers_own_message() -> None:
    assert parse_tpd(TPD_MESSAGE) == (200000, 199248)


def test_a_per_minute_refusal_is_not_mistaken_for_a_daily_one() -> None:
    assert parse_tpd(TPM_MESSAGE) is None


def test_an_unrelated_error_yields_nothing() -> None:
    assert parse_tpd("connection reset by peer") is None
    assert parse_tpd("") is None


def test_remaining_daily_tokens_from_a_refusal() -> None:
    limit, used = parse_tpd(TPD_MESSAGE)
    assert limit - used == 752


# ---------- the daily figure is never invented ----------


def test_the_daily_line_says_unknown_when_the_provider_has_not_said() -> None:
    text = budget().render()
    assert DAILY_UNKNOWN in text
    assert "unknown unless the provider states it" in text


def test_a_full_request_counter_does_not_become_a_daily_token_figure() -> None:
    """The exact mistake that was removed: 998 of 1000 requests meant nothing about tokens."""
    text = budget(requests_remaining=998).render()
    assert DAILY_UNKNOWN in text
    assert "199," not in text, "no inferred daily figure anywhere in the output"


def test_an_almost_empty_request_counter_also_says_unknown() -> None:
    text = budget(requests_remaining=2).render()
    assert DAILY_UNKNOWN in text


def test_the_request_counter_is_labelled_as_not_daily() -> None:
    assert "not a daily counter" in budget().render()
    assert "2m52.8s" in budget().render(), "its reset window is shown, which is why"


def test_a_figure_the_provider_stated_is_shown_as_such() -> None:
    text = budget(daily_tokens_remaining=600).render()
    assert "600 left, as stated by the provider" in text
    assert DAILY_UNKNOWN not in text


# ---------- what it does report ----------


def test_the_minute_figures_are_reported_with_their_reset() -> None:
    text = budget().render()
    assert "7,927 of 8,000 left" in text
    assert "resets in 547ms" in text


def test_the_local_spend_is_reported_with_the_file_it_came_from() -> None:
    text = budget(spent_today=45_000, calls_today=37).render()
    assert "45,000 tokens over 37 call(s)" in text
    assert ".codeatlas/token-spend.json" in text


def test_a_day_with_nothing_recorded_warns_rather_than_implying_a_fresh_budget() -> None:
    b = budget()
    b.warnings.append("nothing recorded today on this machine; the account may still have been used elsewhere")
    assert "WARNING" in b.render()
    assert "used elsewhere" in b.render()


@pytest.mark.parametrize("remaining", [0, 600, 199_000])
def test_any_provider_stated_figure_is_printed_verbatim(remaining: int) -> None:
    assert f"{remaining:,} left, as stated by the provider" in budget(daily_tokens_remaining=remaining).render()
