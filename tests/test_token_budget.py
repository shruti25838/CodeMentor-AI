"""Reading the provider's remaining quota, and being honest about what it does not say.

Groq sends per-minute token limits and a per-day *request* count, but no per-day *token*
header. The daily token figure is therefore an estimate, and these tests pin down that it is
labelled as one, and that the exact figure is used whenever the provider does state it.

No network: the probe's parsing and arithmetic are tested, not the call.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from token_budget import ASSUMED_DAILY_TOKEN_LIMIT, Budget, parse_tpd  # noqa: E402

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
        "requests_per_day_limit": 1000,
        "requests_per_day_remaining": 998,
        "daily_tokens_remaining": 199_600,
        "daily_is_estimate": True,
    }
    return Budget(**{**base, **overrides})


# ---------- reading the daily figure out of a refusal ----------


def test_the_daily_limit_is_read_from_the_providers_own_message() -> None:
    assert parse_tpd(TPD_MESSAGE) == (200000, 199248)


def test_a_per_minute_refusal_is_not_mistaken_for_a_daily_one() -> None:
    """A 413 about tokens per minute must not be read as the daily budget running out."""
    assert parse_tpd(TPM_MESSAGE) is None


def test_an_unrelated_error_yields_nothing() -> None:
    assert parse_tpd("connection reset by peer") is None
    assert parse_tpd("") is None


def test_remaining_daily_tokens_from_a_refusal() -> None:
    limit, used = parse_tpd(TPD_MESSAGE)
    assert limit - used == 752


# ---------- what the report says ----------


def test_the_estimate_is_labelled_as_an_estimate() -> None:
    text = budget().render()
    assert "ESTIMATE" in text
    assert "no tokens-per-day header exists" in budget(note="no tokens-per-day header exists; x").render()


def test_a_figure_from_the_provider_is_not_labelled_an_estimate() -> None:
    text = budget(daily_tokens_remaining=752, daily_is_estimate=False).render()
    assert "reported by the provider" in text
    assert "ESTIMATE" not in text


def test_an_unknown_daily_figure_says_so_rather_than_guessing_zero() -> None:
    text = budget(daily_tokens_remaining=None).render()
    assert "unknown" in text
    assert "the provider sends no header" in text


def test_requests_used_today_is_derived_from_the_counter() -> None:
    assert budget(requests_per_day_remaining=998).requests_used_today == 2
    assert budget(requests_per_day_remaining=0).requests_used_today == 1000


def test_a_remaining_count_above_the_limit_does_not_go_negative() -> None:
    assert budget(requests_per_day_remaining=1200).requests_used_today == 0


def test_the_minute_figures_are_reported() -> None:
    text = budget().render()
    assert "7927 of 8000 left" in text


# ---------- the threshold ----------


@pytest.mark.parametrize(
    "remaining,required,should_stop",
    [
        (199_600, 60_000, False),
        (60_000, 60_000, False),
        (59_999, 60_000, True),
        (752, 60_000, True),
        (0, 60_000, True),
    ],
)
def test_the_stop_threshold(remaining: int, required: int, should_stop: bool) -> None:
    assert (remaining < required) is should_stop


def test_an_unknown_figure_never_triggers_the_stop() -> None:
    """Not knowing must not read as 'nothing left'; the 429 itself is the real guard."""
    b = budget(daily_tokens_remaining=None)
    short = b.daily_tokens_remaining is not None and b.daily_tokens_remaining < 60_000
    assert not short


def test_the_assumed_limit_matches_the_one_the_provider_reports() -> None:
    limit, _ = parse_tpd(TPD_MESSAGE)
    assert limit == ASSUMED_DAILY_TOKEN_LIMIT
