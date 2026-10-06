"""The on-disk count of tokens spent today, and the probe that no longer guesses.

The provider sends no tokens-per-day header. An earlier version of the probe inferred one
from `x-ratelimit-remaining-requests`, which refills every few minutes: it read 996 of 1000
and reported about 199,200 daily tokens left, minutes before the provider refused a call
saying 199,400 of 200,000 were already spent. The count is kept locally now, and the probe
says "unknown" rather than inventing a figure.

A fake provider stands in for every call; nothing here reaches the network.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from codeatlas.observability import token_ledger
from codeatlas.observability.agent_metrics import AgentUsageCallback, agent_run
from codeatlas.observability.token_ledger import record, spent_today, within_budget

MODEL = "openai/gpt-oss-20b"


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path / ".codeatlas" / "token-spend.json"
    monkeypatch.setenv(token_ledger.ENV_PATH, str(path))
    return path


class FakeProvider(BaseChatModel):
    """Reports usage the way a real provider does, without being one."""

    state: Any = None

    def __init__(self, input_tokens: int = 100, output_tokens: int = 25, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("S", (), {"inp": input_tokens, "out": output_tokens})()

    @property
    def _llm_type(self) -> str:
        return "fake-provider"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(
            content="answer",
            usage_metadata={
                "input_tokens": self.state.inp,
                "output_tokens": self.state.out,
                "total_tokens": self.state.inp + self.state.out,
            },
            response_metadata={"model_name": MODEL},
        )
        return ChatResult(generations=[ChatGeneration(message=message)], llm_output={"model_name": MODEL})


# ---------- counting ----------


def test_nothing_spent_before_anything_runs(ledger) -> None:
    spend = spent_today(MODEL)
    assert spend.total == 0
    assert spend.calls == 0


def test_one_call_is_counted(ledger) -> None:
    record(MODEL, 100, 25)
    spend = spent_today(MODEL)

    assert spend.input_tokens == 100
    assert spend.output_tokens == 25
    assert spend.total == 125
    assert spend.calls == 1


def test_calls_add_up(ledger) -> None:
    for _ in range(4):
        record(MODEL, 1000, 500)
    assert spent_today(MODEL).total == 6000
    assert spent_today(MODEL).calls == 4


def test_models_are_counted_separately(ledger) -> None:
    record(MODEL, 100, 50)
    record("other/model", 7000, 1000)

    assert spent_today(MODEL).total == 150
    assert spent_today("other/model").total == 8000
    assert spent_today().total == 8150, "no model means every model"


def test_a_call_with_no_usage_is_not_counted(ledger) -> None:
    record(MODEL, 0, 0)
    assert spent_today(MODEL).calls == 0


def test_the_total_survives_a_new_process(ledger) -> None:
    """A long run spans restarts; a counter held in memory would forget."""
    record(MODEL, 5000, 1000)
    reloaded = json.loads(ledger.read_text(encoding="utf-8"))

    assert reloaded["models"][MODEL]["input"] == 5000
    assert spent_today(MODEL).total == 6000


def test_yesterdays_total_does_not_count_against_today(ledger) -> None:
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps({"day": yesterday, "models": {MODEL: {"input": 190_000, "output": 9_000, "calls": 90}}}),
        encoding="utf-8",
    )
    assert spent_today(MODEL).total == 0


def test_a_new_day_starts_the_count_again(ledger) -> None:
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"day": yesterday, "models": {MODEL: {"input": 5, "output": 5, "calls": 1}}}), "utf-8")

    record(MODEL, 100, 20)

    assert spent_today(MODEL).total == 120
    assert json.loads(ledger.read_text(encoding="utf-8"))["day"] == date.today().isoformat()


# ---------- it must never break a request ----------


def test_an_unreadable_file_reads_as_nothing_spent(ledger) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{ not json", encoding="utf-8")
    assert spent_today(MODEL).total == 0


def test_a_half_written_file_is_replaced_rather_than_crashing(ledger) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"day": "2020', encoding="utf-8")
    record(MODEL, 10, 5)
    assert spent_today(MODEL).total == 15


def test_a_path_that_cannot_be_written_does_not_raise(monkeypatch, tmp_path) -> None:
    # A directory where the file should be: writing fails, the call still goes through.
    blocked = tmp_path / "blocked.json"
    blocked.mkdir()
    monkeypatch.setenv(token_ledger.ENV_PATH, str(blocked))
    record(MODEL, 100, 50)  # must not raise


# ---------- the limit a long run stops on ----------


@pytest.mark.parametrize(
    "spent,limit,allowed,left",
    [
        (0, 150_000, True, 150_000),
        (100_000, 150_000, True, 50_000),
        (149_999, 150_000, True, 1),
        (150_000, 150_000, False, 0),
        (160_000, 150_000, False, 0),
    ],
)
def test_the_limit_decides_when_to_stop(ledger, spent: int, limit: int, allowed: bool, left: int) -> None:
    if spent:
        record(MODEL, spent, 0)
    assert within_budget(limit, MODEL) == (allowed, left)


def test_a_limit_of_zero_never_stops(ledger) -> None:
    record(MODEL, 999_999, 0)
    assert within_budget(0, MODEL) == (True, 0)


def test_the_default_limit_leaves_room_below_a_typical_daily_allowance() -> None:
    from token_budget import DEFAULT_MAX_DAILY_TOKENS

    assert DEFAULT_MAX_DAILY_TOKENS == 150_000
    assert DEFAULT_MAX_DAILY_TOKENS < 200_000, "room left over for live testing"


# ---------- every model call is counted ----------


def test_a_model_call_adds_to_the_ledger(ledger) -> None:
    model = FakeProvider(input_tokens=321, output_tokens=123).with_config({"callbacks": [AgentUsageCallback()]})
    with agent_run("mentor"):
        model.invoke("hello")

    spend = spent_today(MODEL)
    assert spend.total == 444
    assert spend.calls == 1


def test_several_calls_accumulate_through_the_callback(ledger) -> None:
    model = FakeProvider(input_tokens=1000, output_tokens=200).with_config({"callbacks": [AgentUsageCallback()]})
    for _ in range(3):
        model.invoke("hello")

    assert spent_today(MODEL).total == 3600
    assert spent_today(MODEL).calls == 3


def test_the_model_name_is_taken_from_the_providers_reply(ledger) -> None:
    model = FakeProvider().with_config({"callbacks": [AgentUsageCallback()]})
    model.invoke("hello")

    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert MODEL in data["models"], "counted against the model that was actually called"


# ---------- the probe no longer invents a daily figure ----------


def test_the_probe_never_infers_a_daily_figure_from_request_counts() -> None:
    text = (ROOT / "scripts" / "token_budget.py").read_text(encoding="utf-8")

    assert "ASSUMED_DAILY_TOKEN_LIMIT" not in text, "the assumed limit is gone"
    assert "x-ratelimit-remaining-requests" in text, "the request counter is still reported"
    # It may be reported, but never multiplied into a daily token estimate.
    assert "share_used" not in text


def test_the_unknown_wording_is_what_the_probe_prints() -> None:
    from token_budget import DAILY_UNKNOWN, Budget

    budget = Budget(
        model=MODEL,
        tokens_per_minute_limit=8000,
        tokens_per_minute_remaining=7927,
        tokens_reset="547ms",
        requests_limit=1000,
        requests_remaining=998,
        requests_reset="2m52.8s",
        spent_today=1234,
        calls_today=5,
        ledger=".codeatlas/token-spend.json",
    )
    text = budget.render()

    assert DAILY_UNKNOWN in text
    assert "unknown unless the provider states it" in text
    assert "not a daily counter" in text, "the request counter is labelled for what it is"
    assert "1,234 tokens over 5 call(s)" in text


def test_a_figure_the_provider_stated_is_shown_as_such() -> None:
    from token_budget import DAILY_UNKNOWN, Budget

    budget = Budget(
        model=MODEL,
        tokens_per_minute_limit=0,
        tokens_per_minute_remaining=0,
        tokens_reset="",
        requests_limit=0,
        requests_remaining=0,
        requests_reset="",
        daily_tokens_remaining=600,
        spent_today=199_400,
        calls_today=103,
        ledger=".codeatlas/token-spend.json",
    )
    text = budget.render()

    assert "600 left, as stated by the provider" in text
    assert DAILY_UNKNOWN not in text


def test_the_daily_limit_is_still_read_from_a_refusal() -> None:
    from token_budget import parse_tpd

    message = "on tokens per day (TPD): Limit 200000, Used 199400, Requested 1503."
    assert parse_tpd(message) == (200000, 199400)
