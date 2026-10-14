"""The parts of scripts/chat_latency_probe.py that decide things, tested without a server.

The probe is the one script here that spends real money, so the two guards that stop it -- the
token cap and the per-minute pacer -- and the refusal detection that ends a run are worth more
than the HTTP plumbing around them.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.services.llm.quota import DAILY, MESSAGES, PER_MINUTE  # noqa: E402
from scripts.chat_latency_probe import (  # noqa: E402
    HEADER,
    QUESTIONS,
    Pacer,
    Row,
    budget_guard,
    format_row,
    mode_of,
    refusal_of,
)


def _ledger(path: Path, model: str, total: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "day": __import__("datetime").date.today().isoformat(),
                "models": {model: {"input": total, "output": 0, "calls": 1}},
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------- the token cap ----------


def test_the_cap_is_measured_against_this_sessions_spend_not_todays(tmp_path):
    """An earlier run's spend today must neither count against this cap nor hide it."""
    ledger = _ledger(tmp_path / "ledger.json", "openai/gpt-oss-20b", 30_000)
    baseline = 30_000

    may_continue, spent, left = budget_guard(40_000, baseline, ledger)
    assert (may_continue, spent, left) == (True, 0, 40_000)

    _ledger(ledger, "openai/gpt-oss-20b", 55_000)
    may_continue, spent, left = budget_guard(40_000, baseline, ledger)
    assert (may_continue, spent, left) == (True, 25_000, 15_000)


def test_the_guard_stops_the_run_once_the_cap_is_reached(tmp_path):
    ledger = _ledger(tmp_path / "ledger.json", "m", 40_000)
    may_continue, spent, left = budget_guard(40_000, 0, ledger)
    assert may_continue is False and spent == 40_000 and left == 0
    # Over, not merely at, the cap: still stopped, and `left` never goes negative.
    _ledger(ledger, "m", 41_000)
    assert budget_guard(40_000, 0, ledger) == (False, 41_000, 0)


def test_the_guard_reserves_the_next_questions_cost_so_the_cap_holds_on_the_way_out(tmp_path):
    """Without the reserve the last question overruns the cap by its own size, which is how the
    first run of this script spent 43,958 under a cap of 40,000."""
    ledger = _ledger(tmp_path / "ledger.json", "m", 38_000)

    # Room by the old rule - 38,000 is under 40,000 - but the next question costs about 5,000.
    assert budget_guard(40_000, 0, ledger)[0] is True
    assert budget_guard(40_000, 0, ledger, reserve=5_000)[0] is False
    # A question that fits exactly is allowed; the cap is a ceiling, not a strict bound.
    assert budget_guard(40_000, 0, ledger, reserve=2_000)[0] is True
    assert budget_guard(40_000, 0, ledger, reserve=2_001)[0] is False


def test_a_missing_ledger_reads_as_nothing_spent(tmp_path):
    assert budget_guard(40_000, 0, tmp_path / "absent.json") == (True, 0, 40_000)


# ---------- the pacer ----------


def test_the_pacer_waits_only_when_the_window_is_full():
    pacer = Pacer(limit=8_000)
    assert pacer.wait() == 0.0  # nothing spent yet, nothing to wait for

    pacer.record(3_000)
    assert pacer.in_window() == 3_000
    # 3,000 spent and the worst question so far was 3,000: 6,000 fits under 8,000.
    assert pacer.estimate() == 3_000
    assert pacer.wait() == 0.0


def test_the_pacer_estimates_the_next_question_from_the_worst_one_so_far():
    pacer = Pacer(limit=8_000)
    pacer.record(2_000)
    pacer.record(5_000)
    assert pacer.estimate() == 5_000


def test_the_pacer_forgets_spend_older_than_the_window():
    pacer = Pacer(limit=8_000, window_seconds=60.0)
    pacer.record(7_000)
    # Age that entry out by hand rather than by sleeping for a minute.
    at, tokens = pacer.spends[0]
    pacer.spends[0] = (at - 61.0, tokens)
    assert pacer.in_window() == 0
    assert pacer.wait() == 0.0


def test_a_limit_of_zero_turns_the_pacer_off():
    pacer = Pacer(limit=0)
    pacer.record(1_000_000)
    assert pacer.wait() == 0.0


# ---------- recognising a refusal ----------


def test_a_quota_refusal_is_recognised_from_the_sentence_the_visitor_is_shown():
    """The app turns a 429 into prose, so there is no error event to watch for."""
    assert refusal_of({"type": "done"}, MESSAGES[PER_MINUTE] + " more text") == PER_MINUTE
    assert refusal_of({"type": "done"}, MESSAGES[DAILY]) == DAILY


def test_an_ordinary_answer_is_not_mistaken_for_a_refusal():
    assert refusal_of({"type": "done"}, "Signer.sign appends the signature to the payload.") == ""


def test_a_stream_error_and_a_model_error_are_both_reported():
    assert refusal_of({"type": "error", "content": "boom"}, "") == "stream error"
    assert refusal_of({"type": "done"}, "Error generating answer: boom") == "model error"


# ---------- the mode comes from the answer, not the command line ----------


def test_the_mode_is_read_off_the_agents_that_ran():
    assert mode_of({"agents_used": ["grounded"]}) == "grounded"
    assert mode_of({"agents_used": ["retrieval", "mentor"]}) == "legacy"
    assert mode_of({"agents_used": ["planner", "retrieval", "mentor"]}) == "legacy"
    assert mode_of({}) == "?"


# ---------- the table ----------


def _row(**overrides) -> Row:
    fields = dict(
        index=8,
        mode="legacy",
        history_turns=4,
        rewrite_ran=1,
        llm_calls=2,
        in_tokens=12_345,
        out_tokens=2_048,
        llm_ms=98_765.4,
        llm_each=[12_345.6, 86_419.8],
        retry_s=123.4,
        first_token_ms=87_654.3,
        total_ms=99_999.9,
        wall_ms=99_999.9,
        spent=14_393,
    )
    fields.update(overrides)
    return Row(**fields)


def test_no_row_is_wider_than_the_report_it_goes_into():
    assert len(HEADER) <= 90
    assert len(format_row(_row())) <= 90


def test_a_field_the_server_did_not_report_reads_as_a_dash_rather_than_a_zero():
    line = format_row(_row(history_turns=None, rewrite_ran=None, first_token_ms=None))
    assert " - " in line
    assert len(line) <= 90


def test_one_model_call_gets_no_per_call_list():
    """With one call the stage total is that call, so repeating it would be noise."""
    line = format_row(_row(llm_calls=1, llm_each=[1234.5]))
    assert line.endswith("123.4")  # the retry column is the last thing on the line
    assert "/" not in line
    assert "/" in format_row(_row())  # two calls, so both durations are shown


def test_the_questions_are_a_fixed_list_with_follow_ups_in_it():
    from codeatlas.services.memory.conversation import Turn
    from codeatlas.services.qa.followup import needs_rewrite

    history = [Turn("earlier", "answer")]
    rewritten = [q for q in QUESTIONS if needs_rewrite(q, history)]
    # Some questions must lean on the conversation and some must not, or the run cannot show
    # what the rewrite call costs.
    assert 0 < len(rewritten) < len(QUESTIONS)
