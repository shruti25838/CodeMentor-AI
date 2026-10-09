"""The estimated-neuron ledger: the UTC day, the estimate, and what it refuses to hold."""

import json
from datetime import UTC, datetime

from codeatlas.services.retrieval import neuron_ledger


def test_the_estimate_is_three_characters_per_token_capped_at_the_window() -> None:
    assert neuron_ledger.estimated_tokens(["abcdef"]) == 2
    assert neuron_ledger.estimated_tokens(["abcde"]) == 2
    assert neuron_ledger.estimated_tokens(["x" * 100_000]) == neuron_ledger.MAX_TOKENS_PER_DOCUMENT
    assert neuron_ledger.estimated_tokens(["abc", "abc"]) == 2
    assert neuron_ledger.estimated_tokens([]) == 0


def test_neurons_are_tokens_at_the_published_price() -> None:
    tokens = neuron_ledger.estimated_tokens(["x" * 3000])
    assert neuron_ledger.estimated_neurons(["x" * 3000]) == tokens * 1841 / 1_000_000


def test_the_day_is_the_utc_day_because_that_is_when_the_allowance_resets() -> None:
    assert neuron_ledger.today() == datetime.now(UTC).date().isoformat()


def test_recording_adds_to_todays_total(tmp_path) -> None:
    path = tmp_path / "neurons.json"
    neuron_ledger.record(1.5, documents=50, path=path)
    neuron_ledger.record(2.25, documents=10, path=path)
    spend = neuron_ledger.spent_today(path)
    assert spend.estimated_neurons == 3.75
    assert spend.documents == 60
    assert spend.calls == 2
    assert spend.day == neuron_ledger.today()


def test_the_ledger_holds_counts_and_nothing_about_what_was_embedded(tmp_path) -> None:
    path = tmp_path / "neurons.json"
    neuron_ledger.record(1.0, documents=3, path=path)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert set(written) == {"days"}
    assert set(written["days"][neuron_ledger.today()]) == {"estimated_neurons", "documents", "calls"}


def test_yesterdays_spend_does_not_count_against_today(tmp_path) -> None:
    path = tmp_path / "neurons.json"
    path.write_text(json.dumps({"days": {"2000-01-01": {"estimated_neurons": 9999, "documents": 1, "calls": 1}}}))
    assert neuron_ledger.spent_today(path).estimated_neurons == 0.0
    assert neuron_ledger.remaining(8000, path) == 8000


def test_a_missing_or_broken_file_reads_as_nothing_spent(tmp_path) -> None:
    assert neuron_ledger.spent_today(tmp_path / "absent.json").estimated_neurons == 0.0
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert neuron_ledger.spent_today(broken).estimated_neurons == 0.0


def test_affordable_is_the_total_after_the_call_not_before(tmp_path) -> None:
    path = tmp_path / "neurons.json"
    neuron_ledger.record(7999.0, documents=1, path=path)
    assert neuron_ledger.affordable(1.0, 8000, path)
    assert not neuron_ledger.affordable(1.5, 8000, path)
    assert neuron_ledger.remaining(8000, path) == 1.0


def test_a_budget_of_zero_affords_nothing(tmp_path) -> None:
    assert not neuron_ledger.affordable(0.001, 0, tmp_path / "neurons.json")


def test_a_write_that_fails_does_not_raise(tmp_path) -> None:
    """An uncounted call is better than an index that dies over bookkeeping."""
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    neuron_ledger.record(1.0, documents=1, path=blocked / "neurons.json")


def test_the_file_keeps_a_bounded_history(tmp_path) -> None:
    path = tmp_path / "neurons.json"
    old = {f"2020-01-{day:02d}": {"estimated_neurons": 1, "documents": 1, "calls": 1} for day in range(1, 29)}
    path.write_text(json.dumps({"days": old}), encoding="utf-8")
    neuron_ledger.record(1.0, documents=1, path=path)
    days = json.loads(path.read_text(encoding="utf-8"))["days"]
    assert len(days) <= neuron_ledger.HISTORY_DAYS
    assert neuron_ledger.today() in days
