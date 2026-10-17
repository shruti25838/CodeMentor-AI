"""The committed benchmarks summary must be exactly what a fresh build produces.

This is the test that stops the summary being edited by hand or going quietly stale. It
rebuilds the whole thing from the committed result files and this repository's own app and
compares the result with `eval/benchmarks.json` byte for byte, so a result file that changes,
a route that is added or a figure typed into the file directly all fail here.

Nothing in this file clones a repository, calls a model or reaches an API.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("build_benchmarks", ROOT / "scripts" / "build_benchmarks.py")
build_benchmarks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_benchmarks)

SUMMARY = ROOT / "eval" / "benchmarks.json"
RESULTS_MD = ROOT / "docs" / "RESULTS.md"


@pytest.fixture(scope="module")
def built() -> dict:
    return build_benchmarks.build()


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the guard itself


def test_the_committed_summary_is_what_a_build_produces(built):
    """The whole point of the file. If this fails, run: python scripts/build_benchmarks.py"""
    assert SUMMARY.read_text(encoding="utf-8") == build_benchmarks.rendered(built)


def test_a_hand_edited_summary_is_caught(tmp_path, built):
    """The comparison is on content, not on the file being present."""
    edited = json.loads(build_benchmarks.rendered(built))
    edited["retrieval"]["providers"][0]["splits"][0]["rows"][0]["hit@1"] = 0.99
    path = tmp_path / "benchmarks.json"
    path.write_text(json.dumps(edited, indent=2) + "\n", encoding="utf-8")

    assert path.read_text(encoding="utf-8") != build_benchmarks.rendered(built)


def test_the_build_is_a_pure_function_of_its_inputs(built):
    """No timestamp, no run id: two builds of the same checkout are identical."""
    assert build_benchmarks.rendered(built) == build_benchmarks.rendered(build_benchmarks.build())
    assert "generated_at" not in build_benchmarks.rendered(built)


def test_the_check_mode_agrees_with_the_committed_file(capsys):
    assert build_benchmarks.main.__module__  # the script is importable without running
    sys.argv = ["build_benchmarks.py", "--check"]
    assert build_benchmarks.main() == 0


def test_the_check_mode_fails_on_a_stale_file(tmp_path, capsys):
    stale = tmp_path / "benchmarks.json"
    stale.write_text('{"schema": 1}\n', encoding="utf-8")
    sys.argv = ["build_benchmarks.py", "--check", "--out", str(stale)]

    assert build_benchmarks.main() == 1
    assert "stale or hand-edited" in capsys.readouterr().out


# ---------------------------------------------------------------- the numbers come from the files


def test_the_retrieval_rows_are_the_committed_result_files(committed):
    for name, split in (("dev", "development"), ("test", "held-out")):
        source = json.loads((ROOT / "eval" / "results" / f"{name}.json").read_text(encoding="utf-8"))
        rows = _split(committed, "hash", split)["rows"]
        overall = next(r for r in rows if r["scope"] == "overall")
        assert overall["hit@1"] == source["overall"]["hit_rate"]["@1"]
        assert overall["hit@5"] == source["overall"]["hit_rate"]["@5"]
        assert overall["p@3"] == source["overall"]["precision"]["@3"]
        assert overall["mrr@10"] == source["overall"]["mrr@10"]
        assert overall["recall@20"] == source["overall"]["recall@20"]


def test_the_cloudflare_rows_are_the_default_configuration_and_ordering(committed):
    source = json.loads((ROOT / "eval" / "results" / "cloudflare" / "test.json").read_text(encoding="utf-8"))
    expected = next(r for r in source["results"] if r["config"] == "mean-2000")["overall"]["keyword"]
    overall = next(r for r in _split(committed, "cloudflare", "held-out")["rows"] if r["scope"] == "overall")

    assert overall["hit@1"] == expected["hit@1"]
    assert overall["mrr@10"] == expected["mrr@10"]
    assert overall["recall@20"] == expected["recall@20"]


def test_a_metric_a_run_did_not_compute_is_null_not_zero(committed):
    """Cloudflare's development run reported no P@3. An absent number must not read as 0.000."""
    dev = _split(committed, "cloudflare", "development")["rows"]
    assert all(row["p@3"] is None for row in dev)
    held_out = _split(committed, "cloudflare", "held-out")["rows"]
    assert all(row["p@3"] is not None for row in held_out)


def test_the_structural_block_is_the_committed_structural_file(committed):
    source = json.loads((ROOT / "eval" / "results" / "structural" / "all.json").read_text(encoding="utf-8"))
    assert committed["structural"]["overall"] == source["overall"]
    assert committed["structural"]["by_kind"] == source["by_kind"]
    assert committed["structural"]["disagreements"] == source["disagreements"]


def test_the_endpoint_counts_are_the_app_as_it_is_now(committed):
    spec = importlib.util.spec_from_file_location("count_endpoints", ROOT / "scripts" / "count_endpoints.py")
    count_endpoints = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(count_endpoints)
    counted = count_endpoints.count()

    endpoints = committed["system"]["endpoints"]
    assert endpoints["routers"] == counted["routers"]
    assert endpoints["routes"] == counted["routes"]
    assert endpoints["open"] == counted["routes_open"]
    assert endpoints["need_a_key"] == counted["routes_needing_key"]


# ---------------------------------------------------------------- provenance and what must not be there


def test_every_block_says_when_it_was_recorded(committed):
    for block in _recorded_blocks(committed):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", block["date"]), block
        assert re.fullmatch(r"[0-9a-f]{7,40}", block["commit"]), block
        assert block["results_entry"]


def test_every_quoted_entry_is_still_in_results_md(committed):
    """A figure quoted from prose names the entry it came from; the entry must still exist."""
    headings = [
        line[4:].strip() for line in RESULTS_MD.read_text(encoding="utf-8").splitlines() if line.startswith("### ")
    ]
    for block in _recorded_blocks(committed):
        entry = block["results_entry"]
        assert any(h.startswith(entry) for h in headings), f"no RESULTS.md entry starts with: {entry}"


def test_every_retrieval_block_carries_its_embedder_and_question_count(committed):
    for provider in committed["retrieval"]["providers"]:
        assert provider["embedder"]
        for split in provider["splits"]:
            assert split["questions"] == 20
            assert split["recorded"]["commit"] and split["recorded"]["date"]


def test_the_heading_note_says_recorded_not_live(committed):
    assert committed["heading_note"] == "Recorded results on small fixed question sets, not live accuracy."


def test_end_to_end_latency_is_named_as_absent_and_no_figure_is_given(committed):
    text = SUMMARY.read_text(encoding="utf-8").lower()
    assert "depends on the model provider and is not benchmarked here" in committed["end_to_end_note"]
    for banned in ("ttfb", "p95_latency", "end_to_end_ms", "total_latency"):
        assert banned not in text, f"the summary must not carry {banned}"


def test_the_summary_carries_nothing_secret_or_private():
    text = SUMMARY.read_text(encoding="utf-8").lower()
    for banned in ("api_key", "api-key", "account_id", "budget", "neuron", "secret", "x-api-key"):
        assert banned not in text, f"the summary must not carry {banned}"


def test_no_question_text_from_any_question_set():
    """Not one question anyone asks appears in the summary - only counts and code identifiers.

    `eval/structural_questions.json` is not in this list because it holds no question text: a
    structural question is a kind and a Python identifier, and the identifiers of the four
    disagreements are shown on purpose, since a disagreement one cannot name is not a finding.
    """
    text = SUMMARY.read_text(encoding="utf-8")
    sets = [
        ROOT / "eval" / "questions" / "itsdangerous.json",
        ROOT / "eval" / "questions" / "flask.json",
        ROOT / "eval" / "answer_smoke_questions.json",
    ]
    for path in sets:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        # The retrieval sets wrap their questions; the other two are plain lists.
        questions = loaded["questions"] if isinstance(loaded, dict) else loaded
        asked = [q["question"] for q in questions]
        assert asked, f"{path.name} has no question text to check against"
        for question in asked:
            assert question not in text, f"{path.name} question text leaked into the summary"


def test_the_docker_image_copies_the_file_the_route_reads():
    assert "eval/benchmarks.json" in (ROOT / "Dockerfile").read_text(encoding="utf-8")


# ---------------------------------------------------------------- helpers


def _split(summary: dict, provider_id: str, split_id: str) -> dict:
    provider = next(p for p in summary["retrieval"]["providers"] if p["id"] == provider_id)
    return next(s for s in provider["splits"] if s["id"] == split_id)


def _recorded_blocks(value) -> list[dict]:
    """Every "recorded" block anywhere in the summary."""
    found: list[dict] = []
    if isinstance(value, dict):
        if "recorded" in value and isinstance(value["recorded"], dict):
            found.append(value["recorded"])
        for item in value.values():
            found += _recorded_blocks(item)
    elif isinstance(value, list):
        for item in value:
            found += _recorded_blocks(item)
    return found


# ---------------------------------------------------------------- the experiments block


def test_every_experiment_run_reaches_the_summary(committed):
    """Four dev runs of five arrangements each. A run that stopped being read would show here."""
    rows = committed["experiments"]["rows"]
    assert len(rows) == 20
    assert {(r["embedder"], r["records"]) for r in rows} == {
        ("hash", "whole"),
        ("hash", "structured"),
        ("cloudflare", "whole"),
        ("cloudflare", "structured"),
    }


def test_the_keep_rule_is_applied_rather_than_asserted(committed):
    """Every verdict is recomputed here from the numbers, against the rule as written."""
    rows = committed["experiments"]["rows"]
    baselines = {r["embedder"]: r["dev"] for r in rows if r["verdict"] == "baseline"}

    for row in rows:
        if row["verdict"] in ("baseline", "chosen"):
            continue
        base = baselines[row["embedder"]]
        met = row["dev"]["mrr@10"] > base["mrr@10"] and row["dev"]["hit@5"] >= base["hit@5"]
        assert met == row["verdict"].startswith("met the rule"), row


def test_exactly_one_chosen_configuration_per_embedder(committed):
    chosen = [r for r in committed["experiments"]["rows"] if r["verdict"] == "chosen"]
    assert sorted((r["embedder"], r["records"], r["arrangement"]) for r in chosen) == [
        ("cloudflare", "structured", "current"),
        ("hash", "structured", "rrf+keyword"),
    ]


def test_held_out_numbers_are_claimed_only_where_they_were_scored(committed):
    for row in committed["experiments"]["rows"]:
        if row["verdict"] == "chosen":
            assert row["held_out"] and row["holdout2"], row
        elif row["verdict"] == "baseline":
            assert row["holdout2"], row
            assert row["held_out"] is None, "the baseline's held-out figure is in the retrieval panel"
        else:
            assert row["held_out"] is None and row["holdout2"] is None, row


def test_the_page_says_the_original_held_out_split_was_reused(committed):
    """Said in the Experiments panel and again beside the retrieval table that shows it."""
    experiments = committed["experiments"]
    assert "reused" in experiments["reused_split_note"].lower()
    assert "holdout2" in experiments["fresh_split_note"]
    assert "fresher" in experiments["fresh_split_note"]
    assert "reused" in experiments["splits"]["held_out"]
    assert "scored once" in experiments["splits"]["holdout2"]

    retrieval = committed["retrieval"]
    assert "reused" in retrieval["splits_note"].lower()
    for provider in retrieval["providers"]:
        held_out = next(s for s in provider["splits"] if s["id"] == "held-out")
        development = next(s for s in provider["splits"] if s["id"] == "development")
        assert held_out["reused"] is True and "reused" in held_out["note"].lower()
        assert development["reused"] is False and development["note"] == ""


def test_the_two_held_out_sets_are_shown_side_by_side(committed):
    """Four configurations, each with both numbers and the source of each, and the gap between."""
    block = committed["experiments"]["splits_compared"]
    rows = block["rows"]

    assert block["metric"] == "mrr@10"
    assert [(r["embedder"], r["verdict"]) for r in rows] == [
        ("hash", "baseline"),
        ("hash", "chosen"),
        ("cloudflare", "baseline"),
        ("cloudflare", "chosen"),
    ]
    for row in rows:
        assert row["held_out"] is not None and row["holdout2"] is not None, row
        assert row["difference"] == round(row["holdout2"] - row["held_out"], 4), row
        assert row["held_out_source"].startswith("eval/results/"), row
        assert row["holdout2_source"].startswith("eval/results/"), row


def test_every_configuration_scores_worse_on_the_fresh_split(committed):
    """The relationship, not the numbers: a set that was never used scores lower than one that was.

    Asserted as a relationship so the table cannot stop showing it. If a later run made the fresh
    split the higher one, this would fail and the wording above would have to be rewritten rather
    than silently becoming false.
    """
    for row in committed["experiments"]["splits_compared"]["rows"]:
        assert row["holdout2"] < row["held_out"], row
        assert row["difference"] < 0, row


def test_the_side_by_side_numbers_come_from_the_committed_result_files(committed):
    """Each number is read back out of the file the row names, so none of them is typed in."""
    for row in committed["experiments"]["splits_compared"]["rows"]:
        fresh = json.loads((ROOT / row["holdout2_source"]).read_text(encoding="utf-8"))
        arrangement = row["change"].rsplit(", ", 1)[1]
        assert fresh["overall"][arrangement]["mrr@10"] == row["holdout2"], row


def test_holdout3_is_a_block_of_its_own_labelled_as_scored_once(committed):
    """The set that had never been scored: five rows, five metrics, and where each came from."""
    block = committed["experiments"]["holdout3"]

    assert block["scored_once"] is True
    assert "scored once" in block["label"]
    assert block["questions"] == 20
    assert block["metrics"] == ["hit@1", "hit@3", "hit@5", "mrr@10", "recall@20"]
    assert len(block["rows"]) == 5
    assert block["recorded"]["commit"] and block["recorded"]["date"] == "2026-10-12"
    for row in block["rows"]:
        assert row["source"].startswith("eval/results/experiments/holdout3-"), row
        assert all(row[metric] is not None for metric in block["metrics"]), row


def test_the_holdout3_rule_is_applied_rather_than_asserted(committed):
    """Every verdict recomputed here from the numbers, against the rule as it was written down.

    The two guard metrics differ by embedder on purpose, and that asymmetry is the part most
    likely to be quietly lost, so it is recomputed rather than read.
    """
    block = committed["experiments"]["holdout3"]
    guards = block["guards"]
    assert guards == {"hash": "hit@5", "cloudflare": "recall@20"}

    for embedder, guard in guards.items():
        rows = [r for r in block["rows"] if r["embedder"] == embedder]
        incumbent = next(r for r in rows if r["configuration"] == "the configuration it was scored against")
        candidate = next(r for r in rows if r["configuration"] == "candidate")
        passes = candidate["mrr@10"] > incumbent["mrr@10"] and candidate[guard] >= incumbent[guard]

        assert candidate["verdict"].startswith("passes") == passes, candidate
        assert guard in candidate["verdict"], candidate
        assert incumbent["verdict"] == "the incumbent"


def test_passing_the_rule_and_being_adopted_are_recorded_as_two_different_things(committed):
    """Both candidates passed holdout3; only one became the default, and the page says which."""
    block = committed["experiments"]["holdout3"]

    assert block["adopted"] == {"hash": True, "cloudflare": False}
    for embedder in block["guards"]:
        candidate = next(r for r in block["rows"] if r["embedder"] == embedder and r["configuration"] == "candidate")
        assert candidate["verdict"].startswith("passes"), candidate
    assert "only the hash one was adopted" in block["note"]


def test_the_ineligible_configuration_is_shown_and_says_it_is_ineligible(committed):
    """Hiding it would hide where the hash gain comes from; showing it unmarked would mislead."""
    rows = committed["experiments"]["holdout3"]["rows"]
    ineligible = [r for r in rows if r["eligible"] is False]

    assert len(ineligible) == 1
    assert ineligible[0] == next(r for r in rows if r["change"] == "whole records, rrf+keyword")
    assert "not eligible" in ineligible[0]["verdict"]
    assert not ineligible[0]["verdict"].startswith("passes")


def test_every_holdout3_number_comes_from_the_file_its_row_names(committed):
    """None of them is typed in: each is read back out of the committed result file."""
    for row in committed["experiments"]["holdout3"]["rows"]:
        data = json.loads((ROOT / row["source"]).read_text(encoding="utf-8"))
        arrangement = row["change"].rsplit(", ", 1)[1]
        scored = data["overall"][arrangement]

        assert data["split"] == "holdout3" and data["questions"] == "holdout3", row
        for metric in committed["experiments"]["holdout3"]["metrics"]:
            assert scored[metric] == row[metric], (row, metric)


def test_holdout3_has_a_column_in_both_experiment_tables(committed):
    """A number is claimed only for the rows that were actually scored on it."""
    assert "never used" in committed["experiments"]["splits"]["holdout3"]
    for row in committed["experiments"]["rows"]:
        if row["verdict"] in ("baseline", "chosen"):
            assert row["holdout3"], row
        else:
            assert row["holdout3"] is None, row

    for row in committed["experiments"]["splits_compared"]["rows"]:
        assert row["holdout3"] is not None, row
        assert "holdout3-" in row["holdout3_source"], row
        fresh = json.loads((ROOT / row["holdout3_source"]).read_text(encoding="utf-8"))
        arrangement = row["change"].rsplit(", ", 1)[1]
        assert fresh["overall"][arrangement]["mrr@10"] == row["holdout3"], row


def test_the_page_does_not_say_one_fresh_set_settles_it(committed):
    """Two sets scored once disagree with each other; the wording has to say so, not pick one."""
    experiments = committed["experiments"]

    assert "do not agree" in experiments["fresh_split_note"]
    assert "All three" in experiments["both_splits_note"]
    assert "no single set" in experiments["both_splits_note"]
    assert "holdout3" in experiments["splits_compared"]["note"]


def test_the_two_question_trial_is_not_a_source_for_the_page(committed):
    """It is kept as a record and must not be drawn under the full comparison's heading."""
    assert build_benchmarks.CHAT_MODE_SOURCES == ("chat-modes.json",)
    assert (ROOT / "eval" / "results" / "experiments" / "chat-modes-trial.json").exists()
    assert committed["experiments"]["chat_modes"]["source"].endswith("chat-modes.json")
    assert "trial" not in committed["experiments"]["chat_modes"]["source"]


def test_the_chat_mode_block_says_how_large_the_run_was(committed):
    chat = committed["experiments"]["chat_modes"]
    assert chat["questions"] == 15
    assert chat["runs"] == 30
    assert sum(mode["runs"] for mode in chat["modes"]) == chat["runs"]
    assert "15 questions, 30 runs" in chat["scale_note"]
    assert "trial" in chat["scale_note"]


def test_a_short_trial_is_refused_rather_than_drawn_as_the_full_run(tmp_path, monkeypatch):
    """Point the builder at the trial file and the block disappears instead of shrinking."""
    monkeypatch.setattr(build_benchmarks, "CHAT_MODE_SOURCES", ("chat-modes-trial.json",))
    assert build_benchmarks.chat_modes() is None


def test_the_chat_mode_comparison_says_what_it_is(committed):
    chat = committed["experiments"]["chat_modes"]
    assert "one local run" in chat["label"]
    assert "not a benchmark" in chat["label"]
    assert chat["model"]
    assert [m["mode"] for m in chat["modes"]] == ["legacy", "grounded"]
    assert "No judge model" in chat["weak_check"]


def test_the_experiments_block_claims_no_end_to_end_latency(committed):
    printed = json.dumps(committed["experiments"]).lower()
    for banned in ("ttfb", "end_to_end", "p95", "latency"):
        assert banned not in printed, f"the experiments block must not claim {banned}"


# ---------------------------------------------------------------- which configuration is the default


def test_the_hash_panel_names_the_configuration_the_server_runs(committed):
    provider = next(p for p in committed["retrieval"]["providers"] if p["id"] == "hash")

    assert provider["default"] is True
    assert provider["configuration"] == "structured records, BM25 fusion on"
    assert "BM25" in provider["note"] and "structured" in provider["note"]


@pytest.mark.parametrize("split_id", ["development", "held-out"])
def test_each_hash_table_says_which_configuration_it_was_measured_under(committed, split_id):
    """Read out of the result file itself, so a file that changed would change the label with it."""
    current = _split(committed, "hash", split_id)
    legacy = _split(committed, "hash", f"{split_id} (legacy)")

    assert current["configuration"] == "structured records, BM25 on"
    assert legacy["configuration"] == "whole records, BM25 off"
    assert current["source"] == f"eval/results/{'dev' if split_id == 'development' else 'test'}.json"
    assert legacy["source"].startswith("eval/results/legacy/")


def test_the_legacy_tables_say_they_are_not_the_current_numbers(committed):
    for split_id in ("development (legacy)", "held-out (legacy)"):
        note = _split(committed, "hash", split_id)["note"]
        assert "before the default change" in note
        assert "not the server's current numbers" in note


def test_the_legacy_held_out_pair_this_file_quotes_is_still_drawn(committed):
    """hit@5 0.750 and MRR@10 0.645: the pair hundreds of lines of docs/RESULTS.md end with."""
    rows = _split(committed, "hash", "held-out (legacy)")["rows"]
    overall = next(r for r in rows if r["scope"] == "overall")

    assert (overall["hit@5"], overall["mrr@10"]) == (0.750, 0.645)


def test_the_default_moved_for_hash_and_did_not_for_cloudflare(committed):
    rows = committed["experiments"]["splits_compared"]["rows"]
    default_rows = {r["embedder"]: r for r in rows if r["is_default"]}

    assert default_rows["hash"]["change"] == "structured records, rrf+keyword"
    assert default_rows["cloudflare"]["change"] == "whole records, current"
    assert default_rows["hash"]["configuration"] == "the default"
    assert next(r for r in rows if r["embedder"] == "hash" and not r["is_default"])["configuration"] == (
        "the previous default"
    )
    assert next(r for r in rows if r["embedder"] == "cloudflare" and not r["is_default"])["configuration"] == (
        "candidate, not adopted"
    )


def test_exactly_one_row_per_embedder_is_the_default(committed):
    for block in (committed["experiments"]["rows"], committed["experiments"]["splits_compared"]["rows"]):
        for embedder in ("hash", "cloudflare"):
            marked = [r for r in block if r["embedder"] == embedder and r["is_default"]]
            assert len(marked) == 1, (embedder, marked)


def test_the_legacy_baseline_row_is_read_from_the_legacy_file(committed):
    """It describes whole records, and eval/results/test.json no longer does."""
    row = next(
        r
        for r in committed["experiments"]["splits_compared"]["rows"]
        if r["embedder"] == "hash" and r["configuration"] == "the previous default"
    )
    legacy = json.loads((ROOT / "eval" / "results" / "legacy" / "test.json").read_text(encoding="utf-8"))

    assert row["held_out_source"] == "eval/results/legacy/test.json"
    assert row["held_out"] == legacy["overall"]["mrr@10"] == 0.645


# ---------- the headline block ----------
#
# The block at the top of the evaluation page, and for most readers the only part of it they
# will see. It is built from the blocks below it rather than from the result files a second
# time, so the thing worth asserting is that it agrees with them: a summary that disagrees
# with the table it summarises is worse than no summary.


def test_the_headline_names_both_embedders_at_their_shipped_configuration(committed):
    rows = committed["headline"]["retrieval"]

    assert [r["embedder"] for r in rows] == ["hash", "cloudflare"]
    assert rows[0]["configuration"] == "structured records, BM25 fusion on"
    assert rows[1]["configuration"] == "whole records, no fusion"
    assert [r["is_default"] for r in rows] == [True, False]


def test_each_headline_row_is_the_experiments_row_the_server_runs(committed):
    """Not the best row, not the first row: the one marked is_default by the builder."""
    for row in committed["headline"]["retrieval"]:
        source = next(
            r for r in committed["experiments"]["rows"] if r["embedder"] == row["embedder"] and r["is_default"]
        )
        for split in ("holdout2", "holdout3"):
            assert row[split]["mrr@10"] == source[split]["mrr@10"], (row["embedder"], split)
            assert row[split]["hit@5"] == source[split]["hit@5"], (row["embedder"], split)
            assert row[split]["questions"] == source[split]["questions"], (row["embedder"], split)


def test_the_headline_shows_the_two_sets_that_were_scored_once(committed):
    """Not the original held-out split, which was reused run after run and flatters everything."""
    headline = committed["headline"]

    assert set(headline["splits"]) == {"holdout2", "holdout3"}
    for row in headline["retrieval"]:
        assert "held_out" not in row and "dev" not in row
        assert set(row["holdout2"]) == {"questions", "mrr@10", "hit@5"}
    assert "scored once each" in headline["splits_note"]
    assert "reused" in headline["splits_note"]


def test_the_headline_reports_only_mrr_at_10_and_hit_at_5(committed):
    assert committed["headline"]["metrics"] == ["mrr@10", "hit@5"]


def test_the_headline_structural_figures_are_the_structural_blocks_own(committed):
    headline = committed["headline"]["structural"]
    overall = committed["structural"]["overall"]

    assert headline["precision"] == overall["precision"]
    assert headline["recall"] == overall["recall"]
    assert headline["questions"] == overall["questions"]
    assert "ast" in headline["title"]


def test_the_headline_timing_says_the_model_is_not_in_it(committed):
    """Required wording: the figure means something else entirely without it."""
    headline = committed["headline"]["timing"]

    assert headline["label"] == "excludes the language model"
    assert headline["warm_total_ms"] == committed["timing"]["current"]["warm_total_ms"]
    assert headline["rerank_ms"] == committed["timing"]["current"]["rerank_ms"]
    assert "End-to-end latency is not shown" in headline["note"]


def test_the_headline_repeats_that_these_are_recorded_results(committed):
    assert committed["headline"]["note"] == committed["heading_note"]
    assert "not live accuracy" in committed["headline"]["note"]


def test_the_headline_is_derived_rather_than_read_a_second_time():
    """`build()` reads each result file once and hands the blocks to `headline`.

    If `headline` called `experiments()` itself, the two could be built from different reads
    of the same folder and a headline figure could drift from its table without any test here
    noticing - every check above compares the two halves of one build.
    """
    source = (ROOT / "scripts" / "build_benchmarks.py").read_text(encoding="utf-8")
    body = source[source.index("def headline(") : source.index("def build()")]

    for builder in ("experiments()", "structural()", "timing()", "_read("):
        assert builder not in body, f"headline() must not call {builder}"
