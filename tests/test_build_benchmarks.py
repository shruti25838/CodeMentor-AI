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
