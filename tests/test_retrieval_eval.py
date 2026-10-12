import json
import subprocess
from pathlib import Path

import pytest
from test_git_loader import _make_source_repo

from codeatlas.services.eval.basic_eval import hit_at_k, precision_at_k, ranking_metrics, reciprocal_rank
from codeatlas.services.eval.retrieval_eval import (
    QuestionSet,
    checkout,
    describe_embedder,
    evaluate_set,
    load_question_set,
    run,
)
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService

QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "eval" / "questions"


def test_hit_precision_and_reciprocal_rank():
    ranked = ["a.py", "b.py", "a.py", "c.py"]
    expected = {"b.py", "c.py"}
    assert hit_at_k(ranked, expected, 1) == 0.0
    assert hit_at_k(ranked, expected, 3) == 1.0
    assert precision_at_k(ranked, expected, 1) == 0.0
    assert precision_at_k(ranked, expected, 3) == pytest.approx(1 / 3)
    assert precision_at_k(ranked, expected, 5) == pytest.approx(2 / 5)  # missing 5th result counts as a miss
    assert reciprocal_rank(ranked, expected) == 0.5
    assert reciprocal_rank(ranked, {"z.py"}) == 0.0


def test_ranking_metrics_average_over_questions():
    m = ranking_metrics([(["x.py"], {"x.py"}), (["y.py", "x.py"], {"x.py"})], ks=(1, 3))
    assert m.questions == 2
    assert m.hit_rate == {1: 0.5, 3: 1.0}
    assert m.precision[1] == 0.5
    assert m.mrr == pytest.approx((1 + 0.5) / 2)
    assert ranking_metrics([], ks=(1,)).hit_rate == {1: 0.0}


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_question_files_are_pinned_and_split(name):
    qs = load_question_set(QUESTIONS_DIR / f"{name}.json")
    assert len(qs.commit) == 40
    assert len(qs.questions) == 20
    assert len({q.id for q in qs.questions}) == 20
    assert len(qs.select("dev")) == 10 and len(qs.select("test")) == 10
    assert all(q.expected_files for q in qs.questions)
    assert not {q.question for q in qs.select("dev")} & {q.question for q in qs.select("test")}


def test_question_without_expected_files_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "name": "x",
                "repo_url": "u",
                "commit": "c",
                "questions": [{"id": "1", "split": "dev", "question": "q", "expected_files": []}],
            }
        )
    )
    with pytest.raises(ValueError):
        load_question_set(path)


SOURCE = {
    "signer.py": "def sign_value(value, key):\n    return hmac_digest(value, key)\n",
    "clock.py": "def expired(timestamp, max_age):\n    return now() - timestamp > max_age\n",
    "colors.py": "def paint(canvas, color):\n    canvas.fill(color)\n",
}


def _question_file(tmp_path: Path, url: str, commit: str) -> Path:
    path = tmp_path / "demo.json"
    questions = [
        {
            "id": "d1",
            "split": "dev",
            "question": "how is a value signed with the key?",
            "expected_files": ["signer.py"],
        },
        {"id": "d2", "split": "test", "question": "when has a timestamp expired?", "expected_files": ["clock.py"]},
    ]
    path.write_text(json.dumps({"name": "demo", "repo_url": url, "commit": commit, "questions": questions}))
    return path


def _head(url: str) -> str:
    out = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, check=True).stdout
    return out.split()[0]


def test_checkout_fetches_the_pinned_commit_once(tmp_path):
    url = _make_source_repo(tmp_path / "src", SOURCE)
    commit = _head(url)
    qs = QuestionSet(name="demo", repo_url=url, commit=commit, questions=[])
    root = checkout(qs, tmp_path / "cache")
    assert (root / "signer.py").is_file()
    marker = root / "untracked-marker"
    marker.write_text("kept")
    assert checkout(qs, tmp_path / "cache") == root
    assert marker.exists()  # reused, not fetched again


def test_evaluate_set_uses_the_real_index_and_reports_relative_paths(tmp_path):
    url = _make_source_repo(tmp_path / "src", SOURCE)
    qs = load_question_set(_question_file(tmp_path, url, _head(url)))
    root = checkout(qs, tmp_path / "cache")
    result = evaluate_set(qs, root, HashEmbeddingService(), "dev")
    assert result.files_indexed == 3
    assert result.metrics.questions == 1
    top = result.per_question[0]["top5"]
    assert top and all("/" not in p and not p.startswith(str(tmp_path)) for p in top)
    assert result.metrics.hit_rate[5] == 1.0


def test_run_reports_embedder_split_and_overall(tmp_path):
    url = _make_source_repo(tmp_path / "src", SOURCE)
    path = _question_file(tmp_path, url, _head(url))
    result = run([path], HashEmbeddingService(), "all", tmp_path / "cache")
    assert result["embedder"].startswith("hash")
    assert result["split"] == "all"
    assert result["overall"]["questions"] == 2
    assert set(result["overall"]) == {"questions", "hit_rate", "precision", "mrr@10", "recall@20"}
    assert set(result["overall"]["hit_rate"]) == {"@1", "@3", "@5"}
    # Question text never goes into the results, only ids.
    assert "signed" not in json.dumps(result["per_question"])


def test_describe_embedder_names_the_model():
    assert "384 dims" in describe_embedder(HashEmbeddingService())


def test_parser_walks_files_in_sorted_order(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from codeatlas.models.repository import Repository
    from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser

    for name in ("b.py", "a.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n")
    original = Path.rglob
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: reversed(list(original(self, pattern))))
    repo = Repository(repo_id="r", name="r", url="u", root_path=str(tmp_path), ingested_at=datetime.now(UTC))
    parsed = TreeSitterAstParser().parse_repository(repo)
    assert [Path(f.path).name for f in parsed.files] == ["a.py", "b.py", "c.py"]


def test_script_exit_code_follows_min_hit_rate(tmp_path):
    import sys

    url = _make_source_repo(tmp_path / "src", SOURCE)
    path = _question_file(tmp_path, url, _head(url))
    script = Path(__file__).resolve().parent.parent / "scripts" / "eval_retrieval.py"
    base = [sys.executable, str(script), "--set", str(path), "--split", "all", "--cache-dir", str(tmp_path / "c")]
    ok = subprocess.run([*base, "--min-hit-rate", "5=0.5"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "ok: overall hit@5" in ok.stdout
    fail = subprocess.run([*base, "--min-hit-rate", "5=1.01"], capture_output=True, text=True)
    assert fail.returncode == 1
    assert "FAIL: overall hit@5" in fail.stdout


# --- recall@20, and what the clip removes ------------------------------------------------


def test_recall_counts_the_whole_candidate_pool_not_the_ranked_answer() -> None:
    """The two are different questions, and the difference is the whole reason recall is here."""
    from codeatlas.services.eval.basic_eval import RECALL_DEPTH, ranking_metrics

    ranked = [("a.py", {"z.py"})]
    pool = [([*[f"f{i}.py" for i in range(19)], "z.py"], {"z.py"})]
    metrics = ranking_metrics(ranked, (1, 3, 5), candidates=pool)
    assert metrics.hit_rate[5] == 0.0
    assert metrics.mrr == 0.0
    assert metrics.recall == 1.0
    assert RECALL_DEPTH == 20


def test_a_file_past_the_pool_depth_does_not_count_as_recalled() -> None:
    from codeatlas.services.eval.basic_eval import ranking_metrics

    pool = [([*[f"f{i}.py" for i in range(20)], "z.py"], {"z.py"})]
    assert ranking_metrics([("a.py", {"z.py"})], (1, 3, 5), candidates=pool).recall == 0.0


def test_without_a_candidate_pool_recall_falls_back_to_what_is_there() -> None:
    from codeatlas.services.eval.basic_eval import ranking_metrics

    assert ranking_metrics([(["z.py"], {"z.py"})], (1, 3, 5)).recall == 1.0


def test_the_clip_counts_records_over_the_limit_per_scope(tmp_path) -> None:

    from codeatlas.models.function_node import FunctionNode
    from codeatlas.models.parsed_repository import ParsedRepository
    from codeatlas.models.source_file import SourceFile
    from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
    from codeatlas.services.retrieval.indexing import CodeIndexService

    source = tmp_path / "big.py"
    source.write_text("\n".join(f"x = {i}" for i in range(400)), encoding="utf-8")
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(source), language="python", size_bytes=source.stat().st_size)],
        functions=[FunctionNode(name="f", file_path=str(source), start_line=1, end_line=2, signature="()")],
    )
    service = CodeIndexService(embedder=HashEmbeddingService(), retriever=FaissCodeRetriever(), max_chars=100)
    counts = service.clipped_counts(parsed)
    assert counts["file"] == {"records": 1, "clipped": 1, "chars": len(source.read_text()), "chars_kept": 100}
    assert counts["function"]["clipped"] == 0


def test_with_no_clip_nothing_is_counted_as_clipped(tmp_path) -> None:

    from codeatlas.models.parsed_repository import ParsedRepository
    from codeatlas.models.source_file import SourceFile
    from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
    from codeatlas.services.retrieval.indexing import CodeIndexService

    source = tmp_path / "big.py"
    source.write_text("x = 1\n" * 500, encoding="utf-8")
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(source), language="python", size_bytes=source.stat().st_size)],
        functions=[],
    )
    service = CodeIndexService(embedder=HashEmbeddingService(), retriever=FaissCodeRetriever(), max_chars=None)
    counts = service.clipped_counts(parsed)
    assert counts["file"]["clipped"] == 0
    assert counts["file"]["chars_kept"] == counts["file"]["chars"]


# ---------------------------------------------------------------- the second held-out set


HOLDOUT2_DIR = QUESTIONS_DIR / "holdout2"


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_holdout2_is_ten_questions_pinned_at_the_same_commit(name):
    original = load_question_set(QUESTIONS_DIR / f"{name}.json")
    fresh = load_question_set(HOLDOUT2_DIR / f"{name}.json")

    assert fresh.commit == original.commit, "holdout2 must describe the same code as the first set"
    assert fresh.repo_url == original.repo_url
    assert len(fresh.questions) == 10
    assert len({q.id for q in fresh.questions}) == 10
    assert all(q.split == "holdout2" for q in fresh.questions)
    assert all(q.expected_files for q in fresh.questions)


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_no_holdout2_question_repeats_one_from_the_first_set(name):
    original = load_question_set(QUESTIONS_DIR / f"{name}.json")
    fresh = load_question_set(HOLDOUT2_DIR / f"{name}.json")

    assert not {q.question for q in fresh.questions} & {q.question for q in original.questions}
    assert not {q.id for q in fresh.questions} & {q.id for q in original.questions}


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_holdout2_questions_are_not_near_duplicates_of_the_first_set(name):
    """A reworded question is still the same question. Word overlap is the cheap check for that.

    This cannot prove two questions ask different things - only a reader can - but it does catch
    the failure that matters here, which is writing a "new" set by shuffling the old wording.
    """
    original = [_words(q.question) for q in load_question_set(QUESTIONS_DIR / f"{name}.json").questions]
    for q in load_question_set(HOLDOUT2_DIR / f"{name}.json").questions:
        words = _words(q.question)
        for other in original:
            overlap = len(words & other) / len(words | other)
            assert overlap < 0.5, f"{q.id} overlaps an existing question by {overlap:.0%} of its words"


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_holdout2_selects_only_under_its_own_split(name):
    fresh = load_question_set(HOLDOUT2_DIR / f"{name}.json")

    assert len(fresh.select("holdout2")) == 10
    assert fresh.select("dev") == [] and fresh.select("test") == []
    assert len(fresh.select("all")) == 10


def test_an_unknown_split_is_still_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "name": "x",
                "repo_url": "u",
                "commit": "c",
                "questions": [{"id": "1", "split": "holdout9", "question": "q", "expected_files": ["a.py"]}],
            }
        )
    )
    with pytest.raises(ValueError):
        load_question_set(path)


# ---------------------------------------------------------------- the third held-out set


HOLDOUT3_DIR = QUESTIONS_DIR / "holdout3"


def _earlier_sets(name: str) -> list:
    """Every question that existed before holdout3 was written, for one repository."""
    return [
        *load_question_set(QUESTIONS_DIR / f"{name}.json").questions,
        *load_question_set(HOLDOUT2_DIR / f"{name}.json").questions,
    ]


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_holdout3_is_ten_questions_pinned_at_the_same_commit(name):
    original = load_question_set(QUESTIONS_DIR / f"{name}.json")
    fresh = load_question_set(HOLDOUT3_DIR / f"{name}.json")

    assert fresh.commit == original.commit, "holdout3 must describe the same code as the earlier sets"
    assert fresh.repo_url == original.repo_url
    assert len(fresh.questions) == 10
    assert len({q.id for q in fresh.questions}) == 10
    assert all(q.split == "holdout3" for q in fresh.questions)
    assert all(q.expected_files for q in fresh.questions)


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_no_holdout3_question_repeats_an_earlier_one(name):
    """Against dev, the original held-out split **and** holdout2 - all thirty, not just the first set."""
    earlier = _earlier_sets(name)
    fresh = load_question_set(HOLDOUT3_DIR / f"{name}.json")

    assert len(earlier) == 30
    assert not {q.question for q in fresh.questions} & {q.question for q in earlier}
    assert not {q.id for q in fresh.questions} & {q.id for q in earlier}


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_holdout3_questions_are_not_near_duplicates_of_an_earlier_one(name):
    """The same cheap reworded-question check holdout2 gets, run against all thirty earlier ones."""
    earlier = [_words(q.question) for q in _earlier_sets(name)]
    for q in load_question_set(HOLDOUT3_DIR / f"{name}.json").questions:
        words = _words(q.question)
        for other in earlier:
            overlap = len(words & other) / len(words | other)
            assert overlap < 0.5, f"{q.id} overlaps an earlier question by {overlap:.0%} of its words"


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_holdout3_selects_only_under_its_own_split(name):
    fresh = load_question_set(HOLDOUT3_DIR / f"{name}.json")

    assert len(fresh.select("holdout3")) == 10
    assert fresh.select("dev") == [] and fresh.select("test") == [] and fresh.select("holdout2") == []
    assert len(fresh.select("all")) == 10


@pytest.mark.parametrize("name", ["itsdangerous", "flask"])
def test_holdout3_expects_files_that_exist_in_the_question_set(name):
    """Every expected path is a real file in the repository the set pins, fetched at that commit."""
    question_set = load_question_set(HOLDOUT3_DIR / f"{name}.json")
    root = checkout(question_set, Path(__file__).resolve().parent.parent / ".codeatlas" / "eval-repos")

    for q in question_set.questions:
        for path in q.expected_files:
            assert (root / path).is_file(), f"{q.id} expects {path}, which is not in the checkout"


def test_holdout3_is_not_wired_into_any_default_question_list():
    """It has not been scored, and nothing should score it by accident.

    The two eval scripts name the question files they read. holdout3 is in neither list, so a
    plain run of either cannot reach it - only an explicit --set can.
    """
    for script in ("eval_retrieval.py", "experiment_retrieval.py"):
        source = (Path(__file__).resolve().parent.parent / "scripts" / script).read_text(encoding="utf-8")
        assert "holdout3" not in source, f"scripts/{script} names holdout3; it must stay unscored"


def _words(question: str) -> set[str]:
    return {w for w in question.lower().replace("?", " ").replace(",", " ").split() if len(w) > 2}
