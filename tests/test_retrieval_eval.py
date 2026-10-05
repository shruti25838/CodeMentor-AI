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
    assert set(result["overall"]) == {"questions", "hit_rate", "precision", "mrr@10"}
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
