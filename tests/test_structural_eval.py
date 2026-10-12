"""The evaluation's own ground-truth logic, checked on fixtures with no network.

scripts/structural_eval.py reads real repositories; these tests cover the parts that decide
what counts as correct, so a scoring bug cannot quietly flatter the call graph.
"""

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import structural_eval  # noqa: E402
from structural_eval import QUESTIONS, REPOS, Scored, Truth, module_name, score  # noqa: E402

from codeatlas.services.analysis.call_graph import CallGraph  # noqa: E402

SOURCES = {
    "src/pkg/__init__.py": "from .core import Engine\n",
    "src/pkg/util.py": "def helper(x):\n    return x\n",
    "src/pkg/core.py": ("from .util import helper\n\nclass Engine:\n    def run(self, v):\n        return helper(v)\n"),
}


@pytest.fixture
def repo(tmp_path):
    for name, text in SOURCES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_module_name_drops_src_and_init(tmp_path) -> None:
    assert module_name(tmp_path / "src" / "pkg" / "core.py", tmp_path) == "pkg.core"
    assert module_name(tmp_path / "src" / "pkg" / "__init__.py", tmp_path) == "pkg"
    assert module_name(tmp_path / "tests" / "test_a.py", tmp_path) == "tests.test_a"


def test_ground_truth_finds_definitions(repo) -> None:
    truth = Truth.build(repo)
    assert truth.definitions["Engine"] == {"src/pkg/core.py:3"}
    assert truth.definitions["helper"] == {"src/pkg/util.py:1"}
    assert truth.qualified["Engine.run"] == {"src/pkg/core.py:4"}


def test_ground_truth_finds_call_sites_by_name(repo) -> None:
    truth = Truth.build(repo)
    assert truth.call_sites["helper"] == {"src/pkg/core.py:5"}


def test_ground_truth_resolves_relative_imports(repo) -> None:
    truth = Truth.build(repo)
    assert truth.importers["pkg.util"] == {"pkg.core"}
    assert truth.importers["pkg.core"] == {"pkg"}


def test_ground_truth_ignores_a_file_that_does_not_parse(tmp_path) -> None:
    (tmp_path / "ok.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_path / "bad.py").write_text("def (((\n", encoding="utf-8")
    assert Truth.build(tmp_path).definitions["a"] == {"ok.py:1"}


# ---------- scoring ----------


def test_the_graph_agrees_with_ast_on_a_small_repo(repo) -> None:
    graph = CallGraph.build(repo)
    truth = Truth.build(repo)
    for kind, subject in (
        ("where_defined", "Engine"),
        ("where_defined", "helper"),
        ("who_calls", "helper"),
        ("who_imports", "pkg.util"),
    ):
        result = score({"kind": kind, "subject": subject}, graph, truth)
        assert result.exact, f"{kind} {subject}: predicted {result.predicted}, expected {result.expected}"


def test_precision_and_recall_are_computed_as_sets() -> None:
    s = Scored({}, predicted={"a", "b"}, expected={"b", "c"})
    assert s.precision == 0.5
    assert s.recall == 0.5
    assert not s.exact


def test_a_miss_lowers_recall_but_not_precision() -> None:
    s = Scored({}, predicted={"a"}, expected={"a", "b"})
    assert s.precision == 1.0
    assert s.recall == 0.5


def test_a_false_positive_lowers_precision() -> None:
    s = Scored({}, predicted={"a", "x"}, expected={"a"})
    assert s.precision == 0.5
    assert s.recall == 1.0


def test_both_empty_counts_as_exact() -> None:
    s = Scored({}, predicted=set(), expected=set())
    assert s.exact and s.precision == 1.0 and s.recall == 1.0


def test_an_unknown_question_kind_is_rejected(repo) -> None:
    with pytest.raises(ValueError, match="unknown question kind"):
        score({"kind": "nonsense", "subject": "x"}, CallGraph.build(repo), Truth.build(repo))


# ---------- the question set itself ----------


def test_the_question_set_is_well_formed() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    assert len(questions) >= 40
    for q in questions:
        assert q["repo"] in REPOS
        assert q["split"] in ("development", "held-out")
        assert q["kind"] in ("who_calls", "who_imports", "where_defined")
        assert q["subject"] and q["note"]


def test_both_repositories_have_about_twenty_questions_each() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for repo_name in REPOS:
        count = sum(1 for q in questions if q["repo"] == repo_name)
        assert 20 <= count <= 30, f"{repo_name} has {count} questions"


def test_both_splits_are_used_for_both_repositories() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for repo_name in REPOS:
        splits = {q["split"] for q in questions if q["repo"] == repo_name}
        assert splits == {"development", "held-out"}


def test_the_questions_are_syntactically_valid_python_names() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for q in questions:
        for part in q["subject"].split("."):
            assert part.isidentifier(), f"{q['subject']} is not a dotted identifier"


def test_the_eval_script_makes_no_model_call() -> None:
    source = (ROOT / "scripts" / "structural_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "LlmProvider" not in names
    for banned in ("langchain", "openai", "get_chat_model"):
        assert banned not in source, f"the evaluation must not use {banned}"


# ---------------------------------------------------------------- the committed result file


class _Args:
    """What as_result reads off the parsed arguments."""

    split = "all"
    repo = "all"


def _scored(kind: str, subject: str, predicted: set[str], expected: set[str], **extra) -> Scored:
    question = {"repo": "itsdangerous", "split": "development", "kind": kind, "subject": subject, **extra}
    return Scored(question, predicted, expected)


def test_as_result_groups_by_split_repo_and_kind() -> None:
    rows = [
        _scored("where_defined", "Signer", {"a.py:1"}, {"a.py:1"}),
        _scored("who_calls", "sign", {"a.py:2"}, {"a.py:2", "a.py:3"}, split="held-out"),
    ]
    result = structural_eval.as_result(rows, _Args())

    assert result["overall"] == {"questions": 2, "precision": 1.0, "recall": 0.75, "exact": 1}
    assert result["by_split"]["development"]["questions"] == 1
    assert result["by_split"]["held-out"]["exact"] == 0
    assert result["by_repo"]["itsdangerous"]["questions"] == 2
    assert result["by_kind"]["where_defined"]["exact"] == 1
    assert result["by_kind"]["who_imports"] == {"questions": 0, "precision": None, "recall": None, "exact": 0}


def test_as_result_lists_each_disagreement_with_both_directions() -> None:
    rows = [
        _scored("where_defined", "Signer", {"a.py:1"}, {"a.py:1"}),
        _scored("who_calls", "sign", {"b.py:9"}, {"a.py:2"}, note="dispatched through self"),
    ]
    result = structural_eval.as_result(rows, _Args())

    assert len(result["disagreements"]) == 1
    only = result["disagreements"][0]
    assert only["subject"] == "sign"
    assert only["note"] == "dispatched through self"
    assert only["ast_found_graph_missed"] == ["a.py:2"]
    assert only["graph_found_ast_did_not"] == ["b.py:9"]


def test_as_result_records_the_pinned_repository_commits() -> None:
    result = structural_eval.as_result([], _Args())
    assert result["repos"] == {name: {"commit": commit} for name, (_, commit) in REPOS.items()}
    assert result["ground_truth"] == "python ast"
    assert result["language"] == "python"


def test_the_committed_result_file_matches_the_recorded_figures() -> None:
    """The file the benchmarks summary reads, against the entry in docs/RESULTS.md."""
    result = json.loads((ROOT / "eval" / "results" / "structural" / "all.json").read_text(encoding="utf-8"))
    assert result["overall"] == {"questions": 49, "precision": 1.0, "recall": 0.9673, "exact": 45}
    assert result["by_split"]["development"]["questions"] == 26
    assert result["by_split"]["held-out"]["questions"] == 23
    assert len(result["disagreements"]) == 4
    # Every disagreement is a miss; the graph never claimed a call that ast did not find.
    assert all(not d["graph_found_ast_did_not"] for d in result["disagreements"])
