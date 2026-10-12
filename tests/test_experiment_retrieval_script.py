"""`scripts/experiment_retrieval.py`'s question-set selection.

The script carries two question lists and names them in its source. A set written later cannot
be added to that source - `test_holdout3_is_not_wired_into_any_default_question_list` forbids it,
so that no default can reach an unscored set by accident - so there has to be a way to name one
on the command line. These tests cover that way, and the label a result file gets from it.
"""

from pathlib import Path

from scripts import experiment_retrieval as script

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions"


def test_the_two_built_in_lists_are_the_original_and_holdout2() -> None:
    assert [p.name for p in script.DEFAULT_SETS] == ["itsdangerous.json", "flask.json"]
    assert {p.parent.name for p in script.HOLDOUT2_SETS} == {"holdout2"}


def test_no_set_named_keeps_the_built_in_lists() -> None:
    assert script.question_files(None, "original") == list(script.DEFAULT_SETS)
    assert script.question_files(None, "holdout2") == list(script.HOLDOUT2_SETS)


def test_a_named_set_replaces_the_list_entirely() -> None:
    """Not added to: a run that says --set scores those files and nothing else."""
    named = [QUESTIONS / "elsewhere" / "itsdangerous.json"]

    assert script.question_files(named, "original") == named
    assert script.question_files(named, "holdout2") == named


def test_a_named_set_is_labelled_by_its_folder() -> None:
    paths = [QUESTIONS / "holdout2" / "itsdangerous.json", QUESTIONS / "holdout2" / "flask.json"]

    assert script.question_label(paths) == "holdout2"


def test_a_label_for_files_from_two_folders_says_so() -> None:
    """Honest rather than wrong: a run over two sets is not either one of them."""
    paths = [QUESTIONS / "itsdangerous.json", QUESTIONS / "holdout2" / "flask.json"]

    assert script.question_label(paths) == "mixed"


def test_a_relative_path_is_labelled_the_same_as_an_absolute_one(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)

    assert script.question_label([Path("eval/questions/holdout2/flask.json")]) == "holdout2"
