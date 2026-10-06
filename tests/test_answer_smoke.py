"""The answer smoke check's own logic: the two string tests and the token pacer.

The check itself needs a real model, so it is not run here. What is tested is the part that
decides pass or fail, and the question set, so a scoring bug cannot quietly flatter a run.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from answer_smoke import QUESTIONS, REPOS, Pacer, judge  # noqa: E402


def item(expected_file: str = "src/flask/helpers.py", fact: str = "send_from_directory") -> dict:
    return {"expected_file": expected_file, "fact": fact}


def result(citations: list[str], answer: str) -> dict:
    return {"citations": citations, "answer": answer}


# ---------- the file test ----------


def test_the_expected_file_cited_exactly_passes() -> None:
    verdict = judge(
        item(), result(["src/flask/helpers.py (lines 543-560) | def send_from_directory("], "use send_from_directory")
    )
    assert verdict["file_ok"] and verdict["passed"]


def test_a_different_file_fails() -> None:
    verdict = judge(item(), result(["src/flask/app.py (lines 1-5) | x"], "use send_from_directory"))
    assert not verdict["file_ok"]
    assert not verdict["passed"]


def test_no_citations_at_all_fails() -> None:
    assert not judge(item(), result([], "use send_from_directory"))["passed"]


def test_the_expected_file_among_several_citations_passes() -> None:
    citations = [
        "src/flask/app.py (lines 1-5) | x",
        "src/flask/helpers.py (lines 543-560) | def send_from_directory(",
    ]
    assert judge(item(), result(citations, "send_from_directory"))["file_ok"]


def test_a_windows_style_citation_path_still_matches() -> None:
    """Display paths are normalised to forward slashes, but a backslash must not fail the check."""
    verdict = judge(item(), result(["src\\flask\\helpers.py (lines 543-560) | x"], "send_from_directory"))
    assert verdict["file_ok"]


def test_a_citation_without_a_line_range_still_matches() -> None:
    assert judge(item(), result(["src/flask/helpers.py | x"], "send_from_directory"))["file_ok"]


# ---------- the fact test ----------


def test_the_fact_must_appear_in_the_answer() -> None:
    cited = ["src/flask/helpers.py (lines 543-560) | x"]
    assert judge(item(), result(cited, "You want send_from_directory here."))["fact_ok"]
    assert not judge(item(), result(cited, "You want some other helper."))["fact_ok"]


def test_the_fact_test_ignores_case() -> None:
    cited = ["src/flask/sessions.py (lines 291-291) | x"]
    verdict = judge(item("src/flask/sessions.py", "cookie-session"), result(cited, 'The salt is "Cookie-Session".'))
    assert verdict["fact_ok"]


def test_both_tests_must_pass() -> None:
    cited_right = ["src/flask/helpers.py (lines 1-2) | x"]
    cited_wrong = ["src/flask/app.py (lines 1-2) | x"]
    assert not judge(item(), result(cited_wrong, "send_from_directory"))["passed"], "right fact, wrong file"
    assert not judge(item(), result(cited_right, "nothing useful"))["passed"], "right file, wrong fact"
    assert judge(item(), result(cited_right, "send_from_directory"))["passed"]


# ---------- the pacer ----------


def test_the_pacer_does_not_wait_when_nothing_has_been_spent() -> None:
    assert Pacer().wait_for(7000) == 0.0


def test_the_pacer_forgets_spending_older_than_a_minute() -> None:
    import time

    pacer = Pacer(budget=7000)
    pacer.window = [(time.monotonic() - 120, 6000)]
    assert pacer.spent_recently() == 0
    assert pacer.wait_for(6000) == 0.0


def test_the_pacer_counts_recent_spending() -> None:
    import time

    pacer = Pacer(budget=7000)
    pacer.window = [(time.monotonic(), 4000)]
    assert pacer.spent_recently() == 4000


def test_recording_adds_to_the_window() -> None:
    pacer = Pacer()
    pacer.record(1234)
    assert pacer.spent_recently() == 1234


# ---------- the question set ----------


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def test_about_fifteen_questions_across_both_repositories(questions) -> None:
    assert 14 <= len(questions) <= 20
    assert {q["repo"] for q in questions} == set(REPOS)


def test_every_question_has_a_file_a_fact_and_a_note(questions) -> None:
    for q in questions:
        assert q["question"].endswith("?")
        assert q["expected_file"].startswith("src/")
        assert q["fact"]
        assert q["note"], "where the fact was read from, so it can be rechecked by hand"


def test_the_facts_are_short_enough_to_appear_verbatim(questions) -> None:
    for q in questions:
        assert len(q["fact"]) <= 24, f"{q['fact']!r} is a phrase, not a fact a model would echo"


def test_no_question_repeats(questions) -> None:
    assert len({q["question"] for q in questions}) == len(questions)


def test_every_expected_file_exists_in_the_cached_clone(questions) -> None:
    """Catches a typo in a path, and a fact written against a file that moved."""
    cache = ROOT / ".codeatlas" / "eval-repos"
    clones = {p.name.split("-")[0]: p for p in cache.glob("*-*")} if cache.exists() else {}
    if not clones:
        pytest.skip("no cached eval clones; run scripts/structural_eval.py first")
    for q in questions:
        clone = clones.get(q["repo"])
        if clone is None:
            continue
        assert (clone / q["expected_file"]).exists(), f"{q['repo']}: {q['expected_file']} is missing"


def test_every_fact_really_appears_in_the_expected_file(questions) -> None:
    """The facts are hand-written; this proves each one was read from the file it names."""
    cache = ROOT / ".codeatlas" / "eval-repos"
    clones = {p.name.split("-")[0]: p for p in cache.glob("*-*")} if cache.exists() else {}
    if not clones:
        pytest.skip("no cached eval clones; run scripts/structural_eval.py first")
    for q in questions:
        clone = clones.get(q["repo"])
        if clone is None:
            continue
        source = (clone / q["expected_file"]).read_text(encoding="utf-8", errors="replace").lower()
        assert q["fact"].lower() in source, f"{q['repo']}: {q['fact']!r} is not in {q['expected_file']}"


def test_the_check_is_not_wired_into_ci() -> None:
    workflows = (ROOT / ".github" / "workflows").glob("*.yml")
    for path in workflows:
        assert "answer_smoke" not in path.read_text(encoding="utf-8"), f"{path.name} runs the smoke check"


# ---------- resuming, and stopping cleanly ----------


def test_a_finished_run_is_identified_by_repo_question_and_mode() -> None:
    from answer_smoke import key_of

    row = {"repo": "flask", "question": "Which class represents the application context?", "mode": "deep"}
    assert key_of(row) == ("flask", "Which class represents the application context?", "deep")


def test_the_same_question_in_the_other_mode_is_a_different_run() -> None:
    from answer_smoke import key_of

    base = {"repo": "flask", "question": "q"}
    assert key_of({**base, "mode": "fast"}) != key_of({**base, "mode": "deep"})


def test_loading_results_from_a_missing_file_gives_nothing(tmp_path) -> None:
    from answer_smoke import load_done

    assert load_done(tmp_path / "nothing.json") == []


def test_a_half_written_results_file_does_not_crash_the_resume(tmp_path) -> None:
    """A run stopped mid-write must not stop the next one from starting."""
    from answer_smoke import load_done

    broken = tmp_path / "smoke.json"
    broken.write_text('[{"repo": "flask", "quest', encoding="utf-8")
    assert load_done(broken) == []


def test_finished_runs_are_read_back(tmp_path) -> None:
    from answer_smoke import key_of, load_done

    path = tmp_path / "smoke.json"
    rows = [{"repo": "flask", "question": "q1", "mode": "fast"}, {"repo": "flask", "question": "q1", "mode": "deep"}]
    path.write_text(json.dumps(rows), encoding="utf-8")

    done = load_done(path)
    assert {key_of(r) for r in done} == {("flask", "q1", "fast"), ("flask", "q1", "deep")}


def test_resume_skips_exactly_the_finished_runs() -> None:
    """The planning step: every question in both modes, minus what is already recorded."""
    from answer_smoke import key_of

    items = [{"repo": "flask", "question": "q1"}, {"repo": "flask", "question": "q2"}]
    done = [{"repo": "flask", "question": "q1", "mode": "fast"}]
    already = {key_of(r) for r in done}

    planned = [(mode, i) for mode in ("fast", "deep") for i in items]
    remaining = [(m, i) for m, i in planned if (i["repo"], i["question"], m) not in already]

    assert len(planned) == 4
    assert ("fast", items[0]) not in remaining
    assert len(remaining) == 3


def test_free_memory_is_readable_or_reported_as_unknown() -> None:
    from answer_smoke import free_megabytes

    free = free_megabytes()
    assert free == -1 or free > 0, "either a real figure, or -1 meaning it could not be read"


def test_an_unknown_memory_figure_never_stops_the_run() -> None:
    """-1 must not be read as 'no memory left'; the guard only fires on a real low figure."""
    for free, min_free, should_stop in [(-1, 600, False), (5000, 600, False), (100, 600, True), (600, 600, False)]:
        assert (0 <= free < min_free) is should_stop
