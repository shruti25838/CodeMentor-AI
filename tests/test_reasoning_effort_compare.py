"""The four-arm comparison summarises only the questions every arm finished.

`scripts/reasoning_effort_compare.py` is a measurement script and needs a key and a network to
run, so nothing here runs it. What is tested is the one piece of arithmetic that decides what a
reader ends up comparing: a run stopped by the token cap ends part way through a question, and
summarising over every row would then compare the arm that went first on six questions against
the others on five. That is not a slow arm or a fast arm, it is an artefact of where the run
stopped, and it happened on the run recorded in docs/RESULTS.md.
"""

from scripts.reasoning_effort_compare import ARMS, arm_name, complete_questions, summarise

FOUR_ARMS = len(ARMS)


def row(question_id: str, mode: str, effort: str, **overrides) -> dict:
    base = {
        "repo": "itsdangerous",
        "question_id": question_id,
        "chat_mode": mode,
        "effort": effort,
        "passed": True,
        "file_ok": True,
        "fact_ok": True,
        "first_token_seconds": 1.0,
        "total_seconds": 2.0,
        "model_calls": 1,
        "tokens": 100,
        "no_answer": False,
    }
    return {**base, **overrides}


def full_set(question_id: str, **overrides) -> list[dict]:
    return [row(question_id, mode, effort, **overrides) for mode, effort in ARMS]


def test_the_four_arms_are_the_two_modes_times_the_two_efforts() -> None:
    assert FOUR_ARMS == 4
    assert {arm_name(m, e) for m, e in ARMS} == {
        "legacy-unset",
        "legacy-low",
        "grounded-unset",
        "grounded-low",
    }


def test_a_question_every_arm_finished_is_complete() -> None:
    assert complete_questions(full_set("q1")) == ["q1"]


def test_a_question_the_cap_cut_short_is_not_complete() -> None:
    rows = full_set("q1") + [row("q2", *ARMS[0])]

    assert complete_questions(rows) == ["q1"]


def test_questions_keep_the_order_they_were_run_in() -> None:
    rows = full_set("q1") + full_set("q2") + full_set("q3")

    assert complete_questions(rows) == ["q1", "q2", "q3"]


def test_the_partial_question_is_left_out_of_every_arm_not_just_its_own() -> None:
    """The bug this guards: one extra row making one arm's median a different sample."""
    rows = full_set("q1") + [row("q2", *ARMS[0], total_seconds=99.0, tokens=9999)]

    first_arm = summarise(rows, *ARMS[0])
    assert first_arm["runs"] == 1
    assert first_arm["questions"] == ["q1"]
    assert first_arm["tokens_total"] == 100, "the cut-short run must not be charged to the arm"
    assert first_arm["total_seconds_max"] == 2.0


def test_every_arm_is_summarised_over_the_same_questions() -> None:
    rows = full_set("q1") + full_set("q2") + [row("q3", *ARMS[0])]

    sets = {arm_name(m, e): summarise(rows, m, e)["questions"] for m, e in ARMS}

    assert len(set(map(tuple, sets.values()))) == 1, sets
    assert next(iter(sets.values())) == ["q1", "q2"]


def test_the_range_is_the_lowest_and_the_highest_first_token() -> None:
    rows = []
    for n, seconds in enumerate([0.5, 1.5, 30.0], start=1):
        rows += [
            row(f"q{n}", mode, effort, first_token_seconds=seconds if (mode, effort) == ARMS[0] else 1.0)
            for mode, effort in ARMS
        ]

    summary = summarise(rows, *ARMS[0])

    assert summary["first_token_seconds_median"] == 1.5
    assert summary["first_token_seconds_min"] == 0.5
    assert summary["first_token_seconds_max"] == 30.0


def test_a_run_with_no_answer_is_counted_but_leaves_the_timing_alone() -> None:
    rows = full_set("q1") + [
        row("q2", mode, effort, no_answer=(mode, effort) == ARMS[0], first_token_seconds=50.0) for mode, effort in ARMS
    ]

    summary = summarise(rows, *ARMS[0])

    assert summary["runs"] == 2
    assert summary["answered"] == 1
    assert summary["no_answer"] == 1
    assert summary["first_token_seconds_max"] == 1.0, "an unanswered run has no first token to time"


def test_nothing_complete_gives_an_empty_arm_rather_than_a_crash() -> None:
    summary = summarise([row("q1", *ARMS[0])], *ARMS[0])

    assert summary["runs"] == 0
    assert summary["first_token_seconds_median"] is None
    assert summary["total_seconds_median"] is None
