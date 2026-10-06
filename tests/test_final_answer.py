"""Which agent's output becomes the answer the visitor sees.

Every graph node used to overwrite `final_answer`, so the last step won. A plan of
retrieval → mentor → analyst showed the analyst's structural facts and threw the mentor's
answer away. The smoke check caught it indirectly: deep mode had the expected fact in only
8 of 15 answers against fast mode's 13 of 15, and four of those failures cited the right
file while missing only the prose, which is what a structural result looks like.

No model is involved anywhere here.
"""

import pytest
from test_agent_pipeline import FakeAgent, plan_of

from codeatlas.services.agents.orchestration import (
    RECALL_PREFIX,
    SEARCH_PREFIX,
    STRUCTURAL_PREFIX,
    AgentOrchestrator,
)

MENTOR = "The signer derives its key with django-concat."
ANALYST = "sign is called from 3 places"
MEMORY = "Earlier you asked about signing."
RETRIEVAL = "Answer: found it\n\nCitations:\n- a.py (lines 1-2) | code"


def build(*steps: tuple[str, str]) -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=FakeAgent("planner", plan_of(*steps)),
        retrieval_agent=FakeAgent("retrieval", RETRIEVAL),
        analyst_agent=FakeAgent("analyst", ANALYST),
        mentor_agent=FakeAgent("mentor", MENTOR),
        memory_agent=FakeAgent("memory", MEMORY),
    )


def ask(*steps: tuple[str, str]):
    return build(*steps).handle_question("How does Signer derive its key?", "repo")


# ---------- the four plan orders ----------


def test_retrieval_mentor_analyst_answers_with_the_mentor() -> None:
    """The regression: the analyst ran last and replaced the answer."""
    result = ask(("retrieval", "find"), ("mentor", "explain"), ("analyst", "summarise"))

    assert result.answer == MENTOR
    assert ANALYST not in result.answer
    assert result.agents_used == ["planner", "retrieval", "mentor", "analyst"]


def test_retrieval_analyst_mentor_answers_with_the_mentor() -> None:
    result = ask(("retrieval", "find"), ("analyst", "summarise"), ("mentor", "explain"))

    assert result.answer == MENTOR
    assert result.agents_used == ["planner", "retrieval", "analyst", "mentor"]


def test_analyst_only_is_labelled_a_structural_result() -> None:
    result = ask(("analyst", "who calls sign"))

    assert result.answer.startswith(STRUCTURAL_PREFIX)
    assert ANALYST in result.answer
    assert "structural result" in result.answer
    assert any("No mentor step" in step for step in result.reasoning_steps)


def test_two_mentor_steps_answer_with_the_later_one() -> None:
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan_of(("mentor", "first"), ("mentor", "second"))),
        retrieval_agent=FakeAgent("retrieval", RETRIEVAL),
        analyst_agent=FakeAgent("analyst", ANALYST),
        mentor_agent=TwoAnswers(),
        memory_agent=FakeAgent("memory", MEMORY),
    )
    result = orchestrator.handle_question("q", "repo")

    assert result.answer == "second answer"
    assert result.agents_used == ["planner", "mentor", "mentor"]


class TwoAnswers(FakeAgent):
    """A mentor whose second answer differs from its first."""

    def __init__(self) -> None:
        super().__init__("mentor")
        self.n = 0

    def run(self, prompt, repo_id=None, **kwargs):
        self.n += 1
        return "first answer" if self.n == 1 else "second answer"

    def answer(self, question, snippets, facts="", history=""):
        return self.run(question)


# ---------- the rule, stated directly ----------


@pytest.mark.parametrize(
    "steps,expected",
    [
        ((("mentor", "a"),), MENTOR),
        ((("mentor", "a"), ("memory", "b")), MENTOR),
        ((("memory", "a"), ("mentor", "b")), MENTOR),
        ((("retrieval", "a"), ("mentor", "b"), ("memory", "c")), MENTOR),
    ],
)
def test_a_plan_with_a_mentor_always_answers_with_the_mentor(steps, expected) -> None:
    assert ask(*steps).answer == expected


@pytest.mark.parametrize(
    "agent,output,prefix",
    [
        ("analyst", ANALYST, STRUCTURAL_PREFIX),
        ("memory", MEMORY, RECALL_PREFIX),
        ("retrieval", RETRIEVAL, SEARCH_PREFIX),
    ],
)
def test_a_plan_without_a_mentor_labels_the_output_for_what_it_is(agent: str, output: str, prefix: str) -> None:
    """Calling a file search or a recalled turn a "structural result" was simply wrong."""
    result = ask((agent, "do it"))

    assert result.answer.startswith(prefix)
    assert output.splitlines()[0] in result.answer


def test_the_three_labels_are_different_from_each_other() -> None:
    assert len({STRUCTURAL_PREFIX, SEARCH_PREFIX, RECALL_PREFIX}) == 3
    assert "structural" in STRUCTURAL_PREFIX
    assert "search result" in SEARCH_PREFIX
    assert "recalled context" in RECALL_PREFIX


def test_only_the_analyst_is_called_structural() -> None:
    assert not ask(("retrieval", "find")).answer.startswith(STRUCTURAL_PREFIX)
    assert not ask(("memory", "recall")).answer.startswith(STRUCTURAL_PREFIX)
    assert ask(("analyst", "facts")).answer.startswith(STRUCTURAL_PREFIX)


def test_a_structural_answer_is_not_silently_passed_off_as_prose() -> None:
    answer = ask(("analyst", "who calls sign")).answer
    assert answer != ANALYST, "the bare output must not stand alone"
    assert "computed from the code" in answer


# ---------- every agent's work is still visible ----------


def test_the_steps_of_every_agent_are_kept() -> None:
    result = ask(("retrieval", "find"), ("mentor", "explain"), ("analyst", "summarise"))
    steps = [s for s in result.reasoning_steps if s.startswith("Step ")]

    assert len(steps) == 3
    assert any(RETRIEVAL.splitlines()[0] in s for s in steps)
    assert any(MENTOR in s for s in steps)
    assert any(ANALYST in s for s in steps), "the analyst's work is shown, just not as the answer"


def test_a_later_step_is_noted_as_having_run_after_the_answer() -> None:
    result = ask(("mentor", "explain"), ("analyst", "summarise"))
    assert any("taken from the mentor step" in s for s in result.reasoning_steps)


def test_no_note_when_the_mentor_ran_last() -> None:
    result = ask(("retrieval", "find"), ("mentor", "explain"))
    assert not any("taken from the mentor step" in s for s in result.reasoning_steps)


def test_the_streamed_events_still_carry_every_agent() -> None:
    orchestrator = build(("retrieval", "find"), ("mentor", "explain"), ("analyst", "summarise"))
    events = list(orchestrator.stream_question("q", "repo"))

    agents = [e["name"] for e in events if e["type"] == "agent"]
    assert agents == ["planner", "retrieval", "mentor", "analyst"]

    final = events[-1]["result"]
    assert final.answer == MENTOR
    assert any(ANALYST in s for s in final.reasoning_steps), "the analyst's output still reaches the website"


# ---------- the fast path is unaffected ----------


def test_the_fast_path_still_answers_with_the_mentor() -> None:
    result = build(("retrieval", "find")).handle_question_fast("q", "repo")
    assert result.answer == MENTOR
    assert result.agents_used == ["retrieval", "mentor"]


def test_an_empty_run_still_says_so() -> None:
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", '{"steps": []}'),
        retrieval_agent=FakeAgent("retrieval", RETRIEVAL),
        analyst_agent=FakeAgent("analyst", ANALYST),
        mentor_agent=FakeAgent("mentor", MENTOR),
        memory_agent=FakeAgent("memory", MEMORY),
    )
    # An empty plan falls back to the default, which has a mentor.
    assert orchestrator.handle_question("q", "repo").answer == MENTOR


# ---------- retrieval searches with the visitor's question, not the planner's wording ----------


class RecordingRetrieval(FakeAgent):
    """Keeps every query it was asked to search for."""

    def __init__(self) -> None:
        super().__init__("retrieval", RETRIEVAL)
        self.queries: list[str] = []

    def retrieve(self, question, repo_id, top_k=5):
        from codeatlas.services.qa.answer_service import GroundedAnswer

        self.queries.append(question)
        return GroundedAnswer(answer="found", citations=["a.py (lines 1-2)"], reasoning_steps=[], snippets=[])


def search_queries_for(plan_json: str, question: str, search_question: str | None = None) -> list[str]:
    retrieval = RecordingRetrieval()
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan_json),
        retrieval_agent=retrieval,
        analyst_agent=FakeAgent("analyst", ANALYST),
        mentor_agent=FakeAgent("mentor", MENTOR),
        memory_agent=FakeAgent("memory", MEMORY),
    )
    orchestrator.handle_question(question, "repo", search_question=search_question)
    return retrieval.queries


QUESTION = "How does Signer derive its key?"


def test_two_planners_wording_the_step_differently_search_the_same() -> None:
    """The planner is a model call, so its wording varied run to run and moved the search."""
    first = search_queries_for(
        plan_of(("retrieval", "Locate the Signer class and its key derivation"), ("mentor", "explain")), QUESTION
    )
    second = search_queries_for(
        plan_of(("retrieval", "find where signing keys come from in this repo"), ("mentor", "explain")), QUESTION
    )

    assert first == second == [QUESTION]


def test_the_planners_instruction_never_reaches_the_search() -> None:
    queries = search_queries_for(plan_of(("retrieval", "SOME PLANNER WORDING"), ("mentor", "b")), QUESTION)
    assert "SOME PLANNER WORDING" not in queries[0]


def test_an_empty_instruction_does_not_change_the_search_either() -> None:
    assert search_queries_for(plan_of(("retrieval", "")), QUESTION) == [QUESTION]


def test_two_retrieval_steps_both_search_the_question() -> None:
    queries = search_queries_for(plan_of(("retrieval", "first wording"), ("retrieval", "second wording")), QUESTION)
    assert queries == [QUESTION, QUESTION]


def test_a_rewritten_follow_up_is_what_search_sees() -> None:
    """A follow-up like "why does it do that?" is useless as a query; the rewrite is used."""
    rewritten = "Why does Signer use a salt?"
    queries = search_queries_for(
        plan_of(("retrieval", "look into it"), ("mentor", "explain")),
        "why does it do that?",
        search_question=rewritten,
    )
    assert queries == [rewritten]


def test_the_planner_still_chooses_which_agents_run() -> None:
    """Only the query is taken out of the planner's hands; routing is still its job."""
    retrieval = RecordingRetrieval()
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan_of(("analyst", "structure"), ("mentor", "explain"))),
        retrieval_agent=retrieval,
        analyst_agent=FakeAgent("analyst", ANALYST),
        mentor_agent=FakeAgent("mentor", MENTOR),
        memory_agent=FakeAgent("memory", MEMORY),
    )
    result = orchestrator.handle_question(QUESTION, "repo")

    assert result.agents_used == ["planner", "analyst", "mentor"]
    assert retrieval.queries == [], "a plan without a retrieval step runs no search"
