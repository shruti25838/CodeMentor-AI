"""No field of the answer result may be lost between the search and the browser.

The default chat path runs its two halves separately -- `search` finds the code, `answer_from`
writes the answer -- and the controller then adds this conversation's notes to
`reasoning_steps`. It used to do that by building a fresh result and naming the fields it
wanted, which named four of the five: `note` was dropped, so the sentence that tells a visitor
their answer came from keyword search never reached the `done` event. That was fixed by
listing the fifth field; this file guards the shape of the mistake rather than that one field.

So nothing here mentions `note` in particular. The tests **enumerate `AnswerResult`'s fields**
and fail if any one of them does not arrive, and they fail just as loudly for a field that
does not exist yet: a sixth field added to the result type with nowhere to go in the event
breaks `test_every_field_of_the_result_type_is_accounted_for` until someone decides where it
belongs. A dropped field cannot be a silent one again.

Every model and every agent here is a fake. No provider is called and no credential is read.
"""

import json
from dataclasses import fields, replace
from types import SimpleNamespace

from test_analyze_indexing import Env

from codeatlas.app.di import get_agent_orchestrator, get_llm_provider
from codeatlas.controllers.ask_controller import _answer_from
from codeatlas.schemas.ask import AskRequest
from codeatlas.services.agents.types import AnswerResult
from codeatlas.services.qa.answer_service import GroundedAnswer
from codeatlas.services.retrieval.snippets import Snippet

QUESTION = "How does add work in pkg/core.py?"

SNIPPETS = [Snippet(path="pkg/core.py", start_line=1, end_line=2, text="def add(a, b):\n    return a + b")]

# One distinctive value per field of AnswerResult. Every one is non-empty and unlike every
# other, because a field whose marked value is the field's own default -- `note=""` -- is a
# field a drop cannot be seen through, which is exactly how the last one went missing.
MARKERS = {
    "answer": "ANSWER-MARKER-7c1d",
    "citations": ["pkg/core.py (lines 1-2) | CITATION-MARKER-2f90"],
    "reasoning_steps": ["STEP-MARKER-44ab"],
    "agents_used": ["AGENT-MARKER-91ce"],
    "note": "NOTE-MARKER-5d3f",
}

# Where the `done` event carries each field. `answer` is absent from it on purpose: it is the
# stream of `token` events that came before, so it is checked against those instead.
DONE_KEY = {
    "citations": "citations",
    "reasoning_steps": "reasoning_steps",
    "agents_used": "agents_used",
    "note": "retrieval_note",
}
STREAMED = "answer"


def marked_result() -> AnswerResult:
    return AnswerResult(**MARKERS)


class MarkedOrchestrator:
    """The default path's two halves: a search that finds something, and a marked answer."""

    def __init__(self) -> None:
        self.searched = 0
        self.answered = 0

    def search(self, question, repo_id):
        self.searched += 1
        return GroundedAnswer(
            answer="",
            citations=list(MARKERS["citations"]),
            reasoning_steps=[],
            snippets=list(SNIPPETS),
            note=MARKERS["note"],
        )

    def answer_from(self, question, retrieved, repo_id, history=""):
        self.answered += 1
        return marked_result()


class NoModel:
    """A provider whose model is never wanted: neither half of this path asks for one."""

    def get_chat_model(self):
        raise AssertionError("the marked path asked for a model")


def _ask(tmp_path) -> tuple[dict, str]:
    """One streamed question down the default path. Returns the done event and the tokens."""
    env = Env(tmp_path)
    overrides = env.client.app.dependency_overrides
    overrides[get_agent_orchestrator] = lambda: MarkedOrchestrator()
    overrides[get_llm_provider] = lambda: NoModel()
    repo_id = env.analyze().json()["repository_id"]

    response = env.client.post("/ask/stream", json={"question": QUESTION, "repo_id": repo_id})
    assert response.status_code == 200, response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done", events[-1]
    streamed = "".join(e["content"] for e in events if e["type"] == "token")
    return events[-1], streamed


# ---------- the enumeration itself ----------


def test_every_field_of_the_result_type_is_accounted_for():
    """The guard on the guard. A new field with no decision made about it fails here first."""
    declared = {f.name for f in fields(AnswerResult)}

    assert declared == set(MARKERS), (
        "AnswerResult's fields and this file's markers have drifted apart; a field with no "
        "marker is a field the tests below cannot notice the loss of"
    )
    assert set(DONE_KEY) | {STREAMED} == declared, (
        "every field either has a key in the done event or is the streamed answer; a new one "
        "needs a decision about where the browser receives it"
    )


# ---------- search() to the done event ----------


def test_no_field_is_lost_between_the_search_and_the_done_event(tmp_path):
    done, streamed = _ask(tmp_path)

    for field in fields(AnswerResult):
        marker = MARKERS[field.name]
        if field.name == STREAMED:
            assert marker in streamed, f"{field.name} never reached the browser"
            continue
        arrived = done[DONE_KEY[field.name]]
        if isinstance(marker, list):
            missing = [value for value in marker if value not in arrived]
            assert not missing, f"{field.name} lost {missing} on the way to the done event"
        else:
            assert arrived == marker, f"{field.name} arrived as {arrived!r}, not {marker!r}"


def test_the_done_event_carries_no_field_under_its_own_name_by_accident(tmp_path):
    """`note` travels as `retrieval_note`; a test reading `done["note"]` would pass on nothing."""
    done, _ = _ask(tmp_path)

    assert "note" not in done
    assert done["retrieval_note"] == MARKERS["note"]


def test_both_halves_ran_and_neither_asked_for_a_model(tmp_path):
    """Nothing above is true of a path that never searched: pins what was actually exercised."""
    env = Env(tmp_path)
    orchestrator = MarkedOrchestrator()
    overrides = env.client.app.dependency_overrides
    overrides[get_agent_orchestrator] = lambda: orchestrator
    overrides[get_llm_provider] = lambda: NoModel()
    repo_id = env.analyze().json()["repository_id"]

    env.client.post("/ask/stream", json={"question": QUESTION, "repo_id": repo_id})

    assert (orchestrator.searched, orchestrator.answered) == (1, 1)


# ---------- the controller's own copy ----------


def _answered(history=(), search_question=None) -> AnswerResult:
    orchestrator = SimpleNamespace(answer_from=lambda *a, **k: marked_result())
    request = AskRequest(question=QUESTION, repo_id="repo-1")
    return _answer_from(orchestrator, request, SimpleNamespace(snippets=SNIPPETS), list(history), search_question)


def test_the_controller_changes_reasoning_steps_and_copies_the_rest():
    """Stated as an equality rather than field by field, which is the mistake being guarded."""
    original = marked_result()

    returned = _answered()

    assert replace(returned, reasoning_steps=original.reasoning_steps) == original


def test_every_other_field_is_identical_to_what_the_orchestrator_returned():
    """The same claim enumerated, so a failure names the field that moved."""
    original = marked_result()

    returned = _answered()

    for field in fields(AnswerResult):
        if field.name == "reasoning_steps":
            continue
        assert getattr(returned, field.name) == getattr(original, field.name), field.name


def test_the_conversation_notes_are_added_without_displacing_the_orchestrators_steps():
    from codeatlas.services.memory.conversation import Turn

    returned = _answered(history=[Turn(question="earlier", answer="reply")], search_question="a standalone question")

    assert returned.reasoning_steps[: len(MARKERS["reasoning_steps"])] == MARKERS["reasoning_steps"]
    assert "Used 1 earlier turn(s) of this conversation." in returned.reasoning_steps
    assert "Searched with the follow-up rewritten as: a standalone question" in returned.reasoning_steps
