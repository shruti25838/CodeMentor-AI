"""The status event sent the moment retrieval finishes, before the model is asked anything.

The wait a visitor sits through is mostly the model, and until now nothing was on the screen
during it but a sentence written before the search even started. This event carries the stage
name and the files that are about to be cited, so the page can say *Found 3 files* and name
them while the model is still thinking.

The rule the whole thing rests on: **the early file list is only sent where it is guaranteed
to equal the final citation list.** In grounded mode it is the same list - `grounded_context`
builds the snippets once and the event, the prompt and the `done` citations are all read from
it. Every other path sends a stage name and nothing else, because on those paths the citations
are not known until the model has spoken, and a file list that the answer then contradicts is
worse than a spinner.

Every model here is a fake; nothing reaches a provider and no credential is read.
"""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from codeatlas.app.di import get_retrieval_service
from codeatlas.services.llm.quota import DAILY, FILES_HEADING, MESSAGES, TRUNCATED_NOTE
from codeatlas.services.qa.grounded_chat import (
    LEGACY,
    STAGE_ANSWERING,
    STAGE_RETRIEVED,
    STAGE_RETRIEVING,
    STAGE_THINKING,
    GroundedContext,
    grounded_context,
    retrieved_event,
    stage_event,
)
from codeatlas.services.retrieval.snippets import Snippet
from tests.test_grounded_chat import (
    CONFIG,
    REPO,
    Chunk,
    FakeStreamingModel,
    _events,
    app_with,  # noqa: F401 - a fixture, used by name
    indexed,  # noqa: F401 - a fixture, used by name
)


class RunnableFakeModel(BaseChatModel):
    """A chat model that can sit in a `prompt | model` chain, for the no-repository path."""

    @property
    def _llm_type(self) -> str:
        return "runnable-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="A decorator wraps a function."))])


def _statuses(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == "status"]


def _stage(events: list[dict], stage: str) -> dict | None:
    return next((e for e in _statuses(events) if e.get("stage") == stage), None)


# ---------------------------------------------------------------- the event on its own


def test_the_event_names_its_stage_and_counts_the_files() -> None:
    context = GroundedContext(
        snippets=[Snippet("a/b.py", 10, 42, "code"), Snippet("c/d.py", 1, 5, "more")],
        citations=["a/b.py (lines 10-42)", "c/d.py (lines 1-5)"],
    )

    event = retrieved_event(context)

    assert event["type"] == "status"
    assert event["stage"] == STAGE_RETRIEVED
    assert event["count"] == 2
    assert event["content"] == "Found 2 files"
    assert event["files"] == [
        {"path": "a/b.py", "start_line": 10, "end_line": 42},
        {"path": "c/d.py", "start_line": 1, "end_line": 5},
    ]


def test_one_file_is_not_called_files() -> None:
    event = retrieved_event(GroundedContext(snippets=[Snippet("a.py", 1, 2, "x")], citations=["a.py"]))

    assert event["content"] == "Found 1 file"


def test_two_ranges_of_one_file_count_as_one_file_and_two_locations() -> None:
    """ "Found 2 files" would be a lie when both windows came out of the same module."""
    context = GroundedContext(
        snippets=[Snippet("a.py", 1, 20, "x"), Snippet("a.py", 40, 60, "y")],
        citations=["a.py (lines 1-20)", "a.py (lines 40-60)"],
    )

    event = retrieved_event(context)

    assert event["count"] == 1
    assert event["content"] == "Found 1 file"
    assert len(event["files"]) == 2


def test_a_search_that_found_nothing_says_so_rather_than_being_skipped() -> None:
    event = retrieved_event(GroundedContext(snippets=[], citations=[]))

    assert event["count"] == 0
    assert event["files"] == []
    assert event["content"] == "Found 0 files"


def test_a_stage_only_event_promises_nothing_about_the_files() -> None:
    event = stage_event(STAGE_RETRIEVING, "Retrieving context...")

    assert event == {"type": "status", "stage": STAGE_RETRIEVING, "content": "Retrieving context..."}
    assert "files" not in event and "count" not in event


def test_the_event_carries_only_paths_and_line_numbers(indexed) -> None:  # noqa: F811
    """No question text, anywhere in it: a response body is something a proxy may log."""
    answers, _ = indexed
    marker = "private-marker-9f2b"
    context = grounded_context(answers, REPO, f"{marker} how does it sign?", top_k=5, total_max_chars=6000)

    event = retrieved_event(context)

    assert marker not in json.dumps(event)
    for entry in event["files"]:
        assert set(entry) == {"path", "start_line", "end_line"}


# ---------------------------------------------------------------- order, through the endpoint


def test_the_order_is_status_then_text_then_done(app_with) -> None:  # noqa: F811
    model = FakeStreamingModel([Chunk("It "), Chunk("signs.")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    kinds = [e["type"] for e in _events(response)]

    assert kinds[0] == "status"
    assert kinds[-1] == "done"
    assert kinds.index("token") > max(i for i, k in enumerate(kinds) if k == "status")


def test_the_early_event_arrives_before_the_model_is_asked_anything(app_with) -> None:  # noqa: F811
    """It is sent between the search and the call, which is the only moment it is both
    knowable and useful."""
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    events = _events(response)
    retrieved = _stage(events, STAGE_RETRIEVED)

    assert retrieved is not None
    assert events.index(retrieved) < next(i for i, e in enumerate(events) if e["type"] == "token")
    assert _stage(events, STAGE_RETRIEVING) is not None
    assert events.index(_stage(events, STAGE_RETRIEVING)) < events.index(retrieved)


def test_the_early_files_are_the_final_citations(app_with) -> None:  # noqa: F811
    """The claim the feature rests on, asserted rather than described."""
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    events = _events(response)
    early = _stage(events, STAGE_RETRIEVED)
    done = events[-1]

    assert done["citations"], "the question has to find something for this test to mean anything"
    assert len(early["files"]) == len(done["citations"])
    for entry, citation in zip(early["files"], done["citations"]):
        assert citation.startswith(f"{entry['path']} (lines {entry['start_line']}-{entry['end_line']})")
    assert early["count"] == len({entry["path"] for entry in early["files"]})


def test_the_early_files_are_also_the_files_the_model_was_shown(app_with) -> None:  # noqa: F811
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    early = _stage(_events(response), STAGE_RETRIEVED)
    prompt = model.messages[-1].content

    for entry in early["files"]:
        assert entry["path"] in prompt


def test_no_question_text_reaches_any_status_event(app_with) -> None:  # noqa: F811
    marker = "private-marker-9f2b"
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": f"{marker} how does it sign?"})

    assert marker not in json.dumps(_statuses(_events(response)))


# ---------------------------------------------------------------- the ways it can end


def test_the_stream_still_ends_correctly_on_a_model_error(app_with) -> None:  # noqa: F811
    """The early event has already gone out; the refusal still reaches the visitor as words."""
    model = FakeStreamingModel([], error=RuntimeError("429 tokens per day (TPD): Limit 200000"))
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    events = _events(response)
    early = _stage(events, STAGE_RETRIEVED)

    assert early is not None and early["files"]
    assert events[-1]["type"] == "done"
    assert events[-1]["citations"]
    text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert MESSAGES[DAILY] in text
    assert FILES_HEADING in text
    assert "429" not in text


def test_a_truncated_answer_still_says_so_after_the_early_event(app_with) -> None:  # noqa: F811
    model = FakeStreamingModel([Chunk("Half an answer"), Chunk("", finish_reason="length")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    events = _events(response)
    assert _stage(events, STAGE_RETRIEVED) is not None
    assert TRUNCATED_NOTE in "".join(e.get("content", "") for e in events if e["type"] == "token")


def test_the_fallback_note_still_reaches_the_done_event(app_with, indexed) -> None:  # noqa: F811
    """The note comes from retrieval, which is the same call the early event is built from."""
    answers, _ = indexed
    model = FakeStreamingModel([Chunk("ok")])
    app = app_with(model)

    class _Noted:
        def retrieve_with_note(self, repo_id, question, top_k=5):
            records, _ = answers.retrieve_with_note(repo_id, question, top_k)
            return records, "Keyword search answered this one."

        def snippets_for(self, records, total_max_chars=None):
            return answers.snippets_for(records, total_max_chars)

    app.dependency_overrides[get_retrieval_service] = lambda: _Noted()
    with TestClient(app) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    events = _events(response)
    assert _stage(events, STAGE_RETRIEVED)["files"]
    assert events[-1]["retrieval_note"] == "Keyword search answered this one."


def test_an_unknown_repository_is_a_404_before_any_event_at_all(app_with) -> None:  # noqa: F811
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": "nope", "question": "how does it sign?"})

    assert response.status_code == 404
    assert _events(response) == [], "a 404 must not be a 200 with a status event in it"
    assert model.calls == 0


# ---------------------------------------------------------------- the other paths


@pytest.mark.parametrize("mode", ["fast", "deep"])
def test_a_path_that_cannot_promise_the_citations_sends_no_file_list(app_with, mode: str) -> None:  # noqa: F811
    """Legacy and deep retrieve inside the agent graph, after model calls of their own, so
    the citations are not knowable at the point this event would be sent."""
    model = FakeStreamingModel([Chunk("ok")])
    legacy = replace(CONFIG, chat_mode=LEGACY)
    with TestClient(app_with(model, config=legacy)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?", "mode": mode})

    events = _events(response)
    statuses = _statuses(events)

    assert statuses, "the visitor is still told the stage"
    assert all("files" not in e for e in statuses)
    assert _stage(events, STAGE_RETRIEVED) is None
    assert events[-1]["type"] == "done"


def test_every_status_event_names_a_stage(app_with) -> None:  # noqa: F811
    model = FakeStreamingModel([Chunk("ok")])
    legacy = replace(CONFIG, chat_mode=LEGACY)
    for config in (CONFIG, legacy):
        with TestClient(app_with(model, config=config)) as client:
            response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})
        for event in _statuses(_events(response)):
            assert event.get("stage"), event


def test_general_mode_has_no_retrieval_and_so_no_file_list(app_with) -> None:  # noqa: F811
    # A real Runnable, because general mode composes `prompt | model` and the streaming fake
    # the other tests use cannot stand in a chain.
    with TestClient(app_with(RunnableFakeModel())) as client:
        response = client.post("/ask/stream", json={"question": "what is a decorator?"})

    events = _events(response)
    assert _stage(events, STAGE_THINKING) is not None
    assert _stage(events, STAGE_RETRIEVED) is None
    assert events[-1]["type"] == "done"


def test_the_legacy_path_still_says_when_it_starts_answering(app_with) -> None:  # noqa: F811
    model = FakeStreamingModel([Chunk("ok")])
    legacy = replace(CONFIG, chat_mode=LEGACY)
    with TestClient(app_with(model, config=legacy)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    assert _stage(_events(response), STAGE_ANSWERING) is not None
