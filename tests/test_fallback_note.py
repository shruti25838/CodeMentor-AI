"""The sentence that tells a visitor their answer came from keyword search, end to end.

When the semantic index cannot answer - the day's allowance is gone, the provider is in a
cooldown, the API did not respond, or this repository has no semantic index at all - the
question is answered from the hash index instead. The answer is still an answer, and the only
thing that distinguishes it from a semantic one, from where the visitor is sitting, is one
sentence in the `done` event's `retrieval_note`.

It is easy to lose. The default path lost it for as long as it rebuilt its result object field
by field, which is a thing a controller does for ordinary reasons and which no test noticed,
because every test of the note stopped at `AnswerService.retrieve_with_note` and every test of
the endpoint used an index that could not fail. These tests go the whole way: a real dual
index over a real file, a semantic embedder that is unavailable in each of the documented
ways, and the note read out of the JSON the browser would receive.

**All three repository paths**, because they assemble the `done` event in three different
places: the default mode, grounded mode and deep mode.

Every model here is a fake and both embedders are fakes; nothing reaches a provider.
"""

import json
from dataclasses import replace  # noqa: E402  - grouped with the stdlib imports above

import pytest
from test_dual_index import FailingEmbedder, _repo
from test_embedder_identity import FakeEmbedder
from test_grounded_chat import CONFIG, FakeProvider

from codeatlas.app.di import (
    get_agent_orchestrator,
    get_conversation_store,
    get_llm_provider,
    get_repo_state_store,
    get_retrieval_service,
)
from codeatlas.app.main import create_app
from codeatlas.services.agents.coding_mentor_agent import CodingMentorAgent
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.agents.retrieval_agent import RetrievalAgent
from codeatlas.services.memory.conversation import ConversationStore
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.qa.grounded_chat import GROUNDED, LEGACY
from codeatlas.services.retrieval.dual_index import FALLBACK_NAME
from codeatlas.services.retrieval.embedding import REASONS, fallback_note
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from tests.test_conversation_memory import RecordingModel

REPO = "r"
QUESTION = "how is a value signed"

# Every reason the visitor-facing note knows. Each produces one sentence; "unindexed" and
# "cap" share theirs, which is deliberate - from the visitor's side they are one situation.
REASONS_IN_ORDER = sorted(REASONS)

# The reasons an embedder reports by raising; the rest come from there being no usable
# semantic index at all, which the service decides before it calls anything.
RAISED_BY_THE_EMBEDDER = ("api", "budget", "cap", "circuit")


class _StateStore:
    """Knows one repository, so the unknown-repo guard lets these questions through."""

    def get(self, repo_id):
        return object() if repo_id == REPO else None

    def has(self, repo_id):
        return repo_id == REPO


def _service(tmp_path, reason: str | None) -> AnswerService:
    """A dual index over one real file, with the semantic half failing in the way asked for.

    `reason=None` builds the ordinary case: the semantic index answers and there is no note.
    """
    repo, parsed = _repo(tmp_path)
    fallback_retriever = FaissCodeRetriever()
    CodeIndexService(
        embedder=HashEmbeddingService(), retriever=fallback_retriever, name=FALLBACK_NAME
    ).index_repository(repo, parsed)
    semantic_retriever = FaissCodeRetriever()
    CodeIndexService(
        embedder=FakeEmbedder(name="semantic", value=0.5), retriever=semantic_retriever, name="cloudflare"
    ).index_repository(repo, parsed)

    if reason is None:
        embedder = FakeEmbedder(name="semantic", value=0.5)
    elif reason in RAISED_BY_THE_EMBEDDER:
        # The index is this embedder's, so the service chooses it and only then finds out.
        embedder = FailingEmbedder(reason=reason)
    else:
        # No usable semantic index for this repository: a different embedder built the one on
        # disk, so the service refuses it before calling anything. This is what a repository
        # over the document cap looks like from here too.
        embedder = FakeEmbedder(name="other", value=0.25)

    return AnswerService(
        retriever=semantic_retriever,
        embedder=embedder,
        llm=None,
        fallback_retriever=fallback_retriever,
        fallback_embedder=HashEmbeddingService(),
    )


def _ask(tmp_path, reason: str | None, mode: str, body: dict | None = None, store=None) -> dict:
    """Ask one question through the real endpoint and return the `done` event."""
    answers = _service(tmp_path, reason)
    llm = RecordingModel()
    mentor = CodingMentorAgent(llm=llm)
    orchestrator = AgentOrchestrator(
        planner=mentor,
        retrieval_agent=RetrievalAgent(answer_service=answers),
        analyst_agent=mentor,
        mentor_agent=mentor,
        memory_agent=mentor,
    )
    config = replace(CONFIG, chat_mode=GROUNDED if mode == "grounded" else LEGACY)
    app = create_app(config)
    app.dependency_overrides[get_llm_provider] = lambda: FakeProvider(llm)
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_retrieval_service] = lambda: answers
    app.dependency_overrides[get_repo_state_store] = _StateStore
    app.dependency_overrides[get_conversation_store] = lambda: store or ConversationStore()

    from fastapi.testclient import TestClient

    payload = {"repo_id": REPO, "question": QUESTION, **(body or {})}
    if mode == "deep":
        payload["mode"] = "deep"
    with TestClient(app) as client:
        response = client.post("/ask/stream", json=payload)
    assert response.status_code == 200, response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["type"] == "done", events[-1]
    return events[-1]


MODES = ("default", "grounded", "deep")


# ---------------------------------------------------------------- the note reaches the browser


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reason", REASONS_IN_ORDER)
def test_the_fallback_note_reaches_the_done_event(tmp_path, mode: str, reason: str) -> None:
    done = _ask(tmp_path, reason, mode)

    assert done["retrieval_note"] == fallback_note(reason), f"{mode}/{reason}"
    assert done["citations"], "the fallback index answered, so there is still something to cite"


@pytest.mark.parametrize("mode", MODES)
def test_no_note_when_the_semantic_index_answered(tmp_path, mode: str) -> None:
    done = _ask(tmp_path, None, mode)

    assert done["retrieval_note"] == ""
    assert done["citations"]


@pytest.mark.parametrize("reason", REASONS_IN_ORDER)
def test_every_note_is_the_sentence_the_visitor_reads_and_nothing_else(tmp_path, reason: str) -> None:
    """A reason is a label from a fixed list; the sentence must not leak the detail beside it.

    `FailingEmbedder` raises with "detail that must not be shown to anyone" as its message.
    """
    done = _ask(tmp_path, reason, "default")

    assert done["retrieval_note"] in REASONS.values()
    assert "detail that must not be shown" not in json.dumps(done)
    assert "cloudflare" not in json.dumps(done).lower()


# ---------------------------------------------------------------- a follow-up keeps it


def test_a_question_with_a_rewrite_call_still_carries_the_note(tmp_path) -> None:
    """The rewrite runs before the search, and the search is what produces the note. A path
    that rebuilt the result after it is where the note went missing in the first place."""
    store = ConversationStore()
    session = "s" * 32
    first = _ask(tmp_path, "budget", "default", body={"session_id": session}, store=store)
    assert first["retrieval_note"] == fallback_note("budget")

    follow_up = _ask(
        tmp_path,
        "budget",
        "default",
        body={"session_id": session, "question": "what about negative numbers?"},
        store=store,
    )

    assert any("rewritten as" in step for step in follow_up["reasoning_steps"]), "the rewrite ran"
    assert follow_up["timings_ms"]["rewrite_ran"] == 1
    assert follow_up["retrieval_note"] == fallback_note("budget")


def test_a_follow_up_with_no_fallback_still_has_no_note(tmp_path) -> None:
    store = ConversationStore()
    session = "t" * 32
    _ask(tmp_path, None, "default", body={"session_id": session}, store=store)
    follow_up = _ask(tmp_path, None, "default", body={"session_id": session, "question": "what about it?"}, store=store)

    assert follow_up["timings_ms"]["rewrite_ran"] == 1
    assert follow_up["retrieval_note"] == ""


# ---------------------------------------------------------------- the whole event


def test_the_done_event_of_a_fallback_answer_carries_no_question_text(tmp_path) -> None:
    marker = "private-marker-9f2b"
    done = _ask(tmp_path, "budget", "default", body={"question": f"{marker} {QUESTION}"})

    assert done["retrieval_note"] == fallback_note("budget")
    assert marker not in json.dumps(done["timings_ms"])
    assert marker not in json.dumps(done["retrieval_note"])


def test_the_two_paths_that_rebuild_their_result_agree_with_the_one_that_does_not(tmp_path) -> None:
    """Deep mode returns the orchestrator's own AnswerResult; the default mode rebuilds one to
    add its conversation notes. The note must survive that rebuild, so the two must agree."""
    default = _ask(tmp_path, "circuit", "default")
    deep = _ask(tmp_path, "circuit", "deep")

    assert default["retrieval_note"] == deep["retrieval_note"] == fallback_note("circuit")
