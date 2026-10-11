"""Grounded chat: one search, one streamed model call, citations that are the prompt's snippets.

Every model here is a fake that yields chunks from a list, so nothing reaches a provider and
no credential is read. What is under test is the ordering the browser sees, that the citations
cannot drift from what the model was shown, and that each way a call can go wrong reaches the
visitor as a sentence rather than as an exception or an empty bubble.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from codeatlas.app.di import (
    get_agent_orchestrator,
    get_conversation_store,
    get_llm_provider,
    get_repo_state_store,
    get_retrieval_service,
)
from codeatlas.app.main import create_app
from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.llm.quota import DAILY, FILES_HEADING, MESSAGES, PER_MINUTE, TRUNCATED_NOTE
from codeatlas.services.memory.conversation import ConversationStore, Turn
from codeatlas.services.qa.answer_service import AnswerService, RetrievalSettings
from codeatlas.services.qa.grounded_chat import (
    CHAT_MODES,
    EMPTY_REPLY,
    GROUNDED,
    LEGACY,
    SYSTEM_PROMPT,
    build_messages,
    grounded_context,
    stream_tokens,
    validate_chat_mode,
)
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.utils.config import AppConfig

REPO = "repo-1"

CONFIG = AppConfig(
    embedding_provider="hash",
    embedding_model="",
    index_dir=".codeatlas/indexes",
    state_dir=".codeatlas/state",
    llm_provider="none",
    llm_model="",
    llm_temperature=0.2,
    api_key=None,
    auth_enabled=False,
    chat_mode=GROUNDED,
)


class Chunk:
    """What a streaming chat model yields. `content` is empty for a reasoning model's thinking."""

    def __init__(self, content: str = "", finish_reason: str | None = None) -> None:
        self.content = content
        self.response_metadata = {"finish_reason": finish_reason} if finish_reason else {}


class FakeStreamingModel:
    def __init__(self, chunks, error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.calls = 0
        self.messages = None

    def stream(self, messages):
        self.calls += 1
        self.messages = messages
        yield from self._chunks
        if self._error is not None:
            raise self._error


class _FakeOrchestrator:
    """Stands in for the agent graph. Legacy and deep mode land here; grounded never does."""

    def handle_question_fast(self, question, repo_id, history="", search_question=None):
        return _Result("legacy answer", [], ["legacy"], ["mentor"])

    def handle_question(self, question, repo_id, session_id=None):
        return _Result("legacy answer", [], ["legacy"], ["mentor"])

    def stream_question(self, question, repo_id, session_id=None, search_question=None):
        yield {"type": "agent", "name": "planner", "detail": "planned"}
        yield {"type": "result", "result": _Result("deep answer", [], ["deep"], ["planner"])}


class _Result:
    def __init__(self, answer, citations, reasoning_steps, agents_used, note=""):
        self.answer = answer
        self.citations = citations
        self.reasoning_steps = reasoning_steps
        self.agents_used = agents_used
        self.note = note


class FakeProvider:
    def __init__(self, model) -> None:
        self._model = model

    def get_chat_model(self):
        return self._model


@pytest.fixture
def indexed(tmp_path):
    """A real index over two small files, so retrieval is the real thing."""
    files = {
        "signer.py": "def sign(value):\n    return derive_key() + value\n",
        "timed.py": "def expired(age):\n    return age > max_age\n",
    }
    paths = {}
    functions = []
    sources = []
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        paths[name] = path
        sources.append(SourceFile(path=str(path), language="python", size_bytes=len(text)))
        functions.append(
            FunctionNode(name=name[:-3], file_path=str(path), start_line=1, end_line=2, signature=f"def {name[:-3]}():")
        )
    repository = Repository(
        repo_id=REPO, name="r", url="u", root_path=str(tmp_path), ingested_at=datetime.now(UTC), commit="c"
    )
    parsed = ParsedRepository(repository_id=REPO, files=sources, functions=functions)
    embedder = HashEmbeddingService(subtokens=True)
    retriever = FaissCodeRetriever()
    CodeIndexService(embedder=embedder, retriever=retriever).index_repository(repository, parsed)
    answers = AnswerService(
        retriever=retriever,
        embedder=embedder,
        llm=None,
        settings=RetrievalSettings(candidates=20, rerank_weight=1.0, drop_stopwords=True),
    )
    return answers, paths


# ---------------------------------------------------------------- the mode setting


def test_legacy_is_the_default_and_both_modes_are_known():
    assert CONFIG.chat_mode == GROUNDED  # this test module opts in
    assert (
        AppConfig(
            embedding_provider="hash",
            embedding_model="",
            index_dir="",
            state_dir="",
            llm_provider="none",
            llm_model="",
            llm_temperature=0.2,
            api_key=None,
            auth_enabled=False,
        ).chat_mode
        == LEGACY
    )
    assert set(CHAT_MODES) == {LEGACY, GROUNDED}


def test_an_unknown_chat_mode_is_rejected_by_name():
    with pytest.raises(ValueError) as raised:
        validate_chat_mode("agents")
    assert "agents" in str(raised.value)
    assert validate_chat_mode(" GROUNDED ") == GROUNDED


# ---------------------------------------------------------------- the prompt and the citations


def test_the_prompt_holds_the_snippets_with_their_paths_and_line_ranges(indexed):
    answers, _ = indexed
    context = grounded_context(answers, REPO, "how does it sign a value?", top_k=5, total_max_chars=6000)
    messages = build_messages("how does it sign a value?", context, history=[])

    assert messages[0].content == SYSTEM_PROMPT
    prompt = messages[-1].content
    assert "how does it sign a value?" in prompt
    for snippet in context.snippets:
        assert snippet.path in prompt
        assert str(snippet.start_line) in prompt


def test_the_citations_are_exactly_the_snippets_in_the_prompt(indexed):
    answers, _ = indexed
    context = grounded_context(answers, REPO, "sign a value", top_k=5, total_max_chars=6000)
    prompt = build_messages("sign a value", context, history=[])[-1].content

    assert len(context.citations) == len(context.snippets)
    for snippet, citation in zip(context.snippets, context.citations):
        assert snippet.location in citation
        assert snippet.location in prompt


def test_a_snippet_dropped_for_size_is_not_cited(indexed):
    """The cap is applied before the citations are made, so the two lists cannot diverge."""
    answers, _ = indexed
    big = grounded_context(answers, REPO, "sign a value", top_k=5, total_max_chars=6000)
    tiny = grounded_context(answers, REPO, "sign a value", top_k=5, total_max_chars=60)

    assert len(tiny.snippets) < len(big.snippets)
    assert len(tiny.citations) == len(tiny.snippets)
    prompt = build_messages("sign a value", tiny, history=[])[-1].content
    for citation in tiny.citations:
        assert citation.split(" | ")[0] in prompt


def test_the_session_history_is_in_the_prompt(indexed):
    answers, _ = indexed
    context = grounded_context(answers, REPO, "and the salt?", top_k=5, total_max_chars=6000)
    history = [Turn(question="how does it sign?", answer="with an HMAC")]

    messages = build_messages("and the salt?", context, history)
    assert [m.content for m in messages[1:3]] == ["how does it sign?", "with an HMAC"]
    assert messages[-1].content.startswith("and the salt?")


def test_a_question_with_no_matches_still_makes_a_prompt(indexed):
    answers, _ = indexed
    context = grounded_context(answers, "no-such-repo", "anything", top_k=5, total_max_chars=6000)
    assert context.snippets == []
    assert "no code snippets" in build_messages("anything", context, [])[-1].content


# ---------------------------------------------------------------- streaming


def _drain(model, context, **kwargs):
    tokens, outcome = [], None
    for token, finished in stream_tokens(model, [], context, **kwargs):
        if finished is not None:
            outcome = finished
        elif token:
            tokens.append(token)
    return tokens, outcome


@pytest.fixture
def context(indexed):
    answers, _ = indexed
    return grounded_context(answers, REPO, "sign a value", top_k=5, total_max_chars=6000)


def test_tokens_are_forwarded_one_at_a_time_in_order(context):
    model = FakeStreamingModel([Chunk("The "), Chunk("signer "), Chunk("works.")])
    tokens, outcome = _drain(model, context)

    assert tokens == ["The ", "signer ", "works."]
    assert outcome.answer == "The signer works."
    assert model.calls == 1


def test_a_reasoning_models_hidden_tokens_are_skipped_and_do_not_count_as_the_first(context):
    """The first visible token can come long after the call started; only it fires the mark."""
    marks = []
    model = FakeStreamingModel([Chunk(""), Chunk(""), Chunk(""), Chunk("Visible.")])
    tokens, outcome = _drain(model, context, on_first_token=lambda: marks.append("first"))

    assert tokens == ["Visible."]
    assert marks == ["first"]
    assert outcome.answer == "Visible."


def test_the_first_token_mark_fires_once(context):
    marks = []
    model = FakeStreamingModel([Chunk("a"), Chunk("b"), Chunk("c")])
    _drain(model, context, on_first_token=lambda: marks.append(1))
    assert marks == [1]


def test_an_empty_reply_is_explained_rather_than_left_blank(context):
    tokens, outcome = _drain(FakeStreamingModel([Chunk(""), Chunk("")]), context)

    assert EMPTY_REPLY in outcome.answer
    assert "".join(tokens) == outcome.answer
    # The files that were found are still offered.
    assert context.snippets[0].location in outcome.answer


def test_a_truncated_answer_says_so(context):
    model = FakeStreamingModel([Chunk("Half an answer"), Chunk("", finish_reason="length")])
    tokens, outcome = _drain(model, context)

    assert outcome.truncated is True
    assert outcome.answer.endswith(TRUNCATED_NOTE)
    assert TRUNCATED_NOTE in "".join(tokens)


def test_a_daily_quota_refusal_becomes_the_plain_message_with_the_files(context):
    model = FakeStreamingModel([], error=RuntimeError("429 tokens per day (TPD): Limit 200000"))
    tokens, outcome = _drain(model, context)

    assert outcome.quota == DAILY
    assert MESSAGES[DAILY] in outcome.answer
    assert FILES_HEADING in outcome.answer
    assert "429" not in outcome.answer, "the provider's own text must not reach the visitor"
    assert "".join(tokens) == outcome.answer


def test_a_per_minute_refusal_is_told_apart_from_a_daily_one(context):
    model = FakeStreamingModel([], error=RuntimeError("429 rate limit reached"))
    _, outcome = _drain(model, context)

    assert outcome.quota == PER_MINUTE
    assert MESSAGES[PER_MINUTE] in outcome.answer


def test_a_failure_part_way_through_keeps_what_was_already_said(context):
    model = FakeStreamingModel([Chunk("The signer ")], error=RuntimeError("connection reset"))
    tokens, outcome = _drain(model, context)

    assert outcome.failed is True
    assert outcome.answer.startswith("The signer")
    assert "connection reset" not in outcome.answer
    assert tokens[0] == "The signer "


# ---------------------------------------------------------------- through the endpoint


def _client(model, config=CONFIG, store=None):
    app = create_app(config)
    app.dependency_overrides[get_llm_provider] = lambda: FakeProvider(model)
    # The legacy path's orchestrator builds a provider client when it is constructed, and
    # grounded mode never touches it; it is replaced so that resolving the dependency does
    # not reach for a credential.
    app.dependency_overrides[get_agent_orchestrator] = lambda: _FakeOrchestrator()
    if store is not None:
        app.dependency_overrides[get_conversation_store] = lambda: store
    return app


def _events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def app_with(indexed):
    answers, _ = indexed

    def build(model, config=CONFIG, store=None):
        app = _client(model, config, store)
        app.dependency_overrides[get_retrieval_service] = lambda: answers
        state = _state_store()
        app.dependency_overrides[get_repo_state_store] = lambda: state
        return app

    return build


def _state_store():
    class _Store:
        def __init__(self):
            self._states = {REPO: object()}

        def get(self, repo_id):
            return self._states.get(repo_id)

        def has(self, repo_id):
            return repo_id in self._states

    return _Store()


def test_the_stream_sends_status_then_tokens_then_done(app_with):
    model = FakeStreamingModel([Chunk("It "), Chunk("signs.")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    events = _events(response)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "status"
    assert kinds[-1] == "done"
    assert [e["content"] for e in events if e["type"] == "token"] == ["It ", "signs."]
    assert kinds.index("token") > kinds.index("status")


def test_the_done_event_carries_the_citations_the_model_was_shown(app_with):
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    done = _events(response)[-1]
    assert done["citations"]
    prompt = model.messages[-1].content
    for citation in done["citations"]:
        assert citation.split(" | ")[0] in prompt
    assert done["agents_used"] == ["grounded"]


def test_the_question_cap_is_still_enforced(app_with):
    with TestClient(app_with(FakeStreamingModel([Chunk("ok")]))) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "x" * 4001})
    assert response.status_code == 422


def test_an_empty_question_is_still_refused(app_with):
    with TestClient(app_with(FakeStreamingModel([Chunk("ok")]))) as client:
        assert client.post("/ask/stream", json={"repo_id": REPO, "question": ""}).status_code == 422


def test_an_unknown_repository_is_still_a_404_before_the_stream_opens(app_with):
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": "nope", "question": "how does it sign?"})

    assert response.status_code == 404
    assert model.calls == 0


def test_the_answer_goes_into_the_session_history(app_with):
    store = ConversationStore()
    model = FakeStreamingModel([Chunk("It signs with an HMAC.")])
    session = "a" * 20
    with TestClient(app_with(model, store=store)) as client:
        client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?", "session_id": session})

    history = store.history(session, REPO)
    assert len(history) == 1
    assert history[0].answer == "It signs with an HMAC."


def test_a_refused_answer_is_not_remembered_as_one(app_with):
    store = ConversationStore()
    model = FakeStreamingModel([], error=RuntimeError("429 tokens per day (TPD)"))
    session = "b" * 20
    with TestClient(app_with(model, store=store)) as client:
        response = client.post(
            "/ask/stream", json={"repo_id": REPO, "question": "how does it sign?", "session_id": session}
        )

    assert MESSAGES[DAILY] in "".join(e.get("content", "") for e in _events(response) if e["type"] == "token")
    assert store.history(session, REPO) == []


def test_legacy_mode_does_not_take_the_grounded_path(app_with):
    """With the default setting the model is never streamed; the agent graph answers instead."""
    model = FakeStreamingModel([Chunk("grounded answer")])
    legacy = replace(CONFIG, chat_mode=LEGACY)
    with TestClient(app_with(model, config=legacy)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    tokens = "".join(e.get("content", "") for e in _events(response) if e["type"] == "token")
    assert "grounded answer" not in tokens


def test_deep_mode_is_unaffected_by_the_grounded_setting(app_with):
    model = FakeStreamingModel([Chunk("grounded answer")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?", "mode": "deep"})

    tokens = "".join(e.get("content", "") for e in _events(response) if e["type"] == "token")
    assert "grounded answer" not in tokens


def test_one_model_call_per_question(app_with):
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        client.post("/ask/stream", json={"repo_id": REPO, "question": "how does it sign?"})

    assert model.calls == 1


def test_no_question_text_reaches_the_timings_or_the_reasoning(app_with):
    marker = "private-marker-9f2b"
    model = FakeStreamingModel([Chunk("ok")])
    with TestClient(app_with(model)) as client:
        response = client.post("/ask/stream", json={"repo_id": REPO, "question": f"{marker} how does it sign?"})

    done = _events(response)[-1]
    assert marker not in json.dumps(done["timings_ms"])
    assert marker not in json.dumps(done["reasoning_steps"])
