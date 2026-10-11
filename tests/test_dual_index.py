"""Two indexes per repository, and the rule for which one answers a question.

The provider that can fail is faked here rather than reached: these tests are about what
happens *around* a failure, which is the part a visitor experiences.
"""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_embedder_identity import FakeEmbedder

from codeatlas.app.di import (
    build_embedder,
    get_answer_service,
    get_code_retriever,
    get_fallback_embedder,
    get_fallback_retriever,
    get_index_service,
    index_options,
)
from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.retrieval.dual_index import FALLBACK_NAME, DualIndexService
from codeatlas.services.retrieval.embedding import REASONS, EmbeddingUnavailableError, fallback_note
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.state.repo_state_store import RepoState, RepoStateStore
from codeatlas.utils.config import load_config

QUESTION = "how is a cookie signed"


class FailingEmbedder(FakeEmbedder):
    """A semantic embedder that is unavailable, in whichever of the documented ways is asked for."""

    def __init__(self, reason: str = "budget", name: str = "semantic", fail_on_query: bool = True) -> None:
        super().__init__(name=name, value=0.5)
        self._reason = reason
        self._fail_on_query = fail_on_query

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailableError(self._reason, "detail that must not be shown to anyone")

    def embed_query(self, text: str) -> list[float]:
        if self._fail_on_query:
            raise EmbeddingUnavailableError(self._reason, "detail that must not be shown to anyone")
        return super().embed_query(text)


def _repo(tmp_path: Path, functions: int = 0) -> tuple[Repository, ParsedRepository]:
    source = tmp_path / "mod.py"
    source.write_text("def sign(value):\n    return value\n", encoding="utf-8")
    repo = Repository("r", "r", "u", str(tmp_path), datetime.now(UTC))
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(source), language="python", size_bytes=source.stat().st_size)],
        functions=[
            FunctionNode(name=f"f{i}", file_path=str(source), start_line=1, end_line=2, signature="()")
            for i in range(functions)
        ],
    )
    return repo, parsed


def _dual(tmp_path: Path, semantic_embedder=None, max_documents: int = 600) -> tuple[DualIndexService, dict]:
    semantic_retriever, fallback_retriever = FaissCodeRetriever(), FaissCodeRetriever()
    semantic_embedder = semantic_embedder or FakeEmbedder(name="semantic", value=0.5)
    fallback_embedder = HashEmbeddingService()
    service = DualIndexService(
        semantic=CodeIndexService(embedder=semantic_embedder, retriever=semantic_retriever, name="cloudflare"),
        fallback=CodeIndexService(embedder=fallback_embedder, retriever=fallback_retriever, name=FALLBACK_NAME),
        semantic_name="cloudflare",
        max_documents=max_documents,
    )
    parts = {
        "semantic_retriever": semantic_retriever,
        "fallback_retriever": fallback_retriever,
        "semantic_embedder": semantic_embedder,
        "fallback_embedder": fallback_embedder,
    }
    return service, parts


# --- indexing --------------------------------------------------------------------------


def test_both_indexes_are_built_and_named(tmp_path) -> None:
    service, parts = _dual(tmp_path)
    assert service.index_repository(*_repo(tmp_path)) == ("cloudflare", "hash")
    assert parts["semantic_retriever"].has_index("r")
    assert parts["fallback_retriever"].has_index("r")


def test_a_semantic_failure_leaves_a_working_repository(tmp_path) -> None:
    """The point of the second index: the repository is still usable when the API is not."""
    service, parts = _dual(tmp_path, semantic_embedder=FailingEmbedder())
    assert service.index_repository(*_repo(tmp_path)) == ("hash",)
    assert parts["fallback_retriever"].has_index("r")
    assert not parts["semantic_retriever"].has_index("r")


def test_a_repository_over_the_cap_is_not_sent_at_all(tmp_path) -> None:
    class Exploding(FakeEmbedder):
        def embed_texts(self, texts):
            raise AssertionError("an over-cap repository must not reach the semantic embedder")

    service, parts = _dual(tmp_path, semantic_embedder=Exploding(), max_documents=3)
    repo, parsed = _repo(tmp_path, functions=5)
    assert service.document_count(parsed) == 6
    assert service.over_cap(parsed)
    assert service.index_repository(repo, parsed) == ("hash",)
    assert parts["fallback_retriever"].has_index("r")


def test_the_cap_counts_documents_not_files(tmp_path) -> None:
    service, _ = _dual(tmp_path, max_documents=3)
    _, parsed = _repo(tmp_path, functions=5)
    assert len(parsed.files) == 1
    assert service.document_count(parsed) == 6


def test_the_index_format_covers_both_indexes(tmp_path) -> None:
    service, _ = _dual(tmp_path)
    other, _ = _dual(tmp_path, semantic_embedder=FakeEmbedder(name="different"))
    assert "hash=" in service.index_format
    assert service.index_format != other.index_format


def test_the_repository_is_answerable_when_the_fallback_is_there(tmp_path) -> None:
    service, _ = _dual(tmp_path, semantic_embedder=FailingEmbedder())
    service.index_repository(*_repo(tmp_path))
    assert service.has_index("r")
    assert not service.has_semantic_index("r")


def test_discarding_removes_both(tmp_path) -> None:
    service, parts = _dual(tmp_path)
    service.index_repository(*_repo(tmp_path))
    service.discard("r")
    assert not parts["semantic_retriever"].has_index("r")
    assert not parts["fallback_retriever"].has_index("r")


def test_a_semantic_failure_logs_its_reason_and_not_its_detail(tmp_path, caplog) -> None:
    caplog.set_level("DEBUG")
    service, _ = _dual(tmp_path, semantic_embedder=FailingEmbedder(reason="budget"))
    service.index_repository(*_repo(tmp_path))
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "budget" in logged
    assert "detail that must not be shown to anyone" not in logged


# --- which index answers a question ------------------------------------------------------


def _answer_service(tmp_path, query_embedder=None, index_semantic: bool = True) -> AnswerService:
    """Both indexes built by embedders that worked, then questions asked with `query_embedder`.

    Splitting the two is the whole point: the failures that matter here happen once the index
    exists and a visitor is waiting, not while it is being built.
    """
    service, parts = _dual(tmp_path)
    if index_semantic:
        service.index_repository(*_repo(tmp_path))
    else:
        # Only the fallback, as an over-cap repository would have.
        CodeIndexService(
            embedder=parts["fallback_embedder"], retriever=parts["fallback_retriever"], name=FALLBACK_NAME
        ).index_repository(*_repo(tmp_path))
    return AnswerService(
        retriever=parts["semantic_retriever"],
        embedder=query_embedder or parts["semantic_embedder"],
        fallback_retriever=parts["fallback_retriever"],
        fallback_embedder=parts["fallback_embedder"],
    )


def test_the_semantic_index_answers_when_it_can_and_says_nothing(tmp_path) -> None:
    records, note = _answer_service(tmp_path).retrieve_with_note("r", QUESTION)
    assert records
    assert note == ""


@pytest.mark.parametrize("reason", ["budget", "circuit", "api"])
def test_a_failure_at_question_time_falls_back_and_tells_the_visitor(tmp_path, reason) -> None:
    service = _answer_service(tmp_path, query_embedder=FailingEmbedder(reason=reason))
    records, note = service.retrieve_with_note("r", QUESTION)
    assert records, "the fallback index must still answer"
    assert note == REASONS[reason]


def test_a_repository_with_only_the_fallback_index_says_so(tmp_path) -> None:
    records, note = _answer_service(tmp_path, index_semantic=False).retrieve_with_note("r", QUESTION)
    assert records
    assert note == REASONS["unindexed"]


def test_a_semantic_index_built_by_a_different_embedder_falls_back(tmp_path) -> None:
    """A mismatch used to mean no answer at all; with a fallback it means the other index."""
    service, parts = _dual(tmp_path)
    service.index_repository(*_repo(tmp_path))
    answer_service = AnswerService(
        retriever=parts["semantic_retriever"],
        embedder=FakeEmbedder(name="a different semantic embedder"),
        fallback_retriever=parts["fallback_retriever"],
        fallback_embedder=parts["fallback_embedder"],
    )
    records, note = answer_service.retrieve_with_note("r", QUESTION)
    assert records
    assert note == REASONS["unindexed"]


def test_the_note_reaches_the_answer(tmp_path) -> None:
    service = _answer_service(tmp_path, query_embedder=FailingEmbedder(reason="budget"))
    assert service.answer("r", QUESTION).note == REASONS["budget"]


def test_an_ordinary_answer_carries_no_note(tmp_path) -> None:
    assert _answer_service(tmp_path).answer("r", QUESTION).note == ""


def test_falling_back_logs_the_reason_but_never_the_question(tmp_path, caplog) -> None:
    caplog.set_level("DEBUG")
    _answer_service(tmp_path, query_embedder=FailingEmbedder(reason="budget")).retrieve_with_note("r", QUESTION)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "budget" in logged
    assert QUESTION not in logged
    assert "detail that must not be shown to anyone" not in logged


def test_without_a_fallback_nothing_changes(tmp_path) -> None:
    """The hash default must behave exactly as it did: a mismatch still returns nothing."""
    service, parts = _dual(tmp_path)
    service.index_repository(*_repo(tmp_path))
    answer_service = AnswerService(
        retriever=parts["semantic_retriever"], embedder=FakeEmbedder(name="a different embedder")
    )
    records, note = answer_service.retrieve_with_note("r", QUESTION)
    assert records == []
    assert note == ""


def test_a_failure_with_no_usable_fallback_returns_nothing_rather_than_nonsense(tmp_path) -> None:
    service, parts = _dual(tmp_path, semantic_embedder=FailingEmbedder())
    service.index_repository(*_repo(tmp_path))
    answer_service = AnswerService(
        retriever=parts["semantic_retriever"],
        embedder=parts["semantic_embedder"],
        fallback_retriever=FaissCodeRetriever(),
        fallback_embedder=parts["fallback_embedder"],
    )
    assert answer_service.retrieve_with_note("r", QUESTION) == ([], "")


@pytest.mark.parametrize("reason", sorted(REASONS))
def test_every_note_is_one_plain_sentence(reason) -> None:
    note = fallback_note(reason)
    assert note.count(".") == 1 and len(note) < 120
    assert "keyword search" in note


# --- the repository records which indexes it has -----------------------------------------


def test_the_indexes_a_repository_has_are_saved_and_read_back(tmp_path) -> None:
    store = RepoStateStore(base_dir=str(tmp_path))
    _, parsed = _repo(tmp_path)
    store.save(
        "r",
        RepoState(parsed_repo=parsed, import_graph=_graph(), root_path=str(tmp_path), indexes=("cloudflare", "hash")),
    )
    assert RepoStateStore(base_dir=str(tmp_path)).get("r").indexes == ("cloudflare", "hash")


def test_state_written_before_this_existed_reads_as_unrecorded(tmp_path) -> None:
    """Empty is not the same as "no indexes", and must not be read as it."""
    store = RepoStateStore(base_dir=str(tmp_path))
    _, parsed = _repo(tmp_path)
    store.save("r", RepoState(parsed_repo=parsed, import_graph=_graph(), root_path=str(tmp_path)))
    path = tmp_path / "r.json"
    payload = path.read_text(encoding="utf-8").replace('"indexes": [],', "")
    path.write_text(payload, encoding="utf-8")
    assert RepoStateStore(base_dir=str(tmp_path)).get("r").indexes == ()


def _graph():
    import networkx as nx

    return nx.DiGraph()


# --- how the application wires it up -----------------------------------------------------


@pytest.fixture(autouse=True)
def _forget_cached_singletons():
    """di caches its singletons, and these tests change the provider between them."""
    for builder in (get_code_retriever, get_fallback_retriever, get_fallback_embedder, get_index_service):
        builder.cache_clear()
    yield
    for builder in (get_code_retriever, get_fallback_retriever, get_fallback_embedder, get_index_service):
        builder.cache_clear()


def test_hash_gets_one_index_and_no_fallback(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_EMBEDDING_PROVIDER", raising=False)
    assert get_fallback_retriever() is None
    assert get_fallback_embedder() is None
    assert isinstance(get_index_service(), CodeIndexService)


def test_cloudflare_gets_two_indexes_in_two_folders(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CODEATLAS_EMBEDDING_PROVIDER", "cloudflare")
    monkeypatch.setenv("CODEATLAS_INDEX_DIR", str(tmp_path / "hash"))
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_INDEX_DIR", str(tmp_path / "semantic"))
    from codeatlas.app.di import get_config

    get_config.cache_clear()
    try:
        assert isinstance(get_index_service(), DualIndexService)
        assert isinstance(get_fallback_embedder(), HashEmbeddingService)
        assert (tmp_path / "hash").is_dir() and (tmp_path / "semantic").is_dir()
    finally:
        get_config.cache_clear()


def test_the_fallback_embedder_is_the_one_the_hash_index_was_built_with() -> None:
    config = replace(load_config(), embedding_provider="cloudflare")
    assert (
        build_embedder(replace(config, embedding_provider="hash")).signature()
        == HashEmbeddingService(lowercase=config.hash_lowercase, subtokens=config.hash_subtokens).signature()
    )


def test_the_answer_service_is_given_the_fallback_only_when_there_is_one(monkeypatch) -> None:
    """Built through the real factory, which means the real LlmProvider has to build something.

    CODEATLAS_LLM_PROVIDER=none is what makes that safe: the app answers with its own
    FallbackChatModel, which needs no key, constructs no client and reaches no network. The
    earlier version of this test left the provider at its default, so the factory built a
    ChatOpenAI - which worked on a machine with a .env and failed in CI, where there is none.
    """
    from codeatlas.app.di import get_config, get_llm_provider
    from codeatlas.services.llm.provider import FallbackChatModel

    monkeypatch.delenv("CODEATLAS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("CODEATLAS_LLM_PROVIDER", "none")
    for cached in (get_config, get_llm_provider, get_answer_service):
        cached.cache_clear()
    try:
        service = get_answer_service()
        # The provider wraps every model with the metrics callback, so unwrap before checking.
        assert isinstance(getattr(service._llm, "bound", service._llm), FallbackChatModel)
        assert service._fallback_retriever is None
        assert service._fallback_embedder is None
    finally:
        for cached in (get_config, get_llm_provider, get_answer_service):
            cached.cache_clear()


# --- the semantic index has its own time limit, and missing it is survivable ---------------


def test_the_cloudflare_provider_gets_its_own_index_time_limit() -> None:
    from codeatlas.app.di import index_options
    from codeatlas.services.retrieval.cloudflare_embedder import DEFAULT_CLOUDFLARE_INDEX_TIMEOUT_SECONDS

    config = replace(load_config(), embedding_provider="cloudflare")
    assert index_options(config)["timeout_seconds"] == DEFAULT_CLOUDFLARE_INDEX_TIMEOUT_SECONDS == 180
    assert index_options(replace(config, embedding_provider="hash"))["timeout_seconds"] == config.index_timeout_seconds


def test_the_index_time_limit_is_configurable_and_checked_at_startup(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_INDEX_TIMEOUT_SECONDS", "300")
    assert load_config().cloudflare_index_timeout_seconds == 300
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_INDEX_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError):
        load_config()


def test_that_limit_is_ignored_by_every_other_provider() -> None:
    base = replace(load_config(), embedding_provider="hash")
    assert index_options(base) == index_options(replace(base, cloudflare_index_timeout_seconds=1))


class SlowEmbedder(FakeEmbedder):
    """An embedder that takes longer than the index is allowed to take."""

    def __init__(self, clock) -> None:
        super().__init__(name="semantic", value=0.5)
        self._clock = clock

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._clock.advance(1000)
        return super().embed_texts(texts)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_a_semantic_index_that_runs_out_of_time_keeps_the_repository_working(tmp_path) -> None:
    """A timeout on the second index is a reason to stop building it, not to lose the analysis."""
    clock = FakeClock()
    semantic_retriever, fallback_retriever = FaissCodeRetriever(), FaissCodeRetriever()
    service = DualIndexService(
        semantic=CodeIndexService(
            embedder=SlowEmbedder(clock),
            retriever=semantic_retriever,
            name="cloudflare",
            timeout_seconds=180,
            clock=clock,
        ),
        fallback=CodeIndexService(embedder=HashEmbeddingService(), retriever=fallback_retriever, name=FALLBACK_NAME),
        semantic_name="cloudflare",
        max_documents=600,
    )
    assert service.index_repository(*_repo(tmp_path)) == ("hash",)
    assert fallback_retriever.has_index("r")
    assert not semantic_retriever.has_index("r")
    assert service.has_index("r")


def test_the_timeout_is_logged_as_an_outcome(tmp_path, caplog) -> None:
    from codeatlas.services.retrieval.dual_index import TIMEOUT

    caplog.set_level("DEBUG")
    clock = FakeClock()
    service = DualIndexService(
        semantic=CodeIndexService(
            embedder=SlowEmbedder(clock),
            retriever=FaissCodeRetriever(),
            name="cloudflare",
            timeout_seconds=180,
            clock=clock,
        ),
        fallback=CodeIndexService(embedder=HashEmbeddingService(), retriever=FaissCodeRetriever(), name=FALLBACK_NAME),
        semantic_name="cloudflare",
        max_documents=600,
    )
    service.index_repository(*_repo(tmp_path))
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert TIMEOUT in logged


def test_a_timeout_on_the_only_index_is_still_fatal(tmp_path) -> None:
    """Nothing here softens the single-index case: there a timeout means no index at all."""
    from codeatlas.services.retrieval.indexing import IndexingTimeoutError

    clock = FakeClock()
    service = CodeIndexService(
        embedder=SlowEmbedder(clock), retriever=FaissCodeRetriever(), timeout_seconds=180, clock=clock
    )
    with pytest.raises(IndexingTimeoutError):
        service.index_repository(*_repo(tmp_path))
