"""An index must only ever be searched by the embedder that built it.

Two embedders' vectors are not comparable, and nothing makes that visible at search time: the
hash embedder and bge-small are both 384 wide, so FAISS answers a mismatched query with
confident nonsense instead of an error. These tests use fake embedders rather than the real
providers, so they run everywhere and the identity is the only thing under test.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_analyze_indexing import Env
from test_index_cache import CommitLoader, _analyze, _stages

from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService, embedder_mismatch


class FakeEmbedder(EmbeddingService):
    """A fixed-width embedder whose identity is whatever it is told, independent of its vectors."""

    def __init__(self, name: str = "fake", value: float = 1.0, width: int = 384) -> None:
        self._name = name
        self._vector = [value] + [0.0] * (width - 1)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector)

    def signature(self) -> str:
        return self._name


class ForgetfulRetriever(FaissCodeRetriever):
    """A retriever that keeps no embedder record, like one written before the guard existed."""

    def stored_embedder(self, repo_id: str) -> str | None:
        return None


def _repo(tmp_path: Path) -> tuple[Repository, ParsedRepository]:
    source = tmp_path / "mod.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    repo = Repository("r", "r", "u", str(tmp_path), datetime.now(UTC))
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(source), language="python", size_bytes=source.stat().st_size)],
        functions=[],
    )
    return repo, parsed


def _indexed(tmp_path: Path, embedder: EmbeddingService, retriever=None):
    retriever = retriever if retriever is not None else FaissCodeRetriever()
    service = CodeIndexService(embedder=embedder, retriever=retriever)
    service.index_repository(*_repo(tmp_path))
    return service, retriever


# --- the identity is written down with the index ---------------------------------------------


def test_the_index_records_which_embedder_built_it(tmp_path):
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"))
    assert retriever.stored_embedder("r") == "fake-a"


def test_an_unknown_repo_has_no_record_at_all(tmp_path):
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"))
    assert retriever.stored_embedder("never-indexed") is None


def test_the_record_survives_a_restart(tmp_path):
    _indexed(tmp_path / "repo", FakeEmbedder("fake-a"), FaissCodeRetriever(base_dir=str(tmp_path / "idx")))
    reopened = FaissCodeRetriever(base_dir=str(tmp_path / "idx"))
    assert reopened.stored_embedder("r") == "fake-a"


def test_the_real_hash_embedder_records_its_full_signature(tmp_path):
    embedder = HashEmbeddingService(subtokens=True)
    _, retriever = _indexed(tmp_path, embedder)
    assert retriever.stored_embedder("r") == embedder.signature()
    assert "subtokens=1" in retriever.stored_embedder("r")


# --- what counts as a mismatch ---------------------------------------------------------------


def test_a_different_embedder_is_a_mismatch():
    assert embedder_mismatch("fake-a", FakeEmbedder("fake-b")) is True


def test_the_same_embedder_is_not():
    assert embedder_mismatch("fake-a", FakeEmbedder("fake-a")) is False


@pytest.mark.parametrize("stored", [None, ""])
def test_no_record_is_not_treated_as_a_mismatch(stored):
    """None means the retriever does not track it; empty means the index predates the record.

    Neither is evidence of a different embedder, so neither blanks out the index.
    """
    assert embedder_mismatch(stored, FakeEmbedder("fake-a")) is False


def test_two_embedders_with_the_same_vectors_still_differ_by_identity(tmp_path):
    """Identity, not output: these two agree on every vector and still must not share an index."""
    a, b = FakeEmbedder("fake-a"), FakeEmbedder("fake-b")
    assert a.embed_query("x") == b.embed_query("x")
    service, retriever = _indexed(tmp_path, a)
    assert service.has_index("r") is True
    assert CodeIndexService(embedder=b, retriever=retriever).has_index("r") is False


def test_the_default_signature_separates_two_embedder_classes():
    """An embedder that does not override signature() still gets its class name, not a blank."""

    class OtherEmbedder(FakeEmbedder):
        signature = EmbeddingService.signature

    assert OtherEmbedder().signature() == "OtherEmbedder"
    assert embedder_mismatch(FakeEmbedder("fake-a").signature(), OtherEmbedder()) is True


# --- the index is not reused by a different embedder ------------------------------------------


def test_has_index_is_false_for_another_embedders_index(tmp_path):
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"))
    other = CodeIndexService(embedder=FakeEmbedder("fake-b"), retriever=retriever)
    assert retriever.has_index("r") is True
    assert other.has_index("r") is False


def test_an_unrecorded_index_is_still_reusable(tmp_path):
    """Existing deployments keep working; the guard applies to indexes written from now on."""
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"), ForgetfulRetriever())
    assert CodeIndexService(embedder=FakeEmbedder("fake-b"), retriever=retriever).has_index("r") is True


def test_the_analyze_cache_misses_when_the_embedder_changed(tmp_path):
    """End to end: /analyze-repo must not hand back an index built by a different embedder."""
    env = Env(tmp_path, embedder=FakeEmbedder("fake-a"), loader=CommitLoader(tmp_path / "repos"))
    first = _analyze(env)
    env.index = CodeIndexService(
        embedder=FakeEmbedder("fake-b"),
        retriever=FaissCodeRetriever(base_dir=str(env.index_dir)),
    )
    second = _analyze(env)
    assert second.json()["repository_id"] != first.json()["repository_id"]
    assert "clone" in _stages(second)
    # And the new embedder's own index is reused on the next request.
    assert _analyze(env).json()["repository_id"] == second.json()["repository_id"]


def test_switching_back_reuses_the_first_embedders_index(tmp_path):
    env = Env(tmp_path, embedder=FakeEmbedder("fake-a"), loader=CommitLoader(tmp_path / "repos"))
    first = _analyze(env).json()["repository_id"]
    env.index = CodeIndexService(FakeEmbedder("fake-b"), FaissCodeRetriever(base_dir=str(env.index_dir)))
    _analyze(env)
    env.index = CodeIndexService(FakeEmbedder("fake-a"), FaissCodeRetriever(base_dir=str(env.index_dir)))
    assert _analyze(env).json()["repository_id"] == first


# --- the question path refuses a mismatched index ---------------------------------------------


def test_a_question_is_not_answered_from_another_embedders_index(tmp_path):
    """The hole this closes: both are 384 wide, so FAISS would return records, not an error."""
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"))
    assert AnswerService(retriever=retriever, embedder=FakeEmbedder("fake-a"), llm=None).retrieve("r", "what")
    assert AnswerService(retriever=retriever, embedder=FakeEmbedder("fake-b"), llm=None).retrieve("r", "what") == []


def test_the_refusal_says_which_embedder_and_what_to_do(tmp_path, caplog):
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"))
    with caplog.at_level("WARNING"):
        AnswerService(retriever=retriever, embedder=FakeEmbedder("fake-b"), llm=None).retrieve("r", "what")
    message = caplog.text
    assert "fake-a" in message and "fake-b" in message
    assert "Analyze it again" in message


def test_a_question_against_an_unrecorded_index_is_still_answered(tmp_path):
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"), ForgetfulRetriever())
    assert AnswerService(retriever=retriever, embedder=FakeEmbedder("fake-b"), llm=None).retrieve("r", "what")


def test_an_unknown_repo_is_not_mistaken_for_a_mismatch(tmp_path):
    _, retriever = _indexed(tmp_path, FakeEmbedder("fake-a"))
    # No index at all: still empty, but through the normal search path, not the guard.
    assert AnswerService(retriever=retriever, embedder=FakeEmbedder("fake-a"), llm=None).retrieve("other", "q") == []
