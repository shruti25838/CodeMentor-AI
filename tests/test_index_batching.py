"""Documents are embedded shortest-first, and that must change the bill and nothing else.

A provider billed per batch charges every text in a call at the length of its longest member,
so batching texts of similar length together costs about half as much. The whole idea rests on
the vectors being unaffected, which was checked against the real API
(`scripts/index_cost_check.py --check-order`, minimum cosine 1.000000). What these tests pin is
the part that is this project's responsibility: that the sorting is actually done, that the
vectors come back attached to the right records, and that nothing about the index changes.
"""

from datetime import UTC, datetime

import pytest

from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService


class BatchRecorder(EmbeddingService):
    """Returns a vector that identifies its text, and remembers the batches it was given."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), float(hash(text) % 1000)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(hash(text) % 1000)]

    def signature(self) -> str:
        return "recorder"


def _service(embedder, **options) -> CodeIndexService:
    return CodeIndexService(embedder=embedder, retriever=FaissCodeRetriever(), **options)


DOCUMENTS = ["x" * 900, "y" * 10, "z" * 400, "w" * 50]


def test_documents_are_embedded_shortest_first() -> None:
    recorder = BatchRecorder()
    _service(recorder, batch_size=2).embed_documents(DOCUMENTS)
    assert recorder.batches == [["y" * 10, "w" * 50], ["z" * 400, "x" * 900]]


def test_batches_hold_texts_of_similar_length_which_is_the_entire_point() -> None:
    """Unsorted, the first batch pairs the longest text with the shortest and pays for both."""
    recorder = BatchRecorder()
    _service(recorder, batch_size=2, sort_by_length=False).embed_documents(DOCUMENTS)
    assert recorder.batches == [["x" * 900, "y" * 10], ["z" * 400, "w" * 50]]


def test_the_vectors_come_back_in_the_order_the_documents_were_given() -> None:
    """The bookkeeping that makes sorting invisible: a vector must stay with its own record."""
    vectors = _service(BatchRecorder(), batch_size=2).embed_documents(DOCUMENTS)
    assert [int(vector[0]) for vector in vectors] == [900, 10, 400, 50]


def test_sorting_and_not_sorting_give_each_document_the_same_vector() -> None:
    sorted_vectors = _service(BatchRecorder(), batch_size=2).embed_documents(DOCUMENTS)
    plain_vectors = _service(BatchRecorder(), batch_size=2, sort_by_length=False).embed_documents(DOCUMENTS)
    assert sorted_vectors == plain_vectors


def test_a_single_batch_is_still_sorted_but_costs_the_same() -> None:
    recorder = BatchRecorder()
    _service(recorder, batch_size=50).embed_documents(DOCUMENTS)
    assert len(recorder.batches) == 1
    assert [len(text) for text in recorder.batches[0]] == [10, 50, 400, 900]


def test_no_documents_make_no_batches() -> None:
    recorder = BatchRecorder()
    assert _service(recorder).embed_documents([]) == []
    assert recorder.batches == []


def test_the_time_limit_is_still_checked_between_batches() -> None:
    checks = []
    _service(BatchRecorder(), batch_size=2).embed_documents(DOCUMENTS, lambda: checks.append(1))
    assert len(checks) == 2


def test_a_deadline_that_fires_stops_the_embedding() -> None:
    def explode() -> None:
        raise TimeoutError("out of time")

    with pytest.raises(TimeoutError):
        _service(BatchRecorder(), batch_size=2).embed_documents(DOCUMENTS, explode)


@pytest.mark.parametrize("sort", [True, False])
def test_sorting_changes_neither_the_signature_nor_the_index_format(sort) -> None:
    """If it touched either, every existing index would be invalidated by a batching change."""
    service = _service(HashEmbeddingService(), max_chars=2000, prefix_metadata=True, sort_by_length=sort)
    assert service.index_format == "hash:384:lowercase=0:subtokens=0|max_chars=2000|prefix=1"


def test_an_indexed_repository_is_the_same_either_way(tmp_path) -> None:
    """End to end: the records and their vectors must not depend on the order they were sent in."""
    source = tmp_path / "mod.py"
    source.write_text("def sign(value):\n    return value\n" + "# padding\n" * 200, encoding="utf-8")
    repo = Repository("r", "r", "u", str(tmp_path), datetime.now(UTC))
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(source), language="python", size_bytes=source.stat().st_size)],
        functions=[FunctionNode(name="sign", file_path=str(source), start_line=1, end_line=2, signature="()")],
    )
    indexed = {}
    for sort in (True, False):
        retriever = FaissCodeRetriever()
        service = CodeIndexService(
            embedder=HashEmbeddingService(), retriever=retriever, prefix_metadata=True, sort_by_length=sort
        )
        service.index_repository(repo, parsed)
        records = retriever._indexes["r"].records
        indexed[sort] = [(record.record_id, record.vector) for record in records]
    assert indexed[True] == indexed[False]
