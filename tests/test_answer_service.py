from pathlib import Path

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.interfaces import CodeRetriever


class StubRetriever(CodeRetriever):
    def __init__(self, records: list[EmbeddingRecord]) -> None:
        self._records = records

    def index(self, repo_id: str, records: list[EmbeddingRecord]) -> None:
        return None

    def search(self, repo_id: str, query_vector: list[float], top_k: int) -> list[EmbeddingRecord]:
        return self._records[:top_k]


class StubEmbedder(EmbeddingService):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


def test_answer_service_citations_include_line_ranges(tmp_path: Path) -> None:
    file_path = tmp_path / "a.py"
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")
    records = [
        EmbeddingRecord(
            record_id=f"{file_path}:1-2",
            scope="function",
            vector=[1.0],
            metadata={
                "path": str(file_path),
                "start_line": "1",
                "end_line": "2",
                "signature": "def foo():",
            },
        )
    ]
    service = AnswerService(
        retriever=StubRetriever(records),
        embedder=StubEmbedder(),
        llm=None,
    )
    result = service.answer(repo_id="repo", question="foo")
    assert result.citations
    assert "lines 1-2" in result.citations[0]
