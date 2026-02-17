from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.eval.basic_eval import EvalQuery, evaluate_retrieval
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


def test_evaluate_retrieval_hits() -> None:
    records = [
        EmbeddingRecord(
            record_id="a.py",
            scope="file",
            vector=[1.0],
            metadata={},
        )
    ]
    result = evaluate_retrieval(
        repo_id="repo",
        retriever=StubRetriever(records),
        embedder=StubEmbedder(),
        queries=[EvalQuery(query="a", expected_record_id="a.py")],
        top_k=1,
    )
    assert result.total == 1
    assert result.hits == 1
