from dataclasses import dataclass

from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.interfaces import CodeRetriever


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_record_id: str


@dataclass(frozen=True)
class EvalResult:
    total: int
    hits: int
    accuracy: float


def evaluate_retrieval(
    repo_id: str,
    retriever: CodeRetriever,
    embedder: EmbeddingService,
    queries: list[EvalQuery],
    top_k: int = 5,
) -> EvalResult:
    hits = 0
    for item in queries:
        vector = embedder.embed_query(item.query)
        records = retriever.search(repo_id, vector, top_k)
        record_ids = {record.record_id for record in records}
        if item.expected_record_id in record_ids:
            hits += 1
    total = len(queries)
    accuracy = hits / total if total else 0.0
    return EvalResult(total=total, hits=hits, accuracy=accuracy)
