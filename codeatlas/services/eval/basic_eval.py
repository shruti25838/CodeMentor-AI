from collections.abc import Sequence
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


# ---------- ranking metrics over file paths ----------
#
# Each question lists the files that should be retrieved. The retriever returns records (whole
# files or single functions), best first; each record is mapped to its file, and a record is
# relevant if its file is expected. Two function records from the same file both count, because
# both go into the prompt.


def hit_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    """1.0 if any of the first k results is relevant, else 0.0."""
    return 1.0 if any(path in expected for path in ranked[:k]) else 0.0


def precision_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    """Relevant results among the first k, divided by k (missing results count as not relevant)."""
    return sum(1 for path in ranked[:k] if path in expected) / k


def reciprocal_rank(ranked: Sequence[str], expected: set[str]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none of the results is relevant."""
    for rank, path in enumerate(ranked, start=1):
        if path in expected:
            return 1.0 / rank
    return 0.0


# The candidate pool a question is reranked from, and so the depth recall is measured at.
# Recall at this depth asks "did the embedder find the right file at all", with the reranker's
# opinion taken out of it: the reranker can only reorder what the pool already contains.
RECALL_DEPTH = 20


@dataclass(frozen=True)
class RankingMetrics:
    questions: int
    hit_rate: dict[int, float]
    precision: dict[int, float]
    mrr: float
    # Share of questions whose expected file is anywhere in the candidate pool.
    recall: float = 0.0


def ranking_metrics(
    results: Sequence[tuple[Sequence[str], set[str]]],
    ks: Sequence[int] = (1, 3, 5),
    candidates: Sequence[tuple[Sequence[str], set[str]]] | None = None,
) -> RankingMetrics:
    """Average hit@k, precision@k and reciprocal rank over (ranked paths, expected paths) pairs.

    `candidates` is the same questions ranked to the full pool depth, and **only recall is taken
    from it**. It is a separate argument rather than a deeper `results` on purpose: the hit
    rates and MRR are defined over the ranked answer the pipeline actually returns, and handing
    them a longer list would quietly change every number this file has ever reported.
    """
    n = len(results)
    if n == 0:
        return RankingMetrics(questions=0, hit_rate=dict.fromkeys(ks, 0.0), precision=dict.fromkeys(ks, 0.0), mrr=0.0)
    pool = candidates if candidates is not None else results
    return RankingMetrics(
        questions=n,
        hit_rate={k: sum(hit_at_k(r, e, k) for r, e in results) / n for k in ks},
        precision={k: sum(precision_at_k(r, e, k) for r, e in results) / n for k in ks},
        mrr=sum(reciprocal_rank(r, e) for r, e in results) / n,
        recall=sum(hit_at_k(r, e, RECALL_DEPTH) for r, e in pool) / len(pool) if pool else 0.0,
    )
