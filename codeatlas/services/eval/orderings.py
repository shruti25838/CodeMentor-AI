"""Three ways to order the same candidates, scored from one index.

The embedder chooses which 20 records a question is answered from; the ordering decides which
of them the visitor actually sees. Those are separate questions and this module keeps them
separate: every ordering here reads the *same* pool, built once, so the comparison is of
orderings alone and **costs nothing extra** - no text is embedded twice.

* **vector** - the embedder's own similarity order (`rerank_weight` 0.0).
* **keyword** - overlap between the question's words and the record's text, which is what the
  server does today (`rerank_weight` 1.0).
* **rrf** - reciprocal rank fusion of the two, the standard combination of two rankings that
  needs no shared score scale: each list contributes 1 / (k + rank) and the sums decide.

One consequence is worth stating before any numbers are read: **recall@20 is identical under
all three**, because they are permutations of one pool. An ordering can move the right file up
or down; it cannot find a file the embedder missed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.qa.answer_service import AnswerService, RetrievalSettings
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.interfaces import CodeRetriever

VECTOR = "vector"
KEYWORD = "keyword"
RRF = "rrf"
ORDERINGS = (VECTOR, KEYWORD, RRF)

# The constant from the original reciprocal rank fusion paper. It damps the top of each list so
# that one ranking's confident first place cannot win on its own.
RRF_K = 60


@dataclass(frozen=True)
class Rankers:
    """The two pipelines an ordering is built from, each already holding the one index.

    Built once per configuration rather than per question, and kept as two objects rather than
    one whose settings are swapped, so nothing here reaches into another object's state.
    """

    vector: AnswerService
    keyword: AnswerService

    @classmethod
    def build(cls, retriever: CodeRetriever, embedder: EmbeddingService, settings: RetrievalSettings) -> Rankers:
        return cls(
            vector=AnswerService(
                retriever=retriever, embedder=embedder, llm=None, settings=replace(settings, rerank_weight=0.0)
            ),
            keyword=AnswerService(
                retriever=retriever, embedder=embedder, llm=None, settings=replace(settings, rerank_weight=1.0)
            ),
        )


def order(ordering: str, rankers: Rankers, repo_id: str, question: str, depth: int) -> list[EmbeddingRecord]:
    """One question's records under one ordering, from the index the rankers already hold."""
    if ordering == VECTOR:
        return rankers.vector.retrieve(repo_id, question, top_k=depth)
    if ordering == KEYWORD:
        return rankers.keyword.retrieve(repo_id, question, top_k=depth)
    if ordering == RRF:
        return fuse(
            [
                rankers.vector.retrieve(repo_id, question, top_k=depth),
                rankers.keyword.retrieve(repo_id, question, top_k=depth),
            ]
        )
    raise ValueError(f"Unknown ordering {ordering!r}; expected one of {', '.join(ORDERINGS)}")


def fuse(rankings: Sequence[Sequence[EmbeddingRecord]], k: int = RRF_K) -> list[EmbeddingRecord]:
    """Reciprocal rank fusion: each list gives every record 1 / (k + rank), and the sums order them.

    Ties keep the order of first appearance, so the result is the same on every run - two
    records with equal scores must not swap places between runs and look like a difference.
    """
    scores: dict[str, float] = {}
    records: dict[str, EmbeddingRecord] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for ranking in rankings:
        for rank, record in enumerate(ranking, start=1):
            key = record.record_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in records:
                records[key] = record
                first_seen[key] = position
                position += 1
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [records[key] for key in ordered]
