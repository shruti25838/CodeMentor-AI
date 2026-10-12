"""Five ways to arrange the same two retrievers, scored from one index.

`orderings.py` compares *orderings* of a single pool: permutations, so recall@20 is identical
under all three and only the ranking moves. This module is for the question BM25 raises, which
is different - a second retriever changes **which** records are in the pool, so recall can move
too, and an arrangement has to be described by both halves: how the pool is found, and how it is
ordered.

    name            pool                              order
    current         vector, `candidates` deep         keyword overlap   <- what the server does
    vector          vector, `candidates` deep         vector similarity
    bm25            BM25 over every record            BM25 score
    rrf             RRF(vector, BM25)                 the fused score
    rrf+keyword     RRF(vector, BM25)                 keyword overlap

`current` is built from exactly the settings the server runs, so it must reproduce the numbers
already in `docs/RESULTS.md` for the same split; a test asserts that it is the same object graph
rather than a reimplementation, and the experiment script prints both so a drift is visible.

The vector index is built once and every arrangement reads it, so comparing them costs no
embedding calls at all - which is what makes the Cloudflare half of the experiment free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.qa.answer_service import AnswerService, RetrievalSettings, _record_text, _without_stopwords
from codeatlas.services.retrieval.bm25 import Bm25Index, Bm25Settings
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.fusion import reciprocal_rank_fusion
from codeatlas.services.retrieval.interfaces import CodeRetriever

CURRENT = "current"
VECTOR = "vector"
BM25 = "bm25"
RRF = "rrf"
RRF_KEYWORD = "rrf+keyword"
ARRANGEMENTS = (CURRENT, BM25, VECTOR, RRF, RRF_KEYWORD)

# What each one is, for the results file and the Benchmarks page. Written here so the
# description and the implementation cannot drift apart.
DESCRIPTIONS = {
    CURRENT: "vector candidates, ordered by keyword overlap (what the server does today)",
    BM25: "BM25 over every record, ordered by BM25 score",
    VECTOR: "vector candidates, ordered by vector similarity",
    RRF: "reciprocal rank fusion of the vector and BM25 rankings",
    RRF_KEYWORD: "reciprocal rank fusion of vector and BM25, then ordered by keyword overlap",
}


@dataclass(frozen=True)
class Arrangements:
    """Everything the five arrangements need, built once per index."""

    keyword: AnswerService
    vector: AnswerService
    fused: AnswerService
    bm25: Bm25Index
    settings: RetrievalSettings

    @classmethod
    def build(
        cls,
        retriever: CodeRetriever,
        embedder: EmbeddingService,
        settings: RetrievalSettings,
        repo_id: str,
    ) -> Arrangements:
        records = list(retriever.records(repo_id))
        return cls(
            keyword=AnswerService(
                retriever=retriever, embedder=embedder, llm=None, settings=replace(settings, rerank_weight=1.0)
            ),
            vector=AnswerService(
                retriever=retriever, embedder=embedder, llm=None, settings=replace(settings, rerank_weight=0.0)
            ),
            fused=AnswerService(
                retriever=retriever,
                embedder=embedder,
                llm=None,
                settings=replace(settings, rerank_weight=1.0, bm25_enabled=True),
            ),
            bm25=Bm25Index(
                records=records,
                texts=[_record_text(record) for record in records],
                settings=Bm25Settings(k1=settings.bm25_k1, b=settings.bm25_b),
            ),
            settings=settings,
        )


def arrange(name: str, built: Arrangements, repo_id: str, question: str, depth: int) -> list[EmbeddingRecord]:
    """One question's records under one arrangement, deepest first, to `depth`."""
    if name == CURRENT:
        return built.keyword.retrieve(repo_id, question, top_k=depth)
    if name == VECTOR:
        return built.vector.retrieve(repo_id, question, top_k=depth)
    if name == BM25:
        # The same question text the embedder is given: the service drops stopwords before it
        # embeds, and BM25's own tokeniser drops them again, so this only matters for being
        # able to say the two retrievers read the same string.
        text = _without_stopwords(question) if built.settings.drop_stopwords else question
        return built.bm25.search(text, depth)
    if name == RRF:
        text = _without_stopwords(question) if built.settings.drop_stopwords else question
        return reciprocal_rank_fusion(
            [built.vector.retrieve(repo_id, question, top_k=depth), built.bm25.search(text, depth)],
            k=built.settings.bm25_rrf_k,
        )[:depth]
    if name == RRF_KEYWORD:
        return built.fused.retrieve(repo_id, question, top_k=depth)
    raise ValueError(f"Unknown arrangement {name!r}; expected one of {', '.join(ARRANGEMENTS)}")
