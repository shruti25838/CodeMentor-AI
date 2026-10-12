"""Reciprocal rank fusion: combine two rankings of the same records without a shared score scale.

This was written for `services/eval/orderings.py`, which fuses the vector order with the keyword
order to compare the three as *orderings* of one pool. It lives here now because the retrieval
path needs it too: BM25 is a second retriever, and fusing its list with the vector list changes
which records are in the pool, not just their order. Both callers use this one implementation,
so an ordering measured in an experiment is the same arithmetic the server would run.

Each list gives every record it holds `1 / (k + rank)` and the sums decide. `k` damps the top of
each list, so one ranking's confident first place cannot win on its own; 60 is the constant from
the original paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from codeatlas.models.embedding_record import EmbeddingRecord

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[EmbeddingRecord]], k: int = DEFAULT_RRF_K
) -> list[EmbeddingRecord]:
    """The fused ranking, best first.

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
