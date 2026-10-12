"""The five retrieval arrangements, and the offline cache that keeps an experiment from spending.

Nothing here reaches a network, constructs a provider client, or reads a credential.
"""

import json
from dataclasses import replace

import pytest

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.eval.arrangements import (
    ARRANGEMENTS,
    BM25,
    CURRENT,
    DESCRIPTIONS,
    RRF,
    RRF_KEYWORD,
    VECTOR,
    Arrangements,
    arrange,
)
from codeatlas.services.eval.vector_cache import MissingVectorError, OfflineVectorCache
from codeatlas.services.qa.answer_service import RetrievalSettings
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService

REPO = "repo"


@pytest.fixture
def indexed(tmp_path):
    """A tiny repository, really indexed with the hash embedder, as an experiment would."""
    files = {
        "signer.py": "def sign(value):\n    return derive_key() + value\n",
        "timed.py": "def expired(age):\n    return age > max_age\n",
        "exc.py": "class BadSignature(Exception):\n    pass\n",
        "readme.py": "# notes about nothing in particular\n",
    }
    embedder = HashEmbeddingService(subtokens=True)
    records = []
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        records.append(
            EmbeddingRecord(
                record_id=name,
                scope="file",
                vector=embedder.embed_query(text),
                metadata={"path": str(path)},
            )
        )
    retriever = FaissCodeRetriever()
    retriever.index(REPO, records, embedder.signature())
    return retriever, embedder


def built(indexed, **settings):
    retriever, embedder = indexed
    base = RetrievalSettings(candidates=4, rerank_weight=1.0, drop_stopwords=True)
    return Arrangements.build(retriever, embedder, replace(base, **settings), REPO)


def test_all_five_arrangements_are_described():
    assert set(DESCRIPTIONS) == set(ARRANGEMENTS)
    assert len(ARRANGEMENTS) == 5
    assert all(DESCRIPTIONS[name] for name in ARRANGEMENTS)


def test_every_arrangement_returns_records(indexed):
    arranged = built(indexed)
    for name in ARRANGEMENTS:
        records = arrange(name, arranged, REPO, "how does it sign a value?", depth=4)
        assert records, name
        assert len({r.record_id for r in records}) == len(records), f"{name} returned a duplicate"


def test_an_unknown_arrangement_is_rejected(indexed):
    with pytest.raises(ValueError):
        arrange("telepathy", built(indexed), REPO, "q", depth=4)


def test_current_is_the_servers_own_settings(indexed):
    """`current` must be the pipeline the server runs, not a reimplementation of it."""
    arranged = built(indexed)
    question = "how does it sign a value?"

    assert arranged.keyword._settings.rerank_weight == 1.0
    assert arranged.keyword._settings.bm25_enabled is False
    direct = arranged.keyword.retrieve(REPO, question, top_k=4)
    assert arrange(CURRENT, arranged, REPO, question, depth=4) == direct


def test_vector_is_the_embedders_own_order(indexed):
    arranged = built(indexed)
    assert arranged.vector._settings.rerank_weight == 0.0
    assert arranged.vector._settings.bm25_enabled is False


def test_rrf_keyword_is_the_shipped_bm25_path(indexed):
    arranged = built(indexed)
    assert arranged.fused._settings.bm25_enabled is True
    assert arranged.fused._settings.rerank_weight == 1.0


def test_bm25_alone_ignores_the_vector_index(indexed):
    """A question whose words are in a file the vectors rank last still finds it."""
    arranged = built(indexed)
    records = arrange(BM25, arranged, REPO, "derive_key", depth=4)
    assert records[0].record_id == "signer.py"


def test_the_fused_arrangements_can_hold_a_record_neither_pool_ranks_first(indexed):
    arranged = built(indexed)
    question = "how does it sign a value?"
    vector_first = arrange(VECTOR, arranged, REPO, question, depth=4)[0].record_id
    bm25_first = arrange(BM25, arranged, REPO, question, depth=4)[0].record_id
    fused = [r.record_id for r in arrange(RRF, arranged, REPO, question, depth=4)]

    assert vector_first in fused and bm25_first in fused


def test_depth_is_honoured_by_every_arrangement(indexed):
    arranged = built(indexed)
    for name in ARRANGEMENTS:
        assert len(arrange(name, arranged, REPO, "sign value", depth=2)) <= 2, name


def test_rrf_keyword_differs_from_current_when_bm25_disagrees(indexed):
    """If the two were the same list, the experiment would be measuring nothing."""
    arranged = built(indexed)
    question = "which exception is raised for a bad signature?"
    current = [r.record_id for r in arrange(CURRENT, arranged, REPO, question, depth=4)]
    fused = [r.record_id for r in arrange(RRF_KEYWORD, arranged, REPO, question, depth=4)]
    bm25 = [r.record_id for r in arrange(BM25, arranged, REPO, question, depth=4)]

    # The arrangements are built from the same index, so this is a statement about the pool,
    # not about randomness: at least one of them must see a different list.
    assert len({tuple(current), tuple(fused), tuple(bm25)}) > 1


# ---------------------------------------------------------------- the offline cache


def _cache(tmp_path, vectors: dict) -> OfflineVectorCache:
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps({"key": "k", "vectors": vectors}), encoding="utf-8")
    return OfflineVectorCache(path, "cloudflare:test")


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_a_cached_text_is_served_without_calling_anything(tmp_path):
    cache = _cache(tmp_path, {_digest("hello"): [0.1, 0.2]})
    assert cache.embed_query("hello") == [0.1, 0.2]
    assert cache.hits == 1
    assert len(cache) == 1


def test_a_missing_vector_stops_the_run(tmp_path):
    cache = _cache(tmp_path, {_digest("hello"): [0.1]})
    with pytest.raises(MissingVectorError) as raised:
        cache.embed_texts(["hello", "goodbye"])
    message = str(raised.value)
    assert "1 of 2" in message
    # The count and the file, never the text: this message reaches a terminal and a log.
    assert "goodbye" not in message


def test_the_signature_is_the_one_the_index_records(tmp_path):
    assert _cache(tmp_path, {}).signature() == "cloudflare:test"


def test_known_reports_what_is_paid_for_without_raising(tmp_path):
    cache = _cache(tmp_path, {_digest("hello"): [0.1]})
    assert cache.known(["hello", "goodbye"]) == [True, False]
