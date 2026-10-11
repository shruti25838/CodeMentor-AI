"""The evaluation vector cache, and the three orderings scored from one index.

Everything a cache does is invisible when it works, so what is tested here is the part that
would be expensive to get wrong: that a hit really makes no call, that a different
configuration really is a different cache, and that an ordering cannot invent recall.
"""

import json
from pathlib import Path

import pytest

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.eval.orderings import KEYWORD, ORDERINGS, RRF, VECTOR, fuse
from codeatlas.services.eval.vector_cache import CachingEmbeddingService, cache_key
from codeatlas.services.retrieval.embedding import EmbeddingService


class CountingEmbedder(EmbeddingService):
    """Records every text it is asked to embed, so a cache hit is visible as an absence."""

    def __init__(self, name: str = "counting") -> None:
        self.seen: list[str] = []
        self._name = name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(len(text))] * 4 for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def signature(self) -> str:
        return self._name


def _cache(tmp_path, inner=None, key: str = "model|pooling=mean|max_chars=2000"):
    return CachingEmbeddingService(inner=inner or CountingEmbedder(), key=key, cache_dir=tmp_path)


def test_a_text_is_embedded_once_and_then_remembered(tmp_path) -> None:
    inner = CountingEmbedder()
    cache = _cache(tmp_path, inner)
    first = cache.embed_texts(["def sign(): ...", "def verify(): ..."])
    second = cache.embed_texts(["def sign(): ...", "def verify(): ..."])
    assert first == second
    assert inner.seen == ["def sign(): ...", "def verify(): ..."]
    assert (cache.hits, cache.misses) == (2, 2)


def test_a_repeated_text_in_one_call_is_embedded_once(tmp_path) -> None:
    inner = CountingEmbedder()
    vectors = _cache(tmp_path, inner).embed_texts(["same", "same", "same"])
    assert inner.seen == ["same"]
    assert vectors == [vectors[0]] * 3


def test_the_cache_survives_a_new_process(tmp_path) -> None:
    """The point of putting it on disk: a rerun after a crash must not pay again."""
    first = CountingEmbedder()
    _cache(tmp_path, first).embed_texts(["paid for once"])
    second = CountingEmbedder()
    _cache(tmp_path, second).embed_texts(["paid for once"])
    assert second.seen == []


def test_a_different_configuration_is_a_different_cache(tmp_path) -> None:
    """A mean-pooled vector returned for a cls run would invent a result that was never measured."""
    mean = CountingEmbedder()
    _cache(tmp_path, mean, key=cache_key("m", "mean", 2000)).embed_texts(["text"])
    cls = CountingEmbedder()
    _cache(tmp_path, cls, key=cache_key("m", "cls", 2000)).embed_texts(["text"])
    assert cls.seen == ["text"]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (cache_key("m", "mean", 2000), cache_key("m", "mean", 1500)),
        (cache_key("m", "mean", 2000), cache_key("m", "cls", 2000)),
        (cache_key("a", "mean", 2000), cache_key("b", "mean", 2000)),
    ],
)
def test_model_pooling_and_clip_all_change_the_key(a, b) -> None:
    assert a != b


def test_the_cache_knows_what_is_already_paid_for(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.embed_texts(["known"])
    assert cache.known(["known", "new"]) == [True, False]


def test_a_query_goes_through_the_same_cache(tmp_path) -> None:
    inner = CountingEmbedder()
    cache = _cache(tmp_path, inner)
    cache.embed_query("how is a cookie signed")
    cache.embed_query("how is a cookie signed")
    assert inner.seen == ["how is a cookie signed"]


def test_the_cache_does_not_change_what_an_index_is(tmp_path) -> None:
    """A cached embedder must have the embedder's signature, or indexes would not match."""
    assert _cache(tmp_path, CountingEmbedder("the-real-one")).signature() == "the-real-one"


def test_the_file_holds_its_key_so_it_can_be_read_by_a_person(tmp_path) -> None:
    cache = _cache(tmp_path, key="model|pooling=cls|max_chars=1500")
    cache.embed_texts(["text"])
    written = json.loads(cache.path.read_text(encoding="utf-8"))
    assert written["key"] == "model|pooling=cls|max_chars=1500"
    assert len(written["vectors"]) == 1


def test_nothing_is_embedded_for_an_empty_list(tmp_path) -> None:
    inner = CountingEmbedder()
    assert _cache(tmp_path, inner).embed_texts([]) == []
    assert inner.seen == []


def test_a_broken_cache_file_is_ignored_rather_than_fatal(tmp_path) -> None:
    cache = _cache(tmp_path)
    cache.path.write_text("{not json", encoding="utf-8")
    assert _cache(tmp_path).embed_texts(["text"])


# --- orderings ---------------------------------------------------------------------------


def _record(record_id: str) -> EmbeddingRecord:
    return EmbeddingRecord(record_id=record_id, scope="file", vector=[], metadata={"path": record_id})


def test_fusion_puts_what_both_rankings_like_first() -> None:
    a = [_record("x"), _record("y"), _record("z")]
    b = [_record("z"), _record("x"), _record("y")]
    assert [r.record_id for r in fuse([a, b])] == ["x", "z", "y"]


def test_fusion_of_one_ranking_with_itself_changes_nothing() -> None:
    ranking = [_record(name) for name in ("a", "b", "c")]
    assert [r.record_id for r in fuse([ranking, ranking])] == ["a", "b", "c"]


def test_fusion_breaks_ties_the_same_way_every_time() -> None:
    """Two records with equal scores must not swap between runs and look like a difference."""
    a = [_record("p"), _record("q")]
    b = [_record("q"), _record("p")]
    assert [r.record_id for r in fuse([a, b])] == [r.record_id for r in fuse([a, b])] == ["p", "q"]


def test_fusion_keeps_every_record_from_both_rankings() -> None:
    fused = fuse([[_record("a"), _record("b")], [_record("c")]])
    assert {r.record_id for r in fused} == {"a", "b", "c"}


def test_an_ordering_cannot_add_a_record_the_pool_does_not_have() -> None:
    """Which is why recall@20 is the same under all three, and only the ranking metrics move."""
    pool = [_record(name) for name in ("a", "b", "c")]
    assert {r.record_id for r in fuse([pool, list(reversed(pool))])} == {r.record_id for r in pool}


def test_the_three_orderings_are_the_ones_named() -> None:
    assert ORDERINGS == (VECTOR, KEYWORD, RRF)


# --- the orderings over a real index -----------------------------------------------------


def test_the_orderings_read_one_pool_and_only_recall_stays_put(tmp_path) -> None:
    """The claim the experiment rests on: ordering moves ranking metrics, never recall."""
    from datetime import UTC, datetime

    from codeatlas.models.parsed_repository import ParsedRepository
    from codeatlas.models.repository import Repository
    from codeatlas.models.source_file import SourceFile
    from codeatlas.services.eval.basic_eval import RECALL_DEPTH, hit_at_k
    from codeatlas.services.eval.orderings import Rankers, order
    from codeatlas.services.qa.answer_service import RetrievalSettings
    from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
    from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
    from codeatlas.services.retrieval.indexing import CodeIndexService

    files = []
    for name, body in {
        "signing.py": "def sign(value):\n    return hmac_signature(value)\n",
        "templates.py": "def render(name):\n    return jinja_environment().get_template(name)\n",
        "routing.py": "def add_url_rule(rule):\n    return url_map.add(rule)\n",
    }.items():
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        files.append(SourceFile(path=str(path), language="python", size_bytes=path.stat().st_size))

    repo = Repository("r", "r", "u", str(tmp_path), datetime.now(UTC))
    parsed = ParsedRepository(repository_id="r", files=files, functions=[])
    retriever = FaissCodeRetriever()
    embedder = HashEmbeddingService(subtokens=True)
    CodeIndexService(embedder=embedder, retriever=retriever, prefix_metadata=True).index_repository(repo, parsed)

    settings = RetrievalSettings(candidates=RECALL_DEPTH, rerank_weight=1.0, drop_stopwords=True)
    rankers = Rankers.build(retriever, embedder, settings)
    expected = {"signing.py"}

    pools = {}
    for ordering in ORDERINGS:
        records = order(ordering, rankers, "r", "how is a value signed", RECALL_DEPTH)
        ranked = [Path(record.metadata["path"]).name for record in records]
        pools[ordering] = ranked

    # Same records, possibly in different orders: recall cannot move, so it is the fair
    # baseline the ordering comparison is read against.
    assert {tuple(sorted(pool)) for pool in pools.values()} == {tuple(sorted(pools[VECTOR]))}
    recalls = {hit_at_k(pool, expected, RECALL_DEPTH) for pool in pools.values()}
    assert len(recalls) == 1


def test_an_unknown_ordering_is_refused(tmp_path) -> None:
    from codeatlas.services.eval.orderings import Rankers, order
    from codeatlas.services.qa.answer_service import RetrievalSettings
    from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
    from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService

    rankers = Rankers.build(FaissCodeRetriever(), HashEmbeddingService(), RetrievalSettings())
    with pytest.raises(ValueError) as raised:
        order("alphabetical", rankers, "r", "q", 20)
    assert "alphabetical" in str(raised.value)
