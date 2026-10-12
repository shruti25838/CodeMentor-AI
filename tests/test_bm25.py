"""BM25 as a second retriever: the scoring, the fusion, and the settings that switch it on.

Nothing here reaches a network or a model. The arithmetic is checked against what BM25 is
defined to do rather than against numbers this implementation happened to print.
"""

import math
from dataclasses import replace

import pytest

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.qa.answer_service import AnswerService, RetrievalSettings
from codeatlas.services.retrieval.bm25 import DEFAULT_B, DEFAULT_K1, Bm25Index, Bm25Settings, tokenize
from codeatlas.services.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from codeatlas.services.retrieval.interfaces import CodeRetriever


def record(record_id: str, path: str = "", **metadata) -> EmbeddingRecord:
    return EmbeddingRecord(
        record_id=record_id,
        scope="file",
        vector=[0.0],
        metadata={"path": path or record_id, **metadata},
    )


def index(*pairs, settings: Bm25Settings | None = None) -> Bm25Index:
    records = [record(name) for name, _ in pairs]
    return Bm25Index(records=records, texts=[text for _, text in pairs], settings=settings)


# ---------------------------------------------------------------- tokenisation


def test_identifiers_contribute_their_parts():
    assert tokenize("verify_signature") == ["verify_signature", "verify", "signature"]
    assert tokenize("URLSafeSerializer") == ["urlsafeserializer", "url", "safe", "serializer"]


def test_english_stopwords_are_dropped_from_both_sides():
    """BM25 scores rarity, and "how does the" is neither rare nor about code."""
    assert tokenize("how does the code sign a value") == ["sign", "value"]


def test_a_repeated_word_is_kept_because_frequency_is_the_point():
    assert tokenize("sign sign sign").count("sign") == 3


def test_a_single_letter_or_punctuation_is_not_a_term():
    assert tokenize("x = 1 + 2; y") == []


# ---------------------------------------------------------------- scoring


def test_a_record_that_mentions_the_term_beats_one_that_does_not():
    bm25 = index(("signer.py", "sign the value"), ("exc.py", "raise an error"))
    assert [r.record_id for r in bm25.search("sign", 5)] == ["signer.py"]


def test_a_term_in_every_record_cannot_decide_the_order():
    """The idf form used here stays positive, so a document never wins by *lacking* a term."""
    bm25 = index(("a.py", "self value"), ("b.py", "self value"), ("c.py", "self value"))
    scores = bm25.scores("self")
    assert len(set(scores.values())) == 1
    assert all(score > 0 for score in scores.values())


def test_a_rarer_term_is_worth_more_than_a_common_one():
    bm25 = index(
        ("a.py", "signature common"),
        ("b.py", "common"),
        ("c.py", "common"),
        ("d.py", "common"),
    )
    # Both terms appear once in a.py (document 0); only the rarer one says anything about it.
    assert bm25.scores("signature")[0] > bm25.scores("common")[0]


def test_term_frequency_saturates_rather_than_scaling_linearly():
    once = index(("a.py", "sign"), ("b.py", "other")).scores("sign")[0]
    ten = index(("a.py", "sign " * 10), ("b.py", "other")).scores("sign")[0]
    assert once < ten < 10 * once


def test_a_long_record_is_penalised_for_its_length():
    """Same single occurrence; the longer document scores lower when b > 0."""
    short = index(("a.py", "sign"), ("b.py", "padding " * 50)).scores("sign")[0]
    long = index(("a.py", "sign " + "padding " * 50), ("b.py", "padding " * 50)).scores("sign")[0]
    assert long < short


def test_length_normalisation_can_be_switched_off_with_b_zero():
    pairs = (("a.py", "sign " + "padding " * 50), ("b.py", "sign"))
    with_b = index(*pairs, settings=Bm25Settings(b=DEFAULT_B)).scores("sign")
    without_b = index(*pairs, settings=Bm25Settings(b=0.0)).scores("sign")
    assert with_b[0] < with_b[1]
    assert without_b[0] == pytest.approx(without_b[1])


def test_the_score_matches_the_bm25_formula():
    """One document, one term: the whole expression is small enough to write out."""
    bm25 = index(("a.py", "sign sign"), ("b.py", "other"), settings=Bm25Settings(k1=1.2, b=0.75))
    n, df, freq, length, avgdl = 2, 1, 2, 2, 1.5
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    norm = 1 - 0.75 + 0.75 * (length / avgdl)
    expected = idf * freq * (1.2 + 1) / (freq + 1.2 * norm)
    assert bm25.scores("sign")[0] == pytest.approx(expected)


def test_a_question_with_no_known_term_finds_nothing():
    bm25 = index(("a.py", "sign the value"))
    assert bm25.search("kubernetes", 5) == []


def test_an_empty_corpus_does_not_divide_by_zero():
    assert len(Bm25Index(records=[], texts=[])) == 0
    assert Bm25Index(records=[], texts=[]).search("anything", 5) == []
    empty = index(("a.py", ""), ("b.py", ""))
    assert empty.search("anything", 5) == []


def test_records_and_texts_must_line_up():
    with pytest.raises(ValueError):
        Bm25Index(records=[record("a.py")], texts=[])


def test_ties_break_by_document_order_so_a_rerun_matches():
    bm25 = index(("a.py", "sign"), ("b.py", "sign"), ("c.py", "sign"))
    first = [r.record_id for r in bm25.search("sign", 3)]
    assert first == ["a.py", "b.py", "c.py"] == [r.record_id for r in bm25.search("sign", 3)]


def test_search_returns_at_most_top_k():
    bm25 = index(("a.py", "sign"), ("b.py", "sign"), ("c.py", "sign"))
    assert len(bm25.search("sign", 2)) == 2


# ---------------------------------------------------------------- fusion


def test_fusion_adds_one_over_k_plus_rank_from_each_list():
    a, b, c = record("a"), record("b"), record("c")
    fused = reciprocal_rank_fusion([[a, b], [b, c]], k=1)
    # b is second in one list and first in the other: 1/3 + 1/2 beats a's 1/2 and c's 1/3.
    assert [r.record_id for r in fused] == ["b", "a", "c"]


def test_fusion_keeps_first_appearance_order_on_a_tie():
    a, b = record("a"), record("b")
    assert [r.record_id for r in reciprocal_rank_fusion([[a], [b]])] == ["a", "b"]
    assert [r.record_id for r in reciprocal_rank_fusion([[b], [a]])] == ["b", "a"]


def test_fusion_of_one_list_keeps_that_list():
    a, b = record("a"), record("b")
    assert [r.record_id for r in reciprocal_rank_fusion([[a, b]])] == ["a", "b"]


# ---------------------------------------------------------------- inside the answer service


class FakeRetriever(CodeRetriever):
    """A retriever whose vector order and record list are both given up front."""

    def __init__(self, records: list[EmbeddingRecord], vector_order: list[str]) -> None:
        self._records = records
        self._vector_order = vector_order
        self.searched = 0

    def index(self, repo_id, records, embedder=""):  # pragma: no cover - not used
        raise NotImplementedError

    def search(self, repo_id, query_vector, top_k):
        self.searched += 1
        by_id = {r.record_id: r for r in self._records}
        return [by_id[name] for name in self._vector_order][:top_k]

    def records(self, repo_id):
        return self._records

    def has_index(self, repo_id):
        return True

    def remove(self, repo_id):  # pragma: no cover - not used
        raise NotImplementedError


class FakeEmbedder:
    def signature(self):
        return "fake"

    def embed_query(self, text):
        return [1.0]

    def embed_texts(self, texts):
        return [[1.0] for _ in texts]


@pytest.fixture
def corpus(tmp_path):
    """Three files on disk, so the service can read each record's text for BM25 and the rerank."""
    files = {
        "signer.py": "def sign(value):\n    return derive_key() + value\n",
        "timed.py": "def expired(age):\n    return age > max_age\n",
        "exc.py": "class BadSignature(Exception):\n    pass\n",
    }
    records = []
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        records.append(record(name, path=str(path)))
    return records


def service(retriever, **settings) -> AnswerService:
    base = RetrievalSettings(candidates=3, rerank_weight=0.0, drop_stopwords=True)
    return AnswerService(retriever=retriever, embedder=FakeEmbedder(), llm=None, settings=replace(base, **settings))


def test_bm25_is_off_by_default_and_the_candidates_are_the_vector_order(corpus):
    retriever = FakeRetriever(corpus, ["exc.py", "timed.py", "signer.py"])
    got = service(retriever).retrieve("repo", "how does it sign a value?", top_k=3)

    assert [r.record_id for r in got] == ["exc.py", "timed.py", "signer.py"]
    assert RetrievalSettings().bm25_enabled is False


def test_switching_bm25_on_pulls_a_lexically_matching_record_up(corpus):
    retriever = FakeRetriever(corpus, ["exc.py", "timed.py", "signer.py"])
    got = service(retriever, bm25_enabled=True).retrieve("repo", "how does it sign a value?", top_k=3)

    # signer.py is last in the vector order and first lexically, so fusion lifts it.
    assert got[0].record_id == "signer.py"


def test_the_pool_does_not_grow_when_bm25_is_on(corpus):
    """Fusion changes which records are in the pool, never how many. The pool depth is
    max(top_k, candidates) either way, so nothing downstream sees a longer list than before."""
    retriever = FakeRetriever(corpus, ["exc.py", "timed.py"])
    plain = service(retriever, candidates=2).retrieve("repo", "sign value", top_k=2)
    fused = service(retriever, bm25_enabled=True, candidates=2).retrieve("repo", "sign value", top_k=2)

    assert len(plain) == len(fused) == 2
    # signer.py was not in the vector pool at all; BM25 put it there, and the pool stayed at two.
    assert "signer.py" in {r.record_id for r in fused}
    assert "signer.py" not in {r.record_id for r in plain}


def test_the_lexical_index_is_built_once_and_reused(corpus):
    retriever = FakeRetriever(corpus, ["exc.py", "timed.py", "signer.py"])
    answers = service(retriever, bm25_enabled=True)

    answers.retrieve("repo", "sign value", top_k=3)
    built = answers._bm25[(id(retriever), "repo")]
    answers.retrieve("repo", "expired age", top_k=3)

    assert answers._bm25[(id(retriever), "repo")] is built


def test_reindexing_replaces_the_lexical_index(corpus):
    retriever = FakeRetriever(corpus, ["exc.py", "timed.py", "signer.py"])
    answers = service(retriever, bm25_enabled=True)
    answers.retrieve("repo", "sign value", top_k=3)
    first = answers._bm25[(id(retriever), "repo")]

    retriever._records = list(corpus)  # a new list, as reindexing produces
    answers.retrieve("repo", "sign value", top_k=3)

    assert answers._bm25[(id(retriever), "repo")] is not first


def test_a_retriever_that_cannot_list_its_records_still_answers(corpus):
    retriever = FakeRetriever(corpus, ["exc.py", "signer.py"])
    retriever.records = lambda repo_id: []
    got = service(retriever, bm25_enabled=True).retrieve("repo", "sign value", top_k=3)

    assert [r.record_id for r in got] == ["exc.py", "signer.py"]


def test_the_defaults_are_the_literature_values():
    settings = RetrievalSettings()
    assert (settings.bm25_k1, settings.bm25_b) == (DEFAULT_K1, DEFAULT_B)
    assert settings.bm25_rrf_k == DEFAULT_RRF_K
    assert settings.bm25_depth == 0
