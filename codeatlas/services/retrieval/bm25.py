"""BM25 over a repository's records, in plain Python, as a second retriever beside the vectors.

Why a second retriever at all. The existing pipeline has one *recall* mechanism - the embedder
chooses the candidate pool - and one *ordering* mechanism, the keyword rerank. The rerank can
only reorder what the pool already holds, so a question whose file the embedder never found
cannot be recovered by any amount of reordering; `docs/RESULTS.md` records exactly that for hash
on held-out flask, where recall@20 and hit@5 are both 0.600. BM25 is a different way of
*finding* records, so fusing it with the vector list changes the pool rather than its order.

What this is, precisely. Okapi BM25 with the usual saturation and length normalisation:

    score(q, d) = sum over terms t in q of  idf(t) * f(t,d) * (k1 + 1)
                                            ---------------------------
                                            f(t,d) + k1 * (1 - b + b * |d| / avgdl)

    idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

`k1` controls how quickly a repeated term stops adding score, `b` how hard a long document is
penalised for its length. The idf form is the one that stays positive for a term in every
document, which matters here: a term like `self` is in nearly every record, and a negative idf
would let a document win by *not* containing a query word.

Tokenisation is the one the rest of the retrieval already uses, so BM25 and the embedder are
reading the same words: identifiers split into their parts (`verify_signature` -> `verify`,
`signature`) and English stopwords dropped. Dropping stopwords matters more here than for the
embedder, because BM25 scores a term's rarity, and "how", "does" and "the" are neither rare nor
about code.

No new dependency: counters, a dictionary of postings, and arithmetic.

Scope. One index per repository, built from the record texts the reranker already reads, and
built **uncapped** - unlike the embedder, BM25 reads the whole of a long file rather than its
opening, because clipping is a cost control for a paid API and nothing here is paid for.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.retrieval.hash_embedder import STOPWORDS, split_identifier

# Okapi BM25's usual starting points. They are the defaults because they are the literature's,
# not because anything here measured them; what this repository measured is in docs/RESULTS.md.
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

# How many of these an AnswerService may hold at once. Unlike k1 and b this one *was* measured
# here: see docs/RESULTS.md for what an index of each of the two evaluation repositories weighs
# and what the whole process weighs around it. Four is two repositories' worth of headroom under
# the dual-index path, where one repository can have a lexical index per index kind.
DEFAULT_MAX_INDEXES = 4

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """The words BM25 scores: identifiers and their parts, lowercased, without English stopwords.

    Term *frequency* is the point of BM25, so this returns a list and not a set: a record that
    says `signature` nine times is more about signatures than one that says it once, up to the
    saturation `k1` imposes.
    """
    tokens: list[str] = []
    for match in _WORD.findall(text):
        lowered = match.lower()
        if lowered not in STOPWORDS:
            tokens.append(lowered)
        for part in split_identifier(match):
            if part != lowered and part not in STOPWORDS:
                tokens.append(part)
    return tokens


@dataclass(frozen=True)
class Bm25Settings:
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B


class Bm25Index:
    """One repository's records, scored lexically. Built once and queried many times."""

    def __init__(
        self,
        records: list[EmbeddingRecord],
        texts: list[str],
        settings: Bm25Settings | None = None,
    ) -> None:
        if len(records) != len(texts):
            raise ValueError("every record needs exactly one text")
        self._settings = settings or Bm25Settings()
        self._records = records
        self._lengths: list[int] = []
        # term -> [(document index, count), ...]
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self._lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self._postings.setdefault(term, []).append((index, count))
        self._count = len(records)
        total = sum(self._lengths)
        # An empty corpus, or one of empty records, must not divide by zero; with avgdl 1.0 and
        # every length 0 the normalisation term collapses to (1 - b), which is finite and equal
        # for every document, so nothing is ranked by an accident of arithmetic.
        self._avgdl = total / self._count if self._count and total else 1.0

    def __len__(self) -> int:
        return self._count

    def scores(self, query: str) -> dict[int, float]:
        """Document index -> BM25 score, for the documents that share at least one query term."""
        k1, b = self._settings.k1, self._settings.b
        scored: dict[int, float] = {}
        # A term repeated in the question is one term: BM25's query-side frequency adds nothing
        # at this length and would let "signature signature" outweigh a rarer word.
        for term in dict.fromkeys(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (self._count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, count in postings:
                norm = 1 - b + b * (self._lengths[index] / self._avgdl)
                scored[index] = scored.get(index, 0.0) + idf * count * (k1 + 1) / (count + k1 * norm)
        return scored

    def search(self, query: str, top_k: int) -> list[EmbeddingRecord]:
        """The best `top_k` records for this question, best first.

        Ties break by document order, so two records with the same score never swap between
        runs and look like a change.
        """
        scored = self.scores(query)
        if not scored:
            return []
        order = sorted(scored, key=lambda index: (-scored[index], index))
        return [self._records[index] for index in order[:top_k]]
