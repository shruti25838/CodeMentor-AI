import hashlib
import re
from math import sqrt

from codeatlas.services.retrieval.embedding import EmbeddingService

# Common English words in questions ("how does the code ..."); they say nothing about which code is meant.
_STOPWORD_TEXT = (
    "a an and are as at be by can code do does for from get got how i if in into is it its me my of "
    "on or so that the their them then there these they this to use used uses using was what when "
    "where which who why will with would you your"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


class HashEmbeddingService(EmbeddingService):
    def __init__(self, dimension: int = 384, lowercase: bool = False, subtokens: bool = False) -> None:
        self._dimension = dimension
        # lowercase: "Signer" and "signer" land in the same bucket.
        self._lowercase = lowercase
        # subtokens: identifiers also count their parts (verify_signature -> verify, signature).
        self._subtokens = subtokens

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def signature(self) -> str:
        return f"hash:{self._dimension}:lowercase={int(self._lowercase)}:subtokens={int(self._subtokens)}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _tokenize(text):
            for term in self._terms(token):
                bucket = _hash_to_bucket(term, self._dimension)
                vector[bucket] += 1.0
        return _normalize(vector)

    def _terms(self, token: str) -> list[str]:
        terms = [token.lower() if self._lowercase else token]
        if self._subtokens:
            parts = split_identifier(token)
            if len(parts) > 1:
                terms.extend(parts)
        return terms


def split_identifier(token: str) -> list[str]:
    """Lowercase parts of a snake_case or CamelCase identifier: URLSafeSerializer -> url, safe, serializer."""
    parts: list[str] = []
    for piece in token.split("_"):
        parts.extend(re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", piece))
    return [part.lower() for part in parts if part]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text)


def _hash_to_bucket(token: str, dimension: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big")
    return value % dimension


def _normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
