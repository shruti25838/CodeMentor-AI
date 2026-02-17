import hashlib
import re
from math import sqrt

from codeatlas.services.retrieval.embedding import EmbeddingService


class HashEmbeddingService(EmbeddingService):
    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _tokenize(text):
            bucket = _hash_to_bucket(token, self._dimension)
            vector[bucket] += 1.0
        return _normalize(vector)


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
