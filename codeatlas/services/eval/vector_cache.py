"""Remember every vector an evaluation paid for, so it is never paid for twice.

An experiment asks the same questions of the same repositories under a handful of settings.
Without a cache, each configuration re-embeds all 1,864 records, the held-out run re-embeds
them again, and a rerun after a typo in the printing code costs another full set - all of it
out of a free allowance that resets once a day and cannot be topped up.

So vectors are kept on disk under `.codeatlas/` (gitignored), keyed by **the settings that
shaped the text and the text itself**: model, pooling, per-document clip, and a SHA-256 of the
string that was sent. Changing any of those is a different key and a real call, which is the
point - a cache that returned a mean-pooled vector for a cls-pooled run would quietly invent
a result, and the measurement would be of nothing.

It is deliberately an evaluation tool and not wired into the server. The server's cache is its
index; this exists because an experiment indexes the same repository many times over.

Saving happens after every call that cost something, not at the end of the run, because the
thing being protected is spend that has already happened.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from codeatlas.services.retrieval.embedding import EmbeddingService

DEFAULT_CACHE_DIR = Path(".codeatlas") / "eval-vectors"


def cache_key(model: str, pooling: str, max_chars: int | None) -> str:
    """What makes two vectors comparable. Anything outside this list must not share a cache file."""
    return f"{model}|pooling={pooling}|max_chars={max_chars}"


class CachingEmbeddingService(EmbeddingService):
    def __init__(self, inner: EmbeddingService, key: str, cache_dir: Path | None = None) -> None:
        self._inner = inner
        self._key = key
        directory = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.json"
        self._vectors: dict[str, list[float]] = {}
        self._load()
        self.hits = 0
        self.misses = 0
        self._logger = logging.getLogger(__name__)

    @property
    def path(self) -> Path:
        return self._path

    def signature(self) -> str:
        """The wrapped embedder's own signature: a cache must never change what an index is."""
        return self._inner.signature()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        wanted = [_digest(text) for text in texts]
        missing = [text for text, key in zip(texts, wanted) if key not in self._vectors]
        # Each distinct text once: a repository with two identical functions should pay once.
        unique: dict[str, str] = {}
        for text in missing:
            unique.setdefault(_digest(text), text)
        if unique:
            self.misses += len(unique)
            fresh = self._inner.embed_texts(list(unique.values()))
            for key, vector in zip(unique.keys(), fresh):
                self._vectors[key] = vector
            self._save()
        self.hits += len(texts) - len(unique)
        return [self._vectors[key] for key in wanted]

    def embed_query(self, text: str) -> list[float]:
        """Through the same cache as a document: a question costs neurons too, and is asked repeatedly."""
        return self.embed_texts([text])[0]

    def known(self, texts: list[str]) -> list[bool]:
        """Which of these are already paid for. Used to estimate what a run will really cost."""
        return [_digest(text) in self._vectors for text in texts]

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        vectors = data.get("vectors") if isinstance(data, dict) else None
        if isinstance(vectors, dict):
            self._vectors = {k: v for k, v in vectors.items() if isinstance(v, list)}

    def _save(self) -> None:
        payload = {"key": self._key, "vectors": self._vectors}
        try:
            self._path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            # Losing the file means paying again, which is bad but is not a reason to lose the run.
            self._logger.warning("Could not write the vector cache: %s", exc)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
