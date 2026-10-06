"""Opt-in semantic embeddings through fastembed (ONNX on CPU, no PyTorch).

Not installed by default: `pip install -r requirements-fastembed.txt`, then set
CODEATLAS_EMBEDDING_PROVIDER=fastembed. The model is downloaded on first use unless the
image was built with the download baked in (see Dockerfile, --build-arg INSTALL_FASTEMBED=1).
"""

from functools import lru_cache

try:
    from fastembed import TextEmbedding  # type: ignore
except Exception:  # pragma: no cover - exercised only where fastembed is absent
    TextEmbedding = None  # type: ignore

from codeatlas.services.retrieval.embedding import EmbeddingService

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
# One document at a time. A transformer on CPU holds the whole batch in memory, and the
# indexer only checks its time limit between batches, so a large batch means both a memory
# spike and a time limit that overshoots by a whole batch.
DEFAULT_FASTEMBED_BATCH_SIZE = 1

NOT_INSTALLED = (
    "fastembed is not installed. Set CODEATLAS_EMBEDDING_PROVIDER=hash, "
    "or install it with `pip install -r requirements-fastembed.txt`."
)


class FastEmbedEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model_name: str = DEFAULT_FASTEMBED_MODEL,
        batch_size: int = DEFAULT_FASTEMBED_BATCH_SIZE,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._cache_dir = cache_dir

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._model()
        return [vector.tolist() for vector in model.embed(texts, batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        model = self._model()
        # query_embed is what the model expects on the question side; for a model that wants no
        # query prefix it is the same as embed, and for one that does it adds the right one.
        return next(iter(model.query_embed(text))).tolist()

    def signature(self) -> str:
        # Batch size is not here: it changes how long indexing takes, not what the vectors are.
        return f"fastembed:{self._model_name}"

    def _model(self) -> "TextEmbedding":
        if TextEmbedding is None:
            raise RuntimeError(NOT_INSTALLED)
        return _get_model(self._model_name, self._cache_dir)


@lru_cache
def _get_model(model_name: str, cache_dir: str | None) -> "TextEmbedding":
    """Built on first use, not in __init__, so constructing the service downloads nothing."""
    if TextEmbedding is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError(NOT_INSTALLED)
    return TextEmbedding(model_name=model_name, cache_dir=cache_dir)
