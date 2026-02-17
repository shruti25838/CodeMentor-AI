from functools import lru_cache

from sentence_transformers import SentenceTransformer

from codeatlas.services.retrieval.embedding import EmbeddingService


class SentenceTransformerEmbeddingService(EmbeddingService):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = _get_model(self._model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = _get_model(self._model_name)
        embedding = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].tolist()


@lru_cache
def _get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)
