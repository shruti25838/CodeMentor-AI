from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def signature(self) -> str:
        """Identifies how text becomes vectors; an index is only reused by an embedder with the same signature."""
        return type(self).__name__
