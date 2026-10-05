from abc import ABC, abstractmethod

from codeatlas.models.embedding_record import EmbeddingRecord


class CodeRetriever(ABC):
    @abstractmethod
    def index(self, repo_id: str, records: list[EmbeddingRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, repo_id: str, query_vector: list[float], top_k: int) -> list[EmbeddingRecord]:
        raise NotImplementedError

    @abstractmethod
    def has_index(self, repo_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def remove(self, repo_id: str) -> None:
        """Drop a repo's index, in memory and on disk."""
        raise NotImplementedError


class GraphRetriever(ABC):
    @abstractmethod
    def neighbors(self, node_id: str) -> list[str]:
        raise NotImplementedError
