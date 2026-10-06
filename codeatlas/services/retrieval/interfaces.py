from abc import ABC, abstractmethod

from codeatlas.models.embedding_record import EmbeddingRecord


class CodeRetriever(ABC):
    @abstractmethod
    def index(self, repo_id: str, records: list[EmbeddingRecord], embedder: str = "") -> None:
        """Store a repo's records. `embedder` is the signature of the embedder that made the vectors."""
        raise NotImplementedError

    def stored_embedder(self, repo_id: str) -> str | None:
        """Signature of the embedder that built this index, "" if the index predates the record.

        None means this retriever does not keep track, and callers must not treat that as a mismatch.
        """
        return None

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
