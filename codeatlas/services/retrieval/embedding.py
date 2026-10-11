from abc import ABC, abstractmethod


class EmbeddingUnavailableError(Exception):
    """This embedder could not answer, and the caller should use the fallback index instead.

    Raised by a provider that depends on something outside this process - an API, a budget, a
    network - and deliberately not by one that cannot fail, like the hash embedder. `reason`
    says which of REASONS applies, and is the only part a visitor ever sees, through
    fallback_note().
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason


# One sentence each, written for a visitor who never asked about embeddings, budgets or
# providers. They say what happened and what was done about it, and nothing else: no provider
# name, no status code, no mention of a quota in units nobody outside this project knows.
REASONS = {
    "budget": "Today's semantic search allowance is used up, so this answer used keyword search.",
    "circuit": "Semantic search is briefly unavailable, so this answer used keyword search.",
    "api": "Semantic search did not respond, so this answer used keyword search.",
    # Both the repository that was over the document cap and the one whose semantic index was
    # built by a different embedder: from the visitor's side these are one situation.
    "unindexed": "Semantic search is not available for this repository, so this answer used keyword search.",
    "cap": "Semantic search is not available for this repository, so this answer used keyword search.",
}
DEFAULT_NOTE = REASONS["api"]


def fallback_note(reason: str) -> str:
    """The sentence a visitor reads when a question was answered from the fallback index."""
    return REASONS.get(reason, DEFAULT_NOTE)


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
