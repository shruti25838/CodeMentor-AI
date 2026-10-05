from abc import ABC, abstractmethod

from codeatlas.models.repository import Repository


class RepositoryLoader(ABC):
    @abstractmethod
    def load(self, repo_url: str) -> Repository:
        raise NotImplementedError

    def remote_head(self, repo_url: str) -> str | None:
        """Commit the repository's default branch points at, without cloning; None if unknown."""
        return None
