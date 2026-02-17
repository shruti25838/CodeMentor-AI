from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def run(self, prompt: str, repo_id: str | None = None) -> str:
        raise NotImplementedError
