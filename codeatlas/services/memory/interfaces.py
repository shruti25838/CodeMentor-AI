from abc import ABC, abstractmethod

from codeatlas.models.agent_memory import AgentMemory


class MemoryStore(ABC):
    @abstractmethod
    def save(self, memory: AgentMemory) -> None:
        raise NotImplementedError

    @abstractmethod
    def list(self, scope: str) -> list[AgentMemory]:
        raise NotImplementedError
