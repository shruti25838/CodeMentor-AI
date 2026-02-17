from codeatlas.models.agent_memory import AgentMemory
from codeatlas.services.memory.interfaces import MemoryStore


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._memories: list[AgentMemory] = []

    def save(self, memory: AgentMemory) -> None:
        self._memories.append(memory)

    def list(self, scope: str) -> list[AgentMemory]:
        return [memory for memory in self._memories if memory.scope == scope]
