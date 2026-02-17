from datetime import datetime
from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.memory.interfaces import MemoryStore
from codeatlas.models.agent_memory import AgentMemory


class MemoryAgent(Agent):
    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        # Simple instruction parsing for MVP
        # "save: <content>" or "list"
        if prompt.startswith("save:"):
            content = prompt[5:].strip()
            memory = AgentMemory(
                memory_id=str(datetime.utcnow().timestamp()),
                scope=f"repo:{repo_id}" if repo_id else "global",
                content=content,
                created_at=datetime.utcnow(),
            )
            self._memory_store.save(memory)
            return "Memory saved."
        
        if prompt == "list":
            scope = f"repo:{repo_id}" if repo_id else "global"
            memories = self._memory_store.list(scope)
            if not memories:
                return "No memories found."
            return "\n".join([f"[{m.created_at}] {m.content}" for m in memories])

        return "MemoryAgent commands: 'save: <text>' or 'list'"
