import json
import os
from datetime import datetime
from typing import List, Any
from pathlib import Path

from codeatlas.models.agent_memory import AgentMemory
from codeatlas.services.memory.interfaces import MemoryStore

class JsonMemoryStore(MemoryStore):
    def __init__(self, file_path: str = "memory.json") -> None:
        self._file_path = file_path
        self._memories: List[AgentMemory] = self._load()

    def _load(self) -> List[AgentMemory]:
        if not os.path.exists(self._file_path):
            return []
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [self._deserialize(item) for item in data]
        except Exception:
            return []

    def _save_to_disk(self) -> None:
        data = [self._serialize(m) for m in self._memories]
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _serialize(self, memory: AgentMemory) -> dict:
        return {
            "memory_id": memory.memory_id,
            "scope": memory.scope,
            "content": memory.content,
            "created_at": memory.created_at.isoformat()
        }

    def _deserialize(self, data: dict) -> AgentMemory:
        return AgentMemory(
            memory_id=data["memory_id"],
            scope=data["scope"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"])
        )

    def save(self, memory: AgentMemory) -> None:
        self._memories.append(memory)
        self._save_to_disk()

    def list(self, scope: str) -> List[AgentMemory]:
        return [memory for memory in self._memories if memory.scope == scope]
