from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AgentMemory:
    memory_id: str
    scope: str
    content: str
    created_at: datetime
