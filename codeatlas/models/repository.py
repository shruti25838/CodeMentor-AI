from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Repository:
    repo_id: str
    name: str
    url: str
    root_path: str
    ingested_at: datetime
