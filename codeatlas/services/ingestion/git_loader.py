import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from codeatlas.models.repository import Repository
from codeatlas.services.ingestion.interfaces import RepositoryLoader


class GitRepositoryLoader(RepositoryLoader):
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or ".codeatlas/repos").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load(self, repo_url: str) -> Repository:
        repo_url_str = str(repo_url)
        repo_id = str(uuid.uuid4())
        repo_dir = self.base_dir / repo_id
        self._clone(repo_url_str, repo_dir)
        name = repo_url_str.rstrip("/").split("/")[-1]
        return Repository(
            repo_id=repo_id,
            name=name,
            url=repo_url_str,
            root_path=str(repo_dir),
            ingested_at=datetime.now(timezone.utc),
        )

    def _clone(self, repo_url: str, repo_dir: Path) -> None:
        subprocess.run(
            ["git", "clone", "--depth", "1", str(repo_url), str(repo_dir)],
            check=True,
            capture_output=True,
        )
