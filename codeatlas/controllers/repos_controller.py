import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends

from codeatlas.app.di import get_repo_state_store
from codeatlas.schemas.repos import ListReposResponse, RepoInfo
from codeatlas.services.state.repo_state_store import RepoStateStore


def _name_from_git_remote(root_path: str, repo_id: str = "") -> str:
    """Try to read the origin remote URL and extract the repo name."""
    # Try the stored root_path first, then local .codeatlas/repos/<id>
    candidates = [root_path]
    if repo_id:
        local_dir = Path(".codeatlas/repos") / repo_id
        if local_dir.exists():
            candidates.insert(0, str(local_dir))
    for cwd in candidates:
        if not Path(cwd).is_dir():
            continue
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                url = result.stdout.strip().rstrip("/")
                return url.split("/")[-1].removesuffix(".git")
        except Exception:
            continue
    return ""

router = APIRouter(prefix="/repos", tags=["repository"])


@router.get("", response_model=ListReposResponse)
def list_repos(
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> ListReposResponse:
    repo_ids = state_store.list_repo_ids()
    repos = []
    for rid in repo_ids:
        state = state_store.get(rid)
        name = ""
        if state:
            name = state.name
            if not name and state.root_path:
                name = _name_from_git_remote(state.root_path, repo_id=rid)
        if not name:
            name = rid[:8]
        repos.append(RepoInfo(repo_id=rid, name=name))
    return ListReposResponse(repo_ids=repo_ids, repos=repos)
