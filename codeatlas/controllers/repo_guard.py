from fastapi import HTTPException

from codeatlas.services.state.repo_state_store import RepoStateStore

REPO_NOT_FOUND = (
    "This repository is not indexed on the server. It may have been removed when the server "
    "restarted. Go back to the home page and index it again."
)


def require_known_repo(state_store: RepoStateStore, repo_id: str) -> None:
    """Raise 404 for an unknown repo id. Call before any language model work."""
    if state_store.get(repo_id) is None:
        raise HTTPException(status_code=404, detail=REPO_NOT_FOUND)
