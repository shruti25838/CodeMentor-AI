from collections import Counter

from fastapi import APIRouter, Depends, HTTPException

from codeatlas.app.di import get_repo_state_store
from codeatlas.schemas.overview import RepoOverviewRequest, RepoOverviewResponse
from codeatlas.services.state.repo_state_store import RepoStateStore

router = APIRouter(prefix="/repo-overview", tags=["repository"])


@router.post("", response_model=RepoOverviewResponse)
def repo_overview(
    request: RepoOverviewRequest,
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> RepoOverviewResponse:
    state = state_store.get(request.repo_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Repository not found")

    language_counts = Counter(file.language for file in state.parsed_repo.files)
    return RepoOverviewResponse(
        repo_id=request.repo_id,
        file_count=len(state.parsed_repo.files),
        function_count=len(state.parsed_repo.functions),
        language_counts=dict(language_counts),
        dependency_edges=state.import_graph.number_of_edges(),
    )
