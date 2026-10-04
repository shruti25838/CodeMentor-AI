from fastapi import APIRouter, Depends

from codeatlas.app.di import get_explain_service, get_repo_state_store
from codeatlas.controllers.repo_guard import require_known_repo
from codeatlas.schemas.explain import ExplainRequest, ExplainResponse
from codeatlas.services.qa.explain_service import CodeExplainService
from codeatlas.services.state.repo_state_store import RepoStateStore

router = APIRouter(prefix="/explain", tags=["qa"])


@router.post("", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    service: CodeExplainService = Depends(get_explain_service),
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> ExplainResponse:
    require_known_repo(state_store, request.repo_id)
    result = service.explain(repo_id=request.repo_id, node_id=request.node_id)
    return ExplainResponse(
        node_id=result.node_id,
        summary=result.summary,
        snippet=result.snippet,
    )
