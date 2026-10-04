from fastapi import APIRouter, Depends

from codeatlas.app.di import get_agent_orchestrator, get_repo_state_store
from codeatlas.controllers.repo_guard import require_known_repo
from codeatlas.schemas.generate import GenerateCodeRequest, GenerateCodeResponse
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.state.repo_state_store import RepoStateStore

router = APIRouter(prefix="/generate-code", tags=["coding"])


@router.post("", response_model=GenerateCodeResponse)
def generate_code(
    request: GenerateCodeRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> GenerateCodeResponse:
    require_known_repo(state_store, request.repo_id)
    result = orchestrator.handle_generation(request.prompt, request.repo_id)
    return GenerateCodeResponse(
        diff=result.diff,
        notes=result.notes,
        citations=result.citations,
    )
