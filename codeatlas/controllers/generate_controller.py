from fastapi import APIRouter, Depends

from codeatlas.app.di import get_agent_orchestrator
from codeatlas.schemas.generate import GenerateCodeRequest, GenerateCodeResponse
from codeatlas.services.agents.orchestration import AgentOrchestrator

router = APIRouter(prefix="/generate-code", tags=["coding"])


@router.post("", response_model=GenerateCodeResponse)
def generate_code(
    request: GenerateCodeRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
) -> GenerateCodeResponse:
    result = orchestrator.handle_generation(request.prompt, request.repo_id)
    return GenerateCodeResponse(
        diff=result.diff,
        notes=result.notes,
        citations=result.citations,
    )
