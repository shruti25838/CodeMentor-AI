from fastapi import APIRouter, Depends

from codeatlas.app.di import get_explain_service
from codeatlas.schemas.explain import ExplainRequest, ExplainResponse
from codeatlas.services.qa.explain_service import CodeExplainService

router = APIRouter(prefix="/explain", tags=["qa"])


@router.post("", response_model=ExplainResponse)
def explain(
    request: ExplainRequest,
    service: CodeExplainService = Depends(get_explain_service),
) -> ExplainResponse:
    result = service.explain(repo_id=request.repo_id, node_id=request.node_id)
    return ExplainResponse(
        node_id=result.node_id,
        summary=result.summary,
        snippet=result.snippet,
    )
