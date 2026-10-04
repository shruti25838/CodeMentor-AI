import logging

from fastapi import APIRouter, Depends, HTTPException

from codeatlas.app.di import (
    get_ast_parser,
    get_dependency_graph_builder,
    get_index_service,
    get_repo_state_store,
    get_repository_loader,
)
from codeatlas.schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse
from codeatlas.services.dependency.interfaces import DependencyGraphBuilder
from codeatlas.services.ingestion.git_loader import RepoCloneError, remove_clone
from codeatlas.services.ingestion.interfaces import RepositoryLoader
from codeatlas.services.parsing.interfaces import AstParser
from codeatlas.services.retrieval.indexing import CodeIndexService, IndexingTimeoutError
from codeatlas.services.state.repo_state_store import RepoState, RepoStateStore

router = APIRouter(prefix="/analyze-repo", tags=["analysis"])
logger = logging.getLogger(__name__)

INDEX_TIMEOUT_DETAIL = (
    "Indexing this repository took too long, so it was stopped and nothing was saved. Try a smaller repository."
)
INDEX_FAILED_DETAIL = "Indexing this repository failed, so nothing was saved. Please try again later."


@router.post("", response_model=AnalyzeRepoResponse)
def analyze_repo(
    request: AnalyzeRepoRequest,
    loader: RepositoryLoader = Depends(get_repository_loader),
    parser: AstParser = Depends(get_ast_parser),
    graph_builder: DependencyGraphBuilder = Depends(get_dependency_graph_builder),
    index_service: CodeIndexService = Depends(get_index_service),
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> AnalyzeRepoResponse:
    try:
        repo = loader.load(request.repo_url)
    except RepoCloneError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    # Parse, graph and index before answering. The repo is saved only when all of it succeeds,
    # so a client never sees a half-ready repo; on any failure the clone and index are removed.
    try:
        parsed = parser.parse_repository(repo)
        dependency_graph = graph_builder.build_import_graph(parsed)
        index_service.index_repository(repo, parsed)
    except Exception as exc:
        index_service.discard(repo.repo_id)
        remove_clone(repo)
        if isinstance(exc, IndexingTimeoutError):
            logger.warning("Indexing timed out for %s: %s", repo.repo_id, exc)
            raise HTTPException(status_code=504, detail=INDEX_TIMEOUT_DETAIL)
        logger.exception("Analysis failed for %s", repo.repo_id)
        raise HTTPException(status_code=500, detail=INDEX_FAILED_DETAIL)

    state_store.save(
        repo.repo_id,
        RepoState(
            parsed_repo=parsed,
            import_graph=dependency_graph,
            root_path=repo.root_path,
            name=repo.name,
            url=repo.url,
        ),
    )
    return AnalyzeRepoResponse(
        repository_id=repo.repo_id,
        file_count=len(parsed.files),
        dependency_edges=len(dependency_graph.edges),
        indexing_status="ready",
    )
