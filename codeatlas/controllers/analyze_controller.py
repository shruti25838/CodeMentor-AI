import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from codeatlas.app.di import (
    get_ast_parser,
    get_dependency_graph_builder,
    get_index_service,
    get_repo_state_store,
    get_repository_loader,
)
from codeatlas.observability.timing import timed_request
from codeatlas.schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse
from codeatlas.services.dependency.interfaces import DependencyGraphBuilder
from codeatlas.services.ingestion.git_loader import RepoCloneError, remove_clone, repo_cache_url
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
    http_request: Request,
    response: Response,
    loader: RepositoryLoader = Depends(get_repository_loader),
    parser: AstParser = Depends(get_ast_parser),
    graph_builder: DependencyGraphBuilder = Depends(get_dependency_graph_builder),
    index_service: CodeIndexService = Depends(get_index_service),
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> AnalyzeRepoResponse:
    with timed_request() as timer:
        result = None
        if http_request.app.state.config.index_cache_enabled:
            result = _from_cache(request.repo_url, loader, index_service, state_store, timer)
        if result is None:
            result = _analyze(request, loader, parser, graph_builder, index_service, state_store, timer)
        response.headers["Server-Timing"] = timer.server_timing()
        timer.log("analyze")
    return result


def _from_cache(repo_url, loader, index_service, state_store, timer) -> AnalyzeRepoResponse | None:
    """Reuse an earlier index of the same repository at the same commit, if one is complete on disk.

    The commit comes from the host (git ls-remote), so a push to the repository gives a miss
    and a fresh index. Any doubt returns None and the caller indexes as usual.
    """
    try:
        key = repo_cache_url(repo_url)
    except RepoCloneError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    with timer.stage("cache_lookup"):
        candidates = [
            (repo_id, state)
            for repo_id, state in state_store.items()
            if state.commit
            and _cache_key(state.url) == key
            and index_service.has_index(repo_id)
            and Path(state.root_path).is_dir()
        ]
    # ls-remote costs a round trip to the host, so only ask when there is something to reuse.
    if not candidates:
        return None
    with timer.stage("remote_head"):
        commit = loader.remote_head(repo_url)
    for repo_id, state in candidates:
        if commit and state.commit == commit:
            logger.info("Index cache hit for %s", repo_id)
            return AnalyzeRepoResponse(
                repository_id=repo_id,
                file_count=len(state.parsed_repo.files),
                dependency_edges=len(state.import_graph.edges),
                indexing_status="ready",
            )
    return None


def _cache_key(url: str) -> str | None:
    try:
        return repo_cache_url(url)
    except RepoCloneError:
        return None


def _analyze(request, loader, parser, graph_builder, index_service, state_store, timer) -> AnalyzeRepoResponse:
    try:
        with timer.stage("clone"):
            repo = loader.load(request.repo_url)
    except RepoCloneError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    # Parse, graph and index before answering. The repo is saved only when all of it succeeds,
    # so a client never sees a half-ready repo; on any failure the clone and index are removed.
    try:
        with timer.stage("parse"):
            parsed = parser.parse_repository(repo)
        with timer.stage("graph"):
            dependency_graph = graph_builder.build_import_graph(parsed)
        # Adds read_files, embed and index_write.
        index_service.index_repository(repo, parsed)
    except Exception as exc:
        index_service.discard(repo.repo_id)
        remove_clone(repo)
        if isinstance(exc, IndexingTimeoutError):
            logger.warning("Indexing timed out for %s: %s", repo.repo_id, exc)
            raise HTTPException(status_code=504, detail=INDEX_TIMEOUT_DETAIL)
        logger.exception("Analysis failed for %s", repo.repo_id)
        raise HTTPException(status_code=500, detail=INDEX_FAILED_DETAIL)

    with timer.stage("save_state"):
        state_store.save(
            repo.repo_id,
            RepoState(
                parsed_repo=parsed,
                import_graph=dependency_graph,
                root_path=repo.root_path,
                name=repo.name,
                url=repo.url,
                commit=repo.commit,
            ),
        )
    return AnalyzeRepoResponse(
        repository_id=repo.repo_id,
        file_count=len(parsed.files),
        dependency_edges=len(dependency_graph.edges),
        indexing_status="ready",
    )
