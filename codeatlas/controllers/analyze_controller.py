from fastapi import APIRouter, BackgroundTasks, Depends

from codeatlas.app.di import (
    get_ast_parser,
    get_dependency_graph_builder,
    get_repository_loader,
    get_index_service,
    get_repo_state_store,
)
from codeatlas.schemas.analyze import AnalyzeRepoRequest, AnalyzeRepoResponse
from codeatlas.services.dependency.interfaces import DependencyGraphBuilder
from codeatlas.services.ingestion.interfaces import RepositoryLoader
from codeatlas.services.parsing.interfaces import AstParser
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.state.repo_state_store import RepoState, RepoStateStore

router = APIRouter(prefix="/analyze-repo", tags=["analysis"])


@router.post("", response_model=AnalyzeRepoResponse)
def analyze_repo(
    request: AnalyzeRepoRequest,
    background_tasks: BackgroundTasks,
    loader: RepositoryLoader = Depends(get_repository_loader),
    parser: AstParser = Depends(get_ast_parser),
    graph_builder: DependencyGraphBuilder = Depends(get_dependency_graph_builder),
    index_service: CodeIndexService = Depends(get_index_service),
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> AnalyzeRepoResponse:
    repo = loader.load(request.repo_url)
    parsed = parser.parse_repository(repo)
    dependency_graph = graph_builder.build_import_graph(parsed)
    background_tasks.add_task(index_service.index_repository, repo, parsed)
    indexing_status = "queued"
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
        indexing_status=indexing_status,
    )
