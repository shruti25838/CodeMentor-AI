from fastapi import APIRouter, Depends

from codeatlas.app.di import get_repo_state_store
from codeatlas.observability.tracker import tracker
from codeatlas.services.state.repo_state_store import RepoStateStore

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/stats")
def get_eval_stats(
    state_store: RepoStateStore = Depends(get_repo_state_store),
):
    """Return aggregated session analytics and repository metrics."""
    stats = tracker.get_stats()

    # Enrich with per-repository stats
    repo_stats = []
    for repo_id, state in state_store._states.items():
        files = state.parsed_repo.files
        functions = state.parsed_repo.functions
        lang_counts: dict[str, int] = {}
        for f in files:
            lang_counts[f.language] = lang_counts.get(f.language, 0) + 1

        repo_stats.append(
            {
                "repo_id": repo_id,
                "name": getattr(state, "name", "") or repo_id[:8],
                "file_count": len(files),
                "function_count": len(functions),
                "dependency_edges": state.import_graph.number_of_edges(),
                "languages": lang_counts,
            }
        )

    stats["repositories"] = repo_stats
    return stats
