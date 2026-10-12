import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request

from codeatlas.app.di import get_repo_state_store
from codeatlas.observability.tracker import tracker
from codeatlas.services.state.repo_state_store import RepoStateStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["evaluation"])

HEADING_NOTE = "Recorded results on small fixed question sets, not live accuracy."
MISSING_NOTE = (
    "Recorded results are not available in this build. The summary file is written by "
    "scripts/build_benchmarks.py and shipped with the image; nothing is measured at request time."
)


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


@router.get("/benchmarks")
def get_benchmarks(request: Request) -> dict:
    """The recorded-results summary, read from disk, plus the two facts only the running server knows.

    Read-only and open, like /eval/stats. Nothing here is measured now: the summary is a file
    `scripts/build_benchmarks.py` wrote from committed result files, and a test regenerates it and
    fails if the committed copy differs. The file is read per request rather than at startup so
    that a build without it answers honestly instead of refusing to boot.
    """
    config = request.app.state.config
    live = {
        # The provider the server would use for the next index, and the ceiling on documents
        # per repository that applies when that provider is the semantic one. Neither is secret:
        # no account id, no token, and no budget or neuron figure is read here.
        "embedding_provider": config.embedding_provider,
        "semantic_document_cap": config.cloudflare_max_documents,
    }

    path = Path(config.benchmarks_file)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Benchmarks summary unavailable at %s: %s", path, exc)
        return {"available": False, "note": MISSING_NOTE, "heading_note": HEADING_NOTE, "live": live}

    return {"available": True, "heading_note": HEADING_NOTE, "live": live, "summary": summary}
