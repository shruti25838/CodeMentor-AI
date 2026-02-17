import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from codeatlas.app.di import get_repo_state_store
from codeatlas.schemas.files import (
    FileContentRequest,
    FileContentResponse,
    FileEntry,
    ListFilesRequest,
    ListFilesResponse,
)
from codeatlas.services.state.repo_state_store import RepoStateStore

router = APIRouter(prefix="/files", tags=["repository"])


@router.post("", response_model=ListFilesResponse)
def list_files(
    request: ListFilesRequest,
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> ListFilesResponse:
    state = state_store.get(request.repo_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Repository not found")

    files = []
    root_path = Path(state.root_path) if state.root_path else None
    
    # Common ignore list
    IGNORE_PATTERNS = {"__pycache__", ".git", ".pytest_cache", ".venv", "node_modules"}

    for source in state.parsed_repo.files:
        path_obj = Path(source.path)
        
        # Skip ignored patterns
        if any(part in IGNORE_PATTERNS for part in path_obj.parts):
            continue

        try:
            # If we have a root_path, use it
            if root_path and root_path.is_absolute():
                rel_path = path_obj.relative_to(root_path).as_posix()
            else:
                # Fallback: try to find the repos directory in the path and split from there
                # This helps with legacy indexed repos
                parts = path_obj.parts
                if "repos" in parts:
                    repos_idx = parts.index("repos")
                    # The folder after 'repos' is the UUID
                    rel_path = "/".join(parts[repos_idx + 2:])
                else:
                    rel_path = path_obj.name
                    
            if rel_path:
                files.append(FileEntry(path=rel_path, language=source.language))
        except (ValueError, IndexError):
            files.append(FileEntry(path=path_obj.name, language=source.language))

    files.sort(key=lambda entry: entry.path)
    return ListFilesResponse(files=files)


_EXT_TO_LANG = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "java": "java",
    "go": "go",
    "rs": "rust",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "rb": "ruby",
    "cs": "csharp",
    "php": "php",
    "kt": "kotlin",
    "swift": "swift",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "md": "markdown",
    "txt": "text",
    "toml": "toml",
    "cfg": "ini",
    "ini": "ini",
    "sh": "bash",
    "bat": "batch",
}


@router.post("/content", response_model=FileContentResponse)
def get_file_content(
    request: FileContentRequest,
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> FileContentResponse:
    state = state_store.get(request.repo_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Search in .codeatlas/repos/<id>/ first, then root_path
    candidates: list[Path] = []
    local_dir = Path(".codeatlas/repos") / request.repo_id
    if local_dir.exists():
        candidates.append(local_dir / request.file_path)
    if state.root_path:
        candidates.append(Path(state.root_path) / request.file_path)

    for candidate in candidates:
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            ext = candidate.suffix.lstrip(".")
            return FileContentResponse(
                path=request.file_path,
                content=content,
                language=_EXT_TO_LANG.get(ext, ext),
                line_count=content.count("\n") + 1,
            )

    raise HTTPException(status_code=404, detail="File not found")
