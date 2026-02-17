import json
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.source_file import SourceFile


@dataclass(frozen=True)
class RepoState:
    parsed_repo: ParsedRepository
    import_graph: nx.DiGraph
    root_path: str
    name: str = ""
    url: str = ""


class RepoStateStore:
    def __init__(self, base_dir: str | None = None) -> None:
        self._states: dict[str, RepoState] = {}
        self._base_dir = Path(base_dir).resolve() if base_dir else None
        self._logger = logging.getLogger(__name__)
        if self._base_dir:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    def save(self, repo_id: str, state: RepoState) -> None:
        self._states[repo_id] = state
        self._persist(repo_id, state)

    def get(self, repo_id: str) -> RepoState | None:
        return self._states.get(repo_id)

    def list_repo_ids(self) -> list[str]:
        return sorted(self._states.keys())

    def _persist(self, repo_id: str, state: RepoState) -> None:
        if not self._base_dir:
            return
        payload = {
            "repo_id": repo_id,
            "root_path": state.root_path,
            "name": state.name,
            "url": state.url,
            "files": [
                {
                    "path": source.path,
                    "language": source.language,
                    "size_bytes": source.size_bytes,
                }
                for source in state.parsed_repo.files
            ],
            "functions": [
                {
                    "name": function.name,
                    "file_path": function.file_path,
                    "start_line": function.start_line,
                    "end_line": function.end_line,
                    "signature": function.signature,
                }
                for function in state.parsed_repo.functions
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "relation": data.get("relation", "imports"),
                }
                for source, target, data in state.import_graph.edges(data=True)
            ],
        }
        target = self._base_dir / f"{repo_id}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._logger.info("Persisted repo state to %s", target)

    def _load_all(self) -> None:
        if not self._base_dir:
            return
        for path in self._base_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            repo_id = payload.get("repo_id")
            if not repo_id:
                continue
            files = [
                SourceFile(
                    path=item["path"],
                    language=item.get("language", ""),
                    size_bytes=item.get("size_bytes", 0),
                )
                for item in payload.get("files", [])
            ]
            functions = [
                FunctionNode(
                    name=item.get("name", ""),
                    file_path=item.get("file_path", ""),
                    start_line=item.get("start_line", 0),
                    end_line=item.get("end_line", 0),
                    signature=item.get("signature", ""),
                )
                for item in payload.get("functions", [])
            ]
            graph = nx.DiGraph()
            for source_file in files:
                graph.add_node(source_file.path)
            for edge in payload.get("edges", []):
                graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation"))
            parsed_repo = ParsedRepository(
                repository_id=repo_id, files=files, functions=functions
            )
            self._states[repo_id] = RepoState(
                parsed_repo=parsed_repo,
                import_graph=graph,
                root_path=payload.get("root_path", ""),
                name=payload.get("name", ""),
                url=payload.get("url", ""),
            )
            self._logger.info("Loaded repo state for %s", repo_id)
