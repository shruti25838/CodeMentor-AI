from pathlib import Path

import networkx as nx

from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.state.repo_state_store import RepoState, RepoStateStore


def test_repo_state_persist_and_load(tmp_path: Path) -> None:
    repo_id = "repo-1"
    files = [SourceFile(path="a.py", language="python", size_bytes=10)]
    functions = [
        FunctionNode(
            name="foo",
            file_path="a.py",
            start_line=1,
            end_line=2,
            signature="def foo():",
        )
    ]
    parsed = ParsedRepository(repository_id=repo_id, files=files, functions=functions)
    graph = nx.DiGraph()
    graph.add_node("a.py")
    graph.add_edge("a.py", "os", relation="imports")
    store = RepoStateStore(base_dir=str(tmp_path))
    store.save(repo_id, RepoState(parsed_repo=parsed, import_graph=graph))

    reloaded = RepoStateStore(base_dir=str(tmp_path))
    state = reloaded.get(repo_id)
    assert state is not None
    assert len(state.parsed_repo.files) == 1
    assert len(state.parsed_repo.functions) == 1
    assert state.import_graph.number_of_edges() == 1
