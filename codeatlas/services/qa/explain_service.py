from dataclasses import dataclass
import logging
from pathlib import Path
import re

from codeatlas.services.state.repo_state_store import RepoStateStore


@dataclass(frozen=True)
class ExplainResult:
    node_id: str
    summary: str
    snippet: str


from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

class CodeExplainService:
    def __init__(self, state_store: RepoStateStore, llm: BaseChatModel) -> None:
        self._state_store = state_store
        self._llm = llm
        self._logger = logging.getLogger(__name__)

    def explain(self, repo_id: str, node_id: str) -> ExplainResult:
        self._logger.info("Explain requested for repo %s node %s", repo_id, node_id)
        state = self._state_store.get(repo_id)
        if state is None:
            return ExplainResult(node_id=node_id, summary="Repository not found.", snippet="")

        path, start, end = _parse_node_id(node_id)
        if not path:
            return ExplainResult(node_id=node_id, summary="Invalid node_id.", snippet="")

        snippet = _read_snippet(Path(path), start, end)
        
        # Use LLM to summarize
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior developer. Summarize the following code snippet concisely."),
            ("human", f"Code from {path}:\n\n{snippet}")
        ])
        try:
            chain = prompt | self._llm
            response = chain.invoke({})
            summary = response.content
        except Exception as e:
            summary = f"Error generating summary: {e}"
            
        return ExplainResult(node_id=node_id, summary=summary, snippet=snippet)


def _parse_node_id(node_id: str) -> tuple[str | None, int | None, int | None]:
    match = re.match(r"^(.*?):(\d+)-(\d+)$", node_id)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return node_id, None, None


def _read_snippet(path: Path, start: int | None, end: int | None) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if start is None or end is None:
        return "\n".join(lines[:200])
    start_idx = max(start - 1, 0)
    end_idx = max(end, start_idx)
    return "\n".join(lines[start_idx:end_idx])


def _summarize(path: str, start: int | None, end: int | None, snippet: str) -> str:
    line_count = len(snippet.splitlines())
    if start is not None and end is not None:
        return f"Snippet from {path} (lines {start}-{end}), {line_count} lines."
    return f"Snippet from {path}, {line_count} lines."
