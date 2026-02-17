from dataclasses import dataclass
import logging
from pathlib import Path
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.interfaces import CodeRetriever


def _clean_display_path(raw_path: str) -> str:
    """Strip .codeatlas/repos/<uuid>/ prefix so users see clean relative paths."""
    marker = ".codeatlas" + ("\\repos\\" if "\\" in raw_path else "/repos/")
    idx = raw_path.find(marker)
    if idx == -1:
        return raw_path
    after_marker = raw_path[idx + len(marker):]
    # Skip the UUID segment (next path component)
    sep = "\\" if "\\" in after_marker else "/"
    parts = after_marker.split(sep, 1)
    if len(parts) > 1:
        return parts[1].replace("\\", "/")
    return after_marker.replace("\\", "/")


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[str]
    reasoning_steps: list[str]


class AnswerService:
    def __init__(
        self,
        retriever: CodeRetriever,
        embedder: EmbeddingService,
        llm: BaseChatModel | None = None,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._llm = llm
        self._logger = logging.getLogger(__name__)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior engineer. Use provided code snippets to answer. "
                    "Cite file paths and line ranges from the snippets.",
                ),
                ("human", "{question}\n\nContext:\n{context}"),
            ]
        )

    def answer(self, repo_id: str, question: str, top_k: int = 5) -> GroundedAnswer:
        self._logger.info("Answering question for repo %s", repo_id)
        query_vector = self._embedder.embed_query(question)
        records = self._retriever.search(repo_id, query_vector, max(top_k, 10))
        records = self._rerank(question, records)[:top_k]
        self._logger.info("Retrieved %s records for repo %s", len(records), repo_id)
        citations = [self._citation_text(record) for record in records]
        answer_lines = self._format_answer(question, records)
        reasoning = [
            f"Embedded query for repo_id={repo_id}.",
            f"Retrieved {len(records)} records.",
        ]
        return GroundedAnswer(
            answer="\n".join(answer_lines),
            citations=citations,
            reasoning_steps=reasoning,
        )

    def _format_answer(self, question: str, records: list[EmbeddingRecord]) -> list[str]:
        if not records:
            return [
                "No relevant code locations found.",
                "Try a different query or re-run analysis.",
            ]

        if self._llm is None:
            lines = ["Top relevant locations:"]
            for record in records:
                lines.append(self._format_record(record))
            return lines

        context = self._build_context(records)
        try:
            chain = self._prompt | self._llm
            response = chain.invoke({"question": question, "context": context})
            return [response.content]
        except Exception as exc:
            self._logger.warning("LLM answer failed, falling back: %s", exc)
            lines = ["Top relevant locations:"]
            for record in records:
                lines.append(self._format_record(record))
            return lines

    def _format_record(self, record: EmbeddingRecord) -> str:
        path = _clean_display_path(record.metadata.get("path", ""))
        if record.scope == "function":
            signature = record.metadata.get("signature", "")
            return f"- function: {path} :: {signature}"
        language = record.metadata.get("language", "")
        label = f"{language} file" if language else "file"
        return f"- {label}: {path}"

    def _citation_text(self, record: EmbeddingRecord) -> str:
        display_path = _clean_display_path(record.metadata.get("path", ""))
        snippet = _record_snippet(record)
        snippet = " ".join(snippet.splitlines()[:2]).strip()
        if len(snippet) > 200:
            snippet = f"{snippet[:200]}..."
        line_range = _line_range(record)
        prefix = f"{display_path}{line_range}"
        return f"{prefix} | {snippet}" if snippet else prefix

    def _build_context(self, records: list[EmbeddingRecord]) -> str:
        chunks: list[str] = []
        for record in records:
            display_path = _clean_display_path(record.metadata.get("path", ""))
            snippet = _record_snippet(record)
            chunks.append(f"[{display_path}]\n{snippet}")
        return "\n\n".join(chunks)

    def _rerank(self, query: str, records: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return records
        scored: list[tuple[float, EmbeddingRecord]] = []
        for record in records:
            snippet = _record_snippet(record)
            score = _overlap_score(query_tokens, snippet)
            # Penalize __init__.py files with little content
            path = record.metadata.get("path", "")
            if path.endswith("__init__.py") and len(snippet.strip()) < 50:
                score *= 0.1
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored]


def _record_snippet(record: EmbeddingRecord) -> str:
    path = record.metadata.get("path")
    if not path:
        return ""
    if record.scope == "function":
        start = _parse_int(record.metadata.get("start_line"))
        end = _parse_int(record.metadata.get("end_line"))
        if start and end:
            return _read_snippet(Path(path), start, end)
    return _safe_read(Path(path))


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_snippet(path: Path, start_line: int, end_line: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = max(start_line - 1, 0)
    end = max(end_line, start)
    return "\n".join(lines[start:end])


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.lower())
    return set(tokens)


def _overlap_score(query_tokens: set[str], text: str) -> float:
    if not text:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(query_tokens & tokens) / len(query_tokens)


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _line_range(record: EmbeddingRecord) -> str:
    start = _parse_int(record.metadata.get("start_line"))
    end = _parse_int(record.metadata.get("end_line"))
    if start is None or end is None:
        return ""
    return f" (lines {start}-{end})"
