import logging
import re
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.observability.agent_metrics import (
    FALLBACK_LOCATIONS,
    classify,
    current_agent,
    record_failure,
)
from codeatlas.observability.timing import stage
from codeatlas.services.llm.quota import classify_quota, quota_answer
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.hash_embedder import STOPWORDS, split_identifier
from codeatlas.services.retrieval.interfaces import CodeRetriever
from codeatlas.services.retrieval.snippets import (
    DEFAULT_SNIPPET_MAX_CHARS,
    DEFAULT_TOTAL_MAX_CHARS,
    Snippet,
    clip_snippet,
    fit_snippets,
    render_snippets,
)


def _clean_display_path(raw_path: str) -> str:
    """Strip .codeatlas/repos/<uuid>/ prefix so users see clean relative paths."""
    marker = ".codeatlas" + ("\\repos\\" if "\\" in raw_path else "/repos/")
    idx = raw_path.find(marker)
    if idx == -1:
        return raw_path
    after_marker = raw_path[idx + len(marker) :]
    # Skip the UUID segment (next path component)
    sep = "\\" if "\\" in after_marker else "/"
    parts = after_marker.split(sep, 1)
    if len(parts) > 1:
        return parts[1].replace("\\", "/")
    return after_marker.replace("\\", "/")


@dataclass(frozen=True)
class RetrievalSettings:
    """How a question's records are chosen. The defaults are what the chat uses."""

    # Records taken from the vector search before reranking (at least top_k).
    candidates: int = 10
    # Rerank score = weight * keyword overlap + (1 - weight) * vector similarity. 1.0 orders by
    # keyword overlap alone (vector order only breaks ties); 0.0 keeps the vector order.
    rerank_weight: float = 1.0
    # Match question words against the parts of identifiers (verify_signature -> verify, signature).
    rerank_subtokens: bool = False
    # Leave common English words ("how", "does", "the") out of the question for search and rerank.
    drop_stopwords: bool = False
    # Drop records from test files (test_*.py, *_test.py, conftest.py, or under a tests/ folder).
    skip_tests: bool = False


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[str]
    reasoning_steps: list[str]
    # The code the answer was built from, in the same order as `citations`. Later agents use
    # these instead of searching again, so what they read matches what the user is shown.
    snippets: list[Snippet] = field(default_factory=list)


class AnswerService:
    def __init__(
        self,
        retriever: CodeRetriever,
        embedder: EmbeddingService,
        llm: BaseChatModel | None = None,
        settings: RetrievalSettings | None = None,
        snippet_max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
        total_max_chars: int = DEFAULT_TOTAL_MAX_CHARS,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        self._llm = llm
        self._settings = settings or RetrievalSettings()
        self._snippet_max_chars = snippet_max_chars
        self._total_max_chars = total_max_chars
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

    def retrieve(self, repo_id: str, question: str, top_k: int = 5) -> list[EmbeddingRecord]:
        """The records a question's answer is built from, best first. The retrieval eval calls this too."""
        settings = self._settings
        if settings.drop_stopwords:
            question = _without_stopwords(question)
        with stage("embed_query"):
            query_vector = self._embedder.embed_query(question)
        with stage("search"):
            records = self._retriever.search(repo_id, query_vector, max(top_k, settings.candidates))
        if settings.skip_tests:
            records = [r for r in records if not _is_test_path(r.metadata.get("path", ""))]
        # Reranking reads each candidate's snippet from disk.
        with stage("rerank"):
            return self._rerank(question, records, query_vector)[:top_k]

    def answer(self, repo_id: str, question: str, top_k: int = 5) -> GroundedAnswer:
        self._logger.info("Answering question for repo %s", repo_id)
        records = self.retrieve(repo_id, question, top_k)
        self._logger.info("Retrieved %s records for repo %s", len(records), repo_id)
        # Read each record's code once, capped, and keep only what fits the context budget.
        # Citations describe exactly these snippets, so the answer and the citations agree.
        snippets = fit_snippets(
            [self._snippet_for(record) for record in records],
            self._total_max_chars,
        )
        citations = [_citation_text(snippet) for snippet in snippets]
        answer_lines = self._format_answer(question, snippets)
        reasoning = [
            f"Embedded query for repo_id={repo_id}.",
            f"Retrieved {len(records)} records, kept {len(snippets)} snippets within the context budget.",
        ]
        return GroundedAnswer(
            answer="\n".join(answer_lines),
            citations=citations,
            reasoning_steps=reasoning,
            snippets=snippets,
        )

    def _format_answer(self, question: str, snippets: list[Snippet]) -> list[str]:
        if not snippets:
            return [
                "No relevant code locations found.",
                "Try a different query or re-run analysis.",
            ]

        if self._llm is None:
            return _location_list(snippets)

        try:
            chain = self._prompt | self._llm
            with stage("llm"):
                response = chain.invoke({"question": question, "context": render_snippets(snippets)})
            return [response.content]
        except Exception as exc:
            # The locations are still true when the model is unavailable or refuses the request,
            # so the caller keeps the snippets either way.
            self._logger.warning("LLM answer failed, falling back to the location list: %s", exc)
            # Two facts worth separating: why the provider refused, and that the answer the
            # user gets is the degraded one.
            agent = current_agent()
            record_failure(agent, classify(exc))
            record_failure(agent, FALLBACK_LOCATIONS)
            # A refused call is told to the visitor as such; the locations go underneath it,
            # under a heading, so the list is never read as the answer.
            kind = classify_quota(exc)
            if kind:
                return [quota_answer(kind, snippets)]
            return _location_list(snippets)

    def _snippet_for(self, record: EmbeddingRecord) -> Snippet:
        """The code a record points at, clipped, with the line range actually kept."""
        display_path = _clean_display_path(record.metadata.get("path", ""))
        path = record.metadata.get("path")
        if not path:
            return Snippet(path=display_path, start_line=0, end_line=0, text="")

        start = _parse_int(record.metadata.get("start_line"))
        end = _parse_int(record.metadata.get("end_line"))
        if record.scope == "function" and start and end:
            text = _read_snippet(Path(path), start, end)
            return clip_snippet(display_path, text, start, end, self._snippet_max_chars)

        # A whole file: read it, but never send more than one snippet's worth. Sending whole
        # files is what pushed a single request past the model's per-minute token allowance.
        text = _safe_read(Path(path))
        line_count = len(text.splitlines()) if text else 0
        return clip_snippet(display_path, text, 1, line_count, self._snippet_max_chars)

    def _rerank(self, query: str, records: list[EmbeddingRecord], query_vector: list[float]) -> list[EmbeddingRecord]:
        subtokens = self._settings.rerank_subtokens
        weight = self._settings.rerank_weight
        query_tokens = _tokenize(query, subtokens)
        if not query_tokens:
            return records
        scored: list[tuple[float, EmbeddingRecord]] = []
        for record in records:
            snippet = _record_text(record)
            score = _overlap_score(query_tokens, snippet, subtokens)
            if weight != 1.0:
                score = weight * score + (1 - weight) * _cosine(query_vector, record.vector)
            # Penalize __init__.py files with little content
            path = record.metadata.get("path", "")
            if path.endswith("__init__.py") and len(snippet.strip()) < 50:
                score *= 0.1
            scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored]


def _location_list(snippets: list[Snippet]) -> list[str]:
    """What the caller gets when there is no model: the locations, which are still true."""
    return ["Top relevant locations:"] + [f"- {snippet.location}" for snippet in snippets]


def _citation_text(snippet: Snippet) -> str:
    """One citation line: the snippet's own location and its first two lines."""
    preview = " ".join(snippet.text.splitlines()[:2]).strip()
    if len(preview) > 200:
        preview = f"{preview[:200]}..."
    return f"{snippet.location} | {preview}" if preview else snippet.location


def _record_text(record: EmbeddingRecord) -> str:
    """A record's code, uncapped; used for reranking, which is local and never sent anywhere."""
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


def _tokenize(text: str, subtokens: bool = False) -> set[str]:
    if not subtokens:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text.lower()))
    tokens: set[str] = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text):
        tokens.add(token.lower())
        tokens.update(split_identifier(token))
    return tokens


def _overlap_score(query_tokens: set[str], text: str, subtokens: bool = False) -> float:
    if not text:
        return 0.0
    tokens = _tokenize(text, subtokens)
    if not tokens:
        return 0.0
    return len(query_tokens & tokens) / len(query_tokens)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = sqrt(sum(x * x for x in a)) * sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _without_stopwords(question: str) -> str:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
    kept = [word for word in words if word.lower() not in STOPWORDS]
    return " ".join(kept) if kept else question


def _is_test_path(path: str) -> bool:
    parts = Path(path.replace("\\", "/")).parts
    name = parts[-1] if parts else ""
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or any(part in ("tests", "test") for part in parts[:-1])
    )


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
