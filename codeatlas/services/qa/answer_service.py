import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field, replace
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
from codeatlas.observability.metrics import RETRIEVAL_FALLBACK
from codeatlas.observability.timing import stage
from codeatlas.services.llm.quota import classify_quota, quota_answer
from codeatlas.services.retrieval.bm25 import DEFAULT_MAX_INDEXES as DEFAULT_BM25_MAX_INDEXES
from codeatlas.services.retrieval.bm25 import Bm25Index, Bm25Settings
from codeatlas.services.retrieval.embedding import EmbeddingService, EmbeddingUnavailableError, fallback_note
from codeatlas.services.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from codeatlas.services.retrieval.hash_embedder import STOPWORDS, split_identifier
from codeatlas.services.retrieval.indexing import embedder_mismatch
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
    # A second, lexical retriever over every record, fused into the vector candidates by
    # reciprocal rank fusion before the rerank runs. Off by default; with it off the BM25 index
    # is never built and the candidate list is exactly what it was.
    bm25_enabled: bool = False
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    # Damping constant for the fusion; see services/retrieval/fusion.py.
    bm25_rrf_k: int = DEFAULT_RRF_K
    # How far down each list is read before fusing. 0 means "as deep as the vector pool".
    bm25_depth: int = 0
    # How many lexical indexes may be held in memory at once, across every repository and every
    # retriever this service searches. A lexical index is built on the first question about a
    # repository and kept, so without a cap this grows with the number of repositories a visitor
    # has analysed and nothing ever gives it back. Over the cap, the least recently *used* index
    # is dropped and rebuilt the next time it is needed: rebuilding is local arithmetic over
    # records that are already in memory, so the cost of being wrong about the cap is time, not
    # a wrong answer. Not a per-index setting - the cache is one cache, so the cap is one number.
    bm25_max_indexes: int = DEFAULT_BM25_MAX_INDEXES


# The settings a second index is allowed to disagree with the first about. Everything outside
# this list is either applied before the index is chosen (drop_stopwords, which also signs the
# cloudflare embedder, so the two indexes could not differ on it even if the code let them) or
# describes the candidate pool, which both indexes fill to the same depth.
PER_INDEX_FIELDS = ("bm25_enabled", "bm25_k1", "bm25_b", "bm25_rrf_k", "bm25_depth")


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[str]
    reasoning_steps: list[str]
    # The code the answer was built from, in the same order as `citations`. Later agents use
    # these instead of searching again, so what they read matches what the user is shown.
    snippets: list[Snippet] = field(default_factory=list)
    # One sentence for the visitor when the fallback index answered instead of the semantic
    # one. Empty on the ordinary path, which is every path when the provider cannot fail.
    note: str = ""


class AnswerService:
    def __init__(
        self,
        retriever: CodeRetriever,
        embedder: EmbeddingService,
        llm: BaseChatModel | None = None,
        settings: RetrievalSettings | None = None,
        snippet_max_chars: int = DEFAULT_SNIPPET_MAX_CHARS,
        total_max_chars: int = DEFAULT_TOTAL_MAX_CHARS,
        fallback_retriever: CodeRetriever | None = None,
        fallback_embedder: EmbeddingService | None = None,
        fallback_settings: RetrievalSettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._embedder = embedder
        # The index that answers when the first one cannot: present only where the first one is
        # able to fail, which today means the Cloudflare provider. Both are needed or neither,
        # since an index and the embedder that built it are one thing.
        self._fallback_retriever = fallback_retriever
        self._fallback_embedder = fallback_embedder
        self._llm = llm
        self._settings = settings or RetrievalSettings()
        # How the fallback index is searched. It is the primary settings with only the fields in
        # PER_INDEX_FIELDS taken from the caller, rather than the caller's object wholesale: a
        # setting that is applied before the index is even chosen cannot be allowed to differ
        # here, because by then it has already been applied.
        self._fallback_settings = (
            self._settings
            if fallback_settings is None
            else replace(self._settings, **{f: getattr(fallback_settings, f) for f in PER_INDEX_FIELDS})
        )
        self._snippet_max_chars = snippet_max_chars
        self._total_max_chars = total_max_chars
        # One BM25 index per repository *per retriever*, built on first use and rebuilt when the
        # repository is indexed again. Keyed by the retriever as well as the repo because under
        # the cloudflare provider one repo_id has two indexes holding different records, and by
        # the record list's identity because reindexing replaces that list: an index built from
        # the old records would answer about code that is no longer there.
        #
        # An OrderedDict in least-recently-used order, newest last, capped at
        # `settings.bm25_max_indexes`. The cap is read from the primary settings and not the
        # fallback's: there is one cache, shared by both retrievers, so there is one cap.
        self._bm25: OrderedDict[tuple[int, str], tuple[int, Bm25Index]] = OrderedDict()
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
        return self.retrieve_with_note(repo_id, question, top_k)[0]

    def retrieve_with_note(self, repo_id: str, question: str, top_k: int = 5) -> tuple[list[EmbeddingRecord], str]:
        """The records, and the sentence to show the visitor when the fallback index answered.

        The order here is the point. Which index *can* answer is decided first, from what is on
        disk; whether the chosen embedder *works* is only knowable by calling it, and a provider
        that reaches an API fails at exactly that moment, with the visitor already waiting. Both
        outcomes land in the same place: the fallback index, and a note saying so.
        """
        settings = self._settings
        if settings.drop_stopwords:
            question = _without_stopwords(question)

        retriever, embedder, reason = self._choose(repo_id)
        if retriever is None:
            return [], ""
        if retriever is self._fallback_retriever:
            settings = self._fallback_settings

        try:
            with stage("embed_query"):
                query_vector = embedder.embed_query(question)
        except EmbeddingUnavailableError as exc:
            # The reason only. A detail may carry whatever the provider wrote into it, and the
            # question itself is right here in scope - neither belongs in a log line.
            self._logger.warning(
                "Semantic search unavailable (%s) for repo %s; using the fallback index", exc.reason, repo_id
            )
            fallback = self._usable_fallback(repo_id)
            if fallback is None:
                return [], ""
            retriever, embedder = fallback
            settings = self._fallback_settings
            reason = exc.reason
            with stage("embed_query"):
                query_vector = embedder.embed_query(question)

        if reason:
            # The reason is one of a fixed list, so the label can never hold anything a visitor
            # typed, a path, or a provider's error text.
            RETRIEVAL_FALLBACK.labels(reason=reason).inc()
        depth = max(top_k, settings.candidates)
        with stage("search"):
            records = retriever.search(repo_id, query_vector, depth)
            if settings.bm25_enabled:
                records = self._fuse_lexical(retriever, repo_id, question, records, depth, settings)
        if settings.skip_tests:
            records = [r for r in records if not _is_test_path(r.metadata.get("path", ""))]
        # Reranking reads each candidate's snippet from disk.
        with stage("rerank"):
            ranked = _dedupe_windows(self._rerank(question, records, query_vector))[:top_k]
        return ranked, fallback_note(reason) if reason else ""

    def _fuse_lexical(
        self,
        retriever: CodeRetriever,
        repo_id: str,
        question: str,
        vector_records: list[EmbeddingRecord],
        depth: int,
        settings: RetrievalSettings,
    ) -> list[EmbeddingRecord]:
        """The vector candidates fused with BM25's, cut back to the same pool depth.

        The pool stays the same size, so nothing downstream sees a longer list than it used to:
        what changes is *which* records are in it. The fusion is symmetric, so a record both
        retrievers rank highly rises, and a record only BM25 found can still enter the pool.

        `settings` is passed in rather than read from `self`, because which index is being
        searched decides whether it is fused at all: see PER_INDEX_FIELDS.
        """
        index = self._bm25_index(retriever, repo_id, settings)
        if index is None or len(index) == 0:
            return vector_records
        lexical = index.search(question, depth if settings.bm25_depth == 0 else settings.bm25_depth)
        if not lexical:
            return vector_records
        fused = reciprocal_rank_fusion([vector_records, lexical], k=settings.bm25_rrf_k)
        return fused[:depth]

    def _bm25_index(self, retriever: CodeRetriever, repo_id: str, settings: RetrievalSettings) -> Bm25Index | None:
        """This repository's lexical index, built once and reused until the repository is reindexed."""
        records = retriever.records(repo_id)
        if not records:
            return None
        key = (id(retriever), repo_id)
        cached = self._bm25.get(key)
        if cached is not None and cached[0] == id(records):
            # Used, so it goes to the newest end: the cap evicts what has been unused longest,
            # not what was built longest ago. A repository asked about every minute must not be
            # thrown away because it was the first one analysed.
            self._bm25.move_to_end(key)
            return cached[1]
        # The full text of each record, uncapped - the same text the rerank reads. BM25 is local
        # and free, so there is no reason to show it only the opening of a long file.
        index = Bm25Index(
            records=list(records),
            texts=[_record_text(record) for record in records],
            settings=Bm25Settings(k1=settings.bm25_k1, b=settings.bm25_b),
        )
        self._bm25[key] = (id(records), index)
        self._bm25.move_to_end(key)
        self._logger.info("Built a BM25 index of %s records for repo %s", len(index), repo_id)
        self._evict_lexical_indexes(self._settings.bm25_max_indexes)
        return index

    def _evict_lexical_indexes(self, cap: int) -> None:
        """Drop the least recently used lexical indexes until the cache is within the cap.

        A whole index at a time, because half of one answers nothing. What is dropped is the
        lexical index alone: the records it pointed at are the retriever's and stay exactly
        where they were, so a visitor whose repository is evicted waits for a rebuild and is
        never told anything different.
        """
        while len(self._bm25) > max(cap, 1):
            evicted, _ = self._bm25.popitem(last=False)
            self._logger.info(
                "Evicted the BM25 index for repo %s; %s of %s kept. It is rebuilt on the next question.",
                evicted[1],
                len(self._bm25),
                cap,
            )

    def _choose(self, repo_id: str) -> tuple[CodeRetriever | None, EmbeddingService, str]:
        """Which index to search, decided before anything is called."""
        stored = self._retriever.stored_embedder(repo_id)
        mismatched = embedder_mismatch(stored, self._embedder)
        if not mismatched and self._retriever.has_index(repo_id):
            return self._retriever, self._embedder, ""

        fallback = self._usable_fallback(repo_id)
        if fallback is not None:
            # A repository over the document cap has only the fallback index, and so does one
            # whose semantic index was built by a different embedder. The visitor sees one note
            # for both, because from where they are sitting it is one situation.
            self._logger.info("No usable semantic index for repo %s; using the fallback index", repo_id)
            return fallback[0], fallback[1], "unindexed"

        if mismatched:
            # Returning nothing is honest; the alternative is confident nonsense (see embedder_mismatch).
            self._logger.warning(
                "Index for repo %s was built by embedder %r, not %r; refusing to search it. Analyze it again.",
                repo_id,
                stored,
                self._embedder.signature(),
            )
            return None, self._embedder, ""
        return self._retriever, self._embedder, ""

    def _usable_fallback(self, repo_id: str) -> tuple[CodeRetriever, EmbeddingService] | None:
        """The fallback pair, if there is one with an index for this repository that it can search."""
        retriever, embedder = self._fallback_retriever, self._fallback_embedder
        if retriever is None or embedder is None or not retriever.has_index(repo_id):
            return None
        if embedder_mismatch(retriever.stored_embedder(repo_id), embedder):
            return None
        return retriever, embedder

    def snippets_for(self, records: list[EmbeddingRecord], total_max_chars: int | None = None) -> list[Snippet]:
        """The code these records point at, each clipped, and only as many as the budget allows.

        Public because grounded chat builds its prompt and its citations from one list of
        snippets, and reaching into a private method to get it would let the two drift apart.
        """
        budget = self._total_max_chars if total_max_chars is None else total_max_chars
        return fit_snippets([self._snippet_for(record) for record in records], budget)

    def answer(self, repo_id: str, question: str, top_k: int = 5) -> GroundedAnswer:
        self._logger.info("Answering question for repo %s", repo_id)
        records, note = self.retrieve_with_note(repo_id, question, top_k)
        self._logger.info("Retrieved %s records for repo %s", len(records), repo_id)
        # Read each record's code once, capped, and keep only what fits the context budget.
        # Citations describe exactly these snippets, so the answer and the citations agree.
        snippets = self.snippets_for(records)
        citations = [citation_text(snippet) for snippet in snippets]
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
            note=note,
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


def _dedupe_windows(records: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
    """One result per function: the best-ranked window of it, in the order ranking put them.

    Under the "structured" record format a long function becomes several overlapping windows,
    so without this a visitor could be shown three slices of one function and nothing else.
    Records written under "whole" carry no `function_id`, so for them this is provably a
    no-op - the list comes back in the same order, unchanged, and a test says so.
    """
    seen: set[str] = set()
    kept: list[EmbeddingRecord] = []
    for record in records:
        function_id = record.metadata.get("function_id")
        if function_id is not None:
            if function_id in seen:
                continue
            seen.add(function_id)
        kept.append(record)
    return kept


def _location_list(snippets: list[Snippet]) -> list[str]:
    """What the caller gets when there is no model: the locations, which are still true."""
    return ["Top relevant locations:"] + [f"- {snippet.location}" for snippet in snippets]


def citation_text(snippet: Snippet) -> str:
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
