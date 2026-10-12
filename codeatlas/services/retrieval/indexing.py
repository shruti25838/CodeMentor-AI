import contextlib
import logging
import time
from collections.abc import Callable
from pathlib import Path

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.observability.timing import stage
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.interfaces import CodeRetriever
from codeatlas.services.retrieval.record_format import (
    DEFAULT_WINDOW_CHARS,
    DEFAULT_WINDOW_OVERLAP,
    STRUCTURED,
    WHOLE,
    file_header,
    windows,
)


class IndexingTimeoutError(Exception):
    """Indexing ran past its time limit; nothing was stored."""


def embedder_mismatch(stored: str | None, embedder: EmbeddingService) -> bool:
    """True when an index is known to have been built by a different embedder than this one.

    Vectors from two embedders are not comparable, and nothing about a FAISS index makes that
    visible: the hash embedder and bge-small are both 384 wide, so searching one with the other's
    query vector returns confident nonsense rather than an error.

    Two cases are deliberately not a mismatch. None means the retriever does not record the
    embedder at all, and "" means the index was written before it was recorded; neither is
    evidence of a different embedder, and treating them as one would blank out every index
    written before this guard existed.
    """
    return bool(stored) and stored != embedder.signature()


class CodeIndexService:
    def __init__(
        self,
        embedder: EmbeddingService,
        retriever: CodeRetriever,
        timeout_seconds: float | None = None,
        batch_size: int = 64,
        clock: Callable[[], float] = time.monotonic,
        max_chars: int | None = None,
        prefix_metadata: bool = False,
        name: str = "",
        sort_by_length: bool = True,
        record_format: str = WHOLE,
        window_chars: int | None = None,
        window_overlap: int = DEFAULT_WINDOW_OVERLAP,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._timeout_seconds = timeout_seconds
        self._batch_size = batch_size
        self._clock = clock
        # Only the first max_chars characters of each file or function are embedded (None = all).
        self._max_chars = max_chars
        # Start each embedded text with the file path (and function name), so they can match too.
        self._prefix_metadata = prefix_metadata
        # What this index is called where a repository records which indexes it has; the
        # embedding provider's name. Empty means the index is not recorded under any name.
        self._name = name
        # Embed the shortest documents first, so each batch holds texts of similar length.
        self._sort_by_length = sort_by_length
        # What a record is: see services/retrieval/record_format.py. 'whole' is today's.
        self._record_format = record_format
        # The window a long function is cut into. Defaults to the per-document clip when one
        # is set, because a window larger than the clip would be embedded by its opening
        # alone and the whole point of windowing would be lost.
        self._window_chars = window_chars or max_chars or DEFAULT_WINDOW_CHARS
        self._window_overlap = window_overlap
        self._logger = logging.getLogger(__name__)

    @property
    def index_format(self) -> str:
        """Everything that decides an index's vectors. Saved with each repo; the index cache needs a match.

        The record format reaches this through the embedder's signature (see
        `record_format.with_record_format`), so it is here without being named twice.
        Under "whole" the signature is the one it has always been, so no stored index
        is invalidated."""
        return f"{self._embedder.signature()}|max_chars={self._max_chars}|prefix={int(self._prefix_metadata)}"

    def has_index(self, repo_id: str) -> bool:
        """True only for an index this service could search: built by an embedder with this signature."""
        return self._retriever.has_index(repo_id) and not embedder_mismatch(
            self._retriever.stored_embedder(repo_id), self._embedder
        )

    def discard(self, repo_id: str) -> None:
        """Remove any stored index for a repo (used to clean up after a failed analysis)."""
        self._retriever.remove(repo_id)

    @property
    def index_names(self) -> tuple[str, ...]:
        return (self._name,) if self._name else ()

    def index_repository(self, repository: Repository, parsed_repo: ParsedRepository) -> tuple[str, ...]:
        """Build this repository's index. Returns the names of the indexes that exist afterwards."""
        self._logger.info("Indexing repository %s", repository.repo_id)
        deadline = None if self._timeout_seconds is None else self._clock() + self._timeout_seconds

        def check_deadline() -> None:
            if deadline is not None and self._clock() > deadline:
                raise IndexingTimeoutError(f"Indexing took longer than {self._timeout_seconds:g} seconds.")

        with stage("read_files"):
            documents, records = self.documents(repository, parsed_repo, check_deadline)

        with stage("embed"):
            embeddings = self.embed_documents(documents, check_deadline)
        check_deadline()
        indexed_records: list[EmbeddingRecord] = []
        for record, vector in zip(records, embeddings):
            indexed_records.append(
                EmbeddingRecord(
                    record_id=record.record_id,
                    scope=record.scope,
                    vector=vector,
                    metadata=record.metadata,
                )
            )

        with stage("index_write"):
            self._retriever.index(repository.repo_id, indexed_records, self._embedder.signature())
        self._logger.info("Indexed %s records for repo %s", len(indexed_records), repository.repo_id)
        return self.index_names

    def embed_documents(
        self,
        documents: list[str],
        check_deadline: Callable[[], None] = lambda: None,
    ) -> list[list[float]]:
        """Vectors for these documents, returned in the order the documents were given in.

        Embedded in batches, so the time limit is checked while embedding rather than only
        before and after - and, unless told otherwise, **shortest first**.

        Length order is worth the bookkeeping because of how a batched provider is billed: a
        call is charged at the length of its longest text, so fifty short functions sent
        alongside one long file cost fifty long files. Sorting puts texts of similar length in
        the same call, which measured at about half the price of file order for a 600-document
        index, for identical vectors.

        "Identical" was checked rather than assumed: the same fifty documents embedded in file
        order and in length order came back with a **minimum cosine of 1.000000** between the
        two vectors of each document (scripts/index_cost_check.py --check-order). Had that come
        back lower, an index would have depended on the order its files happened to be read in,
        and this would not have been safe at any price.

        Nothing about the index changes: not the vectors, not the signature, not index_format.
        """
        order = (
            sorted(range(len(documents)), key=lambda i: len(documents[i]))
            if self._sort_by_length
            else list(range(len(documents)))
        )
        vectors: list[list[float]] = [[] for _ in documents]
        for start in range(0, len(order), self._batch_size):
            check_deadline()
            chunk = order[start : start + self._batch_size]
            embedded = self._embedder.embed_texts([documents[index] for index in chunk])
            for index, vector in zip(chunk, embedded):
                vectors[index] = vector
        return vectors

    def documents(
        self,
        repository: Repository,
        parsed_repo: ParsedRepository,
        check_deadline: Callable[[], None] = lambda: None,
    ) -> tuple[list[str], list[EmbeddingRecord]]:
        """Exactly the texts this service would embed, with the record each belongs to.

        Public because working out what a run will cost, or what the clip throws away, should
        not mean reaching into a private method or reimplementing the preparation and hoping it
        still matches.
        """
        documents, records = self._collect(parsed_repo, check_deadline)
        root = Path(repository.root_path)
        return [self._prepare(doc, record, root) for doc, record in zip(documents, records)], records

    def clipped_counts(self, parsed_repo: ParsedRepository) -> dict[str, dict[str, int]]:
        """Per scope ("file"/"function"): how many records there are, how many the clip cuts, and how much text.

        Measured on the source text, before the path prefix is added, because that is the part
        the clip is deciding about. With no clip set, nothing is cut and every count is zero.
        """
        documents, records = self._collect(parsed_repo, lambda: None)
        counts: dict[str, dict[str, int]] = {}
        for document, record in zip(documents, records):
            entry = counts.setdefault(record.scope, {"records": 0, "clipped": 0, "chars": 0, "chars_kept": 0})
            entry["records"] += 1
            entry["chars"] += len(document)
            kept = len(document) if self._max_chars is None else min(len(document), self._max_chars)
            entry["chars_kept"] += kept
            if self._max_chars is not None and len(document) > self._max_chars:
                entry["clipped"] += 1
        return counts

    def _prepare(self, document: str, record: EmbeddingRecord, root: Path) -> str:
        if self._max_chars is not None:
            document = document[: self._max_chars]
        if self._prefix_metadata:
            path = record.metadata.get("path", "")
            with contextlib.suppress(ValueError):
                path = Path(path).relative_to(root).as_posix()
            label = " ".join(v for v in (path, record.metadata.get("name", "")) if v)
            document = f"{label}\n{document}"
        return document

    def _collect(
        self, parsed_repo: ParsedRepository, check_deadline: Callable[[], None]
    ) -> tuple[list[str], list[EmbeddingRecord]]:
        documents: list[str] = []
        records: list[EmbeddingRecord] = []
        structured = self._record_format == STRUCTURED
        by_file: dict[str, list] = {}
        if structured:
            for function in parsed_repo.functions:
                by_file.setdefault(function.file_path, []).append(function)

        for source_file in parsed_repo.files:
            check_deadline()
            content = _safe_read(Path(source_file.path))
            if structured:
                signatures = [f.signature for f in sorted(by_file.get(source_file.path, []), key=_start_line)]
                content = file_header(content, source_file.path, signatures, self._window_chars)
            documents.append(content)
            records.append(
                EmbeddingRecord(
                    record_id=source_file.path,
                    scope="file",
                    vector=[],
                    metadata={"path": source_file.path, "language": source_file.language},
                )
            )

        for function in parsed_repo.functions:
            check_deadline()
            snippet = _read_snippet(Path(function.file_path), function.start_line, function.end_line)
            function_id = f"{function.file_path}:{function.start_line}-{function.end_line}"
            base = {
                "path": function.file_path,
                "name": function.name,
                "signature": function.signature,
            }
            if not structured:
                documents.append(snippet)
                records.append(
                    EmbeddingRecord(
                        record_id=function_id,
                        scope="function",
                        vector=[],
                        metadata={
                            **base,
                            "start_line": str(function.start_line),
                            "end_line": str(function.end_line),
                        },
                    )
                )
                continue

            cut = windows(snippet, function.start_line, self._window_chars, self._window_overlap)
            for number, window in enumerate(cut, start=1):
                documents.append(window.text)
                records.append(
                    EmbeddingRecord(
                        record_id=function_id if len(cut) == 1 else f"{function_id}#w{number}",
                        scope="function",
                        vector=[],
                        metadata={
                            **base,
                            "start_line": str(window.start_line),
                            "end_line": str(window.end_line),
                            # Which function this window belongs to, and how many there are.
                            # Ranking keeps one result per function_id; see AnswerService.
                            "function_id": function_id,
                            "window": f"{number}/{len(cut)}",
                        },
                    )
                )

        return documents, records


def _start_line(function) -> int:
    return function.start_line


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
