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


class IndexingTimeoutError(Exception):
    """Indexing ran past its time limit; nothing was stored."""


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
        self._logger = logging.getLogger(__name__)

    @property
    def index_format(self) -> str:
        """Everything that decides an index's vectors. Saved with each repo; the index cache needs a match."""
        return f"{self._embedder.signature()}|max_chars={self._max_chars}|prefix={int(self._prefix_metadata)}"

    def has_index(self, repo_id: str) -> bool:
        return self._retriever.has_index(repo_id)

    def discard(self, repo_id: str) -> None:
        """Remove any stored index for a repo (used to clean up after a failed analysis)."""
        self._retriever.remove(repo_id)

    def index_repository(self, repository: Repository, parsed_repo: ParsedRepository) -> None:
        self._logger.info("Indexing repository %s", repository.repo_id)
        deadline = None if self._timeout_seconds is None else self._clock() + self._timeout_seconds

        def check_deadline() -> None:
            if deadline is not None and self._clock() > deadline:
                raise IndexingTimeoutError(f"Indexing took longer than {self._timeout_seconds:g} seconds.")

        with stage("read_files"):
            documents, records = self._collect(parsed_repo, check_deadline)
            root = Path(repository.root_path)
            documents = [self._prepare(doc, record, root) for doc, record in zip(documents, records)]

        # Embed in batches so the time limit is checked while embedding, not only before and after.
        embeddings: list[list[float]] = []
        with stage("embed"):
            for start in range(0, len(documents), self._batch_size):
                check_deadline()
                embeddings.extend(self._embedder.embed_texts(documents[start : start + self._batch_size]))
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
            self._retriever.index(repository.repo_id, indexed_records)
        self._logger.info("Indexed %s records for repo %s", len(indexed_records), repository.repo_id)

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

    @staticmethod
    def _collect(
        parsed_repo: ParsedRepository, check_deadline: Callable[[], None]
    ) -> tuple[list[str], list[EmbeddingRecord]]:
        documents: list[str] = []
        records: list[EmbeddingRecord] = []

        for source_file in parsed_repo.files:
            check_deadline()
            content = _safe_read(Path(source_file.path))
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
            documents.append(snippet)
            records.append(
                EmbeddingRecord(
                    record_id=f"{function.file_path}:{function.start_line}-{function.end_line}",
                    scope="function",
                    vector=[],
                    metadata={
                        "path": function.file_path,
                        "name": function.name,
                        "signature": function.signature,
                        "start_line": str(function.start_line),
                        "end_line": str(function.end_line),
                    },
                )
            )

        return documents, records


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
