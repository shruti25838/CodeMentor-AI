"""Two indexes per repository: the semantic one, and the hash one that always works.

A semantic provider that calls an API can fail in ways a local embedder cannot - a spent
allowance, a timeout, an outage - and those failures arrive while a visitor is waiting. The
answer to that is not more retries; it is a second index, built at the same time, from the
same records, that needs nothing but this process.

So the hash index is built **first and always**, and the semantic one is attempted after it.
If the semantic index cannot be built the repository is still fully usable, which is why the
order is that way round and not the other: a repository is never left with only the index that
might not be there when a question arrives.

What this class does **not** do is decide which index answers a question. That is
AnswerService, because by then the interesting failures are the ones that happen at question
time, not at index time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.observability.metrics import SEMANTIC_INDEX_NEURONS, SEMANTIC_INDEX_OUTCOMES
from codeatlas.services.retrieval.embedding import EmbeddingUnavailableError
from codeatlas.services.retrieval.indexing import CodeIndexService, IndexingTimeoutError

FALLBACK_NAME = "hash"

# How the attempt at a semantic index ended. These name outcomes for a log line and a metric,
# not sentences for a visitor: what a visitor is told is decided at question time, from what
# turned out to be on disk, and is deliberately a shorter list.
OK = "ok"
BUDGET = "budget"
FAILURE = "failure"
BREAKER = "breaker"
TIMEOUT = "timeout"
CAP = "cap"
OUTCOMES = (OK, BUDGET, FAILURE, BREAKER, TIMEOUT, CAP)

# EmbeddingUnavailableError reasons, which describe one call, mapped onto outcomes, which
# describe a whole indexing attempt.
_OUTCOME_FOR_REASON = {"budget": BUDGET, "circuit": BREAKER, "api": FAILURE}


@dataclass(frozen=True)
class _Usage:
    """What the semantic provider reported spending, flattened so this module needs no import of it."""

    calls: int = 0
    seconds: float = 0.0
    estimated_neurons: float = 0.0


class DualIndexService:
    def __init__(
        self,
        semantic: CodeIndexService,
        fallback: CodeIndexService,
        semantic_name: str,
        max_documents: int,
        fallback_name: str = FALLBACK_NAME,
        usage: Callable[[], object] | None = None,
    ) -> None:
        self._semantic = semantic
        self._fallback = fallback
        self._semantic_name = semantic_name
        self._fallback_name = fallback_name
        # A repository bigger than this gets the fallback index only. The cap is on documents,
        # not on files, because a file with forty functions is forty documents to embed.
        self._max_documents = max_documents
        # What the semantic provider spent, asked for once per analysis. None for a provider
        # that has nothing to report, which is every provider that does not call an API.
        self._usage = usage
        self._logger = logging.getLogger(__name__)

    @property
    def index_format(self) -> str:
        """Both formats, so the index cache only reuses a repository that has both as they are now."""
        return f"{self._semantic.index_format}+{self._fallback_name}={self._fallback.index_format}"

    @property
    def semantic_name(self) -> str:
        return self._semantic_name

    def has_index(self, repo_id: str) -> bool:
        """Whether this repository can be answered at all - that is, whether the fallback is there."""
        return self._fallback.has_index(repo_id)

    def has_semantic_index(self, repo_id: str) -> bool:
        return self._semantic.has_index(repo_id)

    def discard(self, repo_id: str) -> None:
        self._semantic.discard(repo_id)
        self._fallback.discard(repo_id)

    def document_count(self, parsed_repo: ParsedRepository) -> int:
        return len(parsed_repo.files) + len(parsed_repo.functions)

    def over_cap(self, parsed_repo: ParsedRepository) -> bool:
        return self.document_count(parsed_repo) > self._max_documents

    def index_repository(self, repository: Repository, parsed_repo: ParsedRepository) -> tuple[str, ...]:
        """Build both indexes and return the names of the ones that exist afterwards.

        The fallback is built first and its failure is the caller's problem, as it always was.
        A semantic failure is not: it is logged with its reason and the repository goes on
        working with one index, because a repository that is harder to search is a far better
        outcome than a repository that could not be analyzed at all.
        """
        self._fallback.index_repository(repository, parsed_repo)
        built = [self._fallback_name]
        documents = self.document_count(parsed_repo)
        self._take_usage()  # anything left over from an earlier analysis is not this one's

        if self.over_cap(parsed_repo):
            self._report(repository.repo_id, CAP, documents)
            return tuple(built)

        try:
            self._semantic.index_repository(repository, parsed_repo)
        except EmbeddingUnavailableError as exc:
            # The reason, never the detail: a detail can carry whatever the provider put in it.
            self._semantic.discard(repository.repo_id)
            self._report(repository.repo_id, _OUTCOME_FOR_REASON.get(exc.reason, FAILURE), documents)
            return tuple(built)
        except IndexingTimeoutError:
            # Deliberately not fatal here, unlike a timeout on the only index there is. The
            # repository already has a working index; taking too long over the second one is a
            # reason to stop building it, not a reason to throw away the analysis and tell a
            # visitor their repository could not be read.
            self._semantic.discard(repository.repo_id)
            self._report(repository.repo_id, TIMEOUT, documents)
            return tuple(built)

        self._report(repository.repo_id, OK, documents)
        return (self._semantic_name, *built)

    def _report(self, repo_id: str, outcome: str, documents: int) -> None:
        """One line per analysis, and the same facts as metrics.

        Counts, seconds, an estimate and one word from a fixed list. No document, no question,
        no URL, no credential - and the repository id, which is the identifier the rest of the
        logs already use and is not a name or an address.
        """
        usage = self._take_usage()
        self._logger.info(
            "Semantic index for repo %s: outcome=%s documents=%d calls=%d api_seconds=%.1f estimated_neurons=%.2f",
            repo_id,
            outcome,
            documents,
            usage.calls,
            usage.seconds,
            usage.estimated_neurons,
        )
        SEMANTIC_INDEX_OUTCOMES.labels(outcome=outcome).inc()
        if usage.estimated_neurons:
            SEMANTIC_INDEX_NEURONS.inc(usage.estimated_neurons)

    def _take_usage(self) -> _Usage:
        if self._usage is None:
            return _Usage()
        taken = self._usage()
        return _Usage(
            calls=int(getattr(taken, "calls", 0)),
            seconds=float(getattr(taken, "seconds", 0.0)),
            estimated_neurons=float(getattr(taken, "estimated_neurons", 0.0)),
        )
