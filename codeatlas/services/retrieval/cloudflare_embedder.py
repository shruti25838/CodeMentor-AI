"""Semantic embeddings from Cloudflare Workers AI, on the free plan, with a way to stop.

Opt in with CODEATLAS_EMBEDDING_PROVIDER=cloudflare; "hash" remains the default. The model is
@cf/baai/bge-small-en-v1.5, 384 dimensions, reached over the REST endpoint with the stdlib
client in services/http/json_client.py. No new dependency, and no SDK.

What this provider is really about is not the embedding call - that part is ten lines - but
the three ways it has to be able to fail, because it is spending a free allowance that resets
once a day and cannot be topped up:

* **Budget.** Every call's estimated cost goes in the ledger first, and when today's total
  would pass the configured budget the provider refuses rather than finding out from
  Cloudflare. The estimate is a ceiling (see neuron_ledger), so this stops early by design.
* **Failure.** Any transport error, any non-2xx, any malformed body is a failure, with no
  attempt to tell one from another. A spent allowance is reported by a status nobody has
  documented, so matching on codes would be guessing; "it did not work" is the only reliable
  reading and it is the one the caller needs anyway.
* **Repetition.** After a few consecutive failures the circuit breaker opens and the provider
  stops calling for a few minutes. A free plan being hammered by a retry loop is how an
  allowance disappears into errors.

Every one of those raises CloudflareUnavailableError, whose `reason` the caller turns into the
short note the visitor sees while the hash index answers instead.

**Nothing here logs the text.** Not documents, not questions, not the URL (it carries the
account id), not a header. Counts, sizes, statuses and reasons only - which is also all the
ledger holds.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeatlas.services.http.json_client import JsonClientError, JsonResponse, post_json
from codeatlas.services.retrieval import neuron_ledger
from codeatlas.services.retrieval.embedding import EmbeddingService, EmbeddingUnavailableError

if TYPE_CHECKING:  # config imports this module's defaults, so the credentials type is a hint only
    from codeatlas.utils.config import CloudflareCredentials

DEFAULT_CLOUDFLARE_MODEL = "@cf/baai/bge-small-en-v1.5"
DEFAULT_CLOUDFLARE_POOLING = "mean"
POOLINGS = ("mean", "cls")
EXPECTED_DIMENSIONS = 384

# The probe showed the model reads about 512 tokens and silently drops the rest, which is
# roughly 2,000 characters of code; a document longer than this is embedded by its opening.
DEFAULT_CLOUDFLARE_MAX_CHARS = 2000
# A ceiling on what one repository may cost. 600 documents is about 1.1 estimated neurons.
DEFAULT_CLOUDFLARE_MAX_DOCUMENTS = 600
# The probe accepted 101 texts in one array, so the circulated 100-item cap is not current;
# 50 is well inside whatever the real limit is and makes a failed batch cheap to re-send.
DEFAULT_CLOUDFLARE_BATCH_SIZE = 50
# A quarter of the 10,000 free neurons a day, which is lower than the headroom alone would
# ask for. The reason is the ledger: it lives on local disk, a free host wipes that on every
# restart, and after a restart the guard believes nothing has been spent today and will allow
# a further full budget. At 2,500 four restart-cycles still fit inside the day's allowance -
# and 2,500 is about nine capped repositories, which is a reasonable day for a demo.
DEFAULT_CLOUDFLARE_DAILY_NEURONS = 2500
DEFAULT_CLOUDFLARE_TIMEOUT_SECONDS = 30.0
# The limit on building a whole semantic index, as opposed to one call. A repository at the
# 600-document cap is twelve calls of fifty, so 180 s allows about fifteen seconds each - room
# for a slow free host and a retry, without leaving a visitor waiting through three minutes of
# an API that is not answering. Separate from the shared indexing limit because only this
# provider's indexing crosses a network, and because missing it is not fatal here.
DEFAULT_CLOUDFLARE_INDEX_TIMEOUT_SECONDS = 180
DEFAULT_CLOUDFLARE_RETRIES = 2
DEFAULT_CLOUDFLARE_FAILURE_THRESHOLD = 3
DEFAULT_CLOUDFLARE_COOLDOWN_SECONDS = 300.0

ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

NO_CREDENTIALS = (
    "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must both be set for "
    "CODEATLAS_EMBEDDING_PROVIDER=cloudflare. Set CODEATLAS_EMBEDDING_PROVIDER=hash to run without them."
)

# Statuses worth trying again: a rate limit and the server's own faults. A 400 or a 401 will
# say the same thing however many times it is asked.
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class CloudflareUnavailableError(EmbeddingUnavailableError):
    """The API could not be used, for one of the reasons in embedding.REASONS.

    A subclass of the provider-agnostic error on purpose: AnswerService falls back on the base
    class, so it needs to know nothing about Cloudflare, and a later provider that can fail the
    same way gets the same fallback without touching the question path.
    """


@dataclass(frozen=True)
class CloudflareSettings:
    """Everything configurable about the provider.

    The last two are not this class's own business - prefix_metadata belongs to indexing and
    drop_stopwords to the query - but each of them changes the text that gets embedded, so both
    are carried here for the signature to include. Nothing that merely reorders results belongs
    in this class: see signature().
    """

    model: str = DEFAULT_CLOUDFLARE_MODEL
    pooling: str = DEFAULT_CLOUDFLARE_POOLING
    max_chars: int = DEFAULT_CLOUDFLARE_MAX_CHARS
    max_documents: int = DEFAULT_CLOUDFLARE_MAX_DOCUMENTS
    batch_size: int = DEFAULT_CLOUDFLARE_BATCH_SIZE
    daily_neurons: float = DEFAULT_CLOUDFLARE_DAILY_NEURONS
    timeout_seconds: float = DEFAULT_CLOUDFLARE_TIMEOUT_SECONDS
    retries: int = DEFAULT_CLOUDFLARE_RETRIES
    failure_threshold: int = DEFAULT_CLOUDFLARE_FAILURE_THRESHOLD
    cooldown_seconds: float = DEFAULT_CLOUDFLARE_COOLDOWN_SECONDS
    prefix_metadata: bool = True
    drop_stopwords: bool = True


class CloudflareEmbeddingService(EmbeddingService):
    def __init__(
        self,
        credentials: CloudflareCredentials | None,
        settings: CloudflareSettings | None = None,
        ledger_path: Path | None = None,
        post: Callable[..., JsonResponse] = post_json,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._credentials = credentials
        self._settings = settings or CloudflareSettings()
        self._ledger_path = ledger_path
        # The seam the tests replace with a fake client; nothing else here knows about HTTP.
        self._post = post
        self._clock = clock
        self._sleep = sleep
        self._logger = logging.getLogger(__name__)
        self._breaker_lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0

    @property
    def settings(self) -> CloudflareSettings:
        return self._settings

    def signature(self) -> str:
        """Everything that decides the vectors, and nothing else, so the index guard can refuse a mismatch.

        In it: the model, the pooling, the per-document clip, whether each document is prefixed
        with its path and name, and whether stopwords are dropped from the question before it is
        embedded. Each of those changes a vector or the text a vector is made from.

        Pooling is in here because it has to be: the probe measured cos = 0.9466 between the same
        text under cls and under mean. That is the dangerous kind of incompatible - near enough
        to look like it is working, far enough to rank worse - and no check downstream would
        catch it.

        **Out of it: everything that only reorders what has already been embedded** - the rerank
        weight, identifier splitting in the reranker, the candidate count, test-file filtering.
        Including one of those would make changing it invalidate every index built under the old
        value, and for this provider invalidating an index means paying to build it again, for
        vectors that would have come back identical.
        """
        s = self._settings
        return (
            f"cloudflare:{s.model}:pooling={s.pooling}:max_chars={s.max_chars}"
            f":prefix={int(s.prefix_metadata)}:stopwords={int(s.drop_stopwords)}"
        )

    def estimated_neurons(self, texts: list[str]) -> float:
        """What embedding these texts would be counted as costing: clipped, and batched as they will be sent."""
        return neuron_ledger.estimated_neurons(
            [self._clip(text) for text in texts], batch_size=self._settings.batch_size
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clipped = [self._clip(text) for text in texts]
        self._check_breaker()
        vectors: list[list[float]] = []
        for start in range(0, len(clipped), self._settings.batch_size):
            vectors.extend(self._embed_batch(clipped[start : start + self._settings.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _clip(self, text: str) -> str:
        return text[: self._settings.max_chars]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        estimate = neuron_ledger.estimated_neurons(batch)
        if not neuron_ledger.affordable(estimate, self._settings.daily_neurons, self._ledger_path):
            spent = neuron_ledger.spent_today(self._ledger_path).estimated_neurons
            raise CloudflareUnavailableError(
                "budget",
                f"{spent:.2f} of {self._settings.daily_neurons:g} estimated neurons already spent today",
            )

        response = self._call_with_retries(batch)
        vectors = _vectors_from(response, expected=len(batch))
        # Counted only after a call that worked. A refused call is not billed, and a call that
        # vanished in transit cannot be counted without inventing the number.
        neuron_ledger.record(estimate, documents=len(batch), path=self._ledger_path)
        return vectors

    def _call_with_retries(self, batch: list[str]) -> JsonResponse:
        last = ""
        for attempt in range(self._settings.retries + 1):
            try:
                response = self._call(batch)
            except JsonClientError as exc:
                last = f"transport {exc}"
                retryable = True
            else:
                if response.ok:
                    self._succeeded()
                    return response
                last = f"HTTP {response.status}"
                retryable = response.status in RETRYABLE_STATUSES
            self._logger.warning(
                "Cloudflare embedding call failed (%s) for a batch of %d documents, attempt %d of %d",
                last,
                len(batch),
                attempt + 1,
                self._settings.retries + 1,
            )
            if not retryable or attempt == self._settings.retries:
                break
            self._sleep(0.5 * (2**attempt))
        self._failed()
        raise CloudflareUnavailableError("api", last)

    def _call(self, batch: list[str]) -> JsonResponse:
        if self._credentials is None:
            raise CloudflareUnavailableError("api", NO_CREDENTIALS)
        body = {"text": batch, "pooling": self._settings.pooling}
        url = ENDPOINT.format(account_id=self._credentials.account_id, model=self._settings.model)
        return self._post(
            url,
            body,
            timeout=self._settings.timeout_seconds,
            headers={"Authorization": f"Bearer {self._credentials.api_token}"},
        )

    def _check_breaker(self) -> None:
        with self._breaker_lock:
            if self._clock() < self._open_until:
                left = self._open_until - self._clock()
                raise CloudflareUnavailableError("circuit", f"{left:.0f}s left of the cooldown")

    def _succeeded(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _failed(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._settings.failure_threshold:
                self._open_until = self._clock() + self._settings.cooldown_seconds
                self._logger.warning(
                    "Cloudflare embeddings failed %d times in a row; not calling again for %gs",
                    self._consecutive_failures,
                    self._settings.cooldown_seconds,
                )


def _vectors_from(response: JsonResponse, expected: int) -> list[list[float]]:
    """The vectors, or CloudflareUnavailableError. A body that is not the expected shape is a failure.

    Checked rather than trusted because a half-right body is worse than an error: a short list
    would silently pair vectors with the wrong records and the index would be quietly wrong.
    """
    body = response.body
    if body.get("success") is not True:
        raise CloudflareUnavailableError("api", "response did not report success")
    result = body.get("result")
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, list) or len(data) != expected:
        raise CloudflareUnavailableError("api", f"expected {expected} vectors, got {_count(data)}")
    vectors: list[list[float]] = []
    for item in data:
        if not isinstance(item, list) or len(item) != EXPECTED_DIMENSIONS:
            raise CloudflareUnavailableError("api", f"expected {EXPECTED_DIMENSIONS}-dimensional vectors")
        try:
            vectors.append([float(value) for value in item])
        except (TypeError, ValueError):
            raise CloudflareUnavailableError("api", "vector held something that is not a number") from None
    return vectors


def _count(data: object) -> str:
    return str(len(data)) if isinstance(data, list) else "none"
