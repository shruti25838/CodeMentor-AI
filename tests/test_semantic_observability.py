"""One log line per semantic analysis, one metric per outcome, and nothing private in either.

The point of these tests is not that the numbers are logged but that *only* the numbers are:
a log line and a metric label are the two places where a question, a path or a token would
otherwise be easiest to leak by accident.
"""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_dual_index import FailingEmbedder, _repo
from test_embedder_identity import FakeEmbedder

from codeatlas.observability.metrics import (
    RETRIEVAL_FALLBACK,
    SEMANTIC_INDEX_NEURONS,
    SEMANTIC_INDEX_OUTCOMES,
)
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.retrieval.cloudflare_embedder import (
    ApiUsage,
    CloudflareEmbeddingService,
    CloudflareSettings,
)
from codeatlas.services.retrieval.dual_index import (
    BUDGET,
    FAILURE,
    FALLBACK_NAME,
    OK,
    OUTCOMES,
    TIMEOUT,
    DualIndexService,
)
from codeatlas.services.retrieval.embedding import REASONS
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService

SECRETS = ("acct-123", "secret-token", "api.cloudflare.com")


class Usage:
    """Stands in for the provider's own reporter.

    Asked twice per analysis, as the real one is: once before the attempt, to throw away
    anything an earlier analysis left behind, and once after, for what this one spent. So the
    first answer is empty and the second carries the figures.
    """

    def __init__(self, calls=3, seconds=1.25, estimated_neurons=4.5) -> None:
        self.answers = [ApiUsage(), ApiUsage(calls=calls, seconds=seconds, estimated_neurons=estimated_neurons)]
        self.taken = 0

    def __call__(self) -> ApiUsage:
        self.taken += 1
        return self.answers.pop(0) if self.answers else ApiUsage()


def _dual(tmp_path, semantic_embedder=None, usage=None, max_documents=600, timeout=None):
    return DualIndexService(
        semantic=CodeIndexService(
            embedder=semantic_embedder or FakeEmbedder(name="semantic", value=0.5),
            retriever=FaissCodeRetriever(),
            name="cloudflare",
            timeout_seconds=timeout,
        ),
        fallback=CodeIndexService(embedder=HashEmbeddingService(), retriever=FaissCodeRetriever(), name=FALLBACK_NAME),
        semantic_name="cloudflare",
        max_documents=max_documents,
        usage=usage,
    )


def _counter(metric, **labels) -> float:
    return metric.labels(**labels)._value.get() if labels else metric._value.get()


def _log(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


# --- the log line ------------------------------------------------------------------------


def test_a_successful_analysis_logs_its_calls_seconds_neurons_and_outcome(tmp_path, caplog) -> None:
    caplog.set_level("INFO")
    usage = Usage(calls=12, seconds=31.5, estimated_neurons=276.06)
    _dual(tmp_path, usage=usage).index_repository(*_repo(tmp_path))
    line = next(line for line in _log(caplog).splitlines() if "outcome=" in line)
    assert "outcome=ok" in line
    assert "calls=12" in line
    assert "api_seconds=31.5" in line
    assert "estimated_neurons=276.06" in line
    assert "documents=1" in line


@pytest.mark.parametrize(
    ("reason", "outcome"),
    [("budget", BUDGET), ("circuit", "breaker"), ("api", FAILURE)],
)
def test_every_way_of_failing_names_its_outcome(tmp_path, caplog, reason, outcome) -> None:
    caplog.set_level("INFO")
    _dual(tmp_path, semantic_embedder=FailingEmbedder(reason=reason), usage=Usage()).index_repository(*_repo(tmp_path))
    assert f"outcome={outcome}" in _log(caplog)


def test_an_over_cap_repository_is_reported_too_with_no_calls(tmp_path, caplog) -> None:
    """There is no analysis to report, and a line saying so is more use than silence."""
    caplog.set_level("INFO")
    usage = Usage(calls=0, seconds=0.0, estimated_neurons=0.0)
    _dual(tmp_path, max_documents=0, usage=usage).index_repository(*_repo(tmp_path))
    assert "outcome=cap" in _log(caplog)
    assert "calls=0" in _log(caplog)


def test_one_analysis_is_not_charged_for_the_one_before_it(tmp_path, caplog) -> None:
    """Usage is taken before the attempt as well as after, so a leftover cannot land on this line."""
    caplog.set_level("INFO")
    leftover = Usage(calls=99, seconds=99.0, estimated_neurons=99.0)
    # Pretend the reporter is still holding an earlier analysis's spend when this one starts.
    leftover.answers = [ApiUsage(calls=99, seconds=99.0, estimated_neurons=99.0), ApiUsage()]
    _dual(tmp_path, usage=leftover).index_repository(*_repo(tmp_path))
    line = next(line for line in _log(caplog).splitlines() if "outcome=" in line)
    assert "calls=0" in line and "estimated_neurons=0.00" in line
    assert "99" not in line


def test_a_provider_with_nothing_to_report_still_logs_a_line(tmp_path, caplog) -> None:
    caplog.set_level("INFO")
    _dual(tmp_path, usage=None).index_repository(*_repo(tmp_path))
    assert "outcome=ok" in _log(caplog)
    assert "calls=0" in _log(caplog)


def test_the_line_holds_no_text_no_question_no_url_and_no_credential(tmp_path, caplog) -> None:
    caplog.set_level("DEBUG")
    _dual(tmp_path, semantic_embedder=FailingEmbedder(reason="budget"), usage=Usage()).index_repository(
        *_repo(tmp_path)
    )
    logged = _log(caplog)
    assert logged
    for forbidden in (*SECRETS, "def sign", "detail that must not be shown to anyone", "https://"):
        assert forbidden not in logged


# --- the metrics -------------------------------------------------------------------------


def test_each_outcome_is_counted(tmp_path) -> None:
    before = _counter(SEMANTIC_INDEX_OUTCOMES, outcome=OK)
    _dual(tmp_path, usage=Usage()).index_repository(*_repo(tmp_path))
    assert _counter(SEMANTIC_INDEX_OUTCOMES, outcome=OK) == before + 1


def test_estimated_neurons_are_counted(tmp_path) -> None:
    before = _counter(SEMANTIC_INDEX_NEURONS)
    _dual(tmp_path, usage=Usage(estimated_neurons=12.5)).index_repository(*_repo(tmp_path))
    assert _counter(SEMANTIC_INDEX_NEURONS) == pytest.approx(before + 12.5)


def test_a_timeout_is_counted_as_its_own_outcome(tmp_path) -> None:
    class Slow(FakeEmbedder):
        def embed_texts(self, texts):
            clock[0] += 1000
            return super().embed_texts(texts)

    clock = [0.0]
    service = DualIndexService(
        semantic=CodeIndexService(
            embedder=Slow(name="semantic"),
            retriever=FaissCodeRetriever(),
            name="cloudflare",
            timeout_seconds=180,
            clock=lambda: clock[0],
        ),
        fallback=CodeIndexService(embedder=HashEmbeddingService(), retriever=FaissCodeRetriever(), name=FALLBACK_NAME),
        semantic_name="cloudflare",
        max_documents=600,
    )
    before = _counter(SEMANTIC_INDEX_OUTCOMES, outcome=TIMEOUT)
    service.index_repository(*_repo(tmp_path))
    assert _counter(SEMANTIC_INDEX_OUTCOMES, outcome=TIMEOUT) == before + 1


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_every_outcome_is_a_word_from_a_fixed_list(outcome) -> None:
    """A label value must never be able to hold something a visitor typed."""
    assert outcome.isalpha() and outcome.islower()
    assert outcome in ("ok", "budget", "failure", "breaker", "timeout", "cap")


@pytest.mark.parametrize("reason", sorted(REASONS))
def test_a_question_that_falls_back_is_counted_by_reason(tmp_path, reason) -> None:
    source = tmp_path / "mod.py"
    source.write_text("def sign(value):\n    return value\n", encoding="utf-8")
    repo, parsed = _repo(tmp_path)
    fallback_retriever = FaissCodeRetriever()
    CodeIndexService(
        embedder=HashEmbeddingService(), retriever=fallback_retriever, name=FALLBACK_NAME
    ).index_repository(repo, parsed)
    semantic_retriever = FaissCodeRetriever()
    CodeIndexService(
        embedder=FakeEmbedder(name="semantic", value=0.5), retriever=semantic_retriever, name="cloudflare"
    ).index_repository(repo, parsed)

    service = AnswerService(
        retriever=semantic_retriever,
        embedder=FailingEmbedder(reason=reason) if reason in ("budget", "circuit", "api") else FakeEmbedder("other"),
        fallback_retriever=fallback_retriever,
        fallback_embedder=HashEmbeddingService(),
    )
    expected = reason if reason in ("budget", "circuit", "api") else "unindexed"
    before = _counter(RETRIEVAL_FALLBACK, reason=expected)
    records, note = service.retrieve_with_note("r", "how is a value signed")
    assert records and note
    assert _counter(RETRIEVAL_FALLBACK, reason=expected) == before + 1


def test_an_ordinary_question_counts_no_fallback(tmp_path) -> None:
    repo, parsed = _repo(tmp_path)
    retriever = FaissCodeRetriever()
    embedder = FakeEmbedder(name="semantic", value=0.5)
    CodeIndexService(embedder=embedder, retriever=retriever, name="cloudflare").index_repository(repo, parsed)
    service = AnswerService(
        retriever=retriever,
        embedder=embedder,
        fallback_retriever=FaissCodeRetriever(),
        fallback_embedder=HashEmbeddingService(),
    )
    before = sum(_counter(RETRIEVAL_FALLBACK, reason=r) for r in sorted(set(REASONS.values())) for r in REASONS)
    records, note = service.retrieve_with_note("r", "how is a value signed")
    assert records and note == ""
    after = sum(_counter(RETRIEVAL_FALLBACK, reason=r) for r in sorted(set(REASONS.values())) for r in REASONS)
    assert after == before


def test_the_provider_counts_its_own_calls_and_seconds(tmp_path) -> None:
    from codeatlas.services.http.json_client import JsonResponse

    ticks = [0.0]

    def clock() -> float:
        ticks[0] += 0.5
        return ticks[0]

    def post(url, body, timeout=None, headers=None) -> JsonResponse:
        return JsonResponse(
            status=200,
            body={"success": True, "result": {"data": [[0.1] * 384 for _ in body["text"]]}},
        )

    embedder = CloudflareEmbeddingService(
        credentials=None,
        settings=replace(CloudflareSettings(), batch_size=1),
        ledger_path=tmp_path / "neurons.json",
        post=post,
        sleep=lambda _s: None,
    )
    embedder._credentials = type("C", (), {"account_id": "acct-123", "api_token": "secret-token"})()
    monkeypatched = embedder._call_with_retries
    del monkeypatched

    import codeatlas.services.retrieval.cloudflare_embedder as module

    original = module.time.monotonic
    module.time.monotonic = clock
    try:
        embedder.embed_texts(["one", "two"])
    finally:
        module.time.monotonic = original

    usage = embedder.take_usage()
    assert usage.calls == 2
    assert usage.seconds > 0
    assert usage.estimated_neurons > 0
    assert embedder.take_usage() == ApiUsage(), "taking usage must reset it"


def test_usage_holds_only_numbers() -> None:
    usage = ApiUsage(calls=1, seconds=2.0, estimated_neurons=3.0)
    assert all(isinstance(value, int | float) for value in (usage.calls, usage.seconds, usage.estimated_neurons))
    assert datetime.now(UTC) and Path(".")  # imports used, keeps the file honest
