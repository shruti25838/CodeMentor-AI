"""The Cloudflare provider, driven entirely by a fake client.

No test here reaches the network. The fake records what it was asked to send, which is how the
batching, the clipping, the pooling and - most of it - the promise that no document text ever
leaves this process by any route but the request body get checked.
"""

from dataclasses import replace

import pytest

from codeatlas.app.di import build_embedder, cloudflare_settings, index_options
from codeatlas.services.http.json_client import JsonClientError, JsonResponse
from codeatlas.services.retrieval import neuron_ledger
from codeatlas.services.retrieval.cloudflare_embedder import (
    DEFAULT_CLOUDFLARE_BATCH_SIZE,
    DEFAULT_CLOUDFLARE_DAILY_NEURONS,
    DEFAULT_CLOUDFLARE_MAX_CHARS,
    DEFAULT_CLOUDFLARE_MAX_DOCUMENTS,
    DEFAULT_CLOUDFLARE_MODEL,
    EXPECTED_DIMENSIONS,
    NO_CREDENTIALS,
    CloudflareEmbeddingService,
    CloudflareSettings,
    CloudflareUnavailableError,
)
from codeatlas.services.retrieval.embedding import REASONS, fallback_note
from codeatlas.services.retrieval.fastembed_embedder import FastEmbedEmbeddingService
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.utils.config import CloudflareCredentials, load_config

CREDENTIALS = CloudflareCredentials(account_id="acct-123", api_token="secret-token")
SECRETS = ("acct-123", "secret-token")


class FakeClient:
    """Stands in for post_json. Answers from a queue of canned replies, and keeps every request."""

    def __init__(self, replies=None) -> None:
        self.replies = list(replies or [])
        self.requests: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, url, body, timeout=None, headers=None) -> JsonResponse:
        self.urls.append(url)
        self.requests.append({"body": body, "timeout": timeout, "headers": headers or {}})
        reply = self.replies.pop(0) if self.replies else ok_for(body["text"])
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def batch_sizes(self) -> list[int]:
        return [len(request["body"]["text"]) for request in self.requests]


def ok_for(texts, dimensions: int = EXPECTED_DIMENSIONS) -> JsonResponse:
    vectors = [[float(i + 1)] * dimensions for i, _ in enumerate(texts)]
    return JsonResponse(
        status=200, body={"success": True, "result": {"shape": [len(texts), dimensions], "data": vectors}}
    )


def refusal(status: int, code: int = 10000) -> JsonResponse:
    return JsonResponse(status=status, body={"success": False, "errors": [{"code": code, "message": "nope"}]})


def service(tmp_path, client=None, clock=None, **overrides) -> CloudflareEmbeddingService:
    settings = replace(CloudflareSettings(), **overrides)
    return CloudflareEmbeddingService(
        credentials=CREDENTIALS,
        settings=settings,
        ledger_path=tmp_path / "neurons.json",
        post=client or FakeClient(),
        clock=clock or (lambda: 0.0),
        sleep=lambda _seconds: None,
    )


def _config(provider: str):
    return replace(load_config(), embedding_provider=provider)


# --- selection and defaults ------------------------------------------------------------


def test_hash_is_still_the_default(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_EMBEDDING_PROVIDER", raising=False)
    assert isinstance(build_embedder(load_config()), HashEmbeddingService)


def test_cloudflare_is_selected_by_name_and_builds_without_calling_anything(monkeypatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "secret-token")
    embedder = build_embedder(_config("cloudflare"))
    assert isinstance(embedder, CloudflareEmbeddingService)


def test_the_defaults_are_the_ones_that_were_decided() -> None:
    settings = cloudflare_settings(_config("cloudflare"))
    assert settings.model == DEFAULT_CLOUDFLARE_MODEL == "@cf/baai/bge-small-en-v1.5"
    assert settings.pooling == "mean"
    assert settings.max_chars == DEFAULT_CLOUDFLARE_MAX_CHARS == 2000
    assert settings.max_documents == DEFAULT_CLOUDFLARE_MAX_DOCUMENTS == 600
    assert settings.batch_size == DEFAULT_CLOUDFLARE_BATCH_SIZE == 50
    assert settings.daily_neurons == DEFAULT_CLOUDFLARE_DAILY_NEURONS == 2500


def test_the_indexing_options_follow_the_provider() -> None:
    options = index_options(_config("cloudflare"))
    assert options["batch_size"] == 50
    assert options["max_chars"] == 2000


def test_the_cloudflare_settings_are_ignored_by_other_providers() -> None:
    """Changing a cloudflare knob must not move a hash index, or the index cache would miss."""
    base = _config("hash")
    loud = replace(base, cloudflare_batch_size=7, cloudflare_max_chars=11, cloudflare_pooling="cls")
    assert index_options(base) == index_options(loud)
    assert build_embedder(base).signature() == build_embedder(loud).signature()


# --- the signature, which is what the index guard compares --------------------------------


def test_the_signature_names_the_model_the_pooling_and_the_clip(tmp_path) -> None:
    signature = service(tmp_path).signature()
    assert signature.startswith("cloudflare:")
    assert DEFAULT_CLOUDFLARE_MODEL in signature
    assert "pooling=mean" in signature
    assert "max_chars=2000" in signature


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "@cf/baai/bge-base-en-v1.5"),
        ("pooling", "cls"),
        ("max_chars", 1500),
        ("prefix_metadata", False),
        ("drop_stopwords", False),
    ],
)
def test_every_setting_that_changes_the_embedded_text_changes_the_signature(tmp_path, field, value) -> None:
    assert service(tmp_path).signature() != service(tmp_path, **{field: value}).signature()


@pytest.mark.parametrize("field", ["batch_size", "timeout_seconds", "retries", "daily_neurons", "max_documents"])
def test_settings_that_do_not_change_a_vector_stay_out_of_the_signature(tmp_path, field) -> None:
    """Otherwise raising a budget or a batch size would invalidate every index that had been paid for."""
    assert service(tmp_path).signature() == service(tmp_path, **{field: 3}).signature()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CODEATLAS_RERANK_SUBTOKENS", "true"),
        ("CODEATLAS_RERANK_WEIGHT", "0.0"),
        ("CODEATLAS_RETRIEVAL_CANDIDATES", "50"),
        ("CODEATLAS_SKIP_TESTS", "true"),
    ],
)
def test_flipping_a_ranking_only_setting_leaves_the_signature_alone(monkeypatch, name, value) -> None:
    """A setting that reorders what is already embedded must not invalidate a paid-for index.

    rerank_subtokens was in the signature when the provider was first written, and it should not
    have been: it splits identifiers while *reranking*, long after the vectors exist. Changing it
    would have made every repository pay to be embedded again for vectors that came back
    identical. The same reasoning covers the other three here.
    """
    for setting in (
        "CODEATLAS_RERANK_SUBTOKENS",
        "CODEATLAS_RERANK_WEIGHT",
        "CODEATLAS_RETRIEVAL_CANDIDATES",
        "CODEATLAS_SKIP_TESTS",
    ):
        monkeypatch.delenv(setting, raising=False)
    before = build_embedder(_config("cloudflare")).signature()
    monkeypatch.setenv(name, value)
    assert build_embedder(_config("cloudflare")).signature() == before


def test_the_signature_names_only_what_makes_a_vector(tmp_path) -> None:
    """Read as a list rather than as a string, so a setting added later has to be thought about."""
    signature = service(tmp_path).signature()
    head, *parts = signature.split(":")
    assert head == "cloudflare"
    assert [part.split("=")[0] for part in parts[1:]] == ["pooling", "max_chars", "prefix", "stopwords"]
    assert "subtokens" not in signature


def test_cloudflare_does_not_share_a_signature_with_any_other_provider(tmp_path) -> None:
    signature = service(tmp_path).signature()
    assert signature != HashEmbeddingService().signature()
    assert signature != FastEmbedEmbeddingService().signature()


def test_cls_and_mean_are_different_indexes(tmp_path) -> None:
    """The probe measured cos 0.9466 between them: close enough to look fine, far enough to rank worse."""
    from codeatlas.services.retrieval.indexing import embedder_mismatch

    mean, cls = service(tmp_path), service(tmp_path, pooling="cls")
    assert embedder_mismatch(mean.signature(), cls)
    assert embedder_mismatch(cls.signature(), mean)
    assert not embedder_mismatch(mean.signature(), mean)


# --- what gets sent --------------------------------------------------------------------


def test_documents_are_sent_in_batches_of_the_configured_size(tmp_path) -> None:
    client = FakeClient()
    vectors = service(tmp_path, client=client, batch_size=50).embed_texts([f"doc {i}" for i in range(120)])
    assert len(vectors) == 120
    assert client.batch_sizes == [50, 50, 20]


def test_each_document_is_clipped_before_it_is_sent(tmp_path) -> None:
    client = FakeClient()
    service(tmp_path, client=client, max_chars=10).embed_texts(["x" * 500])
    assert client.requests[0]["body"]["text"] == ["x" * 10]


def test_the_pooling_is_sent_explicitly_rather_than_left_to_the_default(tmp_path) -> None:
    """The endpoint's default is mean; sending it anyway means the index matches the setting."""
    client = FakeClient()
    service(tmp_path, client=client, pooling="cls").embed_texts(["doc"])
    assert client.requests[0]["body"]["pooling"] == "cls"


def test_the_model_and_the_account_are_in_the_url_and_the_token_is_in_the_header(tmp_path) -> None:
    client = FakeClient()
    service(tmp_path, client=client).embed_texts(["doc"])
    assert client.urls[0].endswith(f"/accounts/acct-123/ai/run/{DEFAULT_CLOUDFLARE_MODEL}")
    assert client.requests[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_the_configured_timeout_is_passed_down(tmp_path) -> None:
    client = FakeClient()
    service(tmp_path, client=client, timeout_seconds=7.5).embed_texts(["doc"])
    assert client.requests[0]["timeout"] == 7.5


def test_no_texts_makes_no_call(tmp_path) -> None:
    client = FakeClient()
    assert service(tmp_path, client=client).embed_texts([]) == []
    assert client.calls == 0


def test_a_query_is_one_vector(tmp_path) -> None:
    vector = service(tmp_path).embed_query("how are cookies signed")
    assert len(vector) == EXPECTED_DIMENSIONS


# --- the budget ------------------------------------------------------------------------


def test_the_call_is_counted_in_the_ledger_after_it_works(tmp_path) -> None:
    path = tmp_path / "neurons.json"
    embedder = service(tmp_path, batch_size=2)
    texts = ["a" * 300, "b" * 300, "c" * 300]
    embedder.embed_texts(texts)
    spend = neuron_ledger.spent_today(path)
    assert spend.documents == 3
    assert spend.calls == 2
    assert spend.estimated_neurons == pytest.approx(neuron_ledger.estimated_neurons(texts))


def test_a_refused_call_is_not_counted(tmp_path) -> None:
    client = FakeClient([refusal(401)])
    with pytest.raises(CloudflareUnavailableError):
        service(tmp_path, client=client).embed_texts(["doc"])
    assert neuron_ledger.spent_today(tmp_path / "neurons.json").calls == 0


def test_the_provider_stops_before_spending_past_the_budget(tmp_path) -> None:
    path = tmp_path / "neurons.json"
    neuron_ledger.record(7999.9, documents=1, path=path)
    client = FakeClient()
    with pytest.raises(CloudflareUnavailableError) as raised:
        service(tmp_path, client=client, daily_neurons=8000).embed_texts(["x" * 5000])
    assert raised.value.reason == "budget"
    assert client.calls == 0, "the point is that nothing is sent, not that the error is caught"


def test_the_budget_stops_part_way_through_a_long_repository(tmp_path) -> None:
    """The first batches go through and the next one is refused, rather than all or nothing."""
    path = tmp_path / "neurons.json"
    one_batch = neuron_ledger.estimated_neurons(["x" * 2000] * 2)
    neuron_ledger.record(8000 - one_batch * 1.5, documents=1, path=path)
    client = FakeClient()
    with pytest.raises(CloudflareUnavailableError) as raised:
        service(tmp_path, client=client, batch_size=2, daily_neurons=8000).embed_texts(["x" * 2000] * 4)
    assert raised.value.reason == "budget"
    assert client.calls == 1


def test_the_estimate_a_repository_would_cost_can_be_asked_for_before_calling(tmp_path) -> None:
    embedder = service(tmp_path, max_chars=2000)
    assert embedder.estimated_neurons(["x" * 100_000]) == neuron_ledger.estimated_neurons(["x" * 2000])


# --- failure, retries and the breaker ----------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 429, 408])
def test_a_server_side_failure_is_retried(tmp_path, status) -> None:
    client = FakeClient([refusal(status)])
    assert service(tmp_path, client=client, retries=2).embed_texts(["doc"])
    assert client.calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_a_refusal_that_will_not_change_is_not_retried(tmp_path, status) -> None:
    client = FakeClient([refusal(status)] * 5)
    with pytest.raises(CloudflareUnavailableError) as raised:
        service(tmp_path, client=client, retries=2).embed_texts(["doc"])
    assert raised.value.reason == "api"
    assert client.calls == 1


def test_a_transport_failure_is_retried_and_then_gives_up(tmp_path) -> None:
    client = FakeClient([JsonClientError("TimeoutError")] * 5)
    with pytest.raises(CloudflareUnavailableError) as raised:
        service(tmp_path, client=client, retries=2).embed_texts(["doc"])
    assert raised.value.reason == "api"
    assert client.calls == 3


@pytest.mark.parametrize(
    "reply",
    [
        JsonResponse(status=200, body={"success": False, "result": {"data": [[0.0] * EXPECTED_DIMENSIONS]}}),
        JsonResponse(status=200, body={"success": True, "result": {"data": []}}),
        JsonResponse(status=200, body={"success": True, "result": {"data": [[0.0] * 7]}}),
        JsonResponse(status=200, body={"success": True, "result": {"data": [["not a number"] * EXPECTED_DIMENSIONS]}}),
        JsonResponse(status=200, body={"success": True}),
        JsonResponse(status=200, body={}),
    ],
)
def test_a_body_that_is_not_the_expected_shape_is_a_failure(tmp_path, reply) -> None:
    """A half-right body is worse than an error: short data would pair vectors with the wrong records."""
    client = FakeClient([reply] * 5)
    with pytest.raises(CloudflareUnavailableError) as raised:
        service(tmp_path, client=client, retries=0).embed_texts(["doc"])
    assert raised.value.reason == "api"


def test_the_breaker_opens_after_repeated_failures_and_then_stops_calling(tmp_path) -> None:
    now = [0.0]
    client = FakeClient([JsonClientError("TimeoutError")] * 50)
    embedder = service(tmp_path, client=client, retries=0, failure_threshold=3, clock=lambda: now[0])

    for _ in range(3):
        with pytest.raises(CloudflareUnavailableError):
            embedder.embed_texts(["doc"])
    assert client.calls == 3

    with pytest.raises(CloudflareUnavailableError) as raised:
        embedder.embed_texts(["doc"])
    assert raised.value.reason == "circuit"
    assert client.calls == 3, "an open breaker must not reach the network at all"


def test_the_breaker_closes_again_after_the_cooldown(tmp_path) -> None:
    now = [0.0]
    client = FakeClient([JsonClientError("TimeoutError")] * 3)
    embedder = service(
        tmp_path, client=client, retries=0, failure_threshold=3, cooldown_seconds=300, clock=lambda: now[0]
    )
    for _ in range(3):
        with pytest.raises(CloudflareUnavailableError):
            embedder.embed_texts(["doc"])

    now[0] = 301.0
    assert embedder.embed_texts(["doc"])
    assert client.calls == 4


def test_a_success_resets_the_failure_count(tmp_path) -> None:
    client = FakeClient([JsonClientError("TimeoutError"), JsonClientError("TimeoutError"), ok_for(["doc"])])
    embedder = service(tmp_path, client=client, retries=0, failure_threshold=3)
    for _ in range(2):
        with pytest.raises(CloudflareUnavailableError):
            embedder.embed_texts(["doc"])
    assert embedder.embed_texts(["doc"])
    # Three failures in a row would have opened it; two then a success must not.
    client.replies = [JsonClientError("TimeoutError")]
    with pytest.raises(CloudflareUnavailableError) as raised:
        embedder.embed_texts(["doc"])
    assert raised.value.reason == "api"


def test_without_credentials_nothing_is_sent_and_the_message_says_what_to_do(tmp_path) -> None:
    client = FakeClient()
    embedder = CloudflareEmbeddingService(
        credentials=None, ledger_path=tmp_path / "neurons.json", post=client, sleep=lambda _s: None
    )
    with pytest.raises(CloudflareUnavailableError) as raised:
        embedder.embed_texts(["doc"])
    assert client.calls == 0
    assert str(raised.value) == NO_CREDENTIALS
    assert "CODEATLAS_EMBEDDING_PROVIDER=hash" in NO_CREDENTIALS


# --- what a visitor is told, and what is never written down ------------------------------


@pytest.mark.parametrize("reason", sorted(REASONS))
def test_every_reason_has_a_note_a_visitor_can_read(reason) -> None:
    note = fallback_note(reason)
    assert note.endswith(".") and "keyword search" in note
    for jargon in ("neuron", "HTTP", "circuit", "API", "Cloudflare"):
        assert jargon not in note


def test_an_unknown_reason_still_gets_a_note() -> None:
    assert fallback_note("something new") == REASONS["api"]


def test_nothing_about_a_failure_names_the_account_or_the_token(tmp_path) -> None:
    client = FakeClient([refusal(401)] * 5)
    with pytest.raises(CloudflareUnavailableError) as raised:
        service(tmp_path, client=client, retries=1).embed_texts(["a secret document"])
    rendered = f"{raised.value} {raised.value!r} {raised.value.reason}"
    for forbidden in (*SECRETS, "a secret document"):
        assert forbidden not in rendered


def test_no_document_text_question_or_credential_is_logged(tmp_path, caplog) -> None:
    """Everything that fails logs something; none of it may be the thing being embedded."""
    caplog.set_level("DEBUG")
    client = FakeClient([refusal(500), JsonClientError("TimeoutError"), refusal(503), refusal(500)])
    embedder = service(tmp_path, client=client, retries=1, failure_threshold=2)
    for _ in range(2):
        with pytest.raises(CloudflareUnavailableError):
            embedder.embed_texts(["how is a cookie signed", "def sign(value): ..."])

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert logged, "a failure that logs nothing would make this test vacuous"
    for forbidden in (*SECRETS, "how is a cookie signed", "def sign", "api.cloudflare.com"):
        assert forbidden not in logged


# --- settings from the environment, rejected at startup rather than at first use ----------


def test_an_unknown_pooling_is_refused_with_the_two_that_exist(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_POOLING", "max")
    with pytest.raises(ValueError) as raised:
        load_config()
    assert "mean" in str(raised.value) and "cls" in str(raised.value)


@pytest.mark.parametrize("pooling", ["mean", "cls", "CLS", " mean "])
def test_a_known_pooling_is_accepted_and_normalized(monkeypatch, pooling) -> None:
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_POOLING", pooling)
    assert load_config().cloudflare_pooling == pooling.strip().lower()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CODEATLAS_CLOUDFLARE_MAX_CHARS", "0"),
        ("CODEATLAS_CLOUDFLARE_MAX_CHARS", "-1"),
        ("CODEATLAS_CLOUDFLARE_MAX_CHARS", "lots"),
        ("CODEATLAS_CLOUDFLARE_MAX_DOCUMENTS", "0"),
        ("CODEATLAS_CLOUDFLARE_BATCH_SIZE", "0"),
        ("CODEATLAS_CLOUDFLARE_DAILY_NEURONS", "-5"),
        ("CODEATLAS_CLOUDFLARE_DAILY_NEURONS", "none"),
        ("CODEATLAS_CLOUDFLARE_TIMEOUT_SECONDS", "0"),
        ("CODEATLAS_CLOUDFLARE_RETRIES", "-1"),
        ("CODEATLAS_CLOUDFLARE_FAILURE_THRESHOLD", "0"),
    ],
)
def test_a_nonsense_number_stops_the_app_starting(monkeypatch, name, value) -> None:
    """The alternative is a crash halfway through indexing, or a silent revert to the default."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as raised:
        load_config()
    assert name in str(raised.value)


def test_retries_may_be_zero_but_the_others_may_not(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_RETRIES", "0")
    assert load_config().cloudflare_retries == 0


def test_an_unset_or_empty_variable_keeps_the_default(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_BATCH_SIZE", "")
    monkeypatch.delenv("CODEATLAS_CLOUDFLARE_MAX_DOCUMENTS", raising=False)
    config = load_config()
    assert config.cloudflare_batch_size == DEFAULT_CLOUDFLARE_BATCH_SIZE
    assert config.cloudflare_max_documents == DEFAULT_CLOUDFLARE_MAX_DOCUMENTS


def test_the_settings_really_reach_the_provider(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_POOLING", "cls")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_MAX_CHARS", "1500")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_BATCH_SIZE", "25")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_DAILY_NEURONS", "100")
    settings = cloudflare_settings(replace(load_config(), embedding_provider="cloudflare"))
    assert (settings.pooling, settings.max_chars, settings.batch_size, settings.daily_neurons) == (
        "cls",
        1500,
        25,
        100.0,
    )


def test_the_providers_estimate_counts_the_padding_it_will_be_billed_for(tmp_path) -> None:
    """A long document in a batch raises what every other document in that batch costs."""
    embedder = service(tmp_path, batch_size=50, max_chars=2000)
    short = ["x" * 30] * 49
    assert embedder.estimated_neurons(short + ["y" * 2000]) > embedder.estimated_neurons(short) * 10


def test_the_estimate_uses_the_batch_size_the_provider_will_send(tmp_path) -> None:
    texts = ["x" * 30] * 9 + ["y" * 1500]
    big = service(tmp_path, batch_size=10).estimated_neurons(texts)
    small = service(tmp_path, batch_size=1).estimated_neurons(texts)
    assert big > small
