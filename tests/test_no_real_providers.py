"""The guard that keeps the suite honest about credentials.

A test that builds a real provider client passes on a machine where one happens to be
configured and fails where one is not, which is how a green local run and a red CI run come
from the same commit. These tests check the guard that makes both machines behave the same.
"""

import os
from dataclasses import replace

import pytest

from codeatlas.services.http.json_client import post_json
from codeatlas.services.llm.provider import FallbackChatModel, LlmProvider
from codeatlas.services.retrieval.cloudflare_embedder import CloudflareEmbeddingService, CloudflareSettings
from codeatlas.utils.config import CloudflareCredentials, load_cloudflare_credentials, load_config
from tests.conftest import BUILDS_A_CLIENT, CREDENTIAL_PREFIXES, CREDENTIAL_VARS


def _leaked() -> list[str]:
    """Names only, never values, and never the environment itself.

    `assert name not in os.environ` would be the obvious way to write this, and it is a trap:
    when it fails, pytest renders the whole environment into the report - which is how a real
    token ends up in a terminal, a CI log and anyone's scrollback. The assertion below can only
    ever print a list of variable names.
    """
    return sorted(name for name in os.environ if name in CREDENTIAL_VARS or name.startswith(CREDENTIAL_PREFIXES))


def test_no_provider_credential_is_in_the_environment() -> None:
    assert _leaked() == []


def test_dotenv_cannot_put_them_back() -> None:
    """The first test to call this used to load the whole file for every test after it."""
    assert load_cloudflare_credentials() is None


def test_building_a_chat_model_fails_loudly() -> None:
    config = replace(load_config(), llm_provider="groq")
    with pytest.raises(AssertionError) as raised:
        LlmProvider(config)._build()
    message = str(raised.value)
    assert "built a real ChatOpenAI" in message
    assert BUILDS_A_CLIENT in message, "the message has to say how to opt in"
    assert "CODEATLAS_LLM_PROVIDER=none" in message, "and what to do instead"


def test_the_openai_provider_is_blocked_too() -> None:
    with pytest.raises(AssertionError):
        LlmProvider(replace(load_config(), llm_provider="openai"))._build()


@pytest.mark.builds_model_client
def test_a_marked_test_may_build_one() -> None:
    """The opt-out, which test_llm_defaults.py uses to check that settings reach the client."""
    import os as _os

    _os.environ["GROQ_API_KEY"] = "not-a-real-key"
    try:
        model = LlmProvider(replace(load_config(), llm_provider="groq"))._build()
        assert getattr(model, "model_name", None)
    finally:
        del _os.environ["GROQ_API_KEY"]


def test_the_offline_model_is_always_allowed() -> None:
    """What a test should use instead: the app's own answer when nothing is configured."""
    model = LlmProvider(replace(load_config(), llm_provider="none"))._build()
    assert isinstance(getattr(model, "bound", model), FallbackChatModel)
    assert model.invoke("hello").content


def test_a_cloudflare_provider_built_without_a_fake_refuses_to_call(tmp_path) -> None:
    """Construction is allowed - several tests only read the signature - but a call is not."""
    embedder = CloudflareEmbeddingService(
        credentials=CloudflareCredentials("acct-123", "secret-token"),
        settings=CloudflareSettings(),
        ledger_path=tmp_path / "neurons.json",
    )
    assert embedder.signature()
    with pytest.raises(AssertionError) as raised:
        embedder.embed_texts(["anything at all"])
    assert "Cloudflare API" in str(raised.value)


def test_a_cloudflare_provider_given_a_fake_is_left_alone(tmp_path) -> None:
    from codeatlas.services.http.json_client import JsonResponse

    def fake(url, body, timeout=None, headers=None) -> JsonResponse:
        return JsonResponse(status=200, body={"success": True, "result": {"data": [[0.1] * 384]}})

    embedder = CloudflareEmbeddingService(
        credentials=CloudflareCredentials("acct-123", "secret-token"),
        settings=CloudflareSettings(),
        ledger_path=tmp_path / "neurons.json",
        post=fake,
    )
    assert len(embedder.embed_texts(["one"])[0]) == 384


def test_the_real_client_is_what_the_guard_swaps_out(tmp_path) -> None:
    """Pins the thing the guard recognises, so renaming the seam cannot silently disable it."""
    embedder = CloudflareEmbeddingService(
        credentials=None, settings=CloudflareSettings(), ledger_path=tmp_path / "neurons.json"
    )
    assert embedder._post is not post_json
