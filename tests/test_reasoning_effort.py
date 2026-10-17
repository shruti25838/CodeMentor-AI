"""CODEATLAS_REASONING_EFFORT: rejected early, sent to two models, sent to nothing else.

Groq documents `reasoning_effort` as supported by `openai/gpt-oss-20b` and
`openai/gpt-oss-120b` only. Sending it anywhere else is at best ignored and at worst a 400,
which on this site reads as every answer being an error, so the three things worth pinning are
that it reaches those two models, that it reaches no others, and that leaving it unset changes
nothing at all about the call.

The last of those is the one that matters most. Unset is the default, every figure in
`docs/RESULTS.md` was measured under it, and "unset" here means the key is absent from the
constructor call - not present and empty, not present and None.

No test here needs a key or a network: the two that look at a real client are marked
`builds_model_client` and supply a plainly fake key, and the rest read
`reasoning_effort_kwargs` directly or go through a fake chat model.
"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from codeatlas.services.llm.provider import LlmProvider, reasoning_effort_kwargs
from codeatlas.utils.config import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    REASONING_EFFORT_MODELS,
    REASONING_EFFORTS,
    AppConfig,
    load_config,
    validate_reasoning_effort,
)

UNSUPPORTED_MODELS = ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "openai/gpt-oss", "gpt-4o-mini")


def config(**overrides) -> AppConfig:
    base = {
        "embedding_provider": "hash",
        "embedding_model": "",
        "index_dir": ".codeatlas/indexes",
        "state_dir": ".codeatlas/state",
        "llm_provider": "groq",
        "llm_model": DEFAULT_LLM_MODEL,
        "llm_temperature": 0.2,
        "api_key": None,
        "auth_enabled": False,
    }
    return AppConfig(**{**base, **overrides})


class FakeChatModel(BaseChatModel):
    """A chat model that records what it was built with, standing in for the real client.

    This is what "a fake client" means for the sent/not-sent tests: the provider builds it
    exactly as it builds `ChatOpenAI`, and the keyword arguments are then readable.
    """

    built_with: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.built_with = kwargs

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def build_with_a_fake_client(monkeypatch, **overrides) -> dict:
    """What the provider would pass to the client, with the client replaced by a fake."""
    monkeypatch.setattr("codeatlas.services.llm.provider.ChatOpenAI", FakeChatModel)
    return LlmProvider(config(**overrides))._build().built_with


# ---------- the setting is read, and unset is the default ----------


def test_unset_is_the_default(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_REASONING_EFFORT", raising=False)
    assert load_config().llm_reasoning_effort is None


@pytest.mark.parametrize("value", REASONING_EFFORTS)
def test_each_of_the_three_values_is_accepted(monkeypatch, value: str) -> None:
    monkeypatch.setenv("CODEATLAS_REASONING_EFFORT", value)
    assert load_config().llm_reasoning_effort == value


def test_an_empty_value_reads_as_unset(monkeypatch) -> None:
    """So that a deployment which sets every line of .env.example still gets the default."""
    monkeypatch.setenv("CODEATLAS_REASONING_EFFORT", "   ")
    assert load_config().llm_reasoning_effort is None


def test_case_and_surrounding_space_do_not_matter(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_REASONING_EFFORT", "  HIGH ")
    assert load_config().llm_reasoning_effort == "high"


@pytest.mark.parametrize("value", ["lo", "none", "off", "maximum", "0", "true", "verylow"])
def test_an_invalid_value_stops_the_app_starting(monkeypatch, value: str) -> None:
    monkeypatch.setenv("CODEATLAS_REASONING_EFFORT", value)
    with pytest.raises(ValueError, match="Unknown reasoning effort"):
        load_config()


def test_the_rejection_names_the_variable_and_every_valid_value(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_REASONING_EFFORT", "extreme")
    with pytest.raises(ValueError) as caught:
        load_config()
    message = str(caught.value)
    assert "CODEATLAS_REASONING_EFFORT" in message
    for value in REASONING_EFFORTS:
        assert value in message
    assert "unset" in message


def test_validate_rejects_what_is_not_a_level() -> None:
    assert validate_reasoning_effort("Low") == "low"
    with pytest.raises(ValueError):
        validate_reasoning_effort("highest")


# ---------- sent for the right models ----------


@pytest.mark.parametrize("model", REASONING_EFFORT_MODELS)
def test_it_is_sent_for_each_model_groq_documents(monkeypatch, model: str) -> None:
    built = build_with_a_fake_client(monkeypatch, llm_model=model, llm_reasoning_effort="low")

    assert built["model_kwargs"] == {"reasoning_effort": "low"}


def test_the_two_documented_models_are_the_ones_groq_names() -> None:
    assert REASONING_EFFORT_MODELS == ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
    assert DEFAULT_LLM_MODEL in REASONING_EFFORT_MODELS


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_whichever_level_is_set_is_the_level_sent(monkeypatch, effort: str) -> None:
    built = build_with_a_fake_client(monkeypatch, llm_reasoning_effort=effort)

    assert built["model_kwargs"] == {"reasoning_effort": effort}


# ---------- not sent otherwise ----------


@pytest.mark.parametrize("model", UNSUPPORTED_MODELS)
def test_it_is_not_sent_for_a_model_that_does_not_document_it(monkeypatch, model: str) -> None:
    built = build_with_a_fake_client(monkeypatch, llm_model=model, llm_reasoning_effort="high")

    assert "model_kwargs" not in built


def test_it_is_not_sent_on_the_openai_provider(monkeypatch) -> None:
    """Even though that branch builds the same class: those two are Groq model names."""
    built = build_with_a_fake_client(
        monkeypatch, llm_provider="openai", llm_model=DEFAULT_LLM_MODEL, llm_reasoning_effort="high"
    )

    assert "model_kwargs" not in built


def test_the_offline_fallback_model_is_unaffected() -> None:
    model = LlmProvider(config(llm_provider="none", llm_reasoning_effort="high"))._build()

    assert model.invoke("hello").content


# ---------- not sent when unset ----------


def test_nothing_is_added_to_the_call_when_it_is_unset(monkeypatch) -> None:
    built = build_with_a_fake_client(monkeypatch)

    assert "model_kwargs" not in built, "unset must mean the key is absent, not present and empty"


def test_unset_means_absent_rather_than_none_or_empty() -> None:
    assert reasoning_effort_kwargs(config()) == {}
    assert reasoning_effort_kwargs(config(llm_reasoning_effort=None)) == {}


@pytest.mark.parametrize("model", REASONING_EFFORT_MODELS)
def test_an_unset_setting_is_not_sent_even_for_a_model_that_takes_it(monkeypatch, model: str) -> None:
    built = build_with_a_fake_client(monkeypatch, llm_model=model)

    assert "model_kwargs" not in built


# ---------- the output token budget is untouched ----------


@pytest.mark.parametrize("effort", [None, *REASONING_EFFORTS])
def test_the_output_budget_is_passed_exactly_as_before(monkeypatch, effort) -> None:
    built = build_with_a_fake_client(monkeypatch, llm_reasoning_effort=effort, llm_max_tokens=1234)

    assert built["max_tokens"] == 1234


def test_the_default_budget_is_unchanged_by_this_setting(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_REASONING_EFFORT", "low")
    assert load_config().llm_max_tokens == DEFAULT_LLM_MAX_TOKENS


# ---------- the real client takes it ----------


@pytest.mark.builds_model_client
def test_the_real_groq_client_carries_it_in_model_kwargs(monkeypatch) -> None:
    """The mechanism itself: langchain-openai 0.1.22 has no reasoning_effort field, so the
    parameter travels in model_kwargs, which that class merges into the request body."""
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    model = LlmProvider(config(llm_reasoning_effort="low"))._build()

    assert model.model_kwargs == {"reasoning_effort": "low"}
    assert model.max_tokens == DEFAULT_LLM_MAX_TOKENS


@pytest.mark.builds_model_client
def test_the_real_groq_client_has_an_empty_model_kwargs_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    model = LlmProvider(config())._build()

    assert model.model_kwargs == {}


def test_the_env_template_documents_the_setting() -> None:
    from pathlib import Path

    template = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")

    assert "CODEATLAS_REASONING_EFFORT=" in template
    for model in REASONING_EFFORT_MODELS:
        assert model in template
