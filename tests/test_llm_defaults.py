"""The default model, its output budget, and what happens when the budget is too small.

The committed default used to be llama-3.1-8b-instant, which no longer exists on the Groq
account this was developed against: every call returned 404 before any answer. The default
is now a reasoning model, which brings its own failure — it spends part of its output budget
thinking before it writes anything, so too small a budget returns an empty answer.

Every test here uses a fake model; none needs a key or a network.
"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from codeatlas.services.agents.coding_mentor_agent import EMPTY_ANSWER, CodingMentorAgent
from codeatlas.services.llm.provider import LlmProvider
from codeatlas.services.retrieval.snippets import Snippet
from codeatlas.utils.config import DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_MODEL, AppConfig, load_config

SNIPPETS = [Snippet("a.py", 1, 2, "def add(a, b):\n    return a + b")]

# Measured against openai/gpt-oss-20b: a 64-token budget left the answer empty because the
# reasoning used all of it, 128 truncated it, and 256 or more completed.
REASONING_OVERHEAD = 200


class ReasoningModel(BaseChatModel):
    """Stands in for a reasoning model: thinks first, and writes only with budget left over.

    `max_tokens` arrives as a constructor argument, the same way the provider sets it.
    """

    state: Any = None

    def __init__(self, max_tokens: int = DEFAULT_LLM_MAX_TOKENS, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("S", (), {"max_tokens": max_tokens})()

    @property
    def _llm_type(self) -> str:
        return "reasoning"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        budget = self.state.max_tokens
        # The reasoning is spent first; whatever is left becomes the visible answer.
        content = "Here is the answer, citing a.py (lines 1-2)." if budget > REASONING_OVERHEAD else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


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


# ---------- the default itself ----------


def test_the_default_model_is_the_one_that_exists(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_LLM_MODEL", raising=False)
    assert load_config().llm_model == "openai/gpt-oss-20b"


def test_the_default_model_is_not_the_retired_one(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_LLM_MODEL", raising=False)
    assert load_config().llm_model != "llama-3.1-8b-instant"


def test_the_env_template_matches_the_code_default() -> None:
    from pathlib import Path

    template = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
    assert f"CODEATLAS_LLM_MODEL={DEFAULT_LLM_MODEL}" in template
    assert f"CODEATLAS_LLM_MAX_TOKENS={DEFAULT_LLM_MAX_TOKENS}" in template


# ---------- the environment still wins ----------


def test_the_model_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_LLM_MODEL", "some/other-model")
    assert load_config().llm_model == "some/other-model"


def test_the_output_budget_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_LLM_MAX_TOKENS", "99")
    assert load_config().llm_max_tokens == 99


def test_an_override_does_not_leak_into_the_next_load(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_LLM_MODEL", "temporary/model")
    assert load_config().llm_model == "temporary/model"
    monkeypatch.delenv("CODEATLAS_LLM_MODEL")
    assert load_config().llm_model == DEFAULT_LLM_MODEL


# ---------- the budget reaches the model ----------


def test_the_default_budget_is_large_enough_for_a_reasoning_model() -> None:
    answer = CodingMentorAgent(llm=ReasoningModel(DEFAULT_LLM_MAX_TOKENS)).answer("q", SNIPPETS)
    assert answer.strip()
    assert answer != EMPTY_ANSWER
    assert "a.py (lines 1-2)" in answer


def test_a_budget_that_is_too_small_gives_an_explanation_not_an_empty_answer() -> None:
    """The case this guards: the reply is empty and the app must not show nothing."""
    answer = CodingMentorAgent(llm=ReasoningModel(64)).answer("q", SNIPPETS)
    assert answer == EMPTY_ANSWER
    assert "CODEATLAS_LLM_MAX_TOKENS" in answer


@pytest.mark.parametrize("budget,expect_empty", [(64, True), (128, True), (256, False), (2048, False)])
def test_the_threshold_behaves_as_measured(budget: int, expect_empty: bool) -> None:
    answer = CodingMentorAgent(llm=ReasoningModel(budget)).answer("q", SNIPPETS)
    assert (answer == EMPTY_ANSWER) is expect_empty


def test_the_default_budget_is_above_the_threshold() -> None:
    assert DEFAULT_LLM_MAX_TOKENS > REASONING_OVERHEAD


def test_the_provider_passes_the_budget_to_the_model(monkeypatch) -> None:
    # Constructing the client needs a key to be present; none is used, no call is made.
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    model = LlmProvider(config(llm_max_tokens=1234))._build()
    assert getattr(model, "max_tokens", None) == 1234


def test_the_provider_passes_the_model_name(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    model = LlmProvider(config(llm_model="some/other-model"))._build()
    assert getattr(model, "model_name", None) == "some/other-model"


def test_the_openai_provider_also_gets_a_budget(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    model = LlmProvider(config(llm_provider="openai", llm_max_tokens=777))._build()
    assert getattr(model, "max_tokens", None) == 777


def test_a_provider_with_no_model_configured_still_answers() -> None:
    model = LlmProvider(config(llm_provider="none"))._build()
    assert model.invoke("hello").content
