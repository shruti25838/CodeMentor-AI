import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from codeatlas.observability.agent_metrics import AgentUsageCallback
from codeatlas.utils.config import REASONING_EFFORT_MODELS, AppConfig


class LlmProvider:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)

    def get_chat_model(self) -> BaseChatModel:
        return self._counted(self._build())

    @staticmethod
    def _counted(model: BaseChatModel) -> BaseChatModel:
        """Attach the metrics callback, so every call is attributed to the running agent."""
        return model.with_config({"callbacks": [AgentUsageCallback()]})

    def _build(self) -> BaseChatModel:
        if self._config.llm_provider == "openai":
            return ChatOpenAI(
                model=self._config.llm_model,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
                # Without this a streamed call reports no usage at all, so every token it spent
                # is missing from the metrics and from the on-disk ledger - and the ledger is
                # what a long run checks before deciding it still has budget.
                stream_usage=True,
            )
        elif self._config.llm_provider == "groq":
            import os

            return ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
                model=self._config.llm_model,
                temperature=self._config.llm_temperature,
                # Without an explicit budget the provider's own default applies; with one
                # that is too small a reasoning model returns an empty answer.
                max_tokens=self._config.llm_max_tokens,
                # See the openai branch: a streamed call is otherwise invisible to the ledger.
                stream_usage=True,
                # Empty unless the setting is on *and* this model takes it; see below.
                **reasoning_effort_kwargs(self._config),
            )
        return FallbackChatModel()


def reasoning_effort_kwargs(config: AppConfig) -> dict:
    """`model_kwargs={"reasoning_effort": ...}` when the setting applies, otherwise nothing.

    **Why model_kwargs and not a field.** The Groq client here is `ChatOpenAI` pointed at
    Groq's OpenAI-compatible endpoint, and the installed langchain-openai (0.1.22) has no
    `reasoning_effort` field - it predates the parameter. `model_kwargs` is that class's own
    mechanism for a body parameter it does not model: whatever is in it is merged into the
    request body alongside `model`, `temperature` and `max_tokens`. So the call Groq receives
    carries `"reasoning_effort": "low"` at the top level, which is where Groq documents it.
    Nothing about the output budget changes; `max_tokens` is still passed exactly as before.

    **Why it is sent to two models and no others.** Groq documents `reasoning_effort` as
    supported only by `openai/gpt-oss-20b` and `openai/gpt-oss-120b`. Sending it to any other
    model is at best ignored and at worst a 400 that turns every answer into an error, so the
    model name is checked here rather than trusted. The provider is checked too: the OpenAI
    branch builds a client against OpenAI's own API, where the same two names do not exist and
    the parameter means something else.

    Returning a dict rather than setting a field is deliberate - `**{}` adds nothing at all to
    the constructor call, so with the setting unset the client is built exactly as it was.
    """
    effort = config.llm_reasoning_effort
    if not effort:
        return {}
    if config.llm_provider != "groq" or config.llm_model not in REASONING_EFFORT_MODELS:
        return {}
    return {"model_kwargs": {"reasoning_effort": effort}}


class FallbackChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fallback"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = "LLM provider not configured. Set CODEATLAS_LLM_PROVIDER and relevant API keys."
        return self._create_chat_result(content)

    def _create_chat_result(self, content: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
