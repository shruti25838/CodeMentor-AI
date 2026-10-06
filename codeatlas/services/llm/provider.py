import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from codeatlas.observability.agent_metrics import AgentUsageCallback
from codeatlas.utils.config import AppConfig


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
            )
        return FallbackChatModel()


class FallbackChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fallback"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = "LLM provider not configured. Set CODEATLAS_LLM_PROVIDER and relevant API keys."
        return self._create_chat_result(content)

    def _create_chat_result(self, content: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
