import pytest
from fastapi import HTTPException

from codeatlas.app.security import verify_api_key
from codeatlas.utils.config import AppConfig


def test_verify_api_key_disabled_allows() -> None:
    config = AppConfig(
        embedding_provider="hash",
        embedding_model="",
        index_dir=".codeatlas/indexes",
        state_dir=".codeatlas/state",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
        api_key=None,
        auth_enabled=False,
    )
    verify_api_key(config, x_api_key=None)


def test_verify_api_key_enabled_rejects_missing_key() -> None:
    config = AppConfig(
        embedding_provider="hash",
        embedding_model="",
        index_dir=".codeatlas/indexes",
        state_dir=".codeatlas/state",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
        api_key="secret",
        auth_enabled=True,
    )
    with pytest.raises(HTTPException):
        verify_api_key(config, x_api_key=None)


def test_verify_api_key_enabled_accepts_valid_key() -> None:
    config = AppConfig(
        embedding_provider="hash",
        embedding_model="",
        index_dir=".codeatlas/indexes",
        state_dir=".codeatlas/state",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
        api_key="secret",
        auth_enabled=True,
    )
    verify_api_key(config, x_api_key="secret")
