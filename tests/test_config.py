from dataclasses import replace

import pytest

from codeatlas.app.di import build_embedder
from codeatlas.utils.config import EMBEDDING_PROVIDERS, load_config


def test_default_embedding_provider_is_hash(monkeypatch) -> None:
    """The default must not require an uninstalled dependency (sentence-transformers
    is not in requirements.txt), or indexing crashes on a fresh install/deploy."""
    monkeypatch.delenv("CODEATLAS_EMBEDDING_PROVIDER", raising=False)
    config = load_config()
    assert config.embedding_provider == "hash"


def test_default_llm_provider_matches_env_example(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_LLM_PROVIDER", raising=False)
    config = load_config()
    assert config.llm_provider == "groq"


def test_unknown_embedding_provider_is_rejected_at_startup(monkeypatch) -> None:
    """A typo must not fall through to a default: the vectors would silently be the wrong kind."""
    monkeypatch.setenv("CODEATLAS_EMBEDDING_PROVIDER", "sentance")
    with pytest.raises(ValueError) as exc:
        load_config()
    message = str(exc.value)
    assert "'sentance'" in message
    assert "hash" in message and "sentence" in message


@pytest.mark.parametrize("provider", EMBEDDING_PROVIDERS)
def test_every_listed_provider_is_accepted_and_can_be_built(monkeypatch, provider) -> None:
    monkeypatch.setenv("CODEATLAS_EMBEDDING_PROVIDER", provider)
    config = load_config()
    assert config.embedding_provider == provider
    # Building must not raise; constructing an embedder loads no model.
    assert build_embedder(config) is not None


@pytest.mark.parametrize("spelling", ["HASH", " hash", "Hash "])
def test_provider_names_are_case_and_space_insensitive(monkeypatch, spelling) -> None:
    monkeypatch.setenv("CODEATLAS_EMBEDDING_PROVIDER", spelling)
    assert load_config().embedding_provider == "hash"


def test_build_embedder_rejects_an_unknown_name_too() -> None:
    """load_config is not the only way an AppConfig is made, so the builder checks as well."""
    config = replace(load_config(), embedding_provider="word2vec")
    with pytest.raises(ValueError, match="word2vec"):
        build_embedder(config)
