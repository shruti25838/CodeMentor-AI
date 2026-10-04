from codeatlas.utils.config import load_config


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
