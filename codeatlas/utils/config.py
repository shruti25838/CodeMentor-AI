import os
from dataclasses import dataclass

from codeatlas.services.retrieval.fastembed_embedder import (
    DEFAULT_FASTEMBED_BATCH_SIZE,
    DEFAULT_FASTEMBED_MODEL,
)

# A reasoning model spends part of its output budget on reasoning before it writes anything
# the user sees. Measured against openai/gpt-oss-20b with a short question: a budget of 64
# tokens produced an empty answer (the reasoning used all 64 and the reply was cut off),
# 128 produced a truncated one, and 256 or more completed. A real answer about code needs
# far more than that — one grounded answer used 878 tokens — so the default is generous.
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
DEFAULT_LLM_MAX_TOKENS = 2048

# Every embedding provider the app knows how to build. A name outside this list is a typo or a
# provider that was never wired up, and either way it must not fall through to some default:
# the vectors would silently be the wrong ones. See codeatlas/app/di.py:build_embedder.
EMBEDDING_PROVIDERS = ("hash", "sentence", "fastembed")


@dataclass(frozen=True)
class AppConfig:
    embedding_provider: str
    embedding_model: str
    index_dir: str
    state_dir: str
    llm_provider: str
    llm_model: str
    llm_temperature: float
    api_key: str | None
    auth_enabled: bool
    llm_max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    clone_timeout_seconds: int = 60
    index_timeout_seconds: int = 120
    max_repo_mb: int = 100
    max_repo_files: int = 5000
    max_download_mb: int = 200
    trusted_proxy_hops: int = 0
    clone_per_client_per_minute: int = 10
    clone_global_per_minute: int = 30
    max_concurrent_clones: int = 3
    llm_per_client_per_minute: int = 60
    llm_global_per_minute: int = 300
    rate_limit_max_clients: int = 10_000
    index_cache_enabled: bool = True
    chat_history_turns: int = 4
    chat_history_tokens: int = 1500
    chat_session_ttl_seconds: int = 1800
    chat_max_sessions: int = 1000
    # Retrieval and indexing settings, chosen with scripts/eval_retrieval.py on the dev questions and
    # checked on the held-out ones (see docs/RESULTS.md). Changing an index setting means earlier
    # indexes are not reused by the index cache.
    retrieval_candidates: int = 20
    rerank_weight: float = 1.0
    rerank_subtokens: bool = False
    drop_stopwords: bool = True
    skip_tests: bool = False
    embed_max_chars: int | None = None
    embed_prefix_metadata: bool = True
    hash_lowercase: bool = False
    hash_subtokens: bool = True
    # Settings below apply only when embedding_provider == "fastembed". That provider runs a real
    # transformer on CPU, which is far slower per document than the hash embedder, so it indexes one
    # document at a time, embeds only the first 3,000 characters of each, and gets its own time
    # limit. None of these three touch any other provider.
    fastembed_model: str = DEFAULT_FASTEMBED_MODEL
    fastembed_batch_size: int = DEFAULT_FASTEMBED_BATCH_SIZE
    fastembed_max_chars: int = 3000
    fastembed_index_timeout_seconds: int = 900
    # Folder of scripts/eval_retrieval.py --json results; read at startup into the retrieval gauges.
    eval_results_dir: str = "eval/results"


def unknown_provider_message(provider: str) -> str:
    return (
        f"Unknown embedding provider {provider!r}. "
        f"Set CODEATLAS_EMBEDDING_PROVIDER to one of: {', '.join(EMBEDDING_PROVIDERS)}. "
        f"'hash' is the default and needs no extra install."
    )


def validate_embedding_provider(provider: str) -> str:
    """The provider name, normalized; a name that is not a known provider raises ValueError."""
    normalized = provider.strip().lower()
    if normalized not in EMBEDDING_PROVIDERS:
        raise ValueError(unknown_provider_message(provider))
    return normalized


def load_config() -> AppConfig:
    max_repo_mb = int(os.getenv("CODEATLAS_MAX_REPO_MB", "100"))
    return AppConfig(
        embedding_provider=validate_embedding_provider(os.getenv("CODEATLAS_EMBEDDING_PROVIDER", "hash")),
        embedding_model=os.getenv("CODEATLAS_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        index_dir=os.getenv("CODEATLAS_INDEX_DIR", ".codeatlas/indexes"),
        state_dir=os.getenv("CODEATLAS_STATE_DIR", ".codeatlas/state"),
        llm_provider=os.getenv("CODEATLAS_LLM_PROVIDER", "groq"),
        llm_model=os.getenv("CODEATLAS_LLM_MODEL", DEFAULT_LLM_MODEL),
        llm_temperature=float(os.getenv("CODEATLAS_LLM_TEMPERATURE", "0.2")),
        llm_max_tokens=int(os.getenv("CODEATLAS_LLM_MAX_TOKENS", str(DEFAULT_LLM_MAX_TOKENS))),
        api_key=os.getenv("CODEATLAS_API_KEY"),
        auth_enabled=os.getenv("CODEATLAS_AUTH_ENABLED", "true").lower() == "true",
        clone_timeout_seconds=int(os.getenv("CODEATLAS_CLONE_TIMEOUT_SECONDS", "60")),
        index_timeout_seconds=int(os.getenv("CODEATLAS_INDEX_TIMEOUT_SECONDS", "120")),
        max_repo_mb=max_repo_mb,
        max_repo_files=int(os.getenv("CODEATLAS_MAX_REPO_FILES", "5000")),
        max_download_mb=int(os.getenv("CODEATLAS_MAX_DOWNLOAD_MB") or 2 * max_repo_mb),
        trusted_proxy_hops=int(os.getenv("CODEATLAS_TRUSTED_PROXY_HOPS", "0")),
        clone_per_client_per_minute=int(os.getenv("CODEATLAS_CLONE_PER_CLIENT_PER_MINUTE", "10")),
        clone_global_per_minute=int(os.getenv("CODEATLAS_CLONE_GLOBAL_PER_MINUTE", "30")),
        max_concurrent_clones=int(os.getenv("CODEATLAS_MAX_CONCURRENT_CLONES", "3")),
        llm_per_client_per_minute=int(os.getenv("CODEATLAS_LLM_PER_CLIENT_PER_MINUTE", "60")),
        llm_global_per_minute=int(os.getenv("CODEATLAS_LLM_GLOBAL_PER_MINUTE", "300")),
        rate_limit_max_clients=int(os.getenv("CODEATLAS_RATE_LIMIT_MAX_CLIENTS", "10000")),
        index_cache_enabled=os.getenv("CODEATLAS_INDEX_CACHE", "true").lower() == "true",
        chat_history_turns=int(os.getenv("CODEATLAS_CHAT_HISTORY_TURNS", "4")),
        chat_history_tokens=int(os.getenv("CODEATLAS_CHAT_HISTORY_TOKENS", "1500")),
        chat_session_ttl_seconds=int(os.getenv("CODEATLAS_CHAT_SESSION_TTL_SECONDS", "1800")),
        chat_max_sessions=int(os.getenv("CODEATLAS_CHAT_MAX_SESSIONS", "1000")),
        eval_results_dir=os.getenv("CODEATLAS_EVAL_RESULTS_DIR", "eval/results"),
        fastembed_model=os.getenv("CODEATLAS_FASTEMBED_MODEL", DEFAULT_FASTEMBED_MODEL),
        fastembed_batch_size=int(os.getenv("CODEATLAS_FASTEMBED_BATCH_SIZE", str(DEFAULT_FASTEMBED_BATCH_SIZE))),
        fastembed_max_chars=int(os.getenv("CODEATLAS_FASTEMBED_MAX_CHARS", "3000")),
        fastembed_index_timeout_seconds=int(os.getenv("CODEATLAS_FASTEMBED_INDEX_TIMEOUT_SECONDS", "900")),
    )
