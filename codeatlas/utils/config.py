import os
from dataclasses import dataclass

# A reasoning model spends part of its output budget on reasoning before it writes anything
# the user sees. Measured against openai/gpt-oss-20b with a short question: a budget of 64
# tokens produced an empty answer (the reasoning used all 64 and the reply was cut off),
# 128 produced a truncated one, and 256 or more completed. A real answer about code needs
# far more than that — one grounded answer used 878 tokens — so the default is generous.
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
DEFAULT_LLM_MAX_TOKENS = 2048


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


def load_config() -> AppConfig:
    max_repo_mb = int(os.getenv("CODEATLAS_MAX_REPO_MB", "100"))
    return AppConfig(
        embedding_provider=os.getenv("CODEATLAS_EMBEDDING_PROVIDER", "hash"),
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
    )
