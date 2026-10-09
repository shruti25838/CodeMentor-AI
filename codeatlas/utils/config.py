import os
from dataclasses import dataclass

from dotenv import load_dotenv

from codeatlas.services.retrieval.cloudflare_embedder import (
    DEFAULT_CLOUDFLARE_BATCH_SIZE,
    DEFAULT_CLOUDFLARE_COOLDOWN_SECONDS,
    DEFAULT_CLOUDFLARE_DAILY_NEURONS,
    DEFAULT_CLOUDFLARE_FAILURE_THRESHOLD,
    DEFAULT_CLOUDFLARE_MAX_CHARS,
    DEFAULT_CLOUDFLARE_MAX_DOCUMENTS,
    DEFAULT_CLOUDFLARE_MODEL,
    DEFAULT_CLOUDFLARE_POOLING,
    DEFAULT_CLOUDFLARE_RETRIES,
    DEFAULT_CLOUDFLARE_TIMEOUT_SECONDS,
    POOLINGS,
)
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
EMBEDDING_PROVIDERS = ("hash", "sentence", "fastembed", "cloudflare")


CLOUDFLARE_ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_API_TOKEN_ENV = "CLOUDFLARE_API_TOKEN"


@dataclass(frozen=True)
class CloudflareCredentials:
    """The two values a Workers AI call needs. Deliberately not fields on AppConfig.

    AppConfig is the object anything can print while working out why a setting is what it is;
    a token inside it would reach a log the first time someone did. Keeping credentials in a
    separate object that nothing logs means that cannot happen by accident, and its repr is
    overridden so that printing this one does not leak either.
    """

    account_id: str
    api_token: str

    def __repr__(self) -> str:
        return "CloudflareCredentials(account_id=<set>, api_token=<redacted>)"


def load_cloudflare_credentials() -> CloudflareCredentials | None:
    """Both credentials from the environment, or None when either is missing.

    .env is loaded here as well as in app/main.py, because a script has no create_app() to do
    it. load_dotenv does not override a variable that is already set, so the environment still
    wins over the file either way.
    """
    load_dotenv()
    account_id = (os.getenv(CLOUDFLARE_ACCOUNT_ID_ENV) or "").strip()
    api_token = (os.getenv(CLOUDFLARE_API_TOKEN_ENV) or "").strip()
    if not account_id or not api_token:
        return None
    return CloudflareCredentials(account_id=account_id, api_token=api_token)


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
    # Settings below apply only when embedding_provider == "cloudflare". That provider calls a
    # paid API on a free allowance, so besides the model and the pooling it has a per-document
    # clip, a ceiling on documents per repository, a batch size, a daily budget in estimated
    # neurons, and the timeout, retry and circuit-breaker settings that decide how quickly it
    # gives up and falls back to the hash index. None of them touch any other provider.
    cloudflare_model: str = DEFAULT_CLOUDFLARE_MODEL
    cloudflare_pooling: str = DEFAULT_CLOUDFLARE_POOLING
    cloudflare_max_chars: int = DEFAULT_CLOUDFLARE_MAX_CHARS
    cloudflare_max_documents: int = DEFAULT_CLOUDFLARE_MAX_DOCUMENTS
    cloudflare_batch_size: int = DEFAULT_CLOUDFLARE_BATCH_SIZE
    cloudflare_daily_neurons: float = DEFAULT_CLOUDFLARE_DAILY_NEURONS
    cloudflare_timeout_seconds: float = DEFAULT_CLOUDFLARE_TIMEOUT_SECONDS
    cloudflare_retries: int = DEFAULT_CLOUDFLARE_RETRIES
    cloudflare_failure_threshold: int = DEFAULT_CLOUDFLARE_FAILURE_THRESHOLD
    cloudflare_cooldown_seconds: float = DEFAULT_CLOUDFLARE_COOLDOWN_SECONDS
    cloudflare_index_dir: str = ".codeatlas/indexes-cloudflare"
    # Folder of scripts/eval_retrieval.py --json results; read at startup into the retrieval gauges.
    eval_results_dir: str = "eval/results"


def unknown_pooling_message(pooling: str) -> str:
    return (
        f"Unknown pooling {pooling!r}. Set CODEATLAS_CLOUDFLARE_POOLING to one of: {', '.join(POOLINGS)}. "
        f"'{DEFAULT_CLOUDFLARE_POOLING}' is the default; the two are not interchangeable, and an index "
        f"built under one must not be searched under the other."
    )


def validate_pooling(pooling: str) -> str:
    normalized = pooling.strip().lower()
    if normalized not in POOLINGS:
        raise ValueError(unknown_pooling_message(pooling))
    return normalized


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
        # Retrieval and indexing, all seven with the defaults that docs/RESULTS.md measured.
        # Changing one of the last two means earlier indexes are not reused by the index cache.
        retrieval_candidates=_positive_int("CODEATLAS_RETRIEVAL_CANDIDATES", 20),
        rerank_weight=_weight("CODEATLAS_RERANK_WEIGHT", 1.0),
        rerank_subtokens=_flag("CODEATLAS_RERANK_SUBTOKENS", False),
        drop_stopwords=_flag("CODEATLAS_DROP_STOPWORDS", True),
        skip_tests=_flag("CODEATLAS_SKIP_TESTS", False),
        # 0 means no limit, which is what None is inside the app.
        embed_max_chars=_non_negative_int("CODEATLAS_EMBED_MAX_CHARS", 0) or None,
        embed_prefix_metadata=_flag("CODEATLAS_EMBED_PREFIX_METADATA", True),
        fastembed_model=os.getenv("CODEATLAS_FASTEMBED_MODEL", DEFAULT_FASTEMBED_MODEL),
        fastembed_batch_size=int(os.getenv("CODEATLAS_FASTEMBED_BATCH_SIZE", str(DEFAULT_FASTEMBED_BATCH_SIZE))),
        fastembed_max_chars=int(os.getenv("CODEATLAS_FASTEMBED_MAX_CHARS", "3000")),
        fastembed_index_timeout_seconds=int(os.getenv("CODEATLAS_FASTEMBED_INDEX_TIMEOUT_SECONDS", "900")),
        cloudflare_model=os.getenv("CODEATLAS_CLOUDFLARE_MODEL", DEFAULT_CLOUDFLARE_MODEL),
        cloudflare_pooling=validate_pooling(os.getenv("CODEATLAS_CLOUDFLARE_POOLING", DEFAULT_CLOUDFLARE_POOLING)),
        cloudflare_max_chars=_positive_int("CODEATLAS_CLOUDFLARE_MAX_CHARS", DEFAULT_CLOUDFLARE_MAX_CHARS),
        cloudflare_max_documents=_positive_int("CODEATLAS_CLOUDFLARE_MAX_DOCUMENTS", DEFAULT_CLOUDFLARE_MAX_DOCUMENTS),
        cloudflare_batch_size=_positive_int("CODEATLAS_CLOUDFLARE_BATCH_SIZE", DEFAULT_CLOUDFLARE_BATCH_SIZE),
        cloudflare_daily_neurons=_positive_float(
            "CODEATLAS_CLOUDFLARE_DAILY_NEURONS", DEFAULT_CLOUDFLARE_DAILY_NEURONS
        ),
        cloudflare_timeout_seconds=_positive_float(
            "CODEATLAS_CLOUDFLARE_TIMEOUT_SECONDS", DEFAULT_CLOUDFLARE_TIMEOUT_SECONDS
        ),
        cloudflare_retries=_non_negative_int("CODEATLAS_CLOUDFLARE_RETRIES", DEFAULT_CLOUDFLARE_RETRIES),
        cloudflare_failure_threshold=_positive_int(
            "CODEATLAS_CLOUDFLARE_FAILURE_THRESHOLD", DEFAULT_CLOUDFLARE_FAILURE_THRESHOLD
        ),
        cloudflare_cooldown_seconds=_positive_float(
            "CODEATLAS_CLOUDFLARE_COOLDOWN_SECONDS", DEFAULT_CLOUDFLARE_COOLDOWN_SECONDS
        ),
        cloudflare_index_dir=os.getenv("CODEATLAS_CLOUDFLARE_INDEX_DIR", ".codeatlas/indexes-cloudflare"),
    )


TRUE_WORDS = ("true", "1", "yes", "on")
FALSE_WORDS = ("false", "0", "no", "off")


def _flag(name: str, default: bool) -> bool:
    """One on/off setting, where anything that is neither stops the app starting.

    Deliberately stricter than the older settings in this file, which read `== "true"` and so
    quietly treat a typo as off. A retrieval setting that silently reverts is worse than most:
    the app runs, the answers change, and a measurement describes something other than what was
    configured.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in TRUE_WORDS:
        return True
    if value in FALSE_WORDS:
        return False
    raise ValueError(f"{name} must be one of {', '.join((*TRUE_WORDS, *FALSE_WORDS))}, not {raw!r}")


def _weight(name: str, default: float) -> float:
    """A rerank weight: 1.0 is keyword overlap alone, 0.0 is vector order alone, nothing else is meaningful."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number between 0 and 1, not {raw!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1, not {raw!r}")
    return value


def _number(name: str, default: float, cast, minimum: float, bound: str):
    """One numeric setting from the environment, rejected at startup rather than at first use.

    A typo in a budget or a batch size is worth refusing to start over: the alternatives are a
    crash in the middle of indexing, or a setting that silently reverts to its default and a
    measurement that quietly describes something other than what was configured.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return cast(default)
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be {bound}, not {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be {bound}, not {raw!r}")
    return value


def _positive_int(name: str, default: int) -> int:
    return _number(name, default, int, 1, "a whole number of at least 1")


def _non_negative_int(name: str, default: int) -> int:
    return _number(name, default, int, 0, "a whole number of at least 0")


def _positive_float(name: str, default: float) -> float:
    return _number(name, default, float, 0.000001, "a number greater than 0")
