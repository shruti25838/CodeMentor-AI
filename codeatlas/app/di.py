from dataclasses import replace
from functools import lru_cache

from codeatlas.services.agents.coding_mentor_agent import CodingMentorAgent
from codeatlas.services.agents.memory_agent import MemoryAgent
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.agents.planner_agent import PlannerAgent
from codeatlas.services.agents.repo_analyst_agent import RepoAnalystAgent
from codeatlas.services.agents.retrieval_agent import RetrievalAgent
from codeatlas.services.dependency.import_graph_builder import ImportGraphBuilder
from codeatlas.services.ingestion.git_loader import GitRepositoryLoader
from codeatlas.services.llm.provider import LlmProvider
from codeatlas.services.memory.conversation import ConversationStore
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser
from codeatlas.services.qa.answer_service import AnswerService, RetrievalSettings
from codeatlas.services.qa.explain_service import CodeExplainService
from codeatlas.services.retrieval.cloudflare_embedder import CloudflareEmbeddingService, CloudflareSettings
from codeatlas.services.retrieval.dual_index import FALLBACK_NAME, DualIndexService
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.fastembed_embedder import FastEmbedEmbeddingService
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.retrieval.record_format import with_record_format
from codeatlas.services.retrieval.sentence_transformer_embedder import (
    SentenceTransformerEmbeddingService,
)
from codeatlas.services.state.repo_state_store import RepoStateStore
from codeatlas.utils.config import AppConfig, load_cloudflare_credentials, load_config, unknown_provider_message


@lru_cache
def get_repository_loader() -> GitRepositoryLoader:
    config = get_config()
    return GitRepositoryLoader(
        timeout_seconds=config.clone_timeout_seconds,
        max_bytes=config.max_repo_mb * 1024 * 1024,
        max_files=config.max_repo_files,
        max_download_bytes=config.max_download_mb * 1024 * 1024,
    )


@lru_cache
def get_ast_parser() -> TreeSitterAstParser:
    return TreeSitterAstParser()


@lru_cache
def get_dependency_graph_builder() -> ImportGraphBuilder:
    return ImportGraphBuilder()


@lru_cache
def get_code_retriever() -> FaissCodeRetriever:
    """The index the selected provider writes and reads.

    Each provider's vectors live in their own folder, because they are not comparable and a
    folder is the cheapest way to be sure they never meet. "cloudflare" therefore gets
    CODEATLAS_CLOUDFLARE_INDEX_DIR, and every other provider keeps CODEATLAS_INDEX_DIR, where
    the hash indexes already are.
    """
    config = get_config()
    if config.embedding_provider == "cloudflare":
        return FaissCodeRetriever(base_dir=config.cloudflare_index_dir)
    return FaissCodeRetriever(base_dir=config.index_dir)


@lru_cache
def get_fallback_retriever() -> FaissCodeRetriever | None:
    """The hash index that answers when the semantic one cannot, or None when there is no need.

    Only a provider that can fail has one. For hash itself this is None, and every question
    path behaves exactly as it did before any of this existed.
    """
    config = get_config()
    if config.embedding_provider != "cloudflare":
        return None
    return FaissCodeRetriever(base_dir=config.index_dir)


@lru_cache
def get_fallback_embedder() -> EmbeddingService | None:
    """The embedder that built the fallback index. An index and its embedder are one thing."""
    config = get_config()
    if config.embedding_provider != "cloudflare":
        return None
    return build_embedder(replace(config, embedding_provider=FALLBACK_NAME))


def build_embedder(config: AppConfig) -> EmbeddingService:
    """The embedder named by config.embedding_provider.

    Every known name is listed here. An unknown name raises rather than falling back to a
    default: a typo must not quietly produce a different kind of vector than the one asked for.
    """
    return with_record_format(_provider_embedder(config), config.record_format_for(config.embedding_provider))


def _provider_embedder(config: AppConfig) -> EmbeddingService:
    if config.embedding_provider == "hash":
        return HashEmbeddingService(lowercase=config.hash_lowercase, subtokens=config.hash_subtokens)
    if config.embedding_provider == "sentence":
        return SentenceTransformerEmbeddingService(model_name=config.embedding_model)
    if config.embedding_provider == "fastembed":
        return FastEmbedEmbeddingService(model_name=config.fastembed_model, batch_size=config.fastembed_batch_size)
    if config.embedding_provider == "cloudflare":
        return CloudflareEmbeddingService(
            credentials=load_cloudflare_credentials(),
            settings=cloudflare_settings(config),
        )
    raise ValueError(unknown_provider_message(config.embedding_provider))


def cloudflare_settings(config: AppConfig) -> CloudflareSettings:
    """The provider's settings, including the two that belong to other layers.

    prefix_metadata and drop_stopwords are carried into the embedder so that its signature
    covers them: the first changes the text of every document, the second the text of every
    question, so an index built under one value must not be searched under the other.

    Settings that only reorder results are deliberately not passed here. Changing one of them
    would otherwise invalidate every index built under the old value, and for this provider
    that means paying to embed the same documents again for identical vectors.
    """
    return CloudflareSettings(
        model=config.cloudflare_model,
        pooling=config.cloudflare_pooling,
        max_chars=config.cloudflare_max_chars,
        max_documents=config.cloudflare_max_documents,
        batch_size=config.cloudflare_batch_size,
        daily_neurons=config.cloudflare_daily_neurons,
        timeout_seconds=config.cloudflare_timeout_seconds,
        retries=config.cloudflare_retries,
        failure_threshold=config.cloudflare_failure_threshold,
        cooldown_seconds=config.cloudflare_cooldown_seconds,
        prefix_metadata=config.embed_prefix_metadata,
        drop_stopwords=config.drop_stopwords,
    )


@lru_cache
def get_embedder() -> EmbeddingService:
    return build_embedder(get_config())


def index_options(config: AppConfig) -> dict:
    """How the index is built, for the provider that is actually selected.

    Two providers differ. "fastembed" runs a transformer on CPU, far slower per document than
    the hash embedder, so it embeds one document at a time, sees only the first 3,000 characters
    of each, and is given a longer time limit. "cloudflare" sends batches over the network and
    clips each document to 2,000 characters, which is below the point where the model stops
    reading anyway. Every other provider keeps the settings tuned for hash, which is still the
    default.
    """
    if config.embedding_provider == "fastembed":
        return {
            "timeout_seconds": config.fastembed_index_timeout_seconds,
            "batch_size": config.fastembed_batch_size,
            "max_chars": config.fastembed_max_chars,
            "prefix_metadata": config.embed_prefix_metadata,
            **_record_options(config),
        }
    if config.embedding_provider == "cloudflare":
        return {
            # Its own limit, not the shared one: this is the only provider whose indexing
            # crosses a network, and the only one for which running out of time is survivable.
            "timeout_seconds": config.cloudflare_index_timeout_seconds,
            "batch_size": config.cloudflare_batch_size,
            "max_chars": config.cloudflare_max_chars,
            "prefix_metadata": config.embed_prefix_metadata,
            **_record_options(config),
        }
    return {
        "timeout_seconds": config.index_timeout_seconds,
        "max_chars": config.embed_max_chars,
        "prefix_metadata": config.embed_prefix_metadata,
        **_record_options(config),
    }


def _record_options(config: AppConfig) -> dict:
    """What a record is, for the index kind this config names.

    The format is per index kind, because the dual-index path builds two indexes of different
    kinds from the same records in the same analysis and they need not agree: `get_index_service`
    builds the fallback half from this same function with `embedding_provider` replaced by
    "hash", so the hash index is formatted as a hash index whatever the semantic one is doing.
    The two window settings stay global - nothing has measured them differing per kind.
    """
    return {
        "record_format": config.record_format_for(config.embedding_provider),
        "window_chars": config.record_window_chars,
        "window_overlap": config.record_window_overlap,
    }


@lru_cache
def get_index_service() -> CodeIndexService | DualIndexService:
    """What /analyze-repo builds a repository's index with.

    For every provider but "cloudflare" this is one index, as it always was. For "cloudflare"
    it is two: the hash index as well, built from the same records, so that a question still
    has something to search when the API cannot be reached, the budget is gone, or the
    repository was too large to embed in the first place.
    """
    config = get_config()
    primary = CodeIndexService(
        embedder=get_embedder(),
        retriever=get_code_retriever(),
        name=config.embedding_provider,
        **index_options(config),
    )
    fallback_embedder = get_fallback_embedder()
    fallback_retriever = get_fallback_retriever()
    if fallback_embedder is None or fallback_retriever is None:
        return primary

    hash_config = replace(config, embedding_provider=FALLBACK_NAME)
    fallback = CodeIndexService(
        embedder=fallback_embedder,
        retriever=fallback_retriever,
        name=FALLBACK_NAME,
        **index_options(hash_config),
    )
    return DualIndexService(
        semantic=primary,
        fallback=fallback,
        semantic_name=config.embedding_provider,
        max_documents=config.cloudflare_max_documents,
        # Only a provider that calls something has anything to report; getattr rather than an
        # isinstance check so a later provider needs no change here to be counted.
        usage=getattr(get_embedder(), "take_usage", None),
    )


def retrieval_settings_for(config: AppConfig, index_kind: str) -> RetrievalSettings:
    """How a question is searched against one kind of index.

    Everything here is global except `bm25_enabled`, which is per index kind for the reason
    docs/RESULTS.md records: fusion is a large gain for hash and a loss for cloudflare, and under
    the cloudflare provider both indexes exist at once, so one global answer cannot be right for
    both. The BM25 parameters themselves stay global - what was measured is on and off.
    """
    return RetrievalSettings(
        candidates=config.retrieval_candidates,
        rerank_weight=config.rerank_weight,
        rerank_subtokens=config.rerank_subtokens,
        drop_stopwords=config.drop_stopwords,
        skip_tests=config.skip_tests,
        bm25_enabled=config.bm25_enabled_for(index_kind),
        bm25_k1=config.bm25_k1,
        bm25_b=config.bm25_b,
        bm25_rrf_k=config.bm25_rrf_k,
        bm25_depth=config.bm25_depth,
        bm25_max_indexes=config.bm25_max_indexes,
    )


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    config = get_config()
    return retrieval_settings_for(config, config.embedding_provider)


@lru_cache
def get_fallback_retrieval_settings() -> RetrievalSettings | None:
    """How the fallback index is searched, or None when there is no fallback index.

    Present for exactly the providers `get_fallback_retriever` gives an index to, and built for
    the kind that index is - "hash" - rather than for the provider the server is running.
    """
    config = get_config()
    if config.embedding_provider != "cloudflare":
        return None
    return retrieval_settings_for(config, FALLBACK_NAME)


@lru_cache
def get_answer_service() -> AnswerService:
    return AnswerService(
        retriever=get_code_retriever(),
        embedder=get_embedder(),
        llm=get_llm_provider().get_chat_model(),
        settings=get_retrieval_settings(),
        fallback_retriever=get_fallback_retriever(),
        fallback_embedder=get_fallback_embedder(),
        fallback_settings=get_fallback_retrieval_settings(),
    )


@lru_cache
def get_retrieval_service() -> AnswerService:
    """The same retrieval as `get_answer_service`, with **no model attached**.

    Grounded chat searches with this and then streams the model itself, so the service it
    retrieves with must not build a chat client: constructing one needs a credential, and
    building it here would mean every /ask/stream request - including one for a repository
    that does not exist - wanted a provider before it could be refused.
    """
    return AnswerService(
        retriever=get_code_retriever(),
        embedder=get_embedder(),
        llm=None,
        settings=get_retrieval_settings(),
        fallback_retriever=get_fallback_retriever(),
        fallback_embedder=get_fallback_embedder(),
        fallback_settings=get_fallback_retrieval_settings(),
    )


@lru_cache
def get_explain_service() -> CodeExplainService:
    return CodeExplainService(state_store=get_repo_state_store(), llm=get_llm_provider().get_chat_model())


@lru_cache
def get_agent_orchestrator() -> AgentOrchestrator:
    llm = get_llm_provider().get_chat_model()
    answer_service = get_answer_service()
    repo_state_store = get_repo_state_store()

    planner = PlannerAgent(llm=llm)
    retrieval_agent = RetrievalAgent(answer_service=answer_service)
    analyst_agent = RepoAnalystAgent(state_store=repo_state_store, llm=llm)
    mentor_agent = CodingMentorAgent(llm=llm, answer_service=answer_service)
    memory_agent = MemoryAgent(conversations=get_conversation_store())

    return AgentOrchestrator(
        planner=planner,
        retrieval_agent=retrieval_agent,
        analyst_agent=analyst_agent,
        mentor_agent=mentor_agent,
        memory_agent=memory_agent,
    )


@lru_cache
def get_conversation_store() -> ConversationStore:
    config = get_config()
    return ConversationStore(
        max_turns=config.chat_history_turns,
        max_tokens=config.chat_history_tokens,
        ttl_seconds=config.chat_session_ttl_seconds,
        max_sessions=config.chat_max_sessions,
    )


@lru_cache
def get_repo_state_store() -> RepoStateStore:
    config = get_config()
    return RepoStateStore(base_dir=config.state_dir)


@lru_cache
def get_config() -> AppConfig:
    return load_config()


@lru_cache
def get_llm_provider() -> LlmProvider:
    return LlmProvider(get_config())
