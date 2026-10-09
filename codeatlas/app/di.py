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
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.fastembed_embedder import FastEmbedEmbeddingService
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
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
    config = get_config()
    return FaissCodeRetriever(base_dir=config.index_dir)


def build_embedder(config: AppConfig) -> EmbeddingService:
    """The embedder named by config.embedding_provider.

    Every known name is listed here. An unknown name raises rather than falling back to a
    default: a typo must not quietly produce a different kind of vector than the one asked for.
    """
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
    """The provider's settings, including the three that belong to other layers.

    prefix_metadata, subtokens and drop_stopwords are carried into the embedder only so that
    its signature covers them, which is what makes the index guard refuse an index built under
    a different retrieval configuration. Changing one of them means the repository has to be
    indexed again, and for this provider that costs neurons - a real price for a real
    guarantee, and the reason the eval's vector cache is keyed on the text instead.
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
        subtokens=config.rerank_subtokens,
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
        }
    if config.embedding_provider == "cloudflare":
        return {
            # The time limit is the shared one. A repository at the 600-document cap is twelve
            # calls of 50, so the default 120 s is ample; a raised cap may need it raised too.
            "timeout_seconds": config.index_timeout_seconds,
            "batch_size": config.cloudflare_batch_size,
            "max_chars": config.cloudflare_max_chars,
            "prefix_metadata": config.embed_prefix_metadata,
        }
    return {
        "timeout_seconds": config.index_timeout_seconds,
        "max_chars": config.embed_max_chars,
        "prefix_metadata": config.embed_prefix_metadata,
    }


@lru_cache
def get_index_service() -> CodeIndexService:
    return CodeIndexService(embedder=get_embedder(), retriever=get_code_retriever(), **index_options(get_config()))


@lru_cache
def get_retrieval_settings() -> RetrievalSettings:
    config = get_config()
    return RetrievalSettings(
        candidates=config.retrieval_candidates,
        rerank_weight=config.rerank_weight,
        rerank_subtokens=config.rerank_subtokens,
        drop_stopwords=config.drop_stopwords,
        skip_tests=config.skip_tests,
    )


@lru_cache
def get_answer_service() -> AnswerService:
    return AnswerService(
        retriever=get_code_retriever(),
        embedder=get_embedder(),
        llm=get_llm_provider().get_chat_model(),
        settings=get_retrieval_settings(),
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
