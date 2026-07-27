"""
FastAPI dependency providers.

Design choice: heavy singletons (Embedder loads a ~1.3GB model, FaissIndex/
BM25Index load their persisted index files) are cached with lru_cache so
they're constructed exactly once per process, not once per request -- the
difference between a 3-second-per-request model load and effectively zero.
Lighter, stateless services (parser, cleaner, chunker) are cheap enough to
construct per-call but are still routed through Depends() so routes stay
untestable-logic-free and every collaborator can be swapped with a fake in
tests via FastAPI's dependency_overrides.
"""

from functools import lru_cache

import httpx
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.llm_client import LLMClient
from app.core.query_cache import QueryCache
from app.services.chat.chat_service import ChatService
from app.services.chat.session_store import ChatHistoryStore
from app.services.parser.cleaner import TextCleaner
from app.services.parser.pdf_parser import PDFParser
from app.services.rag.chunker import SemanticChunker
from app.services.rag.embeddings.embedder import Embedder
from app.services.rag.ingestion_service import IngestionService
from app.services.rag.metadata.store import MetadataStore
from app.services.rag.ner.ner_service import NERService
from app.services.rag.query_pipeline_service import QueryPipelineService
from app.services.rag.query_rewriter.rewriter import QueryRewriter
from app.services.rag.reranker.reranker import Reranker
from app.services.rag.retriever.bm25_index import BM25Index
from app.services.rag.retriever.faiss_index import FaissIndex
from app.services.rag.retriever.hybrid_retriever import HybridRetriever
from app.services.rag.generation.answer_service import AnswerService
from app.services.rag.generation.confidence_scorer import ConfidenceScorer
from app.services.rag.generation.context_compressor import ContextCompressor
from app.services.web_search.tavily_client import TavilyClient


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings())


@lru_cache
def get_faiss_index() -> FaissIndex:
    return FaissIndex(get_settings())


@lru_cache
def get_bm25_index() -> BM25Index:
    return BM25Index(get_settings())


@lru_cache
def get_metadata_store() -> MetadataStore:
    return MetadataStore(get_settings())


def get_ingestion_service(
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    bm25_index: BM25Index = Depends(get_bm25_index),
    metadata_store: MetadataStore = Depends(get_metadata_store),
) -> IngestionService:
    return IngestionService(
        parser=PDFParser(),
        cleaner=TextCleaner(),
        chunker=SemanticChunker(settings),
        embedder=embedder,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        metadata_store=metadata_store,
    )


# --- Phase 2: query pipeline ---------------------------------------------


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient(get_settings())


@lru_cache
def get_ner_service() -> NERService:
    return NERService(get_settings())


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings())


def get_query_rewriter(
    llm_client: LLMClient = Depends(get_llm_client),
) -> QueryRewriter:
    return QueryRewriter(llm_client)


def get_hybrid_retriever(
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
    faiss_index: FaissIndex = Depends(get_faiss_index),
    bm25_index: BM25Index = Depends(get_bm25_index),
) -> HybridRetriever:
    return HybridRetriever(settings, embedder, faiss_index, bm25_index)


def get_query_pipeline_service(
    rewriter: QueryRewriter = Depends(get_query_rewriter),
    ner_service: NERService = Depends(get_ner_service),
    hybrid_retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: Reranker = Depends(get_reranker),
    metadata_store: MetadataStore = Depends(get_metadata_store),
) -> QueryPipelineService:
    return QueryPipelineService(
        rewriter=rewriter,
        ner_service=ner_service,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        metadata_store=metadata_store,
    )


# --- Phase 3: answer generation ------------------------------------------


@lru_cache
def get_context_compressor() -> ContextCompressor:
    return ContextCompressor(get_settings())


@lru_cache
def get_confidence_scorer() -> ConfidenceScorer:
    return ConfidenceScorer(get_settings())


@lru_cache
def get_http_client() -> httpx.AsyncClient:
    """
    Shared, connection-pooled async HTTP client for outbound calls (Tavily).
    Constructed once, reused for the life of the process; closed in main.py's
    shutdown handler. See app/services/web_search/tavily_client.py for why
    this matters for latency.
    """
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    return httpx.AsyncClient(limits=limits)


def get_tavily_client(
    settings: Settings = Depends(get_settings),
    http_client: httpx.AsyncClient = Depends(get_http_client),
) -> TavilyClient:
    return TavilyClient(settings, http_client)


def get_answer_service(
    settings: Settings = Depends(get_settings),
    llm_client: LLMClient = Depends(get_llm_client),
    compressor: ContextCompressor = Depends(get_context_compressor),
    confidence_scorer: ConfidenceScorer = Depends(get_confidence_scorer),
    tavily_client: TavilyClient = Depends(get_tavily_client),
) -> AnswerService:
    return AnswerService(
        settings=settings,
        llm_client=llm_client,
        compressor=compressor,
        confidence_scorer=confidence_scorer,
        tavily_client=tavily_client,
    )


# --- Phase 4: chat, sessions, caching ------------------------------------


@lru_cache
def get_chat_history_store() -> ChatHistoryStore:
    return ChatHistoryStore(get_settings())


@lru_cache
def get_query_cache() -> QueryCache:
    return QueryCache(get_settings())


def get_chat_service(
    settings: Settings = Depends(get_settings),
    query_pipeline: QueryPipelineService = Depends(get_query_pipeline_service),
    answer_service: AnswerService = Depends(get_answer_service),
    history_store: ChatHistoryStore = Depends(get_chat_history_store),
    query_cache: QueryCache = Depends(get_query_cache),
) -> ChatService:
    return ChatService(
        settings=settings,
        query_pipeline=query_pipeline,
        answer_service=answer_service,
        history_store=history_store,
        query_cache=query_cache,
    )