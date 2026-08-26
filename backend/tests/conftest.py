"""
tests/conftest.py — shared fixtures for the entire test suite.

Execution model
───────────────
• Unit tests (tests/unit/) instantiate service classes directly with a
  lightweight `test_settings` fixture.  No app startup, no HTTP, no DB.

• API tests (tests/api/) use FastAPI's TestClient.  Every heavy singleton
  (Embedder, FaissIndex, BM25Index, Reranker, LLMClient, NERService,
  MetadataStore, IngestionService, ChatService) is replaced via
  app.dependency_overrides so no real model is loaded and no real API is called.

Environment variables must be set *before* any app module is imported,
because config.py reads them at class-definition time via pydantic-settings.
"""

import os

# ── Force test-safe environment BEFORE any app import ────────────────────────
# These prevent LLMClient from raising ValueError (no key) and stop the
# startup event from trying to reach Langfuse over the network.
os.environ.setdefault("LLM_API_KEY", "test-fake-key-not-real")
os.environ.setdefault("GROQ_API_KEY", "test-fake-key-not-real")
os.environ["LANGFUSE_ENABLED"] = "false"  # force-disable, don't use setdefault

# ── Imports (after env is set) ────────────────────────────────────────────────
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

# Clear the lru_cache BEFORE main.py runs so the module-level
# `settings = get_settings()` in main.py picks up our env vars.
from app.core.config import get_settings, Settings
get_settings.cache_clear()

from app.main import app  # noqa: E402  (intentional late import)
from app.api.dependencies import (
    get_bm25_index,
    get_chat_service,
    get_embedder,
    get_faiss_index,
    get_ingestion_service,
    get_llm_client,
    get_metadata_store,
    get_ner_service,
    get_reranker,
)
from app.models.schemas import ChatResponse, UploadResponse
from app.services.rag.ner.ner_service import ExtractedEntities


# ── Shared settings fixture ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Lightweight Settings object for unit tests.
    Uses default storage paths (not overriding upload_dir/faiss_dir here,
    because unit tests don't touch the file system at all).
    Small chunk sizes so chunker tests run on short strings.
    """
    return Settings(
        llm_api_key="test-fake-key",
        langfuse_enabled=False,
        enable_web_fallback=False,
        chunk_size_tokens=100,
        chunk_overlap_tokens=15,
        min_chunk_tokens=5,
    )


# ── API test client fixture ───────────────────────────────────────────────────

@pytest.fixture
def api_client(tmp_path: Path):
    """
    FastAPI TestClient with ALL heavy dependencies swapped for fast stubs.
    - tmp_path: pytest's built-in function-scoped temp directory.
      Upload tests write files here; they are cleaned up automatically.
    - No real models loaded, no network calls made.
    - dependency_overrides is cleared after each test so tests are isolated.
    """
    # Create directories the upload route needs
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    faiss_dir = tmp_path / "faiss_index"
    faiss_dir.mkdir()

    _settings = Settings(
        llm_api_key="test-fake-key",
        langfuse_enabled=False,
        enable_web_fallback=False,
        upload_dir=upload_dir,
        faiss_dir=faiss_dir,
        metadata_db_path=tmp_path / "metadata.db",
    )

    # ── Fake factories (called once per request by FastAPI DI) ────────────────

    def _fake_faiss():
        m = MagicMock()
        m.index = MagicMock()
        m.index.ntotal = 0
        m.search.return_value = []
        m.add.return_value = None
        return m

    def _fake_bm25():
        m = MagicMock()
        m.chunk_ids = []
        m.search.return_value = []
        m.add.return_value = None
        return m

    def _fake_metadata_store():
        m = MagicMock()
        m.list_documents.return_value = []
        m.get_chunks_by_ids.return_value = []
        m.create_document.return_value = None
        m.save_chunks.return_value = None
        m.mark_ready.return_value = None
        m.mark_failed.return_value = None
        return m

    def _fake_ingestion_service():
        m = MagicMock()
        m.ingest_document.return_value = UploadResponse(
            document_id="test-doc-id",
            filename="test.pdf",
            num_pages=5,
            num_chunks=12,
            status="ready",
        )
        return m

    def _fake_chat_service():
        m = MagicMock()
        m.handle_message = AsyncMock(
            return_value=ChatResponse(
                answer="Net profit was Rs 125,682 thousand.",
                confidence_score=0.85,
                citations=[],
                used_web_fallback=False,
                web_sources=[],
                rewritten_query="net profit for the quarter",
                session_id="test-session-id",
            )
        )

        async def _stream_tokens(*args, **kwargs):
            yield "Net profit "
            yield "was Rs 125,682 thousand."

        m.handle_message_stream = AsyncMock(
            return_value=("test-session-id", _stream_tokens())
        )
        return m

    # ── Apply overrides ───────────────────────────────────────────────────────
    overrides = {
        get_settings:          lambda: _settings,
        get_embedder:          lambda: MagicMock(
                                   embed_query=MagicMock(return_value=[0.0] * 1024),
                                   embed_documents=MagicMock(return_value=[[0.0] * 1024]),
                               ),
        get_faiss_index:       _fake_faiss,
        get_bm25_index:        _fake_bm25,
        get_metadata_store:    _fake_metadata_store,
        get_ingestion_service: _fake_ingestion_service,
        get_llm_client:        lambda: MagicMock(
                                   complete=AsyncMock(return_value="Mocked LLM answer."),
                                   model_name="test-model",
                               ),
        get_reranker:          lambda: MagicMock(
                                   rerank=MagicMock(return_value=[]),
                                   _skip_threshold=0.026,
                                   settings=MagicMock(rerank_top_k=5),
                               ),
        get_ner_service:       lambda: MagicMock(
                                   extract=MagicMock(return_value=ExtractedEntities())
                               ),
        get_chat_service:      _fake_chat_service,
    }

    app.dependency_overrides.update(overrides)

    # raise_server_exceptions=True (default) — exceptions propagate so test
    # failures are clearly reported rather than silently becoming 500s.
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    # Always clean up so overrides don't leak between tests
    app.dependency_overrides.clear()
