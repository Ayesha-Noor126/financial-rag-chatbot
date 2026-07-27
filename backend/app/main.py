"""
FastAPI application entrypoint.

Startup sequence (relevant to prompt versioning)
─────────────────────────────────────────────────
1. init_langfuse(settings) — creates the Langfuse() client AND calls
   prompt_manager.init() so the prompt layer has a valid reference.
2. prompt_manager.warm() — pre-fetches all three managed prompts concurrently
   in the thread pool.  This means the first real request always hits an
   already-populated in-process cache and pays zero network latency for
   prompt retrieval.  If Langfuse is unreachable at startup, warm() logs a
   warning and every prompt falls back to its hardcoded constant — the app
   starts and serves requests normally.

The shutdown handler flushes pending Langfuse trace events and closes the
shared httpx connection pool used by TavilyClient.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.dependencies import get_http_client
from app.api.routes import chat, debug_answer, debug_query, document, health, history, upload
from app.core import prompt_manager
from app.core.config import get_settings
from app.core.langfuse_client import flush_langfuse, get_langfuse, init_langfuse

# Import the hardcoded fallback strings so warm() can pass them to
# prompt_manager.warm() — no network call is needed to build the fallback map.
from app.services.rag.generation.answer_prompt import SYSTEM_PROMPT as _ANSWER_FALLBACK
from app.services.rag.generation.answer_service import WEB_SYSTEM_PROMPT as _WEB_FALLBACK
from app.services.rag.query_rewriter.rewriter import SYSTEM_PROMPT as _REWRITER_FALLBACK

settings = get_settings()
app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    # Step 1 — initialise Langfuse observability client.
    # This also calls prompt_manager.init() internally so the prompt layer
    # is ready before we attempt the warm-up below.
    init_langfuse(settings)

    lf = get_langfuse()
    if lf:
        logger.info(
            "Langfuse active — host={}, key_set={}",
            settings.langfuse_host,
            bool(settings.langfuse_public_key),
        )

    # Step 2 — pre-fetch all managed prompts concurrently.
    # Maps each Langfuse prompt name to its hardcoded fallback string.
    # warm() fetches in parallel via asyncio.gather; any individual failure
    # is caught and logged without aborting the startup sequence.
    await prompt_manager.warm(
        prompt_names=[
            settings.langfuse_prompt_answer_system,
            settings.langfuse_prompt_web_system,
            settings.langfuse_prompt_rewriter_system,
        ],
        fallbacks={
            settings.langfuse_prompt_answer_system: _ANSWER_FALLBACK,
            settings.langfuse_prompt_web_system:    _WEB_FALLBACK,
            settings.langfuse_prompt_rewriter_system: _REWRITER_FALLBACK,
        },
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    # Flush pending Langfuse trace events before the process exits.
    flush_langfuse()
    # Close the shared httpx connection pool used by TavilyClient.
    await get_http_client().aclose()


app.include_router(upload.router,       tags=["ingestion"])
app.include_router(chat.router,         tags=["chat"])
app.include_router(history.router,      tags=["chat"])
app.include_router(document.router,     tags=["ingestion"])
app.include_router(health.router,       tags=["system"])
app.include_router(debug_query.router,  tags=["debug"])
app.include_router(debug_answer.router, tags=["debug"])
