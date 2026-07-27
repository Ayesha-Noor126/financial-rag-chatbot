"""
Chat orchestrator -- the service behind POST /chat.

Design choice: this is the ONLY place that knows about session/conversation
concerns (history persistence, query caching). QueryPipelineService and
AnswerService stay history-agnostic and stateless, so they remain directly
reusable from the Phase 2/3 debug routes without dragging session plumbing
into them.

Caching policy: only cache confident, document-grounded answers (see
handle_message). A low-confidence or web-fallback answer might reflect a
fast-changing fact (stock price, news) that shouldn't be served stale on a
later identical question within the cache TTL.
"""
"""
Chat orchestrator -- the service behind POST /chat.
...
"""

from collections.abc import AsyncIterator

from loguru import logger

from app.core.config import Settings
from app.core.query_cache import QueryCache
from app.models.schemas import ChatResponse, MessageRole
from app.services.chat.session_store import ChatHistoryStore
from app.services.rag.generation.answer_service import METADATA_SENTINEL, AnswerService
from app.services.rag.query_pipeline_service import QueryPipelineService


class ChatService:
    def __init__(
        self,
        settings: Settings,
        query_pipeline: QueryPipelineService,
        answer_service: AnswerService,
        history_store: ChatHistoryStore,
        query_cache: QueryCache,
    ):
        self.settings = settings
        self.query_pipeline = query_pipeline
        self.answer_service = answer_service
        self.history_store = history_store
        self.query_cache = query_cache

    async def handle_message(
        self, session_id: str | None, message: str, trace=None
    ) -> ChatResponse:
        sid = self.history_store.ensure_session(session_id)
        if trace:
            # route created the trace before sid existed (session_id may have
            # been None on a first message) -- backfill it now so this trace
            # groups correctly with the rest of the conversation in Langfuse.
            trace.update(session_id=sid)
        self.history_store.add_message(sid, MessageRole.USER, message)

        cached = self.query_cache.get(message)
        if cached is not None:
            logger.info(f"Query cache hit for session {sid}")
            response = cached.model_copy(update={"session_id": sid})
            if trace:
                trace.span(name="query-cache-hit", output={"cached": True}).end()
        else:
            history = self.history_store.get_recent_turns(sid, n=3)
            pipeline_result = await self.query_pipeline.run(message, history=history, trace=trace)
            response = await self.answer_service.generate(pipeline_result, trace=trace)
            response.session_id = sid

            if (
                response.confidence_score >= self.settings.confidence_threshold
                and not response.used_web_fallback
            ):
                self.query_cache.set(message, response)

        self.history_store.add_message(
            sid, MessageRole.ASSISTANT, response.answer, citations=response.citations
        )
        return response

    async def handle_message_stream(
        self, session_id: str | None, message: str, trace=None
    ) -> tuple[str, AsyncIterator[str]]:
        """
        Returns (session_id, token_stream). The caller (the /chat route)
        wraps token_stream into a StreamingResponse. Cache lookups are
        skipped for streaming requests -- a cache hit would have to be
        "replayed" as a single synthetic chunk anyway, which defeats the
        purpose of asking for a stream, so streaming always goes live.
        """
        sid = self.history_store.ensure_session(session_id)
        if trace:
            trace.update(session_id=sid)
        self.history_store.add_message(sid, MessageRole.USER, message)
        history = self.history_store.get_recent_turns(sid, n=3)
        pipeline_result = await self.query_pipeline.run(message, history=history, trace=trace)

        async def token_stream() -> AsyncIterator[str]:
            full_answer = ""
            final_response: ChatResponse | None = None
            async for piece in self.answer_service.generate_stream(pipeline_result, trace=trace):
                if piece.startswith(METADATA_SENTINEL):
                    final_response = ChatResponse.model_validate_json(
                        piece[len(METADATA_SENTINEL):]
                    )
                    yield piece
                else:
                    full_answer += piece
                    yield piece

            if final_response is not None:
                self.history_store.add_message(
                    sid,
                    MessageRole.ASSISTANT,
                    full_answer,
                    citations=final_response.citations,
                )

        return sid, token_stream()