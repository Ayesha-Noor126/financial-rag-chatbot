"""
Answer generation orchestrator — async + streaming.

Langfuse Prompt Versioning changes (minimal, targeted)
───────────────────────────────────────────────────────
Two prompts previously hardcoded as module-level string constants are now
fetched at call time via the prompt manager:

  SYSTEM_PROMPT      → answer_prompt.get_answer_system_prompt()
                        (fin-rag-answer-system in Langfuse)
  WEB_SYSTEM_PROMPT  → prompt_manager.get(settings.langfuse_prompt_web_system,
                                           fallback=WEB_SYSTEM_PROMPT)
                        (fin-rag-web-system in Langfuse)

Both calls are async and return instantly from the in-process cache on the
hot path (after the startup warm-up in main.py).  The first call per process
lifetime may take one network round-trip (~20–50 ms) but that is hidden inside
the startup warm-up, not in request latency.

Everything else — tracing, eval, citations, streaming, web fallback, no-answer
detection — is identical to the previous version.  No business logic changed.
"""

import re
import time
from collections.abc import AsyncIterator

from loguru import logger

from app.core import prompt_manager
from app.core.config import Settings
from app.core.llm_client import LLMClient
from app.models.schemas import ChatResponse, Citation, WebSource
from app.services.rag.evaluation.eval_service import RAGEvaluationService
from app.services.rag.generation.answer_prompt import (
    NO_ANSWER_PHRASE,
    build_user_prompt,
    get_answer_system_prompt,
)
from app.services.rag.generation.confidence_scorer import ConfidenceScorer
from app.services.rag.generation.context_compressor import CompressedContext, ContextCompressor
from app.services.rag.query_pipeline_service import QueryPipelineResult
from app.services.web_search.tavily_client import TavilyClient

WEB_INTENT_PATTERN = re.compile(
    r"\b(latest|recent|current|currently|today|this week|this month|"
    r"stock price|share price|news|now)\b",
    re.IGNORECASE,
)

CITATION_MARKER_RE = re.compile(r"\[Source (\d+)\]")
METADATA_SENTINEL = "\n[[METADATA]]"

# Hardcoded fallback for the web-search system prompt.
# Langfuse name: fin-rag-web-system (set in settings.langfuse_prompt_web_system).
# seed_prompts.py registers this exact string as the first version.
WEB_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the provided web search snippets. Be concise and factual. "
    "If the snippets do not contain enough information to answer, say so clearly. "
    "Do not invent numbers or facts."
)


class AnswerService:
    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        compressor: ContextCompressor,
        confidence_scorer: ConfidenceScorer,
        tavily_client: TavilyClient,
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.compressor = compressor
        self.confidence_scorer = confidence_scorer
        self.tavily_client = tavily_client
        self.eval_service = RAGEvaluationService()

    # ── Public: non-streaming ────────────────────────────────────────────

    async def generate(self, pipeline_result: QueryPipelineResult, trace=None) -> ChatResponse:
        compressed = self.compressor.compress(pipeline_result.reranked_chunks)
        if trace:
            trace.span(
                name="context-compression",
                input={"num_reranked_chunks": len(pipeline_result.reranked_chunks)},
                output={"num_compressed_chunks": len(compressed)},
            ).end()

        if not compressed:
            return await self._no_answer_response(pipeline_result, confidence=0.0, trace=trace)

        confidence = self.confidence_scorer.score(pipeline_result.entities, compressed)
        if trace:
            trace.span(name="confidence-scoring", output={"confidence": confidence}).end()

        answer_text = await self._generate_grounded_answer(
            pipeline_result.rewritten_query, compressed, trace=trace
        )
        is_no_answer = NO_ANSWER_PHRASE in answer_text
        citations = [] if is_no_answer else self._extract_citations(answer_text, compressed)

        needs_web = self._needs_web_fallback(pipeline_result, confidence, is_no_answer)
        web_sources = (
            await self._tavily_search(pipeline_result.rewritten_query, trace=trace)
            if needs_web else []
        )

        if is_no_answer and web_sources:
            answer_text = await self._generate_web_answer(
                pipeline_result.rewritten_query, web_sources, trace=trace
            )

        eval_metrics = self.eval_service.evaluate(
            question=pipeline_result.original_query,
            answer=answer_text,
            contexts=[c.chunk.text for c in compressed],
            expected_answer=None,
        )
        if trace:
            trace.score(name="faithfulness", value=eval_metrics["faithfulness"])
            trace.score(name="answer_relevancy", value=eval_metrics["answer_relevancy"])

        response = ChatResponse(
            answer=answer_text,
            confidence_score=confidence,
            citations=citations,
            used_web_fallback=bool(web_sources),
            web_sources=web_sources,
            rewritten_query=pipeline_result.rewritten_query,
            faithfulness=eval_metrics["faithfulness"],
            answer_relevancy=eval_metrics["answer_relevancy"],
            precision=eval_metrics["precision"],
            recall=eval_metrics["recall"],
        )
        if trace:
            trace.update(
                output=answer_text,
                metadata={
                    "confidence": confidence,
                    "used_web_fallback": bool(web_sources),
                    "num_citations": len(citations),
                },
            )
        return response

    # ── Public: streaming ────────────────────────────────────────────────

    async def generate_stream(
        self, pipeline_result: QueryPipelineResult, trace=None
    ) -> AsyncIterator[str]:
        compressed = self.compressor.compress(pipeline_result.reranked_chunks)
        if not compressed:
            no_answer = await self._no_answer_response(pipeline_result, confidence=0.0, trace=trace)
            yield no_answer.answer
            yield METADATA_SENTINEL + no_answer.model_dump_json()
            return

        confidence = self.confidence_scorer.score(pipeline_result.entities, compressed)

        # Fetch the live system prompt once per stream call.
        # Returns from the in-process cache; never blocks the event loop.
        system_prompt = await get_answer_system_prompt()
        user_prompt = build_user_prompt(pipeline_result.rewritten_query, compressed)

        full_text = ""
        generation = None
        if trace:
            generation = trace.generation(
                name="grounded-answer-stream",
                model=getattr(self.llm_client, "model_name", "llama-3.3-70b-versatile"),
                input=user_prompt,
            )
        start = time.perf_counter()
        try:
            async for token in self.llm_client.stream_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=600,
            ):
                full_text += token
                yield token
        except Exception:
            logger.exception("Streaming answer generation failed")
            full_text = NO_ANSWER_PHRASE
            yield full_text
        finally:
            if generation:
                generation.end(
                    output=full_text,
                    metadata={"latency_seconds": round(time.perf_counter() - start, 3)},
                )

        is_no_answer = NO_ANSWER_PHRASE in full_text
        citations = [] if is_no_answer else self._extract_citations(full_text, compressed)
        needs_web = self._needs_web_fallback(pipeline_result, confidence, is_no_answer)
        web_sources = (
            await self._tavily_search(pipeline_result.rewritten_query, trace=trace)
            if needs_web else []
        )

        if is_no_answer and web_sources:
            full_text = await self._generate_web_answer(
                pipeline_result.rewritten_query, web_sources, trace=trace
            )

        eval_metrics = self.eval_service.evaluate(
            question=pipeline_result.original_query,
            answer=full_text,
            contexts=[c.chunk.text for c in compressed],
            expected_answer=None,
        )

        final = ChatResponse(
            answer=full_text,
            confidence_score=confidence,
            citations=citations,
            used_web_fallback=bool(web_sources),
            web_sources=web_sources,
            rewritten_query=pipeline_result.rewritten_query,
            faithfulness=eval_metrics["faithfulness"],
            answer_relevancy=eval_metrics["answer_relevancy"],
            precision=eval_metrics["precision"],
            recall=eval_metrics["recall"],
        )
        if trace:
            trace.update(
                output=full_text,
                metadata={
                    "confidence": confidence,
                    "used_web_fallback": bool(web_sources),
                },
            )
        yield METADATA_SENTINEL + final.model_dump_json()

    # ── Private helpers ──────────────────────────────────────────────────

    def _needs_web_fallback(
        self, pipeline_result: QueryPipelineResult, confidence: float, is_no_answer: bool
    ) -> bool:
        return self.settings.enable_web_fallback and (
            confidence < self.settings.confidence_threshold
            or bool(WEB_INTENT_PATTERN.search(pipeline_result.original_query))
            or is_no_answer
        )

    async def _generate_grounded_answer(
        self, query: str, compressed: list[CompressedContext], trace=None
    ) -> str:
        # Fetch the live system prompt (in-process cache; non-blocking).
        system_prompt = await get_answer_system_prompt()
        user_prompt = build_user_prompt(query, compressed)

        generation = None
        if trace:
            generation = trace.generation(
                name="grounded-answer",
                model=getattr(self.llm_client, "model_name", "llama-3.3-70b-versatile"),
                input=user_prompt,
            )
        try:
            result = (
                await self.llm_client.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=600,
                )
            ).strip()
            if generation:
                generation.end(output=result)
            return result
        except Exception:
            logger.exception("Answer generation LLM call failed")
            if generation:
                generation.end(output=NO_ANSWER_PHRASE, level="ERROR")
            return NO_ANSWER_PHRASE

    def _extract_citations(
        self, answer_text: str, compressed: list[CompressedContext]
    ) -> list[Citation]:
        cited_indices = {int(m) - 1 for m in CITATION_MARKER_RE.findall(answer_text)}
        citations = []
        for idx in sorted(cited_indices):
            if 0 <= idx < len(compressed):
                chunk = compressed[idx].chunk
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        filename=chunk.filename,
                    )
                )
        return citations

    async def _tavily_search(self, query: str, trace=None) -> list:
        span = trace.span(name="tavily-web-search", input={"query": query}) if trace else None
        results = await self.tavily_client.search(query)
        if span:
            span.end(output={"num_results": len(results)})
        return results

    async def _generate_web_answer(self, query: str, web_sources: list, trace=None) -> str:
        if not web_sources:
            return NO_ANSWER_PHRASE

        sources_block = "\n\n".join(
            f"[Web {i + 1}] {source.title}\n{source.snippet}\nURL: {source.url}"
            for i, source in enumerate(web_sources)
        )

        # Fetch the live web system prompt (in-process cache; non-blocking).
        web_system_prompt = await prompt_manager.get(
            self.settings.langfuse_prompt_web_system,
            fallback=WEB_SYSTEM_PROMPT,
        )

        generation = None
        if trace:
            generation = trace.generation(
                name="web-fallback-answer",
                model=getattr(self.llm_client, "model_name", "llama-3.3-70b-versatile"),
                input=sources_block,
            )
        try:
            result = (
                await self.llm_client.complete(
                    system_prompt=web_system_prompt,
                    user_prompt=(
                        f"Web search results:\n\n{sources_block}\n\n---\n\nQuestion: {query}"
                    ),
                    max_tokens=400,
                )
            ).strip()
            if generation:
                generation.end(output=result)
            return result
        except Exception:
            logger.exception("Web answer generation failed")
            fallback = (
                "I couldn't find this in your uploaded document. "
                "Here are relevant web results:\n\n"
                + "\n".join(f"- {s.title}: {s.url}" for s in web_sources[:3])
            )
            if generation:
                generation.end(output=fallback, level="ERROR")
            return fallback

    async def _no_answer_response(
        self, pipeline_result: QueryPipelineResult, confidence: float, trace=None
    ) -> ChatResponse:
        web_sources = []
        if self.settings.enable_web_fallback:
            web_sources = await self._tavily_search(pipeline_result.rewritten_query, trace=trace)

        answer_text = NO_ANSWER_PHRASE
        if web_sources:
            answer_text = await self._generate_web_answer(
                pipeline_result.rewritten_query, web_sources, trace=trace
            )

        return ChatResponse(
            answer=answer_text,
            confidence_score=confidence,
            citations=[],
            used_web_fallback=bool(web_sources),
            web_sources=web_sources,
            rewritten_query=pipeline_result.rewritten_query,
        )
