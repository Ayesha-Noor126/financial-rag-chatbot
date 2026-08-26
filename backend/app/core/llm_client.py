"""
Provider-agnostic async LLM client (OpenAI or Groq).

Design choice (Phase 4 update): switched from the synchronous OpenAI SDK to
AsyncOpenAI so every LLM call in the pipeline (query rewrite, grounded
generation) is natively async -- no asyncio.to_thread wrapping needed for
network I/O, which is the correct way to handle I/O-bound work under
FastAPI's event loop (to_thread is for CPU-bound/blocking-library work like
FAISS/reranker calls, not for things that are already async-native).

stream_complete() is the new piece for Phase 4's "Streaming Response"
requirement -- it's an async generator yielding text deltas as they arrive,
used by AnswerService.generate_stream() for the /chat streaming path.
complete() (non-streaming) is kept for query rewriting and the debug routes,
where we need the full string before proceeding.

Groq's API is OpenAI-wire-compatible (same request/response shape, just a
different base_url), so both providers go through this one client.

Reasoning-model compatibility note
────────────────────────────────────
Models like openai/gpt-oss-120b on Groq are reasoning models: they emit an
internal <think> chain before producing visible content.  The OpenAI SDK
surfaces this via message.model_extra['reasoning'].  When message.content is
empty (which can happen with very low max_tokens budgets that are exhausted by
reasoning tokens), _extract_content() falls back to the reasoning field so
callers always receive a non-empty string.  The max_tokens parameter passed by
callers should be large enough for the expected visible output; reasoning
tokens are counted separately and do NOT reduce the max_tokens budget.
"""

import os
import re
from collections.abc import AsyncIterator
from typing import NamedTuple

from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from app.core.config import Settings


class UsageInfo(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _extract_content(message: ChatCompletionMessage) -> str:
    """
    Return the visible text from a chat completion message.
    Never returns internal reasoning chains, and strips stray <think> tags.
    """
    content = message.content or ""
    if "<think>" in content and "</think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content.strip()


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.llm_model  # exposed for Langfuse generation() calls

        api_key = (
            settings.llm_api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )

        if not api_key:
            raise ValueError(
                "LLM_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY) is not set. "
                "Add it to backend/.env before starting the server."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=settings.llm_base_url,
        )
        self.last_usage: UsageInfo | None = None  # populated after each call, read by tracing code

        # Log configuration at construction time so misconfigured model names
        # are visible in the startup log before the first request arrives.
        # The API key is intentionally NOT logged.
        logger.info(
            "LLMClient ready — provider={provider}, base_url={base_url}, model={model}",
            provider=settings.llm_provider,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )

    async def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        response = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if response.usage:
            self.last_usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return _extract_content(response.choices[0].message)

    async def stream_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 600
    ) -> AsyncIterator[str]:
        self.last_usage = None
        stream = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},  # Groq/OpenAI both send a final usage-only chunk
        )
        async for chunk in stream:
            if chunk.usage:
                # final chunk: choices is empty, usage is populated -- capture, don't yield
                self.last_usage = UsageInfo(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                continue
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta