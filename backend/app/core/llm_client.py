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
"""
"""
Provider-agnostic async LLM client (OpenAI or Groq).
...
"""

from collections.abc import AsyncIterator
from typing import NamedTuple

from openai import AsyncOpenAI

from app.core.config import Settings


class UsageInfo(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.llm_model  # exposed for Langfuse generation() calls
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.last_usage: UsageInfo | None = None  # populated after each call, read by tracing code

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
        return response.choices[0].message.content or ""

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