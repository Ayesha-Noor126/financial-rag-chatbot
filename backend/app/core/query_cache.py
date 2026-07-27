"""
Query-level response cache.

Design choice: caches the FULL ChatResponse (not just embeddings, which
Phase 1's Embedder already caches at a finer grain) keyed on normalized
question text. This is the "avoid unnecessary LLM calls" latency win from
the spec -- a repeated identical question skips rewriting, retrieval,
reranking, AND generation entirely, not just the embedding step.

We deliberately only cache when the answer was document-grounded and
confident (see ChatService) -- a low-confidence or web-fallback answer
might be a fast-changing fact (news, stock price) that shouldn't be served
stale from cache later.

TTLCache from cachetools is not thread-safe by construction, and our async
routes may hit this from multiple asyncio.to_thread worker threads
concurrently, so a lock guards every access.
"""

from threading import Lock

from cachetools import TTLCache

from app.core.config import Settings
from app.models.schemas import ChatResponse


class QueryCache:
    def __init__(self, settings: Settings):
        self._cache: TTLCache = TTLCache(maxsize=256, ttl=settings.query_cache_ttl_seconds)
        self._lock = Lock()

    def _key(self, query: str) -> str:
        return query.strip().lower()

    def get(self, query: str) -> ChatResponse | None:
        with self._lock:
            return self._cache.get(self._key(query))

    def set(self, query: str, response: ChatResponse) -> None:
        with self._lock:
            self._cache[self._key(query)] = response