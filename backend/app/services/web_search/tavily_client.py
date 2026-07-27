"""
Tavily web search client -- async, connection-pooled (Phase 4 update).

Design choice: switched from `requests` to a shared `httpx.AsyncClient`
instance, created once at app startup and injected here rather than built
per-call. This is what satisfies the spec's "Connection pooling" latency
requirement for this integration specifically -- Tavily is called inline,
synchronously blocking a /chat turn, so paying for a fresh TCP+TLS
handshake on every call (as a new requests.post() would) adds real,
avoidable latency. httpx.AsyncClient keeps a pool of warm connections and
reuses them across calls.

Still called ONLY from AnswerService's fallback branch, never from the main
retrieval path -- Tavily is last-resort, triggered by low confidence or
explicit web-intent keywords.
"""

import httpx
from loguru import logger

from app.core.config import Settings
from app.models.schemas import WebSource

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class TavilyClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient):
        self.settings = settings
        self.http_client = http_client

    async def search(self, query: str) -> list[WebSource]:
        if not self.settings.tavily_api_key:
            logger.warning("Tavily search requested but TAVILY_API_KEY is not set")
            return []

        try:
            response = await self.http_client.post(
                TAVILY_ENDPOINT,
                json={
                    "api_key": self.settings.tavily_api_key,
                    "query": query,
                    "search_depth": self.settings.tavily_search_depth,
                    "max_results": self.settings.tavily_max_results,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError:
            logger.exception("Tavily search failed")
            return []

        return [
            WebSource(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:500],
            )
            for item in data.get("results", [])
        ]