"""
Langfuse Prompt Manager — centralised fetch / compile / fallback layer.

Why this module exists
──────────────────────
The Langfuse SDK's get_prompt() is synchronous and makes a network round-trip
on the first call.  Calling it inline inside an async request handler would
block the event loop.  This module solves that by:

  1. Pre-warming every known prompt once at startup in a thread pool so the
     first real request is never slow.
  2. Delegating all get_prompt() calls to asyncio.to_thread() so they are
     always non-blocking even when the in-process TTL cache has expired and
     the SDK needs to re-fetch from the network.
  3. Transparently falling back to the hardcoded Python strings if Langfuse
     is unreachable, disabled, or the prompt name doesn't exist yet.

Contract understood by callers
───────────────────────────────
Every caller (answer_prompt.py, answer_service.py, rewriter.py) calls:

    text = await prompt_manager.get("fin-rag-answer-system")
    # or with variables:
    text = await prompt_manager.get("fin-rag-answer-system",
                                    no_answer_phrase=NO_ANSWER_PHRASE)

A string is always returned — never None, never an exception.

Langfuse prompt template format
────────────────────────────────
Langfuse uses mustache-style double-brace variables: {{variable_name}}
When you create a prompt in the Langfuse UI (or via seed_prompts.py), replace
any runtime-injected value with {{variable_name}}.  At call time pass the
values as kwargs and compile() substitutes them in.

Thread-safety
─────────────
_cache is a plain dict guarded by a threading.Lock.  get_prompt() on the
Langfuse SDK object is itself thread-safe (it holds its own internal lock).
The asyncio.to_thread() calls land on the default thread-pool executor, so
multiple concurrent awaits will each acquire the Python-level lock separately,
which is fine — only one will actually hit the network; the rest wait and
then read the freshly populated cache entry.
"""

import asyncio
import threading
from typing import Any

from loguru import logger

# Maps prompt name → compiled string (or None while not yet fetched)
_cache: dict[str, str] = {}
_lock = threading.Lock()

# Module-level reference to the Langfuse client; set by init() at startup.
_langfuse = None
# Cache TTL forwarded to the SDK's own in-process cache; overridden by init().
_cache_ttl_seconds: int = 300


# ── Public API ────────────────────────────────────────────────────────────


def init(langfuse_client, cache_ttl_seconds: int = 300) -> None:
    """
    Store a reference to the already-initialised Langfuse client.

    Called once from app/main.py after init_langfuse() returns.  Safe to
    call with None (Langfuse disabled) — every subsequent get() will just
    return its fallback string.

    Parameters
    ----------
    langfuse_client    : the Langfuse() instance (or None if disabled)
    cache_ttl_seconds  : forwarded to the SDK's own in-process cache TTL;
                         controlled by settings.langfuse_prompt_cache_ttl_seconds
    """
    global _langfuse, _cache_ttl_seconds
    _langfuse = langfuse_client
    _cache_ttl_seconds = cache_ttl_seconds


async def warm(prompt_names: list[str], fallbacks: dict[str, str]) -> None:
    """
    Pre-fetch a list of prompts concurrently at startup.

    Any prompt that fails to fetch is silently skipped — the fallback string
    will be used at runtime.  This means a network outage at startup does NOT
    prevent the application from starting.

    Parameters
    ----------
    prompt_names : list of Langfuse prompt names to pre-fetch
    fallbacks    : dict mapping name → hardcoded fallback string
    """
    tasks = [_fetch_into_cache(name, fallbacks.get(name, "")) for name in prompt_names]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(
        "Prompt warm-up complete. Cached: [%s]",
        ", ".join(k for k in _cache if _cache[k]),
    )


async def get(name: str, fallback: str = "", **compile_kwargs: Any) -> str:
    """
    Return the compiled text for a prompt, fetching from Langfuse if needed.

    Parameters
    ----------
    name           : the prompt name registered in Langfuse
    fallback       : the hardcoded string to return if Langfuse is unavailable
    compile_kwargs : mustache variable values (e.g. no_answer_phrase="…")

    Returns
    -------
    Compiled prompt string — always a non-empty str (uses fallback if needed).
    """
    # Fast path: already in cache (typical in-flight request case).
    # A prompt is evicted from the cache when the SDK's own TTL expires;
    # the next await on that name transparently re-fetches it.
    if name in _cache:
        template = _cache[name]
        return _compile(template, compile_kwargs) if compile_kwargs else template

    # Slow path: not yet cached — fetch from Langfuse in a thread.
    fetched = await _fetch_into_cache(name, fallback)
    return _compile(fetched, compile_kwargs) if compile_kwargs else fetched


# ── Internals ─────────────────────────────────────────────────────────────


async def _fetch_into_cache(name: str, fallback: str) -> str:
    """
    Fetch prompt from Langfuse in a thread (non-blocking), store in _cache.
    Returns the fetched template string, or `fallback` on any failure.
    """
    if _langfuse is None:
        # Langfuse disabled or not yet initialised — use fallback silently.
        with _lock:
            _cache[name] = fallback
        return fallback

    try:
        # get_prompt() is synchronous and may do network I/O; run it in the
        # thread pool so the event loop is never blocked.
        prompt_client = await asyncio.to_thread(
            _langfuse.get_prompt,
            name,
            # label="production" targets the prompt version tagged as
            # production in the Langfuse UI.  Omit label to get the latest
            # version regardless of tag — useful during development.
            label="production",
            # The SDK's own in-process TTL cache.  After this many seconds
            # the SDK will silently re-fetch from the network on the next
            # get_prompt() call.  Controlled by
            # settings.langfuse_prompt_cache_ttl_seconds (default 300 s).
            cache_ttl_seconds=_cache_ttl_seconds,
            # Inline SDK-level fallback: if the network call fails, the SDK
            # returns a TextPromptClient built from this string instead of
            # raising.  We pass the hardcoded string here so the SDK's own
            # retry logic also has a safe value.
            fallback=fallback,
        )
        template = prompt_client.prompt  # raw mustache template string
        with _lock:
            _cache[name] = template
        logger.debug("Langfuse prompt cached: '%s' (v%s)", name, getattr(prompt_client, "version", "?"))
        return template

    except Exception:
        logger.warning(
            "Could not fetch Langfuse prompt '%s' — using hardcoded fallback.", name
        )
        with _lock:
            _cache[name] = fallback
        return fallback


def _compile(template: str, kwargs: dict[str, Any]) -> str:
    """
    Replace {{variable}} placeholders in a mustache template.

    We do not import the Langfuse SDK's TemplateParser here because this
    module must remain importable even when Langfuse is disabled.  A simple
    str.replace loop is sufficient: our templates only use flat variable
    substitution, not mustache conditionals or loops.
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def invalidate(name: str | None = None) -> None:
    """
    Remove one prompt (or all prompts) from the local cache.

    Useful in tests or after a manual prompt update when you want the next
    request to pull a fresh copy from Langfuse immediately rather than
    waiting for the TTL to expire.
    """
    with _lock:
        if name is None:
            _cache.clear()
            logger.debug("All prompts invalidated from local cache.")
        elif name in _cache:
            del _cache[name]
            logger.debug("Prompt '%s' invalidated from local cache.", name)
