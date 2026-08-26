"""
Langfuse client singleton.

Responsibilities
────────────────
• init_langfuse()   — create the Langfuse() instance once at startup and store
                      it in the module-level _langfuse_client variable.
• get_langfuse()    — return that instance (or None if disabled/failed).
• flush_langfuse()  — flush pending trace events before process exit.

Prompt management is intentionally separated into app/core/prompt_manager.py
so this file stays focused on observability concerns (tracing, scoring).
init_langfuse() calls prompt_manager.init() after constructing the client so
the prompt layer always has a valid (or None) reference without needing its
own import of the Langfuse class.
"""

from langfuse import Langfuse
from loguru import logger

from app.core import prompt_manager
from app.core.config import Settings

_langfuse_client: Langfuse | None = None


def init_langfuse(settings: Settings) -> Langfuse | None:
    """
    Initialise the Langfuse observability client and the prompt manager.

    Called once from app/main.py on startup.  Returns the client instance so
    main.py can log a confirmation message, but callers that only need the
    client should use get_langfuse() instead.

    Both init paths (enabled / disabled) call prompt_manager.init() so the
    prompt manager always has a consistent state:
      • enabled  → prompt_manager gets a live Langfuse client and will fetch
                   prompts from the network.
      • disabled → prompt_manager gets None and will always use hardcoded
                   fallbacks, keeping the app fully functional offline.
    """
    global _langfuse_client

    pub_key = (settings.langfuse_public_key or "").strip()
    sec_key = (settings.langfuse_secret_key or "").strip()
    is_dummy = (
        not pub_key
        or not sec_key
        or "your_public_key_here" in pub_key
        or "your_secret_key_here" in sec_key
        or pub_key == "pk-lf-your_public_key_here"
    )

    if not settings.langfuse_enabled or is_dummy:
        if is_dummy and settings.langfuse_enabled:
            logger.info("Langfuse disabled — placeholder or missing API keys detected")
        else:
            logger.info("Langfuse disabled via settings")
        _langfuse_client = None
        prompt_manager.init(None, cache_ttl_seconds=settings.langfuse_prompt_cache_ttl_seconds)
        return None

    try:
        _langfuse_client = Langfuse(
            public_key=pub_key,
            secret_key=sec_key,
            host=settings.langfuse_host,
        )
        logger.info(
            "Langfuse client initialised successfully (host={})", settings.langfuse_host
        )
    except Exception:
        logger.exception("Langfuse initialisation failed")
        _langfuse_client = None

    # Always initialise the prompt manager, even if the client failed to
    # construct — it will use None and fall back to hardcoded strings.
    prompt_manager.init(
        _langfuse_client,
        cache_ttl_seconds=settings.langfuse_prompt_cache_ttl_seconds,
    )
    return _langfuse_client


def get_langfuse() -> Langfuse | None:
    """Return the module-level Langfuse client (None if disabled or failed)."""
    return _langfuse_client


def flush_langfuse() -> None:
    """Flush all pending trace events to Langfuse before process exit."""
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
            logger.info("Langfuse flushed successfully")
        except Exception:
            logger.exception("Failed to flush Langfuse")
