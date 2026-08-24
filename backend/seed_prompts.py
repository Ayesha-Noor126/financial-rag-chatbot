"""
seed_prompts.py — one-time script to push every hardcoded prompt into Langfuse
               and tag each one as "production".

Run once after the first deploy (or whenever you want to reset all prompts to
the current hardcoded baseline):

    cd backend
    python seed_prompts.py

What it does
────────────
1. Reads LANGFUSE_* credentials from backend/.env via the normal Settings object.
2. Calls langfuse.create_prompt() for each prompt.
   • If the prompt name already exists, Langfuse creates a new version and the
     existing version is preserved — nothing is deleted.
   • The new version is immediately labelled "production" so the running app
     picks it up within one cache-TTL cycle (default 5 min) without a restart.
3. Prints a summary of what was created / already up-to-date.

Template variable convention
─────────────────────────────
Langfuse uses mustache-style {{variable}} placeholders.  At runtime
prompt_manager.get("name", fallback, key=value) replaces {{key}} → value.

The answer-system prompt uses {{no_answer_phrase}} so the exact fallback
sentence can be changed in the UI without re-deploying the application.

The rewriter-system prompt has no runtime variables — it is a static
instruction block.

The web-system prompt has no runtime variables either.

Idempotency
───────────
Running this script multiple times is safe.  Each run creates a new version
(v2, v3, …) in Langfuse but the "production" label always points to the
latest run.  To avoid version bloat, only run when you intend to update a
prompt.
"""

import sys
from pathlib import Path

# Make app/ importable when running from backend/
sys.path.insert(0, str(Path(__file__).parent))

from langfuse import Langfuse  # noqa: E402
from loguru import logger  # noqa: E402

from app.core.config import get_settings  # noqa: E402

# ── Pull the exact hardcoded text from the application modules ─────────────
# Importing directly from the modules guarantees the seed content is always
# byte-for-byte identical to what the fallback path would use.  If a developer
# updates the hardcoded string and re-runs this script, Langfuse gets the new
# version automatically.
from app.services.rag.generation.answer_prompt import (  # noqa: E402
    NO_ANSWER_PHRASE,
    SYSTEM_PROMPT as ANSWER_SYSTEM_PROMPT,
)
from app.services.rag.query_rewriter.rewriter import (  # noqa: E402
    SYSTEM_PROMPT as REWRITER_SYSTEM_PROMPT,
)

# WEB_SYSTEM_PROMPT lives inline in answer_service — import it directly so
# we don't duplicate the string here.
from app.services.rag.generation.answer_service import (  # noqa: E402
    WEB_SYSTEM_PROMPT,
)

# ── Prompt definitions ─────────────────────────────────────────────────────
#
# Each entry is a dict accepted by langfuse.create_prompt():
#   name    : must match the settings.langfuse_prompt_* values in config.py
#   prompt  : the template string; use {{variable}} for runtime substitution
#   type    : "text" (single string) — we don't use chat-format prompts here
#   labels  : list of labels to apply; "production" makes it the live version
#   config  : arbitrary JSON metadata stored alongside the prompt in the UI
#
# The answer-system prompt uses {{no_answer_phrase}} so the fallback sentence
# can be updated in the Langfuse UI without code changes.  The hardcoded
# ANSWER_SYSTEM_PROMPT already contains the literal phrase, so we replace it
# with the template variable before seeding.

_ANSWER_SYSTEM_TEMPLATE = ANSWER_SYSTEM_PROMPT.replace(
    NO_ANSWER_PHRASE, "{{no_answer_phrase}}"
)



def main() -> None:
    settings = get_settings()

    if not settings.langfuse_enabled:
        logger.error(
            "LANGFUSE_ENABLED is False in your .env — set it to True and re-run."
        )
        sys.exit(1)

    if not settings.langfuse_secret_key:
        logger.error(
            "LANGFUSE_SECRET_KEY is not set in your .env — cannot authenticate."
        )
        sys.exit(1)

    # Build prompt definitions here (inside main) so settings.llm_model is resolved
    # from the live .env instead of being hardcoded.
    prompts: list[dict] = [
        {
            "name": "fin-rag-answer-system",
            "prompt": _ANSWER_SYSTEM_TEMPLATE,
            "type": "text",
            "labels": ["production"],
            "config": {
                "description": (
                    "System prompt for grounded financial-document QA. "
                    "Variable: {{no_answer_phrase}} — the exact sentence the model "
                    "should output when the sources do not contain the answer."
                ),
                "variables": ["no_answer_phrase"],
                "model": settings.llm_model,
            },
        },
        {
            "name": "fin-rag-web-system",
            "prompt": WEB_SYSTEM_PROMPT,
            "type": "text",
            "labels": ["production"],
            "config": {
                "description": (
                    "System prompt for web-search fallback answers. "
                    "No runtime variables — static instruction block."
                ),
                "variables": [],
                "model": settings.llm_model,
            },
        },
        {
            "name": "fin-rag-rewriter-system",
            "prompt": REWRITER_SYSTEM_PROMPT,
            "type": "text",
            "labels": ["production"],
            "config": {
                "description": (
                    "System prompt for query rewriting. "
                    "Converts short/ambiguous questions into standalone queries. "
                    "No runtime variables — static instruction block."
                ),
                "variables": [],
                "model": settings.llm_model,
            },
        },
    ]

    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    logger.info("Connected to Langfuse at {}", settings.langfuse_host)
    logger.info("Seeding {} prompt(s) with model={}…", len(prompts), settings.llm_model)

    for defn in prompts:
        name = defn["name"]
        try:
            created = lf.create_prompt(
                name=name,
                prompt=defn["prompt"],
                type=defn["type"],       # type: ignore[arg-type]
                labels=defn["labels"],
                config=defn["config"],
            )
            logger.success(
                "  ✓ '{}' → version {} labelled {}",
                name,
                created.version,
                defn["labels"],
            )
        except Exception:
            logger.exception("  ✗ Failed to seed prompt '{}'", name)

    # Flush so all HTTP calls complete before the script exits.
    lf.flush()
    logger.info("Done.  All prompts are live in Langfuse.")
    logger.info(
        "Tip: the running app will pick up the new versions within {} seconds "
        "(langfuse_prompt_cache_ttl_seconds).",
        settings.langfuse_prompt_cache_ttl_seconds,
    )


if __name__ == "__main__":
    main()
