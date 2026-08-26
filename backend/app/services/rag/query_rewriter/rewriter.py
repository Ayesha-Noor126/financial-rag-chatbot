"""
Query rewriting — async, conversation-aware.

Langfuse Prompt Versioning change
──────────────────────────────────
The system prompt is no longer read directly from the SYSTEM_PROMPT constant
inside rewrite().  Instead, rewrite() calls prompt_manager.get() which:

  1. Returns the live "fin-rag-rewriter-system" prompt from Langfuse (cached
     in-process, TTL = settings.langfuse_prompt_cache_ttl_seconds).
  2. Falls back to the local SYSTEM_PROMPT constant if Langfuse is unreachable
     or the prompt has not been seeded yet.

The SYSTEM_PROMPT constant is preserved here so:
  • seed_prompts.py can import it to register the first version.
  • The fallback path works without any network dependency.

Everything else (async design, history truncation to 6 turns, exception
handling, return-original-on-failure) is unchanged.
"""

from loguru import logger

from app.core import prompt_manager
from app.core.config import get_settings
from app.core.llm_client import LLMClient

# ── Hardcoded fallback ─────────────────────────────────────────────────────
# Also seeded into Langfuse as the first version by seed_prompts.py.
# Langfuse name: fin-rag-rewriter-system
# No {{variables}} — this is a static instruction block.

SYSTEM_PROMPT = """You rewrite short, ambiguous, or conversational financial-document \
questions into fully explicit, standalone questions, without changing their meaning.

Rules:
- Preserve the original intent exactly. Do not add facts, years, or company \
names that weren't stated or clearly implied by the conversation.
- Normalize company names and financial metrics into clear, proper financial terms \
(e.g., capitalize company names like "netsol technologies" -> "Netsol Technologies", \
and standard financial terms like "earning per share" -> "earnings per share").
- If recent conversation is provided, use it ONLY to resolve pronouns, \
ellipsis, or follow-up references ("what about Europe?", "and last year?"). \
Do not pull in unrelated details from earlier turns.
- If the question is already clear and specific, return it unchanged.
- CRITICAL: If the question contains a specific number for a calculation \
(e.g. "earnings of 50 shares", "profit for 200 units", "revenue for 30%"), \
you MUST preserve that exact number in the rewritten question. Never drop or \
change numerical quantities the user specified.
- Output ONLY the rewritten question. No preamble, no quotes, no explanation.

Example:
Input: what is the earning per share of netsol technologies in 2017
Output: What was the earnings per share of Netsol Technologies in 2017 or for any reported period in 2017?

Example:
Input: How much profit?
Output: What was the company's net profit after tax?

Example (calculation with number):
Input: what is the earning of 50 shares?
Output: What is the total earnings for 50 shares, given the earnings per share stated in the document?

Example (with conversation context):
Recent conversation:
user: What was the revenue in Asia in 2023?
assistant: Asia revenue in 2023 was CHF 12.1 billion [Source 1].
Latest question: What about Europe?
Output: What was the company's revenue in Europe in 2023?
"""


class QueryRewriter:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def rewrite(self, query: str, history: list[tuple[str, str]] | None = None) -> str:
        """
        Rewrite `query` into a fully explicit standalone question.

        Fetches the live system prompt from Langfuse (in-process cache,
        non-blocking) and falls back to SYSTEM_PROMPT if unavailable.

        On any LLM failure the original query is returned unchanged so a
        failed rewrite degrades retrieval quality slightly rather than
        breaking the whole pipeline.
        """
        try:
            # Build the user turn first so we only do it once.
            user_prompt = query
            if history:
                history_block = "\n".join(
                    f"{role}: {content}" for role, content in history[-6:]
                )
                user_prompt = (
                    f"Recent conversation:\n{history_block}\n\nLatest question: {query}"
                )

            # Fetch system prompt — returns from in-process cache on the hot path.
            settings = get_settings()
            system_prompt = await prompt_manager.get(
                settings.langfuse_prompt_rewriter_system,
                fallback=SYSTEM_PROMPT,
            )

            rewritten = await self.llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=300,
            )
            cleaned = rewritten.strip().strip('"').strip("'").strip()
            # If the output contains multiple lines (e.g. intro text), pick the last non-empty line
            if "\n" in cleaned:
                lines = [line.strip().strip('"').strip("'") for line in cleaned.splitlines() if line.strip()]
                cleaned = lines[-1] if lines else query
            return cleaned if cleaned else query

        except Exception:
            logger.exception("Query rewrite failed, falling back to original query")
            return query
