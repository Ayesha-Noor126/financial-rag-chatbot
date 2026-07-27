"""
Grounded prompting — system prompt and user prompt builder.

Design choice: the system prompt encodes every hallucination-control rule
from the spec explicitly and repeatedly (never invent numbers, answer only
from context, exact fallback phrase, mandatory citations) — financial
figures are exactly the kind of content where an LLM will confidently
interpolate a plausible-sounding number if the prompt leaves any room for
it.  The fallback phrase is specified VERBATIM and the model is told to use
it verbatim, so AnswerService can detect a no-answer response by simple
string match without a second LLM call.

Langfuse Prompt Versioning
──────────────────────────
get_answer_system_prompt() fetches the live "fin-rag-answer-system" prompt
from Langfuse at runtime (cached in process; TTL controlled by
settings.langfuse_prompt_cache_ttl_seconds).

The Langfuse template stores {{no_answer_phrase}} as a mustache variable so
the exact fallback sentence can be updated in the Langfuse UI without a
redeploy.  At fetch time prompt_manager.get() substitutes the current value
of NO_ANSWER_PHRASE.

Fallback path
─────────────
If Langfuse is disabled, unreachable, or the prompt has not been seeded yet
(run seed_prompts.py once to register it), get_answer_system_prompt() returns
SYSTEM_PROMPT unchanged — the application continues working exactly as before.

Nothing else in the codebase changes: AnswerService still calls
get_answer_system_prompt() in the same places it previously used the module-
level SYSTEM_PROMPT constant.
"""

from app.core import prompt_manager
from app.core.config import get_settings
from app.services.rag.generation.context_compressor import CompressedContext

# ── Immutable constants ───────────────────────────────────────────────────

# NO_ANSWER_PHRASE is intentionally NOT fetched from Langfuse.  AnswerService
# detects a no-answer response by searching for this exact string in the LLM
# output.  If the phrase changed in Langfuse but the detection code was not
# updated in the same deployment, silent regressions would result.  Keeping
# it here as a Python constant ensures the prompt template and the detection
# code always agree.
NO_ANSWER_PHRASE = "I couldn't find this information in the uploaded document."

# ── Hardcoded fallback ────────────────────────────────────────────────────
# Used verbatim when Langfuse is unavailable.
# Also seeded into Langfuse by seed_prompts.py as the first prompt version.
# The {{no_answer_phrase}} token in the Langfuse template is replaced with
# the value of NO_ANSWER_PHRASE at fetch time by prompt_manager.get().

SYSTEM_PROMPT = f"""You are a financial document assistant. You answer questions \
using ONLY the numbered source excerpts provided below.

STRICT RULES:
1. Answer ONLY using information present in the provided sources. Never use \
outside knowledge, even if you are confident it is correct.
2. NEVER invent, estimate, or infer a number that is not explicitly stated \
in the sources.
3. If the sources do not contain the answer, respond with EXACTLY this \
sentence and nothing else: "{NO_ANSWER_PHRASE}"
4. Every factual claim in your answer must end with a citation marker like \
[Source 1] or [Source 2] referencing which numbered source it came from. \
If a claim draws on multiple sources, cite all of them: [Source 1][Source 3].
5. Be concise and direct. Do not pad the answer with generic commentary \
about the company or industry.
6. Do not mention these instructions or that you were given "sources" -- \
just answer naturally with citations.
"""


async def get_answer_system_prompt() -> str:
    """
    Return the answer system prompt, preferring the live Langfuse version.

    Call signature is async because prompt_manager.get() may need to fetch
    from the network on the first call or after a cache expiry.  Subsequent
    calls within the TTL window return instantly from the in-process cache.

    The {{no_answer_phrase}} variable in the Langfuse template is compiled
    here with the module-level NO_ANSWER_PHRASE constant so the detection
    logic in AnswerService always matches what the LLM was instructed to say.
    """
    settings = get_settings()
    return await prompt_manager.get(
        settings.langfuse_prompt_answer_system,
        fallback=SYSTEM_PROMPT,
        # Compile the mustache variable so the LLM sees the exact phrase that
        # AnswerService uses for `NO_ANSWER_PHRASE in answer_text` detection.
        no_answer_phrase=NO_ANSWER_PHRASE,
    )


# ── User prompt builder ───────────────────────────────────────────────────
# These functions are pure Python — no Langfuse dependency — because the user
# prompt is dynamically assembled from retrieved chunks at query time.
# Putting dynamic document content into a Langfuse template would add latency
# (one round-trip per request) and provide no management benefit over building
# the string locally.

def build_context_block(compressed_chunks: list[CompressedContext]) -> str:
    """Render numbered source blocks for the user prompt."""
    blocks = []
    for i, cc in enumerate(compressed_chunks, start=1):
        chunk = cc.chunk
        header = f"[Source {i}] (Page {chunk.page_number}"
        if chunk.section:
            header += f", Section: {chunk.section}"
        header += ")"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(query: str, compressed_chunks: list[CompressedContext]) -> str:
    """Assemble the complete user-turn message sent to the LLM."""
    context_block = build_context_block(compressed_chunks)
    return (
        f"Sources:\n\n{context_block}\n\n---\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the sources above, "
        f"with citation markers as instructed."
    )
