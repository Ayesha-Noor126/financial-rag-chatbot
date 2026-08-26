"""
Unit tests for answer prompt builders (app/services/rag/generation/answer_prompt.py).

What we test
────────────
• NO_ANSWER_PHRASE is a non-empty string (immutable safety check).
• build_context_block() renders [Source N | Page P] headers correctly.
• build_context_block() includes the section when present.
• build_context_block() omits section when absent.
• build_context_block() separates multiple sources with the --- divider.
• build_user_prompt() includes the query under "Question:".
• build_user_prompt() includes "Sources:" and the context block.
• SYSTEM_PROMPT contains the NO_ANSWER_PHRASE (so detection logic matches).

No external I/O — these are pure string-building functions.
"""

import pytest

from app.services.rag.generation.answer_prompt import (
    NO_ANSWER_PHRASE,
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)
from app.services.rag.generation.context_compressor import CompressedContext
from app.services.rag.metadata.store import ChunkRow


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compressed(
    text: str,
    chunk_id: str = "c1",
    page_number: int = 3,
    section: str | None = None,
    score: float = 1.0,
    chunk_index: int = 0,
) -> CompressedContext:
    row = ChunkRow(
        chunk_id=chunk_id,
        document_id="doc-1",
        filename="financial.pdf",
        page_number=page_number,
        section=section,
        text=text,
        token_count=len(text.split()),
        chunk_index=chunk_index,
    )
    return CompressedContext(chunk=row, score=score, merged_from=[chunk_id])


# ── NO_ANSWER_PHRASE ──────────────────────────────────────────────────────────

class TestNoAnswerPhrase:
    def test_is_non_empty_string(self):
        assert isinstance(NO_ANSWER_PHRASE, str)
        assert len(NO_ANSWER_PHRASE) > 0

    def test_contained_in_system_prompt(self):
        """
        AnswerService detects no-answer responses with `NO_ANSWER_PHRASE in answer_text`.
        The system prompt must instruct the model to use this exact phrase, which
        means it must appear verbatim in SYSTEM_PROMPT.
        """
        assert NO_ANSWER_PHRASE in SYSTEM_PROMPT


# ── build_context_block() ────────────────────────────────────────────────────

class TestBuildContextBlock:
    def test_single_chunk_has_source_header(self):
        cc = _compressed("Revenue was Rs 832 million.", page_number=5)
        block = build_context_block([cc])
        assert "[Source 1 | Page 5]" in block

    def test_source_numbering_starts_at_1(self):
        chunks = [
            _compressed("First passage.", chunk_id="c1", page_number=1),
            _compressed("Second passage.", chunk_id="c2", page_number=2),
        ]
        block = build_context_block(chunks)
        assert "[Source 1 | Page 1]" in block
        assert "[Source 2 | Page 2]" in block

    def test_section_included_when_present(self):
        cc = _compressed("Balance sheet data.", section="Balance Sheet", page_number=7)
        block = build_context_block([cc])
        assert "Section: Balance Sheet" in block

    def test_section_omitted_when_none(self):
        cc = _compressed("Some text without section.", section=None, page_number=4)
        block = build_context_block([cc])
        assert "Section:" not in block

    def test_chunk_text_appears_in_block(self):
        text = "Profit after taxation was Rs 125,682 thousand."
        cc = _compressed(text, page_number=10)
        block = build_context_block([cc])
        assert text in block

    def test_multiple_chunks_separated_by_divider(self):
        chunks = [
            _compressed("First.", chunk_id="c1", page_number=1),
            _compressed("Second.", chunk_id="c2", page_number=2),
        ]
        block = build_context_block(chunks)
        assert "---" in block  # the separator between sources

    def test_empty_list_returns_empty_string(self):
        assert build_context_block([]) == ""


# ── build_user_prompt() ───────────────────────────────────────────────────────

class TestBuildUserPrompt:
    def test_query_appears_under_question_label(self):
        query = "What was the net profit for Q3 2017?"
        cc = _compressed("Profit was Rs 125,682 thousand.", page_number=10)
        prompt = build_user_prompt(query, [cc])
        assert f"Question: {query}" in prompt

    def test_sources_label_present(self):
        cc = _compressed("Some text.", page_number=1)
        prompt = build_user_prompt("What is revenue?", [cc])
        assert "Sources:" in prompt

    def test_context_block_embedded_in_prompt(self):
        text = "Revenue was Rs 832 million."
        cc = _compressed(text, page_number=3)
        prompt = build_user_prompt("What is revenue?", [cc])
        assert text in prompt

    def test_citation_instruction_present(self):
        """The user prompt should remind the model to use citation markers."""
        cc = _compressed("Some text.", page_number=1)
        prompt = build_user_prompt("Any question?", [cc])
        assert "citation" in prompt.lower()
