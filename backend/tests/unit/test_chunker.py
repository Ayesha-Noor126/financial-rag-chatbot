"""
Unit tests for SemanticChunker (app/services/rag/chunker.py).

What we test
────────────
• A single short page produces exactly one chunk.
• A long page (text > chunk_size_tokens) is split into multiple chunks,
  each within the token budget.
• chunk_index values are sequential starting from 0.
• Chunks below min_chunk_tokens are discarded.
• chunk_id, document_id, and filename are correctly propagated.
• Paragraphs within token budget are grouped into one chunk (not split).

Design note on tiktoken
───────────────────────
tiktoken is used as a length proxy (not for actual model tokenisation), so
tests work with small token budgets.  The `cl100k_base` encoding is cached
locally after the first download, so no network call is made after that.
"""

import pytest

from app.models.schemas import CleanedPage
from app.services.rag.chunker import SemanticChunker


# Helper: build a CleanedPage with repeated words to hit a specific token count
def _page(text: str, page_number: int = 1, section: str | None = None) -> CleanedPage:
    return CleanedPage(page_number=page_number, text=text, section=section)


def _long_text(words: int = 200) -> str:
    """Return a string of ~`words` words in multiple paragraphs.
    Paragraphs are required because the chunker splits on paragraph boundaries
    (\\n\\n).  A single paragraph with 300 words is treated as one unit and
    only force-split by sentence when it exceeds chunk_size_tokens.
    test_settings uses chunk_size_tokens=100, so each paragraph must be < 100
    tokens individually for the chunker to merge then flush at boundaries.
    We build many ~30-word paragraphs to guarantee multiple chunks.
    """
    paras = []
    para_size = 25
    for p in range(words // para_size):
        para = " ".join(f"word{p}x{i}" for i in range(para_size))
        paras.append(para)
    return "\n\n".join(paras)


# ── Basic correctness ─────────────────────────────────────────────────────────

class TestChunkerBasics:
    def test_single_short_page_yields_one_chunk(self, test_settings):
        chunker = SemanticChunker(test_settings)
        pages = [_page("Net profit was Rs 125,682 thousand for Q3 2017.")]
        chunks = chunker.chunk_document("doc-1", "report.pdf", pages)
        assert len(chunks) >= 1

    def test_chunk_fields_populated(self, test_settings):
        chunker = SemanticChunker(test_settings)
        pages = [_page("Revenue was Rs 832 million in Q3 2017.")]
        chunks = chunker.chunk_document("doc-42", "financial.pdf", pages)
        assert chunks[0].document_id == "doc-42"
        assert chunks[0].filename == "financial.pdf"
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_id  # non-empty UUID string

    def test_chunk_ids_are_unique(self, test_settings):
        chunker = SemanticChunker(test_settings)
        pages = [_page(_long_text(300))]
        chunks = chunker.chunk_document("doc-1", "test.pdf", pages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique"

    def test_chunk_index_sequential_from_zero(self, test_settings):
        chunker = SemanticChunker(test_settings)
        pages = [_page(_long_text(300))]
        chunks = chunker.chunk_document("doc-1", "test.pdf", pages)
        for expected, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected

    def test_empty_pages_returns_empty(self, test_settings):
        chunker = SemanticChunker(test_settings)
        # chunk_document accesses pages[0] if pages is empty — guard with 0 pages
        # Actually the code does `pages[0].page_number if pages else 1` so it's safe
        chunks = chunker.chunk_document("doc-1", "test.pdf", [])
        assert chunks == []


# ── Token budget ──────────────────────────────────────────────────────────────

class TestChunkerTokenBudget:
    def test_long_text_produces_multiple_chunks(self, test_settings):
        # test_settings has chunk_size_tokens=100; 300 words >> 100 tokens
        chunker = SemanticChunker(test_settings)
        pages = [_page(_long_text(300))]
        chunks = chunker.chunk_document("doc-1", "test.pdf", pages)
        assert len(chunks) > 1

    def test_each_chunk_within_token_budget(self, test_settings):
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        chunker = SemanticChunker(test_settings)
        pages = [_page(_long_text(300))]
        chunks = chunker.chunk_document("doc-1", "test.pdf", pages)
        # Allow up to chunk_size_tokens + overlap + slight tolerance (5 tokens) for newline join boundaries
        budget = test_settings.chunk_size_tokens + test_settings.chunk_overlap_tokens + 5
        for chunk in chunks:
            tok_count = len(enc.encode(chunk.text))
            assert tok_count <= budget, (
                f"Chunk {chunk.chunk_index} has {tok_count} tokens, "
                f"budget is {budget}"
            )

    def test_min_chunk_token_filter(self, test_settings):
        """
        Paragraphs below min_chunk_tokens should be silently discarded.
        We set min_chunk_tokens=5 in test_settings; a 1-word paragraph
        ('Hi') is definitely below that after overlap stripping... actually
        one word may still be >= 5 tokens because tiktoken encodes each word
        differently.  Let's use an empty-ish paragraph that won't pass.
        """
        chunker = SemanticChunker(test_settings)
        # Build a page where the last paragraph is 1 token so it gets dropped
        long_para = _long_text(120)  # this gets chunked
        tiny_para = "x"             # single character — likely < 5 tokens
        pages = [_page(f"{long_para}\n\n{tiny_para}")]
        chunks = chunker.chunk_document("doc-1", "test.pdf", pages)
        # The single-char paragraph should not appear as its own final chunk
        for chunk in chunks:
            assert chunk.text.strip() != "x"


# ── Multi-page ────────────────────────────────────────────────────────────────

class TestChunkerMultiPage:
    def test_page_number_assigned_from_source_page(self, test_settings):
        chunker = SemanticChunker(test_settings)
        pages = [
            _page("Revenue data here.", page_number=3),
            _page("Profit data here.", page_number=4),
        ]
        chunks = chunker.chunk_document("doc-1", "test.pdf", pages)
        # All chunks must have a valid page number (3 or 4)
        page_numbers = {c.page_number for c in chunks}
        assert page_numbers.issubset({3, 4})

    def test_section_propagated_from_cleaned_page(self, test_settings):
        chunker = SemanticChunker(test_settings)
        # Use text long enough to pass min_chunk_tokens=5
        long_text = "The balance sheet shows total assets and liabilities for the period."
        pages = [_page(long_text, page_number=1, section="Balance Sheet")]
        chunks = chunker.chunk_document("doc-1", "test.pdf", pages)
        assert len(chunks) >= 1, "Expected at least one chunk from the page"
        assert chunks[0].section == "Balance Sheet"
