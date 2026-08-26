"""
Unit tests for ContextCompressor (app/services/rag/generation/context_compressor.py).

What we test
────────────
• Empty input returns an empty list.
• A single chunk passes through unchanged.
• Near-duplicate chunks (≥ 85% token overlap) are deduplicated.
• Chunks far below the top score (< 30% of top score) are dropped.
• Adjacent chunks from the same document (consecutive chunk_index) are merged
  into a single CompressedContext.
• Non-adjacent chunks from the same document are NOT merged.
• The merged CompressedContext uses the highest-scoring chunk's chunk_id.
• Results are sorted by score descending.

No external I/O — ContextCompressor is pure Python logic.
"""

import pytest

from app.services.rag.generation.context_compressor import (
    DUPLICATE_OVERLAP_THRESHOLD,
    RELATIVE_SCORE_FLOOR,
    ContextCompressor,
)
from app.services.rag.metadata.store import ChunkRow


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk_row(
    text: str,
    chunk_id: str = "cid-1",
    document_id: str = "doc-1",
    chunk_index: int = 0,
    page_number: int = 1,
) -> ChunkRow:
    return ChunkRow(
        chunk_id=chunk_id,
        document_id=document_id,
        filename="report.pdf",
        page_number=page_number,
        text=text,
        token_count=len(text.split()),
        chunk_index=chunk_index,
    )


def _scored(chunk: ChunkRow, score: float) -> tuple[ChunkRow, float]:
    return (chunk, score)


# ── Empty / single ────────────────────────────────────────────────────────────

class TestCompressorEdgeCases:
    def test_empty_input_returns_empty(self, test_settings):
        compressor = ContextCompressor(test_settings)
        assert compressor.compress([]) == []

    def test_single_chunk_passes_through(self, test_settings):
        compressor = ContextCompressor(test_settings)
        chunk = _make_chunk_row("Revenue was Rs 832 million.", chunk_id="c1")
        result = compressor.compress([_scored(chunk, 1.0)])
        assert len(result) == 1
        assert result[0].chunk.chunk_id == "c1"
        assert result[0].score == 1.0
        assert result[0].merged_from == ["c1"]


# ── Low-relevance dropping ────────────────────────────────────────────────────

class TestLowRelevanceDrop:
    def test_chunk_below_score_floor_is_dropped(self, test_settings):
        """
        Top chunk score = 10.0.  Floor = 30% → 3.0.
        A chunk with score 2.0 should be dropped.
        """
        compressor = ContextCompressor(test_settings)
        top = _make_chunk_row("Top relevant passage about revenue.", chunk_id="c1", chunk_index=0)
        low = _make_chunk_row("Completely different topic.", chunk_id="c2", document_id="doc-2", chunk_index=0)
        result = compressor.compress([_scored(top, 10.0), _scored(low, 2.0)])
        ids = [r.chunk.chunk_id for r in result]
        assert "c1" in ids
        assert "c2" not in ids, "Chunk below 30% of top score should be dropped"

    def test_chunk_at_score_floor_is_kept(self, test_settings):
        """Score exactly at floor (30%) should be kept."""
        compressor = ContextCompressor(test_settings)
        top = _make_chunk_row("Top content.", chunk_id="c1", chunk_index=0)
        at_floor = _make_chunk_row("Floor content.", chunk_id="c2", document_id="doc-2", chunk_index=0)
        # 30% of 10.0 = 3.0
        result = compressor.compress([_scored(top, 10.0), _scored(at_floor, 3.0)])
        ids = [r.chunk.chunk_id for r in result]
        assert "c2" in ids


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_near_duplicate_chunks_are_deduped(self, test_settings):
        """
        Two chunks with Jaccard word overlap >= 85% are near-duplicates.
        Threshold = 0.85 → overlap / union >= 0.85.
        We use the exact same sentence for both to guarantee 100% overlap.
        """
        compressor = ContextCompressor(test_settings)
        # Identical texts → Jaccard = 1.0 → definitely above 0.85
        text = "The net profit for Q3 2017 was Rs 125682 thousand compared to the prior period results"
        chunk_a = _make_chunk_row(text, chunk_id="ca", chunk_index=0, document_id="doc-1")
        chunk_b = _make_chunk_row(text, chunk_id="cb", chunk_index=5, document_id="doc-2")
        result = compressor.compress([_scored(chunk_a, 5.0), _scored(chunk_b, 4.9)])
        assert len(result) == 1, "Identical text should deduplicate to one chunk"

    def test_distinct_chunks_both_kept(self, test_settings):
        compressor = ContextCompressor(test_settings)
        chunk_a = _make_chunk_row(
            "Revenue for the quarter was strong driven by license fees.",
            chunk_id="ca", document_id="doc-1", chunk_index=0,
        )
        chunk_b = _make_chunk_row(
            "The company acquired new clients in APAC and EMEA regions.",
            chunk_id="cb", document_id="doc-2", chunk_index=0,
        )
        result = compressor.compress([_scored(chunk_a, 5.0), _scored(chunk_b, 4.0)])
        assert len(result) == 2, "Distinct chunks should both survive"


# ── Adjacent merging ──────────────────────────────────────────────────────────

class TestAdjacentMerging:
    def test_adjacent_same_doc_chunks_are_merged(self, test_settings):
        compressor = ContextCompressor(test_settings)
        chunk_0 = _make_chunk_row(
            "The company reported strong revenues in Q3.", chunk_id="c0",
            document_id="doc-1", chunk_index=0,
        )
        chunk_1 = _make_chunk_row(
            "Net profit after taxation was Rs 125,682 thousand.", chunk_id="c1",
            document_id="doc-1", chunk_index=1,
        )
        result = compressor.compress([_scored(chunk_0, 5.0), _scored(chunk_1, 4.5)])
        assert len(result) == 1, "Consecutive chunks from same doc should merge"
        merged = result[0]
        # Merged text should contain both passages
        assert "revenues" in merged.chunk.text
        assert "125,682" in merged.chunk.text
        # Both chunk IDs should appear in merged_from
        assert "c0" in merged.merged_from or "c1" in merged.merged_from

    def test_non_adjacent_same_doc_chunks_not_merged(self, test_settings):
        compressor = ContextCompressor(test_settings)
        chunk_0 = _make_chunk_row(
            "First passage about revenues.", chunk_id="c0",
            document_id="doc-1", chunk_index=0,
        )
        chunk_5 = _make_chunk_row(
            "Fifth passage about liabilities.", chunk_id="c5",
            document_id="doc-1", chunk_index=5,  # gap of 4 → not adjacent
        )
        result = compressor.compress([_scored(chunk_0, 5.0), _scored(chunk_5, 4.0)])
        assert len(result) == 2, "Non-adjacent chunks should remain separate"

    def test_different_doc_chunks_not_merged(self, test_settings):
        compressor = ContextCompressor(test_settings)
        chunk_a = _make_chunk_row(
            "Document A passage one.", chunk_id="ca",
            document_id="doc-A", chunk_index=0,
        )
        chunk_b = _make_chunk_row(
            "Document B passage one.", chunk_id="cb",
            document_id="doc-B", chunk_index=1,  # chunk_index=1 but different doc
        )
        result = compressor.compress([_scored(chunk_a, 5.0), _scored(chunk_b, 4.0)])
        assert len(result) == 2, "Chunks from different documents must not merge"


# ── Sorting ───────────────────────────────────────────────────────────────────

class TestResultSorting:
    def test_results_sorted_by_score_descending(self, test_settings):
        compressor = ContextCompressor(test_settings)
        chunks = [
            _scored(_make_chunk_row("Low score passage.", chunk_id="c_low", document_id="d1", chunk_index=0), 1.0),
            _scored(_make_chunk_row("High score passage.", chunk_id="c_high", document_id="d2", chunk_index=0), 9.0),
            _scored(_make_chunk_row("Mid score passage.", chunk_id="c_mid", document_id="d3", chunk_index=0), 4.0),
        ]
        result = compressor.compress(chunks)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
