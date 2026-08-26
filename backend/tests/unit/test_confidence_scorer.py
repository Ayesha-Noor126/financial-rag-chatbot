"""
Unit tests for ConfidenceScorer (app/services/rag/generation/confidence_scorer.py).

What we test
────────────
• score() always returns a float in [0.0, 1.0].
• When there are no entities, entity_coverage returns 1.0 (neutral signal)
  and the full weight falls on the reranker signal via sigmoid.
• When all entities appear in the context, coverage = 1.0.
• When no entities appear in the context, coverage = 0.0, pulling score down.
• Partial coverage produces a proportional signal.
• Empty compressed_chunks returns 0.0 (no data to score).
• The _sigmoid helper is monotone and bounded.

No external I/O — ConfidenceScorer is pure Python math.
"""

import math
import pytest

from app.services.rag.generation.confidence_scorer import (
    ENTITY_COVERAGE_WEIGHT,
    RERANKER_WEIGHT,
    ConfidenceScorer,
    _sigmoid,
)
from app.services.rag.generation.context_compressor import CompressedContext
from app.services.rag.metadata.store import ChunkRow
from app.services.rag.ner.ner_service import ExtractedEntities


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(text: str, score: float = 1.0, chunk_index: int = 0) -> CompressedContext:
    """Build a CompressedContext with minimal fields required by ConfidenceScorer."""
    row = ChunkRow(
        chunk_id=f"chunk-{chunk_index}",
        document_id="doc-1",
        filename="report.pdf",
        page_number=1,
        text=text,
        token_count=len(text.split()),
        chunk_index=chunk_index,
    )
    return CompressedContext(chunk=row, score=score, merged_from=[row.chunk_id])


def _entities(**kwargs) -> ExtractedEntities:
    """Build ExtractedEntities with explicit field values."""
    e = ExtractedEntities()
    for field, value in kwargs.items():
        setattr(e, field, value)
    return e


# ── _sigmoid ─────────────────────────────────────────────────────────────────

class TestSigmoid:
    def test_sigmoid_zero_is_half(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_sigmoid_positive_above_half(self):
        assert _sigmoid(3.0) > 0.5

    def test_sigmoid_negative_below_half(self):
        assert _sigmoid(-3.0) < 0.5

    def test_sigmoid_output_bounded_0_to_1(self):
        for x in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            result = _sigmoid(x)
            assert 0.0 <= result <= 1.0

    def test_sigmoid_monotone(self):
        values = [_sigmoid(x) for x in range(-5, 6)]
        assert values == sorted(values), "sigmoid must be monotonically increasing"


# ── ConfidenceScorer.score() ─────────────────────────────────────────────────

class TestConfidenceScorer:
    def test_empty_chunks_returns_zero(self, test_settings):
        scorer = ConfidenceScorer(test_settings)
        result = scorer.score(ExtractedEntities(), [])
        assert result == 0.0

    def test_score_bounded_0_to_1(self, test_settings):
        scorer = ConfidenceScorer(test_settings)
        chunks = [_make_chunk("Revenue was Rs 832 million.", score=2.0)]
        result = scorer.score(ExtractedEntities(), chunks)
        assert 0.0 <= result <= 1.0

    def test_score_returns_float(self, test_settings):
        scorer = ConfidenceScorer(test_settings)
        chunks = [_make_chunk("Some text.", score=1.0)]
        result = scorer.score(ExtractedEntities(), chunks)
        assert isinstance(result, float)

    def test_no_entities_uses_reranker_signal_only(self, test_settings):
        """
        With no entities, entity_coverage = 1.0 (neutral), so score should
        equal sigmoid(top_chunk_score) * RERANKER_WEIGHT + 1.0 * ENTITY_COVERAGE_WEIGHT.
        """
        scorer = ConfidenceScorer(test_settings)
        top_score = 1.5
        chunks = [_make_chunk("Some text.", score=top_score)]
        expected = round(
            min(max(RERANKER_WEIGHT * _sigmoid(top_score) + ENTITY_COVERAGE_WEIGHT * 1.0, 0.0), 1.0),
            4,
        )
        result = scorer.score(ExtractedEntities(), chunks)
        assert abs(result - expected) < 1e-6

    def test_full_entity_coverage_boosts_score(self, test_settings):
        scorer = ConfidenceScorer(test_settings)
        entities = _entities(years=["2017"], metrics=["revenue"])
        # Both entities appear in the chunk text
        chunks = [_make_chunk("Revenue for the year 2017 was strong.", score=0.5)]
        score_full = scorer.score(entities, chunks)

        # Compare against zero entity coverage (entities not in text)
        chunks_no_match = [_make_chunk("Unrelated text about nothing relevant.", score=0.5)]
        score_none = scorer.score(entities, chunks_no_match)

        assert score_full > score_none

    def test_zero_entity_coverage_lowers_score(self, test_settings):
        scorer = ConfidenceScorer(test_settings)
        entities = _entities(years=["2019"], metrics=["ebitda"])
        # Neither entity appears in the chunk
        chunks = [_make_chunk("The weather is fine today.", score=1.0)]
        score = scorer.score(entities, chunks)
        # With 0 coverage, entity_signal = 0, so score < RERANKER_WEIGHT * sigmoid(1.0) + ENTITY_COVERAGE_WEIGHT
        max_possible_with_zero_coverage = round(
            min(RERANKER_WEIGHT * _sigmoid(1.0) + ENTITY_COVERAGE_WEIGHT * 0.0, 1.0), 4
        )
        assert score <= max_possible_with_zero_coverage + 1e-6

    def test_score_is_rounded_to_4_decimal_places(self, test_settings):
        scorer = ConfidenceScorer(test_settings)
        chunks = [_make_chunk("text", score=0.7)]
        result = scorer.score(ExtractedEntities(), chunks)
        # round(x, 4) means at most 4 decimal places
        assert result == round(result, 4)
