"""
Confidence scoring.

Design choice: confidence is derived from two independent, cheap signals
rather than asking the LLM to self-report a confidence number (LLMs are
notoriously badly calibrated at self-reported confidence -- they'll say
"90% confident" on a hallucinated figure just as readily as on a grounded
one). Instead:

1. Reranker signal: the cross-encoder's top relevance score, passed through
   a sigmoid to map its unbounded logit range into [0, 1]. This reflects
   "how well does the best available chunk actually match the question" --
   independent of what the LLM later does with that chunk.

2. Entity coverage: what fraction of the entities extracted from the query
   (company, metric, year, region) actually appear (as substrings) in the
   compressed context. A question asking about "2019 net income" where none
   of the retrieved chunks mention "2019" is a strong signal the answer, if
   generated anyway, may be pulled from the wrong period.

Final confidence is a weighted blend, weighted toward the reranker signal
since it's the more reliable of the two (entity coverage is a coarse
heuristic -- e.g. it can't detect that a chunk mentions the wrong year's
"2019" in a comparison table).
"""

import math
import re

from app.core.config import Settings
from app.services.rag.generation.context_compressor import CompressedContext
from app.services.rag.ner.ner_service import ExtractedEntities

RERANKER_WEIGHT = 0.7
ENTITY_COVERAGE_WEIGHT = 0.3


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class ConfidenceScorer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def score(
        self,
        entities: ExtractedEntities,
        compressed_chunks: list[CompressedContext],
    ) -> float:
        if not compressed_chunks:
            return 0.0

        reranker_signal = _sigmoid(compressed_chunks[0].score)
        entity_signal = self._entity_coverage(entities, compressed_chunks)

        confidence = (
            RERANKER_WEIGHT * reranker_signal + ENTITY_COVERAGE_WEIGHT * entity_signal
        )
        return round(min(max(confidence, 0.0), 1.0), 4)

    def _entity_coverage(
        self, entities: ExtractedEntities, compressed_chunks: list[CompressedContext]
    ) -> float:
        all_terms = (
            entities.companies + entities.metrics + entities.years + entities.regions
        )
        if not all_terms:
            # No extractable entities in the query (e.g. a vague question) --
            # this signal simply doesn't apply, so don't let it drag
            # confidence down. Neutral 1.0 lets the reranker signal dominate.
            return 1.0

        combined_text = " ".join(cc.chunk.text.lower() for cc in compressed_chunks)
        matched = 0
        for term in all_terms:
            term_lower = term.lower().strip()
            if not term_lower:
                continue
            if term_lower in combined_text:
                matched += 1
            else:
                # Check significant sub-words (length > 3) for multi-word or compound terms
                words = [w for w in re.findall(r"\b[a-z]{4,}\b", term_lower)]
                if words and any(w in combined_text for w in words):
                    matched += 1

        return matched / len(all_terms)