"""
Context compression.

Design choice: this runs AFTER reranking, on the already-small top-5 set --
its job isn't to filter for relevance (the reranker did that), it's to make
the 5 chunks cheaper and cleaner for the LLM to consume:

1. Dedupe near-identical chunks. This happens when two chunks came from
   different retrievers (FAISS vs BM25) but landed on overlapping text due
   to the 100-token chunk overlap from Phase 1 -- without dedup, the LLM
   prompt would contain the same sentence twice and could double-cite it.

2. Merge adjacent chunks from the same document. If chunk_index 4 and 5 of
   the same document both made the top-5, they're almost certainly part of
   one continuous passage -- merging them (and stripping the duplicated
   overlap region) gives the LLM one coherent passage instead of two
   fragments with a repeated seam, which measurably reduces citation
   confusion ("which of these two chunks is the number actually in?").

3. Drop chunks far below the top reranker score. The reranker returns
   exactly rerank_top_k chunks regardless of whether the 5th one is
   actually relevant -- if its score is much lower than the top result, it's
   likely noise that dilutes the prompt rather than helping it.
"""

from dataclasses import dataclass

from app.core.config import Settings
from app.services.rag.metadata.store import ChunkRow

# A chunk is dropped if its score falls below this fraction of the top
# chunk's score. 0.3 is deliberately lenient -- we only want to cut clear
# outliers, not second-guess the reranker's ordering of genuinely close calls.
RELATIVE_SCORE_FLOOR = 0.3

# Two chunks are treated as near-duplicates if this fraction of their
# (word-level) tokens overlap.
DUPLICATE_OVERLAP_THRESHOLD = 0.85


@dataclass
class CompressedContext:
    chunk: ChunkRow
    score: float
    merged_from: list[str]  # chunk_ids folded into this one, for debugging/audit


class ContextCompressor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def compress(
        self, scored_chunks: list[tuple[ChunkRow, float]]
    ) -> list[CompressedContext]:
        if not scored_chunks:
            return []

        filtered = self._drop_low_relevance(scored_chunks)
        deduped = self._dedupe(filtered)
        merged = self._merge_adjacent(deduped)
        return merged

    # -- internals ---------------------------------------------------

    def _drop_low_relevance(
        self, scored_chunks: list[tuple[ChunkRow, float]]
    ) -> list[tuple[ChunkRow, float]]:
        top_score = scored_chunks[0][1]
        if top_score <= 0:
            # Reranker scores can be negative logits for an entirely
            # irrelevant candidate set; the ratio floor doesn't make sense
            # against a non-positive top score, so skip filtering rather
            # than accidentally dropping everything.
            return scored_chunks
        return [
            (chunk, score)
            for chunk, score in scored_chunks
            if score >= top_score * RELATIVE_SCORE_FLOOR
        ]

    def _dedupe(
        self, scored_chunks: list[tuple[ChunkRow, float]]
    ) -> list[tuple[ChunkRow, float]]:
        kept: list[tuple[ChunkRow, float]] = []
        kept_token_sets: list[set[str]] = []

        for chunk, score in scored_chunks:
            tokens = set(chunk.text.lower().split())
            is_duplicate = False
            for existing_tokens in kept_token_sets:
                if not tokens or not existing_tokens:
                    continue
                overlap = len(tokens & existing_tokens) / len(tokens | existing_tokens)
                if overlap >= DUPLICATE_OVERLAP_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append((chunk, score))
                kept_token_sets.append(tokens)

        return kept

    def _merge_adjacent(
        self, scored_chunks: list[tuple[ChunkRow, float]]
    ) -> list[CompressedContext]:
        # Group by document, sort by chunk_index, then merge runs where
        # consecutive chunks differ by exactly 1 in chunk_index.
        by_doc: dict[str, list[tuple[ChunkRow, float]]] = {}
        for chunk, score in scored_chunks:
            by_doc.setdefault(chunk.document_id, []).append((chunk, score))

        results: list[CompressedContext] = []
        for doc_chunks in by_doc.values():
            doc_chunks.sort(key=lambda cs: cs[0].chunk_index)

            run: list[tuple[ChunkRow, float]] = [doc_chunks[0]]
            for current in doc_chunks[1:]:
                prev_chunk = run[-1][0]
                if current[0].chunk_index == prev_chunk.chunk_index + 1:
                    run.append(current)
                else:
                    results.append(self._flush_run(run))
                    run = [current]
            results.append(self._flush_run(run))

        results.sort(key=lambda c: c.score, reverse=True)
        return results

    def _flush_run(self, run: list[tuple[ChunkRow, float]]) -> CompressedContext:
        if len(run) == 1:
            chunk, score = run[0]
            return CompressedContext(chunk=chunk, score=score, merged_from=[chunk.chunk_id])

        # Merge text in chunk_index order. We don't attempt to precisely
        # strip the exact overlapping token span here (that would require
        # re-tokenizing with tiktoken to match Phase 1's chunker) -- instead
        # we accept the modest duplication from the ~100-token overlap
        # region, since it's small relative to a merged 1200-1600 token
        # passage and costs far less than the citation-confusion problem
        # merging solves in the first place.
        merged_text = "\n\n".join(c.text for c, _ in run)
        best_chunk, best_score = max(run, key=lambda cs: cs[1])

        merged_chunk = ChunkRow(
            chunk_id=best_chunk.chunk_id,  # cite the highest-scoring member
            document_id=best_chunk.document_id,
            filename=best_chunk.filename,
            page_number=run[0][0].page_number,  # first page of the run
            section=best_chunk.section,
            text=merged_text,
            token_count=sum(c.token_count for c, _ in run),
            chunk_type=best_chunk.chunk_type,
            chunk_index=best_chunk.chunk_index,
        )
        avg_score = sum(s for _, s in run) / len(run)
        return CompressedContext(
            chunk=merged_chunk,
            score=avg_score,
            merged_from=[c.chunk_id for c, _ in run],
        )