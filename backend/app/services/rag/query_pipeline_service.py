"""
Query pipeline orchestrator -- async (Phase 4 update).
...
"""

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from app.services.rag.metadata.store import ChunkRow, MetadataStore
from app.services.rag.ner.ner_service import ExtractedEntities, NERService
from app.services.rag.query_rewriter.rewriter import QueryRewriter
from app.services.rag.reranker.reranker import Reranker
from app.services.rag.retriever.hybrid_retriever import HybridRetriever

ENTITY_MATCH_BOOST = 2.5


@dataclass
class QueryPipelineResult:
    original_query: str
    rewritten_query: str
    entities: ExtractedEntities
    reranked_chunks: list[tuple[ChunkRow, float]] = field(default_factory=list)


class QueryPipelineService:
    def __init__(
        self,
        rewriter: QueryRewriter,
        ner_service: NERService,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        metadata_store: MetadataStore,
    ):
        self.rewriter = rewriter
        self.ner_service = ner_service
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.metadata_store = metadata_store

    def _apply_entity_boost(
        self,
        scored: list[tuple[ChunkRow, float]],
        entities: ExtractedEntities,
    ) -> list[tuple[ChunkRow, float]]:
        terms = [
            t.lower()
            for t in (entities.metrics + entities.companies + entities.years + entities.regions)
        ]
        if not terms:
            return scored

        boosted: list[tuple[ChunkRow, float]] = []
        for chunk, score in scored:
            text = chunk.text.lower()
            matches = sum(1 for term in terms if term in text)
            if matches:
                score += ENTITY_MATCH_BOOST * matches
            boosted.append((chunk, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    async def run(
        self,
        raw_query: str,
        history: list[tuple[str, str]] | None = None,
        trace=None,
    ) -> QueryPipelineResult:
        query = raw_query.strip()

        logger.info(f"Rewriting query: {query!r}")
        span = trace.span(name="query-rewrite", input={"query": query}) if trace else None
        rewritten = await self.rewriter.rewrite(query, history=history)
        if span:
            span.end(output={"rewritten_query": rewritten})
        logger.info(f"Rewritten: {rewritten!r}")

        entities = self.ner_service.extract(rewritten)
        logger.info(f"Entities: {entities.to_dict()}")
        if trace:
            trace.span(name="ner-extraction", output=entities.to_dict()).end()

        entity_terms = (
            entities.metrics + entities.companies + entities.years + entities.regions
        )

        retrieval_span = (
            trace.span(name="hybrid-retrieval", input={"query": rewritten, "entity_terms": entity_terms})
            if trace else None
        )
        retrieved = await self.hybrid_retriever.retrieve(rewritten, entity_terms=entity_terms)
        chunk_ids = [r.chunk_id for r in retrieved]
        if retrieval_span:
            retrieval_span.end(output={"num_retrieved": len(retrieved)})

        chunk_rows = await asyncio.to_thread(self.metadata_store.get_chunks_by_ids, chunk_ids)

        order = {cid: i for i, cid in enumerate(chunk_ids)}
        chunk_rows.sort(key=lambda c: order.get(c.chunk_id, len(order)))

        # Pass the top RRF score to the reranker so it can apply its
        # conditional skip: when both FAISS and BM25 strongly agree on the
        # top chunk, the expensive cross-encoder forward passes are bypassed.
        top_rrf_score = retrieved[0].rrf_score if retrieved else None
        skipped = (
            top_rrf_score is not None
            and top_rrf_score >= self.reranker._skip_threshold
        )

        rerank_span = trace.span(
            name="rerank",
            input={
                "num_candidates": len(chunk_rows),
                "top_rrf_score": round(top_rrf_score, 5) if top_rrf_score else None,
                "skip_threshold": self.reranker._skip_threshold,
            },
        ) if trace else None

        reranked = await asyncio.to_thread(
            self.reranker.rerank,
            rewritten,
            chunk_rows,
            None,           # top_k — reranker uses settings.rerank_top_k
            top_rrf_score=top_rrf_score,
        )
        reranked = self._apply_entity_boost(reranked, entities)[
            : self.reranker.settings.rerank_top_k
        ]
        if rerank_span:
            rerank_span.end(output={
                "num_final": len(reranked),
                "cross_encoder_skipped": skipped,
                # Confidence from the top reranker score (normalized to 0-1 range
                # via sigmoid so Langfuse shows a meaningful 0-100% value)
                "top_rerank_score": round(reranked[0][1], 4) if reranked else None,
            })

        return QueryPipelineResult(
            original_query=raw_query,
            rewritten_query=rewritten,
            entities=entities,
            reranked_chunks=reranked,
        )