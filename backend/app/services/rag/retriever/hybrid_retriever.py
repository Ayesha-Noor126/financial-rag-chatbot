"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF) -- async + parallelized (Phase 4).

Design choice on RRF itself (unchanged from Phase 2): raw FAISS scores
(cosine similarity, 0-1) and BM25 scores (unbounded, corpus-dependent) are
not comparable on the same scale, so we fuse by rank position rather than
score value:

    RRF_score(chunk) = sum over each retriever of  1 / (k + rank)

Phase 4 change -- genuine parallelism: BM25 search needs only the raw query
text, so it can start immediately. FAISS search needs the query embedded
first. Rather than doing embed -> faiss sequentially THEN bm25 sequentially
after that (paying for both full latencies back to back), we kick off BM25
immediately and run the embed+FAISS branch concurrently via asyncio.gather.
This overlaps BM25's search time with the embedding model's inference time,
which is the real latency win the spec's "Parallel BM25 + Vector Search"
requirement is asking for -- not just calling both, but overlapping them.

Both FAISS and BM25 search are blocking/CPU-bound calls, so they're pushed
onto the asyncio thread pool via asyncio.to_thread rather than awaited
directly (they have no native async API).
"""

import asyncio
from dataclasses import dataclass

from app.core.config import Settings
from app.services.rag.embeddings.embedder import Embedder
from app.services.rag.retriever.bm25_index import BM25Index
from app.services.rag.retriever.faiss_index import FaissIndex


@dataclass
class RetrievedChunk:
    chunk_id: str
    rrf_score: float
    faiss_rank: int | None
    bm25_rank: int | None


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        faiss_index: FaissIndex,
        bm25_index: BM25Index,
    ):
        self.settings = settings
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.k = settings.rrf_k_constant

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        entity_terms: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval_top_k
        search_query = query
        if entity_terms:
            search_query = f"{query} {' '.join(entity_terms)}"

        bm25_task = asyncio.to_thread(self.bm25_index.search, search_query, top_k)
        faiss_task = self._faiss_search(query, top_k)
        bm25_results, faiss_results = await asyncio.gather(bm25_task, faiss_task)

        faiss_ranks = {cid: rank for rank, (cid, _) in enumerate(faiss_results)}
        bm25_ranks = {cid: rank for rank, (cid, _) in enumerate(bm25_results)}
        all_chunk_ids = set(faiss_ranks) | set(bm25_ranks)

        fused: list[RetrievedChunk] = []
        for chunk_id in all_chunk_ids:
            score = 0.0
            f_rank = faiss_ranks.get(chunk_id)
            b_rank = bm25_ranks.get(chunk_id)
            if f_rank is not None:
                score += 1.0 / (self.k + f_rank + 1)  # +1: ranks are 0-indexed
            if b_rank is not None:
                score += 1.0 / (self.k + b_rank + 1)
            fused.append(
                RetrievedChunk(
                    chunk_id=chunk_id, rrf_score=score, faiss_rank=f_rank, bm25_rank=b_rank
                )
            )

        fused.sort(key=lambda c: c.rrf_score, reverse=True)
        return fused[:top_k]

    async def _faiss_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_vector = await asyncio.to_thread(self.embedder.embed_query, query)
        return await asyncio.to_thread(self.faiss_index.search, query_vector, top_k)