import asyncio
import sys

from app.core.config import get_settings
from app.core.llm_client import LLMClient
from app.services.rag.embeddings.embedder import Embedder
from app.services.rag.generation.confidence_scorer import ConfidenceScorer
from app.services.rag.generation.context_compressor import ContextCompressor
from app.services.rag.metadata.store import MetadataStore
from app.services.rag.ner.ner_service import NERService
from app.services.rag.query_pipeline_service import QueryPipelineService
from app.services.rag.query_rewriter.rewriter import QueryRewriter
from app.services.rag.reranker.reranker import Reranker
from app.services.rag.retriever.bm25_index import BM25Index
from app.services.rag.retriever.faiss_index import FaissIndex
from app.services.rag.retriever.hybrid_retriever import HybridRetriever


def safe(text: str, n: int = 120) -> str:
    return text[:n].encode("ascii", "replace").decode("ascii").replace("\n", " ")


async def main() -> None:
    settings = get_settings()
    embedder = Embedder(settings)
    faiss = FaissIndex(settings)
    bm25 = BM25Index(settings)
    store = MetadataStore(settings)

    query = "What was the company's total revenue?"
    print("QUERY:", query)
    print("FAISS total:", faiss.index.ntotal)
    print("BM25 total:", len(bm25.chunk_ids))

    vec = embedder.embed_query(query)
    faiss_hits = faiss.search(vec, 10)
    bm25_hits = bm25.search(query, 10)
    print("\nFAISS top 10:")
    for rank, (cid, score) in enumerate(faiss_hits, 1):
        rows = store.get_chunks_by_ids([cid])
        if rows:
            r = rows[0]
            print(
                f"  {rank}. page={r.page_number} score={score:.4f} "
                f"rev={'revenue' in r.text.lower()} {safe(r.text)}"
            )

    print("\nBM25 top 10:")
    for rank, (cid, score) in enumerate(bm25_hits, 1):
        rows = store.get_chunks_by_ids([cid])
        if rows:
            r = rows[0]
            print(
                f"  {rank}. page={r.page_number} score={score:.4f} "
                f"rev={'revenue' in r.text.lower()} {safe(r.text)}"
            )

    retriever = HybridRetriever(settings, embedder, faiss, bm25)
    fused = await retriever.retrieve(query)
    print("\nRRF fused top 10:")
    for rank, hit in enumerate(fused[:10], 1):
        rows = store.get_chunks_by_ids([hit.chunk_id])
        if rows:
            r = rows[0]
            print(
                f"  {rank}. page={r.page_number} rrf={hit.rrf_score:.4f} "
                f"rev={'revenue' in r.text.lower()} {safe(r.text)}"
            )

    pipeline = QueryPipelineService(
        QueryRewriter(LLMClient(settings)),
        NERService(settings),
        retriever,
        Reranker(settings),
        store,
    )
    result = await pipeline.run("what is the total revenue?")
    print("\nRERANKED top 5:")
    for i, (chunk, score) in enumerate(result.reranked_chunks, 1):
        print(
            f"  {i}. page={chunk.page_number} score={score:.4f} "
            f"rev={'revenue' in chunk.text.lower()} {safe(chunk.text)}"
        )

    compressed = ContextCompressor(settings).compress(result.reranked_chunks)
    conf = ConfidenceScorer(settings).score(result.entities, compressed)
    print("\nCONFIDENCE:", conf)
    print("COMPRESSED chunks with revenue:", sum(1 for c in compressed if "revenue" in c.chunk.text.lower()))


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
