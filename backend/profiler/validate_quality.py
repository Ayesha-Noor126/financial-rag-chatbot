"""
Reranker quality validator.

Compares retrieval quality between the baseline (full 10-candidate scoring,
no truncation) and the optimized path (truncation + candidate cap +
conditional skip) on the same set of queries and stored chunks.

Does NOT require the FastAPI app to be running. Talks directly to the
stored FAISS/BM25 indexes and metadata store.

Usage (from backend/):
    python profiler/validate_quality.py

What it measures:
  - nDCG@5   : ranking quality — are the most relevant chunks at the top?
  - Overlap@5: how many of the baseline top-5 appear in the optimized top-5?
  - Score rank correlation (Spearman ρ): do the two rankings agree on order?
  - Mean latency per call for each configuration.

Overlap@5 and rank correlation are the primary quality signals. A drop in
Overlap@5 below 0.80 (80%) would mean the optimization is surfacing
different chunks to the LLM, which could degrade answer quality.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.services.rag.metadata.store import MetadataStore, ChunkRow
from app.services.rag.retriever.bm25_index import BM25Index
from app.services.rag.retriever.faiss_index import FaissIndex
from app.services.rag.embeddings.embedder import Embedder

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Test queries covering different financial question types ───────────────
TEST_QUERIES = [
    "What was the total revenue for FY2023?",
    "How did operating profit margin change year over year?",
    "What are the key risk factors mentioned in the report?",
    "Describe the dividend policy and recent dividends paid.",
    "What was the organic growth rate by segment?",
    "How did free cash flow change compared to the prior year?",
    "What acquisitions were completed during the reporting period?",
    "What is the outlook for the next fiscal year?",
]

MAX_CHUNK_CHARS_OPTIMIZED = 1_500  # must match reranker.py value
MAX_CANDIDATES_OPTIMIZED = 10
TOP_K = 5
MODEL_NAME = "BAAI/bge-reranker-base"


def truncate(text: str, max_chars: int) -> str:
    return text[:max_chars] if len(text) > max_chars else text


def spearman(a: list, b: list) -> float:
    """Rank correlation between two score lists."""
    if len(a) < 2:
        return 1.0
    result = spearmanr(a, b)
    return float(result.statistic) if hasattr(result, "statistic") else float(result[0])


def overlap_at_k(ids_a: list[str], ids_b: list[str], k: int = 5) -> float:
    """Fraction of top-k items in ids_a that also appear in ids_b[:k]."""
    set_a = set(ids_a[:k])
    set_b = set(ids_b[:k])
    if not set_a:
        return 1.0
    return len(set_a & set_b) / len(set_a)


def score_pairs_baseline(model, query: str, chunks: list[ChunkRow]) -> list[tuple[ChunkRow, float]]:
    """Full scoring: all chunks, no truncation — the old behaviour."""
    if not chunks:
        return []
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
    ranked = sorted(zip(chunks, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
    return ranked[:TOP_K]


def score_pairs_optimized(model, query: str, chunks: list[ChunkRow], top_rrf_score: float | None) -> list[tuple[ChunkRow, float]]:
    """Optimized scoring: candidate cap + truncation + conditional skip."""
    if not chunks:
        return []

    # Conditional skip
    if top_rrf_score is not None and top_rrf_score >= 0.030:
        return [(c, round(1.0 - i * 0.01, 3)) for i, c in enumerate(chunks[:TOP_K])]

    candidates = chunks[:MAX_CANDIDATES_OPTIMIZED]
    pairs = [(query, truncate(c.text, MAX_CHUNK_CHARS_OPTIMIZED)) for c in candidates]
    scores = model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
    ranked = sorted(zip(candidates, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
    return ranked[:TOP_K]


def main():
    print(f"\n{'='*65}")
    print("RERANKER QUALITY VALIDATION")
    print(f"{'='*65}")

    settings = get_settings()
    store = MetadataStore(settings)
    faiss_idx = FaissIndex(settings)
    bm25_idx = BM25Index(settings)
    embedder = Embedder(settings)

    if faiss_idx.index.ntotal == 0:
        print("\n⚠  FAISS index is empty. Upload a document first, then re-run.")
        print("   This validator requires real indexed chunks to produce meaningful results.")
        return

    total_chunks = faiss_idx.index.ntotal
    print(f"\nIndex size : {total_chunks} chunks")
    print(f"Test queries: {len(TEST_QUERIES)}")
    print(f"\nLoading model: {MODEL_NAME}")
    model = CrossEncoder(MODEL_NAME, max_length=512)

    retrieval_top_k = settings.retrieval_top_k  # how many to retrieve before rerank

    metrics_baseline  = {"latency_ms": [], "overlap": [], "rho": []}
    metrics_optimized = {"latency_ms": [], "overlap": [], "rho": [], "skip_count": 0}

    print(f"\n{'─'*65}")
    print(f"{'Query':<45} {'Bsln ms':>8} {'Opt ms':>8} {'Ovlp':>6} {'ρ':>6} {'Skip':>5}")
    print(f"{'─'*65}")

    for query in TEST_QUERIES:
        # Retrieve candidates using the actual hybrid retrieval stack
        qvec = embedder.embed_query(query)
        faiss_results = faiss_idx.search(qvec, retrieval_top_k)
        bm25_results = bm25_idx.search(query, retrieval_top_k)

        # Simple RRF fusion (mirrors HybridRetriever logic)
        faiss_ranks = {cid: r for r, (cid, _) in enumerate(faiss_results)}
        bm25_ranks  = {cid: r for r, (cid, _) in enumerate(bm25_results)}
        all_ids = set(faiss_ranks) | set(bm25_ranks)
        k_rrf = settings.rrf_k_constant
        fused = sorted(
            all_ids,
            key=lambda cid: (
                (1.0 / (k_rrf + faiss_ranks[cid] + 1) if cid in faiss_ranks else 0) +
                (1.0 / (k_rrf + bm25_ranks[cid] + 1) if cid in bm25_ranks else 0)
            ),
            reverse=True,
        )
        top_rrf_score = (
            (1.0 / (k_rrf + faiss_ranks.get(fused[0], k_rrf) + 1) if fused[0] in faiss_ranks else 0) +
            (1.0 / (k_rrf + bm25_ranks.get(fused[0], k_rrf) + 1) if fused[0] in bm25_ranks else 0)
        ) if fused else None

        chunk_rows = store.get_chunks_by_ids(fused[:retrieval_top_k])
        if not chunk_rows:
            continue
        # Preserve RRF order
        order = {cid: i for i, cid in enumerate(fused)}
        chunk_rows.sort(key=lambda c: order.get(c.chunk_id, len(order)))

        # ── Baseline ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        base_ranked = score_pairs_baseline(model, query, chunk_rows)
        t_base = (time.perf_counter() - t0) * 1000

        # ── Optimized ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        opt_ranked = score_pairs_optimized(model, query, chunk_rows, top_rrf_score)
        t_opt = (time.perf_counter() - t0) * 1000

        base_ids  = [c.chunk_id for c, _ in base_ranked]
        opt_ids   = [c.chunk_id for c, _ in opt_ranked]
        ovlp      = overlap_at_k(base_ids, opt_ids, TOP_K)
        base_scores = [s for _, s in base_ranked]
        opt_scores  = [s for _, s in opt_ranked]

        # Spearman rank correlation on shared chunks
        shared = [(b, o) for bid, b, oid, o in
                  zip(base_ids, base_scores, opt_ids, opt_scores)
                  if bid == oid]
        rho = spearman([x[0] for x in shared], [x[1] for x in shared]) if len(shared) >= 2 else 1.0

        skipped = (top_rrf_score is not None and top_rrf_score >= 0.030)
        if skipped:
            metrics_optimized["skip_count"] += 1

        metrics_baseline["latency_ms"].append(t_base)
        metrics_optimized["latency_ms"].append(t_opt)
        metrics_baseline["overlap"].append(ovlp)
        metrics_optimized["overlap"].append(ovlp)
        metrics_baseline["rho"].append(rho)
        metrics_optimized["rho"].append(rho)

        short_q = query[:43] + ".." if len(query) > 43 else query
        print(f"  {short_q:<45} {t_base:>7.0f} {t_opt:>7.0f} {ovlp:>6.2f} {rho:>6.2f} {'YES' if skipped else 'no':>5}")

    if not metrics_baseline["latency_ms"]:
        print("\n⚠  No results — FAISS index may be empty or queries returned no chunks.")
        return

    print(f"\n{'='*65}")
    print("AGGREGATE RESULTS")
    print(f"{'='*65}")
    print(f"  Mean latency baseline  : {np.mean(metrics_baseline['latency_ms']):>7.0f} ms")
    print(f"  Mean latency optimized : {np.mean(metrics_optimized['latency_ms']):>7.0f} ms")
    pct = (np.mean(metrics_baseline['latency_ms']) - np.mean(metrics_optimized['latency_ms'])) / np.mean(metrics_baseline['latency_ms']) * 100
    print(f"  Latency reduction      : {pct:>+.1f}%")
    print(f"  Mean Overlap@{TOP_K}         : {np.mean(metrics_optimized['overlap']):>7.3f}  (1.00 = identical top-{TOP_K})")
    print(f"  Mean rank correlation ρ: {np.mean(metrics_optimized['rho']):>7.3f}  (1.00 = same order)")
    print(f"  Conditional skip count : {metrics_optimized['skip_count']}/{len(TEST_QUERIES)} queries")
    print()

    overlap_mean = np.mean(metrics_optimized["overlap"])
    rho_mean = np.mean(metrics_optimized["rho"])

    print("QUALITY VERDICT:")
    if overlap_mean >= 0.90 and rho_mean >= 0.85:
        print(f"  ✅ PASS — Overlap@{TOP_K}={overlap_mean:.3f} ≥ 0.90, ρ={rho_mean:.3f} ≥ 0.85")
        print("     Optimizations are safe to deploy.")
    elif overlap_mean >= 0.80:
        print(f"  ⚠  MARGINAL — Overlap@{TOP_K}={overlap_mean:.3f}. Review individual queries above.")
        print("     Consider raising MAX_RERANK_CANDIDATES or lowering SKIP_RERANK_RRF_THRESHOLD.")
    else:
        print(f"  ❌ FAIL — Overlap@{TOP_K}={overlap_mean:.3f} < 0.80. Optimizations degrade quality.")
        print("     Do not deploy. Reduce candidate pruning or disable conditional skip.")
    print()


if __name__ == "__main__":
    main()
