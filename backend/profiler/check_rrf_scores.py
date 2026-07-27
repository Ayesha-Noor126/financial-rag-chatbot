"""Print the actual RRF scores produced for each test query so we can
calibrate SKIP_RERANK_RRF_THRESHOLD correctly."""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from app.core.config import get_settings
from app.services.rag.embeddings.embedder import Embedder
from app.services.rag.retriever.faiss_index import FaissIndex
from app.services.rag.retriever.bm25_index import BM25Index

QUERIES = [
    "What was the total revenue for FY2023?",
    "What are the key risk factors mentioned in the report?",
    "What was the organic growth rate by segment?",
    "What is the outlook for the next fiscal year?",
]

settings = get_settings()
faiss_idx = FaissIndex(settings)
bm25_idx  = BM25Index(settings)
embedder  = Embedder(settings)
k         = settings.rrf_k_constant
top_k     = settings.retrieval_top_k

print(f"\n{'Query':<50} {'Top RRF':>10} {'#2 RRF':>10} {'Gap':>8}")
print("─" * 82)
for q in QUERIES:
    qvec = embedder.embed_query(q)
    f_res = faiss_idx.search(qvec, top_k)
    b_res = bm25_idx.search(q, top_k)
    f_ranks = {cid: r for r, (cid, _) in enumerate(f_res)}
    b_ranks = {cid: r for r, (cid, _) in enumerate(b_res)}
    all_ids = set(f_ranks) | set(b_ranks)
    fused = sorted(all_ids,
        key=lambda cid: (
            (1/(k + f_ranks[cid]+1) if cid in f_ranks else 0) +
            (1/(k + b_ranks[cid]+1) if cid in b_ranks else 0)
        ), reverse=True)
    def rrf(cid):
        return ((1/(k+f_ranks[cid]+1) if cid in f_ranks else 0) +
                (1/(k+b_ranks[cid]+1) if cid in b_ranks else 0))
    scores = [rrf(cid) for cid in fused[:5]]
    gap = scores[0] - scores[1] if len(scores) > 1 else 0
    short = q[:48] + ".." if len(q) > 48 else q
    print(f"  {short:<50} {scores[0]:>10.5f} {scores[1]:>10.5f} {gap:>8.5f}")

print(f"\nNote: threshold in code = 0.030 (2/61 ≈ 0.0328 is the theoretical max for rank-1 in both)")
print("      If no scores reach 0.030, the threshold is too high for this corpus.")
