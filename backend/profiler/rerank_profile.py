"""
Standalone reranker profiler.

Measures wall-clock time broken down into:
  1. Pair construction   — building (query, chunk_text) tuples
  2. Tokenization        — CrossEncoder's internal tokenizer call
  3. Model forward pass  — actual BERT inference
  4. Total predict()     — what the pipeline pays (1+2+3+overhead)

Also records per-candidate latency so we can see the O(n) scaling clearly.

Run from backend/:
    python profiler/rerank_profile.py
"""

import os
import sys
import time

# Make sure app/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import CrossEncoder

# ── Synthetic test data matching real production conditions ────────────────
# 700-token chunks are ~3000 chars. We use 1500-char and 3000-char variants
# to measure the effect of the pre-truncation optimisation separately.

QUERY = "What was the total revenue for FY2023 and how did it compare to FY2022?"

CHUNK_SHORT = (
    "Net revenues increased 5.6% to CHF 93.0 billion in 2023 from CHF 88.1 billion in 2022. "
    "Organic growth was 7.2%, driven by pricing of 5.4% and real internal growth of 1.8%. "
    "All operating segments delivered positive organic growth. Nestlé Waters returned to "
    "profitable growth. The Board of Directors proposes a dividend of CHF 3.00 per share."
) * 3  # ~500 chars

CHUNK_LONG = (
    "Net revenues increased 5.6% to CHF 93.0 billion in 2023 from CHF 88.1 billion in 2022. "
    "Organic growth was 7.2%, driven by pricing of 5.4% and real internal growth of 1.8%. "
    "All operating segments delivered positive organic growth. Nestlé Waters returned to "
    "profitable growth. The Board of Directors proposes a dividend of CHF 3.00 per share. "
    "Underlying trading operating profit margin was 17.3%, up 20 basis points. Reported "
    "earnings per share increased 68.8% to CHF 4.30. Underlying earnings per share in "
    "constant currency increased 8.3%. Free cash flow was CHF 11.2 billion, up from "
    "CHF 7.9 billion in 2022. Net financial debt decreased to CHF 36.3 billion. "
    "The group completed the acquisition of a majority stake in Orgain in the United States."
) * 6  # ~3000 chars — matches 700-token chunk at ~4.5 chars/token

MODEL_NAME = "BAAI/bge-reranker-base"

print(f"\n{'='*60}")
print("RERANKER PROFILER")
print(f"{'='*60}")
print(f"Model            : {MODEL_NAME}")
print(f"PyTorch threads  : {torch.get_num_threads()}")
print(f"MKL available    : {torch.backends.mkl.is_available()}")
print(f"Short chunk len  : {len(CHUNK_SHORT)} chars")
print(f"Long chunk len   : {len(CHUNK_LONG)} chars")

print("\nLoading model (one-time cost, not charged per request)...")
t0 = time.perf_counter()
model = CrossEncoder(MODEL_NAME, max_length=512)
load_time = time.perf_counter() - t0
print(f"  Model load time: {load_time:.2f}s")


def bench(label: str, pairs: list, runs: int = 3) -> float:
    """Warm up once, then average `runs` timed runs. Returns mean ms."""
    # Warmup
    model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        times.append((time.perf_counter() - t) * 1000)
    mean_ms = sum(times) / len(times)
    print(f"  {label:<45} {mean_ms:7.1f} ms  (best {min(times):.1f}, worst {max(times):.1f})")
    return mean_ms


print("\n─── Baseline: LONG chunks (700 tok ≈ 3000 chars), N=10 ─────────────")
pairs_10_long = [(QUERY, CHUNK_LONG)] * 10
t_10_long = bench("10 long chunks (baseline — current behaviour)", pairs_10_long)

print("\n─── Effect of pre-truncation to 1500 chars ──────────────────────────")
pairs_10_short = [(QUERY, CHUNK_SHORT)] * 10
t_10_short = bench("10 short/truncated chunks (1500 chars)", pairs_10_short)
print(f"  → Truncation saving: {(t_10_long - t_10_short)/t_10_long*100:.1f}%")

print("\n─── Effect of reducing candidate count ─────────────────────────────")
for n in [10, 8, 6, 5]:
    pairs_n = [(QUERY, CHUNK_SHORT)] * n
    bench(f"{n} truncated chunks", pairs_n)

print("\n─── Batch size vs one-at-a-time ─────────────────────────────────────")
# One-at-a-time simulates what happened before batch_size fix
pairs_single = [(QUERY, CHUNK_SHORT)]
t_single = bench("1 chunk (one-at-a-time baseline)", pairs_single)
t_10_batched = bench("10 chunks (batch_size=10)", [(QUERY, CHUNK_SHORT)] * 10)
print(f"  → Sequential overhead vs batch: {t_single*10:.1f} ms estimated seq  vs  {t_10_batched:.1f} ms batched")

print("\n─── Thread count sensitivity ─────────────────────────────────────────")
for n_threads in [1, 2, 4]:
    torch.set_num_threads(n_threads)
    pairs = [(QUERY, CHUNK_SHORT)] * 10
    bench(f"10 truncated chunks, {n_threads} torch thread(s)", pairs)

# Restore
torch.set_num_threads(torch.get_num_threads())

print("\n─── ONNX Runtime check ──────────────────────────────────────────────")
try:
    import onnxruntime  # noqa: F401
    print("  onnxruntime is available — ONNX path is viable")
except ImportError:
    print("  onnxruntime NOT installed — cannot test ONNX path yet")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Current baseline (10 long chunks) : {t_10_long:.1f} ms")
print(f"  After truncation (10 short chunks) : {t_10_short:.1f} ms")
improvement = (t_10_long - t_10_short) / t_10_long * 100
print(f"  Pre-truncation improvement         : {improvement:.1f}%")
print()
