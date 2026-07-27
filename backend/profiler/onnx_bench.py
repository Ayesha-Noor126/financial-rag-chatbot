"""
ONNX export + benchmark for bge-reranker-base.

Exports the model to ONNX format (FP32 first, then INT8 via dynamic
quantization), then benchmarks all three variants:
  A. PyTorch FP32  (current production baseline)
  B. ONNX FP32     (CPU graph optimizations, fused ops)
  C. ONNX INT8     (dynamic quantization: weights int8, activations fp32)

Run from backend/:
    python profiler/onnx_bench.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_DIR = Path("profiler/onnx_models")
ONNX_FP32_PATH = ONNX_DIR / "bge_reranker_fp32.onnx"
ONNX_INT8_PATH = ONNX_DIR / "bge_reranker_int8.onnx"

QUERY = "What was the total revenue for FY2023 and how did it compare to FY2022?"
CHUNK = (
    "Net revenues increased 5.6% to CHF 93.0 billion in 2023 from CHF 88.1 billion in 2022. "
    "Organic growth was 7.2%, driven by pricing of 5.4% and real internal growth of 1.8%. "
    "All operating segments delivered positive organic growth. Nestlé Waters returned to "
    "profitable growth. The Board of Directors proposes a dividend of CHF 3.00 per share. "
    "Underlying trading operating profit margin was 17.3%, up 20 basis points."
) * 2  # ~1000 chars — represents truncated production chunk

MAX_LENGTH = 512
N_PAIRS = 10
BENCH_RUNS = 5
ONNX_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def bench_pytorch(model: CrossEncoder, pairs: list, runs: int) -> float:
    """Warm up then average `runs` timed calls. Returns mean ms."""
    model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        times.append((time.perf_counter() - t) * 1000)
    return sum(times) / len(times)


def bench_onnx(session, tokenizer, pairs: list, runs: int) -> tuple[float, list[float]]:
    """Tokenize + run ONNX session. Returns (mean_ms, scores)."""
    texts_a = [q for q, _ in pairs]
    texts_b = [c for _, c in pairs]
    encodings = tokenizer(
        texts_a,
        texts_b,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    # ONNX model was exported with token_type_ids; add zeros if tokenizer omits it
    if "token_type_ids" not in encodings:
        encodings["token_type_ids"] = np.zeros_like(encodings["input_ids"])
    # Get the exact input names the session expects and filter to only those
    session_input_names = {inp.name for inp in session.get_inputs()}
    inputs = {k: v.astype(np.int64) for k, v in encodings.items() if k in session_input_names}

    # Warmup
    session.run(None, inputs)

    times = []
    for _ in range(runs):
        t = time.perf_counter()
        outputs = session.run(None, inputs)
        times.append((time.perf_counter() - t) * 1000)

    scores = outputs[0].squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]
    return sum(times) / len(times), scores


# ── Step 1: Load PyTorch model ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("ONNX BENCHMARK FOR bge-reranker-base")
print(f"{'='*60}")
print(f"Pairs per call   : {N_PAIRS}")
print(f"Benchmark runs   : {BENCH_RUNS}")
print(f"Chunk length     : {len(CHUNK)} chars")

print("\n[1/5] Loading PyTorch cross-encoder...")
ce_model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pairs = [(QUERY, CHUNK)] * N_PAIRS

print("[2/5] Benchmarking PyTorch FP32 baseline...")
t_pytorch = bench_pytorch(ce_model, pairs, BENCH_RUNS)
print(f"  PyTorch FP32   : {t_pytorch:.1f} ms")

# ── Step 2: Export to ONNX FP32 ───────────────────────────────────────────
print("\n[3/5] Exporting to ONNX FP32...")
if not ONNX_FP32_PATH.exists():
    hf_model = ce_model.model
    hf_model.eval()

    # Build a representative dummy input
    sample = tokenizer(
        f"{QUERY} [SEP] {CHUNK}",
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    dummy_ids = sample["input_ids"]
    dummy_mask = sample["attention_mask"]
    dummy_type = sample.get("token_type_ids", torch.zeros_like(dummy_ids))

    with torch.no_grad():
        torch.onnx.export(
            hf_model,
            args=(dummy_ids, dummy_mask, dummy_type),
            f=str(ONNX_FP32_PATH),
            opset_version=17,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits":         {0: "batch"},
            },
        )
    print(f"  Exported → {ONNX_FP32_PATH} ({ONNX_FP32_PATH.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"  Using cached {ONNX_FP32_PATH}")

# ── Step 3: INT8 dynamic quantization ─────────────────────────────────────
print("\n[4/5] Applying INT8 dynamic quantization...")
if not ONNX_INT8_PATH.exists():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(
        str(ONNX_FP32_PATH),
        str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8,
        # Quantize all MatMul ops (the dominant cost in BERT attention layers)
        op_types_to_quantize=["MatMul"],
        per_channel=False,   # per-tensor is faster on CPU; per-channel is more accurate
        reduce_range=False,  # reduce_range=True for AVX2 targets; False is safer cross-CPU
    )
    print(f"  Quantized → {ONNX_INT8_PATH} ({ONNX_INT8_PATH.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"  Using cached {ONNX_INT8_PATH}")

# ── Step 4: Benchmark ONNX variants ───────────────────────────────────────
print("\n[5/5] Benchmarking ONNX variants...")
import onnxruntime as ort

so_fp32 = ort.SessionOptions()
so_fp32.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so_fp32.intra_op_num_threads = torch.get_num_threads()
sess_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), sess_options=so_fp32, providers=["CPUExecutionProvider"])

so_int8 = ort.SessionOptions()
so_int8.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so_int8.intra_op_num_threads = torch.get_num_threads()
sess_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), sess_options=so_int8, providers=["CPUExecutionProvider"])

t_onnx_fp32, scores_fp32 = bench_onnx(sess_fp32, tokenizer, pairs, BENCH_RUNS)
t_onnx_int8, scores_int8 = bench_onnx(sess_int8, tokenizer, pairs, BENCH_RUNS)

print(f"  ONNX FP32      : {t_onnx_fp32:.1f} ms")
print(f"  ONNX INT8      : {t_onnx_int8:.1f} ms")

# ── Score accuracy check ───────────────────────────────────────────────────
# All pairs are identical, so all scores should be equal.
# What we care about is that INT8 scores don't diverge significantly from FP32.
print("\n─── Score accuracy: INT8 vs FP32 ────────────────────────────────────")
pt_scores = ce_model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
for i, (pt, fp32, int8) in enumerate(zip(pt_scores, scores_fp32, scores_int8)):
    print(f"  Pair {i+1}: PyTorch={pt:.4f}  ONNX-FP32={fp32:.4f}  ONNX-INT8={int8:.4f}  "
          f"Δ(int8-fp32)={abs(int8-fp32):.4f}")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"  A. PyTorch FP32 (baseline) : {t_pytorch:7.1f} ms")
print(f"  B. ONNX FP32               : {t_onnx_fp32:7.1f} ms  ({(t_pytorch-t_onnx_fp32)/t_pytorch*100:+.1f}%)")
print(f"  C. ONNX INT8               : {t_onnx_int8:7.1f} ms  ({(t_pytorch-t_onnx_int8)/t_pytorch*100:+.1f}%)")
print()
print("Score deviation INT8 vs FP32 (max acceptable ≈ 0.05 for ranking stability):")
max_delta = max(abs(i - f) for i, f in zip(scores_int8, scores_fp32))
print(f"  Max |INT8 - FP32| = {max_delta:.5f}  → {'SAFE' if max_delta < 0.05 else 'EXCEEDS THRESHOLD'}")
print()
