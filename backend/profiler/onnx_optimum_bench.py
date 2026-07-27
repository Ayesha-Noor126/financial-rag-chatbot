"""
ONNX benchmark using Hugging Face Optimum — the CORRECT approach.

Manual torch.onnx.export on CrossEncoder produces wrong scores because it
only exports the transformer backbone, not the classification head's sigmoid
normalization that CrossEncoder.predict() applies. Optimum's
ORTModelForSequenceClassification exports the full pipeline correctly.

Run from backend/:
    python profiler/onnx_optimum_bench.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_DIR = Path("profiler/onnx_models")
ONNX_OPTIMUM_FP32 = ONNX_DIR / "optimum_fp32"
ONNX_OPTIMUM_INT8 = ONNX_DIR / "optimum_int8"

QUERY = "What was the total revenue for FY2023 and how did it compare to FY2022?"
CHUNK = (
    "Net revenues increased 5.6% to CHF 93.0 billion in 2023 from CHF 88.1 billion in 2022. "
    "Organic growth was 7.2%, driven by pricing of 5.4% and real internal growth of 1.8%. "
    "All operating segments delivered positive organic growth. Nestlé Waters returned to "
    "profitable growth. The Board of Directors proposes a dividend of CHF 3.00 per share. "
    "Underlying trading operating profit margin was 17.3%, up 20 basis points."
) * 2

MAX_LENGTH = 512
N_PAIRS = 10
BENCH_RUNS = 5
ONNX_DIR.mkdir(parents=True, exist_ok=True)

pairs = [(QUERY, CHUNK)] * N_PAIRS
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"\n{'='*60}")
print("OPTIMUM ONNX BENCHMARK")
print(f"{'='*60}")
print(f"ORT version: {ort.__version__}")
print(f"Providers  : {ort.get_available_providers()}")


def tokenize_pairs(pairs):
    texts_a = [q for q, _ in pairs]
    texts_b = [c for _, c in pairs]
    return tokenizer(
        texts_a, texts_b,
        padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="np",
    )


def run_ort_session(session, pairs, runs=BENCH_RUNS):
    enc = tokenize_pairs(pairs)
    session_inputs = {inp.name for inp in session.get_inputs()}
    inputs = {k: v.astype(np.int64) for k, v in enc.items() if k in session_inputs}
    # Warmup
    out = session.run(None, inputs)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        out = session.run(None, inputs)
        times.append((time.perf_counter() - t) * 1000)
    scores = out[0].squeeze().tolist()
    if isinstance(scores, float):
        scores = [scores]
    return sum(times) / len(times), scores


# ── Baseline PyTorch ───────────────────────────────────────────────────────
print("\n[1/4] PyTorch FP32 baseline...")
ce = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)
ce.predict(pairs, batch_size=N_PAIRS, show_progress_bar=False)  # warmup
times_pt = []
for _ in range(BENCH_RUNS):
    t = time.perf_counter()
    pt_scores = ce.predict(pairs, batch_size=N_PAIRS, show_progress_bar=False)
    times_pt.append((time.perf_counter() - t) * 1000)
t_pytorch = sum(times_pt) / len(times_pt)
print(f"  PyTorch FP32 : {t_pytorch:.1f} ms")

# ── Optimum ONNX FP32 export ───────────────────────────────────────────────
print("\n[2/4] Optimum ONNX FP32 export...")
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    if not (ONNX_OPTIMUM_FP32 / "model.onnx").exists():
        print("  Exporting with optimum (this takes ~30s the first time)...")
        ort_model_fp32 = ORTModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            export=True,
            provider="CPUExecutionProvider",
        )
        ort_model_fp32.save_pretrained(str(ONNX_OPTIMUM_FP32))
        tokenizer.save_pretrained(str(ONNX_OPTIMUM_FP32))
        print(f"  Saved → {ONNX_OPTIMUM_FP32}")
    else:
        print(f"  Using cached {ONNX_OPTIMUM_FP32}")
        ort_model_fp32 = ORTModelForSequenceClassification.from_pretrained(
            str(ONNX_OPTIMUM_FP32),
            provider="CPUExecutionProvider",
        )

    # Benchmark via raw ORT session (avoids Python overhead of optimum's forward())
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = torch.get_num_threads()
    sess_fp32 = ort.InferenceSession(
        str(ONNX_OPTIMUM_FP32 / "model.onnx"),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    t_fp32, scores_fp32 = run_ort_session(sess_fp32, pairs)
    print(f"  ONNX FP32    : {t_fp32:.1f} ms  ({(t_pytorch-t_fp32)/t_pytorch*100:+.1f}%)")

    # ── Optimum ONNX INT8 quantization ─────────────────────────────────────
    print("\n[3/4] INT8 quantization via optimum...")
    if not (ONNX_OPTIMUM_INT8 / "model_quantized.onnx").exists():
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        quantizer = ORTQuantizer.from_pretrained(str(ONNX_OPTIMUM_FP32))
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
        )
        quantizer.quantize(
            quantization_config=qconfig,
            save_dir=str(ONNX_OPTIMUM_INT8),
        )
        print(f"  Saved → {ONNX_OPTIMUM_INT8}")
    else:
        print(f"  Using cached {ONNX_OPTIMUM_INT8}")

    int8_model_path = ONNX_OPTIMUM_INT8 / "model_quantized.onnx"
    if not int8_model_path.exists():
        # Fall back to non-quantized name
        candidates = list(ONNX_OPTIMUM_INT8.glob("*.onnx"))
        int8_model_path = candidates[0] if candidates else None

    if int8_model_path and int8_model_path.exists():
        so_int8 = ort.SessionOptions()
        so_int8.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so_int8.intra_op_num_threads = torch.get_num_threads()
        sess_int8 = ort.InferenceSession(
            str(int8_model_path),
            sess_options=so_int8,
            providers=["CPUExecutionProvider"],
        )
        t_int8, scores_int8 = run_ort_session(sess_int8, pairs)
        print(f"  ONNX INT8    : {t_int8:.1f} ms  ({(t_pytorch-t_int8)/t_pytorch*100:+.1f}%)")

        # Score accuracy check
        print("\n─── Score accuracy (first 3 pairs) ─────────────────────────────")
        for i in range(3):
            pt = float(pt_scores[i])
            fp32 = scores_fp32[i] if isinstance(scores_fp32, list) else float(scores_fp32)
            int8 = scores_int8[i] if isinstance(scores_int8, list) else float(scores_int8)
            print(f"  PyTorch={pt:.4f}  ONNX-FP32={fp32:.4f}  ONNX-INT8={int8:.4f}  "
                  f"Δ(int8-pt)={abs(int8-pt):.4f}")
        max_delta_fp32 = max(abs(float(pt_scores[i]) - scores_fp32[i]) for i in range(min(3, len(scores_fp32))))
        max_delta_int8 = max(abs(float(pt_scores[i]) - scores_int8[i]) for i in range(min(3, len(scores_int8))))
        print(f"\n  Max Δ FP32 vs PyTorch  : {max_delta_fp32:.5f}  {'SAFE' if max_delta_fp32 < 0.05 else 'CHECK'}")
        print(f"  Max Δ INT8 vs PyTorch  : {max_delta_int8:.5f}  {'SAFE' if max_delta_int8 < 0.05 else 'CHECK (ranking may still be correct)'}")
    else:
        print("  Could not find quantized model file.")
        t_int8 = None

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  PyTorch FP32   : {t_pytorch:7.1f} ms  (baseline)")
    print(f"  ONNX FP32      : {t_fp32:7.1f} ms  ({(t_pytorch-t_fp32)/t_pytorch*100:+.1f}%)")
    if t_int8:
        print(f"  ONNX INT8      : {t_int8:7.1f} ms  ({(t_pytorch-t_int8)/t_pytorch*100:+.1f}%)")
    print()

except ImportError as e:
    print(f"  optimum import failed: {e}")
    print("  Install with: pip install optimum[onnxruntime]")

print("\n[4/4] Done.")
