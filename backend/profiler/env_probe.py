"""Probe CPU, PyTorch thread config, and ONNX availability."""
import os
import platform

import torch

print("=== CPU / Runtime Environment ===")
print("Python platform :", platform.platform())
print("Processor       :", platform.processor())
print("PyTorch version :", torch.__version__)
print("MKL available   :", torch.backends.mkl.is_available())
print("OpenMP available:", torch.backends.openmp.is_available())
print("CPU threads     :", torch.get_num_threads())
print("Intraop threads :", torch.get_num_interop_threads())
print("OMP_NUM_THREADS :", os.environ.get("OMP_NUM_THREADS", "not set"))
print("MKL_NUM_THREADS :", os.environ.get("MKL_NUM_THREADS", "not set"))

try:
    import onnxruntime as ort
    print("\n=== ONNX Runtime ===")
    print("Version  :", ort.__version__)
    print("Providers:", ort.get_available_providers())
except ImportError:
    print("\nONNX Runtime : NOT installed")

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa: F401
    print("optimum[onnxruntime] : installed")
except ImportError:
    print("optimum[onnxruntime] : NOT installed")
