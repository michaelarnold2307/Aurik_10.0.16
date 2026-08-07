"""PyInstaller Runtime-Hook — Thread-Limit-Konfiguration.

Setzt OS-Level-Umgebungsvariablen für OpenMP, OpenBLAS und MKL,
bevor irgendeine ML-Bibliothek (ONNX Runtime, PyTorch, scipy) den
Thread-Pool initialisiert.

Referenz: §2.37 CPU-Aware Pipeline Scheduling (copilot-instructions.md)
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS = cpu_count()
"""

import logging
import os

logger = logging.getLogger(__name__)

# Use ALL logical CPUs for maximum throughput on Ryzen 7 (8C/16T).
# For ONNX sessions, intra/inter_op_num_threads is set additionally
# via make_session_options() — these vars are the OS-level safety net.
_n = str(os.cpu_count() or 1)

os.environ.setdefault("OMP_NUM_THREADS", _n)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _n)
os.environ.setdefault("MKL_NUM_THREADS", _n)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _n)

# GPU policy: ml_device_manager decides at runtime (ROCm or DirectML if available).
# Do NOT unconditionally suppress CUDA_VISIBLE_DEVICES here — that would hide ROCm.
# On systems without a supported GPU, MLDeviceManager falls back to CPU automatically.U automatically.

# §2.37 InterOp-Thread-Pool — limits parallel operator dispatch in PyTorch.
# Set to 4 to balance inter-operator and intra-operator parallelism on Ryzen 7.
try:
    import torch as _torch

    _torch.set_num_interop_threads(4)
except Exception:
    logger.warning("runtime_hook_threading.py::unknown Ersatzpfad", exc_info=True)
    pass  # torch optional at hook time (not yet imported)
