"""
Optimiertes DSP-Modul für AURIK v8.
==================================

High-performance DSP operations using:
- NumExpr: 2× speedup for vectorized operations
- Cython: 3-5× speedup for critical loops
- pyFFTW: 1.5-2× speedup for FFT operations

Combined speedup: 2-5× faster DSP processing
"""

# NumExpr optimizations (always available)
from .numexpr_ops import OptimizedDSP, hard_threshold, soft_threshold, spectral_gate

# FFT caching
try:
    from .fft_cache import CachedFFT, irfft, istft, rfft, stft

    HAS_PYFFTW = True
except ImportError:
    HAS_PYFFTW = False
    CachedFFT = None  # type: ignore
    rfft = None  # type: ignore[assignment]
    irfft = None  # type: ignore[assignment]
    stft = None  # type: ignore[assignment]
    istft = None  # type: ignore[assignment]

# Cython loops (requires compilation)
try:
    from . import cython_loops  # type: ignore[attr-defined]

    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    cython_loops = None

__all__ = [
    "HAS_CYTHON",
    # Flags
    "HAS_PYFFTW",
    # FFT
    "CachedFFT",
    # NumExpr
    "OptimizedDSP",
    # Cython
    "cython_loops",
    "hard_threshold",
    "irfft",
    "istft",
    "rfft",
    "soft_threshold",
    "spectral_gate",
    "stft",
]
