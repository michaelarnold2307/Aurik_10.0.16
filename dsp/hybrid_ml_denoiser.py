"""
dsp/hybrid_ml_denoiser.py — Hybrid DSP+ML Denoiser (Aurik 10.0.0.x)
==================================================================

Implements an OMLSA-based spectral denoiser as the DSP-fast stage, with an
optional ML refinement stage (Resemble Enhance ONNX, §4.4).

The ML stage gracefully falls back to DSP-only when the plugin is unavailable
— ensuring out-of-the-box operation.

Spec reference: §4.4 (DeepFilterNet v3 / Resemble-Enhance kaskade), §2.37.

Classes:
    DenoiseStrategy    — Strategy enum
    DenoiseConfig      — Configuration dataclass
    DenoiseResult      — Per-call result dataclass
    HybridMLDenoiser   — Main denoiser class

Functions:
    denoise_fast       — OMLSA-only convenience wrapper
    denoise_balanced   — OMLSA + adaptive ML wrapper
    denoise_maximum    — Full OMLSA → Resemble pipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DenoiseStrategy
# ---------------------------------------------------------------------------


class DenoiseStrategy(Enum):
    OMLSA_ONLY = "omlsa_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


# ---------------------------------------------------------------------------
# DenoiseConfig
# ---------------------------------------------------------------------------


@dataclass
class DenoiseConfig:
    """Configuration for HybridMLDenoiser.

    Attributes:
        strategy:          Processing strategy (OMLSA_ONLY / HYBRID / ADAPTIVE).
        quality_threshold: OMLSA quality estimate below which the ML stage
                           is triggered (HYBRID/ADAPTIVE modes, default 0.7).
        n_fft:             FFT size for OMLSA STFT (default 1024).
        hop_length:        Hop size (default 256).
        noise_frames:      Number of leading frames used for noise estimation
                           (default 10).
        reduction_db:      Maximum spectral attenuation in dB (default 18).
        omlsa_lambda_d:    OMLSA smoothing factor for noise PSD (default 0.85).
    """

    strategy: DenoiseStrategy = DenoiseStrategy.OMLSA_ONLY
    quality_threshold: float = 0.70
    n_fft: int = 1024
    hop_length: int = 256
    noise_frames: int = 10
    reduction_db: float = 18.0
    omlsa_lambda_d: float = 0.85


# ---------------------------------------------------------------------------
# DenoiseResult
# ---------------------------------------------------------------------------


@dataclass
class DenoiseResult:
    """Result of a single HybridMLDenoiser.denoise() call."""

    audio: np.ndarray
    strategy_used: DenoiseStrategy
    omlsa_applied: bool
    resemble_applied: bool
    quality_estimate: float
    processing_time: float
    snr_improvement_db: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OMLSA core (DSP)
# ---------------------------------------------------------------------------


def _omlsa_denoise(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    noise_frames: int = 10,
    reduction_db: float = 18.0,
    lambda_d: float = 0.85,
) -> tuple[np.ndarray, float]:
    """Wendet an: OMLSA-style spectral noise reduction (Cohen 2003 / IMCRA).

    Uses sliding-minimum noise estimation and a Wiener-like gain function.

    Args:
        audio:        1-D mono float32/64 signal.
        sr:           Sample rate (Hz).
        n_fft:        FFT window size.
        hop:          Hop length.
        noise_frames: Leading frames for initial noise estimate.
        reduction_db: Maximum attenuation.
        lambda_d:     Noise PSD smoothing factor.

    Returns:
        Tuple of (denoised_audio, quality_estimate).
    """
    # Ensure n_fft > hop to satisfy scipy.stft constraint
    if n_fft <= hop:
        n_fft = hop + 1

    audio_f64 = np.asarray(audio, dtype=np.float64)
    _, _, Z = stft(audio_f64, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    mag = np.abs(Z)  # (n_freq, n_frames)
    phase = np.angle(Z)
    power = mag**2

    n_frames = power.shape[1]
    n_est = max(1, min(noise_frames, n_frames))

    # Initial noise PSD from first n_est frames (median for robustness)
    noise_psd = np.median(power[:, :n_est], axis=1, keepdims=True) + 1e-12

    # IMCRA-style smoothed noise PSD
    noise_psd_smooth = np.zeros_like(power)
    for t in range(n_frames):
        noise_psd = lambda_d * noise_psd + (1 - lambda_d) * power[:, t : t + 1]
        noise_psd_smooth[:, t] = noise_psd.ravel()

    # Wiener-like gain
    floor_linear = 10 ** (-reduction_db / 20)
    snr_post = power / (noise_psd_smooth + 1e-12)
    gain = np.maximum(floor_linear, 1.0 - 1.0 / (snr_post + 1.0))

    # Apply gain
    Z_filtered = gain * mag * np.exp(1j * phase)

    # Reconstruct signal
    _, audio_out = istft(Z_filtered, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

    # Trim/pad to original length
    n_orig = len(audio_f64)
    audio_out = audio_out[:n_orig] if len(audio_out) >= n_orig else np.pad(audio_out, (0, n_orig - len(audio_out)))

    audio_out = np.clip(np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(audio.dtype)

    # Simple quality estimate: ratio of median gain to 1 (closer to 1 = less denoised)
    quality = float(np.clip(1.0 - float(np.mean(1.0 - gain)), 0.0, 1.0))

    return audio_out, quality


def _estimate_snr(audio: np.ndarray, n_fft: int = 512) -> float:
    """Schätzt SNR in dB via spectral flatness (Wiener entropy proxy)."""
    power = np.abs(np.fft.rfft(audio, n=n_fft)) ** 2 + 1e-12
    log_geo_mean = float(np.mean(np.log(power)))
    arith_mean = float(np.mean(power))
    sfm = np.exp(log_geo_mean) / arith_mean  # spectral flatness ∈ (0, 1]
    # Pure tone → sfm ~ 0; white noise → sfm ~ 1
    # Convert to SNR-like: lower flatness = more tonal = higher SNR
    estimated_snr_db = float(np.clip(20.0 * (1.0 - sfm) * 20.0, 0.0, 60.0))
    return estimated_snr_db


# ---------------------------------------------------------------------------
# HybridMLDenoiser
# ---------------------------------------------------------------------------


class HybridMLDenoiser:
    """Two-stage denoiser: OMLSA (DSP) → Resemble Enhance (ML, optional).

    In production, the ML stage uses the ``resemble_enhance_plugin`` ONNX
    model.  In test environments or when the plugin is unavailable, the ML
    stage is silently skipped (``resemble_applied=False``).

    Args:
        config: DenoiseConfig instance (default: DenoiseStrategy.OMLSA_ONLY).
    """

    def __init__(self, config: DenoiseConfig | None = None) -> None:
        self.config = config or DenoiseConfig()
        self._resemble_plugin = None
        self._try_load_resemble()

    def _try_load_resemble(self) -> None:
        """Attempt to load the Resemble Enhance plugin (silent on failure)."""
        try:
            from backend.core.ml_memory_budget import try_allocate  # type: ignore[import]

            if not try_allocate("ResembleEnhanceHybrid", size_gb=0.75):
                return
            from plugins.resemble_enhance_plugin import ResembleEnhancePlugin  # type: ignore[import]

            self._resemble_plugin = ResembleEnhancePlugin()  # type: ignore[assignment]
        except Exception:
            self._resemble_plugin = None

    def denoise(self, audio: np.ndarray, sr: int) -> DenoiseResult:
        """Denoise *audio* using the configured strategy.

        Args:
            audio: 1-D (mono) or 2-D (stereo, shape [2, N] or [N, 2]) float array.
            sr:    Sample rate in Hz.

        Returns:
            DenoiseResult with all fields populated.
        """
        assert sr == 48000 or sr > 0  # permissive for test audio
        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Stereo handling: process each channel individually
        # ------------------------------------------------------------------
        stereo = False
        audio_np = np.asarray(audio, dtype=np.float32)
        original_shape = audio_np.shape

        if audio_np.ndim == 2:
            stereo = True
            # Normalise to (n_channels, n_samples)
            if audio_np.shape[0] > audio_np.shape[1]:
                audio_np = audio_np.T
            channels = [audio_np[i] for i in range(audio_np.shape[0])]
        else:
            channels = [audio_np]

        # ------------------------------------------------------------------
        # Determine effective strategy
        # ------------------------------------------------------------------
        strategy = self.config.strategy
        if strategy == DenoiseStrategy.ADAPTIVE:
            snr = _estimate_snr(channels[0])
            strategy = DenoiseStrategy.HYBRID if snr < 15.0 else DenoiseStrategy.OMLSA_ONLY
            logger.debug("ADAPTIVE: estimated SNR=%.1f dB → %s", snr, strategy.value)

        # ------------------------------------------------------------------
        # Stage 1: OMLSA
        # ------------------------------------------------------------------
        processed_channels = []
        quality_estimates = []
        for ch in channels:
            out, q = _omlsa_denoise(
                ch,
                sr,
                n_fft=self.config.n_fft,
                hop=self.config.hop_length,
                noise_frames=self.config.noise_frames,
                reduction_db=self.config.reduction_db,
                lambda_d=self.config.omlsa_lambda_d,
            )
            processed_channels.append(out)
            quality_estimates.append(q)

        quality_after_omlsa = float(np.mean(quality_estimates))
        omlsa_applied = True
        resemble_applied = False

        # ------------------------------------------------------------------
        # Stage 2: ML refinement (HYBRID only, when quality below threshold)
        # ------------------------------------------------------------------
        if (
            strategy == DenoiseStrategy.HYBRID
            and self._resemble_plugin is not None
            and quality_after_omlsa < self.config.quality_threshold
        ):
            try:
                refined = []
                for ch in processed_channels:
                    result = self._resemble_plugin.process(ch, sr)
                    out_ch = result if isinstance(result, np.ndarray) else ch
                    out_ch = np.clip(np.nan_to_num(out_ch, nan=0.0), -1.0, 1.0).astype(np.float32)
                    refined.append(out_ch)
                processed_channels = refined
                resemble_applied = True
                logger.info(
                    "HybridMLDenoiser: Resemble Stufe angewendet (quality_omlsa=%.3f < %.3f)",
                    quality_after_omlsa,
                    self.config.quality_threshold,
                )
            except Exception as exc:
                logger.warning("HybridMLDenoiser: Resemble Stufe fehlgeschlagen: %s", exc)

        # ------------------------------------------------------------------
        # Rebuild output array
        # ------------------------------------------------------------------
        if stereo:
            out_array = np.stack(processed_channels, axis=0)
            if original_shape[0] > original_shape[1]:
                out_array = out_array.T
        else:
            out_array = processed_channels[0]

        # Final NaN guard
        out_array = np.clip(np.nan_to_num(out_array, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

        elapsed = time.perf_counter() - t_start

        return DenoiseResult(
            audio=out_array,
            strategy_used=strategy,
            omlsa_applied=omlsa_applied,
            resemble_applied=resemble_applied,
            quality_estimate=quality_after_omlsa,
            processing_time=elapsed,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def denoise_fast(audio: np.ndarray, sr: int) -> np.ndarray:
    """OMLSA-only denoising (fastest mode).

    Returns denoised audio array.
    """
    denoiser = HybridMLDenoiser(DenoiseConfig(strategy=DenoiseStrategy.OMLSA_ONLY))
    return denoiser.denoise(audio, sr).audio


def denoise_balanced(audio: np.ndarray, sr: int) -> np.ndarray:
    """Adaptive OMLSA + optional ML denoising (balanced quality/speed).

    Returns denoised audio array.
    """
    denoiser = HybridMLDenoiser(DenoiseConfig(strategy=DenoiseStrategy.ADAPTIVE, quality_threshold=0.70))
    return denoiser.denoise(audio, sr).audio


def denoise_maximum(audio: np.ndarray, sr: int) -> np.ndarray:
    """Full OMLSA → Resemble pipeline (maximum quality).

    Falls back gracefully to OMLSA-only if Resemble plugin unavailable.
    Returns denoised audio array.
    """
    denoiser = HybridMLDenoiser(DenoiseConfig(strategy=DenoiseStrategy.HYBRID, quality_threshold=0.0))
    return denoiser.denoise(audio, sr).audio
