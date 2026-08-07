"""
PhraseBoundaryGuard — §Gap3 Phrase-Boundary Aware DSP (Aurik 10.0.0.x)
======================================================================

Detects phrase boundaries in vocal/music material and provides a
strength-modulation helper so DSP phases can taper their processing
near phrase boundaries (preventing audible clicks and processing
artifacts across musical phrases).

Phrase boundary = onset of silence/breath between phrases
Algorithm:
  1. Compute RMS energy envelope (frame 512, hop 256)
  2. Low-pass smooth envelope (window 20 frames)
  3. Find local minima below `boundary_threshold_dbfs` = −30 dBFS
  4. Merge nearby minima (< 200 ms apart) → boundary sample indices

DSP helper `apply_phrase_boundary_taper()`:
  - Fades processing strength to 0 within `taper_ms` of each boundary
  - Returns a per-sample modulation envelope [0, 1]

Usage in phases:
    from backend.core.dsp.phrase_boundary_guard import (
        detect_phrase_boundaries, apply_phrase_boundary_taper
    )
    boundaries = detect_phrase_boundaries(audio, sr)
    mod_env = apply_phrase_boundary_taper(audio, boundaries, sr, taper_ms=20)
    # DSP output: output = dry + (wet - dry) * mod_env[:, np.newaxis]

All functions are non-blocking (return safe defaults on error).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRAME_N: int = 512
_HOP_N: int = 256
_SMOOTH_FRAMES: int = 20
_BOUNDARY_THRESHOLD_DBFS: float = -30.0
_MIN_BOUNDARY_GAP_S: float = 0.20  # merge boundaries closer than this


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_phrase_boundaries(audio: np.ndarray, sr: int) -> list[int]:
    """Erkennt phrase boundaries as sample indices.

    Args:
        audio: Input audio (mono or stereo, any length).
        sr:    Sample rate of ``audio``.

    Returns:
        Sorted list of sample-index positions where phrase boundaries occur.
        Empty list on failure (non-blocking).
    """
    try:
        mono = _to_mono(audio)
        n = len(mono)
        if n < _FRAME_N * 4:
            return []

        # RMS envelope
        n_frames = (n - _FRAME_N) // _HOP_N + 1
        rms_db = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            seg = mono[i * _HOP_N : i * _HOP_N + _FRAME_N]
            rms = float(np.sqrt(np.mean(seg**2) + 1e-10))
            rms_db[i] = float(20.0 * np.log10(rms + 1e-10))

        # Smooth envelope
        kernel = np.ones(_SMOOTH_FRAMES, dtype=np.float32) / _SMOOTH_FRAMES
        rms_smooth = np.convolve(rms_db, kernel, mode="same")

        # Find local minima below threshold
        _thr = _BOUNDARY_THRESHOLD_DBFS
        _min_gap_frames = max(1, int(_MIN_BOUNDARY_GAP_S * sr / _HOP_N))
        candidates: list[int] = []
        for i in range(1, n_frames - 1):
            if rms_smooth[i] < _thr and rms_smooth[i] <= rms_smooth[i - 1] and rms_smooth[i] <= rms_smooth[i + 1]:
                candidates.append(i)

        if not candidates:
            return []

        # Merge nearby candidates
        merged_frames: list[int] = [candidates[0]]
        for c in candidates[1:]:
            if c - merged_frames[-1] < _min_gap_frames:
                # Keep the quieter one
                if rms_smooth[c] < rms_smooth[merged_frames[-1]]:
                    merged_frames[-1] = c
            else:
                merged_frames.append(c)

        # Convert to sample indices (center of frame)
        boundaries = [int(f * _HOP_N + _FRAME_N // 2) for f in merged_frames]
        logger.debug("PhraseBoundaryGuard: erkannt %d boundaries", len(boundaries))
        return boundaries

    except Exception as exc:
        logger.debug("erkennen_phrase_boundaries nicht blockierend: %s", exc)
        return []


def apply_phrase_boundary_taper(
    audio: np.ndarray,
    boundaries: list[int],
    sr: int,
    taper_ms: float = 20.0,
) -> np.ndarray:
    """Erstellt a per-sample modulation envelope [0, 1] that fades to 0 at boundaries.

    Use this envelope to blend dry/wet:
        output = dry + (wet - dry) * mod_env[:, np.newaxis]  # stereo
        output = dry + (wet - dry) * mod_env                 # mono

    Args:
        audio:      Input audio (used only for length).
        boundaries: Sample indices from ``detect_phrase_boundaries``.
        sr:         Sample rate.
        taper_ms:   Fade-out duration in ms around each boundary.

    Returns:
        1-D ndarray of shape (n_samples,), dtype float32, values in [0, 1].
        All-ones if boundaries is empty (no-op).
    """
    try:
        n = audio.shape[-1] if audio.ndim > 1 else len(audio)
        if not boundaries:
            return np.ones(n, dtype=np.float32)  # type: ignore[no-any-return]

        taper_n = int(taper_ms * sr / 1000.0)
        taper_n = max(taper_n, 1)
        env = np.ones(n, dtype=np.float32)

        fade_out = np.linspace(1.0, 0.0, taper_n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, taper_n, dtype=np.float32)

        for b in boundaries:
            # Fade out: [b - taper_n, b]
            pre_start = max(0, b - taper_n)
            pre_end = min(n, b)
            fo_slice = fade_out[max(0, taper_n - (b - pre_start)) :]
            l = pre_end - pre_start
            env[pre_start:pre_end] = np.minimum(env[pre_start:pre_end], fo_slice[:l])

            # Fade in: [b, b + taper_n]
            post_start = max(0, b)
            post_end = min(n, b + taper_n)
            fi_slice = fade_in[: post_end - post_start]
            l = post_end - post_start
            env[post_start:post_end] = np.minimum(env[post_start:post_end], fi_slice[:l])

        return np.nan_to_num(env, nan=1.0, posinf=1.0, neginf=0.0)  # type: ignore[no-any-return]

    except Exception as exc:
        logger.debug("anwenden_phrase_boundary_taper nicht blockierend: %s", exc)
        n = audio.shape[-1] if audio.ndim > 1 else len(audio)
        return np.ones(n, dtype=np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)  # type: ignore[no-any-return]
    if audio.ndim == 2:
        if audio.shape[0] == 2 and audio.shape[1] > 2:
            return audio.mean(axis=0).astype(np.float32)  # type: ignore[no-any-return]
        return audio.mean(axis=-1).astype(np.float32)  # type: ignore[no-any-return]
    return audio.flatten().astype(np.float32)  # type: ignore[no-any-return]
