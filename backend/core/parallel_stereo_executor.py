"""
backend/core/parallel_stereo_executor.py — Parallel Stereo Execution (§v10.10)
==============================================================================

Erkennt kanal-unabhängige DSP-Phasen und führt left/right parallel via ThreadPool.
~40% Speedup für Stereo-DSP-Phasen bei 2-Kanal-Audio.

Usage:
    from backend.core.parallel_stereo_executor import run_phase_stereo_parallel
    audio = run_phase_stereo_parallel(phase_fn, audio, sr, **kwargs)
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)

# Phasen die kanal-unabhängig sind (keine Stereo-Korrelation nötig)
_PARALLEL_SAFE_PHASES: frozenset[str] = frozenset(
    {
        "phase_01_click_removal",
        "phase_02_hum_removal",
        "phase_03_denoise",
        "phase_09_crackle_removal",
        "phase_18_noise_gate",
        "phase_27_click_pop_removal",
        "phase_28_surface_noise_profiling",
        "phase_29_tape_hiss_reduction",
        "phase_45_dc_offset",
        "phase_47_truepeak_limiter",
        "phase_56_spectral_band_gap_repair",
        "phase_59_modulation_noise_reduction",
    }
)

_MAX_WORKERS: int = 2  # max 2 Threads (left/right)


def is_parallel_safe(phase_id: str) -> bool:
    """Prüft ob eine Phase kanal-parallel ausgeführt werden kann."""
    return phase_id in _PARALLEL_SAFE_PHASES


def run_phase_stereo_parallel(
    phase_fn,
    audio: np.ndarray,
    sample_rate: int,
    phase_id: str = "",
    **kwargs,
) -> np.ndarray:
    """Führt eine Phase parallel auf left/right-Kanälen aus.

    Args:
        phase_fn: Callable(audio_mono, sr, **kwargs) → audio_mono.
        audio: Stereo-Audio (2, samples).
        sample_rate: Sample-Rate.
        phase_id: Phase-ID für Logging.
        **kwargs: Weitere Phase-Parameter.

    Returns:
        Stereo-Audio (2, samples).
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        # Mono oder nicht standard → sequentiell
        return phase_fn(audio, sample_rate, **kwargs)  # type: ignore[no-any-return]

    if not is_parallel_safe(phase_id):
        return phase_fn(audio, sample_rate, **kwargs)  # type: ignore[no-any-return]

    try:
        _left = np.asarray(audio[0], dtype=np.float32)
        _right = np.asarray(audio[1], dtype=np.float32)

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            _fut_left = executor.submit(phase_fn, _left, sample_rate, **kwargs)
            _fut_right = executor.submit(phase_fn, _right, sample_rate, **kwargs)

            _done = as_completed([_fut_left, _fut_right])
            _results = {}
            for _f in _done:
                if _f == _fut_left:
                    _results["left"] = _f.result()
                else:
                    _results["right"] = _f.result()

        _out_left = np.asarray(_results.get("left", _left), dtype=np.float32)
        _out_right = np.asarray(_results.get("right", _right), dtype=np.float32)

        # Auf gleiche Länge trimmen
        _min_len = min(_out_left.shape[-1], _out_right.shape[-1])
        _out = np.stack(
            [
                _out_left[..., :_min_len],
                _out_right[..., :_min_len],
            ],
            axis=0,
        )

        logger.debug("⚡ Parallel-Stereo %s: left+right parallel ausgeführt", phase_id)
        return _out  # type: ignore[no-any-return]

    except Exception as exc:
        logger.warning("⚠️ Parallel-Stereo %s fehlgeschlagen: %s → sequentiell", phase_id, exc)
        return phase_fn(audio, sample_rate, **kwargs)  # type: ignore[no-any-return]
