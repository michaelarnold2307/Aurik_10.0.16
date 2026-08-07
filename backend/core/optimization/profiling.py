"""
optimization/profiling.py – Performance-Profiler und Qualitätsvalidator.
======================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profiles the runtime of each stage in a BalancedAudioProcessor pipeline.

    Parameters
    ----------
    processor:
        A ``BalancedAudioProcessor`` (or any object with ``.sr``, ``.efficiency``,
        ``.vocal_enhancer`` attributes and a ``.process()`` method).
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def profile_pipeline(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Führt aus: the processor and record per-component timings.

        Returns
        -------
        dict with keys ``"total_rt"`` (real-time factor) and
        ``"components"`` (list of ``{"name": str, "rt_factor": float}``).
        """
        audio_f32 = np.asarray(audio, dtype=np.float32)
        duration_s = len(audio_f32) / sr

        components: list[dict[str, Any]] = []
        errors: list[str] = []

        # --- efficiency stage ---
        efficiency = getattr(self.processor, "efficiency", None)
        if efficiency is not None:
            t0 = time.perf_counter()
            try:
                efficiency.process(audio_f32, sr, use_multicore=False)
            except Exception as exc:
                errors.append(f"efficiency:{exc}")
                logger.debug("PerformanceProfiler: efficiency Stufe fehlgeschlagen: %s", exc)
            dt = time.perf_counter() - t0
            components.append({"name": "efficiency", "rt_factor": dt / (duration_s + 1e-12)})

        # --- vocal enhancer stage ---
        vocal_enhancer = getattr(self.processor, "vocal_enhancer", None)
        if vocal_enhancer is not None:
            t0 = time.perf_counter()
            try:
                vocal_enhancer.process(audio_f32, sr)
            except Exception as exc:
                errors.append(f"vocal_enhancer:{exc}")
                logger.debug("PerformanceProfiler: vocal_enhancer Stufe fehlgeschlagen: %s", exc)
            dt = time.perf_counter() - t0
            components.append({"name": "vocal_enhancer", "rt_factor": dt / (duration_s + 1e-12)})

        # --- full pipeline ---
        t0 = time.perf_counter()
        try:
            self.processor.process(audio_f32, sr)
        except Exception as exc:
            errors.append(f"pipeline:{exc}")
            logger.debug("PerformanceProfiler: full pipeline fehlgeschlagen: %s", exc)
        total_time = time.perf_counter() - t0
        total_rt = total_time / (duration_s + 1e-12)

        return {"total_rt": float(total_rt), "components": components, "errors": errors}


class QualityValidator:
    """Misst subjective quality improvement after optimization.

    Parameters
    ----------
    model:
        Optional quality-scoring model (may be ``None`` — DSP fallback used).
    processor:
        ``BalancedAudioProcessor`` instance.
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    def validate_optimization(
        self,
        audio: np.ndarray,
        sr: int,
        reference: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Berechnet quality scores before and after processing.

        Returns
        -------
        dict with ``"optimized_quality"`` and ``"improvement"`` (both floats).
        """
        audio_f32 = np.asarray(audio, dtype=np.float32)

        # Baseline quality: simple crest-factor proxy
        rms_in = float(np.sqrt(np.mean(audio_f32**2))) + 1e-12
        peak_in = float(np.max(np.abs(audio_f32))) + 1e-12
        baseline = float(np.clip(rms_in / peak_in, 0.0, 1.0))

        try:
            processed = self.processor.process(audio_f32, sr)
        except Exception as exc:
            logger.debug("QualityValidator: processor fehlgeschlagen, falling back to passthrough: %s", exc)
            processed = audio_f32

        processed = np.asarray(processed, dtype=np.float32)
        rms_out = float(np.sqrt(np.mean(processed**2))) + 1e-12
        peak_out = float(np.max(np.abs(processed))) + 1e-12
        optimized_quality = float(np.clip(rms_out / peak_out, 0.0, 1.0))

        improvement = float(optimized_quality - baseline)

        return {
            "optimized_quality": optimized_quality,
            "improvement": improvement,
        }
