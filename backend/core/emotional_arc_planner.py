"""§ArcPlan EmotionalArcPlanner — Dramaturgie-getriebene Schutz-Gewicht-Planung (v10.0.0).

Computes a time-indexed protection-weight vector BEFORE phase execution.
Unlike EmotionalArcPreservationMetric (which measures arc AFTER the pipeline),
the ArcPlanner plans phase strengths PROACTIVELY based on VFAResult zones.

Protection weights (higher = more conservative / lower NR strength allowed):
  frisson_zones:   1.5  — maximal protection (emotional climax / goosebumps)
  whisper_zones:   1.4  — very fragile passages, high musical information
  tension_zones:   1.3  — emotional build-up, critical phrasing
  passaggio_zones: 1.2  — register transitions, acoustically delicate
  release_zones:   1.1  — resolution passages
  normal voiced:   1.0
  silent/instr:    0.7

Weights are Gaussian-smoothed (σ = 1.5 s) to avoid hard strength transitions.

Usage in UV3 (after VocalFocusAnalyzer):
    from backend.core.emotional_arc_planner import get_emotional_arc_planner
    _arc_plan = get_emotional_arc_planner().plan(audio, sr, _restoration_context)
    _restoration_context["arc_protection_weights"] = _arc_plan

Phases read arc weight via:
    _arc = kwargs.get("_restoration_context", {}).get("arc_protection_weights")
    if _arc:
        _weight = _arc.weight_at(segment_start_s, segment_end_s)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_RESOLUTION_S: float = 0.1  # 100 ms weight grid
_SMOOTH_SIGMA_S: float = 1.5  # Gaussian σ for weight smoothing
_RMS_SILENCE_THRESHOLD: float = 1e-3  # ~-60 dBFS — treated as silent


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ArcPlan:
    """Time-indexed protection weight array."""

    weights: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=np.float32))
    resolution_s: float = _RESOLUTION_S
    duration_s: float = 0.0

    def weight_at(self, start_s: float, end_s: float) -> float:
        """Mean protection weight for the time range [start_s, end_s].

        Returns 1.0 if the plan is empty or indices are out of range.
        """
        if len(self.weights) == 0:
            return 1.0
        res = max(self.resolution_s, 1e-6)
        i_start = int(start_s / res)
        i_end = int(end_s / res) + 1
        i_start = max(0, min(i_start, len(self.weights) - 1))
        i_end = max(i_start + 1, min(i_end, len(self.weights)))
        return float(np.mean(self.weights[i_start:i_end]))

    def to_dict(self) -> dict:
        """Serialisiert the arc plan summary as a dictionary for UV3 metadata."""
        return {
            "duration_s": self.duration_s,
            "resolution_s": self.resolution_s,
            "n_frames": int(len(self.weights)),
            "mean_weight": float(np.mean(self.weights)),
            "max_weight": float(np.max(self.weights)),
            "min_weight": float(np.min(self.weights)),
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class EmotionalArcPlanner:
    """Singleton-Planer – thread-safe, nicht-blockierend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def plan(
        self,
        audio: np.ndarray,
        sr: int,
        restoration_context: dict | None = None,
    ) -> ArcPlan:
        """Berechnet an ArcPlan from audio and restoration context.

        Args:
            audio:               Input audio (mono or stereo, any SR).
            sr:                  Sample rate (analysis module — no assert sr==48000).
            restoration_context: Dict with zone lists from VocalFocusAnalyzer.

        Returns:
            ArcPlan with protection weights; on error returns uniform weight 1.0.
        """
        with self._lock:
            try:
                return self._plan_impl(audio, sr, dict(restoration_context or {}))
            except Exception as exc:
                logger.debug("EmotionalArcPlanner nicht blockierend: %s", exc)
                n = max(1, int(len(audio) / max(sr, 1) / _RESOLUTION_S))
                dur = len(audio) / max(sr, 1)
                return ArcPlan(weights=np.ones(n, dtype=np.float32), duration_s=dur)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _plan_impl(self, audio: np.ndarray, sr: int, ctx: dict) -> ArcPlan:
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim == 2:
            if mono.shape[0] <= 2 and mono.shape[0] < mono.shape[1]:
                mono = np.mean(mono, axis=0)
            else:
                mono = np.mean(mono, axis=1)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        duration_s = len(mono) / max(sr, 1)
        n_frames = max(1, int(np.ceil(duration_s / _RESOLUTION_S)))
        weights = np.ones(n_frames, dtype=np.float32)

        # Mark silent frames with lower weight
        hop = max(1, int(_RESOLUTION_S * sr))
        for i in range(n_frames):
            start = i * hop
            end = min(start + hop, len(mono))
            if end <= start:
                continue
            rms = float(np.sqrt(np.mean(mono[start:end] ** 2)))
            if rms < _RMS_SILENCE_THRESHOLD:
                weights[i] = 0.7

        # Apply zone weights (higher priority zones overwrite lower ones via maximum)
        # Apply in ascending priority order so higher-priority zones win
        for zone_key, zone_weight in [
            ("release_zones", 1.1),
            ("passaggio_zones", 1.2),
            ("tension_zones", 1.3),
            ("whisper_zones", 1.4),
            ("frisson_zones", 1.5),
        ]:
            self._apply_zone_weight(weights, n_frames, ctx.get(zone_key, []), zone_weight)

        # Gaussian smoothing — avoids abrupt strength transitions at zone boundaries
        sigma_frames = max(1, int(_SMOOTH_SIGMA_S / _RESOLUTION_S))
        if sigma_frames > 1 and len(weights) > sigma_frames * 2:
            try:
                from scipy.ndimage import gaussian_filter1d  # pylint: disable=import-outside-toplevel

                weights = gaussian_filter1d(weights.astype(np.float64), sigma=sigma_frames).astype(np.float32)
            except Exception as e:
                logger.warning("emotional_arc_planner.py::_plan_impl Ersatzpfad: %s", e)
                pass  # smoothing is optional

        weights = np.clip(weights, 0.5, 1.5).astype(np.float32)

        logger.debug(
            "EmotionalArcPlanner: dur=%.1fs n=%d mean_w=%.2f max_w=%.2f",
            duration_s,
            n_frames,
            float(np.mean(weights)),
            float(np.max(weights)),
        )
        return ArcPlan(weights=weights, resolution_s=_RESOLUTION_S, duration_s=duration_s)

    @staticmethod
    def _apply_zone_weight(
        weights: np.ndarray,
        n_frames: int,
        zones: Sequence,
        zone_weight: float,
    ) -> None:
        """Setzt weight to max(current, zone_weight) for all frames inside zones."""
        for zone in zones or []:
            try:
                if isinstance(zone, (list, tuple)) and len(zone) >= 2:
                    t_start, t_end = float(zone[0]), float(zone[1])
                elif isinstance(zone, dict):
                    t_start = float(zone.get("start", zone.get("t_start", 0.0)))  # type: ignore[arg-type]
                    t_end = float(zone.get("end", zone.get("t_end", 0.0)))  # type: ignore[arg-type]
                else:
                    continue
                i_s = max(0, int(t_start / _RESOLUTION_S))
                i_e = min(n_frames, int(t_end / _RESOLUTION_S) + 1)
                if i_e > i_s:
                    weights[i_s:i_e] = np.maximum(weights[i_s:i_e], zone_weight)
            except Exception as e:
                logger.warning("emotional_arc_planner.py::_anwenden_zone_weight Ersatzpfad: %s", e)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_planner_holder: list[EmotionalArcPlanner | None] = [None]
_planner_lock = threading.Lock()


def get_emotional_arc_planner() -> EmotionalArcPlanner:
    """Thread-safe singleton factory."""
    if _planner_holder[0] is None:
        with _planner_lock:
            if _planner_holder[0] is None:
                _planner_holder[0] = EmotionalArcPlanner()
    instance = _planner_holder[0]
    assert instance is not None
    return instance
