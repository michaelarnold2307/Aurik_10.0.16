"""§v10.303.19 Phase-0 Goal-Recalibration-Cache.

Vermeidet wiederholte measure_all()-Aufrufe (je ~8s) für dasselbe Material.
Wird von unified_restorer_v3._execute_pipeline() nach Phase 0 aufgerufen.

API:
    from plugins.phase0_goal_cache import get_goal_cache

    cache = get_goal_cache()
    baseline = cache.get_or_measure(mat_key, audio, sr, measure_fn)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_instance: GoalRecalibrationCache | None = None
_lock = threading.Lock()


class GoalRecalibrationCache:
    """Thread-sicherer Singleton-Cache für Goal-Recalibration-Ergebnisse."""

    def __init__(self):
        self._cache: dict[str, dict[str, float]] = {}

    def get(self, material_key: str) -> dict[str, float] | None:
        return self._cache.get(material_key)

    def put(self, material_key: str, baseline: dict[str, float]) -> None:
        self._cache[material_key] = dict(baseline)
        logger.debug("Goal-Cache stored: %s (%d goals)", material_key, len(baseline))

    def get_or_measure(
        self,
        material_key: str,
        audio: Any,
        sr: int,
        measure_fn: Any,
    ) -> dict[str, float]:
        """Holt Goal-Baseline aus Cache oder misst neu.

        Args:
            material_key: Material-String (z.B. "mp3_high")
            audio: np.ndarray
            sr: Sample-Rate
            measure_fn: measure_all-Funktion

        Returns:
            Dict[str, float] mit Goal-Scores.
        """
        _cached = self._cache.get(material_key)
        if _cached is not None:
            logger.debug("§v10.303.19 Goal-Cache HIT: %s", material_key)
            return dict(_cached)

        logger.debug("§v10.303.19 Goal-Cache MISS: measuring %s...", material_key)
        _goals = measure_fn(audio, sr, material_key=material_key, mode="restoration")
        _baseline = {}
        for _g in _goals:
            _baseline[_g.name] = _g.score
        self._cache[material_key] = _baseline
        return _baseline

    def clear(self) -> None:
        self._cache.clear()


def get_goal_cache() -> GoalRecalibrationCache:
    """Thread-sicherer Singleton."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GoalRecalibrationCache()
    return _instance
