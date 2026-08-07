"""
Material-Adaptive Learner — UCB1 pro Material, backed by SelfLearningOptimizer.

Nicht zwei parallele Lernsysteme, sondern EINS: SelfLearningOptimizer
instanziiert pro Materialtyp. Phase-Level und Stage-Level Lernen
schreiben in dieselbe Instanz.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MATERIALS = [
    "vinyl",
    "shellac",
    "tape",
    "reel_tape",
    "cd_digital",
    "mp3_low",
    "mp3_high",
    "cassette",
    "streaming",
    "dat",
    "minidisc",
    "wire_recording",
    "wax_cylinder",
    "lacquer_disc",
    "unknown",
]


class MaterialAdaptiveLearner:
    """Ein Lernsystem: SelfLearningOptimizer-Instanzen pro Material."""

    def __init__(self, persist_dir: str = "~/.aurik/learning"):
        self._dir = Path(persist_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._optimizers: dict[str, Any] = {}

    def _get_optimizer(self, material: str) -> Any:
        """Lazy-load SelfLearningOptimizer für ein Material."""
        mat = material if material in MATERIALS else "unknown"
        if mat not in self._optimizers:
            try:
                from backend.core.self_learning_optimizer import SelfLearningOptimizer

                self._optimizers[mat] = SelfLearningOptimizer()
                logger.info("MaterialAdaptiveLearner: SLO für %s initialisiert", mat)
            except Exception as e:
                logger.warning("SelfLearningOptimizer nicht verfügbar: %s", e)
                self._optimizers[mat] = None
        return self._optimizers[mat]

    def record(self, material: str, action: str, reward: float):
        """Zeichnet Belohnung auf — delegiert an materialspezifischen SLO."""
        opt = self._get_optimizer(material)
        if opt is not None:
            try:
                opt.update(action, reward)
            except Exception as e:
                logger.warning("MaterialAdaptiveLearner aufzeichnen: %s", e)

    def suggest_strength(self, material: str, default: float = 0.5) -> float:
        """Schlägt optimale Stärke vor basierend auf Lernhistorie."""
        opt = self._get_optimizer(material)
        if opt is not None and hasattr(opt, "get_best_action"):
            try:
                return float(opt.get_best_action().get("strength", default))
            except Exception as e:
                logger.warning("MaterialAdaptiveLearner suggest: %s", e)
        return default

    def get_stats(self, material: str) -> dict:
        opt = self._get_optimizer(material)
        if opt is not None and hasattr(opt, "get_stats"):
            try:
                return opt.get_stats()  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning("MaterialAdaptiveLearner stats: %s", e)
        return {}


_learner: MaterialAdaptiveLearner | None = None
_learner_lock = threading.Lock()


def get_learner() -> MaterialAdaptiveLearner:
    global _learner
    if _learner is None:
        with _learner_lock:
            if _learner is None:
                _learner = MaterialAdaptiveLearner()
    return _learner
