"""Cross-Run-Learning — Aurik lernt von jedem Song.

Speichert pro (Material, Era, Genre) die erfolgreichsten Phasen-Parameter.
Beim nächsten ähnlichen Song werden sie als Prior geladen.
Beschleunigt wiederholte Restaurierungen und verbessert Konsistenz.

Persistenz: ~/.aurik/cross_run_learning.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
STORE_PATH = Path.home() / ".aurik" / "cross_run_learning.json"


class CrossRunLearning:
    """Singleton: Sammlung aller gelernten Parameter aus bisherigen Läufen."""

    _instance: CrossRunLearning | None = None

    def __new__(cls) -> CrossRunLearning:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        self._entries: list[dict] = []
        self._by_material: dict[str, list[dict]] = {}
        self._by_era: dict[int, list[dict]] = {}
        if STORE_PATH.exists():
            try:
                with open(STORE_PATH) as f:
                    data = json.load(f)
                self._entries = data.get("entries", [])
                for e in self._entries:
                    mat = e.get("material", "unknown")
                    era = e.get("era_decade", 2000)
                    self._by_material.setdefault(mat, []).append(e)
                    self._by_era.setdefault(era, []).append(e)
            except Exception:
                self._entries = []

    def _save(self) -> None:
        STORE_PATH.parent.mkdir(exist_ok=True)
        with open(STORE_PATH, "w") as f:
            json.dump({"entries": self._entries[-200:], "updated": time.time()}, f, indent=2)

    def record_run(
        self,
        material: str,
        era: int,
        quality: float,
        phase_strengths: dict[str, float],
        phase_order: list[str],
        rt_factor: float,
        presence_score: float = 0.0,
    ) -> None:
        entry = {
            "material": material,
            "era_decade": era,
            "quality": round(quality, 1),
            "phase_strengths": phase_strengths,
            "phase_order": phase_order,
            "rt_factor": round(rt_factor, 1),
            "presence_score": round(presence_score, 2),
            "timestamp": time.time(),
        }
        self._entries.append(entry)
        self._by_material.setdefault(material, []).append(entry)
        self._by_era.setdefault(era, []).append(entry)
        self._save()

    def get_prior(self, material: str, era: int | None = None) -> dict[str, Any] | None:
        """Besten Prior für gegebenes Material + Era finden."""
        candidates = self._by_material.get(material, [])
        if era is not None:
            candidates = [c for c in candidates if abs(c.get("era_decade", 0) - era) <= 20]
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.get("quality", 0))
        return {
            "phase_strengths": best["phase_strengths"],
            "phase_order": best["phase_order"],
            "prior_quality": best["quality"],
            "based_on_n_runs": len(candidates),
        }

    def get_material_stats(self, material: str) -> dict[str, Any]:
        entries = self._by_material.get(material, [])
        if not entries:
            return {}
        qualities = [e["quality"] for e in entries]
        rts = [e["rt_factor"] for e in entries]
        return {
            "n_runs": len(entries),
            "avg_quality": round(float(np.mean(qualities)), 1),
            "best_quality": round(float(np.max(qualities)), 1),
            "avg_rt": round(float(np.mean(rts)), 1),
            "trend": "improving" if len(qualities) >= 3 and qualities[-1] > qualities[0] else "stable",
        }
