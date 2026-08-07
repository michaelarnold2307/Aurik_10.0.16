"""
PhaseImpactRecorder — §CROWN Self-Supervised Learning
======================================================

Misst nach JEDER Phase das Quality-Delta und speichert es.
Jeder Song macht Aurik besser.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "phase_impacts")


@dataclass
class PhaseImpact:
    phase_id: str = ""
    material: str = "unknown"
    era: int = 0
    strength: float = 1.0
    mode: str = "restoration"
    quality_delta: float = 0.0
    delta_norm: float = 0.0
    timestamp: float = field(default_factory=time.time)


class PhaseImpactRecorder:
    def __init__(self):
        self._session_impacts: list[PhaseImpact] = []
        self._session_id = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(DATA_DIR, exist_ok=True)

    def record(self, material="unknown", era=0, phase_id="", quality_delta=0.0, mode="restoration", strength=1.0):
        impact = PhaseImpact(
            phase_id=phase_id, material=material, era=era, strength=strength, mode=mode, quality_delta=quality_delta
        )
        self._session_impacts.append(impact)
        if abs(quality_delta) > 0.05:
            logger.debug("PhaseImpact: %s | %s/%d → Δ=%.3f", phase_id, material, era, quality_delta)
        return impact

    def query(self, material="", era=0, phase_id="", mode=""):
        results = []
        for impact in self._session_impacts:
            if material and impact.material != material:
                continue
            if era and impact.era != era:
                continue
            if phase_id and impact.phase_id != phase_id:
                continue
            if mode and impact.mode != mode:
                continue
            results.append(
                {
                    "material": impact.material,
                    "era": impact.era,
                    "phase_id": impact.phase_id,
                    "mode": impact.mode,
                    "delta": impact.quality_delta,
                }
            )
        return results

    def flush(self):
        if not self._session_impacts:
            return
        try:
            path = os.path.join(DATA_DIR, f"impacts_{self._session_id}.json")
            data = [
                {"material": i.material, "era": i.era, "phase_id": i.phase_id, "delta": i.quality_delta, "mode": i.mode}
                for i in self._session_impacts
            ]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("PhaseImpactRecorder: %d impacts gespeichert to %s", len(data), path)
        except Exception as e:
            logger.debug("PhaseImpactRecorder flush fehlgeschlagen: %s", e)


_recorder = None


def get_phase_impact_recorder():
    global _recorder
    if _recorder is None:
        _recorder = PhaseImpactRecorder()
    return _recorder
