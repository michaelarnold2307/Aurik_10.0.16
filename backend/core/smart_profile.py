"""SmartProfile — §INCREMENTAL #1: Strategie-Export/Import.

Speichert die erfolgreichsten Strategien aus dem PhaseImpactRecorder
als JSON-Profil und lädt sie für ähnliche Songs wieder.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROFILE_DIR = Path(__file__).parent.parent.parent / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SmartProfile:
    """Gespeicherte Strategie für eine Material-Kombination."""

    material: str = ""
    era: int = 0
    genre: str = ""
    mode: str = "restoration"
    best_strategies: list[str] = field(default_factory=list)
    avg_improvement: float = 0.0
    n_samples: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "material": self.material,
            "era": self.era,
            "genre": self.genre,
            "mode": self.mode,
            "best_strategies": self.best_strategies,
            "avg_improvement": self.avg_improvement,
            "n_samples": self.n_samples,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SmartProfile:
        return cls(
            **{k: d.get(k, "") for k in ["material", "genre", "mode"]},
            era=d.get("era", 0),
            best_strategies=d.get("best_strategies", []),
            avg_improvement=d.get("avg_improvement", 0.0),
            n_samples=d.get("n_samples", 0),
            confidence=d.get("confidence", 0.0),
        )


def build_profile(material: str, era: int = 0, genre: str = "", mode: str = "restoration") -> SmartProfile:
    """Extrahiert bestes Profil aus PhaseImpactRecorder-Daten."""
    try:
        from backend.core.phase_impact_predictor import get_phase_impact_predictor
        from backend.core.phase_impact_recorder import get_phase_impact_recorder

        rec = get_phase_impact_recorder()
        pred = get_phase_impact_predictor()

        # Query all strategies for this material profile
        profiles_5 = ["passthrough", "light", "balanced", "deep", "full"]
        deltas = {}
        for strategy in profiles_5:
            result = pred.predict(material=material, era=era, phase_id=strategy, mode=mode)
            deltas[strategy] = result.predicted_delta

        # Sortiere nach Delta (beste zuerst)
        best = sorted(deltas.items(), key=lambda x: -x[1])
        best_strategies = [s for s, d in best if d > 0][:3]

        profile = SmartProfile(
            material=material,
            era=era,
            genre=genre,
            mode=mode,
            best_strategies=best_strategies,
            avg_improvement=sum(d for _, d in best) / max(len(best), 1),
            n_samples=rec._session_impacts.__len__() if hasattr(rec, "_session_impacts") else 0,
            confidence=min(1.0, len(best_strategies) / 3.0),
        )
        return profile
    except Exception as e:
        logger.debug("build_Profil fehlgeschlagen: %s", e)
        return SmartProfile(material=material, era=era, genre=genre)


def save_profile(profile: SmartProfile, path: str | None = None) -> str:
    """Speichert Profil als JSON."""
    if path is None:
        key = f"{profile.material}_{profile.era}_{profile.mode}"
        path = str(PROFILE_DIR / f"{key}.json")
    with open(path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)
    return path


def load_profile(material: str, era: int = 0, mode: str = "restoration") -> SmartProfile | None:
    """Lädt ein gespeichertes Profil."""
    key = f"{material}_{era}_{mode}"
    path = PROFILE_DIR / f"{key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return SmartProfile.from_dict(json.load(f))
