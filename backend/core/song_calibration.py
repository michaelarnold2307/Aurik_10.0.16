"""backend/core/song_calibration.py — §v10.700 I5.

SongCalibrationProfile: materialadaptives Kalibrierungsprofil
(global_scalar + family_scalars) vor Phasenkette.
Abgeleitet aus Restorability + Genre + Defekt-Profil.

§03 ROADMAP: spezifiziert, jetzt implementiert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.calibration_context import get_calibration_context

logger = logging.getLogger(__name__)


@dataclass
class SongCalibrationProfile:
    """Materialadaptives Kalibrierungsprofil für einen Song."""

    restorability_score: float = 65.0
    genre: str = "unknown"
    material: str = "vinyl"
    transfer_chain_depth: int = 1

    # Globale Skalierung
    global_scalar: float = 1.0

    # Pro Phase-Familie Skalierung
    family_scalars: dict[str, float] = field(default_factory=dict)

    # Abgeleitete Werte
    confidence: float = 0.5
    recommended_strength: float = 0.35

    def __post_init__(self):
        self._calibrate()

    def _calibrate(self) -> None:
        """Berechnet global_scalar und family_scalars aus den Eingangsdaten."""
        rs = float(np.clip(self.restorability_score, 10.0, 100.0))
        depth = max(1, int(self.transfer_chain_depth))

        # ── Global Scalar ──
        # Höhere Restorability → konservativer (weniger Processing nötig)
        # Tiefere Chain → aggressiver (mehr Defekte erwartet)
        self.global_scalar = float(
            np.clip(
                0.90 - (rs - 50.0) * 0.004 + (depth - 1) * 0.05,
                0.55,
                1.30,
            )
        )

        # ── Family Scalars ──
        self.family_scalars = {
            "click_removal": self._compute_family(rs, depth, 1.0, 0.15),
            "denoise": self._compute_family(rs, depth, 1.1, 0.20),
            "hum_removal": self._compute_family(rs, depth, 0.8, 0.05),
            "eq_correction": self._compute_family(rs, depth, 0.9, 0.10),
            "compression": self._compute_family(rs, depth, 0.7, -0.05),
            "enhancement": self._compute_family(rs, depth, 1.0, 0.10),
            "mastering": self._compute_family(rs, depth, 0.85, 0.05),
            "repair": self._compute_family(rs, depth, 1.2, 0.30),
        }

        # ── Confidence ──
        self.confidence = round(float(np.clip(0.30 + rs * 0.007, 0.20, 0.95)), 2)

        # ── Recommended Strength ──
        if rs >= 90:
            base = 0.20
        elif rs >= 60:
            base = 0.35
        elif rs >= 30:
            base = 0.40
        else:
            base = 0.45
        self.recommended_strength = round(float(np.clip(base * (1.0 + (depth - 1) * 0.08), 0.10, 0.60)), 3)

    @staticmethod
    def _compute_family(rs: float, depth: int, base: float, noise_penalty: float) -> float:
        """Berechnet family_scalar für eine Phase-Familie."""
        rs_factor = float(np.clip(0.80 + (100.0 - rs) * 0.004, 0.80, 1.20))
        depth_factor = float(np.clip(1.0 + (depth - 1) * noise_penalty, 0.60, 1.50))
        return round(base * rs_factor * depth_factor, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "restorability_score": self.restorability_score,
            "genre": self.genre,
            "material": self.material,
            "transfer_chain_depth": self.transfer_chain_depth,
            "global_scalar": self.global_scalar,
            "family_scalars": dict(self.family_scalars),
            "confidence": self.confidence,
            "recommended_strength": self.recommended_strength,
        }


def compute_calibration(
    restorability_score: float,
    material: str = "vinyl",
    transfer_chain_depth: int | None = None,
    genre: str = "unknown",
) -> SongCalibrationProfile:
    """Factory-Funktion für SongCalibrationProfile."""
    if transfer_chain_depth is None:
        _ctx = get_calibration_context()
        transfer_chain_depth = _ctx.transfer_chain_depth if _ctx is not None else 1
    return SongCalibrationProfile(
        restorability_score=float(np.clip(restorability_score, 10.0, 100.0)),
        material=str(material),
        transfer_chain_depth=max(1, int(transfer_chain_depth)),
        genre=str(genre),
    )
