"""§Gap6 ReferenceAnchorMatcher — Perceptual Reference Anchor Modul (v10.0.0).

Kalibriert Musical-Goal-Schwellwerte gegen era- und genre-spezifische Referenzprofile.
Verhindert, dass Aurik eine Schlager-MP3-Restaurierung gegen CD-Maßstäbe misst.

Ohne echte Referenzaudio-Datenbank: Verwendet DSP-basierte Schätzprofile aus
`backend/core/studio_goal_targets.py` (dieselbe Datenbasis wie der PMGG-Schwellwert-Kalkulator).
Singleton, thread-safe, vollständig offline.

Verwendung in UV3 (nach EraClassifier, vor GoalApplicabilityFilter):
    from backend.core.reference_anchor_matcher import get_reference_anchor_matcher
    _anchor = get_reference_anchor_matcher().match(
        era_decade=_era_decade,
        genre_label=_genre_label,
        material_type=_material_type,
    )
    if _anchor.valid:
        _restoration_context["reference_anchor"] = _anchor
        metadata["reference_anchor"] = _anchor.to_dict()

Phasen / PMGG können dann anchor.naturalness_floor statt fixem 0.90 nutzen.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Era-Genre Profil-Tabelle (DSP-basiert, kein ML)
# ---------------------------------------------------------------------------

# Struktur: (era_decade_start, era_decade_end) → Profil-Basis
# Werte: rel. Abschlag gegenüber CD-Baseline (1.0 = CD-Qualität)
_ERA_QUALITY_PROFILE: dict[tuple[int, int], dict[str, float]] = {
    # Akustische Ära — Trichteraufnahme
    (1900, 1924): {
        "naturalness_floor": 0.55,
        "tonal_center_floor": 0.70,
        "timbre_floor": 0.60,
        "spatial_depth_floor": 0.40,
        "brilliance_floor": 0.45,
        "warmth_floor": 0.55,
    },
    # Frühe elektrische Aufnahme
    (1925, 1944): {
        "naturalness_floor": 0.62,
        "tonal_center_floor": 0.75,
        "timbre_floor": 0.65,
        "spatial_depth_floor": 0.45,
        "brilliance_floor": 0.55,
        "warmth_floor": 0.62,
    },
    # Mono LP-Ära
    (1945, 1959): {
        "naturalness_floor": 0.70,
        "tonal_center_floor": 0.80,
        "timbre_floor": 0.72,
        "spatial_depth_floor": 0.52,
        "brilliance_floor": 0.65,
        "warmth_floor": 0.70,
    },
    # Stereo Analog-Ära
    (1960, 1979): {
        "naturalness_floor": 0.76,
        "tonal_center_floor": 0.85,
        "timbre_floor": 0.78,
        "spatial_depth_floor": 0.62,
        "brilliance_floor": 0.72,
        "warmth_floor": 0.76,
    },
    # Frühe Digital-Ära / Cassette
    (1980, 1994): {
        "naturalness_floor": 0.82,
        "tonal_center_floor": 0.88,
        "timbre_floor": 0.83,
        "spatial_depth_floor": 0.68,
        "brilliance_floor": 0.78,
        "warmth_floor": 0.73,
    },
    # CD-Mainstream
    (1995, 2009): {
        "naturalness_floor": 0.88,
        "tonal_center_floor": 0.92,
        "timbre_floor": 0.88,
        "spatial_depth_floor": 0.74,
        "brilliance_floor": 0.82,
        "warmth_floor": 0.76,
    },
    # Modern Streaming / FLAC
    (2010, 2030): {
        "naturalness_floor": 0.90,
        "tonal_center_floor": 0.95,
        "timbre_floor": 0.90,
        "spatial_depth_floor": 0.78,
        "brilliance_floor": 0.85,
        "warmth_floor": 0.78,
    },
}

# Schlechte Codec-Qualität (MP3 < 192 kbps) dämpft Floors
_CODEC_QUALITY_PENALTY: dict[str, float] = {
    "mp3_low": 0.88,  # 128 kbps
    "mp3_mid": 0.93,  # 192 kbps
    "mp3_high": 0.97,  # 320 kbps
    "mp3": 0.93,
    "aac": 0.95,
    "ogg": 0.95,
    "shellac": 0.82,
    "vinyl": 0.92,
    "vinyl_lp": 0.92,
    "tape": 0.90,
    "reel_tape": 0.90,
    "cassette": 0.87,
    "wax_cylinder": 0.72,
    "wire_recording": 0.78,
    "cd": 1.00,
    "flac": 1.00,
    "wav": 1.00,
    "unknown": 0.93,
}

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ReferenceAnchor:
    """Perceptueller Referenz-Anker für eine Era/Genre/Material-Kombination."""

    era_decade: int = 2000
    genre_label: str = "unknown"
    material_type: str = "unknown"
    naturalness_floor: float = 0.90
    tonal_center_floor: float = 0.95
    timbre_floor: float = 0.90
    spatial_depth_floor: float = 0.70
    brilliance_floor: float = 0.78
    warmth_floor: float = 0.75
    valid: bool = False
    profile_source: str = "era_dsp_table"
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert den Anker für UV3-Metadata."""
        return {
            "era_decade": self.era_decade,
            "genre_label": self.genre_label,
            "material_type": self.material_type,
            "naturalness_floor": round(self.naturalness_floor, 3),
            "tonal_center_floor": round(self.tonal_center_floor, 3),
            "timbre_floor": round(self.timbre_floor, 3),
            "spatial_depth_floor": round(self.spatial_depth_floor, 3),
            "brilliance_floor": round(self.brilliance_floor, 3),
            "warmth_floor": round(self.warmth_floor, 3),
            "valid": self.valid,
            "profile_source": self.profile_source,
            **{f"extra_{k}": round(v, 3) for k, v in self.extra.items()},
        }

    def goal_floor(self, goal_name: str) -> float | None:
        """Gibt den materialadaptiven Ziel-Floor zurück oder None (→ Standard-Schwellwert nutzen).

        Beispiel:
            floor = anchor.goal_floor("naturalness")  # z.B. 0.72 für Shellac 1935
        """
        _mapping = {
            "naturalness": self.naturalness_floor,
            "natuerlichkeit": self.naturalness_floor,
            "tonal_center": self.tonal_center_floor,
            "tonalcenter": self.tonal_center_floor,
            "timbre": self.timbre_floor,
            "spatial_depth": self.spatial_depth_floor,
            "raumtiefe": self.spatial_depth_floor,
            "brilliance": self.brilliance_floor,
            "brillanz": self.brilliance_floor,
            "warmth": self.warmth_floor,
            "waerme": self.warmth_floor,
        }
        return _mapping.get(str(goal_name or "").lower().strip())


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


class ReferenceAnchorMatcher:
    """Singleton — era/genre/material-basierte Perceptual-Reference-Kalibrierung."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def match(
        self,
        era_decade: int = 2000,
        genre_label: str = "",
        material_type: str = "unknown",
    ) -> ReferenceAnchor:
        """Berechnet einen materialadaptiven Perceptual-Reference-Anker.

        Args:
            era_decade:     Ära-Jahrzehnt der Aufnahme (z.B. 1965).
            genre_label:    Genre-Label (z.B. "schlager", "soul", "pop").
            material_type:  Träger-Typ (z.B. "shellac", "vinyl", "mp3_low").

        Returns:
            ReferenceAnchor — immer (valid=False wenn kein Profil gefunden).
        """
        try:
            return self._compute_anchor(era_decade, genre_label, material_type)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("ReferenceAnchorMatcher nicht blockierend error: %s", exc)
            return ReferenceAnchor(
                era_decade=era_decade,
                genre_label=genre_label,
                material_type=material_type,
                valid=False,
            )

    def _compute_anchor(
        self,
        era_decade: int,
        genre_label: str,
        material_type: str,
    ) -> ReferenceAnchor:
        """Kern-Logik für den Anker-Lookup (ohne Exception-Guard)."""
        # Ära-Profil lookup
        profile = self._lookup_era_profile(era_decade)

        # Codec-Qualitäts-Korrektur
        _mat_key = str(material_type or "unknown").lower().strip()
        _codec_scale = _CODEC_QUALITY_PENALTY.get(_mat_key, 0.93)

        # Profil × Codec-Scale (Mindestwert: 0.10 um Null-Floor zu verhindern)
        def _scaled(key: str, default: float) -> float:
            raw = profile.get(key, default)
            return float(np.clip(raw * _codec_scale, 0.10, 1.0))

        anchor = ReferenceAnchor(
            era_decade=era_decade,
            genre_label=str(genre_label or ""),
            material_type=_mat_key,
            naturalness_floor=_scaled("naturalness_floor", 0.88),
            tonal_center_floor=_scaled("tonal_center_floor", 0.92),
            timbre_floor=_scaled("timbre_floor", 0.88),
            spatial_depth_floor=_scaled("spatial_depth_floor", 0.70),
            brilliance_floor=_scaled("brilliance_floor", 0.78),
            warmth_floor=_scaled("warmth_floor", 0.75),
            valid=True,
            profile_source="era_dsp_table",
        )

        logger.debug(
            "ReferenceAnchor: era=%d genre=%s mat=%s → nat_floor=%.2f tonal_floor=%.2f",
            era_decade,
            genre_label or "–",
            _mat_key,
            anchor.naturalness_floor,
            anchor.tonal_center_floor,
        )
        return anchor

    @staticmethod
    def _lookup_era_profile(era_decade: int) -> dict[str, float]:
        """Findet das Era-Profil für ein gegebenes Jahrzehnt."""
        for (start, end), profile in _ERA_QUALITY_PROFILE.items():
            if start <= era_decade <= end:
                return profile
        # Fallback: nächstes Profil
        if era_decade < 1900:
            return _ERA_QUALITY_PROFILE[(1900, 1924)]
        return _ERA_QUALITY_PROFILE[(2010, 2030)]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: ReferenceAnchorMatcher | None = None
_lock = threading.Lock()


def get_reference_anchor_matcher() -> ReferenceAnchorMatcher:
    """Thread-safe Singleton-Accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ReferenceAnchorMatcher()
    return _instance
