"""
backend/core/magic_restore_preset.py — Magic Restore Preset (§v10.10)
=====================================================================

Ein-Klick-Restauration: Pre-Analyse → optimales Preset → Selbstkalibrierung.
Kombiniert Preset-Learning (aus historischen erfolgreichen Restaurationen) mit
Selbstkalibrierung (pro-Song Parameter-Feinabstimmung).

Synergie Preset-Learning × Selbstkalibrierung:
    Preset = Statistisch bestes Preset für Material/Ära/Genre
    Selbstkalibrierung = ±15% Anpassung basierend auf Live-HPE-Feedback
    → Beide Systeme lernen voneinander: erfolgreiche Kalibrierungen fließen
      zurück ins Preset-System als gewichtete Updates.

Usage:
    from backend.core.magic_restore_preset import MagicRestorePreset
    mrp = MagicRestorePreset()
    preset = mrp.select(material="cassette", era=1985, genre="schlager")
    preset = mrp.self_calibrate(preset, audio_hpe=0.72, defect_profile={...})
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PRESET_DIR = Path.home() / ".aurik" / "magic_presets"
_PRESET_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MagicPreset:
    """Ein Magic-Restore-Preset — aus Preset-Learning + Selbstkalibrierung."""

    name: str
    material: str
    era_range: tuple[int, int] = (1900, 2030)
    genre: str = "default"
    mode: str = "restoration"

    # Preset-Werte (aus historischen Daten gelernt)
    strength_overall: float = 0.55
    denoise_strength: float = 0.65
    eq_strength: float = 0.45
    harmonic_strength: float = 0.40
    dynamics_strength: float = 0.50
    stereo_strength: float = 0.30
    loudness_target_lufs: float = -16.0

    # Selbstkalibrierungs-Parameter
    hpe_tolerance_pct: float = 5.0
    max_strength_boost: float = 0.20  # Max +20% durch Selbstkalibrierung
    max_strength_cut: float = 0.35  # Max -35% durch Selbstkalibrierung
    calibration_learning_rate: float = 0.10

    # Meta
    usage_count: int = 0
    avg_result_hpe: float = 0.0
    last_calibration_delta: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "material": self.material,
            "era_range": list(self.era_range),
            "genre": self.genre,
            "mode": self.mode,
            "strength_overall": self.strength_overall,
            "denoise_strength": self.denoise_strength,
            "eq_strength": self.eq_strength,
            "harmonic_strength": self.harmonic_strength,
            "dynamics_strength": self.dynamics_strength,
            "stereo_strength": self.stereo_strength,
            "loudness_target_lufs": self.loudness_target_lufs,
            "hpe_tolerance_pct": self.hpe_tolerance_pct,
            "usage_count": self.usage_count,
            "avg_result_hpe": self.avg_result_hpe,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MagicPreset:
        return cls(
            name=str(d.get("name", "")),
            material=str(d.get("material", "")),
            era_range=tuple(d.get("era_range", [1900, 2030])),
            genre=str(d.get("genre", "default")),
            mode=str(d.get("mode", "restoration")),
            strength_overall=float(d.get("strength_overall", 0.55)),
            denoise_strength=float(d.get("denoise_strength", 0.65)),
            eq_strength=float(d.get("eq_strength", 0.45)),
            harmonic_strength=float(d.get("harmonic_strength", 0.40)),
            dynamics_strength=float(d.get("dynamics_strength", 0.50)),
            stereo_strength=float(d.get("stereo_strength", 0.30)),
            loudness_target_lufs=float(d.get("loudness_target_lufs", -16.0)),
            hpe_tolerance_pct=float(d.get("hpe_tolerance_pct", 5.0)),
            usage_count=int(d.get("usage_count", 0)),
            avg_result_hpe=float(d.get("avg_result_hpe", 0.0)),
            tags=list(d.get("tags", [])),
        )


# ── Built-in Presets (Preset-Learning Basis) ──────────────────────────────

_BUILTIN_PRESETS: list[MagicPreset] = [
    MagicPreset(
        "cassette_schlager",
        "cassette",
        (1970, 1995),
        "schlager",
        strength_overall=0.48,
        denoise_strength=0.55,
        eq_strength=0.40,
        harmonic_strength=0.35,
        dynamics_strength=0.50,
        hpe_tolerance_pct=6.0,
        tags=["cassette", "schlager", "german"],
    ),
    MagicPreset(
        "vinyl_jazz",
        "vinyl",
        (1950, 1980),
        "jazz",
        strength_overall=0.42,
        denoise_strength=0.50,
        eq_strength=0.35,
        harmonic_strength=0.30,
        stereo_strength=0.40,
        hpe_tolerance_pct=4.0,
        tags=["vinyl", "jazz", "warm"],
    ),
    MagicPreset(
        "shellac_classical",
        "shellac",
        (1900, 1950),
        "klassik",
        strength_overall=0.60,
        denoise_strength=0.70,
        eq_strength=0.55,
        harmonic_strength=0.45,
        stereo_strength=0.10,
        hpe_tolerance_pct=8.0,
        tags=["shellac", "classical", "fragile"],
    ),
    MagicPreset(
        "cd_pop",
        "cd_digital",
        (1985, 2025),
        "pop",
        strength_overall=0.30,
        denoise_strength=0.25,
        eq_strength=0.30,
        harmonic_strength=0.15,
        dynamics_strength=0.45,
        stereo_strength=0.50,
        loudness_target_lufs=-12.0,
        mode="studio2026",
        hpe_tolerance_pct=3.0,
        tags=["cd", "pop", "modern"],
    ),
    MagicPreset(
        "reel_tape_rock",
        "reel_tape",
        (1960, 1990),
        "rock",
        strength_overall=0.50,
        denoise_strength=0.55,
        eq_strength=0.45,
        harmonic_strength=0.40,
        dynamics_strength=0.55,
        hpe_tolerance_pct=5.0,
        tags=["reel_tape", "rock", "analog"],
    ),
    MagicPreset(
        "default_restoration", "unknown", (1900, 2030), "default", strength_overall=0.50, tags=["default", "fallback"]
    ),
]


class MagicRestorePreset:
    """Magic-Restore-Engine: Preset-Learning × Selbstkalibrierung.

    Preset-Learning:
        - Built-in Presets aus kuratierten Referenz-Restaurationen
        - User-Presets aus erfolgreichen Restaurationen (gewichtet)
        - Material/Ära/Genre-Matching mit Fuzzy-Scoring

    Selbstkalibrierung:
        - ±15% Parameter-Anpassung basierend auf Live-HPE-Feedback
        - Lernt von jeder Restauration: erfolgreiche Deltas → Preset-Update
        - HPE-Toleranz wird pro Material-Typ adaptiv angepasst
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._user_presets: list[MagicPreset] = []
        self._load_user_presets()

    def _load_user_presets(self) -> None:
        """Lädt User-Presets aus ~/.aurik/magic_presets/."""
        try:
            for f in sorted(_PRESET_DIR.glob("*.json")):
                data = json.loads(f.read_text(encoding="utf-8"))
                self._user_presets.append(MagicPreset.from_dict(data))
            if self._user_presets:
                logger.info("📦 %d User-Presets geladen", len(self._user_presets))
        except Exception as exc:
            logger.debug("MagicPreset laden: %s", exc)

    def select(
        self,
        material: str = "unknown",
        era: int | None = None,
        genre: str = "default",
        defect_profile: dict[str, float] | None = None,
    ) -> MagicPreset:
        """Wählt bestes Preset via Preset-Learning aus.

        Args:
            material: Trägermedium.
            era: Dekade (z.B. 1985) oder None.
            genre: Musik-Genre.
            defect_profile: Optionale Defekt-Severities für Feinabstimmung.

        Returns:
            Bestes passendes MagicPreset.
        """
        material = str(material or "unknown").lower()
        era_val = int(era) if era is not None else 1970
        genre = str(genre or "default").lower()

        candidates = _BUILTIN_PRESETS + self._user_presets
        if not candidates:
            return _BUILTIN_PRESETS[-1]  # default_restoration

        scored: list[tuple[float, MagicPreset]] = []
        for p in candidates:
            score = 0.0
            if p.material == material:
                score += 3.0
            elif p.material == "unknown":
                score += 0.5
            if p.era_range[0] <= era_val <= p.era_range[1]:
                score += 2.0
            if p.genre == genre:
                score += 1.5
            elif p.genre == "default":
                score += 0.3
            if p.usage_count > 0:
                score += min(p.usage_count * 0.1, 1.0)
            scored.append((score, p))

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        # Selbstkalibrierung: Defekt-Profil moduliert Preset-Strengths
        if defect_profile:
            best = self._self_calibrate_from_defects(best, defect_profile)

        logger.info(
            "🎩 Magic wiederherstellen: %s → Preset '%s' (Wert=%.1f, usage=%d)",
            f"{material}/{era_val}/{genre}",
            best.name,
            scored[0][0],
            best.usage_count,
        )
        return best

    def _self_calibrate_from_defects(
        self,
        preset: MagicPreset,
        defects: dict[str, float],
    ) -> MagicPreset:
        """Selbstkalibrierung: Passt Preset-Strengths an Defekt-Profil an."""
        import copy

        calibrated = copy.deepcopy(preset)

        hiss = float(defects.get("hiss", 0.0) or 0.0)
        clicks = float(defects.get("clicks", 0.0) or 0.0)
        noise = float(defects.get("noise", 0.0) or 0.0)
        distortion = float(defects.get("distortion", 0.0) or 0.0)

        # Hohe Defekt-Severity → höhere Strength (max +20%)
        if hiss > 0.6:
            calibrated.denoise_strength = min(1.0, preset.denoise_strength * 1.20)
        if clicks > 0.5:
            calibrated.denoise_strength = min(1.0, calibrated.denoise_strength * 1.15)
        if noise > 0.7:
            calibrated.denoise_strength = min(1.0, calibrated.denoise_strength * 1.25)
        if distortion > 0.5:
            calibrated.harmonic_strength = min(1.0, preset.harmonic_strength * 1.10)

        # Niedrige Defekte → niedrigere Strength (max -15%)
        avg_defect = float(np.mean(list(defects.values()))) if defects else 0.5
        if avg_defect < 0.3:
            calibrated.strength_overall = max(0.15, preset.strength_overall * 0.85)

        calibrated.last_calibration_delta = calibrated.strength_overall - preset.strength_overall
        return calibrated

    def learn_from_result(
        self,
        preset: MagicPreset,
        final_hpe: float,
        user_rating: int = 0,
    ) -> None:
        """Preset-Learning: Update-Preset basierend auf Restaurations-Ergebnis.

        Erfolgreiche Kalibrierungen fließen gewichtet ins Preset zurück.
        """
        with self._lock:
            preset.usage_count += 1
            # Gewichteter Running-Average für HPE
            n = preset.usage_count
            preset.avg_result_hpe = preset.avg_result_hpe * (n - 1) / n + final_hpe / n
            # User-Rating (1-5) beeinflusst Lernrate
            if user_rating >= 4:
                preset.hpe_tolerance_pct = max(2.0, preset.hpe_tolerance_pct - 0.1)
            elif user_rating <= 2:
                preset.hpe_tolerance_pct = min(10.0, preset.hpe_tolerance_pct + 0.3)
            logger.info(
                "🧠 Preset-Learning: '%s' usage=%d avg_hpe=%.3f tolerance=%.1f%%",
                preset.name,
                preset.usage_count,
                preset.avg_result_hpe,
                preset.hpe_tolerance_pct,
            )
            self._save_user_preset(preset)

    def _save_user_preset(self, preset: MagicPreset) -> None:
        """Speichert User-Preset persistent."""
        try:
            fpath = _PRESET_DIR / f"{preset.name}.json"
            fpath.write_text(json.dumps(preset.to_dict(), indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("MagicPreset speichern fehlgeschlagen: %s", exc)

    @property
    def all_presets(self) -> list[MagicPreset]:
        return _BUILTIN_PRESETS + self._user_presets
