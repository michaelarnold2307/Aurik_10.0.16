"""
backend/core/reference_track_calibrator.py — Reference-Track Auto-Calibration (§v10.10)
=======================================================================================

Extrahiert klangliche Eigenschaften aus einem Referenz-Track und kalibriert
Song-Goals automatisch. Kombiniert Preset (Referenz-Analyse) mit Selbstkalibrierung
(feine Anpassung an das Zielmaterial).

Synergie Preset × Selbstkalibrierung:
    1. Preset = Was WILL ich erreichen? (Referenz-Track-Analyse)
    2. Selbstkalibrierung = Was KANN ich erreichen? (Material-Floor + HPE-Gate)
    3. Ergebnis = Optimaler Kompromiss zwischen Wunsch und Machbarkeit

Usage:
    from backend.core.reference_track_calibrator import ReferenceTrackCalibrator
    cal = ReferenceTrackCalibrator()
    goals = cal.calibrate_from_reference(
        reference_audio, reference_sr,
        target_material="cassette",
        self_calibrate=True,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReferenceProfile:
    """Extrahiertes klangliches Profil eines Referenz-Tracks (Preset)."""

    integrated_lufs: float = -16.0
    spectral_tilt_db_oct: float = -3.0
    stereo_width: float = 0.70
    crest_factor_db: float = 12.0
    brilliance: float = 0.65
    warmth: float = 0.70
    dynamic_range_db: float = 10.0
    hpe_score: float = 0.85


@dataclass
class CalibratedGoals:
    """Kalibrierte Song-Goals: Preset-Ziel + Selbstkalibrierungs-Anpassung."""

    # Preset-seitig (aus Referenz-Track)
    preset_lufs: float = -16.0
    preset_tilt: float = -3.0
    preset_brilliance: float = 0.65
    preset_warmth: float = 0.70

    # Selbstkalibrierung (Material-Floor + HPE-Gate)
    material_lufs_floor: float = -20.0
    material_brilliance_ceiling: float = 0.60
    material_warmth_floor: float = 0.55
    material_hpe_floor: float = 0.60

    # Finale Ziele (Preset ∩ Selbstkalibrierung)
    target_lufs: float = -16.0
    target_tilt: float = -3.0
    target_brilliance: float = 0.60
    target_warmth: float = 0.65

    # Metadaten
    confidence: float = 0.80
    self_calibration_applied: bool = False
    warnings: list[str] = field(default_factory=list)


class ReferenceTrackCalibrator:
    """Kalibriert Song-Goals aus Referenz-Track + Material-Selbstkalibrierung.

    Zweistufiger Prozess:
        1. Preset: Referenz-Track analysieren → Wunsch-Profil
        2. Selbstkalibrierung: Material-Floor prüfen → Machbarkeits-Profil
        3. Merge: Schnittmenge aus Wunsch und Machbarkeit
    """

    def calibrate_from_reference(
        self,
        reference_audio: np.ndarray,
        reference_sr: int,
        target_material: str = "unknown",
        *,
        self_calibrate: bool = True,
    ) -> CalibratedGoals:
        """Haupt-Kalibrierungsmethode.

        Args:
            reference_audio: Audio-Daten des Referenz-Tracks.
            reference_sr: Sample-Rate.
            target_material: Ziel-Trägermedium (z.B. 'cassette').
            self_calibrate: Ob Selbstkalibrierung aktiv sein soll.

        Returns:
            CalibratedGoals mit finalen Zielwerten.
        """
        # Stufe 1: Preset — Referenz-Track analysieren
        _ref_profile = self._analyze_reference(reference_audio, reference_sr)
        logger.info(
            "🎯 Referenz Preset: LUFS=%.1f tilt=%.1f brill=%.2f warm=%.2f",
            _ref_profile.integrated_lufs,
            _ref_profile.spectral_tilt_db_oct,
            _ref_profile.brilliance,
            _ref_profile.warmth,
        )

        # Stufe 2: Selbstkalibrierung — Material-Floor aus Calibration-Matrix
        _material_floor = self._get_material_floor(target_material)
        _material_ceiling = self._get_material_ceiling(target_material)

        # Stufe 3: Merge — Preset ∩ Selbstkalibrierung
        goals = CalibratedGoals(
            preset_lufs=_ref_profile.integrated_lufs,
            preset_tilt=_ref_profile.spectral_tilt_db_oct,
            preset_brilliance=_ref_profile.brilliance,
            preset_warmth=_ref_profile.warmth,
            material_lufs_floor=_material_floor.get("lufs", -20.0),
            material_brilliance_ceiling=_material_ceiling.get("brilliance", 0.60),
            material_warmth_floor=_material_floor.get("warmth", 0.55),
            material_hpe_floor=_material_floor.get("hpe", 0.60),
        )

        if self_calibrate:
            # Selbstkalibrierung: Preset-Ziele auf Material-Machbarkeit begrenzen
            goals.target_lufs = max(_ref_profile.integrated_lufs, _material_floor.get("lufs", -20.0))
            goals.target_tilt = float(
                np.clip(
                    _ref_profile.spectral_tilt_db_oct,
                    -6.0,
                    0.0,
                )
            )
            goals.target_brilliance = min(
                _ref_profile.brilliance,
                _material_ceiling.get("brilliance", 0.60),
            )
            goals.target_warmth = max(
                _ref_profile.warmth,
                _material_floor.get("warmth", 0.55),
            )
            goals.self_calibration_applied = True

            # Warnungen wenn Preset-Wunsch nicht erreichbar
            if _ref_profile.brilliance > _material_ceiling.get("brilliance", 0.60):
                goals.warnings.append(
                    f"Brillanz-Wunsch ({_ref_profile.brilliance:.2f}) über Material-Ceiling "
                    f"({_material_ceiling.get('brilliance', 0.60):.2f}) — auf Ceiling begrenzt"
                )
            if _ref_profile.integrated_lufs < _material_floor.get("lufs", -20.0):
                goals.warnings.append(
                    f"LUFS-Wunsch ({_ref_profile.integrated_lufs:.1f}) unter Material-Floor "
                    f"({_material_floor.get('lufs', -20.0):.1f}) — auf Floor angehoben"
                )
        else:
            goals.target_lufs = _ref_profile.integrated_lufs
            goals.target_tilt = _ref_profile.spectral_tilt_db_oct
            goals.target_brilliance = _ref_profile.brilliance
            goals.target_warmth = _ref_profile.warmth

        goals.confidence = self._compute_confidence(_ref_profile, _material_floor)
        logger.info(
            "🎯 kalibriert Goals (self=%s): LUFS=%.1f tilt=%.1f brill=%.2f warm=%.2f (conf=%.0f%%)",
            goals.self_calibration_applied,
            goals.target_lufs,
            goals.target_tilt,
            goals.target_brilliance,
            goals.target_warmth,
            goals.confidence * 100,
        )

        return goals

    def _analyze_reference(self, audio: np.ndarray, sr: int) -> ReferenceProfile:
        """Extrahiert klangliches Profil aus Referenz-Track (Preset-Seite)."""
        _mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        _rms = float(np.sqrt(np.mean(_mono**2)) + 1e-12)

        # LUFS (vereinfacht via RMS)
        _lufs = float(20.0 * np.log10(_rms + 1e-10))

        # Spectral Tilt (via FFT)
        try:
            _n_fft = 2048
            _spec = np.abs(np.fft.rfft(_mono[:sr], n=_n_fft))
            _freqs = np.fft.rfftfreq(_n_fft, 1.0 / sr)
            _lo = float(np.mean(_spec[(_freqs >= 200) & (_freqs <= 500)]))
            _hi = float(np.mean(_spec[(_freqs >= 4000) & (_freqs <= 8000)]))
            _tilt = float(np.log2(_hi / max(_lo, 1e-10)) * 3.0) if _lo > 1e-10 else -3.0
        except Exception:
            _tilt = -3.0

        # Stereo Width
        if audio.ndim == 2 and audio.shape[0] == 2:
            _corr = float(np.corrcoef(audio[0, :sr], audio[1, :sr])[0, 1])
            _width = float(np.clip(1.0 - abs(_corr), 0.0, 1.0))
        else:
            _width = 0.0

        # Crest Factor
        _peak = float(np.max(np.abs(_mono[:sr])))
        _crest = float(20.0 * np.log10(_peak / _rms)) if _rms > 1e-10 else 12.0

        return ReferenceProfile(
            integrated_lufs=_lufs,
            spectral_tilt_db_oct=_tilt,
            stereo_width=_width,
            crest_factor_db=_crest,
        )

    def _get_material_floor(self, material: str) -> dict[str, float]:
        """Selbstkalibrierung: Material-spezifische Qualitäts-Floors."""
        try:
            from backend.core.calibration_matrix import get_material_floor as _gmf

            return dict(_gmf(material) or {})  # type: ignore
        except Exception:
            pass  # Material floor lookup best-effort — non-critical
        return {"lufs": -20.0, "warmth": 0.55, "hpe": 0.60}

    def _get_material_ceiling(self, material: str) -> dict[str, float]:
        """Selbstkalibrierung: Material-spezifische Qualitäts-Ceilings."""
        _ceilings = {
            "shellac": {"brilliance": 0.40, "stereo_width": 0.20},
            "vinyl": {"brilliance": 0.75, "stereo_width": 0.80},
            "cassette": {"brilliance": 0.55, "stereo_width": 0.60},
            "tape": {"brilliance": 0.60, "stereo_width": 0.65},
            "reel_tape": {"brilliance": 0.70, "stereo_width": 0.70},
            "cd_digital": {"brilliance": 0.90, "stereo_width": 0.95},
        }
        return _ceilings.get(material.lower(), {"brilliance": 0.60, "stereo_width": 0.60})

    def _compute_confidence(
        self,
        ref: ReferenceProfile,
        floor: dict[str, float],
    ) -> float:
        """Berechnet Konfidenz: wie gut passt das Preset zum Material?"""
        _score = 0.80
        _lufs_gap = abs(ref.integrated_lufs - floor.get("lufs", -20.0))
        if _lufs_gap > 6.0:
            _score -= 0.15
        elif _lufs_gap > 3.0:
            _score -= 0.05
        return float(np.clip(_score, 0.30, 1.0))
