"""§AK: AuraPreserver — Bewahrt die klangliche Aura/Atmosphäre einer Aufnahme.

"Aura" ist die Summe aus:
  - Era-Aesthetics (Epochen-typischer Klang)
  - Emotional-Presence (emotionale Wirkung)
  - Artist-Intent (künstlerische Absicht)
  - Character-Patina (gewollte Unvollkommenheit)

Maximalstufen-Architektur:
  1. Aura-Fingerprinting — Era+Genre+Material+Emotion als Vektor
  2. Aura-Delta-Monitoring — Abweichung nach jeder Phase messen
  3. Aura-Restoration — Automatische Rückführung bei Drift
  4. Integration mit ArtisticIntent + EmotionalArc + EraSignatures
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AuraProfile:
    """Fingerabdruck der klanglichen Aura einer Aufnahme."""

    era_decade: int = 1970
    era_label: str = ""
    genre: str = "unknown"
    material: str = "unknown"
    warmth_score: float = 0.7
    brilliance_score: float = 0.45
    crest_factor: float = 8.0
    stereo_width: float = 0.85
    hf_rolloff_hz: float = 16000.0
    noise_floor_db: float = -60.0
    emotional_arousal: float = 0.5
    emotional_valence: float = 0.5
    character_preservation: float = 1.0
    confidence: float = 0.7


@dataclass
class AuraReport:
    """Bericht über Aura-Erhalt nach Verarbeitung."""

    aura_preserved: bool = True
    aura_drift_score: float = 0.0  # 0.0=perfekt, 1.0=komplett verloren
    warmth_drift_db: float = 0.0
    brilliance_drift_db: float = 0.0
    crest_drift_pct: float = 0.0
    emotional_arc_correlation: float = 1.0
    character_drift: float = 0.0
    era_authenticity: float = 1.0
    warnings: list[str] = field(default_factory=list)


class AuraPreserver:
    """Bewahrt die klangliche Aura über den gesamten Restaurationsprozess.

    Wird in UV3 nach der Pipeline-Initialisierung aufgerufen, um einen
    Baseline-Fingerabdruck zu nehmen, und nach jeder großen Phase um
    die Aura-Abweichung zu messen.
    """

    def __init__(self) -> None:
        self._baseline: AuraProfile | None = None

    def fingerprint(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        era_decade: int = 1970,
        era_label: str = "",
        genre: str = "unknown",
        material: str = "unknown",
        confidence: float = 0.7,
    ) -> AuraProfile:
        """Nimmt den Aura-Fingerabdruck einer Aufnahme.

        Args:
            audio: Original-Audio (float32)
            sr: Sample-Rate
            era_decade: Dekade (z.B. 1970)
            era_label: Ära-Label (z.B. "1970s")
            genre: Genre (z.B. "jazz")
            material: Material (z.B. "vinyl")
            confidence: Confidence der Erkennung
        """
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
        profile = AuraProfile(
            era_decade=era_decade,
            era_label=era_label,
            genre=genre,
            material=material,
            confidence=confidence,
        )

        # ── Akustische Messungen ──
        # Crest-Faktor
        peak = float(np.max(np.abs(mono))) + 1e-10
        rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
        profile.crest_factor = 20.0 * np.log10(peak / rms)

        # HF-Rolloff (Frequenz bei -20 dB vom Peak)
        fft = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / sr)
        max_mag = np.max(fft) + 1e-10
        rolloff_mask = fft < (max_mag * 0.1)  # -20 dB
        if np.any(rolloff_mask):
            first_rolloff = np.argmax(rolloff_mask)
            if first_rolloff > 0 and first_rolloff < len(freqs):
                profile.hf_rolloff_hz = float(freqs[first_rolloff])

        # Wärme (100-500 Hz Energie / Gesamt)
        warm_band = (freqs >= 100) & (freqs <= 500)
        total = float(np.sum(fft)) + 1e-10
        profile.warmth_score = float(np.sum(fft[warm_band])) / total

        # Brillanz (8-16 kHz / Gesamt)
        bright_band = (freqs >= 8000) & (freqs <= 16000)
        profile.brilliance_score = float(np.sum(fft[bright_band])) / total

        # Noise Floor (Median der leisesten 10% der FFT-Bins)
        sorted_mags = np.sort(fft[fft > 1e-10])
        if len(sorted_mags) > 10:
            noise_bins = sorted_mags[: max(10, len(sorted_mags) // 10)]
            profile.noise_floor_db = float(20.0 * np.log10(np.median(noise_bins) + 1e-12))

        # Stereo Width (L/R Differenz / Summe)
        if audio.ndim >= 2 and audio.shape[0] >= 2:
            diff = np.mean(np.abs(audio[0] - audio[1]))
            sum_ = np.mean(np.abs(audio[0] + audio[1])) + 1e-10
            profile.stereo_width = float(diff / sum_)

        self._baseline = profile
        return profile

    def measure_drift(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        emotional_arc: Any = None,
        character_score: float = 1.0,
    ) -> AuraReport:
        """Misst die Aura-Abweichung vom Baseline-Fingerabdruck.

        Args:
            audio: Verarbeitetes Audio
            sr: Sample-Rate
            emotional_arc: Optionales EmotionalArc-Ergebnis
            character_score: Character-Preservation-Score (0-1)
        """
        report = AuraReport()
        if self._baseline is None:
            report.warnings.append("No baseline — fingerprint() first")
            return report

        # Aktuelle Messung
        current = self.fingerprint(
            audio,
            sr,
            era_decade=self._baseline.era_decade,
            era_label=self._baseline.era_label,
            genre=self._baseline.genre,
            material=self._baseline.material,
            confidence=self._baseline.confidence,
        )

        # ── Drift-Berechnung ──
        # Wärme-Drift
        if self._baseline.warmth_score > 1e-10:
            report.warmth_drift_db = float(20.0 * np.log10(current.warmth_score / self._baseline.warmth_score))

        # Brillanz-Drift
        if self._baseline.brilliance_score > 1e-10:
            report.brilliance_drift_db = float(
                20.0 * np.log10(current.brilliance_score / self._baseline.brilliance_score)
            )

        # Crest-Drift
        if self._baseline.crest_factor > 0:
            report.crest_drift_pct = float(
                abs(current.crest_factor - self._baseline.crest_factor) / self._baseline.crest_factor * 100.0
            )

        # Emotional-Arc
        if emotional_arc is not None:
            try:
                if hasattr(emotional_arc, "correlation"):
                    report.emotional_arc_correlation = float(emotional_arc.correlation)
                elif isinstance(emotional_arc, (int, float)):
                    report.emotional_arc_correlation = float(emotional_arc)
            except Exception as e:
                logger.warning("aura_preserver.py::measure_drift Ersatzpfad: %s", e)

        # Character
        report.character_drift = 1.0 - character_score

        # Era-Authenticity (basierend auf HF-Rolloff)
        if self._baseline.hf_rolloff_hz > 100:
            hf_drift = abs(current.hf_rolloff_hz - self._baseline.hf_rolloff_hz)
            report.era_authenticity = max(0.0, 1.0 - hf_drift / self._baseline.hf_rolloff_hz)

        # ── Aura-Drift-Score (gewichteter Mittelwert) ──
        drift_components = [
            abs(report.warmth_drift_db) / 3.0,  # max 3 dB = 1.0
            abs(report.brilliance_drift_db) / 3.0,  # max 3 dB = 1.0
            report.crest_drift_pct / 20.0,  # max 20% = 1.0
            (1.0 - report.emotional_arc_correlation) * 2.0,  # corr=0.5 = 1.0
            report.character_drift,
            (1.0 - report.era_authenticity),
        ]
        report.aura_drift_score = float(np.clip(np.mean(drift_components), 0.0, 1.0))
        report.aura_preserved = report.aura_drift_score < 0.3  # < 30% Drift = preserved

        if not report.aura_preserved:
            report.warnings.append(
                f"Aura drift: {report.aura_drift_score:.1%} "
                f"(warmth={report.warmth_drift_db:+.1f}dB "
                f"brill={report.brilliance_drift_db:+.1f}dB "
                f"crest={report.crest_drift_pct:.0f}%)"
            )

        return report

    def get_baseline(self) -> AuraProfile | None:
        return self._baseline

    def has_baseline(self) -> bool:
        return self._baseline is not None
