"""§AL: PrecisionDefectLocator — Sub-Sample Edge Refinement + Overlap Resolution.

Schließt die Präzisionslücken der Defekterkennung:
  1. Sub-Sample Edge Refinement — Zero-Crossing + Envelope-Matching an Defektgrenzen
  2. Overlap Resolution — Mehrere Defekte in derselben Region priorisieren
  3. Sub-Type Classification — Click/Crackle/Hum-Feindifferenzierung
  4. Confidence Calibration — Heuristik → kalibrierte Confidence

Integriert mit DefectScanner als Post-Processing-Schicht.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RefinedDefect:
    """Präzise lokalisierter Defekt mit sub-sample Genauigkeit."""

    start_sample: int
    end_sample: int
    start_sub: float  # Sub-sample offset (0.0-1.0)
    end_sub: float  # Sub-sample offset (0.0-1.0)
    defect_type: str
    sub_type: str = ""  # Feindifferenzierung
    severity: float = 0.0
    peak_db: float = -60.0
    confidence: float = 0.7
    overlap_group: int = -1
    priority: int = 0  # Höhere Zahl = höhere Priorität


@dataclass
class LocatorReport:
    refined: list[RefinedDefect] = field(default_factory=list)
    edges_refined: int = 0
    overlaps_resolved: int = 0
    subtypes_assigned: int = 0
    confidence_calibrated: int = 0


class PrecisionDefectLocator:
    """Verfeinert Defekt-Positionen auf sub-sample Genauigkeit."""

    # Prioritäten für Overlap-Resolution (höher = wichtiger)
    PRIORITY = {
        "CLIPS": 10,
        "DROPOUTS": 9,
        "TAPE_SPLICE_ARTIFACT": 9,
        "CLICKS": 8,
        "CRACKLE": 7,
        "HUM": 6,
        "WOW": 6,
        "FLUTTER": 6,
        "HIGH_FREQ_NOISE": 4,
        "LOW_FREQ_RUMBLE": 4,
        "HISS": 3,
        "DIGITAL_ARTIFACTS": 5,
        "COMPRESSION_ARTIFACTS": 5,
        "NOISE": 3,
        "SURFACE_NOISE": 3,
    }

    def __init__(self) -> None:
        self._reports: list[LocatorReport] = []

    def refine_edges(
        self,
        audio: np.ndarray,
        sr: int,
        defects: list[dict],
    ) -> list[RefinedDefect]:
        """Verfeinert Defekt-Grenzen auf sub-sample Genauigkeit.

        Algorithmus:
        1. Zero-Crossing-Suche: Nächster Nulldurchgang als natürliche Grenze
        2. Envelope-Kontinuität: Grenze dort wo RMS-Hüllkurve stationär wird
        3. Sub-sample-Offset via lineare Interpolation

        Args:
            audio: (samples,) oder (channels, samples) float32
            sr: Sample-Rate
            defects: Liste von {'type': str, 'start_s': float, 'end_s': float, ...}
        """
        mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio
        report = LocatorReport()
        refined = []

        for d in defects:
            s0 = int(d.get("start_s", 0) * sr)
            s1 = int(d.get("end_s", 0) * sr)
            dtype = str(d.get("type", "UNKNOWN"))
            sev = float(d.get("severity", 0.5))
            peak = float(d.get("peak_db", -60))
            conf = float(d.get("confidence", 0.7))

            # ── Edge Refinement ──
            # Suche ±5ms um die Original-Grenze
            search_win = int(sr * 0.005)
            s0_ref, s0_sub = self._refine_edge(mono, s0, -1, search_win)
            s1_ref, s1_sub = self._refine_edge(mono, s1, 1, search_win)

            if s0_ref != s0 or s1_ref != s1:
                report.edges_refined += 1

            # ── Sub-Type Classification ──
            sub_type = self._classify_subtype(mono, s0_ref, s1_ref, sr, dtype)

            # ── Confidence Calibration ──
            calib_conf = self._calibrate_confidence(mono, s0_ref, s1_ref, sr, dtype, conf)

            refined.append(
                RefinedDefect(
                    start_sample=s0_ref,
                    end_sample=s1_ref,
                    start_sub=s0_sub,
                    end_sub=s1_sub,
                    defect_type=dtype,
                    sub_type=sub_type,
                    severity=sev,
                    peak_db=peak,
                    confidence=calib_conf,
                    priority=self.PRIORITY.get(dtype, 5),
                )
            )

        # ── Overlap Resolution ──
        refined = self._resolve_overlaps(refined, report)

        self._reports.append(report)
        return refined

    def _refine_edge(self, audio: np.ndarray, sample: int, direction: int, window: int) -> tuple[int, float]:
        """Verfeinert eine Defekt-Grenze via Zero-Crossing + Envelope.

        direction: -1 = Start-Grenze (rückwärts suchen)
                   +1 = End-Grenze (vorwärts suchen)
        Returns: (refined_sample, sub_sample_offset)
        """
        n = len(audio)
        best_sample = sample
        best_sub = 0.0

        # Methode 1: Zero-Crossing (für impulsive Defekte)
        for offset in range(window):
            idx = sample + direction * offset
            if idx <= 0 or idx >= n - 1:
                break
            # Suche Nulldurchgang
            if audio[idx] * audio[idx - 1] <= 0:
                # Sub-sample via lineare Interpolation
                if abs(audio[idx] - audio[idx - 1]) > 1e-10:
                    frac = audio[idx - 1] / (audio[idx - 1] - audio[idx])
                    best_sub = float(np.clip(frac, 0.0, 1.0))
                best_sample = idx
                break

        # Methode 2: Envelope-Stationarität (für stationäre Defekte)
        # Prüfe ob RMS in 5ms Fenster stabil ist
        check_win = min(window // 2, int(len(audio) * 0.001))
        if check_win >= 4 and best_sample == sample:
            best_rms_var = float("inf")
            for offset in range(-window, window):
                idx = sample + offset
                if idx < check_win or idx > n - check_win:
                    continue
                pre = audio[idx - check_win : idx]
                post = audio[idx : idx + check_win]
                pre_rms = float(np.sqrt(np.mean(pre**2) + 1e-12))
                post_rms = float(np.sqrt(np.mean(post**2) + 1e-12))
                rms_jump = abs(post_rms - pre_rms)
                if rms_jump < best_rms_var:
                    best_rms_var = rms_jump
                    best_sample = idx

        return best_sample, best_sub

    def _classify_subtype(self, audio: np.ndarray, s0: int, s1: int, sr: int, dtype: str) -> str:
        """Feindifferenzierung von Defekt-Untertypen."""
        seg = audio[s0:s1]
        if len(seg) < 8:
            return ""

        if dtype in ("CLICKS", "CLICK", "CLICK_POP"):
            # Click-Subtypen nach Dauer + Spektrum
            duration_ms = (s1 - s0) / sr * 1000
            if duration_ms < 0.5:
                return "impulse"  # Extrem kurzer Impuls
            elif duration_ms < 2.0:
                return "tick"  # Kurzer Tick
            elif duration_ms < 5.0:
                # Prüfe ob Oberflächen-Knistern (breitbandig)
                fft = np.abs(np.fft.rfft(seg))
                spectral_flatness = float(np.exp(np.mean(np.log(fft + 1e-10))) / (np.mean(fft) + 1e-10))
                if spectral_flatness > 0.6:
                    return "surface_click"  # Vinyl-Oberfläche
                return "pop"  # Längerer Pop
            return "scratch"

        if dtype in ("CRACKLE",):
            # Crackle-Subtypen
            fft = np.abs(np.fft.rfft(seg))
            ratio_high = float(np.sum(fft[len(fft) // 2 :])) / (float(np.sum(fft)) + 1e-10)
            if ratio_high > 0.4:
                return "vinyl_crackle"
            return "tape_crackle"

        if dtype in ("HUM",):
            # Hum-Subtypen
            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
            # Prüfe ob 50Hz oder 60Hz dominant
            for hz, label in [(50, "hum_50hz"), (60, "hum_60hz"), (100, "hum_100hz"), (120, "hum_120hz")]:
                band = (freqs >= hz - 2) & (freqs <= hz + 2)
                if np.any(band) and np.mean(fft[band]) > np.mean(fft) * 3:
                    return label
            return "hum_broadband"

        return ""

    def _calibrate_confidence(self, audio: np.ndarray, s0: int, s1: int, sr: int, dtype: str, raw_conf: float) -> float:
        """Kalibriert Confidence via Signal-Qualität im Defekt-Bereich."""
        seg = audio[s0:s1]
        if len(seg) < 8:
            return raw_conf

        # SNR im Defekt-Bereich
        rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
        peak = float(np.max(np.abs(seg))) + 1e-12
        crest = peak / rms

        # Hoher Crest-Faktor = klarer Defekt (höhere Confidence)
        if dtype in ("CLICKS", "CLICK_POP", "CLICK"):
            crest_factor = min(1.0, crest / 10.0)
            return float(np.clip(raw_conf * (0.8 + 0.2 * crest_factor), 0.3, 0.98))

        # Stationäre Defekte: Confidence sinkt wenn Signal zu leise
        if rms < 0.001:
            return float(np.clip(raw_conf * 0.7, 0.2, 0.8))

        return raw_conf

    def _resolve_overlaps(self, defects: list[RefinedDefect], report: LocatorReport) -> list[RefinedDefect]:
        """Löst überlappende Defekte auf — priorisiert nach Schwere/Seltenheit."""
        if len(defects) < 2:
            return defects

        # Sortiere nach Priorität (höchste zuerst)
        sorted_defects = sorted(defects, key=lambda d: (d.priority, d.severity), reverse=True)

        # Gruppiere überlappende Defekte
        groups: list[list[RefinedDefect]] = []
        for d in sorted_defects:
            placed = False
            for g in groups:
                # Prüfe Überlappung mit erstem Defekt der Gruppe
                g0 = g[0]
                if d.start_sample <= g0.end_sample and d.end_sample >= g0.start_sample:
                    g.append(d)
                    placed = True
                    report.overlaps_resolved += 1
                    break
            if not placed:
                groups.append([d])

        # Nur den höchst-priorisierten Defekt pro Gruppe behalten
        resolved: list[Any] = []
        for g in groups:
            d = g[0]
            d.overlap_group = len(resolved)
            resolved.append(d)

        return resolved

    def get_reports(self) -> list[LocatorReport]:
        return list(self._reports)
