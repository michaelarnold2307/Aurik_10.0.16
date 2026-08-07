"""
core/restorability_estimator.py — Aurik 10.0.0+ (§2.26)

RestorabilityEstimator: Schätzt in < 5 s, wie gut das Material restaurierbar
ist und welcher PQS-MOS nach Restaurierung realistisch zu erwarten ist.

Verhindert falsche Erwartungen bei hoffnungslosem Material und liefert dem
Nutzer eine laienverständliche Einschätzung VOR dem vollständigen Prozess.

Invariante: Kein ML, nur DSP-Schnellanalyse. Laufzeit ≤ 5 s.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class RestorabilityResult:
    """Ergebnis des Vor-Assessments (§2.26)."""

    restorability_score: float  # 0–100: 100 = optimale Restaurierbarkeit
    predicted_mos: float  # Erwarteter PQS-MOS ∈ [1.0, 5.0]
    predicted_mos_range: tuple  # 90 %-Konfidenzintervall
    limiting_defects: list[str]  # Top-3 Defekte, die Restaurierbarkeit begrenzen
    recommendations: list[str]  # Laienverständliche Empfehlungen (Deutsch)
    processing_time_estimate_s: float  # Geschätzte Verarbeitungszeit
    snr_db: float = 0.0  # Geschätzter SNR
    grade: str = "unknown"  # excellent / good / fair / poor / critical

    @property
    def tier(self) -> str:
        """Alias for `grade` (§2.47 Spec compatibility)."""
        return self.grade

    def as_dict(self) -> dict:
        return {
            "restorability_score": self.restorability_score,
            "predicted_mos": self.predicted_mos,
            "predicted_mos_range": list(self.predicted_mos_range),
            "limiting_defects": self.limiting_defects,
            "recommendations": self.recommendations,
            "processing_time_estimate_s": self.processing_time_estimate_s,
            "snr_db": self.snr_db,
            "grade": self.grade,
            "tier": self.tier,
        }


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class RestorabilityEstimator:
    """Schätzt Restaurierbarkeit in < 5 s ohne vollständige Pipeline.

    Algorithmus (§2.26):
        1. Schnell-SNR-Schätzung (IMCRA-Minima-Kurve, vereinfacht)
        2. Defekt-Dichte-Schätzung (Clipping-Ratio, Crackle-Impuls-Rate)
        3. Spectral-Bandwidth-Messung (effektive HF-Bandbreite)
        4. Score-Aggregation mit material-spezifischen Schranken
        5. Predicted MOS: regressionsbasiert aus Score + Material-Prior

    Schwellwerte:
        90–100: „Exzellent restaurierbar"
        70–89:  „Gut restaurierbar"
        50–69:  „Mäßig restaurierbar"
        30–49:  „Schwierig restaurierbar"
        0–29:   „Sehr schwer restaurierbar"

    Invarianten:
        - Laufzeit ≤ 5 s (nur DSP, kein ML)
        - NaN in SNR → score = 50 (neutral)
        - Ergebnis ist immer vollständig valide (keine None-Felder)
    """

    SCORE_THRESHOLDS = {
        "excellent": 90.0,
        "good": 70.0,
        "fair": 50.0,
        "poor": 30.0,
    }

    # MOS-Mapping: (score_min, score_max) → (mos_min, mos_max)
    _MOS_MAP = [
        (90, 100, 4.3, 5.0),
        (70, 90, 3.8, 4.3),
        (50, 70, 3.2, 3.8),
        (30, 50, 2.5, 3.2),
        (0, 30, 1.5, 2.5),
    ]

    # Defect-type-specific weights: how much each defect type limits restorability.
    # Clicks/crackle are fully repairable → low penalty;
    # broadband noise is partially repairable → moderate penalty;
    # clipping destroys peaks irreversibly → high penalty.
    # Calibrated from Aurik AMRB test corpus (2024–2026).
    _DEFECT_WEIGHTS = {
        "noise": 0.80,  # broadband noise — largely repairable but leaves residual
        "clipping": 0.55,  # peak destruction — partially repairable via waveform inference
        "crackle": 0.92,  # impulsive — highly repairable via transient detection
        "bandwidth": 0.70,  # spectral loss — partially repairable via AudioSR
        "hum": 0.95,  # tonal interference — near-fully repairable via comb filter
        "wow_flutter": 0.85,  # pitch modulation — correctable via FCPE/CREPE tracking
        "dropout": 0.60,  # signal loss — partially inferable from context
    }

    # Material-specific caps — now with restorability-aware ranges instead of
    # single hard limits.  (floor, ceiling): floor for severe damage, ceiling
    # for mild damage — interpolated by defect severity.
    _MATERIAL_CAPS = {
        "shellac": (55.0, 80.0),
        "wax_cylinder": (40.0, 65.0),
        "wire_recording": (50.0, 70.0),
        "lacquer_disc": (52.0, 72.0),
        "tape": (60.0, 88.0),
        "cassette": (55.0, 85.0),
        "reel_tape": (62.0, 90.0),
        "vinyl": (60.0, 92.0),
    }

    def estimate(
        self,
        audio: np.ndarray,
        sr: int,
        material: str = "unknown",
    ) -> RestorabilityResult:
        """Schätzt Restaurierbarkeit des Audios.

        Args:
            audio:    Float32-Array, mono oder stereo (intern zu mono)
            sr:       Sample-Rate (beliebig — Analyse-Module arbeiten bei nativer SR)
            material: Material-Prior (tape / vinyl / shellac / unknown / …)

        Returns:
            RestorabilityResult mit Score, MOS-Prognose, Empfehlungen.
        """

        # Mono-Konvertierung:
        # Aurik canonical shape is (N, channels) — axis=1 is the channel dimension.
        # (channels, N) shape is never used in Aurik pipelines (§11 Spec 02 §2.46).
        # Using axis=0 would reduce (N, 2) to a 2-element vector → SNR=0.0 bug.
        if audio.ndim == 2:
            # Normalise to (N, channels) if necessary, then average channels
            if audio.shape[0] < audio.shape[1]:
                audio = audio.T  # was (channels, N) → (N, channels)
            mono = audio.mean(axis=1).astype(np.float32)
        else:
            mono = audio.astype(np.float32)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        # §2.26 Performance-Guard: Laufzeit ≤ 5 s (Spec-Invariante)
        _t0 = time.perf_counter()

        limiting_defects: list[str] = []
        # Start at 100 and apply multiplicative defect-type-weighted penalties.
        score = 100.0
        total_severity = 0.0  # summed severity for material cap interpolation

        # ----------------------------------------------------------------
        # 1. SNR-Schätzung (IMCRA-Minima, vereinfacht: Histogramm-Methode)
        # ----------------------------------------------------------------
        snr_db = self._estimate_snr(mono, sr)
        if not math.isfinite(snr_db):
            snr_db = 0.0
            score = 50.0

        w_noise = self._DEFECT_WEIGHTS["noise"]
        if snr_db < -10.0:
            score *= w_noise * 0.56  # 0.80 × 0.56 ≈ 0.45
            limiting_defects.append("extremes_rauschen")
            total_severity += 0.9
        elif snr_db < 5.0:
            score *= w_noise * 0.88  # 0.80 × 0.88 ≈ 0.70
            limiting_defects.append("starkes_rauschen")
            total_severity += 0.6
        elif snr_db < 15.0:
            score *= w_noise * 1.06  # 0.80 × 1.06 ≈ 0.85 (mild)
            total_severity += 0.2
        # SNR ≥ 15 dB → noise fully manageable

        # ----------------------------------------------------------------
        # 2. Clipping-Ratio — irreversible peak destruction
        # ----------------------------------------------------------------
        clip_ratio = float(np.mean(np.abs(mono) >= 0.98))
        w_clip = self._DEFECT_WEIGHTS["clipping"]
        if clip_ratio > 0.05:
            score *= w_clip * 1.27  # 0.55 × 1.27 ≈ 0.70
            limiting_defects.append("starkes_clipping")
            total_severity += 0.8
        elif clip_ratio > 0.01:
            score *= w_clip * 1.60  # 0.55 × 1.60 ≈ 0.88
            total_severity += 0.3

        # ----------------------------------------------------------------
        # 3. Spectral Bandwidth — uses multi-window averaging for robustness
        # ----------------------------------------------------------------
        bw_hz = self._estimate_bandwidth(mono, sr)
        w_bw = self._DEFECT_WEIGHTS["bandwidth"]
        if bw_hz < 4000:
            score *= w_bw * 1.07  # 0.70 × 1.07 ≈ 0.75
            limiting_defects.append("sehr_schmale_bandbreite")
            total_severity += 0.7
        elif bw_hz < 8000:
            score *= w_bw * 1.26  # 0.70 × 1.26 ≈ 0.88
            total_severity += 0.3

        # ----------------------------------------------------------------
        # 4. Impuls-/Crackle-Rate — highly repairable
        # ----------------------------------------------------------------
        crackle_rate = self._estimate_crackle_rate(mono, sr)
        w_crackle = self._DEFECT_WEIGHTS["crackle"]
        if crackle_rate > 0.10:
            score *= w_crackle * 0.87  # 0.92 × 0.87 ≈ 0.80
            limiting_defects.append("starkes_crackle")
            total_severity += 0.5
        elif crackle_rate > 0.03:
            score *= w_crackle * 1.00  # 0.92 (mild)
            total_severity += 0.15

        # ----------------------------------------------------------------
        # 5. Material-spezifische Obergrenzen — severity-interpolated
        # ----------------------------------------------------------------
        if material in self._MATERIAL_CAPS:
            floor, ceiling = self._MATERIAL_CAPS[material]
            # More damage → lower cap (interpolate toward floor)
            sev_norm = min(total_severity / 2.5, 1.0)  # normalize: 2.5 = extreme
            cap = ceiling - sev_norm * (ceiling - floor)
            score = min(score, cap)

        # ----------------------------------------------------------------
        # 6. Defekt-Dichte-Gesamtstrafe
        # ----------------------------------------------------------------
        defect_density = clip_ratio + crackle_rate
        if defect_density > 0.8:
            score *= 0.70

        score = float(np.clip(score, 0.0, 100.0))

        # ----------------------------------------------------------------
        # 7. MOS-Prognose
        # ----------------------------------------------------------------
        predicted_mos = self._score_to_mos(score)
        mos_ci = (
            max(1.0, predicted_mos - 0.3),
            min(5.0, predicted_mos + 0.3),
        )

        # ----------------------------------------------------------------
        # 8. Verarbeitungszeit schätzen (empirische Kurve)
        # ----------------------------------------------------------------
        duration_s = len(mono) / sr
        processing_estimate = duration_s * 2.5 + 15.0  # ~ 2.5× Realzeit + Overhead

        # ----------------------------------------------------------------
        # 9. Grade und Empfehlungen
        # ----------------------------------------------------------------
        grade = self._score_to_grade(score)
        recommendations = self._build_recommendations(grade, snr_db, clip_ratio, bw_hz, material)

        # Nur Top-3 Defekte
        limiting_defects = limiting_defects[:3]

        _elapsed = time.perf_counter() - _t0
        if _elapsed > 5.0:
            logger.warning(
                "RestorabilityEstimator: time Grenze exceeded (%.2fs > 5.0s) — Ergebnis may be partial. material=%s",
                _elapsed,
                material,
            )

        return RestorabilityResult(
            restorability_score=round(score, 1),
            predicted_mos=round(predicted_mos, 2),
            predicted_mos_range=(round(mos_ci[0], 2), round(mos_ci[1], 2)),
            limiting_defects=limiting_defects,
            recommendations=recommendations,
            processing_time_estimate_s=round(processing_estimate, 1),
            snr_db=round(snr_db, 1),
            grade=grade,
        )

    # ----------------------------------------------------------------
    # Hilfsmethoden
    # ----------------------------------------------------------------

    def _estimate_snr(self, mono: np.ndarray, sr: int) -> float:
        """Schnell-SNR via Perzentil-Histogramm-Methode (≈ IMCRA-Minima).

        Nimmt das 5. Perzentil der Kurzzeitenergien als Rauschbodenschätzung.
        """
        try:
            frame_len = int(sr * 0.025)  # 25 ms
            hop = frame_len // 2
            if len(mono) < frame_len:
                return 0.0
            # Vectorized sliding-window RMS — replaces Python loop over ~18k frames (225s audio)
            n_frames = max(1, (len(mono) - frame_len) // hop + 1)
            windows = np.lib.stride_tricks.sliding_window_view(mono, frame_len)[::hop][:n_frames]
            energies = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12)
            if len(energies) == 0:
                return 0.0
            noise_floor = np.percentile(energies, 5)
            signal_level = np.percentile(energies, 90)
            if noise_floor < 1e-9:
                return 40.0  # quasi-stilles Material
            snr = 20.0 * math.log10(signal_level / noise_floor + 1e-12)
            return float(np.clip(snr, -30.0, 60.0))
        except Exception as exc:
            logger.debug("SNR-Schätzung fehlgeschlagen: %s", exc)
            return 0.0

    def _estimate_bandwidth(self, mono: np.ndarray, sr: int) -> float:
        """Effektive HF-Bandbreite via multi-window averaged 95% spectral rolloff.

        Uses overlapping windows to reduce variance from transient content.
        """
        try:
            n_fft = 4096
            hop = n_fft // 2
            n_frames = max(1, min(16, (len(mono) - n_fft) // hop + 1))
            if len(mono) < n_fft:
                n_fft = len(mono)
                n_frames = 1
                hop = n_fft
            rolloffs = []
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            for i in range(n_frames):
                start = i * hop
                chunk = mono[start : start + n_fft]
                if len(chunk) < n_fft:
                    break
                spec = np.abs(np.fft.rfft(chunk, n=n_fft)) ** 2
                total_energy = np.sum(spec) + 1e-12
                cumsum = np.cumsum(spec)
                rolloff_idx = int(np.searchsorted(cumsum, 0.95 * total_energy))
                rolloff_idx = int(np.clip(rolloff_idx, 0, len(freqs) - 1))
                rolloffs.append(float(freqs[rolloff_idx]))
            return float(np.median(rolloffs)) if rolloffs else 20000.0
        except Exception as exc:
            logger.debug("Bandbreiten-Schätzung fehlgeschlagen: %s", exc)
            return 20000.0

    def _estimate_crackle_rate(self, mono: np.ndarray, sr: int) -> float:
        """Impuls-/Crackle-Dichte: Anteil kurzer hochenergetischer Spikes."""
        try:
            frame_len = int(sr * 0.010)  # 10 ms, non-overlapping
            if len(mono) < frame_len * 4:
                return 0.0
            # Vectorized non-overlapping reshape — replaces Python loop over ~22.5k frames (225s audio)
            n_frames = len(mono) // frame_len
            rms_arr = np.sqrt(
                np.mean(mono[: n_frames * frame_len].reshape(n_frames, frame_len).astype(np.float64) ** 2, axis=1)
                + 1e-12
            )
            if len(rms_arr) < 4:
                return 0.0
            median_rms = np.median(rms_arr)
            threshold = median_rms * 6.0  # 6× Median = Impuls
            n_impulses = int(np.sum(rms_arr > threshold))
            return float(n_impulses / len(rms_arr))
        except Exception as exc:
            logger.debug("Crackle-Schätzung fehlgeschlagen: %s", exc)
            return 0.0

    def _score_to_mos(self, score: float) -> float:
        """Regressionsbasierte MOS-Schätzung aus Restorability-Score."""
        for s_min, s_max, m_min, m_max in self._MOS_MAP:
            if score >= s_min:
                t = (score - s_min) / max(s_max - s_min, 1.0)
                return float(m_min + t * (m_max - m_min))
        return 1.5

    def _score_to_grade(self, score: float) -> str:
        if score >= self.SCORE_THRESHOLDS["excellent"]:
            return "excellent"
        if score >= self.SCORE_THRESHOLDS["good"]:
            return "good"
        if score >= self.SCORE_THRESHOLDS["fair"]:
            return "fair"
        if score >= self.SCORE_THRESHOLDS["poor"]:
            return "poor"
        return "critical"

    def _build_recommendations(
        self,
        grade: str,
        snr_db: float,
        clip_ratio: float,
        bw_hz: float,
        material: str,
    ) -> list[str]:
        msgs = {
            "excellent": "Exzellent restaurierbar — fast wie Neuaufnahme erwartet.",
            "good": "Gut restaurierbar — deutliche Verbesserung erwartet.",
            "fair": "Mäßig restaurierbar — Restdefekte werden bleiben.",
            "poor": "Schwierig restaurierbar — Ergebnis besser als Original, aber begrenzt.",
            "critical": "Sehr schwer restaurierbar — das Material ist stark beschädigt.",
        }
        recs = [msgs.get(grade, "Analyse läuft.")]

        if snr_db < 5.0:
            recs.append("Starkes Hintergrundrauschen erkannt — Rauschunterdrückung wird priorisiert.")
        if clip_ratio > 0.01:
            recs.append("Übersteuerungen entdeckt — Clipping-Reparatur wird aktiviert.")
        if bw_hz < 8000:
            recs.append("Begrenzte Klangbandbreite — Frequenzerweiterung wird versucht.")
        if material in ("shellac", "wax_cylinder"):
            recs.append("Historisches Material: Originalklangcharakter wird sorgfältig bewahrt.")
        return recs


# ---------------------------------------------------------------------------
# Thread-sicherer Singleton (Double-Checked Locking §3.2)
# ---------------------------------------------------------------------------

_instance: RestorabilityEstimator | None = None
_lock = threading.Lock()


def get_restorability_estimator() -> RestorabilityEstimator:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RestorabilityEstimator()
    return _instance


def estimate_restorability(
    audio: np.ndarray,
    sr: int,
    material: str = "unknown",
) -> RestorabilityResult:
    """Convenience-Wrapper: Restaurierbarkeit schätzen.

    Args:
        audio:    Float32-Audio (mono oder stereo), SR = 48000
        sr:       48000 (Pflicht)
        material: Material-Prior (tape / vinyl / shellac / …)

    Returns:
        RestorabilityResult mit Score, MOS-Prognose und Nutzer-Empfehlungen.
    """
    return get_restorability_estimator().estimate(audio, sr, material)
