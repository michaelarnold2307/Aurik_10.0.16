"""
backend/core/temporal_quality_coherence.py
Aurik 10.0.0 -- Spec §2.16: TemporalQualityCoherenceMetric

PQS-MOS-Konsistenz ueber Zeitachse gemaess Spec §2.16.
MOS-Spanne <= 0.30, sigma(MOS) <= 0.15.
Min. 3 Segmente (Dateien < 25 s werden nicht geprueft).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

TEMPORAL_CONSISTENCY_THRESHOLD: float = 0.30
SIGMA_THRESHOLD: float = 0.15
MIN_FILE_DURATION_S: float = 25.0
SEGMENT_DURATION_S: float = 10.0
SEGMENT_HOP_S: float = 5.0

# §2.54 Material-adaptive thresholds — lossy codecs show high spectral variance
# across segments (bitrate-allocation fluctuations), so tighter thresholds cause
# false-positive warnings without indicating real quality inconsistency.
# §9.12.8 — vintage analog media (shellac, vinyl, tape) have intrinsic amplitude
# dynamics (fade-ins, quiet verses, loud choruses) that cause per-segment SNR proxy
# variation that is NOT a quality inconsistency. Wider thresholds prevent false failure.
_MATERIAL_SPAN_THRESHOLD: dict[str, float] = {
    "mp3_low": 2.50,  # 128 kbps: aggressive bitrate allocation variance
    "mp3_mid": 1.80,  # 192 kbps
    "mp3_high": 1.20,  # 320 kbps
    "lossy_low": 2.50,
    "lossy_mid": 1.80,
    "lossy_high": 1.20,
    "ogg": 1.50,
    "aac": 1.50,
    "cassette": 1.80,  # §v10.0.4: 1.20→1.80 — Kassette hat Dropouts+Oxide-Shed (gemessen: span=1.762)
    # §9.12.8 vintage additions
    "shellac": 2.00,  # Shellac: high noise floor + significant amplitude variation
    "wax_cylinder": 2.50,  # Wax cylinder: very high noise + narrow BW
    "wire_recording": 2.00,
    "vinyl": 1.00,  # Vinyl: moderate noise; better dynamic range than shellac
    "vinyl_lp": 1.00,
    "tape": 1.60,  # §v10.0.4: 1.20→1.60 — analoges Band hat natürliche Varianz
    "reel_tape": 1.60,
    "default": 0.30,
}
_MATERIAL_SIGMA_THRESHOLD: dict[str, float] = {
    "mp3_low": 1.20,
    "mp3_mid": 0.90,
    "mp3_high": 0.60,
    "lossy_low": 1.20,
    "lossy_mid": 0.90,
    "lossy_high": 0.60,
    "ogg": 0.70,
    "aac": 0.70,
    "cassette": 0.60,
    # §9.12.8 vintage additions
    "shellac": 0.90,
    "wax_cylinder": 1.10,
    "wire_recording": 0.90,
    "vinyl": 0.50,
    "vinyl_lp": 0.50,
    "tape": 0.60,
    "reel_tape": 0.60,
    "default": 0.15,
}


# ---------------------------------------------------------------------------
# Ergebnis-Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TemporalCoherenceResult:
    """Spec §2.16: Ergebnis der Temporalen Qualitaetskohaerenzmessung."""

    passed: bool  # True wenn max_span <= 0.30 UND sigma <= 0.15
    max_span: float  # max(MOS) - min(MOS) ueber alle Segmente
    sigma: float  # Standardabweichung sigma(MOS)
    mean_mos: float  # Mittlere PQS-MOS ueber alle Segmente
    n_segments: int  # Anzahl ausgewerteter Segmente
    segment_scores: list[float] = field(default_factory=list)  # MOS pro Segment
    skipped: bool = False  # True wenn Datei zu kurz (< 25 s)
    warnings: list[str] = field(default_factory=list)
    message: str = ""  # Laienverständliche Zusammenfassung (Deutsch)

    @property
    def passes(self) -> bool:
        """Alias fuer passed — Rueckwaertskompatibilitaet (§2.16)."""
        return self.passed


# ---------------------------------------------------------------------------
# Klasse
# ---------------------------------------------------------------------------


class TemporalQualityCoherenceMetric:
    """Spec §2.16: Prueft PQS-MOS-Konsistenz ueber die Zeitachse.

    Algorithmus:
        1. Audio in ueberlappende 10-s-Segmente (Hop 5 s)
        2. PQS-MOS-Schaetzung pro Segment (schnelle DSP-Schätzung)
        3. max_span = max(MOS) - min(MOS) <= 0.30
        4. sigma(MOS) <= 0.15
    """

    TEMPORAL_CONSISTENCY_THRESHOLD: float = TEMPORAL_CONSISTENCY_THRESHOLD
    SIGMA_THRESHOLD: float = SIGMA_THRESHOLD
    MIN_FILE_DURATION_S: float = MIN_FILE_DURATION_S

    def measure(
        self,
        audio: np.ndarray,
        sr: int,
        material_key: str | None = None,
        transfer_chain: list[str] | None = None,
    ) -> TemporalCoherenceResult:
        """Misst temporale Qualitaetskohaerentz.

        Args:
            audio:        float32 1-D oder 2-D
            sr:           Sample-Rate (muss 48000 sein)
            material_key: Optionaler Materialtyp-Key für adaptive Thresholds
                          (z.B. 'mp3_low', 'vinyl', 'cd_digital').
            transfer_chain: Optionale Tonträgerkette für §v10.303.15 Multi-Carrier
                          (permissivster Schwellwert aller Carrier in der Kette).

        Returns:
            TemporalCoherenceResult
        """
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
        if audio.ndim == 2:
            # Handle (2, N) channels-first (UV3) and (N, 2) samples-first
            audio = (
                audio.mean(axis=0) if (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]) else audio.mean(axis=1)
            )

        # §2.54 Material-adaptive thresholds
        _mat = str(material_key).lower().replace("-", "_") if material_key else "default"
        span_threshold = _MATERIAL_SPAN_THRESHOLD.get(_mat, _MATERIAL_SPAN_THRESHOLD["default"])
        sigma_threshold = _MATERIAL_SIGMA_THRESHOLD.get(_mat, _MATERIAL_SIGMA_THRESHOLD["default"])
        # §v10.303.15 Multi-Carrier: permissivsten Schwellwert aller Carrier
        if transfer_chain and len(transfer_chain) > 1:
            _chain_span, _chain_sigma = span_threshold, sigma_threshold
            for _c in transfer_chain:
                _ck = str(_c).lower().replace("-", "_")
                _chain_span = max(_chain_span, _MATERIAL_SPAN_THRESHOLD.get(_ck, span_threshold))
                _chain_sigma = max(_chain_sigma, _MATERIAL_SIGMA_THRESHOLD.get(_ck, sigma_threshold))
            if _chain_span > span_threshold:
                span_threshold, sigma_threshold = _chain_span, _chain_sigma
                logger.debug(
                    "§v10.303.15 Multi-Carrier TQC: span=%.2f sigma=%.2f (chain: %s)",
                    span_threshold,
                    sigma_threshold,
                    transfer_chain,
                )

        duration_s = len(audio) / max(sr, 1)
        if duration_s < self.MIN_FILE_DURATION_S:
            return TemporalCoherenceResult(
                passed=True,
                max_span=0.0,
                sigma=0.0,
                mean_mos=4.0,
                n_segments=0,
                skipped=True,
                warnings=["Datei zu kurz fuer temporale Kohaarenzmessung (< 25 s)."],
            )

        seg_len = int(SEGMENT_DURATION_S * sr)
        hop_len = int(SEGMENT_HOP_S * sr)
        scores: list[float] = []
        i = 0
        while i + seg_len <= len(audio):
            seg = audio[i : i + seg_len]
            score = self._quick_mos(seg, sr)
            if math.isfinite(score):
                scores.append(score)
            else:
                logger.debug("Segment MOS nicht endlich – uebersprungen")
            i += hop_len

        if len(scores) < 3:
            return TemporalCoherenceResult(
                passed=True,
                max_span=0.0,
                sigma=0.0,
                mean_mos=float(np.mean(scores)) if scores else 4.0,
                n_segments=len(scores),
                skipped=True,
                warnings=["Zu wenige Segmente fuer temporale Kohaerenz (< 3)."],
            )

        arr = np.array(scores, dtype=np.float32)
        max_span = float(arr.max() - arr.min())
        sigma = float(arr.std())
        mean_mos = float(arr.mean())
        passes = (max_span <= span_threshold) and (sigma <= sigma_threshold)

        return TemporalCoherenceResult(
            passed=passes,
            max_span=max_span,
            sigma=sigma,
            mean_mos=mean_mos,
            n_segments=len(scores),
            segment_scores=scores,
        )

    # ------------------------------------------------------------------
    # Interne Hilfsmethode: schnelle MOS-Schätzung via SNR-Proxy
    # ------------------------------------------------------------------

    def _quick_mos(self, seg: np.ndarray, sr: int) -> float:  # pylint: disable=unused-argument
        """Schnelle DSP-basierte MOS-Schätzung (kein ML); gibt float in [1,5] zurueck.

        §9.12.8: Lautstärke-normalisiert — misst SNR relativ zum Segment-eigenen RMS-Peak
        (nicht absolut). Dadurch sind stille Intros und laute Choruse vergleichbar.
        """
        if len(seg) == 0:
            return 4.0
        rms = float(np.sqrt(np.mean(seg**2)))
        if rms < 1e-8:
            return 4.0  # Stille → neutral
        # §9.12.8 Amplitude-Normalization: normalize segment to RMS=0.1 before computing
        # SNR proxy. This ensures that a quiet verse and a loud chorus get the same MOS
        # if their noise-to-signal ratio is identical — removes loudness-induced MOS span.
        norm_seg = seg / (rms + 1e-10) * 0.1

        frame = 1024
        if len(norm_seg) < frame:
            return 4.0
        n_frames = len(norm_seg) // frame
        frame_rms = np.array([np.sqrt(np.mean(norm_seg[k * frame : (k + 1) * frame] ** 2)) for k in range(n_frames)])
        noise_floor = float(np.percentile(frame_rms, 5)) + 1e-10
        signal_level = float(np.percentile(frame_rms, 95))
        # Guard: signal_level muss > noise_floor sein (Dirac oder Stille-Artefakt)
        if signal_level <= noise_floor or not math.isfinite(signal_level / noise_floor):
            return 4.0
        snr_db = 20.0 * math.log10(signal_level / noise_floor)
        # Mapping SNR → MOS-Schätzung
        mos = 1.0 + 4.0 * self._sigmoid((snr_db - 15.0) / 10.0)
        return float(np.clip(mos, 1.0, 5.0))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: TemporalQualityCoherenceMetric | None = None
_lock = threading.Lock()


def get_temporal_coherence_metric() -> TemporalQualityCoherenceMetric:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TemporalQualityCoherenceMetric()
    return _instance


def measure_temporal_coherence(
    audio: np.ndarray,
    sr: int,
    material_key: str | None = None,
    transfer_chain: list[str] | None = None,
) -> TemporalCoherenceResult:
    """Convenience-Wrapper.

    §v10.303.15 Multi-Carrier: Wenn transfer_chain gegeben ist, wird der
    permissivste Schwellwert aller Carrier in der Kette verwendet.
    Bei 3 analogen + 1 digitalen Träger ist die Qualitätsvarianz
    NATÜRLICH höher — der strengste Schwellwert würde falsch alarmieren.
    """
    return get_temporal_coherence_metric().measure(
        audio,
        sr,
        material_key=material_key,
        transfer_chain=transfer_chain,
    )


# Convenience-Alias (Spec §3.2)
get_temporal_quality_coherence = get_temporal_coherence_metric

__all__ = [
    "MIN_FILE_DURATION_S",
    "SEGMENT_DURATION_S",
    "SEGMENT_HOP_S",
    "SIGMA_THRESHOLD",
    "TEMPORAL_CONSISTENCY_THRESHOLD",
    "TemporalCoherenceResult",
    "TemporalQualityCoherenceMetric",
    "get_temporal_coherence_metric",
    "get_temporal_quality_coherence",
    "measure_temporal_coherence",
]
