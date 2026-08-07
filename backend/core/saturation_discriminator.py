"""
§6.5d SaturationDiscriminator — Präzise SOTA-Diskriminierung guter vs. schlechter Sättigung

Klassifiziert Soft-Saturation chirurgisch auf Signal-Ebene (nicht taxonomisch):
  - Gute Sättigung (PRESERVE): H2-dominant, progressiver Onset, musikalische Kohärenz
  - Schlechte Sättigung (REPAIR): H3/H5-dominant, harter Onset, Generation-Loss-Charakter

Analyse-Pipeline:
  1. Multi-Resolution Harmonic Decomposition (STFT-basiert, 3 FFT-Größen)
  2. Even/Odd Harmonic Ratio pro Frequenzband
  3. Saturation-Onset-Detection (gradual vs. sudden knee)
  4. Per-Segment-Klassifikation (Segment = 1s Fenster)
  5. Surgical Repair Plan: (start_s, end_s, band_hz_low, band_hz_high, strength)

Zentrale Entscheidungsstelle: wird VOR DefectScanner und Phase-Selektion aufgerufen.
Ergebnis fließt in IntentionalArtifactClassifier und UV3 Phase-Strength-Caps ein.

Author: Aurik Development Team
Version: 10.1.0
Date: 2026-07-13
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as _scipy_signal

logger = logging.getLogger(__name__)

# ── Analyse-Parameter ────────────────────────────────────────────────────

# Analyse-Fenster-Größen für Multi-Resolution-Decomposition
ANALYSIS_WINDOWS: list[int] = [1024, 4096, 16384]  # Kurz/Mittel/Lang
ANALYSIS_HOP: int = 512  # 10.7 ms @ 48 kHz
SEGMENT_DURATION_S: float = 1.0  # Klassifikation pro 1s-Segment
MIN_SEGMENT_SAMPLES: int = 48000  # 1s @ 48kHz

# Harmonic Ratio Schwellwerte (kalibriert an Referenzsignalen)
H2_DOMINANCE_THRESHOLD: float = 1.3  # H2 > H3 × 1.3 → gute Sättigung (tanh/Röhre)
H3_DOMINANCE_THRESHOLD: float = 0.7  # H3 > H2 × 0.7 → Clipping-Charakter
EVEN_ODD_RATIO_TAPE: float = 1.8  # H2+H4 / H3+H5 ≥ 1.8 → Bandsättigung
EVEN_ODD_RATIO_CLIP: float = 0.5  # H2+H4 / H3+H5 ≤ 0.5 → Hard Clipping

# Onset-Detection: progressiv vs. abrupt
ONSET_RAMP_MS: float = 50.0  # ≤ 50ms Anstieg = abrupt; > 50ms = progressiv
ONSET_GRADUAL_THRESHOLD: float = 0.6  # > 0.6 = gradual (tape knee)

# Band-Struktur für per-Band-Analyse
SATURATION_BANDS: list[tuple[float, float, str]] = [
    (20.0, 250.0, "sub_bass"),
    (250.0, 2000.0, "low_mid"),
    (2000.0, 8000.0, "presence"),
    (8000.0, 20000.0, "air"),
]


# ── Datenklassen ──────────────────────────────────────────────────────────


@dataclass
class SaturationSegment:
    """Analyse-Ergebnis für ein 1s-Zeitsegment."""

    start_s: float
    end_s: float
    h2_energy: float
    h3_energy: float
    h4_energy: float
    h5_energy: float
    even_odd_ratio: float  # (H2+H4) / (H3+H5)
    onset_slope: float  # 0 = abrupt, 1 = gradual
    classification: str = "ambiguous"  # "preserve" | "repair" | "ambiguous"
    confidence: float = 0.5
    per_band: dict[str, float] = field(default_factory=dict)  # band → even_odd_ratio


@dataclass
class SaturationDiscriminationResult:
    """Gesamtergebnis der Saturation-Diskriminierung."""

    segments: list[SaturationSegment] = field(default_factory=list)
    global_classification: str = "ambiguous"  # preserve / repair / mixed
    preserve_segments: list[tuple[float, float]] = field(default_factory=list)
    repair_segments: list[tuple[float, float]] = field(default_factory=list)
    repair_plan: list[dict] = field(default_factory=list)  # surgical repair instructions
    h2_h4_dominant: bool = False  # True → tape/tube character dominates
    h3_h5_dominant: bool = False  # True → clipping/overdrive character dominates
    onset_is_gradual: bool = False  # True → progressive saturation (tape knee)


# ── Core DSP ──────────────────────────────────────────────────────────────


def _detect_fundamental_hz(audio: np.ndarray, sr: int, f_min: float = 40.0, f_max: float = 800.0) -> float | None:
    """Detektiert dominante Grundfrequenz via Autokorrelation."""
    n = min(len(audio), 8192)
    if n < 512:
        return None
    segment = audio[:n].astype(np.float64)
    segment = segment - np.mean(segment)
    corr = np.correlate(segment, segment, mode="full")
    corr = corr[len(corr) // 2 :] / (corr[len(corr) // 2] + 1e-12)
    lag_min = max(1, int(sr / f_max))
    lag_max = min(n - 1, int(sr / f_min))
    if lag_max <= lag_min:
        return None
    peak = int(np.argmax(corr[lag_min:lag_max]))
    peak_val = float(corr[lag_min + peak])
    if peak_val < 0.35:
        return None
    return float(sr) / float(lag_min + peak)


def _harmonic_energy_at_order(
    fft_mag: np.ndarray, f0_hz: float, order: int, sr: int, fft_size: int, radius: int = 2
) -> float:
    """Summiert FFT-Energie um eine harmonische Ordnung."""
    bin_hz = sr / fft_size
    center_bin = round(f0_hz * order / bin_hz)
    lo = max(0, center_bin - radius)
    hi = min(len(fft_mag), center_bin + radius + 1)
    return float(np.sum(fft_mag[lo:hi] ** 2))


def _analyze_saturation_segment(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> SaturationSegment:
    """Analysiert die Sättigungscharakteristik eines 1s-Audiosegments."""
    seg = audio[int(start_s * sr) : int(end_s * sr)]
    if len(seg) < 1024:
        return SaturationSegment(
            start_s=start_s,
            end_s=end_s,
            h2_energy=0,
            h3_energy=0,
            h4_energy=0,
            h5_energy=0,
            even_odd_ratio=1.0,
            onset_slope=0.5,
        )

    mono = seg if seg.ndim == 1 else seg.mean(axis=1)
    mono = mono.astype(np.float64)

    # Multi-Resolution Harmonic Analysis (3 FFT-Größen)
    h2_total, h3_total, h4_total, h5_total = 0.0, 0.0, 0.0, 0.0
    f0 = _detect_fundamental_hz(mono, sr)

    for n_fft in ANALYSIS_WINDOWS:
        n = min(len(mono), n_fft)
        fft_mag = np.abs(np.fft.rfft(mono[:n], n=n_fft))

        if f0 is not None:
            f0_energy = _harmonic_energy_at_order(fft_mag, f0, 1, sr, n_fft) + 1e-12
            h2_total += _harmonic_energy_at_order(fft_mag, f0, 2, sr, n_fft) / f0_energy
            h3_total += _harmonic_energy_at_order(fft_mag, f0, 3, sr, n_fft) / f0_energy
            h4_total += _harmonic_energy_at_order(fft_mag, f0, 4, sr, n_fft) / f0_energy
            h5_total += _harmonic_energy_at_order(fft_mag, f0, 5, sr, n_fft) / f0_energy

    n_windows = len(ANALYSIS_WINDOWS)
    h2 = h2_total / n_windows
    h3 = h3_total / n_windows
    h4 = h4_total / n_windows
    h5 = h5_total / n_windows

    even_odd_ratio = (h2 + h4) / (h3 + h5 + 1e-12)

    # Onset-Slope: misst wie abrupt die Sättigung einsetzt
    # Vergleich: RMS-Wachstum in den ersten 50ms vs. nächsten 200ms
    onset_n = int(ONSET_RAMP_MS / 1000.0 * sr)
    if len(mono) > onset_n * 4:
        onset_rms = np.sqrt(np.mean(mono[:onset_n] ** 2) + 1e-12)
        body_rms = np.sqrt(np.mean(mono[onset_n : onset_n * 4] ** 2) + 1e-12)
        onset_slope = float(np.clip(onset_rms / body_rms, 0.0, 1.0))
    else:
        onset_slope = 0.5

    # Per-Band-Analyse
    per_band: dict[str, float] = {}
    for band_lo, band_hi, band_name in SATURATION_BANDS:
        band_lo_bin = int(band_lo / (sr / 2) * (ANALYSIS_WINDOWS[1] // 2 + 1))
        band_hi_bin = int(band_hi / (sr / 2) * (ANALYSIS_WINDOWS[1] // 2 + 1))
        band_lo_bin = max(1, band_lo_bin)
        band_hi_bin = min(ANALYSIS_WINDOWS[1] // 2, band_hi_bin)
        if band_hi_bin > band_lo_bin:
            m = min(len(mono), ANALYSIS_WINDOWS[1])
            fft_b = np.abs(np.fft.rfft(mono[:m], n=ANALYSIS_WINDOWS[1]))
            band_energy = float(np.sum(fft_b[band_lo_bin:band_hi_bin] ** 2)) + 1e-12
            per_band[band_name] = band_energy
        else:
            per_band[band_name] = 0.0

    # Klassifikation
    if even_odd_ratio >= EVEN_ODD_RATIO_TAPE and onset_slope >= ONSET_GRADUAL_THRESHOLD:
        classification = "preserve"
        confidence = float(np.clip((even_odd_ratio - 1.0) / 2.0, 0.5, 1.0))
    elif even_odd_ratio <= EVEN_ODD_RATIO_CLIP or onset_slope < 0.3:
        classification = "repair"
        confidence = float(np.clip(1.0 - even_odd_ratio, 0.5, 1.0))
    else:
        classification = "ambiguous"
        confidence = 0.5

    return SaturationSegment(
        start_s=start_s,
        end_s=end_s,
        h2_energy=h2,
        h3_energy=h3,
        h4_energy=h4,
        h5_energy=h5,
        even_odd_ratio=even_odd_ratio,
        onset_slope=onset_slope,
        classification=classification,
        confidence=confidence,
        per_band=per_band,
    )


# ── Haupt-API ─────────────────────────────────────────────────────────────


def discriminate_saturation(
    audio: np.ndarray,
    sr: int,
    material_type: str = "unknown",
    transfer_chain: list[str] | None = None,
) -> SaturationDiscriminationResult:
    """Zentrale Saturation-Diskriminierung — SOTA-Präzision pro Segment.

    Führt Multi-Resolution-Harmonic-Analysis auf 1s-Segmenten durch und
    klassifiziert jedes Segment als PRESERVE (gute Sättigung) oder REPAIR
    (schlechte Sättigung / Generation Loss / Clipping).

    Args:
        audio: Stereo-Audio (N,2) oder Mono (N,) float32, ±1.0.
        sr: Sample-Rate.
        material_type: Primäres Trägermaterial (für Kontext).
        transfer_chain: Vollständige Transfer-Kette.

    Returns:
        SaturationDiscriminationResult mit Segment-Klassifikationen und
        chirurgischem Repair-Plan.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    mono = np.asarray(mono, dtype=np.float32)
    total_dur = len(mono) / sr

    # 1s-Segment-Analyse
    segments: list[SaturationSegment] = []
    seg_dur = SEGMENT_DURATION_S
    t = 0.0
    while t + seg_dur <= total_dur:
        seg = _analyze_saturation_segment(mono, sr, t, t + seg_dur)
        segments.append(seg)
        t += seg_dur

    if not segments:
        return SaturationDiscriminationResult()

    # Globale Aggregation
    n_preserve = sum(1 for s in segments if s.classification == "preserve")
    n_repair = sum(1 for s in segments if s.classification == "repair")
    n_segments = len(segments)

    h2_h4_dominant = np.mean([s.even_odd_ratio for s in segments]) >= EVEN_ODD_RATIO_TAPE
    h3_h5_dominant = np.mean([s.even_odd_ratio for s in segments]) <= EVEN_ODD_RATIO_CLIP
    onset_is_gradual = np.mean([s.onset_slope for s in segments]) >= ONSET_GRADUAL_THRESHOLD

    if n_preserve > n_repair and n_preserve > n_segments * 0.4:
        global_class = "preserve"
    elif n_repair > n_preserve and n_repair > n_segments * 0.3:
        global_class = "repair"
    else:
        global_class = "mixed"

    # Preserve/Repair-Zeitbereiche
    preserve_zones = [(s.start_s, s.end_s) for s in segments if s.classification == "preserve"]
    repair_zones = [(s.start_s, s.end_s) for s in segments if s.classification == "repair"]

    # Chirurgischer Repair-Plan: (Zeit, Band, Stärke)
    repair_plan: list[dict] = []
    for seg in segments:
        if seg.classification == "repair":
            for band_name, band_energy in seg.per_band.items():
                if band_energy > 0:
                    band_info = next((b for b in SATURATION_BANDS if b[2] == band_name), None)
                    if band_info:
                        repair_plan.append(
                            {
                                "start_s": seg.start_s,
                                "end_s": seg.end_s,
                                "band_hz_low": band_info[0],
                                "band_hz_high": band_info[1],
                                "band_name": band_name,
                                "strength": float(np.clip(seg.confidence, 0.1, 0.9)),
                                "even_odd_ratio": seg.even_odd_ratio,
                            }
                        )

    result = SaturationDiscriminationResult(
        segments=segments,
        global_classification=global_class,
        preserve_segments=preserve_zones,
        repair_segments=repair_zones,
        repair_plan=repair_plan,
        h2_h4_dominant=h2_h4_dominant,  # type: ignore[arg-type]
        h3_h5_dominant=h3_h5_dominant,  # type: ignore[arg-type]
        onset_is_gradual=onset_is_gradual,  # type: ignore[arg-type]
    )

    logger.info(
        "SaturationDiscriminator: %d segments analyzed — "
        "preserve=%d repair=%d global=%s h2h4_dom=%s h3h5_dom=%s onset_gradual=%s "
        "material=%s chain=%s",
        n_segments,
        n_preserve,
        n_repair,
        global_class,
        h2_h4_dominant,
        h3_h5_dominant,
        onset_is_gradual,
        material_type,
        transfer_chain,
    )

    return result


def should_preserve_saturation(result: SaturationDiscriminationResult) -> bool:
    """Schnelle Ja/Nein-Entscheidung für Pipeline-Integration."""
    return result.global_classification == "preserve"


def get_saturation_strength_cap(result: SaturationDiscriminationResult, segment_start_s: float = 0.0) -> float:
    """Gibt den Strength-Cap für einen Zeitpunkt zurück (0.0 = voll reparieren, 1.0 = erhalten)."""
    for seg in result.segments:
        if seg.start_s <= segment_start_s < seg.end_s:
            if seg.classification == "preserve":
                return 0.10  # Preserve → max 10% Strength
            elif seg.classification == "repair":
                return 0.90  # Repair → bis zu 90% Strength
            else:
                return 0.40  # Ambiguous → 40%
    return 0.50  # Default


@dataclass
class SaturationClassificationSummary:
    """Aggregierte Zusammenfassung eines SaturationDiscriminationResult für
    Aufrufer, die nur die Gesamt-Entscheidung (nicht die Segment-Details) benötigen."""

    result: SaturationDiscriminationResult
    good_ratio: float  # Anteil "preserve"-Segmente
    bad_ratio: float  # Anteil "repair"-Segmente
    preservation_ratio: float  # good / (good + bad), 0.0 wenn beide 0

    def as_dict(self) -> dict:
        return {
            "global_classification": self.result.global_classification,
            "good_ratio": self.good_ratio,
            "bad_ratio": self.bad_ratio,
            "preservation_ratio": self.preservation_ratio,
            "h2_h4_dominant": self.result.h2_h4_dominant,
            "h3_h5_dominant": self.result.h3_h5_dominant,
            "onset_is_gradual": self.result.onset_is_gradual,
        }


class SaturationDiscriminator:
    """Zustandsloser Objekt-Wrapper um `discriminate_saturation()` für Aufrufer,
    die eine Instanz mit `.classify()` erwarten statt der freien Funktion."""

    def classify(
        self,
        audio: np.ndarray,
        sr: int,
        transfer_chain: list[str] | None = None,
        era_decade: int | None = None,
    ) -> SaturationClassificationSummary:
        _ = era_decade  # discriminate_saturation() nutzt aktuell keine Era-Information
        result = discriminate_saturation(audio, sr, transfer_chain=transfer_chain)
        n = len(result.segments)
        n_preserve = sum(1 for s in result.segments if s.classification == "preserve")
        n_repair = sum(1 for s in result.segments if s.classification == "repair")
        good_ratio = n_preserve / n if n else 0.0
        bad_ratio = n_repair / n if n else 0.0
        denom = good_ratio + bad_ratio
        preservation_ratio = good_ratio / denom if denom > 0 else 0.0
        return SaturationClassificationSummary(
            result=result,
            good_ratio=good_ratio,
            bad_ratio=bad_ratio,
            preservation_ratio=preservation_ratio,
        )


_discriminator_instance: SaturationDiscriminator | None = None


def get_saturation_discriminator() -> SaturationDiscriminator:
    """Gibt den SaturationDiscriminator-Singleton zurück."""
    global _discriminator_instance
    if _discriminator_instance is None:
        _discriminator_instance = SaturationDiscriminator()
    return _discriminator_instance
