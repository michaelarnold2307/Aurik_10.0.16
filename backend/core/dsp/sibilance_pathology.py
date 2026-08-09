"""
Sibilance-Pathology-Classifier — Lücke 4 (v10.0.0.x)
====================================================

Klassifiziert Sibillanten-Zeitfenster in drei grundlegend verschiedene Zustände,
bevor phase_43 (ML-De-Esser) und phase_19 (DSP-De-Esser) eingreifen:

    NATURAL     — charakteristische starke Sibilanz des Sängers → Schutz, kein Eingriff
    MASKED_HISS — Bandrauschen schimmert durch S-Fenster → freq-selektive NR only
    DISTORTED   — Bandsättigungsverzerrung auf Frikativen → spektrale Reparatur des Klirr

ALGORITHMUS:
    1. Sibilanz-Segmenterkennung: Energie > threshold im Band [4–13 kHz] → S-Kandidaten
    2. THD-Analyse im S-Band: Klirr-Spektrum aller Harmonischen → DISTORTED wenn THD > 6 %
    3. Spectral-Flatness-Analyse: Rauschanteile in S-Band → MASKED_HISS wenn SFM > 0.65
    4. Energie-Stabilität: Natürliche S-Laute haben schnellen Anstieg, stabilen Pegel
       → NATURAL bei steiler Onset-Flanke ohne Klirr/Rauschen

Rückgabe: Liste von SibilanceSegment mit Typ, Zeitfenster, Empfehlung für phase_43.

Aufruf (non-blocking):
    from backend.core.dsp.sibilance_pathology import classify_sibilance_pathology
    segs = classify_sibilance_pathology(audio, sr=48000)
    # Injiziert via kwargs in phase_43: sibilance_pathology=segs

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0.x — Lücke 4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import scipy.signal as sp_sig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typen
# ---------------------------------------------------------------------------


class SibilanceType(str, Enum):
    """Pathologie-Klasse eines Sibilanz-Segments."""

    NATURAL = "natural"  # Charakteristische Sänger-Sibilanz → Schutz
    MASKED_HISS = "masked_hiss"  # Rauschen maskiert unter S-Laut → freq-selektive NR
    DISTORTED = "distorted"  # Sättigungsklirr → spektrale Reparatur


@dataclass
class SibilanceSegment:
    """Ein klassifiziertes Sibilanz-Zeitfenster."""

    start_s: float
    end_s: float
    sibilance_type: SibilanceType
    thd_ratio: float = 0.0  # Klirrfaktor im S-Band (0–1)
    sfm: float = 0.0  # Spectral-Flatness-Measure (0–1)
    onset_slope_db_ms: float = 0.0  # Anstiegssteilheit dB/ms
    recommended_action: str = ""
    """
    Empfehlung für phase_43:
      'protect'          NATURAL  → strength_cap = 0, kein Eingriff
      'nr_only'          MASKED_HISS → nur NR im S-Band-Fenster
      'spectral_repair'  DISTORTED → Klirr-Partial-Entfernung
    """

    def __post_init__(self) -> None:
        _ACTION_MAP = {
            SibilanceType.NATURAL: "protect",
            SibilanceType.MASKED_HISS: "nr_only",
            SibilanceType.DISTORTED: "spectral_repair",
        }
        if not self.recommended_action:
            self.recommended_action = _ACTION_MAP[self.sibilance_type]


# ---------------------------------------------------------------------------
# Interne DSP-Hilfsfunktionen
# ---------------------------------------------------------------------------

# S-Band Grenzen — alle relevanten Sibilanztypen
_SIB_LOW_HZ = 4_000.0
_SIB_HIGH_HZ = 13_000.0

# Minimale Segmentdauer (zu kurze Bursts = Transienten, kein S-Laut)
_MIN_SEGMENT_S = 0.030  # 30 ms

# THD-Schwellen
_THD_DISTORTED_THRESH = 0.06  # > 6 % → Sättigungsklirr
_SFM_HISS_THRESH = 0.60  # > 0.60 SFM → Rauschen dominiert


def _bandpass_energy(mono: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    """Hüllkurve der Energie in [low, high] Hz, 10 ms-Frames."""
    nyq = sr / 2.0
    sos = sp_sig.butter(4, [low / nyq, min(high / nyq, 0.98)], btype="band", output="sos")
    filtered = sp_sig.sosfiltfilt(sos, mono)
    frame_len = max(1, int(sr * 0.010))  # 10 ms
    n_frames = len(filtered) // frame_len
    energy = np.array([float(np.mean(filtered[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(n_frames)])
    return np.maximum(energy, 1e-30)  # type: ignore[no-any-return]


def _spectral_flatness(spectrum: np.ndarray) -> float:
    """Spectral Flatness Measure ∈ [0, 1] — 1 = weißes Rauschen."""
    s = np.abs(spectrum) + 1e-12
    log_mean = np.mean(np.log(s))
    arith_mean = np.mean(s)
    sfm = float(np.exp(log_mean) / (arith_mean + 1e-12))
    return float(np.clip(sfm, 0.0, 1.0))


def _thd_estimate(mono_segment: np.ndarray, sr: int, f0_hz: float) -> float:
    """
    Klirrfaktor-Schätzung für bekanntes f0 im S-Band-Segment.
    THD ≈ Energie(Harmonische 2–6) / Energie(Fundamental)
    Wenn kein f0 bekannt: Schätzung über spektrale Peakstruktur.
    """
    if len(mono_segment) < 64:
        return 0.0
    N = min(len(mono_segment), 4096)
    spectrum = np.fft.rfft(mono_segment[:N] * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    mag = np.abs(spectrum)

    if f0_hz <= 0.0:
        # Kein F0 bekannt: suche stärksten Peak im S-Band als Referenz
        sib_mask = (freqs >= _SIB_LOW_HZ) & (freqs <= _SIB_HIGH_HZ)
        if not np.any(sib_mask):
            return 0.0
        peak_idx = int(np.argmax(mag * sib_mask))
        f0_hz = float(freqs[peak_idx])

    if f0_hz < 100.0:
        return 0.0

    def _band_energy(center: float, width: float = 200.0) -> float:
        mask = (freqs >= center - width) & (freqs <= center + width)
        return float(np.sum(mag[mask] ** 2)) + 1e-30

    fund_energy = _band_energy(f0_hz)
    harm_energy = sum(_band_energy(f0_hz * k) for k in range(2, 7) if f0_hz * k < freqs[-1])
    thd = float(np.sqrt(harm_energy / fund_energy))
    return float(np.clip(thd, 0.0, 1.0))


def _onset_slope_db_ms(energy_frames: np.ndarray, onset_idx: int) -> float:
    """Anstiegssteilheit in dB/ms ab onset_idx (Fenster: 3 Frames = 30 ms)."""
    win = 3
    if onset_idx + win >= len(energy_frames):
        return 0.0
    e_start = energy_frames[onset_idx] + 1e-30
    e_end = energy_frames[min(onset_idx + win, len(energy_frames) - 1)] + 1e-30
    db_rise = 10.0 * np.log10(e_end / e_start)
    return float(db_rise / (win * 10.0))  # 10 ms pro Frame


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def classify_sibilance_pathology(
    audio: np.ndarray,
    sr: int = 48_000,
    f0_hz: float = 0.0,
    *,
    sib_low_hz: float = _SIB_LOW_HZ,
    sib_high_hz: float = _SIB_HIGH_HZ,
    energy_threshold_db: float = -42.0,
) -> list[SibilanceSegment]:
    """
    Klassifiziert Sibilanz-Segmente im Audio in NATURAL / MASKED_HISS / DISTORTED.

    Non-blocking: Exception → leere Liste (kein Crash, kein Veto in Phase 43/19).

    Args:
        audio:               Mono oder Stereo float32/64
        sr:                  Abtastrate (muss 48 000 sein für Phasen-Integration)
        f0_hz:               Vokal-Grundfrequenz für THD-Schätzung (0 = auto-detect)
        sib_low_hz:          Untere S-Band-Grenze (Hz)
        sib_high_hz:         Obere S-Band-Grenze (Hz)
        energy_threshold_db: Schwelle für Sibilanz-Erkennung (dBFS RMS)

    Returns:
        Liste von SibilanceSegment, chronologisch sortiert.
        Leer wenn: kein Vokalinhalt, zu kurzes Audio, Exception.
    """
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Mono-Extraktion
        if audio.ndim == 2:
            mono = audio.mean(axis=0).astype(np.float64)
        else:
            mono = audio.astype(np.float64)

        if len(mono) < int(sr * 0.1):  # < 100 ms → zu kurz
            return []

        # S-Band-Energie-Hüllkurve (10 ms Frames)
        sib_energy = _bandpass_energy(mono, sr, sib_low_hz, sib_high_hz)
        sib_energy_db = 10.0 * np.log10(sib_energy + 1e-30)
        frame_dur_s = 0.010  # 10 ms pro Frame

        # Sibilanz-Frames: Energie über Schwelle
        active = sib_energy_db > energy_threshold_db

        # Segmentierung: zusammenhängende aktive Runs
        segments: list[SibilanceSegment] = []
        in_segment = False
        seg_start = 0

        for i, is_active in enumerate(active):
            if is_active and not in_segment:
                seg_start = i
                in_segment = True
            elif not is_active and in_segment:
                seg_end = i
                in_segment = False
                _process_segment(
                    mono,
                    sr,
                    sib_energy,
                    sib_energy_db,
                    seg_start,
                    seg_end,
                    frame_dur_s,
                    f0_hz,
                    sib_low_hz,
                    sib_high_hz,
                    segments,
                )

        # Offenes Segment am Ende
        if in_segment:
            _process_segment(
                mono,
                sr,
                sib_energy,
                sib_energy_db,
                seg_start,
                len(active),
                frame_dur_s,
                f0_hz,
                sib_low_hz,
                sib_high_hz,
                segments,
            )

        logger.debug(
            "sibilance_pathology: %d segments — natural=%d masked_hiss=%d distorted=%d",
            len(segments),
            sum(1 for s in segments if s.sibilance_type == SibilanceType.NATURAL),
            sum(1 for s in segments if s.sibilance_type == SibilanceType.MASKED_HISS),
            sum(1 for s in segments if s.sibilance_type == SibilanceType.DISTORTED),
        )
        return segments

    except Exception as exc:
        logger.warning("§G23 sibilance_pathology: DSP-Ersatzpfad — %s", exc, exc_info=True)
        return []


def _process_segment(
    mono: np.ndarray,
    sr: int,
    sib_energy: np.ndarray,
    _sib_energy_db: np.ndarray,
    seg_start: int,
    seg_end: int,
    frame_dur_s: float,
    f0_hz: float,
    sib_low_hz: float,
    sib_high_hz: float,
    out: list[SibilanceSegment],
) -> None:
    """Analysiert ein Sibilanz-Segment und klassifiziert es."""
    dur_s = (seg_end - seg_start) * frame_dur_s
    if dur_s < _MIN_SEGMENT_S:
        return  # Zu kurz — Transient, kein S-Laut

    start_s = seg_start * frame_dur_s
    end_s = seg_end * frame_dur_s

    # Audiosamples extrahieren
    s0 = int(start_s * sr)
    s1 = min(int(end_s * sr), len(mono))
    segment_audio = mono[s0:s1]

    if len(segment_audio) < 32:
        return

    # 1. Spectral Flatness Measure im S-Fenster
    N = min(len(segment_audio), 2048)
    spectrum = np.fft.rfft(segment_audio[:N] * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    sib_mask = (freqs >= sib_low_hz) & (freqs <= sib_high_hz)
    sfm = _spectral_flatness(spectrum[sib_mask]) if np.any(sib_mask) else 0.5

    # 2. THD-Schätzung
    thd = _thd_estimate(segment_audio, sr, f0_hz)

    # 3. Onset-Steilheit (dB/ms)
    onset_slope = _onset_slope_db_ms(sib_energy, seg_start)

    # Klassifikation (hierarchisch)
    if thd > _THD_DISTORTED_THRESH:
        sib_type = SibilanceType.DISTORTED
    elif sfm > _SFM_HISS_THRESH:
        sib_type = SibilanceType.MASKED_HISS
    else:
        # Kein Klirr, kein dominierendes Rauschen → natürliche Sibilanz
        sib_type = SibilanceType.NATURAL

    out.append(
        SibilanceSegment(
            start_s=start_s,
            end_s=end_s,
            sibilance_type=sib_type,
            thd_ratio=thd,
            sfm=sfm,
            onset_slope_db_ms=onset_slope,
        )
    )


# ---------------------------------------------------------------------------
# Aggregat-Hilfsfunktion für Phase-Integration
# ---------------------------------------------------------------------------


def get_sibilance_pathology_summary(
    segments: list[SibilanceSegment],
) -> dict[str, Any]:
    """
    Kompaktes Summary für Phase-43/19-Integration.

    Returns:
        dict mit:
          'dominant_type': häufigster Typ als str
          'natural_fraction': Anteil NATURAL-Segmente (0–1)
          'distorted_fraction': Anteil DISTORTED-Segmente (0–1)
          'protected_zones': [(start_s, end_s)] für NATURAL-Segmente
          'repair_zones': [(start_s, end_s)] für DISTORTED-Segmente
    """
    if not segments:
        return {
            "dominant_type": "unknown",
            "natural_fraction": 0.0,
            "distorted_fraction": 0.0,
            "protected_zones": [],
            "repair_zones": [],
        }

    n_total = len(segments)
    n_natural = sum(1 for s in segments if s.sibilance_type == SibilanceType.NATURAL)
    n_distorted = sum(1 for s in segments if s.sibilance_type == SibilanceType.DISTORTED)
    n_hiss = n_total - n_natural - n_distorted

    type_counts = {
        SibilanceType.NATURAL: n_natural,
        SibilanceType.MASKED_HISS: n_hiss,
        SibilanceType.DISTORTED: n_distorted,
    }
    dominant_type = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]

    return {
        "n_total": n_total,
        "n_natural": n_natural,
        "n_masked_hiss": n_hiss,
        "n_distorted": n_distorted,
        "dominant_type": dominant_type.value,
        "natural_fraction": n_natural / n_total,
        "distorted_fraction": n_distorted / n_total,
        "protected_zones": [(s.start_s, s.end_s) for s in segments if s.sibilance_type == SibilanceType.NATURAL],
        "repair_zones": [(s.start_s, s.end_s) for s in segments if s.sibilance_type == SibilanceType.DISTORTED],
    }
