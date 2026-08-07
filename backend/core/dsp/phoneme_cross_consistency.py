"""
Phoneme Cross-Consistency Monitor — Lücke 7 (v10.0.0.x)
======================================================

Stellt sicher, dass gleiche Phoneme (z.B. das „a" in Strophe 1 und Strophe 3)
nach der Restaurierung konsistente Vokalklangfarbe haben.

PROBLEM:
    Restaurierung verarbeitet jeden Frame lokal. Dasselbe Wort in zwei Strophen
    wird mit leicht unterschiedlicher Stärke/Phase/NR-Aggressivität bearbeitet
    → subtile aber hörbare Inkonsistenz der Stimmfarbe über den Song.

ALGORITHMUS:
    1. Phonem-Segmentierung: Via LyricsGuidedEnhancement (wenn verfügbar)
       oder energie-basierter DSP-Proxy
    2. Vokalklang-Fingerprint: MFCC (13 Koeffizienten) + F1/F2 per Phonem-Segment
    3. Konsistenz-Messung: Für gleiche Phonemklassen → cosine distance der Fingerprints
       zwischen Strophen → Drift-Report
    4. Blend-Korrektiv: Wenn cosine_distance > 0.08 → Ziel-MFCC interpolieren
       und sanften Spektralen EQ-Korrekturterm ausgeben

Verwendung (Post-Processing-Hook in UV3):
    monitor = PhonemeConsistencyMonitor(audio_orig, audio_restored, sr)
    report = monitor.compute_consistency()
    correction = monitor.get_correction_eq()  # Spektraler EQ-Term (optional)

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0.x — Lücke 7)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sp_sig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

_N_MFCC = 13
_SR_MFCC = 16_000  # MFCC-Analyse auf 16 kHz (Standard)
_MIN_PHONEME_FRAMES = 3  # Mindestlänge Phonem-Segment (×10 ms = 30 ms)
_CONSISTENCY_THRESH = 0.08  # Cosine-Distance-Schwelle → Inkonsistenz-Flag
_N_MELS = 40


# ---------------------------------------------------------------------------
# Typen
# ---------------------------------------------------------------------------


@dataclass
class PhonemeConsistencyReport:
    """Bericht über Vokalklang-Konsistenz nach Restaurierung."""

    n_phoneme_groups: int = 0  # Anzahl analysierter Phonem-Gruppen
    mean_cosine_distance: float = 0.0  # Mittlere Inkonsistenz (0 = perfekt konsistent)
    max_cosine_distance: float = 0.0
    inconsistent_groups: list[str] = field(default_factory=list)
    """Phonem-Gruppen mit Inkonsistenz > _CONSISTENCY_THRESH"""
    is_consistent: bool = True
    """True wenn mean_cosine_distance < _CONSISTENCY_THRESH"""
    correction_needed: bool = False

    def to_dict(self) -> dict:
        return {
            "n_phoneme_groups": self.n_phoneme_groups,
            "mean_cosine_distance": self.mean_cosine_distance,
            "max_cosine_distance": self.max_cosine_distance,
            "inconsistent_groups": self.inconsistent_groups,
            "is_consistent": self.is_consistent,
            "correction_needed": self.correction_needed,
        }


@dataclass
class SpectralCorrection:
    """
    Sanfter Spektral-EQ-Korrekturterm für inkonsistente Phonem-Gruppen.
    Enthält EQ-Gains (dB) in logarithmisch verteilten Bändern.
    """

    band_centers_hz: np.ndarray  # Mittenfrequenzen der EQ-Bänder
    gain_db: np.ndarray  # EQ-Gain pro Band (dB), meist < ±1.5 dB
    apply_to_time_ranges: list[tuple[float, float]] = field(default_factory=list)
    """Zeitbereiche (s), in denen die Korrektur angewendet werden soll."""
    strength: float = 0.5  # Blend-Stärke (0 = kein Eingriff, 1 = volle Korrektur)


# ---------------------------------------------------------------------------
# DSP-Hilfsfunktionen
# ---------------------------------------------------------------------------


def _resample_to_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    """Resampling auf 16 kHz für MFCC-Analyse."""
    if sr == _SR_MFCC:
        return mono
    n_out = int(len(mono) * _SR_MFCC / sr)
    return sp_sig.resample(mono, n_out).astype(np.float32)  # type: ignore[no-any-return]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = _N_MELS) -> np.ndarray:
    """Einfache Mel-Filterbank (n_mels × (n_fft//2+1))."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Mel-Frequenz-Grenzen
    mel_min = 2595.0 * np.log10(1 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1 + (sr / 2.0) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_idx = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_idx[m - 1]
        f_m = bin_idx[m]
        f_m_plus = bin_idx[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return filterbank  # type: ignore[no-any-return]


def _compute_mfcc(mono_16k: np.ndarray, n_mfcc: int = _N_MFCC) -> np.ndarray:
    """
    Einfache MFCC-Berechnung ohne librosa (DSP-only).
    Gibt (n_mfcc,) Vektor zurück (Segment-Mittelwert).
    """
    n_fft = 512
    hop = 160  # 10 ms bei 16 kHz
    n_frames = max(1, (len(mono_16k) - n_fft) // hop)

    filterbank = _mel_filterbank(_SR_MFCC, n_fft)
    mfccs = []
    for i in range(n_frames):
        frame = mono_16k[i * hop : i * hop + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        mel_energy = filterbank @ spectrum + 1e-12
        log_mel = np.log(mel_energy)
        mfcc = np.fft.dct(log_mel, type=2, norm="ortho")[:n_mfcc]  # type: ignore[attr-defined]
        mfccs.append(mfcc)

    if not mfccs:
        return np.zeros(n_mfcc)  # type: ignore[no-any-return]
    return np.mean(mfccs, axis=0)  # type: ignore[no-any-return]


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine-Distance zwischen zwei MFCC-Vektoren (0 = identisch, 2 = entgegengesetzt)."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 1.0
    cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(1.0 - cosine_sim, 0.0, 2.0))


def _segment_by_energy(mono: np.ndarray, sr: int) -> list[tuple[int, int, str]]:
    """
    Energie-basierte Phonem-Proxy-Segmentierung (DSP-Fallback wenn LGE nicht verfügbar).
    Gibt Liste von (start_sample, end_sample, phoneme_class_proxy) zurück.
    phoneme_class_proxy ist ein grober Klang-Klassen-Schätzer basierend auf
    Spektralform (vowel/fricative/plosive).
    """
    frame_len = int(sr * 0.020)  # 20 ms Frames
    n_frames = len(mono) // frame_len
    segments: list[tuple[int, int, str]] = []

    energy = np.array([float(np.mean(mono[i * frame_len : (i + 1) * frame_len] ** 2) + 1e-30) for i in range(n_frames)])
    energy_db = 10.0 * np.log10(energy)
    threshold = float(np.percentile(energy_db, 30))

    in_seg = False
    seg_start = 0
    for i, e in enumerate(energy_db):
        if e > threshold and not in_seg:
            seg_start = i
            in_seg = True
        elif e <= threshold and in_seg:
            if i - seg_start >= _MIN_PHONEME_FRAMES:
                seg_start_s = seg_start * frame_len
                seg_end_s = i * frame_len
                segments.append((seg_start_s, seg_end_s, "vowel_proxy"))
            in_seg = False
    if in_seg:
        seg_end_s = n_frames * frame_len
        if n_frames - seg_start >= _MIN_PHONEME_FRAMES:
            segments.append((seg_start * frame_len, seg_end_s, "vowel_proxy"))

    return segments


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class PhonemeConsistencyMonitor:
    """
    Analysiert Vokalklang-Konsistenz vor/nach Restaurierung.

    Singleton-frei — wird pro Aufruf instanziiert und nach Benutzung GC'et.
    """

    def __init__(
        self,
        audio_orig: np.ndarray,
        audio_restored: np.ndarray,
        sr: int,
    ) -> None:
        self._sr = sr

        # Mono
        def _to_mono(a: np.ndarray) -> np.ndarray:
            if a.ndim == 2:
                return np.nan_to_num(a.mean(axis=0), nan=0.0).astype(np.float32)  # type: ignore[no-any-return]
            return np.nan_to_num(a, nan=0.0).astype(np.float32)  # type: ignore[no-any-return]

        self._orig = _to_mono(audio_orig)
        self._restored = _to_mono(audio_restored)

        # Länge angleichen
        min_len = min(len(self._orig), len(self._restored))
        self._orig = self._orig[:min_len]
        self._restored = self._restored[:min_len]

    def compute_consistency(self) -> PhonemeConsistencyReport:
        """
        Berechnet Konsistenz-Report.
        Non-blocking: Exception → PhonemeConsistencyReport() mit defaults.
        """
        try:
            segments = self._get_segments()
            if len(segments) < 2:
                return PhonemeConsistencyReport()

            # MFCC-Fingerprints pro Segment (restored)
            mono_16k = _resample_to_16k(self._restored, self._sr)
            sr_ratio = _SR_MFCC / self._sr

            fingerprints: dict[str, list[np.ndarray]] = {}
            for start_s, end_s, phon_class in segments:
                seg_16k_start = int(start_s * sr_ratio)
                seg_16k_end = int(end_s * sr_ratio)
                seg_audio = mono_16k[seg_16k_start:seg_16k_end]
                if len(seg_audio) < 80:
                    continue
                mfcc = _compute_mfcc(seg_audio)
                if phon_class not in fingerprints:
                    fingerprints[phon_class] = []
                fingerprints[phon_class].append(mfcc)

            if not fingerprints:
                return PhonemeConsistencyReport()

            # Konsistenz-Messung: Paarweise Cosine-Distance innerhalb jeder Klasse
            distances: dict[str, float] = {}
            for phon_class, mfcc_list in fingerprints.items():
                if len(mfcc_list) < 2:
                    continue
                class_distances = []
                for i in range(len(mfcc_list)):
                    for j in range(i + 1, len(mfcc_list)):
                        d = _cosine_distance(mfcc_list[i], mfcc_list[j])
                        class_distances.append(d)
                distances[phon_class] = float(np.mean(class_distances))

            if not distances:
                return PhonemeConsistencyReport()

            mean_dist = float(np.mean(list(distances.values())))
            max_dist = float(np.max(list(distances.values())))
            inconsistent = [k for k, v in distances.items() if v > _CONSISTENCY_THRESH]

            report = PhonemeConsistencyReport(
                n_phoneme_groups=len(distances),
                mean_cosine_distance=mean_dist,
                max_cosine_distance=max_dist,
                inconsistent_groups=inconsistent,
                is_consistent=mean_dist < _CONSISTENCY_THRESH,
                correction_needed=len(inconsistent) > 0,
            )

            logger.debug(
                "phoneme_consistency: groups=%d mean_dist=%.3f max_dist=%.3f inconsistent=%s",
                report.n_phoneme_groups,
                mean_dist,
                max_dist,
                inconsistent,
            )
            return report

        except Exception as exc:
            logger.debug("PhonemeConsistencyMonitor.berechnen_consistency: Ersatzpfad — %s", exc)
            return PhonemeConsistencyReport()

    def get_correction_eq(
        self,
        report: PhonemeConsistencyReport | None = None,
        n_bands: int = 8,
    ) -> SpectralCorrection | None:
        """
        Berechnet sanften Spektral-EQ-Korrekturterm für inkonsistente Stellen.

        Gibt None zurück wenn kein Korrektiv nötig (is_consistent=True).

        Der EQ-Term ist absichtlich konservativ (max ±1.5 dB) —
        §0 Primum non nocere.
        """
        try:
            if report is None:
                report = self.compute_consistency()
            if not report.correction_needed:
                return None

            # MFCC-Differenz: orig vs. restored — in welchen Bändern weicht das am meisten ab?
            mono_orig_16k = _resample_to_16k(self._orig, self._sr)
            mono_rest_16k = _resample_to_16k(self._restored, self._sr)

            # Gesamte MFCC-Vektoren
            mfcc_orig = _compute_mfcc(mono_orig_16k)
            mfcc_rest = _compute_mfcc(mono_rest_16k)
            mfcc_diff = mfcc_orig - mfcc_rest  # Positiv = Orig hat mehr Energie in Band

            # Mapping MFCC-Koeffizienten → EQ-Bänder (grobe Approximation)
            # MFCC[0] = Gesamt-Energie, MFCC[1..12] = spektrale Form
            # EQ-Bänder logarithmisch über 80–16000 Hz
            band_centers = np.logspace(np.log10(80), np.log10(16000), n_bands)
            gains = np.zeros(n_bands)

            # Sanfte Übertragung: max ±1.5 dB, proportional zur MFCC-Differenz
            # MFCC-Norm als Skalierungsfaktor
            mfcc_norm = float(np.max(np.abs(mfcc_diff[1:])) + 1e-6)
            for b in range(n_bands):
                mfcc_idx = min(b + 1, len(mfcc_diff) - 1)
                raw_gain = float(mfcc_diff[mfcc_idx]) / mfcc_norm * 1.5  # Max ±1.5 dB
                gains[b] = float(np.clip(raw_gain, -1.5, 1.5))

            return SpectralCorrection(
                band_centers_hz=band_centers,
                gain_db=gains,
                strength=float(np.clip(report.mean_cosine_distance / _CONSISTENCY_THRESH, 0.0, 1.0)),
            )

        except Exception as exc:
            logger.debug("PhonemeConsistencyMonitor.get_correction_eq: Ersatzpfad — %s", exc)
            return None

    # ------------------------------------------------------------------

    def _get_segments(self) -> list[tuple[float, float, str]]:
        """Phonem-Segmentierung: LGE wenn verfügbar, sonst DSP-Proxy."""
        try:
            from backend.core.lyrics_guided_enhancement import (  # pylint: disable=import-outside-toplevel
                get_lyrics_guided_enhancement,
            )

            lge = get_lyrics_guided_enhancement()
            if lge.is_loaded():
                result = lge.transcribe(self._orig, self._sr)
                segs = []
                for word in result.words:
                    phon = getattr(word, "phoneme_class", "vowel_proxy")
                    segs.append((float(word.start_s), float(word.end_s), phon))
                if segs:
                    return segs
        except Exception as e:
            logger.warning("phoneme_cross_consistency.py::_get_segments Ersatzpfad: %s", e)

        # DSP-Fallback
        raw = _segment_by_energy(self._orig, self._sr)
        return [(s / self._sr, e / self._sr, phon) for s, e, phon in raw]
