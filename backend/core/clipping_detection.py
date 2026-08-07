#!/usr/bin/env python3
"""
Clipping-Erkennung — §6.3 Spec: CLIPPING vs. SOFT_SATURATION Diskriminierung.

Critical distinction (copilot-instructions.md §6.3):
    SOFT_SATURATION = Tube/Tape character → PRESERVE
    CLIPPING        = Amplitude damage    → REPAIR

Algorithm:
    1. flat_tops_pct: fraction of samples at hard-clip boundary (|x| ≥ 0.999)
    2. THD analysis via STFT-based harmonic energy estimation:
       - THD_odd  = energy at odd harmonics  (clipping signature: square-wave-like)
       - THD_even = energy at even harmonics (saturation signature: tanh-like)
    3. Classification rule (normative per §6.3):
       CLIPPING:        flat_tops > 0.1 % AND THD_odd > THD_even × 1.5
       SOFT_SATURATION: flat_tops < 0.1 % OR  THD_even > THD_odd

Singleton: thread-safe, Double-Checked Locking (§3.x).

Author: Aurik Development Team
Version: 10.0.0
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np

from backend.core.core_utils import fft_autocorr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (normative per §6.3)
# ---------------------------------------------------------------------------

FLAT_TOPS_THRESHOLD_PCT: float = 0.1  # > this → potential hard clipping
FLAT_TOPS_STRONG_CLIP_PCT: float = 1.5  # > this → CLIPPING unabhängig von THD (polyphon-sicher)
FLAT_TOPS_CLIP_BOUNDARY: float = 0.999  # samples within this range are "at ceiling"

# Sub-Ceiling-Clipping-Erkennung (Hard-Clipping unter 0.999, z.B. ±0.85 / ±0.92)
# np.clip / Hardware erzeugt viele identische float32-Werte exakt am Clip-Level.
# Soft-Saturation (tanh) oder komprimiertes Audiomaterial: float32-Plateau-
# Rounding erzeugt ~80 identische Maximalwerte je Sekunde (≈ 0.17 %).
# Hard-Clipping bei 5 % geclippt: ~2 400 identische Werte (5 %) — deutlich mehr.
SUBCEIL_LOW: float = 0.75  # Min. abs_max für Sub-Ceiling-Prüfung
SUBCEIL_MIN_IDENTICAL: int = 200  # Min. absolute identische Samples am Peak
SUBCEIL_MIN_IDENTICAL_PCT: float = 0.5  # Min. prozentualer Anteil am Gesamtsignal
THD_ODD_DOMINANCE_FACTOR: float = 1.5  # THD_odd > THD_even × this → CLIPPING
ANALYSIS_WINDOW_SAMPLES: int = 2048  # FFT window size for harmonic analysis
ANALYSIS_HOP_SAMPLES: int = 512  # hop between analysis windows
FUNDAMENTAL_MIN_HZ: float = 40.0  # lowest fundamental to search (E1 ~ 41 Hz)
FUNDAMENTAL_MAX_HZ: float = 800.0  # highest fundamental (G5 ~ 784 Hz)
MAX_HARMONIC_ORDER: int = 10  # analyse up to this harmonic order
HARMONIC_BIN_RADIUS: int = 1  # bins around harmonic centre to include


class ClippingType(Enum):
    """Discrimination result per §6.3."""

    CLIPPING = "clipping"  # Hard amplitude clipping — REPAIR (→ phase_23)
    SOFT_SATURATION = "soft_saturation"  # Tube/Tape saturation character — PRESERVE


@dataclass
class ClippingAnalysisResult:
    """Full analysis result returned by classify_clipping()."""

    clipping_type: ClippingType
    flat_tops_pct: float  # % of samples at ±1.0 boundary
    thd_odd: float  # Total odd-harmonic energy (summed, normalised)
    thd_even: float  # Total even-harmonic energy (summed, normalised)
    confidence: float  # 0.0–1.0 — distance from decision boundary
    is_clipping: bool  # True when clipping_type == CLIPPING
    sub_ceiling_level: float = 0.0  # Amplitude-Histogramm Sub-Ceiling-Clip-Level (0.0 wenn nicht erkannt)

    @property
    def should_repair(self) -> bool:
        """True when the pipeline should activate Clipping-Repair (phase_23)."""
        return self.is_clipping

    @property
    def should_preserve(self) -> bool:
        """True when the pipeline should skip Clipping-Repair and preserve saturation."""
        return not self.is_clipping


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertiert (N,) or (N, C) audio to 1-D mono float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def _flat_tops_pct(mono: np.ndarray, boundary: float = FLAT_TOPS_CLIP_BOUNDARY) -> float:
    """
    Gibt the percentage of samples sitting at or above the clip boundary zurück.

    A sample is considered 'at the ceiling' when |x| >= boundary.
    For hard digital clipping this can exceed several percent.
    For analogue soft saturation this is typically < 0.05 %.
    """
    if len(mono) == 0:
        return 0.0
    ratio = float(np.mean(np.abs(mono) >= boundary))
    return ratio * 100.0


def _find_dominant_fundamental_hz(mono: np.ndarray, sr: int) -> float | None:
    """
    Schätzt dominant fundamental frequency via normalised autocorrelation (AMDF).

    Returns None when no clear fundamental within [FUNDAMENTAL_MIN_HZ, FUNDAMENTAL_MAX_HZ]
    is detectable (polyphonic / noise-dominated material).
    """
    n = min(len(mono), 4096)
    if n < 256:
        return None

    segment = mono[:n].astype(np.float64)
    # Normalised autocorrelation — FFT-based O(N log N)
    corr = fft_autocorr(segment)
    if np.max(np.abs(corr)) < 1e-8:
        return None
    corr = corr / (corr[0] + 1e-12)

    lag_min = max(1, int(sr / FUNDAMENTAL_MAX_HZ))
    lag_max = min(n - 1, int(sr / FUNDAMENTAL_MIN_HZ))
    if lag_max <= lag_min:
        return None

    search_region = corr[lag_min : lag_max + 1]
    peak_rel = int(np.argmax(search_region))
    peak_lag = lag_min + peak_rel
    peak_val = float(corr[peak_lag])

    # Require a strong correlation peak to confirm a clear fundamental
    if peak_val < 0.40:
        return None

    return float(sr) / float(peak_lag)


def _harmonic_energies(
    fft_mag: np.ndarray,
    fundamental_hz: float,
    sr: int,
    fft_size: int,
) -> tuple[float, float]:
    """
    Sum FFT magnitude energy at odd and even harmonics above the fundamental.

    Returns (thd_odd_energy, thd_even_energy) normalised by fundamental energy.
    Energy is summed over bins [k-HARMONIC_BIN_RADIUS, k+HARMONIC_BIN_RADIUS]
    around each harmonic centre bin.
    """
    bin_hz = sr / fft_size
    fundamental_bin = max(1, round(fundamental_hz / bin_hz))

    # Fundamental energy (normalisation)
    f_lo = max(0, fundamental_bin - HARMONIC_BIN_RADIUS)
    f_hi = min(len(fft_mag), fundamental_bin + HARMONIC_BIN_RADIUS + 1)
    fundamental_energy = float(np.sum(fft_mag[f_lo:f_hi] ** 2)) + 1e-12

    odd_energy = 0.0
    even_energy = 0.0

    for n in range(2, MAX_HARMONIC_ORDER + 1):
        harmonic_bin = round(fundamental_hz * n / bin_hz)
        if harmonic_bin >= len(fft_mag):
            break
        lo = max(0, harmonic_bin - HARMONIC_BIN_RADIUS)
        hi = min(len(fft_mag), harmonic_bin + HARMONIC_BIN_RADIUS + 1)
        energy = float(np.sum(fft_mag[lo:hi] ** 2))
        if n % 2 == 0:
            even_energy += energy
        else:
            odd_energy += energy

    return odd_energy / fundamental_energy, even_energy / fundamental_energy


def _polyphonic_odd_even_estimate(mono: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Broadband odd/even harmonic ratio for polyphonic material without clear fundamental.

    Approach: window the signal, detect tonal peaks in each frame, and accumulate
    harmonic energy estimates over all detected peaks. Returns averaged ratio.

    Falls back to (1.0, 1.0) — neutral — if no tonal peaks are reliably found.
    """
    fft_size = ANALYSIS_WINDOW_SAMPLES
    window = np.hanning(fft_size)
    n_frames = max(1, (len(mono) - fft_size) // ANALYSIS_HOP_SAMPLES + 1)
    n_frames = min(n_frames, 32)  # cap at 32 frames for speed

    odd_acc = 0.0
    even_acc = 0.0
    valid_frames = 0

    for k in range(n_frames):
        start = k * ANALYSIS_HOP_SAMPLES
        end = start + fft_size
        if end > len(mono):
            break
        frame = mono[start:end] * window
        if np.max(np.abs(frame)) < 0.01:
            continue  # silent frame

        fft_mag = np.abs(np.fft.rfft(frame, n=fft_size))
        # Find peak (potential local fundamental)
        peak_bin = int(np.argmax(fft_mag[1:]) + 1)  # skip DC
        peak_hz = float(peak_bin) * sr / fft_size

        if peak_hz < FUNDAMENTAL_MIN_HZ or peak_hz > FUNDAMENTAL_MAX_HZ:
            continue

        o, e = _harmonic_energies(fft_mag, peak_hz, sr, fft_size)
        odd_acc += o
        even_acc += e
        valid_frames += 1

    if valid_frames == 0:
        return 1.0, 1.0

    return odd_acc / valid_frames, even_acc / valid_frames


def _compute_thd(mono: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Berechnet (thd_odd, thd_even) for the audio signal.

    Strategy:
    1. Try single stable fundamental via autocorrelation (works for voice/single-instrument).
    2. Fall back to polyphonic frame-by-frame analysis.
    """
    fundamental_hz = _find_dominant_fundamental_hz(mono, sr)

    if fundamental_hz is not None:
        # Use global FFT for single-fundamental case
        n = min(len(mono), 16384)
        fft_mag = np.abs(np.fft.rfft(mono[:n], n=n))
        return _harmonic_energies(fft_mag, fundamental_hz, sr, n)
    else:
        return _polyphonic_odd_even_estimate(mono, sr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_clipping(audio: np.ndarray, sr: int) -> ClippingType:
    """
    Discriminate CLIPPING from SOFT_SATURATION via harmonic analysis (§6.3).

    Classification rules (normative):
        CLIPPING:        flat_tops > 0.1 % AND THD_odd > THD_even × 1.5
        SOFT_SATURATION: flat_tops < 0.1 % OR  THD_even > THD_odd

    SOFT_SATURATION → Pipeline SKIPS clipping repair completely.
    CLIPPING        → Pipeline activates phase_23 (spectral repair / declipping).

    Args:
        audio: Audio signal as float32 numpy array, shape (N,) or (N, 2).
               Expected range [-1.0, 1.0].
        sr:    Sample rate (any valid rate, e.g. 44100, 48000).

    Returns:
        ClippingType.CLIPPING or ClippingType.SOFT_SATURATION.
    """
    return analyse_clipping(audio, sr).clipping_type


# Band-Pile-Ratio-Schwelle für Methode 2 (DAW-Brickwall-Limiter-Erkennung).
# DAW-Limiter häuft Samples in einem schmalen Band nahe dem Clip-Level an
# (natürliche Signale haben eine flachere Amplitudenverteilung nahe dem Maximum).
# Kalibrierung (2s polyphones Signal bei 48 kHz, 48% Overdrive):
#   Sinus 440 Hz (kein Clip):           band_pile_ratio ≈ 3.1  → nicht erkannt
#   tanh(2.5×sin) ~0.987:               band_pile_ratio ≈ 4.2  → nicht erkannt
#   Musik polyphon (kein Clip):          band_pile_ratio ≈ 2.7  → nicht erkannt
#   DAW-Limiter @0.88, ±0.003 Streuung: band_pile_ratio ≈ 23   → erkannt ✓
BAND_PILE_RATIO_THRESHOLD: float = 8.0


def detect_sub_ceiling_clipping(audio: np.ndarray, amplitude_bins: int = 1000) -> tuple[bool, float]:
    """Erkennt Sub-Ceiling-Clipping durch zwei komplementäre Methoden.

    Methode 1 — Adjacent-Ratio-Histogramm (präzise, für exaktes np.clip / Hardware):
    Hard-Clipping setzt Samples auf EXAKT einen float32-Wert → massiver Spike im
    Argmax-Bin. Adjacent-Ratio ≥ 20 → Hard-Clip erkannt.
        tanh(2.5×sin), 0.5 s  → ratio ≈  4.8  → nicht erkannt
        Hard-Clip @0.88, 2 s  → ratio ≈ 3500  → erkannt

    Methode 2 — Band-Pile-Ratio (DAW-Brickwall-Limiter / Loudness-War):
    DAW-Limiter häuft Samples in einem schmalen Band nahe dem Clip-Level an
    (auch wenn Samples leicht gestreut sind, z.B. ±0.003). Die Ratio aus
    Samples im engen Peak-Band [peak-0.005, peak] vs. dem breiteren Band darunter
    [peak-0.02, peak-0.005] unterscheidet Clipping (ratio ≈ 23) von natürlichen
    Signalen (Sinus: 3.1; tanh: 4.2; komprimierte Musik: 2.7).
        Sinus 440 Hz (kein Clip):           ratio ≈  3.1  → nicht erkannt
        tanh(2.5×sin) ~0.987:               ratio ≈  4.2  → nicht erkannt
        Musik polyphon (kein Clip, @0.88):   ratio ≈  2.7  → nicht erkannt
        DAW-Limiter @0.88 (±0.003 Streuung): ratio ≈ 23.0  → erkannt ✓

    Returns:
        (is_sub_ceiling_clip, clip_level) — clip_level = 0.0 wenn kein Clipping
    """
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=0 if audio.shape[0] < audio.shape[1] else 1)
    mono = np.asarray(mono, dtype=np.float32)
    abs_audio = np.abs(mono)
    peak = float(np.percentile(abs_audio, 99.9))

    # Nur wenn Peak im Sub-Ceiling-Bereich (0.80 bis 0.998)
    if peak < 0.80 or peak > 0.998:
        return False, 0.0

    # Histogramm im oberen 20%-Amplitudenbereich
    search_min = peak * 0.80
    upper_vals = abs_audio[abs_audio >= search_min]
    if len(upper_vals) < 100:
        return False, 0.0

    hist, edges = np.histogram(upper_vals, bins=amplitude_bins, range=(search_min, peak + 1e-6))
    centers = (edges[:-1] + edges[1:]) / 2

    # Fokus: Top-5%-Bereich des Histogramms
    top_5pct_idx = int(len(hist) * 0.95)
    top_region = hist[top_5pct_idx:]
    if len(top_region) < 5:
        return False, 0.0

    # Argmax-Bin (potenzieller Clipping-Spike) und zweit-häufigster Bin
    sorted_top_desc = np.sort(top_region)[::-1]
    top_bin_count = float(sorted_top_desc[0])
    second_bin_count = float(sorted_top_desc[1]) if len(sorted_top_desc) > 1 else 1.0

    if top_bin_count < 10:
        return False, 0.0

    # Adjacent-Ratio: Hard-Clipping-Spike = top_bin ≫ second_bin.
    # Soft-Saturation: top_bin ≈ 2–10 × second_bin (glatter Anstieg).
    adjacent_ratio = top_bin_count / (second_bin_count + 0.5)

    # Schwellwert: kalibriert an 440 Hz-Sinussignalen
    # (tanh → 4.8; Hard-Clip → 3500; sicherer Trennbereich 20–500)
    ADJACENT_RATIO_THRESHOLD = 20.0

    if adjacent_ratio >= ADJACENT_RATIO_THRESHOLD:
        top_peak_abs_idx = top_5pct_idx + int(np.argmax(top_region))
        clip_level = float(centers[top_peak_abs_idx])
        return True, clip_level

    # --- Methode 2: Band-Pile-Ratio (DAW-Brickwall-Limiter-Erkennung) ---
    # DAW-Limiter häuft Samples im schmalen Band [peak-0.005, peak] an.
    # Band 1 (Pile):    [peak - 0.005, peak]       — 0.5%-Streifen nahe dem Maximum
    # Band 2 (Natural): [peak - 0.020, peak-0.005] — 1.5%-Streifen darunter (3× breiter)
    # Normiertes Verhältnis: b1 / (b2/3). Clipping: ~23; natürlich: ~2.7–4.2.
    if len(abs_audio) >= 512:
        b1 = float(np.sum((abs_audio >= peak - 0.005) & (abs_audio <= peak)))
        b2 = float(np.sum((abs_audio >= peak - 0.020) & (abs_audio < peak - 0.005)))
        band_pile_ratio = b1 / (b2 / 3.0 + 0.5)
        logger.debug(
            "erkennen_sub_ceiling M2: band_pile_Verhaeltnis=%.2f (b1=%.0f, b2=%.0f, peak=%.4f)",
            band_pile_ratio,
            b1,
            b2,
            peak,
        )
        if band_pile_ratio >= BAND_PILE_RATIO_THRESHOLD:
            logger.info(
                "erkennen_sub_ceiling: Band-Pile-Clip erkannt (Verhaeltnis=%.1f ≥ %.1f, peak=%.4f)",
                band_pile_ratio,
                BAND_PILE_RATIO_THRESHOLD,
                peak,
            )
            return True, peak

    return False, 0.0


def analyse_clipping(audio: np.ndarray, sr: int) -> ClippingAnalysisResult:
    """
    Full clipping analysis with intermediate metrics.

    Returns ClippingAnalysisResult with flat_tops_pct, thd_odd, thd_even,
    confidence, and the final ClippingType decision.

    Args:
        audio: Audio signal, float32, shape (N,) or (N, C).
        sr:    Sample rate (any valid rate, e.g. 44100, 48000).
    """
    if sr < 8000 or sr > 192000:
        logger.warning("ClippingDetector: unusual SR=%d, results may be unreliable", sr)

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    mono = _to_mono(audio)

    if len(mono) == 0 or np.max(np.abs(mono)) < 1e-8:
        # Silence or empty signal — no clipping possible
        return ClippingAnalysisResult(
            clipping_type=ClippingType.SOFT_SATURATION,
            flat_tops_pct=0.0,
            thd_odd=0.0,
            thd_even=0.0,
            confidence=1.0,
            is_clipping=False,
        )

    # --- Step 1: flat top detection ---
    flat_pct = _flat_tops_pct(mono)
    logger.debug("ClippingDetector: flat_tops_pct=%.4f%%", flat_pct)

    # --- Step 1b: Sub-Ceiling-Clipping via identische float32-Samples ---
    # Erkennt Hard-Clipping unterhalb der digitalen Ceiling (abs_max < 0.999),
    # z.B. Loudness-War-Clipping bei ±0.85 oder ±0.92.
    # np.clip / Hardware-Clipping: viele Samples exakt auf denselben float32-Wert
    # gesetzt → deutlich mehr identische Maximalwerte als bei Soft-Saturation.
    # tanh/Kompression: float32-Plateau-Rounding → ~0.17 % identische Maximalwerte.
    # Guard: ≥ 200 identische Samples UND ≥ 0.5 % des Gesamtsignals.
    _sub_ceiling_clip = False
    if flat_pct < FLAT_TOPS_THRESHOLD_PCT:
        _abs_max = float(np.max(np.abs(mono)))
        if SUBCEIL_LOW <= _abs_max < FLAT_TOPS_CLIP_BOUNDARY:
            _n_total = len(mono)
            _n_identical = int(np.sum(np.abs(mono) == np.float32(_abs_max)))
            _pct_identical = _n_identical / max(_n_total, 1) * 100.0
            logger.debug(
                "ClippingDetector sub-ceiling: abs_max=%.4f n_identical=%d pct=%.4f%%",
                _abs_max,
                _n_identical,
                _pct_identical,
            )
            if _n_identical >= SUBCEIL_MIN_IDENTICAL and _pct_identical >= SUBCEIL_MIN_IDENTICAL_PCT:
                _sub_ceiling_clip = True
                logger.info(
                    "ClippingDetector: Sub-Ceiling-Clipping erkannt (abs_max=%.4f n_identical=%d pct=%.2f%%)",
                    _abs_max,
                    _n_identical,
                    _pct_identical,
                )

    # --- Step 2: harmonic distortion analysis ---
    thd_odd, thd_even = _compute_thd(mono, sr)
    thd_odd = float(np.clip(thd_odd, 0.0, 1e6))
    thd_even = float(np.clip(thd_even, 0.0, 1e6))
    logger.debug("ClippingDetector: thd_odd=%.4f thd_even=%.4f", thd_odd, thd_even)

    # --- Step 3: classify (normative rule §6.3) ---
    flat_tops_exceeded = flat_pct > FLAT_TOPS_THRESHOLD_PCT
    odd_dominant = thd_odd > thd_even * THD_ODD_DOMINANCE_FACTOR

    # Bei stark ausgeprägten Flat-Tops (> 1.5 %) ist Hard-Clipping zwingend:
    # Nur np.clip / hardware-Clipping erzeugt so viele identische Samples exakt
    # an der digitalen Ceiling (0.999+). Polyphonisches Musik-Material hat von
    # Natur aus viele gerade Harmonics (thd_even > thd_odd), was den THD-Test
    # für stark geklippte Signale unzuverlässig macht. Ab 1.5 % Flat-Tops ist
    # die Evidenz für Hard-Clipping so stark, dass THD keine zusätzliche
    # Sicherheit bietet (Spec §6.3 — polyphon-sichere Erweiterung).
    strong_flat_tops = flat_pct > FLAT_TOPS_STRONG_CLIP_PCT

    is_clipping = (flat_tops_exceeded and (odd_dominant or strong_flat_tops)) or _sub_ceiling_clip
    clipping_type = ClippingType.CLIPPING if is_clipping else ClippingType.SOFT_SATURATION

    # --- Step 3b: Amplitude-Histogramm Sub-Ceiling-Erkennung (Loudness-War, ±0.80–0.998) ---
    # Ergänzt identische-float32-Erkennung (Step 1b): erkennt auch Material, bei dem
    # Clipping-Level-Samples nicht exakt identisch sind (z.B. DAW-Saturation bei ±0.85–0.97).
    _subceil_hist_level: float = 0.0
    if not is_clipping and flat_pct < FLAT_TOPS_STRONG_CLIP_PCT:
        _hist_clip, _hist_level = detect_sub_ceiling_clipping(mono)
        if _hist_clip:
            is_clipping = True
            clipping_type = ClippingType.CLIPPING
            _subceil_hist_level = _hist_level
            logger.info(
                "ClippingDetector: Histogramm-Sub-Ceiling-Clipping erkannt (clip_level=%.4f)",
                _subceil_hist_level,
            )

    # --- Confidence: distance from both thresholds ---
    # When thd_odd==1.0 AND thd_even==1.0 with flat_tops near 0, the
    # polyphonic estimator found no valid frames → no harmonic evidence.
    # Confidence should reflect this near-zero certainty.
    _no_harmonic_data = abs(thd_odd - 1.0) < 0.001 and abs(thd_even - 1.0) < 0.001 and flat_pct < 0.01
    if _no_harmonic_data:
        confidence = 0.0
        logger.debug(
            "ClippingDetector: no harmonic data verfuegbar (polyphonic/no-leeren-fundamental) — confidence=0.0"
        )
    else:
        flat_distance = abs(flat_pct - FLAT_TOPS_THRESHOLD_PCT) / max(flat_pct, FLAT_TOPS_THRESHOLD_PCT, 1e-6)
        if thd_odd + thd_even < 1e-8:
            thd_ratio_distance = 1.0
        else:
            thd_ratio = thd_odd / (thd_even * THD_ODD_DOMINANCE_FACTOR + 1e-12)
            thd_ratio_distance = abs(math.log(max(thd_ratio, 1e-6)))
            thd_ratio_distance = min(thd_ratio_distance / 2.0, 1.0)  # normalise to [0,1]

        confidence = float(np.clip(min(flat_distance, thd_ratio_distance + 0.1), 0.0, 1.0))

    logger.info(
        "ClippingDetector: Ergebnis=%s flat_tops=%.3f%% odd=%.3f even=%.3f confidence=%.2f",
        clipping_type.value,
        flat_pct,
        thd_odd,
        thd_even,
        confidence,
    )

    return ClippingAnalysisResult(
        clipping_type=clipping_type,
        flat_tops_pct=flat_pct,
        thd_odd=thd_odd,
        thd_even=thd_even,
        confidence=confidence,
        is_clipping=is_clipping,
        sub_ceiling_level=_subceil_hist_level,
    )


# ---------------------------------------------------------------------------
# Singleton (§3.x — thread-safe, Double-Checked Locking)
# ---------------------------------------------------------------------------


class ClippingClassifier:
    """
    Stateless wrapper providing the singleton access point per §3.x.

    All state lives in module-level functions; this class exists only to
    satisfy the Singleton-Pattern requirement and provide a discoverable API.
    """

    def classify(self, audio: np.ndarray, sr: int) -> ClippingType:
        """Classify audio as CLIPPING or SOFT_SATURATION."""
        return classify_clipping(audio, sr)

    def analyse(self, audio: np.ndarray, sr: int) -> ClippingAnalysisResult:
        """Gibt full analysis result including intermediate metrics zurück."""
        return analyse_clipping(audio, sr)


_instance: ClippingClassifier | None = None
_lock = threading.Lock()


def get_clipping_classifier() -> ClippingClassifier:
    """Gibt the singleton ClippingClassifier instance (thread-safe) zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ClippingClassifier()
    return _instance
