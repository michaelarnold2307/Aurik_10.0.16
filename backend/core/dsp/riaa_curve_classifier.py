"""
backend/core/dsp/riaa_curve_classifier.py
Aurik 10.0.0 — Spec §6.6 (v10.0.0): RIAA-Kurven-Klassifikation — normative Metriken.

Bayesianische Klassifikation des Disc-EQ-Kurventyps (pre-RIAA / RIAA-Varianten)
aus dem Audiofile-Spektrum. Ergänzt §6.3a (Zeitkonstanten) um Klassifikations-
Metriken.

Messung: Spektraler Tilt in dB/Oktave über 250–8000 Hz (Fenster: Hann, FFT 4096).
Toleranzband: ±1.0 dB/oct (±1.5 bei SNR < 10 dB).
Konfidenz-Schwelle: ≥ 0.70 → Kurve bestätigt; < 0.70 → "unknown" (konservativ).
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# §6.6a Kanonische Slope-Profile (normative Quelle)
# ---------------------------------------------------------------------------

#: (slope_dB_oct, bass_boost_db_at_100hz, hf_cut_freq_hz, bass_turnover_hz)
_SlotEntry = tuple[float, float, float, float]

RIAA_SLOPE_PROFILES: dict[str, _SlotEntry | None] = {
    "riaa": (-5.0, +13.7, 2122.0, 3183.0),  # IEC 1994 Standard (τ: 75/318/3180 µs)
    "nab": (-4.5, +13.7, 1590.0, 3183.0),  # NAB Broadcast (τ: 100/318/3180 µs)
    "columbia": (-3.5, +16.0, 1590.0, 1590.0),  # Columbia Records 1948 (Bassbetont)
    "aes": (-4.5, +12.0, 3183.0, 3183.0),  # AES 1951 (weniger Bass-Boost)
    "capitol": (-3.7, +15.0, 1590.0, 2122.0),  # Capitol Records 1949
    "london": (-5.2, +13.7, 2122.0, 3183.0),  # London/Decca (mehr HF-Boost als RIAA)
    "ccir": (-4.9, +13.7, 3183.0, 3183.0),  # CCIR/EBU Rundfunk
    "unknown": None,  # Bayes-Prior: uniform über alle Kurven
}

#: Konfidenz-Schwelle für bestätigte Kurve (§6.6)
_CONFIDENCE_THRESHOLD: float = 0.70

#: FFT-Größe für spektrale Messung (Spec: 4096 bei 48 kHz)
_FFT_SIZE: int = 4096

#: Messbereich für Slope (Hz)
_SLOPE_F_LOW: float = 250.0
_SLOPE_F_HIGH: float = 8000.0


# ---------------------------------------------------------------------------
# Hilfsfunktionen (interne API)
# ---------------------------------------------------------------------------


def _measure_spectral_slope(audio: np.ndarray, sr: int) -> float:
    """Misst spektralen Tilt in dB/Oktave über [250, 8000] Hz via Welch-PSD.

    Args:
        audio:  Mono float-Array (bereits in mono konvertiert).
        sr:     Sample-Rate (erwartet 48000).

    Returns:
        Slope in dB/Oktave (negativ = abfallend zu hohen Frequenzen hin).
    """
    try:
        freqs, psd = sp_signal.welch(audio, sr, nperseg=_FFT_SIZE, window="hann")
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

        # Nur Bereich 250–8000 Hz verwenden
        mask = (freqs >= _SLOPE_F_LOW) & (freqs <= _SLOPE_F_HIGH)
        if np.sum(mask) < 4:
            return 0.0

        f_sel = freqs[mask]
        p_sel = psd_db[mask]

        # Log-Oktaven-Regressionssteigung (log2-Skalierung)
        log2_f = np.log2(f_sel / f_sel[0])
        if np.std(log2_f) < 1e-6:
            return 0.0

        # Linearer Fit: slope = dB / Oktave
        slope, _ = np.polyfit(log2_f, p_sel, 1)
        return float(slope)
    except Exception as exc:
        logger.debug("_measure_spectral_slope fehlgeschlagen: %s", exc)
        return 0.0


def _measure_bass_boost_at_100hz(audio: np.ndarray, sr: int) -> float:
    """Misst relativen Bass-Boost bei 100 Hz gegenüber 1 kHz-Referenz in dB.

    Args:
        audio:  Mono float-Array.
        sr:     Sample-Rate.

    Returns:
        Bass-Boost in dB (positiv = Bass stärker als 1 kHz).
    """
    try:
        freqs, psd = sp_signal.welch(audio, sr, nperseg=_FFT_SIZE, window="hann")
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

        # 100 Hz Bin und 1 kHz Referenz
        idx_100 = int(np.argmin(np.abs(freqs - 100.0)))
        idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))

        return float(psd_db[idx_100] - psd_db[idx_1k])
    except Exception as exc:
        logger.debug("_measure_bass_boost_at_100hz fehlgeschlagen: %s", exc)
        return 0.0


def _find_hf_turnover_freq(audio: np.ndarray, sr: int) -> float:
    """Schätzt die HF-Turnover-Frequenz in Hz (erste Frequenz mit -3 dB Abfall).

    Args:
        audio:  Mono float-Array.
        sr:     Sample-Rate.

    Returns:
        Turnover-Frequenz in Hz. Fallback: 2122 Hz (RIAA-Standard).
    """
    try:
        freqs, psd = sp_signal.welch(audio, sr, nperseg=_FFT_SIZE, window="hann")
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

        # Referenz: 1 kHz Level
        idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))
        ref_db = float(psd_db[idx_1k])

        # Suche erste Frequenz > 1 kHz, wo Level um 3 dB abgefallen ist
        mask = freqs > 1000.0
        if np.sum(mask) == 0:
            return 2122.0

        f_above = freqs[mask]
        p_above = psd_db[mask]

        below_threshold = p_above < (ref_db - 3.0)
        if not np.any(below_threshold):
            return float(f_above[-1])  # kein Abfall → höchste Freq

        first_idx = int(np.argmax(below_threshold))
        return float(f_above[first_idx])
    except Exception as exc:
        logger.debug("_find_hf_turnover_freq fehlgeschlagen: %s", exc)
        return 2122.0


def _get_era_riaa_priors(era_decade: int | None) -> dict[str, float]:
    """Gibt era-adaptive Prior-Gewichte für Bayes-Klassifikation zurück.

    §6.6:
    - era_decade ≤ 1950: Prior für columbia/capitol/aes erhöht (×2.5)
    - era_decade ≥ 1960: Prior für riaa erhöht (×3.0), columbia/capitol/aes reduziert (×0.3)

    Returns:
        Dict mit Kurvenname → Prior-Multiplikator. Fehlende Einträge = 1.0.
    """
    if era_decade is None:
        return {}
    if era_decade <= 1950:
        return {
            "columbia": 2.5,
            "capitol": 2.5,
            "aes": 2.5,
            "riaa": 0.5,  # RIAA noch nicht standardisiert vor 1954
        }
    if era_decade >= 1960:
        return {
            "riaa": 3.0,
            "columbia": 0.3,
            "capitol": 0.3,
            "aes": 0.3,
        }
    # 1951–1959: Übergangszeit — neutrale Priors
    return {
        "riaa": 1.5,
        "aes": 1.2,
        "nab": 1.2,
    }


# ---------------------------------------------------------------------------
# Hauptfunktion: §6.6 classify_riaa_curve
# ---------------------------------------------------------------------------


def classify_riaa_curve(audio: np.ndarray, sr: int, era_decade: int | None = None) -> str:
    """Bayesianische RIAA-Kurvenklassifikation (§6.6).

    P(curve | audio) ∝ P(audio | curve) × P(curve | era_decade)
    Likelihood aus: slope_match + bass_turnover_match + hf_rolloff_match.
    Konfidenz-Schwelle: ≥ 0.70 → Kurve bestätigt; < 0.70 → "unknown" (konservativ).

    Args:
        audio:       Eingangs-Audio (mono oder stereo, float32/float64).
        sr:          Sample-Rate (bevorzugt 48000 für Analyse).
        era_decade:  Aufnahme-Jahrzehnt (4-stellige Zahl, z. B. 1948).
                     None → neutrale Priors.

    Returns:
        Kurvenname aus RIAA_SLOPE_PROFILES oder "unknown".
    """
    assert sr > 0, "SR muss positiv sein"

    # Mono-Konvertierung (robust für (N,) und (2,N) und (N,2))
    if audio.ndim == 2:
        _ch_first = audio.shape[0] == 2 and audio.shape[1] > 2
        mono = audio.mean(axis=0) if _ch_first else audio.mean(axis=1)
    else:
        mono = audio
    mono = mono[: min(len(mono), sr * 20)].astype(np.float64)

    if len(mono) < _FFT_SIZE:
        logger.debug("classify_riaa_curve: Audio zu kurz (%d samples) → 'unknown'", len(mono))
        return "unknown"

    # Messung (spec §6.6a)
    slope = _measure_spectral_slope(mono, sr)
    bass_boost = _measure_bass_boost_at_100hz(mono, sr)
    hf_turnover = _find_hf_turnover_freq(mono, sr)

    logger.debug(
        "classify_riaa_curve: slope=%.2f dB/oct bass_boost=%.1f dB hf_turnover=%.0f Hz era=%s",
        slope,
        bass_boost,
        hf_turnover,
        era_decade,
    )

    # Likelihoods (Gauß-Kernel per Spec)
    likelihoods: dict[str, float] = {}
    for curve, profile in RIAA_SLOPE_PROFILES.items():
        if profile is None:
            likelihoods[curve] = 0.1  # uniform Prior für "unknown"
            continue
        s_ref, b_ref, h_ref, _bt_ref = profile
        d_slope = abs(slope - s_ref) / 1.0  # Toleranz 1 dB/oct
        d_bass = abs(bass_boost - b_ref) / 3.0  # Toleranz 3 dB
        d_hf = abs(hf_turnover - h_ref) / 500.0  # Toleranz 500 Hz
        likelihoods[curve] = float(np.exp(-0.5 * (d_slope**2 + d_bass**2 + d_hf**2)))

    # Era-Prior anwenden
    era_priors = _get_era_riaa_priors(era_decade)
    posteriors: dict[str, float] = {k: likelihoods[k] * era_priors.get(k, 1.0) for k in likelihoods}

    # Normierung
    total = sum(posteriors.values())
    if total < 1e-12:
        return "unknown"
    posteriors = {k: v / total for k, v in posteriors.items()}

    best_curve = max(posteriors, key=lambda k: posteriors[k])
    best_conf = posteriors[best_curve]

    logger.debug(
        "classify_riaa_curve: best=%s conf=%.3f (Schwelle=%.2f)",
        best_curve,
        best_conf,
        _CONFIDENCE_THRESHOLD,
    )

    return best_curve if best_conf >= _CONFIDENCE_THRESHOLD else "unknown"


def classify_riaa_curve_with_confidence(
    audio: np.ndarray,
    sr: int,
    era_decade: int | None = None,
) -> tuple[str, float]:
    """Wie classify_riaa_curve, gibt zusätzlich die Konfidenz zurück.

    Returns:
        Tuple (curve_name, confidence ∈ [0, 1]).
    """
    assert sr > 0, "SR muss positiv sein"

    if audio.ndim == 2:
        _ch_first = audio.shape[0] == 2 and audio.shape[1] > 2
        mono = audio.mean(axis=0) if _ch_first else audio.mean(axis=1)
    else:
        mono = audio
    mono = mono[: min(len(mono), sr * 20)].astype(np.float64)

    if len(mono) < _FFT_SIZE:
        return "unknown", 0.0

    slope = _measure_spectral_slope(mono, sr)
    bass_boost = _measure_bass_boost_at_100hz(mono, sr)
    hf_turnover = _find_hf_turnover_freq(mono, sr)

    likelihoods: dict[str, float] = {}
    for curve, profile in RIAA_SLOPE_PROFILES.items():
        if profile is None:
            likelihoods[curve] = 0.1
            continue
        s_ref, b_ref, h_ref, _bt_ref = profile
        d_slope = abs(slope - s_ref) / 1.0
        d_bass = abs(bass_boost - b_ref) / 3.0
        d_hf = abs(hf_turnover - h_ref) / 500.0
        likelihoods[curve] = float(np.exp(-0.5 * (d_slope**2 + d_bass**2 + d_hf**2)))

    era_priors = _get_era_riaa_priors(era_decade)
    posteriors: dict[str, float] = {k: likelihoods[k] * era_priors.get(k, 1.0) for k in likelihoods}
    total = sum(posteriors.values())
    if total < 1e-12:
        return "unknown", 0.0
    posteriors = {k: v / total for k, v in posteriors.items()}

    best_curve = max(posteriors, key=lambda k: posteriors[k])
    best_conf = float(posteriors[best_curve])

    return (best_curve if best_conf >= _CONFIDENCE_THRESHOLD else "unknown"), best_conf


# ---------------------------------------------------------------------------
# Singleton-Wrapper (für konsistente Nutzung)
# ---------------------------------------------------------------------------


class RiaaCurveClassifier:
    """Singleton-Wrapper für classify_riaa_curve (§6.6)."""

    def classify(self, audio: np.ndarray, sr: int, era_decade: int | None = None) -> str:
        """Siehe :func:`classify_riaa_curve`."""
        return classify_riaa_curve(audio, sr, era_decade)

    def classify_with_confidence(self, audio: np.ndarray, sr: int, era_decade: int | None = None) -> tuple[str, float]:
        """Siehe :func:`classify_riaa_curve_with_confidence`."""
        return classify_riaa_curve_with_confidence(audio, sr, era_decade)


_instance: RiaaCurveClassifier | None = None
_lock = threading.Lock()


def get_riaa_curve_classifier() -> RiaaCurveClassifier:
    """Thread-sicherer Singleton-Accessor."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RiaaCurveClassifier()
    return _instance
