"""NMR-Feedback — Noise-to-Mask-Ratio als aktive FeedbackChain-Zielgröße (§2.62+).

NMR[k] = noise_power_bark[k] / masking_threshold_bark[k] pro Bark-Band (24 Bänder).
NMR ≤ 1.0: Rauschen unterhalb der Hörschwelle (maskiert → kein NR nötig).
NMR > 1.0: Rauschen hörbar → NR-Strength erhöhen.

Kanonische Nutzung als FeedbackChain-Loop-Zielgröße (§2.62+):
    from backend.core.dsp.nmr_feedback import compute_nmr_score, NMRResult, get_nmr_feedback

    result = compute_nmr_score(audio, sr)
    # nmr_above_masking_fraction > 0.30 → NR-Stärke um recommended_nr_strength_delta erhöhen
    # nmr_above_masking_fraction < 0.05 → NR-Phase überspringen (§2.45 Minimal-Intervention)

VERBOTEN (V40):
    NR-Phase ohne NMR-Score-Messung wenn FeedbackChain aktiv:
    → compute_nmr_score(pre_nr_audio, sr) VOR NR-Phase;
    → result.recommended_nr_strength_delta auf base_strength addieren;
    → Falls result.ok → §2.45 Minimal-Intervention aktiv
"""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as _sp_signal

from backend.core.dsp.psychoacoustics import BARK_EDGES_HZ, N_BARK

logger = logging.getLogger(__name__)

# ── Schwellwerte §NMR ─────────────────────────────────────────────────────────
_NMR_OPTIMAL_FRACTION = 0.05  # Anteil hörb. Bänder unter dem alles gut ist
_NMR_HIGH_FRACTION = 0.30  # Ab hier: NR-Stärke erhöhen
_MAX_NR_DELTA = 0.50  # Maximal empfohlene Stärkeveränderung
_SPREADING_SLOPE_DB = 25.0  # Terhardt-Spreading-Abfall in dB/Bark

# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: NMRFeedback | None = None
_lock = threading.Lock()


@dataclass
class NMRResult:
    """Ergebnis der NMR-Berechnung über alle 24 Bark-Bänder.

    Attributes:
        nmr_per_band: NMR je Bark-Band, shape [24].
            Werte > 1.0 = Band mit hörbarem Rauschen.
        nmr_above_masking_fraction: Anteil Bänder mit NMR > 1.0. In [0, 1].
        global_nmr_score: Normierter Score [0, 1]: 0 = optimal maskiert, 1 = voll hörbar.
        recommended_nr_strength_delta: Empfohlenes Stärke-Delta für NR [-0.5, +0.5].
            Positiv → mehr NR nötig; Negativ → §2.45 Minimal-Intervention.
        noise_floor_db: Geschätzter globaler Rauschboden in dBFS.
        ok: True wenn nmr_above_masking_fraction < 0.10 (≤ 2 Bänder hörbar).
    """

    nmr_per_band: np.ndarray = field(default_factory=lambda: np.ones(N_BARK, dtype=np.float32))
    nmr_above_masking_fraction: float = 0.10
    global_nmr_score: float = 0.10
    recommended_nr_strength_delta: float = 0.0
    noise_floor_db: float = -60.0
    ok: bool = True


class NMRFeedback:
    """Singleton-Wrapper für NMR-Feedback-Berechnungen."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def compute(
        self,
        audio: np.ndarray,
        sr: int,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> NMRResult:
        """Thread-sichere NMR-Berechnung."""
        with self._lock:
            return compute_nmr_score(audio, sr, n_fft=n_fft, hop_length=hop_length)


def get_nmr_feedback() -> NMRFeedback:
    """Singleton-Zugriff auf NMRFeedback."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = NMRFeedback()
    return _instance


def compute_nmr_score(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> NMRResult:
    """Berechnet NMR (Noise-to-Mask-Ratio) pro Bark-Band.

    Algorithmus (Zwicker & Fastl 1999, §4.2):
      1. STFT → Leistungsspektrum (n_freq × n_frames)
      2. Rauschschätzung via Minimum-Statistics (10th Percentile über Zeit)
      3. ATH nach ISO 226:2003 Terhardt-Formel als absoluter Maskierungsboden
      4. Vereinfachte Terhardt-Spreading-Funktion auf aktive Signalkomponenten
      5. Gesamtmaskierungsschwelle = max(ATH, Spreading-Summe)
      6. NMR[k] = noise_power_bark[k] / masking_power_bark[k]
      7. Aggregation zu recommended_nr_strength_delta

    Args:
        audio: Mono oder Stereo [2,N] / [N,2]. Wird intern zu Mono konvertiert.
        sr: Abtastrate in Hz. KEINE sr==48000-Assertion (Analyse-Modul).
        n_fft: FFT-Fenstergröße.
        hop_length: Hop-Länge in Samples.

    Returns:
        NMRResult mit NMR pro Band + FeedbackChain-Empfehlungen.
    """
    _fallback = NMRResult()
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            # channels-first [2,N] oder channels-last [N,2]
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                audio = audio.mean(axis=0)
            elif audio.shape[1] == 2 and audio.shape[0] > 2:
                audio = audio.mean(axis=1)
            else:
                audio = audio.mean(axis=0)
        audio = np.asarray(audio, dtype=np.float64)

        n_samples = len(audio)
        if n_samples < n_fft:
            return _fallback

        # ── 1. STFT → Leistungsspektrum ───────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, zxx = _sp_signal.stft(
                audio,
                fs=float(sr),
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
                window="hann",
            )
        power = np.abs(zxx) ** 2  # (n_freq, n_frames)
        n_freq = power.shape[0]
        freqs = np.linspace(0.0, sr / 2.0, n_freq, dtype=np.float64)
        f_khz = np.maximum(freqs / 1000.0, 0.001)

        # ── 2. Rauschschätzung: Minimum-Statistics (10th Percentile) ──────────
        noise_power = np.percentile(power, 10.0, axis=1)  # (n_freq,)
        noise_power = np.maximum(noise_power, 1e-20)
        noise_floor_db = float(10.0 * np.log10(noise_power.mean() + 1e-20))

        # ── 3. ATH nach Terhardt 1979 (ISO 226:2003 vereinfacht) ──────────────
        ath_db = 3.64 * np.power(f_khz, -0.8) - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * np.power(f_khz, 4.0)
        ath_db = np.clip(ath_db, -20.0, 80.0)
        ath_power = np.power(10.0, ath_db / 10.0)

        # ── 4. Terhardt-Spreading auf aktive Signalkomponenten ────────────────
        signal_power = np.mean(power, axis=1)
        signal_power = np.maximum(signal_power, 1e-20)
        signal_db = 10.0 * np.log10(signal_power)

        bark = 13.0 * np.arctan(0.76 * f_khz) + 3.5 * np.arctan((f_khz / 7.5) ** 2)
        active_mask = signal_power > ath_power

        if active_mask.sum() > 0:
            # Spreading-Funktion: Terhardt 1979 — exponentieller Abfall ±Bark
            bark_active = bark[active_mask]
            db_active = signal_db[active_mask]
            # Vectorisiert: (n_freq, n_active) — Spreading pro Masker
            bark_diff = bark[:, np.newaxis] - bark_active[np.newaxis, :]  # +: upward, -: downward
            # Upward masking (positive Bark-Differenz) steiler als downward
            spread_db = np.where(
                bark_diff >= 0,
                db_active[np.newaxis, :] - _SPREADING_SLOPE_DB * bark_diff,
                db_active[np.newaxis, :] - 10.0 * np.abs(bark_diff),
            )
            spread_db = np.clip(spread_db, -80.0, None)
            masking_power = np.power(10.0, spread_db / 10.0).sum(axis=1)
        else:
            masking_power = np.zeros(n_freq, dtype=np.float64)

        # ── 5. Gesamtmaskierungsschwelle = max(ATH, Spreading) ────────────────
        masking_power = np.maximum(masking_power, ath_power)
        masking_power = np.maximum(masking_power, 1e-20)

        # ── 6. NMR pro Bark-Band aggregieren ──────────────────────────────────
        nmr_per_band = np.zeros(N_BARK, dtype=np.float32)
        for k in range(N_BARK):
            f_low = BARK_EDGES_HZ[k]
            f_high = BARK_EDGES_HZ[k + 1]
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if band_mask.any():
                noise_k = float(noise_power[band_mask].mean())
                masking_k = float(masking_power[band_mask].mean())
                nmr_per_band[k] = float(np.clip(noise_k / (masking_k + 1e-20), 0.0, 100.0))

        nmr_per_band = np.nan_to_num(nmr_per_band, nan=1.0, posinf=10.0, neginf=0.0)

        # ── 7. Aggregierte Metriken ────────────────────────────────────────────
        nmr_above_masking_fraction = float(np.mean(nmr_per_band > 1.0))
        global_nmr_score = float(np.clip(nmr_above_masking_fraction, 0.0, 1.0))

        # Stärke-Delta: linear proportional zur Überschreitung über optimal
        recommended_nr_strength_delta = float(
            np.clip(nmr_above_masking_fraction - _NMR_OPTIMAL_FRACTION, -_MAX_NR_DELTA, +_MAX_NR_DELTA)
        )

        logger.debug(
            "NMR: above_masking=%.2f global=%.2f delta=%.3f noise_floor=%.1f dBFS",
            nmr_above_masking_fraction,
            global_nmr_score,
            recommended_nr_strength_delta,
            noise_floor_db,
        )

        return NMRResult(
            nmr_per_band=nmr_per_band,
            nmr_above_masking_fraction=nmr_above_masking_fraction,
            global_nmr_score=global_nmr_score,
            recommended_nr_strength_delta=recommended_nr_strength_delta,
            noise_floor_db=float(noise_floor_db),
            ok=bool(nmr_above_masking_fraction < 0.10),
        )

    except Exception as exc:
        logger.warning("NMR-Wert-Berechnung fehlgeschlagen: %s — Ersatzpfad", exc)
        return _fallback


def recommend_nr_strength(
    nmr_result: NMRResult,
    base_strength: float,
    min_strength: float = 0.05,
    max_strength: float = 1.0,
) -> float:
    """Empfiehlt NR-Stärke basierend auf NMR-Feedback-Score.

    §2.45 Minimal-Intervention:
    - nmr_result.ok → base_strength × 0.5 (kaum hörbares Rauschen, weniger NR)
    - nmr_result.nmr_above_masking_fraction > 0.30 → base_strength erhöhen

    Args:
        nmr_result: Ergebnis von compute_nmr_score().
        base_strength: Basis-Stärke aus Phase-Strength-Oracle [0, 1].
        min_strength: Untere Grenze (non-blocking, kein totaler Bypass).
        max_strength: Obere Grenze.

    Returns:
        Empfohlene NR-Stärke in [min_strength, max_strength].
    """
    adjusted = float(base_strength) + nmr_result.recommended_nr_strength_delta
    # §2.45: Wenn Rauschen gut maskiert → auf Minimum-Intervention reduzieren
    if nmr_result.ok and adjusted > 0.3:
        adjusted = max(adjusted * 0.5, min_strength)
    return float(np.clip(adjusted, min_strength, max_strength))
