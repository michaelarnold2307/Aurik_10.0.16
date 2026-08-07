"""Zwicker-Metriken — Rauigkeit (Roughness) und Fluktuationsstärke (Fluctuation Strength).

Psychoakustische Qualitätsmetriken nach Zwicker & Fastl (1999), §10.1+§10.2:

**Rauigkeit (Roughness, Einheit: asper)**:
  Entsteht durch AM-Modulation mit 15–300 Hz (Dissonanz, Klingeln, Intermodulation,
  Wow/Flutter-Residuen nach Phase_12). Spitze bei ~70 Hz AM-Rate.
  Referenz: 1 asper = 100% AM 1 kHz-Ton bei 60 dB SPL, 70 Hz AM-Rate.

**Fluktuationsstärke (Fluctuation Strength, Einheit: vacil)**:
  Entsteht durch AM-Modulation mit 0.5–20 Hz (NR-Pumpen, Kompressor-Atmen).
  Spitze bei ~4 Hz. Referenz: 1 vacil = 100% AM 1 kHz bei 60 dB SPL, 4 Hz AM-Rate.

VERBOTEN (V42):
  NR-Phase (phase_03 / phase_29) ohne anschließenden Roughness/Fluctuation-Check
  bei `panns_singing ≥ 0.35`:
  → check_roughness_regression(pre, post, sr) MUSS nach NR-Phase aufgerufen werden
  → fluctuation_strength_regression: True → Dry-Wet-Blend × 0.80 (NR-Pumpen erkannt)
  → roughness_regression: True → Dry-Wet-Blend × 0.90 (Intermod-Rauschen gestiegen)

Grenzwerte für Regression (non-blocking WARNING, kein Veto):
  - roughness_asper_post > roughness_asper_pre × 1.10 → roughness_regression = True
  - fluctuation_strength_post > fluctuation_strength_pre × 1.20 → pumping_detected = True

Kanonische Nutzung (UV3 post-NR-Phase hook):
    from backend.core.dsp.zwicker_metrics import check_roughness_regression, ZwickerMetricsResult

    result = check_roughness_regression(audio_pre, audio_post, sr)
    if result.roughness_regression:
        metadata["roughness_regression_warning"] = True
        blend = 0.90  # sanfte Rücknahme
    if result.pumping_detected:
        metadata["pumping_detected_warning"] = True
        blend = min(blend, 0.80)  # stärker rückgenommen (Pumpen deutlich hörbar)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal as _sp_signal

logger = logging.getLogger(__name__)

# ── Modulations-Frequenzbereiche (Zwicker & Fastl 1999) ───────────────────────
_ROUGHNESS_MOD_HZ_LOW = 15.0  # Hz — unterste Modulations-Frequenz für Rauigkeit
_ROUGHNESS_MOD_HZ_HIGH = 300.0  # Hz — oberste Modulations-Frequenz für Rauigkeit
_ROUGHNESS_PEAK_HZ = 70.0  # Hz — maximale Rauhigkeits-Sensitivität

_FLUCTUATION_MOD_HZ_LOW = 0.5  # Hz — unterste Modulations-Frequenz für Fluktuationsstärke
_FLUCTUATION_MOD_HZ_HIGH = 20.0  # Hz — oberste Modulations-Frequenz für Fluktuationsstärke
_FLUCTUATION_PEAK_HZ = 4.0  # Hz — maximale Fluktuations-Sensitivität

# ── Regressions-Schwellen ─────────────────────────────────────────────────────
_ROUGHNESS_REGRESSION_RATIO = 1.10  # 10% Zunahme → WARNING
_PUMPING_REGRESSION_RATIO = 1.20  # 20% Zunahme → WARNING (Pumpen erkannt)

# ── Kalibrierungskonstanten (empirisch, Musik bei -18 dBFS → ~0.5 asper) ──────
# 1 asper Referenz: 100% AM 1 kHz 60 dB SPL 70 Hz — normiertes PSD-Integral ~5e-4
_ASPER_CALIBRATION = 2000.0  # Umrechnung rohes PSD-Integral → asper
_VACIL_CALIBRATION = 50000.0  # Umrechnung rohes PSD-Integral → vacil


@dataclass
class ZwickerMetricsResult:
    """Ergebnis der Zwicker-Rauigkeits- und Fluktuationsstärken-Messung.

    Attributes:
        roughness_asper: Rauigkeit in asper. Typisch < 0.5 für restaurierte Musik.
        fluctuation_strength_vacil: Fluktuationsstärke in vacil. Typisch < 0.10 nach NR.
        roughness_regression: True wenn Rauigkeit nach Phase > vor Phase × 1.10.
        pumping_detected: True wenn Fluktuationsstärke > vor Phase × 1.20.
        roughness_asper_reference: Referenzwert (pre-phase) für Regression-Check.
        fluctuation_vacil_reference: Referenzwert (pre-phase) für Regression-Check.
    """

    roughness_asper: float = 0.0
    fluctuation_strength_vacil: float = 0.0
    roughness_regression: bool = False
    pumping_detected: bool = False
    roughness_asper_reference: float = 0.0
    fluctuation_vacil_reference: float = 0.0


def compute_roughness_asper(audio: np.ndarray, sr: int) -> float:
    """Berechnet die Rauigkeit (Roughness) in asper.

    Algorithmus (Daniel & Weber 1997 vereinfacht):
      1. Mono-Konvertierung
      2. Hilbert-Transformation → obere Hüllkurve (Amplitude-Envelope)
      3. Welch-PSD der Hüllkurve
      4. Bereich 15–300 Hz extrahieren + Gauß-Gewichtung um 70 Hz
      5. Gewichtetes PSD-Integral → asper (via Kalibrierungskonstante)

    Args:
        audio: Mono oder Stereo-Signal. KEINE sr-Assertion (Analyse-Modul).
        sr: Abtastrate in Hz. Mindestens 200 Hz (für 100 Hz AM-Auflösung nötig).

    Returns:
        Roughness in asper [0, 10].
    """
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                audio = audio.mean(axis=0)
            elif audio.shape[1] == 2 and audio.shape[0] > 2:
                audio = audio.mean(axis=1)
            else:
                audio = audio.mean(axis=0)
        audio = np.asarray(audio, dtype=np.float32)

        min_samples = max(int(sr * 0.1), 256)
        if len(audio) < min_samples:
            return 0.0

        # ── Hilbert-Hüllkurve ────────────────────────────────────────────────
        envelope = np.abs(_sp_signal.hilbert(audio.astype(np.float64)))
        envelope = np.maximum(envelope, 1e-12)

        # ── Welch-PSD der Hüllkurve ───────────────────────────────────────────
        # nperseg: 4× die Nyquist-Auflösung für 15 Hz (sr/15/2 = sr/30 Samples)
        # Mindest-nperseg: 256 Samples
        nperseg = min(int(sr / 5), max(256, len(envelope) // 4))
        nperseg = max(nperseg, 256)
        nperseg = min(nperseg, len(envelope))

        freqs_env, psd_env = _sp_signal.welch(
            envelope,
            fs=float(sr),
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
        )
        psd_env = np.nan_to_num(psd_env, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Rauigkeits-Band 15–300 Hz ─────────────────────────────────────────
        rough_mask = (freqs_env >= _ROUGHNESS_MOD_HZ_LOW) & (freqs_env <= _ROUGHNESS_MOD_HZ_HIGH)
        if not rough_mask.any():
            return 0.0

        rough_freqs = freqs_env[rough_mask]
        rough_psd = psd_env[rough_mask]

        # Gauß-Gewichtung mit Peak bei 70 Hz (Terhardt 1978)
        rough_weight = np.exp(-0.5 * ((rough_freqs - _ROUGHNESS_PEAK_HZ) / 80.0) ** 2)

        total_weight = float(rough_weight.sum())
        if total_weight < 1e-10:
            return 0.0

        raw_roughness = float(np.sum(rough_weight * rough_psd) / total_weight)
        asper = float(np.clip(raw_roughness * _ASPER_CALIBRATION, 0.0, 10.0))

        return asper

    except Exception as exc:
        logger.debug("berechnen_roughness_asper nicht blockierend: %s", exc)
        return 0.0


def compute_fluctuation_strength_vacil(audio: np.ndarray, sr: int) -> float:
    """Berechnet die Fluktuationsstärke in vacil.

    Algorithmus (Zwicker & Fastl 1999, §10.2 vereinfacht):
      1. Mono-Konvertierung
      2. Hilbert-Transformation → obere Hüllkurve
      3. Welch-PSD der Hüllkurve
      4. Bereich 0.5–20 Hz extrahieren + Lorentz-Gewichtung um 4 Hz
      5. Gewichtetes PSD-Integral → vacil

    Args:
        audio: Mono oder Stereo-Signal.
        sr: Abtastrate. Mindestens 50 Hz für 4 Hz AM-Auflösung.

    Returns:
        Fluctuation Strength in vacil [0, 10].
    """
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                audio = audio.mean(axis=0)
            elif audio.shape[1] == 2 and audio.shape[0] > 2:
                audio = audio.mean(axis=1)
            else:
                audio = audio.mean(axis=0)
        audio = np.asarray(audio, dtype=np.float32)

        min_samples = max(int(sr * 0.5), 256)
        if len(audio) < min_samples:
            return 0.0

        # ── Hilbert-Hüllkurve ────────────────────────────────────────────────
        envelope = np.abs(_sp_signal.hilbert(audio.astype(np.float64)))
        envelope = np.maximum(envelope, 1e-12)

        # ── Welch-PSD der Hüllkurve ───────────────────────────────────────────
        # Für 0.5 Hz Auflösung: nperseg ≥ 2 × sr Samples (= 2 Sekunden)
        nperseg = min(int(sr * 2.0), max(512, len(envelope) // 2))
        nperseg = max(nperseg, 512)
        nperseg = min(nperseg, len(envelope))

        freqs_env, psd_env = _sp_signal.welch(
            envelope,
            fs=float(sr),
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window="hann",
        )
        psd_env = np.nan_to_num(psd_env, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Fluktuations-Band 0.5–20 Hz ──────────────────────────────────────
        fluct_mask = (freqs_env >= _FLUCTUATION_MOD_HZ_LOW) & (freqs_env <= _FLUCTUATION_MOD_HZ_HIGH)
        if not fluct_mask.any():
            return 0.0

        fluct_freqs = freqs_env[fluct_mask]
        fluct_psd = psd_env[fluct_mask]

        # Lorentz-Gewichtung (Breitband-Sensitivitätskurve) mit Peak bei 4 Hz
        # W(f) = 1 / (1 + ((f - 4) / 4)^2)
        fluct_weight = 1.0 / (1.0 + ((fluct_freqs - _FLUCTUATION_PEAK_HZ) / 4.0) ** 2)

        total_weight = float(fluct_weight.sum())
        if total_weight < 1e-10:
            return 0.0

        raw_fluctuation = float(np.sum(fluct_weight * fluct_psd) / total_weight)
        vacil = float(np.clip(raw_fluctuation * _VACIL_CALIBRATION, 0.0, 10.0))

        return vacil

    except Exception as exc:
        logger.debug("berechnen_fluctuation_strength_vacil nicht blockierend: %s", exc)
        return 0.0


def compute_zwicker_metrics(audio: np.ndarray, sr: int) -> ZwickerMetricsResult:
    """Berechnet Rauigkeit und Fluktuationsstärke für ein Audio-Signal.

    Args:
        audio: Mono oder Stereo-Signal.
        sr: Abtastrate in Hz.

    Returns:
        ZwickerMetricsResult mit asper- und vacil-Werten (ohne Regression-Flags).
    """
    asper = compute_roughness_asper(audio, sr)
    vacil = compute_fluctuation_strength_vacil(audio, sr)
    return ZwickerMetricsResult(
        roughness_asper=asper,
        fluctuation_strength_vacil=vacil,
        roughness_regression=False,
        pumping_detected=False,
    )


def check_roughness_regression(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    sr: int,
) -> ZwickerMetricsResult:
    """Prüft ob eine Phase Rauigkeit oder NR-Pumpen eingeführt hat.

    Berechnet Zwicker-Metriken für pre- und post-phase Audio und vergleicht.
    Grenzwerte:
      - roughness_regression: roughness_post > roughness_pre × 1.10 (+10%)
      - pumping_detected: fluctuation_post > fluctuation_pre × 1.20 (+20%)

    Beide Flags sind non-blocking WARNINGs (kein Veto). Empfohlene Reaktion:
      - roughness_regression → Dry-Wet-Blend × 0.90
      - pumping_detected → Dry-Wet-Blend × 0.80

    Args:
        audio_pre: Signal vor der Phase.
        audio_post: Signal nach der Phase.
        sr: Abtastrate in Hz.

    Returns:
        ZwickerMetricsResult mit Regression-Flags und Referenzwerten.
    """
    try:
        pre_asper = compute_roughness_asper(audio_pre, sr)
        post_asper = compute_roughness_asper(audio_post, sr)

        pre_vacil = compute_fluctuation_strength_vacil(audio_pre, sr)
        post_vacil = compute_fluctuation_strength_vacil(audio_post, sr)

        roughness_regression = bool(post_asper > pre_asper * _ROUGHNESS_REGRESSION_RATIO)
        pumping_detected = bool(post_vacil > pre_vacil * _PUMPING_REGRESSION_RATIO)

        if roughness_regression:
            logger.warning(
                "§V42 Rauigkeits-Regression: pre=%.4f asper → post=%.4f asper (+%.1f%%) → Blend × 0.90",
                pre_asper,
                post_asper,
                100.0 * (post_asper / (pre_asper + 1e-10) - 1.0),
            )
        if pumping_detected:
            logger.warning(
                "§V42 NR-Pumpen erkannt: pre=%.5f vacil → post=%.5f vacil (+%.1f%%) → Blend × 0.80",
                pre_vacil,
                post_vacil,
                100.0 * (post_vacil / (pre_vacil + 1e-10) - 1.0),
            )

        return ZwickerMetricsResult(
            roughness_asper=post_asper,
            fluctuation_strength_vacil=post_vacil,
            roughness_regression=roughness_regression,
            pumping_detected=pumping_detected,
            roughness_asper_reference=pre_asper,
            fluctuation_vacil_reference=pre_vacil,
        )

    except Exception as exc:
        logger.debug("Pruefung_roughness_regression nicht blockierend: %s", exc)
        return ZwickerMetricsResult()
