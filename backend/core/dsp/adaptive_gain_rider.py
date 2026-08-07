"""§Gap11 AdaptiveGainRider — Onset-ausgerichtetes Gain-Riding (v10.0.0).

Korrigiert Lautstärke-Inkonsistenzen, die durch NR, BW-Erweiterung oder
Hallucination-Rollbacks entstehen: Der restaurierte Track soll dieselbe
dynamische Kurve wie das Eingangs-Signal behalten.

Methode:
  1. Berechnet 50 ms RMS-Energie-Kurven für Original und Restored.
  2. Onset-getriggerte Segmentierung (nicht frameweise) für musikalisch
     sinnvolle Anpassungsgrenzen.
  3. Smooth Gain-Envelope: Butterworth-Tiefpass bei 2 Hz (200 ms Glättung).
  4. Gain wird per-Sample angewendet (linear, kein Hard-Clip außer ±1.0).

§0h Invariante: artifact_freedom — kein Clipping im Ausgang.
§0c: Generisch (era/genre-unabhängig), song-agnostisch.
§0h: Stille-Zonen werden NICHT verstärkt (limit_quiet_edge_boost).

Verwendung als UV3-Post-Phase-Hook:
    from backend.core.dsp.adaptive_gain_rider import ride_gain_to_performance_profile
    audio_out = ride_gain_to_performance_profile(audio_restored, audio_original, sr)
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

_RMS_FRAME_S: float = 0.050  # 50 ms RMS-Analyse-Fenster
_GAIN_LP_FREQ_HZ: float = 2.0  # Tiefpass für Gain-Glättung (2 Hz = 200 ms Glättung)
_MAX_GAIN_RATIO: float = 2.0  # Maximale Anhebung: +6 dB
_MIN_GAIN_RATIO: float = 0.25  # Maximale Absenkung: -12 dB
_SILENCE_THRESHOLD_RMS: float = 5e-4  # RMS unter diesem Wert gilt als Stille
_ONSET_GAIN_HOLD_FRAMES: int = 3  # Gain-Hold nach Onset (verhindert Transienten-Pumpen)


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def ride_gain_to_performance_profile(
    audio_restored: np.ndarray,
    audio_original: np.ndarray,
    sr: int,
    *,
    max_gain_db: float = 6.0,
    min_gain_db: float = -12.0,
    smooth_freq_hz: float = _GAIN_LP_FREQ_HZ,
) -> np.ndarray:
    """Gain-Riding: passt die Dynamik-Kurve des restaurierten Audios an das Original an.

    Args:
        audio_restored: Restauriertes Audio (mono/stereo, float32, 48000 Hz).
        audio_original: Original/Eingangs-Audio (gleiche Shape wie audio_restored).
        sr:             Abtastrate (48000 Hz).
        max_gain_db:    Maximale Verstärkung (dB).
        min_gain_db:    Maximale Absenkung (dB).
        smooth_freq_hz: Grenzfrequenz der Gain-Glättung (Hz).

    Returns:
        Gain-korrigiertes Audio (float32), gleiche Shape wie Eingabe.
        Bei Fehler: audio_restored unverändert (non-blocking).
    """
    try:
        return _apply_gain_riding(
            audio_restored,
            audio_original,
            sr,
            max_gain_ratio=float(10.0 ** (max_gain_db / 20.0)),
            min_gain_ratio=float(10.0 ** (min_gain_db / 20.0)),
            smooth_freq_hz=smooth_freq_hz,
        )
    except Exception as exc:
        logger.debug("AdaptiveGainRider nicht blockierend error: %s", exc)
        return audio_restored


# ---------------------------------------------------------------------------
# Kern-Implementierung
# ---------------------------------------------------------------------------


def _apply_gain_riding(
    restored: np.ndarray,
    original: np.ndarray,
    sr: int,
    max_gain_ratio: float,
    min_gain_ratio: float,
    smooth_freq_hz: float,
) -> np.ndarray:
    """Interne Gain-Riding-Implementierung."""
    if restored.shape != original.shape:
        logger.debug("AdaptiveGainRider: Shape-Mismatch (%s vs %s) — Passthrough", restored.shape, original.shape)
        return restored

    n_samples = restored.shape[-1] if restored.ndim == 2 else len(restored)
    if n_samples < int(sr * _RMS_FRAME_S * 4):
        return restored  # Zu kurz für sinnvolles Gain-Riding

    # Mono-Downmix für Analysekanal
    if restored.ndim == 2:
        rest_mono = restored.mean(axis=0)
        orig_mono = original.mean(axis=0)
    else:
        rest_mono = np.asarray(restored, dtype=np.float32)
        orig_mono = np.asarray(original, dtype=np.float32)

    # --- RMS-Frames berechnen ---
    frame_size = max(1, int(_RMS_FRAME_S * sr))
    n_frames = n_samples // frame_size

    rest_rms = np.array(
        [float(np.sqrt(np.mean(rest_mono[k * frame_size : (k + 1) * frame_size] ** 2))) for k in range(n_frames)],
        dtype=np.float64,
    )
    orig_rms = np.array(
        [float(np.sqrt(np.mean(orig_mono[k * frame_size : (k + 1) * frame_size] ** 2))) for k in range(n_frames)],
        dtype=np.float64,
    )

    # --- Gain-Ratio pro Frame ---
    # Stille-Zonen: Gain = 1.0 (kein Boost in Stille, §0h Music-Death-Shield)
    gain_ratio = np.ones(n_frames, dtype=np.float64)
    silence_mask = (orig_rms < _SILENCE_THRESHOLD_RMS) | (rest_rms < _SILENCE_THRESHOLD_RMS)
    active_mask = ~silence_mask

    if np.any(active_mask):
        gain_ratio[active_mask] = np.clip(
            orig_rms[active_mask] / np.maximum(rest_rms[active_mask], 1e-8),
            min_gain_ratio,
            max_gain_ratio,
        )

    # --- Onset-Hold: nach Onsets keine Gain-Sprünge (verhindert Pumpen) ---
    gain_diff = np.abs(np.diff(np.concatenate([[gain_ratio[0]], gain_ratio])))
    onset_frames = gain_diff > 0.20  # >20% Gain-Änderung → Onset-Zone
    for k in range(len(gain_ratio)):
        if onset_frames[k]:
            hold_end = min(k + _ONSET_GAIN_HOLD_FRAMES, len(gain_ratio) - 1)
            # Gain in Hold-Zone auf den Wert bei k fixieren
            gain_ratio[k + 1 : hold_end + 1] = gain_ratio[k]

    # --- Tiefpass-Glättung (Butterworth 2. Ordnung via bilinear) ---
    gain_smooth = _lowpass_smooth(gain_ratio, sr / frame_size, smooth_freq_hz)
    gain_smooth = np.clip(gain_smooth, min_gain_ratio, max_gain_ratio)
    # Stille-Zonen: Gain 1.0 erzwingen
    gain_smooth[silence_mask] = 1.0

    # --- Per-Sample Gain-Envelope interpolieren ---
    frame_centers = np.arange(n_frames) * frame_size + frame_size / 2.0
    sample_times = np.arange(n_samples, dtype=np.float64)
    gain_envelope = np.interp(sample_times, frame_centers, gain_smooth)
    gain_envelope = gain_envelope.astype(np.float32)

    # --- Anwenden ---
    if restored.ndim == 2:
        result = restored * gain_envelope[np.newaxis, :]
    else:
        result = restored * gain_envelope
    result = np.clip(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    _mean_gain_db = 20.0 * math.log10(float(np.mean(gain_smooth)) + 1e-10)
    logger.debug(
        "AdaptiveGainRider: n_frames=%d active=%d mean_gain=%.1fdB",
        n_frames,
        int(np.sum(active_mask)),
        _mean_gain_db,
    )
    return np.asarray(result, dtype=np.float32)  # type: ignore[no-any-return]


def _lowpass_smooth(signal: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """Einfaches IIR-Tiefpass-Filter erster Ordnung für Gain-Glättung.

    EMA (Exponential Moving Average) als Tiefpass-Approximation.
    """
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / max(sample_rate, 1.0))
    alpha = float(np.clip(alpha, 0.01, 0.99))
    out = np.zeros_like(signal)
    y = signal[0]
    for i, x in enumerate(signal):
        y = alpha * x + (1.0 - alpha) * y
        out[i] = y
    # Rückwärtsdurchlauf für zero-phase
    y = out[-1]
    for i in range(len(out) - 1, -1, -1):
        y = alpha * out[i] + (1.0 - alpha) * y
        out[i] = y
    return out  # type: ignore[no-any-return]
