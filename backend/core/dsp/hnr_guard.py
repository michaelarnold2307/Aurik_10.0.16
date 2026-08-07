"""
§2.35c HNR-Guard — Harmonics-to-Noise Ratio Gate für NR-Phasen (v10.0.0)

Zweck: Verhindert, dass Denoise-Phasen die natürliche Stimmrauigkeit
(Vocal Fry, Breathiness, Stimmklang-Charakter) vollständig beseitigen.

Psychoakustik: Stimmrauigkeit (HNR 5–15 dB) ist ein wesentlicher Teil
des Vokal-Timbres. Übermäßiges NR erhöht den HNR auf >20 dB und erzeugt
"klinischen" Klang. ΔHNR > +3 dB → Dry/Wet-Blend zur Rauigkeits-Erhaltung.

§0h Invariante: Kein Guard überschreibt Artefakt-Freiheit. Wenn Dry-Blend
ein Artefakt aufweist, bleibt der NR-Output unverändert (besser NR als
Rausch-Artefakt).
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Maximale akzeptable HNR-Verbesserung durch NR (§2.35c).
# Werte > 3 dB bedeuten, der Algorithmus hat mehr als Rauschen entfernt —
# er hat auch harmonische Rauigkeit getilgt.
_HNR_DELTA_THRESHOLD_DB = 3.0

# Minimale Audiolänge für HNR-Schätzung (ms)
_MIN_AUDIO_MS = 100

# Fensterbreite für ACF-basierte HNR-Schätzung (ms)
_ACF_FRAME_MS = 40.0

# Hop-Schrittweite (ms)
_ACF_HOP_MS = 10.0

# Minimum Energie für stimmhafte Frames
_VOICED_RMS_THRESHOLD = 1e-3


def compute_hnr(audio: npt.NDArray[np.float32], sr: int) -> float:
    """
    Berechnet den Harmonics-to-Noise Ratio (HNR) in dB.

    Methode: ACF-basiert (Boersma 1993 vereinfacht).
    - HNR = 10 * log10(r(T0) / (r(0) - r(T0)))
    - r(τ) = normierte Autokorrelation des Signals
    - T0 = geschätzte Grundfrequenz-Periode via Peak-ACF

    Wertebereich:
    - HNR ~5 dB : rauhe, heisere Stimmen (Vocal Fry)
    - HNR ~10 dB: normale Sprechstimme
    - HNR ~20+ dB: sehr klare/saubere Singstimme
    - HNR = 0 dB: Rauschen dominiert (kein tonaler Anteil)

    Args:
        audio: float32 Mono-Signal
        sr:    Abtastrate in Hz

    Returns:
        HNR in dB. Bei Fehler/ungestimmtem Material: 0.0
    """
    mono: np.ndarray
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1 if audio.shape[1] <= 8 else 0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    n_min = int(_MIN_AUDIO_MS / 1000.0 * sr)
    if len(mono) < n_min:
        return 0.0

    frame_len = int(_ACF_FRAME_MS / 1000.0 * sr)
    hop_len = int(_ACF_HOP_MS / 1000.0 * sr)

    if frame_len < 64:
        return 0.0

    # Fensterbasierter Pitch-Bereich: F0 20–500 Hz → Lag-Bereich
    lag_min = max(1, int(sr / 500.0))
    lag_max = min(frame_len // 2, int(sr / 20.0))

    hnr_values: list[float] = []

    for fi in range(0, len(mono) - frame_len, hop_len):
        frame = mono[fi : fi + frame_len]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < _VOICED_RMS_THRESHOLD:
            continue

        frame_w = frame * np.hanning(len(frame))
        # Normierte ACF via FFT (schneller als direkte Berechnung)
        fft_len = 2 * frame_len
        spectrum = np.fft.rfft(frame_w, n=fft_len)
        acf_full = np.fft.irfft(np.abs(spectrum) ** 2).real
        acf_norm_val = acf_full[0]
        if acf_norm_val < 1e-12:
            continue

        acf = acf_full / acf_norm_val  # normierte ACF r(0)=1.0

        # Peak-Suche im Pitch-Lag-Bereich
        acf_window = acf[lag_min : lag_max + 1]
        if len(acf_window) == 0:
            continue

        peak_idx = int(np.argmax(acf_window))
        r_t0 = float(acf_window[peak_idx])

        # HNR = 10 log10(r(T0) / (1 - r(T0)))
        r_t0 = np.clip(r_t0, 0.0, 0.9999)
        if r_t0 <= 0.0:
            continue

        hnr_frame = 10.0 * np.log10(r_t0 / max(1.0 - r_t0, 1e-10))
        hnr_values.append(float(np.clip(hnr_frame, -10.0, 40.0)))

    if not hnr_values:
        return 0.0

    # Robuster Median über alle stimmhaften Frames
    return float(np.median(hnr_values))


def check_hnr_delta(
    audio_pre: npt.NDArray[np.float32],
    audio_post: npt.NDArray[np.float32],
    sr: int,
) -> dict[str, object]:
    """
    Prüft ob NR zu viel Stimmrauigkeit entfernt hat (ΔHNR > Schwellwert).

    Args:
        audio_pre:  Audio VOR NR-Phase
        audio_post: Audio NACH NR-Phase
        sr:         Abtastrate in Hz

    Returns:
        dict mit:
          'hnr_pre':      HNR vor NR in dB
          'hnr_post':     HNR nach NR in dB
          'delta_hnr':    Differenz (positiv = HNR gestiegen = Rauigkeit reduziert)
          'over_cleaned': bool — True wenn ΔHNR > _HNR_DELTA_THRESHOLD_DB
          'blend_ratio':  Empfohlener Dry-Anteil [0,1] um Rauigkeit zu restaurieren
    """
    hnr_pre = compute_hnr(audio_pre, sr)
    hnr_post = compute_hnr(audio_post, sr)
    delta = hnr_post - hnr_pre

    over_cleaned = delta > _HNR_DELTA_THRESHOLD_DB

    # Blend-Empfehlung: proportional zur Überschreitung
    # Ziel: ΔHNR auf Schwellwert begrenzen
    if over_cleaned and delta > 0.0:
        # Wie viel "Dry" brauchen wir um von hnr_post auf hnr_pre + Schwelle zu kommen?
        target_hnr = hnr_pre + _HNR_DELTA_THRESHOLD_DB
        # Linear angenähert: blend_ratio = (hnr_post - target_hnr) / (hnr_post - hnr_pre)
        blend_ratio = float(np.clip((hnr_post - target_hnr) / max(delta, 0.01), 0.05, 0.60))
    else:
        blend_ratio = 0.0

    result: dict[str, object] = {
        "hnr_pre": float(hnr_pre),
        "hnr_post": float(hnr_post),
        "delta_hnr": float(delta),
        "over_cleaned": bool(over_cleaned),
        "blend_ratio": float(blend_ratio),
    }

    if over_cleaned:
        logger.debug(
            "§HNR-Guard: ΔHNR=+%.1f dB (%.1f→%.1f dB) > Schwelle %.1f dB — Dry-Blend %.0f%% empfohlen",
            delta,
            hnr_pre,
            hnr_post,
            _HNR_DELTA_THRESHOLD_DB,
            blend_ratio * 100.0,
        )

    return result


def apply_hnr_blend(
    audio_pre: npt.NDArray[np.float32],
    audio_post: npt.NDArray[np.float32],
    sr: int,
) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
    """
    Wendet HNR-Guard an und mischt Dry-Signal ein falls nötig.

    Gibt den (möglicherweise korrigierten) Audio-Output und das Diagnose-Dict zurück.
    Wenn kein Over-Cleaning vorliegt, wird audio_post unverändert zurückgegeben.

    §0h Invariante: Wenn Dry-Blend ein neues Artefakt einführt (NaN/Inf nach Mix),
    wird audio_post unverändert zurückgegeben.

    Args:
        audio_pre:  Audio VOR NR-Phase
        audio_post: Audio NACH NR-Phase
        sr:         Abtastrate in Hz

    Returns:
        (korrigiertes_audio, diagnose_dict)
    """
    diag = check_hnr_delta(audio_pre, audio_post, sr)

    if not diag.get("over_cleaned", False):
        return audio_post, diag

    _br_raw = diag.get("blend_ratio", 0.0)
    blend_ratio = float(_br_raw) if isinstance(_br_raw, (int, float)) else 0.0
    if blend_ratio <= 0.0:
        return audio_post, diag

    # Längenanpassung (§2.61 Output-Length-Guard)
    n_out = (
        len(audio_post)
        if audio_post.ndim == 1
        else audio_post.shape[0]
        if audio_post.shape[0] <= 8
        else audio_post.shape[-1]
    )
    n_pre = len(audio_pre) if audio_pre.ndim == 1 else audio_pre.shape[-1]
    if n_pre > n_out:
        audio_pre_aligned = audio_pre[..., :n_out]
    elif n_pre < n_out:
        pad_width = [(0, 0)] * audio_pre.ndim
        pad_width[-1] = (0, n_out - n_pre)
        audio_pre_aligned = np.pad(audio_pre, pad_width, mode="edge")
    else:
        audio_pre_aligned = audio_pre

    try:
        blended = (1.0 - blend_ratio) * audio_post + blend_ratio * audio_pre_aligned
        blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)
        blended = np.clip(blended, -1.0, 1.0).astype(np.float32)
        logger.debug(
            "§HNR-Guard: Dry-Blend %.0f%% angewendet — Stimmrauigkeit restauriert",
            blend_ratio * 100.0,
        )
        return blended, diag
    except Exception as exc:
        logger.debug("§HNR-Guard: Blend fehlgeschlagen (%s) — NR-Ausgabe unverändert", exc)
        return audio_post, diag
