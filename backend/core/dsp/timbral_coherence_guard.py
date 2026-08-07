"""
§Cross-Segment Timbral Coherence Guard — konsistenter Rauschboden (v10.0.0)

Zweck: Stellt sicher, dass der Rauschboden-Charakter (spektrale Form, Textur)
über alle Segmente eines Songs konsistent bleibt. Verhindert, dass NR-Phasen
in ruhigen Passagen stärker angreifen als in lauten, was zu hörbaren
"Pumping"- oder "Switching"-Artefakten führt.

§0a Restoration: Rauschboden-*Niveau* UND -*Textur* des originalen
Aufnahmemediums bewahren. Spectral Form des Restrauschens muss dem
Trägerprofil entsprechen.

Implementierung:
1. Segmentiere Song in 30s-Fenster
2. Bestimme das "sauberste" Segment (höchstes SNR, niedrigste Defect-Severity)
3. Extrahiere dessen Spektral-Noise-Profil als Referenz
4. Stelle sicher, dass alle anderen Segmente nicht unter diesen Rauschboden fallen
   (verhindert "Over-Cleaning" bei lauten Passagen, "Under-Cleaning" bei leisen)
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Segmentlänge für Profil-Extraktion (s)
_PROFILE_SEGMENT_S = 30.0

# Hop zwischen Segmenten (s)
_PROFILE_HOP_S = 15.0

# Mindestlänge für Analyse (s)
_MIN_DURATION_S = 5.0

# Frequenz-Bins für spektrales Profil (logarithmisch verteilt)
_N_FREQ_BINS = 32

# Mindest-Kohärenz-Score (§0a Spec: ≥ 0.80 Pflicht)
COHERENCE_SCORE_MIN = 0.80


def _estimate_noise_spectrum(
    audio: npt.NDArray[np.float32],
    sr: int,
    n_bins: int = _N_FREQ_BINS,
) -> npt.NDArray[np.float64]:
    """
    Schätzt das Rausch-Spektralprofil eines Audio-Segments.

    Methode: Tiefstes 10. Percentile des STFT-Magnitudenspektrums
    (über Zeitachse), logarithmisch auf n_bins Bänder reduziert.

    Returns:
        Logarithmisches Spektralprofil (n_bins,), normiert [0,1]
    """
    mono: np.ndarray
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1 if audio.shape[1] <= 8 else 0).astype(np.float64)
    else:
        mono = audio.astype(np.float64)

    if len(mono) < 512:
        return np.zeros(n_bins)

    try:
        hop_length = 512
        n_fft = 2048
        win = np.hanning(n_fft)
        # STFT frames
        frames = []
        for i in range(0, len(mono) - n_fft, hop_length):
            frame = mono[i : i + n_fft] * win
            spec = np.abs(np.fft.rfft(frame))
            frames.append(spec)

        if not frames:
            return np.zeros(n_bins)

        stft_mag = np.array(frames)  # (T, F)
        # 10. Percentile über Zeit (Rauschboden-Schätzung)
        noise_profile = np.percentile(stft_mag, 10, axis=0)

        n_freqs = noise_profile.shape[0]
        # Log-Mel-ähnliche Frequenz-Aggregation auf n_bins
        freq_edges = np.logspace(np.log10(max(20.0, sr / n_fft)), np.log10(sr / 2.0), n_bins + 1)
        bin_freqs = np.linspace(0, sr / 2.0, n_freqs)

        aggregated = np.zeros(n_bins)
        for k in range(n_bins):
            mask = (bin_freqs >= freq_edges[k]) & (bin_freqs < freq_edges[k + 1])
            if np.any(mask):
                aggregated[k] = float(np.mean(noise_profile[mask]))

        # Normieren [0,1]
        max_val: float = float(np.max(aggregated))
        if max_val > 1e-10:
            aggregated /= max_val

        return aggregated

    except Exception as exc:
        logger.debug("_estimate_noise_spectrum Fehler (unkritisch): %s", exc)
        return np.zeros(n_bins)


def extract_song_noise_profile(
    audio: npt.NDArray[np.float32],
    sr: int,
) -> npt.NDArray[np.float64]:
    """
    Extrahiert das Referenz-Rauschprofil des Songs.

    Wählt das Segment mit niedrigstem Energieniveau (stille Passagen
    haben weniger Musiksignal → Rauschboden ist sichtbarer).

    Args:
        audio: Eingangssignal (mono/stereo)
        sr:    Abtastrate

    Returns:
        Spektralprofil (n_bins,) als Rausch-Referenz
    """
    duration_s = audio.shape[-1] / sr if audio.ndim == 2 else len(audio) / sr
    if duration_s < _MIN_DURATION_S:
        return _estimate_noise_spectrum(audio, sr)

    seg_len = int(_PROFILE_SEGMENT_S * sr)
    hop_len = int(_PROFILE_HOP_S * sr)
    n_samples = audio.shape[-1] if audio.ndim == 2 else len(audio)

    # Segmente und deren RMS
    segment_rms: list[tuple[float, int]] = []  # (rms, start)
    for start in range(0, n_samples - seg_len, hop_len):
        if audio.ndim == 2:
            seg = audio[:, start : start + seg_len]
        else:
            seg = audio[start : start + seg_len]
        rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        segment_rms.append((rms, start))

    if not segment_rms:
        return _estimate_noise_spectrum(audio, sr)

    # Niedrigstes Quartil (ruhigste Segmente = Rauschboden-Sichtbarkeit)
    segment_rms.sort(key=lambda x: x[0])
    n_quiet = max(1, len(segment_rms) // 4)
    quiet_profiles: list[npt.NDArray[np.float64]] = []

    for _, start in segment_rms[:n_quiet]:
        if audio.ndim == 2:
            seg = audio[:, start : start + seg_len]
        else:
            seg = audio[start : start + seg_len]
        profile = _estimate_noise_spectrum(seg.astype(np.float32), sr)
        quiet_profiles.append(profile)

    # Median-Profil aus ruhigen Segmenten
    profile_arr = np.array(quiet_profiles)
    reference_profile: npt.NDArray[np.float64] = np.asarray(np.median(profile_arr, axis=0), dtype=np.float64)

    logger.debug(
        "§CSTC song_noise_Profil: %d Segmente analysiert, Referenz aus %d ruhigsten",
        len(segment_rms),
        n_quiet,
    )
    return reference_profile


def compute_timbral_coherence_score(
    audio_processed: npt.NDArray[np.float32],
    reference_profile: npt.NDArray[np.float64],
    sr: int,
) -> dict[str, object]:
    """
    Prüft ob das prozessierte Audio denselben Rauschboden-Charakter hat
    wie der Referenz-Song-Profile.

    §0a: Spektrale Form des Restrauschens muss dem Träger entsprechen
    (Kohärenz-Score ≥ 0.80 Pflicht).

    Args:
        audio_processed:  Prozessiertes Audio (nach NR-Phasen)
        reference_profile: Referenz-Rauschprofil aus extract_song_noise_profile()
        sr:               Abtastrate

    Returns:
        dict mit 'coherence_score', 'spectral_drift', 'pass' (bool)
    """
    post_profile = _estimate_noise_spectrum(audio_processed, sr, n_bins=len(reference_profile))

    # Spektrale Korrelation (Pearson)
    if np.std(reference_profile) < 1e-8 or np.std(post_profile) < 1e-8:
        coherence = 1.0 if np.allclose(reference_profile, post_profile, atol=1e-4) else 0.5
    else:
        coherence = float(
            np.dot(reference_profile - reference_profile.mean(), post_profile - post_profile.mean())
            / (
                np.linalg.norm(reference_profile - reference_profile.mean())
                * np.linalg.norm(post_profile - post_profile.mean())
                + 1e-10
            )
        )
        coherence = float(np.clip(coherence, 0.0, 1.0))

    # Spektrale Drift (L1-Distanz der normierten Profile)
    spectral_drift = float(np.mean(np.abs(reference_profile - post_profile)))

    result: dict[str, object] = {
        "coherence_score": coherence,
        "spectral_drift": spectral_drift,
        "pass": bool(coherence >= COHERENCE_SCORE_MIN),
    }

    if not result["pass"]:
        logger.warning(
            "§CSTC Timbral-Kohärenz FAIL: Wert=%.2f < %.2f (spectral_drift=%.3f) — Rauschboden-Textur verändert",
            coherence,
            COHERENCE_SCORE_MIN,
            spectral_drift,
        )
    else:
        logger.debug(
            "§CSTC Timbral-Kohärenz OK: Wert=%.2f (spectral_drift=%.3f)",
            coherence,
            spectral_drift,
        )

    return result


def apply_timbral_coherence_correction(
    audio_processed: npt.NDArray[np.float32],
    reference_profile: npt.NDArray[np.float64],
    sr: int,
    max_correction_db: float = 6.0,
) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
    """
    Korrigiert spektrale Drift im Rauschboden durch frequenzselektive Anpassung.

    Bei Kohärenz-Fail wird eine frequenzselektive Gain-Korrektur angewendet,
    die die spektrale Form des Rauschbodens zur Referenz hin anpasst.

    §0h: Maximale Korrektur = max_correction_db. Korrektur ist additiv
    (Boost verbotener Frequenzen) oder subtraktiv (Dämpfung überbetonter).
    Kein Klippen, kein Artefakt.

    Args:
        audio_processed:   Prozessiertes Audio
        reference_profile: Referenz-Rauschprofil
        sr:                Abtastrate
        max_correction_db: Max. EQ-Korrektur in dB

    Returns:
        (korrigiertes_audio, diagnose_dict)
    """
    diag = compute_timbral_coherence_score(audio_processed, reference_profile, sr)

    if diag.get("pass", True):
        return audio_processed, diag

    # Frequenzselektive Korrektur via Spektral-Multiplikation
    try:
        mono: np.ndarray
        is_stereo = audio_processed.ndim == 2
        if is_stereo:
            # Verwende Mittelkanal für Profil
            if audio_processed.shape[0] == 2 and audio_processed.shape[1] > 2:
                mono = np.mean(audio_processed, axis=0).astype(np.float64)
                is_channels_first = True
            else:
                mono = np.mean(audio_processed, axis=1).astype(np.float64)
                is_channels_first = False
        else:
            mono = audio_processed.astype(np.float64)
            is_channels_first = False

        post_profile = _estimate_noise_spectrum(audio_processed, sr, n_bins=len(reference_profile))

        # Gain-Kurve: ratio reference/post pro Band (gemessen in Noise-Profil)
        ratio = np.where(post_profile > 1e-10, reference_profile / np.clip(post_profile, 1e-10, None), 1.0)

        # Auf dB begrenzen
        ratio_db = np.clip(20.0 * np.log10(np.clip(ratio, 1e-5, None)), -max_correction_db, max_correction_db)
        gain = 10.0 ** (ratio_db / 20.0)

        # Gain-Interpolation auf volle Frequenzauflösung (STFT)
        n_fft = 2048
        n_freqs = n_fft // 2 + 1
        n_bins = len(gain)
        gain_full = np.interp(
            np.linspace(0, n_bins - 1, n_freqs),
            np.arange(n_bins),
            gain,
        )

        # Anwenden via STFT → spectral multiplication → iSTFT
        hop_length = 512
        window = np.hanning(n_fft)
        out = np.zeros_like(mono)
        norm = np.zeros_like(mono)

        for i in range(0, len(mono) - n_fft, hop_length):
            frame = mono[i : i + n_fft] * window
            spec = np.fft.rfft(frame)
            spec_corrected = spec * gain_full
            frame_out = np.fft.irfft(spec_corrected).real[:n_fft] * window
            out[i : i + n_fft] += frame_out
            norm[i : i + n_fft] += window**2

        # OLA-Normierung
        norm = np.where(norm > 1e-8, norm, 1.0)
        out /= norm

        # Rückkonstruktion auf Original-Format
        out_f = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out_f = np.clip(out_f, -1.0, 1.0)

        if is_stereo:
            if is_channels_first:
                # Anwenden auf beide Kanäle
                out_stereo = audio_processed.copy()
                for ch in range(audio_processed.shape[0]):
                    out_stereo[ch] = np.clip(audio_processed[ch] * (out_f / (mono + 1e-10)), -1.0, 1.0)
                corrected = out_stereo.astype(np.float32)
            else:
                out_stereo = audio_processed.copy()
                for ch in range(audio_processed.shape[1]):
                    out_stereo[:, ch] = np.clip(audio_processed[:, ch] * (out_f / (mono + 1e-10)), -1.0, 1.0)
                corrected = out_stereo.astype(np.float32)
        else:
            corrected = out_f.astype(np.float32)

        logger.debug(
            "§CSTC Timbral-Kohärenz-Korrektur angewendet (max_db=%.1f dB)",
            max_correction_db,
        )
        diag["correction_applied"] = True
        return corrected, diag

    except Exception as exc:
        logger.debug("§CSTC Timbral-Kohärenz-Korrektur fehlgeschlagen (unkritisch): %s", exc)
        diag["correction_applied"] = False
        return audio_processed, diag
