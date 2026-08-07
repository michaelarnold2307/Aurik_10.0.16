"""Spectrogram Provider — Backend-Endpoint für GUI-Spektrogramm-Daten.

§Spektrogramm-Feedback: Liefert Frequenzspektrum-Daten als JSON für die
GUI-Spektrogramm-Anzeige. Berechnet Kurzzeit-Fourier-Transformation (STFT)
und gibt Frequenz-/Zeit-/Amplitude-Matrizen zurück.

Die GUI ruft diesen Endpoint auf, um Spektrogramme von Original und
restauriertem Audio nebeneinander darzustellen (Before/After).

Nutzung:
    from backend.core.spectrogram_provider import compute_spectrogram_data

    data = compute_spectrogram_data(audio, sr=48000)
    # data["frequencies"] → Frequenz-Achse (Hz)
    # data["times"] → Zeit-Achse (s)
    # data["magnitudes"] → dB-Spektrogramm (2D-Array)
    # data["max_db"], data["min_db"] → Dynamikbereich

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Konstanten ───────────────────────────────────────────────────────────────
DEFAULT_FFT_SIZE: int = 2048
DEFAULT_HOP_SIZE: int = 512
DEFAULT_DB_RANGE: float = 80.0  # Dynamikbereich in dB


@dataclass
class SpectrogramData:
    """Spektrogramm-Daten für GUI-Rendering."""

    frequencies: list[float]  # Hz, Frequenz-Achse
    times: list[float]  # Sekunden, Zeit-Achse
    magnitudes: list[list[float]]  # dB, 2D-Array [freq][time]
    max_db: float
    min_db: float
    sample_rate: int
    duration_s: float
    fft_size: int
    hop_size: int
    n_freqs: int
    n_times: int


def compute_spectrogram_data(
    audio: np.ndarray,
    sr: int = 48000,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    db_range: float = DEFAULT_DB_RANGE,
    max_freq: float | None = None,
    downsample_factor: int = 1,
) -> SpectrogramData:
    """Berechnet Spektrogramm-Daten für GUI-Darstellung.

    Args:
        audio:               float32/np.ndarray, mono [N] oder stereo [N, C].
        sr:                  Abtastrate in Hz.
        fft_size:            FFT-Fenstergröße (Default: 2048).
        hop_size:            Hop-Größe (Default: 512, ~75% Overlap).
        db_range:            Dynamikbereich in dB.
        max_freq:            Maximale Frequenz (None = Nyquist).
        downsample_factor:   Zeit- und Frequenz-Downsampling für GUI (1=voll).

    Returns:
        SpectrogramData mit Frequenzen, Zeiten und dB-Magnituden.
    """
    # ── Mono konvertieren ────────────────────────────────────────────────
    audio_mono = audio
    if audio.ndim > 1:
        audio_mono = np.mean(audio, axis=-1)

    audio_mono = audio_mono.astype(np.float32).flatten()
    n_samples = len(audio_mono)

    # ── STFT berechnen ────────────────────────────────────────────────────
    window = np.hanning(fft_size)

    n_frames = (n_samples - fft_size) // hop_size + 1
    if n_frames <= 0:
        raise ValueError(f"Audio zu kurz für STFT: {n_samples} samples, fft_size={fft_size}")

    # Nur reale Frequenz-Bins (0 bis Nyquist)
    n_freqs = fft_size // 2 + 1

    # Magnituden-Matrix bauen
    spec = np.zeros((n_freqs, n_frames), dtype=np.float32)

    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        end = start + fft_size
        frame = audio_mono[start:end] * window
        spectrum = np.abs(np.fft.rfft(frame, n=fft_size))
        spec[:, frame_idx] = spectrum

    # ── dB-Konvertierung ──────────────────────────────────────────────────
    # Referenz: max magnitude → 0 dB
    ref = np.max(spec) if np.max(spec) > 0 else 1e-10
    spec_db = 20 * np.log10(np.maximum(spec / ref, 1e-10))
    spec_db = np.clip(spec_db, -db_range, 0)

    # ── Frequenz- und Zeit-Achse ──────────────────────────────────────────
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sr)
    times = np.arange(n_frames) * hop_size / sr

    # Frequenz-Begrenzung
    if max_freq is not None:
        freq_mask = freqs <= max_freq
        freqs = freqs[freq_mask]
        spec_db = spec_db[freq_mask, :]

    # ── Downsampling für GUI ──────────────────────────────────────────────
    if downsample_factor > 1:
        # Zeit-Downsampling
        spec_db = spec_db[:, ::downsample_factor]
        times = times[::downsample_factor]
        # Frequenz-Downsampling
        spec_db = spec_db[::downsample_factor, :]
        freqs = freqs[::downsample_factor]

    # ── Listen konvertieren für JSON-Serialisierung ──────────────────────
    freq_list = [round(float(f), 1) for f in freqs]
    time_list = [round(float(t), 3) for t in times]

    # Magnituden als 2D-Liste (freq-major: [freq][time])
    mag_list: list[list[float]] = []
    for fi in range(spec_db.shape[0]):
        row = [round(float(v), 2) for v in spec_db[fi, :].tolist()]
        mag_list.append(row)

    duration_s = n_samples / sr

    return SpectrogramData(
        frequencies=freq_list,
        times=time_list,
        magnitudes=mag_list,
        max_db=0.0,
        min_db=-db_range,
        sample_rate=sr,
        duration_s=duration_s,
        fft_size=fft_size,
        hop_size=hop_size,
        n_freqs=len(freq_list),
        n_times=len(time_list),
    )


def compute_before_after_spectrograms(
    original: np.ndarray,
    restored: np.ndarray,
    sr: int = 48000,
    fft_size: int = 2048,
    hop_size: int = 512,
    downsample_factor: int = 2,
) -> dict[str, Any]:
    """Berechnet Original- und Restauriert-Spektrogramme für GUI.

    Args:
        original:  Original-Audio (vor Restaurierung).
        restored:  Restauriertes Audio (nach Aurik).
        sr:        Abtastrate.
        fft_size:  FFT-Größe.
        hop_size:  Hop-Größe.
        downsample_factor: Downsampling für GUI-Performance.

    Returns:
        Dict mit "original" und "restored" SpectrogramData (als Dict).
    """
    try:
        spec_orig = compute_spectrogram_data(
            original,
            sr,
            fft_size,
            hop_size,
            downsample_factor=downsample_factor,
        )
    except ValueError as e:
        logger.warning("Originalsignal-Spektrogramm konnte nicht berechnet werden: %s", e)
        spec_orig = None

    try:
        spec_rest = compute_spectrogram_data(
            restored,
            sr,
            fft_size,
            hop_size,
            downsample_factor=downsample_factor,
        )
    except ValueError as e:
        logger.warning("Restauriert-Spektrogramm konnte nicht berechnet werden: %s", e)
        spec_rest = None

    result: dict[str, Any] = {}

    if spec_orig is not None:
        result["original"] = {
            "frequencies": spec_orig.frequencies,
            "times": spec_orig.times,
            "magnitudes": spec_orig.magnitudes,
            "max_db": spec_orig.max_db,
            "min_db": spec_orig.min_db,
            "duration_s": spec_orig.duration_s,
            "n_freqs": spec_orig.n_freqs,
            "n_times": spec_orig.n_times,
        }

    if spec_rest is not None:
        result["restored"] = {
            "frequencies": spec_rest.frequencies,
            "times": spec_rest.times,
            "magnitudes": spec_rest.magnitudes,
            "max_db": spec_rest.max_db,
            "min_db": spec_rest.min_db,
            "duration_s": spec_rest.duration_s,
            "n_freqs": spec_rest.n_freqs,
            "n_times": spec_rest.n_times,
        }

    result["sample_rate"] = sr
    result["fft_size"] = fft_size
    result["hop_size"] = hop_size

    return result


def spectrogram_to_json(data: SpectrogramData) -> str:
    """Konvertiert SpectrogramData nach JSON (für REST-Endpoint)."""
    import json

    return json.dumps(data.__dict__, ensure_ascii=False)


__all__ = [
    "SpectrogramData",
    "compute_spectrogram_data",
    "compute_before_after_spectrograms",
    "spectrogram_to_json",
]
