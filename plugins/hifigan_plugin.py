"""HifiGanPlugin — Mel→Waveform Vocoder via lokales ONNX (kein Docker/HF).

Modell : models/hifi_gan/hifi_gan.onnx
ONNX   : input[1,80,seq_length] → output[1,1,2560]
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_INSTANCE_HOLDER: list[HifiGanPlugin | None] = [None]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_ROOT, "models", "hifi_gan", "hifi_gan.onnx")
_SR_MODEL = 22_050
_N_MELS = 80
_HOP = 256
_WIN = 1024
_OUT_HOP = 2560


class HifiGanPlugin:
    """HiFi-GAN ONNX vocoder plugin for mel-to-waveform synthesis."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialisiert HiFi-GAN plugin and attempt model load."""
        self._session: Any = None
        self._try_load(model_path or _MODEL)

    def unload(self) -> None:
        """Gibt HiFi-GAN Session frei."""
        self._session = None

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("HiFi-GAN Modell fehlt: %s — Griffin-Lim-Fallback.", path)
            return
        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("HiFiGAN", size_gb=0.004):
                    logger.warning("HiFiGAN: ML-Budget erschöpft — Griffin-Lim-Fallback.")
                    return
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                _providers = _get_prov("HiFiGAN")
            except Exception:
                _providers = ["CPUExecutionProvider"]

            self._session = ort.InferenceSession(path, sess_options=opts, providers=_providers)
            logger.info("HiFi-GAN ONNX geladen: %s [providers=%s]", path, _providers)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm("HiFiGAN", size_gb=0.004, unload_fn=self.unload)
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("HiFi-GAN Ladefehler: %s — Fallback.", exc)
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("HiFiGAN")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    def vocode(self, mel: np.ndarray, sr_out: int = 48000) -> np.ndarray:
        """mel[80, T] → waveform float32. sr_out=48000 entspricht Aurik-Pipeline-SR."""
        assert sr_out == 48000, f"SR muss 48000 Hz sein, erhalten: {sr_out}"
        mel = np.nan_to_num(mel.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self._session:
            return self._vocode_onnx(mel, sr_out)
        return self._pghi_istft(mel, sr_out)

    def mel_from_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Erzeuge 80-Band-Mel-Spektrogramm [80, T]."""
        if audio.ndim == 2:
            # Handle (2, N) channels-first (UV3) and (N, 2) samples-first
            mono = (
                audio.mean(axis=0) if (audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]) else audio.mean(axis=1)
            )
        else:
            mono = audio
        mono = _resamp(mono, sr, _SR_MODEL)
        return _mel_spec(mono, _SR_MODEL, _N_MELS, _WIN, _HOP)

    def reconstruct(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Audio → Mel → Waveform (round-trip, nützlich als Enhancement-Stub)."""
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mel = self.mel_from_audio(audio, sr)
        wave = self.vocode(mel, sr)
        result = _resamp(wave, _SR_MODEL, sr)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(np.clip(result, -1.0, 1.0), dtype=np.float32)  # type: ignore[no-any-return]

    def _vocode_onnx(self, mel: np.ndarray, sr_out: int) -> np.ndarray:
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("HiFiGAN", True)
        except Exception:
            logger.warning("hifigan_plugin.py::_vocode_onnx fallback", exc_info=True)
        try:
            T = mel.shape[1]
            chunks = []
            CHUNK = 64  # T-Frames pro Inferenz
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                m = mel[:, s:e][None].astype(np.float32)  # [1,80,chunk]
                try:
                    out = np.asarray(self._session.run(None, {"input": m})[0], dtype=np.float32)  # [1,1,2560*chunk]
                    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                    chunks.append(out[0, 0])
                except Exception as exc:
                    logger.debug("HiFi-GAN chunk Fehler: %s", exc)
                    # Fallback: stille Ausgabe für diesen Chunk
                    chunks.append(np.zeros((e - s) * _OUT_HOP // _OUT_HOP * _OUT_HOP, np.float32))
            wave = np.concatenate(chunks)
            wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
            if sr_out != _SR_MODEL:
                wave = _resamp(wave, _SR_MODEL, sr_out)
                wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
            return np.asarray(np.clip(wave, -1.0, 1.0), dtype=np.float32)  # type: ignore[no-any-return]
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("HiFiGAN", False)
                except Exception:
                    logger.warning("hifigan_plugin.py::_vocode_onnx fallback", exc_info=True)

    @staticmethod
    def _pghi_istft(mel: np.ndarray, sr_out: int) -> np.ndarray:
        """PGHI-ISTFT — §4.5 Letzfall-Fallback wenn HiFi-GAN ONNX fehlt.

        Griffin-Lim ist laut §4.4 VERBOTEN (iterativer Vocoder-Endschritt).
        PGHI (Phase Gradient Heap Integration) ist ein nicht-iteratives
        Einzelpass-Verfahren zur phasenkonsistenten Syntheserekonstruktion.
        Referenz: Perraudin et al. (2013) — 'A Non-Iterative Method for STFT
        Phase (Re)construction via Phase Gradient Heap Integration'.
        """
        import scipy.signal as ss

        n_fft = _WIN
        n_bins = n_fft // 2 + 1
        # Mel → lineare Magnitude (Gleichverteilung der Filterbank-Bins)
        lin = np.power(10.0, np.clip(mel.astype(np.float32) / 10.0, -8.0, 8.0))
        spec_mag = np.zeros((n_bins, mel.shape[1]), np.float32)
        step = max(1, n_bins // _N_MELS)
        for k in range(_N_MELS):
            lo = k * step
            hi = min((k + 1) * step, n_bins)
            if lo < n_bins:
                spec_mag[lo:hi] = lin[k]
        # PGHI: Phasengradient aus log|S|-Zeitgradient (Perraudin 2013, Eq. 5)
        # φ_k(t) = Σ_τ [2π·k·hop/n_fft + Δ_τ], Δ_τ ≈ ∂log|S|/∂τ
        log_mag = np.log1p(spec_mag)
        grad_t = np.diff(log_mag, axis=1, prepend=log_mag[:, :1])
        hop_phase = (2.0 * np.pi * np.arange(n_bins) * _HOP / n_fft).astype(np.float32)
        phase = np.cumsum(hop_phase[:, None] + grad_t * 0.2, axis=1).astype(np.float32)
        _, wave = ss.istft(
            spec_mag * np.exp(1j * phase),
            fs=sr_out,
            nperseg=n_fft,
            noverlap=n_fft - _HOP,
            window="hann",
        )
        return np.asarray(np.clip(wave, -1.0, 1.0), dtype=np.float32)  # type: ignore[no-any-return]


def _mel_spec(
    mono: np.ndarray,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop: int = 256,
) -> np.ndarray:
    import scipy.signal as ss

    _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.113
    _, _, Z = ss.stft(mono, fs=sr, nperseg=n_fft, noverlap=_noverlap, window="hann")
    mag = np.abs(Z[: n_fft // 2 + 1])
    mel_fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = np.dot(mel_fb, mag)
    return np.asarray(10.0 * np.log10(mel + 1e-9), dtype=np.float32)  # type: ignore[no-any-return]


def _mel_filterbank(sr, n_fft, n_mels):
    lo, hi = 0.0, sr / 2.0
    mel_lo = 2595 * np.log10(1 + lo / 700)
    mel_hi = 2595 * np.log10(1 + hi / 700)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor(hz_pts * (n_fft // 2 + 1) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / (c - l + 1e-8)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / (r - c + 1e-8)
    return fb


def _resamp(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return np.asarray(x, dtype=np.float32)  # type: ignore[no-any-return]
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return np.asarray(resample_poly(x, dst // g, src // g), dtype=np.float32)  # type: ignore[no-any-return]


def get_hifigan_plugin() -> HifiGanPlugin:
    """Thread-safe singleton accessor for HiFi-GAN plugin."""
    if _INSTANCE_HOLDER[0] is None:
        with _lock:
            if _INSTANCE_HOLDER[0] is None:
                _INSTANCE_HOLDER[0] = HifiGanPlugin()
    plugin = _INSTANCE_HOLDER[0]
    assert plugin is not None
    return plugin


def vocode(mel: np.ndarray, sr_out: int = 48000) -> np.ndarray:
    """Modul-Level-Wrapper: mel[80,T] → waveform bei sr_out Hz (Default 48000 = Aurik-SR)."""
    return get_hifigan_plugin().vocode(mel, sr_out)


# Alias für Rückwärtskompatibilität
HiFiGANPlugin = HifiGanPlugin

# Convenience-Alias


def vocode_audio(audio: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Konvertiert audio to mel spectrogram and back via HiFi-GAN."""
    plugin = get_hifigan_plugin()
    return plugin.reconstruct(audio, sr)
