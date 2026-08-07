"""Aurik 10.0.0 — Standalone DSP-Modelle (scipy/numpy, kein torch).

Werden von ModelManager.multi_stage_enhancement() als Fallback-Kette verwendet:
  denoiser  → sibilant → authenticity
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DenoiserModel:
    """Spektrale Subtraktion als Rauschunterdrückung."""

    def process(self, audio: np.ndarray, context: dict) -> np.ndarray:
        try:
            from scipy.signal import istft, stft

            sr = int(context.get("sr", 44100))
            nperseg = 1024
            noverlap = nperseg * 3 // 4
            _, _, Zxx = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
            # Rauschschätzung über die 5% leisesten Frames
            mag = np.abs(Zxx)
            frame_energy = np.sum(mag, axis=0)
            n_noise = max(1, int(0.05 * mag.shape[1]))
            noise_frames = np.argsort(frame_energy)[:n_noise]
            noise_profile = np.mean(mag[:, noise_frames], axis=1, keepdims=True)
            alpha = float(context.get("alpha", 1.5))
            beta = float(context.get("beta", 0.002))
            gain_sq = np.maximum(mag**2 - alpha * noise_profile**2, beta * mag**2)
            gain = np.sqrt(gain_sq) / (mag + 1e-12)
            Zxx_out = Zxx * gain
            _, y = istft(Zxx_out, fs=sr, nperseg=nperseg, noverlap=noverlap)
            n = len(audio)
            if len(y) >= n:
                return y[:n].astype(audio.dtype)  # type: ignore[no-any-return]
            return np.pad(y, (0, n - len(y))).astype(audio.dtype)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("dummy_models.py::verarbeiten Ersatzpfad: %s", e)
            return audio


class SibilantModel:
    """Spektraler De-Esser (Sibilanten-Unterdrückung 4–12 kHz)."""

    def process(self, audio: np.ndarray, context: dict) -> np.ndarray:
        try:
            from dsp.deesser_ml import MLDeEsser

            reduction_db = float(context.get("reduction_db", 6.0))
            sr = int(context.get("sr", 44100))
            de_esser = MLDeEsser(reduction_db=reduction_db)
            return de_esser.process(audio, sr).astype(audio.dtype)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("dummy_models.py::verarbeiten Ersatzpfad: %s", e)
            return audio


class AuthenticityModel:
    """Weiche Sättigung zur Authentizitäts-Verstärkung (Tape-Sättigung)."""

    def process(self, audio: np.ndarray, context: dict) -> np.ndarray:
        try:
            drive = float(context.get("drive", 1.5))
            # Weiche Sättigung: y = tanh(drive * x) / tanh(drive)
            norm = float(np.tanh(drive))
            if norm < 1e-6:
                return audio
            saturated = np.tanh(drive * audio.astype(np.float64)) / norm
            # Subtiler Mix: 80% Original + 20% Sättigung
            mix = float(context.get("mix", 0.2))
            return (audio * (1.0 - mix) + saturated * mix).astype(audio.dtype)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("dummy_models.py::verarbeiten Ersatzpfad: %s", e)
            return audio
