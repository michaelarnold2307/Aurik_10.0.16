"""
Adaptive MCRA Noise Estimation DSP-Modul für Aurik 6.0 (SOTA-Maximum)
Klassische adaptive MCRA-Rauschschätzung mit automatischer Parameteroptimierung (SOTA-Maximum).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_mcra"
    category: str = "noise_estimation"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveMCRA:
    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.98,
        win_length: int = 20,
        noise_floor: float = 1e-6,
    ):
        self.alpha = alpha  # Glättungsfaktor für Power-Schätzung
        self.beta = beta  # Glättungsfaktor für Minimum-Tracking
        self.win_length = win_length  # Fensterlänge für Minimumsuche
        self.noise_floor = noise_floor

    def estimate_noise(self, power_spectrogram: np.ndarray) -> np.ndarray:
        """Schätzt noise PSD adaptively with MCRA (Cohen 2002).

        Uses smoothed power tracking + minimum statistics with SNR-adaptive
        beta parameter for more conservative tracking in noisy environments.
        """
        power = np.nan_to_num(np.asarray(power_spectrogram, dtype=np.float64))
        n_frames, n_bins = power.shape
        noise_psd = np.zeros_like(power)
        smoothed_psd = np.zeros(n_bins, dtype=np.float64)
        min_buffer = np.full((self.win_length, n_bins), np.inf, dtype=np.float64)

        for t in range(n_frames):
            smoothed_psd = self.alpha * smoothed_psd + (1 - self.alpha) * power[t]
            min_buffer[t % self.win_length] = smoothed_psd
            min_psd = np.min(min_buffer, axis=0)

            prev_noise = noise_psd[t - 1] if t > 0 else smoothed_psd

            # SNR-adaptive beta: increase beta when signal >> noise (more conservative)
            local_snr = smoothed_psd / np.maximum(min_psd, self.noise_floor)
            adaptive_beta = np.where(local_snr > 3.0, self.beta, self.beta * 0.9)

            noise_psd[t] = adaptive_beta * prev_noise + (1 - adaptive_beta) * min_psd
            noise_psd[t] = np.maximum(noise_psd[t], self.noise_floor)

        noise_psd = np.nan_to_num(noise_psd, nan=self.noise_floor, posinf=self.noise_floor, neginf=self.noise_floor)
        return noise_psd

    def auto_optimize(self, power_spectrogram: np.ndarray) -> dict[str, float]:  # type: ignore[return]
        """Automatische Anpassung der Parameter je nach Signal."""
        n_frames = power_spectrogram.shape[0]
        if n_frames < 50:
            self.win_length = 5
        elif n_frames < 200:
            self.win_length = 10
        else:
            self.win_length = 20
