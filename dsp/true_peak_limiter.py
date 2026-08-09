"""
true_peak_limiter.py - ITU-R BS.1770-4 Compliant True Peak Limiter

Prevents inter-sample peaks that occur during digital-to-analog conversion.
Critical for streaming platforms (Spotify, YouTube) and broadcast (EBU R128).

Features:
- 4x Oversampling for inter-sample peak detection
- Lookahead for transparent limiting
- Soft/Hard knee modes
- True Peak measurement in dBTP

Author: AURIK Development Team
Version: 1.0.0
Date: 10. Februar 2026
"""

import numpy as np
from scipy import signal


class TruePeakLimiter:
    """
    ITU-R BS.1770-4 compliant True Peak Limiter.

    Prevents inter-sample peaks by oversampling, detecting peaks,
    and applying transparent gain reduction with lookahead.

    Parameters
    ----------
    ceiling_dbtp : float
        Maximum allowed True Peak level in dBTP (default: -1.0)
        EBU R128 recommends -1.0 dBTP for broadcast
    attack_ms : float
        Attack time in milliseconds (default: 0.1)
    release_ms : float
        Release time in milliseconds (default: 100)
    lookahead_ms : float
        Lookahead time in milliseconds (default: 5.0)
    knee_db : float
        Soft knee width in dB (default: 0.0 = hard knee)
    oversample_factor : int
        Oversampling factor for peak detection (default: 4)
    safety_margin_db : float
        Additional headroom margin subtracted from ceiling_dbtp before gain
        reduction (default: 0.5 dB). Guards against oversampling filter
        transients and inter-sample peaks that survive 4× upsampling.
        Set to 0.0 for a hard ceiling with no extra guard-band.
    """

    def __init__(
        self,
        ceiling_dbtp: float = -1.0,
        attack_ms: float = 0.1,
        release_ms: float = 100.0,
        lookahead_ms: float = 5.0,
        knee_db: float = 0.0,
        oversample_factor: int = 4,
        safety_margin_db: float = 0.5,
    ):
        self.ceiling_dbtp = ceiling_dbtp
        self.safety_margin_db = safety_margin_db
        # Effective ceiling includes an explicit guard-band (default 0.5 dB).
        # Accounts for oversampling filter transients and residual inter-sample peaks.
        self.ceiling_linear = 10 ** ((ceiling_dbtp - safety_margin_db) / 20.0)
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.lookahead_ms = lookahead_ms
        self.knee_db = knee_db
        self.oversample_factor = oversample_factor

    def _upsample(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        Upsample audio by oversample_factor using polyphase filtering.

        Returns
        -------
        upsampled : np.ndarray
            Upsampled audio
        sr_up : int
            Upsampled sample rate
        """
        if audio.ndim == 1:
            # Mono
            upsampled = signal.resample_poly(audio, up=self.oversample_factor, down=1, window=("kaiser", 5.0))
        else:
            # Stereo - process each channel
            upsampled = np.zeros((audio.shape[0], audio.shape[1] * self.oversample_factor))
            for ch in range(audio.shape[0]):
                upsampled[ch, :] = signal.resample_poly(
                    audio[ch, :], up=self.oversample_factor, down=1, window=("kaiser", 5.0)
                )

        sr_up = sr * self.oversample_factor
        return upsampled, sr_up

    def _downsample(self, audio_upsampled: np.ndarray, sr: int) -> np.ndarray:
        """
        Downsample audio back to original sample rate.
        """
        if audio_upsampled.ndim == 1:
            # Mono
            downsampled = signal.resample_poly(
                audio_upsampled, up=1, down=self.oversample_factor, window=("kaiser", 5.0)
            )
        else:
            # Stereo
            downsampled = np.zeros((audio_upsampled.shape[0], audio_upsampled.shape[1] // self.oversample_factor))
            for ch in range(audio_upsampled.shape[0]):
                downsampled[ch, :] = signal.resample_poly(
                    audio_upsampled[ch, :], up=1, down=self.oversample_factor, window=("kaiser", 5.0)
                )

        return downsampled  # type: ignore[no-any-return]

    def _compute_gain_reduction(self, audio_upsampled: np.ndarray, sr_up: int) -> np.ndarray:
        """
        Berechnet gain reduction envelope with lookahead.

        Returns
        -------
        gain_reduction : np.ndarray
            Gain reduction factors (0.0-1.0) for each sample
        """
        # Compute peak envelope (absolute values)
        if audio_upsampled.ndim == 1:
            peak_env = np.abs(audio_upsampled)
        else:
            # Stereo: use maximum of both channels
            peak_env = np.max(np.abs(audio_upsampled), axis=0)

        # Lookahead buffer: detect future peaks
        lookahead_samples = int(self.lookahead_ms * sr_up / 1000.0)
        if lookahead_samples > 0:
            # Apply maximum filter (running max over lookahead window)
            # This looks ahead and finds the maximum peak in the upcoming window
            from scipy.ndimage import maximum_filter1d

            peak_env = maximum_filter1d(peak_env, size=lookahead_samples, mode="constant")

        # Compute required gain reduction
        gain_reduction = np.ones_like(peak_env)

        # Where peaks exceed ceiling, compute reduction
        mask = peak_env > self.ceiling_linear
        if np.any(mask):
            gain_reduction[mask] = self.ceiling_linear / peak_env[mask]

        # Apply soft knee if configured
        if self.knee_db > 0:
            knee_linear = 10 ** (self.knee_db / 20.0)
            knee_start = self.ceiling_linear / knee_linear

            # Soft transition zone
            knee_mask = (peak_env > knee_start) & (peak_env <= self.ceiling_linear)
            if np.any(knee_mask):
                # Smooth interpolation in knee region
                knee_factor = (peak_env[knee_mask] - knee_start) / (self.ceiling_linear - knee_start)
                knee_reduction = 1.0 - (1.0 - self.ceiling_linear / peak_env[knee_mask]) * knee_factor
                gain_reduction[knee_mask] = knee_reduction

        # Apply attack/release smoothing
        attack_samples = int(self.attack_ms * sr_up / 1000.0)
        release_samples = int(self.release_ms * sr_up / 1000.0)

        # Simple peak-hold with exponential decay
        smoothed_gain = np.ones_like(gain_reduction)
        current_gain = 1.0

        for i in range(len(gain_reduction)):
            target_gain = gain_reduction[i]

            if target_gain < current_gain:
                # Attack (fast reduction)
                if attack_samples > 0:
                    alpha = 1.0 - np.exp(-1.0 / attack_samples)
                    current_gain = current_gain * (1 - alpha) + target_gain * alpha
                else:
                    current_gain = target_gain
            else:
                # Release (slow return)
                if release_samples > 0:
                    alpha = 1.0 - np.exp(-1.0 / release_samples)
                    current_gain = current_gain * (1 - alpha) + target_gain * alpha
                else:
                    current_gain = target_gain

            smoothed_gain[i] = current_gain

        return smoothed_gain  # type: ignore[no-any-return]

    def measure_true_peak(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst True Peak level in dBTP (ITU-R BS.1770-4).

        Parameters
        ----------
        audio : np.ndarray
            Audio signal (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        true_peak_dbtp : float
            True Peak level in dBTP
        """
        # Upsample for inter-sample peak detection
        audio_up, _sr_up = self._upsample(audio, sr)

        # Find maximum absolute value
        peak_linear: float = float(np.max(np.abs(audio_up)))

        # Convert to dBTP
        true_peak_dbtp = 20 * np.log10(peak_linear) if peak_linear > 0 else -np.inf

        return true_peak_dbtp

    def process(self, audio: np.ndarray, sr: int, return_metrics: bool = False) -> tuple[np.ndarray, dict | None]:
        """
        Wendet an: true peak limiting to audio.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono: (samples,) or stereo: (channels, samples))
        sr : int
            Sample rate in Hz
        return_metrics : bool
            If True, return metrics dictionary

        Returns
        -------
        audio_limited : np.ndarray
            Limited audio with same shape as input
        metrics : dict, optional
            Processing metrics if return_metrics=True:
            - 'true_peak_input_dbtp': Input true peak level
            - 'true_peak_output_dbtp': Output true peak level
            - 'gain_reduction_max_db': Maximum gain reduction applied
            - 'samples_limited': Number of samples that were limited
        """
        # Input validation
        if audio.size == 0:
            if return_metrics:
                return audio, {
                    "true_peak_input_dbtp": -np.inf,
                    "true_peak_output_dbtp": -np.inf,
                    "gain_reduction_max_db": 0.0,
                    "samples_limited": 0,
                }
            return audio, None

        # Measure input true peak
        tp_input = self.measure_true_peak(audio, sr)

        # If already below ceiling, no processing needed
        if tp_input <= self.ceiling_dbtp:
            if return_metrics:
                return audio, {
                    "true_peak_input_dbtp": tp_input,
                    "true_peak_output_dbtp": tp_input,
                    "gain_reduction_max_db": 0.0,
                    "samples_limited": 0,
                }
            return audio, None

        # Store original shape
        original_shape = audio.shape
        is_mono = audio.ndim == 1

        # Convert to (channels, samples) format if needed
        if is_mono:
            audio = audio.reshape(1, -1)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            # (samples, channels) → (channels, samples)
            audio = audio.T

        # Upsample
        audio_up, sr_up = self._upsample(audio, sr)

        # Compute gain reduction envelope
        gain_envelope = self._compute_gain_reduction(audio_up, sr_up)

        # Apply gain reduction
        if audio_up.ndim == 1:
            audio_limited_up = audio_up * gain_envelope
        else:
            # Apply same gain to all channels
            audio_limited_up = audio_up * gain_envelope[np.newaxis, :]

        # Downsample back
        audio_limited = self._downsample(audio_limited_up, sr)

        # Restore original shape
        if is_mono:
            audio_limited = audio_limited.flatten()
        elif original_shape[0] > original_shape[1]:
            # Back to (samples, channels)
            audio_limited = audio_limited.T

        # Measure output metrics
        if return_metrics:
            tp_output = self.measure_true_peak(audio_limited, sr)
            gr_max_db = -20 * np.log10(np.min(gain_envelope))
            samples_limited = np.sum(gain_envelope < 0.999)

            metrics = {
                "true_peak_input_dbtp": tp_input,
                "true_peak_output_dbtp": tp_output,
                "gain_reduction_max_db": gr_max_db,
                "samples_limited": int(samples_limited),
            }
            audio_limited = np.clip(np.nan_to_num(audio_limited, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
                audio.dtype
            )
            return audio_limited, metrics

        audio_limited = np.clip(np.nan_to_num(audio_limited, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
            audio.dtype
        )
        return audio_limited, None
