"""
dialog_intelligibility.py - Dialog/Speech Intelligibility Enhancement for AURIK 6.0

Improves speech clarity and intelligibility for podcasts, interviews, voice-over.

Key techniques:
- Formant-aware EQ (boost 2-6 kHz for consonants)
- Sibilance control (de-ess 6-8 kHz)
- Nasality reduction (notch 800-1200 Hz)
- Presence boost (3-5 kHz)
- Mid-range clarity (200-800 Hz cleanup)

References:
- iZotope Dialogue Match
- Cedar Cambridge DNS Two
- Waves Clarity VX Pro
"""

import logging
from typing import Any

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt

logger = logging.getLogger(__name__)


class DialogIntelligibilityEnhancer:
    """
    Enhance speech intelligibility for dialog, podcasts, interviews.

    Optimizes frequency response for maximum clarity and presence.
    """

    def __init__(
        self,
        presence_boost_db: float = 3.0,
        consonant_boost_db: float = 4.0,
        nasality_reduction_db: float = -3.0,
        sibilance_control_db: float = -2.0,
        mud_reduction_db: float = -2.0,
    ):
        """
        Initialisiert Dialog Intelligibility Enhancer.

        Args:
            presence_boost_db: Boost at 3-5 kHz (presence range)
            consonant_boost_db: Boost at 4-6 kHz (consonants)
            nasality_reduction_db: Reduction at 800-1200 Hz (nasality)
            sibilance_control_db: Reduction at 6-8 kHz (sibilance)
            mud_reduction_db: Reduction at 200-400 Hz (mud/boxiness)
        """
        self.presence_boost_db = presence_boost_db
        self.consonant_boost_db = consonant_boost_db
        self.nasality_reduction_db = nasality_reduction_db
        self.sibilance_control_db = sibilance_control_db
        self.mud_reduction_db = mud_reduction_db

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Wendet Dialog-Verständlichkeitsverbesserung an.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate

        Returns:
            Tuple of (enhanced audio, metrics dict)
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        is_stereo = audio.ndim == 2

        if is_stereo:
            left = self._enhance_channel(audio[0], sr)
            right = self._enhance_channel(audio[1], sr)
            result = np.stack([left, right], axis=0)
        else:
            result = self._enhance_channel(audio, sr)

        # Calculate intelligibility improvement
        clarity_before = self._measure_clarity(audio, sr)
        clarity_after = self._measure_clarity(result, sr)
        improvement_db = 20 * np.log10((clarity_after + 1e-10) / (clarity_before + 1e-10))
        improvement_db = np.nan_to_num(improvement_db, nan=0.0, posinf=0.0, neginf=0.0)

        # Final guards
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)

        metrics = {
            "clarity_before": float(clarity_before),
            "clarity_after": float(clarity_after),
            "improvement_db": float(improvement_db),
            "presence_boost_db": self.presence_boost_db,
            "consonant_boost_db": self.consonant_boost_db,
            "nasality_reduction_db": self.nasality_reduction_db,
            "sibilance_control_db": self.sibilance_control_db,
        }

        return result, metrics

    def _enhance_channel(self, channel: np.ndarray, sr: int) -> np.ndarray:
        """
        Enhance single audio channel for dialog intelligibility.

        5-stage processing:
        1. Mud reduction (200-400 Hz)
        2. Nasality reduction (800-1200 Hz)
        3. Presence boost (3-5 kHz)
        4. Consonant boost (4-6 kHz)
        5. Sibilance control (6-8 kHz)
        """
        result = channel.copy()

        # Stage 1: Mud reduction (200-400 Hz)
        if abs(self.mud_reduction_db) > 0.1:
            result = self._apply_bandpass_gain(result, sr, freq_range=(200, 400), gain_db=self.mud_reduction_db)

        # Stage 2: Nasality reduction (800-1200 Hz notch)
        if abs(self.nasality_reduction_db) > 0.1:
            result = self._apply_notch_filter(
                result,
                sr,
                center_freq=1000.0,
                q=2.0,
                gain_db=self.nasality_reduction_db,  # Nasality center
            )

        # Stage 3: Presence boost (3-5 kHz)
        if self.presence_boost_db > 0.1:
            result = self._apply_bandpass_gain(result, sr, freq_range=(3000, 5000), gain_db=self.presence_boost_db)

        # Stage 4: Consonant boost (4-6 kHz)
        if self.consonant_boost_db > 0.1:
            result = self._apply_bandpass_gain(result, sr, freq_range=(4000, 6000), gain_db=self.consonant_boost_db)

        # Stage 5: Sibilance control (6-8 kHz)
        if abs(self.sibilance_control_db) > 0.1:
            result = self._apply_bandpass_gain(result, sr, freq_range=(6000, 8000), gain_db=self.sibilance_control_db)

        # Final NaN/Inf-Guard and clipping
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0)

        return result

    def _apply_bandpass_gain(
        self, signal: np.ndarray, sr: int, freq_range: tuple[float, float], gain_db: float
    ) -> np.ndarray:
        """
        Wendet an: gain to a specific frequency band.

        Process:
        1. Extract frequency band with bandpass filter
        2. Apply gain
        3. Mix back with original signal
        """
        nyquist = sr / 2
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist

        # Ensure valid range
        low = max(low, 0.001)
        high = min(high, 0.999)

        # Bandpass filter (4th order Butterworth)
        sos = butter(4, [low, high], btype="band", output="sos")
        band_signal = sosfilt(sos, signal)

        # Apply gain
        gain_linear = 10 ** (gain_db / 20.0)
        band_signal_gained = band_signal * (gain_linear - 1.0)

        # Mix back
        result = signal + band_signal_gained

        return result  # type: ignore[no-any-return]

    def _apply_notch_filter(
        self, signal: np.ndarray, sr: int, center_freq: float, q: float, gain_db: float
    ) -> np.ndarray:
        """
        Wendet an: notch filter (parametric EQ cut).

        Args:
            signal: Input signal
            sr: Sample rate
            center_freq: Center frequency (Hz)
            q: Q factor (bandwidth)
            gain_db: Gain reduction (negative dB)
        """
        # Design notch filter
        b, _a = iirnotch(center_freq, q, sr)

        # Apply filter
        filtered = np.zeros_like(signal)
        for _ in range(int(abs(gain_db) / 6)):  # Cascade for stronger effect
            filtered = np.convolve(signal if _ == 0 else filtered, b, mode="same")

        # Mix based on gain_db
        mix_factor = abs(gain_db) / 6.0  # Normalize
        mix_factor = np.clip(mix_factor, 0.0, 1.0)
        result = signal * (1 - mix_factor) + filtered * mix_factor

        return result  # type: ignore[no-any-return]

    def _measure_clarity(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst speech clarity (energy in 2-6 kHz range).

        Higher energy in this range = better intelligibility.
        """
        audio_mono = np.mean(audio, axis=0) if audio.ndim == 2 else audio

        # Extract clarity range (2-6 kHz)
        nyquist = sr / 2
        low = 2000.0 / nyquist
        high = 6000.0 / nyquist

        sos = butter(4, [low, high], btype="band", output="sos")
        clarity_band = sosfilt(sos, audio_mono)

        # RMS energy
        clarity = np.sqrt(np.mean(clarity_band**2))
        return clarity  # type: ignore[no-any-return]


def create_dialog_intelligibility_enhancer(
    presence_boost_db: float = 3.0,
    consonant_boost_db: float = 4.0,
    nasality_reduction_db: float = -3.0,
    sibilance_control_db: float = -2.0,
    mud_reduction_db: float = -2.0,
) -> DialogIntelligibilityEnhancer:
    """Factory-Funktion zum Erstellen einer DialogIntelligibilityEnhancer-Instanz."""
    return DialogIntelligibilityEnhancer(
        presence_boost_db=presence_boost_db,
        consonant_boost_db=consonant_boost_db,
        nasality_reduction_db=nasality_reduction_db,
        sibilance_control_db=sibilance_control_db,
        mud_reduction_db=mud_reduction_db,
    )


# Example usage
if __name__ == "__main__":
    import soundfile as sf

    from backend.file_import import load_audio_file

    # Load test audio
    _res = load_audio_file("dialog_test.wav")
    audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]

    # Create enhancer
    enhancer = create_dialog_intelligibility_enhancer(
        presence_boost_db=3.0,
        consonant_boost_db=4.0,
        nasality_reduction_db=-3.0,
        sibilance_control_db=-2.0,
        mud_reduction_db=-2.0,
    )

    # Process
    enhanced, metrics = enhancer.process(audio, sr)

    logger.info("Dialog Intelligibility Enhancement angewendet:")
    logger.info("  Clarity Improvement: %.1f dB", metrics["improvement_db"])
    logger.info("  Presence Boost: %.1f dB", metrics["presence_boost_db"])
    logger.info("  Consonant Boost: %.1f dB", metrics["consonant_boost_db"])

    # Save result
    sf.write("enhanced_dialog.wav", enhanced, sr)
