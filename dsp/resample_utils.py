"""
Professional Audio Resampling Utilities for AURIK v8
=====================================================

AURIK arbeitet intern konsistent mit 48 kHz:
- Standard für professionelle Musik-Restauration
- Nyquist-Frequenz 24 kHz (abdeckt alle hörbaren Frequenzen)
- Kompatibel mit Broadcast-Standards (48 kHz / 96 kHz)
- Optimale Frequenzauflösung für DSP-Filter

Resampling erfolgt:
1. Input → 48 kHz (bei Programmstart)
2. Vor ML-Modellen mit spezifischen Requirements (z.B. Wav2Vec2 @ 16 kHz)
3. Nach ML-Modellen zurück auf 48 kHz
4. Output optional auf Original-SR oder 48 kHz (Default: 48 kHz)

Author: AURIK Development Team
Version: 2.0.0
Date: 9. Februar 2026
"""

import logging

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# Professional audio standard
AURIK_STANDARD_SR = 48000


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Intern: resampling using scipy.signal.resample_poly (high-quality, polyphase filtering).

    Args:
        audio: Audio signal (mono)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio

    # Calculate rational resampling factors
    from math import gcd

    common_divisor = gcd(orig_sr, target_sr)
    up = target_sr // common_divisor
    down = orig_sr // common_divisor

    # Use polyphase resampling (high-quality anti-aliasing)
    resampled = scipy_signal.resample_poly(audio, up, down, axis=0)

    return resampled  # type: ignore[no-any-return]


class AudioResampler:
    """
    Professional audio resampler with anti-aliasing and quality control.

    Features:
    - Anti-aliasing filters
    - Quality presets (draft/good/best)
    - Stereo/Mono support
    - Consistent 48 kHz standard

    Example:
        >>> resampler = AudioResampler(quality='good')
        >>> audio_48k, sr_48k = resampler.to_standard(audio, sr=44100)
        >>> # Process at 48 kHz...
        >>> audio_out = resampler.from_standard(audio_48k, target_sr=44100)
    """

    def __init__(self, quality: str = "good"):
        """
        Initialisiert resampler.

        Args:
            quality: Quality preset (for compatibility, always uses high-quality scipy)
        """
        self.quality = quality
        self.standard_sr = AURIK_STANDARD_SR

        logger.debug("AudioResampler initialisiert: standard=%s Hz", self.standard_sr)

    def to_standard(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """
        Resample audio to AURIK standard (48 kHz).

        Args:
            audio: Input audio (mono or stereo in format (N_samples, N_channels) or (N_channels, N_samples))
            sr: Current sample rate

        Returns:
            Tuple of (resampled_audio, 48000)
        """
        if sr == self.standard_sr:
            return audio, sr

        logger.debug("Resampling %s Hz → %s Hz", sr, self.standard_sr)

        # Handle stereo/mono
        if audio.ndim == 2:
            # Auto-detect array orientation: (samples, channels) vs (channels, samples)
            # If shape[0] < shape[1], it's likely (channels, samples) format
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Transpose to (samples, channels) format
                logger.debug("Auto-transposing audio from %s (channels, samples) to (samples, channels)", audio.shape)
                audio = audio.T

            # Stereo: resample each channel (format: (N_samples, N_channels))
            int(audio.shape[0] * self.standard_sr / sr)
            resampled_channels = []
            for ch in range(audio.shape[1]):
                resampled_ch = _resample_audio(audio[:, ch], sr, self.standard_sr)
                resampled_channels.append(resampled_ch)

            # Stelle sicher, dass alle Kanäle exakt gleich lang sind (minimale Länge)
            min_length = min(len(ch) for ch in resampled_channels)
            resampled = np.zeros((min_length, audio.shape[1]), dtype=audio.dtype)
            for ch_idx, ch_data in enumerate(resampled_channels):
                resampled[:, ch_idx] = ch_data[:min_length]
        else:
            # Mono
            resampled = _resample_audio(audio, sr, self.standard_sr)

        return resampled, self.standard_sr

    def from_standard(self, audio: np.ndarray, target_sr: int) -> tuple[np.ndarray, int]:
        """
        Resample from AURIK standard (48 kHz) to target sample rate.

        Args:
            audio: Audio at 48 kHz (format: (N_samples, N_channels) or (N_channels, N_samples))
            target_sr: Desired output sample rate

        Returns:
            Tuple of (resampled_audio, target_sr)
        """
        if target_sr == self.standard_sr:
            return audio, target_sr

        logger.debug("Resampling %s Hz → %s Hz", self.standard_sr, target_sr)

        # Handle stereo/mono
        if audio.ndim == 2:
            # Auto-detect array orientation: (samples, channels) vs (channels, samples)
            # If shape[0] < shape[1], it's likely (channels, samples) format
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Transpose to (samples, channels) format
                logger.debug("Auto-transposing audio from %s (channels, samples) to (samples, channels)", audio.shape)
                audio = audio.T

            # Stereo: resample each channel (format: (N_samples, N_channels))
            new_length = int(audio.shape[0] * target_sr / self.standard_sr)
            resampled = np.zeros((new_length, audio.shape[1]), dtype=audio.dtype)
            for ch in range(audio.shape[1]):
                resampled[:, ch] = _resample_audio(audio[:, ch], self.standard_sr, target_sr)
        else:
            # Mono
            resampled = _resample_audio(audio, self.standard_sr, target_sr)

        return resampled, target_sr

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
        """
        Direct resampling from any sample rate to any other.

        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Tuple of (resampled_audio, target_sr)
        """
        if orig_sr == target_sr:
            return audio, target_sr

        logger.debug("Direct resampling %s Hz → %s Hz", orig_sr, target_sr)

        # Handle stereo/mono
        if audio.ndim == 2:
            # Stereo: resample each channel
            new_length = int(len(audio) * target_sr / orig_sr)
            resampled = np.zeros((new_length, audio.shape[1]), dtype=audio.dtype)
            for ch in range(audio.shape[1]):
                resampled[:, ch] = _resample_audio(audio[:, ch], orig_sr, target_sr)
        else:
            # Mono
            resampled = _resample_audio(audio, orig_sr, target_sr)

        return resampled, target_sr


def ensure_sr(audio: np.ndarray, sr: int, target_sr: int = 48000) -> tuple[np.ndarray, int]:
    """
    Legacy function - ensures audio has target sample rate.

    Usage: For simple cases. Prefer AudioResampler for professional use.
    """
    if sr == target_sr:
        return audio, sr

    if audio.ndim == 2:
        new_length = int(len(audio) * target_sr / sr)
        audio_rs = np.zeros((new_length, audio.shape[1]), dtype=audio.dtype)
        for ch in range(audio.shape[1]):
            audio_rs[:, ch] = _resample_audio(audio[:, ch], sr, target_sr)
    else:
        audio_rs = _resample_audio(audio, sr, target_sr)

    return audio_rs, target_sr


def process_with_resampling(
    process_func,
    audio: np.ndarray,
    sr: int,
    required_sr: int | None = None,
    return_to_standard: bool = True,
    quality: str = "good",
    **kwargs,
) -> tuple[np.ndarray, int]:
    """
    Verarbeitet audio with automatic resampling for components that need specific sample rates.

    Example: Wav2Vec2 needs 16 kHz, but AURIK works at 48 kHz internally.

    Args:
        process_func: Processing function that takes (audio, sr, **kwargs)
        audio: Input audio (at any sample rate)
        sr: Current sample rate
        required_sr: Sample rate required by process_func (None = use current)
        return_to_standard: Return to 48 kHz after processing
        quality: Resampling quality
        **kwargs: Additional arguments for process_func

    Returns:
        Tuple of (processed_audio, sample_rate)

    Example:
        >>> # PhonemeDetector needs 16 kHz, but we work at 48 kHz
        >>> phonemes, sr_out = process_with_resampling(
        ...     phoneme_detector.detect,
        ...     audio_48k, sr=48000,
        ...     required_sr=16000,
        ...     return_to_standard=True
        ... )
        >>> # phonemes are computed at 16 kHz, but sr_out = 48000
    """
    resampler = AudioResampler(quality=quality)

    # Step 1: Resample to required SR if needed
    if required_sr and sr != required_sr:
        logger.debug("Resampling for processing: %s Hz → %s Hz", sr, required_sr)
        audio_proc, sr_proc = resampler.resample(audio, sr, required_sr)
    else:
        audio_proc, sr_proc = audio, sr

    # Step 2: Process
    result = process_func(audio_proc, sr_proc, **kwargs)

    # Step 3: Return to standard if requested
    if return_to_standard and sr_proc != AURIK_STANDARD_SR:
        logger.debug("Returning to standard: %s Hz → %s Hz", sr_proc, AURIK_STANDARD_SR)
        result, sr_out = resampler.resample(result, sr_proc, AURIK_STANDARD_SR)
    else:
        sr_out = sr_proc

    return result, sr_out
