"""
AURIK Spectral Repair/Inpainting Module

Repairs digital artifacts including:
- MP3 compression artifacts (spectral holes above 16 kHz)
- Packet loss (spectral dropouts)
- Digital clicks/glitches
- Codec artifacts (AAC, Vorbis bandwidth limitations)

Approach:
- Spectral hole detection (missing frequency bands)
- Harmonic extrapolation (synthesize missing harmonics)
- Spectral smoothing (repair codec artifacts)
- Transient-aware processing (preserve musical details)

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.signal

try:
    from dsp.pghi import pghi_reconstruct_from_stft as _pghi_reconstruct_from_stft

    _PGHI_AVAILABLE = True
except ImportError:
    _PGHI_AVAILABLE = False


@dataclass
class SpectralRepairConfig:
    """Configuration for spectral repair processing."""

    # Spectral hole detection
    hole_threshold_db: float = -60.0
    """Threshold for detecting spectral holes (dB below peak)."""

    min_hole_width_hz: float = 500.0
    """Minimum width of spectral hole to repair (Hz)."""

    # Harmonic extrapolation
    extrapolation_order: int = 8
    """Number of harmonics to analyze for extrapolation."""

    extrapolation_strength: float = 0.7
    """Strength of harmonic extrapolation (0.0-1.0)."""

    decay_db_per_octave: float = -6.0
    """Harmonic decay slope for hole repair (dB/octave). Default -6.0 (neutral).
    Material-adaptive overrides: -9.0 (vocal), -4.0 (bright instrumental),
    -3.0 (vintage brass). Passed via SpectralRepairConfig at call site.
    """

    # Codec artifact repair
    smooth_bandwidth_hz: float = 100.0
    """Smoothing bandwidth for codec artifact repair (Hz)."""

    # Processing
    fft_size: int = 4096
    """FFT size for spectral analysis."""

    hop_size: int = 1024
    """Hop size for STFT processing."""


class SpectralRepair:
    """
    Spectral Repair/Inpainting for Digital Artifacts.

    Repairs:
    - MP3 artifacts (spectral holes above 16 kHz)
    - Packet loss (spectral dropouts)
    - Digital clicks/glitches
    - Codec artifacts (AAC, Vorbis bandwidth limitations)
    """

    def __init__(self, config: SpectralRepairConfig | None = None):
        """
        Initialisiert spectral repair processor.

        Args:
            config: Spectral repair configuration (uses defaults if None)
        """
        self.config = config or SpectralRepairConfig()
        self._metrics: dict[str, Any] = {}

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet audio with spectral repair.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate (Hz)

        Returns:
            Repaired audio
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        # Reset metrics
        self._metrics = {"holes_detected": 0, "holes_repaired": 0, "frequency_ranges": [], "artifacts_smoothed": 0}

        # Handle stereo
        if audio.ndim == 2:
            # Process each channel
            left = self._process_mono(audio[:, 0], sr)
            right = self._process_mono(audio[:, 1], sr)
            return np.column_stack([left, right])
        else:
            return self._process_mono(audio, sr)

    def _process_mono(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """Verarbeitet mono audio."""
        # STFT
        f, _t, Zxx = scipy.signal.stft(
            audio,
            fs=sr,
            nperseg=self.config.fft_size,
            noverlap=self.config.fft_size - self.config.hop_size,
            boundary="even",
        )

        # Detect and repair spectral holes
        Zxx_repaired = self._detect_and_repair_holes(Zxx, f, sr)

        # Smooth codec artifacts
        Zxx_smoothed = self._smooth_codec_artifacts(Zxx_repaired, f, sr)

        # Phase-coherent reconstruction: PGHI (Perraudin 2013) preferred over
        # scipy.signal.istft, which reuses stale phases after spectral modification.
        if _PGHI_AVAILABLE:
            audio_repaired = _pghi_reconstruct_from_stft(
                Zxx_smoothed,
                sr=sr,
                win_size=self.config.fft_size,
                hop=self.config.hop_size,
            )
        else:
            _, audio_repaired = scipy.signal.istft(
                Zxx_smoothed, fs=sr, nperseg=self.config.fft_size, noverlap=self.config.fft_size - self.config.hop_size
            )

        # Ensure same length as input
        if len(audio_repaired) > len(audio):
            audio_repaired = audio_repaired[: len(audio)]
        elif len(audio_repaired) < len(audio):
            audio_repaired = np.pad(audio_repaired, (0, len(audio) - len(audio_repaired)), mode="constant")

        # NaN/Inf-Guard + Clipping
        audio_repaired = np.nan_to_num(audio_repaired, nan=0.0, posinf=0.0, neginf=0.0)
        audio_repaired = np.clip(audio_repaired, -1.0, 1.0)

        return audio_repaired.astype(np.float64)

    def _detect_and_repair_holes(
        self, Zxx: npt.NDArray[np.complex128], f: npt.NDArray[np.float64], sr: int
    ) -> npt.NDArray[np.complex128]:
        """
        Erkennt and repair spectral holes.

        Common scenarios:
        - MP3: Hole above 16 kHz (HF cutoff)
        - AAC 128kbps: Hole above 15.5 kHz
        - Opus/Vorbis: Variable cutoff 12-20 kHz
        """
        # Compute magnitude spectrum (averaged over time)
        mag_spec = np.abs(Zxx).mean(axis=1)
        mag_spec_db = 20 * np.log10(mag_spec + 1e-10)

        # Find peak magnitude (reference level)
        peak_db = mag_spec_db.max()

        # Detect holes: regions below threshold
        hole_mask = mag_spec_db < (peak_db + self.config.hole_threshold_db)

        # Find contiguous hole regions
        hole_regions = self._find_contiguous_regions(hole_mask, f, self.config.min_hole_width_hz)

        if not hole_regions:
            return Zxx

        # Repair each hole
        Zxx_repaired = Zxx.copy()
        for start_idx, end_idx in hole_regions:
            start_freq = f[start_idx]
            end_freq = f[end_idx]

            # Only repair holes in upper frequency range (> 2 kHz)
            # (Low frequency holes are likely intentional)
            if start_freq > 2000:
                Zxx_repaired = self._repair_hole_region(Zxx_repaired, f, start_idx, end_idx, sr)

                self._metrics["holes_repaired"] += 1
                self._metrics["frequency_ranges"].append((float(start_freq), float(end_freq)))

        self._metrics["holes_detected"] = len(hole_regions)

        return Zxx_repaired

    def _find_contiguous_regions(
        self, mask: npt.NDArray[np.bool_], f: npt.NDArray[np.float64], min_width_hz: float
    ) -> list:
        """Findet zusammenhängende Regionen in binärer Maske."""
        regions = []
        in_region = False
        start_idx = 0

        for i, val in enumerate(mask):
            if val and not in_region:
                # Start of region
                start_idx = i
                in_region = True
            elif not val and in_region:
                # End of region
                end_idx = i - 1
                width_hz = f[end_idx] - f[start_idx]

                if width_hz >= min_width_hz:
                    regions.append((start_idx, end_idx))

                in_region = False

        # Handle last region
        if in_region:
            end_idx = len(mask) - 1
            width_hz = f[end_idx] - f[start_idx]
            if width_hz >= min_width_hz:
                regions.append((start_idx, end_idx))

        return regions

    def _repair_hole_region(
        self, Zxx: npt.NDArray[np.complex128], f: npt.NDArray[np.float64], start_idx: int, end_idx: int, sr: int
    ) -> npt.NDArray[np.complex128]:
        """
        Repariert spectral hole using harmonic extrapolation.

        Method:
        1. Analyze harmonics below hole
        2. Extrapolate harmonic pattern into hole
        3. Blend with original signal
        """
        # Find reference region (just below hole)
        ref_width = min(end_idx - start_idx, 50)  # Up to 50 bins
        ref_start = max(0, start_idx - ref_width)
        ref_end = start_idx

        if ref_start >= ref_end:
            return Zxx

        # Extract reference spectrum
        ref_spec = Zxx[ref_start:ref_end, :]
        np.abs(ref_spec).mean(axis=0)  # Average magnitude
        np.angle(ref_spec)

        # Synthesize missing content using harmonic decay model
        # Typical HF decay: -6 dB/octave for harmonics
        hole_length = end_idx - start_idx

        for i in range(hole_length):
            freq_idx = start_idx + i

            # Frequency of current bin
            freq_hz = f[freq_idx]

            # Find corresponding lower frequency (1 octave down)
            ref_freq_hz = freq_hz / 2.0
            ref_freq_idx = np.argmin(np.abs(f - ref_freq_hz))

            if ref_freq_idx < start_idx:
                # Extrapolate with decay
                octaves_up = np.log2(freq_hz / f[ref_freq_idx])
                decay_db = self.config.decay_db_per_octave * octaves_up
                decay_linear = 10 ** (decay_db / 20.0)

                # Apply decay + add noise for naturalness
                Zxx[freq_idx, :] = Zxx[ref_freq_idx, :] * decay_linear * self.config.extrapolation_strength + Zxx[
                    freq_idx, :
                ] * (1.0 - self.config.extrapolation_strength)

        return Zxx

    def _smooth_codec_artifacts(
        self, Zxx: npt.NDArray[np.complex128], f: npt.NDArray[np.float64], sr: int
    ) -> npt.NDArray[np.complex128]:
        """
        Glättet codec artifacts (quantization noise, ringing).

        Method:
        - Apply gentle spectral smoothing in codec artifact bands
        - Preserve transients (only smooth steady-state regions)
        """
        # Compute temporal variation (to detect transients)
        mag = np.abs(Zxx)
        temporal_var = np.std(mag, axis=1)

        # Only smooth regions with low temporal variation (steady-state)
        steady_mask = temporal_var < temporal_var.mean()

        # Smooth spectrum in steady regions
        mag_smoothed = mag.copy()

        # Spectral smoothing kernel (frequency domain)
        smooth_bins = int(self.config.smooth_bandwidth_hz / (sr / self.config.fft_size))
        smooth_bins = max(3, smooth_bins)

        np.ones(smooth_bins) / smooth_bins

        # Vectorised: uniform_filter1d replaces the per-frame np.convolve loop.
        # Equivalent to a box kernel along the frequency axis but ~n_frames× faster.
        from scipy.ndimage import uniform_filter1d as _uf1d

        smoothed_all = _uf1d(mag, size=smooth_bins, axis=0, mode="reflect")
        mag_smoothed[steady_mask, :] = smoothed_all[steady_mask, :]

        # Count artifacts smoothed
        self._metrics["artifacts_smoothed"] = int(np.sum(steady_mask))

        # Reconstruct with smoothed magnitude
        phase = np.angle(Zxx)
        Zxx_smoothed = mag_smoothed * np.exp(1j * phase)

        return Zxx_smoothed  # type: ignore[no-any-return]

    def get_metrics(self) -> dict[str, Any]:
        """
        Gibt zurück: processing metrics.

        Returns:
            Dictionary with metrics:
            - holes_detected: Number of spectral holes detected
            - holes_repaired: Number of spectral holes repaired
            - frequency_ranges: List of (start_hz, end_hz) tuples for repaired holes
            - artifacts_smoothed: Number of frequency bins with artifacts smoothed
        """
        return self._metrics.copy()


def process_spectral_repair(
    audio: npt.NDArray[np.float64], sr: int, strength: float = 0.7, hole_threshold_db: float = -60.0
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """
    Convenience function for spectral repair.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate (Hz)
        strength: Repair strength (0.0-1.0)
        hole_threshold_db: Threshold for detecting spectral holes

    Returns:
        Tuple of (repaired_audio, metrics)
    """
    config = SpectralRepairConfig(hole_threshold_db=hole_threshold_db, extrapolation_strength=strength)

    processor = SpectralRepair(config)
    repaired = processor.process(audio, sr)
    metrics = processor.get_metrics()

    return repaired, metrics


if __name__ == "__main__":
    # Demo: Test spectral repair on synthetic signal with holes
    logger.info("AURIK Spectral Repair Demo")
    logger.info(str("=" * 60))

    # Generate test signal (1 second at 48 kHz)
    sr = 48000
    t = np.linspace(0, 1, sr)

    # Multi-harmonic signal (fundamental + harmonics)
    signal = np.zeros_like(t)
    fundamental = 440  # A4
    for h in range(1, 11):  # 10 harmonics
        signal += 0.5 * np.sin(2 * np.pi * fundamental * h * t) / h

    # Simulate MP3 compression: remove content above 16 kHz
    butter_coeffs = scipy.signal.butter(8, 16000.0 / (sr / 2.0), btype="low", output="ba")
    if not isinstance(butter_coeffs, tuple) or len(butter_coeffs) != 2:
        raise RuntimeError("scipy.signal.butter returned unexpected coefficient format")

    b = np.asarray(butter_coeffs[0], dtype=np.float64)
    a = np.asarray(butter_coeffs[1], dtype=np.float64)
    signal_compressed = scipy.signal.lfilter(b, a, signal)

    logger.info("\nEingabe:")
    logger.info("  - Sample Rate: %s Hz", sr)
    logger.info("  - Duration: 1.0 seconds")
    logger.info("  - Content: Multi-harmonic signal (440 Hz fundamental, 10 harmonics)")
    logger.info("  - Simulated Damage: MP3-style lowpass (cutoff 16 kHz)")

    # Process with spectral repair
    config = SpectralRepairConfig(hole_threshold_db=-60.0, min_hole_width_hz=1000.0, extrapolation_strength=0.7)
    processor = SpectralRepair(config)
    signal_repaired = processor.process(signal_compressed, sr)
    metrics = processor.get_metrics()

    logger.info("\nSpectral Repair Results:")
    logger.info("  - Holes erkannt: %s", metrics["holes_detected"])
    logger.info("  - Holes Repaired: %s", metrics["holes_repaired"])
    if metrics["frequency_ranges"]:
        logger.info("  - Frequency Ranges:")
        for start_hz, end_hz in metrics["frequency_ranges"]:
            logger.info("    * %.0f - %.0f Hz", start_hz, end_hz)
    logger.info("  - Artifacts Smoothed: %s bins", metrics["artifacts_smoothed"])

    logger.info("\n✅ Spectral Repair Demo vollstaendig")
