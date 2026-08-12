#!/usr/bin/env python3
"""
Bark Scale Processor - Psychoacoustic Foundation
================================================

24 Critical Bands nach Zwicker & Fastl (1990).
Extrahiert aus Ultimate De-Esser für universelle Verwendung.

Die Bark-Skala repräsentiert die kritischen Bänder des menschlichen Gehörs.
Jedes Band hat ~1 Bark Breite und enthält ~1300 Haarzellen in der Cochlea.

Anwendungen:
- Perceptual Audio Coding
- Masking Analysis
- Frequency Weighting
- Spectral Shaping
- Quality Metrics

Autor: Aurik v8.0 - Psychoacoustic Core
Lizenz: Apache 2.0
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BARK SCALE DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 24 Bark-Scale Critical Bands (Zwicker & Fastl, 1990)
# Format: (bark_index, lower_freq_hz, upper_freq_hz)
BARK_BANDS_24 = [
    (0, 20, 100),  # Band 0: Sub-bass
    (1, 100, 200),  # Band 1: Bass
    (2, 200, 300),  # Band 2: Bass
    (3, 300, 400),  # Band 3: Low-mid
    (4, 400, 510),  # Band 4: Low-mid
    (5, 510, 630),  # Band 5: Mid
    (6, 630, 770),  # Band 6: Mid
    (7, 770, 920),  # Band 7: Mid
    (8, 920, 1080),  # Band 8: Mid
    (9, 1080, 1270),  # Band 9: Mid
    (10, 1270, 1480),  # Band 10: Upper-mid
    (11, 1480, 1720),  # Band 11: Upper-mid
    (12, 1720, 2000),  # Band 12: Upper-mid
    (13, 2000, 2320),  # Band 13: Presence
    (14, 2320, 2700),  # Band 14: Presence
    (15, 2700, 3150),  # Band 15: Presence
    (16, 3150, 3700),  # Band 16: Brilliance
    (17, 3700, 4400),  # Band 17: Brilliance
    (18, 4400, 5300),  # Band 18: Brilliance
    (19, 5300, 6400),  # Band 19: Air
    (20, 6400, 7700),  # Band 20: Air
    (21, 7700, 9500),  # Band 21: Air
    (22, 9500, 12000),  # Band 22: Ultra
    (23, 12000, 15500),  # Band 23: Ultra
]


@dataclass
class BarkBand:
    """
    Represents a single Bark-Scale Critical Band.

    Attributes:
        index: Bark band index (0-23)
        center_hz: Center frequency in Hz
        lower_hz: Lower boundary frequency in Hz
        upper_hz: Upper boundary frequency in Hz
        bandwidth_hz: Bandwidth in Hz
    """

    index: int
    center_hz: float
    lower_hz: float
    upper_hz: float
    bandwidth_hz: float

    @property
    def center_bark(self) -> float:
        """Gibt the Bark value of the center frequency zurück."""
        return hz_to_bark(self.center_hz)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BarkBand(index={self.index}, "
            f"center={self.center_hz:.0f}Hz, "
            f"range=[{self.lower_hz:.0f}, {self.upper_hz:.0f}]Hz, "
            f"bandwidth={self.bandwidth_hz:.0f}Hz)"
        )


@dataclass
class BarkSpectrum:
    """
    Represents the energy distribution across Bark bands.

    Attributes:
        energies: Energy per bark band (normalized)
        energies_db: Energy per bark band in dB
        bands: List of BarkBand objects
        sample_rate: Sample rate of original audio
        total_energy: Total energy across all bands
    """

    energies: np.ndarray  # Shape: (24,)
    energies_db: np.ndarray  # Shape: (24,)
    bands: list[BarkBand]
    sample_rate: int
    total_energy: float

    def get_energy_in_range(self, lower_hz: float, upper_hz: float) -> float:
        """
        Gibt zurück: total normalized energy in frequency range.

        Args:
            lower_hz: Lower frequency bound
            upper_hz: Upper frequency bound

        Returns:
            Summed normalized energy
        """
        energy = 0.0
        for band, band_energy in zip(self.bands, self.energies):
            # Check if band overlaps with range
            if band.upper_hz >= lower_hz and band.lower_hz <= upper_hz:
                energy += band_energy
        return energy

    def get_peak_band(self) -> tuple[BarkBand, float]:
        """
        Gibt zurück: the bark band with maximum energy.

        Returns:
            (peak_band, peak_energy)
        """
        peak_idx = np.argmax(self.energies)
        return self.bands[peak_idx], self.energies[peak_idx]

    def get_spectral_centroid_bark(self) -> float:
        """
        Calculate spectral centroid in Bark scale.

        Returns:
            Centroid in Bark units
        """
        bark_values = np.array([band.center_bark for band in self.bands])
        return float(np.sum(bark_values * self.energies) / (np.sum(self.energies) + 1e-10))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BARK SCALE CONVERSION FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def hz_to_bark(freq_hz: float) -> float:
    """
    Konvertiert frequency in Hz to Bark scale.

    Uses the Zwicker & Terhardt (1980) formula.

    Args:
        freq_hz: Frequency in Hz

    Returns:
        Bark value (0-24)
    """
    # Zwicker & Terhardt (1980) formula
    bark = 13 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500) ** 2)
    return bark  # type: ignore[no-any-return]


def bark_to_hz(bark: float) -> float:
    """
    Konvertiert Bark scale to frequency in Hz.

    Uses the Schroeder et al. (1979) approximation (inverse of Zwicker).

    Args:
        bark: Bark value (0-24)

    Returns:
        Frequency in Hz
    """
    # Schroeder et al. (1979) approximation
    freq_hz = 1960 * (bark + 0.53) / (26.28 - bark)
    return freq_hz


def get_bark_bands(num_bands: int = 24) -> list[BarkBand]:
    """
    Gibt zurück: list of BarkBand objects.

    Args:
        num_bands: Number of bands (default: 24)

    Returns:
        List of BarkBand objects
    """
    if num_bands != 24:
        raise ValueError("Only 24-band Bark scale is currently supported")

    bands = []
    for bark_idx, lower, upper in BARK_BANDS_24:
        center = (lower + upper) / 2
        bandwidth = upper - lower

        band = BarkBand(index=bark_idx, center_hz=center, lower_hz=lower, upper_hz=upper, bandwidth_hz=bandwidth)
        bands.append(band)

    return bands


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BARK SCALE PROCESSOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BarkScaleProcessor:
    """
    Verarbeitet audio on a perceptual Bark scale.

    Features:
    - 24 Critical Bands analysis
    - Energy distribution computation
    - Band-wise filtering
    - Perceptual spectrum extraction

    Example:
        >>> processor = BarkScaleProcessor()
        >>> spectrum = processor.analyze(audio, sr=48000)
        logger.debug("Peak band: %.0f Hz", spectrum.get_peak_band()[0].center_hz)
    """

    def __init__(self, num_bands: int = 24):
        """
        Initialisiert Bark Scale Processor.

        Args:
            num_bands: Number of Bark bands (default: 24)
        """
        self.num_bands = num_bands
        self.bands = get_bark_bands(num_bands)
        logger.debug("BarkScaleProcessor initialisiert with %s bands", num_bands)

    def analyze(self, audio: np.ndarray, sr: int, window: str = "hamming", normalize: bool = True) -> BarkSpectrum:
        """
        Analysiert Audio und berechnet das Bark-Spektrum.

        Args:
            audio: Input audio (mono)
            sr: Sample rate
            window: Window function ('hamming', 'hann', 'blackman')
            normalize: Whether to normalize energies to sum to 1

        Returns:
            BarkSpectrum object
        """
        # Apply window
        if window == "hamming":
            windowed = audio * np.hamming(len(audio))
        elif window == "hann":
            windowed = audio * np.hanning(len(audio))
        elif window == "blackman":
            windowed = audio * np.blackman(len(audio))
        else:
            windowed = audio

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Compute energy per Bark band
        energies = np.zeros(self.num_bands)
        for i, band in enumerate(self.bands):
            mask = (freqs >= band.lower_hz) & (freqs <= band.upper_hz)
            energies[i] = np.sum(spectrum[mask])

        total_energy: float = float(np.sum(energies))

        # Normalize if requested
        energies_normalized = energies / total_energy if normalize and total_energy > 1e-10 else energies

        # Convert to dB
        energies_db = 10 * np.log10(energies + 1e-10)

        return BarkSpectrum(
            energies=energies_normalized,
            energies_db=energies_db,
            bands=self.bands,
            sample_rate=sr,
            total_energy=total_energy,
        )

    def filter_bark_band(self, audio: np.ndarray, sr: int, bark_index: int, order: int = 8) -> np.ndarray:
        """
        Filtert audio to isolate a specific Bark band.

        Args:
            audio: Input audio
            sr: Sample rate
            bark_index: Index of Bark band (0-23)
            order: Filter order (default: 8)

        Returns:
            Filtered audio (band-pass filtered)
        """
        if bark_index < 0 or bark_index >= self.num_bands:
            raise ValueError(f"bark_index must be in range [0, {self.num_bands - 1}]")

        band = self.bands[bark_index]
        nyquist = sr / 2

        # Normalize frequencies
        low_norm = max(0.01, band.lower_hz / nyquist)
        high_norm = min(0.99, band.upper_hz / nyquist)

        if low_norm >= high_norm:
            logger.warning("Invalid band [%s]: %s >= %s", bark_index, low_norm, high_norm)
            return np.zeros_like(audio)  # type: ignore[no-any-return]

        # Design bandpass filter
        sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")

        # Apply filter
        filtered = sosfilt(sos, audio)

        return filtered  # type: ignore[no-any-return]

    def filter_bark_range(
        self, audio: np.ndarray, sr: int, lower_bark: int, upper_bark: int, order: int = 8
    ) -> np.ndarray:
        """
        Filtert audio to isolate a range of Bark bands.

        Args:
            audio: Input audio
            sr: Sample rate
            lower_bark: Lower Bark index (inclusive)
            upper_bark: Upper Bark index (inclusive)
            order: Filter order (default: 8)

        Returns:
            Filtered audio
        """
        if lower_bark < 0 or upper_bark >= self.num_bands or lower_bark > upper_bark:
            raise ValueError(f"Invalid Bark range: [{lower_bark}, {upper_bark}]")

        # Get frequency bounds
        lower_hz = self.bands[lower_bark].lower_hz
        upper_hz = self.bands[upper_bark].upper_hz

        nyquist = sr / 2
        low_norm = max(0.01, lower_hz / nyquist)
        high_norm = min(0.99, upper_hz / nyquist)

        # Design bandpass filter
        sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")

        # Apply filter
        filtered = sosfilt(sos, audio)

        return filtered  # type: ignore[no-any-return]

    def synthesize_from_bark(
        self, bark_spectrum: BarkSpectrum, audio_length: int, phase: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Synthesize audio from Bark spectrum (inverse transform).

        Args:
            bark_spectrum: BarkSpectrum object
            audio_length: Length of output audio
            phase: Optional phase spectrum (random if None)

        Returns:
            Synthesized audio
        """
        sr = bark_spectrum.sample_rate
        freqs = np.fft.rfftfreq(audio_length, 1 / sr)
        spectrum = np.zeros(len(freqs), dtype=complex)

        # Reconstruct FFT spectrum from Bark energies
        for band, energy in zip(bark_spectrum.bands, bark_spectrum.energies):
            mask = (freqs >= band.lower_hz) & (freqs <= band.upper_hz)
            magnitude = np.sqrt(energy * bark_spectrum.total_energy)

            if phase is not None:
                # Use provided phase
                spectrum[mask] = magnitude * np.exp(1j * phase[mask])
            else:
                # Random phase
                random_phase = np.random.uniform(0, 2 * np.pi, int(np.sum(mask)))
                spectrum[mask] = magnitude * np.exp(1j * random_phase)

        # IFFT
        audio = np.fft.irfft(spectrum, n=audio_length)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(audio, -1.0, 1.0)  # type: ignore[no-any-return]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_bark_spectrum(audio: np.ndarray, sr: int, normalize: bool = True) -> BarkSpectrum:
    """
    Quick bark spectrum analysis.

    Args:
        audio: Input audio (mono)
        sr: Sample rate
        normalize: Normalize energies

    Returns:
        BarkSpectrum object
    """
    processor = BarkScaleProcessor()
    return processor.analyze(audio, sr, normalize=normalize)


if __name__ == "__main__":
    """Demo bark scale processor"""
    logger.debug("\n" + "=" * 70)
    logger.debug("BARK SCALE PROCESSOR - Demo")
    logger.debug("=" * 70 + "\n")

    # Generate test signal
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Multi-frequency signal
    audio = (
        np.sin(2 * np.pi * 500 * t)  # 500 Hz (Bark ~5)
        + np.sin(2 * np.pi * 2000 * t)  # 2 kHz (Bark ~13)
        + np.sin(2 * np.pi * 8000 * t)  # 8 kHz (Bark ~21)
    )
    audio = audio / np.abs(audio).max()

    # Analyze
    processor = BarkScaleProcessor()
    spectrum = processor.analyze(audio, sr)

    logger.debug("Bark Spectrum Analyse:")
    logger.debug("  Sample Rate: %s Hz", spectrum.sample_rate)
    logger.debug("  Total Energy: %.2e", spectrum.total_energy)
    logger.debug("  Spectral Centroid: %.2f Bark", spectrum.get_spectral_centroid_bark())

    peak_band, peak_energy = spectrum.get_peak_band()
    logger.debug("\n  Peak Band: %s", peak_band)
    logger.debug("  Peak Energy: %.4f", peak_energy)

    logger.debug("\n  Top 5 Bands by Energy:")
    top_indices = np.argsort(spectrum.energies)[-5:][::-1]
    for idx in top_indices:
        band = spectrum.bands[idx]
        energy = spectrum.energies[idx]
        logger.debug("    Band %s: %.0f Hz - Energy: %.4f", idx, band.center_hz, energy)

    # Test filtering
    logger.debug("\n  Testing Band Filtering...")
    filtered_band_5 = processor.filter_bark_band(audio, sr, bark_index=5)
    filtered_band_13 = processor.filter_bark_band(audio, sr, bark_index=13)
    filtered_band_21 = processor.filter_bark_band(audio, sr, bark_index=21)

    logger.debug("    Band 5 (500 Hz): RMS = %.4f", np.sqrt(np.mean(filtered_band_5**2)))
    logger.debug("    Band 13 (2 kHz): RMS = %.4f", np.sqrt(np.mean(filtered_band_13**2)))
    logger.debug("    Band 21 (8 kHz): RMS = %.4f", np.sqrt(np.mean(filtered_band_21**2)))

    logger.debug("\n" + "=" * 70)
    logger.debug("Demo vollstaendig!")
    logger.debug("=" * 70 + "\n")
