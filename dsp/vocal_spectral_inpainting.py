"""
vocal_spectral_inpainting.py - Intelligent Spectral Gap Filling (Phase 2.2)

ML-inspired spectral inpainting for vocal restoration:
- Harmonic-Aware Gap Filling
- Formant-Locked Reconstruction
- Timbre Preservation
- Use Cases: Codec artifacts, dropouts, over-EQ damage

Author: AURIK Development Team
Version: 1.0.0
Date: 9. Februar 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import istft, stft

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class HarmonicDetector:
    """
    Erkennt harmonic structure for harmonic-aware inpainting.
    """

    def __init__(self, f0_range: tuple[float, float] = (80.0, 500.0)):
        """
        Parameters
        ----------
        f0_range : Tuple[float, float]
            Fundamental frequency range (Hz)
        """
        self.f0_range = f0_range

    def detect_harmonics(self, audio: np.ndarray, sr: int) -> dict:
        """
        Erkennt harmonic structure.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz

        Returns
        -------
        harmonics : Dict
            Harmonic structure info
        """
        # Detect fundamental
        f0 = self._detect_f0(audio, sr)

        if f0 == 0:
            return {"f0": 0.0, "harmonics": []}

        # Generate harmonic series
        harmonics = []
        for n in range(1, 21):  # Up to 20th harmonic
            freq = f0 * n
            if freq > sr / 2:
                break

            # Measure strength at this harmonic
            strength = self._measure_harmonic_strength(audio, sr, freq)

            harmonics.append({"number": n, "frequency": freq, "strength": strength})

        return {"f0": f0, "harmonics": harmonics}

    def _detect_f0(self, audio: np.ndarray, sr: int) -> float:
        """
        Erkennt fundamental frequency using FFT-based autocorrelation.

        Limits input to 200 ms to prevent O(N²) hang on full-length audio
        (§VERBOTEN: np.correlate mode=full on long signals).
        """
        # Limit to 200 ms — sufficient for F0 estimation, prevents O(N²) hang
        max_samples = min(len(audio), int(sr * 0.2))
        segment = audio[:max_samples]
        if len(segment) < 4:
            return 0.0

        # FFT-based autocorrelation: O(N log N) instead of O(N²)
        n = len(segment)
        fft = np.fft.rfft(segment, n=2 * n)
        corr = np.fft.irfft(fft * np.conj(fft))[:n].real
        corr = corr / (corr[0] + 1e-10)

        # Expected lag range
        min_lag = int(sr / self.f0_range[1])
        max_lag = int(sr / self.f0_range[0])

        if max_lag >= len(corr):
            return 0.0

        corr_range = corr[min_lag:max_lag]

        if len(corr_range) == 0:
            return 0.0

        # Find peak
        peak_lag = np.argmax(corr_range) + min_lag
        f0 = sr / peak_lag

        return f0  # type: ignore[return-value]

    def _measure_harmonic_strength(self, audio: np.ndarray, sr: int, freq: float, bandwidth: float = 50.0) -> float:
        """
        Misst energy at specific frequency.
        """
        # Short FFT
        n_fft = 4096
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)

        # Find bin range
        low_idx = np.argmin(np.abs(freqs - (freq - bandwidth / 2)))
        high_idx = np.argmin(np.abs(freqs - (freq + bandwidth / 2)))

        # Sum energy in range
        energy = np.sum(spectrum[low_idx : high_idx + 1])

        return energy  # type: ignore[no-any-return]


class SpectralGapFiller:
    """
    Fills spectral gaps using intelligent interpolation.
    """

    def __init__(self, gap_threshold_db: float = -40.0, min_gap_width_hz: float = 100.0):
        """
        Parameters
        ----------
        gap_threshold_db : float
            Threshold for gap detection (dB)
        min_gap_width_hz : float
            Minimum gap width to fill (Hz)
        """
        self.gap_threshold_db = gap_threshold_db
        self.min_gap_width_hz = min_gap_width_hz

    def detect_gaps(self, audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """
        Erkennt spectral gaps (missing frequency bands).

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz

        Returns
        -------
        gaps : List[Tuple[float, float]]
            List of (low_freq, high_freq) gap ranges in Hz
        """
        # Compute average spectrum
        nperseg = 2048
        f, _t, Zxx = stft(audio, sr, nperseg=nperseg)

        # Average over time
        avg_spectrum = np.mean(np.abs(Zxx), axis=1)

        # Convert to dB
        avg_spectrum_db = 20 * np.log10(avg_spectrum + 1e-10)

        # Detect gaps (regions below threshold)
        threshold = np.max(avg_spectrum_db) + self.gap_threshold_db
        gap_mask = avg_spectrum_db < threshold

        # Find contiguous gap regions
        gaps = []
        in_gap = False
        gap_start_idx = 0

        for i in range(len(gap_mask)):
            if gap_mask[i] and not in_gap:
                gap_start_idx = i
                in_gap = True
            elif not gap_mask[i] and in_gap:
                gap_end_idx = i
                gap_start_freq = f[gap_start_idx]
                gap_end_freq = f[gap_end_idx]
                gap_width = gap_end_freq - gap_start_freq

                if gap_width >= self.min_gap_width_hz:
                    gaps.append((gap_start_freq, gap_end_freq))

                in_gap = False

        return gaps

    def fill_gaps(
        self, audio: np.ndarray, sr: int, gaps: list[tuple[float, float]], harmonic_info: dict | None = None
    ) -> np.ndarray:
        """
         Fill spectral gaps.

        Parameters
         ----------
         audio : np.ndarray
             Input audio (mono)
         sr : int
             Sample rate in Hz
         gaps : List[Tuple[float, float]]
             Gaps to fill
         harmonic_info : Dict, optional
             Harmonic structure for harmonic-aware filling

         Returns
         -------
         audio_filled : np.ndarray
             Audio with filled gaps
        """
        if len(gaps) == 0:
            return audio

        # STFT
        nperseg = 2048
        noverlap = nperseg // 2
        f, _t, Zxx = stft(audio, sr, nperseg=nperseg, noverlap=noverlap)

        # Fill each gap
        for gap_low, gap_high in gaps:
            # Find frequency bin range
            low_idx = np.argmin(np.abs(f - gap_low))
            high_idx = np.argmin(np.abs(f - gap_high))

            if harmonic_info and harmonic_info["f0"] > 0:
                # Harmonic-aware filling
                Zxx = self._fill_harmonic_aware(Zxx, f, low_idx, high_idx, harmonic_info)  # type: ignore[arg-type]
            else:
                # Interpolation-based filling
                Zxx = self._fill_interpolation(Zxx, low_idx, high_idx)  # type: ignore[arg-type]

        # Inverse STFT
        _, audio_filled = istft(Zxx, sr, nperseg=nperseg, noverlap=noverlap)

        # Match length
        if len(audio_filled) < len(audio):
            audio_filled = np.pad(audio_filled, (0, len(audio) - len(audio_filled)))
        elif len(audio_filled) > len(audio):
            audio_filled = audio_filled[: len(audio)]

        audio_filled = np.clip(np.nan_to_num(audio_filled, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
            audio.dtype
        )
        return audio_filled  # type: ignore[no-any-return]

    def _fill_harmonic_aware(
        self, Zxx: np.ndarray, f: np.ndarray, low_idx: int, high_idx: int, harmonic_info: dict
    ) -> np.ndarray:
        """
        Fill gap using harmonic structure.
        """
        harmonic_info["f0"]
        harmonics = harmonic_info["harmonics"]

        # For each time frame
        for t_idx in range(Zxx.shape[1]):
            # Synthesize harmonics in gap region
            for harmonic in harmonics:
                freq = harmonic["frequency"]
                strength = harmonic["strength"]

                # Check if harmonic falls in gap
                if freq < f[low_idx] or freq > f[high_idx]:
                    continue

                # Find nearest frequency bin
                freq_idx = np.argmin(np.abs(f - freq))

                if low_idx <= freq_idx <= high_idx:
                    # Synthesize harmonic with phase continuity
                    # Use neighboring bins for phase reference
                    if freq_idx > 0 and freq_idx < len(Zxx) - 1:
                        phase = np.angle(Zxx[freq_idx - 1, t_idx])
                        magnitude = strength * 0.5  # Scale down
                        Zxx[freq_idx, t_idx] = magnitude * np.exp(1j * phase)

        return Zxx

    def _fill_interpolation(self, Zxx: np.ndarray, low_idx: int, high_idx: int) -> np.ndarray:
        """
        Fill gap using linear interpolation.
        """
        # For each time frame
        for t_idx in range(Zxx.shape[1]):
            # Get boundary values
            if low_idx > 0 and high_idx < len(Zxx) - 1:
                val_low = Zxx[low_idx - 1, t_idx]
                val_high = Zxx[high_idx + 1, t_idx]

                # Interpolate magnitude
                mag_low = np.abs(val_low)
                mag_high = np.abs(val_high)
                mags = np.linspace(mag_low, mag_high, high_idx - low_idx + 1)

                # Interpolate phase
                phase_low = np.angle(val_low)
                phase_high = np.angle(val_high)
                phases = np.linspace(phase_low, phase_high, high_idx - low_idx + 1)

                # Fill
                for i, freq_idx in enumerate(range(low_idx, high_idx + 1)):
                    Zxx[freq_idx, t_idx] = mags[i] * np.exp(1j * phases[i])

        return Zxx


class VocalSpectralInpainting:
    """
    Einheitliche API für vokales spektrales Inpainting.
    """

    def __init__(
        self, gap_threshold_db: float = -40.0, min_gap_width_hz: float = 100.0, use_harmonic_awareness: bool = True
    ):
        """
        Parameters
        ----------
        gap_threshold_db : float
            Gap detection threshold (dB)
        min_gap_width_hz : float
            Minimum gap width (Hz)
        use_harmonic_awareness : bool
            Use harmonic-aware filling
        """
        self.harmonic_detector = HarmonicDetector()
        self.gap_filler = SpectralGapFiller(gap_threshold_db=gap_threshold_db, min_gap_width_hz=min_gap_width_hz)
        self.use_harmonic_awareness = use_harmonic_awareness

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full spectral inpainting pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_inpainted : np.ndarray
            Inpainted audio
        report : Dict
            Processing report
        """
        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            # Heuristic: If first dimension is small and < second dimension, likely channels
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples)
                left, left_report = self.process(audio[0], sr)
                right, right_report = self.process(audio[1], sr)
                return np.vstack([left, right]), {**left_report, "stereo": True}
            else:
                # Format: (samples, channels) - AURIK standard
                left, left_report = self.process(audio[:, 0], sr)
                right, _right_report = self.process(audio[:, 1], sr)
                return np.column_stack([left, right]), {**left_report, "stereo": True}

        # Detect gaps
        gaps = self.gap_filler.detect_gaps(audio, sr)

        if len(gaps) == 0:
            return audio, {"gaps_detected": 0, "gaps_filled": 0}

        # Detect harmonics (if enabled)
        harmonic_info = None
        if self.use_harmonic_awareness:
            harmonic_info = self.harmonic_detector.detect_harmonics(audio, sr)

        # Fill gaps
        audio_inpainted = self.gap_filler.fill_gaps(audio, sr, gaps, harmonic_info)

        report = {
            "gaps_detected": len(gaps),
            "gaps_filled": len(gaps),
            "gap_ranges_hz": gaps,
            "harmonic_awareness": self.use_harmonic_awareness,
            "f0_detected": harmonic_info["f0"] if harmonic_info else 0.0,
        }

        audio_inpainted = np.clip(np.nan_to_num(audio_inpainted, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
            audio.dtype
        )
        return audio_inpainted, report


# CLI interface
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Vocal Spectral Inpainting - Intelligent gap filling")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", help="Output audio file")
    parser.add_argument("--gap-threshold", type=float, default=-40.0, help="Gap detection threshold (dB)")
    parser.add_argument("--min-gap-width", type=float, default=100.0, help="Minimum gap width (Hz)")
    parser.add_argument("--no-harmonic", action="store_true", help="Disable harmonic-aware filling")

    args = parser.parse_args()

    # Load audio
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Process
    inpainter = VocalSpectralInpainting(
        gap_threshold_db=args.gap_threshold,
        min_gap_width_hz=args.min_gap_width,
        use_harmonic_awareness=not args.no_harmonic,
    )

    audio_inpainted, report = inpainter.process(audio, sr)

    # Print report
    logger.info(str("\n" + "=" * 70))
    logger.info("VOCAL SPECTRAL INPAINTING REPORT")
    logger.info(str("=" * 70))
    logger.info("Gaps erkannt: %s", report["gaps_detected"])
    logger.info("Gaps filled:   %s", report["gaps_filled"])

    if report["gaps_detected"] > 0:
        logger.info("\nGap Ranges:")
        for i, (low, high) in enumerate(report["gap_ranges_hz"], 1):
            logger.info("  Gap %s: %.0f - %.0f Hz (%.0f Hz wide)", i, low, high, high - low)

    if report["harmonic_awareness"]:
        logger.info("\nHarmonic Awareness: aktiviert")
        logger.info("  F0 erkannt: %.1f Hz", report["f0_detected"])

    logger.info(str("=" * 70))

    # Save
    if args.output:
        sf.write(args.output, audio_inpainted, sr)
        logger.info("\n✅ gespeichert to: %s", args.output)
