"""
vocal_presence_enhancer.py - Vocal Presence Enhancement (Phase 2.2)

Professional vocal clarity and presence enhancement:
- Harmonic Enhancement (natural brilliance)
- Air Band Processing (12-20 kHz clarity)
- Broadcast Clarity (3-8 kHz intelligibility)
- Vocal Saturation (subtle warmth & analog character)

Author: AURIK Development Team
Version: 1.0.0
Date: 9. Februar 2026
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)
from scipy import signal
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore", category=RuntimeWarning)


class HarmonicEnhancer:
    """
    Enhances harmonics for natural brilliance without harshness.
    """

    def __init__(
        self,
        fundamental_range: tuple[float, float] = (80.0, 500.0),
        harmonic_gain_db: float = 2.0,
        max_harmonic: int = 8,
    ):
        """
        Parameters
        ----------
        fundamental_range : Tuple[float, float]
            Expected fundamental frequency range (Hz)
        harmonic_gain_db : float
            Gain applied to harmonics (dB)
        max_harmonic : int
            Maximum harmonic to enhance
        """
        self.fundamental_range = fundamental_range
        self.harmonic_gain_db = harmonic_gain_db
        self.max_harmonic = max_harmonic

    def enhance(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance harmonics for natural brilliance.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_enhanced : np.ndarray
            Enhanced audio
        metrics : Dict
            Enhancement metrics
        """
        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples)
                left = self.enhance(audio[0], sr)[0]
                right = self.enhance(audio[1], sr)[0]
                return np.vstack([left, right]), {"stereo": True}
            else:
                # Format: (samples, channels) - AURIK standard
                left = self.enhance(audio[:, 0], sr)[0]
                right = self.enhance(audio[:, 1], sr)[0]
                return np.column_stack([left, right]), {"stereo": True}

        # Detect fundamental frequency
        f0 = self._detect_fundamental(audio, sr)

        if f0 < self.fundamental_range[0] or f0 > self.fundamental_range[1]:
            # No valid fundamental detected
            return audio, {"fundamental_hz": 0.0, "harmonics_enhanced": 0}

        # Enhance harmonics
        audio_enhanced = audio.copy()
        harmonics_enhanced = 0

        for n in range(2, self.max_harmonic + 1):
            harmonic_freq = f0 * n

            if harmonic_freq > sr / 2:
                break

            # Apply subtle boost to harmonic
            audio_enhanced = self._boost_harmonic(audio_enhanced, sr, harmonic_freq, self.harmonic_gain_db / n)
            harmonics_enhanced += 1

        # NaN/Inf-Guard + Clipping
        audio_enhanced = np.nan_to_num(audio_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        audio_enhanced = np.clip(audio_enhanced, -1.0, 1.0)

        metrics = {"fundamental_hz": f0, "harmonics_enhanced": harmonics_enhanced, "gain_db": self.harmonic_gain_db}

        return audio_enhanced, metrics

    def _detect_fundamental(self, audio: np.ndarray, sr: int) -> float:
        """
        Erkennt fundamental frequency using autocorrelation.
        """
        # Find peaks in expected lag range
        min_lag = int(sr / self.fundamental_range[1])
        max_lag = int(sr / self.fundamental_range[0])

        _n = len(audio)
        if max_lag >= _n or min_lag >= max_lag:
            return 0.0

        # Efficient O(n·order) autocorrelation — only lags 0..max_lag (avoids O(n²) full correlate)
        corr = np.array([np.dot(audio[: _n - k], audio[k:]) for k in range(max_lag + 1)])

        corr_range = corr[min_lag:max_lag]

        if len(corr_range) == 0:
            return 0.0

        # Find peak
        peak_lag = np.argmax(corr_range) + min_lag
        f0 = sr / peak_lag

        return f0  # type: ignore[return-value]

    def _boost_harmonic(self, audio: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
        """
        Wendet an: subtle boost to specific harmonic frequency.
        """
        # Narrow bandwidth (5% of frequency)
        bandwidth = freq * 0.05
        Q = freq / bandwidth
        A = 10 ** (gain_db / 40)

        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        audio = np.asarray(signal.lfilter(b, a, audio), dtype=np.float64)

        return audio


class AirBandProcessor:
    """
    Verarbeitet "air band" (12-20 kHz) for clarity and openness.
    """

    def __init__(self, frequency_hz: float = 15000.0, gain_db: float = 2.5, smooth: bool = True):
        """
        Parameters
        ----------
        frequency_hz : float
            Air band center frequency (12-20 kHz)
        gain_db : float
            Gain in dB
        smooth : bool
            Apply smooth high-shelf instead of peak
        """
        self.frequency_hz = frequency_hz
        self.gain_db = gain_db
        self.smooth = smooth

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Verarbeitet air band for clarity.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_processed : np.ndarray
            Processed audio
        metrics : Dict
            Processing metrics
        """
        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples)
                left = self.process(audio[0], sr)[0]
                right = self.process(audio[1], sr)[0]
                return np.vstack([left, right]), {"stereo": True}
            else:
                # Format: (samples, channels) - AURIK standard
                left = self.process(audio[:, 0], sr)[0]
                right = self.process(audio[:, 1], sr)[0]
                return np.column_stack([left, right]), {"stereo": True}

        # Check if air band is present in sample rate
        nyquist = sr / 2
        if self.frequency_hz > nyquist:
            # Sample rate too low for air band processing
            return audio, {"error": "Sample rate too low for air band"}

        # Apply enhancement
        if self.smooth:
            audio_processed = self._apply_high_shelf(audio, sr, self.frequency_hz, self.gain_db)
        else:
            audio_processed = self._apply_air_peak(audio, sr, self.frequency_hz, self.gain_db)

        metrics = {
            "frequency_hz": self.frequency_hz,
            "gain_db": self.gain_db,
            "type": "high_shelf" if self.smooth else "peak",
        }

        return audio_processed, metrics

    def _apply_high_shelf(self, audio: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
        """
        Wendet an: high-shelf filter.
        """
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / 0.707 - 1) + 2)

        cos_w0 = np.cos(w0)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        audio = np.asarray(signal.lfilter(b, a, audio), dtype=np.float64)

        return audio

    def _apply_air_peak(self, audio: np.ndarray, sr: int, freq: float, gain_db: float) -> np.ndarray:
        """
        Wendet an: peak EQ at air band frequency.
        """
        bandwidth = 4000  # Wide bandwidth for natural sound
        Q = freq / bandwidth
        A = 10 ** (gain_db / 40)

        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        audio = np.asarray(signal.lfilter(b, a, audio), dtype=np.float64)

        return audio


class BroadcastClarityEnhancer:
    """
    Enhances broadcast clarity in 3-8 kHz range (intelligibility).
    """

    def __init__(self, presence_freq_hz: float = 5000.0, gain_db: float = 3.0, q_factor: float = 1.5):
        """
        Parameters
        ----------
        presence_freq_hz : float
            Presence frequency (3-8 kHz)
        gain_db : float
            Gain in dB
        q_factor : float
            Q factor for EQ
        """
        self.presence_freq_hz = presence_freq_hz
        self.gain_db = gain_db
        self.q_factor = q_factor

    def enhance(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance broadcast clarity.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_enhanced : np.ndarray
            Enhanced audio
        metrics : Dict
            Enhancement metrics
        """
        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples)
                left = self.enhance(audio[0], sr)[0]
                right = self.enhance(audio[1], sr)[0]
                return np.vstack([left, right]), {"stereo": True}
            else:
                # Format: (samples, channels) - AURIK standard
                left = self.enhance(audio[:, 0], sr)[0]
                right = self.enhance(audio[:, 1], sr)[0]
                return np.column_stack([left, right]), {"stereo": True}

        # Apply presence boost
        audio_enhanced = self._apply_presence_eq(audio, sr, self.presence_freq_hz, self.q_factor, self.gain_db)

        # Measure enhancement
        presence_before = self._measure_presence(audio, sr)
        presence_after = self._measure_presence(audio_enhanced, sr)

        metrics = {
            "frequency_hz": self.presence_freq_hz,
            "gain_db": self.gain_db,
            "presence_before": presence_before,
            "presence_after": presence_after,
            "improvement_db": 20 * np.log10(presence_after / (presence_before + 1e-10)),
        }

        return audio_enhanced, metrics

    def _apply_presence_eq(self, audio: np.ndarray, sr: int, freq: float, Q: float, gain_db: float) -> np.ndarray:
        """
        Wendet an: presence EQ boost.
        """
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        audio = np.asarray(signal.lfilter(b, a, audio), dtype=np.float64)

        return audio

    def _measure_presence(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst energy in presence band (3-8 kHz).
        """
        # Bandpass filter
        nyquist = sr / 2
        low = min(3000 / nyquist, 0.98)
        high = min(8000 / nyquist, 0.99)

        sos = butter(4, [low, high], btype="bandpass", output="sos")
        presence_band = sosfilt(sos, audio)

        # RMS energy
        presence_energy = np.sqrt(np.mean(presence_band**2))

        return presence_energy  # type: ignore[no-any-return]


class VocalSaturation:
    """
    Applies subtle vocal saturation for warmth and analog character.
    """

    def __init__(self, drive_db: float = 3.0, mix: float = 0.3):
        """
        Parameters
        ----------
        drive_db : float
            Drive amount in dB
        mix : float
            Wet/dry mix (0.0-1.0)
        """
        self.drive_db = drive_db
        self.mix = np.clip(mix, 0.0, 1.0)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Wendet an: subtle saturation.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_saturated : np.ndarray
            Saturated audio
        metrics : Dict
            Processing metrics
        """
        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples)
                left = self.process(audio[0], sr)[0]
                right = self.process(audio[1], sr)[0]
                return np.vstack([left, right]), {"stereo": True}
            else:
                # Format: (samples, channels) - AURIK standard
                left = self.process(audio[:, 0], sr)[0]
                right = self.process(audio[:, 1], sr)[0]
                return np.column_stack([left, right]), {"stereo": True}

        if np.max(np.abs(audio)) <= 0.0:
            return audio, {"error": "Silent audio"}

        # Apply drive without peak normalization
        drive_linear = 10 ** (self.drive_db / 20)
        audio_driven = audio * drive_linear

        # Soft clipping (tanh saturation)
        audio_saturated = np.tanh(audio_driven)

        # Mix with dry signal
        audio_mixed = self.mix * audio_saturated + (1 - self.mix) * audio

        # Measure harmonics added
        thd_before = self._measure_thd(audio, sr)
        thd_after = self._measure_thd(audio_mixed, sr)

        metrics = {
            "drive_db": self.drive_db,
            "mix": self.mix,
            "thd_before": thd_before,
            "thd_after": thd_after,
            "harmonics_added": thd_after - thd_before,
        }

        return audio_mixed, metrics

    def _measure_thd(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Total Harmonic Distortion (simplified).
        """
        # FFT
        n_fft = len(audio)
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft))

        # Fundamental energy (assume low frequency is fundamental)
        fundamental_energy = np.sum(spectrum[: len(spectrum) // 10] ** 2)

        # Harmonic energy (rest of spectrum)
        harmonic_energy = np.sum(spectrum[len(spectrum) // 10 :] ** 2)

        # THD ratio
        thd = np.sqrt(harmonic_energy / fundamental_energy) if fundamental_energy > 0 else 0.0

        return thd


class VocalPresenceEnhancer:
    """
    Einheitliche API für Vokal-Präsenz-Verbesserung.
    """

    def __init__(
        self,
        harmonic_gain_db: float = 2.0,
        air_gain_db: float = 2.5,
        presence_gain_db: float = 3.0,
        saturation_mix: float = 0.3,
    ):
        """
        Parameters
        ----------
        harmonic_gain_db : float
            Harmonic enhancement gain
        air_gain_db : float
            Air band gain (12-20 kHz)
        presence_gain_db : float
            Broadcast presence gain (3-8 kHz)
        saturation_mix : float
            Saturation wet/dry mix
        """
        self.harmonic_enhancer = HarmonicEnhancer(harmonic_gain_db=harmonic_gain_db)
        self.air_processor = AirBandProcessor(gain_db=air_gain_db)
        self.clarity_enhancer = BroadcastClarityEnhancer(gain_db=presence_gain_db)
        self.saturation = VocalSaturation(mix=saturation_mix)

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full vocal presence enhancement pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_enhanced : np.ndarray
            Enhanced audio
        report : Dict
            Processing report
        """
        # Stage 1: Harmonic enhancement
        audio_harmonics, harmonics_metrics = self.harmonic_enhancer.enhance(audio, sr)

        # Stage 2: Broadcast clarity
        audio_clarity, clarity_metrics = self.clarity_enhancer.enhance(audio_harmonics, sr)

        # Stage 3: Air band
        audio_air, air_metrics = self.air_processor.process(audio_clarity, sr)

        # Stage 4: Subtle saturation
        audio_final, saturation_metrics = self.saturation.process(audio_air, sr)

        report = {
            "harmonic_enhancement": harmonics_metrics,
            "broadcast_clarity": clarity_metrics,
            "air_band": air_metrics,
            "vocal_saturation": saturation_metrics,
        }

        return audio_final, report


# CLI interface
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Vocal Presence Enhancer - Professional vocal clarity")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", help="Output audio file")
    parser.add_argument("--harmonic-gain", type=float, default=2.0, help="Harmonic enhancement gain (dB)")
    parser.add_argument("--air-gain", type=float, default=2.5, help="Air band gain (dB)")
    parser.add_argument("--presence-gain", type=float, default=3.0, help="Presence gain (dB)")
    parser.add_argument("--saturation", type=float, default=0.3, help="Saturation mix (0.0-1.0)")

    args = parser.parse_args()

    # Load audio
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Process
    enhancer = VocalPresenceEnhancer(
        harmonic_gain_db=args.harmonic_gain,
        air_gain_db=args.air_gain,
        presence_gain_db=args.presence_gain,
        saturation_mix=args.saturation,
    )

    audio_enhanced, report = enhancer.process(audio, sr)

    # Print report
    logger.info(str("\n" + "=" * 70))
    logger.info("VOCAL PRESENCE ENHANCER REPORT")
    logger.info(str("=" * 70))

    logger.info("\n[Harmonic Enhancement]")
    harm = report["harmonic_enhancement"]
    if "fundamental_hz" in harm:
        logger.info("  Fundamental:       %.1f Hz", harm["fundamental_hz"])
        logger.info("  Harmonics verbessert: %s", harm["harmonics_enhanced"])
        logger.info("  Gain angewendet:      %.1f dB", harm["gain_db"])

    logger.info("\n[Broadcast Clarity]")
    clar = report["broadcast_clarity"]
    if "improvement_db" in clar:
        logger.info("  Frequency:         %.0f Hz", clar["frequency_hz"])
        logger.info("  Improvement:       %.1f dB", clar["improvement_db"])

    logger.info("\n[Air Band]")
    air = report["air_band"]
    if "frequency_hz" in air:
        logger.info("  Frequency:         %.0f Hz", air["frequency_hz"])
        logger.info("  Gain:              %.1f dB", air["gain_db"])
        logger.info("  Type:              %s", air["type"])

    logger.info("\n[Vocal Saturation]")
    sat = report["vocal_saturation"]
    if "harmonics_added" in sat:
        logger.info("  Harmonics added:   %.4f", sat["harmonics_added"])
        logger.info("  Mix:               %s", format(sat["mix"], ".1%"))

    logger.info(str("=" * 70))

    # Save
    if args.output:
        sf.write(args.output, audio_enhanced, sr)
        logger.info("\n✅ gespeichert to: %s", args.output)
