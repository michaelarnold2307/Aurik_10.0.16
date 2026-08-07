"""
==============================

Fügt "Air" und "Presence" hinzu - die feinen High-Frequency Details,
die modernem Audio "Raum zum Atmen" geben.

Was ist "Air"?
--------------
- Ultra-High Frequencies (12-20 kHz)
- Subtile HF Details, die Musik "öffnen"
- Der "Raum" bzw. "Space" zwischen Instrumenten
- Beispiel: Das leise "Sssss" von Becken, das Rauschen von Bändern

Was ist "Presence"?
-------------------
- Upper-Mid / Low-High Frequencies (4-8 kHz)
- Klarheit und "Nähe" von Vocals/Instrumenten
- "Vorne im Mix" Sound
- Beispiel: Vocal-Konsonanten (S, T, K), Snare-Attack

Ziel:
-----
- +1-2 dB @ 12 kHz (Air)
- +1-1.5 dB @ 5-6 kHz (Presence)
- Sanfte, musikalische EQ-Kurven (kein Harshness!)
- Optional: Micro-Reverb für "Space" (< 50ms)

Wissenschaftliche Grundlagen:
-----------------------------
- Katz (2014): "Mastering Audio"
- Owsinski (2014): "The Mixing Engineer's Handbook"

Autor: AURIK Phase 2.0 - Psychoakustische Exzellenz
Datum: 13. Februar 2026
"""

import logging

logger = logging.getLogger(__name__)

import warnings

import librosa
import numpy as np
from scipy import signal

warnings.filterwarnings("ignore", category=RuntimeWarning)


class AirPresenceEnhancer:
    """
    Air & Presence Enhancer

    Fügt subtile High-Frequency Details hinzu für "Offenheit" und "Klarheit".

    Parameters
    ----------
    air_gain_db : float
        Gain für Air band (12-20 kHz), typical 1-2 dB
    presence_gain_db : float
        Gain für Presence band (4-8 kHz), typical 1-1.5 dB
    add_micro_reverb : bool
        Fügt Micro-Reverb hinzu für "Space" (<50ms)
    micro_reverb_mix : float
        Micro-Reverb Dry/Wet (0.0-1.0), typical 0.10-0.15
    smooth_transitions : bool
        Sanfte EQ-Übergänge (verhindert Harshness)
    """

    def __init__(
        self,
        air_gain_db: float = 1.5,
        presence_gain_db: float = 1.0,
        add_micro_reverb: bool = True,
        micro_reverb_mix: float = 0.12,
        smooth_transitions: bool = True,
    ):
        self.air_gain_db = np.clip(air_gain_db, 0.0, 3.0)
        self.presence_gain_db = np.clip(presence_gain_db, 0.0, 2.0)
        self.add_micro_reverb = add_micro_reverb
        self.micro_reverb_mix = np.clip(micro_reverb_mix, 0.0, 0.3)
        self.smooth_transitions = smooth_transitions

        # EQ parameters
        self.air_freq = 12000  # Hz - Center of Air band
        self.presence_freq = 5500  # Hz - Center of Presence band
        self.air_q = 0.7 if smooth_transitions else 1.0  # Broader = smoother
        self.presence_q = 0.9 if smooth_transitions else 1.2

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, float]]:
        """
        Wendet Air- und Präsenz-Verbesserung an.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        enhanced : np.ndarray
            Enhanced audio
        report : dict
            Processing report
        """
        if sr <= 0:
            raise ValueError(f"Sample rate must be > 0 Hz, got {sr}")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Store original format
        is_stereo = audio.ndim == 2

        if is_stereo:
            # Process channels separately
            left = self._process_channel(audio[:, 0], sr)
            right = self._process_channel(audio[:, 1], sr)
            enhanced = np.stack([left, right], axis=1)
        else:
            enhanced = self._process_channel(audio, sr)

        # Measure before/after
        report = self._generate_report(audio, enhanced, sr)

        # Final safety: NaN/Inf-Guard and clipping
        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced = np.clip(enhanced, -1.0, 1.0)

        return enhanced, report

    def _process_channel(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Verarbeitet single channel.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio signal
        sr : int
            Sample rate

        Returns
        -------
        enhanced : np.ndarray
            Enhanced mono audio
        """
        # Start with original
        enhanced = audio.copy()

        # 1. Air Enhancement (High-Shelf @ 12 kHz)
        if self.air_gain_db > 0:
            enhanced = self._apply_high_shelf(enhanced, sr, freq=self.air_freq, gain_db=self.air_gain_db, q=self.air_q)

        # 2. Presence Enhancement (Bell @ 5.5 kHz)
        if self.presence_gain_db > 0:
            enhanced = self._apply_bell(
                enhanced,
                sr,
                freq=self.presence_freq,
                gain_db=self.presence_gain_db,
                q=self.presence_q,
            )

        # 3. Micro-Reverb for "Space" (optional)
        if self.add_micro_reverb and self.micro_reverb_mix > 0:
            enhanced = self._apply_micro_reverb(enhanced, sr, mix=self.micro_reverb_mix)

        # NaN/Inf-Guard after all processing
        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced = np.clip(enhanced, -1.0, 1.0)

        return enhanced

    def _apply_high_shelf(self, audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
        """
        Wendet an: High-Shelf EQ.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        freq : float
            Shelf frequency in Hz
        gain_db : float
            Gain in dB
        q : float
            Q factor (0.5-1.0 = broad, >1.0 = narrow)

        Returns
        -------
        filtered : np.ndarray
            Filtered audio
        """
        # Convert gain to linear
        gain_linear = 10 ** (gain_db / 20)

        # Robert Bristow-Johnson Audio EQ Cookbook: High-Shelf biquad
        A = gain_linear
        w0 = 2.0 * np.pi * freq / sr
        if w0 >= np.pi:
            return audio * gain_linear  # freq too high
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)
        sqrt_A = np.sqrt(max(A, 1e-10))

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

        # Normalize by a0
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        filtered = signal.sosfilt(sos, audio)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        return filtered  # type: ignore[no-any-return]

    def _apply_bell(self, audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
        """
        Wendet Bell-(Peaking)-EQ an.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        freq : float
            Center frequency in Hz
        gain_db: float
            Gain in dB
        q : float
            Q factor (bandwidth)

        Returns
        -------
        filtered : np.ndarray
            Filtered audio
        """
        # Robert Bristow-Johnson Audio EQ Cookbook: Peaking (Bell) biquad
        A = 10.0 ** (gain_db / 40.0)  # sqrt of linear gain
        w0 = 2.0 * np.pi * freq / sr
        if w0 >= np.pi:
            return audio  # freq too high for this sample rate
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        filtered = signal.sosfilt(sos, audio)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        return filtered  # type: ignore[no-any-return]

    def _apply_micro_reverb(self, audio: np.ndarray, sr: int, mix: float) -> np.ndarray:
        """
        Wendet an: Micro-Reverb (<50ms) for "Space.

        Simple early reflections simulation.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate
        mix : float
            Dry/wet mix (0.0-1.0)

        Returns
        -------
        reverbed : np.ndarray
            Audio with micro-reverb
        """
        # Create early reflections (< 50ms)
        # Delays: 10ms, 20ms, 35ms, 48ms (golden ratios)
        delays_ms = [10, 20, 35, 48]
        gains = [0.3, 0.25, 0.15, 0.10]  # Decaying amplitude

        # Start with dry signal
        wet = np.zeros_like(audio)

        for delay_ms, gain in zip(delays_ms, gains):
            delay_samples = int(delay_ms * sr / 1000)

            if delay_samples < len(audio):
                # Create delayed copy
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain

                # Add to wet signal
                wet += delayed

        # Mix dry and wet
        reverbed = (1.0 - mix) * audio + mix * wet

        return reverbed

    def _generate_report(self, original: np.ndarray, enhanced: np.ndarray, sr: int) -> dict[str, float]:
        """
        Generiert processing report.

        Parameters
        ----------
        original : np.ndarray
            Original audio
        enhanced : np.ndarray
            Enhanced audio
        sr : int
            Sample rate

        Returns
        -------
        report : dict
            Processing metrics
        """
        # Convert to mono for analysis
        if original.ndim == 2:
            orig_mono = np.mean(original, axis=0)
            enh_mono = np.mean(enhanced, axis=0)
        else:
            orig_mono = original
            enh_mono = enhanced

        # Compute STFT
        stft_orig = librosa.stft(orig_mono, n_fft=4096, hop_length=1024)
        stft_enh = librosa.stft(enh_mono, n_fft=4096, hop_length=1024)

        mag_orig = np.abs(stft_orig)
        mag_enh = np.abs(stft_enh)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        # Measure energy in Air band (12-20 kHz)
        air_mask = (freqs >= 12000) & (freqs <= 20000)
        air_energy_orig = np.mean(mag_orig[air_mask] ** 2)
        air_energy_enh = np.mean(mag_enh[air_mask] ** 2)
        air_boost_db = 10 * np.log10(air_energy_enh / (air_energy_orig + 1e-10))

        # Measure energy in Presence band (4-8 kHz)
        presence_mask = (freqs >= 4000) & (freqs <= 8000)
        presence_energy_orig = np.mean(mag_orig[presence_mask] ** 2)
        presence_energy_enh = np.mean(mag_enh[presence_mask] ** 2)
        presence_boost_db = 10 * np.log10(presence_energy_enh / (presence_energy_orig + 1e-10))

        # Measure overall RMS change
        rms_orig = np.sqrt(np.mean(orig_mono**2))
        rms_enh = np.sqrt(np.mean(enh_mono**2))
        overall_gain_db = 20 * np.log10(rms_enh / (rms_orig + 1e-10))

        report = {
            "air_gain_applied_db": self.air_gain_db,
            "presence_gain_applied_db": self.presence_gain_db,
            "air_boost_measured_db": air_boost_db,
            "presence_boost_measured_db": presence_boost_db,
            "overall_gain_db": overall_gain_db,
            "micro_reverb_applied": self.add_micro_reverb,
            "micro_reverb_mix": self.micro_reverb_mix if self.add_micro_reverb else 0.0,
        }

        return report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def enhance_air_presence(
    audio: np.ndarray,
    sr: int,
    air_gain_db: float = 1.5,
    presence_gain_db: float = 1.0,
    add_micro_reverb: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Convenience function to enhance Air & Presence

    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sr : int
        Sample rate in Hz
    air_gain_db : float
        Air band gain (12-20 kHz), 1-2 dB recommended
    presence_gain_db : float
        Presence band gain (4-8 kHz), 1-1.5 dB recommended
    add_micro_reverb : bool
        Add micro-reverb for space

    Returns
    -------
    enhanced : np.ndarray
        Enhanced audio
    report : dict
        Processing report

    Examples
    --------
    >>> import numpy as np
    >>> import soundfile as sf
    >>> audio, sr = sf.read("vocal.wav")
    >>> enhanced, report = enhance_air_presence(audio, sr)
    >>> print(f"Air Boost: {report['air_boost_measured_db']:.1f} dB")
    >>> print(f"Presence Boost: {report['presence_boost_measured_db']:.1f} dB")
    >>> sf.write("vocal_enhanced.wav", enhanced, sr)
    """
    enhancer = AirPresenceEnhancer(
        air_gain_db=air_gain_db,
        presence_gain_db=presence_gain_db,
        add_micro_reverb=add_micro_reverb,
    )
    return enhancer.process(audio, sr)


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    import sys

    import soundfile as sf

    from backend.file_import import load_audio_file

    if len(sys.argv) < 2:
        logger.info("Usage: python air_presence_enhancer.py <audio_file> [Ausgabe_file]")
        logger.info("Options:")
        logger.info("  --air-gain <dB>       Air band gain (default: 1.5)")
        logger.info("  --presence-gain <dB>  Presence band gain (default: 1.0)")
        logger.info("  --no-reverb           Disable micro-reverb")
        sys.exit(1)

    # Load audio
    audio_path = sys.argv[1]
    logger.info("Processing: %s", audio_path)
    _res = load_audio_file(audio_path)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Parse options
    air_gain = 1.5
    presence_gain = 1.0
    add_reverb = True

    for i, arg in enumerate(sys.argv):
        if arg == "--air-gain" and i + 1 < len(sys.argv):
            air_gain = float(sys.argv[i + 1])
        elif arg == "--presence-gain" and i + 1 < len(sys.argv):
            presence_gain = float(sys.argv[i + 1])
        elif arg == "--no-reverb":
            add_reverb = False

    # Enhance
    logger.info("")
    logger.info("🎵 ENHANCING:")
    logger.info("  Air Gain: %.1f dB @ 12 kHz", air_gain)
    logger.info("  Presence Gain: %.1f dB @ 5.5 kHz", presence_gain)
    logger.info("  Micro-Reverb: %s", "Enabled" if add_reverb else "Disabled")

    enhanced, report = enhance_air_presence(
        audio, sr, air_gain_db=air_gain, presence_gain_db=presence_gain, add_micro_reverb=add_reverb
    )

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("AIR & PRESENCE ENHANCEMENT REPORT")
    logger.info("=" * 70)
    logger.info("Air Boost (Measured):       %+.2f dB", report["air_boost_measured_db"])
    logger.info("Presence Boost (Measured):  %+.2f dB", report["presence_boost_measured_db"])
    logger.info("Overall Gain:               %+.2f dB", report["overall_gain_db"])
    logger.info("Micro-Reverb Mix:           %.1f%%", report["micro_reverb_mix"] * 100)
    logger.info("=" * 70)

    # Save
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        output_path = sys.argv[2]
    else:
        output_path = audio_path.replace(".wav", "_air_presence.wav")

    sf.write(output_path, enhanced, sr)
    logger.info("")
    logger.info("✅ verbessert audio gespeichert: %s", output_path)
