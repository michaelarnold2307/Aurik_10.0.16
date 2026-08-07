import logging

logger = logging.getLogger(__name__)

"""
Soundstage Depth Enhancer - Erzeugt räumliche Tiefe durch Early Reflections & Reverb Tail Shaping

Dieser Enhancer erzeugt eine dreidimensionale Soundstage durch:
1. **Foreground** (0-10ms): Direkt-Sound, keine Processing
2. **Midground** (10-50ms): Early Reflections für Raumgröße
3. **Background** (50-300ms): Diffuse Reverb für Tiefe & Ambience

**Wissenschaftliche Grundlage:**
- ITD (Interaural Time Difference): 0-700 μs für Stereo-Width
- Early Reflections (10-80ms): Wahrnehmung von Raumgröße
- Late Reverb (80-300ms): Wahrnehmung von Raum-Tiefe
- HF Damping: Simulation von Luftabsorption (Distance Cue)

**References:**
- Begault, D. R. (1994). "3-D Sound for Virtual Reality and Multimedia"
- Blauert, J. (1997). "Spatial Hearing: The Psychophysics of Human Sound Localization"
- Zahorik, P. (2002). "Assessing auditory distance perception using virtual acoustics"

**Author:** GitHub Copilot @ Claude Sonnet 4.5
**Date:** 13. Februar 2026
"""

from dataclasses import dataclass

import numpy as np
import scipy.signal as signal


@dataclass
class SoundstageDepthReport:
    """Report der Soundstage Depth Enhancement"""

    foreground_level: float  # 0.0-1.0
    midground_level: float
    background_level: float
    early_reflections_delay_ms: float
    reverb_rt60_seconds: float
    hf_damping_hz: float
    depth_score: float  # 0.0-1.0 (Vorher → Nachher)


class SoundstageDepthEnhancer:
    """
    Erzeugt räumliche Tiefe durch Multi-Layer Spatial Processing

    **Conceptual Model:**
    ```
    FOREGROUND (Direct Sound)
        ↓
    MIDGROUND (Early Reflections: 10-50ms)
        ↓
    BACKGROUND (Diffuse Reverb: 50-300ms, HF Damped)
    ```

    **Parameters:**
    - depth_amount: 0.0-1.0 (0.5 = Subtle, 1.0 = Full 3D Effect)
    - room_size: 0.0-1.0 (0.3 = Small, 0.7 = Large Concert Hall)
    - hf_damping_hz: High-frequency damping für Distance Cues (default: 8000 Hz)
    """

    def __init__(
        self,
        depth_amount: float = 0.5,
        room_size: float = 0.5,
        hf_damping_hz: float = 8000.0,
        early_reflections_delay_ms: float = 15.0,
        reverb_rt60: float = 0.3,
        foreground_level: float = 0.70,
        midground_level: float = 0.20,
        background_level: float = 0.10,
    ):
        """
        Args:
            depth_amount: Stärke des Depth-Effekts (0.0-1.0)
            room_size: Raumgröße (0.0-1.0), beeinflusst RT60 & Delay
            hf_damping_hz: Cutoff-Frequenz für HF Damping (simuliert Luftabsorption)
            early_reflections_delay_ms: Delay der ersten Reflections (10-50ms typisch)
            reverb_rt60: Reverb Time (RT60 in Sekunden)
            foreground_level: Mix-Level für Direkt-Sound (0.0-1.0)
            midground_level: Mix-Level für Early Reflections (0.0-1.0)
            background_level: Mix-Level für Diffuse Reverb (0.0-1.0)
        """
        self.depth_amount = np.clip(depth_amount, 0.0, 1.0)
        self.room_size = np.clip(room_size, 0.0, 1.0)
        self.hf_damping_hz = hf_damping_hz
        self.early_reflections_delay_ms = early_reflections_delay_ms
        self.reverb_rt60 = reverb_rt60

        # Mix Levels (normalisiert zu 1.0 total)
        total = foreground_level + midground_level + background_level
        self.foreground_level = foreground_level / total
        self.midground_level = midground_level / total
        self.background_level = background_level / total

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, SoundstageDepthReport]:
        """
        Wendet Soundstage Depth Enhancement an

        Args:
            audio: Input audio (mono oder stereo)
            sr: Sample Rate

        Returns:
            (enhanced_audio, report)
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        # Sicherstellen, dass Audio stereo ist
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)
        elif audio.ndim == 2 and audio.shape[0] == 2:
            audio = audio.T  # (2, N) → (N, 2)

        # === LAYER 1: FOREGROUND (Direct Sound) ===
        foreground = audio.copy()

        # === LAYER 2: MIDGROUND (Early Reflections) ===
        midground = self._apply_early_reflections(audio, sr)

        # === LAYER 3: BACKGROUND (Diffuse Reverb) ===
        background = self._apply_diffuse_reverb(audio, sr)

        # === MIX mit Depth Amount ===
        enhanced = (
            self.foreground_level * foreground
            + self.midground_level * midground * self.depth_amount
            + self.background_level * background * self.depth_amount
        )

        # NaN/Inf-Guard
        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalisieren (Peak = 1.0)
        peak = np.max(np.abs(enhanced))
        if peak > 0:
            enhanced = enhanced / peak

        # Final clip
        enhanced = np.clip(enhanced, -1.0, 1.0)

        # === REPORT ===
        depth_score_before = self._measure_depth_score(audio, sr)
        depth_score_after = self._measure_depth_score(enhanced, sr)

        report = SoundstageDepthReport(
            foreground_level=self.foreground_level,
            midground_level=self.midground_level,
            background_level=self.background_level,
            early_reflections_delay_ms=self.early_reflections_delay_ms,
            reverb_rt60_seconds=self.reverb_rt60,
            hf_damping_hz=self.hf_damping_hz,
            depth_score=(depth_score_after - depth_score_before),
        )

        return enhanced, report

    def _apply_early_reflections(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Erzeugt Early Reflections für Midground-Layer

        Early Reflections Pattern (nach Barron, 2005):
        - First Reflection: 10-15ms (Rückwand)
        - Second Reflection: 20-25ms (Seitenwände)
        - Third Reflection: 30-40ms (Boden/Decke)

        Jede Reflection hat:
        - Delay
        - Attenuation (-6 dB pro Reflection)
        - Stereo Offset (für Räumlichkeit)
        """
        # Base delay in samples
        delay_samples = int(self.early_reflections_delay_ms * sr / 1000.0)

        # Pattern: 3 Reflections mit unterschiedlichen Delays & Gains
        reflections = [
            (1.0, 0.0),  # First: center
            (1.5, -0.2),  # Second: leicht links (-0.2 = 20% Pan)
            (2.0, +0.2),  # Third: leicht rechts
        ]

        output = np.zeros_like(audio)

        for delay_mult, pan_offset in reflections:
            delay = int(delay_samples * delay_mult)
            gain = 0.5**delay_mult  # -6 dB pro Reflection

            # Delay Audio
            delayed = np.roll(audio, delay, axis=0)
            delayed[:delay] = 0  # Zero out wrapped samples

            # Apply Pan (Stereo Width)
            if audio.shape[1] == 2:
                pan_left = 0.5 - pan_offset / 2
                pan_right = 0.5 + pan_offset / 2
                delayed[:, 0] *= pan_left
                delayed[:, 1] *= pan_right

            output += delayed * gain

        # Low-pass filter (Reflections verlieren High-Frequencies)
        # Cutoff = 12 kHz (natürliche HF Absorption)
        output = self._apply_lowpass(output, cutoff_hz=12000, sr=sr)

        # NaN/Inf-Guard
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        return output

    def _apply_diffuse_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Erzeugt Diffuse Reverb für Background-Layer

        **Algorithm:** Schroeder Reverberator (4 Allpass + 4 Comb Filters)

        **Psychoacoustic Properties:**
        - RT60: Room Size (0.1s = Small, 0.5s = Large)
        - HF Damping: Distance Cue (fernen Sounds verlieren High-Frequencies)
        - Diffusion: Smooth Reverb Tail (keine Echo-Artefakte)

        **References:**
        - Schroeder, M. R. (1962). "Natural Sounding Artificial Reverberation"
        - Jot, J. M. (1992). "Digital Delay Networks for Designing Artificial Reverberators"
        """
        # RT60 adaptiert an Room Size
        rt60 = self.reverb_rt60 * (0.5 + 0.5 * self.room_size)

        # Diffuse Reverb = Multiple Allpass + Comb Filters
        output = audio.copy()

        # Allpass Filters (für Diffusion)
        allpass_delays = [347, 113, 37, 59]  # Prime numbers (minimiert Artefakte)
        for delay_samples in allpass_delays:
            output = self._allpass_filter(output, delay_samples, gain=0.5)

        # Comb Filters (für Reverb Tail)
        comb_delays = [1687, 1601, 2053, 2251]  # Prime numbers
        comb_output = np.zeros_like(audio)

        for delay_samples in comb_delays:
            # Calculate feedback gain für gewünschten RT60
            # Formula: g = 10^(-3 * delay_seconds / RT60)
            delay_seconds = delay_samples / sr
            feedback_gain = 10 ** (-3 * delay_seconds / rt60)
            feedback_gain = min(feedback_gain, 0.95)  # Stability

            comb_output += self._comb_filter(output, delay_samples, feedback_gain)

        output = comb_output / len(comb_delays)

        # HF Damping (simuliert Luftabsorption)
        output = self._apply_lowpass(output, cutoff_hz=self.hf_damping_hz, sr=sr)

        # NaN/Inf-Guard
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        return output

    def _allpass_filter(self, audio: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
        """
        Allpass Filter für Reverb Diffusion

        Transfer Function: H(z) = (g + z^-M) / (1 + g*z^-M)
        """
        if delay_samples <= 0:
            return np.asarray(audio, dtype=np.float64)

        x = np.asarray(audio, dtype=np.float64)
        b = np.zeros(delay_samples + 1, dtype=np.float64)
        a = np.zeros(delay_samples + 1, dtype=np.float64)
        # H(z) = (g + z^-M) / (1 + g z^-M)
        b[0] = gain
        b[-1] = 1.0
        a[0] = 1.0
        a[-1] = gain

        if x.ndim == 2:
            y = np.empty_like(x)
            for ch in range(x.shape[1]):
                y[:, ch] = signal.lfilter(b, a, x[:, ch])
            return y
        return signal.lfilter(b, a, x)  # type: ignore[no-any-return]

    def _comb_filter(self, audio: np.ndarray, delay_samples: int, feedback_gain: float) -> np.ndarray:
        """
        Comb Filter für Reverb Tail

        Transfer Function: H(z) = 1 / (1 - g*z^-M)
        """
        if delay_samples <= 0:
            return np.asarray(audio, dtype=np.float64)

        x = np.asarray(audio, dtype=np.float64)
        b = np.array([1.0], dtype=np.float64)
        a = np.zeros(delay_samples + 1, dtype=np.float64)
        # H(z) = 1 / (1 - g z^-M)
        a[0] = 1.0
        a[-1] = -feedback_gain

        if x.ndim == 2:
            y = np.empty_like(x)
            for ch in range(x.shape[1]):
                y[:, ch] = signal.lfilter(b, a, x[:, ch])
            return y
        return signal.lfilter(b, a, x)  # type: ignore[no-any-return]

    def _apply_lowpass(self, audio: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
        """
        Low-pass Butterworth Filter (2nd Order)
        """
        nyquist = sr / 2
        cutoff_norm = cutoff_hz / nyquist
        cutoff_norm = min(cutoff_norm, 0.99)  # Stability

        sos = np.asarray(signal.butter(2, cutoff_norm, btype="low", output="sos"), dtype=np.float64)

        # Apply per channel
        if audio.ndim == 2:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered[:, ch] = signal.sosfiltfilt(sos, audio[:, ch])
            return filtered
        else:
            return signal.sosfiltfilt(sos, audio)  # type: ignore[no-any-return]

    def _measure_depth_score(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst wahrgenommene räumliche Tiefe

        **Cues für Depth Perception:**
        1. **Reverb Presence** (50-300ms Energy)
        2. **HF Roll-off** (fernen Sounds fehlen High-Frequencies)
        3. **Temporal Smearing** (Reverb Tail Länge)

        Returns:
            depth_score: 0.0 = no depth, 1.0 = full 3D depth
        """
        # === 1. Reverb Presence (Energy in 50-300ms) ===
        # Autocorrelation für Reverb Detection
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        autocorr = signal.fftconvolve(mono, mono[::-1], mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Energy in 50-300ms range
        start_idx = int(0.050 * sr)
        end_idx = int(0.300 * sr)
        if end_idx > len(autocorr):
            end_idx = len(autocorr)

        reverb_energy = np.sum(np.abs(autocorr[start_idx:end_idx]))
        direct_energy = np.sum(np.abs(autocorr[:start_idx]))

        reverb_ratio = reverb_energy / (direct_energy + 1e-10)
        reverb_score = min(1.0, reverb_ratio / 0.5)  # Normalize

        # === 2. HF Roll-off (Spectral Centroid) ===
        fft = np.fft.rfft(mono)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(mono), 1 / sr)

        # Spectral Centroid
        centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)

        # Lower centroid = More HF roll-off = More depth
        # Typical: 2000 Hz = Low (distant), 8000 Hz = High (close)
        hf_score = 1.0 - min(1.0, (centroid - 2000) / 6000)

        # === 3. Weighted Combination ===
        depth_score = 0.6 * reverb_score + 0.4 * hf_score

        return float(np.clip(depth_score, 0.0, 1.0))


# === CONVENIENCE FUNCTION ===
def enhance_soundstage_depth(
    audio: np.ndarray, sr: int, depth_amount: float = 0.5, room_size: float = 0.5
) -> tuple[np.ndarray, SoundstageDepthReport]:
    """
    Convenience function für Soundstage Depth Enhancement

    Args:
        audio: Input audio (mono oder stereo)
        sr: Sample Rate
        depth_amount: 0.0-1.0 (0.3 = Subtle, 0.7 = Dramatic)
        room_size: 0.0-1.0 (0.3 = Small Room, 0.7 = Concert Hall)

    Returns:
        (enhanced_audio, report)

    Example:
        >>> enhanced, report = enhance_soundstage_depth(audio, sr, depth_amount=0.6)
        >>> logger.info(f"Depth improved by: {report.depth_score:.1%}")
    """
    enhancer = SoundstageDepthEnhancer(depth_amount=depth_amount, room_size=room_size)
    return enhancer.process(audio, sr)


if __name__ == "__main__":
    # === DEMO / UNIT TEST ===
    logger.info("🎵 Soundstage Depth Enhancer - Demo")
    logger.info(str("=" * 60))

    # Generate test signal (stereo white noise burst)
    sr = 48000
    duration = 2.0
    samples = int(duration * sr)

    # Stereo white noise (Mono-like: same L/R)
    audio = np.random.randn(samples, 2) * 0.1

    logger.info("Eingabe: %s samples, %s channels, %s Hz", samples, audio.shape[1], sr)
    logger.info("Duration: %.1f seconds", duration)

    # Apply enhancement
    enhanced, report = enhance_soundstage_depth(audio, sr, depth_amount=0.6, room_size=0.5)

    logger.info("\n✅ Soundstage Depth Enhancement vollstaendig!")
    logger.info("  • Foreground Level: %.1%", report.foreground_level)
    logger.info("  • Midground Level: %.1%", report.midground_level)
    logger.info("  • Background Level: %.1%", report.background_level)
    logger.info("  • Early Reflections Delay: %.1f ms", report.early_reflections_delay_ms)
    logger.info("  • Reverb RT60: %.2f seconds", report.reverb_rt60_seconds)
    logger.info("  • HF Damping Cutoff: %.0f Hz", report.hf_damping_hz)
    logger.info("  • Depth Wert Improvement: %+.2f (0-1 scale)", report.depth_score)

    logger.info("\n✨ 3D Soundstage mit räumlicher Tiefe erzeugt!")
