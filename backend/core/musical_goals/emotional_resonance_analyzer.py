"""
Emotional Resonance Analyzer - Misst & Enhanced emotionale Qualität von Audio

Dieser Analyzer erweitert die vorhandene EmotionalitaetMetric mit:
1. **Vocal Warmth**: Energy im 200-800 Hz Band (Vocals)
2. **Dynamic Expression**: Mikrodynamik + Makrodynamik kombiniert
3. **Harmonic Richness**: Even Harmonics (Wärme) vs. Odd Harmonics (Härte)
4. **Temporal Flow**: Smooth vs. Choppy (Musikfluss)
5. **Air & Presence**: High-Frequency Detail (12-20 kHz)

**Wissenschaftliche Grundlage:**
- Vocal Warmth: Fundamental Frequencies von Vocals (männlich: 85-180 Hz, weiblich: 165-255 Hz)
- Expression: Psychoakustische "Lebendigkeit" durch Dynamics
- Harmonic Richness: Even Harmonics (2f, 4f) = Pleasant, Odd (3f, 5f) = Harsh
- Temporal Flow: Variationskoeffizient der Frame-Energie
- Air: Perception von "Openness" & "Airiness" @ High-Frequencies

**References:**
- Juslin, P. & Laukka, P. (2003). "Communication of emotions in vocal expression"
- Rossing, T. (2007). "Springer Handbook of Acoustics" (Harmonic Analysis)
- Moore, B. C. J. (2012). "An Introduction to the Psychology of Hearing"

**Enhancement Strategy:**
- Wenn Warmth niedrig → +2 dB @ 400 Hz (subtle Low-Mid Boost)
- Wenn Richness niedrig → Tube-Style Saturation (Even Harmonics)
- Wenn Air niedrig → +1 dB @ 12 kHz (High-Shelf)
- Wenn Expression niedrig → Gentle Expansion (Mikrodynamik bewahren)

**Author:** GitHub Copilot @ Claude Sonnet 4.5
**Date:** 13. Februar 2026
"""

import logging
from dataclasses import dataclass
from typing import cast

from backend.core.audio_utils import safe_filtfilt  # §v10.101 padlen-guard

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


@dataclass
class EmotionalResonanceAnalysis:
    """Analyse-Report der emotionalen Resonanz."""

    vocal_warmth: float  # 0.0-1.0
    dynamic_expression: float  # 0.0-1.0
    harmonic_richness: float  # 0.0-1.0
    temporal_flow: float  # 0.0-1.0
    air_presence: float  # 0.0-1.0
    emotional_resonance_score: float  # 0.0-1.0 (Weighted Combination)


@dataclass
class EmotionalEnhancementReport:
    """Report des Emotional Enhancement"""

    warmth_boost_db: float
    harmonic_saturation_gain: float
    air_boost_db: float
    expansion_applied: bool
    resonance_improvement: float  # Delta Score (Before → After)


class EmotionalResonanceAnalyzer:
    """
    Misst emotionale Resonanz von Audio mit 5 Faktoren

    **Score Calculation:**
    ```
    Emotional Resonance =
        0.30 * Vocal Warmth +
        0.25 * Dynamic Expression +
        0.20 * Harmonic Richness +
        0.15 * Temporal Flow +
        0.10 * Air & Presence
    ```

    **Thresholds (für "High Emotional Resonance"):**
    - Vocal Warmth: ≥ 0.70
    - Dynamic Expression: ≥ 0.75
    - Harmonic Richness: ≥ 0.60
    - Temporal Flow: ≥ 0.65
    - Air & Presence: ≥ 0.70
    - **Total Score**: ≥ 0.70
    """

    def __init__(self, threshold: float = 0.70):
        """
        Args:
            threshold: Minimum score für "High Emotional Resonance" (0.70 typisch)
        """
        self.threshold = threshold

    def analyze(self, audio: np.ndarray, sr: int) -> EmotionalResonanceAnalysis:
        """
        Analysiert emotionale Resonanz.

        Args:
            audio: Input audio (mono oder stereo)
            sr: Sample Rate

        Returns:
            EmotionalResonanceAnalysis
        """
        # Convert to mono for analysis
        audio_mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio

        # Measure 5 factors
        vocal_warmth = self._measure_vocal_warmth(audio_mono, sr)
        dynamic_expression = self._measure_dynamic_expression(audio_mono, sr)
        harmonic_richness = self._measure_harmonic_richness(audio_mono, sr)
        temporal_flow = self._measure_temporal_flow(audio_mono, sr)
        air_presence = self._measure_air_presence(audio_mono, sr)

        # Weighted combination
        emotional_score = (
            0.30 * vocal_warmth
            + 0.25 * dynamic_expression
            + 0.20 * harmonic_richness
            + 0.15 * temporal_flow
            + 0.10 * air_presence
        )

        return EmotionalResonanceAnalysis(
            vocal_warmth=vocal_warmth,
            dynamic_expression=dynamic_expression,
            harmonic_richness=harmonic_richness,
            temporal_flow=temporal_flow,
            air_presence=air_presence,
            emotional_resonance_score=emotional_score,
        )

    def _measure_vocal_warmth(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Vocal Warmth (200-800 Hz Energy)

        **Rationale:**
        - Fundamentals von männlichen Vocals: 85-180 Hz
        - Fundamentals von weiblichen Vocals: 165-255 Hz
        - Formants (Vokalklang): 200-800 Hz

        Hohe Energy im 200-800 Hz Band = Warm, Rich, Intimate
        """
        # FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Extract 200-800 Hz band
        mask_warmth = (freqs >= 200) & (freqs <= 800)
        warmth_energy: float = float(np.sum(magnitude[mask_warmth]))

        # Total energy
        total_energy: float = float(np.sum(magnitude))

        # Ratio
        warmth_ratio = warmth_energy / (total_energy + 1e-10)

        # Normalize: Typical range 0.05-0.20, clip to 0-1
        warmth_score = min(1.0, warmth_ratio / 0.15)

        return float(np.clip(warmth_score, 0.0, 1.0))

    def _measure_dynamic_expression(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Dynamic Expression (Lebendigkeit)

        Kombiniert:
        1. **Makrodynamik**: Peak-to-RMS Ratio (whole file)
        2. **Mikrodynamik**: Frame-by-Frame Variance (50ms windows)

        Hohe Dynamik = High Expression
        """
        # === 1. Makrodynamik (Peak-to-RMS) ===
        peak: float = float(np.max(np.abs(audio)))
        rms = np.sqrt(np.mean(audio**2))

        dynamic_range_db = 20 * np.log10(peak / rms) if rms > 0 else 0.0

        # Normalize: Typical range 6-18 dB, 12 dB = 0.5
        macro_score = min(1.0, dynamic_range_db / 18.0)

        # === 2. Mikrodynamik (Frame Variance) ===
        frame_size = int(0.050 * sr)  # 50ms
        hop_size = frame_size // 4

        frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))

        # Variationskoeffizient (CV = σ / μ)
        frame_mean = np.mean(frame_rms)
        frame_std = np.std(frame_rms)

        cv = frame_std / frame_mean if frame_mean > 0 else 0.0

        # Normalize: Typical CV 0.1-0.6, 0.4 = 0.5
        micro_score = min(1.0, cv / 0.6)

        # === Weighted Combination ===
        expression_score = 0.4 * macro_score + 0.6 * micro_score

        return float(np.clip(expression_score, 0.0, 1.0))

    def _measure_harmonic_richness(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Harmonic Richness (Even vs. Odd Harmonics)

        **Principle:**
        - Even Harmonics (2f, 4f, 6f): Musical, Warm, Pleasant
        - Odd Harmonics (3f, 5f, 7f): Harsh, Metallic

        Score = (Even Power * 1.5 - Odd Power * 0.5) / Total Power

        High Even/Low Odd = High Richness
        """
        # FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Find fundamental (dominant low-frequency peak)
        # Limit search to 50-500 Hz (typical fundamental range)
        mask_fundamental = (freqs >= 50) & (freqs <= 500)

        if np.sum(mask_fundamental) == 0:
            return 0.5  # Neutral if no fundamental

        f0_idx = np.argmax(magnitude[mask_fundamental])
        f0_idx += np.where(mask_fundamental)[0][0]  # Absolute index
        f0 = freqs[f0_idx]

        # Measure harmonics (2nd-7th)
        even_power = 0.0
        odd_power = 0.0
        total_harmonic_power = 0.0

        for n in range(2, 8):  # 2nd to 7th harmonic
            harmonic_freq = f0 * n

            # Find closest FFT bin
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))

            if harmonic_idx < len(magnitude):
                power = magnitude[harmonic_idx] ** 2
                total_harmonic_power += power

                if n % 2 == 0:
                    even_power += power
                else:
                    odd_power += power

        # Score calculation
        if total_harmonic_power > 0:
            richness_score = (even_power * 1.5 - odd_power * 0.5) / total_harmonic_power
            richness_score = (richness_score + 1.0) / 2.0  # Normalize to 0-1
        else:
            richness_score = 0.5  # Neutral

        return float(np.clip(richness_score, 0.0, 1.0))

    def _measure_temporal_flow(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Temporal Flow (Smooth vs. Choppy)

        **Principle:**
        - Smooth Flow: Gradual Energy changes, natural progression
        - Choppy Flow: Abrupt changes, disrupted flow

        Measurement: Spectral Flux (Frame-to-Frame Spectral Change)
        Low Flux = Smooth = High Score
        """
        # Short-Time Fourier Transform
        hop_length = 512
        n_fft = 2048

        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

        # Spectral Flux (Frame-to-Frame difference)
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))

        # Mean flux
        mean_flux = np.mean(flux)

        # Normalize: Low flux = High flow
        # Typical range: 0.01-0.10
        flow_score = 1.0 - min(1.0, mean_flux / 0.10)  # type: ignore[operator]

        return float(np.clip(flow_score, 0.0, 1.0))

    def _measure_air_presence(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst Air & Presence (12-20 kHz Energy)

        **Principle:**
        - "Air": Perception von Openness, Space
        - High-Frequency Detail @ 12-20 kHz

        High Energy @ HF = High Air
        """
        # FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Extract 12-20 kHz band
        mask_air = (freqs >= 12000) & (freqs <= 20000)

        if np.sum(mask_air) == 0:
            return 0.0  # No HF content

        air_energy: float = float(np.sum(magnitude[mask_air]))

        # Total energy
        total_energy: float = float(np.sum(magnitude))

        # Ratio
        air_ratio = air_energy / (total_energy + 1e-10)

        # Normalize: Typical range 0.001-0.01
        air_score = min(1.0, air_ratio / 0.01)

        return float(np.clip(air_score, 0.0, 1.0))


class EmotionalResonanceEnhancer:
    """
    Enhanced emotionale Resonanz durch subtle Processing

    **Enhancement Strategy:**
    1. Low Warmth → +2 dB @ 400 Hz (Vocal Boost)
    2. Low Richness → Tube-Style Saturation (Even Harmonics)
    3. Low Air → +1 dB @ 12 kHz (High-Shelf)
    4. Low Expression → Gentle Expansion (Preserve Mikrodynamik)
    """

    def __init__(
        self,
        warmth_boost_db: float = 2.0,
        harmonic_saturation_gain: float = 0.15,
        air_boost_db: float = 1.0,
        expansion_threshold: float = -40.0,
        expansion_ratio: float = 1.2,
    ) -> None:
        """
        Args:
            warmth_boost_db: Vocal warmth boost (400 Hz Bell)
            harmonic_saturation_gain: Tube saturation intensity (0.0-0.3)
            air_boost_db: High-shelf boost @ 12 kHz
            expansion_threshold: Expansion threshold dB
            expansion_ratio: 1:ratio (1.2 = gentle)
        """
        self.warmth_boost_db = warmth_boost_db
        self.harmonic_saturation_gain = harmonic_saturation_gain
        self.air_boost_db = air_boost_db
        self.expansion_threshold = expansion_threshold
        self.expansion_ratio = expansion_ratio

    def enhance(
        self, audio: np.ndarray, sr: int, current_analysis: EmotionalResonanceAnalysis
    ) -> tuple[np.ndarray, EmotionalEnhancementReport]:
        """
        Wendet Emotional Enhancement an (adaptive basierend auf Analysis)

        Args:
            audio: Input audio (mono oder stereo)
            sr: Sample Rate
            current_analysis: Aktuelle Emotional Resonance Analysis

        Returns:
            (enhanced_audio, report)
        """
        enhanced = audio.copy()

        # Flags
        warmth_applied = False
        saturation_applied = False
        air_applied = False
        expansion_applied = False

        actual_warmth_boost = 0.0
        actual_saturation_gain = 0.0
        actual_air_boost = 0.0

        # === 1. Vocal Warmth Enhancement ===
        if current_analysis.vocal_warmth < 0.70:
            # Bell Filter @ 400 Hz
            warmth_needed = (0.70 - current_analysis.vocal_warmth) * self.warmth_boost_db
            actual_warmth_boost = min(warmth_needed, self.warmth_boost_db)

            enhanced = self._apply_bell_filter(enhanced, sr, center_freq=400, gain_db=actual_warmth_boost, q=1.0)
            warmth_applied = True

        # === 2. Harmonic Richness Enhancement ===
        if current_analysis.harmonic_richness < 0.60:
            # Tube-Style Saturation
            richness_needed = (0.60 - current_analysis.harmonic_richness) * self.harmonic_saturation_gain
            actual_saturation_gain = min(richness_needed, self.harmonic_saturation_gain)

            enhanced = self._apply_tube_saturation(enhanced, gain=actual_saturation_gain)
            saturation_applied = True

        # === 3. Air & Presence Enhancement ===
        if current_analysis.air_presence < 0.70:
            # High-Shelf @ 12 kHz
            air_needed = (0.70 - current_analysis.air_presence) * self.air_boost_db
            actual_air_boost = min(air_needed, self.air_boost_db)

            enhanced = self._apply_high_shelf(enhanced, sr, cutoff_freq=12000, gain_db=actual_air_boost)
            air_applied = True

        # === 4. Dynamic Expression Enhancement ===
        if current_analysis.dynamic_expression < 0.75:
            # Gentle Expansion (bewahrt Mikrodynamik)
            enhanced = self._apply_expansion(
                enhanced, threshold_db=self.expansion_threshold, ratio=self.expansion_ratio
            )
            expansion_applied = True

        # Normalize
        peak: float = float(np.max(np.abs(enhanced)))
        if peak > 0:
            enhanced = enhanced / peak

        # === REPORT — Re-Analyse für Resonanz-Improvement ===
        analyzer = EmotionalResonanceAnalyzer(threshold=self.warmth_boost_db)
        post_analysis = analyzer.analyze(enhanced, sr)
        resonance_improvement = float(
            post_analysis.emotional_resonance_score - current_analysis.emotional_resonance_score
        )

        report = EmotionalEnhancementReport(
            warmth_boost_db=actual_warmth_boost if warmth_applied else 0.0,
            harmonic_saturation_gain=actual_saturation_gain if saturation_applied else 0.0,
            air_boost_db=actual_air_boost if air_applied else 0.0,
            expansion_applied=expansion_applied,
            resonance_improvement=resonance_improvement,
        )

        return enhanced, report

    def _apply_bell_filter(
        self, audio: np.ndarray, sr: int, center_freq: float, gain_db: float, q: float = 1.0
    ) -> np.ndarray:
        """
        Peaking EQ (Bell Filter)
        """
        # Design peaking filter
        nyquist = sr / 2
        freq_norm = center_freq / nyquist

        # IIR peaking filter
        b, a = signal.iirpeak(freq_norm, Q=q, fs=sr)

        # Apply gain
        if audio.ndim == 2:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered[:, ch] = safe_filtfilt(b, a, audio[:, ch])
        else:
            filtered = safe_filtfilt(b, a, audio)

        # Mix with dry (apply gain)
        gain_linear = 10 ** (gain_db / 20.0)
        enhanced = audio + (filtered - audio) * (gain_linear - 1.0)

        return enhanced  # type: ignore[no-any-return]

    def _apply_tube_saturation(self, audio: np.ndarray, gain: float = 0.15) -> np.ndarray:
        """
        Tube-Style Soft Saturation (Even Harmonics)

        Formula: tanh(x * gain) / tanh(gain)
        """
        if gain < 0.01:
            return audio

        # Soft clip with tanh
        drive = 1.0 + gain * 3.0  # Scale gain
        saturated = np.tanh(audio * drive) / np.tanh(drive)

        # Mix: 85% dry + 15% saturated
        mix = 0.15
        enhanced = (1 - mix) * audio + mix * saturated

        return enhanced  # type: ignore[no-any-return]

    def _apply_high_shelf(self, audio: np.ndarray, sr: int, cutoff_freq: float, gain_db: float) -> np.ndarray:
        """
        High-Shelf Filter (Boost/Cut High-Frequencies)
        """
        # Design shelving filter
        nyquist = sr / 2
        cutoff_norm = cutoff_freq / nyquist
        cutoff_norm = min(cutoff_norm, 0.99)

        # Calculate gain
        gain_linear = 10 ** (gain_db / 20.0)

        # High-shelf with scipy
        # Note: scipy doesn't have direct high-shelf, use custom
        # Simple approximation: High-pass + mix
        b, a = cast(tuple[np.ndarray, np.ndarray], signal.butter(2, cutoff_norm, btype="high", output="ba"))

        if audio.ndim == 2:
            filtered = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                filtered[:, ch] = safe_filtfilt(b, a, audio[:, ch])
        else:
            filtered = safe_filtfilt(b, a, audio)

        # Mix to apply gain
        enhanced = audio + (filtered * (gain_linear - 1.0))

        return enhanced  # type: ignore[no-any-return]

    def _apply_expansion(self, audio: np.ndarray, threshold_db: float = -40.0, ratio: float = 1.2) -> np.ndarray:
        """
        Gentle Expander (Preserve Mikrodynamik)

        Formula:
        if level < threshold:
            output = input * ratio
        else:
            output = input
        """
        # Convert to dB
        threshold_linear = 10 ** (threshold_db / 20.0)

        # Simplified frame-based expansion
        frame_size = 2048
        hop_size = frame_size // 2

        # Process per channel
        if audio.ndim == 2:
            expanded = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                channel = audio[:, ch]

                # Frame
                frames = librosa.util.frame(channel, frame_length=frame_size, hop_length=hop_size)
                frame_rms = np.sqrt(np.mean(frames**2, axis=0))

                # Expansion gain
                expansion_gain = np.ones_like(frame_rms)
                below_threshold = frame_rms < threshold_linear
                expansion_gain[below_threshold] = ratio

                # Apply gain per frame
                expanded_channel = np.zeros_like(channel)
                window_sum = np.zeros_like(channel)

                for i, gain in enumerate(expansion_gain):
                    start = i * hop_size
                    end = min(start + frame_size, len(channel))

                    expanded_channel[start:end] += frames[: end - start, i] * gain
                    window_sum[start:end] += 1.0

                # Normalize overlap-add
                window_sum[window_sum == 0] = 1.0
                expanded_channel /= window_sum

                expanded[:, ch] = expanded_channel
        else:
            # Mono
            frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size)
            frame_rms = np.sqrt(np.mean(frames**2, axis=0))

            expansion_gain = np.ones_like(frame_rms)
            below_threshold = frame_rms < threshold_linear
            expansion_gain[below_threshold] = ratio

            expanded = np.zeros_like(audio)
            window_sum = np.zeros_like(audio)

            for i, gain in enumerate(expansion_gain):
                start = i * hop_size
                end = min(start + frame_size, len(audio))

                expanded[start:end] += frames[: end - start, i] * gain
                window_sum[start:end] += 1.0

            window_sum[window_sum == 0] = 1.0
            expanded /= window_sum

        return expanded  # type: ignore[no-any-return]


# === CONVENIENCE FUNCTION ===
def analyze_and_enhance_emotional_resonance(
    audio: np.ndarray, sr: int, threshold: float = 0.70
) -> tuple[np.ndarray, EmotionalResonanceAnalysis, EmotionalEnhancementReport]:
    """
    Convenience function: Analyze + Enhance Emotional Resonance

    Args:
        audio: Input audio
        sr: Sample Rate
        threshold: Threshold für "High Emotional Resonance"

    Returns:
        (enhanced_audio, analysis, enhancement_report)

    Example:
        >>> enhanced, analysis, report = analyze_and_enhance_emotional_resonance(audio, sr)
        logger.debug("Emotional Resonance: %.1%", analysis.emotional_resonance_score)
        logger.debug("Warmth Boost: %.1f dB", report.warmth_boost_db)
    """
    # Analyze
    analyzer = EmotionalResonanceAnalyzer(threshold=threshold)
    analysis = analyzer.analyze(audio, sr)

    # Enhance (nur wenn unter Threshold)
    if analysis.emotional_resonance_score < threshold:
        enhancer = EmotionalResonanceEnhancer()
        enhanced, report = enhancer.enhance(audio, sr, analysis)
    else:
        # Already good, skip enhancement
        enhanced = audio
        report = EmotionalEnhancementReport(
            warmth_boost_db=0.0,
            harmonic_saturation_gain=0.0,
            air_boost_db=0.0,
            expansion_applied=False,
            resonance_improvement=0.0,
        )

    return enhanced, analysis, report


if __name__ == "__main__":
    # === DEMO / UNIT TEST ===
    logger.debug("💎 Emotional Resonance Analyzer & Enhancer - Demo")
    logger.debug("=" * 60)

    # Generate test signal (sine wave + noise)
    sr = 48000
    duration = 3.0
    samples = int(duration * sr)

    # Base signal: 220 Hz sine (A3 note) + harmonics
    t = np.linspace(0, duration, samples)
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # Fundamental
    audio += 0.15 * np.sin(2 * np.pi * 440 * t)  # 2nd harmonic (even)
    audio += 0.05 * np.sin(2 * np.pi * 660 * t)  # 3rd harmonic (odd)

    # Add subtle noise
    audio += 0.02 * np.random.randn(samples)

    # Make stereo
    audio = np.stack([audio, audio], axis=1)

    logger.debug("Eingabe: %s samples, %s channels, %s Hz", samples, audio.shape[1], sr)
    logger.debug("Duration: %.1f seconds", duration)
    logger.debug("Signal: A3 (220 Hz) + Harmonics")

    # Analyze & Enhance
    enhanced, analysis, report = analyze_and_enhance_emotional_resonance(audio, sr)

    logger.debug("\n✅ Emotional Resonance Analyse:")
    logger.debug("  • Vocal Warmth: %.1f", analysis.vocal_warmth)
    logger.debug("  • Dynamic Expression: %.1f", analysis.dynamic_expression)
    logger.debug("  • Harmonic Richness: %.1f", analysis.harmonic_richness)
    logger.debug("  • Temporal Flow: %.1f", analysis.temporal_flow)
    logger.debug("  • Air & Presence: %.1f", analysis.air_presence)
    logger.debug("  • Overall Wert: %.1f", analysis.emotional_resonance_score)

    logger.debug("\n✨ Emotional Enhancement angewendet:")
    logger.debug("  • Warmth Boost: %.1f dB @ 400 Hz", report.warmth_boost_db)
    logger.debug("  • Harmonic Saturation: %.1f", report.harmonic_saturation_gain)
    logger.debug("  • Air Boost: %.1f dB @ 12 kHz", report.air_boost_db)
    logger.debug("  • Expansion angewendet: %s", report.expansion_applied)

    logger.debug("\n💎 Emotionale Resonanz optimiert!")
