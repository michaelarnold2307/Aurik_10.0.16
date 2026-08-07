"""
AURIK Harmonic Character Analyzer
==================================

Analysiert harmonische Charakteristik - die "Farbe" der Harmonischen.

Die Wahrheit über Harmonische:
------------------------------
NICHT alle Harmonischen sind schlecht! THD (Total Harmonic Distortion) misst
NUR die Menge, aber nicht die QUALITÄT der Harmonischen.

**Even Harmonics (2f, 4f, 6f):**
- Musikalisch, Warm, Pleasant
- Oktav-Verwandt (konsona nt)
- Beispiel: Röhrenverstärker (Tube Saturation)
- Effekt: "Wärme", "Fülle", "Analog-Feeling"

**Odd Harmonics (3f, 5f, 7f):**
- Dissonant, Harsh, Metallic
- Nicht-Oktav (inkonsistent)
- Beispiel: Transistor-Clipping (Hard Clipping)
- Effekt: "Schärfe", "Aggressivität", "Digital-Härte"

Messung:
-------
- Separierung von Even vs. Odd Harmonics
- Harmonic Richness Score (Even = GUT, Odd = NEUTRAL/SCHLECHT)
- Optimal: 3-8% Even Harmonics, <1% Odd Harmonics

Score: 0.0 = Poor (harsh odd harmonics), 1.0 = Excellent (warm even harmonics)
Threshold: 0.75

Wissenschaftliche Grundlagen:
-----------------------------
- Katz (2014): "Mastering Audio"
- Huber & Runstein (2017): "Modern Recording Techniques"
- Colletti (2013): "The Art of Digital Audio Recording"

Autor: AURIK Phase 2.0 - Psychoakustische Exzellenz
Datum: 13. Februar 2026
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class HarmonicAnalysis:
    """Result of harmonic character analysis"""

    harmonic_richness_score: float  # 0.0-1.0 (1.0 = optimal even harmonics)
    even_harmonics_ratio: float  # Ratio of even harmonic energy
    odd_harmonics_ratio: float  # Ratio of odd harmonic energy
    total_thd: float  # Total Harmonic Distortion (for reference)
    warmth_score: float  # Even harmonics = warmth
    harshness_penalty: float  # Odd harmonics = harshness
    passed: bool  # True if score >= threshold
    details: dict[str, float]


class HarmonicCharacterAnalyzer:
    """
    Harmonic Character Analyzer

    Unterscheidet zwischen positiven (even) und negativen (odd) Harmonischen.

    Parameters
    ----------
    threshold : float
        Minimum harmonic richness score (0.75 recommended)
    n_harmonics : int
        Number of harmonics to analyze (6 default: 2f-7f)
    """

    def __init__(self, threshold: float = 0.75, n_harmonics: int = 6):
        self.threshold = threshold
        self.n_harmonics = n_harmonics

    def analyze(self, audio: np.ndarray, sr: int, return_details: bool = True) -> HarmonicAnalysis:
        """
        Analysiert den harmonischen Charakter.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal (mono or stereo)
        sr : int
            Sample rate in Hz
        return_details : bool
            If True, return detailed metrics

        Returns
        -------
        HarmonicAnalysis
            Complete harmonic analysis
        """
        # Convert to mono for analysis
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        # Normalize audio for consistent analysis (§0 Peak-Guard: 99.9th percentile)
        from backend.core.core_utils import safe_peak_amplitude

        audio_mono = audio_mono / (safe_peak_amplitude(audio_mono) + 1e-10)

        # Analyze harmonics
        (
            even_ratio,
            odd_ratio,
            total_thd,
            harmonic_details,
        ) = self._analyze_harmonics(audio_mono, sr)

        # Compute warmth score (even harmonics = good)
        # Optimal: 3-8% even harmonics
        if 0.03 <= even_ratio <= 0.08:
            warmth_score = 1.0  # Perfect amount
        elif even_ratio < 0.03:
            # Too little warmth
            warmth_score = even_ratio / 0.03
        else:
            # Too much (potential over-saturation)
            warmth_score = 1.0 - min(1.0, (even_ratio - 0.08) / 0.10)

        # Compute harshness penalty (odd harmonics = bad)
        # Optimal: <1% odd harmonics
        if odd_ratio < 0.01:
            harshness_penalty = 0.0  # No penalty
        elif odd_ratio < 0.03:
            # Acceptable but penalize slightly
            harshness_penalty = (odd_ratio - 0.01) / 0.02 * 0.3
        else:
            # High penalty for harsh odd harmonics
            harshness_penalty = 0.3 + min(0.7, (odd_ratio - 0.03) / 0.05 * 0.7)

        # Harmonic richness score
        # Formula: Warmth - Harshness Penalty
        harmonic_richness_score = warmth_score - harshness_penalty

        # Clip to [0, 1]
        harmonic_richness_score = np.clip(harmonic_richness_score, 0.0, 1.0)

        passed = harmonic_richness_score >= self.threshold

        details = {
            "even_harmonics_percentage": even_ratio * 100,
            "odd_harmonics_percentage": odd_ratio * 100,
            "total_thd_percentage": total_thd * 100,
            "warmth_contribution": warmth_score,
            "harshness_penalty": harshness_penalty,
            **harmonic_details,  # Include per-harmonic details
        }

        return HarmonicAnalysis(
            harmonic_richness_score=harmonic_richness_score,
            even_harmonics_ratio=even_ratio,
            odd_harmonics_ratio=odd_ratio,
            total_thd=total_thd,
            warmth_score=warmth_score,
            harshness_penalty=harshness_penalty,
            passed=passed,
            details=details,
        )

    def _analyze_harmonics(self, audio: np.ndarray, sr: int) -> tuple[float, float, float, dict[str, float]]:
        """
        Analysiert gerade vs. ungerade Obertöne mittels FFT.

        Returns
        -------
        even_ratio : float
            Even harmonic energy ratio
        odd_ratio : float
            Odd harmonic energy ratio
        total_thd : float
            Total Harmonic Distortion
        details : dict
            Per-harmonic details
        """
        # Compute FFT
        n_fft = 8192  # High resolution for accurate harmonic detection
        fft = np.fft.rfft(audio, n=n_fft)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)

        # Find fundamental frequency (dominant frequency in 50-2000 Hz)
        fund_mask = (freqs >= 50) & (freqs <= 2000)
        fund_idx = np.argmax(magnitude[fund_mask]) + np.where(fund_mask)[0][0]
        f0 = freqs[fund_idx]
        fund_power = magnitude[fund_idx] ** 2

        if fund_power < 1e-10:
            # No significant fundamental (likely noise or percussion)
            return 0.0, 0.0, 0.0, {}

        # Detect harmonics (2f, 3f, 4f, ..., 7f)
        harmonic_powers = []
        for n in range(2, self.n_harmonics + 2):  # 2nd to (n+1)th harmonic
            harmonic_freq = f0 * n

            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(freqs - harmonic_freq))

            # Search in neighborhood (±5% tolerance)
            search_range = int(0.05 * freq_idx)
            start_idx = max(0, freq_idx - search_range)
            end_idx = min(len(freqs), freq_idx + search_range + 1)

            # Find peak in search range
            if end_idx > start_idx:  # type: ignore[operator]
                local_peak_idx = np.argmax(magnitude[start_idx:end_idx]) + start_idx  # type: ignore[call-overload]
                harmonic_power = magnitude[local_peak_idx] ** 2
            else:
                harmonic_power = 0.0

            harmonic_powers.append(harmonic_power)

        # Separate even and odd harmonics
        even_harmonics = [harmonic_powers[i] for i in range(0, len(harmonic_powers), 2)]  # 2f, 4f, 6f
        odd_harmonics = [harmonic_powers[i] for i in range(1, len(harmonic_powers), 2)]  # 3f, 5f, 7f

        even_power: float = float(np.sum(even_harmonics))
        odd_power: float = float(np.sum(odd_harmonics))
        total_harmonic_power = even_power + odd_power

        # Compute ratios (relative to fundamental)
        even_ratio = even_power / (fund_power + 1e-10)
        odd_ratio = odd_power / (fund_power + 1e-10)
        total_thd = np.sqrt(total_harmonic_power / (fund_power + 1e-10))

        # Per-harmonic details
        details = {
            "fundamental_freq_hz": f0,
            "fundamental_power": fund_power,
            "2nd_harmonic_power": harmonic_powers[0] if len(harmonic_powers) > 0 else 0.0,
            "3rd_harmonic_power": harmonic_powers[1] if len(harmonic_powers) > 1 else 0.0,
            "4th_harmonic_power": harmonic_powers[2] if len(harmonic_powers) > 2 else 0.0,
            "5th_harmonic_power": harmonic_powers[3] if len(harmonic_powers) > 3 else 0.0,
            "6th_harmonic_power": harmonic_powers[4] if len(harmonic_powers) > 4 else 0.0,
            "7th_harmonic_power": harmonic_powers[5] if len(harmonic_powers) > 5 else 0.0,
        }

        return even_ratio, odd_ratio, total_thd, details

    def check_preservation(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[bool, float, dict[str, float]]:
        """
        Prüft if processing improves/maintains harmonic character.

        Parameters
        ----------
        original : np.ndarray
            Original audio
        processed : np.ndarray
            Processed audio
        sr : int
            Sample rate

        Returns
        -------
        passed : bool
            True if harmonic character improved/maintained
        improvement : float
            Harmonic richness improvement (positive = better)
        details : dict
            Detailed comparison
        """
        orig_analysis = self.analyze(original, sr, return_details=True)
        proc_analysis = self.analyze(processed, sr, return_details=True)

        improvement = proc_analysis.harmonic_richness_score - orig_analysis.harmonic_richness_score

        # Processing should improve harmonic character (or keep it same)
        # Allow small degradation (0.05) for edge cases
        passed = improvement >= -0.05

        details = {
            "original_harmonic_score": orig_analysis.harmonic_richness_score,
            "processed_harmonic_score": proc_analysis.harmonic_richness_score,
            "improvement": improvement,
            "original_even_harmonics": orig_analysis.even_harmonics_ratio * 100,
            "processed_even_harmonics": proc_analysis.even_harmonics_ratio * 100,
            "original_odd_harmonics": orig_analysis.odd_harmonics_ratio * 100,
            "processed_odd_harmonics": proc_analysis.odd_harmonics_ratio * 100,
            "original_thd": orig_analysis.total_thd * 100,
            "processed_thd": proc_analysis.total_thd * 100,
        }

        return passed, improvement, details

    def suggest_enhancement(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Suggest harmonic enhancement strategy based on analysis

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns
        -------
        suggestions : dict
            Enhancement suggestions
        """
        analysis = self.analyze(audio, sr)

        suggestions = {
            "needs_warmth": analysis.even_harmonics_ratio < 0.03,
            "needs_de_harshness": analysis.odd_harmonics_ratio > 0.02,
            "optimal": analysis.passed,
            "recommended_action": None,
            "saturation_gain": 0.0,  # For adding warmth
            "de_harsh_strength": 0.0,  # For reducing harshness
        }

        if analysis.even_harmonics_ratio < 0.03:
            # Needs warmth - suggest tube-style saturation
            warmth_deficit = 0.03 - analysis.even_harmonics_ratio
            suggestions["recommended_action"] = "add_warmth"  # type: ignore[assignment]
            suggestions["saturation_gain"] = min(0.3, warmth_deficit * 5.0)  # Scale to gain (0-0.3)

        elif analysis.odd_harmonics_ratio > 0.02:
            # Needs de-harshness - suggest odd harmonic filtering
            harshness_excess = analysis.odd_harmonics_ratio - 0.01
            suggestions["recommended_action"] = "reduce_harshness"  # type: ignore[assignment]
            suggestions["de_harsh_strength"] = min(1.0, harshness_excess * 20.0)  # Scale to strength (0-1)

        else:
            suggestions["recommended_action"] = "none"  # type: ignore[assignment]  # Already optimal

        return suggestions


# =============================================================================
# HARMONIC ENHANCEMENT (OPTIONAL - FOR STUDIO_2026 MODE)
# =============================================================================


class MusicalHarmonicEnhancer:
    """
    Harmonic Enhancer for STUDIO_2026 Mode

    Adds subtle even harmonics (warmth) while suppressing odd harmonics (harshness).

    Parameters
    ----------
    saturation_gain : float
        Gain for tube-style saturation (0.0-0.3)
    mix : float
        Dry/wet mix (0.0-1.0, default 0.15)
    """

    def __init__(self, saturation_gain: float = 0.1, mix: float = 0.15) -> None:
        self.saturation_gain = np.clip(saturation_gain, 0.0, 0.3)
        self.mix = np.clip(mix, 0.0, 1.0)

    def enhance(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Fügt hinzu: musical even harmonics.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal
        sr : int
            Sample rate

        Returns
        -------
        enhanced : np.ndarray
            Enhanced audio with warmth
        report : dict
            Processing report
        """
        # Soft saturation (tanh) - generates primarily 2nd harmonic
        # Formula: tanh(x * gain) / tanh(gain)
        # This normalizes output amplitude while adding harmonics

        if self.saturation_gain > 0:
            # Apply saturation with normalization
            saturated = np.tanh(audio * (1 + self.saturation_gain)) / np.tanh(1 + self.saturation_gain)

            # Mix dry (clean) and wet (saturated)
            enhanced = (1 - self.mix) * audio + self.mix * saturated
        else:
            enhanced = audio

        # Analyze before/after
        analyzer = HarmonicCharacterAnalyzer()
        before = analyzer.analyze(audio, sr)
        after = analyzer.analyze(enhanced, sr)

        report = {
            "saturation_gain": self.saturation_gain,
            "mix": self.mix,
            "even_harmonics_before": before.even_harmonics_ratio * 100,
            "even_harmonics_after": after.even_harmonics_ratio * 100,
            "warmth_improvement": (after.warmth_score - before.warmth_score),
            "harmonic_richness_before": before.harmonic_richness_score,
            "harmonic_richness_after": after.harmonic_richness_score,
        }

        return enhanced, report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def analyze_harmonic_character(audio: np.ndarray, sr: int, threshold: float = 0.75) -> HarmonicAnalysis:
    """
    Convenience function to analyze harmonic character

    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sr : int
        Sample rate in Hz
    threshold : float
        Minimum acceptable harmonic richness score

    Returns
    -------
    HarmonicAnalysis
        Complete harmonic analysis

    Examples
    --------
    >>> import numpy as np
    >>> import soundfile as sf
    >>> result = load_audio_file("vocal.wav")
    audio, sr = result["audio"], result["sr"]
    >>> analysis = analyze_harmonic_character(audio, sr)
    logger.debug("Harmonic Richness: %.2f", analysis.harmonic_richness_score)
    logger.debug("Even Harmonics: %.1f%%", analysis.even_harmonics_ratio*100)
    logger.debug("Odd Harmonics: %.1f%%", analysis.odd_harmonics_ratio*100)
    """
    analyzer = HarmonicCharacterAnalyzer(threshold=threshold)
    return analyzer.analyze(audio, sr)


def enhance_harmonic_warmth(
    audio: np.ndarray, sr: int, saturation_gain: float = 0.1, mix: float = 0.15
) -> tuple[np.ndarray, dict]:
    """
    Fügt hinzu: musical warmth via even harmonic enhancement.

    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sr : int
        Sample rate
    saturation_gain : float
        Saturation amount (0.0-0.3)
    mix : float
        Dry/wet mix (0.0-1.0)

    Returns
    -------
    enhanced : np.ndarray
        Enhanced audio
    report : dict
        Processing report
    """
    enhancer = MusicalHarmonicEnhancer(saturation_gain=saturation_gain, mix=mix)
    return enhancer.enhance(audio, sr)


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    import sys

    import soundfile as sf

    if len(sys.argv) < 2:
        logger.debug("Usage: python harmonic_character_analyzer.py <audio_file> [--verbessern]")
        sys.exit(1)

    # Load audio
    audio_path = sys.argv[1]
    logger.debug("analysiere: %s", audio_path)
    from backend.file_import import load_audio_file

    _res = load_audio_file(audio_path)
    audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]

    # Analyze
    analysis = analyze_harmonic_character(audio, sr)

    # Report
    logger.debug("\n" + "=" * 70)
    logger.debug("HARMONIC CHARACTER Analyse")
    logger.debug("=" * 70)
    logger.debug("Harmonic Richness Wert:  %.3f (Schwelle: 0.75)", analysis.harmonic_richness_score)
    logger.debug("Status: %s", "✅ PASSED" if analysis.passed else "❌ FAILED")
    logger.debug("")
    logger.debug("Harmonic Distribution:")
    logger.debug("  Even Harmonics (2f,4f,6f):  %.2f%% (optimal: 3-8%%)", analysis.even_harmonics_ratio * 100)
    logger.debug("  Odd Harmonics (3f,5f,7f):   %.2f%% (optimal: <1%%)", analysis.odd_harmonics_ratio * 100)
    logger.debug("  Total THD:                  %.2f%%", analysis.total_thd * 100)
    logger.debug("")
    logger.debug("Character Scores:")
    logger.debug("  Warmth (Even):              %.3f", analysis.warmth_score)
    logger.debug("  Harshness Penalty (Odd):    %.3f", analysis.harshness_penalty)
    logger.debug("")
    logger.debug("Per-Harmonic Power:")
    for key, value in analysis.details.items():
        if "harmonic_power" in key or "fundamental" in key:
            logger.debug("  %s: %.6f", key, value)
    logger.debug("=" * 70)

    # Suggestions
    analyzer = HarmonicCharacterAnalyzer()
    suggestions = analyzer.suggest_enhancement(audio, sr)

    if suggestions["recommended_action"] != "none":
        logger.debug("\n💡 SUGGESTION: %s", suggestions["recommended_action"])
        if suggestions["recommended_action"] == "add_warmth":
            logger.debug("   Recommended Saturation Gain: %.2f", suggestions["saturation_gain"])
        elif suggestions["recommended_action"] == "reduce_harshness":
            logger.debug("   Recommended De-Harsh Strength: %.2f", suggestions["de_harsh_strength"])

    # Optional: Enhance
    if "--enhance" in sys.argv and suggestions["recommended_action"] == "add_warmth":
        logger.debug("\n🎵 ENHANCING...")
        enhanced, report = enhance_harmonic_warmth(audio, sr, saturation_gain=suggestions["saturation_gain"])
        output_path = audio_path.replace(".wav", "_enhanced.wav")
        sf.write(output_path, enhanced, sr)
        logger.debug("✅ verbessert audio gespeichert: %s", output_path)
        logger.debug(
            f"   Even Harmonics: {report['even_harmonics_before']:.2f}% → {report['even_harmonics_after']:.2f}%"
        )
        logger.debug("   Warmth Improvement: +%.3f", report["warmth_improvement"])
