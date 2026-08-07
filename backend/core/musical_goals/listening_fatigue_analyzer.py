"""
AURIK Listening Fatigue Analyzer
=================================

Misst Faktoren, die zur Hör-Ermüdung führen (Listening Fatigue).

Listening Fatigue = Die subtile Ermüdung/Stress, die Hörer nach 30-60 Minuten
                    empfinden, auch wenn die Musik technisch "perfekt" ist.

Kritische Faktoren:
1. Harshness (3-8 kHz) - Aggressive Mid-High Frequencies
2. Intermodulation Distortion (IMD) - Non-linear Interaction zwischen Frequenzen
3. Spectral Roughness - "Rauhigkeit" im Frequenzspektrum
4. Bark Scale Balance - Psychoakustische Frequenzverteilung
5. Temporal Masking - Zeitliche Maskierungseffekte

Score: 0.0 = High Fatigue (schlecht), 1.0 = No Fatigue (perfekt)
Threshold: 0.90 (sehr strict - Listening Comfort ist kritisch!)

Wissenschaftliche Grundlagen:
- Zwicker & Fastl (2006): "Psychoacoustics: Facts and Models"
- Moore (2012): "An Introduction to the Psychology of Hearing"
- ISO 532-1: Loudness (Zwicker Method)

Autor: AURIK Phase 2.0 - Psychoakustische Exzellenz
Datum: 13. Februar 2026
"""

import logging
import warnings
from dataclasses import dataclass

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class FatigueAnalysis:
    """Result of listening fatigue analysis"""

    fatigue_score: float  # 0.0-1.0 (1.0 = no fatigue)
    harshness_score: float  # 0.0-1.0 (1.0 = no harshness)
    imd_score: float  # 0.0-1.0 (1.0 = no IMD)
    roughness_score: float  # 0.0-1.0 (1.0 = smooth)
    bark_balance_score: float  # 0.0-1.0 (1.0 = balanced)
    temporal_masking_score: float  # 0.0-1.0 (1.0 = no issues)
    passed: bool  # True if fatigue_score >= threshold
    details: dict[str, float]


class ListeningFatigueAnalyzer:
    """
    Listening Fatigue Analyzer

    Misst psychoakustische Faktoren, die zur Hör-Ermüdung führen.

    Parameters
    ----------
    threshold : float
        Minimum fatigue score (0.90 recommended)
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

        # Critical frequency ranges
        self.harshness_band = (3000, 8000)  # "Presence" band - can be harsh
        self.sibilance_band = (5000, 10000)  # S-sounds - often harsh
        self.air_band = (12000, 20000)  # "Air" - should be smooth

        # Bark scale boundaries (Hz) - Psychoacoustic critical bands
        self.bark_boundaries = [
            20,
            100,
            200,
            300,
            400,
            510,
            630,
            770,
            920,
            1080,
            1270,
            1480,
            1720,
            2000,
            2320,
            2700,
            3150,
            3700,
            4400,
            5300,
            6400,
            7700,
            9500,
            12000,
            15500,
            20000,
        ]

    def analyze(self, audio: np.ndarray, sr: int, return_details: bool = True) -> FatigueAnalysis:
        """
        Analysiert Hörmüdigkeitsfaktoren.

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
        FatigueAnalysis
            Complete fatigue analysis
        """
        # Convert to mono for analysis
        if audio.ndim > 1:
            # Kürzere Achse = Kanal-Achse (funktioniert für (n,2) UND (2,n))
            channel_axis = int(np.argmin(audio.shape))
            audio_mono = np.mean(audio, axis=channel_axis)
        else:
            audio_mono = audio

        # Normalize audio for consistent analysis (§0 Peak-Guard: 99.9th percentile)
        from backend.core.core_utils import safe_peak_amplitude

        audio_mono = audio_mono / (safe_peak_amplitude(audio_mono) + 1e-10)

        # Measure individual factors
        harshness_score = self._measure_harshness(audio_mono, sr)
        imd_score = self._measure_imd(audio_mono, sr)
        roughness_score = self._measure_spectral_roughness(audio_mono, sr)
        bark_balance_score = self._measure_bark_balance(audio_mono, sr)
        temporal_masking_score = self._measure_temporal_masking(audio_mono, sr)

        # Weighted combination (prioritize harshness & IMD)
        fatigue_score = (
            0.35 * harshness_score  # Harshness ist KRITISCH
            + 0.25 * imd_score  # IMD ist sehr wichtig
            + 0.20 * roughness_score  # Roughness wichtig
            + 0.15 * bark_balance_score  # Balance wichtig
            + 0.05 * temporal_masking_score  # Temporal weniger kritisch
        )

        # Clip to [0, 1]
        fatigue_score = np.clip(fatigue_score, 0.0, 1.0)

        passed = fatigue_score >= self.threshold

        details = {
            "harshness_3_8khz": 1.0 - harshness_score,  # Invert für "Problem"-Darstellung
            "imd_distortion": 1.0 - imd_score,
            "spectral_roughness": 1.0 - roughness_score,
            "bark_imbalance": 1.0 - bark_balance_score,
            "temporal_masking_issues": 1.0 - temporal_masking_score,
        }

        return FatigueAnalysis(
            fatigue_score=fatigue_score,
            harshness_score=harshness_score,
            imd_score=imd_score,
            roughness_score=roughness_score,
            bark_balance_score=bark_balance_score,
            temporal_masking_score=temporal_masking_score,
            passed=passed,
            details=details,
        )

    def _measure_harshness(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure harshness in critical mid-high frequency band (3-8 kHz)

        Harshness entsteht durch:
        - Übertriebene Energie in 3-8 kHz (Presence Band)
        - Scharfe Sibilanz in 5-10 kHz
        - Fehlende Balance zwischen Brillanz und Harshness

        Returns
        -------
        float
            Harshness score: 0.0 = very harsh, 1.0 = no harshness
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=4096, hop_length=1024)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        # Full spectrum energy
        full_energy: float = float(np.sum(magnitude**2))

        # Harshness band energy (3-8 kHz)
        harshness_mask = (freqs >= self.harshness_band[0]) & (freqs <= self.harshness_band[1])
        harshness_energy: float = float(np.sum(magnitude[harshness_mask] ** 2))

        # Sibilance band energy (5-10 kHz)
        sibilance_mask = (freqs >= self.sibilance_band[0]) & (freqs <= self.sibilance_band[1])
        sibilance_energy: float = float(np.sum(magnitude[sibilance_mask] ** 2))

        # Air band energy (12-20 kHz) - should be present but smooth
        air_mask = (freqs >= self.air_band[0]) & (freqs <= self.air_band[1])
        air_energy: float = float(np.sum(magnitude[air_mask] ** 2))

        # Calculate ratios
        harshness_ratio = harshness_energy / (full_energy + 1e-10)
        sibilance_ratio = sibilance_energy / (full_energy + 1e-10)
        air_ratio = air_energy / (full_energy + 1e-10)

        # Optimal ranges (empirically determined)
        # Harshness: 0.05-0.15 (5-15% of total energy = OK)
        # Sibilance: 0.02-0.08 (2-8% = OK)
        # Air: 0.01-0.05 (1-5% = OK)

        # Score harshness (penalize excess)
        if harshness_ratio < 0.05:
            harsh_score = 0.8  # Too little presence (dull)
        elif harshness_ratio <= 0.15:
            harsh_score = 1.0  # Optimal
        elif harshness_ratio <= 0.25:
            harsh_score = 0.7  # Slightly harsh
        else:
            harsh_score = 0.3  # Very harsh

        # Score sibilance
        if sibilance_ratio < 0.02:
            sib_score = 0.9  # OK (not much sibilance)
        elif sibilance_ratio <= 0.08:
            sib_score = 1.0  # Optimal
        elif sibilance_ratio <= 0.15:
            sib_score = 0.6  # Harsh sibilance
        else:
            sib_score = 0.2  # Very harsh

        # Score air (should be smooth, not excessive)
        if air_ratio < 0.005:
            air_score = 0.7  # Too little air (dull)
        elif air_ratio <= 0.05:
            air_score = 1.0  # Optimal
        else:
            air_score = 0.8  # Slightly harsh highs

        # Combined harshness score (weighted)
        harshness_score = 0.50 * harsh_score + 0.35 * sib_score + 0.15 * air_score

        return harshness_score

    def _measure_imd(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure Intermodulation Distortion (IMD)

        IMD entsteht durch non-lineare Interaktion zwischen Frequenzen.
        Z.B. zwei Töne bei 1000 Hz und 2000 Hz erzeugen IMD-Produkte bei:
        - 3000 Hz (2000 + 1000)
        - 1000 Hz (2000 - 1000)
        - etc.

        IMD ist sehr ermüdend für das Ohr, aber schwer zu messen.
        Wir nutzen einen vereinfachten Ansatz über Spectral Flux.

        Returns
        -------
        float
            IMD score: 0.0 = high IMD, 1.0 = low IMD
        """
        # Compute spectral flux (frame-to-frame spectral change)
        # High flux can indicate IMD products appearing/disappearing

        # STFT with overlap
        hop_length = 512
        stft = librosa.stft(audio, n_fft=2048, hop_length=hop_length)
        magnitude = np.abs(stft)

        # Compute spectral flux (L2-norm of spectral difference)
        flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        flux = np.sqrt(flux)

        # Normalize by frame energy
        frame_energy = np.sum(magnitude[:, 1:] ** 2, axis=0)
        flux_normalized = flux / (frame_energy + 1e-10)

        # Guard: leerer flux (z. B. sehr kurzes Audio < 1 STFT-Frame)
        if flux_normalized.size == 0:
            return 1.0  # Kein Flux → kein messbares IMD → Excellent

        # High flux = potential IMD (or legitimate dynamic content)
        # We need to distinguish between:
        # - Legitimate transients (musical, OK)
        # - IMD artifacts (non-musical, BAD)

        # Use 90th percentile as "typical high flux"
        flux_90th = np.percentile(flux_normalized, 90)

        # Empirical thresholds (based on testing)
        # Low IMD: flux_90th < 0.5
        # Medium: 0.5 - 1.0
        # High: > 1.0

        if flux_90th < 0.5:
            imd_score = 1.0  # Excellent
        elif flux_90th < 1.0:
            imd_score = 0.8  # Good
        elif flux_90th < 2.0:
            imd_score = 0.6  # Acceptable
        else:
            imd_score = 0.4  # Poor

        return imd_score

    def _measure_spectral_roughness(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure spectral roughness (Rauhigkeit)

        Roughness entsteht durch:
        - Unregelmäßige Spektralverteilung
        - "Peaks" und "Notches" im Frequenzgang
        - Fehlende spektrale "Glätte"

        Returns
        -------
        float
            Roughness score: 0.0 = very rough, 1.0 = smooth
        """
        # Compute power spectrum
        magnitude = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Focus on audible range (20 Hz - 20 kHz)
        audible_mask = (freqs >= 20) & (freqs <= 20000)
        magnitude_audible = magnitude[audible_mask]

        # Smooth spectrum (what "ideal" would look like)
        # Use median filter to get smooth version
        window_size = 101  # Odd number for median
        magnitude_smooth = signal.medfilt(magnitude_audible, kernel_size=window_size)

        # Compute roughness as deviation from smooth
        roughness = np.abs(magnitude_audible - magnitude_smooth)
        roughness_mean = np.mean(roughness)

        # Normalize by signal amplitude
        signal_amplitude = np.mean(magnitude_audible)
        roughness_normalized = roughness_mean / (signal_amplitude + 1e-10)

        # Empirical thresholds
        # Low roughness: < 0.1
        # Medium: 0.1 - 0.3
        # High: > 0.3

        if roughness_normalized < 0.1:
            roughness_score = 1.0  # Very smooth
        elif roughness_normalized < 0.3:
            roughness_score = 0.8  # Smooth
        elif roughness_normalized < 0.5:
            roughness_score = 0.6  # Acceptable
        else:
            roughness_score = 0.4  # Rough

        return roughness_score

    def _measure_bark_balance(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure Bark scale balance (psychoacoustic frequency distribution)

        Die Bark-Skala repräsentiert die kritischen Bänder des menschlichen Gehörs.
        Eine ausgewogene Verteilung über die Bark-Bänder ist wichtig für
        ermüdungsfreies Hören.

        Returns
        -------
        float
            Bark balance score: 0.0 = imbalanced, 1.0 = balanced
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=4096, hop_length=1024)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        # Compute energy in each Bark band
        bark_energies = []
        for i in range(len(self.bark_boundaries) - 1):
            f_low = self.bark_boundaries[i]
            f_high = self.bark_boundaries[i + 1]

            # Mask for this bark band
            bark_mask = (freqs >= f_low) & (freqs < f_high)
            bark_energy: float = float(np.sum(magnitude[bark_mask] ** 2))
            bark_energies.append(bark_energy)

        bark_energies = np.array(bark_energies)  # type: ignore[assignment]

        # Normalize energies
        bark_energies_norm = bark_energies / (np.sum(bark_energies) + 1e-10)

        # Compute balance as inverse of standard deviation
        # Low std = balanced, high std = imbalanced

        bark_std = np.std(bark_energies_norm)

        # Empirical thresholds
        # Excellent balance: std < 0.02
        # Good: 0.02 - 0.04
        # Acceptable: 0.04 - 0.06
        # Poor: > 0.06

        if bark_std < 0.02:
            balance_score = 1.0
        elif bark_std < 0.04:
            balance_score = 0.9
        elif bark_std < 0.06:
            balance_score = 0.7
        else:
            balance_score = 0.5

        return balance_score

    def _measure_temporal_masking(self, audio: np.ndarray, sr: int) -> float:
        """
        Measure temporal masking effects

        Temporal Masking = Ein lautes Signal maskiert leise Signale kurz davor/danach.
        Zu viel Maskierung führt zu Ermüdung (Ohr muss "härter arbeiten").

        Returns
        -------
        float
            Temporal masking score: 0.0 = high masking, 1.0 = low masking
        """
        # Compute envelope (instantaneous amplitude)
        analytic_signal: np.ndarray = signal.hilbert(audio)  # type: ignore[assignment]
        envelope = np.abs(analytic_signal)

        # Smooth envelope (10ms window)
        window_size = int(0.01 * sr)
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Compute frame-by-frame dynamic range
        frame_length = int(0.050 * sr)  # 50ms frames
        hop_length = frame_length // 2

        # Guard: Audio zu kurz für Frame-Analyse
        if len(envelope_smooth) < frame_length:
            return 1.0  # Kein Temporal Masking angenommen
        frames = librosa.util.frame(envelope_smooth, frame_length=frame_length, hop_length=hop_length)
        frame_max = np.max(frames, axis=0)
        frame_min = np.min(frames, axis=0)

        # Dynamic range per frame (dB)
        frame_dr = 20 * np.log10((frame_max + 1e-10) / (frame_min + 1e-10))

        # High DR = good (less masking)
        # Low DR = bad (more masking)

        mean_dr = np.mean(frame_dr)

        # Empirical thresholds
        # Excellent: > 30 dB
        # Good: 20-30 dB
        # Acceptable: 10-20 dB
        # Poor: < 10 dB

        if mean_dr > 30:
            masking_score = 1.0
        elif mean_dr > 20:
            masking_score = 0.9
        elif mean_dr > 10:
            masking_score = 0.7
        else:
            masking_score = 0.5

        return masking_score

    def check_preservation(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[bool, float, dict[str, float]]:
        """
        Prüft if processed audio is less fatiguing than original.

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
            True if processed is less/equally fatiguing
        improvement : float
            Fatigue score improvement (positive = better)
        details : dict
            Detailed comparison
        """
        orig_analysis = self.analyze(original, sr, return_details=True)
        proc_analysis = self.analyze(processed, sr, return_details=True)

        improvement = proc_analysis.fatigue_score - orig_analysis.fatigue_score

        # Processing should reduce fatigue (or keep it same)
        # Allow small increase (0.05) for edge cases
        passed = improvement >= -0.05

        details = {
            "original_fatigue_score": orig_analysis.fatigue_score,
            "processed_fatigue_score": proc_analysis.fatigue_score,
            "improvement": improvement,
            "original_harshness": 1.0 - orig_analysis.harshness_score,
            "processed_harshness": 1.0 - proc_analysis.harshness_score,
            "original_imd": 1.0 - orig_analysis.imd_score,
            "processed_imd": 1.0 - proc_analysis.imd_score,
        }

        return passed, improvement, details


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def analyze_listening_fatigue(audio: np.ndarray, sr: int, threshold: float = 0.90) -> FatigueAnalysis:
    """
    Convenience function to analyze listening fatigue

    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sr : int
        Sample rate in Hz
    threshold : float
        Minimum acceptable fatigue score

    Returns
    -------
    FatigueAnalysis
        Complete fatigue analysis

    Examples
    --------
    >>> import numpy as np
    >>>     >>> result = load_audio_file("audio.wav")
    audio, sr = result["audio"], result["sr"]
    >>> analysis = analyze_listening_fatigue(audio, sr)
    logger.debug("Fatigue Score: %.2f", analysis.fatigue_score)
    logger.debug("Passed: %s", analysis.passed)
    >>> if not analysis.passed:
    ...     logger.debug("Warning: High listening fatigue detected!")
    logger.debug("  Harshness: %.2f", 1.0 - analysis.harshness_score)
    logger.debug("  IMD: %.2f", 1.0 - analysis.imd_score)
    """
    analyzer = ListeningFatigueAnalyzer(threshold=threshold)
    return analyzer.analyze(audio, sr)


def check_fatigue_preservation(original: np.ndarray, processed: np.ndarray, sr: int) -> tuple[bool, float, dict]:
    """
    Prüft if processing maintains/improves listening comfort.

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
        True if processed is less/equally fatiguing
    improvement : float
        Fatigue improvement (positive = better)
    details : dict
        Detailed comparison
    """
    analyzer = ListeningFatigueAnalyzer()
    return analyzer.check_preservation(original, processed, sr)


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.debug("Usage: python listening_fatigue_analyzer.py <audio_file>")
        sys.exit(1)

    # Load audio
    audio_path = sys.argv[1]
    logger.debug("analysiere: %s", audio_path)
    from backend.file_import import load_audio_file

    _res = load_audio_file(audio_path)
    audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]

    # Analyze
    analysis = analyze_listening_fatigue(audio, sr)

    # Report
    logger.debug("\n" + "=" * 70)
    logger.debug("LISTENING FATIGUE Analyse")
    logger.debug("=" * 70)
    logger.debug("Overall Fatigue Wert: %.3f (Schwelle: 0.90)", analysis.fatigue_score)
    logger.debug("Status: %s", "✅ PASSED" if analysis.passed else "❌ FAILED")
    logger.debug("")
    logger.debug("Component Scores (1.0 = Perfect):")
    logger.debug("  Harshness (3-8 kHz):        %.3f", analysis.harshness_score)
    logger.debug("  IMD (Distortion):           %.3f", analysis.imd_score)
    logger.debug("  Spectral Roughness:         %.3f", analysis.roughness_score)
    logger.debug("  Bark Scale Balance:         %.3f", analysis.bark_balance_score)
    logger.debug("  Temporal Masking:           %.3f", analysis.temporal_masking_score)
    logger.debug("")
    logger.debug("Problem Indicators (0.0 = No Problem):")
    for key, value in analysis.details.items():
        logger.debug("  %s: %.3f", key, value)
    logger.debug("=" * 70)

    if not analysis.passed:
        logger.debug("\n⚠️  WARNING: High listening fatigue erkannt!")
        logger.debug("Recommendation: anwenden fatigue reduction processing")
