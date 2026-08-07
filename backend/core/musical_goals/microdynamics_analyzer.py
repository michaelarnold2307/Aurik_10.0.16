"""
AURIK Microdynamics Analyzer
=============================

Misst Mikrodynamik - die feinen dynamischen Nuancen, die Musik lebendig machen.

Mikrodynamik vs. Makrodynamik:
- **Makrodynamik:** Gesamter Dynamic Range (Peak-to-RMS über ganze Datei)
- **Mikrodynamik:** LOKALE Dynamic Variationen (Frame-by-Frame, 10-100ms)

Mikrodynamik ist entscheidend für:
- "Lebendigen" vs. "Toten" Klang
- Natürliche Expression (Piano-Anschlag, Vocal-Vibrato)
- Emotionale Tiefe
- "Atmende" Qualität

Messung:
- Frame-by-Frame RMS Variance (10-100ms Fenster)
- Envelope Modulation Depth
- Local Crest Factor Variability
- Transient Density & Diversity

Score: 0.0 = Flat/Dead, 1.0 = Highly Expressive
Threshold: 0.70 (high variability = good!)

Wissenschaftliche Grundlagen:
- Katz (2014): "Mastering Audio: The Art and the Science"
- Vickers (2010): "Automatic Long-Term Loudness and Dynamics Matching"

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
class MicrodynamicsAnalysis:
    """Result of microdynamics analysis"""

    microdynamics_score: float  # 0.0-1.0 (1.0 = highly expressive)
    frame_variance_score: float  # RMS variance across frames
    envelope_modulation_score: float  # Envelope modulation depth
    crest_variability_score: float  # Local crest factor changes
    transient_diversity_score: float  # Transient density & variation
    passed: bool  # True if score >= threshold
    details: dict[str, float]


class MicrodynamicsAnalyzer:
    """
    Microdynamics Analyzer

    Misst lokale dynamische Variationen, die Musik lebendig machen.

    Parameters
    ----------
    threshold : float
        Minimum microdynamics score (0.70 recommended)
    frame_size_ms : float
        Frame size for analysis in milliseconds (50ms default)
    """

    def __init__(self, threshold: float = 0.70, frame_size_ms: float = 50.0):
        self.threshold = threshold
        self.frame_size_ms = frame_size_ms

    def analyze(self, audio: np.ndarray, sr: int, return_details: bool = True) -> MicrodynamicsAnalysis:
        """
        Analysiert die Mikrodynamik.

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
        MicrodynamicsAnalysis
            Complete microdynamics analysis
        """
        # Convert to mono for analysis
        if audio.ndim > 1:
            # Kuerze Achse = Kanal-Achse (funktioniert fuer (n,2) UND (2,n))
            channel_axis = int(np.argmin(audio.shape))
            audio_mono = np.mean(audio, axis=channel_axis)
        else:
            audio_mono = audio

        # Normalize audio for consistent analysis (§0 Peak-Guard: 99.9th percentile)
        from backend.core.core_utils import safe_peak_amplitude

        audio_mono = audio_mono / (safe_peak_amplitude(audio_mono) + 1e-10)

        # Measure individual components
        frame_variance_score = self._measure_frame_variance(audio_mono, sr)
        envelope_modulation_score = self._measure_envelope_modulation(audio_mono, sr)
        crest_variability_score = self._measure_crest_variability(audio_mono, sr)
        transient_diversity_score = self._measure_transient_diversity(audio_mono, sr)

        # Weighted combination
        microdynamics_score = (
            0.35 * frame_variance_score  # Most important
            + 0.30 * envelope_modulation_score  # Very important
            + 0.20 * crest_variability_score  # Important
            + 0.15 * transient_diversity_score  # Supplementary
        )

        # Clip to [0, 1]
        microdynamics_score = np.clip(microdynamics_score, 0.0, 1.0)

        passed = microdynamics_score >= self.threshold

        details = {
            "frame_rms_variance": frame_variance_score,
            "envelope_modulation_depth": envelope_modulation_score,
            "local_crest_factor_variability": crest_variability_score,
            "transient_density": transient_diversity_score,
        }

        return MicrodynamicsAnalysis(
            microdynamics_score=microdynamics_score,
            frame_variance_score=frame_variance_score,
            envelope_modulation_score=envelope_modulation_score,
            crest_variability_score=crest_variability_score,
            transient_diversity_score=transient_diversity_score,
            passed=passed,
            details=details,
        )

    def _measure_frame_variance(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst RMS variance across frames.

        High variance = high microdynamics = good!

        Returns
        -------
        float
            Frame variance score: 0.0 = flat, 1.0 = highly variable
        """
        # Frame parameters
        frame_length = int(self.frame_size_ms * sr / 1000)
        hop_length = frame_length // 4  # 75% overlap for smooth analysis

        # Frame audio
        if len(audio) < frame_length:
            return 0.5  # Audio zu kurz: neutraler Score
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)

        # Compute RMS per frame
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))

        # Filter out silence (below -60 dB)
        silence_threshold = 10 ** (-60 / 20)  # -60 dB
        active_frames = frame_rms > silence_threshold

        if np.sum(active_frames) < 10:
            # Not enough active frames
            return 0.5

        frame_rms_active = frame_rms[active_frames]

        # Compute variance (in linear domain)
        variance = np.var(frame_rms_active)

        # Convert to dB for interpretation
        mean_rms = np.mean(frame_rms_active)
        variance_db = 20 * np.log10((np.sqrt(variance) / mean_rms) + 1e-10)

        # Empirical thresholds (based on testing)
        # High microdynamics: variance > 3 dB
        # Medium: 1.5 - 3 dB
        # Low: < 1.5 dB

        if variance_db > 3.0:
            score = 1.0  # Excellent
        elif variance_db > 1.5:
            score = 0.7 + (variance_db - 1.5) / 1.5 * 0.3  # Linear interpolation
        else:
            score = variance_db / 1.5 * 0.7  # Linear scaling

        return np.clip(score, 0.0, 1.0)  # type: ignore[no-any-return]

    def _measure_envelope_modulation(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst envelope modulation depth.

        Envelope = Instantaneous amplitude (via Hilbert transform)
        Modulation = How much the envelope varies

        Returns
        -------
        float
            Envelope modulation score: 0.0 = flat, 1.0 = highly modulated
        """
        # Compute envelope via Hilbert transform
        analytic_signal: np.ndarray = signal.hilbert(audio)  # type: ignore[assignment]
        envelope = np.abs(analytic_signal)

        # Smooth envelope (remove very fast fluctuations)
        window_size = int(0.01 * sr)  # 10ms smoothing
        envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

        # Detect envelope peaks and valleys
        peaks, _ = signal.find_peaks(envelope_smooth, distance=int(0.05 * sr))
        valleys, _ = signal.find_peaks(-envelope_smooth, distance=int(0.05 * sr))

        if len(peaks) < 5 or len(valleys) < 5:
            # Not enough modulation
            return 0.3

        # Compute modulation depth (peak-to-valley ratio)
        peak_amplitudes = envelope_smooth[peaks]
        valley_amplitudes = envelope_smooth[valleys]

        # Compute mean peak and valley
        mean_peak = np.mean(peak_amplitudes)
        mean_valley = np.mean(valley_amplitudes)

        # Modulation depth in dB
        modulation_depth_db = 20 * np.log10(mean_peak / (mean_valley + 1e-10))

        # Empirical thresholds
        # High modulation: > 12 dB
        # Medium: 6 - 12 dB
        # Low: < 6 dB

        if modulation_depth_db > 12.0:
            score = 1.0
        elif modulation_depth_db > 6.0:
            score = 0.6 + (modulation_depth_db - 6.0) / 6.0 * 0.4
        else:
            score = modulation_depth_db / 6.0 * 0.6

        return np.clip(score, 0.0, 1.0)  # type: ignore[no-any-return]

    def _measure_crest_variability(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst local crest factor variability.

        Crest Factor = Peak / RMS (frame-by-frame)
        Variability = How much crest factor changes

        High variability = expressive dynamics

        Returns
        -------
        float
            Crest variability score: 0.0 = constant, 1.0 = highly variable
        """
        # Frame parameters
        frame_length = int(self.frame_size_ms * sr / 1000)
        hop_length = frame_length // 4

        # Frame audio
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)

        # Compute crest factor per frame
        frame_peak = np.max(np.abs(frames), axis=0)
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))

        crest_factor = frame_peak / (frame_rms + 1e-10)

        # Filter out silence
        silence_threshold = 10 ** (-60 / 20)
        active_frames = frame_rms > silence_threshold

        if np.sum(active_frames) < 10:
            return 0.5

        crest_factor_active = crest_factor[active_frames]

        # Compute crest factor variability (standard deviation)
        crest_std = np.std(crest_factor_active)

        # Empirical thresholds
        # High variability: std > 2.0
        # Medium: 1.0 - 2.0
        # Low: < 1.0

        if crest_std > 2.0:
            score = 1.0
        elif crest_std > 1.0:
            score = 0.6 + (crest_std - 1.0) / 1.0 * 0.4  # type: ignore[assignment]
        else:
            score = crest_std / 1.0 * 0.6  # type: ignore[assignment]

        return np.clip(score, 0.0, 1.0)  # type: ignore[no-any-return]

    def _measure_transient_diversity(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst transient density and diversity.

        Transients = Short, sharp attacks (drums, piano, etc.)
        Density = How many transients per second
        Diversity = How varied are the transient strengths

        Returns
        -------
        float
            Transient diversity score: 0.0 = few/monotone, 1.0 = many/diverse
        """
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)  # type: ignore[attr-defined]

        # Detect transients (peaks in onset envelope)
        transient_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=512, backtrack=False)  # type: ignore[attr-defined]

        if len(transient_frames) < 5:
            # Very few transients
            return 0.2

        # Transient density (transients per second)
        duration = len(audio) / sr
        transient_density = len(transient_frames) / duration

        # Transient strength diversity (variance of onset strengths)
        transient_strengths = onset_env[transient_frames]
        strength_variance = np.var(transient_strengths)
        strength_mean = np.mean(transient_strengths)
        strength_cv = np.sqrt(strength_variance) / (strength_mean + 1e-10)  # Coefficient of variation

        # Score density
        # Optimal: 1-10 transients/sec
        if 1.0 <= transient_density <= 10.0:
            density_score = 1.0
        elif transient_density < 1.0:
            density_score = transient_density  # Linear scaling
        else:
            # Too many transients (noise?)
            density_score = 1.0 - min(1.0, (transient_density - 10.0) / 10.0)

        # Score diversity (coefficient of variation)
        # High CV = diverse (good)
        # Optimal: CV > 0.5
        diversity_score = 1.0 if strength_cv > 0.5 else strength_cv / 0.5

        # Combined score (equal weighting)
        transient_score = 0.5 * density_score + 0.5 * diversity_score

        return np.clip(transient_score, 0.0, 1.0)  # type: ignore[no-any-return]

    def check_preservation(
        self, original: np.ndarray, processed: np.ndarray, sr: int
    ) -> tuple[bool, float, dict[str, float]]:
        """
        Prüft if processed audio maintains microdynamics.

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
            True if microdynamics preserved (within 10% loss)
        loss : float
            Microdynamics loss (negative = loss, positive = gain)
        details : dict
            Detailed comparison
        """
        orig_analysis = self.analyze(original, sr, return_details=True)
        proc_analysis = self.analyze(processed, sr, return_details=True)

        loss = proc_analysis.microdynamics_score - orig_analysis.microdynamics_score

        # Allow 10% loss (0.1 points)
        max_allowed_loss = 0.10
        passed = loss >= -max_allowed_loss

        details = {
            "original_microdynamics": orig_analysis.microdynamics_score,
            "processed_microdynamics": proc_analysis.microdynamics_score,
            "loss": loss,
            "max_allowed_loss": max_allowed_loss,
            "original_frame_variance": orig_analysis.frame_variance_score,
            "processed_frame_variance": proc_analysis.frame_variance_score,
            "original_envelope_modulation": orig_analysis.envelope_modulation_score,
            "processed_envelope_modulation": proc_analysis.envelope_modulation_score,
        }

        return passed, loss, details


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def analyze_microdynamics(
    audio: np.ndarray, sr: int, threshold: float = 0.70, frame_size_ms: float = 50.0
) -> MicrodynamicsAnalysis:
    """
    Convenience function to analyze microdynamics

    Parameters
    ----------
    audio : np.ndarray
        Audio signal
    sr : int
        Sample rate in Hz
    threshold : float
        Minimum acceptable microdynamics score
    frame_size_ms : float
        Frame size for analysis in milliseconds

    Returns
    -------
    MicrodynamicsAnalysis
        Complete microdynamics analysis

    Examples
    --------
    >>> import numpy as np
    >>>     >>> result = load_audio_file("piano.wav")
    audio, sr = result["audio"], result["sr"]
    >>> analysis = analyze_microdynamics(audio, sr)
    logger.debug("Microdynamics Score: %.2f", analysis.microdynamics_score)
    logger.debug("Passed: %s", analysis.passed)
    >>> if analysis.microdynamics_score < 0.5:
    ...     logger.debug("Warning: Flat/dead sound - low microdynamics!")
    """
    analyzer = MicrodynamicsAnalyzer(threshold=threshold, frame_size_ms=frame_size_ms)
    return analyzer.analyze(audio, sr)


def check_microdynamics_preservation(original: np.ndarray, processed: np.ndarray, sr: int) -> tuple[bool, float, dict]:
    """
    Prüft if processing preserves microdynamics.

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
        True if microdynamics preserved
    loss : float
        Microdynamics change (negative = loss)
    details : dict
        Detailed comparison
    """
    analyzer = MicrodynamicsAnalyzer()
    return analyzer.check_preservation(original, processed, sr)


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.debug("Usage: python microdynamics_analyzer.py <audio_file>")
        sys.exit(1)

    # Load audio
    audio_path = sys.argv[1]
    logger.debug("analysiere: %s", audio_path)
    from backend.file_import import load_audio_file

    _res = load_audio_file(audio_path)
    audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]

    # Analyze
    analysis = analyze_microdynamics(audio, sr)

    # Report
    logger.debug("\n" + "=" * 70)
    logger.debug("MICRODYNAMICS Analyse")
    logger.debug("=" * 70)
    logger.debug("Overall Microdynamics Wert: %.3f (Schwelle: 0.70)", analysis.microdynamics_score)
    logger.debug("Status: %s", "✅ PASSED" if analysis.passed else "❌ FAILED")
    logger.debug("")
    logger.debug("Component Scores (1.0 = Perfect):")
    logger.debug("  Frame RMS Variance:         %.3f", analysis.frame_variance_score)
    logger.debug("  Envelope Modulation:        %.3f", analysis.envelope_modulation_score)
    logger.debug("  Crest Factor Variability:   %.3f", analysis.crest_variability_score)
    logger.debug("  Transient Diversity:        %.3f", analysis.transient_diversity_score)
    logger.debug("")
    logger.debug("Detail Metrics:")
    for key, value in analysis.details.items():
        logger.debug("  %s: %.3f", key, value)
    logger.debug("=" * 70)

    if not analysis.passed:
        logger.debug("\n⚠️  WARNING: Low microdynamics erkannt!")
        logger.debug("Recommendation: Sound may be flat/dead. Consider dynamic expansion.")
    else:
        logger.debug("\n✅ Excellent microdynamics - expressive, living sound!")
