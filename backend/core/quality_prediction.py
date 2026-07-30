"""
core/quality_prediction.py
Quality Prediction System
=========================

Ferrari Edition - Premium Qualitäts-Vorhersage:
- Pre-processing Quality Estimation
- Expected Improvement Prediction
- Confidence Scoring
- Result Quality Validation
- Processing Time Estimation
- Quality Gates (Early Stopping)

Version: 2.0.0 "Limited Edition - Qualitäts-Fokus"
Author: AURIK Team
Date: 10. Februar 2026
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Quality metrics that can be predicted."""

    SNR = "snr"  # Signal-to-Noise Ratio (dB)
    DYNAMIC_RANGE = "dynamic_range"  # Dynamic Range (dB)
    THD = "thd"  # Total Harmonic Distortion (%)
    CLARITY = "clarity"  # Perceptual Clarity (0-1)
    WARMTH = "warmth"  # Tonal Warmth (0-1)
    BRIGHTNESS = "brightness"  # High Frequency Energy (0-1)
    NATURALNESS = "naturalness"  # Natural Sound (0-1)
    AUTHENTICITY = "authenticity"  # Period Authenticity (0-1)
    PLEASANTNESS = "pleasantness"  # §v10 Human Pleasantness (0-1) — PRIMÄR


class QualityLevel(Enum):
    """Quality assessment levels."""

    POOR = "poor"  # <40%
    FAIR = "fair"  # 40-60%
    GOOD = "good"  # 60-80%
    EXCELLENT = "excellent"  # 80-95%
    PRISTINE = "pristine"  # >95%


@dataclass
class QualityEstimate:
    """
    Quality estimation for audio.
    """

    # Overall quality
    overall_score: float  # 0-100
    quality_level: QualityLevel

    # Metric predictions
    snr_db: float
    dynamic_range_db: float
    thd_percent: float
    clarity: float  # 0-1

    # Perceptual qualities
    warmth: float  # 0-1
    brightness: float  # 0-1
    naturalness: float  # 0-1
    authenticity: float  # 0-1

    # Confidence
    confidence: float  # 0-1

    # Details
    bandwidth_hz: tuple[float, float]  # (low, high)
    has_artifacts: bool
    artifact_types: list[str] = field(default_factory=list)


@dataclass
class ImprovementPrediction:
    """
    Predicted improvement from processing.
    """

    # Expected improvements
    snr_improvement_db: float
    dynamic_range_improvement_db: float
    thd_reduction_percent: float
    clarity_improvement: float  # 0-1

    # Expected final metrics
    final_snr_db: float
    final_dynamic_range_db: float
    final_thd_percent: float
    final_clarity: float

    # Perceptual improvements
    warmth_improvement: float
    brightness_improvement: float
    naturalness_improvement: float

    # Confidence in prediction
    confidence: float  # 0-1

    # Processing details
    estimated_processing_time_sec: float
    recommended_modules: list[str]
    quality_gates: dict[str, float]  # Metric -> target value


@dataclass
class QualityValidation:
    """
    Validierung von vorhergesagter vs. tatsächlicher Qualität.
    """

    # Prediction accuracy
    snr_accuracy_db: float  # |predicted - actual|
    clarity_accuracy: float  # |predicted - actual|
    time_accuracy_sec: float  # |predicted - actual|

    # Overall prediction quality
    prediction_score: float  # 0-100 (how accurate was prediction)

    # Details
    predicted_quality: QualityEstimate
    actual_quality: QualityEstimate
    improvement: ImprovementPrediction


class QualityAnalyzer:
    """
    Analysiert die Audioqualität.

    Features:
    - SNR estimation
    - Dynamic range measurement
    - THD estimation
    - Perceptual quality metrics
    - Artifact detection
    """

    def analyze_quality(
        self,
        audio: np.ndarray,
        sample_rate: int,
        reference: np.ndarray | None = None,
        *,
        perceptual_score: float | None = None,
    ) -> QualityEstimate:
        """
        Analysiert die Audioqualität — perzeptuell gewichtet für menschliches Hören.

        Args:
            audio: Audio signal
            sample_rate: Sample rate
            reference: Optional reference signal for comparison
            perceptual_score: Optional MUSHRA/OQS-Score (0-100) für perzeptuelle Gewichtung

        Returns:
            QualityEstimate with comprehensive metrics
        """
        # Stereo → Mono (alle FFT-basierten Methoden erwarten 1D)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1 if audio.shape[1] <= audio.shape[0] else 0)

        # Basic metrics
        snr_db = self._estimate_snr(audio)
        dynamic_range_db = self._measure_dynamic_range(audio)
        thd_percent = self._estimate_thd(audio, sample_rate)

        # Perceptual metrics
        clarity = self._measure_clarity(audio, sample_rate)
        warmth = self._measure_warmth(audio, sample_rate)
        brightness = self._measure_brightness(audio, sample_rate)
        naturalness = self._measure_naturalness(audio, sample_rate, reference)
        authenticity = self._measure_authenticity(audio, sample_rate)

        # Bandwidth
        bandwidth = self._measure_bandwidth(audio, sample_rate)

        # Artifacts
        has_artifacts, artifact_types = self._detect_artifacts(audio, sample_rate)

        # Overall score (perzeptuell gewichtet, §v10.101)
        overall_score = self._calculate_overall_score(
            snr_db,
            dynamic_range_db,
            thd_percent,
            clarity,
            warmth,
            brightness,
            naturalness,
            perceptual_score=perceptual_score,
        )

        # Quality level
        quality_level = self._determine_quality_level(overall_score)

        # Confidence (based on signal characteristics)
        confidence = self._calculate_confidence(audio, snr_db)

        return QualityEstimate(
            overall_score=overall_score,
            quality_level=quality_level,
            snr_db=snr_db,
            dynamic_range_db=dynamic_range_db,
            thd_percent=thd_percent,
            clarity=clarity,
            warmth=warmth,
            brightness=brightness,
            naturalness=naturalness,
            authenticity=authenticity,
            confidence=confidence,
            bandwidth_hz=bandwidth,
            has_artifacts=has_artifacts,
            artifact_types=artifact_types,
        )

    # === Metric Calculations ===

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Schätzt Signal-to-Noise Ratio.

        Uses signal power vs noise floor estimation.
        Improved: Uses bottom 5% of amplitude envelope as noise floor.
        """
        # Compute amplitude envelope (absolute value smoothed)
        envelope = np.abs(audio)

        # Noise floor = bottom 5% of envelope (excluding zeros)
        non_zero_env = envelope[envelope > 1e-10]
        if len(non_zero_env) < 10:
            return 100.0  # Very clean or silent

        noise_floor_rms = np.percentile(non_zero_env, 5)

        # Signal RMS (overall)
        signal_rms = np.sqrt(np.mean(audio**2))

        # SNR in dB
        if noise_floor_rms > 1e-10:
            snr_db = 20 * np.log10(signal_rms / noise_floor_rms)
        else:
            snr_db = 100.0  # Very clean signal

        return float(np.clip(snr_db, 0, 100))

    def _measure_dynamic_range(self, audio: np.ndarray) -> float:
        """
        Misst dynamic range (peak to noise floor).
        """
        peak: float = float(np.max(np.abs(audio)))
        noise_floor = np.percentile(np.abs(audio), 10)

        dr_db = 20 * np.log10(peak / noise_floor) if noise_floor > 1e-10 else 100.0

        return float(np.clip(dr_db, 0, 120))

    def _estimate_thd(self, audio: np.ndarray, sr: int) -> float:
        """
        Schätzt Total Harmonic Distortion.

        Simplified: ratio of high-frequency energy to mid-frequency.
        """
        # FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)

        # Power spectrum
        power = np.abs(fft) ** 2

        # Fundamental (200-2000 Hz)
        fundamental_mask = (freqs >= 200) & (freqs <= 2000)
        fundamental_power: float = float(np.sum(power[fundamental_mask]))

        # Harmonics (2000-10000 Hz)
        harmonic_mask = (freqs > 2000) & (freqs <= 10000)
        harmonic_power: float = float(np.sum(power[harmonic_mask]))

        # THD
        thd = 100 * harmonic_power / fundamental_power if fundamental_power > 0 else 0.0

        return float(np.clip(thd, 0, 100))

    def _measure_clarity(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst perceptual clarity (high-frequency energy balance).
        """
        # FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        power = np.abs(fft) ** 2

        # HF energy (4-16 kHz)
        hf_mask = (freqs >= 4000) & (freqs <= 16000)
        hf_power: float = float(np.sum(power[hf_mask]))

        # Total energy
        total_power: float = float(np.sum(power))

        # Clarity = HF ratio (well-balanced).
        # The original formula  1 - |0.20 - r| / 0.20  goes negative whenever
        # hf_ratio > 0.40 (e.g. bright modern MP3 with intact 4–16 kHz content),
        # which clamps to 0.0 and triggers a false "Clarity too low" gate failure.
        #
        # Fix: use a soft Gaussian-shaped score centred at 0.20 with σ=0.15 so
        # that any reasonable HF balance (0.05 – 0.60) scores ≥ 0.15, and
        # perfectly balanced material (hf_ratio ≈ 0.20) scores close to 1.0.
        # A completely energy-free signal still scores 0.0.
        if total_power > 0:
            hf_ratio = hf_power / total_power
            # Gaussian centred at 0.20, σ=0.25 → score in (0, 1].
            # σ=0.25 keeps bright modern MP3 (hf_ratio ~0.40–0.50) above 0.60,
            # while still penalising extreme HF imbalance (muffled: ~0.05, over-bright: ~0.80).
            clarity = float(np.exp(-0.5 * ((hf_ratio - 0.20) / 0.25) ** 2))
        else:
            clarity = 0.0

        return float(np.clip(clarity, 0.0, 1.0))

    def _measure_warmth(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst tonal warmth (vocal/instrument body, not only bass weight).
        """
        # Probe-runs and ultra-low-energy snippets should not fail warmth gates.
        # They do not contain enough spectral evidence for a meaningful warmth score.
        rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2))) if len(audio) > 0 else 0.0
        if len(audio) < 256 or rms < 1e-5:
            return 0.70

        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        power = np.abs(fft) ** 2

        # Psychoakustische Wärme liegt bei Musik/Gesang im Körperband. Der alte
        # 60-250-Hz-Proxy wertete historische oder vokal-dominierte Quellen ohne
        # Bassfundament fälschlich als eiskalt, obwohl 200-800 Hz intakt waren.
        body_mask = (freqs >= 120) & (freqs <= 800)
        presence_mask = (freqs > 800) & (freqs <= 3000)
        broad_mask = (freqs >= 60) & (freqs <= min(6000, sr / 2))

        body_power: float = float(np.sum(power[body_mask]))
        presence_power: float = float(np.sum(power[presence_mask]))
        broad_power: float = float(np.sum(power[broad_mask]))

        if broad_power > 0:
            body_ratio = body_power / broad_power
            body_to_presence = body_power / (body_power + presence_power + 1e-20)
            ratio_score = float(np.clip(body_ratio / 0.35, 0.0, 1.0))
            balance_score = float(np.exp(-0.5 * ((body_to_presence - 0.55) / 0.35) ** 2))
            warmth = 0.65 * ratio_score + 0.35 * balance_score
        else:
            # Silence / 2-sample probe → return neutral pass value.
            # Returning 0.0 falsely triggers warmth gates during multi-pass
            # quick evaluation on dummy or sub-millisecond audio.
            warmth = 0.70

        return float(np.clip(warmth, 0, 1))

    def _measure_brightness(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst brightness (high-frequency presence).
        """
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        power = np.abs(fft) ** 2

        # HF energy (8-16 kHz)
        hf_mask = (freqs >= 8000) & (freqs <= 16000)
        hf_power: float = float(np.sum(power[hf_mask]))

        total_power: float = float(np.sum(power))

        if total_power > 0:
            hf_ratio = hf_power / total_power
            brightness = min(hf_ratio / 0.05, 1.0)
        else:
            brightness = 0.0

        return float(np.clip(brightness, 0, 1))

    def _measure_naturalness(self, audio: np.ndarray, sr: int, reference: np.ndarray | None = None) -> float:
        """
        Misst naturalness via octave-band spectral smoothness.

        Self-calibrating: when a reference (original/unprocessed) audio is
        provided, the threshold is derived from the reference's own residual
        variance.  This prevents false-zero scores for inherently bandlimited
        or noisy sources (cassette, shellac) that legitimately deviate from
        the 1/f model.

        Without reference: uses absolute threshold 0.80 (reasonable default).
        With reference:   threshold = max(0.80, ref_residual_var × 1.5).
                          "No worse than 50% above the source's natural state."

        Returns 0.0–1.0 (1.0 = envelope smoothness matches/exceeds reference).
        """
        residual_var = self._compute_octave_residual_var(audio)
        if residual_var is None:
            return 0.75

        # Calibrate threshold from reference if available
        if reference is not None:
            ref_rv = self._compute_octave_residual_var(reference)
            if ref_rv is not None and ref_rv > 0:
                threshold = max(0.80, ref_rv * 1.5)
            else:
                threshold = 0.80
        else:
            threshold = 0.80

        naturalness = max(0.0, 1.0 - min(residual_var / threshold, 1.0))
        return float(np.clip(naturalness, 0, 1))

    def _compute_octave_residual_var(self, audio: np.ndarray) -> float | None:
        """Compute octave-band residual variance (1/f-tilt removed)."""
        rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2))) if len(audio) > 0 else 0.0
        if len(audio) < 256 or rms < 1e-5:
            return None

        fft = np.fft.rfft(audio)
        power = np.abs(fft) ** 2

        n_bands = 8
        band_powers: list[float] = []
        for i in range(n_bands):
            start = len(power) // (2 ** (n_bands - i))
            end = len(power) // (2 ** (n_bands - i - 1))
            if end > start:
                band_powers.append(float(np.mean(power[start:end])))
            else:
                band_powers.append(0.0)

        _bp = np.array(band_powers, dtype=np.float64)
        if np.sum(_bp) <= 0:
            return None

        log_powers = np.log10(_bp + 1e-20)
        x = np.arange(n_bands, dtype=np.float64)
        coeffs = np.polyfit(x, log_powers, 1)
        tilt_line = np.polyval(coeffs, x)
        residual = log_powers - tilt_line
        return float(np.var(residual))

    def _measure_authenticity(self, audio: np.ndarray, sr: int) -> float:
        """
        Misst period authenticity (analog characteristics).
        """
        # Analog has:
        # - Slight noise floor
        # - Harmonic distortion
        # - Bandwidth limitations

        snr = self._estimate_snr(audio)
        thd = self._estimate_thd(audio, sr)
        bandwidth = self._measure_bandwidth(audio, sr)

        # Authentic analog: SNR 50-70 dB, THD 0.5-2%, BW < 18 kHz
        snr_score = 1.0 - abs(snr - 60) / 60  # Optimal at 60 dB
        thd_score = min(thd / 2.0, 1.0)  # Some distortion is good
        bw_score = 1.0 if bandwidth[1] < 18000 else 0.7

        authenticity = (snr_score + thd_score + bw_score) / 3

        return float(np.clip(authenticity, 0, 1))

    def _measure_bandwidth(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """
        Misst effective bandwidth (low, high).
        """
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / sr)
        power = np.abs(fft) ** 2

        # Noise floor (10th percentile)
        noise_floor = np.percentile(power, 10)

        # Low cutoff (first bin > noise floor)
        valid = power > noise_floor * 3
        low_cutoff = freqs[np.argmax(valid)] if np.any(valid) else 20.0

        # High cutoff (last bin > noise floor)
        high_cutoff = freqs[len(valid) - np.argmax(valid[::-1]) - 1] if np.any(valid) else sr / 2

        return (float(low_cutoff), float(high_cutoff))

    def _detect_artifacts(self, audio: np.ndarray, sr: int) -> tuple[bool, list[str]]:
        """
        Erkennt audio artifacts.
        """
        artifacts = []

        # Clipping
        if np.max(np.abs(audio)) > 0.99:
            artifacts.append("clipping")

        # DC offset
        dc = np.mean(audio)
        if abs(dc) > 0.01:
            artifacts.append("dc_offset")

        # Dropouts (sudden silence)
        rms_windows = []
        win_size = sr // 10  # 100ms windows
        for i in range(0, len(audio) - win_size, win_size):
            rms = np.sqrt(np.mean(audio[i : i + win_size] ** 2))
            rms_windows.append(rms)

        if len(rms_windows) > 0:
            median_rms = np.median(rms_windows)
            if np.any(np.array(rms_windows) < median_rms * 0.1):
                artifacts.append("dropouts")

        return (len(artifacts) > 0, artifacts)

    # === Helper Methods ===

    def _calculate_overall_score(
        self,
        snr: float,
        dr: float,
        thd: float,
        clarity: float,
        warmth: float,
        brightness: float,
        naturalness: float,
        *,
        perceptual_score: float | None = None,
    ) -> float:
        """
        Calculate overall quality score (0-100) — perzeptuell gewichtet.

        über technische Messwerte. SNR und THD sind unterhalb der Hörschwelle irrelevant.
        Wenn ein MUSHRA/OQS-Perceptual-Score verfügbar ist, wird dieser als
        direktes Hörer-Modell mit 35% gewichtet.

        Gewichte (Perceptual Mode, mit MUSHRA):
        - MUSHRA/OQS:      35%  ← direktes menschliches Hörmodell
        - Naturalness:     20%  ← Unnatürlichkeit wird sofort wahrgenommen
        - Warmth:          15%  ← "musikalisch, kraftvoll"
        - Clarity:         15%  ← Sprach-/Gesangsverständlichkeit
        - SNR:              5%  ← nur bei extrem schlechtem Rauschabstand relevant
        - Dynamic Range:    5%  ← Crest-Faktor ist perzeptuell relevanter
        - THD:              5%  ← unter ~1% unhörbar

        Gewichte (Technical Fallback, ohne MUSHRA):
        - SNR: 20%, DR: 15%, THD: 10%, Clarity: 20%, Warmth: 15%, Naturalness: 20%
        """
        snr_norm = min(snr / 80, 1.0)
        dr_norm = min(dr / 90, 1.0)
        thd_norm = 1.0 - min(thd / 5.0, 1.0)

        if perceptual_score is not None and perceptual_score > 0:
            # Perceptual Mode: MUSHRA als Ground Truth
            _mushra_norm = float(np.clip(perceptual_score / 100.0, 0.0, 1.0))
            score = (
                0.35 * _mushra_norm
                + 0.20 * naturalness
                + 0.15 * warmth
                + 0.15 * clarity
                + 0.05 * snr_norm
                + 0.05 * dr_norm
                + 0.05 * thd_norm
            )
        else:
            # Technical Fallback: reduziert technische, erhöht perzeptuelle Anteile
            score = (
                0.20 * snr_norm + 0.15 * dr_norm + 0.10 * thd_norm + 0.20 * clarity + 0.15 * warmth + 0.20 * naturalness
            )

        return float(score * 100)

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Bestimmt quality level from score."""
        if score >= 95:
            return QualityLevel.PRISTINE
        elif score >= 80:
            return QualityLevel.EXCELLENT
        elif score >= 60:
            return QualityLevel.GOOD
        elif score >= 40:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR

    def _calculate_confidence(self, audio: np.ndarray, snr: float) -> float:
        """
        Calculate confidence in quality estimation.

        Higher confidence for:
        - Higher SNR
        - Longer audio
        - Stable signal
        """
        # SNR confidence (higher SNR = more confident)
        snr_confidence = min(snr / 60, 1.0)

        # Length confidence (longer = more confident)
        duration = len(audio) / 48000  # Assume 48kHz
        length_confidence = min(duration / 2.0, 1.0)  # 2s = full confidence

        # Stability confidence (low variance = more confident)
        variance = np.var(audio)
        stability_confidence = 1.0 - min(variance, 1.0)  # type: ignore[operator]

        confidence = (snr_confidence + length_confidence + stability_confidence) / 3

        return float(np.clip(confidence, 0.1, 1.0))


class QualityPredictor:
    """
    Predict quality improvements from processing.

    Features:
    - Expected improvement prediction
    - Processing time estimation
    - Module recommendation
    - Quality gates
    """

    def __init__(self, quality_analyzer: QualityAnalyzer | None = None):
        """Initialisiert quality predictor."""
        self.analyzer = quality_analyzer or QualityAnalyzer()

        # Module improvement factors (heuristic, ML-ready)
        self.module_improvements = {
            "DCBlocker": {"snr": 2.0, "clarity": 0.05, "time": 0.01},
            "NoiseReduction": {"snr": 10.0, "clarity": 0.15, "time": 0.5},
            "ClickRemover": {"snr": 5.0, "clarity": 0.20, "time": 0.3},
            "CrackleSuppressor": {"snr": 4.0, "clarity": 0.15, "time": 0.2},
            "TapeSpecialist": {"snr": 8.0, "dr": 5.0, "clarity": 0.10, "time": 0.4},
            "DigitalRestoration": {"clarity": 0.25, "naturalness": 0.15, "time": 0.3},
            "Equalizer": {"warmth": 0.10, "brightness": 0.10, "time": 0.05},
            "Compressor": {"dr": -5.0, "clarity": -0.05, "time": 0.1},
        }

    def predict_improvement(
        self,
        current_quality: QualityEstimate,
        planned_modules: list[str],
        forensic_analysis: dict[str, Any] | None = None,
    ) -> ImprovementPrediction:
        """
        Predict improvement from processing.

        Args:
            current_quality: Current audio quality
            planned_modules: Modules to be applied
            forensic_analysis: Optional forensic analysis

        Returns:
            ImprovementPrediction with expected improvements
        """
        # Initialize improvements
        snr_improvement = 0.0
        dr_improvement = 0.0
        thd_reduction = 0.0
        clarity_improvement = 0.0
        warmth_improvement = 0.0
        brightness_improvement = 0.0
        naturalness_improvement = 0.0
        processing_time = 0.0

        # Sum improvements from each module
        for module_name in planned_modules:
            if module_name in self.module_improvements:
                factors = self.module_improvements[module_name]

                # Apply diminishing returns for low-quality input
                quality_factor = current_quality.overall_score / 100

                snr_improvement += factors.get("snr", 0) * (1 - quality_factor * 0.5)
                dr_improvement += factors.get("dr", 0) * (1 - quality_factor * 0.5)
                clarity_improvement += factors.get("clarity", 0)
                warmth_improvement += factors.get("warmth", 0)
                brightness_improvement += factors.get("brightness", 0)
                naturalness_improvement += factors.get("naturalness", 0)
                processing_time += factors.get("time", 0.1)

        # Forensic-based adjustments
        if forensic_analysis:
            quality_assessment = forensic_analysis.get("quality_assessment", "GOOD")
            if quality_assessment == "POOR":
                # More improvement possible
                snr_improvement *= 1.3
                clarity_improvement *= 1.2
            elif quality_assessment == "EXCELLENT":
                # Less improvement possible
                snr_improvement *= 0.7
                clarity_improvement *= 0.8

        # Calculate final metrics
        final_snr = min(current_quality.snr_db + snr_improvement, 100)
        final_dr = min(current_quality.dynamic_range_db + dr_improvement, 120)
        final_thd = max(current_quality.thd_percent - thd_reduction, 0)
        final_clarity = min(current_quality.clarity + clarity_improvement, 1.0)

        # Confidence (based on current quality and number of modules)
        confidence = self._calculate_prediction_confidence(current_quality, len(planned_modules))

        # Quality gates (target values for early stopping)
        quality_gates = {
            "snr_db": 70.0,  # Stop if SNR > 70 dB
            "clarity": 0.85,  # Stop if clarity > 85%
            "overall_score": 90.0,  # Stop if overall > 90
        }

        return ImprovementPrediction(
            snr_improvement_db=snr_improvement,
            dynamic_range_improvement_db=dr_improvement,
            thd_reduction_percent=thd_reduction,
            clarity_improvement=clarity_improvement,
            final_snr_db=final_snr,
            final_dynamic_range_db=final_dr,
            final_thd_percent=final_thd,
            final_clarity=final_clarity,
            warmth_improvement=warmth_improvement,
            brightness_improvement=brightness_improvement,
            naturalness_improvement=naturalness_improvement,
            confidence=confidence,
            estimated_processing_time_sec=processing_time,
            recommended_modules=self._recommend_modules(current_quality, forensic_analysis),
            quality_gates=quality_gates,
        )

    def _calculate_prediction_confidence(self, quality: QualityEstimate, num_modules: int) -> float:
        """Calculate confidence in prediction."""
        # Higher confidence for:
        # - Higher current quality (easier to predict)
        # - Fewer modules (less uncertainty)

        quality_factor = quality.confidence
        module_factor = 1.0 - min(num_modules / 10, 0.5)  # Penalty for many modules

        confidence = (quality_factor + module_factor) / 2

        return float(np.clip(confidence, 0.3, 0.95))

    def _recommend_modules(self, quality: QualityEstimate, forensic_analysis: dict[str, Any] | None) -> list[str]:
        """Recommend modules based on quality analysis."""
        recommended = []

        # Always start with DC blocker
        recommended.append("DCBlocker")

        # Based on quality level
        if quality.quality_level in [QualityLevel.POOR, QualityLevel.FAIR]:
            recommended.append("NoiseReduction")

        # Based on artifacts
        if "clicks" in quality.artifact_types:
            recommended.append("ClickRemover")
        if "crackle" in quality.artifact_types:
            recommended.append("CrackleSuppressor")

        # Based on metrics
        if quality.snr_db < 50:
            recommended.append("NoiseReduction")
        if quality.clarity < 0.6:
            recommended.append("Equalizer")
        if quality.warmth < 0.5:
            recommended.append("Equalizer")

        # Forensic-based
        if forensic_analysis:
            medium = forensic_analysis.get("medium_type", "").upper()
            if medium in ["VINYL", "TAPE", "CASSETTE"]:
                recommended.append("TapeSpecialist")
            if medium in ["LOSSY", "MP3", "AAC"]:
                recommended.append("DigitalRestoration")

        # Remove duplicates, preserve order
        seen = set()
        unique = []
        for module in recommended:
            if module not in seen:
                seen.add(module)
                unique.append(module)

        return unique


class QualityPredictionSystem:
    """
    Complete Quality Prediction System.

    Ferrari Edition - Premium Qualitäts-Management:
    - Pre-processing quality estimation
    - Improvement prediction
    - Result validation
    - Quality gates
    """

    VERSION = "2.0.0"

    def __init__(self):
        """Initialisiert quality prediction system."""
        self.analyzer = QualityAnalyzer()
        self.predictor = QualityPredictor(self.analyzer)

        logger.info("QualityPredictionSystem initialized (v2.0.0 Ferrari Edition)")

    def estimate_quality(self, audio: np.ndarray, sample_rate: int) -> QualityEstimate:
        """
        Schätzt audio quality before processing.

        Args:
            audio: Input audio
            sample_rate: Sample rate

        Returns:
            QualityEstimate with comprehensive metrics
        """
        return self.analyzer.analyze_quality(audio, sample_rate)

    def predict_processing_outcome(
        self,
        audio: np.ndarray,
        sample_rate: int,
        planned_modules: list[str],
        forensic_analysis: dict[str, Any] | None = None,
    ) -> tuple[QualityEstimate, ImprovementPrediction]:
        """
        Predict outcome of processing.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            planned_modules: Modules to be applied
            forensic_analysis: Optional forensic analysis

        Returns:
            (current_quality, improvement_prediction)
        """
        # Analyze current quality
        current_quality = self.analyzer.analyze_quality(audio, sample_rate)

        # Predict improvement
        improvement = self.predictor.predict_improvement(current_quality, planned_modules, forensic_analysis)

        logger.info(
            f"Quality Prediction: {current_quality.overall_score:.1f}/100 → "
            f"{current_quality.overall_score + improvement.snr_improvement_db * 0.5:.1f}/100 "
            f"(confidence={improvement.confidence:.2f})"
        )

        return (current_quality, improvement)

    def validate_prediction(
        self, predicted: ImprovementPrediction, actual_audio: np.ndarray, sample_rate: int
    ) -> QualityValidation:
        """
        Validiert prediction against actual result.

        Args:
            predicted: Predicted improvement
            actual_audio: Actual processed audio
            sample_rate: Sample rate

        Returns:
            QualityValidation with accuracy metrics
        """
        # Analyze actual quality
        actual_quality = self.analyzer.analyze_quality(actual_audio, sample_rate)

        # Prediction accuracy
        snr_accuracy = abs(predicted.final_snr_db - actual_quality.snr_db)
        clarity_accuracy = abs(predicted.final_clarity - actual_quality.clarity)

        # Prediction score (0-100)
        snr_score = 100 * (1 - min(snr_accuracy / 20, 1.0))  # ±20 dB tolerance
        clarity_score = 100 * (1 - clarity_accuracy)  # ±100% tolerance

        prediction_score = (snr_score + clarity_score) / 2

        logger.info(
            f"Prediction Validation: {prediction_score:.1f}/100 accuracy "
            f"(SNR: ±{snr_accuracy:.1f} dB, Clarity: ±{clarity_accuracy:.2f})"
        )

        # Create dummy predicted quality for validation
        predicted_quality = QualityEstimate(
            overall_score=predicted.final_snr_db * 0.8,  # Simplified
            quality_level=QualityLevel.GOOD,
            snr_db=predicted.final_snr_db,
            dynamic_range_db=predicted.final_dynamic_range_db,
            thd_percent=predicted.final_thd_percent,
            clarity=predicted.final_clarity,
            warmth=0.5,
            brightness=0.5,
            naturalness=0.5,
            authenticity=0.5,
            confidence=predicted.confidence,
            bandwidth_hz=(20, 20000),
            has_artifacts=False,
        )

        return QualityValidation(
            snr_accuracy_db=snr_accuracy,
            clarity_accuracy=clarity_accuracy,
            time_accuracy_sec=0.0,  # Not tracked yet
            prediction_score=prediction_score,
            predicted_quality=predicted_quality,
            actual_quality=actual_quality,
            improvement=predicted,
        )

    def check_quality_gate(self, current_quality: QualityEstimate, target_gates: dict[str, float]) -> tuple[bool, str]:
        """
        Prüft if quality gates are met (early stopping).

        Args:
            current_quality: Current audio quality
            target_gates: Target values for metrics

        Returns:
            (gate_met, reason)
        """
        failures: list[str] = []

        if "snr_db" in target_gates and current_quality.snr_db < target_gates["snr_db"]:
            failures.append(f"SNR {current_quality.snr_db:.1f} < {target_gates['snr_db']:.1f} dB")

        if "clarity" in target_gates and current_quality.clarity < target_gates["clarity"]:
            failures.append(f"Clarity {current_quality.clarity:.2f} < {target_gates['clarity']:.2f}")

        if "overall_score" in target_gates and current_quality.overall_score < target_gates["overall_score"]:
            failures.append(f"Overall {current_quality.overall_score:.1f} < {target_gates['overall_score']:.1f}")

        if failures:
            return (False, "Quality gates not met, continue processing")

        if target_gates:
            return (True, "All quality gates met")

        return (False, "No quality gates configured")


# === Convenience Functions ===


def create_quality_prediction_system() -> QualityPredictionSystem:
    """Erstellt a quality prediction system with default settings."""
    return QualityPredictionSystem()
