"""
deesser_safety.py - HIPS-Compliant De-Esser Safety Wrapper

Wraps de-essing DSP operations with comprehensive safety checks:
- Sibilant presence detection
- Vocal profile validation (female/male/child)
- Consonant clarity preservation
- Intelligibility protection
- No over-processing artifacts

This ensures de-essing reduces harshness without destroying speech clarity.

Author: AURIK Team
Version: 1.0.0
Date: 7. Februar 2026
Phase: 1 Week 5-6
"""

from pathlib import Path

import numpy as np
import scipy.signal as signal

from .safety_wrapper_template import (
    BaseSafetyWrapper,
    PostCheckResult,
    PreCheckResult,
    compute_correlation,
    compute_energy_ratio,
    validate_audio_basic,
)

# ============================================================================
# SIBILANCE ANALYSIS UTILITIES
# ============================================================================


def detect_sibilance(audio: np.ndarray, sr: int) -> tuple[bool, float, dict[str, float]]:
    """
    Erkennt sibilance in audio.

    Sibilants (s, sh, ch, z) are concentrated in 4-10 kHz range.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate

    Returns:
        (has_sibilance, intensity, band_energies): Detection result
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Sibilant frequency bands
    s_band = (freqs >= 4000) & (freqs <= 10000)  # Primary sibilance
    lower_band = (freqs >= 2000) & (freqs < 4000)  # Context
    upper_band = (freqs > 10000) & (freqs <= 16000)  # Harmonics

    s_energy = np.mean(spectrum[s_band]) if np.any(s_band) else 0.0
    lower_energy = np.mean(spectrum[lower_band]) if np.any(lower_band) else 0.0
    upper_energy = np.mean(spectrum[upper_band]) if np.any(upper_band) else 0.0
    total_energy = np.mean(spectrum)

    # Sibilance detection: s-band should be elevated
    if total_energy == 0:
        return False, 0.0, {}

    s_ratio = s_energy / total_energy

    # Has sibilance if s-band is prominent
    has_sibilance = s_ratio > 0.15
    intensity = float(np.clip(s_ratio / 0.3, 0.0, 1.0))  # Normalize to 0-1

    band_energies = {
        "sibilant_band_energy": float(s_energy),
        "lower_band_energy": float(lower_energy),
        "upper_band_energy": float(upper_energy),
        "sibilance_ratio": float(s_ratio),
    }

    return has_sibilance, intensity, band_energies


def classify_vocal_profile(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Classify vocal profile (female/male/child).

    Different profiles have different sibilant characteristics:
    - Female: 7-11 kHz
    - Male: 5-9 kHz
    - Child: 9-13 kHz

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (profile, confidence): Vocal profile and confidence
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Fundamental frequency estimation (simplified)
    # Look for dominant frequency in 80-400 Hz range
    f0_band = (freqs >= 80) & (freqs <= 400)

    if not np.any(f0_band):
        return "male", 0.5  # Default

    f0_spectrum = spectrum[f0_band]
    f0_freqs = freqs[f0_band]

    f0 = f0_freqs[np.argmax(f0_spectrum)]

    # Classify based on F0
    # Male: 85-180 Hz
    # Female: 165-255 Hz
    # Child: 250-400 Hz

    if f0 < 160:
        profile = "male"
        confidence = 1.0 - abs(f0 - 130) / 130
    elif f0 < 240:
        profile = "female"
        confidence = 1.0 - abs(f0 - 200) / 200
    else:
        profile = "child"
        confidence = 1.0 - abs(f0 - 320) / 320

    confidence = float(np.clip(confidence, 0.3, 1.0))

    return profile, confidence


def measure_consonant_clarity(audio: np.ndarray, sr: int) -> float:
    """
    Misst consonant clarity (high-frequency transient energy).

    Consonants have sharp transients in 2-8 kHz range.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Clarity score (0.0-1.0)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # High-pass filter at 2 kHz to isolate consonants
    sos = signal.butter(4, 2000, "hp", fs=sr, output="sos")
    consonant_band = signal.sosfilt(sos, audio)

    # Measure transient energy using envelope
    analytic = signal.hilbert(consonant_band)
    analytic_arr = np.asarray(analytic, dtype=np.complex128)
    envelope = np.abs(analytic_arr)

    # Transient detection: look for sharp attacks
    diff = np.diff(envelope)
    transient_strength = np.mean(np.abs(diff))

    # Normalize to 0-1 range (empirical)
    clarity = np.clip(transient_strength * 1000, 0.0, 1.0)

    return float(clarity)


def compute_intelligibility_score(audio: np.ndarray, sr: int) -> float:
    """
    Berechnet speech intelligibility score.

    Based on energy distribution across critical bands for speech.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Intelligibility score (0.0-1.0)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Critical bands for speech intelligibility
    bands = [
        (300, 1000),  # Fundamental + F1
        (1000, 3000),  # F2, F3
        (3000, 6000),  # F4, fricatives
        (6000, 10000),  # Sibilants
    ]

    band_energies = []
    for low, high in bands:
        band = (freqs >= low) & (freqs <= high)
        if np.any(band):
            energy = np.mean(spectrum[band])
            band_energies.append(energy)
        else:
            band_energies.append(0.0)

    # Intelligibility requires balanced energy across bands
    # Too much imbalance = poor intelligibility

    if max(band_energies) == 0:
        return 0.0

    # Normalize
    band_energies = np.array(band_energies) / max(band_energies)

    # Good intelligibility: all bands have reasonable energy
    # Score based on minimum band energy
    intelligibility = np.min(band_energies)

    return float(np.clip(intelligibility, 0.0, 1.0))


# ============================================================================
# DE-ESSER SAFETY WRAPPER
# ============================================================================


class DeEsserSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for de-essing.

    Ensures:
    - Only processes audio with detected sibilance
    - Adapts to vocal profile (female/male/child)
    - Preserves consonant clarity and intelligibility
    - No over-processing (lisping artifacts)
    - No spectral holes in high frequencies
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        min_sibilance_intensity: float = 0.2,
        min_intelligibility: float = 0.4,
    ):
        """
        Initialisiert De-Esser Safety Wrapper.

        Args:
            processor_func: De-essing function (audio, sr, profile, depth_db) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            min_sibilance_intensity: Minimum sibilance to process
            min_intelligibility: Minimum intelligibility threshold
        """
        super().__init__(
            module_name="DeEsser",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=0.5,
            quality_threshold=0.65,
        )

        self.min_sibilance_intensity = min_sibilance_intensity
        self.min_intelligibility = min_intelligibility

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for de-essing."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Detect sibilance
        has_sibilance, intensity, band_energies = detect_sibilance(audio, sr)
        metadata["has_sibilance"] = has_sibilance
        metadata["sibilance_intensity"] = intensity  # type: ignore[assignment]
        metadata.update(band_energies)  # type: ignore[arg-type]

        if not has_sibilance:
            return PreCheckResult(passed=False, confidence=0.0, reasons=["No sibilance detected in audio"])

        if intensity < self.min_sibilance_intensity:
            return PreCheckResult(
                passed=False,
                confidence=intensity,
                reasons=[f"Sibilance intensity too low: {intensity:.2f} (min {self.min_sibilance_intensity})"],
            )

        # Classify vocal profile
        profile, profile_conf = classify_vocal_profile(audio, sr)
        metadata["vocal_profile"] = profile  # type: ignore[assignment]
        metadata["profile_confidence"] = profile_conf  # type: ignore[assignment]

        if profile_conf < 0.5:
            warnings.append(f"Uncertain vocal profile classification: {profile} (confidence {profile_conf:.2f})")

        # Measure initial consonant clarity
        clarity_before = measure_consonant_clarity(audio, sr)
        metadata["consonant_clarity_before"] = clarity_before  # type: ignore[assignment]

        # Measure initial intelligibility
        intelligibility_before = compute_intelligibility_score(audio, sr)
        metadata["intelligibility_before"] = intelligibility_before  # type: ignore[assignment]

        if intelligibility_before < self.min_intelligibility:
            warnings.append(
                f"Low initial intelligibility: {intelligibility_before:.2f}. De-essing may further reduce clarity."
            )

        # Validate depth parameter
        depth_db = params.get("depth_db", 0.0)

        if depth_db < 0:
            return PreCheckResult(
                passed=False, confidence=intensity, reasons=[f"Invalid depth: {depth_db} dB (must be positive)"]
            )

        if depth_db > 20:
            warnings.append(f"Very aggressive de-essing: {depth_db} dB. Risk of over-processing.")

        return PreCheckResult(passed=True, confidence=intensity, warnings=warnings, metadata=metadata)

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in de-essing for this audio."""
        # Base confidence from sibilance detection
        sibilance_intensity = pre_check.metadata.get("sibilance_intensity", 0.5)

        # Profile classification confidence
        profile_conf = pre_check.metadata.get("profile_confidence", 0.5)

        # Higher confidence = clearer sibilance + clear vocal profile
        confidence = sibilance_intensity * 0.6 + profile_conf * 0.4

        # Penalty for very aggressive processing
        depth_db = params.get("depth_db", 0.0)
        if depth_db > 12:
            depth_penalty = (depth_db - 12) / 20  # Up to 40% penalty
            confidence *= 1.0 - 0.4 * depth_penalty

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after de-essing."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # 1. Check sibilance reduction
        _has_sib_before, intensity_before, _ = detect_sibilance(original, sr)
        _has_sib_after, intensity_after, _ = detect_sibilance(processed, sr)

        metrics["sibilance_before"] = intensity_before
        metrics["sibilance_after"] = intensity_after

        if intensity_after >= intensity_before:
            side_effects.append(f"Sibilance not reduced: {intensity_before:.2f} -> {intensity_after:.2f}")

        reduction = intensity_before - intensity_after
        metrics["sibilance_reduction"] = float(reduction)

        # 2. Check consonant clarity preservation
        clarity_before = measure_consonant_clarity(original, sr)
        clarity_after = measure_consonant_clarity(processed, sr)

        metrics["consonant_clarity_before"] = clarity_before
        metrics["consonant_clarity_after"] = clarity_after

        clarity_loss = clarity_before - clarity_after
        if clarity_loss > 0.3:  # More than 30% clarity loss
            issues.append(f"Excessive consonant clarity loss: {clarity_loss:.2f}")

        # 3. Check intelligibility preservation
        intelligibility_before = compute_intelligibility_score(original, sr)
        intelligibility_after = compute_intelligibility_score(processed, sr)

        metrics["intelligibility_before"] = intelligibility_before
        metrics["intelligibility_after"] = intelligibility_after

        intelligibility_loss = intelligibility_before - intelligibility_after
        if intelligibility_loss > 0.2:  # More than 20% loss
            issues.append(f"Excessive intelligibility loss: {intelligibility_loss:.2f}")

        # 4. Energy preservation in non-sibilant bands
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio < 0.8 or energy_ratio > 1.05:
            side_effects.append(f"Unexpected energy change: {energy_ratio:.2%}")

        # 5. Correlation (overall signal similarity)
        correlation = compute_correlation(original.flatten(), processed.flatten())
        metrics["correlation"] = correlation

        if correlation < 0.85:
            side_effects.append(f"Low correlation with original: {correlation:.2f}")

        # 6. Check for spectral holes
        orig_spectrum = np.abs(np.fft.rfft(original.flatten()))
        proc_spectrum = np.abs(np.fft.rfft(processed.flatten()))

        # Look for frequencies that disappeared
        spectral_ratio = proc_spectrum / (orig_spectrum + 1e-8)
        holes = np.sum(spectral_ratio < 0.1) / len(spectral_ratio)
        metrics["spectral_holes_ratio"] = float(holes)

        if holes > 0.05:  # More than 5% of spectrum gone
            issues.append(f"Spectral holes detected: {holes:.1%} of frequencies removed")

        passed = len(issues) == 0

        return PostCheckResult(
            passed=passed,
            quality_score=0.0,  # Will be computed separately
            issues=issues,
            side_effects=side_effects,
            metrics=metrics,
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet overall quality score."""
        scores = []

        # Sibilance reduction effectiveness
        sib_before = post_check.metrics.get("sibilance_before", 1.0)
        sib_after = post_check.metrics.get("sibilance_after", 1.0)

        if sib_before > 0:
            reduction_score = (sib_before - sib_after) / sib_before
            scores.append(np.clip(reduction_score, 0.0, 1.0))

        # Consonant clarity preservation
        clarity_before = post_check.metrics.get("consonant_clarity_before", 1.0)
        clarity_after = post_check.metrics.get("consonant_clarity_after", 1.0)

        clarity_preservation = clarity_after / (clarity_before + 1e-8)
        scores.append(np.clip(clarity_preservation, 0.0, 1.0))

        # Intelligibility preservation
        intel_before = post_check.metrics.get("intelligibility_before", 1.0)
        intel_after = post_check.metrics.get("intelligibility_after", 1.0)

        intel_preservation = intel_after / (intel_before + 1e-8)
        scores.append(np.clip(intel_preservation, 0.0, 1.0))

        # Correlation
        correlation = post_check.metrics.get("correlation", 0.0)
        scores.append(correlation)

        # Weighted average (emphasize intelligibility preservation)
        weights = [0.3, 0.3, 0.3, 0.1]  # reduction, clarity, intel, correlation
        quality = np.average(scores, weights=weights)

        return float(np.clip(quality, 0.0, 1.0))
