"""
formant_shifter_safety.py - HIPS-Compliant Formant Shifter Safety Wrapper

Wraps formant shifting DSP operations with comprehensive safety checks:
- Voice presence validation
- Formant bounds enforcement (±500 Hz safe range)
- Timbre preservation verification
- Singer's formant protection (2.8-3.5 kHz)

This ensures formant shifting never degrades voice quality or creates
unnatural artifacts.

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
    compute_spectral_centroid,
    validate_audio_basic,
)

# ============================================================================
# FORMANT ANALYSIS UTILITIES
# ============================================================================


def detect_formants_lpc(audio: np.ndarray, sr: int, n_formants: int = 5) -> np.ndarray:
    """
    Erkennt formants using Linear Predictive Coding (LPC).

    Args:
        audio: Input audio (mono)
        sr: Sample rate
        n_formants: Number of formants to detect

    Returns:
        Array of formant frequencies [F1, F2, F3, F4, F5] in Hz
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Pre-emphasis to boost high frequencies
    pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # LPC order (rule of thumb: 2 + sr/1000, minimum 16 per §VERBOTEN)
    lpc_order = max(16, min(40, int(2 + sr / 1000)))

    # Compute LPC coefficients

    # Autocorrelation method for LPC
    a = _lpc_autocorr(pre_emphasized, lpc_order)

    # Find roots of LPC polynomial
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0]  # Keep only positive frequencies

    # Convert roots to frequencies
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))

    # Sort by frequency
    freqs = np.sort(freqs)

    # Take first n_formants
    if len(freqs) < n_formants:
        # Pad with zeros if not enough formants detected
        formants = np.zeros(n_formants)
        formants[: len(freqs)] = freqs
    else:
        formants = freqs[:n_formants]

    return formants


def _lpc_autocorr(signal: np.ndarray, order: int) -> np.ndarray:
    """Berechnet LPC coefficients using autocorrelation method."""
    # Compute autocorrelation — FFT-based O(N log N)
    from backend.core.core_utils import fft_autocorr

    r = fft_autocorr(signal, max_lag=order)

    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]

    for i in range(1, order + 1):
        lambda_val = -np.sum(a[:i] * r[i:0:-1]) / e
        a[1 : i + 1] += lambda_val * a[i - 1 :: -1]
        a[i] = lambda_val
        e *= 1 - lambda_val**2

    return a


def compute_formant_dispersion(formants: np.ndarray) -> float:
    """
    Berechnet formant dispersion (spacing between formants).
    Higher dispersion = clearer formant structure.

    Args:
        formants: Array of formant frequencies

    Returns:
        Dispersion score (0.0-1.0)
    """
    if len(formants) < 2:
        return 0.0

    # Filter out zero formants
    valid_formants = formants[formants > 0]

    if len(valid_formants) < 2:
        return 0.0

    # Compute spacing
    spacings = np.diff(valid_formants)

    # Good formant structure has spacing of ~1000-1500 Hz
    # Score based on how close to ideal
    ideal_spacing = 1200  # Hz
    dispersion = 1.0 - np.mean(np.abs(spacings - ideal_spacing)) / ideal_spacing
    dispersion = np.clip(dispersion, 0.0, 1.0)

    return float(dispersion)


def detect_voice_presence(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Erkennt if audio contains voice/vocals.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (is_voice, confidence): Voice detection result and confidence
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Voice characteristics:
    # 1. Fundamental frequency in 80-400 Hz range (speech/singing)
    # 2. Strong formant structure (F1: 300-1000 Hz, F2: 800-3000 Hz)
    # 3. Harmonic structure

    # Compute spectrogram
    f, _t, Sxx = signal.spectrogram(audio, sr, nperseg=2048)

    # Check for energy in voice frequency range (80-400 Hz)
    voice_band = (f >= 80) & (f <= 400)
    voice_energy = np.mean(Sxx[voice_band, :])

    total_energy = np.mean(Sxx)
    voice_ratio = voice_energy / (total_energy + 1e-8)

    # Check for formant energy (300-3000 Hz)
    formant_band = (f >= 300) & (f <= 3000)
    formant_energy = np.mean(Sxx[formant_band, :])
    formant_ratio = formant_energy / (total_energy + 1e-8)

    # Simple heuristic: voice if both ratios are significant
    is_voice = (voice_ratio > 0.1) and (formant_ratio > 0.3)
    confidence = (voice_ratio + formant_ratio) / 2

    return is_voice, float(np.clip(confidence, 0.0, 1.0))


def detect_singers_formant(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Erkennt Singer's Formant (2.8-3.5 kHz resonance).
    Important for trained singers - must be preserved.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (has_singers_formant, strength): Detection result and strength
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Singer's formant range: 2.8-3.5 kHz
    sf_band = (freqs >= 2800) & (freqs <= 3500)

    if not np.any(sf_band):
        return False, 0.0

    sf_energy = np.mean(spectrum[sf_band])

    # Compare to surrounding bands
    lower_band = (freqs >= 2000) & (freqs < 2800)
    upper_band = (freqs > 3500) & (freqs <= 4500)

    surrounding_energy = (np.mean(spectrum[lower_band]) + np.mean(spectrum[upper_band])) / 2

    # Singer's formant should be prominently elevated
    if surrounding_energy == 0:
        return False, 0.0

    ratio = sf_energy / surrounding_energy

    has_sf = ratio > 1.5  # At least 50% elevated
    strength = float(np.clip(ratio / 3.0, 0.0, 1.0))  # Normalize

    return has_sf, strength


# ============================================================================
# FORMANT SHIFTER SAFETY WRAPPER
# ============================================================================


class FormantShifterSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for formant shifting.

    Ensures:
    - Only processes audio with clear voice presence
    - Enforces safe formant shift range (±500 Hz)
    - Preserves voice timbre and identity
    - Protects Singer's Formant when present
    - No metallic or robotic artifacts introduced
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        max_shift_hz: float = 500.0,
        min_voice_confidence: float = 0.6,
    ):
        """
        Initialisiert Formant Shifter Safety Wrapper.

        Args:
            processor_func: Formant shifting function (audio, sr, shift_hz) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            max_shift_hz: Maximum allowed formant shift (Hz)
            min_voice_confidence: Minimum voice confidence to proceed
        """
        super().__init__(
            module_name="FormantShifter",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=min_voice_confidence,
            quality_threshold=0.7,
        )

        self.max_shift_hz = max_shift_hz
        self.min_voice_confidence = min_voice_confidence

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for formant shifting."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Check for voice presence
        is_voice, voice_conf = detect_voice_presence(audio, sr)
        metadata["voice_detected"] = is_voice
        metadata["voice_confidence"] = voice_conf  # type: ignore[assignment]

        if not is_voice:
            return PreCheckResult(passed=False, confidence=voice_conf, reasons=["No voice detected in audio"])

        if voice_conf < self.min_voice_confidence:
            return PreCheckResult(
                passed=False, confidence=voice_conf, reasons=[f"Voice confidence too low: {voice_conf:.2f}"]
            )

        # Validate shift parameter
        shift_hz = params.get("shift_hz", 0.0)

        if abs(shift_hz) > self.max_shift_hz:
            return PreCheckResult(
                passed=False,
                confidence=voice_conf,
                reasons=[f"Shift too large: {shift_hz} Hz (max ±{self.max_shift_hz} Hz)"],
            )

        # Detect formants
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        formants = detect_formants_lpc(audio_mono, sr, n_formants=5)
        metadata["formants_hz"] = formants.tolist()

        dispersion = compute_formant_dispersion(formants)
        metadata["formant_dispersion"] = dispersion  # type: ignore[assignment]

        if dispersion < 0.3:
            warnings.append("Weak formant structure detected - proceed with caution")

        # Check for Singer's Formant
        has_sf, sf_strength = detect_singers_formant(audio_mono, sr)
        metadata["has_singers_formant"] = has_sf
        metadata["singers_formant_strength"] = sf_strength  # type: ignore[assignment]

        if has_sf:
            warnings.append(
                f"Singer's Formant detected (strength: {sf_strength:.2f}). Will be protected during processing."
            )

        return PreCheckResult(passed=True, confidence=voice_conf, warnings=warnings, metadata=metadata)

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in formant shifting for this audio."""
        # Base confidence from voice detection
        voice_conf = pre_check.metadata.get("voice_confidence", 0.5)

        # Formant structure quality
        dispersion = pre_check.metadata.get("formant_dispersion", 0.5)

        # Higher confidence for clear formants
        confidence = voice_conf * 0.6 + dispersion * 0.4

        # Penalty for very large shifts (less confident in extreme processing)
        shift_hz = abs(params.get("shift_hz", 0.0))
        shift_penalty = shift_hz / self.max_shift_hz  # 0.0-1.0
        confidence *= 1.0 - 0.3 * shift_penalty  # Up to 30% penalty

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after formant shifting."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # Mono for analysis
        if original.ndim > 1:
            orig_mono = np.mean(original, axis=0)
            proc_mono = np.mean(processed, axis=0)
        else:
            orig_mono = original
            proc_mono = processed

        # 1. Check voice presence preserved
        is_voice_after, voice_conf_after = detect_voice_presence(proc_mono, sr)
        metrics["voice_confidence_after"] = voice_conf_after

        if not is_voice_after:
            issues.append("Voice characteristics lost after processing")

        # 2. Energy preservation
        energy_ratio = compute_energy_ratio(orig_mono, proc_mono)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio < 0.7 or energy_ratio > 1.3:
            side_effects.append(f"Significant energy change: {energy_ratio:.2%}")

        # 3. Correlation (signal similarity)
        correlation = compute_correlation(orig_mono, proc_mono)
        metrics["correlation"] = correlation

        if correlation < 0.8:
            side_effects.append(f"Low correlation with original: {correlation:.2f}")

        # 4. Formant shift verification
        formants_before = detect_formants_lpc(orig_mono, sr)
        formants_after = detect_formants_lpc(proc_mono, sr)

        metrics["formants_before"] = formants_before.tolist()
        metrics["formants_after"] = formants_after.tolist()

        # Check if formants actually shifted
        shift_hz = params.get("shift_hz", 0.0)
        actual_shift = np.mean(formants_after[:3]) - np.mean(formants_before[:3])
        metrics["actual_formant_shift_hz"] = float(actual_shift)
        metrics["requested_shift_hz"] = shift_hz

        shift_error = abs(actual_shift - shift_hz)
        if shift_error > 100:  # More than 100 Hz error
            side_effects.append(f"Formant shift error: requested {shift_hz} Hz, got {actual_shift:.0f} Hz")

        # 5. Singer's Formant preservation (if present)
        has_sf_before = params.get("__pre_check_singers_formant", False)
        if has_sf_before:
            has_sf_after, _sf_strength_after = detect_singers_formant(proc_mono, sr)
            metrics["singers_formant_preserved"] = has_sf_after

            if not has_sf_after:
                issues.append("Singer's Formant was lost during processing")

        # 6. Check for metallic/robotic artifacts
        spectral_centroid_before = compute_spectral_centroid(orig_mono, sr)
        spectral_centroid_after = compute_spectral_centroid(proc_mono, sr)

        metrics["spectral_centroid_before"] = spectral_centroid_before
        metrics["spectral_centroid_after"] = spectral_centroid_after

        centroid_ratio = spectral_centroid_after / spectral_centroid_before
        if centroid_ratio > 1.5 or centroid_ratio < 0.67:
            side_effects.append(f"Significant brightness change: {centroid_ratio:.2f}x")

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

        # Voice preservation
        voice_conf = post_check.metrics.get("voice_confidence_after", 0.0)
        scores.append(voice_conf)

        # Energy preservation
        energy_ratio = post_check.metrics.get("energy_ratio", 1.0)
        energy_score = 1.0 - abs(1.0 - energy_ratio)  # Closer to 1.0 = better
        scores.append(np.clip(energy_score, 0.0, 1.0))

        # Correlation
        correlation = post_check.metrics.get("correlation", 0.0)
        scores.append(correlation)

        # Formant shift accuracy
        requested = post_check.metrics.get("requested_shift_hz", 0.0)
        actual = post_check.metrics.get("actual_formant_shift_hz", 0.0)

        if requested != 0:
            shift_accuracy = 1.0 - min(abs(actual - requested) / abs(requested), 1.0)
            scores.append(shift_accuracy)

        # Weighted average
        quality = np.mean(scores)

        return float(np.clip(quality, 0.0, 1.0))
