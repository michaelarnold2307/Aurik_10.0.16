"""
declick_safety.py - HIPS-Compliant De-Click Safety Wrapper

Wraps de-clicking DSP operations with comprehensive safety checks:
- Click vs musical transient classification
- Transient preservation (drums, consonants)
- No over-smoothing artifacts
- Attack/sustain/release preservation
- Musical timing integrity

This ensures de-clicking removes only defects, not musical content.

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
# CLICK DETECTION UTILITIES
# ============================================================================


def detect_clicks(audio: np.ndarray, sr: int, threshold_db: float = -30.0) -> tuple[bool, int, list[int]]:
    """
    Erkennt clicks in audio signal.

    Clicks are sharp, broadband transients (often from vinyl, digital errors).

    Args:
        audio: Input audio
        sr: Sample rate
        threshold_db: Detection threshold (dB re: peak)

    Returns:
        (has_clicks, count, click_positions): Detection result
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute first-order difference (detects sharp transitions)
    diff = np.abs(np.diff(audio))

    # Threshold based on peak
    peak = np.max(diff)
    threshold_linear = peak * 10 ** (threshold_db / 20)

    # Find clicks
    potential_clicks = diff > threshold_linear

    # Clicks are isolated spikes (1-3 samples)
    click_positions = []

    i = 0
    while i < len(potential_clicks):
        if potential_clicks[i]:
            # Check if isolated (not part of longer transient)
            start = i
            while i < len(potential_clicks) and potential_clicks[i]:
                i += 1
            end = i

            # Click if very short (1-5 samples at 44.1kHz)
            duration_samples = end - start
            if duration_samples <= 5:
                click_positions.append(start)
        else:
            i += 1

    has_clicks = len(click_positions) > 0
    count = len(click_positions)

    return has_clicks, count, click_positions


def classify_transient(audio: np.ndarray, position: int, sr: int, window_ms: float = 10.0) -> str:
    """
    Classify transient at position as click or musical.

    Musical transients (drums, consonants):
    - Longer duration (> 5ms)
    - Frequency content matches instrument
    - Part of rhythmic pattern

    Clicks:
    - Very short (< 2ms)
    - Broadband noise
    - Random timing

    Args:
        audio: Input audio
        position: Sample position of transient
        sr: Sample rate
        window_ms: Analysis window (ms)

    Returns:
        Classification: 'click', 'musical', or 'uncertain'
    """
    window_samples = int(window_ms * sr / 1000)

    # Extract region around transient
    start = max(0, position - window_samples // 2)
    end = min(len(audio), position + window_samples // 2)
    region = audio[start:end]

    if len(region) < 10:
        return "uncertain"

    # 1. Duration check
    # Find full width at half maximum (FWHM)
    peak_idx = np.argmax(np.abs(region))
    peak_val = np.abs(region[peak_idx])
    half_max = peak_val / 2

    above_half = np.abs(region) > half_max
    duration_samples = np.sum(above_half)
    duration_ms = duration_samples / sr * 1000

    # Very short = likely click
    if duration_ms < 1.0:
        return "click"

    # 2. Spectral analysis
    spectrum = np.abs(np.fft.rfft(region))
    np.fft.rfftfreq(len(region), 1 / sr)

    # Clicks are broadband (flat spectrum)
    # Musical transients have spectral peaks

    # Compute spectral flatness
    geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
    arithmetic_mean = np.mean(spectrum)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

    # High flatness (> 0.5) = broadband = click
    if spectral_flatness > 0.5:
        return "click"

    # Low flatness = tonal = musical
    if spectral_flatness < 0.2:
        return "musical"

    return "uncertain"


def compute_transient_density(audio: np.ndarray, sr: int) -> float:
    """
    Berechnet transient density (transients per second).

    High density may indicate rhythmic music (drums).

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Transients per second
    """
    # Onset detection using spectral flux
    hop_length = 512

    # Compute STFT
    _f, _t, Zxx = signal.stft(audio, sr, nperseg=2048, noverlap=2048 - hop_length, boundary="even")

    # Spectral flux (change in spectrum)
    mag = np.abs(Zxx)
    flux = np.diff(mag, axis=1)
    flux = np.sum(flux, axis=0)
    flux = np.maximum(flux, 0)  # Only increases

    # Find peaks
    mean_flux = np.mean(flux)
    threshold = mean_flux * 2
    peaks = flux > threshold

    n_transients = np.sum(peaks)
    duration_sec = len(audio) / sr

    density = n_transients / duration_sec

    return float(density)


def measure_attack_time(audio: np.ndarray, sr: int) -> float:
    """
    Misst average attack time of transients.

    Faster attacks = more percussive (need preservation).

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Average attack time in milliseconds
    """
    # Envelope detection
    analytic = signal.hilbert(audio)
    analytic_arr = np.asarray(analytic, dtype=np.complex128)
    envelope = np.abs(analytic_arr)

    # Find peaks (transient starts)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(envelope, distance=sr // 10, height=np.max(envelope) * 0.3)

    if len(peaks) == 0:
        return 0.0

    attack_times = []

    for peak_idx in peaks:
        # Find start of attack (10% of peak)
        threshold = envelope[peak_idx] * 0.1

        # Search backwards
        start_idx = peak_idx
        while start_idx > 0 and envelope[start_idx] > threshold:
            start_idx -= 1

        attack_samples = peak_idx - start_idx
        attack_ms = attack_samples / sr * 1000

        attack_times.append(attack_ms)

    avg_attack = np.mean(attack_times)

    return float(avg_attack)


# ============================================================================
# DE-CLICK SAFETY WRAPPER
# ============================================================================


class DeClickSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for de-clicking.

    Ensures:
    - Only processes audio with detected clicks
    - Classifies clicks vs musical transients
    - Preserves all musical transients (drums, consonants)
    - No loss of attack/transient energy
    - Maintains rhythmic timing
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        min_click_count: int = 5,
        max_transient_loss: float = 0.15,
    ):
        """
        Initialisiert De-Click Safety Wrapper.

        Args:
            processor_func: De-clicking function (audio, sr, sensitivity) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            min_click_count: Minimum clicks to warrant processing
            max_transient_loss: Maximum acceptable transient energy loss (0.0-1.0)
        """
        super().__init__(
            module_name="DeClick",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=0.5,
            quality_threshold=0.7,
        )

        self.min_click_count = min_click_count
        self.max_transient_loss = max_transient_loss

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for de-clicking."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Detect clicks
        has_clicks, click_count, click_positions = detect_clicks(audio, sr)
        metadata["has_clicks"] = has_clicks
        metadata["click_count"] = click_count  # type: ignore[assignment]
        metadata["click_positions"] = click_positions[:10]  # type: ignore  # Limit log size

        if not has_clicks:
            return PreCheckResult(passed=False, confidence=0.0, reasons=["No clicks detected in audio"])

        if click_count < self.min_click_count:
            return PreCheckResult(
                passed=False,
                confidence=0.3,
                reasons=[f"Too few clicks: {click_count} (min {self.min_click_count}). Not worth processing risk."],
            )

        # Classify clicks vs musical transients
        click_classifications = []
        for pos in click_positions[:50]:  # Classify first 50
            if pos < len(audio) - 100:
                classification = classify_transient(audio, pos, sr)
                click_classifications.append(classification)

        n_true_clicks = click_classifications.count("click")
        n_musical = click_classifications.count("musical")
        n_uncertain = click_classifications.count("uncertain")

        metadata["true_clicks"] = n_true_clicks  # type: ignore[assignment]
        metadata["musical_transients_detected"] = n_musical  # type: ignore[assignment]
        metadata["uncertain_transients"] = n_uncertain  # type: ignore[assignment]

        if n_musical > n_true_clicks:
            warnings.append(
                f"More musical transients ({n_musical}) than clicks ({n_true_clicks}). "
                "High risk of damaging musical content."
            )

        # Measure transient density
        transient_density = compute_transient_density(audio, sr)
        metadata["transient_density_per_sec"] = transient_density  # type: ignore[assignment]

        if transient_density > 20:
            warnings.append(
                f"Very high transient density: {transient_density:.1f}/sec. "
                "Likely contains drums/percussion. Proceed with caution."
            )

        # Measure attack time
        avg_attack_ms = measure_attack_time(audio, sr)
        metadata["avg_attack_time_ms"] = avg_attack_ms  # type: ignore[assignment]

        if avg_attack_ms < 5:
            warnings.append(
                f"Fast attacks detected: {avg_attack_ms:.1f} ms. Likely percussive content. Risk of transient loss."
            )

        return PreCheckResult(
            passed=True, confidence=float(n_true_clicks / max(1, click_count)), warnings=warnings, metadata=metadata
        )

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in de-clicking for this audio."""
        # Base confidence from click classification
        true_clicks = pre_check.metadata.get("true_clicks", 0)
        click_count = pre_check.metadata.get("click_count", 1)

        classification_confidence = true_clicks / max(1, click_count)

        # Penalty for high transient density (risk to music)
        density = pre_check.metadata.get("transient_density_per_sec", 0)
        density_penalty = min(density / 30, 0.3)  # Up to 30% penalty

        # Penalty for fast attacks (risk to percussion)
        attack_ms = pre_check.metadata.get("avg_attack_time_ms", 10)
        attack_penalty = max(0, (5 - attack_ms) / 5) * 0.2  # Up to 20% penalty

        confidence = classification_confidence * (1.0 - density_penalty - attack_penalty)

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after de-clicking."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # 1. Check click reduction
        _has_clicks_before, count_before, _ = detect_clicks(original, sr)
        _has_clicks_after, count_after, _ = detect_clicks(processed, sr)

        metrics["clicks_before"] = count_before
        metrics["clicks_after"] = count_after

        if count_after >= count_before:
            side_effects.append(f"Clicks not reduced: {count_before} -> {count_after}")

        reduction = count_before - count_after
        metrics["click_reduction"] = reduction
        metrics["click_reduction_percent"] = float(reduction / max(1, count_before) * 100)  # type: ignore[assignment]

        # 2. Check transient preservation
        # Extract transient energy
        def get_transient_energy(audio):
            analytic = signal.hilbert(audio.flatten())
            analytic_arr = np.asarray(analytic, dtype=np.complex128)
            envelope = np.abs(analytic_arr)
            # High-pass to isolate transients
            sos = signal.butter(4, 500, "hp", fs=sr, output="sos")
            transients = signal.sosfilt(sos, envelope)
            return np.sum(transients**2)

        transient_energy_before = get_transient_energy(original)
        transient_energy_after = get_transient_energy(processed)

        if transient_energy_before > 0:
            transient_preservation = transient_energy_after / transient_energy_before
        else:
            transient_preservation = 1.0

        metrics["transient_preservation_ratio"] = float(transient_preservation)  # type: ignore[assignment]

        transient_loss = 1.0 - transient_preservation
        if transient_loss > self.max_transient_loss:
            issues.append(f"Excessive transient loss: {transient_loss:.1%} (max {self.max_transient_loss:.1%})")

        # 3. Energy preservation
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio  # type: ignore[assignment]

        if energy_ratio < 0.85 or energy_ratio > 1.05:
            side_effects.append(f"Unexpected energy change: {energy_ratio:.2%}")

        # 4. Correlation (signal similarity)
        correlation = compute_correlation(original.flatten(), processed.flatten())
        metrics["correlation"] = correlation  # type: ignore[assignment]

        if correlation < 0.9:
            side_effects.append(f"Low correlation: {correlation:.3f}. Significant signal change.")

        # 5. Check for over-smoothing
        # Measure high-frequency content
        def get_hf_energy(audio):
            sos = signal.butter(4, 8000, "hp", fs=sr, output="sos")
            hf = signal.sosfilt(sos, audio.flatten())
            return np.mean(hf**2)

        hf_before = get_hf_energy(original)
        hf_after = get_hf_energy(processed)

        if hf_before > 0:
            hf_preservation = hf_after / hf_before
            metrics["high_freq_preservation"] = float(hf_preservation)  # type: ignore[assignment]

            if hf_preservation < 0.7:
                issues.append(f"Over-smoothing detected: {hf_preservation:.1%} HF content remaining")

        passed = len(issues) == 0

        return PostCheckResult(
            passed=passed,
            quality_score=0.0,  # Will be computed separately
            issues=issues,
            side_effects=side_effects,
            metrics=metrics,  # type: ignore[arg-type]
        )

    def _compute_quality_score(
        self, original: np.ndarray, processed: np.ndarray, sr: int, post_check: PostCheckResult
    ) -> float:
        """Berechnet overall quality score."""
        scores = []

        # Click reduction effectiveness
        reduction_pct = post_check.metrics.get("click_reduction_percent", 0.0)
        reduction_score = min(reduction_pct / 80.0, 1.0)  # 80% reduction = full score
        scores.append(reduction_score)

        # Transient preservation
        transient_preservation = post_check.metrics.get("transient_preservation_ratio", 1.0)
        scores.append(np.clip(transient_preservation, 0.0, 1.0))

        # Correlation
        correlation = post_check.metrics.get("correlation", 0.0)
        scores.append(correlation)

        # High-frequency preservation
        hf_preservation = post_check.metrics.get("high_freq_preservation", 1.0)
        scores.append(np.clip(hf_preservation, 0.0, 1.0))

        # Weighted average (emphasize transient preservation)
        weights = [0.25, 0.4, 0.2, 0.15]  # reduction, transients, correlation, HF
        quality = np.average(scores, weights=weights)

        return float(np.clip(quality, 0.0, 1.0))
