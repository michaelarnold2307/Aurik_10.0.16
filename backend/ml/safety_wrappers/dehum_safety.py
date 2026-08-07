"""
dehum_safety.py - HIPS-Compliant De-Hum Safety Wrapper

Wraps de-humming DSP operations with comprehensive safety checks:
- 50/60 Hz hum detection (+ harmonics)
- Bass frequency preservation (musical content below 100 Hz)
- No comb filtering artifacts
- Harmonic series validation
- Phase coherence maintenance

This ensures de-humming removes electrical interference without damaging bass.

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
# HUM DETECTION UTILITIES
# ============================================================================


def detect_hum(
    audio: np.ndarray, sr: int, fundamental_hz: float | None = None
) -> tuple[bool, float, float, list[float]]:
    """
    Erkennt electrical hum in audio signal.

    Hum characteristics:
    - Tonal (pure sine waves)
    - At 50 Hz or 60 Hz (+ harmonics: 100, 120, 150, 180, 200, 240...)
    - Constant amplitude over time

    Args:
        audio: Input audio
        sr: Sample rate
        fundamental_hz: Expected hum frequency (50 or 60), or None to detect

    Returns:
        (has_hum, hum_frequency, hum_severity, harmonic_frequencies)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Detect fundamental if not provided
    if fundamental_hz is None:
        # Check for 50 Hz or 60 Hz peak
        candidates = [50, 60]
        candidate_energies = []

        for candidate in candidates:
            idx = np.argmin(np.abs(freqs - candidate))
            if idx < len(spectrum):
                # Sum energy in ±2 Hz band
                band = (freqs >= candidate - 2) & (freqs <= candidate + 2)
                energy = np.sum(spectrum[band] ** 2)
                candidate_energies.append(energy)
            else:
                candidate_energies.append(0.0)

        # Choose stronger candidate
        fundamental_hz = 50 if candidate_energies[0] > candidate_energies[1] else 60

    # Find harmonics
    max_harmonic = int(sr / 2 / fundamental_hz)  # Up to Nyquist
    harmonic_energies = []
    harmonic_frequencies = []

    for n in range(1, min(max_harmonic, 11)):  # Up to 10th harmonic
        harmonic_freq = fundamental_hz * n

        if harmonic_freq > sr / 2:
            break

        idx = np.argmin(np.abs(freqs - harmonic_freq))

        # Sum energy in ±1 Hz band
        band = (freqs >= harmonic_freq - 1) & (freqs <= harmonic_freq + 1)
        energy = np.sum(spectrum[band] ** 2)

        harmonic_energies.append(energy)
        harmonic_frequencies.append(harmonic_freq)

    # Check if energy is concentrated at harmonic frequencies
    total_energy = np.sum(spectrum**2)
    hum_energy = np.sum(harmonic_energies)

    hum_ratio = hum_energy / (total_energy + 1e-10)

    # Has hum if > 5% of energy at harmonic frequencies
    has_hum = hum_ratio > 0.05

    # Severity based on ratio
    severity = min(hum_ratio / 0.3, 1.0)  # Normalize to 0-1

    return has_hum, float(fundamental_hz), float(severity), harmonic_frequencies


def check_bass_content(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Prüft if audio has significant musical bass content (<100 Hz).

    Important to distinguish from hum.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (has_bass, bass_energy_ratio)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Filter to bass range (20-100 Hz)
    sos = signal.butter(4, [20, 100], "bp", fs=sr, output="sos")
    bass = signal.sosfilt(sos, audio)

    # Compute energy
    bass_energy = np.mean(bass**2)
    total_energy = np.mean(audio**2)

    bass_ratio = bass_energy / (total_energy + 1e-10)

    # Has bass if > 10% of energy
    has_bass = bass_ratio > 0.1

    return has_bass, float(bass_ratio)


def detect_comb_filtering(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Erkennt comb filtering artifacts.

    Comb filtering creates spectral notches at regular intervals.
    Can be introduced by aggressive notch filtering.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (has_combing, severity)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # OPTIMIZATION: Limit analysis to 5 seconds for performance
    # Comb filtering patterns are detectable in short samples
    max_samples = int(5 * sr)
    if len(audio) > max_samples:
        # Use middle section for better representation
        start = (len(audio) - max_samples) // 2
        audio = audio[start : start + max_samples]

    # Compute spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    np.fft.rfftfreq(len(audio), 1 / sr)

    # Look for regular dips in spectrum
    # Comb filtering creates nulls at multiples of delay time

    # Compute spectral envelope (smoothed)
    from scipy.ndimage import gaussian_filter1d

    envelope = gaussian_filter1d(spectrum, sigma=10)

    # Find dips (spectrum much lower than envelope)
    dips = spectrum < envelope * 0.5

    # Count dips
    n_dips = 0
    in_dip = False

    for i in range(len(dips)):
        if dips[i] and not in_dip:
            n_dips += 1
            in_dip = True
        elif not dips[i]:
            in_dip = False

    # Many dips = combing
    has_combing = n_dips > 20
    severity = min(n_dips / 50, 1.0)

    return has_combing, float(severity)


def measure_harmonic_series_strength(audio: np.ndarray, sr: int) -> float:
    """
    Misst strength of harmonic series in audio.

    Musical sounds have strong harmonic series.
    Hum also has harmonics but different pattern.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Harmonic strength (0.0-1.0)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Autocorrelation for pitch detection — FFT-based O(N log N)
    from backend.core.core_utils import fft_autocorr

    autocorr = fft_autocorr(audio)
    autocorr = autocorr / (autocorr[0] + 1e-10)  # Normalize

    # Find first significant peak (indicates pitch)
    min_lag = int(sr / 500)  # Highest expected F0 (500 Hz)
    max_lag = int(sr / 30)  # Lowest expected F0 (30 Hz)

    if max_lag >= len(autocorr):
        return 0.0

    search_region = autocorr[min_lag:max_lag]

    if len(search_region) == 0:
        return 0.0

    max_autocorr = np.max(search_region)

    # Strong harmonics if autocorrelation peak > 0.5
    harmonic_strength = np.clip(max_autocorr, 0.0, 1.0)

    return float(harmonic_strength)


def check_phase_coherence(audio: np.ndarray) -> float:
    """
    Prüft phase coherence (for stereo signals).

    De-humming should preserve phase relationships.

    Args:
        audio: Input audio (stereo)

    Returns:
        Phase coherence (0.0-1.0)
    """
    if audio.ndim == 1:
        return 1.0  # Mono = always coherent

    if audio.shape[0] != 2:
        return 1.0  # Not stereo

    left = audio[0]
    right = audio[1]

    # Guarded Pearson correlation — avoids NaN on silent/constant channels (§VERBOTEN: np.corrcoef)
    _l = left.astype(float)
    _r = right.astype(float)
    _lc = _l - float(np.mean(_l))
    _rc = _r - float(np.mean(_r))
    _nl = float(np.linalg.norm(_lc))
    _nr = float(np.linalg.norm(_rc))
    if _nl < 1e-12 or _nr < 1e-12:
        return 1.0  # Silence in one channel → treat as coherent (conservative)
    correlation = float(np.dot(_lc, _rc) / (_nl * _nr))
    correlation = float(np.clip(correlation, -1.0, 1.0))

    # Perfect correlation = 1.0 (perfect coherence)
    # Zero correlation = 0.5 (no relationship)
    # Negative correlation = 0.0 (out of phase)

    coherence = (correlation + 1.0) / 2.0

    return float(np.clip(coherence, 0.0, 1.0))


# ============================================================================
# DE-HUM SAFETY WRAPPER
# ============================================================================


class DeHumSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for de-humming.

    Ensures:
    - Only processes audio with detected hum (50/60 Hz + harmonics)
    - Preserves musical bass content (kick drums, bass guitar)
    - No comb filtering artifacts introduced
    - Maintains harmonic series of musical content
    - Preserves phase coherence (stereo)
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        min_hum_severity: float = 0.2,
        max_bass_loss: float = 0.15,
    ):
        """
        Initialisiert De-Hum Safety Wrapper.

        Args:
            processor_func: De-humming function (audio, sr, fundamental_hz) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            min_hum_severity: Minimum hum severity to warrant processing
            max_bass_loss: Maximum acceptable bass energy loss
        """
        super().__init__(
            module_name="DeHum",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=0.5,
            quality_threshold=0.7,
        )

        self.min_hum_severity = min_hum_severity
        self.max_bass_loss = max_bass_loss

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for de-humming."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Detect hum
        fundamental_hz = params.get("fundamental_hz")
        has_hum, detected_fundamental, severity, harmonics = detect_hum(audio, sr, fundamental_hz)

        metadata["has_hum"] = has_hum
        metadata["hum_frequency"] = detected_fundamental  # type: ignore[assignment]
        metadata["hum_severity"] = severity  # type: ignore[assignment]
        metadata["hum_harmonics"] = harmonics  # type: ignore[assignment]

        if not has_hum:
            return PreCheckResult(passed=False, confidence=0.0, reasons=["No hum detected in audio"])

        if severity < self.min_hum_severity:
            return PreCheckResult(
                passed=False,
                confidence=severity,
                reasons=[f"Hum too weak: {severity:.2f} (min {self.min_hum_severity}). Not worth processing risk."],
            )

        # Check for bass content
        has_bass, bass_ratio = check_bass_content(audio, sr)
        metadata["has_bass_content"] = has_bass
        metadata["bass_energy_ratio"] = bass_ratio  # type: ignore[assignment]

        if has_bass:
            warnings.append(f"Significant bass content detected: {bass_ratio:.1%}. Risk of damaging musical low end.")

        # Check harmonic series
        harmonic_strength = measure_harmonic_series_strength(audio, sr)
        metadata["harmonic_series_strength"] = harmonic_strength  # type: ignore[assignment]

        if harmonic_strength > 0.7:
            warnings.append(
                f"Strong harmonic series: {harmonic_strength:.2f}. Audio likely contains musical bass instruments."
            )

        # Check phase coherence (if stereo)
        phase_coherence = check_phase_coherence(audio)
        metadata["phase_coherence_before"] = phase_coherence  # type: ignore[assignment]

        if phase_coherence < 0.7:
            warnings.append(f"Low phase coherence: {phase_coherence:.2f}. Stereo image may be fragile.")

        return PreCheckResult(passed=True, confidence=severity, warnings=warnings, metadata=metadata)

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in de-humming for this audio."""
        # Base confidence from hum severity
        severity = pre_check.metadata.get("hum_severity", 0.5)

        # Penalty for bass content (risk of damage)
        bass_ratio = pre_check.metadata.get("bass_energy_ratio", 0.0)
        bass_penalty = bass_ratio * 0.3  # Up to 30% penalty

        # Penalty for strong harmonics (risk to music)
        harmonic_strength = pre_check.metadata.get("harmonic_series_strength", 0.0)
        harmonic_penalty = (harmonic_strength - 0.5) * 0.2 if harmonic_strength > 0.5 else 0

        confidence = severity * (1.0 - bass_penalty - harmonic_penalty)

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after de-humming."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # 1. Check hum reduction
        _, _, severity_before, _ = detect_hum(original, sr)
        _, _, severity_after, _ = detect_hum(processed, sr)

        metrics["hum_severity_before"] = severity_before
        metrics["hum_severity_after"] = severity_after

        if severity_after >= severity_before:
            side_effects.append(f"Hum not reduced: {severity_before:.2f} -> {severity_after:.2f}")

        reduction = severity_before - severity_after
        metrics["hum_reduction"] = float(reduction)
        metrics["hum_reduction_percent"] = float(reduction / max(severity_before, 1e-6) * 100)

        # 2. Check bass preservation
        has_bass_before, bass_ratio_before = check_bass_content(original, sr)
        _has_bass_after, bass_ratio_after = check_bass_content(processed, sr)

        metrics["bass_ratio_before"] = bass_ratio_before
        metrics["bass_ratio_after"] = bass_ratio_after

        if has_bass_before:
            bass_loss = (bass_ratio_before - bass_ratio_after) / bass_ratio_before
            metrics["bass_loss_percent"] = float(bass_loss * 100)

            if bass_loss > self.max_bass_loss:
                issues.append(f"Excessive bass loss: {bass_loss:.1%} (max {self.max_bass_loss:.1%})")

        # 3. Check for comb filtering
        has_combing_before, combing_before = detect_comb_filtering(original, sr)
        has_combing_after, combing_after = detect_comb_filtering(processed, sr)

        metrics["comb_filtering_before"] = combing_before
        metrics["comb_filtering_after"] = combing_after

        if has_combing_after and not has_combing_before:
            issues.append(f"Comb filtering introduced: severity {combing_after:.2f}")
        elif combing_after > combing_before + 0.2:
            issues.append(f"Comb filtering increased: {combing_before:.2f} -> {combing_after:.2f}")

        # 4. Harmonic series preservation
        harmonic_before = measure_harmonic_series_strength(original, sr)
        harmonic_after = measure_harmonic_series_strength(processed, sr)

        metrics["harmonic_strength_before"] = harmonic_before
        metrics["harmonic_strength_after"] = harmonic_after

        if harmonic_before > 0.5 and harmonic_after < harmonic_before * 0.8:
            side_effects.append(f"Harmonic series weakened: {harmonic_before:.2f} -> {harmonic_after:.2f}")

        # 5. Phase coherence (stereo)
        phase_before = check_phase_coherence(original)
        phase_after = check_phase_coherence(processed)

        metrics["phase_coherence_before"] = phase_before
        metrics["phase_coherence_after"] = phase_after

        if phase_after < phase_before - 0.1:
            side_effects.append(f"Phase coherence degraded: {phase_before:.2f} -> {phase_after:.2f}")

        # 6. Energy preservation
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio < 0.85 or energy_ratio > 1.05:
            side_effects.append(f"Unexpected energy change: {energy_ratio:.2%}")

        # 7. Correlation
        correlation = compute_correlation(original.flatten(), processed.flatten())
        metrics["correlation"] = correlation

        if correlation < 0.9:
            side_effects.append(f"Low correlation: {correlation:.3f}. Significant signal change.")

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

        # Hum reduction effectiveness
        reduction_pct = post_check.metrics.get("hum_reduction_percent", 0.0)
        reduction_score = min(reduction_pct / 70.0, 1.0)  # 70% reduction = full score
        scores.append(max(reduction_score, 0.0))

        # Bass preservation
        bass_loss_pct = post_check.metrics.get("bass_loss_percent", 0.0)
        bass_score = 1.0 - (bass_loss_pct / 100.0)
        scores.append(np.clip(bass_score, 0.0, 1.0))

        # No comb filtering
        combing_after = post_check.metrics.get("comb_filtering_after", 0.0)
        combing_score = 1.0 - combing_after
        scores.append(np.clip(combing_score, 0.0, 1.0))

        # Correlation
        correlation = post_check.metrics.get("correlation", 0.0)
        scores.append(correlation)

        # Weighted average (emphasize bass and comb filtering)
        weights = [0.3, 0.3, 0.2, 0.2]  # reduction, bass, combing, correlation
        quality = np.average(scores, weights=weights)

        return float(np.clip(quality, 0.0, 1.0))
