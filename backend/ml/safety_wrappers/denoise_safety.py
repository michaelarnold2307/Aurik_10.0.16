"""
denoise_safety.py - HIPS-Compliant De-Noise Safety Wrapper

Wraps de-noising DSP operations with comprehensive safety checks:
- Noise type classification (white, hiss, hum, room tone)
- Signal preservation validation
- No "birdie" artifacts (musical noise)
- Spectral balance preservation
- Time-domain continuity

This ensures de-noising reduces noise without creating synthetic artifacts.

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
# NOISE DETECTION UTILITIES
# ============================================================================


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """
    Schätzt Signal-to-Noise Ratio (SNR).

    Uses simple energy-based method:
    - Signal = peaks (top 25%)
    - Noise = valleys (bottom 25%)

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        SNR in dB
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute short-time energy
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = frame_length // 2

    n_frames = (len(audio) - frame_length) // hop_length

    if n_frames < 10:
        return 20.0  # Default reasonable SNR

    energies = []
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        energy = np.mean(frame**2)
        energies.append(energy)

    energies = np.array(energies)  # type: ignore[assignment]

    # Signal = top 25% of energy
    signal_threshold = np.percentile(energies, 75)
    signal_energy = np.mean(energies[energies >= signal_threshold])

    # Noise = bottom 25% of energy
    noise_threshold = np.percentile(energies, 25)
    noise_energy = np.mean(energies[energies <= noise_threshold])

    if noise_energy == 0:
        return 60.0  # Very high SNR

    snr = 10 * np.log10(signal_energy / noise_energy)

    return float(snr)


def classify_noise_type(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """
    Classify type of noise in audio.

    Types:
    - white: Broadband, flat spectrum
    - hiss: High-frequency emphasis (tape hiss)
    - hum: Low-frequency tonal (50/60 Hz + harmonics)
    - room_tone: Mid-frequency ambient

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (noise_type, confidence): Classification result
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Estimate noise floor (quiet sections)
    frame_length = int(0.1 * sr)  # 100ms
    hop_length = frame_length // 2

    n_frames = (len(audio) - frame_length) // hop_length

    if n_frames < 5:
        return "white", 0.5

    # Find quietest frames (likely noise-only)
    frame_energies = []
    frames = []

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        energy = np.mean(frame**2)
        frame_energies.append(energy)
        frames.append(frame)

    # Take bottom 20% quietest frames
    threshold = np.percentile(frame_energies, 20)
    quiet_frames = [f for f, e in zip(frames, frame_energies) if e <= threshold]

    if len(quiet_frames) == 0:
        return "white", 0.3

    # Average noise spectrum
    noise_spectra = [np.abs(np.fft.rfft(f)) for f in quiet_frames]
    avg_noise_spectrum = np.mean(noise_spectra, axis=0)
    freqs = np.fft.rfftfreq(frame_length, 1 / sr)

    # Analyze spectrum shape

    # 1. Check for hum (50/60 Hz peaks)
    hum_freqs = [50, 60, 100, 120, 150, 180]
    hum_energy = 0
    for hum_f in hum_freqs:
        idx = np.argmin(np.abs(freqs - hum_f))
        if idx < len(avg_noise_spectrum):
            hum_energy += avg_noise_spectrum[idx]

    total_energy = np.sum(avg_noise_spectrum)
    hum_ratio = hum_energy / (total_energy + 1e-10)

    if hum_ratio > 0.3:
        return "hum", 0.8

    # 2. Check for hiss (high-frequency emphasis)
    # Split spectrum into bands
    low_band = (freqs >= 20) & (freqs < 1000)
    mid_band = (freqs >= 1000) & (freqs < 4000)
    high_band = (freqs >= 4000) & (freqs < 12000)

    low_energy = np.mean(avg_noise_spectrum[low_band]) if np.any(low_band) else 0
    mid_energy = np.mean(avg_noise_spectrum[mid_band]) if np.any(mid_band) else 0
    high_energy = np.mean(avg_noise_spectrum[high_band]) if np.any(high_band) else 0

    if high_energy > 1.5 * mid_energy:
        return "hiss", 0.7

    # 3. Check for room tone (mid-frequency)
    if mid_energy > 1.3 * low_energy and mid_energy > 1.3 * high_energy:
        return "room_tone", 0.6

    # 4. Default to white noise (flat spectrum)
    # Compute spectral flatness
    geometric_mean = np.exp(np.mean(np.log(avg_noise_spectrum + 1e-10)))
    arithmetic_mean = np.mean(avg_noise_spectrum)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

    if spectral_flatness > 0.5:
        return "white", 0.8

    return "white", 0.4


def detect_birdie_artifacts(audio: np.ndarray, sr: int) -> tuple[bool, float]:
    """
    Erkennt "birdie" artifacts (musical noise from aggressive noise reduction).

    Birdies are tonal, brief, random-pitch artifacts.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        (has_birdies, severity): Detection result
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute spectrogram
    f, _t, Sxx = signal.spectrogram(audio, sr, nperseg=2048, noverlap=1536)

    # Birdies appear as isolated spectral peaks in time-frequency
    # Look for peaks that are:
    # 1. Short duration (< 50ms)
    # 2. Narrow bandwidth (< 500 Hz)
    # 3. High energy relative to surroundings

    # For each time frame, find isolated peaks
    birdie_count = 0

    for i in range(Sxx.shape[1]):
        frame = Sxx[:, i]

        # Find peaks
        from scipy.signal import find_peaks

        peaks, _properties = find_peaks(frame, prominence=np.max(frame) * 0.3)

        # Count narrow, prominent peaks
        for peak_idx in peaks:
            peak_freq = f[peak_idx]

            # Check if isolated (not part of harmonic series)
            # Look for other peaks nearby
            nearby = np.abs(f - peak_freq) < 500
            nearby_peaks = np.sum(frame[nearby] > frame[peak_idx] * 0.5)

            if nearby_peaks <= 2:  # Isolated
                birdie_count += 1

    # Normalize by time
    duration_sec = len(audio) / sr
    birdies_per_sec = birdie_count / duration_sec

    has_birdies = birdies_per_sec > 5  # More than 5 per second
    severity = min(birdies_per_sec / 20, 1.0)  # Normalize

    return has_birdies, float(severity)


def measure_spectral_balance(audio: np.ndarray, sr: int) -> dict[str, float]:
    """
    Misst spectral balance across frequency bands.

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Dict with energy per band
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Standard bands
    bands = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mid": (250, 500),
        "mid": (500, 2000),
        "high_mid": (2000, 4000),
        "presence": (4000, 6000),
        "brilliance": (6000, 20000),
    }

    balance = {}
    total_energy = np.sum(spectrum**2)

    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        if np.any(band_mask):
            band_energy = np.sum(spectrum[band_mask] ** 2)
            balance[band_name] = float(band_energy / (total_energy + 1e-10))
        else:
            balance[band_name] = 0.0

    return balance


def check_temporal_continuity(audio: np.ndarray, sr: int) -> float:
    """
    Prüft temporal continuity (no gaps or dropouts).

    Args:
        audio: Input audio
        sr: Sample rate

    Returns:
        Continuity score (0.0-1.0)
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Compute envelope
    analytic = signal.hilbert(audio)
    analytic_arr = np.asarray(analytic, dtype=np.complex128)
    envelope = np.abs(analytic_arr)

    # Look for sudden drops (gaps)
    frame_length = int(0.01 * sr)  # 10ms
    hop_length = frame_length // 2

    n_frames = (len(envelope) - frame_length) // hop_length

    if n_frames < 10:
        return 1.0

    frame_energies = []
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = envelope[start:end]
        energy = np.mean(frame)
        frame_energies.append(energy)

    frame_energies = np.array(frame_energies)  # type: ignore[assignment]

    # Count sudden drops (> 20 dB)
    frame_energies_db = 20 * np.log10(frame_energies + 1e-10)  # type: ignore[operator]
    drops = np.abs(np.diff(frame_energies_db)) > 20

    n_drops = np.sum(drops)
    drop_rate = n_drops / n_frames

    continuity = 1.0 - min(drop_rate * 5, 1.0)  # Penalize drops

    return float(continuity)


# ============================================================================
# DE-NOISE SAFETY WRAPPER
# ============================================================================


class DeNoiseSafety(BaseSafetyWrapper):
    """
    HIPS-compliant safety wrapper for de-noising.

    Ensures:
    - Only processes audio with sufficient noise (SNR < threshold)
    - Adapts to noise type (white, hiss, hum, room tone)
    - Preserves signal content (no over-reduction)
    - No birdie artifacts (musical noise)
    - Maintains spectral balance
    - Ensures temporal continuity
    """

    def __init__(
        self,
        processor_func,
        enable_logging: bool = True,
        log_dir: Path | None = None,
        max_snr_db: float = 30.0,
        min_snr_db: float = 5.0,
        max_birdie_tolerance: float = 0.3,
    ):
        """
        Initialisiert De-Noise Safety Wrapper.

        Args:
            processor_func: De-noising function (audio, sr, noise_type, strength) -> audio
            enable_logging: Enable audit trail
            log_dir: Audit log directory
            max_snr_db: Maximum SNR (above this, no processing needed)
            min_snr_db: Minimum SNR (below this, too noisy to process reliably)
            max_birdie_tolerance: Maximum acceptable birdie severity
        """
        super().__init__(
            module_name="DeNoise",
            module_version="1.0.0",
            processor_func=processor_func,
            enable_logging=enable_logging,
            log_dir=log_dir,
            confidence_threshold=0.5,
            quality_threshold=0.65,
        )

        self.max_snr_db = max_snr_db
        self.min_snr_db = min_snr_db
        self.max_birdie_tolerance = max_birdie_tolerance

    def _validate_pre_conditions(self, audio: np.ndarray, sr: int, **params) -> PreCheckResult:
        """Validiert pre-conditions for de-noising."""
        # Basic audio validation
        is_valid, errors = validate_audio_basic(audio)

        if not is_valid:
            return PreCheckResult(passed=False, confidence=0.0, reasons=errors)

        warnings = []
        metadata = {}

        # Estimate SNR
        snr = estimate_snr(audio, sr)
        metadata["snr_db"] = snr

        if snr > self.max_snr_db:
            return PreCheckResult(
                passed=False, confidence=0.0, reasons=[f"SNR too high: {snr:.1f} dB. Audio is already clean."]
            )

        if snr < self.min_snr_db:
            warnings.append(f"Very low SNR: {snr:.1f} dB. Noise reduction may struggle to preserve signal.")

        # Classify noise type
        noise_type, noise_conf = classify_noise_type(audio, sr)
        metadata["noise_type"] = noise_type  # type: ignore[assignment]
        metadata["noise_classification_confidence"] = noise_conf

        if noise_conf < 0.5:
            warnings.append(f"Uncertain noise classification: {noise_type} (confidence {noise_conf:.2f})")

        # Measure spectral balance
        balance_before = measure_spectral_balance(audio, sr)
        metadata["spectral_balance_before"] = balance_before  # type: ignore[assignment]

        # Check temporal continuity
        continuity = check_temporal_continuity(audio, sr)
        metadata["temporal_continuity_before"] = continuity

        if continuity < 0.8:
            warnings.append(f"Poor temporal continuity: {continuity:.2f}. Audio may have gaps or dropouts.")

        # Validate processing strength
        strength = params.get("strength", 0.5)

        if strength > 0.8:
            warnings.append(f"Very aggressive de-noising: strength {strength:.2f}. High risk of artifacts.")

        return PreCheckResult(passed=True, confidence=noise_conf, warnings=warnings, metadata=metadata)

    def _assess_epistemic_confidence(self, audio: np.ndarray, sr: int, pre_check: PreCheckResult, **params) -> float:
        """Assess confidence in de-noising for this audio."""
        # Base confidence from noise classification
        noise_conf = pre_check.metadata.get("noise_classification_confidence", 0.5)

        # SNR-based confidence (optimal SNR: 10-25 dB)
        snr = pre_check.metadata.get("snr_db", 20)

        if snr < 10:
            snr_factor = snr / 10  # Lower confidence for very noisy
        elif snr > 25:
            snr_factor = (self.max_snr_db - snr) / (self.max_snr_db - 25)
        else:
            snr_factor = 1.0  # Optimal range

        # Temporal continuity factor
        continuity = pre_check.metadata.get("temporal_continuity_before", 1.0)

        # Penalty for aggressive processing
        strength = params.get("strength", 0.5)
        strength_penalty = max(0, (strength - 0.7) / 0.3) * 0.3  # Up to 30% penalty

        confidence = noise_conf * 0.4 + snr_factor * 0.3 + continuity * 0.3
        confidence *= 1.0 - strength_penalty

        return float(np.clip(confidence, 0.0, 1.0))

    def _validate_post_conditions(
        self, original: np.ndarray, processed: np.ndarray, sr: int, **params
    ) -> PostCheckResult:
        """Validiert post-conditions after de-noising."""
        issues = []
        side_effects = []
        metrics = {}

        # Ensure same shape
        if original.shape != processed.shape:
            issues.append(f"Shape mismatch: {original.shape} -> {processed.shape}")
            return PostCheckResult(passed=False, quality_score=0.0, issues=issues)

        # 1. Check SNR improvement
        snr_before = estimate_snr(original, sr)
        snr_after = estimate_snr(processed, sr)

        metrics["snr_before"] = snr_before
        metrics["snr_after"] = snr_after
        metrics["snr_improvement_db"] = float(snr_after - snr_before)

        if snr_after <= snr_before:
            side_effects.append(f"SNR not improved: {snr_before:.1f} -> {snr_after:.1f} dB")

        # 2. Check for birdie artifacts
        has_birdies_before, severity_before = detect_birdie_artifacts(original, sr)
        has_birdies_after, severity_after = detect_birdie_artifacts(processed, sr)

        metrics["birdie_severity_before"] = severity_before
        metrics["birdie_severity_after"] = severity_after

        if has_birdies_after and not has_birdies_before:
            issues.append(f"Birdie artifacts introduced: severity {severity_after:.2f}")
        elif severity_after > severity_before + 0.2:
            issues.append(f"Birdie artifacts increased: {severity_before:.2f} -> {severity_after:.2f}")

        if severity_after > self.max_birdie_tolerance:
            issues.append(f"Excessive birdies: {severity_after:.2f} (max {self.max_birdie_tolerance})")

        # 3. Spectral balance preservation
        balance_before = measure_spectral_balance(original, sr)
        balance_after = measure_spectral_balance(processed, sr)

        metrics["spectral_balance_before"] = balance_before  # type: ignore[assignment]
        metrics["spectral_balance_after"] = balance_after  # type: ignore[assignment]

        # Check each band
        for band in balance_before:
            ratio = balance_after[band] / (balance_before[band] + 1e-10)

            if ratio < 0.5 or ratio > 2.0:
                side_effects.append(f"Spectral imbalance in {band}: {ratio:.2f}x")

        # 4. Temporal continuity
        continuity_before = check_temporal_continuity(original, sr)
        continuity_after = check_temporal_continuity(processed, sr)

        metrics["temporal_continuity_before"] = continuity_before
        metrics["temporal_continuity_after"] = continuity_after

        if continuity_after < continuity_before - 0.1:
            issues.append(f"Temporal continuity degraded: {continuity_before:.2f} -> {continuity_after:.2f}")

        # 5. Energy preservation
        energy_ratio = compute_energy_ratio(original, processed)
        metrics["energy_ratio"] = energy_ratio

        if energy_ratio < 0.7 or energy_ratio > 1.1:
            side_effects.append(f"Unexpected energy change: {energy_ratio:.2%}")

        # 6. Correlation
        correlation = compute_correlation(original.flatten(), processed.flatten())
        metrics["correlation"] = correlation

        if correlation < 0.85:
            side_effects.append(f"Low correlation: {correlation:.3f}. Significant signal change.")

        # 7. High-frequency preservation (check for over-smoothing)
        centroid_before = compute_spectral_centroid(original.flatten(), sr)
        centroid_after = compute_spectral_centroid(processed.flatten(), sr)

        metrics["spectral_centroid_before"] = centroid_before
        metrics["spectral_centroid_after"] = centroid_after

        centroid_ratio = centroid_after / centroid_before
        if centroid_ratio < 0.7:
            issues.append(f"Excessive high-frequency loss: {centroid_ratio:.2f}x brightness")

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

        # SNR improvement
        snr_improvement = post_check.metrics.get("snr_improvement_db", 0.0)
        snr_score = min(snr_improvement / 10.0, 1.0)  # 10 dB improvement = full score
        scores.append(max(snr_score, 0.0))

        # Birdie-free (lower is better)
        birdie_severity = post_check.metrics.get("birdie_severity_after", 0.0)
        birdie_score = 1.0 - birdie_severity
        scores.append(np.clip(birdie_score, 0.0, 1.0))

        # Temporal continuity
        continuity = post_check.metrics.get("temporal_continuity_after", 1.0)
        scores.append(continuity)

        # Correlation
        correlation = post_check.metrics.get("correlation", 0.0)
        scores.append(correlation)

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # SNR, birdies, continuity, correlation
        quality = np.average(scores, weights=weights)

        return float(np.clip(quality, 0.0, 1.0))
