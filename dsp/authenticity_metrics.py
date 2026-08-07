"""
Authenticity Metrics
====================

Objektive Messung der Original-Charakter-Erhaltung.

Misst wie sehr sich Audio vom Original unterscheidet:
- Spectral Deviation (Frequenz-Charakteristik)
- Dynamic Range Change (Lautheit-Dynamik)
- Transient Correlation (Attack-Preservation)

Safeguard #4: Objective Authenticity Metrics
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AuthenticityMetrics:
    """
    Messung der Authentizität (Original-Charakter-Erhaltung).

    Verwendung:
        is_authentic, warnings, metrics = AuthenticityMetrics.authenticity_check(
            audio_original, audio_processed, sr
        )
    """

    # Thresholds für Authenticity
    SPECTRAL_DEVIATION_WARNING = 0.1  # Sichtbare Änderung
    SPECTRAL_DEVIATION_CRITICAL = 0.3  # Character altered!

    DYNAMIC_RANGE_WARNING = 20  # 20% change
    DYNAMIC_RANGE_CRITICAL = 40  # 40% change

    TRANSIENT_CORRELATION_WARNING = 0.9  # < 0.9
    TRANSIENT_CORRELATION_CRITICAL = 0.7  # < 0.7

    @staticmethod
    def measure_spectral_deviation(audio_original: np.ndarray, audio_enhanced: np.ndarray, sr: int) -> float:
        """
        Spectral Flatness Deviation.

        Misst wie sehr sich das Frequenz-Spektrum verändert hat.

        Returns:
            deviation: 0.0 = identisch, > 0.3 = stark verändert
        """
        if sr <= 0:
            raise ValueError(f"Sample rate must be > 0 Hz, got {sr}")
        audio_original = np.nan_to_num(audio_original, nan=0.0, posinf=0.0, neginf=0.0)
        audio_enhanced = np.nan_to_num(audio_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

        # STFT
        n_fft = min(2048, max(64, len(audio_original)))
        D_original = np.abs(librosa.stft(audio_original, n_fft=n_fft))
        n_fft_enh = min(2048, max(64, len(audio_enhanced)))
        D_enhanced = np.abs(librosa.stft(audio_enhanced, n_fft=n_fft_enh))

        # Normalized Spectral Difference über alle Frequenzen und Zeit
        # Nutze logarithmische Skala (dB) für perceptually-relevant Messung
        D_original_db = librosa.amplitude_to_db(D_original, ref=np.max)
        D_enhanced_db = librosa.amplitude_to_db(D_enhanced, ref=np.max)

        # Mean Absolute Error in dB
        deviation = np.mean(np.abs(D_enhanced_db - D_original_db))
        deviation = np.nan_to_num(deviation, nan=0.0, posinf=0.0, neginf=0.0)

        return float(deviation)

    @staticmethod
    def measure_dynamic_range_change(audio_original: np.ndarray, audio_enhanced: np.ndarray) -> float:
        """
        Crest Factor Deviation.

        Crest Factor = Peak / RMS
        Misst wie sich die Dynamik verändert hat.

        Returns:
            change_percent: % change in Crest Factor
        """
        audio_original = np.nan_to_num(audio_original, nan=0.0, posinf=0.0, neginf=0.0)
        audio_enhanced = np.nan_to_num(audio_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

        def crest_factor(audio):
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio**2))
            return peak / (rms + 1e-10)

        crest_original = crest_factor(audio_original)
        crest_enhanced = crest_factor(audio_enhanced)

        change_percent = abs(crest_enhanced - crest_original) / (crest_original + 1e-10) * 100
        change_percent = np.nan_to_num(change_percent, nan=0.0, posinf=0.0, neginf=0.0)

        return float(change_percent)

    @staticmethod
    def measure_transient_preservation(audio_original: np.ndarray, audio_enhanced: np.ndarray, sr: int) -> float:
        """
        Onset Strength Correlation.

        Misst wie gut Transienten (Drum-Attacks, etc.) erhalten wurden.

        Returns:
            correlation: 1.0 = perfekt erhalten, < 0.9 = verändert
        """
        if sr <= 0:
            raise ValueError(f"Sample rate must be > 0 Hz, got {sr}")
        audio_original = np.nan_to_num(audio_original, nan=0.0, posinf=0.0, neginf=0.0)
        audio_enhanced = np.nan_to_num(audio_enhanced, nan=0.0, posinf=0.0, neginf=0.0)

        # Onset Strength Envelopes
        onsets_orig = librosa.onset.onset_strength(y=audio_original, sr=sr)
        onsets_enh = librosa.onset.onset_strength(y=audio_enhanced, sr=sr)

        # Ensure same length
        min_len = min(len(onsets_orig), len(onsets_enh))
        onsets_orig = onsets_orig[:min_len]
        onsets_enh = onsets_enh[:min_len]

        # Correlation
        if len(onsets_orig) < 2:
            return 1.0  # Too short to measure

        # Guarded Pearson correlation — avoids NaN on silent/constant signals (§VERBOTEN: np.corrcoef)
        _a_g = np.asarray(onsets_orig, dtype=np.float64)
        _b_g = np.asarray(onsets_enh, dtype=np.float64)
        _a_g = _a_g - _a_g.mean()
        _b_g = _b_g - _b_g.mean()
        _denom_g = (np.linalg.norm(_a_g) * np.linalg.norm(_b_g)) + 1e-12
        correlation = float(np.dot(_a_g, _b_g) / _denom_g)

        return float(correlation)

    @staticmethod
    def measure_spectral_tilt_change(audio_original: np.ndarray, audio_enhanced: np.ndarray, sr: int) -> float:
        """
        Spectral Tilt Change.

        Misst Änderung in Bass/Treble-Balance.

        Returns:
            tilt_change_db: dB change in spectral tilt
        """

        def spectral_tilt(audio, sr):
            # Compute spectrum
            n_fft = min(2048, max(64, len(audio)))
            D = np.abs(librosa.stft(audio, n_fft=n_fft))
            spectrum = np.mean(D, axis=1)  # Average über Zeit

            # Low vs High Energy (Bass vs Treble)
            freq_bins = len(spectrum)
            low_cutoff = int(freq_bins * 0.2)  # Untere 20%
            high_cutoff = int(freq_bins * 0.8)  # Obere 20%

            low_energy = np.sum(spectrum[:low_cutoff])
            high_energy = np.sum(spectrum[high_cutoff:])

            tilt = high_energy / (low_energy + 1e-10)
            return tilt

        tilt_original = spectral_tilt(audio_original, sr)
        tilt_enhanced = spectral_tilt(audio_enhanced, sr)

        # Change in dB
        tilt_change_db = 20 * np.log10(tilt_enhanced / (tilt_original + 1e-10))

        return abs(tilt_change_db)  # type: ignore[no-any-return]

    @staticmethod
    def measure_stereo_image_change(audio_original: np.ndarray, audio_enhanced: np.ndarray) -> float:
        """
        Stereo Width Change (nur für Stereo-Signale).

        Returns:
            width_change: Relative change in stereo width (0.0 = no change)
        """
        if audio_original.ndim == 1 or audio_enhanced.ndim == 1:
            return 0.0  # Mono

        def stereo_width(audio_stereo):
            L = audio_stereo[0]
            R = audio_stereo[1]

            # Mid-Side
            mid = (L + R) / 2
            side = (L - R) / 2

            mid_energy = np.sum(mid**2)
            side_energy = np.sum(side**2)

            width = side_energy / (mid_energy + 1e-10)
            return width

        width_original = stereo_width(audio_original)
        width_enhanced = stereo_width(audio_enhanced)

        width_change = abs(width_enhanced - width_original) / (width_original + 1e-10)

        return width_change  # type: ignore[no-any-return]

    @staticmethod
    def authenticity_check(
        audio_original: np.ndarray,
        audio_enhanced: np.ndarray,
        sr: int,
        verbose: bool = True,
    ) -> tuple[bool, list[str], dict[str, float]]:
        """
        Comprehensive Authenticity Check.

        Args:
            audio_original: Original audio
            audio_enhanced: Processed audio
            sr: Sample rate
            verbose: Log warnings

        Returns:
            (is_authentic, warnings, metrics)
                is_authentic: True if all checks pass
                warnings: List of warning messages
                metrics: Dict of measured values
        """
        warnings = []
        metrics = {}

        # 1. Spectral Deviation
        spec_dev = AuthenticityMetrics.measure_spectral_deviation(audio_original, audio_enhanced, sr)
        metrics["spectral_deviation"] = spec_dev

        if spec_dev > AuthenticityMetrics.SPECTRAL_DEVIATION_CRITICAL:
            warnings.append(
                f"❌ CRITICAL: Spectral Deviation {spec_dev:.2f} dB (> {AuthenticityMetrics.SPECTRAL_DEVIATION_CRITICAL})"
            )
        elif spec_dev > AuthenticityMetrics.SPECTRAL_DEVIATION_WARNING:
            warnings.append(f"⚠️ Moderate Spectral Change: {spec_dev:.2f} dB")

        # 2. Dynamic Range Change
        dr_change = AuthenticityMetrics.measure_dynamic_range_change(audio_original, audio_enhanced)
        metrics["dynamic_range_change_percent"] = dr_change

        if dr_change > AuthenticityMetrics.DYNAMIC_RANGE_CRITICAL:
            warnings.append(
                f"❌ CRITICAL: Dynamic Range changed by {dr_change:.1f}% (> {AuthenticityMetrics.DYNAMIC_RANGE_CRITICAL}%)"
            )
        elif dr_change > AuthenticityMetrics.DYNAMIC_RANGE_WARNING:
            warnings.append(f"⚠️ Dynamic Range changed by {dr_change:.1f}%")

        # 3. Transient Preservation
        trans_corr = AuthenticityMetrics.measure_transient_preservation(audio_original, audio_enhanced, sr)
        metrics["transient_correlation"] = trans_corr

        if trans_corr < AuthenticityMetrics.TRANSIENT_CORRELATION_CRITICAL:
            warnings.append(
                f"❌ CRITICAL: Transients heavily altered: correlation {trans_corr:.2f} (< {AuthenticityMetrics.TRANSIENT_CORRELATION_CRITICAL})"
            )
        elif trans_corr < AuthenticityMetrics.TRANSIENT_CORRELATION_WARNING:
            warnings.append(f"⚠️ Transients altered: correlation {trans_corr:.2f}")

        # 4. Spectral Tilt
        tilt_change = AuthenticityMetrics.measure_spectral_tilt_change(audio_original, audio_enhanced, sr)
        metrics["spectral_tilt_change_db"] = tilt_change

        if tilt_change > 6.0:  # > 6dB change in tonal balance
            warnings.append(f"⚠️ Spectral Tilt changed by {tilt_change:.1f} dB (tonal balance altered)")

        # 5. Stereo Width (if stereo)
        if audio_original.ndim > 1:
            width_change = AuthenticityMetrics.measure_stereo_image_change(audio_original, audio_enhanced)
            metrics["stereo_width_change"] = width_change

            if width_change > 0.5:  # 50% change
                warnings.append(f"⚠️ Stereo Width changed by {width_change * 100:.0f}%")

        # Overall Verdict
        is_authentic = len(warnings) == 0

        if verbose and not is_authentic:
            logger.warning("Authenticity Pruefung WARNINGS:")
            for warning in warnings:
                logger.warning("  %s", warning)

        return is_authentic, warnings, metrics

    @staticmethod
    def compare_ab(
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sr: int,
        labels: tuple[str, str] = ("A", "B"),
    ) -> str:
        """
        A/B Comparison Report.

        Args:
            audio_a: First audio (e.g., original)
            audio_b: Second audio (e.g., enhanced)
            sr: Sample rate
            labels: Tuple of labels for A and B

        Returns:
            report: Formatted comparison report
        """
        _, _, metrics = AuthenticityMetrics.authenticity_check(audio_a, audio_b, sr, verbose=False)

        report = f"\n{'=' * 60}\n"
        report += f"A/B COMPARISON: {labels[0]} vs {labels[1]}\n"
        report += f"{'=' * 60}\n\n"

        report += f"Spectral Deviation:     {metrics['spectral_deviation']:.3f} dB\n"
        report += f"Dynamic Range Change:   {metrics['dynamic_range_change_percent']:.1f}%\n"
        report += f"Transient Correlation:  {metrics['transient_correlation']:.3f}\n"
        report += f"Spectral Tilt Change:   {metrics['spectral_tilt_change_db']:.2f} dB\n"

        if "stereo_width_change" in metrics:
            report += f"Stereo Width Change:    {metrics['stereo_width_change'] * 100:.0f}%\n"

        # Interpretation
        spec_dev = metrics["spectral_deviation"]
        if spec_dev < AuthenticityMetrics.SPECTRAL_DEVIATION_WARNING:
            report += (
                f"\n✅ Character PRESERVED (Spectral Deviation < {AuthenticityMetrics.SPECTRAL_DEVIATION_WARNING})\n"
            )
        elif spec_dev < AuthenticityMetrics.SPECTRAL_DEVIATION_CRITICAL:
            report += f"\n⚠️ Moderate character change (Spectral Deviation {spec_dev:.2f} dB)\n"
        else:
            report += f"\n❌ Character ALTERED (Spectral Deviation {spec_dev:.2f} dB > {AuthenticityMetrics.SPECTRAL_DEVIATION_CRITICAL})\n"

        report += f"{'=' * 60}\n"

        return report


# ============================================================================
# Unit Test
# ============================================================================

if __name__ == "__main__":
    # Test Authenticity Metrics
    sr = 48000
    duration = 3.0
    samples = int(duration * sr)

    # Generate clean signal
    t = np.linspace(0, duration, samples)
    audio_original = np.sin(2 * np.pi * 440 * t) * 0.3

    # Test 1: No change
    audio_identical = audio_original.copy()
    is_auth, warns, metrics = AuthenticityMetrics.authenticity_check(audio_original, audio_identical, sr)
    logger.warning("Test 1 (Identical): Authentic = %s, Warnings = %s", is_auth, len(warns))
    assert is_auth, "Identical audio should be authentic"

    # Test 2: Minor change (authentic)
    audio_minor = audio_original * 1.05  # 5% level change
    is_auth, warns, metrics = AuthenticityMetrics.authenticity_check(audio_original, audio_minor, sr)
    logger.info("Test 2 (Minor Change): Authentic = %s, Spectral Dev = %.3f", is_auth, metrics["spectral_deviation"])

    # Test 3: Major change (not authentic)
    audio_major = audio_original + np.random.randn(samples) * 0.5
    is_auth, warns, metrics = AuthenticityMetrics.authenticity_check(audio_original, audio_major, sr)
    logger.warning("Test 3 (Major Change): Authentic = %s, Warnings = %s", is_auth, len(warns))

    # A/B Comparison Report
    report = AuthenticityMetrics.compare_ab(audio_original, audio_minor, sr, labels=("Original", "Minor Change"))
    logger.info(str(report))

    logger.info("\n✓ All tests passed")
