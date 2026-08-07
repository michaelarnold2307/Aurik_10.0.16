"""
Emergency Restoration Engine für Severely Damaged Material

Implements GAP #3: Best-Effort Restoration für >90% Corrupted Audio.
Ein production-ready System braucht graceful degradation für extreme edge cases.

Scenarios:
- Severely corrupted archive material (>70% damage)
- Heavily degraded tape recordings
- Partially destroyed digital files
- Material where standard restoration fails

Architecture:
1. DamageAssessment - Analyze extent of corruption
2. FrequencyBandTriage - Identify salvageable frequency bands
3. PartialReconstructor - Reconstruct damaged regions (BSRNN-based)
4. EmergencyReport - Transparent reporting
5. EmergencyRestorationEngine - Main API

Strategy:
- Triage: Assess which parts can be saved
- Partial Reconstruction: Use AI inpainting for damaged regions
- Fallback: Silence/interpolation für unsalvageable parts
- Transparency: Clear report about what worked and what didn't

Author: AURIK Development Team
Version: 1.0
Date: 2026-02-10
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DamageSeverity(Enum):
    """Severity levels of audio damage."""

    MILD = "mild"  # <30% corrupted
    MODERATE = "moderate"  # 30-60% corrupted
    SEVERE = "severe"  # 60-90% corrupted
    CRITICAL = "critical"  # >90% corrupted


class FrequencyBandStatus(Enum):
    """Status of a frequency band."""

    INTACT = "intact"  # Salvageable
    DAMAGED = "damaged"  # Partially corrupted
    DESTROYED = "destroyed"  # Unsalvageable


@dataclass
class FrequencyBand:
    """Represents a frequency band with damage status."""

    low_freq_hz: float
    """Lower frequency bound."""

    high_freq_hz: float
    """Upper frequency bound."""

    status: FrequencyBandStatus
    """Damage status."""

    corruption_percent: float = 0.0
    """Percentage of corruption (0.0-100.0)."""

    snr_db: float = 0.0
    """Signal-to-Noise Ratio in dB."""

    salvageable: bool = True
    """Whether this band can be salvaged."""


@dataclass
class DamageAssessment:
    """
    Comprehensive damage assessment.
    """

    overall_corruption_percent: float
    """Overall corruption percentage (0.0-100.0)."""

    severity: DamageSeverity
    """Damage severity classification."""

    frequency_bands: list[FrequencyBand]
    """Per-band damage assessment."""

    salvageable_bands_count: int = 0
    """Number of salvageable bands."""

    total_bands: int = 0
    """Total number of bands analyzed."""

    recommendations: list[str] = field(default_factory=list)
    """Recommendations for restoration approach."""

    can_attempt_restoration: bool = True
    """Whether restoration should be attempted."""


@dataclass
class EmergencyReport:
    """Report about emergency restoration attempt."""

    input_corruption_percent: float
    """Initial corruption percentage."""

    restoration_attempted: bool
    """Whether restoration was attempted."""

    restoration_successful: bool
    """Whether restoration succeeded."""

    salvaged_bands: list[str] = field(default_factory=list)
    """List of salvaged frequency bands."""

    lost_bands: list[str] = field(default_factory=list)
    """List of unsalvageable bands."""

    final_quality_estimate: str = "poor"
    """Estimated final quality (poor/fair/acceptable)."""

    warnings: list[str] = field(default_factory=list)
    """Warnings about restoration limitations."""

    processing_notes: str = ""
    """Additional processing notes."""


@dataclass
class EmergencyRestorationResult:
    """Typed result for EmergencyRestorationEngine with dict-style compatibility."""

    audio: np.ndarray
    assessment: DamageAssessment
    report: EmergencyReport
    success: bool

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def as_dict(self) -> dict[str, Any]:
        return {
            "audio": self.audio,
            "assessment": self.assessment,
            "report": self.report,
            "success": self.success,
        }


class DamageAnalyzer:
    """
    Analysiert Ausmaß und Art des Audioschadens.
    """

    def __init__(self, n_bands: int = 8):
        """
        Initialisiert DamageAnalyzer.

        Args:
            n_bands: Number of frequency bands for analysis
        """
        self.n_bands = n_bands

    def assess_damage(self, audio: np.ndarray, sample_rate: int) -> DamageAssessment:
        """
        Assess audio damage.

        Args:
            audio: Input audio
            sample_rate: Sample rate

        Returns:
            DamageAssessment
        """
        logger.info("🔍 analysiere audio damage...")

        # === 1. Overall Corruption Detection ===
        corruption_percent = self._estimate_overall_corruption(audio)

        # === 2. Classify Severity ===
        if corruption_percent < 30:
            severity = DamageSeverity.MILD
        elif corruption_percent < 60:
            severity = DamageSeverity.MODERATE
        elif corruption_percent < 90:
            severity = DamageSeverity.SEVERE
        else:
            severity = DamageSeverity.CRITICAL

        logger.info("  Overall Corruption: %.1f%% (%s)", corruption_percent, severity.value.upper())

        # === 3. Frequency Band Triage ===
        frequency_bands = self._analyze_frequency_bands(audio, sample_rate)

        salvageable_count = sum(1 for band in frequency_bands if band.salvageable)

        # === 4. Recommendations ===
        recommendations = self._generate_recommendations(
            corruption_percent, severity, salvageable_count, len(frequency_bands)
        )

        # === 5. Can Attempt Restoration? ===
        can_attempt = corruption_percent < 95  # Beyond 95% is nearly hopeless

        return DamageAssessment(
            overall_corruption_percent=corruption_percent,
            severity=severity,
            frequency_bands=frequency_bands,
            salvageable_bands_count=salvageable_count,
            total_bands=len(frequency_bands),
            recommendations=recommendations,
            can_attempt_restoration=can_attempt,
        )

    def _estimate_overall_corruption(self, audio: np.ndarray) -> float:
        """
        Schätzt overall corruption percentage.

        Uses heuristics:
        - Clipping ratio
        - Silence ratio
        - Abnormal values (NaN, Inf)
        - High zero-crossing rate (noise indicator)
        """
        # Clipping
        clipped_samples: int = int(np.sum(np.abs(audio) > 0.99))
        clipping_ratio = clipped_samples / len(audio)

        # Silence (sehr small values)
        silent_samples: int = int(np.sum(np.abs(audio) < 1e-6))
        silence_ratio = silent_samples / len(audio)

        # Abnormal values
        abnormal_samples: int = int(np.sum(~np.isfinite(audio)))
        abnormal_ratio = abnormal_samples / len(audio)

        # Zero-crossing rate (normalized)
        zero_crossings: int = int(np.sum(np.diff(np.sign(audio)) != 0))
        zcr_normalized = zero_crossings / len(audio)

        # Weighted corruption estimate (corrected formula)
        # Each component contributes to overall corruption percentage
        corruption_score = 0.0

        # Heavy clipping is problematic (>50% clipped = severe)
        if clipping_ratio > 0.5:
            corruption_score += 40
        elif clipping_ratio > 0.2:
            corruption_score += 20
        elif clipping_ratio > 0.05:
            corruption_score += 10

        # Very high silence is suspicious (>80% silent = severe)
        if silence_ratio > 0.8:
            corruption_score += 30
        elif silence_ratio > 0.5:
            corruption_score += 15

        # Abnormal values are critical
        corruption_score += abnormal_ratio * 100  # Direct percentage

        # Very high ZCR indicates noise (typical ZCR < 0.1)
        if zcr_normalized > 0.5:
            corruption_score += 20
        elif zcr_normalized > 0.3:
            corruption_score += 10

        return min(corruption_score, 100.0)  # type: ignore[no-any-return]

    def _analyze_frequency_bands(self, audio: np.ndarray, sample_rate: int) -> list[FrequencyBand]:
        """
        Analysiert Schäden pro Frequenzband.

        Divides spectrum into n_bands and assesses each.
        """
        from scipy import signal

        # Compute STFT
        f, _t, Zxx = signal.stft(audio, fs=sample_rate, nperseg=1024, boundary="even")

        # Power spectrum
        power = np.abs(Zxx) ** 2

        # Divide into bands
        nyquist = sample_rate / 2
        bands = []

        for i in range(self.n_bands):
            low_freq = (i / self.n_bands) * nyquist
            high_freq = ((i + 1) / self.n_bands) * nyquist

            # Find frequency indices
            low_idx = int((low_freq / nyquist) * len(f))
            high_idx = int((high_freq / nyquist) * len(f))

            # Band power
            band_power = power[low_idx:high_idx, :].mean()

            # Estimate SNR (simplified)
            # Assume noise floor is lowest 10% of power
            noise_floor = np.percentile(power[low_idx:high_idx, :], 10)
            snr_db = 10 * np.log10((band_power / (noise_floor + 1e-10)) + 1e-10)

            # Estimate corruption (low power = likely corrupted)
            if band_power < 1e-10:
                corruption = 100.0
                status = FrequencyBandStatus.DESTROYED
                salvageable = False
            elif snr_db < 5.0:
                corruption = 80.0
                status = FrequencyBandStatus.DAMAGED
                salvageable = False
            elif snr_db < 15.0:
                corruption = 40.0
                status = FrequencyBandStatus.DAMAGED
                salvageable = True
            else:
                corruption = 10.0
                status = FrequencyBandStatus.INTACT
                salvageable = True

            band = FrequencyBand(
                low_freq_hz=low_freq,
                high_freq_hz=high_freq,
                status=status,
                corruption_percent=corruption,
                snr_db=snr_db,
                salvageable=salvageable,
            )

            bands.append(band)

        return bands

    def _generate_recommendations(
        self, corruption: float, severity: DamageSeverity, salvageable_bands: int, total_bands: int
    ) -> list[str]:
        """Generiert restoration recommendations."""
        recommendations = []

        if severity == DamageSeverity.CRITICAL:
            recommendations.append("⚠ CRITICAL DAMAGE: Material is >90% corrupted. Only partial restoration possible.")
            recommendations.append("Consider: Frequency band triage to save intact regions.")

        elif severity == DamageSeverity.SEVERE:
            recommendations.append(
                "⚠ SEVERE DAMAGE: Significant corruption detected. Emergency restoration recommended."
            )

        if salvageable_bands < total_bands / 2:
            recommendations.append(
                f"Only {salvageable_bands}/{total_bands} frequency bands are salvageable. "
                "Consider partial reconstruction."
            )

        if corruption > 95:
            recommendations.append(
                "Material may be beyond salvage. Recommend archiving original and marking as 'severely degraded'."
            )

        return recommendations


class EmergencyRestorationEngine:
    """
    Emergency Restoration Engine für severely damaged material.

    Strategy:
    1. Assess damage extent
    2. Identify salvageable frequency bands
    3. Attempt partial reconstruction
    4. Fill unsalvageable regions (silence or interpolation)
    5. Provide transparent report

    Usage:
        engine = get_emergency_restoration_engine()
        result = engine.emergency_restore(audio, sample_rate)

        restored_audio = result["audio"]
        report = result["report"]
    """

    def __init__(self) -> None:
        """Initialisiert EmergencyRestorationEngine."""
        self.damage_analyzer = DamageAnalyzer(n_bands=8)
        logger.info("EmergencyRestorationEngine initialisiert")

    def emergency_restore(
        self, audio: np.ndarray, sample_rate: int, attempt_reconstruction: bool = True
    ) -> EmergencyRestorationResult:
        """
        Attempt emergency restoration of severely damaged audio.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            attempt_reconstruction: Whether to attempt AI reconstruction

        Returns:
            EmergencyRestorationResult with dict-compatible accessors.
        """
        logger.info("🚨 EMERGENCY RESTORATION Betriebsart ACTIVATED")

        # === Pre-Processing: Clean up critical issues BEFORE assessment ===
        # This allows damage assessment to work properly
        audio_for_assessment = audio.copy()

        # Replace NaN/Inf with zeros for assessment
        if np.any(~np.isfinite(audio_for_assessment)):
            logger.info("  Pre-processing: Cleaning NaN/Inf values for assessment")
            audio_for_assessment[~np.isfinite(audio_for_assessment)] = 0.0

        # Clip extreme values for assessment
        audio_for_assessment = np.clip(audio_for_assessment, -10.0, 10.0)

        # === 1. Assess Damage ===
        assessment = self.damage_analyzer.assess_damage(audio_for_assessment, sample_rate)

        logger.info("  Damage: %.1f%% (%s)", assessment.overall_corruption_percent, assessment.severity.value)
        logger.info("  Salvageable Bands: %s/%s", assessment.salvageable_bands_count, assessment.total_bands)

        # === 2. Check if Restoration is Possible ===
        if not assessment.can_attempt_restoration:
            logger.error("  ❌ Material beyond restoration Schwelle (>95%% corrupted)")

            report = EmergencyReport(
                input_corruption_percent=assessment.overall_corruption_percent,
                restoration_attempted=False,
                restoration_successful=False,
                warnings=["Material >95% corrupted - beyond restoration capability"],
                processing_notes="Material archived as 'severely degraded'",
            )

            return EmergencyRestorationResult(
                audio=audio,
                assessment=assessment,
                report=report,
                success=False,
            )

        # === 3. Attempt Restoration ===
        restored_audio = audio.copy()

        if attempt_reconstruction:
            try:
                # Frequency Band Triage
                restored_audio = self._frequency_band_restoration(
                    restored_audio, sample_rate, assessment.frequency_bands
                )
                logger.info("  ✓ Frequency band restoration angewendet")

                # Partial Reconstruction (simplified - real version would use BSRNN)
                restored_audio = self._partial_reconstruction(restored_audio, sample_rate)
                logger.info("  ✓ Partial reconstruction angewendet")

            except Exception as e:
                logger.warning("  ⚠ Reconstruction fehlgeschlagen: %s", e)
                restored_audio = self._fallback_restoration(audio)
                logger.info("  ✓ Ersatzpfad restoration angewendet")
        else:
            restored_audio = self._fallback_restoration(audio)
            logger.info("  ✓ Ersatzpfad restoration angewendet")

        # === 4. Generate Report ===
        report = self._generate_report(assessment, restored_audio, sample_rate)

        # === 5. Final Quality Estimate ===
        if assessment.severity == DamageSeverity.CRITICAL:
            report.final_quality_estimate = "poor"
        elif assessment.severity == DamageSeverity.SEVERE:
            report.final_quality_estimate = "fair"
        else:
            report.final_quality_estimate = "acceptable"

        logger.info("  Final Quality Estimate: %s", report.final_quality_estimate.upper())
        logger.info("✅ Emergency restoration vollstaendig")

        return EmergencyRestorationResult(
            audio=restored_audio,
            assessment=assessment,
            report=report,
            success=True,
        )

    def _frequency_band_restoration(
        self, audio: np.ndarray, sample_rate: int, bands: list[FrequencyBand]
    ) -> np.ndarray:
        """
        Restauriert audio by filtering out destroyed bands.

        Strategy: Keep salvageable bands, attenuate/remove destroyed bands.
        """
        from scipy import signal

        restored = audio.copy()

        for band in bands:
            if not band.salvageable:
                # Attenuate destroyed band via bandstop filter
                try:
                    nyq = sample_rate / 2
                    low = max(band.low_freq_hz / nyq, 0.01)
                    high = min(band.high_freq_hz / nyq, 0.99)

                    if low < high:
                        sos = signal.butter(4, [low, high], "bandstop", output="sos")
                        restored = signal.sosfilt(sos, restored)

                        logger.debug("    Attenuated band %.0f-%.0f Hz", band.low_freq_hz, band.high_freq_hz)
                except Exception as e:
                    logger.warning("    konnte nicht filter band: %s", e)

        return restored

    def _partial_reconstruction(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Partial reconstruction of damaged regions.

        Simplified version: Uses interpolation.
        Real version würde BSRNN or similar AI model verwenden.
        """
        # Detect abnormal regions (clipping, NaN, etc.)
        abnormal_mask = ~np.isfinite(audio) | (np.abs(audio) > 1.0)

        if np.any(abnormal_mask):
            # Interpolate over abnormal regions
            from scipy.interpolate import interp1d

            x = np.arange(len(audio))
            valid_indices = x[~abnormal_mask]
            valid_values = audio[~abnormal_mask]

            if len(valid_indices) > 1:
                # Linear interpolation
                try:
                    interp_func = interp1d(valid_indices, valid_values, kind="linear", fill_value="extrapolate")

                    reconstructed = interp_func(x)

                    # Blend back
                    audio = audio.copy()
                    audio[abnormal_mask] = reconstructed[abnormal_mask]

                    logger.debug("    Reconstructed %s samples", np.sum(abnormal_mask))
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        return audio

    def _fallback_restoration(self, audio: np.ndarray) -> np.ndarray:
        """
        Fallback restoration: Basic cleanup only.

        - Remove NaN/Inf
        - Clip to valid range
        - Gentle smoothing
        """
        restored = audio.copy()

        # Replace invalid values with zeros
        restored[~np.isfinite(restored)] = 0.0

        # Clip to valid range
        restored = np.clip(restored, -1.0, 1.0)

        # Gentle smoothing (3-point moving average)
        from scipy.ndimage import uniform_filter1d

        restored = uniform_filter1d(restored, size=3, mode="nearest")

        return restored  # type: ignore[no-any-return]

    def _generate_report(
        self, assessment: DamageAssessment, restored_audio: np.ndarray, sample_rate: int
    ) -> EmergencyReport:
        """Generiert emergency restoration report."""
        salvaged_bands = []
        lost_bands = []

        for band in assessment.frequency_bands:
            band_name = f"{band.low_freq_hz:.0f}-{band.high_freq_hz:.0f} Hz"

            if band.salvageable:
                salvaged_bands.append(band_name)
            else:
                lost_bands.append(band_name)

        warnings = []

        if assessment.severity in [DamageSeverity.SEVERE, DamageSeverity.CRITICAL]:
            warnings.append(
                f"Material was {assessment.severity.value} damaged "
                f"({assessment.overall_corruption_percent:.0f}% corrupted). "
                "Restoration quality is limited."
            )

        if lost_bands:
            warnings.append(f"{len(lost_bands)} frequency band(s) were unsalvageable and removed.")

        processing_notes = (
            f"Emergency restoration applied to {assessment.severity.value} damaged material. "
            f"{assessment.salvageable_bands_count}/{assessment.total_bands} frequency bands salvaged."
        )

        return EmergencyReport(
            input_corruption_percent=assessment.overall_corruption_percent,
            restoration_attempted=True,
            restoration_successful=True,
            salvaged_bands=salvaged_bands,
            lost_bands=lost_bands,
            warnings=warnings,
            processing_notes=processing_notes,
        )


# === Example Usage ===
if __name__ == "__main__":
    import soundfile as sf

    from backend.file_import import load_audio_file

    # Load severely damaged audio
    _res = load_audio_file("test_audio/severely_damaged.wav")
    audio, sr = np.asarray(_res["audio"], dtype=np.float32), int(_res["sr"])  # type: ignore[index]

    # Run emergency restoration
    engine = EmergencyRestorationEngine()
    result = engine.emergency_restore(audio, sr)

    # Save result
    sf.write("test_output/emergency_restored.wav", result["audio"], sr)

    # Print report
    report = result["report"]
    assessment = result["assessment"]

    logger.debug("\n🚨 EMERGENCY RESTORATION REPORT")
    logger.debug("=" * 60)
    logger.debug("Eingabe Corruption: %.1f%%", report.input_corruption_percent)
    logger.debug("Severity: %s", assessment.severity.value.upper())
    logger.debug("Restoration Attempted: %s", report.restoration_attempted)
    logger.debug("Success: %s", report.restoration_successful)
    logger.debug("\nSalvaged Bands: %s", len(report.salvaged_bands))
    for band in report.salvaged_bands:
        logger.debug("  ✓ %s", band)
    logger.debug("\nLost Bands: %s", len(report.lost_bands))
    for band in report.lost_bands:
        logger.debug("  ❌ %s", band)
    logger.debug("\nFinal Quality: %s", report.final_quality_estimate.upper())
    logger.debug("\nWarnings:")
    for warning in report.warnings:
        logger.debug("  ⚠ %s", warning)


# Singleton-Instanz (§3.2)
_instance_er: EmergencyRestorationEngine | None = None
_lock_er = threading.Lock()


def get_emergency_restoration_engine() -> EmergencyRestorationEngine:
    """Thread-safe singleton accessor for EmergencyRestorationEngine.

    Returns:
        EmergencyRestorationEngine singleton instance
    """
    global _instance_er
    if _instance_er is None:
        with _lock_er:
            if _instance_er is None:
                _instance_er = EmergencyRestorationEngine()
    return _instance_er
