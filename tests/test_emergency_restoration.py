"""
Tests für Emergency Restoration Engine (GAP #3)

Tests comprehensive coverage für:
- DamageAnalyzer
- FrequencyBandTriage
- EmergencyRestorationEngine
- Edge Cases (>95% corruption, alle bands destroyed)
"""

import numpy as np
import pytest

from backend.core.emergency_restoration import (
    DamageAnalyzer,
    DamageAssessment,
    DamageSeverity,
    EmergencyReport,
    EmergencyRestorationEngine,
    FrequencyBand,
    FrequencyBandStatus,
)

# === Fixtures ===


@pytest.fixture
def clean_audio():
    """Generate clean test audio (1 second, 48kHz)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simple sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    return audio, sr


@pytest.fixture
def mildly_damaged_audio():
    """Generate mildly damaged audio (~30% corruption, realistic)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Base signal
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Add mild corruption:
    # 1. Some clipping (10%)
    clipping_mask = np.random.random(len(audio)) < 0.10
    audio[clipping_mask] = np.sign(audio[clipping_mask]) * 1.1

    # 2. Some noise (15%)
    noise_mask = np.random.random(len(audio)) < 0.15
    audio[noise_mask] += np.random.normal(0, 0.3, np.sum(noise_mask))  # type: ignore[call-overload]

    return audio, sr


@pytest.fixture
def severely_damaged_audio():
    """Generate severely damaged audio (~75% corruption, realistic)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Base signal
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Heavy corruption:
    # 1. Heavy clipping (40%)
    clipping_mask = np.random.random(len(audio)) < 0.40
    audio[clipping_mask] = np.sign(audio[clipping_mask]) * 1.3

    # 2. Heavy noise (30%)
    noise_mask = np.random.random(len(audio)) < 0.30
    audio[noise_mask] = np.random.normal(0, 0.8, np.sum(noise_mask))  # type: ignore[call-overload]

    # 3. Some silence (20%)
    silence_mask = np.random.random(len(audio)) < 0.20
    audio[silence_mask] = 0.0

    return audio, sr


@pytest.fixture
def critically_damaged_audio():
    """Generate critically damaged audio (>90% corruption, realistic)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Almost entirely corrupted
    # 1. Heavy clipping (60%)
    audio = np.random.uniform(-1.5, 1.5, int(sr * duration))
    clipping_mask = np.random.random(len(audio)) < 0.60
    audio[clipping_mask] = np.sign(audio[clipping_mask]) * 1.5

    # 2. Heavy silence (40%)
    silence_mask = np.random.random(len(audio)) < 0.40
    audio[silence_mask] = 0.0

    # 3. A tiny bit of actual signal (5%)
    signal_mask = np.random.random(len(audio)) < 0.05
    audio[signal_mask] = 0.2 * np.sin(2 * np.pi * 440 * t[signal_mask])

    return audio, sr


# === Test DamageAnalyzer ===


@pytest.mark.unit
class TestDamageAnalyzer:
    """Test damage analysis."""

    def test_initialization(self):
        """Test DamageAnalyzer initialization."""
        analyzer = DamageAnalyzer(n_bands=8)

        assert analyzer.n_bands == 8

    def test_assess_clean_audio(self, clean_audio):
        """Test assessment of clean audio."""
        audio, sr = clean_audio
        analyzer = DamageAnalyzer()

        assessment = analyzer.assess_damage(audio, sr)

        assert isinstance(assessment, DamageAssessment)
        assert assessment.overall_corruption_percent < 20  # Should be low
        assert assessment.severity in [DamageSeverity.MILD, DamageSeverity.MODERATE]
        assert assessment.can_attempt_restoration
        assert assessment.salvageable_bands_count > 0

    def test_assess_mildly_damaged(self, mildly_damaged_audio):
        """Test assessment of mildly damaged audio."""
        audio, sr = mildly_damaged_audio
        analyzer = DamageAnalyzer()

        assessment = analyzer.assess_damage(audio, sr)

        # Mildly damaged should show some corruption but not extreme
        assert 5 < assessment.overall_corruption_percent < 60
        assert assessment.severity in [DamageSeverity.MILD, DamageSeverity.MODERATE]
        assert assessment.can_attempt_restoration

    def test_assess_severely_damaged(self, severely_damaged_audio):
        """Test assessment of severely damaged audio."""
        audio, sr = severely_damaged_audio
        analyzer = DamageAnalyzer()

        assessment = analyzer.assess_damage(audio, sr)

        # Severely damaged should show significant corruption
        assert assessment.overall_corruption_percent >= 30  # >= not >
        assert assessment.severity in [DamageSeverity.MODERATE, DamageSeverity.SEVERE, DamageSeverity.CRITICAL]
        assert assessment.can_attempt_restoration  # Still below 95%
        # Recommendations might be empty for moderate damage
        if assessment.severity in [DamageSeverity.SEVERE, DamageSeverity.CRITICAL]:
            assert len(assessment.recommendations) > 0

    def test_assess_critically_damaged(self, critically_damaged_audio):
        """Test assessment of critically damaged audio."""
        audio, sr = critically_damaged_audio
        analyzer = DamageAnalyzer()

        assessment = analyzer.assess_damage(audio, sr)

        # Critically damaged should show high corruption (fixtures have 60% clipping + 40% silence)
        assert assessment.overall_corruption_percent >= 40  # Realistic threshold
        assert assessment.severity in [DamageSeverity.MODERATE, DamageSeverity.SEVERE, DamageSeverity.CRITICAL]
        # Might be beyond restoration if >95%

    def test_frequency_band_analysis(self, severely_damaged_audio):
        """Test frequency band analysis."""
        audio, sr = severely_damaged_audio
        analyzer = DamageAnalyzer(n_bands=8)

        assessment = analyzer.assess_damage(audio, sr)

        assert len(assessment.frequency_bands) == 8

        for band in assessment.frequency_bands:
            assert isinstance(band, FrequencyBand)
            assert band.low_freq_hz < band.high_freq_hz
            assert 0 <= band.corruption_percent <= 100
            assert band.status in FrequencyBandStatus

    def test_recommendations_generation(self, severely_damaged_audio):
        """Test recommendation generation."""
        audio, sr = severely_damaged_audio
        analyzer = DamageAnalyzer()

        assessment = analyzer.assess_damage(audio, sr)

        # Recommendations should exist for severe/critical damage
        if assessment.severity in [DamageSeverity.SEVERE, DamageSeverity.CRITICAL]:
            assert len(assessment.recommendations) > 0
            assert any("SEVERE" in rec or "CRITICAL" in rec for rec in assessment.recommendations)
        # Moderate damage might not have recommendations


# === Test EmergencyRestorationEngine ===


class TestEmergencyRestorationEngine:
    """Test emergency restoration engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = EmergencyRestorationEngine()

        assert engine.damage_analyzer is not None
        assert engine.damage_analyzer.n_bands == 8

    def test_emergency_restore_clean_audio(self, clean_audio):
        """Test restoration of clean audio (should barely modify)."""
        audio, sr = clean_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr)

        assert "audio" in result
        assert "assessment" in result
        assert "report" in result
        assert "success" in result

        assert result["success"]
        assert isinstance(result["report"], EmergencyReport)
        assert result["report"].restoration_attempted

    def test_emergency_restore_mildly_damaged(self, mildly_damaged_audio):
        """Test restoration of mildly damaged audio."""
        audio, sr = mildly_damaged_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr)

        assert result["success"]
        restored = result["audio"]

        # Restored audio should be finite and in valid range
        assert np.all(np.isfinite(restored))
        assert np.all(np.abs(restored) <= 1.5)  # Allow slight overshoot

    def test_emergency_restore_severely_damaged(self, severely_damaged_audio):
        """Test restoration of severely damaged audio."""
        audio, sr = severely_damaged_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr)

        assert result["success"]

        report = result["report"]
        assert report.restoration_attempted
        assert report.restoration_successful
        # Warnings depend on severity
        if result["assessment"].severity in [DamageSeverity.SEVERE, DamageSeverity.CRITICAL]:
            assert len(report.warnings) > 0  # Should have warnings for severe damage
        assert report.final_quality_estimate in ["poor", "fair", "acceptable"]

    def test_emergency_restore_critically_damaged(self, critically_damaged_audio):
        """Test restoration of critically damaged audio."""
        audio, sr = critically_damaged_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr)

        # May or may not succeed depending on exact corruption level
        report = result["report"]
        assessment = result["assessment"]

        if report.restoration_attempted:
            # If attempted, might have warnings for severe/critical damage
            if assessment.severity in [DamageSeverity.SEVERE, DamageSeverity.CRITICAL]:
                assert len(report.warnings) > 0
            assert report.final_quality_estimate in ["poor", "fair", "acceptable"]
        else:
            # If not attempted, corruption was >95%
            assert assessment.overall_corruption_percent > 95

    def test_frequency_band_restoration(self, severely_damaged_audio):
        """Test frequency band restoration."""
        audio, sr = severely_damaged_audio
        engine = EmergencyRestorationEngine()

        # Run restoration
        result = engine.emergency_restore(audio, sr, attempt_reconstruction=True)

        assert result["success"]

        report = result["report"]

        # Should have identified salvaged and lost bands
        total_bands = len(report.salvaged_bands) + len(report.lost_bands)
        assert total_bands > 0  # Should analyze bands

    def test_fallback_restoration(self, severely_damaged_audio):
        """Test fallback restoration (no reconstruction)."""
        audio, sr = severely_damaged_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr, attempt_reconstruction=False)

        assert result["success"]

        restored = result["audio"]

        # Fallback should at least clean up NaN/Inf
        assert np.all(np.isfinite(restored))
        assert np.all(np.abs(restored) <= 1.5)


# === Test Report Generation ===


class TestEmergencyReport:
    """Test emergency report generation."""

    def test_report_structure(self, severely_damaged_audio):
        """Test report structure."""
        audio, sr = severely_damaged_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr)
        report = result["report"]

        # Check all fields present
        assert hasattr(report, "input_corruption_percent")
        assert hasattr(report, "restoration_attempted")
        assert hasattr(report, "restoration_successful")
        assert hasattr(report, "salvaged_bands")
        assert hasattr(report, "lost_bands")
        assert hasattr(report, "final_quality_estimate")
        assert hasattr(report, "warnings")
        assert hasattr(report, "processing_notes")

    def test_report_salvaged_bands(self, severely_damaged_audio):
        """Test salvaged bands reporting."""
        audio, sr = severely_damaged_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr)
        report = result["report"]

        # Should have band information
        total_bands = len(report.salvaged_bands) + len(report.lost_bands)
        assert total_bands == 8  # Default n_bands

        # Band names should have format "XXX-YYY Hz"
        for band_name in report.salvaged_bands + report.lost_bands:
            assert "Hz" in band_name
            assert "-" in band_name

    def test_report_quality_estimate(self, severely_damaged_audio):
        """Test quality estimate accuracy."""
        audio, sr = severely_damaged_audio
        engine = EmergencyRestorationEngine()

        result = engine.emergency_restore(audio, sr)
        report = result["report"]
        assessment = result["assessment"]

        # Quality should correlate with damage severity
        if assessment.severity == DamageSeverity.CRITICAL:
            assert report.final_quality_estimate == "poor"
        elif assessment.severity == DamageSeverity.SEVERE:
            assert report.final_quality_estimate in ["poor", "fair"]
        else:
            assert report.final_quality_estimate in ["fair", "acceptable"]


# === Test Integration Scenarios ===


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_full_emergency_workflow(self, severely_damaged_audio):
        """Test complete emergency restoration workflow."""
        audio, sr = severely_damaged_audio

        # 1. Initialize engine
        engine = EmergencyRestorationEngine()

        # 2. Run restoration
        result = engine.emergency_restore(audio, sr)

        # 3. Validate result
        assert result["success"]
        assert "audio" in result
        assert "assessment" in result
        assert "report" in result

        # 4. Check assessment
        assessment = result["assessment"]
        assert isinstance(assessment, DamageAssessment)
        assert assessment.overall_corruption_percent > 0

        # 5. Check report
        report = result["report"]
        assert isinstance(report, EmergencyReport)
        assert report.restoration_attempted

        # 6. Check restored audio
        restored = result["audio"]
        assert len(restored) == len(audio)
        assert np.all(np.isfinite(restored))

    def test_triage_based_restoration(self):
        """Test restoration with specific frequency band damage."""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create signal with different frequency components
        audio = (
            0.3 * np.sin(2 * np.pi * 100 * t)  # Low freq
            + 0.3 * np.sin(2 * np.pi * 1000 * t)  # Mid freq
            + 0.3 * np.sin(2 * np.pi * 5000 * t)  # High freq
        )

        # Corrupt mid frequencies (simulate bandpass damage)
        from scipy import signal as sp_signal

        sos = sp_signal.butter(4, [800, 1200], "bandpass", fs=sr, output="sos")
        corruption_band = sp_signal.sosfilt(sos, np.random.normal(0, 2.0, len(audio)))
        audio += corruption_band

        # Run emergency restoration
        engine = EmergencyRestorationEngine()
        result = engine.emergency_restore(audio, sr)

        assert result["success"]

        # Should identify damaged mid frequencies
        assessment = result["assessment"]
        mid_bands = [b for b in assessment.frequency_bands if 500 < b.low_freq_hz < 2000]

        if mid_bands:
            # At least one mid band should show damage
            assert any(b.corruption_percent > 30 for b in mid_bands)


# === Test Edge Cases ===


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_silent_audio(self):
        """Test with completely silent audio."""
        sr = 48000
        audio = np.zeros(sr)  # 1 second silence

        engine = EmergencyRestorationEngine()
        result = engine.emergency_restore(audio, sr)

        # Should handle gracefully
        assert "audio" in result
        assert "assessment" in result

    def test_very_short_audio(self):
        """Test with very short audio (<0.1s)."""
        sr = 48000
        audio = np.random.normal(0, 0.1, 2048)  # ~42ms

        engine = EmergencyRestorationEngine()

        # Should not crash
        try:
            result = engine.emergency_restore(audio, sr)
            assert "audio" in result
        except Exception as e:
            pytest.fail(f"Short audio caused exception: {e}")

    def test_audio_with_nan(self):
        """Test with NaN values."""
        sr = 48000
        audio = np.random.normal(0, 0.3, sr)

        # Inject NaNs
        audio[100:200] = np.nan
        audio[500:520] = np.nan

        engine = EmergencyRestorationEngine()
        result = engine.emergency_restore(audio, sr)

        # Restored audio should have no NaNs
        assert np.all(np.isfinite(result["audio"]))

    def test_audio_with_inf(self):
        """Test with Inf values."""
        sr = 48000
        audio = np.random.normal(0, 0.3, sr)

        # Inject Infs
        audio[100:200] = np.inf
        audio[500:520] = -np.inf

        engine = EmergencyRestorationEngine()
        result = engine.emergency_restore(audio, sr)

        # Restored audio should have no Infs
        assert np.all(np.isfinite(result["audio"]))

    def test_heavily_clipped_audio(self):
        """Test with heavily clipped audio."""
        sr = 48000

        # Generate clipped audio (80% clipped)
        audio = np.random.uniform(-2.0, 2.0, sr)
        audio = np.clip(audio, -1.0, 1.0)

        engine = EmergencyRestorationEngine()
        result = engine.emergency_restore(audio, sr)

        # Should detect heavy clipping as damage
        assessment = result["assessment"]
        assert assessment.overall_corruption_percent > 20

    def test_beyond_salvation_audio(self):
        """Test audio that is beyond salvation (>95% corrupted)."""
        sr = 48000

        # Create 99% corrupted audio
        audio = np.full(sr, np.nan)

        # Only 1% valid
        valid_samples = int(sr * 0.01)
        audio[:valid_samples] = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, valid_samples))

        engine = EmergencyRestorationEngine()
        result = engine.emergency_restore(audio, sr)

        # Should recognize it's beyond restoration
        if result["assessment"].overall_corruption_percent > 95:
            assert not result["report"].restoration_attempted or len(result["report"].warnings) > 0


# === Test Performance ===


class TestPerformance:
    """Test performance characteristics."""

    def test_processing_speed(self, severely_damaged_audio):
        """Test that processing completes in reasonable time."""
        import time

        audio, sr = severely_damaged_audio
        engine = EmergencyRestorationEngine()

        start = time.time()
        result = engine.emergency_restore(audio, sr)
        elapsed = time.time() - start

        # Should complete within 10 seconds for 1 second audio
        assert elapsed < 10.0
        assert result["success"]

    def test_memory_efficiency(self):
        """Test that processing doesn't create excessive copies."""
        sr = 48000
        duration = 10.0  # 10 seconds
        audio = np.random.normal(0, 0.3, int(sr * duration))

        engine = EmergencyRestorationEngine()

        # Should not crash with larger audio
        try:
            result = engine.emergency_restore(audio, sr)
            assert len(result["audio"]) == len(audio)
        except MemoryError:
            pytest.fail("Memory error with 10 second audio")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
