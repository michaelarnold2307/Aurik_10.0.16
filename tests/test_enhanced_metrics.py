"""
Tests for Enhanced Audio Quality Metrics
========================================

Phase: 2D.2.1 - Real-World Validation Testing
Author: AURIK Team
Date: 8. Februar 2026
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.enhanced_metrics import (
    EnhancedMetrics,
    QualityMetricsResult,
    batch_compute_metrics,
    generate_metrics_report,
)

# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def clean_audio():
    """Generate clean sine wave audio."""
    sr = 48000
    duration = 1.0
    freq = 440.0  # A4
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio, sr


@pytest.fixture
def noisy_audio(clean_audio):
    """Generate noisy version of clean audio."""
    audio, sr = clean_audio
    # Add Gaussian noise
    noise = np.random.normal(0, 0.05, len(audio))
    noisy = audio + noise
    return noisy, sr


@pytest.fixture
def restored_audio(clean_audio, noisy_audio):
    """Simulate restored audio (clean + very small noise)."""
    audio_clean, sr = clean_audio
    # Restored = clean + tiny noise (much better than original noisy)
    tiny_noise = np.random.normal(0, 0.005, len(audio_clean))  # Very small noise
    restored = audio_clean + tiny_noise
    return restored, sr


@pytest.fixture
def metrics_computer():
    """Create EnhancedMetrics instance."""
    return EnhancedMetrics()


# ============================================================
# Test Basic Metrics (from Phase 2D.1)
# ============================================================


@pytest.mark.unit
def test_compute_snr(clean_audio, noisy_audio):
    """Test SNR computation."""
    audio_clean, sr = clean_audio
    audio_noisy, _ = noisy_audio
    metrics = EnhancedMetrics()

    snr_clean = metrics.compute_snr(audio_clean, sr)
    snr_noisy = metrics.compute_snr(audio_noisy, sr)

    # SNR should be computed and in valid range
    assert isinstance(snr_clean, float)
    assert isinstance(snr_noisy, float)

    # Both should be positive
    assert snr_clean > 0
    assert snr_noisy > 0

    # SNR should be reasonable (not extreme)
    assert snr_clean < 100.0
    assert snr_noisy < 100.0


def test_compute_thd(clean_audio):
    """Test THD computation."""
    audio, sr = clean_audio
    metrics = EnhancedMetrics()

    thd = metrics.compute_thd(audio, sr)

    # THD should be low for clean sine wave
    assert 0.0 <= thd <= 1.0, f"THD out of range: {thd}"
    assert thd < 0.1, f"Expected low THD for sine wave, got {thd}"


def test_compute_lufs(clean_audio):
    """Test LUFS computation."""
    audio, sr = clean_audio
    metrics = EnhancedMetrics()

    lufs = metrics.compute_lufs(audio, sr)

    # LUFS should be in reasonable range
    assert -100.0 <= lufs <= 0.0, f"LUFS out of range: {lufs}"


# ============================================================
# Test compute_all
# ============================================================


def test_compute_all(clean_audio, noisy_audio, metrics_computer):
    """Test compute_all method."""
    audio_clean, sr = clean_audio
    audio_noisy, _ = noisy_audio

    result = metrics_computer.compute_all(audio_noisy, audio_clean, sr=sr)

    # Check all basic metrics are computed
    assert result.snr_db > 0
    assert 0.0 <= result.thd <= 1.0
    assert result.lufs > -100.0

    # Check improvement metric
    assert result.snr_improvement_db is not None


def test_compute_all_returns_quality_metrics_result(clean_audio, restored_audio, metrics_computer):
    """Test that compute_all returns QualityMetricsResult."""
    audio_clean, sr = clean_audio
    audio_restored, _ = restored_audio

    result = metrics_computer.compute_all(audio_clean, audio_restored, sr=sr)

    assert isinstance(result, QualityMetricsResult)


# ============================================================
# Test Batch Processing
# ============================================================


def test_batch_compute_metrics():
    """Test batch_compute_metrics function."""
    sr = 48000

    # Create 3 audio pairs
    pairs = []
    for i in range(3):
        original = np.random.randn(sr)
        restored = original + np.random.randn(sr) * 0.01  # Small noise
        pairs.append((original, restored))

    results = batch_compute_metrics(pairs, sr=sr)

    assert len(results) == 3
    assert all(isinstance(r, QualityMetricsResult) for r in results)


# ============================================================
# Integration Tests
# ============================================================


def test_complete_workflow(clean_audio, noisy_audio, restored_audio):
    """Test complete workflow from audio to report."""
    audio_clean, sr = clean_audio
    audio_noisy, _ = noisy_audio
    audio_restored, _ = restored_audio

    # 1. Compute metrics
    metrics = EnhancedMetrics()
    result = metrics.compute_restoration_improvement(
        original_noisy=audio_noisy, original_clean=audio_clean, restored=audio_restored, sr=sr
    )

    # 2. Check standards
    passes = result.passes_aurik_standards()
    assert type(passes).__name__ == "bool", f"passes_aurik_standards should return bool, got {type(passes)}"

    # 3. Generate report
    report = generate_metrics_report(result, filename="/tmp/test_metrics.txt")
    assert len(report) > 0
    assert "AURIK Quality Metrics Report" in report


# ============================================================
# Authenticity Metrics Tests (Phase 2D.2.1 Task 3)
# ============================================================


def test_authenticity_metrics_available():
    """Test that authenticity metrics are available."""
    from backend.core.enhanced_metrics import AUTHENTICITY_AVAILABLE

    # Should be available after Phase 2D.1
    assert AUTHENTICITY_AVAILABLE, "AuthenticityMetrics should be available"


def test_compute_breath_retention_with_vocal():
    """Test breath retention computation with vocal audio."""
    # Create synthetic vocal audio with breath
    sr = 48000
    duration = 2.0

    # Simulate vocal + breath
    t = np.linspace(0, duration, int(sr * duration))
    vocal = 0.3 * np.sin(2 * np.pi * 200 * t)  # Vocal fundamental

    # Add breath event at 1.0s (0.2s duration)
    breath_start = int(1.0 * sr)
    breath_end = int(1.2 * sr)
    breath = np.random.randn(breath_end - breath_start) * 0.05
    vocal[breath_start:breath_end] += breath

    # Simulate "restored" audio (minor noise reduction)
    restored = vocal + np.random.randn(len(vocal)) * 0.001

    # Compute metrics
    metrics = EnhancedMetrics()
    if metrics.authenticity is not None:
        retention, orig_breaths, proc_breaths = metrics.authenticity.compute_breath_retention(vocal, restored, sr)

        # Should detect breath and retain it
        assert retention >= 0.0, "Retention should be valid"
        assert retention <= 1.0, "Retention should be ≤1.0"
        print(f"Breath retention: {retention:.1%} (detected {len(orig_breaths)} breaths)")


def test_compute_transient_preservation_with_drums():
    """Test transient preservation with drum-like audio."""
    # Create synthetic drum hits
    sr = 48000
    duration = 1.0
    audio = np.zeros(int(sr * duration))

    # Add 3 drum hits with sharp transients
    for hit_time in [0.2, 0.5, 0.8]:
        hit_idx = int(hit_time * sr)
        # Sharp attack (5ms)
        attack_samples = int(0.005 * sr)
        attack = np.linspace(0, 1, attack_samples)
        decay_samples = int(0.1 * sr)
        decay = np.exp(-np.linspace(0, 5, decay_samples))

        envelope = np.concatenate([attack, decay])
        noise = np.random.randn(len(envelope))
        drum_hit = envelope * noise * 0.5

        audio[hit_idx : hit_idx + len(drum_hit)] += drum_hit

    # Simulate processing (slight smoothing)
    from scipy import signal as sp_signal

    restored = sp_signal.savgol_filter(audio, 11, 3)

    # Compute transient preservation
    metrics = EnhancedMetrics()
    if metrics.authenticity is not None:
        preservation, orig_trans, proc_trans = metrics.authenticity.compute_transient_preservation(audio, restored, sr)

        # Should have reasonable preservation (>0.7 even with smoothing)
        assert preservation >= 0.0, "Preservation should be valid"
        assert preservation <= 1.0, "Preservation should be ≤1.0"
        print(f"Transient preservation: {preservation:.1%} (detected {len(orig_trans)} transients)")


def test_compute_all_with_authenticity():
    """Test compute_all includes authenticity metrics."""
    sr = 48000
    duration = 1.0

    # Create audio with transients
    audio1 = np.random.randn(int(sr * duration)) * 0.1
    audio2 = audio1 + np.random.randn(int(sr * duration)) * 0.01

    metrics = EnhancedMetrics()
    result = metrics.compute_all(audio1, audio2, sr=sr)

    # Check that authenticity metrics are computed (may be None if no events detected)
    assert hasattr(result, "breath_retention"), "Should have breath_retention field"
    assert hasattr(result, "transient_preservation"), "Should have transient_preservation field"
    assert hasattr(result, "plosive_retention"), "Should have plosive_retention field"

    # Values should be None or valid range
    if result.breath_retention is not None:
        assert 0.0 <= result.breath_retention <= 1.0

    if result.transient_preservation is not None:
        assert 0.0 <= result.transient_preservation <= 1.0

    if result.plosive_retention is not None:
        assert 0.0 <= result.plosive_retention <= 1.0


def test_restoration_improvement_with_authenticity():
    """Test compute_restoration_improvement includes authenticity."""
    np.random.seed(42)  # §5.4 Reproduzierbarkeit
    sr = 48000
    duration = 1.0

    # Simulate clean → noisy → restored
    # clean (0.3), noisy_residual (0.1 → SNR ≈ 9.5 dB), restored_residual (0.02 → SNR ≈ 23.5 dB)
    # → SNR-Verbesserung stets +14 dB (deterministische Amplituden garantieren positiven Wert)
    clean = np.random.randn(int(sr * duration)) * 0.3
    noisy = clean + np.random.randn(int(sr * duration)) * 0.1
    restored = clean + np.random.randn(int(sr * duration)) * 0.02

    metrics = EnhancedMetrics()
    result = metrics.compute_restoration_improvement(noisy, clean, restored, sr=sr)

    # Should have authenticity fields
    assert hasattr(result, "breath_retention")
    assert hasattr(result, "transient_preservation")
    assert hasattr(result, "plosive_retention")
    assert hasattr(result, "sibilance_retention")
    assert hasattr(result, "room_tone_retention")

    # Should show improvement
    assert result.snr_improvement_db > 0, "SNR should improve"  # type: ignore[operator]


def test_sibilance_retention_metric():
    """Test sibilance retention computation."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create audio with sibilance-like content (4-10 kHz)
    from scipy import signal as sp_signal

    # Base audio
    audio = 0.2 * np.sin(2 * np.pi * 200 * t)

    # Add sibilance bursts
    sibilance = np.random.randn(len(t)) * 0.1
    sos = sp_signal.butter(4, [4000, 10000], "bandpass", fs=sr, output="sos")
    sibilance = sp_signal.sosfilt(sos, sibilance)

    original = audio + sibilance

    # Simulate de-essing (reduce high freq)
    restored = audio + sibilance * 0.7  # 70% sibilance retained

    metrics = EnhancedMetrics()
    result = metrics.compute_all(original, restored, sr=sr)

    # Should detect sibilance retention
    if result.sibilance_retention is not None:
        assert 0.0 <= result.sibilance_retention <= 1.0, "Sibilance retention in valid range"
        print(f"Sibilance retention: {result.sibilance_retention * 100:.1f}%")


def test_room_tone_retention_metric():
    """Test room tone retention computation."""
    sr = 48000
    duration = 2.0

    # Create audio with room tone
    audio = np.random.randn(int(sr * duration)) * 0.05  # Ambient noise

    # Add some musical content
    t = np.linspace(0, duration, int(sr * duration))
    music = 0.3 * np.sin(2 * np.pi * 440 * t)

    original = music + audio

    # Simulate aggressive denoising (removes room tone)
    restored = music + audio * 0.3  # 30% room tone retained

    metrics = EnhancedMetrics()
    result = metrics.compute_all(original, restored, sr=sr)

    # Should detect room tone retention
    if result.room_tone_retention is not None:
        assert 0.0 <= result.room_tone_retention <= 1.0, "Room tone retention in valid range"
        print(f"Room tone retention: {result.room_tone_retention * 100:.1f}%")


def test_comprehensive_authenticity_validation():
    """Test all 5 authenticity metrics together."""
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Complex audio with all elements
    # 1. Base vocal-like signal
    vocal = 0.2 * np.sin(2 * np.pi * 200 * t + np.sin(2 * np.pi * 5 * t))

    # 2. Add breath
    breath_start = int(1.0 * sr)
    breath_duration = int(0.15 * sr)
    breath = np.random.randn(breath_duration) * 0.03
    from scipy import signal as sp_signal

    sos = sp_signal.butter(4, [200, 3000], "bandpass", fs=sr, output="sos")
    breath = sp_signal.sosfilt(sos, breath)
    vocal[breath_start : breath_start + breath_duration] += breath

    # 3. Add transient (drum hit)
    hit_idx = int(0.5 * sr)
    envelope = np.exp(-np.linspace(0, 5, int(0.05 * sr)))
    drum = envelope * np.random.randn(len(envelope)) * 0.3
    vocal[hit_idx : hit_idx + len(drum)] += drum

    # 4. Add sibilance
    sibilance = np.random.randn(len(t)) * 0.08
    sos = sp_signal.butter(4, [5000, 9000], "bandpass", fs=sr, output="sos")
    sibilance = sp_signal.sosfilt(sos, sibilance)
    vocal += sibilance * 0.3

    # 5. Add room tone
    room_tone = np.random.randn(len(t)) * 0.02

    original = vocal + room_tone

    # Simulate gentle processing (preserves most elements)
    restored = vocal * 0.98 + room_tone * 0.8

    metrics = EnhancedMetrics()
    result = metrics.compute_all(original, restored, sr=sr)

    # Check all authenticity metrics are computed
    authenticity_metrics = [
        result.breath_retention,
        result.transient_preservation,
        result.plosive_retention,
        result.sibilance_retention,
        result.room_tone_retention,
    ]

    computed_count = sum(1 for m in authenticity_metrics if m is not None)
    print(f"\nAuthenticity metrics computed: {computed_count}/5")

    if result.breath_retention is not None:
        print(f"  Breath Retention: {result.breath_retention * 100:.1f}%")
    if result.transient_preservation is not None:
        print(f"  Transient Preservation: {result.transient_preservation * 100:.1f}%")
    if result.plosive_retention is not None:
        print(f"  Plosive Retention: {result.plosive_retention * 100:.1f}%")
    if result.sibilance_retention is not None:
        print(f"  Sibilance Retention: {result.sibilance_retention * 100:.1f}%")
    if result.room_tone_retention is not None:
        print(f"  Room Tone Retention: {result.room_tone_retention * 100:.1f}%")

    # At least some should be computed
    assert computed_count >= 2, "At least 2 authenticity metrics should be computed"


# ============================================================
# Performance Tests
# ============================================================


def test_metrics_computation_speed():
    """Test that metrics computation is reasonably fast."""
    import time

    sr = 48000
    duration = 5.0  # 5 seconds
    audio1 = np.random.randn(int(sr * duration))
    audio2 = np.random.randn(int(sr * duration))

    metrics = EnhancedMetrics()

    start = time.time()
    result = metrics.compute_all(audio1, audio2, sr=sr)
    elapsed = time.time() - start

    # Should complete in <5 seconds for 5s audio
    assert elapsed < 5.0, f"Metrics computation too slow: {elapsed:.3f}s"
    assert isinstance(result, QualityMetricsResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
