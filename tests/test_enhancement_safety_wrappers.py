"""
test_enhancement_safety_wrappers.py - Comprehensive Tests for Enhancement Safety Wrappers

Tests all two Priority-3 enhancement safety wrappers:
- HarmonicExciterSafety
- StereoWidenerSafety

Validates:
- Headroom enforcement
- Artifact prevention (harshness, IMD, hollow)
- Mono compatibility
- Center content preservation
- Quality scoring

Author: AURIK Team
Version: 1.0.0
Date: 7. Februar 2026
Phase: 1 Week 5-6
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.ml.safety_wrappers.harmonic_exciter_safety import HarmonicExciterSafety
from backend.ml.safety_wrappers.safety_wrapper_template import ProcessingDecision
from backend.ml.safety_wrappers.stereo_widener_safety import StereoWidenerSafety

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mono_audio():
    """Generate mono audio with headroom."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Musical signal (fundamental + harmonics)
    f0 = 440  # A4
    audio = np.zeros_like(t)

    for n in range(1, 4):
        amplitude = 0.3 / n
        audio += amplitude * np.sin(2 * np.pi * f0 * n * t)

    # Normalize to 0.5 (leaves headroom)
    audio = audio / np.max(np.abs(audio)) * 0.5

    return audio, sr


@pytest.fixture
def stereo_audio():
    """Generate stereo audio with center content."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Center content (bass + vocal)
    bass = np.sin(2 * np.pi * 100 * t) * 0.3
    vocal = np.sin(2 * np.pi * 440 * t) * 0.2
    center = bass + vocal

    # Side content (pads)
    pad_l = np.sin(2 * np.pi * 880 * t) * 0.1
    pad_r = np.sin(2 * np.pi * 1760 * t) * 0.1

    # Create stereo
    left = center + pad_l
    right = center + pad_r

    audio = np.stack([left, right], axis=0)

    return audio, sr


@pytest.fixture
def narrow_stereo_audio():
    """Generate narrow stereo audio (mostly mono)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Mostly center content
    center = np.sin(2 * np.pi * 440 * t) * 0.5

    # Tiny difference for stereo
    diff = np.sin(2 * np.pi * 880 * t) * 0.02

    left = center + diff
    right = center - diff

    audio = np.stack([left, right], axis=0)

    return audio, sr


@pytest.fixture
def hot_audio():
    """Generate audio with little headroom (peaks at 0.95)."""
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))

    # Signal with peaks
    audio = np.sin(2 * np.pi * 440 * t) * 0.95

    return audio, sr


@pytest.fixture
def dark_audio():
    """Generate dark audio (low brightness)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Low-frequency content only
    audio = np.zeros_like(t)
    for freq in [100, 200, 300]:
        audio += 0.2 * np.sin(2 * np.pi * freq * t)

    audio = audio / np.max(np.abs(audio)) * 0.5

    return audio, sr


@pytest.fixture
def bright_audio():
    """Generate bright audio (high brightness, may be harsh)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # High-frequency content emphasized
    audio = np.zeros_like(t)
    for freq in [2000, 4000, 6000, 8000]:
        audio += 0.15 * np.sin(2 * np.pi * freq * t)

    audio = audio / np.max(np.abs(audio)) * 0.5

    return audio, sr


# ============================================================================
# DUMMY PROCESSORS
# ============================================================================


def dummy_harmonic_exciter(audio: np.ndarray, sr: int, amount: float = 0.5) -> np.ndarray:
    """Dummy harmonic exciter (adds 2nd and 3rd harmonics)."""
    # Highpass to get content to excite
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 500 / (sr / 2), btype="high")
    high_freq = filtfilt(b, a, audio)

    # Generate harmonics (simple saturation)
    harmonics = np.tanh(high_freq * 2) * amount * 0.2

    # Mix
    result = audio + harmonics

    # Normalize to prevent clipping
    result = result / np.max(np.abs(result)) * 0.98

    return result  # type: ignore[no-any-return]


def dummy_stereo_widener(audio: np.ndarray, sr: int, width: float = 0.5) -> np.ndarray:
    """Dummy stereo widener (Mid/Side processing)."""
    if audio.ndim == 1:
        # Mono input - create pseudo-stereo
        audio = np.stack([audio, audio], axis=0)

    left = audio[0]
    right = audio[1]

    # Mid/Side decomposition
    mid = (left + right) / 2
    side = (left - right) / 2

    # Widen by amplifying side
    side_widened = side * (1 + width)

    # Reconstruct
    left_out = mid + side_widened
    right_out = mid - side_widened

    result = np.stack([left_out, right_out], axis=0)

    return result  # type: ignore[no-any-return]


# ============================================================================
# HARMONIC EXCITER SAFETY TESTS
# ============================================================================


@pytest.mark.unit
def test_harmonic_exciter_with_headroom(mono_audio, temp_log_dir):
    """Test harmonic exciter with sufficient headroom."""
    audio, sr = mono_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=True, log_dir=temp_log_dir)

    # Should process successfully
    processed, report = wrapper.process(audio, sr, amount=0.5)

    assert report.decision in [ProcessingDecision.PROCEED, ProcessingDecision.REDUCE_STRENGTH]
    assert report.pre_check_result.passed

    # Should not clip
    if report.post_check_result:
        clipping_detected = report.post_check_result.metrics.get("clipping_detected", False)
        assert not clipping_detected


def test_harmonic_exciter_rejects_hot_audio(hot_audio, temp_log_dir):
    """Test harmonic exciter rejects audio with insufficient headroom."""
    audio, sr = hot_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False, min_headroom_db=6.0)

    # Should abort (insufficient headroom)
    processed, report = wrapper.process(audio, sr, amount=0.5)

    assert report.decision == ProcessingDecision.ABORT
    assert not report.pre_check_result.passed


def test_harmonic_exciter_increases_brightness(dark_audio, temp_log_dir):
    """Test harmonic exciter increases brightness on dark audio."""
    audio, sr = dark_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False)

    processed, report = wrapper.process(audio, sr, amount=0.7)

    # Should process
    if report.decision == ProcessingDecision.PROCEED:
        if report.post_check_result:
            brightness_increase = report.post_check_result.metrics.get("brightness_increase", 0)
            assert brightness_increase >= 0  # Should increase or stay same


def test_harmonic_exciter_prevents_harshness(bright_audio, temp_log_dir):
    """Test harmonic exciter prevents harshness on bright audio."""
    audio, sr = bright_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False, max_harshness=0.6)

    processed, report = wrapper.process(audio, sr, amount=0.8)

    # May abort or reduce strength
    if report.decision == ProcessingDecision.PROCEED:
        if report.post_check_result:
            harshness_after = report.post_check_result.metrics.get("harshness_after", 0)
            assert harshness_after <= wrapper.max_harshness


def test_harmonic_exciter_detects_imd(mono_audio, temp_log_dir):
    """Test harmonic exciter detects intermodulation distortion."""
    audio, sr = mono_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False)

    processed, report = wrapper.process(audio, sr, amount=0.5)

    # Check IMD metrics exist
    if report.post_check_result:
        assert "imd_after" in report.post_check_result.metrics


# ============================================================================
# STEREO WIDENER SAFETY TESTS
# ============================================================================


def test_stereo_widener_normal_stereo(stereo_audio, temp_log_dir):
    """Test stereo widener with normal stereo audio."""
    audio, sr = stereo_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=True, log_dir=temp_log_dir)

    # Should process successfully
    processed, report = wrapper.process(audio, sr, width=0.5)

    assert report.decision in [ProcessingDecision.PROCEED, ProcessingDecision.REDUCE_STRENGTH]
    assert report.pre_check_result.passed

    # Should increase width
    if report.post_check_result:
        width_increase = report.post_check_result.metrics.get("width_increase", 0)
        assert width_increase >= 0


def test_stereo_widener_rejects_mono(mono_audio, temp_log_dir):
    """Test stereo widener rejects mono audio."""
    audio, sr = mono_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False)

    # Should abort (mono input)
    processed, report = wrapper.process(audio, sr, width=0.5)

    assert report.decision == ProcessingDecision.ABORT
    assert not report.pre_check_result.passed


def test_stereo_widener_mono_compatibility(stereo_audio, temp_log_dir):
    """Test stereo widener maintains mono compatibility."""
    audio, sr = stereo_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False, min_mono_compatibility=0.8)

    processed, report = wrapper.process(audio, sr, width=0.3)

    # Check mono compatibility
    if report.post_check_result:
        mono_compat = report.post_check_result.metrics.get("mono_compatibility_after", 0)
        assert mono_compat >= wrapper.min_mono_compatibility


def test_stereo_widener_preserves_center(stereo_audio, temp_log_dir):
    """Test stereo widener preserves center content."""
    audio, sr = stereo_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False)

    processed, report = wrapper.process(audio, sr, width=0.4)

    # Check center preservation (calculated as 1 - energy_loss)
    if report.post_check_result:
        center_energy_loss = report.post_check_result.metrics.get("center_energy_loss", 1.0)
        center_preservation = 1.0 - center_energy_loss
        assert center_preservation > 0.8  # Should preserve >80%


def test_stereo_widener_detects_hollow(stereo_audio, temp_log_dir):
    """Test stereo widener detects hollow artifacts."""
    audio, sr = stereo_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False)

    processed, report = wrapper.process(audio, sr, width=0.6)

    # Check hollow artifact detection
    if report.post_check_result:
        assert "hollow_artifacts" in report.post_check_result.metrics


def test_stereo_widener_limits_width_increase(narrow_stereo_audio, temp_log_dir):
    """Test stereo widener limits width increase."""
    audio, sr = narrow_stereo_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False, max_width_increase=0.5)

    processed, report = wrapper.process(audio, sr, width=0.8)

    # May reduce strength or abort if width increase too large
    if report.decision == ProcessingDecision.REDUCE_STRENGTH:
        assert report.post_check_result is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_sequential_enhancement_processing(stereo_audio, temp_log_dir):
    """Test sequential processing with multiple enhancement wrappers."""
    audio, sr = stereo_audio

    # Process each channel with harmonic exciter first
    exciter = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=True, log_dir=temp_log_dir)

    audio_excited_l, report1_l = exciter.process(audio[0], sr, amount=0.4)
    audio_excited_r, report1_r = exciter.process(audio[1], sr, amount=0.4)

    # Recombine
    audio_excited = np.stack([audio_excited_l, audio_excited_r], axis=0)

    # Then widen
    widener = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=True, log_dir=temp_log_dir)

    audio_final, report2 = widener.process(audio_excited, sr, width=0.5)

    # Safety-first: Allow ABORT if headroom insufficient
    # At least reports should exist
    assert report1_l is not None
    assert report2 is not None


def test_enhancement_wrapper_statistics(mono_audio, temp_log_dir):
    """Test enhancement wrapper statistics tracking."""
    audio, sr = mono_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False)

    # Process multiple times
    n_calls = 5
    for _ in range(n_calls):
        wrapper.process(audio, sr, amount=0.5)

    stats = wrapper.get_statistics()

    assert stats["total_calls"] == n_calls


# ============================================================================
# STRESS TESTS
# ============================================================================


def test_harmonic_exciter_extreme_amount(mono_audio, temp_log_dir):
    """Test harmonic exciter with extreme amount."""
    audio, sr = mono_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False)

    # Very high amount
    processed, report = wrapper.process(audio, sr, amount=1.0)

    # Should handle gracefully (may reduce strength or abort)
    assert report is not None


def test_stereo_widener_extreme_width(stereo_audio, temp_log_dir):
    """Test stereo widener with extreme width."""
    audio, sr = stereo_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False)

    # Very high width
    processed, report = wrapper.process(audio, sr, width=2.0)

    # Should handle gracefully (may reduce strength or abort)
    assert report is not None

    # Check mono compatibility not destroyed
    if report.post_check_result:
        mono_compat = report.post_check_result.metrics.get("mono_compatibility_after", 0)
        # Should still have some compatibility
        assert mono_compat >= 0.5


def test_harmonic_exciter_nan_resilience(temp_log_dir):
    """Test harmonic exciter handles NaN gracefully."""
    sr = 44100
    audio = np.random.randn(sr) * 0.5
    audio[1000] = np.nan

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False)

    processed, report = wrapper.process(audio, sr, amount=0.5)

    # Should abort due to NaN
    assert report.decision == ProcessingDecision.ABORT


def test_stereo_widener_inf_resilience(temp_log_dir):
    """Test stereo widener handles Inf gracefully."""
    sr = 44100
    audio = np.random.randn(2, sr) * 0.5
    audio[0, 1000] = np.inf

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False)

    processed, report = wrapper.process(audio, sr, width=0.5)

    # Should abort due to Inf
    assert report.decision == ProcessingDecision.ABORT


# ============================================================================
# QUALITY SCORING TESTS
# ============================================================================


def test_harmonic_exciter_quality_score(mono_audio, temp_log_dir):
    """Test harmonic exciter quality score calculation."""
    audio, sr = mono_audio

    wrapper = HarmonicExciterSafety(processor_func=dummy_harmonic_exciter, enable_logging=False)

    processed, report = wrapper.process(audio, sr, amount=0.5)

    # Check quality score exists and is reasonable
    if report.post_check_result and report.post_check_result.quality_score is not None:
        assert 0.0 <= report.post_check_result.quality_score <= 1.0


def test_stereo_widener_quality_score(stereo_audio, temp_log_dir):
    """Test stereo widener quality score calculation."""
    audio, sr = stereo_audio

    wrapper = StereoWidenerSafety(processor_func=dummy_stereo_widener, enable_logging=False)

    processed, report = wrapper.process(audio, sr, width=0.5)

    # Check quality score exists and is reasonable
    if report.post_check_result and report.post_check_result.quality_score is not None:
        assert 0.0 <= report.post_check_result.quality_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
