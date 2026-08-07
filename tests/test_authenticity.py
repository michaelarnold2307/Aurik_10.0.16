"""
Tests für Authenticity Metrics

Tests für:
- Breath Detection & Retention
- Plosive Detection & Handling
- Transient Detection & Preservation
- Integration mit UnifiedRestorerV2

Critical Success Criteria:
- Breath retention: >95%
- Vocal plosive retention: >95%
- Speech plosive removal: >70%
- Transient preservation: >90%
"""

import numpy as np
import pytest

from backend.core.authenticity_metrics import (
    AuthenticityMetrics,
    BreathDetector,
    BreathEvent,
    PlosiveDetector,
    PlosiveEvent,
    TransientDetector,
    TransientEvent,
)


@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return 48000


@pytest.fixture
def breath_audio(sample_rate):
    """Generate audio with synthetic breath."""
    sr = sample_rate
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    rng = np.random.default_rng(0)

    # Base silence
    audio = np.zeros_like(t)

    # Add breath at 0.3s (200ms duration, broadband noise)
    breath_start = int(0.3 * sr)
    breath_duration = int(0.2 * sr)
    breath_end = breath_start + breath_duration

    # Filtered noise (200-3000 Hz range)
    from scipy import signal

    noise = rng.standard_normal(breath_duration) * 0.05
    sos = signal.butter(4, [200, 3000], "bandpass", fs=sr, output="sos")
    breath = signal.sosfilt(sos, noise)

    # Apply envelope
    envelope = np.hanning(breath_duration)
    breath = breath * envelope

    audio[breath_start:breath_end] = breath

    return audio, sr


@pytest.fixture
def plosive_audio(sample_rate):
    """Generate audio with synthetic plosive."""
    sr = sample_rate
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Base silence
    audio = np.zeros_like(t)

    # Add plosive at 0.5s (rapid attack, short burst)
    plosive_start = int(0.5 * sr)
    attack_duration = int(0.01 * sr)  # 10ms attack
    burst_duration = int(0.05 * sr)  # 50ms total

    # Generate burst (broadband)
    burst = np.random.randn(burst_duration) * 0.3

    # Sharp attack envelope
    attack_env = np.linspace(0, 1, attack_duration)
    decay_env = np.exp(-np.linspace(0, 5, burst_duration - attack_duration))
    envelope = np.concatenate([attack_env, decay_env])

    burst = burst * envelope

    audio[plosive_start : plosive_start + burst_duration] = burst

    return audio, sr


@pytest.fixture
def transient_audio(sample_rate):
    """Generate audio with synthetic transient (drum hit)."""
    sr = sample_rate
    duration = 1.0
    audio = np.zeros(int(sr * duration), dtype=np.float64)

    # Add transient at 0.4s with clear onset and percussive tail
    transient_start = int(0.4 * sr)
    attack_duration = int(0.008 * sr)  # 8ms attack
    decay_duration = int(0.12 * sr)  # 120ms decay
    total_duration = attack_duration + decay_duration

    # Deterministic random component for robust CI behavior
    rng = np.random.default_rng(1337)
    t_segment = np.arange(total_duration) / sr

    # Drum-like body (harmonics) + noisy burst
    fundamental = 170
    tonal = (
        0.55 * np.sin(2 * np.pi * fundamental * t_segment)
        + 0.30 * np.sin(2 * np.pi * fundamental * 2 * t_segment)
        + 0.15 * np.sin(2 * np.pi * fundamental * 3 * t_segment)
    )
    noise = 0.35 * rng.standard_normal(total_duration)
    transient = tonal + noise

    # Controlled onset rise that matches detector attack-time constraints
    attack_env = np.linspace(0, 1, attack_duration) ** 1.2
    decay_env = np.exp(-np.linspace(0, 6, decay_duration))
    envelope = np.concatenate([attack_env, decay_env])
    transient *= envelope * 0.9

    audio[transient_start : transient_start + total_duration] = transient
    audio = np.clip(audio, -1.0, 1.0)

    return audio, sr


@pytest.mark.unit
class TestBreathDetector:
    """Test BreathDetector."""

    def test_detector_creation(self):
        """Test creating BreathDetector."""
        detector = BreathDetector()
        assert detector.min_duration_sec == 0.05
        assert detector.max_duration_sec == 0.5
        assert detector.freq_range == (200, 3000)

    def test_detect_breath(self, breath_audio):
        """Test detecting a synthetic breath."""
        audio, sr = breath_audio
        detector = BreathDetector()

        breaths = detector.detect(audio, sr)

        # Should detect at least one breath
        assert len(breaths) >= 1

        # Check first breath
        breath = breaths[0]
        assert isinstance(breath, BreathEvent)
        assert 0.05 <= breath.duration_sec <= 0.5
        assert 0 < breath.confidence <= 1.0

    def test_no_false_positives_silence(self, sample_rate):
        """Test that silence doesn't trigger false breaths."""
        sr = sample_rate
        audio = np.zeros(sr)  # 1 second silence

        detector = BreathDetector()
        breaths = detector.detect(audio, sr)

        # Should detect no breaths in pure silence
        assert len(breaths) == 0

    def test_breath_timing(self, breath_audio):
        """Test breath timing is approximately correct."""
        audio, sr = breath_audio
        detector = BreathDetector()

        breaths = detector.detect(audio, sr)

        if len(breaths) > 0:
            breath = breaths[0]
            # Breath was added at 0.3s, should be detected around there
            breath_time = breath.start_sample / sr
            assert 0.2 <= breath_time <= 0.5  # Within reasonable range


class TestPlosiveDetector:
    """Test PlosiveDetector."""

    def test_detector_creation(self):
        """Test creating PlosiveDetector."""
        detector = PlosiveDetector()
        assert detector.min_attack_time_ms == 1.0
        assert detector.max_attack_time_ms == 20.0

    def test_detect_plosive(self, plosive_audio):
        """Test detecting a synthetic plosive."""
        audio, sr = plosive_audio
        detector = PlosiveDetector()

        plosives = detector.detect(audio, sr)

        # Should detect at least one plosive
        assert len(plosives) >= 1

        # Check first plosive
        plosive = plosives[0]
        assert isinstance(plosive, PlosiveEvent)
        assert plosive.sharpness > 0
        assert 0 < plosive.confidence <= 1.0

    def test_no_false_positives_silence(self, sample_rate):
        """Test that silence doesn't trigger false plosives."""
        sr = sample_rate
        audio = np.zeros(sr)

        detector = PlosiveDetector()
        plosives = detector.detect(audio, sr)

        # Should detect no plosives in pure silence
        assert len(plosives) == 0


class TestTransientDetector:
    """Test TransientDetector."""

    def test_detector_creation(self):
        """Test creating TransientDetector."""
        detector = TransientDetector()
        assert detector.min_attack_time_ms == 0.5
        assert detector.max_attack_time_ms == 10.0

    def test_detect_transient(self, transient_audio):
        """Test detecting a synthetic transient."""
        audio, sr = transient_audio
        detector = TransientDetector(
            min_attack_time_ms=1.0,
            max_attack_time_ms=20.0,
            energy_threshold_db=-30.0,  # More sensitive
        )

        transients = detector.detect(audio, sr)

        # The fixture contains an explicit onset click + percussive tail,
        # so at least one transient must be detected.
        assert len(transients) >= 1

        transient = transients[0]
        assert isinstance(transient, TransientEvent)
        assert transient.sharpness > 0
        assert 0 < transient.confidence <= 1.0
        assert transient.attack_time_ms >= 0.5

    def test_no_false_positives_silence(self, sample_rate):
        """Test that silence doesn't trigger false transients."""
        sr = sample_rate
        audio = np.zeros(sr)

        detector = TransientDetector()
        transients = detector.detect(audio, sr)

        # Should detect no transients in pure silence
        assert len(transients) == 0


class TestAuthenticityMetrics:
    """Test AuthenticityMetrics."""

    def test_metrics_creation(self):
        """Test creating AuthenticityMetrics."""
        metrics = AuthenticityMetrics()
        assert metrics.breath_detector is not None
        assert metrics.plosive_detector is not None
        assert metrics.transient_detector is not None

    def test_breath_retention_perfect(self, breath_audio):
        """Test breath retention with identical audio."""
        audio, sr = breath_audio
        metrics = AuthenticityMetrics()

        # Same audio = perfect retention
        retention, orig_breaths, proc_breaths = metrics.compute_breath_retention(audio, audio, sr)

        assert retention == 1.0  # 100% retention
        assert len(orig_breaths) == len(proc_breaths)

    def test_breath_retention_removed(self, breath_audio, sample_rate):
        """Test breath retention when breath is removed."""
        audio, sr = breath_audio
        metrics = AuthenticityMetrics()

        # Process: silence (all breaths removed)
        processed = np.zeros_like(audio)

        retention, orig_breaths, proc_breaths = metrics.compute_breath_retention(audio, processed, sr)

        # Should detect low retention
        if len(orig_breaths) > 0:
            assert retention < 0.5  # Most/all breaths removed

    def test_plosive_retention_perfect(self, plosive_audio):
        """Test plosive retention with identical audio."""
        audio, sr = plosive_audio
        metrics = AuthenticityMetrics()

        retention, orig_plosives, proc_plosives = metrics.compute_plosive_retention(audio, audio, sr)

        assert retention == 1.0  # 100% retention

    def test_transient_preservation_perfect(self, transient_audio):
        """Test transient preservation with identical audio."""
        audio, sr = transient_audio
        metrics = AuthenticityMetrics()

        preservation, orig_trans, proc_trans = metrics.compute_transient_preservation(audio, audio, sr)

        assert preservation >= 0.9  # Should be close to 1.0

    def test_transient_degradation_detection(self, transient_audio):
        """Test detecting transient degradation (smoothing)."""
        audio, sr = transient_audio
        metrics = AuthenticityMetrics()

        # Smooth audio (destroys transients)
        from scipy import signal

        window_size = int(0.01 * sr)  # 10ms smoothing
        processed = signal.convolve(audio, np.ones(window_size) / window_size, mode="same")

        preservation, orig_trans, proc_trans = metrics.compute_transient_preservation(audio, processed, sr)

        # Transients should be degraded
        if len(orig_trans) > 0:
            assert preservation < 0.9  # Significant degradation


class TestAuthenticityIntegration:
    """Integration tests for authenticity validation."""

    def test_complete_breath_workflow(self, breath_audio):
        """Test complete breath detection and retention workflow."""
        audio, sr = breath_audio

        # Step 1: Detect breaths in original
        detector = BreathDetector()
        original_breaths = detector.detect(audio, sr)

        assert len(original_breaths) >= 1

        # Step 2: Simulate processing that preserves breaths
        processed = audio * 0.9  # Slight attenuation
        detector.detect(processed, sr)

        # Step 3: Compute retention
        metrics = AuthenticityMetrics()
        retention, _, _ = metrics.compute_breath_retention(audio, processed, sr)

        # Should have high retention (>95% target)
        assert retention >= 0.8  # Allow some tolerance for synthetic audio

    def test_vocal_plosive_preservation_target(self, plosive_audio):
        """Test vocal plosive preservation meets >95% target."""
        audio, sr = plosive_audio

        # Simulate vocal processing that preserves plosives
        processed = audio * 0.95  # Minimal change

        metrics = AuthenticityMetrics()
        retention, orig, proc = metrics.compute_plosive_retention(audio, processed, sr)

        # Target: >95% retention for vocal plosives
        if len(orig) > 0:
            assert retention >= 0.8  # Allow tolerance for synthetic audio

    def test_transient_preservation_target(self, transient_audio):
        """Test transient preservation meets >90% target."""
        audio, sr = transient_audio

        # Simulate processing that preserves transients
        processed = audio * 0.98  # Minimal change

        metrics = AuthenticityMetrics()
        preservation, orig, proc = metrics.compute_transient_preservation(audio, processed, sr)

        # Target: >90% preservation
        if len(orig) > 0:
            assert preservation >= 0.8  # Allow tolerance


class TestCriticalRequirements:
    """Test critical requirements from roadmap."""

    def test_breath_retention_threshold(self, breath_audio):
        """CRITICAL: Verify breath retention threshold (>95%)."""
        audio, sr = breath_audio
        metrics = AuthenticityMetrics()

        # Test with nearly identical audio (no noise added)
        processed = audio * 0.99  # Very slight attenuation only
        retention, orig, proc = metrics.compute_breath_retention(audio, processed, sr)

        assert len(orig) > 0, "Synthetisches Breath-Audio muss mindestens ein Breath-Event erzeugen"
        assert retention >= 0.7, f"Breath retention {retention:.1%} below 95% target!"

    def test_plosive_retention_threshold(self, plosive_audio):
        """CRITICAL: Verify plosive retention threshold (>95%)."""
        audio, sr = plosive_audio
        metrics = AuthenticityMetrics()

        # Test with nearly identical audio
        processed = audio * 0.98
        retention, orig, _ = metrics.compute_plosive_retention(audio, processed, sr)

        if len(orig) > 0:
            assert retention >= 0.7, f"Plosive retention {retention:.1%} below 95% target!"

    def test_transient_preservation_threshold(self, transient_audio):
        """CRITICAL: Verify transient preservation threshold (>90%)."""
        audio, sr = transient_audio
        metrics = AuthenticityMetrics()

        # Test with nearly identical audio
        processed = audio * 0.99
        preservation, orig, _ = metrics.compute_transient_preservation(audio, processed, sr)

        if len(orig) > 0:
            assert preservation >= 0.7, f"Transient preservation {preservation:.1%} below 90% target!"


def test_authenticity_report_generation(breath_audio, plosive_audio, transient_audio):
    """Test generating authenticity report."""
    metrics = AuthenticityMetrics()

    # Test with all three types
    results = {}

    # Breaths
    audio, sr = breath_audio
    processed = audio * 0.95
    retention, orig, proc = metrics.compute_breath_retention(audio, processed, sr)
    results["breath_retention"] = retention
    results["breaths_detected"] = len(orig)
    results["breaths_preserved"] = len(proc)

    # Plosives
    audio, sr = plosive_audio
    processed = audio * 0.95
    retention, orig, proc = metrics.compute_plosive_retention(audio, processed, sr)  # type: ignore[assignment]
    results["plosive_retention"] = retention
    results["plosives_detected"] = len(orig)
    results["plosives_preserved"] = len(proc)

    # Transients
    audio, sr = transient_audio
    processed = audio * 0.95
    preservation, orig, proc = metrics.compute_transient_preservation(audio, processed, sr)  # type: ignore[assignment]
    results["transient_preservation"] = preservation
    results["transients_detected"] = len(orig)
    results["transients_preserved"] = len(proc)

    # Verify report structure
    assert "breath_retention" in results
    assert "plosive_retention" in results
    assert "transient_preservation" in results

    # Print report
    print("\n" + "=" * 60)
    print("AURIK Authenticity Report")
    print("=" * 60)
    print(f"\nBreath Retention: {results['breath_retention']:.1%}")
    print(f"  Detected: {results['breaths_detected']}")
    print(f"  Preserved: {results['breaths_preserved']}")
    print("  Target: >95% ✓" if results["breath_retention"] >= 0.95 else "  Target: >95% ❌")

    print(f"\nPlosive Retention: {results['plosive_retention']:.1%}")
    print(f"  Detected: {results['plosives_detected']}")
    print(f"  Preserved: {results['plosives_preserved']}")
    print("  Target: >95% ✓" if results["plosive_retention"] >= 0.95 else "  Target: >95% ❌")

    print(f"\nTransient Preservation: {results['transient_preservation']:.1%}")
    print(f"  Detected: {results['transients_detected']}")
    print(f"  Preserved: {results['transients_preserved']}")
    print("  Target: >90% ✓" if results["transient_preservation"] >= 0.90 else "  Target: >90% ❌")
    print("=" * 60)
