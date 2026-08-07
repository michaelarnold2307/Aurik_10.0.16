import pytest

#!/usr/bin/env python3
"""
Phase 31 ML-Hybrid Integration Test
====================================

Tests the ML-Hybrid Speed/Pitch Correction with YIN + CREPE.

Test Scenarios:
1. FAST mode: YIN DSP only
2. BALANCED mode: Adaptive (YIN → CREPE if confidence <0.7)
3. BALANCED mode: Adaptive (YIN + optional CREPE)

Expected Behavior:
- FAST: Pure DSP, ~0.5× RT
- BALANCED: Adaptive ML, ~1.0× RT (skip CREPE if YIN confident)
- BALANCED: Adaptive ML, ~1.5-2× RT (YIN + optional CREPE)
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase


def create_test_audio_with_speed_error(
    duration: float = 3.0, sr: int = 48000, speed_error_percent: float = 3.0
) -> np.ndarray:
    """
    Create test audio with simulated speed error.

    Args:
        duration: Duration in seconds
        sr: Sample rate
        speed_error_percent: Speed error percentage (e.g., 3.0 for 3%)

    Returns:
        Audio with speed error
    """
    t = np.linspace(0, duration, int(duration * sr))

    # Reference: A440 Hz
    reference_pitch = 440.0

    # Simulate speed error (too fast playback)
    # E.g., 3% too fast → pitch is 440 * 1.03 = 453.2 Hz
    speed_ratio = 1.0 + (speed_error_percent / 100.0)
    played_pitch = reference_pitch * speed_ratio

    # Create audio with wrong pitch
    audio = 0.4 * np.sin(2 * np.pi * played_pitch * t)
    audio += 0.2 * np.sin(2 * np.pi * played_pitch * 2 * t)  # 2nd harmonic
    audio += 0.1 * np.sin(2 * np.pi * played_pitch * 3 * t)  # 3rd harmonic

    # Add envelope (attack/decay)
    envelope = np.ones_like(audio)
    attack = int(0.1 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    release = int(0.2 * sr)
    envelope[-release:] = np.linspace(1, 0, release)
    audio *= envelope

    return audio  # type: ignore[no-any-return]


@pytest.mark.unit
def test_phase_31_mode(phase, audio, sr, mode: str):
    """Test Phase 31 with specific quality mode."""
    print(f"\n{'=' * 60}")
    print(f"Test: {mode.upper()} Mode")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        result = phase.process(
            audio=audio,
            sample_rate=sr,
            material_type="vinyl",  # Vinyl has 5% max error, correction strength 0.8
            reference_pitch=440.0,
            quality_mode=mode,
        )

        elapsed = time.time() - start_time

        # Extract metadata
        metadata = result.metadata if result else {}
        modifications = result.modifications if result else {}

        # Check processing
        processing = modifications.get("processing", "unknown")
        detected_pitch = modifications.get("detected_pitch", metadata.get("detected_pitch", 0.0))
        confidence = modifications.get("confidence", metadata.get("confidence", 0.0))

        # ML-Hybrid specific
        strategy = metadata.get("strategy", "unknown")
        yin_applied = metadata.get("yin_applied", False)
        crepe_applied = metadata.get("crepe_applied", False)
        algorithm_version = metadata.get("algorithm_version", "unknown")
        quality_mode = metadata.get("quality_mode", mode)

        print(f"✅ Processing: {processing}")
        print(f"   Algorithm Version: {algorithm_version}")
        print(f"   Quality Mode: {quality_mode}")
        print(f"   Strategy: {strategy}")
        print(f"   YIN Applied: {yin_applied}")
        print(f"   CREPE Applied: {crepe_applied}")
        print(f"   Detected Pitch: {detected_pitch:.2f} Hz")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Time: {elapsed:.2f}s ({elapsed * sr / len(audio):.2f}× RT)")

        # Check if speed error was detected
        if "speed_error_percent" in modifications:
            speed_error = modifications["speed_error_percent"]
            print(f"   Speed Error: {speed_error:.2f}%")

        # Validation
        if mode == "fast":
            assert strategy in ["yin_only", "yin_fallback"], f"FAST mode should use YIN only, got {strategy}"
            assert not crepe_applied, "FAST mode should not apply CREPE"
        elif mode == "balanced":
            assert strategy in [
                "adaptive",
                "yin_fallback",
            ], f"BALANCED mode should use adaptive strategy, got {strategy}"
            # CREPE may or may not be applied depending on YIN confidence
        elif mode == "balanced":
            assert strategy in [
                "adaptive",
                "yin_fallback",
            ], f"BALANCED mode should use adaptive strategy, got {strategy}"
            assert yin_applied, "BALANCED mode should apply YIN"

        print(f"✅ {mode.upper()} Mode Test: PASSED")
        return True

    except Exception as e:
        print(f"❌ {mode.upper()} Mode Test: FAILED")
        print(f"   Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run Phase 31 ML-Hybrid integration tests."""
    print("\n" + "=" * 80)
    print("PHASE 31: SPEED/PITCH CORRECTION ML-HYBRID INTEGRATION TEST")
    print("=" * 80)

    # Create test audio with 3% speed error
    print("\n📊 Creating test audio...")
    print("   Simulating 3% speed error (440 Hz → 453.2 Hz)")
    sr = 48000
    audio = create_test_audio_with_speed_error(duration=3.0, sr=sr, speed_error_percent=3.0)
    print(f"✅ Created {len(audio) / sr:.1f}s test audio @ {sr} Hz")

    # Initialize phase
    phase = SpeedPitchCorrectionPhase()

    # Test all modes
    results = {}
    for mode in ["fast", "balanced"]:
        results[mode] = test_phase_31_mode(phase, audio, sr, mode)

    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for mode, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {mode.upper():15} {status}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Phase 31 ML-Hybrid Integration Complete!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
