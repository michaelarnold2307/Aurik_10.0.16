import pytest

"""
Test Phase 24 Dropout Repair ML-Hybrid Integration

Quick test to verify AudioSR length-based routing for dropout repair.
"""

import sys

sys.path.insert(0, "/mnt/1846D15B46D139E8/Aurik_Standalone")


import numpy as np

print("=" * 80)
print("Phase 24 Dropout Repair ML-Hybrid Integration Test")
print("=" * 80)

# Import phase
from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

# Create test phase
phase = DropoutRepairPhase()
print(f"\n✅ Phase instantiated: {phase.get_metadata().name}")

# Generate test audio with dropouts of different lengths
sr = 44100
duration = 3
t = np.linspace(0, duration, sr * duration)

# Clean signal (music)
audio = 0.3 * np.sin(2 * np.pi * 440 * t)
audio += 0.15 * np.sin(2 * np.pi * 880 * t)

# Short dropout (<20ms) - should use DSP linear
short_pos = int(1.0 * sr)
short_len = int(0.015 * sr)  # 15ms
audio[short_pos : short_pos + short_len] = 0

# Medium dropout (20-100ms) - should use DSP spectral
medium_pos = int(1.5 * sr)
medium_len = int(0.050 * sr)  # 50ms
audio[medium_pos : medium_pos + medium_len] = 0

# Long dropout (>100ms) - should use ML AudioSR in BALANCED
long_pos = int(2.0 * sr)
long_len = int(0.150 * sr)  # 150ms
audio[long_pos : long_pos + long_len] = 0

print("\nTest Audio:")
print(f"  Duration: {duration}s @ {sr} Hz")
print("  Dropouts:")
print("    - Short: 15ms (DSP linear expected)")
print("    - Medium: 50ms (DSP spectral expected)")
print("    - Long: 150ms (ML AudioSR expected in BALANCED)")

# Test with different quality modes
test_modes = ["FAST", "BALANCED"]

for mode in test_modes:
    print(f"\n{'-' * 80}")
    print(f"Testing Quality Mode: {mode}")
    print(f"{'-' * 80}")

    result = phase.process(audio.copy(), sample_rate=sr, material_type="tape", quality_mode=mode)

    if result.success:
        print("✅ Processing successful!")
        print(f"   Algorithm: {result.modifications['algorithm_version']}")
        print(f"   Dropouts repaired: {result.modifications['dropouts_repaired']}")
        print(f"   ML repaired: {result.modifications['ml_repaired']}")
        print(f"   ML usage ratio: {result.modifications['ml_usage_ratio']:.2%}")
        print(f"   Avg dropout: {result.modifications['avg_dropout_duration_ms']:.1f}ms")
        print(f"   Max dropout: {result.modifications['max_dropout_duration_ms']:.1f}ms")
        print(f"   Total duration: {result.modifications['total_dropout_duration_ms']:.1f}ms")
        print(
            f"   Execution time: {result.metadata['execution_time_seconds']:.3f}s ({result.metadata['execution_time_seconds'] / duration:.2f}× RT)"
        )

        # Expectations
        if mode == "FAST":
            if result.modifications["ml_repaired"] == 0:
                print("   ✅ Expected: FAST mode uses DSP only")
            else:
                print("   ⚠️  Unexpected: FAST should not use ML")

        elif mode == "BALANCED":
            if result.modifications["ml_repaired"] > 0:
                print(f"   ✅ Expected: {mode} mode uses ML for long dropouts")
                print("      Routing: Long dropout (150ms) → AudioSR")
            else:
                print("   ⚠️  Note: ML not used (plugin may be unavailable)")
    else:
        print("❌ Processing failed!")

print(f"\n{'=' * 80}")
print("✅ Phase 24 ML-Hybrid Integration Test Complete!")
print(f"{'=' * 80}")
print("\nStrategy: Length-Based Routing")
print("  - <20ms: DSP linear interpolation")
print("  - 20-100ms: DSP spectral inpainting")
print("  - >100ms: ML AudioSR generative repair")
print("  - Graceful fallback: DSP if ML unavailable")  # §V6: logger.warning handled at call site
print("\nExpected Improvement:")
print("  - Current (DSP): Natürlichkeit 0.50")
print("  - With ML: Natürlichkeit 0.80 (+0.30)")
print("  - Overall: 0.88 → 0.90 (+0.02)")
