#!/usr/bin/env python3
"""
Mini-Test für R11 P1/P2-Regression.
Nutzt nur 10s real audio ür schnelle Diagnose
"""

import sys

sys.path.insert(0, ".")

if __name__ != "__main__":
    import pytest

    pytest.skip("Mini-R11-Skript ist nur fuer manuelle Ausfuehrung gedacht.", allow_module_level=True)

from pathlib import Path

import numpy as np

# Find a test audio file
test_audio_dir = Path("test_audio")
if not test_audio_dir.exists():
    print("test_audio dir not found")
    sys.exit(1)

audio_files = list(test_audio_dir.glob("*.wav")) + list(test_audio_dir.glob("*.mp3"))
if not audio_files:
    print(f"No audio files found in {test_audio_dir}")
    sys.exit(1)

test_file = sorted(audio_files)[0]
print(f"Using test audio: {test_file}")

# Load audio (only first 10 seconds for speed)
import librosa

audio, sr = librosa.load(test_file, sr=48000, mono=False, duration=10)
if audio.ndim == 1:
    audio = audio.reshape(-1, 1)
elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
    audio = audio.T

print(f"Audio loaded: shape={audio.shape}, sr={sr}")

from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
from backend.core.performance_guard import QualityMode

# Run restoration
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

original = audio.copy()
original_mono = np.mean(original, axis=1) if original.ndim == 2 else original

# Fast restoration
cfg = RestorationConfig(
    mode=QualityMode.FAST,
    enable_performance_guard=False,
    enable_phase_gate=False,
    enable_phase_skipping=False,
)
restorer = UnifiedRestorerV3(config=cfg)
result = restorer.restore(audio, sample_rate=sr, mode="fast", ml_runtime_budget_s=3.0)  # type: ignore[arg-type]
restored = np.asarray(result.audio)
if restored.ndim == 2 and restored.shape[0] < restored.shape[1]:
    restored = restored.T

restored_mono = np.mean(restored, axis=1) if restored.ndim == 2 else restored

# Measure goals
checker = MusicalGoalsChecker(mode="restoration")
goals_before = checker.measure_all(original_mono, sr)  # type: ignore[arg-type]
goals_after = checker.measure_all(restored_mono, sr, reference=original_mono)  # type: ignore[arg-type]

# P1/P2 regression check (same as R11 test)
p1p2 = ["natuerlichkeit", "authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation"]
goal_floors = {
    "natuerlichkeit": 0.72,
    "authentizitaet": 0.72,
    "tonal_center": 0.50,
    "timbre_authentizitaet": 0.72,
    "artikulation": 0.72,
}

print("\n=== P1/P2 REGRESSION TEST (R11) ===")
failures = []
for goal in p1p2:
    before = float(goals_before.get(goal, 0.0))
    after = float(goals_after.get(goal, 0.0))
    floor = float(goal_floors.get(goal, 0.70))

    threshold = max(floor, before - 0.30)
    passed = after >= threshold

    status = "PASS" if passed else "FAIL"
    print(f"{goal:25s}: {before:6.4f} → {after:6.4f}  (floor={floor:.2f}, threshold={threshold:.4f}) [{status}]")

    if not passed:
        failures.append((goal, before, after, threshold))

if failures:
    print(f"\n!!! {len(failures)} GOAL(S) FAILED !!!")
    for goal, before, after, threshold in failures:
        print(f"  - {goal}: {before:.4f} → {after:.4f}, need ≥ {threshold:.4f}")
    sys.exit(1)
else:
    print("\n✅ All P1/P2 goals pass R11 regression check!")
    sys.exit(0)
