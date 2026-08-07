#!/usr/bin/env python3
"""
Phase 2.3 Real-World Validation - Minimal Version
Tests all 6 Phase 2.3 components on real audio files
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsp.bass_enhancement import BassEnhancementSystem
from dsp.brass_enhancement import BrassEnhancementSystem
from dsp.drums_enhancement import DrumsEnhancementSystem
from dsp.guitar_enhancement import GuitarEnhancementSystem
from dsp.piano_restoration import PianoRestorationSystem
from dsp.spatial_enhancement import SpatialEnhancementSystem

print("🎵 Phase 2.3 Real-World Validation")
print("=" * 60)

# Find audio files
audio_dirs = [Path("input"), Path("test_audio"), Path("audio_examples")]
audio_files: list[Any] = []
for d in audio_dirs:
    if d.exists():
        audio_files.extend(d.glob("*.wav"))
        audio_files.extend(d.glob("*.mp3"))

print(f"Found {len(audio_files)} audio files\n")

# Initialize all systems once
print("Initializing Phase 2.3 systems...")
systems = {
    "Bass": BassEnhancementSystem(),
    "Drums": DrumsEnhancementSystem(),
    "Guitar": GuitarEnhancementSystem(),
    "Piano": PianoRestorationSystem(),
    "Brass": BrassEnhancementSystem(),
    "Spatial": SpatialEnhancementSystem(),
}
print(f"✓ {len(systems)} systems ready\n")

# Test each file
results = []
for i, file in enumerate(audio_files[:5], 1):  # Test first 5 files
    print(f"[{i}/{min(5, len(audio_files))}] {file.name}")

    try:
        # Load
        audio, sr = sf.read(file)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)
        print(f"  ✓ Loaded: {audio.shape[0]} samples @ {sr} Hz")

        # Test each system
        for name, system in systems.items():
            try:
                processed, report = system.process(audio, sr)  # type: ignore[attr-defined]
                rms_orig = np.sqrt(np.mean(audio**2))
                rms_proc = np.sqrt(np.mean(processed**2))
                gain_db = 20 * np.log10(rms_proc / (rms_orig + 1e-10))
                print(f"  ✓ {name:8s}: {gain_db:+.2f} dB")
                results.append({"file": file.name, "system": name, "gain_db": gain_db, "success": True})
            except Exception as e:
                print(f"  ✗ {name:8s}: {str(e)[:40]}")
                results.append({"file": file.name, "system": name, "error": str(e)[:40], "success": False})
        print()

    except Exception as e:
        print(f"  ✗ Failed to load: {str(e)[:40]}\n")

# Summary
print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
total = len(results)
success = sum(1 for r in results if r.get("success", False))
print(f"Total tests: {total}")
print(f"Successful:  {success} ({success / total * 100:.1f}%)")
print(f"Failed:      {total - success} ({(total - success) / total * 100:.1f}%)")

# Per-system stats
print("\nPer-System Stats:")
for system_name in systems:
    sys_results = [r for r in results if r["system"] == system_name]
    sys_success = sum(1 for r in sys_results if r.get("success", False))
    sys_total = len(sys_results)
    if sys_total > 0:
        print(f"  {system_name:8s}: {sys_success}/{sys_total} ({sys_success / sys_total * 100:.0f}%)")

print("\n✅ Validation complete!")
