#!/usr/bin/env python3
"""
Test: Complete Phase Selection V3
Testet die erweiterte Phase Selection Logic mit allen 25 Phasen.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

import numpy as np

from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType
from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@pytest.mark.timeout(300)
def test_all_material_types():
    """Teste Phase Selection für alle Material-Typen"""
    v3 = UnifiedRestorerV3()
    scanner = DefectScanner()

    # Test-Audio: 30s Stereo mit mittleren Defekten
    test_audio = np.random.randn(44100 * 30, 2) * 0.1

    materials = [MaterialType.SHELLAC, MaterialType.VINYL, MaterialType.TAPE, MaterialType.CD_DIGITAL]

    print("=" * 80)
    print("TEST: Phase Selection Complete - All Material Types")
    print("=" * 80)
    print()

    for material in materials:
        print(f"\n{'=' * 80}")
        print(f"Material: {material.value.upper()}")
        print(f"{'=' * 80}")

        # Defect Scan
        t0 = time.perf_counter()
        defects = scanner.scan(test_audio, material)  # type: ignore[arg-type]
        scan_time = time.perf_counter() - t0

        # Phase Selection
        selected = v3._select_phases(defects)

        # Statistics
        missing = [pid for pid in selected if pid not in v3.phase_metadata]
        prof_phases = [pid for pid in selected if "v2" in pid or "professional" in pid]

        print(f"Scan Time: {scan_time:.3f}s")
        print(f"Detected: {material.value}")
        print(f"Selected Phases: {len(selected)}")
        print(f"Professional Level: {len(prof_phases)}/{len(selected)} ({len(prof_phases) * 100 // len(selected)}%)")
        print(f"All phases exist: {'✅ YES' if len(missing) == 0 else '❌ NO'}")

        if missing:
            print(f"\n❌ FEHLER: {len(missing)} fehlende Phasen:")
            for pid in missing:
                print(f"   - {pid}")
            pytest.fail(f"{len(missing)} fehlende Phasen: {missing}")

        print(f"\n{'Phase':<50} {'Priority':<10} {'Level'}")
        print("-" * 75)
        for i, phase_id in enumerate(selected, 1):
            prio = v3.phase_metadata[phase_id]["priority"]
            level = "PROFESSIONAL" if ("v2" in phase_id or "professional" in phase_id) else "Standard"
            print(f"{i:2d}. {phase_id:<47} {prio:2d}          {level}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Phase Selection Complete")
    print("=" * 80)


@pytest.mark.timeout(300)
def test_severe_defects():
    """Teste Phase Selection mit schweren Defekten"""
    print("\n\n" + "=" * 80)
    print("TEST: Severe Defects - Maximum Phase Count")
    print("=" * 80)

    v3 = UnifiedRestorerV3()
    scanner = DefectScanner()

    # Audio mit hohen Defekt-Werten
    test_audio = np.random.randn(44100 * 30, 2) * 0.5  # Höherer Noise-Level

    defects = scanner.scan(test_audio, MaterialType.SHELLAC)  # type: ignore[arg-type]

    # Erhöhe manuell alle Severity-Werte
    for defect_type in DefectType:
        if defect_type in defects.scores:
            defects.scores[defect_type].severity = min(0.9, defects.scores[defect_type].severity * 2.0)

    selected = v3._select_phases(defects)

    print(f"Material: {defects.material_type.value}")
    print(f"Selected Phases: {len(selected)}")
    print("Expected: ~20-25 phases for severe defects")
    print()

    print("Top Defects:")
    for score in defects.get_top_defects(5):
        print(f"  - {score.defect_type.value}: {score.severity:.2f}")

    if len(selected) < 10:
        print("\n⚠️ WARNING: Too few phases selected for severe defects!")
        pytest.fail(f"Too few phases selected: {len(selected)} < 10")

    print(f"\n✅ PASS: {len(selected)} phases selected for severe defects")


if __name__ == "__main__":
    success = True

    # Test 1: All Material Types
    success &= test_all_material_types()

    # Test 2: Severe Defects
    success &= test_severe_defects()

    if success:
        print("\n\n" + "🎉" * 40)
        print("ALL TESTS PASSED - Phase Selection is Production-Ready!")
        print("🎉" * 40)
        sys.exit(0)
    else:
        print("\n\n❌ TESTS FAILED")
        sys.exit(1)
