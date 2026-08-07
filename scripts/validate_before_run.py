#!/usr/bin/env python3
"""
§2.59 Comprehensive Pre-Pipeline Validation (2026-07-09)

Führt ALLE Checks aus, bevor der User einen Pipeline-Run startet.
Fängt jeden Fehler, der sonst erst nach Minuten im Run sichtbar würde.

Usage: python scripts/validate_before_run.py
Exit 0 = bereit, Exit 1 = Fehler — NICHT starten
"""

import os
import sys
import time

sys.path.insert(0, ".")

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def check(name: str, fn) -> bool:
    global CHECKS_PASSED, CHECKS_FAILED
    try:
        ok = fn()
        if ok:
            print(f"  ✅ {name}")
            CHECKS_PASSED += 1
            return True
        else:
            print(f"  ❌ {name}")
            CHECKS_FAILED += 1
            return False
    except Exception as e:
        print(f"  💥 {name}: {e}")
        CHECKS_FAILED += 1
        return False


def import_check(module: str, attr: str) -> bool:
    mod = __import__(module, fromlist=[attr])
    getattr(mod, attr)
    return True


def test_critical_imports() -> bool:
    modules = [
        ("backend.core.defect_scanner", "DefectType"),
        ("backend.core.defect_scanner", "MaterialType"),
        ("backend.core.unified_restorer_v3", "UnifiedRestorerV3"),
        ("backend.core.phase_pruner", "IntelligentPhasePruner"),
        ("backend.core.defect_manifest", "get_defect_manifest"),
        ("backend.core.defect_contract_validator", "run_contract_validation"),
        ("backend.core.safe_dict", "SafeDict"),
        ("backend.core.quality_mode", "QualityModeConfig"),
        ("backend.core.quality_mode", "validate_mode"),
        ("backend.core.surgical_defect_analyzer", "SurgicalDefectAnalyzer"),
        ("backend.core.surgical_repair", "SurgicalRepair"),
        ("backend.core.periodic_health", "get_health_collector"),
        ("backend.core.vocal_distortion_sentinel", "VocalDistortionSentinel"),
        ("backend.core.song_goal_importance", "estimate_goal_importance"),
        ("forensics.medium_detector", "MediumDetector"),
        ("denker.aurik_denker", "get_aurik_denker"),
        ("denker.defekt_denker", "get_defekt_denker"),
        ("denker.phase_interaction_denker", "get_phase_interaction_denker"),
    ]
    all_ok = True
    for mod, attr in modules:
        if not import_check(mod, attr):
            all_ok = False
    return all_ok


def test_phase_imports() -> bool:

    phase_dir = "backend/core/phases"
    phases = sorted(f for f in os.listdir(phase_dir) if f.startswith("phase_") and f.endswith(".py"))

    all_ok = True
    for phase_file in phases:
        phase_name = phase_file.replace(".py", "")
        try:
            __import__(f"backend.core.phases.{phase_name}", fromlist=[""])
        except Exception as e:
            print(f"    ❌ {phase_name}: {e}")
            all_ok = False
    return all_ok


def test_song_calibration_integrity() -> bool:
    from unittest.mock import MagicMock

    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = MagicMock(spec=UnifiedRestorerV3)
    uv3._restoration_context = {"source_fidelity_bandwidth_target_hz": 13006.0}

    for material in ["vinyl", "cassette", "reel_tape", "mp3_low", "cd_digital"]:
        profile = UnifiedRestorerV3._build_song_calibration_profile(  # type: ignore[misc]
            uv3,
            material_type=material,  # type: ignore[arg-type]
            mode="restoration",  # type: ignore[arg-type]
            restorability_score=63.5,
            input_snr_db=14.3,
            max_defect_severity=0.6,
            pipeline_confidence=0.75,
        )
        gs = profile.get("global_scalar", -1)
        if not (0.5 <= gs <= 1.5):
            print(f"    ❌ {material}: global_scalar={gs} out of bounds")
            return False
    return True


def test_defect_manifest_integrity() -> bool:
    from backend.core.defect_manifest import get_defect_manifest
    from backend.core.defect_scanner import DefectType

    dm = get_defect_manifest()
    all_defects = {e.value for e in DefectType}

    for dv in all_defects:
        entry = dm.get(dv)
        if entry is None:
            print(f"    ❌ DefectType '{dv}' not in DefectManifest")
            return False
        # Every phase referenced must be valid
        for phase_id in entry.phases:
            if not phase_id.startswith("phase_"):
                print(f"    ❌ {dv} → invalid phase: {phase_id}")
                return False
    return True


def test_contract_validator() -> bool:
    from backend.core.defect_contract_validator import run_contract_validation

    result = run_contract_validation()
    return result["ok"]  # type: ignore[no-any-return]


def main() -> int:
    print("=" * 55)
    print("Aurik Pre-Pipeline Validation")
    print("=" * 55)

    t0 = time.monotonic()

    print("\n1. Critical Imports (18 modules):")
    check("18/18 modules", test_critical_imports)

    print("\n2. Phase Imports (66 phases):")
    check("66/66 phases importable", test_phase_imports)

    print("\n3. SongCalibration (5 materials):")
    check("global_scalar in [0.5, 1.5]", test_song_calibration_integrity)

    print("\n4. DefectManifest Integrity:")
    check("all DefectTypes mapped", test_defect_manifest_integrity)

    print("\n5. ContractValidator:")
    check("0 cross-module violations", test_contract_validator)

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 55}")
    print(f"Passed: {CHECKS_PASSED}  Failed: {CHECKS_FAILED}  ({elapsed:.1f}s)")

    if CHECKS_FAILED == 0:
        print("✅ UV3 BEREIT — Pipeline kann gestartet werden")
        return 0
    else:
        print("❌ FEHLER GEFUNDEN — Pipeline wird crashen. BITTE VORHER FIXEN.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
