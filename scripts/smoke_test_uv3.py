"""
§2.59 UV3 Pre-Flight Smoke Test (2026-07-09)

Importiert und testet UV3-Kernfunktionen mit synthetischen Daten.
Fängt NameError, ImportError und andere Runtime-Fehler VOR dem
echten Pipeline-Run ab.

Usage: python scripts/smoke_test_uv3.py
Exit 0 = OK, Exit 1 = Fehler gefunden
"""

import sys

sys.path.insert(0, ".")
#!/usr/bin/env python3


import sys
import time


def test_import_chain() -> int:
    """Testet den kompletten Import-Chain, der im echten Run durchlaufen wird."""
    errors = 0

    modules = [
        ("PhasePruner", "backend.core.phase_pruner", "IntelligentPhasePruner"),
        ("DefectManifest", "backend.core.defect_manifest", "get_defect_manifest"),
        ("ContractValidator", "backend.core.defect_contract_validator", "run_contract_validation"),
        ("SafeDict", "backend.core.safe_dict", "SafeDict"),
        ("QualityMode", "backend.core.quality_mode", "QualityModeConfig"),
        ("SurgicalDefectAnalyzer", "backend.core.surgical_defect_analyzer", "SurgicalDefectAnalyzer"),
        ("SurgicalRepair", "backend.core.surgical_repair", "SurgicalRepair"),
        ("PeriodicHealth", "backend.core.periodic_health", "RunHealth"),
        ("VocalSentinel", "backend.core.vocal_distortion_sentinel", "VocalDistortionSentinel"),
        ("SongGoalImportance", "backend.core.song_goal_importance", "estimate_goal_importance"),
        ("SongCalibration (UV3)", "backend.core.unified_restorer_v3", "UnifiedRestorerV3"),
    ]

    for name, module, attr in modules:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors += 1

    return errors


def test_calibration_profile() -> int:
    """Testet _build_song_calibration_profile mit realistischen Werten."""
    try:
        from unittest.mock import MagicMock

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        uv3 = MagicMock(spec=UnifiedRestorerV3)
        uv3._restoration_context = {"source_fidelity_bandwidth_target_hz": 13006.0}

        profile = UnifiedRestorerV3._build_song_calibration_profile(  # type: ignore[misc]
            uv3,
            material_type="vinyl",  # type: ignore[arg-type]
            mode="restoration",  # type: ignore[arg-type]
            restorability_score=63.5,
            input_snr_db=14.3,
            max_defect_severity=0.63,
            pipeline_confidence=0.75,
            era_decade=1970,
            defect_scores={"wow": 1.0, "flutter": 1.0, "bandwidth_loss": 1.0, "clicks": 0.8},
            genre_label="Deutscher Schlager",
        )

        gs = profile.get("global_scalar", -1)
        assert 0.5 <= gs <= 1.5, f"global_scalar={gs} out of range"
        print(f"  ✅ SongCalibration: global_scalar={gs:.3f}")
        return 0
    except Exception as e:
        print(f"  ❌ SongCalibration: {e}")
        import traceback

        traceback.print_exc()
        return 1


def test_contract_validator() -> int:
    """Testet den ContractValidator."""
    try:
        from backend.core.defect_contract_validator import run_contract_validation

        result = run_contract_validation()
        if result["ok"]:
            print(f"  ✅ ContractValidator: OK ({result['violations']} violations)")
        else:
            print(f"  ⚠️  ContractValidator: {result['violations']} violations")
        return 0
    except Exception as e:
        print(f"  ❌ ContractValidator: {e}")
        return 1


def main() -> int:
    print("UV3 Pre-Flight Smoke Test")
    print("=" * 50)

    t0 = time.monotonic()
    errors = 0

    print("\n1. Import Chain:")
    errors += test_import_chain()

    print("\n2. SongCalibration:")
    errors += test_calibration_profile()

    print("\n3. ContractValidator:")
    errors += test_contract_validator()

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 50}")
    if errors == 0:
        print(f"✅ ALL PASSED ({elapsed:.1f}s) — UV3 bereit für den Run")
        return 0
    else:
        print(f"❌ {errors} ERROR(S) ({elapsed:.1f}s) — UV3 wird im Run crashen. BITTE VORHER FIXEN.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
