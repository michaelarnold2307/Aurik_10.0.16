#!/usr/bin/env python3
"""Golden Case Audit: Musikalische Ziele bei echtem Gesang.

Testet: Werden alle 15 musikalischen Ziele bei einer Restaurierung
mit echter Musik + Gesang auf ihre Schwellwerte erreicht?

Testdatei: test_audio/vocals/opera_sibilance.wav (Opernsänger mit Sibilanz-Artefakt)
Mode: Restoration (original audio fidelity)
"""

import sys
from pathlib import Path

import librosa
import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Backend imports
sys.path.insert(0, str(_WORKSPACE_ROOT))

# Pylint C0413 is intentional here: the script can be executed directly via
# `python audit/audit_golden_case_vocal.py`, so the workspace root must be added
# before importing backend modules.
from backend.core.calibration_matrix import CANONICAL_THRESHOLDS_RESTORATION  # pylint: disable=wrong-import-position
from backend.core.musical_goals.musical_goals_metrics import (  # pylint: disable=wrong-import-position
    MusicalGoalsChecker,
)
from backend.core.unified_restorer_v3 import UnifiedRestorerV3  # pylint: disable=wrong-import-position
from backend.file_import import load_audio_file  # pylint: disable=wrong-import-position


def run_audit():
    """Lade Audio, starte Restaurierung, messe Goals."""

    def _meta_get(d: dict, *path: str, default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)  # type: ignore[assignment]
            if cur is None:
                return default
        return cur

    audio_path = _WORKSPACE_ROOT / "test_audio" / "vocals" / "opera_sibilance.wav"
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return 1

    # Load audio
    print(f"📁 Loading: {audio_path}")
    load_result = load_audio_file(str(audio_path))
    if load_result is None or load_result.get("error"):
        print(
            f"❌ Failed to load audio: {load_result.get('error', 'unknown error') if load_result else 'None returned'}"
        )
        return 1
    audio = load_result["audio"]
    sr = load_result["sr"]
    print(f"   ✓ {len(audio) / sr:.1f}s @ {sr} Hz")

    # Run restoration
    print("\n🎵 Running Restoration (Restoration mode)...")
    uv3 = UnifiedRestorerV3()
    result = uv3.restore(
        audio,
        sr,
        mode="restoration",
        progress_callback=lambda pct: print(f"   {pct:3.0f}%", end="\r", flush=True),
    )
    print("   ✓ Restoration complete")

    # Measure Musical Goals
    print("\n📊 Measuring 15 Musical Goals...")
    checker = MusicalGoalsChecker()
    # Measure at the restoration output sample rate to avoid metric distortion.
    out_sr = int(_meta_get(result.metadata, "sample_rate", "output", default=sr) or sr)
    ref_audio = audio
    if out_sr != sr:
        ref_audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=out_sr)
        print(f"   ℹ️ Reference resampled: {sr} -> {out_sr} Hz")
    goals = checker.measure_all(result.audio, out_sr, reference=ref_audio)

    # Display results with pass/fail against canonical thresholds
    print("\n" + "=" * 80)
    print(f"{'Goal':<25} {'Score':>8} {'Threshold':>10} {'Status':>10} {'Material Ceiling':>15}")
    print("=" * 80)

    material = result.metadata.get("defect_analysis", {}).get("material", "unknown")
    canonic = CANONICAL_THRESHOLDS_RESTORATION  # Restoration mode fallback
    # Prefer adaptive thresholds from pipeline (material/restorability-adjusted)
    _mg_meta = result.metadata.get("musical_goals", {})
    _adaptive_thr = _mg_meta.get("thresholds") if isinstance(_mg_meta, dict) else None
    if not isinstance(_adaptive_thr, dict) or not _adaptive_thr:
        _adaptive_thr = {}

    passed = 0
    failed = 0

    for goal_name, goal_score in goals.items():
        goal_key = goal_name.lower()
        # Adaptive threshold first (material + restorability calibrated),
        # fall back to canonical if not populated for this goal.
        threshold = _adaptive_thr.get(goal_key) or canonic.get(goal_key, 0.70)
        canonical_thr = canonic.get(goal_key, 0.70)

        status = "✅ PASS" if goal_score >= threshold else "❌ FAIL"
        if goal_score >= threshold:
            passed += 1
        else:
            failed += 1

        print(f"{goal_name:<25} {goal_score:8.3f} {threshold:10.3f} {status:>10} {canonical_thr:15.3f}")

    print("=" * 80)
    print(f"\n🎯 Summary: {passed}/14 goals passed (threshold)")
    print(f"   Material: {material}")
    print(f"   Duration: {len(result.audio) / out_sr:.1f}s")
    print(f"   Output SR: {out_sr}")
    _rest_score = _meta_get(result.metadata, "defect_analysis", "restorability_score", default=None)
    if _rest_score is None:
        _rest_score = _meta_get(result.metadata, "song_calibration", "restorability_score", default=None)
    if _rest_score is None:
        _rest_score = _meta_get(result.metadata, "recovery_certainty", "restorability_score", default="N/A")
    print(f"   Restorability Score: {_rest_score}")

    # Print detailed metadata
    print("\n📋 Metadata Snapshot:")
    for key, meta_path in [
        ("primary_material", ("defect_analysis", "material")),
        ("transfer_chain", ("defect_analysis", "transfer_chain")),
        ("restorability_score", ("defect_analysis", "restorability_score")),
        ("restorability_score_song_cal", ("song_calibration", "restorability_score")),
        ("restorability_score_recovery", ("recovery_certainty", "restorability_score")),
        ("carrier_chain_recovery_ratio", ("carrier_chain_recovery_ratio",)),
    ]:
        # Extract nested keys: ("defect_analysis", "material") → metadata["defect_analysis"]["material"]
        val = result.metadata
        for k in meta_path:
            val = val.get(k, {}) if isinstance(val, dict) else None
            if val is None:
                break
        if isinstance(val, (list, dict)):
            print(f"   {key}: {str(val)[:60]}...")
        else:
            print(f"   {key}: {val if val is not None else 'N/A'}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(run_audit())
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(2)
