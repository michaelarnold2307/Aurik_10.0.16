#!/usr/bin/env python3
"""End-to-End Lag-Test: Lädt den Song, misst Lag an jeder Pipeline-Stufe."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/media/michael/Software 4TB/Aurik_Standalone")

AUDIO_FILE = "/home/michael/Musik/Elke Best - 30 Sekunden.mp3"


def measure_lag(audio, sr, label):
    """Multi-Point Lag-Messung."""
    from backend.file_import import _estimate_interchannel_lag_multi_point, _estimate_interchannel_lag_samples

    single = _estimate_interchannel_lag_samples(audio, sr)
    multi = _estimate_interchannel_lag_multi_point(audio, sr, num_points=5)
    print(
        f"  [{label}] single={single:>6} samples ({single / sr * 1000:>6.1f}ms) | "
        f"multi_median={multi['median_lag']:>6} spread={multi['max_spread']:>6} "
        f"consistent={str(multi['consistent']):>5} | "
        f"points=[{', '.join(f'{p[0] * 100:.0f}%→{p[1]}' for p in multi['points'])}]"
    )
    return single, multi


def test_import_stage():
    """Stage 0: Load file via Aurik import."""
    print("\n=== STAGE 0: Import ===")
    from backend.file_import import load_audio_file

    result = load_audio_file(AUDIO_FILE, target_sr=48000)
    audio = result["audio"]  # type: ignore[index]
    sr = result["sr"]  # type: ignore[index]
    print(
        f"  Loaded: shape={audio.shape}, sr={sr}, lag_before={result.get('interchannel_lag_samples_before')}, lag_after={result.get('interchannel_lag_samples_after')}"  # type: ignore[union-attr]
    )
    measure_lag(audio, sr, "AFTER_IMPORT")
    return audio, sr


def test_stcg_pre_pipeline(audio, sr):
    """Stage 1: Simulate UV3's STCG pre_pipeline."""
    print("\n=== STAGE 1: STCG Pre-Pipeline ===")
    from backend.core.stereo_temporal_coherence_guard import get_stereo_temporal_coherence_guard

    stcg = get_stereo_temporal_coherence_guard()
    # This is what UV3 calls at line ~10874
    corrected = stcg.correct_interchannel_delay(audio, sr, phase_id="pre_pipeline")
    measure_lag(corrected, sr, "AFTER_PRE_PIPELINE")
    return corrected, sr


def test_phase12_simulation(audio, sr):
    """Stage 2: Simulate Phase 12 processing (introduce drift, then correct)."""
    print("\n=== STAGE 2: Phase 12 Simulation ===")
    from backend.core.stereo_temporal_coherence_guard import get_stereo_temporal_coherence_guard

    stcg = get_stereo_temporal_coherence_guard()

    # Sim: Phase 12 processes in chunks, STCG pre-chunking corrects first
    audio = stcg.correct_interchannel_delay(audio, sr, phase_id="phase_12_pre_chunking")
    measure_lag(audio, sr, "AFTER_P12_PRE_CHUNKING")

    # Sim: introduce per-chunk drift (10 samples per 5s chunk)
    chunk_s = 5
    chunk_n = int(sr * chunk_s)
    n_chunks = audio.shape[0] // chunk_n
    for ci in range(n_chunks):
        start = ci * chunk_n
        end = min(start + chunk_n, audio.shape[0])
        audio[start:end, 1] = np.roll(audio[start:end, 1], 10)

    measure_lag(audio, sr, f"AFTER_P12_SIM_DRIFT ({n_chunks} chunks)")

    # Sim: _preserve_phase_loudness STCG at end
    audio = stcg.correct_interchannel_delay(audio, sr, phase_id="phase_12_wow_flutter_fix")
    measure_lag(audio, sr, "AFTER_P12_STCG_POST")
    return audio, sr


def test_post_pipeline(audio, sr):
    """Stage 3: Simulate G14 Post-Pipeline with retries."""
    print("\n=== STAGE 3: G14 Post-Pipeline (3 retries) ===")
    from backend.core.stereo_temporal_coherence_guard import get_stereo_temporal_coherence_guard
    from backend.file_import import _estimate_interchannel_lag_multi_point

    stcg = get_stereo_temporal_coherence_guard()

    for retry in range(3):
        lag_profile = _estimate_interchannel_lag_multi_point(audio, sr, num_points=3)
        lag_pre = lag_profile["median_lag"]
        if abs(lag_pre) <= 50:
            print(f"  Retry {retry + 1}: below threshold (median={lag_pre}) — DONE")
            break
        print(f"  Retry {retry + 1}: pre={lag_pre}, spread={lag_profile['max_spread']} — correcting")
        audio = stcg.correct_interchannel_delay(audio, sr, phase_id="post_pipeline")
        lag_post = _estimate_interchannel_lag_multi_point(audio, sr, num_points=3)
        print(f"  Retry {retry + 1}: post={lag_post['median_lag']}, spread={lag_post['max_spread']}")

    measure_lag(audio, sr, "AFTER_POST_PIPELINE")
    return audio, sr


def check_stereo_correlation(audio, sr):
    """G15: Verify stereo correlation."""
    l = audio[:, 0].astype(np.float64)
    r = audio[:, 1].astype(np.float64)
    corr = float(np.corrcoef(l[: sr * 5], r[: sr * 5])[0, 1])  # First 5s
    corr_full = float(np.corrcoef(l, r)[0, 1])
    print(f"\n  G15: corr(first_5s)={corr:.4f}, corr(full)={corr_full:.4f} — {'OK' if corr > 0.5 else 'WARNING'}")
    return corr > 0.5


if __name__ == "__main__":
    print(f"Testing: {AUDIO_FILE}")

    # Stage 0
    audio, sr = test_import_stage()

    # Check: is lag reasonable after import?
    from backend.file_import import _estimate_interchannel_lag_multi_point

    lag0 = _estimate_interchannel_lag_multi_point(audio, sr)
    if abs(lag0["median_lag"]) > 100:
        print(f"\n*** ROOT CAUSE: Import did NOT correct lag. median={lag0['median_lag']} ***")
        print("*** The import STCG correction failed or was not applied. ***")
    else:
        print(f"\n✓ Import OK: median lag = {lag0['median_lag']} samples")

    # Stage 1
    audio, sr = test_stcg_pre_pipeline(audio, sr)
    lag1 = _estimate_interchannel_lag_multi_point(audio, sr)
    if abs(lag1["median_lag"]) > 100:
        print(f"\n*** ISSUE: Pre-pipeline STCG did not correct lag. median={lag1['median_lag']} ***")

    # Stage 2
    audio, sr = test_phase12_simulation(audio, sr)
    lag2 = _estimate_interchannel_lag_multi_point(audio, sr)

    # Stage 3
    audio, sr = test_post_pipeline(audio, sr)

    # Final check
    ok = check_stereo_correlation(audio, sr)

    print(f"\n=== RESULT: {'ALL CHECKS PASSED' if ok else 'ISSUES DETECTED'} ===")
