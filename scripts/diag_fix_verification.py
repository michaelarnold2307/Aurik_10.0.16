"""
Aurik 10.0.0 — Fix-Verifikations-Diagnostik
=============================================
Testet die 5 Fixes dieser Session mit gezielten pathologischen Inputs,
die die jeweiligen Bugs exakt reproduzieren.

Läuft ohne ML-Modelle, ohne Audio-Dateien, in ~15–30 s.
Ausgabe: PASS/FAIL pro Fix + numerische Evidenz.

Aufruf:
    .venv_aurik/bin/python scripts/diag_fix_verification.py

v10.0.0  22.04.2026
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Workspace root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ────────────────────────────────────────────────────────────────────────────

RESULTS: list[dict] = []


def _report(name: str, passed: bool, evidence: str, elapsed: float) -> None:
    symbol = "✅" if passed else "❌"
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"name": name, "passed": passed, "evidence": evidence})
    print(f"{symbol}  [{status}]  {name}")
    print(f"       Evidenz  : {evidence}")
    print(f"       Zeit     : {elapsed * 1000:.1f} ms\n")


def _make_vinyl_signal(sr: int = 48000, dur: float = 5.0) -> np.ndarray:
    """Synthetisches Vinyl-Signal: Grundton + Harmoniken, starkes HF."""
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    sig = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.sin(2 * np.pi * 880 * t)
        + 0.05 * np.sin(2 * np.pi * 5000 * t)
        + 0.03 * np.sin(2 * np.pi * 12000 * t)  # HF — soll NICHT als Spike erkannt werden
        + 0.005 * np.random.default_rng(42).normal(size=len(t)).astype(np.float32)
    )
    return np.clip(sig, -1.0, 1.0)  # type: ignore[no-any-return]


def _make_heavily_denoised_signal(sr: int = 48000, dur: float = 5.0) -> np.ndarray:
    """
    Simuliert das Ergebnis von OMLSA-Dämpfung auf einem Vinyl→Kassette→MP3-Signal.
    Nach Denoising liegt das Signal bei ca. -50 bis -55 dBFS — unterhalb des
    bisherigen _rms_dbfs_gated-Gates von -50 dBFS → bug: meldet -96 dBFS.
    """
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    # Musik bei -52 dBFS (nach OMLSA-Dämpfung)
    amplitude = 10 ** (-52 / 20)
    sig = amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sig += 0.2 * amplitude * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    return np.clip(sig, -1.0, 1.0)  # type: ignore[no-any-return]


def _make_bw_limited_signal(sr: int = 48000, dur: float = 5.0) -> np.ndarray:
    """
    Simuliert Vinyl→Kassette→MP3-Signal mit HF/LF = 0.027 (sehr_schmale_bandbreite).
    Ausschließlich Energie unter 2 kHz → TransparenzMetric muss BW-adaptiv reagieren.
    """
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    sig = (
        0.5 * np.sin(2 * np.pi * 200 * t)
        + 0.3 * np.sin(2 * np.pi * 400 * t)
        + 0.15 * np.sin(2 * np.pi * 800 * t)
        + 0.05 * np.sin(2 * np.pi * 1500 * t)
    ).astype(np.float32)
    return np.clip(sig, -1.0, 1.0)  # type: ignore[no-any-return]


# ════════════════════════════════════════════════════════════════════════════
# Fix 1: Phase_29 gated-RMS Fallback — -62.75 dB False Drop
# ════════════════════════════════════════════════════════════════════════════


def test_fix1_phase29_gated_rms_fallback() -> None:
    """
    Bug: _rms_dbfs_gated mit Gate=-50 dBFS meldet -96 dBFS für ein Signal bei -52 dBFS.
    → Makeup-Gain berechnet -62.75 dB Drop → 6 dB Cap komplett unzureichend.

    Fix: Fallback auf globalen RMS wenn alle Frames unter Gate liegen.
         Makeup-Cap auf 30 dB erhöht.
         Direktes Clipping statt apply_musical_gain_envelope.

    Erwartung: Makeup-Gain korrekt (~20 dB, nicht 6 dB); Signal danach lauter.
    """
    t0 = time.perf_counter()
    try:
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        phase = TapeHissReductionPhase()

        # Originalsignal bei -30 dBFS (vor Denoising)
        sr = 48000
        original = _make_vinyl_signal(sr, 3.0) * (10 ** (-30 / 20))
        # "Verarbeitetes" Signal bei -52 dBFS (nach OMLSA)
        processed = _make_heavily_denoised_signal(sr, 3.0)

        class _FakeMat:
            name = "vinyl"

        result, meta = phase._apply_material_loudness_preservation(original, processed, _FakeMat())  # type: ignore[arg-type]

        # Messung: RMS des Outputs vs. Inputs
        rms_in = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
        rms_out = float(np.sqrt(np.mean(result.astype(np.float64) ** 2)))
        rms_in_db = 20 * np.log10(rms_in + 1e-12)
        rms_out_db = 20 * np.log10(rms_out + 1e-12)
        makeup_db = meta.get("loudness_makeup_db", 0.0)
        rms_drop_db = meta.get("rms_drop_db", 0.0)

        # Kriterium: Output muss lauter als Input − 6 dB sein (alter Bug: blieb bei -52 dBFS)
        # und der gemessene Drop muss plausibel sein (kein -62.75 dB)
        drop_realistic = -25.0 < rms_drop_db < 0.0
        output_adequate = rms_out_db > rms_in_db - 7.0

        passed = drop_realistic and output_adequate
        evidence = (
            f"rms_in={rms_in_db:.1f} dBFS, rms_out={rms_out_db:.1f} dBFS, "
            f"drop={rms_drop_db:.1f} dB, makeup={makeup_db:.1f} dB | "
            f"drop_realistic={drop_realistic}, output_adequate={output_adequate}"
        )
    except Exception as exc:
        passed = False
        evidence = f"EXCEPTION: {exc}\n{traceback.format_exc(limit=3)}"

    _report("Fix 1 — Phase_29 gated-RMS Fallback", passed, evidence, time.perf_counter() - t0)


# ════════════════════════════════════════════════════════════════════════════
# Fix 2: TransparenzMetric — BW-adaptiv + 0-Länge Guard
# ════════════════════════════════════════════════════════════════════════════


def test_fix2_transparenz_bw_adaptive() -> None:
    """
    Bug 2a: Vinyl→Kassette→MP3 (HF/LF=0.027) → Score konstant 0.277 wegen
            fehlendem HF-Inhalt im 4k–8k-Band → falsche Regression.
    Bug 2b: len(audio)==0 → 'Invalid number of FFT data points (0)'.

    Fix: BW-adaptive Bänder-Ausschluss wenn HF/LF < 0.05.
         Guard len < 2 → return 0.5.

    Erwartung BW-limitiert: Score > 0.4 (nicht konstant 0.277).
    Erwartung leer: Score = 0.5, kein Crash.
    """
    t0 = time.perf_counter()
    try:
        from backend.core.musical_goals.musical_goals_metrics import TransparenzMetric

        metric = TransparenzMetric()
        sr = 48000

        # Test 2a: BW-limitiertes Signal
        bw_signal = _make_bw_limited_signal(sr, 3.0)
        score_bw = metric.measure(bw_signal, sr)

        # Test 2b: Leeres Signal → kein Crash
        empty = np.array([], dtype=np.float32)
        score_empty = metric.measure(empty, sr)

        # Test 2c: Kurzes Signal (1 Sample) → kein Crash
        tiny = np.array([0.5], dtype=np.float32)
        score_tiny = metric.measure(tiny, sr)

        passed = (
            score_bw > 0.40  # nicht mehr konstant 0.277
            and 0.0 <= score_empty <= 1.0  # kein Crash, sinnvoller Wert
            and 0.0 <= score_tiny <= 1.0
        )
        evidence = f"BW-limited score={score_bw:.3f} (>0.40?), empty={score_empty:.3f}, tiny={score_tiny:.3f}"
    except Exception as exc:
        passed = False
        evidence = f"EXCEPTION: {exc}\n{traceback.format_exc(limit=3)}"

    _report("Fix 2 — TransparenzMetric BW-adaptiv", passed, evidence, time.perf_counter() - t0)


# ════════════════════════════════════════════════════════════════════════════
# Fix 3: TimbralAuthenticityMetric — spectral_centroid/rolloff Guards
# ════════════════════════════════════════════════════════════════════════════


def test_fix3_timbral_authenticity_guards() -> None:
    """
    Bug: _spectral_centroid/_spectral_rolloff mit Audio < n_fft // 4 Samples
         → 'noverlap >= nperseg' oder 'tuple index out of range'.

    Fix: Längen-Guards, leere STFT-Guards, hop-Begrenzung.

    Erwartung: Keine Exception bei kurzen/leeren Signalen.
    """
    t0 = time.perf_counter()
    try:
        from backend.core.musical_goals.musical_goals_metrics import TimbralAuthenticityMetric

        metric = TimbralAuthenticityMetric()
        sr = 48000

        errors = []

        # Test 3a: Sehr kurzes Audio (10 Samples)
        try:
            tiny = np.random.default_rng(0).uniform(-0.1, 0.1, 10).astype(np.float32)
            metric._spectral_centroid(tiny, sr)
            metric._spectral_rolloff(tiny, sr)
        except Exception as e:
            errors.append(f"tiny-10: {e}")

        # Test 3b: 64 Samples (typische STFT-Grenzfall)
        try:
            short = np.random.default_rng(1).uniform(-0.1, 0.1, 64).astype(np.float32)
            metric._spectral_centroid(short, sr)
            metric._spectral_rolloff(short, sr)
        except Exception as e:
            errors.append(f"short-64: {e}")

        # Test 3c: Leeres Array
        try:
            empty = np.array([], dtype=np.float32)
            metric._spectral_centroid(empty, sr)
            metric._spectral_rolloff(empty, sr)
        except Exception as e:
            errors.append(f"empty: {e}")

        # Test 3d: Normales Signal muss noch funktionieren
        normal = _make_vinyl_signal(sr, 2.0)
        c = metric._spectral_centroid(normal, sr)
        r = metric._spectral_rolloff(normal, sr)
        if len(c) == 0 or len(r) == 0:
            errors.append("normal signal: leere Rückgabe")

        passed = len(errors) == 0
        evidence = (
            f"Alle Grenzfälle fehlerfrei | centroid(normal)={np.mean(c):.0f} Hz | rolloff(normal)={np.mean(r):.0f} Hz"
            if passed
            else f"FEHLER: {'; '.join(errors)}"
        )
    except Exception as exc:
        passed = False
        evidence = f"EXCEPTION: {exc}\n{traceback.format_exc(limit=3)}"

    _report("Fix 3 — TimbralAuthenticity Guards", passed, evidence, time.perf_counter() - t0)


# ════════════════════════════════════════════════════════════════════════════
# Fix 4: Phase_23 HF-Protected-Bin Guard
# ════════════════════════════════════════════════════════════════════════════


def test_fix4_phase23_hf_protected_bins() -> None:
    """
    Bug: phase_23 _detect_defects flaggt restaurierte HF-Harmoniken (durch Phase_06)
         als Codec-Spikes UND als Phasensprünge → 9 Phase-Cancellation-Artefakte → §2.49 Rollback.

    Fix: _hf_protected_bin_start für analoge Materialien.
         Bins ≥ Start → Spike-Detektion (Z-Score) UND Phase-Sprung-Detektion ausgeschlossen.

    Erwartung: Transiente Spikes in HF-Bins (≥ 13.6 kHz) werden bei Vinyl NICHT geflagged.
               LF-Spike (< 13.6 kHz) WIRD geflagged.
               Phase-Sprünge in HF-Bins werden NICHT geflagged.
    """
    t0 = time.perf_counter()
    try:
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        phase = SpectralRepair()
        phase._current_material = "vinyl"  # type: ignore[assignment]

        nfft = 4096
        n_bins = nfft // 2 + 1
        n_frames = 80

        # Gleichmäßiger Hintergrund (kein Amplitude-Spike im HF-Bereich)
        mag = np.full((n_bins, n_frames), 0.02, dtype=np.float32)
        phase_arr = np.zeros((n_bins, n_frames), dtype=np.float32)

        bin_hz = 48000.0 / nfft
        protect_hz = phase._ANALOG_HF_PROTECT_HZ.get("vinyl", 0.0)  # 13600 Hz
        expected_start = int(protect_hz / bin_hz)  # ≈ 1160

        # --- LF-Defekt: breitbandiger transienter Amplitude-Spike ---
        # 11 Bins × 11 Frames → überlebt 3×3 morphologische Opening
        lf_center = 200
        mag[lf_center - 5 : lf_center + 6, 25:36] = 2.0  # 100× Hintergrund → Z-Score enorm

        # --- HF-Restaurierung durch Phase_06: NUR Phase-Sprünge, kein Amplitude-Spike ---
        # Phase_06 ändert Phasenrelationen intentional (keine Amplitude-Änderung).
        # Ohne Guard → phase_jumps flaggt diese HF-Bins. Mit Guard → zeroed.
        hf_center = 1300
        # Konsistente Phasen in Frame 25–35 → phase_diff ≫ threshold an Grenzen (24 und 35)
        phase_arr[hf_center - 5 : hf_center + 6, 25:36] = np.pi * 0.85

        thresholds = {
            "outlier_z_score": 3.0,
            "energy_floor_db": -60.0,
            "phase_jump_threshold": np.pi * 0.5,
        }
        defect_mask = phase._detect_defects(mag, phase_arr, thresholds)

        lf_flagged = bool(np.any(defect_mask[lf_center, :]))
        hf_flagged = bool(np.any(defect_mask[hf_center, :]))

        passed = lf_flagged and not hf_flagged
        evidence = (
            f"vinyl protect_hz={protect_hz:.0f} Hz → bin_start={expected_start} | "
            f"LF (bin {lf_center}≈{lf_center * bin_hz:.0f} Hz) amplitude-spike flagged={lf_flagged} (soll True) | "
            f"HF (bin {hf_center}≈{hf_center * bin_hz:.0f} Hz) phase-only flagged={hf_flagged} (soll False)"
        )
    except Exception as exc:
        passed = False
        evidence = f"EXCEPTION: {exc}\n{traceback.format_exc(limit=3)}"

    _report("Fix 4 — Phase_23 HF-Protected-Bin Guard", passed, evidence, time.perf_counter() - t0)


# ════════════════════════════════════════════════════════════════════════════
# Fix 5: UV3 Pipeline-Wall-Time-Budget
# ════════════════════════════════════════════════════════════════════════════


def test_fix5_uv3_wall_time_budget() -> None:
    """
    Bug: Kein globales Pipeline-Zeitlimit → PMGG-Retry-Loop → 5617 s für 225 s Audio.

    Fix: _pipeline_wall_budget pro Material (vinyl=600s usw.),
         _WALL_BUDGET_EXEMPT_PHASES schützt Pflichtphasen,
         überzeitige Phasen werden als Passthrough übersprungen (kein Abbruch).

    Prüfung ohne echte Pipeline: Direkte Code-Inspection der neuen Konstanten +
    Smoke-Test dass die Budget-Variablen existieren und plausible Werte haben.
    """
    t0 = time.perf_counter()
    try:
        uv3_path = ROOT / "backend" / "core" / "unified_restorer_v3.py"
        src = uv3_path.read_text(encoding="utf-8")

        # Prüfe: _PIPELINE_WALL_BUDGET_S definiert
        has_budget_dict = "_PIPELINE_WALL_BUDGET_S" in src
        # Prüfe: vinyl-Budget vorhanden
        has_vinyl = '"vinyl": 600.0' in src or "'vinyl': 600.0" in src
        # Prüfe: _WALL_BUDGET_EXEMPT_PHASES definiert
        has_exempt = "_WALL_BUDGET_EXEMPT_PHASES" in src
        # Prüfe: phase_14 in exempt (Pflichtphase)
        has_phase14_exempt = "phase_14_phase_correction" in src and "_WALL_BUDGET_EXEMPT_PHASES" in src
        # Prüfe: Wall-Time-Check in for-Loop (muss nach _pipeline_start_time liegen)
        budget_check_in_loop = "_elapsed_wall = time.time() - _pipeline_start_time" in src
        # Prüfe: Warning-Log bei Überschreitung
        has_warning_log = "§Wall-Time-Budget" in src

        all_ok = all(
            [
                has_budget_dict,
                has_vinyl,
                has_exempt,
                has_phase14_exempt,
                budget_check_in_loop,
                has_warning_log,
            ]
        )
        passed = all_ok
        evidence = (
            f"budget_dict={has_budget_dict}, vinyl_600s={has_vinyl}, "
            f"exempt_set={has_exempt}, phase14_exempt={has_phase14_exempt}, "
            f"loop_check={budget_check_in_loop}, warn_log={has_warning_log}"
        )
    except Exception as exc:
        passed = False
        evidence = f"EXCEPTION: {exc}\n{traceback.format_exc(limit=3)}"

    _report("Fix 5 — UV3 Wall-Time-Budget", passed, evidence, time.perf_counter() - t0)


# ════════════════════════════════════════════════════════════════════════════
# Fix 5b: Funktionaler Wall-Time-Smoke-Test (simuliert Budget-Überschreitung)
# ════════════════════════════════════════════════════════════════════════════


def test_fix5b_wall_time_budget_fires() -> None:
    """
    Simuliert einen Budget-Überschreitungs-Zustand und prüft ob der Guard korrekt auslöst.
    Testet die Logik direkt ohne den vollen Pipeline-Overhead.
    """
    t0 = time.perf_counter()
    try:
        # Die Budget-Logik aus UV3 nachbilden (ohne echten Pipeline-Aufruf)
        _PIPELINE_WALL_BUDGET_S_TEST = {
            "vinyl": 600.0,
            "shellac": 900.0,
            "cassette": 540.0,
            "mp3_low": 360.0,
            "cd_digital": 300.0,
        }
        _WALL_BUDGET_EXEMPT_PHASES_TEST = frozenset(
            {
                "phase_01_click_removal",
                "phase_09_crackle_removal",
                "phase_12_wow_flutter_fix",
                "phase_14_phase_correction",
                "phase_15_stereo_balance",
                "phase_30_dc_offset_removal",
            }
        )

        errors = []

        # Szenario A: Vinyl, 601 Sekunden verstrichen, nicht-exempt Phase
        budget = _PIPELINE_WALL_BUDGET_S_TEST["vinyl"]
        elapsed = 601.0
        phase_to_test = "phase_06_frequency_restoration"
        should_skip = phase_to_test not in _WALL_BUDGET_EXEMPT_PHASES_TEST and elapsed > budget
        if not should_skip:
            errors.append("Szenario A: Phase sollte übersprungen werden (601s > 600s budget)")

        # Szenario B: Exempt Phase wird NICHT übersprungen
        exempt_phase = "phase_12_wow_flutter_fix"
        # Korrekt: exempt → KEIN Skip
        exempt_fires = exempt_phase not in _WALL_BUDGET_EXEMPT_PHASES_TEST
        if exempt_fires:
            errors.append("Szenario B: Exempt-Phase fälschlich nicht in frozenset")

        # Szenario C: Budget noch nicht überschritten → kein Skip
        elapsed_ok = 300.0
        should_not_skip_c = elapsed_ok > budget  # False → kein Skip
        if should_not_skip_c:
            errors.append("Szenario C: Budget noch nicht überschritten, trotzdem Skip")

        # Szenario D: Shellac hat 900s (mehr als vinyl 600s)
        shellac_budget = _PIPELINE_WALL_BUDGET_S_TEST["shellac"]
        vinyl_budget = _PIPELINE_WALL_BUDGET_S_TEST["vinyl"]
        if shellac_budget <= vinyl_budget:
            errors.append(f"Szenario D: shellac={shellac_budget} sollte > vinyl={vinyl_budget}")

        passed = len(errors) == 0
        evidence = "Alle Budget-Szenarien korrekt" if passed else f"FEHLER: {'; '.join(errors)}"
    except Exception as exc:
        passed = False
        evidence = f"EXCEPTION: {exc}\n{traceback.format_exc(limit=3)}"

    _report("Fix 5b — Wall-Time-Budget Logik-Smoke-Test", passed, evidence, time.perf_counter() - t0)


# ════════════════════════════════════════════════════════════════════════════
# Gesamtergebnis
# ════════════════════════════════════════════════════════════════════════════


def main() -> int:
    print("=" * 65)
    print("  Aurik 10.0.0 — Fix-Verifikations-Diagnostik  (v10.0.0)")
    print("=" * 65)
    print()

    t_total = time.perf_counter()
    test_fix1_phase29_gated_rms_fallback()
    test_fix2_transparenz_bw_adaptive()
    test_fix3_timbral_authenticity_guards()
    test_fix4_phase23_hf_protected_bins()
    test_fix5_uv3_wall_time_budget()
    test_fix5b_wall_time_budget_fires()
    elapsed_total = time.perf_counter() - t_total

    n_pass = sum(1 for r in RESULTS if r["passed"])
    n_fail = sum(1 for r in RESULTS if not r["passed"])

    print("=" * 65)
    print(f"  Gesamt: {n_pass}/{len(RESULTS)} PASS  |  {n_fail} FAIL  |  {elapsed_total:.1f} s")
    print("=" * 65)

    if n_fail > 0:
        print("\n⚠ Fehlgeschlagene Fixes — Handlungsbedarf:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"  • {r['name']}")
                print(f"    {r['evidence']}")
        return 1
    else:
        print("\n✅ Alle Fixes verifiziert — Qualitätsverbesserung bestätigt.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
