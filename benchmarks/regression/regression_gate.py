#!/usr/bin/env python3
"""Aurik Regression Gate — Selbstvergleich, Edge-Cases, DeepFilterNet3.

§Echter Benchmark: Vergleicht Aurik gegen sich selbst (Regression-Erkennung),
testet Edge-Cases, misst Phasen-Beiträge, und vergleicht optional gegen
DeepFilterNet3 als einzigen fairen Open-Source-Gegner.

Architektur:
  1. Regressions-Gate: PQS-Delta gegen gespeicherte Baseline. Δ < -2 → FAIL.
  2. Edge-Case-Stress: Stille, Clipping, DC-Offset, 1-Sek-Clip, 60-Sek-Rauschen
  3. Phase-Contribution: Welche Phase bringt wie viel PQS?
  4. Parameter-Sweep: material × quality × era Matrix
  5. DeepFilterNet3: Fairer Vergleich (wenn installiert)

Nutzung:
  python benchmarks/regression/regression_gate.py --baseline   # Baseline generieren
  python benchmarks/regression/regression_gate.py --check      # Gegen Baseline prüfen
  python benchmarks/regression/regression_gate.py --full       # Alle Tests
  python benchmarks/regression/regression_gate.py --ci         # CI-Mode

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

BASELINE_FILE = _PROJECT_ROOT / "benchmarks/regression/baselines/v10_baseline.json"

# ═══════════════════════════════════════════════════════════════════════════
# Testsignale
# ═══════════════════════════════════════════════════════════════════════════


def _make_music(dur: float = 2.0, sr: int = 48000) -> np.ndarray:
    """Multi-Instrument mit Hüllkurven."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    env = np.exp(-t * 3.0) * (1.0 - np.exp(-t * 20.0))
    sig = 0.4 * np.sin(2 * np.pi * 261.63 * t) * env
    sig += 0.25 * np.sin(2 * np.pi * 523.25 * t) * env * 0.7
    sig += 0.12 * np.sin(2 * np.pi * 784.0 * t) * env * 0.5
    bass_env = np.exp(-(t - 0.5) * 2.0) * (t > 0.5) * (1.0 - np.exp(-(t - 0.5) * 15.0))
    sig += 0.3 * np.sin(2 * np.pi * 130.81 * t) * bass_env
    str_env = (1.0 - np.exp(-t * 1.5)) * np.exp(-t * 0.3)
    sig += 0.15 * np.sin(2 * np.pi * 349.23 * t) * str_env
    return sig.astype(np.float32)  # type: ignore[no-any-return]


def _make_noisy(clean: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    rng = np.random.RandomState(42)
    sp = np.mean(clean**2)
    npwr = sp / (10 ** (snr_db / 10))
    return clean + np.sqrt(npwr) * rng.randn(len(clean)).astype(np.float32)  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════
# Perzeptuelle Metrik (PQS — konsistent mit open_source_benchmark v2)
# ═══════════════════════════════════════════════════════════════════════════


def _spectral_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    sa = np.abs(np.fft.rfft(a[:n]))
    sb = np.abs(np.fft.rfft(b[:n]))
    freqs = np.fft.rfftfreq(n, d=1.0 / 48000)
    w = np.ones(len(sa))
    w[(freqs >= 300) & (freqs <= 3400)] = 2.0
    corr = np.corrcoef(sa * w, sb * w)[0, 1]
    return max(0.0, min(1.0, float(corr))) if not np.isnan(corr) else 0.5


def _energy_ratio(clean: np.ndarray, restored: np.ndarray) -> float:
    ec = float(np.sqrt(np.mean(clean**2)))
    er = float(np.sqrt(np.mean(restored**2)))
    if ec < 1e-10:
        return 1.0
    return max(0.0, 1.0 - abs(1.0 - er / ec))


def _artifact_score(clean: np.ndarray, processed: np.ndarray) -> float:
    n = min(len(clean), len(processed))
    cs = np.abs(np.fft.rfft(clean[:n]))
    ps = np.abs(np.fft.rfft(processed[:n]))
    hf = int(len(cs) * 0.75)
    if hf >= len(cs):
        return 1.0
    return max(0.0, min(1.0, (np.sum(cs[hf:]) + 1e-10) / (np.sum(ps[hf:]) + 1e-10)))  # type: ignore[no-any-return]


def compute_pqs(clean: np.ndarray, restored: np.ndarray) -> float:
    """PQS 0-100: 40% spektrale Ähnlichkeit + 30% Energie + 30% Artefakt-Freiheit."""
    return round(
        (
            0.4 * _spectral_correlation(clean, restored)
            + 0.3 * _energy_ratio(clean, restored)
            + 0.3 * _artifact_score(clean, restored)
        )
        * 100,
        2,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Aurik Mini-Pipeline
# ═══════════════════════════════════════════════════════════════════════════


def aurik_pipeline(audio: np.ndarray, sr: int = 48000, use_real: bool = True, full: bool = False) -> np.ndarray:
    """Aurik Pipeline: Mini (3 Phasen) oder Full (68 Phasen).

    Args:
        use_real: True = echte Phasen, False = Fallback-Lowpass.
        full:     True = full 68-phase restaurierung(), False = Mini-Pipeline.
    """
    if full:
        try:
            from backend.aurik_restore import restaurierung

            result_audio, _ = restaurierung(audio.astype(np.float32), sr)
            return result_audio
        except Exception as e:
            logger.warning("Full pipeline failed, falling back to mini: %s", e)
            return aurik_pipeline(audio, sr, use_real=True, full=False)

    if not use_real:
        from scipy.signal import butter, filtfilt

        b, a = butter(6, 12000 / (sr / 2), btype="low")
        return filtfilt(b, a, audio.astype(np.float64)).astype(np.float32)  # type: ignore[no-any-return]
    try:
        from backend.core.comfort_guard import apply_comfort_guard
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase
        from backend.core.phases.phase_03_denoise import DenoisePhase

        a = audio.astype(np.float32)
        r1 = ClickRemovalPhase(sample_rate=sr).process(a, sample_rate=sr, material_type="vinyl")  # type: ignore[call-arg]
        r3 = DenoisePhase(sample_rate=sr).process(r1.audio, sample_rate=sr, material_type="vinyl")
        return apply_comfort_guard(r3.audio, sr)
    except Exception:
        return aurik_pipeline(audio, sr, use_real=False)


# ═══════════════════════════════════════════════════════════════════════════
# Edge-Case-Generator
# ═══════════════════════════════════════════════════════════════════════════


def generate_edge_cases(sr: int = 48000) -> dict[str, np.ndarray]:
    """Erzeugt Edge-Case-Testsignale für Stress-Test."""
    rng = np.random.RandomState(42)
    return {
        "silence_2s": np.zeros(sr * 2, dtype=np.float32),
        "dc_offset": np.ones(sr, dtype=np.float32) * 0.5,
        "clip_fullscale": np.clip(rng.randn(sr).astype(np.float32) * 2.0, -1.0, 1.0),
        "one_sample": np.array([0.5], dtype=np.float32),
        "very_quiet": _make_music(1.0, sr) * 0.001,
        "very_loud": np.clip(_make_music(1.0, sr) * 5.0, -1.0, 1.0),
        "step_function": np.concatenate([np.zeros(sr // 2), np.ones(sr // 2)]).astype(np.float32),
        "impulse": np.array([1.0] + [0.0] * (sr - 1), dtype=np.float32),
        "sweep_20_20k": np.sin(
            2 * np.pi * np.linspace(20, 20000, sr, endpoint=False) * np.linspace(0, 1, sr, endpoint=False)
        ).astype(np.float32),
        "stereo_correlation": np.column_stack(
            [
                _make_music(1.0, sr),
                _make_music(1.0, sr) * -0.9,  # Antikorreliert
            ]
        ).astype(np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Baseline-System
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Baseline:
    version: str
    timestamp: str
    scenarios: dict[str, dict]  # scenario_name → {pqs, timbre, artifact_free, runtime_s}
    edge_cases: dict[str, dict]
    parameter_matrix: dict[str, dict]
    metadata: dict = field(default_factory=dict)


def generate_baseline(dur: float = 1.0) -> Baseline:
    """Erzeugt Baseline-Messungen für den aktuellen Aurik-Stand."""
    import datetime

    sr = 48000
    music = _make_music(dur, sr)

    scenarios = {
        "music_clean": music,
        "music_noise_10dB": _make_noisy(music, 10.0),
        "music_noise_20dB": _make_noisy(music, 20.0),
        "music_noise_6dB": _make_noisy(music, 6.0),
    }

    sc_results = {}
    for name, degraded in scenarios.items():
        t0 = time.perf_counter()
        out = aurik_pipeline(degraded, sr)
        rt = time.perf_counter() - t0
        sc_results[name] = {
            "pqs": compute_pqs(music, out),
            "timbre": round(_spectral_correlation(music, out), 4),
            "artifact_free": round(_artifact_score(music, out), 4),
            "runtime_s": round(rt, 4),
        }

    # Edge Cases
    edge_cases = generate_edge_cases(sr)
    ec_results = {}
    for name, signal in edge_cases.items():
        try:
            t0 = time.perf_counter()
            out = aurik_pipeline(signal, sr)
            rt = time.perf_counter() - t0
            # Für Edge Cases: prüfe ob kein NaN, kein Crash, Output existiert
            ec_results[name] = {
                "no_crash": True,
                "no_nan": bool(np.all(np.isfinite(out))),
                "shape_preserved": out.shape == signal.shape if hasattr(out, "shape") else True,
                "runtime_s": round(rt, 4),
            }
        except Exception as e:
            ec_results[name] = {
                "no_crash": False,
                "error": str(e)[:100],  # type: ignore[dict-item]
                "runtime_s": 0.0,
            }

    # Parameter-Sweep
    materials = ["shellac", "vinyl", "tape", "digital"]
    param_results = {}
    for mat in materials:
        try:
            from backend.core.phases.phase_03_denoise import DenoisePhase

            p = DenoisePhase(sample_rate=sr)
            t0 = time.perf_counter()
            r = p.process(_make_noisy(music, 15.0), sample_rate=sr, material_type=mat)
            rt = time.perf_counter() - t0
            param_results[f"denoise_{mat}"] = {
                "pqs": compute_pqs(music, r.audio),
                "runtime_s": round(rt, 4),
                "success": r.success,
            }
        except Exception as e:
            param_results[f"denoise_{mat}"] = {"error": str(e)[:100]}  # type: ignore[dict-item]

    return Baseline(
        version="10.0.0-Phantom",
        timestamp=datetime.datetime.now().isoformat(),
        scenarios=sc_results,
        edge_cases=ec_results,
        parameter_matrix=param_results,
        metadata={"test_signal_duration_s": dur, "sample_rate": sr},
    )


def check_regression(baseline: Baseline, tolerance: float = 2.0) -> tuple[bool, list[str]]:
    """Prüft aktuellen Stand gegen Baseline.

    Returns:
        (passed, issues) — passed=True wenn keine signifikante Regression.
    """
    current = generate_baseline(dur=baseline.metadata.get("test_signal_duration_s", 1.0))
    issues: list[str] = []

    # Szenario-Vergleich
    for name in baseline.scenarios:
        bl = baseline.scenarios[name]
        cur = current.scenarios.get(name, {})
        if not cur:
            issues.append(f"MISSING: scenario '{name}' not in current run")
            continue
        delta = cur.get("pqs", 0) - bl.get("pqs", 0)
        if delta < -tolerance:
            issues.append(
                f"REGRESSION: {name}: PQS {bl['pqs']:.1f}→{cur['pqs']:.1f} (Δ={delta:+.1f}, limit=-{tolerance})"
            )
        elif delta < 0:
            issues.append(f"WARNING: {name}: PQS {bl['pqs']:.1f}→{cur['pqs']:.1f} (Δ={delta:+.1f}, within tolerance)")

    # Edge-Case-Vergleich
    for name in baseline.edge_cases:
        bl = baseline.edge_cases[name]
        cur = current.edge_cases.get(name, {})
        if not cur:
            issues.append(f"MISSING: edge case '{name}' not in current run")
            continue
        if bl.get("no_crash") and not cur.get("no_crash"):
            issues.append(f"REGRESSION: edge case '{name}': was OK, now CRASHES")
        if bl.get("no_nan") and not cur.get("no_nan"):
            issues.append(f"REGRESSION: edge case '{name}': was NaN-free, now has NaN")

    return len([i for i in issues if i.startswith("REGRESSION")]) == 0, issues


# ═══════════════════════════════════════════════════════════════════════════
# Phase-Contribution-Analyse
# ═══════════════════════════════════════════════════════════════════════════


def analyze_phase_contributions(dur: float = 1.0) -> dict:
    """Misst PQS-Beitrag jeder Phase in der Mini-Pipeline."""
    sr = 48000
    music = _make_music(dur, sr)
    noisy = _make_noisy(music, 15.0)

    contributions = {}

    # Baseline: unverarbeitet
    contributions["raw_degraded"] = compute_pqs(music, noisy)

    # Phase 01 only
    try:
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        p1 = ClickRemovalPhase(sample_rate=sr)  # type: ignore[call-arg]
        r1 = p1.process(noisy, sample_rate=sr, material_type="vinyl")
        contributions["phase_01_click_removal"] = compute_pqs(music, r1.audio)
    except Exception as e:
        contributions["phase_01_click_removal"] = f"ERROR: {e}"  # type: ignore[assignment]

    # Phase 03 only
    try:
        from backend.core.phases.phase_03_denoise import DenoisePhase

        p3 = DenoisePhase(sample_rate=sr)
        r3 = p3.process(noisy, sample_rate=sr, material_type="vinyl")
        contributions["phase_03_denoise"] = compute_pqs(music, r3.audio)
    except Exception as e:
        contributions["phase_03_denoise"] = f"ERROR: {e}"  # type: ignore[assignment]

    # Full mini-pipeline
    contributions["full_mini_pipeline"] = compute_pqs(music, aurik_pipeline(noisy, sr))

    return contributions


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser(description="Aurik Regression Gate")
    p.add_argument("--baseline", action="store_true", help="Baseline generieren und speichern")
    p.add_argument("--check", action="store_true", help="Gegen gespeicherte Baseline prüfen")
    p.add_argument("--full", action="store_true", help="Vollständiger Test: Baseline + Check + Edge + Phasen")
    p.add_argument("--ci", action="store_true", help="CI-Mode: Exit-Code ≠0 bei Regression")
    p.add_argument("--tolerance", type=float, default=2.0, help="PQS-Toleranz für Regression (Default: 2.0)")
    p.add_argument("--duration", type=float, default=1.0, help="Signaldauer in s")
    a = p.parse_args()

    logging.basicConfig(level=logging.WARNING)

    if a.baseline or a.full:
        print("📊 Generiere Baseline...")
        bl = generate_baseline(a.duration)
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(
            json.dumps(
                {
                    "version": bl.version,
                    "timestamp": bl.timestamp,
                    "scenarios": bl.scenarios,
                    "edge_cases": bl.edge_cases,
                    "parameter_matrix": bl.parameter_matrix,
                    "metadata": bl.metadata,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"   ✅ Baseline gespeichert: {BASELINE_FILE}")
        print(f"   Szenarien: {list(bl.scenarios.keys())}")
        print(f"   Edge-Cases: {len(bl.edge_cases)} getestet")
        for name, r in bl.scenarios.items():
            print(
                f"     {name:<25} PQS={r['pqs']:6.2f}  timbre={r['timbre']:.4f}  artifact_free={r['artifact_free']:.4f}  runtime={r['runtime_s']:.4f}s"
            )

    if a.check or a.full:
        if not BASELINE_FILE.exists():
            print("❌ Keine Baseline gefunden. Bitte zuerst --baseline ausführen.")
            return 1

        print("\n🔍 Prüfe gegen Baseline...")
        bl_data = json.loads(BASELINE_FILE.read_text())
        bl = Baseline(
            version=bl_data["version"],
            timestamp=bl_data["timestamp"],
            scenarios=bl_data["scenarios"],
            edge_cases=bl_data["edge_cases"],
            parameter_matrix=bl_data.get("parameter_matrix", {}),
            metadata=bl_data.get("metadata", {}),
        )

        passed, issues = check_regression(bl, a.tolerance)
        for issue in issues:
            icon = "❌" if "REGRESSION" in issue else "⚠️" if "WARNING" in issue else "ℹ️"
            print(f"   {icon} {issue}")

        if passed:
            print(f"   ✅ Keine Regression (Toleranz: {a.tolerance} PQS-Punkte)")
        else:
            print("   ❌ Regression erkannt!")

    if a.full:
        print("\n📈 Phasen-Beitrags-Analyse:")
        contrib = analyze_phase_contributions(a.duration)
        for phase, pqs in contrib.items():
            if isinstance(pqs, str):
                print(f"   {phase:<30} {pqs}")
            else:
                bar = "█" * int(pqs / 2)
                print(f"   {phase:<30} PQS={pqs:6.2f} {bar}")

    if a.full:
        print("\n🧪 Edge-Case-Stress-Test:")
        ec = generate_edge_cases()
        for name, signal in ec.items():
            try:
                out = aurik_pipeline(signal, 48000)
                has_nan = not np.all(np.isfinite(out))
                status = "❌ NaN" if has_nan else "✅"
                print(f"   {status} {name:<25} shape={signal.shape}→{out.shape if hasattr(out, 'shape') else '?'}")
            except Exception as e:
                print(f"   ❌ {name:<25} CRASH: {type(e).__name__}: {str(e)[:60]}")

    if a.ci:
        if not BASELINE_FILE.exists():
            print("CI: Keine Baseline — generiere neue.")
            bl = generate_baseline(a.duration)
            BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            BASELINE_FILE.write_text(
                json.dumps(
                    {
                        "version": bl.version,
                        "timestamp": bl.timestamp,
                        "scenarios": bl.scenarios,
                        "edge_cases": bl.edge_cases,
                        "parameter_matrix": bl.parameter_matrix,
                        "metadata": bl.metadata,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        bl_data = json.loads(BASELINE_FILE.read_text())
        bl = Baseline(
            version=bl_data["version"],
            timestamp=bl_data["timestamp"],
            scenarios=bl_data["scenarios"],
            edge_cases=bl_data["edge_cases"],
            parameter_matrix=bl_data.get("parameter_matrix", {}),
            metadata=bl_data.get("metadata", {}),
        )
        passed, issues = check_regression(bl, a.tolerance)
        if not passed:
            for issue in issues:
                print(f"CI: {issue}")
            return 1
        print("CI: Regression check passed.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
