"""§v10.995: Das Evaluations-System testet sich selbst — Ehrlichkeit ist Pflicht.

Schlüsselprinzip: Auch VERSCHLECHTERUNGEN müssen als solche berichtet werden.
Diese Tests pinnen genau das fest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.core.evaluation_system import (
    CaseMetrics,
    EvalCase,
    EvalReport,
    EvaluationSystem,
    GateResult,
    ListeningTestExporter,
    compute_objective_metrics,
    discover_corpus_cases,
    gate_competitive_multi,
    gate_regression,
)


def _synthetic_case(seed: int = 0, *, mode: str = "improved") -> EvalCase:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, 48000, endpoint=False, dtype=np.float32)
    clean = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (rng.standard_normal(len(t)) * 0.05).astype(np.float32)
    damaged = clean + noise
    if mode == "improved":
        restored = clean
    elif mode == "degraded":
        restored = damaged + noise * 2.0
    else:
        restored = damaged.copy()
    return EvalCase(
        case_id=f"syn_{seed}",
        material="synthetic",
        damaged=damaged,
        clean=clean,
        restored=restored,
        sample_rate=48000,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Ehrlichkeit: alle drei Verdicts werden korrekt berichtet
# ═══════════════════════════════════════════════════════════════════════════════


def test_improved_case_reported_improved():
    metrics = compute_objective_metrics(_synthetic_case(0, mode="improved"))
    assert metrics.verdict == "improved"
    assert metrics.snr_delta_db > 10
    assert metrics.mse_reduction_pct > 90


def test_degraded_case_reported_degraded_not_filtered():
    """Ehrlichkeits-Regel: Verschlechterung wird NICHT weggefiltert."""
    metrics = compute_objective_metrics(_synthetic_case(1, mode="degraded"))
    assert metrics.verdict == "degraded"
    assert metrics.snr_delta_db < 0


def test_neutral_case_reported_neutral():
    metrics = compute_objective_metrics(_synthetic_case(2, mode="neutral"))
    assert metrics.verdict == "neutral"


def test_metrics_defensive_without_restored():
    case = _synthetic_case(3, mode="improved")
    case.restored = None
    metrics = compute_objective_metrics(case)
    assert metrics.verdict == "neutral"
    assert metrics.snr_delta_db == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Gates — die EINEN Entscheidungsregeln
# ═══════════════════════════════════════════════════════════════════════════════


def test_regression_gate_fails_on_any_degraded_case():
    improved = [compute_objective_metrics(_synthetic_case(i, mode="improved")) for i in range(3)]
    degraded = compute_objective_metrics(_synthetic_case(9, mode="degraded"))
    gate = gate_regression(improved + [degraded])
    assert not gate.passed
    assert "syn_9" in gate.details["degraded_cases"]


def test_regression_gate_passes_when_all_improved():
    improved = [compute_objective_metrics(_synthetic_case(i, mode="improved")) for i in range(3)]
    assert gate_regression(improved).passed


def test_competitive_gate_seven_of_ten():
    scenarios = [(f"s{i}", 75.0 if i < 7 else 60.0, 71.0) for i in range(10)]
    gate = gate_competitive_multi(scenarios)
    assert gate.passed
    assert gate.details["won"] == 7


def test_competitive_gate_fails_below_seven():
    scenarios = [(f"s{i}", 75.0 if i < 6 else 60.0, 71.0) for i in range(10)]
    gate = gate_competitive_multi(scenarios)
    assert not gate.passed
    assert gate.details["won"] == 6


# ═══════════════════════════════════════════════════════════════════════════════
# Report-Schema — EIN Format für alles, JSON-Roundtrip
# ═══════════════════════════════════════════════════════════════════════════════


def test_report_json_roundtrip(tmp_path):
    system = EvaluationSystem()
    report = system.run_objective([_synthetic_case(i, mode="improved") for i in range(3)])
    assert report.verdict == "PASS"
    path = report.save(tmp_path / "eval.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["case_count"] == 3
    assert data["aggregate"]["degraded"] == 0
    assert data["gates"][0]["name"] == "regression"
    loaded = EvalReport.load(path)
    assert loaded.verdict == "PASS"
    assert len(loaded.cases) == 3


def test_report_verdict_fails_with_degraded_case(tmp_path):
    system = EvaluationSystem()
    report = system.run_objective(
        [_synthetic_case(i, mode="improved") for i in range(2)] + [_synthetic_case(9, mode="degraded")]
    )
    assert report.verdict == "FAIL"


# ═══════════════════════════════════════════════════════════════════════════════
# Kontrollierter Hörvergleich — Doppelblind-fähig
# ═══════════════════════════════════════════════════════════════════════════════


def test_listening_export_writes_pairs_key_and_scoresheet(tmp_path):
    exporter = ListeningTestExporter(tmp_path / "lt")
    case = _synthetic_case(0, mode="improved")
    exporter.export_pair(case.case_id, case.restored, case.clean, case.sample_rate)
    exporter.export_pair("syn_1", _synthetic_case(1).restored, _synthetic_case(1).clean, 48000)

    sheet = exporter.write_scoresheet()
    key = exporter.write_key()

    assert (tmp_path / "lt" / "syn_0" / "A.wav").exists()
    assert (tmp_path / "lt" / "syn_0" / "B.wav").exists()
    assert "syn_0" in sheet.read_text(encoding="utf-8")
    assert "syn_1" in sheet.read_text(encoding="utf-8")
    key_data = json.loads(key.read_text(encoding="utf-8"))
    # Jeder Fall hat genau eine restored- und eine reference-Datei (randomisiert)
    for case_id in ("syn_0", "syn_1"):
        mapping = key_data[case_id]
        assert set(mapping.values()) == {"restored", "reference"}


# ═══════════════════════════════════════════════════════════════════════════════
# Korpus-Discovery — echte Aufnahmen korrekt gepaart
# ═══════════════════════════════════════════════════════════════════════════════


def test_corpus_discovery_pairs_damaged_clean_restored(tmp_path):
    import wave

    def _wav(path: Path, seconds: float = 0.05) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = int(48000 * seconds)
        pcm = (np.sin(np.linspace(0, 10, n)) * 8000).astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm.tobytes())

    _wav(tmp_path / "vinyl" / "damaged" / "a.wav")
    _wav(tmp_path / "vinyl" / "clean" / "a.wav")
    _wav(tmp_path / "vinyl" / "restored" / "a.wav")
    _wav(tmp_path / "vinyl" / "damaged" / "b.wav")  # ohne clean-Pendant → ignoriert

    cases = discover_corpus_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0]["case_id"] == "vinyl_a"
    assert cases[0]["restored_path"] is not None

    limited = discover_corpus_cases(tmp_path, limit=1)
    assert len(limited) == 1


def test_corpus_discovery_prefix_convention(tmp_path):
    """Reale Korpus-Konvention: <song>_<decade>_<defekt> ↔ <song>_<decade>_clean."""
    import wave

    def _wav(path: Path, seconds: float = 0.05) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        n = int(48000 * seconds)
        pcm = (np.sin(np.linspace(0, 10, n)) * 8000).astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm.tobytes())

    _wav(tmp_path / "vinyl" / "damaged" / "vinyl_blues_1950s_crackle.wav")
    _wav(tmp_path / "vinyl" / "clean" / "vinyl_blues_1950s_clean.wav")
    _wav(tmp_path / "vinyl" / "restored" / "vinyl_blues_1950s_crackle.wav")
    # anderer Defekt desselben Songs → paart auf dieselbe clean-Referenz
    _wav(tmp_path / "vinyl" / "damaged" / "vinyl_blues_1950s_hiss.wav")

    cases = discover_corpus_cases(tmp_path)
    assert len(cases) == 2
    for c in cases:
        assert "clean.wav" in c["clean_path"]
    assert cases[0]["restored_path"] is not None
    # hiss-Variante hat kein restored-Pendant
    assert cases[1]["restored_path"] is None
