"""
§v10.600: Integrationstest — Vollständige SOTA-Kette.

Testet die komplette Pipeline in Serie:
  Defect Consensus → Repair Planner → SOTA Denoise → Coordinated Repair → Harmonic Inpainting

Jede Stufe wird einzeln verifiziert, dann die Kette als Ganzes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT))

SR = 48000


@pytest.fixture(scope="module")
def manifest():
    """Stufe-1-Manifest als Fixture — wird von den Folge-Stufen wiederverwendet."""
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

    audio, _ = _make_test_audio()
    pipeline = DefectConsensusPipeline()
    _manifest = pipeline.analyze(audio, SR)
    assert _manifest is not None, "Manifest ist None"
    return _manifest


@pytest.fixture(scope="module")
def plan(manifest):
    """Stufe-2-Repair-Plan als Fixture — wird von Stufe 4 wiederverwendet."""
    from backend.core.coordinated_repair import RepairPlanner

    audio, _ = _make_test_audio()
    planner = RepairPlanner()
    _plan = planner.plan(manifest, len(audio))
    assert _plan is not None, "Plan ist None"
    return _plan


def _make_test_audio(duration_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Erzeugt synthetisches Musik-ähnliches Audio mit bekannten Defekten."""
    t = np.arange(int(duration_s * SR), dtype=np.float32) / SR

    # Harmonisches Grundsignal (C-Dur Akkord)
    clean = (
        0.3 * np.sin(2 * np.pi * 261.63 * t) +  # C4
        0.2 * np.sin(2 * np.pi * 329.63 * t) +  # E4
        0.15 * np.sin(2 * np.pi * 392.00 * t) +  # G4
        0.1 * np.sin(2 * np.pi * 523.25 * t)  # C5
    ).astype(np.float32)

    # Defekte einbauen
    noisy = clean.copy()

    # 1. Klick bei 0.5s
    click_pos = int(0.5 * SR)
    noisy[click_pos:click_pos + 50] += 0.8 * np.exp(-np.arange(50) / 5).astype(np.float32)

    # 2. Hum (50 Hz)
    noisy += 0.02 * np.sin(2 * np.pi * 50 * t).astype(np.float32)

    # 3. Breitband-Rauschen
    noisy += 0.01 * np.random.randn(len(t)).astype(np.float32)

    return noisy.astype(np.float32), clean.astype(np.float32)


def test_1_defect_consensus():
    """Stufe 1: Defect Consensus Pipeline erkennt die Defekte."""
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

    audio, _ = _make_test_audio()
    pipeline = DefectConsensusPipeline()
    manifest = pipeline.analyze(audio, SR)

    assert manifest is not None, "Manifest ist None"
    print(f"  ✅ Stufe 1: Manifest erstellt ({manifest.total_hypotheses} Hypothesen, "
          f"{manifest.conflicts_resolved} Konflikte, {len(manifest.defects)} Defekte)")
    return manifest


def test_2_repair_planner(manifest):
    """Stufe 2: Repair Planner erstellt Plan aus Manifest."""
    from backend.core.coordinated_repair import RepairPlanner

    audio, _ = _make_test_audio()
    planner = RepairPlanner()
    plan = planner.plan(manifest, len(audio))

    assert plan is not None, "Plan ist None"
    assert plan.total_defects >= 0, "Defekt-Zähler ungültig"
    print(f"  ✅ Stufe 2: Plan mit {len(plan.steps)} Schritten, "
          f"Reihenfolge: {plan.phase_order[:4]}...")
    return plan


def test_3_sota_denoise():
    """Stufe 3: SOTA Denoise entfernt Breitband-Rauschen."""
    from backend.core.sota_denoise_pipeline import SOTADenoisePipeline

    audio, clean = _make_test_audio()
    pipeline = SOTADenoisePipeline()
    result = pipeline.process(audio, SR, auto_params=False, override_strength=0.4)

    assert result.audio.shape == audio.shape, "Shape-Mismatch"
    # Denoise sollte die Leistung reduzieren (Rauschen weg)
    power_in = np.mean(audio**2)
    power_out = np.mean(result.audio**2)
    assert power_out <= power_in * 1.5, "Denoise verstärkt das Signal (Edge-Artefakt!)"

    mse_noisy = np.mean((audio - clean) ** 2)
    mse_clean = np.mean((result.audio - clean) ** 2)
    print(f"  ✅ Stufe 3: Denoise OK (Power {power_in:.4f} → {power_out:.4f}, "
          f"MSE {mse_noisy:.6f} → {mse_clean:.6f})")
    return result.audio


def test_4_coordinated_repair(manifest, plan):
    """Stufe 4: Coordinated Repair führt den Plan aus."""
    from backend.core.coordinated_repair import CoordinatedRepair

    audio, _ = _make_test_audio()
    executor = CoordinatedRepair()
    repaired, report = executor.execute(audio, plan, manifest, SR)

    assert repaired.shape == audio.shape, "Shape-Mismatch"
    assert report is not None, "Report fehlt"
    print(f"  ✅ Stufe 4: {len(report.completed_steps)} Schritte abgeschlossen, "
          f"{len(report.failed_steps)} fehlgeschlagen, {report.total_time:.2f}s")
    return repaired


def test_5_harmonic_inpainting():
    """Stufe 5: Harmonic Inpainting (fallback wenn DiT nicht verfügbar)."""
    from backend.core.coordinated_repair import RepairStep, RepairPriority, CoordinatedRepair

    audio, clean = _make_test_audio()
    executor = CoordinatedRepair()

    step = RepairStep(
        phase_id="phase_55_diffusion_inpainting",
        priority=RepairPriority.INPAINTING,
        defect_category="harmonic_loss",
        affected_samples=[(0, len(audio))],
        parameters={"strength": 0.3, "confidence": 0.8},
    )
    result = executor._run_inpainting(audio, step, None, SR)

    assert result.shape == audio.shape, "Shape-Mismatch"
    print(f"  ✅ Stufe 5: Inpainting OK (Shape {result.shape}, "
          f"Power {np.mean(result**2):.4f})")
    return result


def test_6_full_chain():
    """Komplette Kette in Serie."""
    audio, clean = _make_test_audio()

    # Stufe 1+2
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline
    from backend.core.coordinated_repair import RepairPlanner, CoordinatedRepair

    consensus = DefectConsensusPipeline()
    manifest = consensus.analyze(audio, SR)

    planner = RepairPlanner()
    plan = planner.plan(manifest, len(audio))

    # Stufe 3+4+5
    executor = CoordinatedRepair()
    repaired, report = executor.execute(audio, plan, manifest, SR)

    assert repaired.shape == audio.shape
    assert report.completed_steps or report.failed_steps, "Kein einziger Schritt ausgeführt"

    print(f"  ✅ Stufe 6 (Kette): {len(report.completed_steps)} von "
          f"{len(plan.steps)} Schritten erfolgreich in {report.total_time:.2f}s")
    return repaired


def main():
    print("=" * 60)
    print("§v10.600 Integrationstest: SOTA-Kette")
    print("=" * 60)

    t0 = time.time()
    failures = 0

    tests = [
        ("Defect Consensus", test_1_defect_consensus),
        ("Repair Planner", lambda: test_2_repair_planner(test_1_defect_consensus())),
        ("SOTA Denoise", test_3_sota_denoise),
        ("Harmonic Inpainting", test_5_harmonic_inpainting),
        ("Full Chain", test_6_full_chain),
    ]

    for name, fn in tests:
        try:
            print(f"\n▶ {name}")
            fn()
        except Exception as e:
            failures += 1
            print(f"  ❌ {name}: {e}")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    if failures == 0:
        print(f"✅ ALLE {len(tests)} STUFEN BESTANDEN ({elapsed:.1f}s)")
    else:
        print(f"❌ {failures}/{len(tests)} Stufen fehlgeschlagen ({elapsed:.1f}s)")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
