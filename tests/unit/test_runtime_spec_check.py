from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.runtime_spec_check import run_check


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _get_mode_check(report: dict) -> dict:
    for chk in report["checks"]:
        if chk["id"] == "mode_contract":
            return chk  # type: ignore[no-any-return]
    raise AssertionError("mode_contract-Check nicht gefunden")


@pytest.mark.unit
def test_mode_contract_accepts_internal_studio2026(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                "run context mode=studio2026",
                "AurikDenker.denke() abgeschlossen",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    mode_chk = _get_mode_check(report)

    assert mode_chk["passed"] is True
    assert "vorhanden" in mode_chk["evidence"]

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["checks"]


def test_mode_contract_accepts_ui_studio_2026(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                'payload: {"mode":"STUDIO_2026"}',
                "AurikDenker.denke() abgeschlossen",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    mode_chk = _get_mode_check(report)

    assert mode_chk["passed"] is True


def _get_check(report: dict, check_id: str) -> dict:
    for chk in report["checks"]:
        if chk["id"] == check_id:
            return chk  # type: ignore[no-any-return]
    raise AssertionError(f"{check_id}-Check nicht gefunden")


def test_phase12_effective_check_fails_on_low_confidence_skip(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                "run context mode=restoration",
                "phase_12_wow_flutter_fix startet",
                "Phase 12: Pitch-Konfidenz zu niedrig (0.000 < 0.25) — keine Korrektur angewandt",
                "AurikDenker.denke() abgeschlossen",
                "§7.5a Phase-DAG FINAL: keine Reihenfolge-Verletzungen",
                "§2.44 HPI(restoration)=0.91",
                "§2.49 Final artifact_freedom=0.960",
                "joy_runtime_index",
                "auto_improvement_recommendations",
                "cluster_policy",
                "Verwende gecachten DefectScan",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    chk = _get_check(report, "phase12_effective_when_started")

    assert chk["passed"] is False
    assert "low-confidence Skip" in chk["evidence"]


def test_phase12_effective_check_passes_when_phase12_not_started(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                "run context mode=restoration",
                "AurikDenker.denke() abgeschlossen",
                "§7.5a Phase-DAG FINAL: keine Reihenfolge-Verletzungen",
                "§2.44 HPI(restoration)=0.91",
                "§2.49 Final artifact_freedom=0.960",
                "joy_runtime_index",
                "auto_improvement_recommendations",
                "cluster_policy",
                "Verwende gecachten DefectScan",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    chk = _get_check(report, "phase12_effective_when_started")

    assert chk["passed"] is True


def test_material_specificity_guard_fails_on_cassette_to_tape_downgrade(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                "run context mode=restoration",
                "🔍 MediumDetector: Material=cassette Konfidenz=0.40 Quelle=cached",
                "🎛️ SongCalibration: material=tape era=1970 tier=fair global=0.983 denoise=0.971 conf=0.29",
                "AurikDenker.denke() abgeschlossen",
                "§7.5a Phase-DAG FINAL: keine Reihenfolge-Verletzungen",
                "§2.44 HPI(restoration)=0.91",
                "§2.49 Final artifact_freedom=0.960",
                "joy_runtime_index",
                "auto_improvement_recommendations",
                "cluster_policy",
                "Verwende gecachten DefectScan",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    chk = _get_check(report, "material_specificity_guard")

    assert chk["passed"] is False
    assert "cassette" in chk["evidence"]


def test_material_specificity_guard_passes_without_downgrade(tmp_path: Path) -> None:
    backend = tmp_path / "backend.log"
    frontend = tmp_path / "frontend.log"
    output = tmp_path / "report.json"

    _write(
        backend,
        "\n".join(
            [
                "AurikDenker.denke() gestartet",
                "run context mode=restoration",
                "🔍 MediumDetector: Material=cassette Konfidenz=0.40 Quelle=cached",
                "🎛️ SongCalibration: material=cassette era=1970 tier=fair global=0.983 denoise=0.971 conf=0.29",
                "AurikDenker.denke() abgeschlossen",
                "§7.5a Phase-DAG FINAL: keine Reihenfolge-Verletzungen",
                "§2.44 HPI(restoration)=0.91",
                "§2.49 Final artifact_freedom=0.960",
                "joy_runtime_index",
                "auto_improvement_recommendations",
                "cluster_policy",
                "Verwende gecachten DefectScan",
            ]
        ),
    )
    _write(frontend, "")

    report = run_check(backend, frontend, output)
    chk = _get_check(report, "material_specificity_guard")

    assert chk["passed"] is True
