from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.core.real_audio_execution_golden_gate import (
    ExecutionCaseResult,
    ExecutionGateThresholds,
    _evaluate_execution_gate,
    _measure_manifest_vqi,
    run_real_audio_execution_golden_gate,
)
from backend.core.real_audio_strategy_golden_gate import StrategyCaseResult


def _case(**overrides):
    base = {
        "case_id": "case",
        "required_phases": ("phase_01_click_removal",),
        "planned_phases": ("phase_01_click_removal",),
        "phases_executed": ("phase_01_click_removal",),
        "phases_skipped": (),
        "missing_required_executions": (),
        "forbidden_executed": (),
        "phase_delta_phases": ("phase_01_click_removal",),
        "missing_phase_deltas": (),
        "artifact_freedom": 0.99,
        "artifact_contract_passed": True,
        "hpi": 0.55,
        "hpi_contract_passed": True,
        "vqi": None,
        "vocal_required": False,
        "vocal_contract_passed": True,
        "export_contract_passed": True,
        "export_strategy": "success",
        "export_blocked": False,
        "degradation_status": "ok",
        "fail_reasons": (),
        "runtime_seconds": 1.0,
        "duration_seconds": 2.0,
        "metadata": {},
    }
    base.update(overrides)
    return ExecutionCaseResult(**base)  # type: ignore[arg-type]


@pytest.mark.unit
def test_execution_gate_aggregates_contract_failures() -> None:
    gate = _evaluate_execution_gate(
        [
            _case(
                missing_required_executions=("phase_01_click_removal",),
                missing_phase_deltas=("phase_01_click_removal",),
                artifact_contract_passed=False,
                hpi_contract_passed=False,
                export_contract_passed=False,
            )
        ],
        ExecutionGateThresholds(),
    )

    assert gate.passed is False
    assert gate.phase_execution_recall == 0.0
    assert gate.phase_delta_coverage == 0.0
    assert gate.artifact_contract_rate == 0.0
    assert gate.hpi_contract_rate == 0.0
    assert gate.export_contract_rate == 0.0
    assert gate.fail_reasons


def test_execution_gate_runtime_duration_floor_prevents_short_clip_overhead_bias() -> None:
    gate = _evaluate_execution_gate(
        [
            _case(
                runtime_seconds=80.0,
                duration_seconds=1.0,
            )
        ],
        ExecutionGateThresholds(max_runtime_factor=25.0, runtime_duration_floor_seconds=4.0),
    )

    assert gate.runtime_factor == pytest.approx(20.0, abs=1e-9)
    assert gate.passed is True


def test_real_audio_execution_gate_with_mocked_uv3(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    audio_path = repo_root / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    manifest = repo_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "target_sample_rate": 48000,
                "execution_thresholds": {
                    "min_phase_execution_recall": 1.0,
                    "min_phase_delta_coverage": 1.0,
                    "min_artifact_contract_rate": 1.0,
                    "min_hpi_contract_rate": 1.0,
                    "min_vocal_contract_rate": 1.0,
                    "min_export_contract_rate": 1.0,
                    "max_runtime_factor": 10.0,
                },
                "cases": [
                    {
                        "case_id": "mock_case",
                        "path": "audio.wav",
                        "material_type": "vinyl",
                        "requires_vocal_gate": True,
                        "description": "Choir restoration case",
                        "required_phases": ["phase_01_click_removal"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    strategy = StrategyCaseResult(
        case_id="mock_case",
        accepted_causes=(),
        cause_top_k=3,
        primary_cause="vinyl_crackle",
        top_causes=("vinyl_crackle",),
        cause_hit=True,
        required_phases=("phase_01_click_removal",),
        missing_required_phases=(),
        forbidden_phases=(),
        forbidden_present=(),
        ordered_before=(),
        order_violations=(),
        reasoner_phases=("phase_01_click_removal",),
        mapper_phases=(),
        combined_phases=("phase_01_click_removal",),
        runtime_seconds=0.1,
        duration_seconds=2.0,
        metadata={},
    )
    monkeypatch.setattr("backend.core.real_audio_execution_golden_gate._scan_strategy_case", lambda *args: strategy)
    monkeypatch.setattr(
        "backend.core.real_audio_execution_golden_gate._load_audio_for_execution",
        lambda *args, **kwargs: (np.zeros(2048, dtype=np.float32), 48000, 2.0),
    )

    class FakeRestorer:
        def __init__(self, config):
            self.config = config

        def restore(self, audio, sample_rate, **kwargs):
            assert kwargs["vocal_material_prior"] is True
            assert kwargs["multi_singer_prior"] is True
            return SimpleNamespace(
                audio=np.zeros(2048, dtype=np.float32),
                phases_executed=["phase_01_click_removal"],
                phases_skipped=[],
                quality_estimate=0.9,
                metadata={
                    "fail_reasons": [],
                    "degradation_status": "ok",
                    "phase_deltas": {"phase_01_click_removal": {"delta": {}}},
                    "artifact_freedom": {"score": 0.99, "passed": True},
                    "holistic_perceptual_gate": {"hpi": 0.55, "passed": True},
                    "vqi": 0.82,
                },
            )

    monkeypatch.setattr("backend.core.real_audio_execution_golden_gate.UnifiedRestorerV3", FakeRestorer)

    report = run_real_audio_execution_golden_gate(
        manifest_path=manifest,
        repo_root=repo_root,
        output_dir=tmp_path / "exports",
    )

    assert report.scanned_cases == 1
    assert report.gate.passed is True
    assert report.gate.phase_execution_recall == 1.0
    assert report.gate.export_contract_rate == 1.0


def test_measure_manifest_vqi_passes_era_profile_for_historical_case(monkeypatch) -> None:
    captured = {"era_profile": None}

    def _fake_get_era_vocal_profile(decade):
        return {"era_decade": decade, "formant_tolerance_db": 2.4}

    def _fake_compute_vqi(**kwargs):
        captured["era_profile"] = kwargs.get("era_profile")
        return {"vqi": 0.8}

    def _fake_get_vqi_material_floor(material, is_studio_2026=False):
        return 0.72

    monkeypatch.setattr(
        "backend.core.musical_goals.era_vocal_profile.get_era_vocal_profile",
        _fake_get_era_vocal_profile,
    )
    monkeypatch.setattr(
        "backend.core.musical_goals.vocal_quality_index.compute_vqi",
        _fake_compute_vqi,
    )
    monkeypatch.setattr(
        "backend.core.musical_goals.vocal_quality_index.get_vqi_material_floor",
        _fake_get_vqi_material_floor,
    )

    raw_case = {
        "case_id": "hist_vocal",
        "description": "vocal restoration case",
        "material_type": "shellac",
        "year": 1932,
    }

    # Die Funktion importiert intern; monkeypatching der Modulattribute mit raising=False
    # reicht als Guard, dass era_profile bei compute_vqi ankommt.
    vqi, floor, source = _measure_manifest_vqi(
        np.zeros(4096, dtype=np.float32),
        np.zeros(4096, dtype=np.float32),
        48000,
        raw_case=raw_case,
    )

    assert source == "manifest_vqi"
    assert vqi == 0.8
    assert floor == pytest.approx(0.72, abs=1e-6)
    assert isinstance(captured["era_profile"], dict)
    assert captured["era_profile"].get("era_decade") == 1930
