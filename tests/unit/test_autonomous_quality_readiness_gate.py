from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit.autonomous_quality_readiness_gate import (
    ReadinessThresholds,
    build_autonomous_quality_readiness,
    generate_autonomous_quality_readiness_report,
)


def _write_manifest(root: Path, material: str, *, file_name: str, vocal: bool = True) -> Path:
    material_dir = root / material
    audio_path = material_dir / "damaged" / file_name
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"RIFF0000WAVE")
    manifest = material_dir / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "corpus_version: 1.0.0",
                f"material: {material}",
                "entries:",
                f"- file: damaged/{file_name}",
                "  duration_s: 1.0",
                "  sample_rate: 48000",
                f"  material: {material}",
                "  era_year: 1970",
                "  genre: vocal",
                "  condition: damaged",
                f"  vocal: {'true' if vocal else 'false'}",
                "  license: CC0",
                "  source_attribution: unit-test synthetic fixture",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def _write_benchmark(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "benchmark_20260804_120000.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-04T12:00:00+00:00",
                "files_total": 2,
                "files_ok": 2,
                "files_failed": 0,
                "corpus_files_total": 2,
                "benchmark_files_selected": 2,
                "coverage_ratio": 1.0,
                "aurik_pipeline_ok": 2,
                "aurik_pipeline_fallbacks": 0,
                "aggregate": {"avg_snr_db": 31.2, "avg_si_sdr_db": 24.5},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_calibration(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration_stage": 2,
                "confidence": 0.91,
                "calibrated_weights": {"mert_similarity": 0.5, "artifact_freedom": 0.5},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_autonomous_quality_readiness_ready_when_all_evidence_present(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_manifest(corpus, "vinyl", file_name="vinyl_case.wav", vocal=True)
    _write_manifest(corpus, "cassette", file_name="cassette_case.wav", vocal=True)
    benchmark_dir = tmp_path / "benchmarks" / "quality" / "results"
    calibration = _write_calibration(tmp_path / "calibration" / "mushra_calibration.json")
    _write_benchmark(benchmark_dir)

    report = build_autonomous_quality_readiness(
        corpus_root=corpus,
        benchmark_results_dir=benchmark_dir,
        calibration_artifact=calibration,
        thresholds=ReadinessThresholds(min_entries=2, min_materials=2, min_vocal_entries=1),
        require_benchmark=True,
        require_panel_calibration=True,
    )

    assert report["status"] == "ready"
    assert report["autonomy"]["manual_action_required"] is False
    assert report["autonomy"]["allowed_user_decisions"] == ["mode_selection"]
    assert report["corpus"]["entries_total"] == 2
    assert report["quality_benchmark"]["ready"] is True
    assert report["mushra_calibration"]["ready"] is True


@pytest.mark.unit
def test_autonomous_quality_readiness_rejects_benchmark_with_pipeline_fallbacks(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_manifest(corpus, "vinyl", file_name="vinyl_case.wav", vocal=True)
    benchmark_dir = tmp_path / "benchmarks" / "quality" / "results"
    calibration = _write_calibration(tmp_path / "calibration" / "mushra_calibration.json")
    benchmark_path = _write_benchmark(benchmark_dir)
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    data["aurik_pipeline_ok"] = 1
    data["aurik_pipeline_fallbacks"] = 1
    benchmark_path.write_text(json.dumps(data), encoding="utf-8")

    report = build_autonomous_quality_readiness(
        corpus_root=corpus,
        benchmark_results_dir=benchmark_dir,
        calibration_artifact=calibration,
        thresholds=ReadinessThresholds(min_entries=1, min_materials=1, min_vocal_entries=1),
        require_benchmark=True,
        require_panel_calibration=True,
    )

    assert report["status"] == "blocked"
    assert "quality_benchmark_missing_or_failed" in report["blocking_reasons"]
    assert report["quality_benchmark"]["aurik_pipeline_fallbacks"] == 1


@pytest.mark.unit
def test_autonomous_quality_readiness_rejects_legacy_benchmark_without_pipeline_telemetry(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_manifest(corpus, "vinyl", file_name="vinyl_case.wav", vocal=True)
    benchmark_dir = tmp_path / "benchmarks" / "quality" / "results"
    calibration = _write_calibration(tmp_path / "calibration" / "mushra_calibration.json")
    benchmark_path = _write_benchmark(benchmark_dir)
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    data.pop("aurik_pipeline_ok")
    data.pop("aurik_pipeline_fallbacks")
    benchmark_path.write_text(json.dumps(data), encoding="utf-8")

    report = build_autonomous_quality_readiness(
        corpus_root=corpus,
        benchmark_results_dir=benchmark_dir,
        calibration_artifact=calibration,
        thresholds=ReadinessThresholds(min_entries=1, min_materials=1, min_vocal_entries=1),
    )

    assert report["status"] == "attention"
    assert "quality_benchmark_missing_pipeline_telemetry" in report["advisory_reasons"]
    assert report["quality_benchmark"]["aurik_pipeline_ok"] == 0
    assert any("current schema" in action for action in report["next_automated_actions"])


@pytest.mark.unit
def test_autonomous_quality_readiness_advisory_when_optional_evidence_missing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_manifest(corpus, "vinyl", file_name="vinyl_case.wav", vocal=True)

    report = build_autonomous_quality_readiness(
        corpus_root=corpus,
        benchmark_results_dir=tmp_path / "missing-results",
        calibration_artifact=tmp_path / "missing" / "mushra_calibration.json",
        thresholds=ReadinessThresholds(min_entries=1, min_materials=1, min_vocal_entries=1),
    )

    assert report["status"] == "attention"
    assert report["blocking_reasons"] == []
    assert "quality_benchmark_missing_pipeline_telemetry" in report["advisory_reasons"]
    assert "mushra_panel_calibration_missing" in report["advisory_reasons"]


@pytest.mark.unit
def test_autonomous_quality_readiness_blocks_missing_corpus_file(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    manifest = _write_manifest(corpus, "vinyl", file_name="missing.wav", vocal=True)
    (manifest.parent / "damaged" / "missing.wav").unlink()

    report = build_autonomous_quality_readiness(
        corpus_root=corpus,
        benchmark_results_dir=tmp_path / "missing-results",
        thresholds=ReadinessThresholds(min_entries=1, min_materials=1, min_vocal_entries=1),
    )

    assert report["status"] == "blocked"
    assert "corpus_not_ready" in report["blocking_reasons"]
    assert report["corpus"]["files_missing_count"] == 1


@pytest.mark.unit
def test_generate_autonomous_quality_readiness_report_writes_json(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_manifest(corpus, "vinyl", file_name="vinyl_case.wav", vocal=True)
    output_json = tmp_path / "audit" / "autonomous_quality_readiness.json"

    report = generate_autonomous_quality_readiness_report(
        corpus_root=corpus,
        benchmark_results_dir=tmp_path / "missing-results",
        output_json=output_json,
        thresholds=ReadinessThresholds(min_entries=1, min_materials=1, min_vocal_entries=1),
    )

    assert output_json.exists()
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["status"] == report["status"]
    assert loaded["autonomy"]["runtime_user_prompts"] is False
