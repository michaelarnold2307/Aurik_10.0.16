from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks.quality.auto_quality_benchmark import AutoQualityBenchmark


@pytest.mark.unit
def test_auto_quality_benchmark_empty_corpus_returns_schema_stable_report(tmp_path: Path) -> None:
    corpus = tmp_path / "empty-corpus"
    corpus.mkdir()

    report = AutoQualityBenchmark(corpus).run_benchmark()

    assert report["corpus"] == str(corpus)
    assert report["files_total"] == 0
    assert report["files_ok"] == 0
    assert report["files_failed"] == 0
    assert report["corpus_files_total"] == 0
    assert report["benchmark_files_selected"] == 0
    assert report["coverage_ratio"] == 0.0
    assert report["aurik_pipeline_ok"] == 0
    assert report["aurik_pipeline_fallbacks"] == 0
    assert report["aggregate"]["avg_snr_db"] == 0.0
    assert report["results"] == []


@pytest.mark.unit
def test_auto_quality_benchmark_propagates_engine_confidence_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRestorer:
        def restore(self, audio: np.ndarray, sr: int, mode: str) -> SimpleNamespace:
            assert sr == 48_000
            assert mode == "restoration"
            return SimpleNamespace(
                audio=audio * 0.5,
                metadata={
                    "degradation_status": "ok",
                    "fail_reason": "",
                    "material_used": "vinyl",
                    "material_detected": "vinyl",
                    "material_confidence": 0.91,
                    "era_decade": 1970,
                    "era_confidence": 0.88,
                    "recording_year": 1977,
                    "year_source": "tag:date",
                    "genre_label": "pop",
                    "genre_confidence": 0.77,
                    "pipeline_confidence": {"confidence": 0.93},
                    "mushra_score": 86.5,
                    "hpi_score": 0.89,
                    "best_possible_reached": True,
                },
            )

    fake_module = ModuleType("backend.core.unified_restorer_v3")
    fake_module.UnifiedRestorerV3 = FakeRestorer  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "backend.core.unified_restorer_v3", fake_module)

    restored, engine = AutoQualityBenchmark(tmp_path)._run_aurik_auto(np.ones(16, dtype=np.float32), 48_000)

    assert np.allclose(restored, np.full(16, 0.5, dtype=np.float32))
    assert engine == {
        "status": "aurik_pipeline",
        "fallback_used": False,
        "error": "",
        "degradation_status": "ok",
        "fail_reason": "",
        "material_used": "vinyl",
        "material_detected": "vinyl",
        "material_confidence": 0.91,
        "era_decade": 1970,
        "era_confidence": 0.88,
        "recording_year": 1977,
        "year_source": "tag:date",
        "genre_label": "pop",
        "genre_confidence": 0.77,
        "pipeline_confidence": 0.93,
        "mushra_score": 86.5,
        "hpi_score": 0.89,
        "best_possible_reached": True,
    }
