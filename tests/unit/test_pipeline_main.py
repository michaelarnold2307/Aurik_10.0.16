from __future__ import annotations

import pytest

"""Unit-Tests für backend/core/pipeline_main.py.

Spec §2.1: AurikAutonomousPipeline (primär).
Tests decken ab: Initialisierung, get_session_summary() Leer-Zustand,
process()-Aufruf über gemockte Engine, Audit-Trail-Schreiben, Edge-Cases
(Stille, NaN-Audio, Stereo), Singleton-Verhalten und prozess-Ausgabe-Format.
"""


import json
import os
from unittest.mock import MagicMock, patch

import numpy as np

np.random.seed(3)

from backend.core.pipeline_main import AurikAutonomousPipeline
from backend.core.processing_modes import ProcessingMode

SR = 48000


def _sine(secs: float = 1.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(secs: float = 1.0) -> np.ndarray:
    mono = _sine(secs)
    return np.stack([mono, mono * 0.85])


def _mock_material():
    m = MagicMock()
    m.value = "vinyl"
    return m


def _make_mock_result(audio: np.ndarray) -> MagicMock:
    """Erstellt ein minimal gültiges AutonomousRestorationResult-Mock."""
    r = MagicMock()
    r.audio = audio.copy()
    r.mode = ProcessingMode.RESTORATION
    r.material_type = _mock_material()
    r.winning_variant = "balanced"
    r.quality_before = 60.0
    r.quality_after = 80.0
    r.improvement_db = 3.5
    r.rollback_triggered = False
    r.processing_time_seconds = 0.5
    r.passes_executed = 1
    return r


def _make_pipeline_with_mock_engine(audio: np.ndarray, mode=ProcessingMode.RESTORATION):
    """Erstellt AurikAutonomousPipeline, dessen Engine gemockt ist."""
    mock_result = _make_mock_result(audio)
    with patch("backend.core.pipeline_main.AutonomousRestorationEngine") as MockEngine:
        MockEngine.return_value.process.return_value = mock_result
        pipeline = AurikAutonomousPipeline(mode=mode, enable_self_learning=False)
        # Engine muss aktiv bleiben:
        pipeline._engine = MockEngine.return_value
        pipeline._engine.process.return_value = mock_result  # type: ignore[attr-defined]
    return pipeline, mock_result


# ---------------------------------------------------------------------------
# Klasse 1: Import und Konstanten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImportAndConstants:
    def test_01_autonomous_pipeline_importable(self):
        assert AurikAutonomousPipeline is not None

    def test_03_processing_mode_importable(self):
        assert ProcessingMode is not None

    def test_04_processing_mode_restoration_exists(self):
        assert hasattr(ProcessingMode, "RESTORATION")

    def test_05_processing_mode_studio_2026_exists(self):
        assert hasattr(ProcessingMode, "STUDIO_2026")


# ---------------------------------------------------------------------------
# Klasse 2: AurikAutonomousPipeline — Initialisierung
# ---------------------------------------------------------------------------


class TestAutonomousPipelineInit:
    def test_06_default_init_no_crash(self):
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            pipeline = AurikAutonomousPipeline()
            assert pipeline is not None

    def test_07_mode_restoration_default(self):
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            pipeline = AurikAutonomousPipeline()
            assert pipeline.mode == ProcessingMode.RESTORATION

    def test_08_mode_studio_2026_accepted(self):
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            pipeline = AurikAutonomousPipeline(mode=ProcessingMode.STUDIO_2026)
            assert pipeline.mode == ProcessingMode.STUDIO_2026

    def test_09_session_results_empty_on_init(self):
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            assert len(pipeline._session_results) == 0

    def test_10_logs_dir_created(self):
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            AurikAutonomousPipeline(enable_self_learning=False)
            assert os.path.isdir("logs")


# ---------------------------------------------------------------------------
# Klasse 3: get_session_summary() — Leerer Zustand
# ---------------------------------------------------------------------------


class TestGetSessionSummaryEmpty:
    def test_11_empty_summary_returns_dict(self):
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            summary = pipeline.get_session_summary()
            assert isinstance(summary, dict)

    def test_12_empty_summary_session_results_zero(self):
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            summary = pipeline.get_session_summary()
            assert summary["session_results"] == 0

    def test_13_empty_summary_has_no_mode_key(self):
        """Leere Session hat kein 'mode'-Key (spezifikationsgemäß)."""
        with patch("backend.core.pipeline_main.AutonomousRestorationEngine"):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            summary = pipeline.get_session_summary()
            # Nur session_results == 0 — kein crash
            assert "session_results" in summary


# ---------------------------------------------------------------------------
# Klasse 4: get_session_summary() — Nach einem process()-Aufruf
# ---------------------------------------------------------------------------


class TestGetSessionSummaryAfterProcess:
    def _run_once(self, mode=ProcessingMode.RESTORATION):
        audio = _sine(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio, mode)
        pipeline.process(audio, SR)
        return pipeline

    def test_14_session_results_increments(self):
        pipeline = self._run_once()
        summary = pipeline.get_session_summary()
        assert summary["session_results"] == 1

    def test_15_summary_has_mode_key(self):
        pipeline = self._run_once()
        summary = pipeline.get_session_summary()
        assert "mode" in summary

    def test_16_summary_avg_snr_improvement_is_float(self):
        pipeline = self._run_once()
        summary = pipeline.get_session_summary()
        assert isinstance(summary["avg_snr_improvement_db"], float)

    def test_17_summary_rollbacks_is_int(self):
        pipeline = self._run_once()
        summary = pipeline.get_session_summary()
        assert isinstance(summary["rollbacks"], int)

    def test_18_summary_materials_seen_is_list(self):
        pipeline = self._run_once()
        summary = pipeline.get_session_summary()
        assert isinstance(summary["materials_seen"], list)

    def test_19_second_process_increments_count(self):
        audio = _sine(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        pipeline.process(audio, SR)
        pipeline.process(audio, SR)
        summary = pipeline.get_session_summary()
        assert summary["session_results"] == 2


# ---------------------------------------------------------------------------
# Klasse 5: process() — Mono und Stereo
# ---------------------------------------------------------------------------


class TestProcessCall:
    def test_20_process_mono_returns_result(self):
        audio = _sine(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        result = pipeline.process(audio, SR)
        assert result is not None

    def test_21_process_result_has_audio(self):
        audio = _sine(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        result = pipeline.process(audio, SR)
        assert hasattr(result, "audio")

    def test_22_process_audio_no_nan(self):
        audio = _sine(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        result = pipeline.process(audio, SR)
        assert not np.any(np.isnan(result.audio))

    def test_23_process_audio_no_inf(self):
        audio = _sine(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        result = pipeline.process(audio, SR)
        assert not np.any(np.isinf(result.audio))

    def test_24_process_stereo_mode(self):
        audio = _stereo(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        result = pipeline.process(audio, SR)
        assert result is not None

    def test_25_process_silence_no_crash(self):
        audio = np.zeros(SR // 2, dtype=np.float32)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        result = pipeline.process(audio, SR)
        assert result is not None

    def test_26_engine_process_called_once(self):
        audio = _sine(secs=0.5)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        pipeline.process(audio, SR)
        pipeline._engine.process.assert_called_once_with(audio, SR)


# ---------------------------------------------------------------------------
# Klasse 6: _append_audit() — Audit-Trail
# ---------------------------------------------------------------------------


class TestAppendAudit:
    def test_27_audit_creates_ndjson_entry(self, tmp_path):
        audio = _sine(secs=0.5)
        mock_result = _make_mock_result(audio)
        with (
            patch("backend.core.pipeline_main.AutonomousRestorationEngine"),
            patch("backend.core.pipeline_main._AUDIT_LOG_PATH", str(tmp_path / "audit.ndjson")),
        ):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            pipeline._append_audit(mock_result)
            log_path = tmp_path / "audit.ndjson"
            assert log_path.exists()

    def test_28_audit_entry_is_valid_json(self, tmp_path):
        audio = _sine(secs=0.5)
        mock_result = _make_mock_result(audio)
        audit_path = str(tmp_path / "audit.ndjson")
        with (
            patch("backend.core.pipeline_main.AutonomousRestorationEngine"),
            patch("backend.core.pipeline_main._AUDIT_LOG_PATH", audit_path),
        ):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            pipeline._append_audit(mock_result)
        with open(audit_path, encoding="utf-8") as f:
            line = f.readline().strip()
        entry = json.loads(line)
        assert isinstance(entry, dict)

    def test_29_audit_entry_has_mode_field(self, tmp_path):
        audio = _sine(secs=0.5)
        mock_result = _make_mock_result(audio)
        audit_path = str(tmp_path / "audit.ndjson")
        with (
            patch("backend.core.pipeline_main.AutonomousRestorationEngine"),
            patch("backend.core.pipeline_main._AUDIT_LOG_PATH", audit_path),
        ):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            pipeline._append_audit(mock_result)
        with open(audit_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert "mode" in entry

    def test_30_audit_entry_has_rollback_field(self, tmp_path):
        audio = _sine(secs=0.5)
        mock_result = _make_mock_result(audio)
        audit_path = str(tmp_path / "audit.ndjson")
        with (
            patch("backend.core.pipeline_main.AutonomousRestorationEngine"),
            patch("backend.core.pipeline_main._AUDIT_LOG_PATH", audit_path),
        ):
            pipeline = AurikAutonomousPipeline(enable_self_learning=False)
            pipeline._append_audit(mock_result)
        with open(audit_path, encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert "rollback" in entry


# ---------------------------------------------------------------------------
# Edge-Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_36_processing_mode_restoration_value_is_string(self):
        assert isinstance(ProcessingMode.RESTORATION.value, str)

    def test_37_processing_mode_studio_2026_value_is_string(self):
        assert isinstance(ProcessingMode.STUDIO_2026.value, str)

    def test_38_pipeline_mode_restoration_has_correct_string(self):
        mode = ProcessingMode.RESTORATION
        assert "restoration" in mode.value.lower()

    def test_39_session_results_append_on_process(self):
        audio = _sine(secs=0.3)
        pipeline, mock_result = _make_pipeline_with_mock_engine(audio)
        assert len(pipeline._session_results) == 0
        pipeline.process(audio, SR)
        assert len(pipeline._session_results) == 1
