"""
Test suite for core/export_workflow.py - Export functions
Tests audio export and audit log export
"""

import os
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.export_workflow import _evaluate_export_quality_gate, export_audio, export_audit_log


@pytest.fixture
def temp_export_dir():
    """Create temporary export directory"""
    temp_dir = tempfile.mkdtemp()
    export_dir = os.path.join(temp_dir, "export")
    logs_dir = os.path.join(temp_dir, "logs")
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Change to temp dir
    orig_dir = os.getcwd()
    os.chdir(temp_dir)

    yield temp_dir

    # Cleanup
    os.chdir(orig_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.unit
def test_export_audio_basic(temp_export_dir):
    """Test basic audio export"""
    audio = np.random.randn(1000) * 0.5
    sr = 48000
    filename = "test_audio.wav"

    export_path = export_audio(audio, sr, filename)

    # Check file was created
    assert os.path.exists(export_path)
    assert export_path == os.path.join("export", filename)


def test_export_audio_stereo(temp_export_dir):
    """Test stereo audio export"""
    audio = np.random.randn(1000, 2) * 0.5  # Stereo
    sr = 44100
    filename = "test_stereo.wav"

    export_path = export_audio(audio, sr, filename)

    # Check file exists
    assert os.path.exists(export_path)


def test_export_audio_different_sample_rates(temp_export_dir):
    """Test audio export with different sample rates"""
    audio = np.random.randn(500) * 0.5

    # Test 44.1 kHz
    path1 = export_audio(audio, 44100, "test_44k.wav")
    assert os.path.exists(path1)

    # Test 48 kHz
    path2 = export_audio(audio, 48000, "test_48k.wav")
    assert os.path.exists(path2)

    # Test 96 kHz
    path3 = export_audio(audio, 96000, "test_96k.wav")
    assert os.path.exists(path3)


def test_export_audio_high_res(temp_export_dir):
    """Test high-resolution audio export"""
    audio = np.random.randn(2000) * 0.5
    sr = 192000
    filename = "test_hires.wav"

    export_path = export_audio(audio, sr, filename)

    assert os.path.exists(export_path)


def test_export_audit_log_basic(temp_export_dir):
    """Test basic audit log export"""
    audit_log = [{"step": "denoising", "status": "success"}, {"step": "restoration", "status": "complete"}]
    filename = "test_audit.log"

    log_path = export_audit_log(audit_log, filename)

    # Check file was created
    assert os.path.exists(log_path)
    assert log_path == os.path.join("logs", filename)


def test_export_audit_log_content(temp_export_dir):
    """Test audit log exports correct content"""
    audit_log = [{"step": "analysis", "result": "vinyl_detected"}, {"step": "filtering", "params": {"freq": 50}}]
    filename = "test_content.log"

    log_path = export_audit_log(audit_log, filename)

    # Read file and check content
    with open(log_path) as f:
        content = f.read()

    assert "analysis" in content
    assert "vinyl_detected" in content
    assert "filtering" in content


def test_export_audit_log_empty(temp_export_dir):
    """Test audit log export with empty log"""
    audit_log: list[Any] = []
    filename = "test_empty.log"

    log_path = export_audit_log(audit_log, filename)

    # File should exist but be empty
    assert os.path.exists(log_path)


def test_export_audit_log_append(temp_export_dir):
    """Test audit log appends to existing file"""
    filename = "test_append.log"

    # First export
    audit_log1 = [{"entry": 1}]
    log_path = export_audit_log(audit_log1, filename)

    # Second export (should append)
    audit_log2 = [{"entry": 2}]
    export_audit_log(audit_log2, filename)

    # Check both entries are in file
    with open(log_path) as f:
        content = f.read()

    assert "entry" in content
    # Should have 2 lines (one per entry)
    lines = content.strip().split("\n")
    assert len(lines) >= 2


def test_export_audio_return_path(temp_export_dir):
    """Test export_audio returns correct path"""
    audio = np.random.randn(500) * 0.5
    sr = 48000
    filename = "test_return.wav"

    path = export_audio(audio, sr, filename)

    # Path should contain export directory and filename
    assert "export" in path
    assert filename in path
    assert path.endswith(filename)


def test_export_audit_log_return_path(temp_export_dir):
    """Test export_audit_log returns correct path"""
    audit_log = [{"test": "data"}]
    filename = "test_return.log"

    path = export_audit_log(audit_log, filename)

    # Path should contain logs directory and filename
    assert "logs" in path
    assert filename in path
    assert path.endswith(filename)


def test_export_audio_zero_signal(temp_export_dir):
    """Test audio export with zero (silent) signal"""
    audio = np.zeros(1000)
    sr = 48000
    filename = "test_silent.wav"

    export_path = export_audio(audio, sr, filename)

    assert os.path.exists(export_path)


def test_export_audit_log_complex_entries(temp_export_dir):
    """Test audit log with complex nested entries"""
    audit_log = [
        {
            "step": "restoration",
            "params": {"mode": "vinyl", "settings": {"denoise": True, "declicker": True}},
            "result": {"quality": 0.92, "snr": 35.5},
        }
    ]
    filename = "test_complex.log"

    log_path = export_audit_log(audit_log, filename)

    # Check file exists and has content
    assert os.path.exists(log_path)
    with open(log_path) as f:
        content = f.read()
    assert "restoration" in content
    assert "denoise" in content


def test_quality_gate_normalizes_plain_fail_reason():
    """Quality-gate payload without structured fail_reasons is normalized deterministically."""
    quality_gate = {
        "passed": False,
        "fail_reason": "PQS unter Mindestschwelle",
        "required_gates": ["musical_goals", "pqs", "oqs"],
    }

    passed, fail_reason, degradation_status, fail_reasons = _evaluate_export_quality_gate(quality_gate)

    assert passed is False
    assert fail_reason == "PQS unter Mindestschwelle"
    assert degradation_status == "blocked"
    assert isinstance(fail_reasons, list)
    assert len(fail_reasons) == 1
    assert fail_reasons[0]["error_code"] == "QUALITY_GATE_FAILED"
    assert fail_reasons[0]["severity"] == "blocked"


def test_quality_gate_prefers_structured_error_code_when_fail_reason_missing():
    """When fail_reason is absent, first structured error code becomes primary fail reason."""
    quality_gate = {
        "passed": False,
        "fail_reasons": [
            {
                "component": "quality_gate",
                "error_code": "PQS_GATE_FAILED",
                "severity": "blocked",
                "exc_msg": "PQS unter Mindestschwelle",
            }
        ],
    }

    passed, fail_reason, degradation_status, fail_reasons = _evaluate_export_quality_gate(quality_gate)

    assert passed is False
    assert fail_reason == "PQS_GATE_FAILED"
    assert degradation_status == "blocked"
    assert isinstance(fail_reasons, list)


def test_export_audio_blocks_failed_gate_without_recovery(temp_export_dir):
    """Failed quality gate without recovery metadata must block export."""
    audio = np.zeros(1000, dtype=np.float32)
    quality_gate = {
        "passed": False,
        "fail_reason": "PQS unter Mindestschwelle",
    }

    with pytest.raises(RuntimeError):
        export_audio(audio, 48_000, "blocked.wav", quality_gate=quality_gate)


def test_export_audio_sets_recovered_strategy(temp_export_dir):
    """Failed gate with successful recovery should export as recovered."""
    audio = np.zeros(1000, dtype=np.float32)
    quality_gate = {
        "passed": False,
        "fail_reason": "PQS unter Mindestschwelle",
        "recovery_attempted": True,
        "best_possible_reached": True,
    }

    path = export_audio(audio, 48_000, "recovered.wav", quality_gate=quality_gate)
    assert os.path.exists(path)
    meta_path = os.path.join("export", "recovered.json")
    assert os.path.exists(meta_path)
    with open(meta_path, encoding="utf-8") as f:
        payload = f.read()
    assert '"export_strategy": "recovered"' in payload


def test_export_audio_fqf_recovered_overrides_passed_gate(temp_export_dir):
    """fallback_quality_floor.status=recovered must force recovered strategy."""
    audio = np.zeros(1000, dtype=np.float32)
    quality_gate = {
        "passed": True,
        "fallback_quality_floor": {
            "triggered": True,
            "status": "recovered",
            "reason": "fallback_quality_floor_recovered_with_checkpoint",
        },
    }

    path = export_audio(audio, 48_000, "fqf_recovered.wav", quality_gate=quality_gate)
    assert os.path.exists(path)
    meta_path = os.path.join("export", "fqf_recovered.json")
    assert os.path.exists(meta_path)
    with open(meta_path, encoding="utf-8") as f:
        payload = f.read()
    assert '"export_strategy": "recovered"' in payload


def test_export_audio_fqf_degraded_overrides_passed_gate(temp_export_dir):
    """fallback_quality_floor.status=degraded must force degraded strategy."""
    audio = np.zeros(1000, dtype=np.float32)
    quality_gate = {
        "passed": True,
        "fallback_quality_floor": {
            "triggered": True,
            "status": "degraded",
            "reason": "fallback_quality_floor_failed_no_compatible_checkpoint",
        },
    }

    path = export_audio(audio, 48_000, "fqf_degraded.wav", quality_gate=quality_gate)
    assert os.path.exists(path)
    meta_path = os.path.join("export", "fqf_degraded.json")
    assert os.path.exists(meta_path)
    with open(meta_path, encoding="utf-8") as f:
        payload = f.read()
    assert '"export_strategy": "degraded"' in payload


# ---------------------------------------------------------------------------
# §2.46b / §2.49 fidelity_guards — ExportMetadata + sidecar JSON
# ---------------------------------------------------------------------------


def test_fidelity_guards_in_sidecar_json(temp_export_dir):
    """ExportMetadata.fidelity_guards must appear in JSON sidecar when set."""
    from backend.core.export_workflow import ExportMetadata, export_audio

    guards = {
        "spectral_tilt_guard": {
            "guard_fired_count": 2,
            "phases_guarded": ["phase_06_frequency_restoration"],
            "max_deviation_db_per_oct": 1.82,
            "max_wet_cap_applied": 0.72,
        },
        "hf_hallucination_guard": {
            "guard_fired_count": 1,
            "phases_guarded": ["phase_07_harmonic_restoration"],
            "max_delta_ratio": 1.15,
            "min_cap_hz": 14500.0,
        },
    }
    meta = ExportMetadata(fidelity_guards=guards)
    audio = np.zeros(1000, dtype=np.float32)
    path = export_audio(audio, 48_000, "fidelity_guard_test.wav", metadata=meta)

    sidecar = os.path.splitext(path)[0] + ".json"
    assert os.path.exists(sidecar), "Sidecar JSON must exist"
    import json

    with open(sidecar, encoding="utf-8") as f:
        payload = json.load(f)

    assert "fidelity_guards" in payload, "fidelity_guards key must be present in sidecar"
    fg = payload["fidelity_guards"]
    assert "spectral_tilt_guard" in fg
    assert "hf_hallucination_guard" in fg
    assert fg["spectral_tilt_guard"]["guard_fired_count"] == 2
    assert fg["hf_hallucination_guard"]["guard_fired_count"] == 1


def test_build_export_metadata_populates_fidelity_guards():
    """bridge.build_export_metadata extracts both guard dicts from result.metadata."""
    from types import SimpleNamespace

    from backend.api.bridge import build_export_metadata

    result = SimpleNamespace(
        metadata={
            "spectral_tilt_guard": {
                "guard_fired_count": 3,
                "phases_guarded": ["phase_06_frequency_restoration", "phase_39_harmonic_ext"],
                "max_deviation_db_per_oct": 2.1,
                "max_wet_cap_applied": 0.65,
            },
            "hf_hallucination_guard": {
                "guard_fired_count": 0,
                "phases_guarded": [],
                "max_delta_ratio": 0.0,
                "min_cap_hz": 24000.0,
            },
        }
    )
    em = build_export_metadata(result, title="TestSong", artist="Testkünstler")
    assert em is not None
    assert em.title == "TestSong"
    assert em.artist == "Testkünstler"
    assert em.fidelity_guards is not None
    stg = em.fidelity_guards.get("spectral_tilt_guard", {})
    assert stg["guard_fired_count"] == 3
    hfg = em.fidelity_guards.get("hf_hallucination_guard", {})
    assert hfg["guard_fired_count"] == 0


def test_build_export_metadata_no_guards_yields_none():
    """build_export_metadata returns fidelity_guards=None when result has no guard data."""
    from types import SimpleNamespace

    from backend.api.bridge import build_export_metadata

    result = SimpleNamespace(metadata={"some_key": "some_value"})
    em = build_export_metadata(result)
    assert em is not None
    assert em.fidelity_guards is None


def test_build_export_metadata_sanitizes_nan():
    """build_export_metadata must sanitize NaN/Inf values in guard dicts."""
    from types import SimpleNamespace

    from backend.api.bridge import build_export_metadata

    result = SimpleNamespace(
        metadata={
            "spectral_tilt_guard": {
                "guard_fired_count": 1,
                "max_deviation_db_per_oct": float("nan"),
                "phases_guarded": ["phase_06_frequency_restoration"],
            }
        }
    )
    em = build_export_metadata(result)
    assert em is not None
    assert em.fidelity_guards is not None
    stg = em.fidelity_guards["spectral_tilt_guard"]
    # NaN must be replaced by 0.0
    assert stg["max_deviation_db_per_oct"] == 0.0
