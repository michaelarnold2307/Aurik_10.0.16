from __future__ import annotations

"""Tests for §2.39 OOM-Recovery-Checkpoint-System.

Verifies:
  - Checkpoint save/load round-trip (audio + JSON)
  - Atomic writes (no partial files on failure)
  - Expired checkpoint cleanup (> 7 days)
  - Orphaned checkpoint cleanup (missing audio WAV)
  - RecoveryCheckpoint dataclass serialisation
  - save_checkpoint under various input conditions
  - find_pending_checkpoints discovery
  - delete_checkpoint cleanup
  - Stereo audio checkpoint round-trip
  - Checkpoint with empty/missing fields
"""


import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sessions_tmp(tmp_path):
    """Override _SESSIONS_DIR to a temporary directory for isolation."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    with patch("backend.core.recovery_checkpoint._SESSIONS_DIR", sessions):
        yield sessions


@pytest.fixture
def sample_mono_audio():
    """1-second mono sine wave at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t), sr


@pytest.fixture
def sample_stereo_audio():
    """1-second stereo audio at 48 kHz (2, N) channel-first layout."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    ch0 = np.sin(2 * np.pi * 440 * t)
    ch1 = np.sin(2 * np.pi * 880 * t)
    return np.stack([ch0, ch1], axis=0), sr


class _FakeDefectResult:
    """Minimal defect result stub for checkpoint serialisation."""

    def __init__(self):
        self.scores = {}
        self.material_type = None
        self.spectral_fingerprint = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveCheckpoint:
    """Tests for save_checkpoint()."""

    def test_save_creates_json_and_wav(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import save_checkpoint

        audio, sr = sample_mono_audio
        result = save_checkpoint(
            input_path="/tmp/test_input.wav",
            output_path="/tmp/test_output.wav",
            current_audio=audio,
            sample_rate=sr,
            phases_executed=["phase_01_click_removal"],
            phases_remaining=["phase_02_hum_removal", "phase_03_denoise"],
            failure_phase="phase_02_hum_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None
        assert os.path.isfile(result)  # JSON file

        # Audio WAV must exist
        audio_wav = result.replace("_oom_checkpoint.json", "_oom_audio.wav")
        assert os.path.isfile(audio_wav)

    def test_save_json_contains_required_fields(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import save_checkpoint

        audio, sr = sample_mono_audio
        result = save_checkpoint(
            input_path="/tmp/song.mp3",
            output_path="/tmp/song_restored.wav",
            current_audio=audio,
            sample_rate=sr,
            phases_executed=["phase_01_click_removal"],
            phases_remaining=["phase_03_denoise"],
            failure_phase="phase_03_denoise",
            mode="maximum",
            defect_result=_FakeDefectResult(),
            era_decade=1960,
            restorability_score=65.0,
            quality_estimate=0.42,
            musical_goals={"brillanz": 0.87, "waerme": 0.82},
        )
        with open(result) as f:  # type: ignore[arg-type]
            data = json.load(f)

        assert data["input_path"] == "/tmp/song.mp3"
        assert data["output_path"] == "/tmp/song_restored.wav"
        assert data["mode"] == "maximum"
        assert data["era_decade"] == 1960
        assert data["restorability_score"] == 65.0
        assert data["quality_estimate_at_failure"] == 0.42
        assert data["musical_goals_at_failure"]["brillanz"] == 0.87
        assert data["failure_phase"] == "phase_03_denoise"
        assert data["failure_reason"] == "MemoryError"
        assert data["sample_rate"] == sr
        assert "timestamp" in data
        assert "aurik_version" in data

    def test_save_stereo_audio_roundtrip(self, sessions_tmp, sample_stereo_audio):
        from backend.core.recovery_checkpoint import (
            load_checkpoint_audio,
            save_checkpoint,
        )

        audio, sr = sample_stereo_audio
        result = save_checkpoint(
            input_path="/tmp/stereo.wav",
            output_path="/tmp/stereo_out.wav",
            current_audio=audio,
            sample_rate=sr,
            phases_executed=[],
            phases_remaining=["phase_01_click_removal"],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None

        # Load checkpoint
        from backend.core.recovery_checkpoint import RecoveryCheckpoint

        with open(result) as f:
            data = json.load(f)
        cp = RecoveryCheckpoint(**{k: v for k, v in data.items() if k in RecoveryCheckpoint.__dataclass_fields__})
        loaded = load_checkpoint_audio(cp)
        assert loaded is not None
        # soundfile returns (N, 2) — original is (2, N)
        if loaded.ndim == 2 and loaded.shape[1] == 2:
            loaded = loaded.T
        assert loaded.shape == audio.shape
        np.testing.assert_allclose(loaded, audio, atol=1e-5)

    def test_save_with_none_defect_result(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import save_checkpoint

        audio, sr = sample_mono_audio
        result = save_checkpoint(
            input_path="/tmp/no_defect.wav",
            output_path="/tmp/no_defect_out.wav",
            current_audio=audio,
            sample_rate=sr,
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=None,
        )
        assert result is not None


class TestFindPendingCheckpoints:
    """Tests for find_pending_checkpoints()."""

    def test_find_returns_valid_checkpoints(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import (
            find_pending_checkpoints,
            save_checkpoint,
        )

        save_checkpoint(
            input_path="/tmp/find_test.wav",
            output_path="/tmp/find_test_out.wav",
            current_audio=sample_mono_audio[0],
            sample_rate=sample_mono_audio[1],
            phases_executed=["phase_01_click_removal"],
            phases_remaining=["phase_02_hum_removal"],
            failure_phase="phase_02_hum_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )

        checkpoints = find_pending_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].input_path == "/tmp/find_test.wav"
        assert checkpoints[0].failure_phase == "phase_02_hum_removal"

    def test_find_ignores_expired_checkpoints(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import (
            find_pending_checkpoints,
            save_checkpoint,
        )

        result = save_checkpoint(
            input_path="/tmp/expired.wav",
            output_path="/tmp/expired_out.wav",
            current_audio=sample_mono_audio[0],
            sample_rate=sample_mono_audio[1],
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None
        # Backdate checkpoint to 8 days ago
        with open(result) as f:
            data = json.load(f)
        data["timestamp"] = time.time() - 8 * 86400
        with open(result, "w") as f:
            json.dump(data, f)

        checkpoints = find_pending_checkpoints()
        assert len(checkpoints) == 0

    def test_find_ignores_orphaned_checkpoints(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import (
            find_pending_checkpoints,
            save_checkpoint,
        )

        result = save_checkpoint(
            input_path="/tmp/orphaned.wav",
            output_path="/tmp/orphaned_out.wav",
            current_audio=sample_mono_audio[0],
            sample_rate=sample_mono_audio[1],
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None
        # Remove audio WAV
        audio_wav = result.replace("_oom_checkpoint.json", "_oom_audio.wav")
        os.remove(audio_wav)

        checkpoints = find_pending_checkpoints()
        assert len(checkpoints) == 0

    def test_find_empty_directory(self, sessions_tmp):
        from backend.core.recovery_checkpoint import find_pending_checkpoints

        assert find_pending_checkpoints() == []


class TestLoadCheckpointAudio:
    """Tests for load_checkpoint_audio()."""

    def test_load_mono_audio(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import (
            RecoveryCheckpoint,
            load_checkpoint_audio,
            save_checkpoint,
        )

        audio, sr = sample_mono_audio
        result = save_checkpoint(
            input_path="/tmp/load_test.wav",
            output_path="/tmp/load_test_out.wav",
            current_audio=audio,
            sample_rate=sr,
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        with open(result) as f:  # type: ignore[arg-type]
            data = json.load(f)
        cp = RecoveryCheckpoint(**{k: v for k, v in data.items() if k in RecoveryCheckpoint.__dataclass_fields__})

        loaded_audio = load_checkpoint_audio(cp)
        assert loaded_audio is not None
        assert loaded_audio.shape == audio.shape
        np.testing.assert_allclose(loaded_audio, audio, atol=1e-5)

    def test_original_audio_is_preferred_over_checkpoint_audio(self, sessions_tmp, sample_mono_audio, tmp_path):
        import soundfile as sf

        from backend.core.recovery_checkpoint import (
            RecoveryCheckpoint,
            load_checkpoint_audio,
            save_checkpoint,
        )

        checkpoint_audio, sr = sample_mono_audio
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        original_audio = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        original_path = tmp_path / "original_source.wav"
        sf.write(str(original_path), original_audio, sr, subtype="FLOAT", format="WAV")

        result = save_checkpoint(
            input_path=str(original_path),
            output_path="/tmp/preferred_out.wav",
            current_audio=checkpoint_audio,
            sample_rate=sr,
            phases_executed=["phase_01_click_removal"],
            phases_remaining=["phase_02_hum_removal"],
            failure_phase="phase_02_hum_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None

        with open(result) as f:
            data = json.load(f)
        cp = RecoveryCheckpoint(**{k: v for k, v in data.items() if k in RecoveryCheckpoint.__dataclass_fields__})

        loaded_audio = load_checkpoint_audio(cp)
        assert loaded_audio is not None
        assert loaded_audio.shape == original_audio.shape
        np.testing.assert_allclose(loaded_audio, original_audio, atol=1e-5)

    def test_original_audio_resampled_to_checkpoint_sr_on_mismatch(self, sessions_tmp, sample_mono_audio, tmp_path):
        import soundfile as sf

        from backend.core.recovery_checkpoint import (
            RecoveryCheckpoint,
            load_checkpoint_audio,
            save_checkpoint,
        )

        checkpoint_audio, checkpoint_sr = sample_mono_audio  # 48 kHz checkpoint baseline
        src_sr = 44100
        t = np.linspace(0, 1.0, src_sr, endpoint=False, dtype=np.float32)
        original_audio = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        original_path = tmp_path / "original_source_44k1.wav"
        sf.write(str(original_path), original_audio, src_sr, subtype="FLOAT", format="WAV")

        result = save_checkpoint(
            input_path=str(original_path),
            output_path="/tmp/preferred_resample_out.wav",
            current_audio=checkpoint_audio,
            sample_rate=checkpoint_sr,
            phases_executed=["phase_01_click_removal"],
            phases_remaining=["phase_02_hum_removal"],
            failure_phase="phase_02_hum_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None

        with open(result) as f:
            data = json.load(f)
        cp = RecoveryCheckpoint(**{k: v for k, v in data.items() if k in RecoveryCheckpoint.__dataclass_fields__})

        loaded_audio = load_checkpoint_audio(cp)
        assert loaded_audio is not None

        # 1 s at 48 kHz expected after SR alignment to checkpoint SR
        assert abs(len(loaded_audio) - checkpoint_sr) <= 2

        # Dominant frequency should remain close to 880 Hz after resampling.
        n = min(len(loaded_audio), checkpoint_sr)
        window = np.hanning(n).astype(np.float32)
        spec = np.fft.rfft((loaded_audio[:n] * window).astype(np.float32))
        freqs = np.fft.rfftfreq(n, d=1.0 / float(checkpoint_sr))
        peak_hz = float(freqs[int(np.argmax(np.abs(spec)))])
        assert abs(peak_hz - 880.0) < 12.0


class TestDeleteCheckpoint:
    """Tests for delete_checkpoint()."""

    def test_delete_removes_files(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import (
            delete_checkpoint,
            save_checkpoint,
        )

        result = save_checkpoint(
            input_path="/tmp/delete_test.wav",
            output_path="/tmp/delete_test_out.wav",
            current_audio=sample_mono_audio[0],
            sample_rate=sample_mono_audio[1],
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None
        audio_wav = result.replace("_oom_checkpoint.json", "_oom_audio.wav")
        assert os.path.isfile(result)
        assert os.path.isfile(audio_wav)

        delete_checkpoint("/tmp/delete_test.wav")

        assert not os.path.isfile(result)
        assert not os.path.isfile(audio_wav)

    def test_delete_nonexistent_is_safe(self, sessions_tmp):
        from backend.core.recovery_checkpoint import delete_checkpoint

        # Should not raise
        delete_checkpoint("/tmp/nonexistent_file.wav")


class TestCleanupExpired:
    """Tests for cleanup_expired_checkpoints()."""

    def test_cleanup_removes_expired(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import (
            cleanup_expired_checkpoints,
            save_checkpoint,
        )

        result = save_checkpoint(
            input_path="/tmp/cleanup_test.wav",
            output_path="/tmp/cleanup_test_out.wav",
            current_audio=sample_mono_audio[0],
            sample_rate=sample_mono_audio[1],
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None
        # Backdate to 10 days
        with open(result) as f:
            data = json.load(f)
        data["timestamp"] = time.time() - 10 * 86400
        with open(result, "w") as f:
            json.dump(data, f)

        removed = cleanup_expired_checkpoints()
        assert removed == 1
        assert not os.path.isfile(result)

    def test_cleanup_keeps_fresh(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import (
            cleanup_expired_checkpoints,
            save_checkpoint,
        )

        save_checkpoint(
            input_path="/tmp/fresh.wav",
            output_path="/tmp/fresh_out.wav",
            current_audio=sample_mono_audio[0],
            sample_rate=sample_mono_audio[1],
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )

        removed = cleanup_expired_checkpoints()
        assert removed == 0


class TestRecoveryCheckpointDataclass:
    """Tests for RecoveryCheckpoint dataclass invariants."""

    def test_dataclass_fields(self):
        from backend.core.recovery_checkpoint import RecoveryCheckpoint

        fields = set(RecoveryCheckpoint.__dataclass_fields__.keys())
        required = {
            "input_path",
            "output_path",
            "phases_executed",
            "phases_remaining",
            "mode",
            "material_type",
            "era_decade",
            "defect_scores",
            "defect_scores_full",
            "restorability_score",
            "spectral_fingerprint",
            "quality_estimate_at_failure",
            "musical_goals_at_failure",
            "audio_wav_path",
            "sample_rate",
            "original_input_path",
            "timestamp",
            "aurik_version",
            "failure_phase",
            "failure_reason",
        }
        assert required.issubset(fields), f"Missing fields: {required - fields}"

    def test_default_failure_reason(self):
        from backend.core.recovery_checkpoint import RecoveryCheckpoint, _get_aurik_version

        cp = RecoveryCheckpoint(
            input_path="",
            output_path="",
            phases_executed=[],
            phases_remaining=[],
            mode="quality",
            material_type="",
            era_decade=None,
            defect_scores={},
            defect_scores_full={},
            restorability_score=None,
            spectral_fingerprint={},
            quality_estimate_at_failure=0.0,
            musical_goals_at_failure={},
            audio_wav_path="",
            sample_rate=48000,
            original_input_path="",
        )
        assert cp.failure_reason == "MemoryError"
        assert cp.aurik_version == _get_aurik_version()


class TestSaveCheckpointFilenameHandling:
    """Edge cases for filename sanitisation."""

    def test_special_characters_in_filename(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import save_checkpoint

        audio, sr = sample_mono_audio
        result = save_checkpoint(
            input_path="/tmp/Ärger & Söhne (Live) [2024].flac",
            output_path="/tmp/output.wav",
            current_audio=audio,
            sample_rate=sr,
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None
        assert os.path.isfile(result)

    def test_very_long_filename(self, sessions_tmp, sample_mono_audio):
        from backend.core.recovery_checkpoint import save_checkpoint

        audio, sr = sample_mono_audio
        long_name = "a" * 300
        result = save_checkpoint(
            input_path=f"/tmp/{long_name}.wav",
            output_path="/tmp/output.wav",
            current_audio=audio,
            sample_rate=sr,
            phases_executed=[],
            phases_remaining=[],
            failure_phase="phase_01_click_removal",
            mode="quality",
            defect_result=_FakeDefectResult(),
        )
        assert result is not None
        # Stem is truncated to 120 chars
        assert len(Path(result).stem) <= 140  # stem + suffix _oom_checkpoint
