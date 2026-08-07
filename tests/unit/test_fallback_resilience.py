"""Tests für Fallback-Resilience nach handwerklichen Fixes (resampy, pyloudnorm)."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest


@pytest.mark.unit
class TestFileImportResilience:
    """Verifies file_import works regardless of resampy availability."""

    def test_resample_to_different_sr(self, tmp_path: Path):
        """Resampling from 44.1k to 48k must succeed (resampy or scipy fallback)."""
        import soundfile as sf

        import backend.file_import as fi

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 22050, endpoint=False)).astype(np.float32)
        test_file = tmp_path / "test.wav"
        sf.write(str(test_file), audio, 44100)

        result = fi.load_audio_file(str(test_file), target_sr=48000)
        assert result.get("error") is None  # type: ignore[union-attr]
        assert result["audio"] is not None, f"Audio should not be None: {result.get('error')}"  # type: ignore[union-attr]
        audio_out = cast(np.ndarray, result["audio"])
        assert audio_out.ndim == 1

    def test_stereo_no_resample_passthrough(self, tmp_path: Path):
        """Stereo file without resampling must pass through unchanged in shape."""
        import soundfile as sf

        import backend.file_import as fi

        audio = np.column_stack(
            [
                np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 24000, endpoint=False)),
                np.cos(2 * np.pi * 440 * np.linspace(0, 0.5, 24000, endpoint=False)),
            ]
        ).astype(np.float32)
        test_file = tmp_path / "stereo.wav"
        sf.write(str(test_file), audio, 48000)

        result = fi.load_audio_file(str(test_file))
        assert result.get("error") is None  # type: ignore[union-attr]
        assert result["audio"] is not None  # type: ignore[index]
        assert result["channels"] >= 2  # type: ignore[index]


class TestLoudnessFallback:
    """Verifies LoudnessAnalyzer degrades gracefully without pyloudnorm."""

    def test_analyze_returns_finite_values(self):
        """LoudnessAnalyzer must return finite values even without pyloudnorm."""
        from backend.core.delivery_standards import LoudnessAnalyzer, LoudnessResult

        analyzer = LoudnessAnalyzer()
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000, endpoint=False)).astype(np.float32) * 0.5

        result = analyzer.analyze(audio, 48000)
        assert isinstance(result, LoudnessResult)
        assert np.isfinite(result.integrated_lufs), f"integrated_lufs={result.integrated_lufs}"
        assert np.isfinite(result.true_peak_dbtp)
        assert np.isfinite(result.sample_peak_dbfs)
        # RMS fallback: values should be in a reasonable range
        assert -80 < result.integrated_lufs < 10, f"LUFS out of range: {result.integrated_lufs}"
        assert result.true_peak_dbtp < 10

    def test_silence_returns_valid_result(self):
        """Silence must not crash the analyzer."""
        from backend.core.delivery_standards import LoudnessAnalyzer, LoudnessResult

        analyzer = LoudnessAnalyzer()
        audio = np.zeros(48000, dtype=np.float32)

        result = analyzer.analyze(audio, 48000)
        assert isinstance(result, LoudnessResult)
        # Silence produces -inf LUFS — that's physically correct, not a crash
        assert not np.isnan(result.integrated_lufs), f"LUFS is NaN: {result.integrated_lufs}"


class TestPsychoacousticMetricsMigration:
    """Verifies PsychoAcousticMetrics is accessible from its new location."""

    def test_import_from_comprehensive_metrics(self):
        """PsychoAcousticMetrics must be importable from comprehensive_metrics."""
        from backend.core.comprehensive_metrics import PsychoAcousticMetrics

        assert PsychoAcousticMetrics is not None

    def test_can_instantiate(self):
        """PsychoAcousticMetrics must be instantiable with valid defaults.

        Note: PsychoAcousticMetrics is deliberately NOT a @dataclass (it serves
        as both data container and calculator, §5/5) — it has a custom
        __init__(sample_rate=48000, **kwargs) that accepts arbitrary field
        names via setattr. Use __annotations__ instead of dataclasses.fields().
        """
        from backend.core.comprehensive_metrics import PsychoAcousticMetrics

        kwargs: dict[str, Any] = {}
        for name, type_ in PsychoAcousticMetrics.__annotations__.items():
            kwargs[name] = 0.5 if "float" in str(type_) else 0
        metrics = PsychoAcousticMetrics(**kwargs)
        assert metrics is not None
        assert hasattr(metrics, "snr_db")
