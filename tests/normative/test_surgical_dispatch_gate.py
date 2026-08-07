import pytest

"""§2.59.14 A1-Test: PhaseInterface.surgical_dispatch() — Zeitfenster-Phasenausführung."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
import numpy as np

from backend.core.phases.phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata


class _MockPhase:
    def __init__(self):
        self._metadata = PhaseMetadata(phase_id="mock", name="Mock", category=PhaseCategory.ENHANCEMENT, priority=1)

    def get_metadata(self):
        return self._metadata

    def process(self, audio, sample_rate, material_type="unknown", **kwargs):
        return audio * 2.0


@pytest.mark.unit
def test_surgical_dispatch_single_zone():
    sr = 48000
    audio = np.random.randn(2, sr * 2).astype(np.float32) * 0.1
    orig = audio.copy()
    result = PhaseInterface.surgical_dispatch(_MockPhase(), audio, sr, "vinyl", time_ranges=[(1.0, 1.1)])  # type: ignore[arg-type]
    assert result.shape == orig.shape
    assert np.allclose(result[:, : int(0.9 * sr)], orig[:, : int(0.9 * sr)], atol=1e-7)


def test_surgical_dispatch_skip_tiny():
    sr = 48000
    audio = np.random.randn(sr).astype(np.float32) * 0.1
    orig = audio.copy()
    result = PhaseInterface.surgical_dispatch(
        _MockPhase(),
        audio,
        sr,
        "vinyl",
        time_ranges=[(0.5, 0.50001)],
        context_ms=0,  # type: ignore[arg-type]
    )
    assert np.allclose(result, orig, atol=1e-7)


def test_surgical_dispatch_mono():
    sr = 48000
    audio = np.random.randn(sr).astype(np.float32) * 0.1
    result = PhaseInterface.surgical_dispatch(_MockPhase(), audio, sr, "vinyl", time_ranges=[(0.5, 0.55)])  # type: ignore[arg-type]
    assert result.ndim == 1


class _SpikePhase:
    def __init__(self):
        self._metadata = PhaseMetadata(phase_id="spike", name="Spike", category=PhaseCategory.ENHANCEMENT, priority=1)

    def get_metadata(self):
        return self._metadata

    def process(self, audio, sample_rate, material_type="unknown", **kwargs):
        return audio * 100.0


def test_surgical_dispatch_clamp():
    sr = 48000
    audio = np.random.randn(sr).astype(np.float32) * 0.01
    result = PhaseInterface.surgical_dispatch(_SpikePhase(), audio, sr, "vinyl", time_ranges=[(0.4, 0.5)])  # type: ignore[arg-type]
    assert np.abs(result).max() / (np.abs(audio).max() + 1e-10) <= 2.1
