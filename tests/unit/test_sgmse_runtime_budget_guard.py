from __future__ import annotations

"""Runtime budget guard tests for SGMSE+ chunked inference.

Ensures long ML paths degrade safely to WPE fallback instead of hanging.
"""


import time
from pathlib import Path

import numpy as np
import pytest

from plugins.sgmse_plugin import SGMSEPlusPlugin, _quarantine_corrupt_torchscript


@pytest.mark.unit
def test_chunked_runtime_budget_triggers_fallback_for_remaining_audio() -> None:
    """When budget is exceeded, remaining tail must be processed by WPE fallback."""
    plugin = object.__new__(SGMSEPlusPlugin)

    # Small synthetic chunk geometry keeps the test fast and deterministic.
    plugin._MAX_CHUNK_SAMPLES_LARGE = 1000
    plugin._MAX_CHUNK_SAMPLES_SMALL = 500
    plugin._OVERLAP_SAMPLES = 10

    fallback_calls: list[int] = []

    def _fast_ram() -> float:
        return 8.0

    def _slow_torch(chunk: np.ndarray, sigma: float) -> np.ndarray:
        time.sleep(0.01)
        return np.clip(np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    def _fallback(tail: np.ndarray, sr: int) -> np.ndarray:
        fallback_calls.append(len(tail))
        return np.zeros_like(tail, dtype=np.float32)

    plugin._get_available_ram_gb = _fast_ram  # type: ignore[method-assign]
    plugin._enhance_torchscript = _slow_torch  # type: ignore[assignment]
    plugin._wpe_fallback = _fallback  # type: ignore[assignment]

    mono = np.ones(4000, dtype=np.float32) * 0.1
    out = plugin._enhance_chunked(mono, sigma=0.5, max_runtime_s=0.001)

    assert out.shape == mono.shape
    assert np.isfinite(out).all()
    assert fallback_calls, "Runtime budget should trigger WPE fallback for the remaining tail."


def test_forward_timeout_guard_raises_timeout() -> None:
    """Single forward call must be bounded by hard timeout."""
    plugin = object.__new__(SGMSEPlusPlugin)

    def _slow_call():
        time.sleep(0.05)
        return 1

    with pytest.raises(TimeoutError):
        plugin._run_with_timeout(_slow_call, timeout_s=0.001)


def test_corrupt_torchscript_quarantine_moves_bad_local_model(tmp_path: Path) -> None:
    """A corrupt local TorchScript must be moved out of the production model path."""
    model_path = tmp_path / "sgmse_plus.ts"
    model_path.write_bytes(b"not a torchscript archive")

    quarantined = _quarantine_corrupt_torchscript(
        model_path,
        RuntimeError("invalid magic number for TorchScript archive"),
    )

    assert quarantined is not None
    assert not model_path.exists()
    assert quarantined.exists()
    assert quarantined.read_bytes() == b"not a torchscript archive"
