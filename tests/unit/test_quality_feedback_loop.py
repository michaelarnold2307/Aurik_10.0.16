from __future__ import annotations

import numpy as np
import pytest

from backend.core.quality_feedback_loop import QualityFeedbackLoop


class _Result:
    def __init__(self, audio: np.ndarray, success: bool = True):
        self.audio = audio
        self.success = success
        self.metadata: dict[str, object] = {}


class _PhaseAlwaysSuccess:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, audio, sample_rate=48000, **kwargs):
        self.calls += 1
        return _Result(np.asarray(audio, dtype=np.float32), success=True)

    def get_metadata(self):
        class _Meta:
            name = "MockPhase"

        return _Meta()


class _PhaseFirstFailThenPass:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs_seen: list[dict] = []

    def process(self, audio, sample_rate=48000, **kwargs):
        self.calls += 1
        self.kwargs_seen.append(dict(kwargs))
        if self.calls == 1:
            return _Result(np.asarray(audio, dtype=np.float32), success=False)
        return _Result(np.asarray(audio, dtype=np.float32), success=True)

    def get_metadata(self):
        class _Meta:
            name = "MockPhase"

        return _Meta()


class _SeqMetrics:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._idx = 0
        self.sample_rate = 48000

    def calculate_naturalness_score(self, audio, reference=None):
        value = self._values[min(self._idx, len(self._values) - 1)]
        self._idx += 1
        return {
            "naturalness_overall": float(value),
            "temporal_smoothness": 1.0,
            "harmonic_coherence": 1.0,
            "noise_floor_consistency": 1.0,
        }


@pytest.mark.unit
def test_feedback_loop_uses_previous_iteration_improvement():
    """Regression: improvement guard must compare against previous iteration, not best-so-far value."""
    audio = np.ones(1024, dtype=np.float32) * 0.1
    phase = _PhaseAlwaysSuccess()

    loop = QualityFeedbackLoop(target_naturalness=0.95, max_iterations=3, min_improvement=0.01)
    loop.metrics = _SeqMetrics([0.50, 0.515, 0.535])  # type: ignore[assignment]

    _ = loop.process_with_feedback(phase, audio, 48000, repair_strength=0.8)  # type: ignore[arg-type]
    assert phase.calls == 3


def test_feedback_loop_fallback_reuses_original_kwargs():
    """Fallback branch must call phase with original kwargs (not adapted/modified state)."""
    audio = np.ones(512, dtype=np.float32) * 0.05
    phase = _PhaseFirstFailThenPass()

    loop = QualityFeedbackLoop(target_naturalness=0.80, max_iterations=2, min_improvement=0.01)
    loop.metrics = _SeqMetrics([0.2, 0.2])  # type: ignore[assignment]

    _ = loop.process_with_feedback(phase, audio, 48000, custom_threshold=0.42, repair_strength=0.7)  # type: ignore[arg-type]

    assert phase.calls == 2
    assert phase.kwargs_seen[1].get("custom_threshold") == 0.42
    assert phase.kwargs_seen[1].get("repair_strength") == 0.7
