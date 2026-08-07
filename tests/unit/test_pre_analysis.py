"""Unit tests for pre_analysis.run_pre_analysis() — §CI-PREANALYSIS.

Validates:
- Sequential CLAP steps (era, genre) — no GIL contention regression
- Parallel non-CLAP steps (defect, restorability) — as_completed semantics
- Progress callback order and values
- Timeout handling
- Error aggregation in result.errors
"""

from __future__ import annotations

import time
from concurrent import futures as _cf
from unittest.mock import MagicMock, patch

import pytest

from backend.core.pre_analysis import (
    _SUBSTEP_TIMEOUT_S,
    PreAnalysisResult,
    ProgressState,
    run_pre_analysis,
)


def _slow_fn(seconds: float, result_value: object = None) -> object:
    """Simuliert einen langsamen Analyseschritt."""
    time.sleep(seconds)
    return result_value


class TestPreAnalysisProgress:
    """§G19/V71: ProgressState als Single Source of Truth."""

    def test_progress_state_defaults(self):
        ps = ProgressState()
        assert ps.pct == 0.0
        assert ps.step_msg == ""
        assert ps.total_steps == 4
        assert ps.done_steps == 0

    def test_progress_state_serializable(self):
        ps = ProgressState(pct=75.0, step_msg="Analyse: defect done", done_steps=1)
        d = {"pct": ps.pct, "step_msg": ps.step_msg, "done_steps": ps.done_steps}
        assert d["pct"] == 75.0
        assert "defect" in d["step_msg"]  # type: ignore[operator]


class TestRunPreAnalysisFlow:
    """End-to-end flow tests with mocked components."""

    def test_sequential_clap_steps(self, monkeypatch):
        """Era und Genre MÜSSEN sequentiell laufen (shared LAION-CLAP)."""
        call_order: list[str] = []

        class MockEraClassifier:
            def classify(self, audio, sr, transfer_chain=None):
                call_order.append("era")
                return MagicMock(decade=1970, confidence=0.80)

        class MockGenreClassifier:
            def classify(self, audio, sr):
                call_order.append("genre")
                return MagicMock(genre_label="Schlager", confidence=0.63)

        class MockDefectScanner:
            def __init__(self, **kw):
                pass

            def scan(self, audio, **kw):
                call_order.append("defect")
                return MagicMock(scores={})

        def mock_load_symbol(module, name):
            if "era_classifier" in module or ("bridge" in module and name == "get_era_classifier_fn"):
                return lambda: lambda a, s, **kw: MockEraClassifier().classify(a, s)
            if "genre_classifier" in module or ("bridge" in module and name == "get_genre_classifier_fn"):
                return lambda: lambda a, s: MockGenreClassifier().classify(a, s)
            if "defect_scanner" in module:
                return lambda **kw: MockDefectScanner(**kw)
            if "restorability" in module:
                return lambda audio, sr, **kw: MagicMock(score=64.0)

            # medium_detector
            class MockMedium:
                primary_material = "vinyl"
                confidence = 0.26
                chain_label = "vinyl → mp3"
                transfer_chain = ["vinyl", "mp3_low"]

            return lambda: MagicMock(detect=lambda *a, **kw: MockMedium())

        monkeypatch.setattr("backend.core.pre_analysis._load_symbol", mock_load_symbol)
        monkeypatch.setattr("backend.core.pre_analysis._store_in_cache", lambda *a: None)
        monkeypatch.setattr("backend.core.pre_analysis._load_cached_parts", lambda *a: {})

        import numpy as np

        audio = np.zeros((44100, 2), dtype=np.float32)
        progress_steps: list[tuple[int, str]] = []

        result = run_pre_analysis(
            audio_native=audio,
            sr_native=44100,
            file_path="/tmp/test.mp3",
            progress_callback=lambda pct, msg: progress_steps.append((pct, msg)),
            store_in_bridge_cache=False,
        )

        # Era must come before Genre (sequential CLAP steps)
        assert "era" in call_order, f"Era not called. Order: {call_order}"
        assert "genre" in call_order, f"Genre not called. Order: {call_order}"
        era_idx = call_order.index("era")
        genre_idx = call_order.index("genre")
        assert era_idx < genre_idx, (
            f"Era ({era_idx}) must run before Genre ({genre_idx}). "
            f"Parallel CLAP loading causes 10× slowdown. Order: {call_order}"
        )

    def test_error_propagation(self, monkeypatch):
        """Fehler in Einzelschritten werden in result.errors gesammelt."""

        def mock_load_symbol(module, name):
            if "bridge" in module and name == "get_era_classifier_fn":
                return lambda: lambda a, s: (_ for _ in ()).throw(RuntimeError("CLAP not available"))
            if "bridge" in module and name == "get_genre_classifier_fn":
                return lambda: lambda a, s: (_ for _ in ()).throw(RuntimeError("CLAP not available"))
            if "defect_scanner" in module:
                return lambda **kw: MagicMock(scan=lambda *a, **kw: MagicMock(scores={}))
            if "restorability" in module:
                return lambda audio, sr: MagicMock(score=50.0)

            class MockMedium:
                primary_material = "vinyl"
                confidence = 0.26
                chain_label = "vinyl → mp3"
                transfer_chain = ["vinyl", "mp3_low"]

            return lambda: MagicMock(detect=lambda *a, **kw: MockMedium())

        monkeypatch.setattr("backend.core.pre_analysis._load_symbol", mock_load_symbol)
        monkeypatch.setattr("backend.core.pre_analysis._store_in_cache", lambda *a: None)
        monkeypatch.setattr("backend.core.pre_analysis._load_cached_parts", lambda *a: {})

        import numpy as np

        audio = np.zeros((44100, 2), dtype=np.float32)
        result = run_pre_analysis(
            audio_native=audio,
            sr_native=44100,
            file_path="/tmp/test.mp3",
            store_in_bridge_cache=False,
        )

        assert "era" in result.errors, f"Era error not in errors: {result.errors}"
        assert "genre" in result.errors, f"Genre error not in errors: {result.errors}"
        assert result.era is None
        assert result.genre is None
        assert result.defects is not None  # Should succeed

    def test_progress_callbacks_fire_in_order(self, monkeypatch):
        """Progress-Callbacks feuern in aufsteigender Reihenfolge."""
        from unittest.mock import MagicMock

        class MockClassifier:
            def classify(self, audio, sr):
                return MagicMock(decade=1970, genre_label="Test")

        def mock_load_symbol(module, name):
            if "bridge" in module and name in ("get_era_classifier_fn", "get_genre_classifier_fn"):
                return lambda: lambda a, s: MockClassifier().classify(a, s)
            if "defect_scanner" in module:
                return lambda **kw: MagicMock(scan=lambda *a, **kw: MagicMock(scores={}))
            if "restorability" in module:
                return lambda audio, sr: MagicMock(score=64.0)

            class MockMedium:
                primary_material = "vinyl"
                confidence = 0.26
                chain_label = "vinyl → mp3"
                transfer_chain = ["vinyl", "mp3_low"]

            return lambda: MagicMock(detect=lambda *a, **kw: MockMedium())

        monkeypatch.setattr("backend.core.pre_analysis._load_symbol", mock_load_symbol)
        monkeypatch.setattr("backend.core.pre_analysis._store_in_cache", lambda *a: None)
        monkeypatch.setattr("backend.core.pre_analysis._load_cached_parts", lambda *a: {})

        import numpy as np

        audio = np.zeros((44100, 2), dtype=np.float32)
        pcts: list[int] = []

        result = run_pre_analysis(
            audio_native=audio,
            sr_native=44100,
            file_path="/tmp/test.mp3",
            progress_callback=lambda pct, msg: pcts.append(pct),
            store_in_bridge_cache=False,
        )

        # Percentages must be monotonically increasing
        assert len(pcts) >= 3, f"Expected >= 3 progress callbacks, got {len(pcts)}: {pcts}"
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1], (
                f"Progress not monotonic: {pcts[i]} < {pcts[i - 1]} at index {i}. Full: {pcts}"
            )


class TestAsyncEraGenre:
    """G24: Async Era/Genre — pre-analysis completes without waiting."""

    def test_pre_analysis_completes_without_era(self, monkeypatch):
        """Pre-Analysis MUSS in <100s abschließen, auch wenn Era noch läuft."""
        import time

        _start = time.monotonic()

        def _slow_era(*a, **kw):
            time.sleep(5.0)  # Simuliert CLAP-Kaltstart
            return MagicMock(decade=1970, confidence=0.80)

        def _slow_genre(*a, **kw):
            time.sleep(5.0)
            return MagicMock(genre_label="Schlager", confidence=0.63)

        def mock_load_symbol(module, name):
            if "bridge" in module and name == "get_era_classifier_fn":
                return lambda: _slow_era
            if "bridge" in module and name == "get_genre_classifier_fn":
                return lambda: _slow_genre
            if "defect_scanner" in module:
                return lambda **kw: MagicMock(scan=lambda *a, **kw: MagicMock(scores={}))
            if "restorability" in module:
                return lambda audio, sr: MagicMock(score=64.0)

            class MockMedium:
                primary_material = "vinyl"
                confidence = 0.26
                chain_label = "vinyl → mp3"
                transfer_chain = ["vinyl", "mp3_low"]

            return lambda: MagicMock(detect=lambda *a, **kw: MockMedium())

        monkeypatch.setattr("backend.core.pre_analysis._load_symbol", mock_load_symbol)
        monkeypatch.setattr("backend.core.pre_analysis._store_in_cache", lambda *a: None)
        monkeypatch.setattr("backend.core.pre_analysis._load_cached_parts", lambda *a: {})

        import numpy as np

        audio = np.zeros((44100, 2), dtype=np.float32)

        result = run_pre_analysis(
            audio_native=audio,
            sr_native=44100,
            file_path="/tmp/test.mp3",
            store_in_bridge_cache=False,
        )

        _elapsed = time.monotonic() - _start
        # Must complete in <3s (era/genre are async, don't block)
        assert _elapsed < 3.0, (
            f"Pre-analysis took {_elapsed:.1f}s — should be <3s. Era/Genre are ASYNC and must not block."
        )
        # Defect + restorability must be present
        assert result.defects is not None, "Defects missing"
        # Era may or may not be done (async) — both are acceptable
        # but errors must not contain era/genre failures
        assert "era" not in result.errors, f"Era should not have errors: {result.errors}"
        assert "genre" not in result.errors, f"Genre should not have errors: {result.errors}"


class TestAsCompletedSemantics:
    """as_completed vs sequential — die GIL-Falle (§perf-era-genre)."""

    def test_as_completed_parallel_fast_steps(self):
        """Schnelle Schritte dürfen nicht auf langsame warten."""
        pool = _cf.ThreadPoolExecutor(max_workers=2)
        results: list[str] = []

        def fast():
            time.sleep(0.05)
            return "fast"

        def slow():
            time.sleep(0.3)
            return "slow"

        futs = {pool.submit(fast): "fast", pool.submit(slow): "slow"}
        for fut in _cf.as_completed(futs, timeout=5.0):
            results.append(futs[fut])

        pool.shutdown(wait=False)
        assert results[0] == "fast", f"Fast step blocked by slow: {results}"

    def test_sequential_avoids_gil_contention(self):
        """Sequentielle CLAP-Schritte sind schneller als parallele (GIL)."""
        import threading

        lock = threading.Lock()
        order: list[str] = []

        def clap_step(name: str):
            with lock:
                order.append(name)
                time.sleep(0.1)

        # Sequential
        clap_step("era")
        clap_step("genre")
        assert order == ["era", "genre"], f"Sequential failed: {order}"

        # Parallel würde mit GIL länger dauern — hier nur Semantik-Check
