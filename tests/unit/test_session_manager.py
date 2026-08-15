"""Session-Manager-Test. Spec 15 paragraph 9.5.
Testet: Acquire/Release, LRU-Eviction, Memory-Limit, Concurrent-Access, Batch-Recycling.

Autor: Aurik 10
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestSessionManager:
    """Testet den InferenceSessionManager."""

    @pytest.fixture(autouse=True)
    def _stub_load(self):
        """Spec 15 §9.5: Diese Tests pruefen Cache-Semantik (LRU, Memory,
        Threading, Recycling) — nicht das echte ONNX-Laden. Der Ladevorgang
        wird daher gestubbt; echte fehlende Modelle fallen weiterhin laut
        in _load_session (onnxruntime NoSuchFile) durch.
        """
        from backend.core.ml.session_manager import InferenceSessionManager

        with patch.object(
            InferenceSessionManager,
            "_load_session",
            return_value=(MagicMock(), 1.0),
        ):
            yield

    def test_acquire_release(self):
        """Acquire/Release-Zyklus."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=2)
        sid = mgr.acquire("test_model", model_path="mock.onnx")
        assert sid is not None
        mgr.release("test_model")
        assert mgr.get_active_count() == 0

    def test_lru_eviction(self):
        """LRU-Eviction: aelteste Session wird verdraengt."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=2)
        mgr.acquire("m1", model_path="m1.onnx")
        mgr.acquire("m2", model_path="m2.onnx")
        mgr.acquire("m3", model_path="m3.onnx")  # Should evict m1
        assert "m1" not in mgr._cache

    def test_memory_limit(self):
        """Memory-Limit-Warnung."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=10, memory_limit_mb=1.0)
        with patch.object(mgr, "get_total_memory_mb", return_value=2500.0):
            mgr.acquire("big_model", model_path="big.onnx")
            assert mgr.get_total_memory_mb() > mgr.memory_limit_mb

    def test_concurrent_access(self):
        """Concurrent-Access: Thread-sicherer Zugriff."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=10)
        errors = []

        def worker(name):
            try:
                mgr.acquire(name, model_path=f"{name}.onnx")
                mgr.release(name)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_batch_recycling(self):
        """Batch-Recycling: Nach N Tracks Sessions leeren."""
        from backend.core.ml.session_manager import InferenceSessionManager

        mgr = InferenceSessionManager(max_sessions=4)
        for i in range(6):
            mgr.acquire(f"b{i}", model_path=f"b{i}.onnx")
        mgr.clear()
        assert mgr.get_active_count() == 0
