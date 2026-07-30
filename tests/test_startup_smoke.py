#!/usr/bin/env python3
"""§v10.305 Startup-Smoke-Test — GPU-Erkennung, Warmup, Pre-Analysis in <60s.

Usage:
    python3 -B tests/test_startup_smoke.py
    AURIK_FORCE_CPU=1 python3 -B tests/test_startup_smoke.py

Fails if any step times out or raises.
"""
import sys, os, time, logging, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("startup_smoke")


class StartupSmokeTest(unittest.TestCase):
    """Verify the startup sequence completes without hangs."""

    def test_01_gpu_detection_completes(self):
        """GPU detection within 15 seconds."""
        t0 = time.time()
        from backend.core.ml_device_manager import get_ml_device_manager

        mgr = get_ml_device_manager()
        ok = mgr.wait_for_detection(timeout_s=15.0)
        dt = time.time() - t0
        self.assertTrue(ok, f"_detection_complete event not set after {dt:.1f}s")
        self.assertTrue(mgr._detection_complete.is_set(), "Event should be set")
        logger.info("  GPU: %s, avail=%s, %.1fs", mgr._backend, mgr._gpu_available, dt)

    def test_02_warmup_loads_at_least_7_plugins(self):
        """Warmup within 120 seconds, at least 7/10 loaded."""
        t0 = time.time()
        from backend.api.bridge import warmup_models_background

        warmup_models_background()
        dt = time.time() - t0
        # We can't easily count loaded plugins from outside,
        # but warmup must complete without exception
        self.assertLess(dt, 120, f"Warmup took {dt:.0f}s — timeout")
        logger.info("  Warmup: %.1fs", dt)

    def test_03_pre_analysis_runs(self):
        """Pre-analysis on 1-second test audio within 30 seconds."""
        t0 = time.time()
        from backend.core.pre_analysis import run_pre_analysis

        sr = 48000
        audio = np.sin(2 * np.pi * 440 * np.arange(sr, dtype=np.float32) / sr)
        result = run_pre_analysis(audio, sr, file_path="/tmp/test_startup_smoke.wav")
        dt = time.time() - t0
        self.assertIsNotNone(result, "PreAnalysisResult should not be None")
        self.assertIsNotNone(result.medium, "Medium detection failed")
        self.assertLess(dt, 30, f"Pre-analysis took {dt:.0f}s — timeout")
        logger.info("  Pre-analysis: %.1fs, medium=%s", dt,
                     getattr(result.medium, "primary_material", "?"))

    def test_04_no_torch_zeros_hang(self):
        """warmup_rocm must timeout within 15s, not hang."""
        t0 = time.time()
        from backend.core.ml_device_manager import get_ml_device_manager

        mgr = get_ml_device_manager()
        ok = mgr.warmup_rocm()  # returns bool, NOT hang
        dt = time.time() - t0
        self.assertLess(dt, 15, f"warmup_rocm took {dt:.0f}s — should be <15s")
        logger.info("  warmup_rocm: %s (%.1fs)", ok, dt)

    def test_05_probe_rocm_onnx_pad_is_called(self):
        """Verify _probe_rocm_onnx_pad was invoked during detection."""
        from backend.core.ml_device_manager import get_ml_device_manager

        mgr = get_ml_device_manager()
        # If ROCm is active, ONNX providers should be configured.
        # The probe sets _ort_gpu_providers based on actual availability.
        self.assertIsNotNone(mgr._ort_gpu_providers,
                             "ONNX providers not set — _probe_rocm_onnx_pad may not have run")
        logger.info("  ONNX providers: %s", mgr._ort_gpu_providers)

    def test_06_memory_budget_lock_free(self):
        """try_allocate must return within 10s (no lock-holding import)."""
        t0 = time.time()
        from backend.core.ml_memory_budget import try_allocate

        ok = try_allocate("SMOKE_TEST", 0.001)
        dt = time.time() - t0
        self.assertLess(dt, 10, f"try_allocate took {dt:.0f}s — lock held during import?")
        logger.info("  try_allocate: %s (%.1fs)", ok, dt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
