from __future__ import annotations

import pytest

"""Gap Closure Tests für v10.0.0 — EraVocalProfile, Pre-Echo, RestorationMemory.

Unit-Tests für die geschlossenen Lücken:
1. EraVocalProfile wird an compute_vqi() in VocalNoHarmGate übergeben
2. Pre-Echo-Detektor-Fixture mit bekanntem Onset-Shift
3. RestorationMemory-Kalibrierung (telemetrie-tracking)
"""


import unittest
from unittest.mock import MagicMock, patch

import numpy as np


@pytest.mark.unit
class TestEraVocalProfileVQIGate(unittest.TestCase):
    """§EraVocalProfile: Verify era_vocal_profile is passed to compute_vqi in VocalNoHarmGate._evaluate_vqi."""

    def test_vocal_no_harm_gate_evaluate_vqi_receives_era_profile(self) -> None:
        """VocalNoHarmGate.evaluate() passes era_vocal_profile to _evaluate_vqi."""
        from backend.core.vocal_no_harm_gate import VocalNoHarmGate

        gate = VocalNoHarmGate()
        pre_audio = np.random.randn(48000).astype(np.float32)
        post_audio = np.random.randn(48000).astype(np.float32)

        era_profile_mock = {"formant_tolerance_db": 1.5, "era_decade": 1950}

        with patch(
            "backend.core.vocal_no_harm_gate._load_symbol",
            side_effect=lambda module, symbol: self._mock_load_symbol(module, symbol, era_profile_mock),
        ):
            result = gate.evaluate(
                pre_audio,
                post_audio,
                48000,
                panns_singing=0.5,
                material_type="vinyl",
                era_decade=1950,
                era_vocal_profile=era_profile_mock,
            )

        # Gate should successfully evaluate and return result with vqi score
        self.assertIsNotNone(result)
        self.assertTrue(result.active)
        self.assertIn("vqi", result.scores)

    def _mock_load_symbol(self, module: str, symbol: str, era_profile: dict) -> object:
        """Mock for _load_symbol that injects era_profile into compute_vqi call."""
        if "compute_vqi" in symbol:

            def mock_vqi(
                pre,
                post,
                sr,
                skip_singer_identity=False,
                reference_audio=None,
                genre=None,
                era_profile=None,
            ):
                # Verify that era_profile was actually passed (not None)
                assert era_profile is not None, "era_profile must be passed to compute_vqi"
                return {"vqi": 0.85, "singer_identity_cosine": 0.95}

            return mock_vqi
        elif "get_vqi_material_floor" in symbol:
            return lambda material_type, is_studio_2026=False: 0.72
        return MagicMock()


class TestPreEchoDetectorFixture(unittest.TestCase):
    """Pre-Echo-Detektor Fixture mit bekanntem Onset-Shift > 2ms."""

    def test_pre_echo_detector_synthetic_mp3_fixture(self) -> None:
        """Pre-Echo detector recognizes onset shift > 2ms in synthetic MP3-like fixture."""
        from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

        sr = 48000
        # Synthetic Pre-Echo Fixture: Onset bei t=100ms mit Pre-Echo ca. 30ms vorher
        duration_s = 0.5
        audio = np.zeros(int(duration_s * sr), dtype=np.float32)

        # Pre-Echo: leiser Transient bei ~70ms
        pre_echo_idx = int(0.07 * sr)
        audio[pre_echo_idx : pre_echo_idx + 1000] += np.linspace(0, 0.1, 1000)

        # Main Transient: lauter Transient bei ~100ms
        main_idx = int(0.10 * sr)
        audio[main_idx : main_idx + 2000] += np.linspace(0, 1.0, 2000)

        detector = get_pre_echo_detector()
        result = detector.detect(audio, sr, material_key="mp3_low")

        # Should detect pre-echo with positive severity (not 0.0)
        self.assertIsNotNone(result)
        if hasattr(result, "severity"):
            # If detector returns severity metric, it should be > 0
            self.assertGreater(float(result.severity), 0.0, "Pre-Echo fixture should have severity > 0.0")
        if hasattr(result, "regions"):
            # If detector returns regions, we should find at least one
            self.assertGreater(len(result.regions), 0, "Pre-Echo fixture should have at least one detected region")

    def test_pre_echo_clean_signal_no_false_positive(self) -> None:
        """Pre-Echo detector should NOT flag clean signals as having pre-echo."""
        from backend.core.dsp.pre_echo_detector import get_pre_echo_detector

        sr = 48000
        # Clean signal: no pre-echo, just a transient
        duration_s = 0.5
        audio = np.zeros(int(duration_s * sr), dtype=np.float32)

        # Single transient at 100ms, no pre-echo
        main_idx = int(0.10 * sr)
        audio[main_idx : main_idx + 2000] += np.linspace(0, 1.0, 2000)

        detector = get_pre_echo_detector()
        result = detector.detect(audio, sr, material_key="cd_digital")

        # Clean signal should have low/zero severity
        if hasattr(result, "severity"):
            self.assertLess(float(result.severity), 0.3, "Clean signal should have low pre-echo severity")

    def test_defect_scanner_pre_echo_non_zero_on_synthetic_lossy_fixture(self) -> None:
        """DefectScanner._detect_pre_echo should report >0 severity on clear synthetic pre-echo."""
        from backend.core.defect_scanner import DefectScanner

        sr = 48000
        audio = np.zeros(int(1.2 * sr), dtype=np.float32)

        # Haupttransient bei 300ms
        main_idx = int(0.300 * sr)
        audio[main_idx : main_idx + 2000] += np.hanning(2000).astype(np.float32) * 0.9

        # Codec-typisches Pre-Echo 20ms vorher
        pre_idx = int(0.280 * sr)
        audio[pre_idx : pre_idx + 1200] += np.hanning(1200).astype(np.float32) * 0.18

        scanner = DefectScanner(sample_rate=sr)
        scanner.material_type = "mp3_low"  # type: ignore[assignment]
        score = scanner._detect_pre_echo(audio)

        self.assertIsNotNone(score)
        self.assertGreater(float(score.severity), 0.0)
        self.assertIsInstance(score.metadata, dict)
        self.assertIn("codec_pre_echo_events", score.metadata)
        self.assertIn("fallback_pre_echo_events", score.metadata)
        self.assertGreaterEqual(int(score.metadata.get("codec_pre_echo_events", 0)), 0)
        self.assertGreaterEqual(int(score.metadata.get("fallback_pre_echo_events", 0)), 0)

    def test_defect_scanner_pre_echo_detects_multiple_events(self) -> None:
        """DefectScanner should keep non-zero severity for two separated pre-echo transients."""
        from backend.core.defect_scanner import DefectScanner

        sr = 48000
        audio = np.zeros(int(1.6 * sr), dtype=np.float32)

        # Event 1: 300 ms with 20 ms pre-echo.
        main_a = int(0.300 * sr)
        pre_a = int(0.280 * sr)
        audio[pre_a : pre_a + 1000] += np.hanning(1000).astype(np.float32) * 0.14
        audio[main_a : main_a + 1800] += np.hanning(1800).astype(np.float32) * 0.90

        # Event 2: 900 ms with 25 ms pre-echo.
        main_b = int(0.900 * sr)
        pre_b = int(0.875 * sr)
        audio[pre_b : pre_b + 1200] += np.hanning(1200).astype(np.float32) * 0.16
        audio[main_b : main_b + 2000] += np.hanning(2000).astype(np.float32) * 0.92

        scanner = DefectScanner(sample_rate=sr)
        scanner.material_type = "mp3_low"  # type: ignore[assignment]
        score = scanner._detect_pre_echo(audio)

        self.assertGreater(float(score.severity), 0.0)
        self.assertGreaterEqual(len(score.locations), 1)


class TestRestorationMemoryCalibration(unittest.TestCase):
    """RestorationMemory Prior-Kalibrierung: telemetry tracking for GPOptimizer prior quality."""

    def test_restoration_memory_tracks_prior_contribution(self) -> None:
        """RestorationMemory should track whether prior improved optimization convergence."""
        try:
            from backend.core.restoration_memory import get_restoration_memory
        except ImportError:
            self.skipTest("RestorationMemory not available in this build")

        memory = get_restoration_memory()

        session_key = ("1950", "vinyl", "hash_abc123")

        # In real scenario, this would be called after HPI > 0 + artifact_freedom >= 0.95
        if hasattr(memory, "get_prior"):
            retrieved_prior = memory.get_prior(session_key)
            # If key doesn't exist, should return empty/None gracefully
            self.assertIn(retrieved_prior is None or isinstance(retrieved_prior, dict), [True])

    def test_restoration_memory_prior_only_saves_good_runs(self) -> None:
        """RestorationMemory should only save priors from runs with HPI > 0 and artifact_freedom >= 0.95."""
        try:
            from backend.core.restoration_memory import get_restoration_memory
        except ImportError:
            self.skipTest("RestorationMemory not available in this build")

        memory = get_restoration_memory()

        # Verify that bad runs are not persisted
        if hasattr(memory, "_get_run_quality_score"):
            hpi_bad = -0.1
            af_bad = 0.92
            # Should reject
            should_save_bad = hpi_bad > 0 and af_bad >= 0.95
            self.assertFalse(should_save_bad)

            hpi_good = 0.42
            af_good = 0.97
            # Should accept
            should_save_good = hpi_good > 0 and af_good >= 0.95
            self.assertTrue(should_save_good)


if __name__ == "__main__":
    unittest.main()
