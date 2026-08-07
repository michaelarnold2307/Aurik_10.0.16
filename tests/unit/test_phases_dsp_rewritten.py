"""
Unit-Tests für DSP-neugeschriebene Phasen mit PhaseResult-Return-Fix.

Abgedeckte Phasen:
  Phase 10  — Multi-Band Parallel Compression (PhaseResult-Return-Fix)
  Phase 34  — Mid/Side Processing (PhaseResult-Return-Fix)

Alle Tests laufen auf synthetischen Arrays (48000 Hz).
Phasen 43–53 werden kanonisch in test_phases_late_ext.py getestet.
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_interface import PhaseResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SR = 48000
_N = SR // 4  # 12000 Samples (0.25s) — statt 1s vermeidet unnötige DSP-Arbeit


@pytest.fixture(scope="class")
def mono():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal(_N).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="class")
def stereo():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal((_N, 2)).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="class")
def stereo_quiet():
    """Sehr leises Stereo-Signal (für De-Esser über Schwelle)."""
    rng = np.random.default_rng(99)
    return np.clip(rng.standard_normal((_N, 2)).astype(np.float32) * 0.05, -1.0, 1.0)


def _assert_phase_result(result, orig_audio, check_clipping: bool = True):
    """Gemeinsame Validierung für alle PhaseResult-Objekte."""
    assert isinstance(result, PhaseResult), f"Erwartet PhaseResult, got {type(result)}"
    assert result.success is True, f"success=False: {result}"
    assert isinstance(result.audio, np.ndarray), "audio muss ndarray sein"
    assert result.audio.shape == orig_audio.shape, f"Audio-Shape geändert: {orig_audio.shape} → {result.audio.shape}"
    assert result.audio.dtype == orig_audio.dtype, f"Dtype geändert: {orig_audio.dtype} → {result.audio.dtype}"
    assert isinstance(result.metadata, dict), "metadata muss dict sein"
    assert isinstance(result.metrics, dict), "metrics muss dict sein"
    assert float(result.execution_time_seconds) >= 0.0, "execution_time muss ≥ 0 sein"
    if check_clipping:
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-4, "Clipping: Audio > 1.0"


# ---------------------------------------------------------------------------
# Phase 10: Multi-Band Compression
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase10Compression:
    def setup_method(self):
        from backend.core.phases.phase_10_compression import CompressionPhase

        self.phase = CompressionPhase()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)  # Makeup-Gain kann > 1.0

    def test_metadata_keys(self, stereo):
        result = self.phase.process(stereo, SR)
        assert "rms_change_db" in result.metrics or "rms_change_db" in result.metadata

    def test_rms_changes(self, stereo):
        result = self.phase.process(stereo, SR)
        # Kompressor sollte RMS leicht reduzieren (negative rms_change_db)
        rms_change = result.metrics.get("rms_change_db", result.metadata.get("rms_change_db", 0))
        assert float(rms_change) <= 2.0, "Kompressor erhöht RMS stark — ungewöhnlich"

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, strength=0.0)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=0.02)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 33: Stereo Width Limiter
# ---------------------------------------------------------------------------


class TestPhase33StereoWidthLimiter:
    def setup_method(self):
        from backend.core.phases.phase_33_stereo_width_limiter import StereoWidthLimiterPhaseV2

        self.phase = StereoWidthLimiterPhaseV2()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)  # type: ignore[arg-type]
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=0.0)  # type: ignore[arg-type]
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=0.02)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)  # type: ignore[arg-type]
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 34: Mid/Side Processing
# ---------------------------------------------------------------------------


class TestPhase34MidSide:
    def setup_method(self):
        from backend.core.phases.phase_34_mid_side_processing import MidSideProcessing

        self.phase = MidSideProcessing()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo, check_clipping=False)  # M/S-Dynamik kann > 1.0

    def test_mono_compatibility_metric(self, stereo):
        result = self.phase.process(stereo, SR)
        mc = result.metrics.get("mono_compatibility")
        assert mc is not None, "mono_compatibility-Metrik fehlt"
        assert 0.0 <= float(mc) <= 1.5, f"mono_compatibility außerhalb Bereich: {mc}"

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=0.0)  # type: ignore[arg-type]
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=0.02)
        assert result.metadata.get("processing") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)  # type: ignore[arg-type]
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 35: Multiband Compression
# ---------------------------------------------------------------------------


class TestPhase35MultibandCompression:
    def setup_method(self):
        from backend.core.phases.phase_35_multiband_compression import MultibandCompressionPhase

        self.phase = MultibandCompressionPhase()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL)
        _assert_phase_result(result, stereo, check_clipping=False)

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, stereo, check_clipping=False)
        assert np.allclose(result.audio, stereo, atol=0.02)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 36: Transient Shaper
# ---------------------------------------------------------------------------


class TestPhase36TransientShaper:
    def setup_method(self):
        from backend.core.phases.phase_36_transient_shaper import TransientShaper

        self.phase = TransientShaper()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=0.02)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 37: Bass Enhancement
# ---------------------------------------------------------------------------


class TestPhase37BassEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_37_bass_enhancement import BassEnhancement

        self.phase = BassEnhancement()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=0.02)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 38: Presence Boost
# ---------------------------------------------------------------------------


class TestPhase38PresenceBoost:
    def setup_method(self):
        from backend.core.phases.phase_38_presence_boost import PresenceBoost

        self.phase = PresenceBoost()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)  # type: ignore[arg-type]
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)  # type: ignore[arg-type]
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=0.02)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)  # type: ignore[arg-type]
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 39: Air Band Enhancement
# ---------------------------------------------------------------------------


class TestPhase39AirBandEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_39_air_band_enhancement import AirBandEnhancement

        self.phase = AirBandEnhancement()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL)
        _assert_phase_result(result, mono, check_clipping=False)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=0.0)
        _assert_phase_result(result, mono, check_clipping=False)
        assert np.allclose(result.audio, mono, atol=0.02)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, MaterialType.VINYL, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6
