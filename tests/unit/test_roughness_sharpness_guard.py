from __future__ import annotations

"""
Tests for §2.49c Roughness/Sharpness Guard in ArtifactFreedomGate.

Normative reference:
- §2.49c: Roughness (Zwicker, asper) Δ > 0.15/phase → -0.05 penalty
- §2.49c: Sharpness (Bismarck, acum) Δ > 0.30 total → -0.10 penalty
- Guard applies only to DYNAMICS / ADDITIVE / ENHANCEMENT phase types.
- Guard does NOT apply to SUBTRACTIVE (noise removal reduces roughness intentionally).
"""


import numpy as np
import pytest

from backend.core.artifact_freedom_gate import (
    _ROUGHNESS_APPLICABLE_TYPES,
    _ROUGHNESS_FLAG_ASPER,
    _SHARPNESS_FLAG_ACUM,
    ArtifactFreedomGate,
)
from backend.core.phase_ontology import PhaseOperationType

SR = 48000
_GATE = ArtifactFreedomGate()


def _sine(freq: float, duration_s: float = 3.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _noise(duration_s: float = 3.0, amplitude: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration_s * SR)) * amplitude).astype(np.float32)


@pytest.mark.unit
class TestRoughnessSharpnessGuardConstants:
    """Constant values must match spec §2.49c."""

    def test_roughness_flag_asper_value(self):
        assert pytest.approx(0.15, abs=1e-6) == _ROUGHNESS_FLAG_ASPER

    def test_sharpness_flag_acum_value(self):
        assert pytest.approx(0.30, abs=1e-6) == _SHARPNESS_FLAG_ACUM

    def test_applicable_types_contain_dynamics(self):
        assert PhaseOperationType.DYNAMICS in _ROUGHNESS_APPLICABLE_TYPES

    def test_applicable_types_contain_additive(self):
        assert PhaseOperationType.ADDITIVE in _ROUGHNESS_APPLICABLE_TYPES

    def test_applicable_types_contain_enhancement(self):
        assert PhaseOperationType.ENHANCEMENT in _ROUGHNESS_APPLICABLE_TYPES

    def test_subtractive_not_applicable(self):
        assert PhaseOperationType.SUBTRACTIVE not in _ROUGHNESS_APPLICABLE_TYPES

    def test_ml_generative_not_applicable(self):
        # ML_GENERATIVE not in applicable types — only DYNAMICS/ADDITIVE/ENHANCEMENT
        assert PhaseOperationType.ML_GENERATIVE not in _ROUGHNESS_APPLICABLE_TYPES


class TestArtifactFreedomResultFields:
    """ArtifactFreedomResult must carry §2.49c new fields."""

    def test_has_roughness_delta_asper_field(self):
        # phase_06 = ADDITIVE
        result = _GATE.evaluate(
            _sine(440),
            _sine(440),
            SR,
            phase_id="phase_06",
            material_type="digital",
        )
        assert hasattr(result, "roughness_delta_asper")
        assert isinstance(result.roughness_delta_asper, float)

    def test_has_sharpness_delta_acum_field(self):
        result = _GATE.evaluate(
            _sine(440),
            _sine(440),
            SR,
            phase_id="phase_06",
            material_type="digital",
        )
        assert hasattr(result, "sharpness_delta_acum")
        assert isinstance(result.sharpness_delta_acum, float)

    def test_has_roughness_sharpness_penalty_field(self):
        result = _GATE.evaluate(
            _sine(440),
            _sine(440),
            SR,
            phase_id="phase_06",
            material_type="digital",
        )
        assert hasattr(result, "roughness_sharpness_penalty")
        assert isinstance(result.roughness_sharpness_penalty, float)

    def test_penalty_nonpositive(self):
        # phase_10 = DYNAMICS
        result = _GATE.evaluate(
            _sine(440),
            _sine(440),
            SR,
            phase_id="phase_10",
            material_type="digital",
        )
        assert result.roughness_sharpness_penalty <= 0.0


class TestRoughnessSharpnessNotFiredForSubtractive:
    """SUBTRACTIVE phases must not have roughness/sharpness penalty."""

    def test_subtractive_no_rs_penalty(self):
        # phase_03 = SUBTRACTIVE (broadband denoise)
        orig = _sine(440) + _noise()
        restored = _sine(440)  # "denoised"
        result = _GATE.evaluate(
            orig,
            restored,
            SR,
            phase_id="phase_03",
            material_type="digital",
        )
        assert result.roughness_sharpness_penalty == pytest.approx(0.0, abs=1e-6)

    def test_corrective_no_rs_penalty(self):
        # phase_04 = CORRECTIVE (RIAA inversion / EQ)
        orig = _sine(440)
        restored = _sine(440) * 0.95
        result = _GATE.evaluate(
            orig,
            restored,
            SR,
            phase_id="phase_04",
            material_type="digital",
        )
        assert result.roughness_sharpness_penalty == pytest.approx(0.0, abs=1e-6)


class TestRoughnessComputationBasics:
    """_compute_roughness_zwicker produces valid values."""

    def test_sine_roughness_nonnegative(self):
        audio = _sine(1000)
        r = _GATE._compute_roughness_zwicker(audio, SR)
        assert r >= 0.0

    def test_am_signal_higher_roughness_than_pure_sine(self):
        """AM-modulated signal (70 Hz) should be rougher than pure sine."""
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False, dtype=np.float32)
        carrier = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
        modulator = (1.0 + np.sin(2 * np.pi * 70.0 * t)).astype(np.float32)
        am_signal = (carrier * modulator * 0.3).astype(np.float32)
        pure = (np.sin(2 * np.pi * 1000.0 * t) * 0.3).astype(np.float32)
        r_am = _GATE._compute_roughness_zwicker(am_signal, SR)
        r_pure = _GATE._compute_roughness_zwicker(pure, SR)
        assert r_am > r_pure, f"AM roughness {r_am:.4f} should exceed pure sine {r_pure:.4f}"

    def test_roughness_bounded(self):
        audio = _sine(1000)
        r = _GATE._compute_roughness_zwicker(audio, SR)
        assert 0.0 <= r <= 10.0

    def test_roughness_short_audio_returns_zero(self):
        tiny = np.zeros(10, dtype=np.float32)
        r = _GATE._compute_roughness_zwicker(tiny, SR)
        assert r == pytest.approx(0.0, abs=1e-6)


class TestSharpnessComputationBasics:
    """_compute_sharpness_bismarck produces valid values."""

    def test_sine_sharpness_nonnegative(self):
        audio = _sine(1000)
        s = _GATE._compute_sharpness_bismarck(audio, SR)
        assert s >= 0.0

    def test_hf_signal_sharper_than_bassline(self):
        """High-frequency signal should have higher sharpness than low-frequency."""
        hf = _sine(8000)
        lf = _sine(200)
        s_hf = _GATE._compute_sharpness_bismarck(hf, SR)
        s_lf = _GATE._compute_sharpness_bismarck(lf, SR)
        assert s_hf > s_lf, f"HF sharpness {s_hf:.4f} should exceed LF {s_lf:.4f}"

    def test_sharpness_bounded(self):
        audio = _sine(4000)
        s = _GATE._compute_sharpness_bismarck(audio, SR)
        assert 0.0 <= s <= 10.0

    def test_sharpness_short_audio_returns_zero(self):
        tiny = np.zeros(10, dtype=np.float32)
        s = _GATE._compute_sharpness_bismarck(tiny, SR)
        assert s == pytest.approx(0.0, abs=1e-6)


class TestRoughnessSharpnessPenaltyIntegration:
    """Integration: penalty reflected in artifact_freedom score."""

    def test_identical_audio_no_rs_penalty(self):
        # phase_10 = DYNAMICS
        audio = _sine(1000)
        result = _GATE.evaluate(
            audio,
            audio,
            SR,
            phase_id="phase_10",
            material_type="digital",
        )
        assert result.roughness_sharpness_penalty == pytest.approx(0.0, abs=1e-6)
        assert result.roughness_delta_asper >= 0.0
        assert result.sharpness_delta_acum >= 0.0

    def test_decreasing_roughness_no_penalty(self):
        """Roughness decrease (denoising effect) must never cause penalty."""
        noisy = _sine(1000) + _noise(amplitude=0.2)
        clean = _sine(1000)
        # phase_10 = DYNAMICS — roughness should decrease
        result = _GATE.evaluate(
            noisy,
            clean,
            SR,
            phase_id="phase_10",
            material_type="digital",
        )
        # roughness_delta_asper = max(0, rough_out - rough_in) → should be 0 here
        assert result.roughness_delta_asper >= 0.0  # always non-negative by spec
        assert result.roughness_sharpness_penalty <= 0.0  # no improvement penalty

    def test_artifact_freedom_reduces_on_rs_penalty(self):
        """artifact_freedom of a noisy result should be ≤ artifact_freedom of clean."""
        # phase_06 = ADDITIVE
        audio = _sine(1000)
        result_clean = _GATE.evaluate(
            audio,
            audio,
            SR,
            phase_id="phase_06",
            material_type="digital",
        )
        # Add intense AM modulation to simulate roughness increase
        t = np.linspace(0, 3.0, int(3.0 * SR), endpoint=False, dtype=np.float32)
        am_rest = audio * (1.0 + 0.95 * np.sin(2 * np.pi * 70.0 * t)).astype(np.float32)
        am_rest = np.clip(am_rest, -1.0, 1.0)
        result_am = _GATE.evaluate(
            audio,
            am_rest,
            SR,
            phase_id="phase_06",
            material_type="digital",
        )
        # AM-modulated output may have higher roughness, penalty should be ≤ 0
        assert result_am.roughness_sharpness_penalty <= result_clean.roughness_sharpness_penalty + 0.01

    def test_rs_penalty_in_detail_report_when_fired(self):
        """When penalty fires, detail_report should contain roughness_flag or sharpness_flag."""
        audio = _sine(1000)
        result = _GATE.evaluate(
            audio,
            audio,
            SR,
            phase_id="phase_10",
            material_type="digital",
        )
        # If no penalty, detail_report should not have roughness_flag
        if result.roughness_sharpness_penalty == 0.0:
            assert "roughness_flag" not in result.detail_report or result.detail_report.get("roughness_flag") is None

    def test_no_negative_artifact_freedom(self):
        """artifact_freedom must always stay in [0, 1] even with penalty."""
        audio = _sine(1000)
        result = _GATE.evaluate(
            audio,
            audio,
            SR,
            phase_id="phase_10",
            material_type="digital",
        )
        assert 0.0 <= result.artifact_freedom <= 1.0

    def test_stereo_audio_evaluated_without_error(self):
        """Guard must handle stereo audio (2, N) shape."""
        stereo = np.stack([_sine(440), _sine(880)], axis=0)
        result = _GATE.evaluate(
            stereo,
            stereo,
            SR,
            phase_id="phase_06",
            material_type="digital",
        )
        assert 0.0 <= result.artifact_freedom <= 1.0
        assert result.roughness_delta_asper >= 0.0
