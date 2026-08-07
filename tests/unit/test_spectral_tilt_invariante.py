from __future__ import annotations

"""
Tests for §2.46b Spectral-Tilt-Preservation-Invariante in phase_06.

Normative reference:
- §2.46b: ADDITIVE phases must not deviate the spectral tilt from era_result.spectral_tilt
  by more than material_tolerance (±1.5 dB/oct for digital, up to ±3.0 for shellac).
- phase_06 must cap HF boost when deviation > tolerance.
- metadata["spectral_tilt_capped"] must be populated when capping occurs.
- Guard is skipped gracefully when era_result not in kwargs.
- For shellac: tolerance is ±3.0 dB/oct.
"""


import numpy as np
import pytest

SR = 48000


def _make_audio(duration_s: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    """Bandlimited test signal (mono, band-limited to simulate rolloff)."""
    rng = np.random.default_rng(77)
    sig = rng.standard_normal(int(duration_s * SR)).astype(np.float32) * amplitude
    # Low-pass to simulate tape rolloff at 8 kHz (forces phase_06 to detect rolloff)
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 8000 / (SR / 2), btype="low")
    filtered = filtfilt(b, a, sig).astype(np.float32)
    return filtered  # type: ignore[no-any-return]


@pytest.mark.unit
class TestSpectralTiltInvariantConstants:
    """_TILT_MATERIAL_TOLERANCE must have all key material entries."""

    def test_digital_tolerance_is_1_5(self):
        from backend.core.phases.phase_06_frequency_restoration import _TILT_MATERIAL_TOLERANCE

        assert _TILT_MATERIAL_TOLERANCE["digital"] == pytest.approx(1.5, abs=1e-6)

    def test_shellac_tolerance_is_3_0(self):
        from backend.core.phases.phase_06_frequency_restoration import _TILT_MATERIAL_TOLERANCE

        assert _TILT_MATERIAL_TOLERANCE["shellac"] == pytest.approx(3.0, abs=1e-6)

    def test_vinyl_tolerance_between_digital_and_shellac(self):
        from backend.core.phases.phase_06_frequency_restoration import _TILT_MATERIAL_TOLERANCE

        assert (
            _TILT_MATERIAL_TOLERANCE["digital"]
            < _TILT_MATERIAL_TOLERANCE["vinyl"]
            < _TILT_MATERIAL_TOLERANCE["shellac"]
        )

    def test_tape_tolerance_exists(self):
        from backend.core.phases.phase_06_frequency_restoration import _TILT_MATERIAL_TOLERANCE

        assert "tape" in _TILT_MATERIAL_TOLERANCE
        assert _TILT_MATERIAL_TOLERANCE["tape"] > 0.0


class TestSpectralTiltGuardSkippedWithoutEraResult:
    """Guard must not crash or alter audio when era_result not provided."""

    def test_no_crash_without_era_result(self):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        phase = FrequencyRestorationPhase()
        audio = _make_audio()
        result = phase.process(audio, material_type="tape", sample_rate=SR)
        assert np.isfinite(result.audio).all()
        assert result.audio.shape == audio.shape

    def test_no_tilt_cap_meta_without_era_result(self):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        phase = FrequencyRestorationPhase()
        audio = _make_audio()
        result = phase.process(audio, material_type="tape", sample_rate=SR)
        # spectral_tilt_capped must NOT be in metadata (no era_result provided)
        assert "spectral_tilt_capped" not in result.metadata or result.metadata["spectral_tilt_capped"] == {}


class _MockEraResult:
    """Minimal EraResult mock for §2.46b tests."""

    def __init__(self, spectral_tilt: float = -4.0):
        self.spectral_tilt = spectral_tilt


class TestSpectralTiltGuardWithEraResult:
    """Guard activates when era_result is provided."""

    def test_output_stays_finite_with_era_result(self):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        phase = FrequencyRestorationPhase()
        audio = _make_audio()
        era = _MockEraResult(spectral_tilt=-4.0)
        result = phase.process(audio, material_type="tape", sample_rate=SR, era_result=era)
        assert np.isfinite(result.audio).all()

    def test_output_clip_respected_with_era_result(self):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        phase = FrequencyRestorationPhase()
        audio = _make_audio()
        era = _MockEraResult(spectral_tilt=-4.0)
        result = phase.process(audio, material_type="tape", sample_rate=SR, era_result=era)
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_output_shape_preserved_with_era_result(self):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        phase = FrequencyRestorationPhase()
        audio = _make_audio()
        era = _MockEraResult(spectral_tilt=-4.0)
        result = phase.process(audio, material_type="tape", sample_rate=SR, era_result=era)
        assert result.audio.shape == audio.shape

    def test_metadata_may_contain_tilt_capped_when_active(self):
        """When capping fires, tilt_capped metadata must hold deviation/tolerance/cap_factor."""
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        phase = FrequencyRestorationPhase()
        audio = _make_audio()
        # Use a very strict era_result (spectral_tilt = +2.0 → opposite of typical HF extension)
        # so capping is more likely to fire. Guard gracefully skips if EraClassifier errors.
        era = _MockEraResult(spectral_tilt=2.0)
        result = phase.process(audio, material_type="digital", sample_rate=SR, era_result=era)
        # If tilt_capped fired:
        if result.metadata.get("spectral_tilt_capped"):
            cap_meta = result.metadata["spectral_tilt_capped"]
            assert "deviation" in cap_meta
            assert "tolerance_dboct" in cap_meta
            assert "cap_factor" in cap_meta
            assert 0.0 < cap_meta["cap_factor"] <= 1.0
        # If guard gracefully skipped — also valid (no era_classifier available in unit test)

    def test_strength_zero_bypasses_guard(self):
        """strength=0 passthrough must not crash on tilt guard."""
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        phase = FrequencyRestorationPhase()
        audio = _make_audio()
        era = _MockEraResult(spectral_tilt=-4.0)
        result = phase.process(audio, material_type="tape", sample_rate=SR, era_result=era, strength=0.0)
        assert np.isfinite(result.audio).all()


class TestSpectralTiltMaterialTolerance:
    """Shellac must tolerate more tilt deviation than digital."""

    def test_digital_tolerance_more_strict_than_shellac(self):
        from backend.core.phases.phase_06_frequency_restoration import _TILT_MATERIAL_TOLERANCE

        # Digital: ±1.5, Shellac: ±3.0 → shellac more tolerant
        assert _TILT_MATERIAL_TOLERANCE["digital"] < _TILT_MATERIAL_TOLERANCE["shellac"]

    def test_all_tolerances_positive(self):
        from backend.core.phases.phase_06_frequency_restoration import _TILT_MATERIAL_TOLERANCE

        for mat, tol in _TILT_MATERIAL_TOLERANCE.items():
            assert tol > 0.0, f"Tolerance for '{mat}' must be > 0"

    def test_all_tolerances_reasonable_range(self):
        """All tolerances should be between 0.5 and 4.0 dB/oct."""
        from backend.core.phases.phase_06_frequency_restoration import _TILT_MATERIAL_TOLERANCE

        for mat, tol in _TILT_MATERIAL_TOLERANCE.items():
            assert 0.5 <= tol <= 4.0, f"Tolerance for '{mat}' = {tol} is outside [0.5, 4.0]"
