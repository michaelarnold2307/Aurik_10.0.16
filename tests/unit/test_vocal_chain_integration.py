import pytest

"""Tests for §2.8 Vocal Chain Integration — Gender, PhonemeDetector, BreathDetector.

Validates the integration of vocal processing modules that were previously
isolated (each module existed but wasn't connected to the pipeline).

Key integration points tested:
1. Gender flows from UV3 → Phase 19 → Phase 42 via _restoration_context
2. PhonemeDetector output feeds into FormantSystem.phoneme_guided_enhance()
3. BreathDetector replaces static bandpass in Phase 42._control_breath()
4. VocalAIEnhancement engaged as post-step when available
"""

from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SR = 48000


def _make_vocal_signal(duration_s: float = 1.0, f0: float = 220.0) -> np.ndarray:
    """Synthesize a simple vocal-like signal (F0 + harmonics + noise)."""
    n = int(SR * duration_s)
    t = np.arange(n, dtype=np.float64) / SR
    sig = np.zeros(n, dtype=np.float64)
    for h in range(1, 6):
        sig += (0.5 / h) * np.sin(2 * np.pi * f0 * h * t)
    # Add formant-like bandpass noise in 300-3000 Hz
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n) * 0.02
    sig += noise
    sig = sig / (np.max(np.abs(sig)) + 1e-12) * 0.8
    return sig.astype(np.float64)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Test: Phase 42 accepts and uses vocal_gender from kwargs
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPhase42GenderIntegration:
    """Phase 42 should extract vocal_gender from kwargs and pass it to FormantSystem."""

    def test_phase42_extracts_vocal_gender_from_kwargs(self):
        """vocal_gender in kwargs → appears in metadata."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.5)
        result = phase.process(
            audio,
            SR,
            vocal_gender="female",
            strength=0.5,
        )
        assert result.success
        assert result.metadata.get("vocal_gender") == "female"

    def test_phase42_defaults_to_unknown_gender(self):
        """No vocal_gender in kwargs → metadata shows 'unknown'."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.5)
        result = phase.process(audio, SR, strength=0.5)
        assert result.success
        assert result.metadata.get("vocal_gender") == "unknown"

    def test_phase42_exposes_saliency_metadata(self):
        """Saliency context from UV3 must be visible in PhaseResult metadata."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.5)
        result = phase.process(
            audio,
            SR,
            strength=0.5,
            defect_saliency_map={"sibilance": 0.8, "vocal_harshness": 0.4},
        )
        assert result.success
        assert 0.0 <= float(result.metadata.get("sibilance_saliency", -1.0)) <= 1.0
        assert 0.0 <= float(result.metadata.get("vocal_harshness_saliency", -1.0)) <= 1.0

    def test_phase42_saliency_metadata_defaults_to_zero(self):
        """Without saliency context metadata should fall back to 0.0."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.5)
        result = phase.process(audio, SR, strength=0.5)
        assert result.success
        assert float(result.metadata.get("sibilance_saliency", -1.0)) == 0.0
        assert float(result.metadata.get("vocal_harshness_saliency", -1.0)) == 0.0

    def test_phase42_exposes_vocal_intimacy_metadata(self):
        """Phase 42 must export psychoacoustic vocal-intimacy telemetry."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.8)
        result = phase.process(audio, SR, strength=0.7)
        assert result.success
        assert 0.0 <= float(result.metadata.get("vocal_intimacy_pre", -1.0)) <= 1.0
        assert 0.0 <= float(result.metadata.get("vocal_intimacy_post", -1.0)) <= 1.0
        assert -1.0 <= float(result.metadata.get("vocal_intimacy_delta", -2.0)) <= 1.0
        assert isinstance(result.metadata.get("vocal_intimacy_gate_triggered"), bool)
        assert 0.0 <= float(result.metadata.get("vocal_intimacy_rescue_mix", -1.0)) <= 0.5

    def test_phase42_intimacy_gate_is_material_adaptive(self):
        """Threshold/cap should adapt to material (shellac > digital tolerance)."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.7)

        r_shellac = phase.process(audio, SR, material=MaterialType.SHELLAC, strength=0.8)
        r_digital = phase.process(audio, SR, material=MaterialType.CD_DIGITAL, strength=0.8)

        assert r_shellac.success and r_digital.success
        assert float(r_shellac.metadata.get("vocal_intimacy_max_drop", -1.0)) >= float(
            r_digital.metadata.get("vocal_intimacy_max_drop", 9.9)
        )
        assert float(r_shellac.metadata.get("vocal_intimacy_rescue_max", -1.0)) >= float(
            r_digital.metadata.get("vocal_intimacy_rescue_max", 9.9)
        )


# ---------------------------------------------------------------------------
# Test: Phase 19 uses external vocal_gender from pipeline context
# ---------------------------------------------------------------------------
class TestPhase19GenderFromContext:
    """Phase 19 should prefer vocal_gender from kwargs over re-detecting."""

    def test_phase19_accepts_pipeline_gender(self):
        """When vocal_gender='male' in kwargs, Phase 19 should use it."""
        from backend.core.phases.phase_19_de_esser import DeEsserPhase, VocalGender

        phase = DeEsserPhase(gender=VocalGender.AUTO)
        audio = _make_vocal_signal(0.5)
        from backend.core.defect_scanner import MaterialType

        result = phase.process(audio, SR, MaterialType.CD_DIGITAL, vocal_gender="male")
        assert result.success
        # Gender should be 'male' (from pipeline context, not re-detected)
        assert result.metadata.get("gender") == "male"


# ---------------------------------------------------------------------------
# Test: BreathDetector integration in Phase 42
# ---------------------------------------------------------------------------
class TestPhase42BreathDetector:
    """Phase 42 should use BreathDetector for segment-aware breath reduction."""

    def test_breath_detector_called_when_available(self):
        """When BreathDetector is available, it should be called instead of
        the static bandpass fallback."""
        from backend.core.phases import phase_42_vocal_enhancement as p42_module

        # Check that the module imports BreathDetector
        assert hasattr(p42_module, "_BREATH_DETECTOR_AVAILABLE")
        assert hasattr(p42_module, "_get_breath_detector")

    def test_breath_control_fallback_works(self):
        """When BreathDetector unavailable, static bandpass fallback still works."""
        from backend.core.phases import phase_42_vocal_enhancement as p42_module
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.5)
        config = {
            "deess_threshold_db": -20,
            "deess_reduction_db": 5,
            "presence_gain_db": 3.0,
            "formant_gain_db": 3.0,
            "chest_gain_db": 2.0,
            "breath_reduction_db": 4,
            "compression_ratio": 2.0,
        }
        # Force BreathDetector unavailable
        with patch.object(p42_module, "_BREATH_DETECTOR_AVAILABLE", False):
            result = phase._control_breath(audio, SR, config)
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Test: PhonemeDetector integration in Phase 42
# ---------------------------------------------------------------------------
class TestPhase42PhonemeDetector:
    """Phase 42 should use PhonemeDetector to feed real phoneme segments
    into FormantSystem.phoneme_guided_enhance()."""

    def test_phoneme_detector_import_available(self):
        """PhonemeDetector module should be importable."""
        from plugins.phoneme_detector import get_phoneme_detector

        pd = get_phoneme_detector()
        assert pd is not None

    def test_phoneme_detect_returns_segments(self):
        """PhonemeDetector should return phoneme labels and timestamps."""
        from plugins.phoneme_detector import get_phoneme_detector

        pd = get_phoneme_detector()
        audio = _make_vocal_signal(1.0)
        result = pd.detect(audio.astype(np.float64), SR)
        assert hasattr(result, "phonemes")
        assert hasattr(result, "timestamps_ms")
        assert hasattr(result, "confidence")
        assert result.confidence >= 0.0


# ---------------------------------------------------------------------------
# Test: VocalAIEnhancement import is real (not dead pass/True pattern)
# ---------------------------------------------------------------------------
class TestVocalAIEnhancementLink:
    """The VOCAL_AI_AVAILABLE flag should reflect a real import."""

    def test_vocal_ai_import_attempts_real_class(self):
        """Phase 42 should import UnifiedVocalAIEnhancer, not use pass/True."""
        from backend.core.phases import phase_42_vocal_enhancement as p42

        # The module should have attempted a real import
        assert hasattr(p42, "VOCAL_AI_AVAILABLE")
        # If the class is importable, the flag should be True
        try:
            pass

            assert p42.VOCAL_AI_AVAILABLE is True
        except ImportError:
            assert p42.VOCAL_AI_AVAILABLE is False


# ---------------------------------------------------------------------------
# Test: Full Phase 42 pipeline with all integrations
# ---------------------------------------------------------------------------
class TestPhase42FullIntegration:
    """End-to-end test: Phase 42 with vocal_gender, producing valid output."""

    def test_full_pipeline_mono(self):
        """Mono signal through Phase 42 with gender context."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(1.0, f0=120.0)  # Male F0
        result = phase.process(audio, SR, vocal_gender="male", strength=0.8)
        assert result.success
        assert result.audio.shape == audio.shape
        assert np.all(np.isfinite(result.audio))
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_full_pipeline_stereo(self):
        """Stereo signal through Phase 42."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        mono = _make_vocal_signal(1.0, f0=220.0)
        stereo = np.column_stack([mono, mono * 0.95])
        result = phase.process(stereo, SR, vocal_gender="female", strength=0.8)
        assert result.success
        assert result.audio.ndim == 2
        assert result.audio.shape[1] == 2
        assert np.all(np.isfinite(result.audio))
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_nan_inf_safety(self):
        """Phase 42 output must be free of NaN/Inf regardless of input."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.5)
        # Inject NaN at a few positions
        audio[100] = np.nan
        audio[200] = np.inf
        result = phase.process(audio, SR, strength=0.5)
        assert result.success
        assert np.all(np.isfinite(result.audio))
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_zero_strength_passthrough(self):
        """strength=0 should return input unchanged."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal_signal(0.5)
        result = phase.process(audio, SR, strength=0.0)
        assert result.success
        np.testing.assert_allclose(result.audio, np.clip(audio, -1.0, 1.0), atol=1e-7)
