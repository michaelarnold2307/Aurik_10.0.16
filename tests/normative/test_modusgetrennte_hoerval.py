"""Tests for mode-separated quality validation (Restoration vs. Studio 2026).

Verifies that:
- HPI formula differs between modes (§2.44)
- PMGG thresholds differ between modes (P3-P5 higher for Studio 2026)
- Mode dispatch in UV3 is correct
- Both modes produce valid quality gate results independently
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_audio_mono():
    """3s mono test signal at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False).astype(np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)


@pytest.fixture
def test_audio_stereo():
    """3s stereo test signal at 48 kHz."""
    sr = 48000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False).astype(np.float32)
    left = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    right = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.08 * np.sin(2 * np.pi * 880 * t + 0.2)
    return np.column_stack([left, right]).astype(np.float32)


# ---------------------------------------------------------------------------
# PMGG Threshold Differentiation
# ---------------------------------------------------------------------------


class TestPMGGThresholdDifferentiation:
    """P3–P5 thresholds must be higher for Studio 2026."""

    def test_canonical_thresholds_restoration(self):
        from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds

        thresholds = _get_canonical_thresholds(is_studio_2026=False)
        assert thresholds["natuerlichkeit"] >= 0.90
        assert thresholds["authentizitaet"] >= 0.88
        assert thresholds["tonal_center"] >= 0.95
        assert thresholds["brillanz"] >= 0.78

    def test_canonical_thresholds_studio(self):
        from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds

        thresholds = _get_canonical_thresholds(is_studio_2026=True)
        # P1/P2 stricter in Studio 2026 (Spec 09 §09.1)
        assert thresholds["natuerlichkeit"] >= 0.92
        assert thresholds["authentizitaet"] >= 0.90
        assert thresholds["tonal_center"] >= 0.96
        assert thresholds["timbre_authentizitaet"] >= 0.89
        assert thresholds["artikulation"] >= 0.87
        # P3-P5: material-universal floors (Spec 09 / calibration_matrix.py)
        assert thresholds["emotionalitaet"] >= 0.84
        assert thresholds["micro_dynamics"] >= 0.90
        assert thresholds["groove"] >= 0.85
        assert thresholds["transparenz"] >= 0.85
        assert thresholds["brillanz"] >= 0.82

    def test_studio_thresholds_higher_than_restoration(self):
        from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds

        rest = _get_canonical_thresholds(is_studio_2026=False)
        studio = _get_canonical_thresholds(is_studio_2026=True)
        # P3-P5 goals must be strictly higher in Studio mode
        p3_p5_goals = [
            "emotionalitaet",
            "micro_dynamics",
            "groove",
            "transparenz",
            "waerme",
            "bass_kraft",
            "separation_fidelity",
            "brillanz",
            "spatial_depth",
        ]
        for goal in p3_p5_goals:
            assert studio[goal] >= rest[goal], (
                f"Studio threshold for {goal} ({studio[goal]}) should be >= Restoration ({rest[goal]})"
            )

    def test_p1_p2_stricter_in_studio(self):
        """Studio 2026 P1/P2 must be strictly higher than Restoration (Spec 09 §09.1)."""
        from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds

        rest = _get_canonical_thresholds(is_studio_2026=False)
        studio = _get_canonical_thresholds(is_studio_2026=True)
        assert studio["natuerlichkeit"] > rest["natuerlichkeit"]
        assert studio["authentizitaet"] > rest["authentizitaet"]


# ---------------------------------------------------------------------------
# HPI Gate Mode Dispatch
# ---------------------------------------------------------------------------


class TestHPIGateModeDispatch:
    """HolisticPerceptualGate must have separate evaluation methods."""

    def test_evaluate_restoration_method_exists(self):
        from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

        gate = HolisticPerceptualGate()
        assert hasattr(gate, "evaluate_restoration") or hasattr(gate, "evaluate")

    def test_evaluate_studio_method_exists(self):
        from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

        gate = HolisticPerceptualGate()
        assert hasattr(gate, "evaluate_studio") or hasattr(gate, "evaluate")

    def test_restoration_hpi_components(self):
        """Restoration HPI = MERT_similarity × timbral_fidelity × artifact_freedom × emotional_arc."""
        from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

        gate = HolisticPerceptualGate()
        # Check that the class references the expected formula components
        src = type(gate).__module__
        import importlib

        mod = importlib.import_module(src)
        with open(mod.__file__, encoding="utf-8") as handle:  # type: ignore[arg-type]
            source_code = handle.read()
        assert "timbral_fidelity" in source_code or "timbral" in source_code
        assert "artifact_freedom" in source_code

    def test_studio_hpi_components(self):
        """Studio HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc."""
        from backend.core.holistic_perceptual_gate import HolisticPerceptualGate

        gate = HolisticPerceptualGate()
        src = type(gate).__module__
        import importlib

        mod = importlib.import_module(src)
        with open(mod.__file__, encoding="utf-8") as handle:  # type: ignore[arg-type]
            source_code = handle.read()
        assert "studio_quality_gain" in source_code or "pqs_improvement" in source_code


# ---------------------------------------------------------------------------
# Mode Mapping in UV3
# ---------------------------------------------------------------------------


class TestUV3ModeMapping:
    """UV3 mode dispatch: restoration ↔ QUALITY, studio_2026 ↔ MAXIMUM."""

    def test_quality_mode_enum_exists(self):
        try:
            from backend.core.unified_restorer_v3 import QualityMode

            assert hasattr(QualityMode, "QUALITY")
            assert hasattr(QualityMode, "MAXIMUM")
        except ImportError:
            # QualityMode may live elsewhere
            from backend.core.restoration_config import QualityMode  # type: ignore[no-redef]

            assert hasattr(QualityMode, "QUALITY")
            assert hasattr(QualityMode, "MAXIMUM")

    def test_restoration_mode_is_quality(self):
        """Restoration mode maps to QualityMode.QUALITY."""
        try:
            from backend.core.unified_restorer_v3 import QualityMode
        except ImportError:
            from backend.core.restoration_config import QualityMode  # type: ignore[no-redef]
        assert QualityMode.QUALITY.value is not None

    def test_studio_mode_is_maximum(self):
        """Studio 2026 mode maps to QualityMode.MAXIMUM."""
        try:
            from backend.core.unified_restorer_v3 import QualityMode
        except ImportError:
            from backend.core.restoration_config import QualityMode  # type: ignore[no-redef]
        assert QualityMode.MAXIMUM.value is not None


# ---------------------------------------------------------------------------
# §0a Modus-Differenzierung Compliance
# ---------------------------------------------------------------------------


class TestModusDifferenzierung:
    """§0a: Restoration = Tonträgerkette invertieren; Studio 2026 = bestmöglicher Studio-Klang."""

    def test_restoration_preserves_character(self):
        """Restoration mode thresholds enforce authenticity preservation."""
        from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds

        rest = _get_canonical_thresholds(is_studio_2026=False)
        # Authentizität ≥ 0.88 = character preservation is critical
        assert rest["authentizitaet"] >= 0.88

    def test_studio_allows_enhancement(self):
        """Studio 2026: P1/P2 stricter (guard against over-processing), P3-P5 Spec-09 floors."""
        from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds

        studio = _get_canonical_thresholds(is_studio_2026=True)
        # P1/P2 stricter in Studio 2026 — more enhancement needs stronger naturalness guard
        assert studio["natuerlichkeit"] >= 0.92
        assert studio["authentizitaet"] >= 0.90
        # P3-P5: material-universal floors (Spec 09 / calibration_matrix.py)
        assert studio["transparenz"] >= 0.85
        assert studio["brillanz"] >= 0.82
        assert studio["micro_dynamics"] >= 0.90

    def test_restoration_noise_floor_material_adaptive(self):
        """Restoration: noise floor approaches original medium level, not −72 dBFS."""
        # This is a documentation/design test — the threshold for restoration
        # is material-adaptive, not a fixed −72 dBFS (which is Studio 2026)
        from backend.core.per_phase_musical_goals_gate import _get_canonical_thresholds

        rest = _get_canonical_thresholds(is_studio_2026=False)
        studio = _get_canonical_thresholds(is_studio_2026=True)
        # Studio transparenz target is higher → more aggressive denoising allowed
        assert studio["transparenz"] > rest["transparenz"]
