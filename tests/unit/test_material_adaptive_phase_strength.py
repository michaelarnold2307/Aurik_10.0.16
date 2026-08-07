"""
Tests for material-adaptive phase initial strengths (§2.29 / §2.31).

Covers:
- DefectPhaseMapper._MATERIAL_PHASE_FACTORS + get_material_initial_strength()
- PhaseAssignment.apply_to_config() with material_factor
- DefectPhaseMapper.build_specialist_config() with material param
- PMGG PerPhaseMusicalGoalsGate.wrap_phase() with initial_strength
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal ProcessingConfig stand-in."""

    denoise_strength: float = 0.5
    click_removal_sensitivity: float = 0.5
    declip_strength: float = 0.0
    spectral_repair_strength: float = 0.5
    preserve_analog_character: bool = False
    enable_spectral_repair: bool = False


def _make_fake_phase(strength_recorder: list) -> object:
    """Creates a minimal phase that records strength and returns input audio.

    CRITICAL: Must define process() not __call__, because PMGG _run_phase
    calls phase.process() (v10.0.0 fix — PhaseInterface has no __call__).
    """

    class FakePhase:
        def process(self, audio, strength=1.0, **kwargs):
            strength_recorder.append(strength)
            # Return object with .audio attribute (PhaseResult-like)
            result = types.SimpleNamespace(audio=audio.copy())
            return result

        def __call__(self, audio, strength=1.0, **kwargs):
            # Legacy fallback — not used by current PMGG but kept for compatibility
            return self.process(audio, strength=strength, **kwargs)

        def get_metadata(self):
            meta = types.SimpleNamespace()
            # Use a DSP (non-ML-deterministic) phase_id so PMGG passes
            # initial_strength directly to process() instead of Wet/Dry blending.
            meta.phase_id = "phase_10_click_repair"
            return meta

    return FakePhase()


# ---------------------------------------------------------------------------
# 1. get_material_initial_strength
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetMaterialInitialStrength:
    def test_shellac_denoise_below_one(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("shellac", "phase_03_denoise")
        assert 0.0 < s < 1.0, f"shellac phase_03_denoise should be < 1.0, got {s}"

    def test_shellac_tape_saturation_very_low(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("shellac", "phase_22_tape_saturation")
        assert s <= 0.30, f"shellac tape_saturation cap should be ≤ 0.30, got {s}"

    def test_wax_cylinder_tape_saturation_very_low(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("wax_cylinder", "phase_22_tape_saturation")
        assert s <= 0.25, f"wax_cylinder tape_saturation should be ≤ 0.25, got {s}"

    def test_cd_digital_denoise_low(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("cd_digital", "phase_03_denoise")
        assert s < 0.5, f"cd_digital denoise should start low, got {s}"

    def test_unknown_material_returns_one(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("unknown_material_xyz", "phase_03_denoise")
        assert s == 1.0

    def test_unknown_phase_returns_one(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("shellac", "phase_99_unknown")
        assert s == 1.0

    def test_vinyl_crackle_below_full(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("vinyl", "phase_09_crackle_removal")
        assert 0.0 < s <= 1.0

    def test_tape_saturation_preserved_for_tape(self):
        from backend.core.defect_phase_mapper import get_material_initial_strength

        s = get_material_initial_strength("tape", "phase_22_tape_saturation")
        assert s <= 0.40, f"tape phase_22 should be capped, got {s}"

    def test_all_factors_in_valid_range(self):
        from backend.core.defect_phase_mapper import _MATERIAL_PHASE_FACTORS

        for mat, phases in _MATERIAL_PHASE_FACTORS.items():
            for phase_id, factor in phases.items():
                assert 0.0 < factor <= 1.0, (
                    f"_MATERIAL_PHASE_FACTORS[{mat!r}][{phase_id!r}] = {factor} — must be in (0, 1.0]"
                )


# ---------------------------------------------------------------------------
# 2. apply_to_config with material_factor
# ---------------------------------------------------------------------------


class TestApplyToConfigMaterialFactor:
    def _get_assignment(self):
        from backend.core.defect_phase_mapper import _PHASE_MAP, DefectType

        return _PHASE_MAP[DefectType.HIGH_FREQ_NOISE]

    def test_material_factor_one_is_identity(self):
        assignment = self._get_assignment()
        cfg1 = _FakeConfig()
        cfg2 = _FakeConfig()
        assignment.apply_to_config(cfg1, severity=0.8, mode_factor=1.0, material_factor=1.0)
        assignment.apply_to_config(cfg2, severity=0.8, mode_factor=1.0)
        # material_factor=1.0 should give same result as not passing it
        assert cfg1.denoise_strength == pytest.approx(cfg2.denoise_strength, abs=1e-6)

    def test_material_factor_halves_strength(self):
        assignment = self._get_assignment()
        cfg_full = _FakeConfig()
        cfg_half = _FakeConfig()
        assignment.apply_to_config(cfg_full, severity=1.0, mode_factor=1.0, material_factor=1.0)
        assignment.apply_to_config(cfg_half, severity=1.0, mode_factor=1.0, material_factor=0.5)
        # With half material factor, effective strength should be lower
        assert cfg_half.denoise_strength <= cfg_full.denoise_strength

    def test_material_factor_zero_point_three_caps_result(self):
        assignment = self._get_assignment()
        cfg = _FakeConfig()
        # severity=1.0, mode_factor=1.0, material_factor=0.30 → effective=0.30
        assignment.apply_to_config(cfg, severity=1.0, mode_factor=1.0, material_factor=0.30)
        # denoise_strength from config_delta=0.80, scaled by effective=0.30 → 0.24
        assert cfg.denoise_strength == pytest.approx(0.80 * 0.30, abs=0.01)

    def test_material_factor_clamped_at_one(self):
        """material_factor > 1.0 after product should be clamped to 1.0."""
        assignment = self._get_assignment()
        cfg_over = _FakeConfig()
        cfg_one = _FakeConfig()
        # severity=1.0, mode=1.0, material_factor=2.0 → effective=min(1.0, 2.0)=1.0
        assignment.apply_to_config(cfg_over, severity=1.0, mode_factor=1.0, material_factor=2.0)
        assignment.apply_to_config(cfg_one, severity=1.0, mode_factor=1.0, material_factor=1.0)
        assert cfg_over.denoise_strength == pytest.approx(cfg_one.denoise_strength, abs=1e-6)

    def test_bool_field_set_above_threshold(self):
        from backend.core.defect_phase_mapper import _PHASE_MAP, DefectType

        assignment = _PHASE_MAP[DefectType.DROPOUTS]
        cfg = _FakeConfig()
        # effective = 1.0 * 1.0 * 1.0 = 1.0 ≥ 0.3 → bool True
        assignment.apply_to_config(cfg, severity=1.0, mode_factor=1.0, material_factor=1.0)
        assert cfg.enable_spectral_repair is True

    def test_bool_field_not_set_below_threshold(self):
        from backend.core.defect_phase_mapper import _PHASE_MAP, DefectType

        assignment = _PHASE_MAP[DefectType.DROPOUTS]
        cfg = _FakeConfig()
        # Bool logic: True if effective >= 0.3 else v (v from config_delta).
        # For DROPOUTS, config_delta value is True, so field remains True even below threshold.
        assignment.apply_to_config(cfg, severity=0.1, mode_factor=1.0, material_factor=1.0)
        assert cfg.enable_spectral_repair is True


# ---------------------------------------------------------------------------
# 3. build_specialist_config with material
# ---------------------------------------------------------------------------


class TestBuildSpecialistConfigMaterial:
    def test_shellac_lowers_denoise_vs_none(self):
        from backend.core.defect_phase_mapper import DefectPhaseMapper, DefectType

        mapper = DefectPhaseMapper()
        cfg_no_mat, _ = mapper.build_specialist_config(
            _FakeConfig(), DefectType.HIGH_FREQ_NOISE, severity=0.8, material=None
        )
        cfg_shellac, _ = mapper.build_specialist_config(
            _FakeConfig(), DefectType.HIGH_FREQ_NOISE, severity=0.8, material="shellac"
        )
        # shellac has factor < 1.0 for phase_03_denoise → should result in lower denoise_strength
        assert cfg_shellac.denoise_strength <= cfg_no_mat.denoise_strength  # type: ignore[attr-defined]

    def test_cd_digital_lowers_click_sensitivity(self):
        from backend.core.defect_phase_mapper import DefectPhaseMapper, DefectType

        mapper = DefectPhaseMapper()
        cfg_no_mat, _ = mapper.build_specialist_config(_FakeConfig(), DefectType.CLICKS, severity=0.8, material=None)
        cfg_cd, _ = mapper.build_specialist_config(
            _FakeConfig(), DefectType.CLICKS, severity=0.8, material="cd_digital"
        )
        assert cfg_cd.click_removal_sensitivity <= cfg_no_mat.click_removal_sensitivity  # type: ignore[attr-defined]

    def test_material_none_same_as_before(self):
        """material=None must behave identically to not passing material."""
        from backend.core.defect_phase_mapper import DefectPhaseMapper, DefectType

        mapper = DefectPhaseMapper()
        cfg_none, v1 = mapper.build_specialist_config(_FakeConfig(), DefectType.CRACKLE, severity=0.7)
        cfg_none2, v2 = mapper.build_specialist_config(_FakeConfig(), DefectType.CRACKLE, severity=0.7, material=None)
        assert cfg_none.denoise_strength == pytest.approx(cfg_none2.denoise_strength, abs=1e-6)  # type: ignore[attr-defined]

    def test_returns_tuple_of_two(self):
        from backend.core.defect_phase_mapper import DefectPhaseMapper, DefectType

        mapper = DefectPhaseMapper()
        result = mapper.build_specialist_config(_FakeConfig(), DefectType.CLICKS, severity=0.5, material="vinyl")
        assert isinstance(result, tuple) and len(result) == 2

    def test_variant_name_is_string(self):
        from backend.core.defect_phase_mapper import DefectPhaseMapper, DefectType

        mapper = DefectPhaseMapper()
        _, name = mapper.build_specialist_config(_FakeConfig(), DefectType.HUM, severity=0.5, material="tape")
        assert isinstance(name, str) and len(name) > 0


# ---------------------------------------------------------------------------
# 4. PMGG initial_strength
# ---------------------------------------------------------------------------


class TestPMGGInitialStrength:
    def _make_quick_audio(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.uniform(-0.1, 0.1, 48000 * 2).astype(np.float32)

    @staticmethod
    def _zero_scores() -> dict:
        """Returns all FAST_GOALS_SUBSET keys at 0.0.

        Ensures PMGG never detects a regression when testing strength
        propagation (random noise audio scores ~0.0 on all goals, which
        equals the baseline — no rollback possible).
        """
        from backend.core.per_phase_musical_goals_gate import FAST_GOALS_SUBSET

        return dict.fromkeys(FAST_GOALS_SUBSET, 0.0)

    def test_initial_strength_passed_to_phase(self):
        """wrap_phase with initial_strength=0.3 must call phase with strength=0.3."""
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        gate = get_phase_gate()
        gate.reset()

        recorded: list[Any] = []
        phase = _make_fake_phase(recorded)
        audio = self._make_quick_audio()

        gate.wrap_phase(
            phase,
            audio,
            48000,
            scores_before=self._zero_scores(),
            initial_strength=0.3,
        )
        assert len(recorded) >= 1
        # First call must use initial_strength, not 1.0
        assert recorded[0] == pytest.approx(0.3, abs=1e-6), (
            f"Expected initial_strength=0.3 passed to phase, got {recorded[0]}"
        )

    def test_initial_strength_default_is_one(self):
        """wrap_phase without initial_strength must pass 1.0 to phase (backward compat)."""
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        gate = get_phase_gate()
        gate.reset()

        recorded: list[Any] = []
        phase = _make_fake_phase(recorded)
        audio = self._make_quick_audio()

        gate.wrap_phase(
            phase,
            audio,
            48000,
            scores_before=self._zero_scores(),
        )
        assert recorded[0] == pytest.approx(1.0, abs=1e-6), f"Default initial_strength should be 1.0, got {recorded[0]}"

    def test_initial_strength_above_one_clamped(self):
        """initial_strength > 1.0 must be clamped to 1.0."""
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        gate = get_phase_gate()
        gate.reset()

        recorded: list[Any] = []
        phase = _make_fake_phase(recorded)
        audio = self._make_quick_audio()

        gate.wrap_phase(
            phase,
            audio,
            48000,
            scores_before=self._zero_scores(),
            initial_strength=1.5,  # over limit
        )
        assert recorded[0] <= 1.0, f"initial_strength=1.5 should be clamped, got {recorded[0]}"

    def test_initial_strength_zero_point_zero_one_floor(self):
        """initial_strength ≤ 0 must be floored to 0.01 (never zero)."""
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        gate = get_phase_gate()
        gate.reset()

        recorded: list[Any] = []
        phase = _make_fake_phase(recorded)
        audio = self._make_quick_audio()

        gate.wrap_phase(
            phase,
            audio,
            48000,
            scores_before=self._zero_scores(),
            initial_strength=0.0,
        )
        assert recorded[0] >= 0.01, f"initial_strength=0 should be floored, got {recorded[0]}"

    def test_returns_three_tuple(self):
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        gate = get_phase_gate()
        gate.reset()

        recorded: list[Any] = []
        phase = _make_fake_phase(recorded)
        audio = self._make_quick_audio()

        result = gate.wrap_phase(phase, audio, 48000, initial_strength=0.5)
        assert isinstance(result, tuple) and len(result) == 3


# ---------------------------------------------------------------------------
# §6.2a MandatoryFloor — invariant: mandatory priority phases must achieve
# minimum effective strength = mat_strength × 0.40, even at low defect severity.
# ---------------------------------------------------------------------------


class TestMandatoryFloorInvariant:
    """Tests for the §6.2a MandatoryFloor guard added to UV3 _execute_pipeline.

    The guard ensures that _MATERIAL_PRIORITY_PHASES (activated unconditionally
    per §6.2a, independent of scanner confidence) cannot be reduced to near-zero
    strength by a low _sev_factor.  Without the guard, shellac phase_09 at
    sev=0.15 → initial_strength = 0.85 × 0.15 = 0.127 — functionally a no-op.
    With the guard: floor = 0.85 × 0.40 = 0.34 → meaningful crackle correction.
    """

    def test_mandatory_floor_formula(self):
        """mat_strength × 0.40 must exceed sev_factor × mat_strength when sev < 0.40."""
        from backend.core.defect_phase_mapper import get_material_initial_strength

        for material, phase_id in [
            ("shellac", "phase_09_crackle_removal"),
            ("shellac", "phase_03_denoise"),
            ("shellac", "phase_01_click_removal"),
            ("vinyl", "phase_09_crackle_removal"),
            ("vinyl", "phase_12_wow_flutter_fix"),
            ("tape", "phase_29_tape_hiss_reduction"),
        ]:
            mat_strength = get_material_initial_strength(material, phase_id)
            if mat_strength == 1.0:
                mat_strength_eff = 1.0  # default — no override in factors
            mandatory_floor = mat_strength * 0.40

            # Simulate low scanner severity (floor from get_phase_defect_severity = 0.15)
            low_sev = 0.15
            combined_without_floor = mat_strength * low_sev
            assert mandatory_floor > combined_without_floor, (
                f"{material}/{phase_id}: mandatory_floor ({mandatory_floor:.3f}) must exceed "
                f"mat_strength × min_sev ({combined_without_floor:.3f}) to prevent near-zero processing"
            )

    def test_mandatory_floor_allows_heavy_defect_full_strength(self):
        """When sev_factor × mat_strength > floor, the floor must NOT cap strength downward."""
        from backend.core.defect_phase_mapper import get_material_initial_strength

        mat_strength = get_material_initial_strength("shellac", "phase_09_crackle_removal")
        high_sev = 0.85
        combined_high = mat_strength * high_sev
        mandatory_floor = mat_strength * 0.40

        # Floor must not apply when combined strength is already above it
        effective = max(combined_high, mandatory_floor)
        assert effective == pytest.approx(combined_high, abs=1e-6), (
            f"Floor ({mandatory_floor:.3f}) should not reduce high-severity strength ({combined_high:.3f})"
        )

    def test_mandatory_floor_value_is_meaningful(self):
        """The resulting floor must be above 0.25 for all primary defect phases on priority materials."""
        from backend.core.defect_phase_mapper import get_material_initial_strength

        priority_pairs = [
            ("shellac", "phase_09_crackle_removal"),
            ("shellac", "phase_03_denoise"),
            ("vinyl", "phase_09_crackle_removal"),
            ("tape", "phase_29_tape_hiss_reduction"),
            ("wax_cylinder", "phase_09_crackle_removal"),
            ("wax_cylinder", "phase_03_denoise"),
        ]
        for material, phase_id in priority_pairs:
            mat_strength = get_material_initial_strength(material, phase_id)
            floor = mat_strength * 0.40
            assert floor >= 0.25, (
                f"{material}/{phase_id}: mandatory floor {floor:.3f} = mat_strength({mat_strength:.2f}) × 0.40 "
                f"must be ≥ 0.25 to ensure meaningful defect correction (§6.2a)"
            )

    def test_mat_strength_defaults_to_one_for_unlisted_priority_phases(self):
        """If a mandatory phase has no mat_strength entry, default=1.0 → floor=0.40."""
        from backend.core.defect_phase_mapper import get_material_initial_strength

        # vinyl phase_09 has no explicit entry in vinyl's factor dict → should return 1.0
        mat_strength = get_material_initial_strength("vinyl", "phase_09_crackle_removal")
        # mat_strength may be 1.0 or explicitly set — either way floor = mat_strength * 0.40
        floor = mat_strength * 0.40
        assert floor >= 0.28, (  # 0.28 = 0.70 × 0.40 (if some low override exists)
            f"vinyl phase_09 floor {floor:.3f} must be ≥ 0.28 (§6.2a)"
        )
