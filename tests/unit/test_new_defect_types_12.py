from __future__ import annotations

"""Tests for 12 new defect types added in v10.0.0.x.

Covers: DefectType enum presence, MATERIAL_SENSITIVITY entries,
detection methods in DefectScanner, CausalDefectReasoner integration
(CAUSES, CAUSE_TO_PHASES, CAUSE_PARAMS, LIKELIHOOD_FNS, MATERIAL_PRIORS),
and the 6 new repair phases (59–64).

≥ 35 tests across enum, detection, causal reasoning, and phase modules.
"""


import importlib
import os

import numpy as np
import pytest

from backend.core.defect_scanner import DefectScanner, DefectType

SR = 48_000

# ──────────────────────────────────────────────────────────────────────────────
# The 12 new DefectType members
# ──────────────────────────────────────────────────────────────────────────────
NEW_DEFECT_TYPES = [
    "MODULATION_NOISE",
    "INNER_GROOVE_DISTORTION",
    "GROOVE_ECHO",
    "CROSSTALK",
    "INTERMODULATION_DISTORTION",
    "TAPE_SPLICE_ARTIFACT",
    "HF_REMANENCE_LOSS",
    "STYLUS_DAMAGE",
    "STICKY_SHED_RESIDUE",
    "MULTIBAND_WOW_FLUTTER",
    "GENERATION_LOSS",
    "MOTOR_INTERFERENCE",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sine(freq: float = 440.0, secs: float = 1.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False, dtype=np.float32)
    return 0.3 * np.sin(2.0 * np.pi * freq * t)  # type: ignore[no-any-return]


def _noise(secs: float = 1.0, amp: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(SR * secs)) * amp).astype(np.float32)


def _test_audio(secs: float = 1.2) -> np.ndarray:
    """Deterministic mixed-content audio for scanner tests."""
    n = int(SR * secs)
    t = np.linspace(0.0, secs, n, endpoint=False, dtype=np.float32)
    x = 0.08 * np.sin(2.0 * np.pi * 220.0 * t)
    x += 0.04 * np.sin(2.0 * np.pi * 880.0 * t)
    x += 0.01 * np.sin(2.0 * np.pi * 50.0 * t)
    rng = np.random.default_rng(42)
    x += 0.005 * rng.standard_normal(n).astype(np.float32)
    return np.clip(x.astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]


def _stereo_audio(secs: float = 1.2) -> np.ndarray:
    mono = _test_audio(secs)
    return np.stack([mono, mono * 0.85], axis=0)


def _inject_long_level_dips(audio: np.ndarray, dip_times: list[float], dip_duration_s: float = 0.18) -> np.ndarray:
    out = audio.copy()
    dip_samples = int(SR * dip_duration_s)
    fade_down = int(0.080 * SR)
    snap_back = int(0.030 * SR)
    hold = max(1, dip_samples - fade_down - snap_back)
    min_gain = 10.0 ** (-12.0 / 20.0)
    env = np.concatenate(
        [
            np.linspace(1.0, min_gain, fade_down, dtype=np.float32),
            np.full(hold, min_gain, dtype=np.float32),
            np.linspace(min_gain, 1.0, snap_back, dtype=np.float32),
        ]
    )[:dip_samples]
    for ts in dip_times:
        start = int(ts * SR)
        end = start + dip_samples
        if end > len(out):
            continue
        out[start:end] *= env
    return out.astype(np.float32)


def _musical_harmonic_stack(secs: float = 4.0, f0: float = 100.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False, dtype=np.float32)
    x = np.zeros_like(t)
    for mult, amp in [(1, 0.22), (2, 0.14), (3, 0.09), (4, 0.06)]:
        x += amp * np.sin(2.0 * np.pi * (f0 * mult) * t)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DefectType Enum — All 12 members exist
# ══════════════════════════════════════════════════════════════════════════════


class TestDefectTypeEnum:
    @pytest.mark.parametrize("name", NEW_DEFECT_TYPES)
    def test_enum_member_exists(self, name: str) -> None:
        member = getattr(DefectType, name, None)
        assert member is not None, f"DefectType.{name} missing"

    def test_enum_members_unique_values(self) -> None:
        values = [getattr(DefectType, n).value for n in NEW_DEFECT_TYPES]
        assert len(values) == len(set(values)), "Duplicate enum values"


# ══════════════════════════════════════════════════════════════════════════════
# 2. MATERIAL_SENSITIVITY — All 12 types present in all materials
# ══════════════════════════════════════════════════════════════════════════════


class TestMaterialSensitivity:
    def test_sensitivity_tables_have_new_types(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        sens = scanner.MATERIAL_SENSITIVITY
        for mat, mat_sens in sens.items():
            for name in NEW_DEFECT_TYPES:
                dt = getattr(DefectType, name)
                assert dt in mat_sens, f"DefectType.{name} missing in MATERIAL_SENSITIVITY[{mat}]"

    def test_sensitivity_values_in_range(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        sens = scanner.MATERIAL_SENSITIVITY
        for mat, mat_sens in sens.items():
            for name in NEW_DEFECT_TYPES:
                dt = getattr(DefectType, name)
                val = mat_sens[dt]
                assert 0.0 <= val <= 3.0, f"MATERIAL_SENSITIVITY[{mat}][{name}] = {val} out of range"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Detection Methods — Scanner produces scores for all 12 types
# ══════════════════════════════════════════════════════════════════════════════


class TestDetectionMethods:
    def test_scan_returns_all_new_defect_types(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        audio = _test_audio()
        result = scanner.scan(audio, SR)
        for name in NEW_DEFECT_TYPES:
            dt = getattr(DefectType, name)
            assert dt in result.scores, f"DefectType.{name} not in scan result"

    @pytest.mark.parametrize("name", NEW_DEFECT_TYPES)
    def test_scan_score_bounds(self, name: str) -> None:
        scanner = DefectScanner(sample_rate=SR)
        audio = _test_audio()
        result = scanner.scan(audio, SR)
        dt = getattr(DefectType, name)
        score = result.scores[dt]
        assert 0.0 <= float(score.severity) <= 1.0, f"{name}: severity out of range"
        assert 0.0 <= float(score.confidence) <= 1.0, f"{name}: confidence out of range"
        assert isinstance(score.locations, list)
        assert isinstance(score.metadata, dict)

    def test_crosstalk_requires_stereo(self) -> None:
        """Crosstalk detection should handle mono gracefully (severity 0)."""
        scanner = DefectScanner(sample_rate=SR)
        mono = _test_audio()
        result = scanner.scan(mono, SR)
        ct = result.scores.get(DefectType.CROSSTALK)
        assert ct is not None
        assert float(ct.severity) == 0.0

    def test_crosstalk_produces_score_on_stereo(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        stereo = _stereo_audio()
        result = scanner.scan(stereo, SR)
        ct = result.scores.get(DefectType.CROSSTALK)
        assert ct is not None
        assert 0.0 <= float(ct.severity) <= 1.0

    def test_modulation_noise_on_clean_signal_low(self) -> None:
        """Clean sine: modulation noise should be low."""
        scanner = DefectScanner(sample_rate=SR)
        audio = _sine(440.0, 1.5).astype(np.float32)
        result = scanner.scan(audio, SR)
        score = result.scores[DefectType.MODULATION_NOISE]
        assert float(score.severity) < 0.30

    def test_motor_interference_on_clean_signal_low(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        audio = _sine(440.0, 1.5).astype(np.float32)
        result = scanner.scan(audio, SR)
        score = result.scores[DefectType.MOTOR_INTERFERENCE]
        assert float(score.severity) < 0.30

    def test_sticky_shed_detects_long_contact_loss_dips(self) -> None:
        from backend.core.defect_scanner import MaterialType

        scanner = DefectScanner(sample_rate=SR)
        audio = _sine(440.0, 8.0).astype(np.float32)
        audio = _inject_long_level_dips(audio, dip_times=[1.0, 2.4, 3.8, 5.2, 6.6], dip_duration_s=0.18)
        result = scanner.scan(audio, SR, material_type=MaterialType.TAPE)
        score = result.scores[DefectType.STICKY_SHED_RESIDUE]
        assert float(score.severity) > 0.10
        assert int(score.metadata.get("n_long_events", 0)) >= 2

    def test_motor_interference_rejects_musical_harmonic_stack(self) -> None:
        from backend.core.defect_scanner import MaterialType

        scanner = DefectScanner(sample_rate=SR)
        audio = _musical_harmonic_stack()
        result = scanner.scan(audio, SR, material_type=MaterialType.VINYL)
        score = result.scores[DefectType.MOTOR_INTERFERENCE]
        assert float(score.severity) < 0.40


# ══════════════════════════════════════════════════════════════════════════════
# 4. CausalDefectReasoner — All 12 causes present
# ══════════════════════════════════════════════════════════════════════════════


class TestCausalReasoner:
    NEW_CAUSES = [
        "modulation_noise",
        "inner_groove_distortion",
        "groove_echo",
        "crosstalk",
        "intermodulation_distortion",
        "tape_splice_artifact",
        "hf_remanence_loss",
        "stylus_damage",
        "sticky_shed_residue",
        "multiband_wow_flutter",
        "generation_loss",
        "motor_interference",
    ]

    def test_all_new_causes_in_CAUSES(self) -> None:
        from backend.core.causal_defect_reasoner import CAUSES

        for cause in self.NEW_CAUSES:
            assert cause in CAUSES, f"{cause} not in CAUSES"

    def test_all_new_causes_have_phases(self) -> None:
        from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES

        for cause in self.NEW_CAUSES:
            assert cause in CAUSE_TO_PHASES, f"{cause} not in CAUSE_TO_PHASES"
            assert isinstance(CAUSE_TO_PHASES[cause], list)
            assert len(CAUSE_TO_PHASES[cause]) > 0, f"CAUSE_TO_PHASES[{cause}] empty"

    def test_all_new_causes_have_params(self) -> None:
        from backend.core.causal_defect_reasoner import CAUSE_PARAMS

        for cause in self.NEW_CAUSES:
            assert cause in CAUSE_PARAMS, f"{cause} not in CAUSE_PARAMS"
            assert isinstance(CAUSE_PARAMS[cause], dict)

    def test_all_new_causes_have_likelihood_fns(self) -> None:
        from backend.core.causal_defect_reasoner import LIKELIHOOD_FNS

        for cause in self.NEW_CAUSES:
            assert cause in LIKELIHOOD_FNS, f"{cause} not in LIKELIHOOD_FNS"
            assert callable(LIKELIHOOD_FNS[cause])

    @pytest.mark.parametrize(
        "material",
        [
            "tape",
            "vinyl",
            "shellac",
            "digital",
            "unknown",
            "mp3_low",
            "mp3_high",
            "aac",
            "cd_digital",
            "streaming",
            "dat",
            "minidisc",
            "wax_cylinder",
            "lacquer_disc",
            "wire_recording",
        ],
    )
    def test_material_priors_have_all_new_causes(self, material: str) -> None:
        from backend.core.causal_defect_reasoner import MATERIAL_PRIORS

        priors = MATERIAL_PRIORS[material]
        for cause in self.NEW_CAUSES:
            assert cause in priors, f"{cause} not in MATERIAL_PRIORS[{material}]"
            val = priors[cause]
            assert 0.0 <= val <= 1.0, f"Prior {material}/{cause} = {val} out of range"


# ══════════════════════════════════════════════════════════════════════════════
# 5. New Repair Phases (59–64) — importable + PhaseInterface compliant
# ══════════════════════════════════════════════════════════════════════════════

NEW_PHASE_MODULES = [
    (
        "backend.core.phases.phase_59_modulation_noise_reduction",
        "ModulationNoiseReductionPhase",
        "phase_59_modulation_noise_reduction",
    ),
    (
        "backend.core.phases.phase_60_inner_groove_distortion_repair",
        "InnerGrooveDistortionRepairPhase",
        "phase_60_inner_groove_distortion_repair",
    ),
    (
        "backend.core.phases.phase_61_groove_echo_cancellation",
        "GrooveEchoCancellationPhase",
        "phase_61_groove_echo_cancellation",
    ),
    (
        "backend.core.phases.phase_62_crosstalk_cancellation",
        "CrosstalkCancellationPhase",
        "phase_62_crosstalk_cancellation",
    ),
    (
        "backend.core.phases.phase_63_intermodulation_reduction",
        "IntermodulationReductionPhase",
        "phase_63_intermodulation_reduction",
    ),
    ("backend.core.phases.phase_64_tape_splice_repair", "TapeSpliceRepairPhase", "phase_64_tape_splice_repair"),
]


class TestNewPhases:
    @pytest.mark.parametrize("modname,classname,phase_id", NEW_PHASE_MODULES)
    def test_phase_importable(self, modname: str, classname: str, phase_id: str) -> None:
        mod = importlib.import_module(modname)
        cls = getattr(mod, classname)
        assert cls is not None

    @pytest.mark.parametrize("modname,classname,phase_id", NEW_PHASE_MODULES)
    def test_phase_has_get_metadata(self, modname: str, classname: str, phase_id: str) -> None:
        mod = importlib.import_module(modname)
        cls = getattr(mod, classname)
        instance = cls()
        meta = instance.get_metadata()
        assert meta.phase_id == phase_id

    @pytest.mark.parametrize("modname,classname,phase_id", NEW_PHASE_MODULES)
    def test_phase_has_process_method(self, modname: str, classname: str, phase_id: str) -> None:
        mod = importlib.import_module(modname)
        cls = getattr(mod, classname)
        instance = cls()
        assert hasattr(instance, "process")
        assert callable(instance.process)

    @pytest.mark.parametrize("modname,classname,phase_id", NEW_PHASE_MODULES)
    def test_phase_process_returns_valid_result(self, modname: str, classname: str, phase_id: str) -> None:
        mod = importlib.import_module(modname)
        cls = getattr(mod, classname)
        instance = cls()

        # Use stereo for crosstalk, mono for others
        if "crosstalk" in phase_id:
            audio = _stereo_audio(1.0)
        else:
            audio = _test_audio(1.0)

        result = instance.process(audio=audio, sr=SR, strength=0.5)
        assert result is not None
        assert hasattr(result, "audio")
        assert hasattr(result, "success")
        assert result.success is True
        # Audio shape preserved
        assert result.audio.shape == audio.shape
        # No NaN / Inf
        assert np.isfinite(result.audio).all(), f"{phase_id}: NaN/Inf in output"
        # Clipping guard
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6, f"{phase_id}: output clipped"

    @pytest.mark.parametrize("modname,classname,phase_id", NEW_PHASE_MODULES)
    def test_phase_process_strength_zero_passthrough(self, modname: str, classname: str, phase_id: str) -> None:
        """Strength 0.0 should return audio unchanged (dry signal)."""
        mod = importlib.import_module(modname)
        cls = getattr(mod, classname)
        instance = cls()

        if "crosstalk" in phase_id:
            audio = _stereo_audio(0.5)
        else:
            audio = _test_audio(0.5)

        result = instance.process(audio=audio, sr=SR, strength=0.0)
        np.testing.assert_allclose(result.audio, audio, atol=1e-5, err_msg=f"{phase_id}: strength=0 not passthrough")

    def test_phase_59_apply_function(self) -> None:
        from backend.core.phases.phase_59_modulation_noise_reduction import apply

        audio = _test_audio(0.5)
        out = apply(audio, SR, strength=0.5)
        assert out.shape == audio.shape
        assert np.isfinite(out).all()

    def test_phase_60_apply_function(self) -> None:
        from backend.core.phases.phase_60_inner_groove_distortion_repair import apply

        audio = _test_audio(0.5)
        out = apply(audio, SR, strength=0.5)
        assert out.shape == audio.shape
        assert np.isfinite(out).all()

    def test_phase_62_mono_passthrough(self) -> None:
        """Crosstalk phase with mono input should pass through."""
        from backend.core.phases.phase_62_crosstalk_cancellation import CrosstalkCancellationPhase

        phase = CrosstalkCancellationPhase()
        mono = _test_audio(0.5)
        result = phase.process(audio=mono, sr=SR, strength=0.5)
        np.testing.assert_allclose(result.audio, mono, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 6. UV3 Phase Selection Integration
# ══════════════════════════════════════════════════════════════════════════════


class TestUV3PhaseSelection:
    """Verify that _select_phases() references the new phase IDs."""

    def test_phase_selection_code_contains_new_phases(self) -> None:
        """Grep UV3 source for new phase references."""
        uv3_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "core", "unified_restorer_v3.py")
        uv3_path = os.path.normpath(uv3_path)
        with open(uv3_path, encoding="utf-8") as f:
            source = f.read()

        new_phase_ids = [
            "phase_59_modulation_noise_reduction",
            "phase_60_inner_groove_distortion_repair",
            "phase_61_groove_echo_cancellation",
            "phase_62_crosstalk_cancellation",
            "phase_63_intermodulation_reduction",
            "phase_64_tape_splice_repair",
        ]
        for pid in new_phase_ids:
            assert pid in source, f"{pid} not found in unified_restorer_v3.py"

    def test_phase_weights_contain_new_phases(self) -> None:
        """Verify _PHASE_WEIGHTS includes entries for phases 59–64."""
        uv3_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "core", "unified_restorer_v3.py")
        uv3_path = os.path.normpath(uv3_path)
        with open(uv3_path, encoding="utf-8") as f:
            source = f.read()

        for i in range(59, 65):
            assert f"phase_{i}" in source, f"phase_{i} not in _PHASE_WEIGHTS"

    def test_defect_type_references_in_uv3(self) -> None:
        """UV3 source must reference all 12 new DefectTypes."""
        uv3_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "core", "unified_restorer_v3.py")
        uv3_path = os.path.normpath(uv3_path)
        with open(uv3_path, encoding="utf-8") as f:
            source = f.read()

        for name in NEW_DEFECT_TYPES:
            assert f"DefectType.{name}" in source, f"DefectType.{name} not referenced in UV3"
