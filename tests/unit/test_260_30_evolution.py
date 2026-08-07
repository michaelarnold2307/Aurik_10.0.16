"""Tests für §2.60–§3.0 Fahrplan, SectionGoalAdapter, Per-Segment-Executor, Closed-Loop PID.

Testet:
  - Fahrplan-Konstruktion mit build_fahrplan()
  - SectionGoalAdapter: get_sections() mit synthetischem Audio
  - Per-Segment-Executor: run_phase_per_segment(), get_segment_strengths_from_fahrplan()
  - Closed-Loop PID: before_phase(), after_phase(), get_state()
  - Source-Aware Fahrplan: get_stem_config(), filter_phases_for_stem()
"""

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════
# §2.60 Fahrplan
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFahrplan:
    """Testet build_fahrplan() und Fahrplan.calibration."""

    def test_build_fahrplan_empty_sections(self):
        from backend.core.fahrplan import build_fahrplan

        fp = build_fahrplan(
            phase_ids=["phase_03_denoise", "phase_19_de_esser", "phase_21_exciter"],
            sections=[],
            goal_priorities={},
        )
        assert len(fp.phase_order) >= 2
        assert len(fp.sections) == 1  # Fallback auf "full"
        assert "full" in fp.sections[0][2]
        assert "3 Phasen" in fp.note or str(len(fp.phase_order)) in fp.note

    def test_fahrplan_calibration_flat(self):
        from backend.core.fahrplan import build_fahrplan

        fp = build_fahrplan(
            phase_ids=["phase_03_denoise", "phase_19_de_esser"],
            sections=[(0, 30, "verse"), (30, 60, "chorus")],
            goal_priorities={"transparenz": 0.8, "waerme": 0.5},
        )
        cal = fp.calibration
        assert "phase_03_denoise" in cal
        assert "phase_19_de_esser" in cal
        for v in cal.values():
            assert 0.0 < v <= 1.2  # Segment-Boosts können > 1.0 ergeben

    def test_fahrplan_per_segment_skip_silence(self):
        from backend.core.fahrplan import build_fahrplan

        fp = build_fahrplan(
            phase_ids=["phase_01_click_removal", "phase_03_denoise"],
            sections=[(0, 10, "silence"), (10, 30, "verse")],
            goal_priorities={},
        )
        instrs = fp.instructions.get("phase_01_click_removal", [])
        assert len(instrs) == 2
        # Silence segment should be skipped
        silence_instr = instrs[0]
        assert silence_instr.skip or silence_instr.strength_mod < 0.15

    def test_perceptual_budget_constant(self):
        from backend.core.fahrplan import PERCEPTUAL_BUDGET

        assert "presence" in PERCEPTUAL_BUDGET
        assert PERCEPTUAL_BUDGET["presence"] > PERCEPTUAL_BUDGET["sub_bass"]


# ═══════════════════════════════════════════════════════════════
# §2.61 SectionGoalAdapter
# ═══════════════════════════════════════════════════════════════


class TestSectionGoalAdapter:
    """Testet get_sections() mit verschiedenen Audio-Längen."""

    def test_short_audio_returns_full(self):
        from backend.core.section_goal_adapter import get_sections

        sr = 48000
        short = np.zeros(sr * 3, dtype=np.float32)  # 3s < 20s min
        sections = get_sections(short, sr, min_duration_s=20.0)
        assert len(sections) == 1
        assert sections[0][2] == "full"

    def test_long_audio_returns_sections(self):
        from backend.core.section_goal_adapter import get_sections

        sr = 16000
        # 30s of noise + 440Hz tone
        t = np.linspace(0, 30, sr * 30, dtype=np.float32)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3 + np.random.randn(sr * 30).astype(np.float32) * 0.02).astype(
            np.float32
        )
        sections = get_sections(audio, sr, min_duration_s=5.0)
        assert len(sections) >= 1
        for start, end, label in sections:
            assert 0.0 <= start < end
            assert label in ("intro", "verse", "chorus", "bridge", "outro", "full", "unknown")

    def test_empty_audio(self):
        from backend.core.section_goal_adapter import get_sections

        sections = get_sections(np.array([], dtype=np.float32), 48000)
        assert len(sections) == 1
        assert sections[0][2] == "full"

    def test_normalize_label(self):
        from backend.core.section_goal_adapter import _normalize_label

        assert _normalize_label("pre-chorus") == "chorus"
        assert _normalize_label("solo") == "bridge"
        assert _normalize_label("fade_out") == "outro"
        assert _normalize_label("unknown") == "full"

    def test_merge_adjacent(self):
        from backend.core.section_goal_adapter import _merge_adjacent

        sections = [(0.0, 10.0, "verse"), (10.0, 20.0, "verse"), (20.0, 30.0, "chorus")]
        merged = _merge_adjacent(sections)
        assert len(merged) == 2
        assert merged[0] == (0.0, 20.0, "verse")


# ═══════════════════════════════════════════════════════════════
# §2.62 Per-Segment Executor
# ═══════════════════════════════════════════════════════════════


class TestPerSegmentExecutor:
    """Testet run_phase_per_segment() und get_segment_strengths_from_fahrplan()."""

    def test_single_segment_fallback(self):
        from backend.core.per_segment_executor import run_phase_per_segment

        audio = np.sin(np.linspace(0, 50, 48000)).astype(np.float32)

        def dummy_phase(a, **kw):
            class R:
                pass

            r = R()
            r.audio = a * float(kw.get("strength", 1.0))  # type: ignore[attr-defined]
            return r

        result = run_phase_per_segment(
            audio,
            48000,
            dummy_phase,
            {"strength": 0.5, "sample_rate": 48000},
            segment_bounds_s=[0.0, 1.0],
            segment_strengths=[0.5],
        )
        assert hasattr(result, "audio")
        assert result.audio.shape == audio.shape

    def test_two_segment_different_strengths(self):
        from backend.core.per_segment_executor import run_phase_per_segment

        sr = 48000
        half = sr // 2
        audio = np.ones(sr, dtype=np.float32)

        def dummy_phase(a, **kw):
            class R:
                pass

            r = R()
            r.audio = a * float(kw.get("strength", 1.0))  # type: ignore[attr-defined]
            return r

        result = run_phase_per_segment(
            audio,
            sr,
            dummy_phase,
            {"strength": 1.0, "sample_rate": sr},
            segment_bounds_s=[0.0, 0.5, 1.0],
            segment_strengths=[0.5, 1.5],
        )
        out = result.audio
        # Erste Hälfte sollte ~0.5 sein
        assert 0.4 < np.mean(out[:half]) < 0.6
        # Zweite Hälfte sollte ~1.5 sein
        assert 1.3 < np.mean(out[half:]) < 1.7

    def test_get_segment_strengths_uniform(self):
        from backend.core.fahrplan import build_fahrplan
        from backend.core.per_segment_executor import get_segment_strengths_from_fahrplan

        fp = build_fahrplan(
            phase_ids=["phase_03_denoise"],
            sections=[(0, 10, "verse"), (10, 20, "verse")],
            goal_priorities={},
        )
        # Both segments are "verse" → same strength → uniform
        info = get_segment_strengths_from_fahrplan(fp, "phase_03_denoise", 0.85)
        assert info is None  # Uniform → None

    def test_get_segment_strengths_non_uniform(self):
        from backend.core.fahrplan import build_fahrplan
        from backend.core.per_segment_executor import get_segment_strengths_from_fahrplan

        fp = build_fahrplan(
            phase_ids=["phase_03_denoise"],
            sections=[(0, 10, "intro"), (10, 30, "verse"), (30, 45, "chorus")],
            goal_priorities={"transparenz": 0.7},
        )
        info = get_segment_strengths_from_fahrplan(fp, "phase_03_denoise", 1.0)
        # Intro gets 0.6x, verse gets 1.15x, chorus gets 1.0x
        assert info is not None
        bounds, strengths = info
        assert len(bounds) == 4  # [0, 10, 30, 45]
        assert len(strengths) == 3
        # At least two different strengths
        assert len({round(s, 2) for s in strengths}) >= 2

    def test_stereo_audio_preserved(self):
        from backend.core.per_segment_executor import run_phase_per_segment

        sr = 48000
        audio = np.ones((2, sr // 2), dtype=np.float32)

        def dummy_phase(a, **kw):
            class R:
                pass

            r = R()
            r.audio = a  # type: ignore[attr-defined]
            return r

        result = run_phase_per_segment(
            audio,
            sr,
            dummy_phase,
            {"strength": 1.0, "sample_rate": sr},
            segment_bounds_s=[0.0, 0.25, 0.5],
            segment_strengths=[0.5, 1.0],
        )
        out = result.audio
        assert out.ndim == 2
        assert out.shape[0] == 2


# ═══════════════════════════════════════════════════════════════
# §2.63 Closed-Loop PID
# ═══════════════════════════════════════════════════════════════


class TestClosedLoopPID:
    """Testet ClosedLoopPIDController."""

    def test_pid_no_targets_disabled(self):
        from backend.core.closed_loop_pid import ClosedLoopPIDController

        pid = ClosedLoopPIDController(None)
        assert not pid._enabled
        assert pid.before_phase("phase_03_denoise", {"natuerlichkeit": 0.5}) == 1.0

    def test_pid_boost_when_under_target(self):
        from backend.core.closed_loop_pid import ClosedLoopPIDController

        pid = ClosedLoopPIDController({"transparenz": 0.85, "waerme": 0.80})
        # phase_04_eq_correction impacts transparenz (+0.04) and waerme (+0.03)
        mult = pid.before_phase("phase_04_eq_correction", {"transparenz": 0.50, "waerme": 0.45})
        # Both targets are way below target → should boost
        assert mult > 1.0, f"Expected boost, got {mult}"

    def test_pid_dampen_when_above_target(self):
        from backend.core.closed_loop_pid import ClosedLoopPIDController

        pid = ClosedLoopPIDController({"transparenz": 0.80, "natuerlichkeit": 0.75})
        # phase_03_denoise impacts transparenz (+0.06) but natuerlichkeit (-0.02)
        mult = pid.before_phase("phase_03_denoise", {"transparenz": 0.90, "natuerlichkeit": 0.85})
        # Both above target → net should dampen
        assert mult < 1.05, f"Expected near or below 1.0, got {mult}"

    def test_pid_integral_accumulates(self):
        from backend.core.closed_loop_pid import ClosedLoopPIDController

        pid = ClosedLoopPIDController({"transparenz": 0.90})
        # First call: big gap → boost
        m1 = pid.before_phase("phase_03_denoise", {"transparenz": 0.30})
        # Second call: still big gap → integral should increase
        m2 = pid.before_phase("phase_03_denoise", {"transparenz": 0.40})
        state = pid.get_state()
        assert state["enabled"]
        # Integral should be non-zero after two calls
        assert any(abs(v) > 0.001 for v in state["integral"].values())

    def test_pid_after_phase_noop(self):
        from backend.core.closed_loop_pid import ClosedLoopPIDController

        pid = ClosedLoopPIDController({"transparenz": 0.80})
        # Should not raise
        pid.after_phase("phase_03_denoise", {"transparenz": 0.75})

    def test_pid_unknown_phase_returns_one(self):
        from backend.core.closed_loop_pid import ClosedLoopPIDController

        pid = ClosedLoopPIDController({"natuerlichkeit": 0.80})
        mult = pid.before_phase("phase_nonexistent_999", {"natuerlichkeit": 0.40})
        assert mult == 1.0

    def test_pid_get_state_rounds_values(self):
        from backend.core.closed_loop_pid import ClosedLoopPIDController

        pid = ClosedLoopPIDController({"transparenz": 0.85})
        pid.before_phase("phase_03_denoise", {"transparenz": 0.50})
        state = pid.get_state()
        for v in state["integral"].values():
            # Should be rounded to 4 decimal places
            assert v == round(v, 4)


# ═══════════════════════════════════════════════════════════════
# §3.0a Source-Aware Fahrplan
# ═══════════════════════════════════════════════════════════════


class TestSourceAwareFahrplan:
    """Testet get_stem_config() und filter_phases_for_stem()."""

    def test_vocals_skips_denoise(self):
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("vocals")
        assert cfg.phase_strengths.get("phase_03_denoise", 1.0) == 0.0
        assert cfg.skip_all_default

    def test_drums_skips_eq_exciter(self):
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("drums")
        assert cfg.phase_strengths.get("phase_04_eq_correction", 1.0) == 0.0
        assert cfg.phase_strengths.get("phase_21_exciter", 1.0) == 0.0

    def test_other_has_full_pipeline(self):
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("other")
        assert not cfg.skip_all_default
        assert cfg.phase_strengths.get("_default", 0.0) > 0.0

    def test_filter_phases_for_stem_vocals(self):
        from backend.core.source_aware_fahrplan import filter_phases_for_stem

        plan = ["phase_03_denoise", "phase_19_de_esser", "phase_21_exciter", "phase_04_eq_correction"]
        filtered = filter_phases_for_stem(plan, "vocals")
        # Denoise and Exciter should be skipped for vocals
        assert "phase_03_denoise" not in filtered
        assert "phase_21_exciter" not in filtered
        assert "phase_19_de_esser" in filtered

    def test_filter_phases_for_stem_drums(self):
        from backend.core.source_aware_fahrplan import filter_phases_for_stem

        plan = ["phase_03_denoise", "phase_08_transient_preservation", "phase_21_exciter"]
        filtered = filter_phases_for_stem(plan, "drums")
        assert "phase_21_exciter" not in filtered
        assert "phase_08_transient_preservation" in filtered

    def test_stem_remix_gains_reasonable(self):
        from backend.core.source_aware_fahrplan import STEM_REMIX_GAINS

        for stem, gain in STEM_REMIX_GAINS.items():
            assert 0.5 < gain < 1.5, f"Gain for {stem} out of range: {gain}"

    def test_unknown_stem_returns_other_config(self):
        from backend.core.source_aware_fahrplan import get_stem_config

        cfg = get_stem_config("nonexistent_stem")
        assert not cfg.skip_all_default  # Falls back to "other"
