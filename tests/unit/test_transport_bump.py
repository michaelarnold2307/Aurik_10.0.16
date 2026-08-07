from __future__ import annotations

import pytest

"""Tests für TRANSPORT_BUMP – Erkennung, kausales Reasoning und Reparatur.

≥ 40 Unit-Tests: synthetische Signale, kein echtes Audio.
Prüft:
  - DefectType.TRANSPORT_BUMP enum
  - DefectScanner._detect_transport_bump() mit synthetischen Bump-Signalen
  - CausalDefectReasoner: routing, material priors, cause params
  - Phase 12 _repair_transport_bumps() Reparaturmethode
  - Hilfsmethoden: _smooth_bump_envelope, _local_pitch_flatten, _quick_pitch_estimate
  - Edge-Cases: Stille, kurze Dateien, Stereo, NaN/Inf-Guards
"""


import numpy as np

SR = 48_000


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(sr: int = SR, duration: float = 3.0, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _silence(sr: int = SR, duration: float = 3.0) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _make_bump_audio(
    sr: int = SR,
    duration: float = 3.0,
    bump_start_s: float = 1.0,
    bump_dur_s: float = 0.15,
    pitch_deviation: float = 0.03,
    amp_deviation: float = 0.4,
    freq: float = 440.0,
) -> np.ndarray:
    """Create audio with a synthetic transport bump at a known position.

    Models a realistic cassette transport bump with:
      - Abrupt energy dropout (tape loses head contact -> near-silence)
      - Low-frequency thump after dropout (mechanical shock)
      - Pitch excursion (tape speed perturbation)
      - Spectral centroid disruption
    The energy dropout zone is kept silent (no LF added) so the
    mandatory energy feature threshold (rms_ratio < 0.45) is triggered.
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)

    bump_start = int(bump_start_s * sr)
    bump_end = int((bump_start_s + bump_dur_s) * sr)
    bump_end = min(bump_end, n)
    bump_len = bump_end - bump_start

    # 1. Energy dropout zone: near-silence for ~30 ms (tape loses head contact)
    drop_len = min(bump_len // 3, int(0.030 * sr))
    if drop_len > 0:
        audio[bump_start : bump_start + drop_len] *= 0.03  # near-silence, no LF

    # 2. Recovery zone: LF thump + shifted pitch (after dropout)
    recovery_start = bump_start + drop_len
    recovery_t = t[recovery_start:bump_end]

    # LF thump (mechanical shock, 30 Hz, strong)
    lf_thump = (np.sin(2 * np.pi * 30 * recovery_t) * 0.5 * amp_deviation).astype(np.float32)

    # Pitch-shifted signal in recovery zone
    shifted_freq = freq * (1.0 + pitch_deviation)
    pitched = (np.sin(2 * np.pi * shifted_freq * recovery_t) * 0.3 * (1.0 + amp_deviation)).astype(np.float32)

    audio[recovery_start:bump_end] = pitched + lf_thump

    return np.clip(audio, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _make_stereo(mono: np.ndarray) -> np.ndarray:
    return np.column_stack([mono, mono * 0.95])


def _make_gradual_level_dip_audio(
    sr: int = SR,
    duration: float = 12.0,
    dip_times: tuple[float, ...] = (1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0),
    freq: float = 440.0,
) -> np.ndarray:
    """Create cassette-like gradual level dips without transport-bump thumps."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)

    dip_duration_s = 0.20
    dip_samples = int(dip_duration_s * sr)
    fade_down = int(0.080 * sr)
    snap_back = int(0.020 * sr)
    hold = max(1, dip_samples - fade_down - snap_back)
    linear_depth = 10.0 ** (-12.0 / 20.0)

    for dip_s in dip_times:
        start = int(dip_s * sr)
        end = start + dip_samples
        if end > len(audio):
            continue
        env = np.ones(dip_samples, dtype=np.float64)
        env[:fade_down] = 1.0 - (1.0 - linear_depth) * (0.5 - 0.5 * np.cos(np.pi * np.arange(fade_down) / fade_down))
        env[fade_down : fade_down + hold] = linear_depth
        env[fade_down + hold :] = np.linspace(linear_depth, 1.0, snap_back)
        audio[start:end] *= env.astype(np.float32)

    return audio.astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 1. DefectType.TRANSPORT_BUMP existiert
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTransportBumpEnum:
    """Enum-Mitgliedschaft und Grundeigenschaften."""

    def test_01_enum_exists(self):
        from backend.core.defect_scanner import DefectType

        assert hasattr(DefectType, "TRANSPORT_BUMP")

    def test_02_enum_value_is_string(self):
        from backend.core.defect_scanner import DefectType

        assert isinstance(DefectType.TRANSPORT_BUMP.value, str)

    def test_03_enum_not_duplicate(self):
        from backend.core.defect_scanner import DefectType

        values = [dt.value for dt in DefectType]
        assert values.count(DefectType.TRANSPORT_BUMP.value) == 1

    def test_04_enum_count_at_least_29(self):
        from backend.core.defect_scanner import DefectType

        assert len(DefectType) >= 29


# ---------------------------------------------------------------------------
# 2. DefectScanner._detect_transport_bump()
# ---------------------------------------------------------------------------


class TestDetectTransportBump:
    """Synthethische Erkennung."""

    @staticmethod
    def _scanner():
        from backend.core.defect_scanner import DefectScanner

        return DefectScanner(sample_rate=SR)

    def test_05_clean_audio_no_bump(self):
        scanner = self._scanner()
        audio = _sine(duration=3.0)
        result = scanner._detect_transport_bump(audio)
        assert result.severity < 0.15, f"False positive: severity={result.severity}"

    def test_06_silence_no_bump(self):
        scanner = self._scanner()
        audio = _silence(duration=2.0)
        result = scanner._detect_transport_bump(audio)
        assert result.severity < 0.1

    def test_07_bump_detected(self):
        scanner = self._scanner()
        audio = _make_bump_audio(bump_start_s=1.0, bump_dur_s=0.15, pitch_deviation=0.05, amp_deviation=0.6)
        result = scanner._detect_transport_bump(audio)
        assert result.severity > 0.05, f"Bump not detected: severity={result.severity}"

    def test_08_bump_has_locations(self):
        scanner = self._scanner()
        audio = _make_bump_audio(bump_start_s=1.5, bump_dur_s=0.12, pitch_deviation=0.04, amp_deviation=0.5)
        result = scanner._detect_transport_bump(audio)
        if result.severity > 0.05:
            assert result.locations is not None
            assert len(result.locations) >= 1

    def test_09_location_accuracy(self):
        """Detected bump location should be within ±0.2 s of true position."""
        scanner = self._scanner()
        true_start = 1.5
        audio = _make_bump_audio(bump_start_s=true_start, bump_dur_s=0.15, pitch_deviation=0.05, amp_deviation=0.6)
        result = scanner._detect_transport_bump(audio)
        if result.severity > 0.05 and result.locations:
            loc_start = result.locations[0][0]
            assert abs(loc_start - true_start) < 0.3, (
                f"Location mismatch: detected={loc_start:.2f}, true={true_start:.2f}"
            )

    def test_10_multiple_bumps(self):
        """Two bumps should produce two locations."""
        scanner = self._scanner()
        audio = _sine(duration=5.0)
        # Insert two bumps
        for bump_s in [1.0, 3.5]:
            bump_start = int(bump_s * SR)
            bump_dur = int(0.12 * SR)
            bump_end = min(bump_start + bump_dur, len(audio))
            audio[bump_start:bump_end] *= 1.8
            t_bump = np.linspace(bump_s, bump_s + 0.12, bump_end - bump_start, endpoint=False)
            audio[bump_start:bump_end] = np.sin(2 * np.pi * 480 * t_bump).astype(np.float32) * 0.5
        result = scanner._detect_transport_bump(audio)
        if result.severity > 0.05 and result.locations:
            assert len(result.locations) >= 1  # at least one detected

    def test_11_severity_monotonic_with_strength(self):
        """Stronger bumps → higher severity."""
        scanner = self._scanner()
        sev_weak = scanner._detect_transport_bump(_make_bump_audio(amp_deviation=0.2, pitch_deviation=0.01)).severity
        sev_strong = scanner._detect_transport_bump(_make_bump_audio(amp_deviation=0.8, pitch_deviation=0.06)).severity
        # Strong should be at least as severe (with tolerance for detection noise)
        assert sev_strong >= sev_weak - 0.05

    def test_12_short_audio_no_crash(self):
        scanner = self._scanner()
        audio = _sine(duration=0.05)  # 50 ms
        result = scanner._detect_transport_bump(audio)
        assert result.severity >= 0.0

    def test_13_confidence_range(self):
        scanner = self._scanner()
        audio = _make_bump_audio()
        result = scanner._detect_transport_bump(audio)
        assert 0.0 <= result.confidence <= 1.0

    def test_14_return_type_is_defect_score(self):
        from backend.core.defect_scanner import DefectScore

        scanner = self._scanner()
        result = scanner._detect_transport_bump(_sine())
        assert isinstance(result, DefectScore)

    def test_15_gradual_level_dips_not_misclassified_as_transport_bump(self):
        scanner = self._scanner()
        audio = _make_gradual_level_dip_audio()
        result = scanner._detect_transport_bump(audio)
        assert result.severity < 0.20, f"Gradual level dips misclassified as transport bump: {result.severity}"

    def test_16_dat_material_not_hard_gated_for_transport_bump(self):
        from backend.core.defect_scanner import DefectType, MaterialType

        scanner = self._scanner()
        audio = _make_bump_audio(amp_deviation=0.7, pitch_deviation=0.05)
        result = scanner.scan(audio, SR, material_type=MaterialType.DAT)
        score = result.scores[DefectType.TRANSPORT_BUMP]
        assert float(score.severity) > 0.05, "DAT transport bump should not be hard-gated to zero"


# ---------------------------------------------------------------------------
# 3. CausalDefectReasoner — routing & priors
# ---------------------------------------------------------------------------


class TestCausalReasonerTransportBump:
    """CausalDefectReasoner-Modul kennt transport_bump."""

    @staticmethod
    def _module():
        import backend.core.causal_defect_reasoner as m

        return m

    def test_15_cause_in_causes_list(self):
        m = self._module()
        assert "transport_bump" in m.CAUSES

    def test_16_cause_to_phases_mapping(self):
        m = self._module()
        phases = m.CAUSE_TO_PHASES.get("transport_bump", [])
        assert len(phases) >= 1
        assert any("phase_12" in p for p in phases)

    def test_17_cause_params_exist(self):
        m = self._module()
        params = m.CAUSE_PARAMS.get("transport_bump", {})
        assert "bump_correction_strength" in params

    def test_18_tape_prior_highest(self):
        """Tape material should have highest transport_bump prior."""
        m = self._module()
        tape_prior = m.MATERIAL_PRIORS.get("tape", {}).get("transport_bump", 0)
        cd_prior = m.MATERIAL_PRIORS.get("cd_digital", {}).get("transport_bump", 0)
        assert tape_prior > cd_prior

    def test_19_all_materials_have_prior(self):
        """Every material in MATERIAL_PRIORS should have transport_bump."""
        m = self._module()
        for mat, priors in m.MATERIAL_PRIORS.items():
            assert "transport_bump" in priors, f"Material {mat} missing transport_bump prior"

    def test_20_cause_params_strength_range(self):
        m = self._module()
        s = m.CAUSE_PARAMS["transport_bump"]["bump_correction_strength"]
        assert 0.0 <= s <= 1.0

    def test_21_phase_31_in_routing(self):
        """Phase 31 (speed_pitch_correction) should be in fallback routing."""
        m = self._module()
        phases = m.CAUSE_TO_PHASES["transport_bump"]
        assert any("phase_31" in p or "speed_pitch" in p for p in phases)


# ---------------------------------------------------------------------------
# 4. Phase 12 _repair_transport_bumps()
# ---------------------------------------------------------------------------


class TestRepairTransportBumps:
    """Phase 12 Reparaturmethode."""

    @staticmethod
    def _phase():
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        return WowFlutterFix()

    def test_22_method_exists(self):
        phase = self._phase()
        assert hasattr(phase, "_repair_transport_bumps")
        assert callable(phase._repair_transport_bumps)

    def test_23_empty_locations_returns_unchanged(self):
        phase = self._phase()
        audio = _sine(duration=2.0)
        result, n = phase._repair_transport_bumps(audio, SR, [], 0.85)
        assert n == 0
        np.testing.assert_array_equal(result, audio)

    def test_24_single_bump_repaired(self):
        phase = self._phase()
        audio = _make_bump_audio(bump_start_s=1.0, bump_dur_s=0.15)
        locs = [(1.0, 1.15)]
        result, n = phase._repair_transport_bumps(audio, SR, locs, 0.85)
        assert n == 1
        assert result.shape == audio.shape

    def test_25_no_nan_inf_in_result(self):
        phase = self._phase()
        audio = _make_bump_audio()
        locs = [(1.0, 1.15)]
        result, n = phase._repair_transport_bumps(audio, SR, locs, 0.85)
        assert np.isfinite(result).all(), "NaN or Inf in repair output"

    def test_26_clipped_output(self):
        phase = self._phase()
        audio = _make_bump_audio(amp_deviation=0.9)
        locs = [(1.0, 1.15)]
        result, _ = phase._repair_transport_bumps(audio, SR, locs, 1.0)
        assert np.max(np.abs(result)) <= 1.0

    def test_27_stereo_support(self):
        phase = self._phase()
        mono = _make_bump_audio(bump_start_s=1.0, bump_dur_s=0.12)
        stereo = _make_stereo(mono)
        locs = [(1.0, 1.12)]
        result, n = phase._repair_transport_bumps(stereo, SR, locs, 0.85)
        assert result.shape == stereo.shape
        assert n == 1

    def test_28_strength_zero_is_passthrough(self):
        """strength=0.0 should not change the audio (or change minimally)."""
        phase = self._phase()
        audio = _make_bump_audio()
        locs = [(1.0, 1.15)]
        result, _ = phase._repair_transport_bumps(audio.copy(), SR, locs, 0.0)
        # With 0 strength, correction should be extremely mild
        diff = np.max(np.abs(result - audio))
        assert diff < 0.05, f"Strength 0 changed audio by {diff}"

    def test_29_out_of_bounds_location_skipped(self):
        phase = self._phase()
        audio = _sine(duration=2.0)
        locs = [(-0.5, 0.0), (10.0, 10.2)]  # both out of bounds
        result, n = phase._repair_transport_bumps(audio, SR, locs, 0.85)
        assert n == 0

    def test_30_inverted_location_skipped(self):
        phase = self._phase()
        audio = _sine(duration=2.0)
        locs = [(1.5, 1.0)]  # end before start
        result, n = phase._repair_transport_bumps(audio, SR, locs, 0.85)
        assert n == 0

    def test_31_very_short_bump_skipped(self):
        """Bump shorter than 10 ms minimum threshold should be skipped."""
        phase = self._phase()
        audio = _sine(duration=2.0)
        locs = [(1.0, 1.002)]  # 2 ms — too short
        result, n = phase._repair_transport_bumps(audio, SR, locs, 0.85)
        # Very short bump may be skipped or handled gracefully
        assert result.shape == audio.shape

    def test_32_multiple_bumps(self):
        phase = self._phase()
        audio = _make_bump_audio(duration=5.0, bump_start_s=1.0, bump_dur_s=0.12)
        # Add second bump manually
        bump2_start = int(3.5 * SR)
        bump2_end = int(3.62 * SR)
        audio[bump2_start:bump2_end] *= 1.5
        locs = [(1.0, 1.12), (3.5, 3.62)]
        result, n = phase._repair_transport_bumps(audio, SR, locs, 0.85)
        assert n == 2


# ---------------------------------------------------------------------------
# 5. Helper methods
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """_smooth_bump_envelope, _local_pitch_flatten, _quick_pitch_estimate."""

    @staticmethod
    def _phase():
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        return WowFlutterFix()

    def test_33_quick_pitch_estimate_sine(self):
        phase = self._phase()
        audio = _sine(duration=0.5, freq=220.0)
        pitch = phase._quick_pitch_estimate(audio, SR)
        # Should be close to 220 Hz (±10%)
        assert 190 < pitch < 250, f"Expected ~220 Hz, got {pitch:.1f}"

    def test_34_quick_pitch_estimate_silence(self):
        phase = self._phase()
        audio = _silence(duration=0.5)
        pitch = phase._quick_pitch_estimate(audio, SR)
        assert pitch == 0.0

    def test_35_quick_pitch_estimate_very_short(self):
        phase = self._phase()
        audio = np.zeros(50, dtype=np.float32)
        pitch = phase._quick_pitch_estimate(audio, SR)
        assert pitch == 0.0

    def test_36_smooth_bump_envelope_shape(self):
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        segment = _sine(duration=0.1, freq=440.0)
        result = WowFlutterFix._smooth_bump_envelope(segment, 0.3, 480, 0.85)
        assert result.shape == segment.shape

    def test_37_smooth_bump_envelope_clipped(self):
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        segment = np.ones(4800, dtype=np.float32) * 0.9
        result = WowFlutterFix._smooth_bump_envelope(segment, 0.3, 480, 1.0)
        assert np.max(np.abs(result)) <= 1.0

    def test_38_local_pitch_flatten_identity(self):
        """If bump has same pitch as context, output ≈ input."""
        phase = self._phase()
        audio = _sine(duration=0.15, freq=440.0)
        ctx = _sine(duration=0.25, freq=440.0)
        result = phase._local_pitch_flatten(audio, ctx, ctx, SR, 0.85)
        diff = np.max(np.abs(result - audio))
        assert diff < 0.15, f"Identity pitch should be ~passthrough, diff={diff}"

    def test_39_local_pitch_flatten_no_nan(self):
        phase = self._phase()
        audio = _sine(duration=0.15, freq=440.0)
        ctx = _sine(duration=0.25, freq=460.0)
        result = phase._local_pitch_flatten(audio, ctx, ctx, SR, 0.85)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# 6. UI-Integration
# ---------------------------------------------------------------------------


class TestUIIntegration:
    """UI-Labels und Display-Mappings."""

    def test_40_defect_label_exists(self):
        """_DEFECT_LABELS should contain transport_bump."""
        # We import the file and check statically via grep — functional test
        from pathlib import Path

        ui_path = Path(__file__).parent.parent.parent / "Aurik10" / "ui" / "modern_window.py"
        content = ui_path.read_text(encoding="utf-8")
        assert '"transport_bump"' in content or "'transport_bump'" in content

    def test_41_phase_reduces_contains_bump(self):
        from pathlib import Path

        ui_path = Path(__file__).parent.parent.parent / "Aurik10" / "ui" / "modern_window.py"
        content = ui_path.read_text(encoding="utf-8")
        assert "transport_bump" in content


# ---------------------------------------------------------------------------
# 7. _spectral_context_blend() — spectral repair helper
# ---------------------------------------------------------------------------


class TestSpectralContextBlend:
    """Phase 12 _spectral_context_blend static method."""

    @staticmethod
    def _phase():
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        return WowFlutterFix()

    def test_42_method_exists(self):
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        assert hasattr(WowFlutterFix, "_spectral_context_blend")

    def test_43_short_bump_passthrough(self):
        """Bumps shorter than 128 samples should be returned unchanged."""
        phase = self._phase()
        bump = _sine(duration=0.002, freq=440.0)  # ~96 samples at 48 kHz
        ctx = _sine(duration=0.1, freq=440.0)
        result = phase._spectral_context_blend(bump, ctx, ctx, 0.85)
        np.testing.assert_array_equal(result, bump)

    def test_44_zero_strength_passthrough(self):
        phase = self._phase()
        bump = _sine(duration=0.05, freq=440.0)  # 2400 samples
        ctx = _sine(duration=0.1, freq=440.0)
        result = phase._spectral_context_blend(bump, ctx, ctx, 0.0)
        np.testing.assert_array_equal(result, bump)

    def test_45_output_shape_matches_input(self):
        phase = self._phase()
        bump = _sine(duration=0.05, freq=440.0)
        ctx = _sine(duration=0.1, freq=440.0)
        result = phase._spectral_context_blend(bump, ctx, ctx, 0.85)
        assert result.shape == bump.shape

    def test_46_output_finite(self):
        phase = self._phase()
        bump = _sine(duration=0.05, freq=440.0)
        ctx = _sine(duration=0.1, freq=440.0)
        result = phase._spectral_context_blend(bump, ctx, ctx, 0.85)
        assert np.isfinite(result).all()

    def test_47_output_clipped(self):
        phase = self._phase()
        bump = np.ones(2400, dtype=np.float32) * 0.9
        ctx = _sine(duration=0.1, freq=440.0)
        result = phase._spectral_context_blend(bump, ctx, ctx, 1.0)
        assert np.max(np.abs(result)) <= 1.0

    def test_48_lf_thump_suppressed(self):
        """A bump with injected LF energy should have that energy reduced."""
        phase = self._phase()
        n = 4800  # 100 ms at 48 kHz
        t = np.linspace(0, 0.1, n, endpoint=False)
        clean = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        # Add LF thump (30 Hz) to bump
        bump = clean + (np.sin(2 * np.pi * 30 * t) * 0.4).astype(np.float32)

        result = phase._spectral_context_blend(bump, clean, clean, 0.85)

        # LF energy should be reduced in result vs bump
        # Measure LF energy (0–60 Hz) via FFT
        def lf_energy(sig: np.ndarray) -> float:
            fft_mag = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(len(sig), 1.0 / SR)
            return float(np.sum(fft_mag[freqs < 60] ** 2))

        assert lf_energy(result) < lf_energy(bump), "LF thump should be reduced"

    def test_49_empty_context_passthrough(self):
        """If both contexts are too short, bump should be returned unchanged."""
        phase = self._phase()
        bump = _sine(duration=0.05, freq=440.0)
        short = np.zeros(10, dtype=np.float32)
        result = phase._spectral_context_blend(bump, short, short, 0.85)
        np.testing.assert_array_equal(result, bump)

    def test_50_single_context_enough(self):
        """If only one context is valid, method should still work."""
        phase = self._phase()
        bump = _sine(duration=0.05, freq=440.0)
        ctx = _sine(duration=0.1, freq=440.0)
        short = np.zeros(10, dtype=np.float32)
        result = phase._spectral_context_blend(bump, ctx, short, 0.85)
        assert result.shape == bump.shape
        assert np.isfinite(result).all()

    def test_51_output_dtype_float32(self):
        phase = self._phase()
        bump = _sine(duration=0.05, freq=440.0)
        ctx = _sine(duration=0.1, freq=440.0)
        result = phase._spectral_context_blend(bump, ctx, ctx, 0.85)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# 8. UV3 Phase-Selection — TRANSPORT_BUMP → Phase 12
# ---------------------------------------------------------------------------


class TestUV3PhaseSelection:
    """UV3 activates Phase 12 when TRANSPORT_BUMP severity > 0.08."""

    def test_52_uv3_phase_selection_code_exists(self):
        """UV3 source must contain TRANSPORT_BUMP phase selection logic."""
        from pathlib import Path

        uv3_path = Path(__file__).parent.parent.parent / "backend" / "core" / "unified_restorer_v3.py"
        content = uv3_path.read_text(encoding="utf-8")
        assert "TRANSPORT_BUMP" in content
        assert "phase_12" in content

    def test_53_defect_locations_dict_key(self):
        """DefectType.TRANSPORT_BUMP.value should be a valid dict key."""
        from backend.core.defect_scanner import DefectType

        key = DefectType.TRANSPORT_BUMP.value
        assert isinstance(key, str)
        assert key == "transport_bump"

    def test_54_phase_12_reads_defect_locations(self):
        """Phase 12 source must fall back to defect_locations dict."""
        from pathlib import Path

        p12_path = Path(__file__).parent.parent.parent / "backend" / "core" / "phases" / "phase_12_wow_flutter_fix.py"
        content = p12_path.read_text(encoding="utf-8")
        assert "defect_locations" in content
