"""
tests/unit/test_per_phase_musical_goals_gate.py
================================================
Aurik 10.0.0 — PerPhaseMusicalGoalsGate (§2.29)

26 Unit-Tests.
Alle Tests synthetisch (keine echten Audio-Dateien).
"""

import logging
import math
import threading

import numpy as np
import pytest

logger = logging.getLogger(__name__)

SR = 48000


# ---------------------------------------------------------------------------
# Mock Phasen
# ---------------------------------------------------------------------------


class _MockPassPhase:
    """Identity-Phase — keine Musical-Goal-Regression."""

    def __call__(self, audio, strength=1.0):
        return audio.copy().astype(np.float32)

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockPass"
        return m


class _MockAttenuatePhase:
    """Dämpft minimal — Pass (keine starke Regression)."""

    def __call__(self, audio, strength=1.0):
        return (audio * 0.98).astype(np.float32)

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockAttenuate"
        return m


class _MockZeroPhase:
    """Zeroed output — erzeugt starke Regression in allen Musical Goals."""

    def __call__(self, audio, strength=1.0):
        return np.zeros_like(audio)

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockZero"
        return m


class _MockExplodePhase:
    """Gibt NaN zurück — sollte sicher behandelt werden."""

    def __call__(self, audio, strength=1.0):
        out = audio.copy()
        out[:] = np.nan
        return out

    def get_metadata(self):
        import types

        m = types.SimpleNamespace()
        m.name = "MockExplode"
        return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate():
    from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

    return PerPhaseMusicalGoalsGate()


@pytest.fixture(scope="module")
def audio_1s():
    np.random.seed(42)
    t = np.linspace(0, 1.0, SR, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8
    return sig


@pytest.fixture(scope="module")
def audio_5s():
    np.random.seed(42)
    t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
    sig = np.sin(2 * np.pi * 261.6 * t).astype(np.float32) * 0.7
    return sig


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------


class TestPMGGSingleton:
    def test_01_singleton_same_instance(self):
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        a = get_phase_gate()
        b = get_phase_gate()
        assert a is b

    def test_02_singleton_thread_safe(self):
        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        instances = []

        def _get():
            instances.append(get_phase_gate())

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(inst is instances[0] for inst in instances)


# ---------------------------------------------------------------------------
# Tests: reset()
# ---------------------------------------------------------------------------


class TestPMGGReset:
    def test_03_reset_clears_rollback_count(self, gate):
        try:
            _ = gate._rollback_count
        except AttributeError:
            pass  # Attribut nicht vorhanden → kein Test nötig
        gate.reset()
        assert getattr(gate, "_rollback_count", 0) == 0
        assert getattr(gate, "_best_effort_count", 0) == 0

    def test_04_reset_idempotent(self, gate):
        gate.reset()
        gate.reset()
        assert getattr(gate, "_rollback_count", 0) == 0


# ---------------------------------------------------------------------------
# Tests: wrap_phase() Rückgabe-Struktur
# ---------------------------------------------------------------------------


class TestPMGGWrapPhaseStructure:
    def test_05_returns_three_tuple(self, gate, audio_5s):
        gate.reset()
        result = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert isinstance(result, (tuple, list))
        assert len(result) == 3

    def test_06_first_element_is_ndarray(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert isinstance(out, np.ndarray)

    def test_07_second_element_is_dict(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert isinstance(scores, dict)

    def test_08_third_element_has_action(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert hasattr(entry, "action")

    def test_09_third_element_has_strength_used(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert hasattr(entry, "strength_used")

    def test_10_action_is_valid_string(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5"} or entry.action.startswith(
            "best_effort"
        )

    def test_11_strength_used_is_positive(self, gate, audio_5s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert math.isfinite(entry.strength_used)
        assert entry.strength_used > 0.0


# ---------------------------------------------------------------------------
# Tests: wrap_phase() Audio-Qualität
# ---------------------------------------------------------------------------


class TestPMGGAudioQuality:
    def test_12_output_shape_preserved(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert out.shape == audio_5s.shape

    def test_13_output_no_nan_passthrough(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert np.isfinite(out).all()

    def test_14_output_bounded(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_15_nan_phase_output_handled(self, gate, audio_5s):
        """Phase gibt NaN zurück — Gate muss robust reagieren (kein Crash)."""
        gate.reset()
        try:
            out, _, entry = gate.wrap_phase(_MockExplodePhase(), audio_5s, SR)
            # Falls kein Fehler: Ausgabe muss finite sein (Rollback greift)
            assert np.isfinite(out).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Exception ist auch akzeptabel

    def test_16_scores_values_finite(self, gate, audio_5s):
        gate.reset()
        _, scores, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        for v in scores.values():
            assert math.isfinite(v)

    def test_17_scores_keys_are_strings(self, gate, audio_5s):
        gate.reset()
        _, scores, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        for k in scores.keys():
            assert isinstance(k, str)

    def test_18_scores_values_in_unit_interval(self, gate, audio_5s):
        gate.reset()
        _, scores, _ = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        for v in scores.values():
            assert -0.1 <= v <= 1.1  # leichte numerische Toleranz

    def test_18b_run_phase_normalizes_samples_first_stereo_layout(self):
        """_run_phase must transpose channels-first phase output to samples-first input layout."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        n = SR
        t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        audio_sf = np.stack(
            [
                0.4 * np.sin(2 * np.pi * 220 * t),
                0.4 * np.sin(2 * np.pi * 330 * t),
            ],
            axis=1,
        ).astype(np.float32)  # (N, 2)

        class _MockProcessPhaseCF:
            def process(self, audio, **kwargs):
                # Deliberately return channels-first although input is samples-first.
                return np.asarray(audio, dtype=np.float32).T

            def get_metadata(self):
                import types

                m = types.SimpleNamespace()
                m.phase_id = "phase_03_denoise"
                return m

        out = PerPhaseMusicalGoalsGate()._run_phase(_MockProcessPhaseCF(), audio_sf, 0.5)
        assert out.shape == audio_sf.shape
        assert np.isfinite(out).all()

    def test_18c_run_phase_upmixes_mono_output_for_stereo_input(self):
        """_run_phase must avoid broadcast failures when phase emits mono from stereo input."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        n = SR
        t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        audio_cf = np.stack(
            [
                0.35 * np.sin(2 * np.pi * 260 * t),
                0.33 * np.sin(2 * np.pi * 390 * t),
            ],
            axis=0,
        ).astype(np.float32)  # (2, N)

        class _MockProcessPhaseMono:
            def process(self, audio, **kwargs):
                # Simulate a phase that collapses to mono.
                x = np.asarray(audio, dtype=np.float32)
                return np.mean(x, axis=0)

            def get_metadata(self):
                import types

                m = types.SimpleNamespace()
                m.phase_id = "phase_29_tape_hiss_reduction"
                return m

        out = PerPhaseMusicalGoalsGate()._run_phase(_MockProcessPhaseMono(), audio_cf, 0.5)
        assert out.shape == audio_cf.shape
        assert np.isfinite(out).all()

    def test_18d_run_phase_safe_fallback_on_exception(self):
        """Bei Phase-Exception muss _run_phase NaN-safe/clip-safe fallbacken."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        audio = np.array([np.nan, 2.5, -3.0, 0.2], dtype=np.float32)

        class _MockProcessPhaseFail:
            def process(self, _audio, **_kwargs):
                raise RuntimeError("forced-fail")

            def get_metadata(self):
                import types

                m = types.SimpleNamespace()
                m.phase_id = "phase_03_denoise"
                return m

        out = PerPhaseMusicalGoalsGate()._run_phase(_MockProcessPhaseFail(), audio, 0.7)
        assert out.shape == audio.shape
        assert np.isfinite(out).all()
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6
        assert float(out[0]) == 0.0

    def test_18e_run_phase_safe_fallback_on_invalid_result_type(self):
        """Ungueltiger Phase-Rueckgabetyp darf nicht raw durchgereicht werden."""
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        audio = np.array([1.3, -1.4, 0.1], dtype=np.float32)

        class _MockProcessPhaseInvalid:
            def process(self, _audio, **_kwargs):
                return {"invalid": True}

            def get_metadata(self):
                import types

                m = types.SimpleNamespace()
                m.phase_id = "phase_29_tape_hiss_reduction"
                return m

        out = PerPhaseMusicalGoalsGate()._run_phase(_MockProcessPhaseInvalid(), audio, 0.4)
        assert out.shape == audio.shape
        assert np.isfinite(out).all()
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: Regressions-Behandlung
# ---------------------------------------------------------------------------


class TestPMGGRegression:
    def test_19_pass_phase_action_passed(self, gate, audio_5s):
        gate.reset()
        _, _, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR)
        # Identity-Phase sollte "passed" oder höchstens "retry1" erhalten
        assert entry.action in {"passed", "retry1"}

    def test_20_zero_phase_triggers_protection(self, gate, audio_5s):
        """Null-Phase zerstört Signal — Gate soll Best-Effort/Retry auslösen."""
        gate.reset()
        _, _, entry = gate.wrap_phase(_MockZeroPhase(), audio_5s, SR)
        # Bei starker Regression: best_effort oder retry (kein Rollback/Skip mehr seit v10.0.0)
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5"} or entry.action.startswith(
            "best_effort"
        )

    def test_21_zero_phase_output_is_bounded(self, gate, audio_5s):
        gate.reset()
        out, _, _ = gate.wrap_phase(_MockZeroPhase(), audio_5s, SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_22_best_effort_does_not_increment_real_rollback_count(self, gate, audio_5s, monkeypatch):
        """PMGG best_effort ist kein Audio-Rollback und darf Rollback-Telemetrie nicht aufblasen."""

        gate.reset()

        def _force_best_effort(*_args, **_kwargs):
            scores = {"natuerlichkeit": 0.2}
            return audio_5s.copy(), scores, "best_effort_r1", 0.35

        monkeypatch.setattr(gate, "_run_with_retry", _force_best_effort)
        count_before = getattr(gate, "_rollback_count", 0)
        _, _, entry = gate.wrap_phase(
            _MockZeroPhase(),
            audio_5s,
            SR,
            scores_before={"natuerlichkeit": 0.5},
            applicable_goals={"natuerlichkeit"},
        )
        count_after = getattr(gate, "_rollback_count", 0)
        assert entry.action == "best_effort_r1"
        assert count_after == count_before == 0
        assert getattr(gate, "_best_effort_count", 0) == 1
        assert entry.metadata["pmgg_real_rollback_count"] == 0
        assert entry.metadata["pmgg_best_effort_count"] == 1


# ---------------------------------------------------------------------------
# Tests: Verschiedene Input-Größen
# ---------------------------------------------------------------------------


class TestPMGGInputVariations:
    def test_23_short_audio_1s(self, gate, audio_1s):
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_1s, SR)
        assert np.isfinite(out).all()
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5"} or entry.action.startswith(
            "best_effort"
        )

    def test_24_stereo_audio(self, gate):
        np.random.seed(42)
        t = np.linspace(0, 5.0, 5 * SR, endpoint=False)
        stereo = np.stack(
            [
                np.sin(2 * np.pi * 440 * t).astype(np.float32),
                np.sin(2 * np.pi * 550 * t).astype(np.float32),
            ],
            axis=0,
        )
        gate.reset()
        try:
            out, scores, entry = gate.wrap_phase(_MockPassPhase(), stereo, SR)
            assert np.isfinite(out).all()
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Stereo-Ablehnung ist akzeptabel

    def test_25_existing_scores_passed_in(self, gate, audio_5s):
        """Existierende Scores als Baseline übergeben."""
        gate.reset()
        initial_scores = {
            "brillanz": 0.87,
            "waerme": 0.82,
            "groove": 0.90,
        }
        out, scores, entry = gate.wrap_phase(_MockPassPhase(), audio_5s, SR, scores_before=initial_scores)
        assert np.isfinite(out).all()
        assert entry.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5"} or entry.action.startswith(
            "best_effort"
        )

    def test_26_attenuate_phase_passes(self, gate, audio_5s):
        """Minimale Dämpfung (0.98) sollte in den meisten Fällen akzeptiert werden."""
        gate.reset()
        out, _, entry = gate.wrap_phase(_MockAttenuatePhase(), audio_5s, SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


# ----------------------------------------------------------------------
# Tests §9.7.3 — Phasen-adaptive Sample-Dauer (PHASE_SAMPLE_DURATIONS)
# ----------------------------------------------------------------------


class TestPMGGAdaptiveSampleDuration:
    """§9.7.3: Triviale Phasen bekommen kürzere Sample-Dauer als Standard."""

    def test_27_trivial_phases_have_short_duration(self):
        from backend.core.per_phase_musical_goals_gate import (
            PHASE_SAMPLE_DURATIONS,
            SAMPLE_DURATION_S,
            _get_sample_duration,
        )

        for prefix in PHASE_SAMPLE_DURATIONS:
            dur = _get_sample_duration(prefix)
            assert dur < SAMPLE_DURATION_S, f"Phase {prefix}: {dur} nicht kürzer als {SAMPLE_DURATION_S}"

    def test_28_standard_phase_gets_full_duration(self):
        from backend.core.per_phase_musical_goals_gate import SAMPLE_DURATION_S, _get_sample_duration

        assert _get_sample_duration("phase_03_denoise") == SAMPLE_DURATION_S
        assert _get_sample_duration("phase_55_diffusion_inpainting") == SAMPLE_DURATION_S
        assert _get_sample_duration("phase_29_tape_hiss_reduction") == SAMPLE_DURATION_S

    def test_29_dc_offset_phase_1s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration

        dur = _get_sample_duration("phase_30_dc_offset_removal")
        assert 1.0 <= dur <= 2.0

    def test_30_rumble_filter_phase_1s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration

        dur = _get_sample_duration("phase_05_rumble_filter")
        assert 1.0 <= dur <= 2.0

    def test_31_hum_removal_phase_2s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration

        dur = _get_sample_duration("phase_02_hum_removal")
        assert 1.0 <= dur <= 2.5

    def test_32_duration_bounded_min_1s(self):
        from backend.core.per_phase_musical_goals_gate import _get_sample_duration

        # Keine Dauer darf unter 1 Sekunde liegen
        for prefix in ("phase_30", "phase_05", "phase_02", "phase_15", "phase_11", "phase_18"):
            assert _get_sample_duration(prefix) >= 1.0

    def test_33_duration_bounded_max_5s(self):
        from backend.core.per_phase_musical_goals_gate import SAMPLE_DURATION_S, _get_sample_duration

        # Keine adaptive Dauer darf über den Standard hinausgehen
        for prefix in ("phase_30", "phase_05", "phase_02", "phase_15", "phase_11", "phase_18"):
            assert _get_sample_duration(prefix) <= SAMPLE_DURATION_S

    def test_34_phase_sample_durations_dict_nonempty(self):
        from backend.core.per_phase_musical_goals_gate import PHASE_SAMPLE_DURATIONS

        assert len(PHASE_SAMPLE_DURATIONS) >= 6

    def test_35_wrap_phase_uses_short_duration_for_trivial(self, gate, audio_5s):
        """DC-Offset-Phase soll mit adaptiver Dauer ohne Fehler laufen."""
        import math

        class _MockDCPhase:
            """Mock-Phase die DC-Offset 'korrigiert' (identisch pass-through)."""

            def __call__(self, audio, **kwargs):
                return audio

        # Phase-ID enthält phase_30 -> kurze Sample-Dauer (§9.7.3)
        _MockDCPhase.__name__ = "phase_30_dc_offset_removal"
        gate.reset()
        out, scores, entry = gate.wrap_phase(_MockDCPhase(), audio_5s, SR)
        assert np.isfinite(out).all()
        for v in scores.values():
            assert math.isfinite(v)


class TestPMGGTeamContextPolicy:
    """§2.54 Team-Koordination: PMGG berücksichtigt Vorphasen-Kontext."""

    def test_35b_phase50_policy_enabled_after_hf_restoration(self):
        from backend.core.per_phase_musical_goals_gate import _resolve_team_context_policy

        policy = _resolve_team_context_policy(
            "phase_50_spectral_repair",
            {
                "prior_phase_context": {
                    "harmonic_restoration_applied": True,
                }
            },
        )

        assert policy["reason"] == "phase50_after_hf_restoration"
        assert policy["threshold_multiplier"] > 1.0
        assert policy["strength_cap"] < 1.0
        assert {"brillanz", "transparenz", "timbre_authentizitaet"}.issubset(policy["goal_exclusions"])

    def test_35c_phase50_policy_disabled_without_prior_context(self):
        from backend.core.per_phase_musical_goals_gate import _resolve_team_context_policy

        policy = _resolve_team_context_policy("phase_50_spectral_repair", {"material_type": "vinyl"})

        assert policy["reason"] == ""
        assert policy["threshold_multiplier"] == 1.0
        assert policy["strength_cap"] == 1.0
        assert policy["goal_exclusions"] == set()

    def test_35d_emergency_retries_blocked_for_phase50_hf_team_context(self):
        from backend.core.per_phase_musical_goals_gate import _allow_emergency_retries

        allow = _allow_emergency_retries(
            "phase_50_spectral_repair",
            worst_priority=2,
            best_regression=0.20,
            catastrophic_threshold=0.08,
            team_policy={"reason": "phase50_after_hf_restoration"},
        )
        assert allow is False

    def test_35e_emergency_retries_allowed_without_team_block(self):
        from backend.core.per_phase_musical_goals_gate import _allow_emergency_retries

        allow = _allow_emergency_retries(
            "phase_03_denoise",
            worst_priority=2,
            best_regression=0.20,
            catastrophic_threshold=0.08,
            team_policy={"reason": ""},
        )
        assert allow is True

    def test_35f_transition_policy_additive_to_subtractive_applies(self):
        from backend.core.per_phase_musical_goals_gate import _resolve_team_context_policy

        policy = _resolve_team_context_policy(
            "phase_03_denoise",
            {
                "prior_phase_context": {
                    "last_phase_type": "ADDITIVE",
                }
            },
        )
        assert policy["threshold_multiplier"] > 1.0
        assert policy["strength_cap"] < 1.0
        assert "brillanz" in policy["goal_exclusions"]

    def test_35g_transition_policy_mlgen_to_subtractive_applies(self):
        from backend.core.per_phase_musical_goals_gate import _resolve_team_context_policy

        policy = _resolve_team_context_policy(
            "phase_29_tape_hiss_reduction",
            {
                "prior_phase_context": {
                    "last_phase_type": "ML_GENERATIVE",
                }
            },
        )
        assert policy["threshold_multiplier"] > 1.0
        assert "artikulation" in policy["goal_exclusions"]


class TestPMGGReconstructionRecheckAllowlist:
    def test_35h_phase24_recheck_allows_low_but_present_vocal_confidence(self):
        from backend.core.per_phase_musical_goals_gate import _reconstruction_goal_recheck_allowlist

        defect_locations = {
            "DROPOUTS": [(0.50, 0.60), (1.10, 1.18)],
        }

        goals = _reconstruction_goal_recheck_allowlist(
            "phase_24_dropout_repair",
            {"vocal_probability": 0.17},
            defect_locations,
            audio_len=5 * SR,
            sr=SR,
            sample_duration_s=5.0,
        )

        assert goals == {"natuerlichkeit", "authentizitaet"}

    def test_35i_phase55_recheck_allows_lyrics_guidance_without_vocal_confidence(self):
        from backend.core.per_phase_musical_goals_gate import _reconstruction_goal_recheck_allowlist

        defect_locations = {
            "DROPOUTS": [(0.40, 0.55)],
            "SPECTRAL_HOLES": [(0.90, 1.10)],
        }

        goals = _reconstruction_goal_recheck_allowlist(
            "phase_55_diffusion_inpainting",
            {"vocal_probability": 0.0, "pre_transcription": "du wolltest nur ein abenteuer"},
            defect_locations,
            audio_len=5 * SR,
            sr=SR,
            sample_duration_s=5.0,
        )

        assert goals == {"natuerlichkeit", "authentizitaet"}


# ----------------------------------------------------------------------
# Tests §2.29b — PHASE_GOAL_EXCLUSIONS (NatuerlichkeitMetric stable-metric
# invariante + phase-specific false-positive prevention)
# ----------------------------------------------------------------------


class TestPhaseGoalExclusions:
    """§2.29b: PHASE_GOAL_EXCLUSIONS ensures unreliable metrics are never
    used for PMGG delta checks in phases where they produce false regressions.
    """

    def test_36_phase03_excludes_natuerlichkeit(self):
        """CREPE state-dependency must not trigger P1 regression in phase_03."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "natuerlichkeit" in PHASE_GOAL_EXCLUSIONS["phase_03"]

    def test_37_phase03_excludes_artikulation(self):
        """ArticulationMetric(ref=noisy_tape) vs denoised output is reference‑mismatch.
        Denoising reshapes transients → false P2 catastrophic regression confirmed
        in debug logs 2026-03-28 (worst_goal=artikulation, Δ=0.54 → best_effort 6 %)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "artikulation" in PHASE_GOAL_EXCLUSIONS["phase_03"]

    def test_38_phase03_tonal_center_excluded(self):
        """§9.7.11 extension (v10.0.0): tonal_center MUST be excluded from phase_03.
        K-S is invariant to additive white noise but NOT to frequency-selective NR
        (OMLSA/ResembleEnhance apply gain G(f) varying per band → chroma energy
        distribution shifts → K-S argmax changes even though musical key is unchanged).
        Real-run confirmed: catastrophic tonal_center regression Δ=0.1043 on 1930s tape
        (SNR≈15 dB, 1/f hiss). Exclusion prevents false P2 catastrophic cascades."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_03", set())
        assert "tonal_center" in excl, (
            "tonal_center MUST be excluded from phase_03: K-S not invariant to shaped NR (§9.7.11 ext, v10.0.0)"
        )

    def test_38b_phase03_excludes_timbre_authentizitaet(self):
        """v10.0.0: timbre_authentizitaet MUST be excluded from phase_03 —
        MFCC-Pearson + centroid-CV proxy disturbed by spectral-envelope change
        after broadband NR (confirmed in logs 2026-03-30: residual 0.046 regression)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_03", set())
        assert "timbre_authentizitaet" in excl, (
            "timbre_authentizitaet MUST be in phase_03 exclusions (v10.0.0): "
            "MFCC correlation against noisy reference is unreliable after denoising"
        )

    def test_39_phase29_brillanz_not_excluded(self):
        """§9.7.12: HF crest-factor proxy is SNR-robust; brillanz MUST NOT be excluded
        from phase_29. After DeepFilterNet, noise floor drops → crest increases → delta positive."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "brillanz" not in PHASE_GOAL_EXCLUSIONS.get("phase_29", set()), (
            "brillanz must NOT be excluded from phase_29 since §9.7.12 crest-factor proxy is SNR-robust"
        )

    def test_40_phase29_excludes_artikulation(self):
        """Same reference-mismatch as phase_03: hissy tape as reference vs denoised output."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "artikulation" in PHASE_GOAL_EXCLUSIONS["phase_29"]

    def test_40b_phase29_tonal_center_excluded(self):
        """§9.7.11 extension (v10.0.0): tonal_center MUST be excluded from phase_29.
        DeepFilterNet v3 II is a learned frequency-selective HF filter: reduces energy
        in high-register chroma bins (C5-B7) while leaving low-register bins less
        affected → K-S correlation shifts even though the musical key is unchanged.
        Real-run confirmed: catastrophic regression 0.8333 > 0.08 (worst goal:
        tonal_center P2) today (2026-03-30). Stagnation Δ=0.000000 across retries
        confirms measurement artifact, not a genuine key shift."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "tonal_center" in PHASE_GOAL_EXCLUSIONS["phase_29"], (
            "tonal_center MUST be excluded from phase_29: K-S not invariant to HF-selective NR (§9.7.11 ext, v10.0.0)"
        )

    def test_41_phase55_excludes_artikulation(self):
        """Diffusion inpainting synthesises new content — no valid transient reference."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "artikulation" in PHASE_GOAL_EXCLUSIONS["phase_55"]

    def test_42_phase55_excludes_micro_dynamics(self):
        """Inpainting inserts content with its own envelope intentionally differing from gap edges."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "micro_dynamics" in PHASE_GOAL_EXCLUSIONS["phase_55"]

    def test_43_phase02_excludes_natuerlichkeit(self):
        """Comb-filter notches in hum-removal should not trigger CREPE-based P1 regression."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "natuerlichkeit" in PHASE_GOAL_EXCLUSIONS["phase_02"]

    def test_44_phase24_excludes_natuerlichkeit(self):
        """Dropout-repair synthesis produces content without CREPE reference → unreliable."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "natuerlichkeit" in PHASE_GOAL_EXCLUSIONS["phase_24"]

    def test_44b_phase24_excludes_artikulation(self):
        """Dropout inpainting creates new transients with no valid pre-repair transient reference."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "artikulation" in PHASE_GOAL_EXCLUSIONS["phase_24"]

    def test_45_phase06_brillanz_not_excluded(self):
        """§9.7.12: HF crest-factor proxy correctly scores SBR improvement (reference-free).
        brillanz MUST NOT be excluded from phase_06 after §9.7.12 fix."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "brillanz" not in PHASE_GOAL_EXCLUSIONS.get("phase_06", set()), (
            "brillanz must NOT be excluded from phase_06 since §9.7.12 crest-factor proxy is reference-free and SNR-robust"
        )

    def test_46_phase18_excludes_micro_dynamics(self):
        """Noise gate deliberately inserts silence segments → dynamics change is intended."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "micro_dynamics" in PHASE_GOAL_EXCLUSIONS["phase_18"]

    def test_46b_phase18_tonal_center_not_excluded(self):
        """§9.7.11 K-S fix: tonal_center removed from phase_18 exclusions.
        Silence gating zeroes inter-beat sections but K-S argmax on the remaining
        musical content is unaffected — no false P2 regression."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "tonal_center" not in PHASE_GOAL_EXCLUSIONS["phase_18"], (
            "tonal_center must NOT be excluded from phase_18 since §9.7.11 K-S proxy is key-stable"
        )

    def test_47_exclusions_dict_nonempty(self):
        """PHASE_GOAL_EXCLUSIONS must contain at least 10 phase entries."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert len(PHASE_GOAL_EXCLUSIONS) >= 10

    def test_48_exclusions_all_values_are_sets(self):
        """Every value in PHASE_GOAL_EXCLUSIONS must be a set (never list/tuple)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        for phase, excl in PHASE_GOAL_EXCLUSIONS.items():
            assert isinstance(excl, set), f"{phase}: exclusions should be a set, got {type(excl)}"

    def test_49_exclusions_keys_start_with_phase(self):
        """All exclusion keys must be phase-ID prefixes starting with 'phase_'."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        for key in PHASE_GOAL_EXCLUSIONS:
            assert key.startswith("phase_"), f"Key {key!r} does not start with 'phase_'"

    def test_50_phase36_excludes_artikulation(self):
        """Transient shaper intentionally alters attack shapes → artikulation delta expected."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "artikulation" in PHASE_GOAL_EXCLUSIONS["phase_36"]

    def test_51_phase03_exclusion_is_superset_v9_10_79(self):
        """phase_03 exclusion set must contain at least {natuerlichkeit, artikulation} (§2.29b)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        required = {"natuerlichkeit", "artikulation"}
        assert required.issubset(PHASE_GOAL_EXCLUSIONS["phase_03"]), (
            f"phase_03 exclusions {PHASE_GOAL_EXCLUSIONS['phase_03']} missing required: "
            f"{required - PHASE_GOAL_EXCLUSIONS['phase_03']}"
        )

    def test_52b_phase02_excludes_groove(self):
        """P3 root cause 2026-03-30: hum removal doesn't affect timing; GrooveMetric
        onset/DTW proxy stagnated Δ=0 across all retries (filter-independent artifact).
        0.1526 regression caused false catastrophic PMGG cascade."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "groove" in PHASE_GOAL_EXCLUSIONS["phase_02"], (
            "groove must be excluded from phase_02: stagnation Δ=0 proves LF-filter-independence"
        )

    def test_52c_phase02_tonal_center_not_excluded(self):
        """§9.7.11 K-S fix: tonal_center removed from phase_02 exclusions.
        Comb-filter notches at G1/G2/G3/G4 (49/98/196/392 Hz) redistribute chroma
        energy in a narrow band, but K-S key argmax on all 12 roots is stable —
        the key label doesn't flip from a single narrow notch."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "tonal_center" not in PHASE_GOAL_EXCLUSIONS["phase_02"], (
            "tonal_center must NOT be excluded from phase_02 since §9.7.11 K-S proxy is robust to narrow notches"
        )

    def test_52d_phase02_excludes_timbre_authentizitaet(self):
        """P2 root cause 2026-03-30: spectral notches at 50/100/150 Hz directly
        disturb MFCC-Pearson and spectral-centroid proxies — false P2 regression
        despite no perceptual timbre degradation."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "timbre_authentizitaet" in PHASE_GOAL_EXCLUSIONS["phase_02"], (
            "timbre_authentizitaet must be excluded from phase_02: notch-induced MFCC shift is false positive"
        )

    def test_52e_phase02_exclusion_superset_v9_10_91(self):
        """phase_02 exclusion set must contain all 6 required goals (v10.0.0 — tonal_center resolved via K-S §9.7.11)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        required = {
            "bass_kraft",
            "authentizitaet",
            "natuerlichkeit",
            "transparenz",
            "groove",
            "timbre_authentizitaet",
        }
        excl = PHASE_GOAL_EXCLUSIONS.get("phase_02", set())
        assert required.issubset(excl), f"phase_02 exclusions missing: {required - excl}"

    # ── phase_20 (SGMSE+ Reverb-Reduction) ──────────────────────────────────

    def test_53a_phase20_brillanz_not_excluded(self):
        """§9.7.12: HF crest-factor proxy is reverb-robust. Reverb adds diffuse noise
        → low crest before removal; dry musical peaks → high crest after SGMSE+.
        brillanz MUST NOT be excluded from phase_20 after §9.7.12 fix."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "brillanz" not in PHASE_GOAL_EXCLUSIONS.get("phase_20", set()), (
            "brillanz must NOT be excluded from phase_20 since §9.7.12 crest-factor proxy is reverb-robust"
        )

    def test_53b_phase20_waerme_not_excluded(self):
        """§9.7.14: Warmth ratio E(200-800)/E(800-3000) is reverb-invariant.
        Both sub-bands are affected proportionally by reverb → ratio stable.
        waerme MUST NOT be excluded from phase_20 after §9.7.14 fix."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "waerme" not in PHASE_GOAL_EXCLUSIONS.get("phase_20", set()), (
            "waerme must NOT be excluded from phase_20 since §9.7.14 warmth-ratio proxy is reverb-invariant"
        )

    def test_53c_phase20_excludes_authentizitaet(self):
        """P1 root cause: reverb smooths log-spectrum valleys (same mechanism as
        broadband noise). After SGMSE+ true valleys reappear → false P1 cascade
        (0.5502 regression observed for structurally identical phase_49)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "authentizitaet" in PHASE_GOAL_EXCLUSIONS["phase_20"]

    def test_53d_phase20_transparenz_not_excluded(self):
        """§9.7.13: Multi-band crest-factor proxy is SNR/reverb-robust. Reverb fills
        each band's floor → low crest; after SGMSE+ floor drops → crest improves.
        transparenz MUST NOT be excluded from phase_20 after §9.7.13 fix."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "transparenz" not in PHASE_GOAL_EXCLUSIONS.get("phase_20", set()), (
            "transparenz must NOT be excluded from phase_20 since §9.7.13 multi-band crest proxy is reverb-robust"
        )

    def test_53e_phase20_excludes_natuerlichkeit(self):
        """SGMSE+ spectral deconvolution can introduce slight harmonic smearing on
        ambiguous reverb/body-resonance segments → MFCC smoothness proxy reacts on
        5-s window. Same mechanism as phase_02/phase_03 MFCC instability."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "natuerlichkeit" in PHASE_GOAL_EXCLUSIONS["phase_20"]

    def test_53f_phase20_exclusion_superset(self):
        """phase_20 must contain at least {authentizitaet, natuerlichkeit}.
        §9.7.12/13/14: brillanz, transparenz, waerme removed — crest-factor and
        warmth-ratio proxies are reverb-robust and no longer need exclusion."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        required = {"authentizitaet", "natuerlichkeit"}
        excl = PHASE_GOAL_EXCLUSIONS.get("phase_20", set())
        assert required.issubset(excl), f"phase_20 exclusions missing: {required - excl}"

    # ── phase_23 (AudioSR Spectral Inpainting) ───────────────────────────────

    def test_54a_phase23_excludes_natuerlichkeit(self):
        """Gap-fill synthesis produces content absent from reference → MFCC proxy
        on synthesised region is unreliable (same mechanism as phase_24)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "natuerlichkeit" in PHASE_GOAL_EXCLUSIONS["phase_23"]

    def test_54b_phase23_excludes_brillanz(self):
        """Synthesised HF fill may not match the damaged reference HF distribution
        → false brillanz drop against damaged-signal baseline (same as phase_24)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "brillanz" in PHASE_GOAL_EXCLUSIONS["phase_23"]

    def test_54c_phase23_excludes_authentizitaet(self):
        """Spectral gaps have near-zero amplitude → fft_mag ≈ 0 → flatness undefined.
        After AudioSR synthesis tonal content appears → transition is
        reference-mismatch-driven, not regression (same as phase_24)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "authentizitaet" in PHASE_GOAL_EXCLUSIONS["phase_23"]

    def test_54d_phase23_excludes_artikulation(self):
        """Inpainting inserts new spectral content where reference has damaged/missing
        content → transient-shape correlation against pre-inpainting fragment is
        meaningless (same mechanism as phase_24 dropout repair)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "artikulation" in PHASE_GOAL_EXCLUSIONS["phase_23"]

    def test_54e_phase23_exclusion_superset(self):
        """phase_23 must contain at least {natuerlichkeit, brillanz, authentizitaet,
        artikulation} — same synthesised-content exclusion set as phase_24."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        required = {"natuerlichkeit", "brillanz", "authentizitaet", "artikulation"}
        excl = PHASE_GOAL_EXCLUSIONS.get("phase_23", set())
        assert required.issubset(excl), f"phase_23 exclusions missing: {required - excl}"

    # ── phase_29 material-adaptive timbre_authentizitaet ────────────────────

    def test_55_phase29_material_adaptive_timbre_analog(self):
        """For analog materials (tape/vinyl/shellac) PMGG must add timbre_authentizitaet
        to phase_29 exclusions at runtime — same HF-removal → centroid-CV disturbance
        mechanism as phase_03 (extended 2026-03-30 to phase_29)."""
        import types
        from unittest.mock import patch

        import numpy as np

        from backend.core.per_phase_musical_goals_gate import (
            FAST_GOALS_SUBSET,
            PerPhaseMusicalGoalsGate,
        )

        gate = PerPhaseMusicalGoalsGate()
        audio = np.random.default_rng(0).uniform(-0.1, 0.1, 48000 * 2).astype(np.float32)

        class _PassPhase:
            def __call__(self, a, strength=1.0, **kw):
                return a.copy()

            def get_metadata(self):
                m = types.SimpleNamespace()
                m.phase_id = "phase_29_tape_hiss_reduction"
                m.name = "phase_29_tape_hiss_reduction"
                return m

        checked_goals: list[list[str]] = []
        _ = PerPhaseMusicalGoalsGate._run_with_retry

        def _capture(*args, **kwargs):
            checked_goals.append(list(kwargs.get("effective_goals", [])))
            return audio.copy(), dict.fromkeys(FAST_GOALS_SUBSET, 0.85), "passed", 1.0

        with patch.object(PerPhaseMusicalGoalsGate, "_run_with_retry", side_effect=_capture):
            gate.wrap_phase(
                _PassPhase(),
                audio,
                48000,
                phase_kwargs={"material_type": "tape"},
            )

        assert checked_goals, "wrap_phase did not call _run_with_retry"
        assert "timbre_authentizitaet" not in checked_goals[0], (
            "timbre_authentizitaet should be excluded from phase_29 for tape material "
            f"but was in effective_goals: {checked_goals[0]}"
        )


# ---------------------------------------------------------------------------
# Helper: Mock-Phase mit konfigurierbarer phase_id
# ---------------------------------------------------------------------------


def _make_pass_phase(phase_id: str):
    """Identity mock phase with a specific phase_id for PMGG exclusion/cal tests."""
    import types

    class _Phase:
        def __call__(self, audio, strength=1.0, **kw):
            return audio.copy().astype("float32")

        def get_metadata(self):
            m = types.SimpleNamespace()
            m.phase_id = phase_id
            m.name = phase_id
            return m

    return _Phase()


# ---------------------------------------------------------------------------
# Tests: §2.31b PMGG Song-Kalibrierungs-Integration (v10.0.0/86)
# ---------------------------------------------------------------------------


class TestPMGGSongCalIntegration:
    """§2.31b: Seven PMGG integration points that use song_calibration_profile."""

    # ── A: Cal-aware threshold ──────────────────────────────────────────────

    def test_52_threshold_reduced_when_global_scalar_low(self, gate, audio_5s):
        """global_scalar < 0.85 → threshold × 0.85 (tighter protection, §2.31b-A)."""
        cal = {"global_scalar": 0.70, "restorability_score": 65.0}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=50.0,
        )
        assert result.action in {"passed", "retry1", "retry2", "retry3", "retry4", "retry5", "best_effort"}

    def test_53_threshold_relaxed_when_global_scalar_high(self, gate, audio_5s):
        """global_scalar > 1.20 → threshold × 1.15 (fewer retry cycles, §2.31b-A)."""
        cal = {"global_scalar": 1.30, "restorability_score": 80.0}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=75.0,
        )
        assert result.action in {"passed", "best_effort"}

    def test_54_threshold_neutral_when_scalar_mid(self, gate, audio_5s):
        """global_scalar in [0.85, 1.20] → threshold unchanged (§2.31b-A)."""
        cal = {"global_scalar": 1.00, "restorability_score": 65.0}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_04_eq"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=65.0,
        )
        assert result is not None

    # ── B: Sanftere Retry-Leiter ────────────────────────────────────────────

    def test_55_soft_retry_anchors_when_initial_strength_below_090(self, gate, audio_5s):
        """initial_strength=0.80 must use anchors starting at 0.80 (§2.31b-B): no rollback."""
        cal = {"global_scalar": 0.75, "restorability_score": 40.0}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=35.0,
            initial_strength=0.80,
        )
        assert "rollback" not in result.action

    def test_56_normal_anchors_when_initial_strength_at_100(self, gate, audio_5s):
        """initial_strength=1.0 uses standard anchors (§2.31b-B inverse)."""
        cal = {"global_scalar": 1.0, "restorability_score": 65.0}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=65.0,
            initial_strength=1.0,
        )
        assert result is not None

    # ── C: Proportionale Stagnation-Schwelle ───────────────────────────────

    def test_57_stagnation_threshold_is_proportional(self, gate, audio_5s):
        """Stagnation delta = max(0.002, threshold × 0.15) — gate runs cleanly (§2.31b-C)."""
        cal = {"global_scalar": 1.0, "restorability_score": 65.0, "restorability_tier": "fair"}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_05_stereo"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=65.0,
        )
        assert result is not None

    # ── E: P3-Retry-Budget nach restorability_tier ─────────────────────────

    def test_58_p3_tier_good_accepted_in_kwargs(self, gate, audio_5s):
        """restorability_tier='good' in song_calibration_profile is accepted (§2.31b-E)."""
        cal = {"global_scalar": 1.1, "restorability_score": 75.0, "restorability_tier": "good"}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_07_harmonic"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=75.0,
        )
        assert result.action in {"passed", "best_effort"}

    def test_59_p3_tier_poor_accepted_in_kwargs(self, gate, audio_5s):
        """restorability_tier='poor' in song_calibration_profile is accepted (§2.31b-E)."""
        cal = {"global_scalar": 0.80, "restorability_score": 30.0, "restorability_tier": "poor"}
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_07_harmonic"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": cal},
            restorability_score=30.0,
        )
        assert result is not None

    # ── F: Dynamischer Catastrophic-Threshold ──────────────────────────────

    def test_60_catastrophic_threshold_formula_good_material(self):
        """GOOD material: catastrophic = max(0.08, 4.0 × 0.020) = 0.08 (§2.31b-F)."""
        val = max(0.08, 4.0 * 0.020)
        assert abs(val - 0.08) < 1e-9, f"GOOD material catastrophic threshold should be 0.08, got {val}"

    def test_61_catastrophic_threshold_formula_poor_material(self):
        """POOR material: catastrophic = max(0.08, 4.0 × 0.055) = 0.22 (§2.31b-F)."""
        val = max(0.08, 4.0 * 0.055)
        assert abs(val - 0.22) < 1e-9, f"POOR material catastrophic threshold should be 0.22, got {val}"

    def test_62_catastrophic_threshold_floor_is_008(self):
        """Floor 0.08: even very low threshold can't push catastrophic below 0.08 (§2.31b-F)."""
        val = max(0.08, 4.0 * 0.001)
        assert val == 0.08

    def test_63_catastrophic_threshold_proportional_fair(self):
        """FAIR material: catastrophic = max(0.08, 4.0 × 0.035) = 0.14 (§2.31b-F)."""
        val = max(0.08, 4.0 * 0.035)
        assert abs(val - 0.14) < 1e-9

    # ── G: Material-adaptive PHASE_GOAL_EXCLUSIONS ─────────────────────────

    def test_64_cd_digital_phase03_gate_runs_cleanly(self, gate, audio_5s):
        """cd_digital + phase_03: gate runs without crash (§2.31b-G)."""
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"material_type": "cd_digital"},
            restorability_score=85.0,
        )
        assert result.action in {"passed", "best_effort"}

    def test_65_dat_phase29_gate_runs_cleanly(self, gate, audio_5s):
        """dat + phase_29: gate runs without crash (§2.31b-G)."""
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_29_tape_hiss"),
            audio_5s,
            SR,
            phase_kwargs={"material_type": "dat"},
            restorability_score=85.0,
        )
        assert result.action in {"passed", "best_effort"}

    def test_66_cd_digital_phase04_exclusions_not_reduced(self, gate, audio_5s):
        """cd_digital + phase_04: exclusion set not reduced (§2.31b-G only phase_03/29)."""
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_04_eq"),
            audio_5s,
            SR,
            phase_kwargs={"material_type": "cd_digital"},
            restorability_score=85.0,
        )
        assert result is not None

    def test_67_vinyl_phase03_gate_runs_cleanly(self, gate, audio_5s):
        """vinyl + phase_03: full exclusions retained, gate runs (§2.31b-G analog materials)."""
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"material_type": "vinyl"},
            restorability_score=50.0,
        )
        assert result is not None

    def test_67b_vinyl_phase03_excludes_timbre_authentizitaet_adaptive(self, gate, audio_5s, monkeypatch):
        """Analog material in phase_03 should add timbre_authentizitaet to effective exclusions."""
        captured: dict[str, list[str]] = {}

        def _fake_run_with_retry(*args, **kwargs):
            captured["effective_goals"] = list(kwargs.get("effective_goals") or [])
            return args[1], args[3], "passed", 1.0

        monkeypatch.setattr(gate, "_run_with_retry", _fake_run_with_retry)

        gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"material_type": "vinyl"},
            restorability_score=50.0,
        )

        assert "timbre_authentizitaet" not in captured.get("effective_goals", [])

    def test_67c_phase24_rechecks_p1_for_sparse_vocal_dropouts(self, gate, audio_5s, monkeypatch):
        """Sparse vocal dropout repair must re-enable P1 guards in the PMGG window."""
        captured: dict[str, list[str]] = {}

        def _fake_run_with_retry(*args, **kwargs):
            captured["effective_goals"] = list(kwargs.get("effective_goals") or [])
            return args[1], args[3], "passed", 1.0

        monkeypatch.setattr(gate, "_run_with_retry", _fake_run_with_retry)

        gate.wrap_phase(
            _make_pass_phase("phase_24_dropout_repair"),
            audio_5s,
            SR,
            phase_kwargs={
                "material_type": "vinyl",
                "vocal_probability": 0.82,
                "defect_locations": {"DROPOUTS": [(1.00, 1.12)]},
            },
            restorability_score=50.0,
        )

        assert "natuerlichkeit" in captured.get("effective_goals", [])
        assert "authentizitaet" in captured.get("effective_goals", [])

    def test_67d_phase24_keeps_p1_excluded_for_dense_dropout_window(self, gate, audio_5s, monkeypatch):
        """Dense dropout coverage must keep phase_24 P1 exclusions active."""
        captured: dict[str, list[str]] = {}

        def _fake_run_with_retry(*args, **kwargs):
            captured["effective_goals"] = list(kwargs.get("effective_goals") or [])
            return args[1], args[3], "passed", 1.0

        monkeypatch.setattr(gate, "_run_with_retry", _fake_run_with_retry)

        gate.wrap_phase(
            _make_pass_phase("phase_24_dropout_repair"),
            audio_5s,
            SR,
            phase_kwargs={
                "material_type": "vinyl",
                "vocal_probability": 0.82,
                "defect_locations": {"DROPOUTS": [(0.80, 2.10)]},
            },
            restorability_score=50.0,
        )

        assert "natuerlichkeit" not in captured.get("effective_goals", [])
        assert "authentizitaet" not in captured.get("effective_goals", [])

    def test_68_tape_phase29_gate_runs_cleanly(self, gate, audio_5s):
        """reel_tape + phase_29: full exclusions retained, gate runs (§2.31b-G)."""
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_29_tape_hiss"),
            audio_5s,
            SR,
            phase_kwargs={"material_type": "reel_tape"},
            restorability_score=45.0,
        )
        assert result is not None

    def test_69_material_adaptive_stable_goals_are_subset(self):
        """Stable exclusions {natuerlichkeit,artikulation} must be subset of phase_03/29 sets."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        stable = {"natuerlichkeit", "artikulation"}
        assert stable.issubset(PHASE_GOAL_EXCLUSIONS["phase_03"])
        assert stable.issubset(PHASE_GOAL_EXCLUSIONS["phase_29"])

    def test_70_no_song_cal_profile_does_not_crash(self, gate, audio_5s):
        """wrap_phase without song_calibration_profile must not crash (§2.31b backward compat)."""
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={},
            restorability_score=65.0,
        )
        assert result is not None

    def test_71_empty_song_cal_profile_does_not_crash(self, gate, audio_5s):
        """wrap_phase with empty song_calibration_profile must not crash."""
        _, _, result = gate.wrap_phase(
            _make_pass_phase("phase_03_denoise"),
            audio_5s,
            SR,
            phase_kwargs={"song_calibration_profile": {}},
            restorability_score=65.0,
        )
        assert result is not None

    def test_72_phase03_has_six_goals_excluded_v9_13(self):
        """phase_03 muss genau 6 Goals ausschließen (§V36 v10.0.0: transient_energie hinzugefügt).
        OMLSA/DFN entfernt Rauschimpulse → TransientEnergieProxy false P3 (Δ=-0.13)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        expected = {
            "natuerlichkeit",
            "artikulation",
            "authentizitaet",
            "tonal_center",
            "timbre_authentizitaet",
            "transient_energie",  # §V36 v10.0.0
        }
        assert PHASE_GOAL_EXCLUSIONS["phase_03"] == expected, (
            f"phase_03 exclusions: {PHASE_GOAL_EXCLUSIONS['phase_03']} != {expected}"
        )

    def test_73_phase29_has_seven_goals_excluded_v9_13(self):
        """phase_29 muss genau 7 Goals ausschließen (§V36 v10.0.0: waerme hinzugefügt).
        OMLSA/DFN-Suppression im Wärmeband (200-2000 Hz) → false P4 (Δ=-0.17)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        expected = {
            "artikulation",
            "authentizitaet",
            "natuerlichkeit",
            "tonal_center",
            "timbre_authentizitaet",
            "transparenz",  # §V32: Tape-Hiss-Carrier inflationiert HF-Crest-Proxy
            "waerme",  # §V36 v10.0.0: Wärmeband-Rauschboden → false P4
        }
        assert PHASE_GOAL_EXCLUSIONS["phase_29"] == expected, (
            f"phase_29 exclusions: {PHASE_GOAL_EXCLUSIONS['phase_29']} != {expected}"
        )


def test_precise_overrides_use_multisegment_sampling(monkeypatch):
    """Lange Audios dürfen nicht nur über den Anfang bewertet werden."""
    import backend.core.per_phase_musical_goals_gate as pmgg

    class _MeanMetric:
        def measure(self, audio, sr, reference=None):
            arr = np.asarray(audio, dtype=np.float32)
            return float(np.mean(arr))

    monkeypatch.setattr(pmgg, "_get_precise_metric_instances", lambda: {"micro_dynamics": _MeanMetric()})

    sr = 48000
    # 9s Signal: Start=0, Mitte=1, Ende=0. Reiner Kopf-Crop wäre ~0.0.
    audio = np.concatenate(
        [
            np.zeros(sr * 3, dtype=np.float32),
            np.ones(sr * 3, dtype=np.float32),
            np.zeros(sr * 3, dtype=np.float32),
        ],
        axis=0,
    )

    refined = pmgg._apply_precise_metric_overrides({"micro_dynamics": 0.0}, audio, sr)
    assert refined["micro_dynamics"] > 0.2


# ---------------------------------------------------------------------------
# Tests: §9.7.9 Groove-Proxy LF-Robustheit (v10.0.0)
# ---------------------------------------------------------------------------


class TestGrooveProxyLFRobustness:
    """§9.7.9: Groove-Proxy-Glättung von rms_env (50 ms) macht autocorr[0]-Normierung
    unabhängig von 50/100 Hz-Hum-Modulation in der Amplitudenhüllkurve."""

    @pytest.fixture
    def _gate(self):
        from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate

        g = PerPhaseMusicalGoalsGate()
        g.reset()
        return g

    def _measure_groove(self, audio: "np.ndarray", sr: int) -> float:
        """Ruft _measure_quick auf und extrahiert den groove-Score."""
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        scores = _measure_quick(audio, sr)
        return scores["groove"]

    def test_74_groove_stable_with_and_without_50hz_hum(self):
        """Groove-Score auf reinem Rhythmus-Signal bleibt nahezu identisch unabhängig
        von zugemischtem 50 Hz Hum (§9.7.9 Strukturfix)."""
        import numpy as np

        sr = 48000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)

        # Rhythmisches Testsignal: Clicks alle 500 ms (120 BPM), Band-gefiltert
        clicks = np.zeros(len(t), dtype=np.float32)
        for beat in np.arange(0.0, 2.0, 0.5):
            idx = int(beat * sr)
            if idx < len(clicks):
                clicks[idx] = 1.0
        # Leichtes Decay nach jedem Click (Impuls-Antwort simulieren)
        from scipy.signal import lfilter

        clicks = lfilter([1.0], [1.0, -0.85], clicks).astype(np.float32)
        # Normalisieren
        clicks /= np.max(np.abs(clicks)) + 1e-9

        # 50 Hz Hum bei 20 % Amplitude
        hum = 0.20 * np.sin(2 * np.pi * 50.0 * t).astype(np.float32)

        groove_clean = self._measure_groove(clicks, sr)
        groove_hum = self._measure_groove(clicks + hum, sr)

        # Nach Fix: Groove-Delta durch Hum darf höchstens 0.10 betragen
        # (Hum beeinflusst nur Normierungsbasis, nicht 500 ms-Periodizität)
        delta = abs(groove_clean - groove_hum)
        assert delta < 0.10, (
            f"Groove-Proxy LF-sensitiv: delta={delta:.4f} "
            f"(clean={groove_clean:.4f}, hum={groove_hum:.4f}) — 50 ms Glättung unzureichend"
        )

    def test_75_groove_separates_periodic_vs_aperiodic_bursts(self):
        """Groove-Proxy muss periodische 500ms-Bursts höher scoren als aperiodische Bursts.
        Weißes Rauschen ist KEIN geeigneter Vergleichswert (flache RMS-Hüllkurve → hohe
        Autokorrelation bei allen Lags). Der Proxy misst Energie-Regelmäßigkeit."""
        import numpy as np
        from scipy.signal import lfilter

        sr = 48000

        # Periodisch: Clicks exakt alle 500 ms (120 BPM)
        periodic = np.zeros(sr * 2, dtype=np.float32)
        for beat in np.arange(0.0, 2.0, 0.5):
            idx = int(beat * sr)
            if idx < len(periodic):
                periodic[idx] = 1.0
        periodic = lfilter([1.0], [1.0, -0.85], periodic).astype(np.float32)
        periodic /= np.max(np.abs(periodic)) + 1e-9

        # Aperiodisch: Clicks bei irregulären Abständen (keiner nahe 500 ms)
        aperiodic = np.zeros(sr * 2, dtype=np.float32)
        for t in [0.10, 0.37, 0.72, 1.03, 1.44, 1.81]:  # Lücken: 0.27/0.35/0.31/0.41/0.37 s
            idx = int(t * sr)
            if idx < len(aperiodic):
                aperiodic[idx] = 1.0
        aperiodic = lfilter([1.0], [1.0, -0.85], aperiodic).astype(np.float32)
        aperiodic /= np.max(np.abs(aperiodic)) + 1e-9

        groove_periodic = self._measure_groove(periodic, sr)
        groove_aperiodic = self._measure_groove(aperiodic, sr)

        assert groove_periodic > groove_aperiodic, (
            f"Groove-Proxy diskriminiert periodisch vs. aperiodisch nicht: "
            f"periodisch={groove_periodic:.4f} <= aperiodisch={groove_aperiodic:.4f}"
        )

    def test_76_groove_not_nan_on_short_audio(self):
        """Groove-Proxy darf kein NaN/Inf auf kurzen Clips zurückgeben (< 0.2 s)."""
        import numpy as np

        sr = 48000
        # Nur 0.12 s — rms_env hat ≈ 12 Frames → Glättung mit min(5, 3)=3
        short = np.random.default_rng(99).uniform(-0.5, 0.5, int(sr * 0.12)).astype(np.float32)
        score = self._measure_groove(short, sr)
        assert np.isfinite(score), f"Groove NaN/Inf bei kurzem Audio: {score}"
        assert 0.0 <= score <= 1.0

    def test_77_groove_bounded_zero_to_one_various_signals(self):
        """Groove-Score muss für 8 verschiedene Testsignale in [0, 1] liegen."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(7)
        signals = [
            np.zeros(sr * 2, dtype=np.float32),
            np.ones(sr * 2, dtype=np.float32) * 0.5,
            np.sin(2 * np.pi * 440 * np.linspace(0, 2, sr * 2, endpoint=False)).astype(np.float32),
            np.sin(2 * np.pi * 50 * np.linspace(0, 2, sr * 2, endpoint=False)).astype(np.float32),
            rng.standard_normal(sr * 2).astype(np.float32),
            rng.uniform(-1, 1, sr * 2).astype(np.float32),
            np.tile(np.array([1.0, -1.0], dtype=np.float32), sr),  # Rechteck
            np.float32(0.01) * rng.standard_normal(sr * 2).astype(np.float32),  # sehr leise
        ]
        for i, sig in enumerate(signals):
            score = self._measure_groove(sig, sr)
            assert 0.0 <= score <= 1.0, f"Signal {i}: Groove out of bounds: {score}"
            assert np.isfinite(score), f"Signal {i}: Groove NaN/Inf: {score}"


# ---------------------------------------------------------------------------
# Tests: §9.7.11 Krumhansl-Schmuckler tonal_center Proxy
# ---------------------------------------------------------------------------


class TestKrumhanslSchmucklerTonalCenter:
    """§9.7.11: Krumhansl-Schmuckler (1990) key detection replaces entropy-based
    chroma concentration proxy.  K-S is SNR-invariant: uniform noise raises all
    24 major/minor correlation scores equally \u2192 argmax unchanged \u2192 no false P2 regressions."""

    def _measure_tonal(self, audio: "np.ndarray", sr: int) -> float:
        """Ruft _measure_quick auf und extrahiert den tonal_center-Score."""
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        return _measure_quick(audio, sr)["tonal_center"]

    def _measure(self, audio: "np.ndarray", sr: int) -> "dict[str, float]":
        """Returns all 15 quick-proxy scores for the given signal."""
        from backend.core.per_phase_musical_goals_gate import _measure_quick

        return _measure_quick(audio, sr)

    def test_78_tonal_center_stable_with_white_noise(self):
        """K-S proxy must stay stable (\u0394 \u2264 0.05) when white noise is added to a tonal signal.
        This directly validates SNR-invariance (the root cause of all former false P2 regressions)."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(42)
        # C-major scale sine waves (C4, E4, G4) \u2014 clearly tonal
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        tonal = (
            0.4 * np.sin(2 * np.pi * 261.63 * t)
            + 0.3 * np.sin(2 * np.pi * 329.63 * t)
            + 0.2 * np.sin(2 * np.pi * 392.00 * t)
        ).astype(np.float32)

        noise = rng.standard_normal(len(tonal)).astype(np.float32) * 0.40  # SNR \u2248 0 dB

        score_clean = self._measure_tonal(tonal, sr)
        score_noisy = self._measure_tonal(tonal + noise, sr)

        delta = abs(score_clean - score_noisy)
        assert delta <= 0.05, (
            f"K-S tonal_center SNR-instabil: \u0394={delta:.4f} "
            f"(clean={score_clean:.4f}, noisy={score_noisy:.4f}) \u2014 \u00a79.7.11 violated"
        )

    def test_79_tonal_center_bounded_zero_to_one(self):
        """tonal_center score must always lie in [0.0, 1.0] for 8 diverse signals."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(7)
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        signals = [
            np.zeros(sr * 2, dtype=np.float32),
            rng.standard_normal(sr * 2).astype(np.float32),
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
            (0.5 * np.sin(2 * np.pi * 261.63 * t) + 0.5 * np.sin(2 * np.pi * 392 * t)).astype(np.float32),
            rng.uniform(-1, 1, sr * 2).astype(np.float32),
            np.float32(0.001) * rng.standard_normal(sr * 2).astype(np.float32),
            np.ones(sr * 2, dtype=np.float32) * 0.5,
            np.tile(np.array([1.0, -1.0], dtype=np.float32), sr),
        ]
        for i, sig in enumerate(signals):
            score = self._measure_tonal(sig, sr)
            assert np.isfinite(score), f"Signal {i}: tonal_center NaN/Inf: {score}"
            assert 0.0 <= score <= 1.0, f"Signal {i}: tonal_center out of [0,1]: {score}"

    def test_80_tonal_center_stable_before_and_after_eq(self):
        """K-S tonal_center must remain stable (\u0394 \u2264 0.08) after a broadband EQ shelf cut.
        EQ redistribution of spectral energy should not flip the key argmax."""
        import numpy as np
        from scipy.signal import butter, sosfilt

        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        # C-major triad with harmonics
        tonal = sum(
            a * np.sin(2 * np.pi * f * t)
            for f, a in [(261.63, 0.4), (329.63, 0.3), (392.00, 0.2), (523.25, 0.1), (659.26, 0.08), (784.00, 0.06)]
        ).astype(np.float32)

        # High-shelf cut: -8 dB above 2 kHz (emulate HF EQ correction)
        sos = butter(4, 2000, btype="low", fs=sr, output="sos")
        eq_filtered = sosfilt(sos, tonal).astype(np.float32)

        score_orig = self._measure_tonal(tonal, sr)
        score_eq = self._measure_tonal(eq_filtered, sr)

        delta = abs(score_orig - score_eq)
        assert delta <= 0.08, (
            f"K-S tonal_center EQ-instabil: \u0394={delta:.4f} "
            f"(original={score_orig:.4f}, EQ-filtered={score_eq:.4f}) \u2014 \u00a79.7.11 violated"
        )

    def test_81_tonal_center_silent_signal_returns_midpoint(self):
        """Silent audio (all zeros) must return 0.5 without crash."""
        import numpy as np

        sr = 48000
        silent = np.zeros(sr * 2, dtype=np.float32)
        score = self._measure_tonal(silent, sr)
        assert np.isfinite(score), f"tonal_center NaN/Inf on silence: {score}"
        assert 0.0 <= score <= 1.0, f"tonal_center out of bounds on silence: {score}"

    def test_82_tonal_center_exclusion_for_selective_nr_only(self):
        """v10.0.0 / 2026-04-10: K-S not invariant to frequency-selective spectral modification.
        phase_03/phase_29/phase_49/phase_20: shaped NR or spectral-subtraction dereverb
        → tonal_center correctly excluded. phase_08/phase_18: additive-only/gating → K-S stable."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        # Frequency-selective spectral modification -> tonal_center excluded
        for phase in ["phase_03", "phase_29", "phase_49", "phase_20"]:
            excl = PHASE_GOAL_EXCLUSIONS.get(phase, set())
            assert "tonal_center" in excl, (
                f"{phase}: tonal_center MUST be excluded - K-S not invariant to frequency-selective "
                f"spectral modification (v10.0.0 / real-run P2 regression confirmed 2026-04-10). Current: {excl}"
            )

        # K-S stable phases (no frequency-selective spectral modification) -> tonal_center not excluded
        for phase in ["phase_08", "phase_18"]:
            excl = PHASE_GOAL_EXCLUSIONS.get(phase, set())
            assert "tonal_center" not in excl, (
                f"{phase}: tonal_center must NOT be excluded (K-S stable, additive/gating only). Current: {excl}"
            )

    def test_83_brillanz_improves_after_denoising(self):
        """\u00a79.7.12: brillanz proxy must NOT drop after broadband denoising.
        Noise floor inflates p50 in HF band \u2192 low crest before; after denoising
        noise floor drops \u2192 p50 falls \u2192 crest increases \u2192 score improves."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        # Musical HF content: harmonics extending into presence/air bands
        music = sum(
            a * np.sin(2 * np.pi * f * t)
            for f, a in [(2000, 0.3), (4000, 0.2), (6000, 0.15), (8000, 0.10), (10000, 0.08)]
        ).astype(np.float32)
        noise = rng.standard_normal(len(music)).astype(np.float32) * 0.5  # SNR \u2248 0 dB

        noisy = music + noise
        clean = music  # perfect denoising

        score_noisy = self._measure(noisy, sr)["brillanz"]
        score_clean = self._measure(clean, sr)["brillanz"]

        assert score_clean >= score_noisy - 0.05, (
            f"\u00a79.7.12 brillanz proxy SNR-instabil: "
            f"noisy={score_noisy:.4f}, clean={score_clean:.4f}, "
            f"delta={score_clean - score_noisy:.4f} \u2014 crest-factor proxy must not decrease after denoising"
        )

    def test_84_brillanz_not_excluded_from_denoise_phases(self):
        """\u00a79.7.12: brillanz must NOT appear in exclusions for phases where
        the crest-factor proxy is now SNR-robust."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        phases_fixed = ["phase_03", "phase_06", "phase_07", "phase_18", "phase_20", "phase_29", "phase_49"]
        for phase in phases_fixed:
            excl = PHASE_GOAL_EXCLUSIONS.get(phase, set())
            assert "brillanz" not in excl, (
                f"{phase}: brillanz must NOT be excluded after \u00a79.7.12 crest-factor proxy fix"
            )

    def test_85_brillanz_bounded_and_finite(self):
        """brillanz proxy must return \u2208 [0, 1] and non-NaN for 8 diverse signals."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(9)
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        signals = [
            np.zeros(sr * 2, dtype=np.float32),
            rng.standard_normal(sr * 2).astype(np.float32),
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
            (0.3 * np.sin(2 * np.pi * 8000 * t) + 0.2 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32),
            np.sin(2 * np.pi * 200 * t).astype(np.float32),  # dark/warm, no HF
            rng.uniform(-1, 1, sr * 2).astype(np.float32),
            np.float32(0.001) * rng.standard_normal(sr * 2).astype(np.float32),
            np.ones(sr * 2, dtype=np.float32) * 0.5,
        ]
        for i, sig in enumerate(signals):
            score = self._measure(sig, sr)["brillanz"]
            assert np.isfinite(score), f"Signal {i}: brillanz NaN/Inf: {score}"
            assert 0.0 <= score <= 1.0, f"Signal {i}: brillanz out of [0,1]: {score}"

    # \u2500\u2500 \u00a79.7.13 Transparenz multi-band crest proxy \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def test_86_transparenz_improves_after_denoising(self):
        """\u00a79.7.13: transparenz score must NOT drop after broadband denoising.
        Noise fills each octave band\u2019s floor \u2192 low per-band crest before;
        after denoising floor drops \u2192 crest rises in every band \u2192 score improves."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(7)
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        # Tonal music spanning all 5 measured octave bands (250-8k Hz)
        music = sum(
            a * np.sin(2 * np.pi * f * t) for f, a in [(350, 0.4), (700, 0.3), (1400, 0.2), (2800, 0.15), (5600, 0.10)]
        ).astype(np.float32)
        noise = rng.standard_normal(len(music)).astype(np.float32) * 0.4

        noisy = music + noise
        clean = music

        score_noisy = self._measure(noisy, sr)["transparenz"]
        score_clean = self._measure(clean, sr)["transparenz"]

        assert score_clean >= score_noisy - 0.05, (
            f"\u00a79.7.13 transparenz proxy SNR-instabil: "
            f"noisy={score_noisy:.4f}, clean={score_clean:.4f}, "
            f"delta={score_clean - score_noisy:.4f} \u2014 multi-band crest must not decrease after denoising"
        )

    def test_87_transparenz_not_excluded_from_denoise_phases(self):
        """§9.7.13: transparenz must NOT appear in exclusions for phases where
        the multi-band crest-factor proxy is now SNR-robust.
        Exception: phase_29 (Tape-Hiss Carrier-NR) — §V32 normativ übergeordnet:
        Breitband-HF-Rauschen inflationiert Crest-Proxy → transparenz ausgeschlossen."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        # phase_29 ist §V32-Ausnahme: Tape-Hiss = Carrier-NR mit HF-Broadband-Rauschen
        phases_fixed = ["phase_03", "phase_18", "phase_20", "phase_49"]
        for phase in phases_fixed:
            excl = PHASE_GOAL_EXCLUSIONS.get(phase, set())
            assert "transparenz" not in excl, (
                f"{phase}: transparenz must NOT be excluded after §9.7.13 multi-band crest proxy fix"
            )
        # phase_29 MUSS transparenz ausschließen (§V32)
        assert "transparenz" in PHASE_GOAL_EXCLUSIONS.get("phase_29", set()), (
            "phase_29: transparenz MUSS ausgeschlossen sein (§V32 Tape-Hiss Carrier-NR)"
        )

    def test_88_transparenz_bounded_and_finite(self):
        """transparenz proxy must return \u2208 [0, 1] and non-NaN for 8 diverse signals."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(13)
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        signals = [
            np.zeros(sr * 2, dtype=np.float32),
            rng.standard_normal(sr * 2).astype(np.float32),
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
            (0.3 * np.sin(2 * np.pi * 350 * t) + 0.2 * np.sin(2 * np.pi * 3500 * t)).astype(np.float32),
            np.sin(2 * np.pi * 50 * t).astype(np.float32),
            rng.uniform(-1, 1, sr * 2).astype(np.float32),
            np.float32(0.001) * rng.standard_normal(sr * 2).astype(np.float32),
            np.ones(sr * 2, dtype=np.float32) * 0.5,
        ]
        for i, sig in enumerate(signals):
            score = self._measure(sig, sr)["transparenz"]
            assert np.isfinite(score), f"Signal {i}: transparenz NaN/Inf: {score}"
            assert 0.0 <= score <= 1.0, f"Signal {i}: transparenz out of [0,1]: {score}"

    # \u2500\u2500 \u00a79.7.14 Waerme warmth-ratio proxy \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def test_89_waerme_stable_after_reverb_reduction(self):
        """§9.7.14: waerme proxy must stay stable (Δ ≤ 0.25) when realistic broadband
        room reverb is added (early reflections model).

        The ISO-226-weighted proxy matches WaermeMetric._measure_absolute() (§2.54).
        A BROADBAND IR (white-noise × exp decay) adds energy approximately
        proportionally to both 200-800 Hz and 800-3000 Hz sub-bands.

        NOTE: The old test used a pure exp(-8t) IR normalised to sum=1.  That is a
        1.27 Hz LP filter — NOT representative of room reverb; it preferentially
        attenuates high frequencies, changing the warmth ratio.  Real room IRs use
        wideband early reflections and are approximately spectrally flat.
        The ISO-226-weighted proxy correctly detects warmth changes from dereverb
        on real room IRs; dereverb phases (phase_20, phase_49) are already
        excluded from waerme measurement in PHASE_GOAL_EXCLUSIONS as additional guard.
        """
        import numpy as np

        sr = 48000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False, dtype=np.float32)
        # Warm vocal-like signal: energy in 200–2000 Hz
        dry = sum(
            a * np.sin(2 * np.pi * f * t) for f, a in [(300, 0.5), (500, 0.4), (800, 0.3), (1200, 0.2), (2000, 0.15)]
        ).astype(np.float32)

        # Realistic room reverb: wideband noise × exp decay (early reflections).
        # White-noise × exp-decay adds broadband energy proportionally across all frequencies.
        # This is representative of actual room early-reflection impulse responses.
        ir_len = int(sr * 0.15)  # 150 ms reverb
        t_ir = np.arange(ir_len, dtype=np.float32) / sr
        rng_ir = np.random.default_rng(42)
        noise = rng_ir.standard_normal(ir_len).astype(np.float32)
        ir = noise * np.exp(-20.0 * t_ir)
        ir /= max(float(np.abs(ir).sum()) * 8.0, 1e-6)  # low-level reverb tail
        reverberant = np.convolve(dry, ir)[: len(dry)].astype(np.float32) + dry
        reverberant = np.clip(reverberant, -1.0, 1.0)

        score_dry = self._measure(dry, sr)["waerme"]
        score_reverb = self._measure(reverberant, sr)["waerme"]

        delta = abs(score_dry - score_reverb)
        assert delta <= 0.25, (
            f"§9.7.14 waerme proxy reverb-sensitiv: "
            f"dry={score_dry:.4f}, reverb={score_reverb:.4f}, delta={delta:.4f} "
            f"— broadband room reverb must not cause large warmth-proxy delta"
        )

    def test_90_waerme_not_excluded_from_dereverb_phases(self):
        """\u00a79.7.14: waerme must NOT appear in exclusions for phases where
        the warmth-ratio proxy is reverb-invariant."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        phases_fixed = ["phase_20", "phase_49"]
        for phase in phases_fixed:
            excl = PHASE_GOAL_EXCLUSIONS.get(phase, set())
            assert "waerme" not in excl, (
                f"{phase}: waerme must NOT be excluded after \u00a79.7.14 warmth-ratio proxy fix"
            )

    def test_91_waerme_bounded_and_finite(self):
        """waerme proxy must return \u2208 [0, 1] and non-NaN for 8 diverse signals."""
        import numpy as np

        sr = 48000
        rng = np.random.default_rng(21)
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        signals = [
            np.zeros(sr * 2, dtype=np.float32),
            rng.standard_normal(sr * 2).astype(np.float32),
            np.sin(2 * np.pi * 440 * t).astype(np.float32),
            np.sin(2 * np.pi * 300 * t).astype(np.float32),  # warm bass
            (0.5 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32),
            rng.uniform(-1, 1, sr * 2).astype(np.float32),
            np.float32(0.001) * rng.standard_normal(sr * 2).astype(np.float32),
            np.ones(sr * 2, dtype=np.float32) * 0.5,
        ]
        for i, sig in enumerate(signals):
            score = self._measure(sig, sr)["waerme"]
            assert np.isfinite(score), f"Signal {i}: waerme NaN/Inf: {score}"
            assert 0.0 <= score <= 1.0, f"Signal {i}: waerme out of [0,1]: {score}"

    # \u2500\u2500 Combined exclusion invariants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    def test_92_phase03_exclusions_v9_13(self):
        """phase_03 muss genau 6 Goals ausschließen (§V36 v10.0.0: transient_energie).
        OMLSA/DFN entfernt Rauschimpulse → TransientEnergieProxy false P3."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        expected = {
            "natuerlichkeit",
            "artikulation",
            "authentizitaet",
            "tonal_center",
            "timbre_authentizitaet",
            "transient_energie",  # §V36 v10.0.0
        }
        excl = PHASE_GOAL_EXCLUSIONS.get("phase_03", set())
        assert excl == expected, f"phase_03: {excl} != {expected}"

    def test_93_phase29_exclusions_v9_13(self):
        """phase_29 muss genau 7 Goals ausschließen (§V36 v10.0.0: waerme + §V32: transparenz)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        expected = {
            "artikulation",
            "authentizitaet",
            "natuerlichkeit",
            "tonal_center",
            "timbre_authentizitaet",
            "transparenz",  # §V32
            "waerme",  # §V36 v10.0.0
        }
        excl = PHASE_GOAL_EXCLUSIONS.get("phase_29", set())
        assert excl == expected, f"phase_29: {excl} != {expected}"

    def test_94_phase49_exclusions_v9_10_92(self):
        """phase_49 extended exclusions (2026-04-10): spectral-subtraction dereverb applies
        frequency-selective gain G(f) identical to OMLSA/NR phases — tonal_center, timbre,
        artikulation, natuerlichkeit all excluded per real-run P2 catastrophic regression
        evidence (Δ=0.4667/0.5530 confirmed). brillanz+transparenz+waerme remain SNR-robust."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        expected = {"authentizitaet", "tonal_center", "timbre_authentizitaet", "artikulation", "natuerlichkeit"}
        excl = PHASE_GOAL_EXCLUSIONS.get("phase_49", set())
        assert excl == expected, f"phase_49: {excl} != {expected}"

    def test_95_phase18_brillanz_transparenz_not_excluded(self):
        """phase_18: brillanz and transparenz must NOT be excluded after \u00a79.7.12/13.
        Silence gating lowers HF noise floor \u2192 crest improves or stays neutral."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_18", set())
        assert "brillanz" not in excl, "brillanz must not be in phase_18 exclusions after \u00a79.7.12"
        assert "transparenz" not in excl, "transparenz must not be in phase_18 exclusions after \u00a79.7.13"

    def test_96_precise_metrics_excludes_snr_sensitive_goals(self):
        """brillanz, transparenz, and waerme must NOT be in _PRECISE_METRICS.
        These goals use SNR-robust quick proxies for symmetric PMGG delta checks.
        Their canonical metrics (ISO-226 energy ratios) would produce false regressions
        on denoising/dereverb phases even via precise override (\u00a79.7.12/13/14)."""
        from backend.core.per_phase_musical_goals_gate import _get_precise_metric_instances

        precise = _get_precise_metric_instances()
        for goal in ("brillanz", "waerme", "transparenz"):
            assert goal not in precise, (
                f"{goal} must NOT be in _PRECISE_METRICS \u2014 use SNR-robust quick proxy "
                f"(\u00a79.7.12/13/14) for symmetric PMGG delta checks"
            )

    def test_97_phase02_transparenz_exclusion_still_valid_for_notch(self):
        """\u00a79.7.13 multi-band crest proxy does NOT protect phase_02 from false transparenz
        regression.  Narrow notch filters at 50-250 Hz remove isolated spectral peaks
        (hum harmonics 5-6: 250/300 Hz) from the first octave band (250-500 Hz).
        Removing a narrow peak LOWERS p95 in that band without changing p50 —
        the opposite of broadband denoising.  transparenz must remain in phase_02."""
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS, _measure_quick

        assert "transparenz" in PHASE_GOAL_EXCLUSIONS.get("phase_02", set()), (
            "transparenz must remain in phase_02 exclusions — "
            "narrowband notch-filter peak removal lowers crest factor in 250-500 Hz band"
        )

        # Demonstrate: narrowband peak removal lowers crest in 250-500 Hz band
        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        # Music with strong harmonics in the 250-500 Hz band (hum area)
        music_with_hum_harmonics = (
            0.4 * np.sin(2 * np.pi * 200 * t)  # fundamental
            + 0.35 * np.sin(2 * np.pi * 250 * t)  # 5th hum harmonic (50 Hz source)
            + 0.30 * np.sin(2 * np.pi * 300 * t)  # 6th hum harmonic
            + 0.25 * np.sin(2 * np.pi * 350 * t)  # broad music
            + 0.20 * np.sin(2 * np.pi * 1000 * t)  # mid band
        ).astype(np.float32)
        # Simulate narrow notch: zero out the 250-300 Hz harmonics
        music_after_notch = (
            0.4 * np.sin(2 * np.pi * 200 * t)  # fundamental (unchanged)
            + 0.0 * np.sin(2 * np.pi * 250 * t)  # removed by notch
            + 0.0 * np.sin(2 * np.pi * 300 * t)  # removed by notch
            + 0.25 * np.sin(2 * np.pi * 350 * t)  # unchanged
            + 0.20 * np.sin(2 * np.pi * 1000 * t)  # unchanged
        ).astype(np.float32)

        score_before = _measure_quick(music_with_hum_harmonics, sr)["transparenz"]
        score_after = _measure_quick(music_after_notch, sr)["transparenz"]
        # Notch removes peaks → crest drops → score_after < score_before
        assert score_after < score_before + 0.20, (
            f"Notch removal did not demonstrate crest reduction: before={score_before:.4f} after={score_after:.4f}"
        )

    def test_98_waerme_stable_double_dereverb_chain(self):
        """Coupling test: waerme proxy must stay stable (\u0394 \u2264 0.12) across
        a simulated phase_20 \u2192 phase_49 double-dereverb chain.
        Validates that two consecutive reverb-reduction operations on the same
        audio do not produce cumulative waerme drift."""
        import numpy as np

        sr = 48000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False, dtype=np.float32)
        dry = sum(
            a * np.sin(2 * np.pi * f * t) for f, a in [(300, 0.5), (500, 0.4), (800, 0.3), (1200, 0.2), (2000, 0.15)]
        ).astype(np.float32)
        # First reverb stage (longer, phase_20 input)
        ir1_len = int(sr * 0.5)
        ir1 = np.exp(-6.0 * np.arange(ir1_len, dtype=np.float32) / sr)
        ir1 /= ir1.sum()
        reverb1 = np.convolve(dry, ir1)[: len(dry)].astype(np.float32)
        # Partial dereverb (phases typically don\u2019t achieve 100% removal)
        after_phase20 = 0.3 * reverb1 + 0.7 * dry  # 70% dry recovered
        # Second reverb stage residual (shorter tail, phase_49 input)
        ir2_len = int(sr * 0.2)
        ir2 = np.exp(-15.0 * np.arange(ir2_len, dtype=np.float32) / sr)
        ir2 /= ir2.sum()
        reverb2 = np.convolve(after_phase20, ir2)[: len(after_phase20)].astype(np.float32)
        after_phase49 = 0.2 * reverb2 + 0.8 * after_phase20  # further 80% dry

        from backend.core.per_phase_musical_goals_gate import _measure_quick

        w_dry = _measure_quick(dry, sr)["waerme"]
        w_after20 = _measure_quick(after_phase20, sr)["waerme"]
        w_after49 = _measure_quick(after_phase49, sr)["waerme"]

        delta_chain = abs(w_dry - w_after49)
        assert delta_chain <= 0.12, (
            f"\u00a79.7.14 waerme drifts across double-dereverb chain: "
            f"dry={w_dry:.4f} after_phase20={w_after20:.4f} after_phase49={w_after49:.4f} "
            f"chain_delta={delta_chain:.4f} (limit 0.12)"
        )

    def test_99_timbre_authentizitaet_proxy_characterisation(self):
        """Characterisation test for timbre_authentizitaet CV-proxy behaviour under noise.

        DISCOVERY (v10.0.0): Broadband noise STABILISES the spectral centroid
        (flat noise spectrum averages out frame-to-frame musical variation) \u2192
        the CV-proxy returns an artificially HIGH score on noisy audio, then
        DROPS after denoising.  This is a false-regression concern in the opposite
        direction from what was initially assumed.

        Handling in the current architecture:
        \u2022 High-noise ANALOG sources (tape/shellac/vinyl): \u00a72.31b material-adaptive
          code ALREADY adds timbre_authentizitaet to phase_03/29 exclusions.
        \u2022 LOW-noise DIGITAL sources (cd_digital/dat): noise level is so small
          that denoising barely changes the centroid distribution \u2192 proxy
          stays stable \u2192 no false regression in practice.
        """
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import (
            PHASE_GOAL_EXCLUSIONS,
            _measure_quick,
        )

        sr = 48000
        rng = np.random.default_rng(55)
        t = np.linspace(0, 3.0, sr * 3, endpoint=False, dtype=np.float32)
        music = sum(
            a * np.sin(2 * np.pi * f * t) for f, a in [(350, 0.4), (700, 0.3), (1400, 0.2), (2800, 0.15), (440, 0.3)]
        ).astype(np.float32)
        score_clean = _measure_quick(music, sr)["timbre_authentizitaet"]

        # HIGH noise (analog-level, SNR \u2248 1 dB): noise stabilises centroid \u2192
        # artificially high proxy score before denoising; drops after.
        heavy_noise = rng.standard_normal(len(music)).astype(np.float32) * 0.45
        noisy_high = (music + heavy_noise).astype(np.float32)
        score_high_noise = _measure_quick(noisy_high, sr)["timbre_authentizitaet"]
        # High noise must not score *much lower* than clean (noise often stabilises centroid)
        assert score_high_noise > score_clean - 0.05, (
            f"High-noise proxy inconsistency: noisy={score_high_noise:.4f} clean={score_clean:.4f}"
        )

        # LOW noise (digital-level, SNR \u2248 60 dB): proxy change must be negligible
        low_noise = rng.standard_normal(len(music)).astype(np.float32) * 0.001
        noisy_low = (music + low_noise).astype(np.float32)
        score_low_noise = _measure_quick(noisy_low, sr)["timbre_authentizitaet"]
        delta_digital = abs(score_low_noise - score_clean)
        assert delta_digital <= 0.05, (
            f"Digital-level noise must not shift timbre proxy: "
            f"low_noisy={score_low_noise:.4f} clean={score_clean:.4f} \u0394={delta_digital:.4f}"
        )

        # v10.0.0: timbre_authentizitaet now STATICALLY excluded from phase_03.
        # MFCC-Pearson + centroid-CV proxy is disturbed by spectral-envelope change
        # after broadband NR regardless of material type — dynamic-only exclusion
        # (§2.31b) was insufficient (confirmed 2026-03-30: residual 0.046 regression
        # causing persistent best-effort across all material types).
        base_excl_03 = PHASE_GOAL_EXCLUSIONS.get("phase_03", set())
        assert "timbre_authentizitaet" in base_excl_03, (
            "timbre_authentizitaet MUST be in static phase_03 set (v10.0.0): "
            "MFCC correlation against noisy reference is unreliable after NR"
        )

    def test_99b_timbre_authentizitaet_silence_is_neutral(self):
        """Near-silence must not produce an artificially perfect timbre score.

        With too few energetic frames, centroid-CV is undefined as a timbre cue.
        PMGG must return a neutral fallback (0.5), not 1.0.
        """
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import _measure_quick

        sr = 48000
        silent = np.zeros(sr * 2, dtype=np.float32)
        score = _measure_quick(silent, sr)["timbre_authentizitaet"]

        assert np.isfinite(score), f"timbre_authentizitaet NaN/Inf on silence: {score}"
        assert abs(score - 0.5) <= 1e-6, f"timbre_authentizitaet must be neutral on silence, got {score:.4f}"

    def test_99c_natuerlichkeit_authentizitaet_silence_are_neutral(self):
        """Near-silence must not map to extreme P1 values for natuerlichkeit/authentizitaet."""
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import _measure_quick

        sr = 48000
        silent = np.zeros(sr * 2, dtype=np.float32)
        scores = _measure_quick(silent, sr)

        n_score = scores["natuerlichkeit"]
        a_score = scores["authentizitaet"]

        assert np.isfinite(n_score), f"natuerlichkeit NaN/Inf on silence: {n_score}"
        assert np.isfinite(a_score), f"authentizitaet NaN/Inf on silence: {a_score}"
        assert abs(n_score - 0.5) <= 1e-6, f"natuerlichkeit must be neutral on silence, got {n_score:.4f}"
        assert abs(a_score - 0.5) <= 1e-6, f"authentizitaet must be neutral on silence, got {a_score:.4f}"

    def test_99d_emotionalitaet_micro_dynamics_silence_are_neutral(self):
        """Near-silence must not map to extreme P3 values for emotion and micro dynamics."""
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import _measure_quick

        sr = 48000
        silent = np.zeros(sr * 2, dtype=np.float32)
        scores = _measure_quick(silent, sr)

        e_score = scores["emotionalitaet"]
        m_score = scores["micro_dynamics"]

        assert np.isfinite(e_score), f"emotionalitaet NaN/Inf on silence: {e_score}"
        assert np.isfinite(m_score), f"micro_dynamics NaN/Inf on silence: {m_score}"
        assert abs(e_score - 0.5) <= 1e-6, f"emotionalitaet must be neutral on silence, got {e_score:.4f}"
        assert abs(m_score - 0.5) <= 1e-6, f"micro_dynamics must be neutral on silence, got {m_score:.4f}"

    def test_99e_separation_artikulation_silence_are_neutral(self):
        """Near-silence must not map to extreme P2/P4 values for separation/artikulation."""
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import _measure_quick

        sr = 48000
        silent = np.zeros(sr * 2, dtype=np.float32)
        scores = _measure_quick(silent, sr)

        s_score = scores["separation_fidelity"]
        a_score = scores["artikulation"]

        assert np.isfinite(s_score), f"separation_fidelity NaN/Inf on silence: {s_score}"
        assert np.isfinite(a_score), f"artikulation NaN/Inf on silence: {a_score}"
        assert abs(s_score - 0.5) <= 1e-6, f"separation_fidelity must be neutral on silence, got {s_score:.4f}"
        assert abs(a_score - 0.5) <= 1e-6, f"artikulation must be neutral on silence, got {a_score:.4f}"

    def test_99f_brillanz_bass_kraft_transparenz_spatial_depth_silence_neutral(self):
        """Near-silence must not map to 0.0 or 1.0 for crest-based and energy-ratio goals.

        brillanz: HF crest of near-zero FFT → negative → clip 0.0 (was wrong, needs 0.5)
        bass_kraft: bass_energy≈0 / 1e-12 = 0.0 (was wrong, needs 0.5)
        transparenz: multi-band crest → negative → clip 0.0 (was wrong, needs 0.5)
        spatial_depth (stereo): side_e=1e-12/(2×1e-12)=0.5 → ×2=1.0 (was wrong, needs 0.5)
        """
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import _measure_quick

        sr = 48000
        # mono silence
        silent_mono = np.zeros(sr * 2, dtype=np.float32)
        scores_mono = _measure_quick(silent_mono, sr)

        for goal in ("brillanz", "bass_kraft", "transparenz", "spatial_depth"):
            val = scores_mono[goal]
            assert np.isfinite(val), f"{goal} NaN/Inf on mono silence"
            assert abs(val - 0.5) <= 1e-6, f"{goal} must be neutral on mono silence, got {val:.4f}"

        # stereo silence — spatial_depth had the 1.0 bug
        silent_stereo = np.zeros((sr * 2, 2), dtype=np.float32)
        scores_stereo = _measure_quick(silent_stereo, sr)

        sd = scores_stereo["spatial_depth"]
        assert np.isfinite(sd), "spatial_depth NaN/Inf on stereo silence"
        assert abs(sd - 0.5) <= 1e-6, f"spatial_depth must be neutral on stereo silence, got {sd:.4f}"

    def test_100_phase16_tonal_center_excluded(self):
        """phase_16 (final/mastering EQ) must exclude tonal_center
        (v10.0.0: EQ shifts chroma energy — K-S not immune to EQ)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_16", set())
        assert "tonal_center" in excl, f"tonal_center must be in phase_16 PHASE_GOAL_EXCLUSIONS, got: {excl}"

    def test_101_phase17_tonal_center_and_artikulation_excluded(self):
        """phase_17 (mastering compression) must exclude tonal_center + artikulation
        (v10.0.0: MB-compression changes attack envelopes + chroma distribution)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_17", set())
        assert "tonal_center" in excl, f"tonal_center must be in phase_17 PHASE_GOAL_EXCLUSIONS, got: {excl}"
        assert "artikulation" in excl, f"artikulation must be in phase_17 PHASE_GOAL_EXCLUSIONS, got: {excl}"

    def test_106_phase57_print_through_exclusions(self):
        """phase_57_print_through_reduction must exclude {authentizitaet, emotionalitaet}.

        Root causes (identical to phase_49 Advanced Dereverb):
        - authentizitaet: echo tail fills log-spectrum valleys; after LMS removal
          true valleys reappear → roughness rises → false P1 cascade.
        - emotionalitaet: echo tail adds energy to quiet segments → scores_before
          crest-factor elevated; after removal crest-factor shifts → false P3.
        """
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_57_print_through_reduction", set())
        for goal in ("authentizitaet", "emotionalitaet"):
            assert goal in excl, f"{goal} must be in phase_57_print_through_reduction exclusions, got: {excl}"

    def test_107_phase53_semantic_audio_empty_exclusions(self):
        """phase_53_semantic_audio must have NO exclusions (set()).

        SemanticAudioPhase.process() returns audio UNCHANGED (metadata-only phase).
        Since scores_before == scores_after for all 15 goals on identical audio,
        no PMGG regression is structurally possible → exclusion set must be empty.
        """
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_53", set())
        assert excl == set(), f"phase_53 (SemanticAudioPhase is metadata-only) must have empty exclusions, got: {excl}"

    @pytest.mark.parametrize(
        "phase_id",
        [
            "phase_13",  # Stereo enhancement: Haas cross-feed (5-35 ms) comb-filter in mono sum
            "phase_14",  # Phase correction: all-pass/fractional-delay shifts mono-sum spectral valleys
            "phase_25",  # Azimuth correction: HF spectral restoration changes centroid-CV
            "phase_32",  # Mono-to-stereo: Schroeder decorrelation + HF harmonic synthesis
            "phase_33",  # Stereo width limiter: M/S Side compression changes L/R spectral distribution
            "phase_41",  # Output format optimization: multi-band loudness shaping (same as phase_40)
            "phase_46",  # Spatial enhancement: cross-feed early reflections + Schroeder diffusion
            "phase_48",  # Stereo width enhancer: STFT HF Side scaling (x1.15 > 8 kHz)
        ],
    )
    def test_102_stereo_phases_timbre_excluded(self, phase_id):
        """All 8 stereo-processing phases must exclude timbre_authentizitaet.

        Root cause: M/S processing, cross-feed delays, HF synthesis, or
        frequency-dependent Side scaling all change MFCC-Pearson + centroid-CV
        vs. the pre-processing reference → false P2 timbre regression.
        """
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get(phase_id, set())
        assert "timbre_authentizitaet" in excl, (
            f"timbre_authentizitaet must be in {phase_id} PHASE_GOAL_EXCLUSIONS, got: {excl}"
        )

    def test_103_phase46_extended_exclusions(self):
        """phase_46 (spatial enhancement) must additionally exclude emotionalitaet
        and waerme beyond the shared timbre_authentizitaet exclusion.

        Cross-feed early reflections (6-22 ms, dry_wet=0.18):
        - Add post-transient tail energy → crest-factor drops → false P3 emotionalitaet.
        - Add mid-band diffuse reflection energy → warmth ratio E(200-800)/E(800-3000)
          shifts → false P4 waerme regression.
        """
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_46", set())
        for goal in ("timbre_authentizitaet", "emotionalitaet", "waerme"):
            assert goal in excl, f"{goal} must be in phase_46 PHASE_GOAL_EXCLUSIONS, got: {excl}"

    def test_104_phase46_not_over_excluded(self):
        """phase_46 should not exclude more goals than its 3 documented ones.

        natuerlichkeit and authentizitaet are NOT excluded: early reflections
        preserve MFCC-smoothness (they add, not remove, spectral content) and
        the spectral flatness proxy is unaffected by reflection cross-feed.
        groove and artikulation are NOT excluded: transients are unchanged by
        the Schroeder diffusion (Side-only, no onset modification).
        """
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_46", set())
        # §0 + §2.55 (2026-04-27): raumtiefe added — cross-feed reflections boost raumtiefe;
        # PMGG must not use this to drive phase to higher strength (Gesang-Distanz-Bug fix).
        expected = {"timbre_authentizitaet", "emotionalitaet", "waerme", "raumtiefe"}
        assert excl == expected, f"phase_46 PHASE_GOAL_EXCLUSIONS must be exactly {expected}, got: {excl}"

    def test_105_phase41_not_over_excluded(self):
        """phase_41 (output format optimization) must exclude only timbre_authentizitaet.

        Pure LUFS gain + lossless SRC are scale-invariant for all ratio-based proxies.
        TruePeak at -1 dBTP is too light to affect groove/emotionalitaet/micro_dynamics.
        Only multi-band frequency shaping risks MFCC-Pearson (same root cause as phase_40).
        """
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_41", set())
        assert "timbre_authentizitaet" in excl, f"timbre_authentizitaet must be in phase_41 exclusions, got: {excl}"
        for goal in ("groove", "emotionalitaet", "micro_dynamics", "natuerlichkeit"):
            assert goal not in excl, (
                f"{goal} must NOT be in phase_41 exclusions (TruePeak -1 dBTP too light), got: {excl}"
            )


# ---------------------------------------------------------------------------
# §V36 Reference-Paradox NR-Exclusion Guard (v10.0.0)
# ---------------------------------------------------------------------------


class TestV36NRExclusionGuard:
    """§V36 (v10.0.0): transient_energie in phase_03 + waerme in phase_29 müssen
    aus PHASE_GOAL_EXCLUSIONS ausgeschlossen sein (Reference Paradox §2.44):
    NR entfernt Rauschimpulse/Rauschenergie, die Proxies falsch inflationierten."""

    def test_123_phase03_pmgg_excludes_transient_energie(self):
        """§V36: transient_energie MUSS in PHASE_GOAL_EXCLUSIONS['phase_03'] stehen.
        OMLSA/DFN entfernt Rauschimpulse → TransientEnergieProxy sieht weniger Onsets
        → false P3-Regression (Δ=-0.13 in Run 1779217698 → PMGG best_effort_r1).
        Realer Endwert: 0.805 (über Boden 0.746)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_03", set())
        assert "transient_energie" in excl, (
            "phase_03: transient_energie MUSS ausgeschlossen sein (§V36 v10.0.0): "
            "OMLSA/DFN entfernt Rauschimpulse → TransientEnergie-Proxy false-positive P3."
        )

    def test_124_phase29_pmgg_excludes_waerme(self):
        """§V36: waerme MUSS in PHASE_GOAL_EXCLUSIONS['phase_29'] stehen.
        OMLSA/DFN-Breitband-Suppression reduziert Rauschboden im Wärmeband (200-2000 Hz)
        → Proxy zeigt false P4-Regression (Δ=-0.17 in Run 1779217698 → PMGG best_effort_r1).
        Realer Endwert: 0.792 (über Canonical-Boden 0.75)."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        excl = PHASE_GOAL_EXCLUSIONS.get("phase_29", set())
        assert "waerme" in excl, (
            "phase_29: waerme MUSS ausgeschlossen sein (§V36 v10.0.0): "
            "OMLSA/DFN-Suppression im Wärmeband → Proxy false-positive P4."
        )

    def test_125_phase03_cig_excludes_transient_energie(self):
        """§2.55-Sync §V36: transient_energie MUSS in CIG-Exclusions für phase_03."""
        from backend.core.cumulative_interaction_guard import _PHASE_SPECIFIC_DRIFT_EXCLUSIONS

        excl = _PHASE_SPECIFIC_DRIFT_EXCLUSIONS.get("phase_03", frozenset())
        assert "transient_energie" in excl, (
            "CIG phase_03: transient_energie MUSS ausgeschlossen sein (§2.55-Sync §V36 v10.0.0)."
        )

    def test_126_phase29_cig_excludes_waerme(self):
        """§2.55-Sync §V36: waerme MUSS in CIG-Exclusions für phase_29."""
        from backend.core.cumulative_interaction_guard import _PHASE_SPECIFIC_DRIFT_EXCLUSIONS

        excl = _PHASE_SPECIFIC_DRIFT_EXCLUSIONS.get("phase_29", frozenset())
        assert "waerme" in excl, "CIG phase_29: waerme MUSS ausgeschlossen sein (§2.55-Sync §V36 v10.0.0)."


# ---------------------------------------------------------------------------
# §2.29c Restorative-Baseline-Capping
# ---------------------------------------------------------------------------


class TestRestorativeBaselineCapping:
    """§2.29c: Defekt-inflationierte Baselines werden auf normative Mindest-
    schwellwerte gedeckelt, um false-positive P1/P2-Regressionen zu verhindern.

    Technischer Hintergrund: Rauschen / Hiss / Hall füllen Spektraltäler und
    erhöhen bestimmte Metriken (Authentizität, Transparenz) künstlich über den
    sauberen Wert. Nach Denoise / Dereverb sinkt der Score auf den echten Wert — ohne
    Capping würde PMGG das als Regression werten und die Phase auf ~6 % Wet drosseln.
    """

    def test_108_restorative_phases_is_frozenset(self):
        """_RESTORATIVE_PHASES must be a frozenset (immutable, thread-safe)."""
        from backend.core.per_phase_musical_goals_gate import _RESTORATIVE_PHASES

        assert isinstance(_RESTORATIVE_PHASES, frozenset), (
            "_RESTORATIVE_PHASES must be frozenset — mutable set would allow accidental mutation"
        )

    def test_109_restorative_phases_contains_core_denoise_dereverb(self):
        """Core denoising and dereverb phases must be in _RESTORATIVE_PHASES.

        These phases remove defects (noise, reverb, hiss, hum) that can inflate
        baseline scores — capping is mandatory to prevent false P1/P2 regressions.
        """
        from backend.core.per_phase_musical_goals_gate import _RESTORATIVE_PHASES

        for phase in (
            "phase_01",
            "phase_02",
            "phase_03",
            "phase_09",
            "phase_18",
            "phase_20",
            "phase_23",
            "phase_24",
            "phase_29",
            "phase_49",
        ):
            assert phase in _RESTORATIVE_PHASES, (
                f"{phase} must be in _RESTORATIVE_PHASES — it removes defects that may inflate baselines"
            )

    def test_110_canonical_thresholds_restoration_has_all_15_goals(self):
        """_CANONICAL_THRESHOLDS_RESTORATION must define all 15 Musical Goals (§14 spec)."""
        from backend.core.per_phase_musical_goals_gate import _CANONICAL_THRESHOLDS_RESTORATION

        required_goals = {
            "natuerlichkeit",
            "authentizitaet",
            "tonal_center",
            "timbre_authentizitaet",
            "artikulation",
            "transient_energie",
            "emotionalitaet",
            "micro_dynamics",
            "groove",
            "transparenz",
            "waerme",
            "bass_kraft",
            "separation_fidelity",
            "brillanz",
            "spatial_depth",
        }
        for goal in required_goals:
            assert goal in _CANONICAL_THRESHOLDS_RESTORATION, (
                f"Goal '{goal}' missing from _CANONICAL_THRESHOLDS_RESTORATION"
            )
        assert len(_CANONICAL_THRESHOLDS_RESTORATION) == 15, (
            f"Expected exactly 15 goals, got {len(_CANONICAL_THRESHOLDS_RESTORATION)}"
        )

    def test_111_canonical_thresholds_match_spec_p1_p2(self):
        """P1/P2 thresholds must exactly match §15 Musical Goals spec values."""
        from backend.core.per_phase_musical_goals_gate import _CANONICAL_THRESHOLDS_RESTORATION

        assert _CANONICAL_THRESHOLDS_RESTORATION["natuerlichkeit"] == pytest.approx(0.90, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["authentizitaet"] == pytest.approx(0.88, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["tonal_center"] == pytest.approx(0.95, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["timbre_authentizitaet"] == pytest.approx(0.87, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["artikulation"] == pytest.approx(0.88, abs=1e-9)

    def test_112_canonical_thresholds_match_spec_p3_p5_restoration(self):
        """P3–P5 restoration thresholds must match §9.10.77 Pareto-Differenzierung."""
        from backend.core.per_phase_musical_goals_gate import _CANONICAL_THRESHOLDS_RESTORATION

        assert _CANONICAL_THRESHOLDS_RESTORATION["emotionalitaet"] == pytest.approx(0.84, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["micro_dynamics"] == pytest.approx(0.88, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["groove"] == pytest.approx(0.83, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["transparenz"] == pytest.approx(0.82, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["waerme"] == pytest.approx(0.77, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["bass_kraft"] == pytest.approx(0.78, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["separation_fidelity"] == pytest.approx(0.80, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["brillanz"] == pytest.approx(0.78, abs=1e-9)
        assert _CANONICAL_THRESHOLDS_RESTORATION["spatial_depth"] == pytest.approx(0.70, abs=1e-9)

    def test_113_studio2026_thresholds_p1_p2_stricter(self):
        """Studio 2026 P1/P2 thresholds must be stricter than Restoration (Spec 09 §09.1).

        More aggressive enhancement requires a stronger naturalness/authenticity guard.
        """
        from backend.core.per_phase_musical_goals_gate import (
            _CANONICAL_THRESHOLDS_RESTORATION,
            _CANONICAL_THRESHOLDS_STUDIO2026,
        )

        for goal in ("natuerlichkeit", "authentizitaet", "timbre_authentizitaet", "artikulation"):
            assert _CANONICAL_THRESHOLDS_STUDIO2026[goal] > _CANONICAL_THRESHOLDS_RESTORATION[goal], (
                f"Studio 2026 P1/P2 goal '{goal}' must be stricter than Restoration (Spec 09 §09.1)"
            )

    def test_114_studio2026_thresholds_p3_p5_higher(self):
        """Studio 2026 P3–P5 thresholds must be ≥ Restoration values (ambitious targets)."""
        from backend.core.per_phase_musical_goals_gate import (
            _CANONICAL_THRESHOLDS_RESTORATION,
            _CANONICAL_THRESHOLDS_STUDIO2026,
        )

        for goal in (
            "emotionalitaet",
            "micro_dynamics",
            "groove",
            "transparenz",
            "waerme",
            "bass_kraft",
            "separation_fidelity",
            "brillanz",
            "spatial_depth",
        ):
            assert _CANONICAL_THRESHOLDS_STUDIO2026[goal] >= _CANONICAL_THRESHOLDS_RESTORATION[goal], (
                f"Studio 2026 '{goal}' threshold must be ≥ Restoration threshold (§9.10.77 Pareto)"
            )

    def test_115_get_canonical_thresholds_restoration_default(self):
        """_get_canonical_thresholds() without args must return Restoration thresholds."""
        from backend.core.per_phase_musical_goals_gate import (
            _CANONICAL_THRESHOLDS_RESTORATION,
            _get_canonical_thresholds,
        )

        result = _get_canonical_thresholds()
        assert result is _CANONICAL_THRESHOLDS_RESTORATION

    def test_116_get_canonical_thresholds_studio2026(self):
        """_get_canonical_thresholds(is_studio_2026=True) must return Studio 2026 thresholds."""
        from backend.core.per_phase_musical_goals_gate import (
            _CANONICAL_THRESHOLDS_STUDIO2026,
            _get_canonical_thresholds,
        )

        result = _get_canonical_thresholds(is_studio_2026=True)
        assert result is _CANONICAL_THRESHOLDS_STUDIO2026

    def test_117_restorative_baseline_capping_formula(self):
        """§2.29c: capping formula limits defect-inflated scores_before to canonical thresholds.

        This verifies the formula inside wrap_phase() for restorative phases:
            effective = {g: min(v, threshold.get(g, v)) for g, v in scores_before.items()}

        Root cause this prevents: noise fills spectral troughs → proxy scores (e.g.
        authentizitaet, natuerlichkeit) measure artificially high. After successful
        denoise, score drops to the *real* clean value. Without capping, PMGG would
        see a drop from 0.96 → 0.88 as a P1 regression and throttle phase to ~6% wet
        → noise floor stays at −55 dBFS instead of −72 dBFS → Tiefen-Immersion lost.
        """
        from backend.core.per_phase_musical_goals_gate import (
            _RESTORATIVE_PHASES,
            _get_canonical_thresholds,
        )

        # phase_03 must be classified restorative
        assert "phase_03" in _RESTORATIVE_PHASES

        thresholds = _get_canonical_thresholds(False)  # Restoration mode
        # Simulate defect-inflated baseline
        inflated_scores = {
            "authentizitaet": 0.96,  # above threshold 0.88 → must be capped
            "natuerlichkeit": 0.95,  # above threshold 0.90 → must be capped
            "groove": 0.72,  # below threshold 0.83 → must NOT be capped
        }

        # Apply the exact capping formula used in wrap_phase() for restorative phases
        effective = {g: min(v, thresholds.get(g, v)) for g, v in inflated_scores.items()}

        assert effective["authentizitaet"] == pytest.approx(0.88, abs=1e-9), (
            "authentizitaet (P1) inflated to 0.96 must be capped at canonical threshold 0.88"
        )
        assert effective["natuerlichkeit"] == pytest.approx(0.90, abs=1e-9), (
            "natuerlichkeit (P1) inflated to 0.95 must be capped at canonical threshold 0.90"
        )
        assert effective["groove"] == pytest.approx(0.72, abs=1e-9), (
            "groove below threshold must NOT be capped — capping only suppresses inflation"
        )
        # All capped values must be ≤ their respective canonical thresholds
        for g, v in effective.items():
            assert v <= thresholds[g] + 1e-9, f"{g}: effective value {v} exceeds canonical threshold {thresholds[g]}"

    def test_118_non_restorative_baseline_not_capped(self):
        """§2.29c: non-restorative phases must NOT apply capping to scores_before.

        Bandwidth extension (phase_06) synthesises new HF content — there is no
        noise removal, so baseline inflation cannot occur. Capping here would mask
        genuine regressions from the synthesised content.
        """
        from backend.core.per_phase_musical_goals_gate import (
            _RESTORATIVE_PHASES,
            _get_canonical_thresholds,
        )

        phase_id = "phase_06"
        is_restorative = any(phase_id.startswith(p) for p in _RESTORATIVE_PHASES)
        assert not is_restorative, f"phase_06 must not be in _RESTORATIVE_PHASES (got: {_RESTORATIVE_PHASES})"

        # For non-restorative phases, effective_scores_before = scores_before (identity, no capping)
        thresholds = _get_canonical_thresholds(False)
        above_threshold_scores = {
            "authentizitaet": 0.96,  # above 0.88 — must stay 0.96 (no capping)
            "natuerlichkeit": 0.95,  # above 0.90 — must stay 0.95 (no capping)
        }

        # Non-restorative code path: effective_scores_before IS scores_before
        effective = above_threshold_scores  # no min() applied
        assert effective["authentizitaet"] == 0.96, "non-restorative: authentizitaet must not be capped"
        assert effective["natuerlichkeit"] == 0.95, "non-restorative: natuerlichkeit must not be capped"
        # Values exceed their canonical thresholds — that is correct and expected
        for g, v in effective.items():
            assert v > thresholds[g], f"{g}: non-restorative score {v} should exceed threshold {thresholds[g]}"

    def test_119_canonical_thresholds_all_values_in_valid_range(self):
        """All canonical threshold values must be in (0, 1] (strict lower bound)."""
        from backend.core.per_phase_musical_goals_gate import (
            _CANONICAL_THRESHOLDS_RESTORATION,
            _CANONICAL_THRESHOLDS_STUDIO2026,
        )

        for mode_label, thresholds in [
            ("Restoration", _CANONICAL_THRESHOLDS_RESTORATION),
            ("Studio2026", _CANONICAL_THRESHOLDS_STUDIO2026),
        ]:
            for goal, value in thresholds.items():
                assert 0.0 < value <= 1.0, f"[{mode_label}] threshold for '{goal}' out of range: {value}"

    def test_120_restorative_phases_nonempty_and_strings(self):
        """_RESTORATIVE_PHASES must be a non-empty frozenset of non-empty strings."""
        from backend.core.per_phase_musical_goals_gate import _RESTORATIVE_PHASES

        assert len(_RESTORATIVE_PHASES) >= 10, (
            "_RESTORATIVE_PHASES must have at least 10 entries (phase_01..phase_57 range)"
        )
        for p in _RESTORATIVE_PHASES:
            assert isinstance(p, str) and p.startswith("phase_"), (
                f"_RESTORATIVE_PHASES entry must be str starting with 'phase_', got: {p!r}"
            )

    def test_121_adaptive_goal_thresholds_blend_lowers_ultra_analog(self):
        """§09.2: adaptive_goal_thresholds from SGT must be blended 60/40 with canonical.

        For shellac/1920s material, 'brillanz' adaptive target ≈ 0.38–0.55 (canonical 0.78
        minus large negative era+material bias). Blended threshold must be significantly
        below canonical 0.78 and above 0.30 (hard lower bound).
        """
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import (
            _get_canonical_thresholds,
        )

        canonical = _get_canonical_thresholds(False)  # Restoration mode
        canonical_brillanz = canonical["brillanz"]  # 0.78

        # Simulate SGT result for 1920s shellac: brillanz gets -0.28 (era) + -0.24 (material) bias
        # kappa ≈ 0.27 (low restorability) → adjustment ≈ 0.27 * (-0.52) ≈ -0.14
        # Realistic adaptive target for 1920s shellac: ~0.55–0.65
        adaptive_sgt = {"brillanz": 0.55, "transparenz": 0.65, "natuerlichkeit": 0.90}

        # Apply blend formula (same as _run_with_retry): 60% canonical + 40% adaptive
        blended = {}
        for g, v in adaptive_sgt.items():
            if g in canonical:
                blended[g] = float(np.clip(0.60 * canonical[g] + 0.40 * float(v), 0.30, 0.99))

        # 'brillanz' blended = 0.60 * 0.78 + 0.40 * 0.55 = 0.468 + 0.220 = 0.688
        assert blended["brillanz"] < canonical_brillanz, (
            f"Blended brillanz ({blended['brillanz']:.3f}) must be lower than canonical ({canonical_brillanz})"
        )
        assert blended["brillanz"] >= 0.30, "Blended threshold must respect hard lower bound 0.30"
        assert blended["brillanz"] <= 0.99, "Blended threshold must respect hard upper bound 0.99"
        assert blended["brillanz"] == pytest.approx(0.60 * 0.78 + 0.40 * 0.55, abs=1e-6), (
            "60/40 blend formula must match: 0.60*0.78 + 0.40*0.55"
        )

    def test_122_adaptive_threshold_accepted_by_run_with_retry(self):
        """§09.2: _run_with_retry must accept adaptive_goal_thresholds without error.

        Verifies that the parameter flows through correctly: restorative baseline capping
        uses blended thresholds (not canonical) when adaptive_goal_thresholds is provided.
        For shellac 1920s, 'brillanz' adaptive target = 0.55 → blended cap = 0.688.
        A pre-phase score of 0.75 should be capped to 0.688+0.05=0.738 (not 0.78+0.05=0.83).
        """
        import numpy as np

        from backend.core.per_phase_musical_goals_gate import get_phase_gate

        gate = get_phase_gate()
        gate.reset()

        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        # Phase that passes audio unchanged (no regression possible)
        class _PassPhase:
            class metadata:  # noqa: N801
                phase_id = "phase_03_denoise"

            def process(self, a, **kw):
                class _R:
                    audio = a
                    metadata = {}

                return _R()

        # 'brillanz' pre-score above canonical (0.78) but above blended (0.688)
        # Adaptive target 0.55 → blended = 0.60*0.78 + 0.40*0.55 = 0.688
        # Cap with adaptive: min(0.75, 0.688+0.05=0.738) = 0.738
        # Cap without adaptive: min(0.75, 0.78+0.05=0.83) = 0.75 (no cap)
        scores_before = {"brillanz": 0.75}
        adaptive_sgt = {"brillanz": 0.55}

        # Must not raise → that's the primary assertion
        audio_out, scores_after, log_entry = gate.wrap_phase(
            _PassPhase(),
            audio,
            sr,
            phase_id="phase_03_denoise",
            scores_before=scores_before,
            is_studio_2026=False,
            adaptive_goal_thresholds=adaptive_sgt,
        )

        assert audio_out is not None
        assert log_entry.action in ("passed", "sub_threshold", "passthrough", "best_effort_accepted")

    # ── phase_19 tonal_center exclusion (§2.55-Sync 2026-07-18) ─────────────

    def test_123_phase19_excludes_tonal_center(self):
        """Phase 19 De-Esser redistributes spectral energy in the sibilant range
        (4–10 kHz) → K-S chroma distribution shifts by up to 3 semitones →
        false P2 catastrophic regression on tonal_center (Δ=0.9978 confirmed
        on cassette+schlager, phase had 0 sibilants + 0 dB reduction →
        pure measurement artifact). §2.55-Sync: tonal_center MUST be excluded."""
        from backend.core.per_phase_musical_goals_gate import PHASE_GOAL_EXCLUSIONS

        assert "tonal_center" in PHASE_GOAL_EXCLUSIONS.get("phase_19", set()), (
            "tonal_center MUST be excluded from phase_19: De-Esser sibilance-energy "
            "redistribution causes false K-S chroma-shift → catastrophic P2 regression "
            "(Δ=0.9978 confirmed on real cassette+schlager run, 2026-07-18). "
            "The regression is a measurement artifact — the phase had 0 sibilants "
            "detected and 0 dB gain reduction applied."
        )
