import pytest

"""
tests/unit/test_cumulative_interaction_guard.py — §2.48 Interaktions-Guard Test-Suite (≥ 25 Tests)
Alle Tests synthetisch, kein ML-Modell erforderlich.
"""

import numpy as np

SR = 48_000


def _audio(dur: float = 1.0, amp: float = 0.3):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _goals(nat=0.92, auth=0.90, tonal=0.96, timbre=0.89, artic=0.87):
    # Keys müssen mit measure_all()-Output übereinstimmen (deutsch, §2.48 Namen-Fix)
    return {
        "natuerlichkeit": nat,
        "authentizitaet": auth,
        "tonal_center": tonal,
        "timbre_authentizitaet": timbre,
        "artikulation": artic,
        "emotion": 0.85,
        "micro_dynamics": 0.88,
        "groove": 0.84,
    }


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.cumulative_interaction_guard import (
        CumulativeInteractionGuard,
        get_interaction_guard,
    )

    assert CumulativeInteractionGuard is not None
    assert get_interaction_guard is not None


def test_01_singleton():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    g1 = get_interaction_guard()
    g2 = get_interaction_guard()
    assert g1 is g2


def test_02_reset_returns_fresh_state():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    assert state is not None
    assert len(state.rollback_log) == 0
    assert state.consecutive_rollbacks == 0
    assert not state.should_stop


def test_03_set_baseline():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    goals = _goals()
    guard.set_pre_pipeline_baseline(state, audio, goals)
    assert state.pre_pipeline_goals == goals
    assert state.best_checkpoint is not None
    assert state.best_checkpoint.phase_id == "__pre_pipeline__"


def test_04_no_rollback_on_improvement():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # Phase that improves goals
    improved = _goals(nat=0.94, auth=0.92)
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        audio,
        improved,
        SR,
    )
    assert not rolled_back
    assert np.array_equal(result_audio, audio)


def test_05_rollback_on_cumulative_drift():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # Use ADDITIVE phase_07: tonal_center NOT excluded.
    # natuerlichkeit/artikulation/timbre_authentizitaet/authentizitaet ARE excluded
    # (§2.44; authentizitaet added §2.55-Sync 2026-08-06).
    # tonal_center drift of -0.18 exceeds any default adaptive tolerance.
    degraded = _goals(tonal=0.78)  # -0.18 drift in tonal_center
    bad_audio = audio * 0.5
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_07_harmonic_restoration",
        bad_audio,
        degraded,
        SR,
    )
    assert rolled_back
    assert len(state.rollback_log) == 1
    assert state.rollback_log[0].phase_id == "phase_07_harmonic_restoration"


def test_06_consecutive_rollbacks_stop_pipeline():
    from backend.core.cumulative_interaction_guard import MAX_CONSECUTIVE_ROLLBACKS, get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # All 5 phases are ADDITIVE/DYNAMICS with NO authentizitaet exclusion in CIG
    # (§2.55 sync: phase_19/phase_21 now have nat excluded in CIG — use saturation/EQ phases instead;
    # phase_07 gained authentizitaet exclusion 2026-08-06 — swapped for phase_06 which only
    # excludes timbre_authentizitaet).
    # authentizitaet drift -0.12 exceeds adaptive tolerance for all of them.
    degraded = _goals(auth=0.78)  # -0.12 drift
    rollback_phases = [
        "phase_06_frequency_restoration",  # CIG: timbre only → auth rollback
        "phase_10_multiband_compression",  # CIG: no entry → auth rollback
        "phase_11_limiter",  # CIG: no entry → auth rollback
        "phase_35_multiband_compression",  # CIG: no entry → auth rollback
        "phase_44_stereo_enhancement",  # CIG: timbre only → auth rollback
    ]
    for i, ph in enumerate(rollback_phases):
        guard.check_after_phase(state, ph, audio * 0.3, degraded, SR)
        if i < MAX_CONSECUTIVE_ROLLBACKS - 1:
            assert not state.should_stop, f"should_stop too early after rollback {i + 1}"
    assert state.should_stop


def test_07_rollback_resets_on_success():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # First: degraded natuerlichkeit via ADDITIVE phase → rollback
    degraded = _goals(tonal=0.78)  # -0.18 drift, no exclusions for phase_07 on tonal_center
    guard.check_after_phase(state, "phase_07_harmonic_restoration", audio * 0.5, degraded, SR)
    assert state.consecutive_rollbacks == 1
    # Second: good → reset counter
    improved = _goals(nat=0.93)
    guard.check_after_phase(state, "phase_21_harmonic_exciter", audio, improved, SR)
    assert state.consecutive_rollbacks == 0


def test_08_critical_pair_denoise_dereverb():
    """§2.55 + §2.44 Reference Paradox: natuerlichkeit ist in den Phase-Exclusions
    von phase_20 und phase_49. Deshalb darf die kritische Paarprüfung
    {phase_03, phase_20} auf natuerlichkeit KEINEN Rollback auslösen.
    Reverb-Entfernung senkt den natuerlichkeit-Score intentional (reverb = 'natürlich'
    für die Metrik) — das ist kein Artefakt, sondern §2.46 Carrier-Chain-Inversion.
    Alte Erwartung (assert rolled_back) prüfte einen §2.55-Verstoß."""
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92)
    guard.set_pre_pipeline_baseline(state, audio, baseline)
    # Execute phase_03 — OK
    state.executed_phases.add("phase_03_denoise")
    # phase_20 triggers critical pair check for natuerlichkeit.
    # natuerlichkeit is in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS["phase_20"] → §2.55:
    # critical pair MUST respect exclusions → NO rollback (Reference Paradox §2.44).
    degraded = _goals(nat=0.88)  # drift = -0.04 — intentionale Dereverb-Drift
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_20_reverb_reduction",
        audio * 0.8,
        degraded,
        SR,
    )
    assert not rolled_back, (
        "§2.55 Verletzung: natuerlichkeit ist in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS['phase_20'] "
        "— kritische Paarprüfung darf dieses Goal nicht blockieren (Reference Paradox §2.44)"
    )


def test_08b_critical_pair_non_excluded_goal_still_fires():
    """Wenn ein nicht-ausgeschlossenes Goal in der kritischen Paar-Schwelle regressiert,
    soll der Guard trotzdem EINEN Rollback auslösen.
    Verwendet phase_35 + phase_40 (micro_dynamics) als Testszenario."""
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92)
    baseline["micro_dynamics"] = 0.90
    guard.set_pre_pipeline_baseline(state, audio, baseline)
    state.executed_phases.add("phase_35_multiband_compression")
    # phase_40 with micro_dynamics regression below pair threshold
    degraded = dict(baseline)
    degraded["micro_dynamics"] = 0.84  # drift = -0.06 < -0.04 pair threshold
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_40_lufs_normalization",
        audio * 0.8,
        degraded,
        SR,
    )
    assert rolled_back, "Critical pair {phase_35, phase_40} auf micro_dynamics soll rollback auslösen"


def test_08c_critical_pair_phase29_transparenz_excluded_no_rollback():
    """§2.55 + §2.44: {phase_29, phase_03} + transparenz darf keinen Rollback auslösen,
    wenn transparenz in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS['phase_29'] liegt.

    Wissenschaftliche Begründung:
    Bandrauschen (cassette/tape) ist breitbandige HF-Energie und inflationiert
    transparenz-Proxies (HF-Crest/centroid-nahe Merkmale). Nach Hiss-Reduktion
    fällt der Proxy intentional auf den physikalisch realen Wert (Reference Paradox),
    ohne dass eine echte Klangverschlechterung vorliegt.
    """
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    state.material_type = "cassette"
    state.restorability_score = 55.0

    audio = _audio()
    baseline = _goals(nat=0.92)
    baseline["transparenz"] = 0.88
    guard.set_pre_pipeline_baseline(
        state,
        audio,
        baseline,
        material_type="cassette",
        restorability_score=55.0,
    )

    # pair counterpart bereits ausgeführt
    state.executed_phases.add("phase_03_denoise")

    # transparenz fällt stark (typisch nach Hiss-Entfernung), darf aber
    # wegen Phase-Exclusion keinen Critical-Pair-Rollback triggern.
    degraded = dict(baseline)
    degraded["transparenz"] = 0.52
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_29_tape_hiss_reduction",
        audio * 0.9,
        degraded,
        SR,
    )
    assert not rolled_back, (
        "§2.55 Verletzung: transparenz ist in _PHASE_SPECIFIC_DRIFT_EXCLUSIONS['phase_29'] "
        "und darf im Critical-Pair-Check {phase_29, phase_03} keinen Rollback auslösen"
    )


def test_09_critical_pair_no_trigger_if_good():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    state.executed_phases.add("phase_03_denoise")
    # phase_20 with good goals — no trigger
    good = _goals(nat=0.91)  # drift = -0.01, ok
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_20_reverb_reduction",
        audio,
        good,
        SR,
    )
    assert not rolled_back


def test_10_stft_phase_tracking():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio(2.0)
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # Execute 3 STFT phases — only SUBTRACTIVE/DYNAMICS/ENHANCEMENT count (§2.48a Architecture Inversion)
    # phase_07_harmonic_restoration is ADDITIVE → not tracked in stft_phases_executed (GDD invalid per §2.48a)
    for pid in ["phase_03_denoise", "phase_50_spectral_repair", "phase_29_tape_hiss_reduction"]:
        state.executed_phases.add(pid)
        guard.check_after_phase(state, pid, audio, _goals(), SR)
    assert len(state.stft_phases_executed) == 3


def test_11_checkpoint_updated_on_improvement():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals(nat=0.92))
    # Better phase
    improved_audio = audio * 1.01
    improved_goals = _goals(nat=0.95)
    guard.check_after_phase(state, "phase_01_click_removal", improved_audio, improved_goals, SR)
    assert state.best_checkpoint is not None
    assert state.best_checkpoint.phase_id == "phase_01_click_removal"


def test_12_metadata_empty_for_clean_run():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    meta = guard.get_rollback_metadata(state)
    assert meta["interaction_rollbacks"] == []
    assert meta["pipeline_stopped_early"] is False


def test_13_metadata_contains_rollback_info():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    guard.set_pre_pipeline_baseline(state, audio, _goals())
    # ADDITIVE phase_07: tonal_center drift triggers rollback (no exclusions)
    degraded = _goals(tonal=0.78)  # -0.18 drift
    guard.check_after_phase(state, "phase_07_harmonic_restoration", audio, degraded, SR)
    meta = guard.get_rollback_metadata(state)
    assert len(meta["interaction_rollbacks"]) == 1
    assert meta["interaction_rollbacks"][0]["phase_id"] == "phase_07_harmonic_restoration"


def test_14_p1_p2_goals_constant():
    from backend.core.cumulative_interaction_guard import P1_P2_GOALS

    # Prüfe deutsche Namen (measure_all()-kompatibel, §2.48 Namen-Fix)
    assert "natuerlichkeit" in P1_P2_GOALS
    assert "authentizitaet" in P1_P2_GOALS
    assert "tonal_center" in P1_P2_GOALS
    assert "timbre_authentizitaet" in P1_P2_GOALS
    assert "artikulation" in P1_P2_GOALS
    assert len(P1_P2_GOALS) == 5


def test_15_stft_phases_constant():
    from backend.core.cumulative_interaction_guard import STFT_PHASES

    assert "phase_03_denoise" in STFT_PHASES
    assert "phase_07_harmonic_restoration" in STFT_PHASES
    assert "phase_29_tape_hiss_reduction" in STFT_PHASES


def test_16_critical_pairs_defined():
    from backend.core.cumulative_interaction_guard import CRITICAL_PAIRS

    assert len(CRITICAL_PAIRS) >= 5
    for pair_phases, guard_goal, description, max_reg in CRITICAL_PAIRS:
        assert isinstance(pair_phases, frozenset)
        assert isinstance(guard_goal, str)
        assert max_reg < 0


def test_17_group_delay_check_short_audio():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard

    guard = CumulativeInteractionGuard()
    short = np.zeros(100, dtype=np.float32)
    # Should return True (OK) for very short audio
    assert guard._check_group_delay(short, short, SR) is True


def test_18_group_delay_identical_audio():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard

    guard = CumulativeInteractionGuard()
    audio = _audio(1.0)
    assert guard._check_group_delay(audio, audio, SR) is True


def test_19_is_better_checkpoint_none():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard, InteractionGuardState

    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    assert guard._is_better_checkpoint(state, _goals()) is True


def test_20_is_better_checkpoint_worse():
    from backend.core.cumulative_interaction_guard import (
        CumulativeInteractionGuard,
        InteractionGuardCheckpoint,
        InteractionGuardState,
    )

    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.best_checkpoint = InteractionGuardCheckpoint(
        audio=_audio(),
        phase_id="best",
        goal_scores=_goals(nat=0.95),
    )
    # Worse goals
    assert guard._is_better_checkpoint(state, _goals(auth=0.78)) is False


def test_21_check_critical_pairs_no_match():
    from backend.core.cumulative_interaction_guard import CumulativeInteractionGuard, InteractionGuardState

    guard = CumulativeInteractionGuard()
    state = InteractionGuardState()
    state.executed_phases = {"phase_01_click_removal"}
    result = guard._check_critical_pairs(state, "phase_02_hum_removal", _goals())
    assert result is None

    def test_22_phase30_authentizitaet_false_positive_is_excluded():
        """phase_30 must not rollback solely on authenticity proxy drop."""
        from backend.core.cumulative_interaction_guard import get_interaction_guard

        guard = get_interaction_guard()
        state = guard.reset()
        audio = _audio()
        baseline = _goals(nat=0.92, auth=0.90, tonal=0.96)
        guard.set_pre_pipeline_baseline(state, audio, baseline)

        # Typical false-positive pattern observed in runtime logs:
        # authenticity drops, but other P1/P2 remain stable.
        current = _goals(nat=0.92, auth=0.52, tonal=0.96)
        _, rolled_back = guard.check_after_phase(
            state,
            "phase_30_dc_offset_removal",
            audio,
            current,
            SR,
        )

        assert not rolled_back

    def test_23_phase05_authentizitaet_false_positive_is_excluded():
        """phase_05 must not rollback solely on authenticity proxy drop."""
        from backend.core.cumulative_interaction_guard import get_interaction_guard

        guard = get_interaction_guard()
        state = guard.reset()
        audio = _audio()
        baseline = _goals(nat=0.92, auth=0.90, tonal=0.96)
        guard.set_pre_pipeline_baseline(state, audio, baseline)

        current = _goals(nat=0.92, auth=0.52, tonal=0.96)
        _, rolled_back = guard.check_after_phase(
            state,
            "phase_05_rumble_filter",
            audio,
            current,
            SR,
        )

        assert not rolled_back


def test_22_should_stop_prevents_further_checks():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    state.should_stop = True
    audio = _audio()
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        audio,
        _goals(),
        SR,
    )
    assert not rolled_back


def test_24_phase30_short_id_authentizitaet_false_positive_is_excluded():
    """Prefix-matching must also work for short IDs (phase_30)."""
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92, auth=0.90, tonal=0.96)
    guard.set_pre_pipeline_baseline(state, audio, baseline)

    current = _goals(nat=0.92, auth=0.52, tonal=0.96)
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_30",
        audio,
        current,
        SR,
    )

    assert not rolled_back


def test_25_phase05_short_id_authentizitaet_false_positive_is_excluded():
    """Prefix-matching must also work for short IDs (phase_05)."""
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92, auth=0.90, tonal=0.96)
    guard.set_pre_pipeline_baseline(state, audio, baseline)

    current = _goals(nat=0.92, auth=0.52, tonal=0.96)
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_05",
        audio,
        current,
        SR,
    )

    assert not rolled_back


def test_25b_phase30_natuerlichkeit_false_positive_is_excluded():
    """phase_30 must not rollback solely on naturalness proxy drop."""
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92, auth=0.90, tonal=0.96)
    guard.set_pre_pipeline_baseline(state, audio, baseline)

    current = _goals(nat=0.70, auth=0.90, tonal=0.96)
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_30_dc_offset_removal",
        audio,
        current,
        SR,
    )

    assert not rolled_back


def test_25c_phase05_natuerlichkeit_false_positive_is_excluded():
    """phase_05 must not rollback solely on naturalness proxy drop."""
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92, auth=0.90, tonal=0.96)
    guard.set_pre_pipeline_baseline(state, audio, baseline)

    current = _goals(nat=0.70, auth=0.90, tonal=0.96)
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_05_rumble_filter",
        audio,
        current,
        SR,
    )

    assert not rolled_back


def test_25d_phase24_natuerlichkeit_false_positive_is_excluded():
    """phase_24 must not rollback solely on naturalness proxy drop after gap filling."""
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    baseline = _goals(nat=0.92, auth=0.90, tonal=0.96)
    guard.set_pre_pipeline_baseline(state, audio, baseline)

    current = _goals(nat=0.52, auth=0.90, tonal=0.96)
    _, rolled_back = guard.check_after_phase(
        state,
        "phase_24_dropout_repair",
        audio,
        current,
        SR,
    )

    assert not rolled_back


def test_23_rollback_returns_checkpoint_audio():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    original_audio = _audio()
    guard.set_pre_pipeline_baseline(state, original_audio, _goals())
    # ADDITIVE phase_07: tonal_center drift triggers rollback (no exclusions)
    bad_audio = original_audio * 0.1
    degraded = _goals(tonal=0.78)  # -0.18 drift
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_07_harmonic_restoration",
        bad_audio,
        degraded,
        SR,
    )
    assert rolled_back
    # Should return checkpoint audio, not the degraded audio
    assert not np.array_equal(result_audio, bad_audio)


def test_24_no_baseline_no_crash():
    from backend.core.cumulative_interaction_guard import get_interaction_guard

    guard = get_interaction_guard()
    state = guard.reset()
    audio = _audio()
    # No baseline set — should not crash
    result_audio, rolled_back = guard.check_after_phase(
        state,
        "phase_03_denoise",
        audio,
        _goals(),
        SR,
    )
    assert not rolled_back


def test_25_max_drift_threshold():
    from backend.core.cumulative_interaction_guard import MAX_CUMULATIVE_DRIFT

    assert MAX_CUMULATIVE_DRIFT == -0.05


def test_26_max_consecutive_rollbacks():
    from backend.core.cumulative_interaction_guard import MAX_CONSECUTIVE_ROLLBACKS

    # §2.54: Notbremse-Fallback = 5 — adaptiver Wert max(5, n_carrier_phases + 2)
    # erlaubt Mehrgenerations-Material (vinyl→tape→mp3) ausreichend Carrier-Phasen.
    assert MAX_CONSECUTIVE_ROLLBACKS == 5


def test_27_interaction_rollback_dataclass():
    from backend.core.cumulative_interaction_guard import InteractionRollback

    rb = InteractionRollback(
        phase_id="phase_03_denoise",
        reason="test",
        drift={"natuerlichkeit": -0.06},
        rolled_back_to="__pre_pipeline__",
    )
    assert rb.phase_id == "phase_03_denoise"
    assert rb.drift["natuerlichkeit"] == -0.06
    assert rb.drift["natuerlichkeit"] == -0.06


def test_28_spectral_subtraction_phases_constant():
    """§2.48: _SPECTRAL_SUBTRACTION_PHASES enthält die 4 betroffenen Phasen."""
    from backend.core.cumulative_interaction_guard import _SPECTRAL_SUBTRACTION_PHASES

    expected = {
        "phase_03_denoise",
        "phase_20_reverb_reduction",
        "phase_29_tape_hiss_reduction",
        "phase_49_advanced_dereverb",
    }
    assert expected == set(_SPECTRAL_SUBTRACTION_PHASES)


def test_29_spectral_threshold_higher_than_standard():
    """§2.48: Spektralsubtraktion-Phasen erhalten 10 ms, Standard 5 ms."""
    from backend.core.cumulative_interaction_guard import (
        MAX_GROUP_DELAY_DEVIATION_MS,
        MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL,
    )

    assert MAX_GROUP_DELAY_DEVIATION_MS == 5.0
    assert MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL == 10.0
    assert MAX_GROUP_DELAY_DEVIATION_MS_SPECTRAL > MAX_GROUP_DELAY_DEVIATION_MS


def test_30_group_delay_identical_audio_spectral_phase_passes():
    """§2.48: Identisches Audio → 0 ms Deviation → immer True, auch für Spektralphase."""
    import numpy as np

    from backend.core.cumulative_interaction_guard import get_interaction_guard

    rng = np.random.default_rng(42)
    sr = 48_000
    audio = rng.standard_normal(sr * 2).astype(np.float32) * 0.1

    guard = get_interaction_guard()
    ok = guard._check_group_delay(audio, audio, sr, phase_id="phase_29_tape_hiss_reduction")
    assert ok, "Identisches Audio muss bei Spektralphase passieren"


def test_31_group_delay_spectral_phase_uses_10ms_threshold():
    """§2.48: Spektralphase erhält 10 ms Threshold — Mock-Kontrolle der internen Messung."""
    from unittest.mock import patch

    import numpy as np

    from backend.core.cumulative_interaction_guard import get_interaction_guard

    rng = np.random.default_rng(42)
    sr = 48_000
    audio = rng.standard_normal(sr * 2).astype(np.float32) * 0.1
    guard = get_interaction_guard()

    # 7 ms in samples = 7 * 48000 / 1000 = 336 samples
    # Mit Mock liefern wir stets 336 als percentile-Wert → 7.0 ms Deviation
    controlled_samples = 7.0 * sr / 1000.0  # 336
    real_percentile = np.percentile
    calls: list[float] = []

    def _mock_percentile(a, q, *args, **kwargs):
        # Nur den gd_valid-Percentile-Call intercepten (1-D float array, q=95)
        if isinstance(a, np.ndarray) and a.ndim == 1 and q == 95:
            calls.append(float(controlled_samples))
            return controlled_samples
        return real_percentile(a, q, *args, **kwargs)

    with patch("backend.core.cumulative_interaction_guard.np.percentile", side_effect=_mock_percentile):
        ok_spectral = guard._check_group_delay(audio, audio, sr, phase_id="phase_29_tape_hiss_reduction")
        ok_standard = guard._check_group_delay(audio, audio, sr, phase_id="phase_07_harmonic_restoration")

    # Muss mindestens 1× intercepted worden sein
    assert len(calls) >= 1, "_mock_percentile nicht aufgerufen"
    # 7 ms < 10 ms → Spektralphase passiert
    assert ok_spectral, "phase_29 soll bei 7 ms (< 10 ms-Threshold) passieren"
    # 7 ms > 5 ms → Standard-Phase scheitert
    assert not ok_standard, "phase_07 soll bei 7 ms (> 5 ms-Threshold) scheitern"


def test_32_group_delay_spectral_phase_fails_above_10ms():
    """§2.48: Auch Spektralphase scheitert bei > 10 ms Deviation."""
    from unittest.mock import patch

    import numpy as np

    from backend.core.cumulative_interaction_guard import get_interaction_guard

    rng = np.random.default_rng(42)
    sr = 48_000
    audio = rng.standard_normal(sr * 2).astype(np.float32) * 0.1
    guard = get_interaction_guard()

    # 11 ms → 528 samples
    controlled_samples = 11.0 * sr / 1000.0
    real_percentile = np.percentile

    def _mock_percentile(a, q, *args, **kwargs):
        if isinstance(a, np.ndarray) and a.ndim == 1 and q == 95:
            return controlled_samples
        return real_percentile(a, q, *args, **kwargs)

    with patch("backend.core.cumulative_interaction_guard.np.percentile", side_effect=_mock_percentile):
        ok = guard._check_group_delay(audio, audio, sr, phase_id="phase_29_tape_hiss_reduction")

    assert not ok, "phase_29 soll bei 11 ms (> 10 ms-Threshold) scheitern"


def test_33_carrier_repair_does_not_increment_consecutive_rollbacks():
    """§2.48 Carrier-Repair-Ausnahme: Rollbacks auf Carrier-Phasen dürfen consecutive_rollbacks
    nicht inkrementieren (§2.44 Reference Paradox).  Bug war: _CARRIER_REPAIR_PHASE_PREFIXES
    definiert aber nie genutzt → Pipeline-Stop nach 3 Carrier-Rollbacks.

    Test-Strategie: phase_25_azimuth_correction ist carrier-repair UND CORRECTIVE
    (nicht SUBTRACTIVE) → natuerlichkeit wird geprüft und kann einen Rollback auslösen.
    Mit straffer cd_digital-Tolerance triggert ein starker natuerlichkeit-Drop einen Rollback.
    Danach muss consecutive_rollbacks 0 geblieben sein.
    """
    import numpy as np

    from backend.core.cumulative_interaction_guard import (
        _CARRIER_REPAIR_PHASE_PREFIXES,
        get_interaction_guard,
    )

    rng = np.random.default_rng(17)
    sr = 48_000
    audio = rng.standard_normal(sr * 2).astype(np.float32) * 0.1
    guard = get_interaction_guard()
    state = guard.reset()

    # Verifikation: phase_25 muss carrier sein
    assert any("phase_25".startswith(p) for p in _CARRIER_REPAIR_PHASE_PREFIXES), (
        "phase_25 muss in _CARRIER_REPAIR_PHASE_PREFIXES sein (Test-Voraussetzung)"
    )

    # cd_digital + hohe Restorability → straffes adaptive_drift_tolerance (~-0.025)
    baseline_goals = {
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        "tonal_center": 0.95,
        "timbre_authentizitaet": 0.87,
        "artikulation": 0.85,
    }
    guard.set_pre_pipeline_baseline(
        state,
        audio,
        baseline_goals,
        material_type="cd_digital",
        restorability_score=95.0,
        defect_severity_mean=0.0,
        n_active_phases=5,
        n_carrier_phases=1,
    )
    assert state.adaptive_drift_tolerance > -0.10, (
        f"cd_digital tolerance sollte eng sein, got {state.adaptive_drift_tolerance}"
    )

    # Starker natuerlichkeit-Drop: -0.30, weit unter Tolerance → Rollback erwartet
    # phase_25 excludiert authentizitaet + timbre_authentizitaet; natuerlichkeit wird geprüft.
    regressed_goals = {
        "natuerlichkeit": 0.60,  # Δ = -0.30, weit unter cd_digital Tolerance
        "authentizitaet": 0.88,  # excluded für phase_25
        "tonal_center": 0.60,  # Δ = -0.35, auch geprüft
        "timbre_authentizitaet": 0.87,  # excluded für phase_25
        "artikulation": 0.85,  # geprüft für phase_25
    }

    before_count = state.consecutive_rollbacks  # == 0
    _, was_rolled = guard.check_after_phase(state, "phase_25_azimuth_correction", audio, regressed_goals, sr)

    # Sanity: Rollback MUSS gefeuert haben (kein False-Negative im Test)
    assert was_rolled, "Rollback hätte bei natuerlichkeit-Drift=-0.30 mit cd_digital-Tolerance feuern sollen"
    # Carrier-Phase: consecutive_rollbacks darf NICHT erhöht sein
    assert state.consecutive_rollbacks == before_count, (
        "Carrier-Repair-Phase phase_25_azimuth_correction hat consecutive_rollbacks inkrementiert "
        f"({before_count} \u2192 {state.consecutive_rollbacks}). Bug: \u00a72.44 Carrier-Repair-Ausnahme fehlt."
    )


def test_34_adaptive_critical_pair_threshold_vinyl_permissive():
    """§2.54: Critical-Pair max_reg muss für Vinyl 3× permissiver sein als CD-Basis.
    Für {phase_29, phase_03} + natuerlichkeit: base=-0.03, vinyl_scale=3.0 → effective ~ -0.09."""

    from backend.core.cumulative_interaction_guard import (
        InteractionGuardState,
        get_interaction_guard,
    )

    guard = get_interaction_guard()

    # State mit Vinyl-Kontext
    state = InteractionGuardState()
    state.material_type = "vinyl"
    state.restorability_score = 55.0

    # CD-State
    state_cd = InteractionGuardState()
    state_cd.material_type = "cd_digital"
    state_cd.restorability_score = 80.0

    base = -0.03
    vinyl_threshold = guard._compute_adaptive_pair_threshold(base, state)
    cd_threshold = guard._compute_adaptive_pair_threshold(base, state_cd)

    # Vinyl muss permissiver (negativer) als CD sein
    assert vinyl_threshold < cd_threshold, (
        f"Vinyl ({vinyl_threshold:.3f}) muss permissiver als CD ({cd_threshold:.3f}) sein"
    )
    # Vinyl: min 2× so permissiv wie Basis
    assert vinyl_threshold <= base * 2.0, f"Vinyl-Threshold ({vinyl_threshold:.3f}) sollte ≤ {base * 2.0:.3f} sein"
    # Nie permissiver als 5× Basis
    assert vinyl_threshold >= base * 5.0, (
        f"Vinyl-Threshold ({vinyl_threshold:.3f}) darf nicht > 5× Basis ({base * 5.0:.3f}) sein"
    )
    # CD: nahe am Basis-Wert (keine große Skalierung)
    assert cd_threshold >= base * 2.0, (
        f"CD-Threshold ({cd_threshold:.3f}) sollte nicht zu weit von Basis ({base:.3f}) geöffnet werden"
    )


def test_35_critical_pair_threshold_respects_song_goal_weights():
    """§2.56: guard_goal weight must influence critical-pair threshold.

    Higher weight for natuerlichkeit => stricter threshold (less negative).
    Lower weight => more permissive threshold (more negative).
    """

    from backend.core.cumulative_interaction_guard import InteractionGuardState, get_interaction_guard

    guard = get_interaction_guard()
    base = -0.03

    state_high = InteractionGuardState()
    state_high.material_type = "vinyl"
    state_high.restorability_score = 55.0
    state_high.goal_weights = {"natuerlichkeit": 2.0}

    state_low = InteractionGuardState()
    state_low.material_type = "vinyl"
    state_low.restorability_score = 55.0
    state_low.goal_weights = {"natuerlichkeit": 0.3}

    thr_high = guard._compute_adaptive_pair_threshold(base, state_high, guard_goal="natuerlichkeit")
    thr_low = guard._compute_adaptive_pair_threshold(base, state_low, guard_goal="natuerlichkeit")

    # Higher weight => stricter threshold => closer to zero (numerically larger)
    assert thr_high > thr_low, f"Expected stricter threshold for high weight: high={thr_high:.3f}, low={thr_low:.3f}"
    # Keep existing safety bounds intact
    assert base * 5.0 <= thr_high <= base
    assert base * 5.0 <= thr_low <= base
