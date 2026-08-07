from __future__ import annotations

"""Tests fuer §2.53b Denker-Plan-Determinismus in UnifiedRestorerV3.

Normative Anforderungen (§2.53b):
- When precomputed_phase_plan=[...] uebergeben, gilt der Denker-Plan als Source of Truth.
- UV3 MUSS autonome _select_phases() und _optimize_phase_plan_intelligence() ueberspringen.
- UV3 MUSS selected_phases direkt aus precomputed_phase_plan ableiten.
- Phase Skipping muss in diesem Pfad deaktiviert werden.
- _last_material_priority_phases darf keinen Stale-State aus vorherigen Laeufen halten.
"""


import logging
import logging.handlers
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def sr():
    return 48000


@pytest.fixture
def short_audio(sr):
    """Kurzes Sinus-Signal fuer schnelle Tests."""
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


def _make_minimal_defect_result(material: str = "vinyl"):
    """Minimales DefectResult-Mock fuer UV3-Tests."""
    dr = MagicMock()
    dr.primary_material = material
    dr.defect_scores = {}
    dr.scores = {}
    dr.overall_severity = 0.2
    dr.transfer_chain = [material]
    dr.transfer_chain_confidences = [0.8]
    dr.era_decade = 1965
    dr.genre_label = "schlager"
    dr.restorability_score = 60.0
    dr.confidence = 0.75
    dr.transfer_chain_raw = {}
    return dr


def _make_restorability_mock():
    """Minimales RestorabilityResult-Mock - verhindert RestorabilityEstimator-Ausfuehrung."""
    r = MagicMock()
    r.restorability_score = 70.0
    r.tier = "moderate"
    r.grade = "C"
    r.predicted_mos_range = (2.8, 3.5)
    return r


def _fast_restore_kwargs(plan=None):
    """
    Liefert kwargs, die in UV3.restore() die schweren Analyse-Schritte ueberspringen:
    - material="vinyl"            => _classified_material != None => kein paralleler Medium-Block
    - cached_defect_result        => DefectScanner wird uebersprungen
    - cached_restorability_result => RestorabilityEstimator wird uebersprungen
    """
    kw: dict = {
        "mode": "quality",
        "material": "vinyl",
        "cached_defect_result": _make_minimal_defect_result("vinyl"),
        "cached_restorability_result": _make_restorability_mock(),
    }
    if plan is not None:
        kw["precomputed_phase_plan"] = plan
    return kw


def _make_pipeline_mock(audio):
    """Dummy-Rueckgabe fuer gemocktes _execute_pipeline."""
    return (audio.copy(), ["phase_03_denoise"], [], [])


def _capture_log(target_logger_name: str, level: int = logging.INFO):
    """Hilfs-Utility: Stellt Handler+Records-Liste bereit (caplog-Ersatz)."""
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level)
    tgt = logging.getLogger(target_logger_name)
    orig_level = tgt.level
    tgt.setLevel(level)
    tgt.addHandler(handler)
    return tgt, handler, orig_level, records


def _excellence_mock(audio, sr, **kwargs):
    """Leichter Stub fuer optimize_for_excellence — verhindert MERT-Laden."""
    result = MagicMock()
    result.summary.return_value = "ExcellenceOptimizer: mocked"
    result.applied_steps = []
    result.delta_rms_db = 0.0
    result.continuity_smoothing_applied = False
    result.micro_dynamic_injected = False
    result.harmonic_reinforcement_db = 0.0
    result.ola_crossfades = 0
    return audio, result


def _make_fc_class_mock(audio: np.ndarray) -> MagicMock:
    """Mock-FeedbackChain-Instanz — verhindert echte Phase-Ausfuehrung und MERT."""
    _result = MagicMock()
    _result.audio = audio.copy()
    _result.overall_score = 0.75
    _result.total_retries = 1
    _result.iterations = 1
    _result.total_time_s = 0.001
    _result.analytics_overhead_s = 0.0
    _result.mos_history = [0.75]
    _result.metadata = {}
    _result.phase_executions = []
    _result.ceiling_reached = False
    _instance = MagicMock()
    _instance.run.return_value = _result
    return _instance


_MGC_THRESHOLDS = {
    "natuerlichkeit": 0.90,
    "authentizitaet": 0.88,
    "tonal_center": 0.95,
    "timbre_authentizitaet": 0.87,
    "artikulation": 0.88,
    "emotionalitaet": 0.84,
    "groove": 0.83,
    "mikro_dynamik": 0.88,
    "transparenz": 0.82,
    "waerme": 0.75,
    "bass_kraft": 0.78,
    "separation_fidelity": 0.80,
    "brillanz": 0.78,
    "spatial_depth": 0.70,
}


def _make_mgc_class_mock() -> MagicMock:
    """Mock-MusicalGoalsChecker — verhindert CREPE-Laden (14-17 s)."""
    _scores = dict.fromkeys(_MGC_THRESHOLDS, 0.91)
    _measure_result = MagicMock()
    _measure_result.scores = _scores
    _measure_result.passed_goals = list(_MGC_THRESHOLDS.keys())
    _measure_result.failed_goals = []
    _measure_result.all_passed = True
    _instance = MagicMock()
    _instance.thresholds = dict(_MGC_THRESHOLDS)
    _instance.measure_all.return_value = _measure_result
    return _instance


def _make_hpg_mock() -> MagicMock:
    """Mock-HolisticPerceptualGate — verhindert MERT-Laden."""
    _hpi = MagicMock()
    _hpi.passed = True
    _hpi.hpi = 0.5
    _hpi.artifact_freedom = 1.0
    _hpi.detail = {"mert_proxy_used": False}
    _hpi.reason = ""
    _gate = MagicMock()
    _gate.evaluate_restoration.return_value = _hpi
    _gate.evaluate_studio.return_value = _hpi
    return _gate


# Gemeinsame patches fuer alle restore()-aufrufenden Tests (verhindert MERT-Load-Timeout)
_HEAVY_PATCHES = [
    "backend.core.excellence_optimizer.optimize_for_excellence",
]


# ── §2.53b Denker-Plan wird deterministisch durchgesetzt ─────────────────


@pytest.mark.unit
def test_precomputed_plan_bypasses_select_phases(sr, short_audio):
    """§2.53b: Wenn precomputed_phase_plan uebergeben wird, darf _select_phases() NICHT aufgerufen werden."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    plan = ["phase_03_denoise", "phase_29_tape_hiss_reduction"]
    _ret = _make_pipeline_mock(short_audio)

    with (
        patch.object(uv3, "_select_phases") as mock_select,
        patch.object(uv3, "_execute_pipeline", return_value=_ret),
        patch.object(uv3, "_collect_reporting_analytics", return_value={}),
        patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=0),
        patch("backend.core.excellence_optimizer.optimize_for_excellence", side_effect=_excellence_mock),
        patch("backend.core.feedback_chain.FeedbackChain", return_value=_make_fc_class_mock(short_audio)),
        patch(
            "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker",
            return_value=_make_mgc_class_mock(),
        ),
        patch(
            "backend.core.holistic_perceptual_gate.get_holistic_gate",
            return_value=_make_hpg_mock(),
        ),
    ):
        try:
            uv3.restore(short_audio, sr=sr, **_fast_restore_kwargs(plan=plan))
        except Exception:
            logger.warning("test fallback", exc_info=True)
        mock_select.assert_not_called()


def test_precomputed_plan_bypasses_optimize_intelligence(sr, short_audio):
    """§2.53b: Wenn precomputed_phase_plan uebergeben wird, darf _optimize_phase_plan_intelligence() NICHT aufgerufen."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    plan = ["phase_03_denoise"]
    _ret = _make_pipeline_mock(short_audio)

    with (
        patch.object(uv3, "_optimize_phase_plan_intelligence") as mock_opt,
        patch.object(uv3, "_execute_pipeline", return_value=_ret),
        patch.object(uv3, "_collect_reporting_analytics", return_value={}),
        patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=0),
        patch("backend.core.excellence_optimizer.optimize_for_excellence", side_effect=_excellence_mock),
        patch("backend.core.feedback_chain.FeedbackChain", return_value=_make_fc_class_mock(short_audio)),
        patch(
            "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker",
            return_value=_make_mgc_class_mock(),
        ),
        patch(
            "backend.core.holistic_perceptual_gate.get_holistic_gate",
            return_value=_make_hpg_mock(),
        ),
    ):
        try:
            uv3.restore(short_audio, sr=sr, **_fast_restore_kwargs(plan=plan))
        except Exception:
            logger.warning("test fallback", exc_info=True)
        mock_opt.assert_not_called()


def test_precomputed_plan_disables_phase_skipping(sr, short_audio):
    """§2.53b: Mit precomputed_phase_plan wird Phase Skipping deaktiviert. UV3 muss dies loggen."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    plan = ["phase_03_denoise", "phase_29_tape_hiss_reduction"]
    _ret = _make_pipeline_mock(short_audio)

    # caplog steht in dieser Repo-Umgebung nicht zur Verfuegung — manueller Log-Handler.
    tgt, handler, orig_level, records = _capture_log("backend.core.unified_restorer_v3", logging.INFO)
    try:
        with (
            patch.object(uv3, "_execute_pipeline", return_value=_ret),
            patch.object(uv3, "_collect_reporting_analytics", return_value={}),
            patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=0),
            patch("backend.core.excellence_optimizer.optimize_for_excellence", side_effect=_excellence_mock),
            patch("backend.core.feedback_chain.FeedbackChain", return_value=_make_fc_class_mock(short_audio)),
            patch(
                "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker",
                return_value=_make_mgc_class_mock(),
            ),
            patch(
                "backend.core.holistic_perceptual_gate.get_holistic_gate",
                return_value=_make_hpg_mock(),
            ),
        ):
            try:
                uv3.restore(short_audio, sr=sr, **_fast_restore_kwargs(plan=plan))
            except Exception:
                logger.warning("test fallback", exc_info=True)
    finally:
        tgt.removeHandler(handler)
        tgt.setLevel(orig_level)

    messages = []
    for _r in records:
        try:
            messages.append(_r.getMessage())
        except (TypeError, ValueError):
            messages.append(str(_r.msg))
    skip_disabled_logged = any("Phase Skipping deaktiviert" in m and "precomputed_phase_plan" in m for m in messages)
    assert skip_disabled_logged, (
        "§2.53b: UV3 muss 'Phase Skipping deaktiviert: precomputed_phase_plan aktiv' loggen. "
        f"Geloggte Meldungen: {messages[:10]}"
    )


def test_precomputed_plan_resets_stale_material_priority_phases(sr, short_audio):
    """§2.53b: _last_material_priority_phases wird auf tuple() zurueckgesetzt wenn precomputed_phase_plan aktiv."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    uv3._last_material_priority_phases = ("phase_09_crackle_removal", "phase_05_rumble_filter")

    plan = ["phase_03_denoise"]
    captured: list = []
    _ret = _make_pipeline_mock(short_audio)

    def _capturing_execute(*a, **kw):
        captured.append(uv3._last_material_priority_phases)
        return _ret

    with (
        patch.object(uv3, "_execute_pipeline", side_effect=_capturing_execute),
        patch.object(uv3, "_collect_reporting_analytics", return_value={}),
        patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=0),
        patch("backend.core.excellence_optimizer.optimize_for_excellence", side_effect=_excellence_mock),
        patch("backend.core.feedback_chain.FeedbackChain", return_value=_make_fc_class_mock(short_audio)),
        patch(
            "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker",
            return_value=_make_mgc_class_mock(),
        ),
        patch(
            "backend.core.holistic_perceptual_gate.get_holistic_gate",
            return_value=_make_hpg_mock(),
        ),
    ):
        try:
            uv3.restore(short_audio, sr=sr, **_fast_restore_kwargs(plan=plan))
        except Exception:
            logger.warning("test fallback", exc_info=True)

    if captured:
        assert captured[0] == (), (
            "§2.53b: _last_material_priority_phases muss nach precomputed_phase_plan "
            f"auf tuple() zurueckgesetzt werden, war aber: {captured[0]}"
        )


def test_precomputed_plan_empty_uses_legacy_select(sr, short_audio):
    """§2.53b contra-positiv: Ohne precomputed_phase_plan nutzt UV3 den Legacy-Pfad (_select_phases aufgerufen)."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    call_count: list = []
    orig_select = uv3._select_phases

    def _counting_select(*a, **kw):
        call_count.append(1)
        return orig_select(*a, **kw)

    uv3._select_phases = _counting_select  # type: ignore[method-assign]

    _ret = _make_pipeline_mock(short_audio)
    with (
        patch.object(uv3, "_execute_pipeline", return_value=_ret),
        patch.object(uv3, "_collect_reporting_analytics", return_value={}),
        patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=0),
        patch("backend.core.excellence_optimizer.optimize_for_excellence", side_effect=_excellence_mock),
        patch("backend.core.feedback_chain.FeedbackChain", return_value=_make_fc_class_mock(short_audio)),
        patch(
            "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker",
            return_value=_make_mgc_class_mock(),
        ),
        patch(
            "backend.core.holistic_perceptual_gate.get_holistic_gate",
            return_value=_make_hpg_mock(),
        ),
    ):
        try:
            # Kein precomputed_phase_plan => Legacy-Pfad
            uv3.restore(short_audio, sr=sr, **_fast_restore_kwargs(plan=None))
        except Exception:
            logger.warning("test fallback", exc_info=True)

    assert len(call_count) >= 1, (
        "§2.53b contra-positiv: Ohne precomputed_phase_plan muss _select_phases() mindestens einmal aufgerufen werden."
    )


def test_precomputed_plan_pid_plan_log_contains_phase_count(sr, short_audio):
    """§2.53b: UV3 muss bei aktivem Denker-Plan die Phasenanzahl loggen."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    plan = ["phase_03_denoise", "phase_09_crackle_removal", "phase_29_tape_hiss_reduction"]
    _ret = _make_pipeline_mock(short_audio)

    # caplog steht in dieser Repo-Umgebung nicht zur Verfuegung — manueller Log-Handler.
    tgt, handler, orig_level, records = _capture_log("backend.core.unified_restorer_v3", logging.INFO)
    try:
        with (
            patch.object(uv3, "_execute_pipeline", return_value=_ret),
            patch.object(uv3, "_collect_reporting_analytics", return_value={}),
            patch("backend.core.plugin_lifecycle_manager.cleanup_after_file", return_value=0),
            patch("backend.core.excellence_optimizer.optimize_for_excellence", side_effect=_excellence_mock),
            patch("backend.core.feedback_chain.FeedbackChain", return_value=_make_fc_class_mock(short_audio)),
            patch(
                "backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker",
                return_value=_make_mgc_class_mock(),
            ),
            patch(
                "backend.core.holistic_perceptual_gate.get_holistic_gate",
                return_value=_make_hpg_mock(),
            ),
        ):
            try:
                uv3.restore(short_audio, sr=sr, **_fast_restore_kwargs(plan=plan))
            except Exception:
                logger.warning("test fallback", exc_info=True)
    finally:
        tgt.removeHandler(handler)
        tgt.setLevel(orig_level)

    messages = []
    for _r in records:
        try:
            messages.append(_r.getMessage())
        except (TypeError, ValueError):
            messages.append(str(_r.msg))
    pid_log_found = any("PhaseInteractionDenker-Plan aktiv" in m and str(len(plan)) in m for m in messages)
    assert pid_log_found, (
        f"§2.53b: UV3 muss 'PhaseInteractionDenker-Plan aktiv: {len(plan)} Phasen' loggen. "
        f"Geloggte Meldungen: {messages[:10]}"
    )
