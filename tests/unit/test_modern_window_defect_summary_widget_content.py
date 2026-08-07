from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QLabel

from Aurik10.ui.modern_window import ModernMainWindow

_APP = QApplication.instance() or QApplication([])


def _mk_dummy_window() -> SimpleNamespace:
    _ = _APP

    calls: dict[str, str] = {}

    def _apply_status_text_style(tone: str) -> None:
        calls["tone"] = tone

    return SimpleNamespace(
        defect_summary_label=QLabel(),
        defect_count_live_label=QLabel(),
        status_text=QLabel(),
        _rest_audio=None,
        _rest_sr=48_000,
        _apply_status_text_style=_apply_status_text_style,
        _update_ab_player_state=lambda: None,
        _update_waveform=lambda _a, _b: None,
        _calls=calls,
    )


@pytest.mark.unit
def test_defect_summary_success_text_is_rendered() -> None:
    w = _mk_dummy_window()
    rr = SimpleNamespace(winning_variant=None)

    ModernMainWindow._apply_quality_defect_summary_and_footer(
        w,  # type: ignore[arg-type]
        restoration_result=rr,
        degradation_status="ok",
        fail_reason="",
        mos_est=4.2,
        quality_after_score=92.0,
        top_causal_cause="",
        causal_conf=0.0,
        quality_before_score=80.0,
        quality_delta=12.0,
        delta_snr=4.2,
        phases_exec_count=12,
        phases_skip_count=3,
        musical_violations=[],
        ceiling_reached=False,
        feedback_retries=1,
        primary_error_code="",
    )

    txt = w.defect_summary_label.text()
    assert "Restaurierung erfolgreich abgeschlossen" in txt
    assert "Klangcharakter und Originalbalance" in txt
    assert "12 Verarbeitungsschritte ausgeführt" in txt


@pytest.mark.unit
def test_defect_summary_passthrough_text_is_rendered() -> None:
    w = _mk_dummy_window()
    rr = SimpleNamespace(winning_variant="clean_digital_pass_through")

    ModernMainWindow._apply_quality_defect_summary_and_footer(
        w,  # type: ignore[arg-type]
        restoration_result=rr,
        degradation_status="ok",
        fail_reason="",
        mos_est=4.8,
        quality_after_score=0.0,
        top_causal_cause="",
        causal_conf=0.0,
        quality_before_score=0.0,
        quality_delta=0.0,
        delta_snr=0.0,
        phases_exec_count=0,
        phases_skip_count=0,
        musical_violations=[],
        ceiling_reached=False,
        feedback_retries=0,
        primary_error_code="",
    )

    txt = w.defect_summary_label.text()
    assert "Saubere Quelle" in txt
    assert "kein Eingriff nötig" in txt
    assert "Overprocessing-Schutz" in txt


@pytest.mark.unit
def test_degraded_status_updates_footer_with_error_code() -> None:
    w = _mk_dummy_window()
    rr = SimpleNamespace(winning_variant=None)

    ModernMainWindow._apply_quality_defect_summary_and_footer(
        w,  # type: ignore[arg-type]
        restoration_result=rr,
        degradation_status="degraded",
        fail_reason="Artifact-Freedom unterschritten",
        mos_est=2.7,
        quality_after_score=55.0,
        top_causal_cause="",
        causal_conf=0.0,
        quality_before_score=60.0,
        quality_delta=-5.0,
        delta_snr=-1.1,
        phases_exec_count=8,
        phases_skip_count=0,
        musical_violations=[],
        ceiling_reached=False,
        feedback_retries=0,
        primary_error_code="AFG_VETO",
    )

    assert w._calls.get("tone") == "warning"
    status_txt = w.status_text.text()
    assert "Verarbeitung mit Einschränkungen" in status_txt
    assert "AFG_VETO" in status_txt
