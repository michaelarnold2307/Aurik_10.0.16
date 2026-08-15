from pathlib import Path

import pytest

GUI_FILE = Path("Aurik10/ui/modern_window.py")
I18N_FILE = Path("Aurik10/i18n/__init__.py")
PROGNOSE_FILE = Path("Aurik10/ui/song_prognose_widget.py")
DEFECT_SCANNER_FILE = Path("backend/core/defect_scanner.py")


@pytest.mark.unit
def test_ui_heartbeat_reassures_during_long_processing_phases() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert 't("status.processing_reassure_analysis")' in src
    assert 't("status.processing_reassure_long_phase")' in src
    assert 't("status.processing_reassure_finalize")' in src
    assert "_time_since_cb >= 8.0" in src
    assert '_user_phase_text = _expl.lstrip(" ·").strip() if _expl else ""' in src
    assert '"Jetzt: {_base}' in src
    assert "Danach: {_next_hint}" in src


def test_ui_status_and_phase_labels_use_eliding_labels() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "class _ElidingLabel(QLabel):" in src
    assert "class _WrappingStatusLabel(QLabel):" in src
    assert 'self._phase_step_label = _ElidingLabel("")' in src
    assert 'self.status_text = _WrappingStatusLabel(t("status.ready"))' in src
    assert "self.status_text.setWordWrap(True)" in src
    assert 'self.stats_label = _ElidingLabel(t("status.stats", pending=0, completed=0, failed=0))' in src


def test_ui_top_row_refresh_prevents_collisions_on_narrow_widths() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "def _refresh_top_info_row(self, window_width: int | None = None) -> None:" in src
    assert '_btn_cancel.setText(t("action.cancel_compact") if _compact else t("action.cancel_full"))' in src
    assert "_stats_should_hide = _very_compact and _running" in src
    assert "self._refresh_top_info_row(w)" in src
    assert "self._refresh_top_info_row()" in src


def test_ui_load_progress_switches_to_finalize_mode_near_98_percent() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "_load_progress = pyqtSignal(float)" in src
    assert "def _on_load_progress_update(self, pct: float) -> None:" in src
    assert "if _preanalysis_pending and _p >= 99.8:" in src
    assert "_bar.setRange(0, 0)" in src
    assert '"⏳ Analyse wird finalisiert …"' in src


def test_defect_scanner_emits_fine_grained_tail_progress() -> None:
    src = DEFECT_SCANNER_FILE.read_text(encoding="utf-8")
    assert 'progress_callback: Optional["Callable[[float, str], None]"]' in src
    assert 'def _prog(pct: float, name: str = "") -> None:' in src
    assert 'def _tail_tick(name: str = "") -> None:' in src
    assert "_DIGITAL_FAST_TAIL" in src
    assert "_ANALOG_DEEP_TAIL" in src
    assert "_tail_start = 93.0" in src
    assert "_tail_start = 88.0" in src
    assert "_prog(_tail_start + _tail_span * _frac, name)" in src


def test_ui_i18n_contains_reassuring_processing_messages() -> None:
    src = I18N_FILE.read_text(encoding="utf-8")
    assert '"status.processing_reassure_analysis": "Aurik prüft die Aufnahme weiter sorgfältig"' in src
    assert (
        '"status.processing_reassure_long_phase": "Rechenintensive Phase aktiv - Fortschritt läuft stabil weiter"'
        in src
    )
    assert (
        '"status.processing_reassure_finalize": "Aurik finalisiert das Ergebnis und sichert alle Qualitätsprüfungen"'
        in src
    )


def test_quality_meter_distinguishes_live_estimate_from_final_measurement() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert 'def set_measurement_state(self, state: str, detail: str = "") -> None:' in src
    assert '"Prognose"' in src
    assert '"Live-Schätzung"' in src
    assert '"Final gemessen"' in src
    assert "self.batch_thread.quality_update.connect(self._set_live_quality_mos)" in src


def test_ab_player_disabled_states_have_actionable_reasons() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "def _audio_output_ready(self) -> tuple[bool, str]:" in src
    assert "def _set_button_enabled_with_reason(button, enabled: bool" in src
    assert "Kein Audio-Ausgabegerät gefunden." in src
    assert "Original-Audio ist noch nicht geladen." in src
    assert "Restaurierter Export liegt noch nicht vor." in src
    assert "Audio-Ausgabe nicht verfügbar" in src


def test_ab_player_uses_loudness_matched_restored_playback() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "def _playback_gated_rms_db(audio: np.ndarray) -> float:" in src
    assert "def _match_playback_loudness_to_reference(" in src
    assert "loudness_match_ref=self._orig_audio" in src
    assert "Pegelmatch" in src
    assert "Export-Audio bleibt unverändert" in src


def test_defect_chips_include_level_defects_separately_from_transport_bump() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    for key, label in (
        ("amplitude_drift", "Pegeldrift"),
        ("tape_head_level_dip", "Bandkopf-Pegelabfall"),
        ("dynamic_compression_excess", "Lautstärkekompression"),
        ("nr_breathing_artifact", "Pegel-Pumpen"),
    ):
        assert f'"{key}"' in src
        assert f'"{label}"' in src

    assert '"transport_bump": "Bandhopser"' in src
    assert 'sev_amplitude_drift = _sev_opt("AMPLITUDE_DRIFT")' in src
    assert 'sev_tape_head_level_dip = _sev_opt("TAPE_HEAD_LEVEL_DIP")' in src
    assert 'sev_nr_breathing = _sev_opt("NR_BREATHING_ARTIFACT")' in src
    assert '"amplitude_drift": round(sev_amplitude_drift * 100.0, 2)' in src
    assert '"tape_head_level_dip": round(sev_tape_head_level_dip * 100.0, 2)' in src
    assert '"nr_breathing_artifact": round(sev_nr_breathing * 100.0, 2)' in src
    assert '"amplitude_drift": "level_drift"' in src
    assert '"tape_head_level_dip": "level_dip"' in src
    assert '"nr_breathing_artifact": "level_pumping"' in src
    assert 'elif effect == "level_drift":' in src
    assert 'elif effect == "level_dip":' in src
    assert 'elif effect == "level_pumping":' in src
    assert '"amplitude_drift": ("Pegeldrift", 5.0, 30.0)' in src
    assert '"tape_head_level_dip": ("Bandkopf-Pegelabfall", 5.0, 30.0)' in src
    assert '"nr_breathing_artifact": ("Pegel-Pumpen", 5.0, 30.0)' in src


def test_release_title_is_professional_not_personalized() -> None:
    src = I18N_FILE.read_text(encoding="utf-8")
    assert '"ui.app_title": "Aurik Professional v{version}"' in src
    assert "für meinen lieben Freund" not in src
    assert "my dear friend" not in src


def test_gui_has_professional_offline_system_check_surface() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "def _collect_professional_system_status(self) -> list[tuple[str, str]]:" in src
    assert "def _show_system_check_dialog(self) -> None:" in src
    assert "Offline-Betrieb" in src
    assert "Bridge-Vertrag" in src
    assert "Audio-Ausgabe" in src
    assert "Lokale Modelle" in src
    assert "Aurik Systemcheck" in src


def test_result_banner_starts_with_clear_professional_verdict() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "Haupturteil" in src
    assert "FREIGEGEBEN" in src
    assert "RECOVERED" in src
    assert "DEGRADED" in src
    assert "BLOCKIERT" in src
    assert "Export freigegeben; technische Details folgen" in src


def test_song_prognose_primary_visible_texts_are_i18n_controlled() -> None:
    src = PROGNOSE_FILE.read_text(encoding="utf-8")
    i18n = I18N_FILE.read_text(encoding="utf-8")
    assert "from Aurik10.i18n import t" in src
    for key in (
        "prognose.header",
        "prognose.section.score",
        "prognose.status.detected",
        "prognose.mode.restoration_recommended",
        "prognose.mode.studio_recommended",
    ):
        assert f't("{key}")' in src
        assert f'"{key}"' in i18n


def test_primary_gui_controls_have_accessibility_metadata() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert 'self.setAccessibleName("Klangqualitätsanzeige")' in src
    assert 'self.btn_play_original.setAccessibleName("Original anhören")' in src
    assert 'self.btn_play_restored.setAccessibleName("Restaurierte Fassung anhören")' in src
    assert 'self.btn_ab_sync.setAccessibleName("Synchronisierter A/B-Loop")' in src
    assert 'self.btn_stop_playback.setAccessibleName("Wiedergabe stoppen")' in src
    assert 'self.btn_help.setAccessibleName("Hilfe und Systemcheck")' in src


def test_gui_runtime_snapshot_centralizes_professional_state() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "class UiRuntimeSnapshot:" in src
    assert "def _capture_runtime_snapshot(self) -> UiRuntimeSnapshot:" in src
    assert "def _format_ab_source_status(snapshot: UiRuntimeSnapshot) -> str:" in src
    assert "processing_active" in src
    assert "quality_state" in src
    assert "result_output_available" in src


def test_result_banner_includes_recovery_diagnostic_center() -> None:
    src = GUI_FILE.read_text(encoding="utf-8")
    assert "def _build_recovery_diagnostic_line(" in src
    assert "Recovery-Diagnose" in src
    assert "Urteil={verdict}" in src
    assert "Pipeline={_deg}" in src
    assert "Export=vorhanden" in src
    assert "A/B={self._format_ab_source_status(_snapshot)}" in src
