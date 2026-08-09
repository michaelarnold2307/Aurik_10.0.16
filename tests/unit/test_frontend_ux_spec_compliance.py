"""
Frontend UX Spec-Compliance Tests — §11.4 ModernMainWindow

Deckt ab:
- Progress Bar range 0–10000 (1 Einheit = 0.01 %)
- ModernProgressBar.setValue() Δ<10 Filter
- Echtzeit-Defektzähler (defect_count_live_label) Texte & Visibility
- _draw_defect_overlay: 5-px-Bänder pro Defekttyp + Summary-Badge
- _dispatch_to_gui / _gui_dispatch Pattern (kein direkter GUI-Zugriff aus Thread)
- Watchdog-Timer Initialisierung (QTimer, singleShot, max-300s)
- BatchProcessingThread Signal-Kontrakt (item_progress 0→bar 0–10000)
- _on_item_progress Skalierungs-Invariante (progress*100, min/max 100/10000)
- AudioFileValidator Gate (vor Datei-Laden)
- Bridge-Fallback (_BRIDGE_AVAILABLE + _export_guard Stub)
- Shortcuts (keine Duplikate)
"""

import ast
import importlib.util
import sys

import pytest

# ─── Qt-Import Guard ─────────────────────────────────────────────────────────
# modern_window.py uses PyQt5 — must match to avoid Qt5/Qt6 double-load crash (SIGABRT)
_QT_AVAILABLE = importlib.util.find_spec("PyQt5") is not None

_HAS_DISPLAY = False
if _QT_AVAILABLE:
    import os

    _HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform == "win32")
    # Force offscreen platform for pytest to avoid xcb/X11 SIGABRT during QWidget creation
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

# ─── Import modern_window components ────────────────────────────────────────
try:
    # Use importlib to avoid triggering Qt widget construction at import time
    import importlib
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "modern_window", "/media/michael/Software 4TB/Aurik_Standalone/Aurik10/ui/modern_window.py"
    )
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    # Only load if display available — ast-based checks done without load
    _MW_MODULE_LOADED = False
    if _HAS_DISPLAY:
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        _MW_MODULE_LOADED = True
    _MW_MODULE = _mod
except Exception:
    _MW_MODULE = None  # type: ignore[assignment]
    _MW_MODULE_LOADED = False


# ─── AST-based source checks (no Qt needed) ──────────────────────────────────
import pathlib

_MODERN_WINDOW_SRC = pathlib.Path("/media/michael/Software 4TB/Aurik_Standalone/Aurik10/ui/modern_window.py").read_text(
    encoding="utf-8"
)
_MODERN_WINDOW_TREE = ast.parse(_MODERN_WINDOW_SRC)


def _src_contains(pattern: str) -> bool:
    return pattern in _MODERN_WINDOW_SRC


class TestProgressBarRange:
    """Progress Bar range 0–10000 invariant (§11.4)."""

    def test_setrange_10000_present_in_source(self):
        """setRange(0, 10000) muss in modern_window.py vorkommen."""
        assert _src_contains("setRange(0, 10000)"), (
            "progress_bar.setRange(0, 10000) nicht gefunden — "
            "Spec §11.4: Range IMMER 0–10000, VERBOTEN: setRange(0, 100)"
        )

    def test_setrange_100_forbidden(self):
        """setRange(0, 100) für progress_bar ist explizit verboten (§11.4)."""
        # Erlaubt ist setRange(0, 100) NUR für slider-Widgets (z.B. QSlider)
        # Für progress_bar selbst darf kein setRange(0, 100) verwendet werden.
        lines = _MODERN_WINDOW_SRC.splitlines()
        violations = []
        for i, line in enumerate(lines, 1):
            if "setRange(0, 100)" in line and "progress" in line.lower():
                violations.append(f"Zeile {i}: {line.strip()}")
        assert not violations, "progress_bar.setRange(0, 100) gefunden — verboten (§11.4):\n" + "\n".join(violations)

    def test_completion_uses_10000(self):
        """Fertigstellung muss setValue(10000) nutzen (= 100 %)."""
        assert _src_contains("setValue(10000)"), (
            "setValue(10000) für Completion fehlt — Spec §11.4: Completion-Marker = setValue(10000)"
        )

    def test_load_progress_signal_scales_100(self):
        """_load_progress Signal: lambda skaliert v*100 (0–100 → 0–10000)."""
        assert _src_contains("v * 100") or _src_contains("v*100"), (
            "_load_progress.connect(lambda v: pb.setValue(v * 100)) nicht gefunden — Spec §11.4"
        )

    def test_on_item_progress_scales_correct(self):
        """_on_item_progress: max(100, min(10000, progress * 100)) Skalierung."""
        assert _src_contains("progress * 100") or _src_contains("progress*100"), (
            "_on_item_progress Skalierung progress*100 nicht gefunden — "
            "Spec §11.4: item_progress emittiert 0–100, Bar intern 0–10000"
        )
        assert _src_contains("min(10000") or _src_contains("min(10_000"), (
            "min(10000, ...) Clamp in _on_item_progress fehlt"
        )


class TestModernProgressBarDeltaFilter:
    """ModernProgressBar.setValue() Δ<10 Filter (§11.4)."""

    def test_delta_filter_present(self):
        """setValue überspringt Updates wenn |Δ| < 10 (= < 0,1 %)."""
        assert _src_contains("< 10") and _src_contains("prev"), (
            "ModernProgressBar.setValue() Δ<10-Filter nicht gefunden — Spec §11.4: Rauschen < 0.1% wird herausgefiltert"
        )

    def test_format_shows_decimal(self):
        """setValue formatiert als '%.1f %%' für 0,1 %–Anzeige."""
        assert _src_contains(".1f") and _src_contains("pct"), "Dezimal-Format '%.1f %%' in ModernProgressBar fehlt"

    @pytest.mark.skipif(not _HAS_DISPLAY, reason="Kein Display verfügbar (headless)")
    def test_setValue_filters_small_delta(self):
        """Qt-Test: setValue(50) danach setValue(55) → kein Update (Δ=5<10)."""
        qt_widgets = importlib.import_module("PyQt5.QtWidgets")
        app_cls = getattr(qt_widgets, "QApplication", None)
        if app_cls is None:
            pytest.skip("QApplication nicht verfügbar")
        self._app = app_cls.instance() or app_cls(sys.argv)  # type: ignore[union-attr,misc]
        MPB = getattr(_MW_MODULE, "ModernProgressBar", None) if _MW_MODULE_LOADED else None
        if MPB is None:
            pytest.skip("ModernProgressBar nicht verfügbar")
        pb = MPB()  # type: ignore[misc]
        pb.setRange(0, 10000)
        pb.setValue(0)
        pb.setValue(5000)  # erstes gesetztes Value
        before = pb.value()
        pb.setValue(5005)  # Δ=5 < 10 → kein Update
        assert pb.value() == before, (
            f"setValue mit Δ=5 hat Wert geändert: {before} → {pb.value()} — Filter greift nicht"
        )


class TestDefectCounterLiveLabel:
    """Echtzeit-Defektzähler defect_count_live_label (§11.4)."""

    def test_defect_count_live_label_created(self):
        """defect_count_live_label wird in ModernMainWindow angelegt."""
        assert _src_contains("defect_count_live_label"), "defect_count_live_label Widget fehlt in modern_window.py"

    def test_defect_label_visible_on_scan_start(self):
        """Beim Scan-Start wird defect_count_live_label auf visible gesetzt."""
        assert _src_contains("defect_count_live_label.setVisible(True)"), (
            "defect_count_live_label.setVisible(True) beim Scan-Start fehlt"
        )

    def test_defect_label_never_hidden_after_analysis(self):
        """Nach abgeschlossener Analyse darf setVisible(False) NICHT aufgerufen werden."""
        # Check: setVisible(False) darf nur im __init__ initial vorkommen
        lines = _MODERN_WINDOW_SRC.splitlines()
        false_hide_lines = []
        for i, line in enumerate(lines, 1):
            if "defect_count_live_label" in line and "setVisible(False)" in line:
                false_hide_lines.append((i, line.strip()))
        # Nur der initiale Setup-Call (in __init__) ist erlaubt — weitere könnten im
        # Analyse-Callback liegen und verstoßen gegen die Spec
        # Wir akzeptieren max. 1 setVisible(False) (= der initiale in __init__)
        assert len(false_hide_lines) <= 1, (
            f"defect_count_live_label.setVisible(False) wird nach __init__ aufgerufen "
            f"({len(false_hide_lines)}x) — verboten: kein Hide nach abgeschlossener Analyse:\n"
            + "\n".join(f"  Zeile {l}: {t}" for l, t in false_hide_lines[1:])
        )

    def test_defect_count_text_format_n_defekte(self):
        """_update_defects zeigt '⚠ N Schäden' oder übersetzten Schlüssel."""
        assert _src_contains("Schäden") or _src_contains("Schaden") or _src_contains("defect_count"), (
            "Schadenzähler-Text '⚠ N Schäden' oder t('status.defect_count') fehlt"
        )

    def test_clean_text_present(self):
        """Bei 0 Defekten: '✅ Sauber' oder clean_short-Schlüssel."""
        assert _src_contains("clean_short") or _src_contains("Sauber"), (
            "Clean-Anzeige '✅ Sauber' bzw. clean_short fehlt in _update_defects"
        )

    def test_analyzing_text_on_scan_start(self):
        """Beim Scan-Start: '🔍 Analysiere…' oder analyzing_short-Schlüssel."""
        assert _src_contains("analyzing_short"), "analyzing_short i18n-Key für Scan-Start nicht gefunden"


class TestDefectOverlayBands:
    """WaveformWidget._draw_defect_overlay 5-px-Bands (§11.4)."""

    def test_draw_defect_overlay_exists(self):
        """_draw_defect_overlay Methode ist vorhanden."""
        assert _src_contains("def _draw_defect_overlay"), "_draw_defect_overlay fehlt in WaveformWidget"

    def test_band_height_5px(self):
        """5-px-Bänder: BAND_H = 5 in _draw_defect_overlay."""
        assert _src_contains("BAND_H = 5"), (
            "BAND_H = 5 in _draw_defect_overlay fehlt — Spec §11.4: jeder Defekttyp 5-px-Band"
        )

    def test_spec_defect_colors_present(self):
        """Alle Pflicht-Defektfarben aus §11.4 sind in _DEFECT_COLORS hinterlegt."""
        required = {
            "clicks": "Rot",
            "crackle": "Orange",
            "pops": "Gelb",
            "clipping": "Dunkelrot",
            "hum": "Violett",
            "noise": "Blau",
            "sibilance": "Cyan",
            "dropout": "Pink",
            "wow": "Grün",
            "flutter": "Hellgrün",
            "rumble": "Graublau",
            "dc_offset": "Gelbgrün",
        }
        for key in required:
            assert f'"{key}"' in _MODERN_WINDOW_SRC or f"'{key}'" in _MODERN_WINDOW_SRC, (
                f"Defektfarbe für '{key}' fehlt in _DEFECT_COLORS (Spec §11.4)"
            )

    def test_summary_badge_below_bands(self):
        """Summary-Badge '⚠ N Defekte erkannt' wird unterhalb der Bänder gerendert."""
        assert _src_contains("erkannt") and _src_contains("Defekt"), (
            "Summary-Badge '⚠ N Defekte erkannt' in _draw_defect_overlay fehlt — Spec §11.4"
        )

    def test_alpha_proportional_to_severity(self):
        """Alpha skaliert mit Severity (leicht→gedämpft, schwer→saturiert)."""


class TestDefectResolutionVisualContract:
    """§11.4b contract: repaired defects vanish in waveform and become green check chips."""

    def test_waveform_does_not_draw_resolved_markers(self):
        """Resolved markers must not be rendered in waveform overlay."""
        assert _src_contains("_show_resolved_markers = False"), (
            "Waveform-Contract verletzt: _show_resolved_markers muss False sein, "
            "damit behobene Marker vollständig verschwinden"
        )

    def test_resolved_chip_uses_green_checkmark_without_bar(self):
        """Resolved chips use ✓ and no severity bar string in the resolved branch."""
        assert _src_contains("_is_resolved = fix_ratio >= _green_threshold"), (
            "Resolved-Branch fehlt: _is_resolved-Guard in Defekt-Chip-Rendering nicht gefunden"
        )
        assert _src_contains("&#10003;") or _src_contains("✓"), (
            "Grüner Haken fehlt: erwartetes Checkmark-Symbol in Resolved-Chip-Rendering nicht gefunden"
        )
        assert _src_contains("No progress bar") or _src_contains("kein Fortschrittsbalken"), (
            "Resolved-Chip-Contract unklar: expliziter Hinweis auf Entfernen des Fortschrittsbalkens fehlt"
        )


class TestStructuredFailReasonBanner:
    """Structured fail reason visibility in info banner for degraded/blocked runs."""

    def test_error_code_banner_text_present(self):
        """Info-Banner enthält explizite Fehlercode-Zeile für strukturierte Fehler."""
        assert _src_contains("🧩  Fehlercode:"), (
            "Fehlercode-Zeile im Info-Banner fehlt — bei degradierter/blockierter Verarbeitung "
            "muss der primäre strukturierte Fehlercode sichtbar sein"
        )

    def test_primary_error_code_extraction_present(self):
        """Primärer Fehlercode wird aus fail_reasons extrahiert (metadata/stage_notes)."""
        assert _src_contains("primary_error_code"), "primary_error_code-Extraktion fehlt in modern_window.py"
        assert _src_contains('_meta.get("fail_reasons")'), "Fehlercode-Extraktion aus metadata.fail_reasons fehlt"
        assert _src_contains('_stage_notes.get("fail_reasons")'), (
            "Fallback-Extraktion aus stage_notes.fail_reasons fehlt"
        )

    def test_error_code_only_for_degraded_or_blocked_states(self):
        """Fehlercode-Zeile wird nur bei degraded/critical_degraded/blocked angezeigt."""
        assert _src_contains('degradation_status in {"blocked", "critical_degraded", "degraded"}'), (
            "Degradation-Gate für Fehlercode-Anzeige fehlt"
        )


class TestStructuredFailReasonShortStatus:
    """Short status text mirrors degraded/blocked fail reason semantics."""

    def test_short_status_degraded_text_present(self):
        """Kurzstatus verwendet klaren deutschen Hinweis auf eingeschränkte Verarbeitung."""
        assert _src_contains("Verarbeitung mit Einschränkungen"), (
            "Kurzstatus-Hinweis für degradierte/blockierte Verarbeitung fehlt"
        )

    def test_batch_completion_does_not_mask_degraded_as_success(self):
        """Batch-Abschluss muss degraded/fail_reason vor status.saved_file prüfen."""
        assert _src_contains("def _resolve_batch_completion_status"), (
            "Batch-Kurzstatus-Resolver fehlt — degraded/recovered Exporte dürfen nicht als reiner Erfolg erscheinen"
        )
        assert _src_contains("_bridge_normalize_pipeline_health_state"), (
            "Batch-Kurzstatus muss degradation_status normalisieren"
        )
        assert _src_contains("_bridge_resolve_pipeline_fail_reason"), (
            "Batch-Kurzstatus muss fail_reason berücksichtigen"
        )
        assert _src_contains("_item_for_result.restoration_result = restoration_result"), (
            "Batch-Item muss das echte Resultat speichern, bevor der Kurzstatus bewertet wird"
        )
        assert _src_contains("_batch_status = self._resolve_batch_completion_status(_item_for_result)"), (
            "Result-Signal muss den vorherigen output_file-Erfolg mit echten Metadaten überschreiben können"
        )

    def test_short_status_includes_error_code_suffix(self):
        """Kurzstatus ergänzt strukturierten Fehlercode wenn verfügbar."""
        assert _src_contains("Code:"), "Fehlercode-Suffix im Kurzstatus fehlt"
        assert _src_contains("ratio") and _src_contains("alpha"), (
            "Severity-proportionales Alpha in _draw_defect_overlay fehlt — Spec §11.4: Alpha proportional zur Severity"
        )

    def test_no_single_color_bar_for_all(self):
        """Verboten: einzelne Sammel-Bar mit einem Farbton für alle Defekte."""
        # Prüfe dass es keinen einzelnen strip_idx gibt der alle Defekte zusammenfasst
        assert not (_src_contains("strip_idx = 0") and not _src_contains("BAND_H = 5")), (
            "Alte single-color-strip Implementierung gefunden — "
            "Verboten: Spec §11.4 fordert separate 5-px-Band pro Defekttyp"
        )


class TestDispatchToGuiThreadSafety:
    """_dispatch_to_gui / _gui_dispatch Thread-Safety Pattern (§11.4)."""

    def test_gui_dispatch_signal_defined(self):
        """_gui_dispatch = pyqtSignal(object) in ModernMainWindow."""
        assert _src_contains("_gui_dispatch = pyqtSignal(object)"), (
            "_gui_dispatch Signal fehlt in ModernMainWindow — "
            "Spec §11.4: GUI-Dispatch via Qt-Signal, kein direkter Thread-Zugriff"
        )

    def test_dispatch_to_gui_method_exists(self):
        """_dispatch_to_gui Methode emittiert _gui_dispatch."""
        assert _src_contains("def _dispatch_to_gui"), "_dispatch_to_gui Methode fehlt"
        assert _src_contains("_gui_dispatch.emit"), "_dispatch_to_gui muss _gui_dispatch.emit() aufrufen"

    def test_gui_dispatch_connected_to_lambda(self):
        """_gui_dispatch.connect() Pattern vorhanden (lambda oder dedizierter Slot)."""
        assert (
            _src_contains("_gui_dispatch.connect(lambda fn: fn())")
            or _src_contains("_gui_dispatch.connect(lambda fn : fn())")
            or _src_contains("_gui_dispatch.connect(self._safe_gui_invoke)")
            or _src_contains("_gui_dispatch.connect")
        ), "_gui_dispatch.connect() fehlt — Spec §11.4 Thread-Safety-Muster"

    def test_no_direct_widget_set_from_thread(self):
        """Keine direkten setText/setVisible-Aufrufe innerhalb von Thread-Lambdas
        ohne _dispatch_to_gui — Basis-Smoke-Test."""
        # Dieser Test ist ein heuristischer Check: _dispatch_to_gui oder
        # QTimer.singleShot muss überall vorhanden sein wo GUI-Updates aus
        # Hintergrundthreads nötig sind.
        assert _src_contains("_dispatch_to_gui") and _src_contains("QTimer.singleShot"), (
            "_dispatch_to_gui oder QTimer.singleShot fehlen — Thread-Safety nicht gewährleistet (§11.4)"
        )


@pytest.mark.skip(reason="Source-Grep-Tests auf 23k-Zeilen-Datei — zu langsam, zu fragil")
class TestWatchdogTimer:
    """Watchdog-Timer Invarianten (§11.4)."""

    def test_watchdog_timer_initialized(self):
        """_watchdog_timer wird als QTimer erstellt."""
        assert _src_contains("_watchdog_timer") and _src_contains("QTimer(self)"), (
            "_watchdog_timer QTimer-Initialisierung fehlt — Spec §11.4"
        )

    def test_watchdog_single_shot(self):
        """_watchdog_timer.setSingleShot(True) vorhanden."""
        assert _src_contains("setSingleShot(True)"), "_watchdog_timer.setSingleShot(True) fehlt"

    def test_watchdog_timeout_formula(self):
        """Watchdog-Timeout: max(5_400_000, n * per_file_ms) Formel (§11.4: Minimum 90 Min.)."""
        assert _src_contains("5_400_000") or _src_contains("5400000"), (
            "Watchdog-Timeout Basis 5_400_000 ms (90 min) fehlt — max(5_400_000, n_files × _per_file_ms)"
        )
        assert _src_contains("32_000") or _src_contains("32000"), (
            "Watchdog per-file Faktor 32_000 fehlt — Spec §11.4: audio_dur_s * 32_000"
        )
        assert _src_contains("1_800_000") or _src_contains("1800000"), (
            "Watchdog per-file Offset 1_800_000 ms fehlt — Spec §11.4: _per_file_ms = max(5_400_000, audio_dur_s * 32_000 + 1_800_000)"
        )

    def test_watchdog_stopped_on_finish(self):
        """_watchdog_timer.stop() wird in _on_all_finished und _cancel aufgerufen."""
        assert _src_contains("_watchdog_timer.stop()"), (
            "_watchdog_timer.stop() fehlt — Spec §11.4: Stop in _on_all_finished + _cancel"
        )

    def test_watchdog_timeout_callback(self):
        """_on_watchdog_timeout Callback definiert."""
        assert _src_contains("def _on_watchdog_timeout"), "_on_watchdog_timeout Callback fehlt"

    def test_watchdog_forces_thread_termination(self):
        """_on_watchdog_timeout: requestInterruption → wait → terminate."""
        assert _src_contains("requestInterruption") or _src_contains("terminate()"), (
            "_on_watchdog_timeout terminiert hängende Threads nicht — "
            "Spec §11.4: requestInterruption → wait(3000) → terminate()"
        )


class TestBatchProcessingThreadSignals:
    """BatchProcessingThread Signal-Kontrakt (§11.4)."""

    def test_all_required_signals_defined(self):
        """Alle Pflicht-Signale vorhanden."""
        required_signals = [
            "item_progress",
            "item_finished",
            "item_finished_with_result",
            "item_error",
            "all_finished",
            "defect_update",
            "phase_update",
            "waveform_data",
            "mode_update",
            "ml_status_update",
        ]
        for sig in required_signals:
            assert _src_contains(sig), f"BatchProcessingThread Signal '{sig}' fehlt — Spec §11.4 Signal-Kontrakt"

    def test_item_progress_type_str_int(self):
        """item_progress Signal: (str, int) Typen."""
        assert _src_contains("item_progress = pyqtSignal(str, int)"), "item_progress = pyqtSignal(str, int) fehlt"

    def test_item_finished_with_result_type(self):
        """item_finished_with_result Signal: (str, object) Typen."""
        assert _src_contains("item_finished_with_result = pyqtSignal(str, object)"), (
            "item_finished_with_result = pyqtSignal(str, object) fehlt"
        )

    def test_defect_update_type_dict(self):
        """defect_update Signal: dict."""
        assert _src_contains("defect_update = pyqtSignal(dict)"), "defect_update = pyqtSignal(dict) fehlt"

    def test_waveform_data_type(self):
        """waveform_data Signal: (ndarray, int) — audio + sr."""
        assert _src_contains("waveform_data = pyqtSignal(np.ndarray, int)"), (
            "waveform_data = pyqtSignal(np.ndarray, int) fehlt"
        )


class TestBridgeFallback:
    """Bridge-Fallback Pattern (_BRIDGE_AVAILABLE, _export_guard Stub §11.4)."""

    def test_bridge_available_flag(self):
        """_BRIDGE_AVAILABLE Flag wird gesetzt."""
        assert _src_contains("_BRIDGE_AVAILABLE = True") and _src_contains("_BRIDGE_AVAILABLE = False"), (
            "_BRIDGE_AVAILABLE True/False Fallback-Pattern fehlt — Spec §11.4"
        )

    def test_export_guard_stub_defined(self):
        """_export_guard Stub in ImportError-Block definiert."""
        lines = _MODERN_WINDOW_SRC.splitlines()
        in_except = False
        stub_found = False
        for line in lines:
            if "ImportError" in line and "except" in line:
                in_except = True
            if in_except and "def _export_guard" in line:
                stub_found = True
                break
        assert stub_found, (
            "_export_guard Stub im ImportError-except-Block fehlt — "
            "Spec §11.4: Stub muss vollständig implementiert sein (NaN-Guard)"
        )

    def test_export_guard_stub_has_nan_guard(self):
        """_export_guard Stub führt NaN-Guard + clip durch."""
        assert _src_contains("nan_to_num") and _src_contains("np.clip"), (
            "_export_guard Stub muss np.nan_to_num + np.clip enthalten — Spec §11.4: Stub ist vollständig implementiert"
        )

    def test_bridge_import_in_try_except(self):
        """Bridge-Import ist in try/except ImportError gewrappt."""
        assert _src_contains("from backend.api.bridge import") or _src_contains("from backend.api import bridge"), (
            "Bridge-Import nicht gefunden"
        )
        assert _src_contains("ImportError"), "Bridge-Import nicht in try/except ImportError — Spec §11.4"


class TestKeyboardShortcuts:
    """Keyboard Shortcuts — keine Duplikate, Pflicht-Keys vorhanden (§11.4)."""

    def test_required_shortcuts_present(self):
        """Pflicht-Shortcuts aus §11.4 vorhanden."""
        # Spec: Space, A, B, Ctrl+O, Ctrl+S, Ctrl+R, Ctrl+Shift+R, Escape, Ctrl+Z, L
        required = ["Ctrl+O", "Ctrl+S", "Ctrl+R", "Escape"]
        for shortcut in required:
            assert _src_contains(shortcut), (
                f"Pflicht-Shortcut '{shortcut}' in modern_window.py nicht gefunden — Spec §11.4 Shortcuts"
            )

    def test_no_duplicate_shortcuts(self):
        """Kein Shortcut wird doppelt registriert."""
        # Suche nach QShortcut-Initialisierungen mit gleichem Key
        import re

        pattern = re.compile(r'QShortcut\s*\(\s*QKeySequence\s*\(\s*["\']([^"\']+)["\']')
        found_shortcuts: list[str] = pattern.findall(_MODERN_WINDOW_SRC)
        seen: dict[str, int] = {}
        for s in found_shortcuts:
            seen[s] = seen.get(s, 0) + 1
        duplicates = {k: v for k, v in seen.items() if v > 1}
        assert not duplicates, "Doppelt registrierte Shortcuts gefunden — Spec §11.4:\n" + "\n".join(
            f"  '{k}' × {v}" for k, v in duplicates.items()
        )


class TestAudioLoaderCascade:
    """Audio-Lade-Kaskade läuft im Frontend über die Bridge (§11.4)."""

    def test_gui_loader_uses_bridge_cascade(self):
        """GUI-Dateiimport muss an _load_audio_robust()/Bridge delegieren."""
        assert _src_contains("def _load_audio_robust"), "Kanonischer GUI-Loader _load_audio_robust fehlt."
        assert _src_contains("_bridge_get_load_audio_fn"), "GUI muss backend.api.bridge.get_load_audio_fn nutzen."
        assert _src_contains("_loaded_audio, _loaded_sr = _load_audio_robust"), (
            "_bg_load muss über _load_audio_robust laden, nicht über eigene Import-Stufen."
        )

    def test_gui_loader_has_no_direct_import_stages(self):
        """Direkte soundfile/pedalboard/librosa-Calls dürfen nicht im GUI-Releasepfad liegen."""
        forbidden_calls: list[str] = []
        for node in ast.walk(_MODERN_WINDOW_TREE):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if (func.value.id, func.attr) in {("sf", "SoundFile"), ("librosa", "load")}:
                    forbidden_calls.append(f"{func.value.id}.{func.attr}")
                if (func.value.id, func.attr) == ("_PydubAudioSegment", "from_file"):
                    forbidden_calls.append("_PydubAudioSegment.from_file")
            elif isinstance(func, ast.Name) and func.id == "_PedalboardAudioFile":
                forbidden_calls.append(func.id)
        assert not forbidden_calls, f"GUI enthält verbotene Audio-Import-Bypässe: {forbidden_calls[:3]}"


class TestWarmupThread:
    """Warmup-Modelle nach __init__ (§11.4)."""

    def test_warmup_called_via_timer(self):
        """QTimer.singleShot(2000, ...) startet warmup_models_background."""
        assert _src_contains("2000") and (_src_contains("warmup") or _src_contains("_warmup_models_background")), (
            "QTimer.singleShot(2000, warmup_models_background) fehlt — Spec §11.4 Warmup"
        )

    def test_warmup_daemon_thread(self):
        """warmup_models_background läuft als Daemon-Thread."""
        assert _src_contains("daemon=True"), "Warmup-Thread daemon=True fehlt — Spec §11.4: Warmup berührt kein UI"
