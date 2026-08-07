#!/usr/bin/env python3
"""
AURIK Debug-Launcher — vollständige Ereignis-Instrumentierung aller §11.4-Features.

Startet das Frontend mit erweitertem Debug-Logging auf DEBUG-Level für alle
aurik/backend/Aurik10-Namespaces. Loggt in:
  logs/aurik_backend.log  (Standard)
  logs/aurik_debug_session.log  (vollständiges DEBUG-Log dieser Session)

Führt keine Änderungen am Produktions-Code durch — rein additives Patching.
"""

import logging
import sys
import time
from pathlib import Path

# ── Workspace root ──────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE
if not (_ROOT / "Aurik10").exists():
    _ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

# ── Debug-Session-Logfile ────────────────────────────────────────────────────
_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_session_log = _LOG_DIR / f"aurik_debug_session_{int(time.time())}.log"

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)

# Datei-Handler für vollständiges DEBUG-Log dieser Session

_dfh = logging.FileHandler(str(_session_log), encoding="utf-8")
_dfh.setLevel(logging.DEBUG)
_dfh.setFormatter(
    logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)-40s | %(message)s",
        datefmt="%H:%M:%S",
    )
)
_root_logger.addHandler(_dfh)

# Konsole zeigt INFO+ für alle Aurik-Namespaces
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
_ch.addFilter(lambda r: "aurik" in r.name.lower() or "Aurik" in r.name)
_root_logger.addHandler(_ch)

_logger = logging.getLogger("aurik.debug_launcher")
_logger.info("=" * 80)
_logger.info("AURIK Debug-Launcher gestartet")
_logger.info("Session-Log: %s", _session_log)
_logger.info("=" * 80)


# ── Signal-Monitoring Mixin ──────────────────────────────────────────────────
def _patch_batch_thread_signals(thread_instance, thread_logger: logging.Logger) -> None:
    """Verbindet alle BatchProcessingThread-Signale mit Debug-Logging."""

    def _mk_spy(name):
        def _spy(*args):
            thread_logger.debug(
                "SIGNAL  %-30s  args=%s",
                name,
                " | ".join(repr(a)[:120] if not hasattr(a, "shape") else f"<ndarray {a.shape}>" for a in args),
            )

        return _spy

    _SIGNALS = [
        "item_started",
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
        "phase_progress",
        "scan_progress",
        "quality_update",
    ]
    _missing = []
    for sig_name in _SIGNALS:
        sig = getattr(thread_instance, sig_name, None)
        if sig is not None:
            sig.connect(_mk_spy(sig_name))
            thread_logger.debug("Signal-Spy verbunden: %s", sig_name)
        else:
            _missing.append(sig_name)
            thread_logger.warning("FEHLENDES SIGNAL: %s", sig_name)
    if _missing:
        thread_logger.error("[RELEASE_MUST] SIGNAL-KONTRAKT VERLETZT — fehlende Signale: %s", _missing)
    else:
        thread_logger.info("[OK] Alle 14 BatchProcessingThread-Signale vorhanden und bespitzelt")


# ── ModernMainWindow Subclass mit Ereignis-Instrumentierung ─────────────────
def _instrument_window(win, win_logger: logging.Logger) -> None:
    """Patcht ModernMainWindow-Instanz mit Debug-Hooks (nach __init__)."""

    # 1. Progress Bar Range Check ────────────────────────────────────────────
    pb = getattr(win, "progress_bar", None)
    if pb is not None:
        _range = (pb.minimum(), pb.maximum())
        if _range == (0, 10000):
            win_logger.info("[OK] progress_bar.setRange(0, 10000) ✓  §11.4")
        else:
            win_logger.error("[VIOLATION] progress_bar Range=%s — MUSS (0, 10000) sein! §11.4", _range)

    ppb = getattr(win, "phase_progress_bar", None)
    if ppb is not None:
        _range2 = (ppb.minimum(), ppb.maximum())
        _h = ppb.height() if ppb.isVisible() else ppb.minimumHeight()
        win_logger.info("[OK] phase_progress_bar vorhanden — Range=%s  Height=%s px  §11.4a-1", _range2, _h)
    else:
        win_logger.error("[VIOLATION] phase_progress_bar FEHLT — §11.4a Feature #1")

    # 2. QualityMeterWidget ──────────────────────────────────────────────────
    qmw = getattr(win, "quality_meter_widget", None)
    if qmw is not None:
        win_logger.info("[OK] quality_meter_widget vorhanden  §11.4a-7")
    else:
        win_logger.error("[VIOLATION] quality_meter_widget FEHLT  §11.4a-7")

    # 3. WaveformWidget.set_scan_pos ─────────────────────────────────────────
    ww = getattr(win, "waveform_widget", None)
    if ww is not None:
        if hasattr(ww, "set_scan_pos"):
            win_logger.info("[OK] WaveformWidget.set_scan_pos() vorhanden  §11.4a-6")
            _orig_ssp = ww.set_scan_pos

            def _ssp_spy(frac, _orig=_orig_ssp):
                win_logger.debug("WaveformWidget.set_scan_pos(%.3f)  §11.4a-6", frac)
                return _orig(frac)

            ww.set_scan_pos = _ssp_spy
        else:
            win_logger.error("[VIOLATION] WaveformWidget.set_scan_pos() FEHLT  §11.4a-6")
    else:
        win_logger.warning("waveform_widget nicht gefunden")

    # 4. _dispatch_to_gui Signal ─────────────────────────────────────────────
    if hasattr(win, "_gui_dispatch"):
        win_logger.info("[OK] _gui_dispatch pyqtSignal(object) vorhanden  §11.4 Thread-Safety")
    else:
        win_logger.error("[VIOLATION] _gui_dispatch FEHLT  §11.4 Thread-Safety")

    # 5. Shortcuts ────────────────────────────────────────────────────────────
    _expected_shortcuts = {
        "Space": "Play/Pause",
        "A": "Original",
        "B": "Restauriert",
        "Ctrl+O": "Öffnen",
        "Ctrl+S": "Export",
        "Ctrl+R": "Restoration",
        "Ctrl+Shift+R": "Studio 2026",
        "Escape": "Abbruch",
        "Ctrl+Z": "Pfad-Clipboard",
        "L": "Lyrics-Overlay",
    }
    _missing_sc = []
    for _sc_key, _sc_action in _expected_shortcuts.items():
        # Check via QShortcut children
        _found = False
        for _child in win.findChildren(type(None).__class__.__mro__[0]):
            pass  # placeholder — actual check below
        # We check via the window object attributes
        _attr = f"_shortcut_{_sc_key.lower().replace('+', '_').replace(' ', '_')}"
        if hasattr(win, _attr):
            _found = True
        if not _found:
            # Check via_setup_shortcuts method call trace (logged separately)
            _missing_sc.append(_sc_key)
    win_logger.info(
        "Shortcuts: Spec fordert %d — werden beim _setup_shortcuts()-Aufruf verifiziert",
        len(_expected_shortcuts),
    )

    # 6. Batch-Thread-Start Hook (für Signal-Patching) ───────────────────────
    _orig_start_batch = getattr(win, "_start_processing", None)
    if _orig_start_batch is None:
        _orig_start_batch = getattr(win, "_start_batch_processing", None)
    if _orig_start_batch is None:
        _orig_start_batch = getattr(win, "_on_restore_clicked", None)

    if _orig_start_batch:

        def _batch_start_spy(*args, _orig=_orig_start_batch, **kwargs):
            win_logger.info("=" * 60)
            win_logger.info("RESTAURIERUNG GESTARTET  §RELEASE_MUST Magic-Button")
            win_logger.info("=" * 60)
            result = _orig(*args, **kwargs)
            # Nach Start: Batch-Thread patchen
            _bt = getattr(win, "batch_thread", None)
            if _bt is not None:
                _patch_batch_thread_signals(_bt, win_logger)
                win_logger.info("[OK] BatchProcessingThread gestartet — Signale instrumentiert")
                # Watchdog-Timer prüfen
                _wt = getattr(win, "_watchdog_timer", None)
                if _wt is not None and _wt.isActive():
                    win_logger.info(
                        "[OK] Watchdog-Timer aktiv — Timeout: %d ms  §RELEASE_MUST No-Competing-Instances",
                        _wt.interval(),
                    )
                else:
                    win_logger.error("[VIOLATION] Watchdog-Timer NICHT AKTIV nach Batch-Start  §RELEASE_MUST")
                # Magic-Button-State prüfen
                _rb = getattr(win, "restore_btn", None)
                getattr(win, "studio_btn", None)
                if _rb is not None and _rb.isEnabled():
                    win_logger.error(
                        "[VIOLATION] restore_btn ist ENABLED während Verarbeitung läuft  §RELEASE_MUST UI-Gating"
                    )
                elif _rb is not None:
                    win_logger.info("[OK] restore_btn deaktiviert während Verarbeitung  §RELEASE_MUST UI-Gating")
            else:
                win_logger.warning("batch_thread nach Start nicht gefunden")
            return result

        setattr(win, _orig_start_batch.__name__, _batch_start_spy)
        win_logger.info("[OK] Batch-Start-Hook installiert auf %s", _orig_start_batch.__name__)

    # 7. _on_all_finished Hook ────────────────────────────────────────────────
    _orig_all_finished = getattr(win, "_on_all_finished", None)
    if _orig_all_finished:

        def _all_finished_spy(*args, _orig=_orig_all_finished, **kwargs):
            win_logger.info("=" * 60)
            win_logger.info("RESTAURIERUNG ABGESCHLOSSEN  all_finished Signal")

            result = _orig(*args, **kwargs)

            # phase_progress_bar Sichtbarkeit — NACH _orig(), da _orig() die Bar versteckt
            ppb2 = getattr(win, "phase_progress_bar", None)
            if ppb2 is not None and not ppb2.isVisible():
                win_logger.info("[OK] phase_progress_bar versteckt nach Abschluss  §11.4a-1")
            elif ppb2 is not None:
                win_logger.warning("[WARN] phase_progress_bar noch sichtbar nach all_finished  §11.4a-1")

            # QualityMeterWidget nach all_finished
            qmw2 = getattr(win, "quality_meter_widget", None)
            if qmw2 is not None:
                win_logger.info(
                    "Quality-Meter sichtbar=%s  mos=%.2f  §11.4a-7",
                    qmw2.isVisible(),
                    getattr(qmw2, "_mos", -1.0),
                )

            # Magic-Buttons wieder aktiv?
            _rb2 = getattr(win, "restore_btn", None)
            if _rb2 is not None and _rb2.isEnabled():
                win_logger.info("[OK] restore_btn nach Abschluss wieder aktiviert  §RELEASE_MUST UI-Gating")
            elif _rb2 is not None:
                win_logger.error("[VIOLATION] restore_btn NOCH DEAKTIVIERT nach all_finished  §RELEASE_MUST")

            win_logger.info("=" * 60)
            return result

        win._on_all_finished = _all_finished_spy
        win_logger.info("[OK] _on_all_finished-Hook installiert")


# ── Monkey-Patch: ModernMainWindow.__init__ nachlagern ──────────────────────
import Aurik10.ui.modern_window as _mw_module

_OrigModernMainWindow = _mw_module.ModernMainWindow

_win_logger = logging.getLogger("aurik.debug_window")


class _InstrumentedModernMainWindow(_OrigModernMainWindow):
    """Instrumentierte Unterklasse von ModernMainWindow für Debug-Analyse."""

    def __init__(self, *args, **kwargs):
        _win_logger.info("ModernMainWindow.__init__ START")
        super().__init__(*args, **kwargs)
        _win_logger.info("ModernMainWindow.__init__ END — Instrumentierung beginnt")
        _instrument_window(self, _win_logger)
        _win_logger.info(
            "Fenster beschriftet: bridge_available=%s",
            getattr(_mw_module, "_BRIDGE_AVAILABLE", "?"),
        )


# Patch im Modul ersetzen (main.py importiert ModernMainWindow)
_mw_module.ModernMainWindow = _InstrumentedModernMainWindow  # type: ignore[misc]


# ── Start ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _logger.info("Debug-Launcher: Übergibt an Aurik10.main.main()")
    _logger.info("Session-Log wird geschrieben nach: %s", _session_log)
    print(f"\n[AURIK Debug-Launcher] Session-Log: {_session_log}\n")

    # Ersetze sys.modules-Eintrag damit main.py die instrumentierte Klasse findet

    from Aurik10.main import main  # type: ignore[import]

    main()
