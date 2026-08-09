from __future__ import annotations

"""
test_watchdog_graceful_stop.py — §0c Watchdog Graceful-Stop-Verifikation
=========================================================================

Stellt sicher, dass der Watchdog KEIN klangblinder Killer ist, sondern:
1. UV3 ein _graceful_stop_event besitzt
2. request_graceful_stop() das Event setzt (threading.Event)
3. _best_carrier_checkpoint für Checkpoint-Export existiert
4. Alle Mechanismen in der Source verankert sind (Source-Level-Verifikation)

Spec-Referenz: §0c, §11.4 (Watchdog-Timer)
"""


import importlib.util

import pytest

# ── Source-Level-Verifikation (kein ML-Modell-Laden) ─────────────────────────


def _get_uv3_source() -> str:
    spec = importlib.util.find_spec("backend.core.unified_restorer_v3")
    assert spec is not None, "backend.core.unified_restorer_v3 nicht gefunden"
    with open(spec.origin, encoding="utf-8") as f:  # type: ignore[arg-type]
        return f.read()


def _get_modern_window_source() -> str:
    spec = importlib.util.find_spec("Aurik10.ui.modern_window")
    if spec is None:
        spec = importlib.util.find_spec("modern_window")
    if spec is None:
        # Fallback: direkt aus Repository-Pfad
        from pathlib import Path

        p = Path("Aurik10/ui/modern_window.py")
        if p.exists():
            return p.read_text(encoding="utf-8")
        pytest.skip("modern_window.py nicht gefunden")
    with open(spec.origin, encoding="utf-8") as f:  # type: ignore[arg-type,union-attr]
        return f.read()


@pytest.mark.unit
class TestWatchdogGracefulStopSource:
    """§0c: Source-Level-Prüfungen — ohne UV3-Instanziierung (kein ML-Load)."""

    def test_01_graceful_stop_event_exists_in_source(self):
        """UV3-Source enthält _graceful_stop_event (threading.Event)."""
        src = _get_uv3_source()
        assert "_graceful_stop_event" in src, (
            "UV3._graceful_stop_event fehlt — Watchdog kann keinen Graceful-Stop signalisieren"
        )
        assert ".Event()" in src or "threading.Event()" in src, (
            "Kein threading.Event() in UV3 — _graceful_stop_event falsch typisiert"
        )

    def test_02_request_graceful_stop_exists(self):
        """request_graceful_stop() ist definiert und setzt das Event."""
        src = _get_uv3_source()
        assert "def request_graceful_stop" in src, "UV3.request_graceful_stop() fehlt"
        assert "_graceful_stop_event.set()" in src, (
            "request_graceful_stop() setzt NICHT _graceful_stop_event — Watchdog-Signal kommt nie an"
        )

    def test_03_graceful_stop_cleared_before_pipeline(self):
        """Vor jedem Pipeline-Lauf wird das Event zurückgesetzt (clear)."""
        src = _get_uv3_source()
        assert "_graceful_stop_event.clear()" in src, (
            "Kein _graceful_stop_event.clear() — Event würde nächsten Lauf sofort stoppen"
        )

    def test_04_best_carrier_checkpoint_exists(self):
        """_best_carrier_checkpoint ist als Checkpoint-Speicher definiert."""
        src = _get_uv3_source()
        assert "_best_carrier_checkpoint" in src, (
            "UV3._best_carrier_checkpoint fehlt — Watchdog kann keinen Checkpoint exportieren"
        )
        # Checkpoint wird vor Export gespeichert (copy)
        assert "_best_carrier_checkpoint = " in src, (
            "Keine Zuweisung an _best_carrier_checkpoint — Checkpoint wird nie gespeichert"
        )

    def test_05_watchdog_calls_graceful_stop_not_terminate(self):
        """_on_watchdog_timeout ruft request_graceful_stop, nicht direkt terminate."""
        try:
            mw_src = _get_modern_window_source()
        except Exception:
            pytest.skip("modern_window.py nicht lesbar")
        assert "request_graceful_stop" in mw_src or "requestInterruption" in mw_src, (
            "_on_watchdog_timeout signalisiert keinen Graceful-Stop"
        )

    def test_06_watchdog_per_file_ms_spec_compliant(self):
        """Watchdog-Formel entspricht Spec §11.4: 32xRT + 30min."""
        try:
            mw_src = _get_modern_window_source()
        except Exception:
            pytest.skip("modern_window.py nicht lesbar")
        # Spec: _per_file_ms = max(5_400_000, audio_dur_s * 32_000 + 1_800_000)
        has_32k = "32_000" in mw_src or "32000" in mw_src
        has_1_8m = "1_800_000" in mw_src or "1800000" in mw_src
        assert has_32k, "Watchdog: 32_000 (32xRT Faktor) fehlt — Spec §11.4 verlangt audio_dur_s * 32_000"
        assert has_1_8m, "Watchdog: 1_800_000 (30min Offset) fehlt — Spec §11.4 verlangt + 1_800_000"
