"""tests/normative/test_e2e_gui_smoke.py — §v10.700 Phase E.

E2E-Integrations-Smoke-Test: Startet die GUI headless, prüft ob sie
initialisiert und grundlegende Widgets existieren.

Nutzung:
  QT_QPA_PLATFORM=offscreen pytest tests/normative/test_e2e_gui_smoke.py -v

CI-Integration:
  Läuft als Teil des Solo-Release-Gates.
"""

from __future__ import annotations

import os
import sys

import pytest

# Erzwinge offscreen backend für headless CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.mark.gui
@pytest.mark.e2e
class TestE2EGuiSmoke:
    """Prüft dass die GUI-Anwendung initialisiert und Basisfunktionalität bereitstellt."""

    @pytest.fixture(autouse=True)
    def _check_qt_available(self):
        """Skip wenn PyQt5 nicht importierbar."""
        try:
            import PyQt5.QtWidgets
        except ImportError:
            pytest.skip("PyQt5 nicht installiert")

    def test_qapplication_initializes(self):
        """QApplication kann headless gestartet werden."""
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        assert app is not None
        # Nicht beenden — andere Tests brauchen die Instanz

    def test_waveform_widget_creates(self):
        """WaveformWidget kann instanziiert werden."""
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        from Aurik10.ui.modern_window import WaveformWidget

        widget = WaveformWidget()
        assert widget is not None
        assert widget.width() > 0 or widget.minimumWidth() > 0

    def test_waveform_widget_accepts_audio(self):
        """WaveformWidget akzeptiert Audio-Daten ohne Crash."""
        import numpy as np
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        from Aurik10.ui.modern_window import WaveformWidget

        widget = WaveformWidget()
        audio = np.random.RandomState(42).randn(48000).astype(np.float32) * 0.1
        widget.set_data(audio, sample_rate=48000)
        assert widget.audio_data is not None

    def test_playhead_methods_exist(self):
        """WaveformWidget hat Loop- und Playhead-Methoden (Phase D)."""
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance() or QApplication(sys.argv)

        from Aurik10.ui.modern_window import WaveformWidget

        widget = WaveformWidget()

        # Playhead
        assert hasattr(widget, "set_playhead_position")
        widget.set_playhead_position(0.5)
        assert widget._playhead_pos == 0.5

        # Color
        assert hasattr(widget, "set_playhead_color")
        widget.set_playhead_color((255, 128, 0))  # Orange

        # Loop region
        assert hasattr(widget, "set_loop_region")
        assert hasattr(widget, "clear_loop_region")
        assert hasattr(widget, "toggle_loop")
        assert hasattr(widget, "get_loop_region")

        widget.set_loop_region(0.2, 0.5)
        region = widget.get_loop_region()
        assert region is not None
        assert abs(region[0] - 0.2) < 0.01
        assert abs(region[1] - 0.5) < 0.01

        toggled = widget.toggle_loop()
        assert isinstance(toggled, bool)

        widget.clear_loop_region()
        assert widget.get_loop_region() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
