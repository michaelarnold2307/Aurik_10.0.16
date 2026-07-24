"""
Aurik10/ui/ab_preview.py — Wellenform-Vorschau in der GUI.

Zeigt die geladene Wellenform vor der Restaurierung an.
Minimalistisch — keine Buttons, keine Modus-Wahl, keine Status-Texte.
Die Wellenform nutzt den gesamten verfügbaren Platz.

v10.0.15: Von A/B-Vorschau-Widget zu schlanker Wellenform-Anzeige vereinfacht.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


class ABPreviewWidget(QtWidgets.QWidget):
    """Wellenform-Vorschau — zeigt das geladene Audio vor der Restaurierung."""

    restoration_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio: np.ndarray | None = None
        self._sr: int = 48000
        self._file_path: str = ""

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Wellenform-Zeichenfläche — nutzt den gesamten Platz
        self._waveform = WaveformCanvas(self)
        self._waveform.setMinimumHeight(120)
        layout.addWidget(self._waveform, 1)

    def set_audio(self, audio: np.ndarray, sr: int, file_path: str):
        """Audio für Wellenform-Anzeige setzen."""
        self._audio = audio.astype(np.float32)
        self._sr = sr
        self._file_path = file_path
        self._waveform.set_audio(self._audio, self._sr)
        self._waveform.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._waveform.update()


class WaveformCanvas(QtWidgets.QWidget):
    """Zeichenfläche für die Wellenform-Darstellung."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio: np.ndarray | None = None
        self._sr: int = 48000
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor("#0d0d1f"))
        self.setPalette(p)

    def set_audio(self, audio: np.ndarray, sr: int):
        self._audio = audio
        self._sr = sr

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Hintergrund
        painter.fillRect(0, 0, w, h, QtGui.QColor("#0d0d1f"))

        if self._audio is None or len(self._audio) == 0:
            # Kein Audio — Fläche bleibt leer für maximale Wellenform-Darstellung
            return

        # Wellenform zeichnen
        audio = self._audio
        if audio.ndim == 2:
            audio = audio.mean(axis=1) if audio.shape[1] < audio.shape[0] else audio.mean(axis=0)
        audio = audio.ravel()

        n_samples = len(audio)
        if n_samples == 0:
            return

        # Downsample für Darstellung
        target_points = min(w * 2, n_samples)
        stride = max(1, n_samples // target_points)
        points = target_points

        center_y = h // 2
        max_amplitude = max(float(np.max(np.abs(audio))), 1e-6)

        # Mittellinie
        painter.setPen(QtGui.QPen(QtGui.QColor("#333"), 1))
        painter.drawLine(0, center_y, w, center_y)

        # Wellenform (obere Hälfte grün, untere gespiegelt)
        painter.setPen(QtGui.QPen(QtGui.QColor("#4CAF50"), 1))
        path = QtGui.QPainterPath()
        first = True

        for i in range(points):
            idx = min(i * stride, n_samples - 1)
            x = int(i * w / max(points - 1, 1))
            val = float(audio[idx]) / max_amplitude
            y = int(center_y - val * (center_y - 10))

            if first:
                path.moveTo(x, y)
                first = False
            else:
                path.lineTo(x, y)

        painter.drawPath(path)

        # Spiegelung (untere Hälfte, heller)
        painter.setPen(QtGui.QPen(QtGui.QColor("#81C784"), 1, QtCore.Qt.PenStyle.DotLine))
        path2 = QtGui.QPainterPath()
        first = True
        for i in range(points):
            idx = min(i * stride, n_samples - 1)
            x = int(i * w / max(points - 1, 1))
            val = float(audio[idx]) / max_amplitude
            y = int(center_y + val * (center_y - 10))
            if first:
                path2.moveTo(x, y)
                first = False
            else:
                path2.lineTo(x, y)
        painter.drawPath(path2)

        # Zeitachse
        painter.setPen(QtGui.QColor("#667"))
        duration_s = n_samples / self._sr
        mins = int(duration_s // 60)
        secs = int(duration_s % 60)
        painter.drawText(10, h - 8, f"{mins}:{secs:02d}")

        painter.end()
