"""
Aurik10/ui/onboarding.py — Erststart-Assistent für Aurik 10.

Zeigt beim ersten Start einen freundlichen 3-Schritt-Wizard der erklärt,
was Aurik macht und wie man es benutzt. Speichert den Status via QSettings,
damit der Wizard nur einmal erscheint.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)

from Aurik10.i18n import t

logger = logging.getLogger(__name__)


class OnboardingWizard(QtWidgets.QDialog):
    finished = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(t("app.name"))
        self.setMinimumSize(560, 420)
        self.resize(600, 460)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.CustomizeWindowHint
            | QtCore.Qt.WindowType.WindowTitleHint
        )

        self._current_page = 0
        self._setup_ui()
        self._apply_theme()

    def paintEvent(self, event):
        """Zeichnet den Splashscreen-Hintergrund ins Onboarding."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._draw_onboarding_background(p, self.width(), self.height())
        p.end()
        super().paintEvent(event)

    def _draw_onboarding_background(self, p: QPainter, w: int, h: int) -> None:
        """Hintergrund-Glows im Stil des Aurik-Splashscreens."""
        corners = 10.0
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), corners, corners)
        p.setClipPath(clip)

        # Deep dark base — match splash: QColor(7, 9, 22)
        p.fillRect(0, 0, w, h, QColor(7, 9, 22))

        # Left warm gold glow ("before / damaged")
        g1 = QRadialGradient(w * 0.15, h * 0.42, w * 0.40)
        g1.setColorAt(0.0, QColor(188, 128, 28, 52))
        g1.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g1))

        # Right cool blue glow ("after / restored")
        g2 = QRadialGradient(w * 0.85, h * 0.42, w * 0.40)
        g2.setColorAt(0.0, QColor(18, 95, 218, 55))
        g2.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g2))

        # Top purple accent
        g3 = QRadialGradient(w * 0.50, 0.0, w * 0.45)
        g3.setColorAt(0.0, QColor(102, 78, 205, 42))
        g3.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g3))

        # Bottom teal glow
        g4 = QRadialGradient(w * 0.50, float(h), w * 0.48)
        g4.setColorAt(0.0, QColor(0, 188, 212, 28))
        g4.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, w, h, QBrush(g4))

        # Gradient border: gold TL → purple mid → blue BR
        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0, QColor(212, 162, 50, 155))
        bg.setColorAt(0.4, QColor(102, 126, 234, 110))
        bg.setColorAt(1.0, QColor(18, 145, 255, 135))
        p.setPen(QPen(QBrush(bg), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.75, 0.75, w - 1.5, h - 1.5), corners, corners)
        p.setClipping(False)

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Stacked pages
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._page_welcome())
        self._stack.addWidget(self._page_how())
        self._stack.addWidget(self._page_ready())
        layout.addWidget(self._stack, 1)

        # Navigation bar
        nav = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav)
        nav_layout.setContentsMargins(20, 12, 20, 16)

        # Dot indicators
        self._dots: list[QtWidgets.QLabel] = []
        dots_widget = QtWidgets.QWidget()
        dots_layout = QtWidgets.QHBoxLayout(dots_widget)
        dots_layout.setSpacing(8)
        for _ in range(3):
            dot = QtWidgets.QLabel("●")
            dot.setStyleSheet("font-size: 14px; color: #555;")
            dots_layout.addWidget(dot)
            self._dots.append(dot)
        nav_layout.addWidget(dots_widget)

        nav_layout.addStretch()

        self._btn_back = QtWidgets.QPushButton(t("onboarding.back"))
        self._btn_back.setVisible(False)
        self._btn_back.clicked.connect(self._prev_page)
        nav_layout.addWidget(self._btn_back)

        self._btn_next = QtWidgets.QPushButton(t("onboarding.next"))
        self._btn_next.clicked.connect(self._next_page)
        nav_layout.addWidget(self._btn_next)

        layout.addWidget(nav)
        self._update_dots()

    def _page_welcome(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 20)

        title = QtWidgets.QLabel(t("onboarding.welcome.title"))
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(t("onboarding.welcome.subtitle"))
        subtitle.setStyleSheet("font-size: 14px; color: #8894A8; margin-top: 8px;")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        intro = QtWidgets.QLabel(t("onboarding.welcome.body"))
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; line-height: 1.6;")
        layout.addWidget(intro)

        layout.addSpacing(20)

        # Features grid
        features = [
            ("🔇", "Entfernt Knackser & Rauschen"),
            ("🎚️", "Automatisch – kein Fachwissen nötig"),
            ("🔒", "100% offline – deine Daten bleiben privat"),
            ("💿", "Für Vinyl, Kassette, CD & MP3"),
        ]

        for icon, text in features:
            row = QtWidgets.QHBoxLayout()
            icon_label = QtWidgets.QLabel(icon)
            icon_label.setStyleSheet("font-size: 20px;")
            row.addWidget(icon_label)
            text_label = QtWidgets.QLabel(text)
            text_label.setStyleSheet("font-size: 13px;")
            row.addWidget(text_label)
            row.addStretch()
            layout.addLayout(row)

        layout.addStretch()
        return page

    def _page_how(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 20)

        title = QtWidgets.QLabel(t("onboarding.how.title"))
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        layout.addSpacing(20)

        # Mode cards
        for mode_id, mode_name, desc, icon, color in [
            (
                "restoration",
                "💿 Restoration",
                "Behutsam · Original bleibt erhalten\nFür Vinyl, Kassetten & Archiv",
                "💿",
                "#FFB300",
            ),
            (
                "studio",
                "🎯 Studio 2026",
                "Modern · Klar & kraftvoll\nFür Spotify, YouTube & Streaming",
                "🎯",
                "#00B0FF",
            ),
        ]:
            card = QtWidgets.QFrame()
            card.setStyleSheet(
                f"background: rgba(14,18,36,0.8); border: 1px solid {color}; border-radius: 10px; padding: 12px;"
            )
            card_layout = QtWidgets.QVBoxLayout(card)
            card_title = QtWidgets.QLabel(f"{icon}  {mode_name}")
            card_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {color};")
            card_layout.addWidget(card_title)
            card_desc = QtWidgets.QLabel(desc.replace("\n", "<br>"))
            card_desc.setStyleSheet("font-size: 12px;")
            card_layout.addWidget(card_desc)
            layout.addWidget(card)

        layout.addSpacing(15)

        steps = QtWidgets.QLabel(
            t("onboarding.how.step_1") + "\n" + t("onboarding.how.step_2") + "\n" + t("onboarding.how.step_3")
        )
        steps.setStyleSheet("font-size: 13px; line-height: 1.8;")
        layout.addWidget(steps)

        layout.addStretch()
        return page

    def _page_ready(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(40, 40, 40, 20)

        title = QtWidgets.QLabel(t("onboarding.ready.title"))
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        layout.addSpacing(15)

        msg = QtWidgets.QLabel(t("onboarding.ready.body"))
        msg.setWordWrap(True)
        msg.setStyleSheet("font-size: 14px; line-height: 1.6;")
        layout.addWidget(msg)

        layout.addSpacing(20)

        self._chk_skip = QtWidgets.QCheckBox(t("onboarding.show_again"))
        layout.addWidget(self._chk_skip)

        layout.addStretch()
        return page

    def _next_page(self):
        if self._current_page < 2:
            self._current_page += 1
            self._stack.setCurrentIndex(self._current_page)
            self._update_dots()

            if self._current_page == 2:
                self._btn_next.setText(t("onboarding.finish"))
                try:
                    self._btn_next.clicked.disconnect()
                except TypeError:
                    pass  # Bereits disconnected oder nie verbunden
                self._btn_next.clicked.connect(self._finish)
        else:
            self._finish()

    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._stack.setCurrentIndex(self._current_page)
            self._update_dots()

    def _update_dots(self):
        for i, dot in enumerate(self._dots):
            color = "#667eea" if i == self._current_page else "#555"
            dot.setStyleSheet(f"font-size: 14px; color: {color};")
        self._btn_back.setVisible(self._current_page > 0)
        if self._current_page < 2:
            self._btn_next.setText(t("onboarding.next"))
        else:
            self._btn_next.setText(t("onboarding.finish"))

    def _finish(self):
        if self._chk_skip.isChecked():
            from PyQt5.QtCore import QSettings

            s = QSettings("AURIK", "AURIK Professional")
            s.setValue("onboarding/shown", True)
        self.finished.emit()
        self.accept()

    def _apply_theme(self):
        self.setStyleSheet("""
            QDialog {
                background: transparent;
                color: #d0d8ff;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background: rgba(102,126,234,0.25);
                color: #fff;
                border: 1px solid rgba(102,126,234,0.6);
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 10pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(102,126,234,0.50);
            }
            QCheckBox {
                color: #8899bb;
                font-size: 10pt;
            }
            QLabel {
                color: #d0d8ff;
            }
        """)


def should_show_onboarding() -> bool:
    """Prüft, ob der Onboarding-Wizard angezeigt werden soll."""
    try:
        from PyQt5.QtCore import QSettings

        s = QSettings("AURIK", "AURIK Professional")
        return not s.value("onboarding/shown", False, type=bool)
    except Exception:
        logger.warning("onboarding.py::should_show_onboarding Ersatzpfad", exc_info=True)
        return True
