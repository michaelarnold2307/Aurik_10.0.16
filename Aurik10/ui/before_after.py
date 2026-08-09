"""§v10.14 P2: Vorher/Nachher-Vergleichs-Widget für das Ergebnis-Panel.

Zeigt Metriken vor und nach der Restaurierung in einem kompakten,
visuell ansprechenden Format: Defekte, Bandbreite, LUFS, Qualität.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class MetricCard(QFrame):
    """Eine einzelne Metrik-Karte mit Label, Vorher- und Nachher-Wert."""

    def __init__(self, title: str, before: str, after: str, improved: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "MetricCard { background: #1A2030; border-radius: 6px; padding: 6px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #7B93B8; font-size: 7pt; font-weight: bold;")
        layout.addWidget(title_label)

        values_row = QHBoxLayout()
        before_label = QLabel(before)
        before_label.setStyleSheet("color: #B8A068; font-size: 9pt;")
        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("color: #4A6080; font-size: 9pt;")
        after_color = "#82B89A" if improved else "#B8A068"
        after_label = QLabel(after)
        after_label.setStyleSheet(f"color: {after_color}; font-size: 9pt; font-weight: bold;")

        values_row.addWidget(before_label)
        values_row.addWidget(arrow_label)
        values_row.addWidget(after_label)
        values_row.addStretch()
        layout.addLayout(values_row)


class BeforeAfterWidget(QWidget):
    """Zeigt eine kompakte Vorher/Nachher-Übersicht der Restaurierung."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 4, 0, 0)
        main.setSpacing(4)

        header = QLabel("Vorher  →  Nachher")
        header.setStyleSheet("color: #AFC3DA; font-size: 9pt; font-weight: bold; padding: 2px 0;")
        main.addWidget(header)

        self._cards_layout = QHBoxLayout()
        self._cards_layout.setSpacing(6)
        main.addLayout(self._cards_layout)
        main.addStretch()

    def load(self, before: dict, after: dict) -> None:
        """Befüllt das Widget mit Vorher/Nachher-Daten.

        Args:
            before: {"defects": int, "bandwidth_hz": float, "lufs": float, "quality": float}
            after:  {"defects": int, "bandwidth_hz": float, "lufs": float, "quality": float}
        """
        # Clear existing cards
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        _def_before = int(before.get("defects", 0))
        _def_after = int(after.get("defects", 0))
        _def_delta = _def_before - _def_after
        self._cards_layout.addWidget(MetricCard(
            "Störungen",
            str(_def_before),
            str(_def_after) + (f" (−{_def_delta})" if _def_delta > 0 else ""),
            improved=_def_after < _def_before,
        ))

        _bw_before = float(before.get("bandwidth_hz", 0))
        _bw_after = float(after.get("bandwidth_hz", 0))
        self._cards_layout.addWidget(MetricCard(
            "Bandbreite",
            f"{_bw_before:.0f} Hz" if _bw_before > 0 else "—",
            f"{_bw_after:.0f} Hz" if _bw_after > 0 else "—",
            improved=_bw_after > _bw_before,
        ))

        _q_before = float(before.get("quality", 0))
        _q_after = float(after.get("quality", 0))
        self._cards_layout.addWidget(MetricCard(
            "Qualität",
            f"{_q_before:.0f}%" if _q_before > 0 else "—",
            f"{_q_after:.0f}%" if _q_after > 0 else "—",
            improved=_q_after > _q_before,
        ))

        _lu_before = float(before.get("lufs", 0))
        _lu_after = float(after.get("lufs", 0))
        if _lu_before != 0 or _lu_after != 0:
            self._cards_layout.addWidget(MetricCard(
                "Lautheit",
                f"{_lu_before:.1f} LUFS" if _lu_before != 0 else "—",
                f"{_lu_after:.1f} LUFS" if _lu_after != 0 else "—",
                improved=abs(_lu_after - (-14.0)) < abs(_lu_before - (-14.0)) if _lu_before != 0 else True,
            ))
