"""§v10.15 Restoration Status Panel
===================================
Rich status display for the restoration pipeline.
Shows current phase with emoji, material badge, progress counter.

Usage in modern_window.py:
    from Aurik10.ui.restoration_status_panel import RestorationStatusPanel
    self._status_panel = RestorationStatusPanel(parent)
    layout.addWidget(self._status_panel)

    # Update from bridge signals:
    self._status_panel.set_phase("phase_03_denoise", 3, 43)
    self._status_panel.set_material("cassette", 1970, "Deutscher Schlager")
"""

from __future__ import annotations

from typing import Callable, cast

from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)  # pylint: disable=no-name-in-module

# §v10.70 Bridge: Phase-Display-Formatter über die Bridge, nicht direkt aus backend.core
from backend.api.bridge import get_phase_display_formatter_fns

_DISPLAY_FNS = get_phase_display_formatter_fns()
get_carrier_display: Callable[..., str] = cast(
    Callable[..., str], _DISPLAY_FNS.get("get_carrier_display", lambda *_a, **_kw: "?")
)
get_era_display: Callable[..., str] = cast(
    Callable[..., str], _DISPLAY_FNS.get("get_era_display", lambda *_a, **_kw: "?")
)
get_phase_display: Callable[..., str] = cast(
    Callable[..., str], _DISPLAY_FNS.get("get_phase_display", lambda *_a, **_kw: "?")
)


class RestorationStatusPanel(QFrame):
    """Rich, log-quality status display for the restoration pipeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("restorationStatusPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(16)

        # Left: Phase icon + name
        self._phase_icon = QLabel("🔄")
        self._phase_icon.setStyleSheet("font-size: 22px;")
        self._phase_name = QLabel("Initialisiere …")
        self._phase_name.setStyleSheet("font-size: 13px; font-weight: 600; color: #d0d0d0;")
        self._phase_counter = QLabel("")
        self._phase_counter.setStyleSheet("font-size: 11px; color: #888;")

        _phase_col = QVBoxLayout()
        _phase_col.setSpacing(1)
        _phase_name_row = QHBoxLayout()
        _phase_name_row.setSpacing(6)
        _phase_name_row.addWidget(self._phase_icon)
        _phase_name_row.addWidget(self._phase_name)
        _phase_name_row.addStretch()
        _phase_col.addLayout(_phase_name_row)

        _info_row = QHBoxLayout()
        _info_row.setSpacing(8)
        _info_row.addWidget(self._phase_counter)
        _info_row.addStretch()
        _phase_col.addLayout(_info_row)

        layout.addLayout(_phase_col, 3)

        # Right: Material badge + era + genre
        self._material_badge = QLabel("")
        self._material_badge.setStyleSheet(
            "background: #2a2a35; color: #b8a068; padding: 3px 10px; "
            "border-radius: 4px; font-size: 11px; font-weight: 500;"
        )
        self._era_badge = QLabel("")
        self._era_badge.setStyleSheet(
            "background: #2a2a35; color: #6890b8; padding: 3px 10px; border-radius: 4px; font-size: 11px;"
        )
        self._genre_badge = QLabel("")
        self._genre_badge.setStyleSheet(
            "background: #2a2a35; color: #68a068; padding: 3px 10px; border-radius: 4px; font-size: 11px;"
        )

        _badge_row = QHBoxLayout()
        _badge_row.setSpacing(6)
        _badge_row.addStretch()
        _badge_row.addWidget(self._material_badge)
        _badge_row.addWidget(self._era_badge)
        _badge_row.addWidget(self._genre_badge)
        layout.addLayout(_badge_row)

    # ── Public API ──────────────────────────────────────────────────

    def set_phase(self, phase_id: str, current: int = 0, total: int = 0) -> None:
        """Update the current phase display."""
        display = get_phase_display(phase_id)
        # Split emoji prefix from name
        parts = display.split(" ", 1)
        if len(parts) == 2 and any(ord(c) > 127 for c in parts[0]):
            self._phase_icon.setText(parts[0])
            self._phase_name.setText(parts[1])
        else:
            self._phase_icon.setText("🔄")
            self._phase_name.setText(display)
        if current > 0 and total > 0:
            self._phase_counter.setText(f"Phase {current}/{total}")
        else:
            self._phase_counter.setText("")

    def set_material(self, material: str, decade: int = 0, genre: str = "") -> None:
        """Update material/era/genre badges."""
        if material:
            self._material_badge.setText(get_carrier_display(material))
            self._material_badge.setVisible(True)
        else:
            self._material_badge.setVisible(False)
        if decade:
            self._era_badge.setText(get_era_display(decade))
            self._era_badge.setVisible(True)
        else:
            self._era_badge.setVisible(False)
        if genre:
            self._genre_badge.setText(f"🎵 {genre}")
            self._genre_badge.setVisible(True)
        else:
            self._genre_badge.setVisible(False)

    def set_complete(self) -> None:
        """Show completion state."""
        self._phase_icon.setText("✅")
        self._phase_name.setText("Restauration abgeschlossen")
        self._phase_counter.setText("")
