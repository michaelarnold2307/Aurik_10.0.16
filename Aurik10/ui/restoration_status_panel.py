"""§v10.15 Restoration Status Panel
===================================
Rich status display for the restoration pipeline.
Shows current phase with emoji, material badge, progress counter,
and §v10.708 live dimensional quality metrics (tonal_center, timbre,
natuerlichkeit, authentizitaet) — die vier Dimensionen, die das
menschliche Ohr tatsächlich wahrnimmt.

Usage in modern_window.py:
    from Aurik10.ui.restoration_status_panel import RestorationStatusPanel
    self._status_panel = RestorationStatusPanel(parent)
    layout.addWidget(self._status_panel)

    # Update from bridge signals:
    self._status_panel.set_phase("phase_03_denoise", 3, 43)
    self._status_panel.set_material("cassette", 1970, "Deutscher Schlager")
    self._status_panel.set_dimensional_quality({
        "tonal_center": 0.85, "timbre_authentizitaet": 0.78,
        "natuerlichkeit": 0.82, "authentizitaet": 0.80
    })
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

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
    """Rich, log-quality status display for the restoration pipeline.

    §v10.708: Zeigt live die vier dimensionalen Qualitätsmetriken,
    die das menschliche Ohr tatsächlich wahrnimmt:
    - tonal_center: Bleibt die Tonart erhalten?
    - timbre_authentizitaet: Klingt die Klangfarbe noch authentisch?
    - natuerlichkeit: Klingt es natürlich oder künstlich?
    - authentizitaet: Ist das Original erkennbar?

    Farbcodierung gegen material-adaptive Schwellwerte (wie Export-Gate).
    """

    # §v10.708: Material-adaptive P1/P2-Schwellwerte (identisch mit Export-Gate)
    _P1P2_THRESHOLDS: dict[str, dict[str, float]] = {
        "shellac": {"tonal_center": 0.65, "timbre_authentizitaet": 0.60,
                     "natuerlichkeit": 0.62, "authentizitaet": 0.60},
        "vinyl": {"tonal_center": 0.74, "timbre_authentizitaet": 0.70,
                  "natuerlichkeit": 0.72, "authentizitaet": 0.70},
        "tape": {"tonal_center": 0.72, "timbre_authentizitaet": 0.68,
                 "natuerlichkeit": 0.70, "authentizitaet": 0.68},
        "cassette": {"tonal_center": 0.70, "timbre_authentizitaet": 0.66,
                      "natuerlichkeit": 0.68, "authentizitaet": 0.66},
        "reel_tape": {"tonal_center": 0.74, "timbre_authentizitaet": 0.70,
                      "natuerlichkeit": 0.72, "authentizitaet": 0.70},
        "digital": {"tonal_center": 0.78, "timbre_authentizitaet": 0.75,
                     "natuerlichkeit": 0.78, "authentizitaet": 0.76},
    }
    _DEFAULT_THRESHOLDS: dict[str, float] = {
        "tonal_center": 0.74, "timbre_authentizitaet": 0.70,
        "natuerlichkeit": 0.72, "authentizitaet": 0.70,
    }

    # Human-readable labels for the 4 dimensions
    _DIM_LABELS: dict[str, str] = {
        "tonal_center": "Tonart",
        "timbre_authentizitaet": "Klangfarbe",
        "natuerlichkeit": "Natürlichkeit",
        "authentizitaet": "Originaltreue",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("restorationStatusPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(52)

        self._current_material: str = "unknown"

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

        layout.addLayout(_phase_col, 2)

        # Center: §v10.708 Dimensional quality metrics
        self._dim_labels: dict[str, QLabel] = {}
        _dim_col = QVBoxLayout()
        _dim_col.setSpacing(1)
        _dim_header = QLabel("Klangqualität")
        _dim_header.setStyleSheet("font-size: 10px; color: #888;")
        _dim_col.addWidget(_dim_header)
        _dim_row = QHBoxLayout()
        _dim_row.setSpacing(8)
        for dim_key in ["tonal_center", "timbre_authentizitaet", "natuerlichkeit", "authentizitaet"]:
            label = QLabel("—")
            label.setStyleSheet(
                "font-size: 11px; padding: 2px 6px; border-radius: 3px; "
                "background: #2a2a35; color: #888;"
            )
            label.setToolTip(self._DIM_LABELS.get(dim_key, dim_key))
            self._dim_labels[dim_key] = label
            _dim_row.addWidget(label)
        _dim_col.addLayout(_dim_row)
        layout.addLayout(_dim_col, 2)

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
        """Update material/era/genre badges + store material for threshold lookup."""
        self._current_material = material.lower() if material else "unknown"
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

    def set_dimensional_quality(self, scores: dict[str, float]) -> None:
        """§v10.708: Live-Update der vier dimensionalen Qualitätsmetriken.

        Args:
            scores: Dict mit Keys tonal_center, timbre_authentizitaet,
                    natuerlichkeit, authentizitaet. Werte 0.0–1.0.
                    Fehlende Keys werden als "—" angezeigt.
        """
        thresholds = self._P1P2_THRESHOLDS.get(
            self._current_material, self._DEFAULT_THRESHOLDS
        )
        for dim_key, label in self._dim_labels.items():
            value = scores.get(dim_key)
            if value is None:
                label.setText("—")
                label.setStyleSheet(
                    "font-size: 11px; padding: 2px 6px; border-radius: 3px; "
                    "background: #2a2a35; color: #888;"
                )
                continue

            threshold = thresholds.get(dim_key, 0.70)
            short_label = self._DIM_LABELS.get(dim_key, dim_key)

            if value >= threshold + 0.05:
                # Grün: deutlich über Schwelle
                color = "#6ab86a"
                bg = "#1a2a1a"
            elif value >= threshold:
                # Gelb: knapp über Schwelle
                color = "#b8a840"
                bg = "#2a2a1a"
            elif value >= threshold - 0.10:
                # Orange: knapp unter Schwelle — Warnung
                color = "#c87830"
                bg = "#2a2010"
            else:
                # Rot: deutlich unter Schwelle — kritisch
                color = "#c84848"
                bg = "#2a1010"

            label.setText(f"{short_label} {value:.0%}")
            label.setStyleSheet(
                f"font-size: 11px; padding: 2px 6px; border-radius: 3px; "
                f"background: {bg}; color: {color};"
            )
            label.setToolTip(
                f"{short_label}: {value:.3f} "
                f"(Schwelle: {threshold:.2f} für {self._current_material})"
            )

    def set_complete(self) -> None:
        """Show completion state."""
        self._phase_icon.setText("✅")
        self._phase_name.setText("Restauration abgeschlossen")
        self._phase_counter.setText("")