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

from PyQt5.QtCore import Qt
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

# §v10.990: Zentrale UI-Palette — keine Hex-Werte mehr direkt in diesem Modul
from Aurik10.ui.ui_constants import (
    BADGE_ERA_TEXT,
    BADGE_GENRE_TEXT,
    BADGE_MATERIAL_TEXT,
    STATUS_CRIT_BG,
    STATUS_CRIT_TEXT,
    STATUS_OK_BG,
    STATUS_OK_TEXT,
    STATUS_ORANGE_BG,
    STATUS_ORANGE_TEXT,
    STATUS_WARN_BG,
    STATUS_WARN_TEXT,
    SURFACE_BG,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

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
        "shellac": {
            "tonal_center": 0.65,
            "timbre_authentizitaet": 0.60,
            "natuerlichkeit": 0.62,
            "authentizitaet": 0.60,
        },
        "vinyl": {"tonal_center": 0.74, "timbre_authentizitaet": 0.70, "natuerlichkeit": 0.72, "authentizitaet": 0.70},
        "tape": {"tonal_center": 0.72, "timbre_authentizitaet": 0.68, "natuerlichkeit": 0.70, "authentizitaet": 0.68},
        "cassette": {
            "tonal_center": 0.70,
            "timbre_authentizitaet": 0.66,
            "natuerlichkeit": 0.68,
            "authentizitaet": 0.66,
        },
        "reel_tape": {
            "tonal_center": 0.74,
            "timbre_authentizitaet": 0.70,
            "natuerlichkeit": 0.72,
            "authentizitaet": 0.70,
        },
        "digital": {
            "tonal_center": 0.78,
            "timbre_authentizitaet": 0.75,
            "natuerlichkeit": 0.78,
            "authentizitaet": 0.76,
        },
    }
    _DEFAULT_THRESHOLDS: dict[str, float] = {
        "tonal_center": 0.74,
        "timbre_authentizitaet": 0.70,
        "natuerlichkeit": 0.72,
        "authentizitaet": 0.70,
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

        layout = QHBoxLayout()
        layout.setSpacing(16)

        _outer = QVBoxLayout(self)
        _outer.setContentsMargins(12, 6, 12, 6)
        _outer.setSpacing(2)
        _outer.addLayout(layout)

        # Left: Phase icon + name
        self._phase_icon = QLabel("🔄")
        self._phase_icon.setStyleSheet("font-size: 22px;")
        self._phase_name = QLabel("Initialisiere …")
        self._phase_name.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {TEXT_PRIMARY};")
        self._phase_counter = QLabel("")
        self._phase_counter.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")

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
        _dim_header.setStyleSheet(f"font-size: 10px; color: {TEXT_MUTED};")
        _dim_col.addWidget(_dim_header)
        _dim_row = QHBoxLayout()
        _dim_row.setSpacing(8)
        for dim_key in ["tonal_center", "timbre_authentizitaet", "natuerlichkeit", "authentizitaet"]:
            label = QLabel("—")
            label.setStyleSheet(
                f"font-size: 11px; padding: 2px 6px; border-radius: 3px; background: {SURFACE_BG}; color: {TEXT_MUTED};"
            )
            label.setToolTip(self._DIM_LABELS.get(dim_key, dim_key))
            self._dim_labels[dim_key] = label
            _dim_row.addWidget(label)
        _dim_col.addLayout(_dim_row)
        layout.addLayout(_dim_col, 2)

        # §v10.990: SOTA-Ketten-Badges (Model Zoo, Consensus, Plan, Guards)
        self._sota_badges: list[QLabel] = []
        _sota_col = QVBoxLayout()
        _sota_col.setSpacing(1)
        _sota_header = QLabel("SOTA-Kette")
        _sota_header.setStyleSheet(f"font-size: 10px; color: {TEXT_MUTED};")
        _sota_col.addWidget(_sota_header)
        _sota_row = QHBoxLayout()
        _sota_row.setSpacing(6)
        for _tooltip in ("Model Zoo", "Consensus", "Repair-Plan", "Guards"):
            _badge = QLabel("—")
            _badge.setStyleSheet(
                f"font-size: 11px; padding: 2px 6px; border-radius: 3px; background: {SURFACE_BG}; color: {TEXT_MUTED};"
            )
            _badge.setToolTip(_tooltip)
            self._sota_badges.append(_badge)
            _sota_row.addWidget(_badge)
        _sota_col.addLayout(_sota_row)
        layout.addLayout(_sota_col, 2)

        # Right: Material badge + era + genre
        self._material_badge = QLabel("")
        self._material_badge.setStyleSheet(
            f"background: {SURFACE_BG}; color: {BADGE_MATERIAL_TEXT}; padding: 3px 10px; "
            "border-radius: 4px; font-size: 11px; font-weight: 500;"
        )
        self._era_badge = QLabel("")
        self._era_badge.setStyleSheet(
            f"background: {SURFACE_BG}; color: {BADGE_ERA_TEXT}; padding: 3px 10px; border-radius: 4px; font-size: 11px;"
        )
        self._genre_badge = QLabel("")
        self._genre_badge.setStyleSheet(
            f"background: {SURFACE_BG}; color: {BADGE_GENRE_TEXT}; padding: 3px 10px; border-radius: 4px; font-size: 11px;"
        )

        _badge_row = QHBoxLayout()
        _badge_row.setSpacing(6)
        _badge_row.addStretch()
        _badge_row.addWidget(self._material_badge)
        _badge_row.addWidget(self._era_badge)
        _badge_row.addWidget(self._genre_badge)
        layout.addLayout(_badge_row)

        # §v10.992: Einwilligungs-Zeile — „Gefunden … · Aurik wird …" (reine Transparenz)
        self._consent_label = QLabel("")
        self._consent_label.setObjectName("consentLabel")
        self._consent_label.setStyleSheet(
            f"font-size: 11px; color: {BADGE_MATERIAL_TEXT}; background: transparent; padding: 2px 2px 0 2px;"
        )
        self._consent_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._consent_label.setVisible(False)
        _outer.addWidget(self._consent_label)

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
        thresholds = self._P1P2_THRESHOLDS.get(self._current_material, self._DEFAULT_THRESHOLDS)
        for dim_key, label in self._dim_labels.items():
            value = scores.get(dim_key)
            if value is None:
                label.setText("—")
                label.setStyleSheet(
                    f"font-size: 11px; padding: 2px 6px; border-radius: 3px; background: {SURFACE_BG}; color: {TEXT_MUTED};"
                )
                continue

            threshold = thresholds.get(dim_key, 0.70)
            short_label = self._DIM_LABELS.get(dim_key, dim_key)

            if value >= threshold + 0.05:
                # Grün: deutlich über Schwelle
                color = STATUS_OK_TEXT
                bg = STATUS_OK_BG
            elif value >= threshold:
                # Gelb: knapp über Schwelle
                color = STATUS_WARN_TEXT
                bg = STATUS_WARN_BG
            elif value >= threshold - 0.10:
                # Orange: knapp unter Schwelle — Warnung
                color = STATUS_ORANGE_TEXT
                bg = STATUS_ORANGE_BG
            else:
                # Rot: deutlich unter Schwelle — kritisch
                color = STATUS_CRIT_TEXT
                bg = STATUS_CRIT_BG

            label.setText(f"{short_label} {value:.0%}")
            label.setStyleSheet(
                f"font-size: 11px; padding: 2px 6px; border-radius: 3px; background: {bg}; color: {color};"
            )
            label.setToolTip(f"{short_label}: {value:.3f} (Schwelle: {threshold:.2f} für {self._current_material})")

    def set_complete(self) -> None:
        """Show completion state."""
        self._phase_icon.setText("✅")
        self._phase_name.setText("Restauration abgeschlossen")
        self._phase_counter.setText("")

    # ── §v10.990 SOTA-Kette ─────────────────────────────────────────

    def _set_sota_badge(self, index: int, text: str, ok: bool | None = None) -> None:
        """Setzt ein SOTA-Badge; ok=None → neutral, True → grün, False → orange."""
        badge = self._sota_badges[index]
        if ok is None:
            color, bg = TEXT_MUTED, SURFACE_BG
        elif ok:
            color, bg = STATUS_OK_TEXT, STATUS_OK_BG
        else:
            color, bg = STATUS_ORANGE_TEXT, STATUS_ORANGE_BG
        badge.setText(text)
        badge.setStyleSheet(
            f"font-size: 11px; padding: 2px 6px; border-radius: 3px; background: {bg}; color: {color};"
        )

    def set_sota_chain(self, status: dict) -> None:
        """§v10.990: Model-Zoo- + Komponenten-Status (bridge.get_sota_chain_status())."""
        if not status:
            return
        zoo = status.get("model_zoo", {})
        total = int(zoo.get("total", 0) or 0)
        by_status = zoo.get("by_status", {}) or {}
        active = int(by_status.get("available", 0) or 0) + int(by_status.get("active", 0) or 0)
        self._set_sota_badge(0, f"🦁 Zoo {total}·{active}", ok=total > 0)

        comps = status.get("components", {}) or {}
        all_ready = all(bool(v) for v in comps.values())
        self._set_sota_badge(1, "🧠 Consensus", ok=bool(comps.get("defect_consensus")))
        self._set_sota_badge(2, "🗺️ Plan", ok=bool(comps.get("repair_planner")))
        self._set_sota_badge(3, "🛡️ Guards", ok=all_ready and bool(comps.get("artifact_guards")))

    def set_consensus_summary(self, summary: dict) -> None:
        """§v10.990: Consensus-Ergebnis (bridge.get_defect_consensus_summary)."""
        if not summary:
            return
        n = int(summary.get("defect_count", 0) or 0)
        mods = int(summary.get("module_count", 0) or 0)
        self._set_sota_badge(1, f"🧠 {n} Defekte·{mods} Mod", ok=True)

    def set_repair_plan_summary(self, summary: dict) -> None:
        """§v10.990: Repair-Plan (bridge.get_repair_plan_summary)."""
        if not summary:
            return
        n = int(summary.get("step_count", 0) or 0)
        self._set_sota_badge(2, f"🗺️ {n} Phasen", ok=n > 0)

    def set_repair_consent(self, consent: dict) -> None:
        """§v10.992: Einwilligungs-Ansicht — zeigt in Alltagssprache, was Aurik tun wird.

        KEINE Interaktion (keine Checkboxen, kein Editieren): reine Transparenz.
        consent = bridge.get_repair_plan_consent(defect_result)
        """
        if not consent:
            self._consent_label.setVisible(False)
            return
        found = consent.get("found", []) or []
        will_do = consent.get("will_do", []) or []
        if not found and not will_do:
            self._consent_label.setVisible(False)
            return
        parts: list[str] = []
        if found:
            parts.append("Gefunden: " + ", ".join(
                f"{f['label']} ({f.get('severity', '')})".rstrip()
                for f in found[:4]
            ))
        if will_do:
            parts.append("Aurik wird: " + " → ".join(will_do[:8]))
        text = "   ·   ".join(parts)
        fm = self._consent_label.fontMetrics()
        _max_w = max(320, self.width() - 48)
        shown = fm.elidedText(text, Qt.TextElideMode.ElideRight, _max_w)
        self._consent_label.setText(shown)
        self._consent_label.setToolTip(text)
        self._consent_label.setVisible(True)

    def set_guard_report(self, report: dict) -> None:
        """§v10.990: Guard-/Loop-Telemetrie (bridge.get_guard_report)."""
        if not report:
            return
        g = report.get("guards", {}) or {}
        fired = (
            int(g.get("truepeak", 0) or 0)
            + int(g.get("pumping", 0) or 0)
            + int(g.get("formant", 0) or 0)
            + int(g.get("spectral", 0) or 0)
        )
        loop = report.get("utmos_loop", {}) or {}
        iters = int(loop.get("iterations", 0) or 0)
        if fired == 0 and iters == 0:
            self._set_sota_badge(3, "🛡️ 0", ok=True)
        else:
            self._set_sota_badge(3, f"🛡️ {fired}·{iters}×", ok=fired == 0)
