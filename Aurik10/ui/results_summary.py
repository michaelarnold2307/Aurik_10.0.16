"""
Aurik10/ui/results_summary.py — Verständliches Ergebnis-Feedback für Laien.

Zeigt nach der Restaurierung in einfacher Sprache, was Aurik getan hat.
Keine technischen Metriken — nur das, was der Nutzer wissen will.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from Aurik10.i18n import t

logger = logging.getLogger(__name__)


class ResultsSummaryDialog(QtWidgets.QDialog):
    """Zeigt das Restaurierungsergebnis in einfacher, verständlicher Sprache."""

    play_requested = QtCore.pyqtSignal()
    open_folder_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        result_data: dict,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._data = result_data
        self._build_ui()
        self.setWindowTitle(t("results.title"))
        self.setMinimumSize(520, 420)
        self.setModal(True)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(28, 24, 28, 24)

        # ── Header ────────────────────────────────────────────────────────
        header = QtWidgets.QLabel(t("results.header_done"))
        header.setStyleSheet("font-size: 18pt; font-weight: bold; color: #82B89A;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # ── File info line ────────────────────────────────────────────────
        d = self._data
        file_name = d.get("file_name", "?")
        duration = d.get("duration_seconds", 0)
        material = d.get("material_detected", "")
        mins = int(duration // 60)
        secs = int(duration % 60)

        info_parts = []
        info_parts.append(f"📂 {file_name}")
        if duration > 0:
            info_parts.append(f"🕐 {mins}:{secs:02d}")
        if material:
            mat_label = t(f"material.{material}") if material else ""
            if mat_label and mat_label != f"material.{material}":
                info_parts.append(f"💿 {mat_label}")
        info_line = QtWidgets.QLabel("  |  ".join(info_parts))
        info_line.setStyleSheet("font-size: 10pt; color: #8894A8;")
        info_line.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        info_line.setWordWrap(True)
        layout.addWidget(info_line)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("background: rgba(130, 184, 154, 0.2); max-height: 1px;")
        layout.addWidget(sep)

        # ── What was done section ─────────────────────────────────────────
        what_label = QtWidgets.QLabel(t("results.what_done"))
        what_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(what_label)

        improvements = self._build_improvements()
        layout.addWidget(improvements)

        # ── Quality indicator ─────────────────────────────────────────────
        # §v10.202: Revert-Prüfung VOR Qualitäts-Anzeige
        was_reverted = d.get("was_reverted", False)
        quality_before = d.get("quality_before", 0)
        quality_after = d.get("quality_after", 0)
        mushra = d.get("mushra_score", 0.0)
        hpi = d.get("hpi_score", 0.0)
        phases = d.get("phases_total", 0)

        if was_reverted:
            revert_reason = d.get("revert_reason", "")
            quality_text = (
                f"⚠️ Bearbeitung verworfen — keine Verbesserung möglich.\n"
                f"Das Original wurde unverändert gespeichert.\n"
                f"Grund: {revert_reason}"
            ) if revert_reason else (
                "⚠️ Bearbeitung verworfen — keine Verbesserung möglich.\n"
                "Das Original wurde unverändert gespeichert."
            )
            _color = "#B8A068"
            _bg = "rgba(184, 160, 104, 0.10)"
        elif quality_after > quality_before and quality_before > 0:
            delta = quality_after - quality_before
            quality_text = (
                f"Restaurierbarkeit: {int(quality_before)}% → Ergebnisqualität: {int(quality_after)}% (+{int(delta)}%)"
            )
            if mushra > 0:
                quality_text += f"\n🎧 Klangqualität: {mushra:.0f}/100"
            if hpi > 0:
                quality_text += f" · 📈 Klangverbesserung: {hpi:.0%}"
            _color = "#82B89A"
            _bg = "rgba(130, 184, 154, 0.08)"
        elif quality_after <= quality_before and quality_before > 0:
            quality_text = (
                f"Restaurierbarkeit: {int(quality_before)}% → Ergebnisqualität: {int(quality_after)}%\n"
                f"Die Aufnahme ist bereits so gut, wie es das Quellmaterial erlaubt."
            )
            _color = "#8894A8"
            _bg = "rgba(136, 148, 168, 0.08)"
        else:
            quality_text = "Qualitätsbewertung nicht verfügbar"
            _color = "#8894A8"
            _bg = "rgba(136, 148, 168, 0.08)"

        quality_label = QtWidgets.QLabel(f"📊 {quality_text}")
        quality_label.setStyleSheet(
            f"font-size: 11pt; color: {_color}; padding: 8px; "
            f"background: {_bg}; border-radius: 6px;"
        )
        quality_label.setWordWrap(True)
        # §v10.207: Laienverständliche Tooltips
        _tooltip_parts = []
        if mushra > 0:
            _tooltip_parts.append("Klangqualität (0-100): Wie gut die Aufnahme für menschliche Ohren klingt — bewertet nach internationalem MUSHRA-Standard")
        if hpi > 0:
            _tooltip_parts.append("Klangverbesserung (0-1): Wie viel besser die restaurierte Version im Vergleich zum Original klingt")
        if _tooltip_parts:
            quality_label.setToolTip("\n\n".join(_tooltip_parts))
        layout.addWidget(quality_label)

        # ── §v10.202: Arbeits-Zusammenfassung ────
        if phases > 0:
            work_label = QtWidgets.QLabel(
                f"🔧 {phases} Phasen ausgeführt"
            )
            work_label.setStyleSheet(
                "font-size: 10pt; color: #8894A8; padding: 4px 0;"
            )
            layout.addWidget(work_label)

        # ── §v10.35 Experience Insights (Joy/Fatigue/Recommendations) ────
        joy_idx = d.get("joy_index", 0.0)
        fatigue_idx = d.get("fatigue_index", 0.0)
        recommendations = d.get("recommendations", [])

        if joy_idx > 0 or fatigue_idx > 0 or recommendations:
            exp_sep = QtWidgets.QFrame()
            exp_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            exp_sep.setStyleSheet("background: rgba(102, 126, 234, 0.2); max-height: 1px;")
            layout.addWidget(exp_sep)

            exp_header = QtWidgets.QLabel(t("results.experience_header"))
            exp_header.setStyleSheet("font-size: 12pt; font-weight: bold;")
            layout.addWidget(exp_header)

            if joy_idx > 0:
                joy_pct = int(joy_idx * 100)
                joy_emoji = "😊" if joy_idx > 0.7 else ("🙂" if joy_idx > 0.4 else "😐")
                joy_label = QtWidgets.QLabel(f"{joy_emoji}  Hörgenuss: {joy_pct}%")
                joy_color = "#82B89A" if joy_idx > 0.6 else ("#C8A84B" if joy_idx > 0.3 else "#B87A7A")
                joy_label.setStyleSheet(f"font-size: 10pt; color: {joy_color}; padding: 4px 0;")
                layout.addWidget(joy_label)

            if fatigue_idx > 0:
                fat_pct = int(fatigue_idx * 100)
                fat_emoji = "😫" if fatigue_idx > 0.6 else ("😐" if fatigue_idx > 0.3 else "😌")
                fat_label = QtWidgets.QLabel(f"{fat_emoji}  Hörermüdung: {fat_pct}%")
                fat_color = "#82B89A" if fatigue_idx < 0.3 else ("#C8A84B" if fatigue_idx < 0.5 else "#B87A7A")
                fat_label.setStyleSheet(f"font-size: 10pt; color: {fat_color}; padding: 4px 0;")
                layout.addWidget(fat_label)

            if recommendations:
                rec_label = QtWidgets.QLabel("💡 " + " · ".join(recommendations[:3]))
                rec_label.setStyleSheet("font-size: 10pt; color: #7B93F0; padding: 4px 0; font-style: italic;")
                rec_label.setWordWrap(True)
                layout.addWidget(rec_label)

        # ── Output path ───────────────────────────────────────────────────
        output = d.get("output_path", "")
        if output:
            fmt = d.get("export_format", "FLAC")
            output_label = QtWidgets.QLabel(t("results.saved_as").format(path=output, fmt=fmt))
            output_label.setStyleSheet("font-size: 10pt; color: #8894A8;")
            output_label.setWordWrap(True)
            layout.addWidget(output_label)

        # ── Spacer ────────────────────────────────────────────────────────
        layout.addStretch()

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)

        listen_btn = QtWidgets.QPushButton(t("results.listen"))
        listen_btn.setStyleSheet(
            "QPushButton { background: #667eea; color: white; border: none;"
            "border-radius: 8px; padding: 10px 24px; font-size: 11pt; font-weight: bold; }"
            "QPushButton:hover { background: #7B93F0; }"
        )
        listen_btn.clicked.connect(self.play_requested.emit)
        btn_row.addWidget(listen_btn)

        if output:
            folder_btn = QtWidgets.QPushButton(t("results.open_folder"))
            folder_btn.setStyleSheet(
                "QPushButton { background: transparent; color: #8894A8;"
                "border: 1px solid rgba(136, 148, 168, 0.3); border-radius: 8px;"
                "padding: 10px 24px; font-size: 11pt; }"
                "QPushButton:hover { border-color: #667eea; color: #c9d1d9; }"
            )
            folder_btn.clicked.connect(lambda: self.open_folder_requested.emit(str(Path(output).parent)))
            btn_row.addWidget(folder_btn)

        ok_btn = QtWidgets.QPushButton(t("results.ok"))
        ok_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #8894A8;"
            "border: 1px solid rgba(136, 148, 168, 0.3); border-radius: 8px;"
            "padding: 10px 24px; font-size: 11pt; }"
            "QPushButton:hover { border-color: #82B89A; color: #82B89A; }"
        )
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _build_improvements(self) -> QtWidgets.QWidget:
        """Baut die menschenlesbare Liste von Verbesserungen."""
        d = self._data
        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)
        vbox.setSpacing(6)
        vbox.setContentsMargins(0, 0, 0, 0)

        items = []

        # Defects
        defects_found = d.get("defects_found", 0)
        defects_fixed = d.get("defects_fixed", 0)
        if defects_fixed > 0:
            items.append(("✓", t("results.defects_fixed").format(n=defects_fixed), "#82B89A"))
        elif defects_found == 0:
            items.append(("✓", t("results.no_defects"), "#82B89A"))

        # Noise reduction
        noise_reduction = d.get("noise_reduction_pct", 0)
        if noise_reduction > 0:
            items.append(("✓", t("results.noise_reduced").format(pct=int(noise_reduction)), "#82B89A"))

        # Clarity improvement
        clarity_delta = d.get("clarity_improvement", 0)
        if clarity_delta > 0:
            items.append(("✓", t("results.clarity_improved"), "#82B89A"))

        # HPE naturalness
        hpe_before = d.get("hpe_before", 0)
        hpe_after = d.get("hpe_after", 0)
        if hpe_after > hpe_before + 0.03:
            items.append(("🎧", t("results.naturalness_improved"), "#7B93B8"))

        # Mode
        mode = d.get("mode", "")
        if mode:
            mode_name = "Studio 2026" if "STUDIO" in str(mode).upper() else "Restoration"
            items.append(("⚙️", t("results.mode_used").format(mode=mode_name), "#8894A8"))

        # Era
        era = d.get("era_detected", "")
        if era:
            items.append(("🕰️", t("results.era_detected").format(era=era), "#8894A8"))

        for icon, text, color in items:
            row = QtWidgets.QHBoxLayout()
            icon_label = QtWidgets.QLabel(icon)
            icon_label.setStyleSheet(f"color: {color}; font-size: 12pt; min-width: 24px;")
            icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            row.addWidget(icon_label)

            text_label = QtWidgets.QLabel(text)
            text_label.setStyleSheet(f"color: {color}; font-size: 10pt;")
            text_label.setWordWrap(True)
            row.addWidget(text_label, 1)
            vbox.addLayout(row)

        vbox.addStretch()
        return widget


def build_results_data(
    *,
    file_name: str = "",
    duration_seconds: float = 0,
    defects_found: int = 0,
    defects_fixed: int = 0,
    quality_before: float = 50,
    quality_after: float = 85,
    material_detected: str = "",
    era_detected: str = "",
    mode: str = "",
    output_path: str = "",
    export_format: str = "FLAC",
    noise_reduction_pct: float = 0,
    clarity_improvement: float = 0,
    hpe_before: float = 0,
    hpe_after: float = 0,
    restoration_result: object = None,
) -> dict:
    """Baut das data-dict für den ResultsSummaryDialog."""
    # §v10.201: Echte Qualitätswerte aus RestorationResult.metadata
    _rmeta = getattr(restoration_result, "metadata", {}) or {}
    _q_raw = float(getattr(restoration_result, "quality_estimate", 0.0) or 0.0)
    _q_after = round(_q_raw * 100, 1)
    _q_before = float(_rmeta.get("restorability_score", 50))
    _mushra = float((_rmeta.get("mushra") or {}).get("mushra_score", 0.0))
    _hpi = float(_rmeta.get("hpi_score", 0.0))
    _reverted = bool((_rmeta.get("do_no_harm") or {}).get("reverted", False))
    _phases = int(_rmeta.get("phases_total", 0))
    _dnh_reason = str((_rmeta.get("do_no_harm") or {}).get("reason", ""))
    return {
        "file_name": file_name,
        "duration_seconds": duration_seconds,
        "defects_found": defects_found,
        "defects_fixed": defects_fixed,
        # §v10.201: Echte Werte aus metadata (überschreiben die Default-Parameter)
        "quality_before": _q_before if _q_before > 0 else quality_before,
        "quality_after": _q_after if _q_after > 0 else quality_after,
        "mushra_score": _mushra,
        "hpi_score": _hpi,
        "was_reverted": _reverted,
        "revert_reason": _dnh_reason,
        "phases_total": _phases,
        "material_detected": material_detected,
        "era_detected": era_detected,
        "mode": mode,
        "output_path": output_path,
        "export_format": export_format,
        "noise_reduction_pct": noise_reduction_pct,
        "clarity_improvement": clarity_improvement,
        "hpe_before": hpe_before,
        "hpe_after": hpe_after,
        "joy_index": 0.0,
        "fatigue_index": 0.0,
        "recommendations": [],
    }
