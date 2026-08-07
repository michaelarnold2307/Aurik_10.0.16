"""Aurik10/ui/plugin_manager.py — §v10.700 H5 GUI.

Plugin-Manager-Widget: Listet installierte Plugins mit Version, Status,
und ermöglicht De-/Aktivierung.

Nutzt PluginRegistry aus backend/core/plugin_registry.py für Discovery.
Kann als eigenständiges Fenster oder als Tab in ModernMainWindow verwendet werden.

Nutzung:
    from Aurik10.ui.plugin_manager import PluginManagerWidget
    widget = PluginManagerWidget()
    widget.show()
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from Aurik10.i18n import t

_REPO_ROOT = Path(__file__).resolve().parents[2]


class PluginManagerWidget(QDialog):
    """Plugin-Manager: Zeigt installierte Plugins, Version, Status, Aktionen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("plugin_manager.title"))
        self.setMinimumSize(600, 400)
        self._refresh_on_show = False
        self._build_ui()
        self._load_plugins()

    def _build_ui(self) -> None:
        """Baut das UI-Layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel(t("plugin_manager.header"))
        header.setStyleSheet("font-size: 14pt; font-weight: bold; color: #B8CCEE;")
        layout.addWidget(header)

        desc = QLabel(t("plugin_manager.description"))
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8899BB; font-size: 9pt;")
        layout.addWidget(desc)

        # Tabelle
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            [
                t("plugin_manager.col_status"),
                t("plugin_manager.col_name"),
                t("plugin_manager.col_version"),
                t("plugin_manager.col_category"),
                t("plugin_manager.col_actions"),
            ]
        )
        _header = self._table.horizontalHeader()
        if _header is not None:
            _header.setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 60)
        self._table.setColumnWidth(2, 80)
        self._table.setColumnWidth(3, 100)
        self._table.setColumnWidth(4, 140)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet("""
            QTableWidget {
                background: rgba(12, 16, 32, 0.9);
                alternate-background-color: rgba(18, 24, 44, 0.9);
                border: 1px solid rgba(80, 100, 160, 0.3);
                border-radius: 8px;
                gridline-color: rgba(60, 80, 140, 0.15);
            }
            QTableWidget::item { color: #D0D8F0; }
            QHeaderView::section {
                background: rgba(30, 40, 70, 0.8);
                color: #8899CC;
                font-weight: bold;
                border: none;
                padding: 6px;
            }
        """)
        layout.addWidget(self._table)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_refresh = QPushButton(t("plugin_manager.refresh"))
        self._btn_refresh.clicked.connect(self._load_plugins)
        self._btn_validate_all = QPushButton(t("plugin_manager.validate_all"))
        self._btn_validate_all.clicked.connect(self._validate_all_plugins)
        self._btn_close = QPushButton(t("plugin_manager.close"))
        self._btn_close.clicked.connect(self.accept)

        for btn in (self._btn_refresh, self._btn_validate_all, self._btn_close):
            btn.setStyleSheet("""
                QPushButton {
                    background: rgba(50, 70, 120, 0.6);
                    border: 1px solid rgba(100, 130, 200, 0.4);
                    border-radius: 6px;
                    color: #C0D0F0;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover { background: rgba(70, 100, 180, 0.8); }
            """)
            btn_layout.addWidget(btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _load_plugins(self) -> None:
        """Lädt und zeigt alle entdeckten Plugins."""
        self._table.setRowCount(0)
        try:
            from backend.api.bridge import get_plugin_registry

            registry = get_plugin_registry()
            registry.reload()
            plugins = registry.list_plugins()
        except ImportError:
            self._add_error_row(t("plugin_manager.registry_unavailable"))
            return

        if not plugins:
            self._add_info_row(t("plugin_manager.none_found"))
            return

        for plugin in plugins:
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Status
            status_item = QTableWidgetItem("✅" if plugin["valid"] else "❌")
            status_item.setToolTip(
                t("plugin_manager.valid")
                if plugin["valid"]
                else t("plugin_manager.errors", errors="; ".join(plugin["errors"]))
            )
            self._table.setItem(row, 0, status_item)

            # Name
            self._table.setItem(row, 1, QTableWidgetItem(plugin["name"]))

            # Version
            self._table.setItem(row, 2, QTableWidgetItem(plugin["version"]))

            # Kategorie (aus manifest.json, default: "general")
            category = plugin.get("category", "general")
            self._table.setItem(row, 3, QTableWidgetItem(category))

            # Aktionen
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(4, 2, 4, 2)
            action_layout.setSpacing(4)

            if plugin["valid"]:
                btn_info = QPushButton("ℹ️")
                btn_info.setFixedSize(28, 28)
                author = plugin.get("author") or t("plugin_manager.author_unknown")
                btn_info.setToolTip(f"{plugin['description']}\n{t('plugin_manager.author', author=author)}")
                btn_info.setStyleSheet(
                    "QPushButton { background: rgba(60,90,160,0.4); border: none; border-radius: 4px; } QPushButton:hover { background: rgba(80,120,220,0.7); }"
                )
                action_layout.addWidget(btn_info)

            btn_enable = QCheckBox(t("plugin_manager.active"))
            btn_enable.setChecked(plugin["valid"])
            btn_enable.setEnabled(plugin["valid"])
            btn_enable.setStyleSheet("color: #A0B8E0;")
            action_layout.addWidget(btn_enable)

            self._table.setCellWidget(row, 4, action_widget)

    def _add_error_row(self, message: str) -> None:
        """Fügt eine Fehlerzeile hinzu."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        item = QTableWidgetItem(f"⚠️ {message}")
        item.setForeground(Qt.GlobalColor.red)
        self._table.setItem(row, 0, item)
        self._table.setSpan(row, 0, 1, 5)

    def _add_info_row(self, message: str) -> None:
        """Fügt eine Info-Zeile hinzu."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        item = QTableWidgetItem(message)
        self._table.setItem(row, 0, item)
        self._table.setSpan(row, 0, 1, 5)

    def _validate_all_plugins(self) -> None:
        """Validiert alle Plugins und zeigt Ergebnis."""
        try:
            from scripts.validate_plugin import validate_all_plugins

            total, failed = validate_all_plugins()
            if failed == 0:
                QMessageBox.information(
                    self,
                    t("plugin_manager.validation_title"),
                    t("plugin_manager.validation_ok", total=total),
                )
            else:
                QMessageBox.warning(
                    self,
                    t("plugin_manager.validation_title"),
                    t("plugin_manager.validation_failed", failed=failed, total=total),
                )
            self._load_plugins()
        except ImportError:
            QMessageBox.warning(self, t("plugin_manager.error_title"), t("plugin_manager.validator_unavailable"))


# ── Standalone-Test ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = PluginManagerWidget()
    widget.show()
    sys.exit(app.exec())
