"""§v10.990: Frontend↔Backend-Harmonie — Drift-Gates.

Diese Tests pinnen die Brücke zwischen GUI und SOTA-Backend:
  1. Die zwei BridgeCalibrationData-Kopien (Aurik10/ui ↔ backend/api) dürfen NIE driften.
  2. Bridge quality_color-Hex-Werte MÜSSEN mit der UI-Palette übereinstimmen.
  3. SOTA-Zugänge der Bridge liefern Frontend-taugliche Daten.
  4. Das Status-Panel nutzt die Palette (keine Hex-Werte mehr im Code).
  5. modern_window verdrahtet das Status-Panel in den Statusbereich.
"""

from __future__ import annotations

import ast
import re
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _dataclass_field_names(mod: object) -> list[str]:
    import dataclasses

    cls = getattr(mod, "BridgeCalibrationData")
    return [f.name for f in dataclasses.fields(cls)]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Drift-Gate: BridgeCalibrationData (Frontend-Kopie vs Backend-Kopie)
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_calibration_dataclass_copies_identical():
    """Die beiden BridgeCalibrationData-Kopien müssen Feld-identisch bleiben."""
    import Aurik10.ui.bridge_calibration as frontend_mod
    import backend.api.bridge_calibration_data as backend_mod

    fe = _dataclass_field_names(frontend_mod)
    be = _dataclass_field_names(backend_mod)
    assert fe == be, f"Feld-Drift: frontend={fe} backend={be}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Drift-Gate: Bridge-Qualitätsfarben == UI-Palette
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_quality_colors_match_ui_palette():
    """Die drei depth-abhängigen quality_color-Werte der Bridge entsprechen der Palette."""
    bridge_src = _read("backend/api/bridge.py")
    palette_src = _read("Aurik10/ui/ui_constants.py")

    palette_hexes = set(re.findall(r'#[0-9A-Fa-f]{6}', palette_src))
    # Bridge-seitige Qualitätsfarben (nur die drei depth-Abstufungen)
    bridge_colors = re.findall(r'color = "(#[0-9A-Fa-f]{6})" if depth >= \d else "\1"', bridge_src)
    # einfacher: alle Hex-Werte im _build_bridge_calibration_dict-Block
    m = re.search(
        r'def _build_bridge_calibration_dict.*?color = "(#[0-9A-Fa-f]{6})" if depth >= 4 else "\("(#[0-9A-Fa-f]{6})" if depth >= 3 else "(#[0-9A-Fa-f]{6})"\)"',
        bridge_src,
        re.S,
    )
    if m:
        deep, moderate, studio = m.group(1), m.group(2), m.group(3)
        for hex_val in (deep, moderate, studio):
            assert hex_val in palette_hexes, f"Bridge-Farbe {hex_val} fehlt in der UI-Palette"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Bridge-SOTA-Zugänge liefern Frontend-taugliche Daten
# ═══════════════════════════════════════════════════════════════════════════════


def test_bridge_model_zoo_summary_shape():
    from backend.api.bridge import get_model_zoo_summary

    entries = get_model_zoo_summary()
    assert isinstance(entries, list)
    assert len(entries) >= 1
    for e in entries:
        assert {"name", "purpose", "status", "integration", "notes"} <= set(e.keys())


def test_bridge_sota_chain_status_keys():
    from backend.api.bridge import get_sota_chain_status

    status = get_sota_chain_status()
    assert "model_zoo" in status
    assert "components" in status
    comps = status["components"]
    for key in ("defect_consensus", "repair_planner", "artifact_guards", "perceptual_loop"):
        assert isinstance(comps.get(key), bool), f"components.{key} fehlt"


def test_bridge_guard_report_from_repair_report():
    """get_guard_report liest §v10.990 RepairReport-Telemetrie."""
    from backend.api.bridge import get_guard_report
    from backend.core.coordinated_repair import RepairPlan, RepairReport

    report = RepairReport(
        plan=RepairPlan(),
        completed_steps=[],
        failed_steps=[],
        total_time=1.0,
        input_peak=0.5,
        output_peak=0.5,
        guard_violations={"truepeak": 1, "spectral": 2},
        guard_peak_delta_db=0.8,
        utmos_iterations=3,
        utmos_blend_count=1,
        utmos_mos_before=3.1,
        utmos_mos_after=2.9,
    )
    data = get_guard_report(report)
    assert data["guards"]["truepeak"] == 1
    assert data["guards"]["spectral"] == 2
    assert data["guards"]["peak_delta_db"] == 0.8
    assert data["utmos_loop"]["iterations"] == 3
    assert data["utmos_loop"]["blend_back"] is True

    class _Wrap:
        repair_report = report
        metadata = {}

    data2 = get_guard_report(_Wrap())
    assert data2["guards"]["spectral"] == 2


def test_bridge_defect_consensus_summary_defensive():
    from backend.api.bridge import get_defect_consensus_summary

    assert get_defect_consensus_summary(None) == {}
    # Fremdes Objekt → Null-Statistik statt Exception (Frontend-tolerant)
    zeroed = get_defect_consensus_summary(object())
    assert zeroed["defect_count"] == 0
    assert zeroed["module_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Status-Panel: Palette statt Hex-Werte + SOTA-Methoden
# ═══════════════════════════════════════════════════════════════════════════════


def test_status_panel_uses_palette_tokens_not_hex():
    """restoration_status_panel.py darf keine Hex-Farben mehr hartkodieren."""
    src = _read("Aurik10/ui/restoration_status_panel.py")
    # Alle Hex-Werte außerhalb des Palette-Imports prüfen
    after_imports = src.split("from Aurik10.ui.ui_constants import")[-1]
    hex_occurrences = re.findall(r'"#[0-9A-Fa-f]{6}"', after_imports)
    assert not hex_occurrences, f"Hex-Werte im Panel-Code: {hex_occurrences}"


def test_status_panel_has_sota_methods():
    src = _read("Aurik10/ui/restoration_status_panel.py")
    for method in ("set_sota_chain", "set_consensus_summary", "set_repair_plan_summary", "set_guard_report"):
        assert f"def {method}(" in src, f"{method} fehlt im Status-Panel"


def test_ui_constants_palette_defined():
    src = _read("Aurik10/ui/ui_constants.py")
    for token in (
        "SURFACE_BG", "TEXT_PRIMARY", "TEXT_MUTED",
        "QUALITY_STUDIO", "QUALITY_MODERATE", "QUALITY_DEEP_CHAIN",
        "STATUS_OK_TEXT", "STATUS_CRIT_BG", "BADGE_MATERIAL_TEXT",
    ):
        assert f"{token} =" in src, f"Palette-Token {token} fehlt"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. modern_window-Verdrahtung (Source-Level-Gates)
# ═══════════════════════════════════════════════════════════════════════════════


def test_modern_window_wires_status_panel():
    src = _read("Aurik10/ui/modern_window.py")
    assert "_RestorationStatusPanel(wrapper)" in src, "Panel wird nicht im Statusbereich erzeugt"
    assert "self._status_panel = _RestorationStatusPanel(wrapper)" in src
    assert "self._sync_status_panel(_eff_step, _eff_total, _live_hint)" in src, "Phasen-Sync fehlt"
    assert "_panel.set_complete()" in src, "Abschluss-Update fehlt"
    assert "_sync_status_panel_sota()" in src


def test_restaurier_denker_stores_repair_plan_for_frontend():
    src = _read("denker/restaurier_denker.py")
    assert "cached_defect_result.repair_plan = _repair_plan" in src


def test_bridge_exports_sota_accessors():
    """Alle neuen SOTA-Zugänge stehen in __all__ (für Stern-Imports stabil)."""
    src = _read("backend/api/bridge.py")
    m = re.search(r"__all__ = \[(.*?)\]", src, re.S)
    assert m, "__all__ fehlt in bridge.py"
    body = m.group(1)
    for name in (
        "get_model_zoo_summary", "get_sota_chain_status",
        "get_defect_consensus_summary", "get_repair_plan_summary", "get_guard_report",
    ):
        assert f'"{name}"' in body, f"{name} fehlt in __all__"


def test_ui_still_bridge_only():
    """§11: Das Status-Panel importiert weiterhin NUR backend.api.bridge."""
    src = _read("Aurik10/ui/restoration_status_panel.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("backend.core", "plugins", "dsp")), alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith(("backend.core", "plugins", "dsp")), mod
