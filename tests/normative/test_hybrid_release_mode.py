from __future__ import annotations

"""Hybrid-Release-Mode CI-Gate — [RELEASE_MUST] (copilot-instructions.md §2.37 / März 2026)

Spec:
    - Statusskripte müssen release_mode ('primary'|'fallback'|'blocked') pro Kernmodul ausgeben.
    - all_runtime_ready=True auch wenn primäre Artefakte fehlen (Fallback aktiv).
    - Zulässige Kaskaden:
        sgmse_plus:    sgmse_plus.ts → wpe_dsp_fallback
        versa:         primary → pqs_dsp_fallback
        flow_matching: primary → cqtdiff_or_diffwave_fallback
    - Quarantänisierte Crash-Kandidaten (RMVPE) dürfen NICHT als primary registriert sein.

KI-Richtlinien-Gate-Tabelle:
    "Hybrid-Release-Mode RELEASE_MUST" → test_hybrid_release_mode.py (diese Datei)

Aufruf: pytest tests/normative/test_hybrid_release_mode.py -v --timeout=30
"""


import importlib
import importlib.util  # type: ignore[import-untyped]
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Script direkt importieren (nicht als installed package)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


def _load_validate_script():
    """Importiert validate_core_model_presence ohne subprocess."""
    spec_path = _SCRIPTS_DIR / "validate_core_model_presence.py"
    if not spec_path.exists():
        pytest.skip(f"validate_core_model_presence.py nicht gefunden: {spec_path}")
    spec = importlib.util.spec_from_file_location("validate_core_model_presence", spec_path)  # type: ignore[union-attr]
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type,union-attr]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runtime_rows():
    """Gibt die runtime-readiness-Zeilen aus dem Statusskript zurück."""
    mod = _load_validate_script()
    return mod._runtime_ready_checks()


# ---------------------------------------------------------------------------
# Tests: release_mode-Feld vorhanden
# ---------------------------------------------------------------------------


_REQUIRED_COMPONENTS = {"fcpe", "sgmse_plus", "versa", "flow_matching", "gacela"}
_VALID_MODES = {"primary", "fallback", "blocked"}


@pytest.mark.unit
def test_all_required_components_present(runtime_rows):
    """Alle normativ geforderten Kernkomponenten erscheinen in den Checks."""
    found = {r["name"] for r in runtime_rows}
    missing = _REQUIRED_COMPONENTS - found
    assert not missing, f"Fehlende Komponenten im Statusskript: {missing}"


def test_every_component_has_release_mode(runtime_rows):
    """Jede Komponente muss ein 'release_mode'-Feld ausgeben."""
    without = [r["name"] for r in runtime_rows if "release_mode" not in r]
    assert not without, (
        f"Komponenten ohne release_mode-Feld: {without} — "
        "Statusskript muss release_mode (primary|fallback|blocked) pro Modul ausgeben."
    )


def test_release_mode_values_are_valid(runtime_rows):
    """release_mode muss 'primary', 'fallback' oder 'blocked' sein."""
    invalid = [(r["name"], r["release_mode"]) for r in runtime_rows if r.get("release_mode") not in _VALID_MODES]
    assert not invalid, f"Ungültige release_mode-Werte: {invalid}"


def test_every_component_has_resolved_by(runtime_rows):
    """Jede Komponente muss einen 'resolved_by'-Eintrag haben."""
    without = [r["name"] for r in runtime_rows if not r.get("resolved_by")]
    assert not without, f"Komponenten ohne resolved_by: {without}"


# ---------------------------------------------------------------------------
# Tests: all_runtime_ready — Fallbacks greifen
# ---------------------------------------------------------------------------


def test_all_runtime_ready_with_fallbacks(runtime_rows):
    """all_runtime_ready muss True sein, wenn Fallback aktiv.

    Spec: Primär-Artefakte können export-technisch blockiert sein — Runtime-Readiness
    muss dennoch True sein (primary OR fallback).
    """
    not_ready = [r["name"] for r in runtime_rows if not bool(r.get("runtime_ready"))]
    assert not not_ready, (
        f"Komponenten nicht runtime-ready (weder primary noch fallback): {not_ready}\n"
        "Statusskript muss all_runtime_ready=True liefern, sobald DSP-Fallback registriert ist."
    )


def test_sgmse_runtime_ready(runtime_rows):
    """sgmse_plus: wpe_dsp_fallback ist immer verfügbar — runtime_ready muss True."""
    row = next((r for r in runtime_rows if r["name"] == "sgmse_plus"), None)
    assert row is not None, "sgmse_plus fehlt in _runtime_ready_checks()"
    assert bool(row["runtime_ready"]), "sgmse_plus ist nicht runtime_ready — wpe_dsp_fallback (lokal) muss greifen!"


def test_versa_runtime_ready(runtime_rows):
    """versa: pqs_dsp_fallback ist immer verfügbar — runtime_ready muss True."""
    row = next((r for r in runtime_rows if r["name"] == "versa"), None)
    assert row is not None, "versa fehlt in _runtime_ready_checks()"
    assert bool(row["runtime_ready"]), "versa ist nicht runtime_ready — pqs_dsp_fallback muss greifen!"


def test_flow_matching_runtime_ready(runtime_rows):
    """flow_matching: cqtdiff_or_diffwave_fallback — runtime_ready muss True wenn Fallback vorhanden."""
    row = next((r for r in runtime_rows if r["name"] == "flow_matching"), None)
    assert row is not None, "flow_matching fehlt in _runtime_ready_checks()"
    # Fallback ist nur verfügbar wenn cqtdiff oder diffwave existieren — skip wenn weder vorhanden
    if not bool(row["primary"]) and not bool(row["fallback"]):
        pytest.skip("flow_matching: Weder primary noch Fallback-Modell vorhanden — CI-Umgebung ohne Modelle")
    assert bool(row["runtime_ready"]), f"flow_matching nicht runtime_ready — release_mode={row.get('release_mode')}"


# ---------------------------------------------------------------------------
# Tests: Fallback-Kaskaden-Konsistenz
# ---------------------------------------------------------------------------


def test_sgmse_fallback_resolved_by(runtime_rows):
    """sgmse_plus ohne primary → resolved_by muss 'wpe_dsp_fallback' oder 'torchscript_fallback' lauten."""
    row = next((r for r in runtime_rows if r["name"] == "sgmse_plus"), None)
    assert row is not None
    if not bool(row["primary"]):
        assert row["resolved_by"] in {
            "wpe_dsp_fallback",
            "torchscript_fallback",
        }, f"sgmse_plus ohne primary hat unbekanntes resolved_by='{row['resolved_by']}'"


def test_versa_fallback_resolved_by(runtime_rows):
    """versa ohne primary → resolved_by muss 'pqs_dsp_fallback' lauten."""
    row = next((r for r in runtime_rows if r["name"] == "versa"), None)
    assert row is not None
    if not bool(row["primary"]):
        assert row["resolved_by"] == "pqs_dsp_fallback", (
            f"versa ohne primary hat unbekanntes resolved_by='{row['resolved_by']}'"
        )


def test_flow_matching_fallback_resolved_by(runtime_rows):
    """flow_matching ohne primary → resolved_by muss 'cqtdiff_or_diffwave_fallback' oder 'missing' lauten."""
    row = next((r for r in runtime_rows if r["name"] == "flow_matching"), None)
    assert row is not None
    if not bool(row["primary"]):
        assert row["resolved_by"] in {
            "cqtdiff_or_diffwave_fallback",
            "missing",
        }, f"flow_matching ohne primary hat unbekanntes resolved_by='{row['resolved_by']}'"


# ---------------------------------------------------------------------------
# Tests: release_mode-Konsistenz mit primary/fallback
# ---------------------------------------------------------------------------


def test_primary_components_have_primary_mode(runtime_rows):
    """Wenn primary=True → release_mode muss 'primary' sein."""
    wrong = [
        (r["name"], r["release_mode"])
        for r in runtime_rows
        if bool(r.get("primary")) and r.get("release_mode") != "primary"
    ]
    assert not wrong, f"Komponenten mit primary=True aber release_mode != 'primary': {wrong}"


def test_blocked_only_if_no_fallback(runtime_rows):
    """release_mode='blocked' darf NUR vorkommen wenn primary=False UND fallback=False."""
    wrong = [
        r["name"]
        for r in runtime_rows
        if r.get("release_mode") == "blocked" and (bool(r.get("primary")) or bool(r.get("fallback")))
    ]
    assert not wrong, f"Komponenten mit release_mode='blocked' aber primary oder fallback verfügbar: {wrong}"


# ---------------------------------------------------------------------------
# Extra: Kein fcpe ohne Fallback blockiert (FCPE Crash-Check)
# ---------------------------------------------------------------------------


def test_fcpe_not_blocked_if_crepe_present(runtime_rows):
    """Wenn crepe.onnx existiert, darf fcpe NICHT blocked sein."""
    row = next((r for r in runtime_rows if r["name"] == "fcpe"), None)
    if row is None:
        pytest.skip("fcpe nicht in runtime_rows")
    _fallback = bool(row.get("fallback")) if isinstance(row, dict) else False
    if _fallback:
        assert row.get("release_mode") != "blocked", "fcpe hat Fallback (crepe/rmvpe), darf nicht 'blocked' sein."
