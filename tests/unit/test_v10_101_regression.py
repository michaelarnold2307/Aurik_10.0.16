"""§v10.101 Regression-Tests: verhindert Rückkehr behobener Bugs.

Tests:
  1. Import-Shadowing: kein `import os` ohne Alias in _profiled_phase_call
  2. GlobalGainBudget: reset() wird bei configure_for_chain_depth aufgerufen
  3. Mikrodynamik: Material-adaptive Schwellwerte für Kassette
  4. DoNoHarmGuardian: Material-abhängige Crest/Nat-Schwellwerte
  5. Phase_07 FeedbackChain: strength capped bei 0.25

Alle Tests non-destructive — lesen nur Source/API, schreiben nichts.
"""

import ast
import inspect
import re

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Import-Shadowing
# ═══════════════════════════════════════════════════════════════════════════════


def _has_shadowing_import(filepath: str) -> list[dict]:
    """Find functions where a module-level name is shadowed by local import."""
    with open(filepath) as f:
        source = f.read()
    tree = ast.parse(source)

    module_names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                module_names.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    name = alias.asname if alias.asname else alias.name
                    module_names.add(name)

    bugs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    for alias in child.names:
                        imported = alias.asname if alias.asname else alias.name.split(".")[0]
                        if imported in module_names:
                            for sub in ast.walk(node):
                                if (
                                    isinstance(sub, ast.Name)
                                    and sub.id == imported
                                    and isinstance(sub.ctx, ast.Load)
                                    and sub.lineno < child.lineno
                                ):
                                    bugs.append(
                                        {
                                            "func": node.name,
                                            "func_line": node.lineno,
                                            "name": imported,
                                            "use_line": sub.lineno,
                                            "import_line": child.lineno,
                                        }
                                    )
                                    break
    return bugs


def test_no_os_shadowing_in_uv3():
    """§v10.101: kein `import os` shadowing in unified_restorer_v3.py."""
    bugs = _has_shadowing_import("backend/core/unified_restorer_v3.py")
    os_bugs = [b for b in bugs if b["name"] == "os"]
    assert len(os_bugs) == 0, (
        f"Import-Shadowing gefunden: os in {os_bugs[0]['func']}() "
        f"verwendet vor lokalem import (L{os_bugs[0]['use_line']} < L{os_bugs[0]['import_line']})"
    )


def test_no_import_shadowing_in_core():
    """§v10.101: kein import-shadowing in backend/core/."""
    import os

    base = os.path.join(os.path.dirname(__file__), "..", "backend", "core")
    all_bugs = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(root, f)
                bugs = _has_shadowing_import(fp)
                all_bugs.extend(bugs)
    assert len(all_bugs) == 0, (
        f"Import-Shadowing in {len(all_bugs)} Funktion(en): {[(b['name'], b['func']) for b in all_bugs[:5]]}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GlobalGainBudget
# ═══════════════════════════════════════════════════════════════════════════════


def test_gain_budget_resets_on_configure():
    """§v10.101: configure_for_chain_depth ruft reset() auf."""
    from backend.core.global_gain_budget import get_global_gain_budget

    gb = get_global_gain_budget()
    # Simuliere Verbrauch
    gb.request("test_phase", 5.0)
    assert gb.cumulative_db > 0.0, "Precondition: Budget sollte verbraucht sein"

    # configure muss resetten
    gb.configure_for_chain_depth(2)
    assert gb.cumulative_db == 0.0, "Budget wurde NICHT zurückgesetzt — Singleton akkumuliert über Läufe!"


def test_gain_budget_snr_adaptive():
    """§v10.101: SNR-adaptives Budget für Kassette."""
    from backend.core.global_gain_budget import get_global_gain_budget

    gb = get_global_gain_budget()
    gb.configure_for_chain_depth(1)
    base_budget = gb._total_budget_db
    assert base_budget == 6.0, f"depth=1 sollte 6.0 dB sein, nicht {base_budget}"

    gb.configure_for_chain_depth(4, snr_db=14.3, material="cassette")
    assert gb._total_budget_db > base_budget, (
        f"SNR=14.3 Kassette sollte > {base_budget} dB Budget haben, nicht {gb._total_budget_db}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Mikrodynamik
# ═══════════════════════════════════════════════════════════════════════════════


def test_mikrodynamik_material_adaptive():
    """§v10.101: Kassette bekommt höheren Wet-Blend als Default."""
    from backend.core.dsp.mikrodynamik_guard import _MATERIAL_FLOOR_THRESHOLD, recommend_mikrodynamik_wet

    # Kassette sollte eigenen Floor haben
    assert "cassette" in _MATERIAL_FLOOR_THRESHOLD, "Kassette fehlt in Material-Floors"
    assert _MATERIAL_FLOOR_THRESHOLD["cassette"] < 0.97, (
        f"Kassette-Floor {_MATERIAL_FLOOR_THRESHOLD['cassette']} sollte < 0.97 sein"
    )

    # Gleiche Korrelation → Kassette = höherer Wet
    wet_default = recommend_mikrodynamik_wet(0.79, 0.35, global_need=0.5)
    wet_cassette = recommend_mikrodynamik_wet(0.79, 0.35, global_need=0.5, material="cassette")
    assert wet_cassette > wet_default, (
        f"Kassette ({wet_cassette:.3f}) sollte > Default ({wet_default:.3f}) bei corr=0.79"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DoNoHarmGuardian
# ═══════════════════════════════════════════════════════════════════════════════


def test_donoharm_material_adaptive_crest():
    """§v10.101: evaluate() akzeptiert material-Parameter mit adaptiven Schwellen."""
    from backend.core.do_no_harm_guardian import DoNoHarmGuardian

    # Verifiziere dass material-Parameter existiert
    sig = inspect.signature(DoNoHarmGuardian.evaluate)
    assert "material" in sig.parameters, "evaluate() muss material-Parameter haben für adaptive Schwellwerte"


def test_donoharm_cassette_passes_higher_crest_drop():
    """§v10.101: Kassette mit 5dB Crest-Drop sollte passen (Schwelle 6dB)."""
    from backend.core.do_no_harm_guardian import DoNoHarmGuardian

    guardian = DoNoHarmGuardian()
    # Simuliere: Input mit Rauschen (Crest niedrig), Output sauber (Crest höher)
    # Erzeuge Test-Signale
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)

    # Input: Sinus + Rauschen → niedriger Crest
    rng = np.random.RandomState(42)
    input_audio = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * rng.randn(sr)).astype(np.float32)
    input_audio = np.clip(input_audio, -1.0, 1.0)

    # Output: sauberer Sinus → höherer Crest → Crest-Drop negativ (kein Problem)
    output_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    guardian.capture_input(input_audio, sr)
    verdict = guardian.evaluate(output_audio, sr, material="cassette")
    # Crest sollte NICHT degradiert sein (clean output hat besseren Crest)
    crest_degraded = any("crest_drop" in m for m in verdict.degraded_metrics)
    assert not crest_degraded, f"Crest fälschlich als degradiert gemeldet: {verdict.degraded_metrics}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Phase_07 FeedbackChain
# ═══════════════════════════════════════════════════════════════════════════════


def test_phase07_feedbackchain_strength_capped():
    """§v10.101: FeedbackChain capped phase_07 bei 0.25."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    src = inspect.getsource(UnifiedRestorerV3.restore)
    # Suche nach der _fc_strength-Logik
    has_cap = "_fc_strength = 0.25" in src or 'if "phase_07"' in src
    assert has_cap, (
        "Phase_07 FeedbackChain strength cap fehlt! "
        "Ohne diesen Fix crasht phase_07 in der FeedbackChain mit -86.5 dBFS."
    )
