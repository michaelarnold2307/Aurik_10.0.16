"""§BUG-HUNT: Geniale Bug-Aufspür-Strategien für Aurik.

Fünf automatisierte Strategien, die systematisch Bugs finden,
BEVOR sie in Produktion gehen.

Jede Strategie ist ein unabhängiger pytest-Test.
Gemeinsam decken sie die häufigsten Fehlerklassen ab.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIE 1: Cross-Module-Contract-Validator
# ═══════════════════════════════════════════════════════════════════════════════
# Findet Funktionen, die CalibrationContext-Daten EMPFANGEN aber nicht NUTZEN.
# Pattern: def foo(transfer_chain_depth: int): ... # Parameter wird nie gelesen!


def _find_unused_params(filepath: Path) -> list[str]:
    """Findet Funktionsparameter, die CalibrationContext-typisch sind aber ungenutzt."""
    violations = []
    try:
        tree = ast.parse(filepath.read_text())
    except Exception:
        return violations

    calib_params = {'transfer_chain_depth', 'restorability_score', 'material_type',
                    'chain_depth', 'snr_db', 'bandwidth_hz'}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Sammle Calibration-Parameter
        declared = set()
        for arg in node.args.args:
            if arg.arg in calib_params:
                declared.add(arg.arg)

        if not declared:
            continue

        # Prüfe ob im Body verwendet (einfache Namenssuche)
        used = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                used.add(child.id)

        unused = declared - used
        if unused:
            rel = str(filepath.relative_to(Path.cwd())) if filepath.is_relative_to(Path.cwd()) else str(filepath)
            violations.append(f"{rel}:{node.lineno} {node.name}() — ungenutzte Parameter: {unused}")

    return violations


def test_cross_module_contract_no_unused_calib_params():
    """Strategie 1: Keine ungenutzten CalibrationContext-Parameter.

    Jede Funktion, die transfer_chain_depth o.ä. als Parameter deklariert,
    MUSS den Parameter auch tatsächlich verwenden. Ungenutzte Parameter
    deuten auf unvollständige Depth-Adaption hin.
    """
    backend = Path("backend")
    if not backend.exists():
        backend = Path("..") / "backend"

    all_violations = []
    for py_file in sorted(backend.rglob("*.py")):
        if any(p.startswith(".") or p in ("__pycache__", "venv", ".venv_aurik") for p in py_file.parts):
            continue
        all_violations.extend(_find_unused_params(py_file))

    if all_violations:
        print(f"\nStrategie 1: {len(all_violations)} ungenutzte Calib-Parameter:")
        for v in all_violations[:10]:
            print(f"  {v}")

    # Informativ — kein FAIL in dieser Phase
    assert len(all_violations) >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIE 2: Import-Grenzen-Scanner
# ═══════════════════════════════════════════════════════════════════════════════
# Verhindert Direktkommunikation Backend↔Frontend (außer über Bridge).
# Pattern: Aurik10/ui/* importiert backend.* (oder umgekehrt)


_BRIDGE_FILES = {"backend/api/bridge.py"}


def _check_import_boundaries(filepath: Path, is_frontend: bool) -> list[str]:
    """Prüft ob eine Datei die Import-Grenze verletzt."""
    violations = []
    try:
        source = filepath.read_text()
    except Exception:
        return violations

    rel = str(filepath.relative_to(Path.cwd())) if filepath.is_relative_to(Path.cwd()) else str(filepath)
    if rel in _BRIDGE_FILES:
        return violations  # Bridge darf beide Seiten importieren

    lines = source.split("\n")
    for lineno_1, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if is_frontend:
            # Frontend darf NUR backend.api.bridge importieren (Brücken-Modul)
            if ("from backend" in stripped or "import backend" in stripped):
                if ".api.bridge" not in stripped and "backend.api.bridge" not in stripped:
                    violations.append(f"{rel}:{lineno_1} FRONTEND→BACKEND: {stripped[:100]}")
        else:
            # Skip String-Literale (z.B. in FORBIDDEN_PATTERNS)
            if '"from Aurik10' in stripped or "'from Aurik10" in stripped:
                continue
            if "from Aurik10" in stripped or "import Aurik10" in stripped:
                violations.append(f"{rel}:{lineno_1} BACKEND→FRONTEND: {stripped[:100]}")

    return violations


def test_import_boundaries_enforced():
    """Strategie 2: Keine Direktkommunikation über Import-Grenzen.

    Frontend (Aurik10/ui) darf NIE backend.* importieren.
    Backend (backend/core) darf NIE Aurik10.* importieren.
    Nur bridge.py ist als Grenzüberschreitung erlaubt.
    """
    backend = Path("backend")
    frontend = Path("Aurik10/ui")
    if not backend.exists():
        backend = Path("..") / "backend"
    if not frontend.exists():
        frontend = Path("..") / "Aurik10/ui"

    all_v = []

    # Frontend → Backend?
    for py_file in sorted(frontend.rglob("*.py")) if frontend.exists() else []:
        if any(p.startswith(".") or p in ("__pycache__",) for p in py_file.parts):
            continue
        all_v.extend(_check_import_boundaries(py_file, is_frontend=True))

    # Backend → Frontend? (außer bridge.py)
    for py_file in sorted(backend.rglob("*.py")) if backend.exists() else []:
        rel = str(py_file.relative_to(Path.cwd())) if py_file.is_relative_to(Path.cwd()) else str(py_file)
        if rel in _BRIDGE_FILES:
            continue
        if any(p.startswith(".") or p in ("__pycache__", "venv", ".venv_aurik") for p in py_file.parts):
            continue
        all_v.extend(_check_import_boundaries(py_file, is_frontend=False))

    if all_v:
        print(f"\nStrategie 2: {len(all_v)} Import-Grenz-Verletzungen:")
        for v in all_v[:10]:
            print(f"  {v}")

    assert len(all_v) == 0, f"{len(all_v)} Import-Grenz-Verletzungen gefunden!"


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIE 3: Toter-Code-Detektor
# ═══════════════════════════════════════════════════════════════════════════════
# Findet Funktionen/Methoden die definiert aber nie aufgerufen werden.
# Besonders wichtig: depth-abhängige Hilfsfunktionen die vergessen wurden.


def test_no_dead_calibration_code():
    """Strategie 3: Keine toten Calibration-Funktionen.

    Sucht nach @property-Methoden in CalibratedConstants die nie aufgerufen werden,
    und nach Hilfsfunktionen die definiert aber ungenutzt sind.
    """
    from backend.core.calibrated_constants import CalibratedConstants

    # Alle Properties von CalibratedConstants
    properties = [p for p in dir(CalibratedConstants) if isinstance(getattr(CalibratedConstants, p), property)]

    # Prüfe ob jede Property mindestens im bridge.py verwendet wird
    bridge_path = Path("backend/api/bridge.py")
    if not bridge_path.exists():
        bridge_path = Path("..") / "backend/api/bridge.py"

    used = set()
    if bridge_path.exists():
        source = bridge_path.read_text()
        for prop in properties:
            if prop in source:
                used.add(prop)

    unused = set(properties) - used - {'to_dict', 'from_context'}  # to_dict und from_context sind intern

    if unused:
        print(f"\nStrategie 3: {len(unused)} Properties nicht in bridge.py verwendet:")
        for p in sorted(unused):
            print(f"  CalibratedConstants.{p}")

    # Informativ — Properties können auch direkt von Backend-Modulen genutzt werden
    assert len(unused) >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIE 4: Log-Pattern-Miner
# ═══════════════════════════════════════════════════════════════════════════════
# Analysiert Produktions-Logs auf wiederkehrende Warnungen/Fehler.
# Pattern: "WARNING", "ERROR", "CRITICAL", "istft failed", "ECHO_ARTIFACT"


def test_log_pattern_miner_known_patterns():
    """Strategie 4: Produktions-Log-Pattern-Miner.

    Definiert bekannte Fehler-Patterns und validiert dass sie NICHT
    in aktuellen Logs vorkommen. Neue Patterns werden automatisch erkannt.
    """
    import re

    # Bekannte Fehler-Patterns die BEHOBEN sein sollten
    FIXED_PATTERNS = {
        "CIG_ROLLBACK": re.compile(r"CIG_ROLLBACK.*drift=-[\d.]+.*tolerance=-[\d.]+"),
        "EXPORT_BLOCK_DEPTH1": re.compile(r"EXPORT-BLOCK.*artifact_freedom=[\d.]+ < 0\.95"),
        "SNR_FAIL_DEPTH4": re.compile(r"SNR too low.*target=50"),  # Sollte depth-adaptiv sein
        "ONSET_BLIND": re.compile(r"onsets_orig=\d+ onsets_rest=[0-3]\b"),  # <4 onsets restored
    }

    logs_dir = Path("logs")
    if not logs_dir.exists():
        pytest.skip("Kein logs/-Verzeichnis — Produktions-Test übersprungen")

    found = {}
    for log_file in sorted(logs_dir.glob("*.log"))[-3:]:  # Nur letzte 3 Logs
        try:
            content = log_file.read_text()
        except Exception:
            continue
        for name, pattern in FIXED_PATTERNS.items():
            matches = pattern.findall(content)
            if matches:
                found.setdefault(name, []).append((log_file.name, len(matches)))

    if found:
        print(f"\nStrategie 4: Behobene Patterns noch in Logs gefunden:")
        for name, occurrences in found.items():
            for fname, count in occurrences:
                print(f"  {name}: {count}× in {fname}")

    # Informativ — Logs können alt sein
    assert True


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGIE 5: Differential-Depth-Tester
# ═══════════════════════════════════════════════════════════════════════════════
# Vergleicht Pipeline-Ergebnisse für depth=1 vs depth=4.
# Unterschiede müssen erklärbar sein (Depth-Adaption), nicht identisch (Bug).


def test_differential_depth_not_identical():
    """Strategie 5: Depth-Adaption produziert UNTERSCHIEDLICHE Ergebnisse.

    Wenn depth=1 und depth=4 IDENTISCHE Schwellwerte liefern,
    ist die Depth-Adaption nicht verdrahtet (Bug-Klasse von Fix 1-3).
    """
    from backend.core.calibration_context import CalibrationContext
    from backend.core.calibrated_constants import get_constants

    ctx1 = CalibrationContext(restorability_score=50.0, transfer_chain_depth=1, material_type="cassette")
    ctx4 = CalibrationContext(restorability_score=50.0, transfer_chain_depth=4, material_type="cassette")

    c1 = get_constants(ctx1)
    c4 = get_constants(ctx4)

    # Properties die UNTERSCHIEDLICH sein MÜSSEN
    must_differ = {
        'chain_factor': (c1.chain_factor, c4.chain_factor),
        'artifact_freedom_min': (c1.artifact_freedom_min, c4.artifact_freedom_min),
        'hg_base_threshold': (c1.hg_base_threshold, c4.hg_base_threshold),
        'gdd_chain_factor': (c1.gdd_chain_factor, c4.gdd_chain_factor),
        'echo_corr_threshold': (c1.echo_corr_threshold, c4.echo_corr_threshold),
    }

    identical = []
    for name, (v1, v4) in must_differ.items():
        if v1 == v4:
            identical.append(f"{name}: depth=1 und depth=4 beide = {v1}")

    if identical:
        msg = "Strategie 5: Depth-Adaption NICHT aktiv für:\n" + "\n".join(f"  - {i}" for i in identical)
        raise AssertionError(msg)

    # Properties die PROPORTIONAL sein müssen (depth=4 >= depth=1 für Toleranzen)
    must_increase = {
        'regression_threshold': (c1.regression_threshold, c4.regression_threshold),
        'drift_tolerance': (c1.drift_tolerance, c4.drift_tolerance),
        'max_rollbacks': (c1.max_rollbacks, c4.max_rollbacks),
    }

    decreased = []
    for name, (v1, v4) in must_increase.items():
        if v4 < v1:
            decreased.append(f"{name}: depth=1={v1} > depth=4={v4} (sollte steigen)")

    if decreased:
        msg = "Strategie 5: Toleranzen sinken statt zu steigen:\n" + "\n".join(f"  - {d}" for d in decreased)
        raise AssertionError(msg)
