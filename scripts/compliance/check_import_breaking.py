#!/usr/bin/env python3
"""Pre-commit hook: Import-Breaking-Change-Detektor.

Prüft ob geänderte Dateien kritische Exports entfernt haben,
die von anderen Dateien importiert werden.

Muster: quality_mode.py wird geändert → 50 Phasen importieren
QualityModeConfig → Hook prüft ob Export noch existiert.

Usage: python scripts/compliance/check_import_breaking.py file1.py file2.py ...
"""

import ast
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Kritische Export-Verträge ──────────────────────────────────────────────
# Datei → {Symbol: [Dateien die es importieren]}
CRITICAL_EXPORTS: dict[str, dict[str, list[str]]] = {
    "backend/core/quality_mode.py": {
        "QualityModeConfig": [
            "backend/core/phases/phase_*.py",
        ],
        "QualityMode": [
            "backend/core/phases/phase_*.py",
        ],
        "is_phase_ml_enabled": [
            "backend/core/phases/phase_*.py",
        ],
        "log_mode_decision": [
            "backend/core/phases/phase_*.py",
        ],
        "validate_mode": [
            "backend/core/unified_restorer_v3.py",
        ],
    },
    "backend/core/defect_manifest.py": {
        "DefectManifest": [
            "backend/core/defect_contract_validator.py",
        ],
        "get_defect_manifest": [
            "backend/core/defect_contract_validator.py",
        ],
    },
    "backend/core/phase_pruner.py": {
        "IntelligentPhasePruner": [
            "backend/core/unified_restorer_v3.py",
        ],
        "_PHASE_DEFECT_REQUIREMENTS": [
            "backend/core/defect_contract_validator.py",
        ],
        "_MATERIAL_SKIP_PHASES": [
            "backend/core/defect_contract_validator.py",
        ],
    },
    "backend/core/safe_dict.py": {
        "SafeDict": [
            "backend/core/song_goal_importance.py",
        ],
    },
    "backend/core/periodic_health.py": {
        "get_health_collector": [
            "backend/core/unified_restorer_v3.py",
        ],
    },
}


def get_exports(filepath: str) -> set[str]:
    """Extrahiert alle exportierten Namen aus einer Python-Datei."""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except Exception:
        logger.warning("check_import_breaking.py::get_exports fallback", exc_info=True)
        return set()

    exports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    exports.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            exports.add(node.target.id)
    return exports


def main() -> None:
    changed_files = set(sys.argv[1:])
    if not changed_files:
        print("Import-Check: ⚠️ keine Dateien zum Prüfen")
        sys.exit(0)

    violations: list[str] = []

    for filepath, required_exports in CRITICAL_EXPORTS.items():
        # Nur prüfen wenn die Quelldatei geändert wurde
        if filepath not in changed_files:
            continue

        actual = get_exports(str(ROOT / filepath))
        for symbol, consumers in required_exports.items():
            if symbol not in actual:
                violations.append(
                    f"BREAKING: {filepath} exportiert '{symbol}' NICHT MEHR — "
                    f"wird importiert von: {', '.join(consumers[:3])}"
                )

    if violations:
        print(f"❌ Import-Breaking-Change: {len(violations)} fehlende Exporte\n")
        for v in violations:
            print(f"  {v}")
        print("\nDatei wurde geändert, aber kritische Exports fehlen.")
        print("Füge die fehlenden Klassen/Funktionen wieder hinzu oder")
        print("aktualisiere CRITICAL_EXPORTS in scripts/compliance/check_import_breaking.py")
        sys.exit(1)

    # §2.59: Strukturelle Validierung — defekt_hint Datenfluss
    structural_violations = []
    for fp in changed_files:
        try:
            with open(fp) as f:
                content = f.read()
        except Exception:
            logger.debug("Import-Breaking-Strukturpruefung uebersprungen", exc_info=True)
            continue
        if "_defekt_hint" in content or "defekt_hint" in content:
            # Check: does file have _defekt_hint = { but no defect_types?
            has_hint = bool(re.search(r"_defekt_hint\s*=\s*\{", content))
            has_types = '"defect_types"' in content
            has_sevs = '"defect_severities"' in content
            if has_hint and (not has_types or not has_sevs):
                structural_violations.append(
                    f"{fp}: _defekt_hint dict present but MISSING "
                    + ("defect_types " if not has_types else "")
                    + ("defect_severities" if not has_sevs else "")
                )
        if 'list(getattr(self, "_active_defekt_hint"' in content:
            structural_violations.append(
                f"{fp}: PhasePruner reads defect_types via list(dict) — use .get('defect_types', [])"
            )
    if structural_violations:
        print(f"❌ Structural violations: {len(structural_violations)}")
        for v in structural_violations:
            print(f"  🚫 {v}")
        sys.exit(1)

    print("Import-Check: ✅ alle kritischen Exports vorhanden, Struktur validiert")
    sys.exit(0)


if __name__ == "__main__":
    main()
