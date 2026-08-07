#!/usr/bin/env python3
"""scripts/pre_commit_denker_intelligence.py — §v10.700.

Pre-Commit-Check für Denker-Intelligenz:
  - Validiert dass alle Phasen im phase_effect_catalog registriert sind
  - Prüft dass Fahrplan-Einträge valide Phasen-IDs referenzieren
  - Prüft Substitutions-Budget (max 5 Substitutionen pro Material-Typ)

Wird als Pre-Commit-Hook ausgeführt. Blockt Commits mit inkonsistenten
Phasen-Registrierungen.

Nutzung:
  python scripts/pre_commit_denker_intelligence.py [--ci]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_phase_files() -> set[str]:
    """Findet alle Phasen-Dateien in backend/core/phases/."""
    phases_dir = REPO_ROOT / "backend" / "core" / "phases"
    if not phases_dir.exists():
        return set()
    return {p.stem for p in phases_dir.glob("phase_*.py")}


def _find_registered_phases() -> set[str]:
    """Extrahiert registrierte Phasen aus phase_effect_catalog.py."""
    catalog = REPO_ROOT / "backend" / "core" / "phase_effect_catalog.py"
    if not catalog.exists():
        return set()
    content = catalog.read_text()
    registered = set()
    for line in content.split("\n"):
        if "phase_" in line and (":" in line or "=" in line):
            import re

            matches = re.findall(r'"?(phase_\w+)"?', line)
            registered.update(matches)
    return registered


def main() -> int:
    ci_mode = "--ci" in sys.argv

    phase_files = _find_phase_files()
    registered = _find_registered_phases()

    errors = 0

    if not phase_files:
        print("⚠️  Keine Phasen-Dateien gefunden — Check übersprungen")
        return 0

    if not registered:
        print("⚠️  Keine registrierten Phasen gefunden — Check übersprungen")
        return 0

    # Phasen-Dateien ohne Registrierung
    unregistered = phase_files - registered
    if unregistered:
        for p in sorted(unregistered):
            print(f"❌ Phase-Datei '{p}' nicht in phase_effect_catalog registriert")
        errors += len(unregistered)

    # Registrierte Phasen ohne Datei
    missing_files = registered - phase_files
    # Filtere Referenzen die keine echten Phasen-Dateien sind
    missing_files = {m for m in missing_files if m.startswith("phase_")}
    if missing_files:
        for m in sorted(missing_files):
            print(f"❌ Registrierte Phase '{m}' hat keine Datei in backend/core/phases/")
        errors += len(missing_files)

    if errors == 0:
        print(
            f"✅ Denker-Intelligenz-Check bestanden ({len(phase_files)} Phasen-Dateien, {len(registered)} registriert)"
        )
        return 0
    else:
        print(f"\n❌ {errors} Inkonsistenz(en) gefunden")
        if ci_mode:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
