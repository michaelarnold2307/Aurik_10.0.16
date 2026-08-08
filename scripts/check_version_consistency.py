#!/usr/bin/env python3
"""Version-Consistency-Check — Sprint D. Spec v10.700 F1.

Prüft dass pyproject.toml ≡ README.md ≡ CHANGELOG.md die gleiche Version haben.
CI-Gate: Exit 0 = konsistent, Exit 1 = Inkonsistenz.

Usage:
    python scripts/check_version_consistency.py [--fix]
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CORE_FILES = ["pyproject.toml", "README.md", "CHANGELOG.md"]


def extract_version(filepath: Path) -> str | None:
    """Extrahiert die Version aus einer Datei."""
    with open(filepath) as f:
        content = f.read()
    # pyproject.toml: version = "10.0.18"
    m = re.search(r'version\s*=\s*"(\d+\.\d+\.\d+)"', content)
    if m:
        return m.group(1)
    # README.md/CHANGELOG.md: **Version:** 10.0.18 or ## 10.0.18 (...)
    m = re.search(r'(?:\*\*Version:?\*\*|##)\s*(\d+\.\d+\.\d+)', content)
    if m:
        return m.group(1)
    return None


def main():
    versions: dict[str, str | None] = {}
    for fname in CORE_FILES:
        fpath = PROJECT_ROOT / fname
        if fpath.exists():
            versions[fname] = extract_version(fpath)

    # Kanonische Quelle: pyproject.toml
    canonical = versions.get("pyproject.toml")
    if not canonical:
        print("❌ Kanonische Version nicht in pyproject.toml gefunden")
        sys.exit(1)

    print(f"Kanonische Version: {canonical}\n")

    ok = True
    for fname in CORE_FILES:
        v = versions.get(fname)
        if v is None:
            print(f"  ❌ {fname}: Keine Version gefunden")
            ok = False
        elif v != canonical:
            print(f"  ❌ {fname}: {v} (erwartet {canonical})")
            ok = False
        else:
            print(f"  ✅ {fname}: {v}")

    if ok:
        print(f"\n✅ Alle {len(CORE_FILES)} Dateien konsistent: {canonical}")
        sys.exit(0)
    else:
        print(f"\n❌ Inkonsistenz gefunden. pyproject.toml = {canonical}")
        print("   Führe 'python scripts/check_version_consistency.py --fix' aus")
        sys.exit(1)


if __name__ == "__main__":
    main()
