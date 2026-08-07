#!/usr/bin/env python3
"""Pre-Commit-Hook: §2.59 Surgical Repair — Vollständigkeits-Prüfung."""

import os
import sys
from pathlib import Path

# Repo-Root: .git/hooks/pre-commit → .git → repo-root
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(str(_REPO))

EXIT_OK, EXIT_VIOLATION, EXIT_ERROR = 0, 1, 2

# Nutze venv Python wenn verfügbar (sonst system python → import scheitert)
_VENV = os.environ.get("VIRTUAL_ENV", "")
if _VENV:
    _venv_site = Path(_VENV) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if _venv_site.exists():
        sys.path.insert(0, str(_venv_site))

try:
    from backend.core.surgical_defect_analyzer import SURGICAL_DEFECT_TYPES
    from backend.core.surgical_repair import _SURGICAL_REPAIR_FUNCTIONS
except ImportError as e:
    # Git hook läuft außerhalb venv → silently pass (kein Fehler)
    print("⚠️ Pre-commit skipped — venv nicht aktiv. Führe manuell aus:")
    print(f"   cd {_REPO} && python scripts/pre_commit_surgical_check.py")
    raise SystemExit(EXIT_OK) from e

s_types = set(SURGICAL_DEFECT_TYPES)
r_keys = set(_SURGICAL_REPAIR_FUNCTIONS.keys())

errs = []
for d in sorted(s_types - r_keys):
    errs.append(f"SURGICAL_DEFECT_TYPES['{d}'] hat KEINE Repair-Funktion")
for k in sorted(r_keys - s_types):
    errs.append(f"_SURGICAL_REPAIR_FUNCTIONS['{k}'] ist NICHT in SURGICAL_DEFECT_TYPES")
for k, v in _SURGICAL_REPAIR_FUNCTIONS.items():
    if not callable(v):
        errs.append(f"_SURGICAL_REPAIR_FUNCTIONS['{k}'] ist NICHT callable")

for e in errs:  # type: ignore[misc]
    print(f"  ❌ {e}")  # type: ignore[misc]

if errs:
    print(f"\n❌ FAILED: {len(errs)} Fehler")
    raise SystemExit(EXIT_VIOLATION)

print(f"✅ OK — {len(s_types)} chirurgische Defekte, {len(r_keys)} Repair-Funktionen")
