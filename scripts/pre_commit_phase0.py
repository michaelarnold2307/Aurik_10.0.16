#!/usr/bin/env python3
"""§v10.303.26 Pre-Commit-Hooks für Phase-0-Integrität.

Prüft vor jedem Commit:
  1. ChainedPhase0Preprocessor importiert sauber
  2. Cache-Schreib-/Lese-Funktion arbeitet korrekt
  3. Hallucination-Guard-Schwellen im definierten Bereich
  4. Keine Syntax-Fehler in Phase-0-Quelldateien

Installation:
  ln -s ../../scripts/pre_commit_phase0.py .git/hooks/pre-commit
  oder: pre-commit install (mit .pre-commit-config.yaml)

Usage:
  python scripts/pre_commit_phase0.py [--staged-only]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Konfiguration ──────────────────────────────────────────────────────

_PHASE0_FILES = [
    "plugins/apollo_phase0_integration.py",
    "plugins/phase0_goal_cache.py",
    "plugins/apollo_plugin.py",
    "plugins/breath_detector.py",
    "plugins/resemble_enhance_plugin.py",
    "plugins/deepfilternet_v3_ii_plugin.py",
]

_GUARD_THRESHOLD_RANGES = {
    "apollo": (0.05, 0.60),
    "deepfilternet": (0.10, 0.80),
    "resemble_enhance": (0.10, 0.70),
}

_CACHE_DIR = os.path.expanduser("~/.aurik/cache/phase0")


def _check_syntax(filepath: str) -> bool:
    """Prüft Python-Syntax einer Datei."""
    try:
        import py_compile

        py_compile.compile(filepath, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print(f"  ❌ SYNTAX-FEHLER in {filepath}: {e}")
        return False


def _check_import() -> bool:
    """Prüft ob ChainedPhase0Preprocessor importierbar ist."""
    try:
        from plugins.apollo_phase0_integration import (
            ApolloPhase0Guard,
            ApolloResult,
            ChainedPhase0Preprocessor,
            DeepFilterNetGuard,
            ResembleEnhanceGuard,
        )

        return True
    except ImportError as e:
        print(f"  ❌ IMPORT-FEHLER: {e}")
        return False
    except SyntaxError as e:
        print(f"  ❌ SYNTAX-FEHLER beim Import: {e}")
        return False


def _check_guard_thresholds() -> bool:
    """Prüft ob Hallucination-Guard-Schwellen im definierten Bereich sind."""
    try:
        from plugins.apollo_phase0_integration import (
            ApolloPhase0Guard,
            DeepFilterNetGuard,
            ResembleEnhanceGuard,
        )

        _checks = {
            "apollo": ApolloPhase0Guard()._hallucination_threshold,
            "deepfilternet": DeepFilterNetGuard()._threshold,
            "resemble_enhance": ResembleEnhanceGuard()._threshold,
        }
        _ok = True
        for _name, _val in _checks.items():
            _lo, _hi = _GUARD_THRESHOLD_RANGES[_name]
            if not (_lo <= _val <= _hi):
                print(f"  ❌ GUARD-SCHWELLE {_name}={_val} außerhalb [{_lo}, {_hi}]")
                _ok = False
        return _ok
    except Exception as e:
        print(f"  ❌ GUARD-CHECK fehlgeschlagen: {e}")
        return False


def _check_cache_integrity() -> bool:
    """Prüft ob Cache-Verzeichnis existiert und beschreibbar ist."""
    try:
        import numpy as np

        os.makedirs(_CACHE_DIR, exist_ok=True)
        _test_file = os.path.join(_CACHE_DIR, "_precommit_test.npz")
        _test_data = np.zeros(100, dtype=np.float32)
        np.savez_compressed(_test_file, audio=_test_data)
        _loaded = np.load(_test_file)
        _ok = np.array_equal(_loaded["audio"], _test_data)
        os.unlink(_test_file)
        if not _ok:
            print("  ❌ CACHE-INTEGRITÄT: Schreiben/Lesen inkonsistent")
        return _ok
    except Exception as e:
        print(f"  ❌ CACHE-CHECK fehlgeschlagen: {e}")
        return False


def _check_phase0_files_syntax() -> bool:
    """Prüft Syntax aller Phase-0-Quelldateien."""
    _ok = True
    for _f in _PHASE0_FILES:
        _path = os.path.join(_PROJECT_ROOT, _f)
        if os.path.exists(_path) and not _check_syntax(_path):
            _ok = False
    return _ok


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    """Führt alle Checks aus. Returns 0 bei Erfolg, 1 bei Fehler."""
    print("=" * 60)
    print("Phase-0 Pre-Commit Integrity Check")
    print("=" * 60)

    _checks = [
        ("Syntax (Phase-0 files)", _check_phase0_files_syntax),
        ("Import (ChainedPhase0Preprocessor)", _check_import),
        ("Guard-Schwellen", _check_guard_thresholds),
        ("Cache-Integrität", _check_cache_integrity),
    ]

    _failures = 0
    for _name, _fn in _checks:
        print(f"\n📋 {_name}...")
        try:
            if _fn():
                print("  ✅ PASS")
            else:
                _failures += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            _failures += 1

    print("\n" + "=" * 60)
    if _failures == 0:
        print("✅ ALLE PHASE-0 CHECKS BESTANDEN")
        return 0
    else:
        print(f"❌ {_failures} CHECK(S) FEHLGESCHLAGEN — Commit verweigert")
        return 1


if __name__ == "__main__":
    sys.exit(main())
