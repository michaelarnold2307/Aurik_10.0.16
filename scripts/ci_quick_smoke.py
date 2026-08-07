#!/usr/bin/env python3
"""scripts/ci_quick_smoke.py — §v10.700 CI Quick Smoke Test.

Schnellster möglicher Health-Check (~5s) für CI-Pipelines.
Testet: Import-Kette, numpy/scipy-Verfügbarkeit, deterministischen Dummy-Run.

Exit 0 = OK, Exit 1 = Fehler.
"""

from __future__ import annotations

import sys
import time


def _check_imports() -> int:
    """Minimale Import-Kette für Aurik Core."""
    errors = 0
    modules = [
        ("numpy", "np"),
        ("scipy.signal", None),
        ("backend.core.unified_restorer_v3", "UnifiedRestorerV3"),
        ("backend.core.defect_manifest", "get_defect_manifest"),
        ("backend.core.safe_dict", "SafeDict"),
    ]
    for module, attr in modules:
        try:
            mod = __import__(module, fromlist=[attr] if attr else [])
            if attr:
                getattr(mod, attr)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            errors += 1
    return errors


def _check_numpy_invariants() -> int:
    """Grundlegende numerische Invarianten."""
    import numpy as np

    rng = np.random.RandomState(42)
    audio = rng.randn(48000).astype(np.float32) * 0.1

    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()
    assert -1.0 < audio.min() < audio.max() < 1.0
    assert -0.5 < audio.mean() < 0.5
    print("  ✅ numpy invariants")
    return 0


def main() -> int:
    print("CI Quick Smoke Test")
    print("=" * 40)
    t0 = time.monotonic()
    errors = 0

    print("\n1. Imports:")
    errors += _check_imports()

    print("\n2. Numeric invariants:")
    try:
        errors += _check_numpy_invariants()
    except Exception as e:
        print(f"  ❌ {e}")
        errors += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 40}")
    if errors == 0:
        print(f"✅ PASS ({elapsed:.1f}s)")
        return 0
    else:
        print(f"❌ {errors} ERROR(S) ({elapsed:.1f}s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
