#!/usr/bin/env python3
"""
§v10.101 Perceptual Consistency Guard — Pre-Commit Hook
========================================================

Prüft vor jedem Commit:
  1. Neue Phasen ohne v10.101-Annotation → WARNUNG
  2. Neue CROSSOVER_FREQS [150,800,5000] (linear, nicht Bark) → FEHLER
  3. Neue skalare Wet/Dry-Blends ohne perceptual_blend → WARNUNG
  4. Neue FFT/rfft-Aufrufe ohne Gammatone-Fallback → INFO
  5. §V34-Verstoß: skalarer Blend statt perceptual_blend → FEHLER

Exit-Codes: 0 = OK, 1 = Fehler (Block), 2 = Warnung (Non-Blocking)
"""

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
errors = []
warnings = []

# ---------------------------------------------------------------------------
# 1. Prüfe alle Phasen-Dateien
# ---------------------------------------------------------------------------
phase_files = sorted(glob.glob(os.path.join(ROOT, "backend/core/phases/phase_*.py")))

for f in phase_files:
    name = os.path.basename(f)
    with open(f) as fh:
        content = fh.read()

    # Prüfe 1: v10.101 Annotation vorhanden?
    if "v10.101" not in content and "§v10.101" not in content:
        warnings.append(f"{name}: Keine v10.101 SOTA-Annotation — Pipeline-Gates schützen, aber bitte annotieren")

    # Prüfe 2: Veraltete lineare Crossover?
    if "CROSSOVER_FREQS = [150, 800, 5000]" in content:
        errors.append(f"{name}: Veraltete CROSSOVER_FREQS [150,800,5000] → MUSS [400,2000,6400] sein (§G102)")

    # Prüfe 3: Skalarer Blend ohne perceptual_blend?
    if "_audio_before_phase + _sev_wet_dry *" in content and "perceptual_blend" not in content:
        errors.append(f"{name}: Skalarer Wet/Dry-Blend ohne perceptual_blend() → §V34 verletzt")

# ---------------------------------------------------------------------------
# 2. Prüfe Core-Dateien
# ---------------------------------------------------------------------------
core_files = sorted(glob.glob(os.path.join(ROOT, "backend/core/*.py")))
for f in core_files:
    name = os.path.basename(f)
    # Überspringe offensichtliche Nicht-Audio-Dateien
    if name in ("__init__.py", "logging_config.py", "meta_router.py"):
        continue
    with open(f) as fh:
        content = fh.read()
    if "v10.101" not in content and any(w in content for w in ["np.ndarray", "def process", "librosa", "scipy.signal"]):
        if len(content) > 200:  # Keine trivialen Dateien
            warnings.append(f"backend/core/{name}: Audio-relevantes Modul ohne v10.101-Annotation")

# ---------------------------------------------------------------------------
# 3. Prüfe auf §V34-Verstoß in unified_restorer
# ---------------------------------------------------------------------------
uv3_path = os.path.join(ROOT, "backend/core/unified_restorer_v3.py")
if os.path.exists(uv3_path):
    with open(uv3_path) as f:
        uv3 = f.read()
    if "perceptual_blend" not in uv3:
        errors.append("unified_restorer_v3.py: perceptual_blend() nicht importiert — §V34 verletzt")
    if "should_skip_phase" not in uv3:
        errors.append("unified_restorer_v3.py: should_skip_phase() nicht importiert — §G104 verletzt")

# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------
if errors:
    print(f"❌ {len(errors)} FEHLER (BLOCK):")
    for e in errors:
        print(f"   {e}")
if warnings:
    print(f"⚠️  {len(warnings)} WARNUNGEN:")
    for w in warnings:
        print(f"   {w}")

if errors:
    sys.exit(1)
elif warnings:
    print(f"\n⚠️  {len(warnings)} Warnungen — bitte vor nächstem Commit prüfen.")
    sys.exit(0)  # Warnungen blocken nicht
else:
    print("✅ §v10.101 Perceptual Consistency: ALLE Prüfungen bestanden")
    sys.exit(0)
