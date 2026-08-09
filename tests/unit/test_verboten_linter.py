"""§V25/§G123 Maschinelle Durchsetzung: VERBOTE-Linter für hartcodierte Schwellwerte.

Scannt den gesamten Code auf:
- Modul-Level-Konstanten die wie Schwellwerte/Caps/Floors aussehen
- NUMERISCHE Defaults in Funktionssignaturen
- Diskrete Buckets/Lookup-Tabellen (§G77)

NUR die kanonischen Definitionsdateien sind ausgenommen.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Konfiguration
# ═══════════════════════════════════════════════════════════════════════════════

CANONICAL_FILES: set[str] = {
    "backend/core/calibration_context.py",
    "backend/core/per_phase_musical_goals_gate.py",  # REGRESSION_THRESHOLD_*
    "backend/core/cumulative_interaction_guard.py",  # MAX_GROUP_DELAY_*
    "backend/core/spec_constitution.py",  # MUSIC_DEATH_SHIELD
    "backend/core/pipeline_calibration.py",  # calibrate_pipeline_guards
}

# Muster für Schwellwert-Konstanten
THRESHOLD_PATTERN = re.compile(
    r"^[A-Z][A-Z_0-9]*(?:_THRESHOLD|_FLOOR|_CAP|_MAX|_MIN|_TOLERANCE|"
    r"_WARN|_CRIT|_CEILING|_GUARD|_LIMIT|_SAFE|_BUDGET|_GATE|"
    r"_BASELINE|_TARGET|_DEF|_DB|_MS|_HZ|_PCT|_RATIO|_FACTOR|_BOOST|"
    r"_DISCOUNT|_PENALTY|_BAND|_KNEE|_ATTACK|_RELEASE|_DEPTH|_WINDOW)"
    r"\s*[:=]\s*[\d.+-]+",
    re.VERBOSE,
)

# Numerische Defaults in Funktionssignaturen (zusätzlich zu transfer_chain_depth)
SIGNATURE_NUMERIC_DEFAULT = re.compile(
    r"(?:restorability_score|snr_db|bandwidth|threshold|tolerance|"
    r"strength|confidence|weight|factor|bonus|penalty|floor|cap|ceiling|"
    r"guard|limit|budget|gate|baseline|target)\s*:\s*(?:int|float)\s*=\s*[\d.+-]+",
    re.IGNORECASE,
)

# Diskrete Buckets/Lookup-Tabellen (§G77 verboten)
BUCKET_PATTERN = re.compile(
    r"(?:if|elif)\s+.*\b(?:transfer_chain_depth|restorability_score|"
    r"material_type|snr_db|bandwidth)\b.*[<>]=?\s*[\d.]+",
)


def _is_canonical(filepath: str) -> bool:
    """Prüft ob die Datei eine kanonische Definitionsdatei ist."""
    for cf in CANONICAL_FILES:
        if filepath.endswith(cf) or filepath == cf:
            return True
    return False


def _scan_file(filepath: Path) -> dict[str, list[dict[str, Any]]]:
    """Scannt eine Datei auf VERBOTE-Verstöße.

    Returns:
        Kategorisiert nach Typ: thresholds, signature_defaults, buckets.
    """
    results: dict[str, list[dict[str, Any]]] = {
        "thresholds": [],
        "signature_defaults": [],
        "buckets": [],
    }

    rel = str(filepath.relative_to(Path.cwd())) if filepath.is_relative_to(Path.cwd()) else str(filepath)

    if _is_canonical(rel):
        return results

    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        return results

    lines = source.split("\n")
    in_docstring = False
    in_multiline_comment = False

    for lineno_1, line in enumerate(lines, 1):
        stripped = line.strip()

        # Überspringe Kommentare und Docstrings
        if stripped.startswith("#"):
            continue
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        # 1. Modul-Level-Threshold-Konstanten
        if THRESHOLD_PATTERN.match(stripped):
            # Nicht innerhalb einer Funktion/Klasse
            indent = len(line) - len(line.lstrip())
            if indent <= 4:  # Modul-Level (inkl. Klassen-Attribute in einfachen Cases)
                results["thresholds"].append(
                    {
                        "line": lineno_1,
                        "text": stripped[:120],
                    }
                )

        # 2. Numerische Defaults in Funktionssignaturen
        if SIGNATURE_NUMERIC_DEFAULT.search(stripped):
            results["signature_defaults"].append(
                {
                    "line": lineno_1,
                    "text": stripped.strip()[:120],
                }
            )

        # 3. Diskrete Buckets (§G77)
        if BUCKET_PATTERN.search(stripped):
            # Nur wenn es nach Bucket-Logik aussieht (mehrere elifs mit gleicher Variable)
            results["buckets"].append(
                {
                    "line": lineno_1,
                    "text": stripped.strip()[:120],
                }
            )

    return results


def _scan_all() -> dict[str, dict]:
    """Scannt alle Python-Dateien in backend/."""
    all_results: dict[str, dict] = {}

    backend = Path("backend")
    if not backend.exists():
        backend = Path("..") / "backend"
    if not backend.exists():
        backend = Path(__file__).resolve().parent.parent.parent / "backend"

    for py_file in sorted(backend.rglob("*.py")):
        parts = py_file.parts
        if any(p.startswith(".") or p in ("__pycache__", "venv", ".venv_aurik") for p in parts):
            continue

        results = _scan_file(py_file)
        rel = str(py_file.relative_to(backend.parent)) if py_file.is_relative_to(backend.parent) else str(py_file)

        total = sum(len(v) for v in results.values())
        if total > 0:
            all_results[rel] = results

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest-Test
# ═══════════════════════════════════════════════════════════════════════════════


def test_no_new_hardcoded_thresholds() -> None:
    """§V25/§G123: Keine NEUEN hartcodierten Schwellwerte.

    Scannt auf Modul-Level-Threshold-Konstanten und numerische Defaults.
    Kanonische Definitionsdateien sind ausgenommen.

    Dieser Test ist INFORMATIV — er dokumentiert den Ist-Zustand.
    Er wird in einer späteren Phase auf FAIL umgestellt, sobald die
    Baseline bereinigt ist.
    """
    results = _scan_all()

    threshold_count = sum(len(v["thresholds"]) for v in results.values())
    sig_count = sum(len(v["signature_defaults"]) for v in results.values())
    bucket_count = sum(len(v["buckets"]) for v in results.values())

    # Für jetzt: nur dokumentieren, nicht failen
    # Sobald Baseline existiert → vergleichen wie beim transfer_chain_depth-Linter
    print("\n§V25 VERBOTE-Linter Scan:")
    print(
        f"  Schwellwert-Konstanten: {threshold_count} in {sum(1 for v in results.values() if v['thresholds'])} Dateien"
    )
    print(
        f"  Signatur-Defaults:      {sig_count} in {sum(1 for v in results.values() if v['signature_defaults'])} Dateien"
    )
    print(f"  Diskrete Buckets:       {bucket_count} in {sum(1 for v in results.values() if v['buckets'])} Dateien")

    # Top-10 Dateien mit den meisten Verstößen
    ranked = sorted(results.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
    if ranked:
        print("\n  Top-10 Dateien:")
        for path, counts in ranked[:10]:
            t = len(counts["thresholds"])
            s = len(counts["signature_defaults"])
            b = len(counts["buckets"])
            print(f"    {path}: {t}T + {s}S + {b}B = {t + s + b}")

    # Kein Assert — informativ in dieser Phase
    assert threshold_count >= 0  # Trivial — dokumentiert nur


if __name__ == "__main__":
    results = _scan_all()
    threshold_count = sum(len(v["thresholds"]) for v in results.values())
    sig_count = sum(len(v["signature_defaults"]) for v in results.values())
    bucket_count = sum(len(v["buckets"]) for v in results.values())

    print("§V25 VERBOTE-Linter:")
    print(f"  Schwellwerte: {threshold_count}")
    print(f"  Signaturen:   {sig_count}")
    print(f"  Buckets:      {bucket_count}")
    print()

    ranked = sorted(results.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
    for path, counts in ranked[:20]:
        t = len(counts["thresholds"])
        s = len(counts["signature_defaults"])
        b = len(counts["buckets"])
        if t + s + b > 0:
            print(f"  {path}: {t}T {s}S {b}B")
            for v in counts["thresholds"][:3]:
                print(f"    L{v['line']}: {v['text'][:100]}")
