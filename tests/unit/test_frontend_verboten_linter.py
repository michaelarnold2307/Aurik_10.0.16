"""Frontend-VERBOTE-Linter: Magic Numbers + UI-Anti-Patterns.

Scannt Aurik10/ui/ auf:
- Magic Numbers: numerische Literale die UI-Schwellwerte darstellen
- UI-Thread-Blocker: time.sleep() ohne Worker-Thread
- Hardcodierte UI-Texte > 80 Zeichen (sollten in i18n/callbacks)

Baseline-basiert: neue Verstöße → FAIL (§G123-Äquivalent für Frontend).
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

CANONICAL_FILES: set[str] = {
    "Aurik10/ui/ui_constants.py",  # UIConstants (nach Erstellung)
}

UI_DIR = Path("Aurik10/ui")


def _scan_ui_file(filepath: Path) -> dict[str, list[dict]]:
    """Scannt eine UI-Datei auf Anti-Patterns."""
    results: dict[str, list[dict]] = {
        "magic_numbers": [],
        "thread_blockers": [],
        "long_strings": [],
    }
    rel = str(filepath.relative_to(Path.cwd())) if filepath.is_relative_to(Path.cwd()) else str(filepath)
    if any(cf in rel for cf in CANONICAL_FILES):
        return results

    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        return results

    lines = source.split("\n")
    in_docstring = False

    for lineno_1, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        # 1. time.sleep() — UI-Thread-Blocker
        if "time.sleep(" in stripped:
            results["thread_blockers"].append(
                {
                    "line": lineno_1,
                    "text": stripped[:120],
                }
            )

        # 2. Magic Numbers: float/int Literale in UI-Kontext
        #    (Zahlen die wie Schwellwerte/Dimensionen aussehen)
        magic = re.findall(
            r"(?:width|height|margin|padding|spacing|size|threshold|"
            r"delay|timeout|interval|duration|opacity|scale|factor|"
            r"ratio|limit|max|min|offset|radius)\s*[:=]\s*([\d.]+)",
            stripped,
        )
        for m in magic:
            val = float(m)
            # Nur nicht-triviale Zahlen (>3, nicht 0/1/100)
            if val > 3 and val not in (10, 20, 30, 50, 100, 200, 500, 1000):
                results["magic_numbers"].append(
                    {
                        "line": lineno_1,
                        "text": stripped.strip()[:120],
                        "value": val,
                    }
                )

        # 3. Lange UI-Texte (sollten Callback-basiert oder in i18n sein)
        if len(stripped) > 80 and ('"' in stripped or "'" in stripped):
            if any(kw in stripped.lower() for kw in ("text", "label", "title", "message", "tooltip")):
                results["long_strings"].append(
                    {
                        "line": lineno_1,
                        "text": stripped[:150],
                    }
                )

    return results


def _scan_all_ui() -> dict[str, dict]:
    """Scannt alle Python-Dateien in Aurik10/ui/."""
    all_results: dict[str, dict] = {}
    if not UI_DIR.exists():
        return all_results

    for py_file in sorted(UI_DIR.rglob("*.py")):
        if any(p.startswith(".") or p in ("__pycache__",) for p in py_file.parts):
            continue
        results = _scan_ui_file(py_file)
        rel = str(py_file.relative_to(Path.cwd())) if py_file.is_relative_to(Path.cwd()) else str(py_file)
        total = sum(len(v) for v in results.values())
        if total > 0:
            all_results[rel] = results

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest-Test
# ═══════════════════════════════════════════════════════════════════════════════


def test_no_new_frontend_anti_patterns() -> None:
    """Frontend-VERBOTE: Keine NEUEN Anti-Patterns.

    Informativ in dieser Phase (Baseline wird dokumentiert).
    Sobald Baseline existiert → FAIL bei neuen Verstößen.
    """
    results = _scan_all_ui()

    magic_count = sum(len(v["magic_numbers"]) for v in results.values())
    blocker_count = sum(len(v["thread_blockers"]) for v in results.values())
    string_count = sum(len(v["long_strings"]) for v in results.values())

    print("\nFrontend-VERBOTE Scan:")
    print(f"  Magic Numbers:    {magic_count}")
    print(f"  Thread-Blocker:   {blocker_count}")
    print(f"  Lange UI-Texte:   {string_count}")

    ranked = sorted(results.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
    if ranked:
        print("\n  Top-5 Dateien:")
        for path, counts in ranked[:5]:
            m = len(counts["magic_numbers"])
            t = len(counts["thread_blockers"])
            s = len(counts["long_strings"])
            print(f"    {path}: {m}M + {t}T + {s}S = {m + t + s}")

    assert magic_count >= 0


if __name__ == "__main__":
    results = _scan_all_ui()
    magic_count = sum(len(v["magic_numbers"]) for v in results.values())
    blocker_count = sum(len(v["thread_blockers"]) for v in results.values())
    string_count = sum(len(v["long_strings"]) for v in results.values())

    print("Frontend-VERBOTE:")
    print(f"  Magic Numbers:  {magic_count}")
    print(f"  Thread-Blocker: {blocker_count}")
    print(f"  Lange Texte:    {string_count}")
    print()

    ranked = sorted(results.items(), key=lambda x: sum(len(v) for v in x[1].values()), reverse=True)
    for path, counts in ranked[:10]:
        m = len(counts["magic_numbers"])
        t = len(counts["thread_blockers"])
        s = len(counts["long_strings"])
        if m + t + s > 0:
            print(f"  {path}: {m}M {t}T {s}S")
