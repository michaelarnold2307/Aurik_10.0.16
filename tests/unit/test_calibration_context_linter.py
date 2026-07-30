"""§G86 Maschinelle Durchsetzung: Linter-Test für CalibrationContext-Verstöße.

Scannt den gesamten Code auf verbotene Default-Muster und vergleicht
gegen eine Baseline-Datei. Neue Verstöße lassen den Test fehlschlagen.

Baseline: tests/unit/calibration_context_linter_baseline.txt
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


FORBIDDEN_DEFAULTS: dict[str, list[str]] = {
    "transfer_chain_depth": [
        "transfer_chain_depth: int = 1",
        "transfer_chain_depth: int=1",
        'transfer_chain_depth: int = 1,',
        'transfer_chain_depth: int=1,',
    ],
}

CANONICAL_FILE = "backend/core/calibration_context.py"


def _scan_all() -> dict[str, list[int]]:
    """Scannt backend/ auf verbotene Defaults.

    Returns:
        {dateipfad: [zeilennummern]} aller Verstöße.
    """
    violations: dict[str, list[int]] = {}

    backend = Path("backend")
    if not backend.exists():
        backend = Path("..") / "backend"
    if not backend.exists():
        backend = Path(__file__).resolve().parent.parent.parent / "backend"

    for py_file in sorted(backend.rglob("*.py")):
        parts = py_file.parts
        if any(p.startswith(".") or p in ("__pycache__", "venv", ".venv_aurik") for p in parts):
            continue

        rel = str(py_file.relative_to(backend.parent)) if py_file.is_relative_to(backend.parent) else str(py_file)
        if rel == CANONICAL_FILE:
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = source.split("\n")
        for lineno_1, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for _param, patterns in FORBIDDEN_DEFAULTS.items():
                for pattern in patterns:
                    if pattern in stripped:
                        violations.setdefault(rel, []).append(lineno_1)
                        break

    return violations


def _baseline_path() -> Path:
    return Path(__file__).resolve().parent / "calibration_context_linter_baseline.txt"


def _compute_hash(violations: dict[str, list[int]]) -> str:
    """Deterministischer Hash über alle Verstöße."""
    items = sorted(violations.items())
    text = "\n".join(f"{path}:{','.join(str(l) for l in sorted(lines))}" for path, lines in items)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _baseline_violations() -> dict[str, list[int]]:
    """Liest die Baseline-Datei ein."""
    bp = _baseline_path()
    if not bp.exists():
        return {}
    result: dict[str, list[int]] = {}
    for line in bp.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        path, linenos = line.split(":", 1)
        result[path] = [int(n) for n in linenos.split(",") if n.strip()]
    return result


def test_calibration_context_linter_no_new_violations() -> None:
    """§G86: Keine NEUEN verbotenen Defaults seit letzter Baseline.

    Vergleicht den aktuellen Scan gegen die gespeicherte Baseline.
    Neue Verstöße (nicht in der Baseline) → FAIL.
    Behobene Verstöße (in Baseline aber nicht mehr im Code) → WARNING im Log.

    Um die Baseline zu aktualisieren:
        python tests/unit/test_calibration_context_linter.py --update-baseline
    """
    current = _scan_all()
    baseline = _baseline_violations()

    new_violations: dict[str, list[int]] = {}
    fixed_violations: dict[str, list[int]] = {}

    for path, lines in current.items():
        base_lines = set(baseline.get(path, []))
        new = sorted(set(lines) - base_lines)
        if new:
            new_violations[path] = new

    for path, lines in baseline.items():
        curr_lines = set(current.get(path, []))
        fixed = sorted(curr_lines - set(lines))  # fixed = in baseline but not current
        if fixed:
            fixed_violations[path] = fixed

    if fixed_violations:
        import logging
        logger = logging.getLogger(__name__)
        for path, lines in sorted(fixed_violations.items()):
            logger.warning("§G86 Behoben: %s (Zeilen %s) — Baseline sollte aktualisiert werden",
                           path, ",".join(str(l) for l in lines))

    if new_violations:
        msg_parts = [f"\n§G86 LINTER: {sum(len(v) for v in new_violations.values())} NEUE verbotene Defaults!\n"]
        for path, lines in sorted(new_violations.items()):
            for line in lines:
                msg_parts.append(f"  {path}:{line}  transfer_chain_depth: int = 1  ← VERBOTEN (§G86)")
        msg_parts.append(
            f"\n  Baseline: {len(sum(baseline.values(), []))} bekannte Verstöße"
            f"\n  Aktuell:  {len(sum(current.values(), []))} Verstöße"
            f"\n  Neu:      {sum(len(v) for v in new_violations.values())}"
            f"\n\n  Abhilfe: Statt 'transfer_chain_depth: int = 1' den Parameter"
            f"\n  aus dem CalibrationContext beziehen oder explizit übergeben."
            f"\n  Der Default darf NUR in calibration_context.py stehen."
        )
        raise AssertionError("\n".join(msg_parts))

    # Test passed — log summary
    current_count = sum(len(v) for v in current.values())
    baseline_count = sum(len(v) for v in baseline.values())
    print(f"§G86 Linter: {current_count} bekannte Defaults (Baseline: {baseline_count}), 0 neue — OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CalibrationContext Linter")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Baseline-Datei aus aktuellem Scan generieren")
    parser.add_argument("--show", action="store_true",
                        help="Alle aktuellen Verstöße anzeigen")
    args = parser.parse_args()

    current = _scan_all()

    if args.update_baseline:
        bp = _baseline_path()
        lines = []
        for path, linenos in sorted(current.items()):
            lines.append(f"{path}:{','.join(str(l) for l in sorted(linenos))}")
        lines.append(f"# Baseline Hash: {_compute_hash(current)}")
        lines.append(f"# {sum(len(v) for v in current.values())} Verstöße in {len(current)} Dateien")
        bp.write_text("\n".join(lines) + "\n")
        print(f"Baseline aktualisiert: {bp}")
        print(f"  {sum(len(v) for v in current.values())} Verstöße in {len(current)} Dateien")

    elif args.show:
        for path, linenos in sorted(current.items()):
            print(f"{path}:{','.join(str(l) for l in sorted(linenos))}")

    else:
        # Nur Scan, kein Test
        print(f"Linter Scan: {sum(len(v) for v in current.values())} Verstöße in {len(current)} Dateien")
        print("Verwende --update-baseline zum Erstellen der Baseline")
        print("Verwende --show zur Anzeige aller Verstöße")
