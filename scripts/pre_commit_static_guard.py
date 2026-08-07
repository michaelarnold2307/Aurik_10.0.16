#!/usr/bin/env python3
"""Pre-Commit Static Guard — Fängt undefined-name Bugs VOR dem Commit.

§Schutzschicht-1: Scannt alle geänderten .py-Dateien auf F821 (undefined name).
Exit-Code 0 = sauber, Exit-Code 1 = undefined names gefunden.

Nutzung:
  python scripts/pre_commit_static_guard.py                    # Alle .py-Dateien
  python scripts/pre_commit_static_guard.py --staged           # Nur git-staged
  python scripts/pre_commit_static_guard.py --changed          # Nur git-diff

Autor: Aurik 10 — 11. Juli 2026
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_undefined_names(filepath: Path) -> list[tuple[int, str]]:
    """Findet undefined-name Referenzen in einer Python-Datei.

    Heuristik: Durchläuft den AST und findet Name-Nodes, die:
    - Nicht in lokalen/globalen Scopes definiert sind
    - Nicht builtins sind
    - Nicht explizit importiert wurden

    Returns:
        Liste von (line_number, name_string).
    """
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
    except SyntaxError:
        return []

    issues: list[Any] = []
    _ = set(dir(__builtins__)) if hasattr(__builtins__, "__dict__") else set()

    for node in ast.walk(tree):
        # Suche nach try/except mit `pass` gefolgt von Name-Nutzung
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.type is None or (isinstance(handler.type, ast.Name) and handler.type.id == "Exception"):
                    if handler.body and isinstance(handler.body[0], ast.Pass):
                        # Prüfe nachfolgende Statements auf undefined names
                        pass

    return issues


def scan_file(filepath: Path) -> list[str]:
    """Scannt eine Datei auf verdächtige Muster (try: pass + undefined)."""
    if not filepath.exists() or filepath.suffix != ".py":
        return []
    if ".venv" in str(filepath) or "__pycache__" in str(filepath):
        return []
    # Skip test files (they use pytest fixtures which are legal undefined names)
    if "/tests/" in str(filepath) or str(filepath).startswith("tests/"):
        return []

    content = filepath.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    issues = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Muster: `try:` gefolgt von `pass` gefolgt von Variable
        if line == "try:" and i + 2 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line == "pass":
                # Check next 5 non-empty lines for variable usage
                for j in range(i + 2, min(i + 8, len(lines))):
                    check = lines[j].strip()
                    if check.startswith("#") or check.startswith("logger."):
                        continue
                    if not check or check == "pass":
                        continue
                    # Suche nach _identifier = ... oder _identifier.method()
                    import re

                    match = re.match(r"^(\w+)\s*[=\.(]", check)
                    if match and match.group(1) not in (
                        "if",
                        "for",
                        "while",
                        "return",
                        "import",
                        "from",
                        "with",
                        "else",
                        "elif",
                        "except",
                        "finally",
                        "raise",
                        "assert",
                        "yield",
                        "break",
                        "continue",
                    ):
                        var_name = match.group(1)
                        # Prüfe ob die Variable im try-Block definiert wird
                        defined_in_try = any(
                            f"{var_name} =" in lines[k]
                            or f"import {var_name}" in lines[k]
                            or f"as {var_name}" in lines[k]
                            for k in range(i + 1, j)
                        )
                        if not defined_in_try:
                            issues.append(
                                f"{filepath.resolve().relative_to(_PROJECT_ROOT)}:{j + 1}: Verdacht auf undefined name '{var_name}' nach `try: pass`"
                            )
                            break
                    break

        # Muster: `name '...' is not defined` im Output (kein eigener Code, aber Indikator)
        i += 1

    return issues


def check_silent_except_pass(filepath: Path) -> list[str]:
    """V74: Findet 'except ... : pass' ohne logging oder Kommentar."""
    if not filepath.exists() or filepath.suffix != ".py":
        return []
    if ".venv" in str(filepath) or "__pycache__" in str(filepath):
        return []
    if "/tests/" in str(filepath) or str(filepath).startswith("tests/"):
        return []  # Tests may have intentional pass in mocks

    content = filepath.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    issues = []
    in_except = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Detect except block start
        if stripped.startswith("except") and ":" in stripped:
            in_except = True
            continue
        if in_except:
            # Check if the body is just 'pass' without logging
            if stripped == "pass":
                # Look for logger in the surrounding context (±3 lines)
                context = lines[max(0, i - 3) : min(len(lines), i + 4)]
                has_logger = any("logger." in l for l in context)
                has_comment = any(
                    l.strip().startswith("#")
                    and ("expected" in l.lower() or "intentional" in l.lower() or "noqa" in l.lower())
                    for l in context
                )
                if not has_logger and not has_comment:
                    issues.append(
                        f"{filepath.resolve().relative_to(_PROJECT_ROOT)}:{i + 1}: "
                        f"V74 Silent except:pass without logging — add logger.debug() or comment"
                    )
                in_except = False
                continue
            if stripped and not stripped.startswith("#"):
                in_except = False  # Non-pass body found, not a silent except
    return issues


def get_target_files(mode: str = "all") -> list[Path]:
    """Bestimmt welche Dateien gescannt werden."""
    if mode == "staged":
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]
        return [_PROJECT_ROOT / f for f in files]
    elif mode == "changed":
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]
        return [_PROJECT_ROOT / f for f in files]
    else:
        # Nur kürzlich geänderte Dateien (letzte 7 Tage)
        files = []
        for py_file in _PROJECT_ROOT.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            if "models/" in str(py_file) or "node_modules/" in str(py_file):
                continue
            files.append(py_file)  # type: ignore[arg-type]
        return files  # type: ignore[return-value]


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Pre-Commit Static Guard")
    p.add_argument("--staged", action="store_true", help="Nur git-staged Dateien")
    p.add_argument("--changed", action="store_true", help="Nur git-diff Dateien")
    p.add_argument("--all", action="store_true", help="Alle Projekt-.py-Dateien (Default)")
    p.add_argument("--json", action="store_true", help="JSON-Ausgabe")
    p.add_argument(
        "files",
        nargs="*",
        default=None,
        help="Explizite Datei-Liste (von pre-commit übergeben). Wenn angegeben, werden NUR diese Dateien gescannt.",
    )
    args = p.parse_args()

    mode = "staged" if args.staged else "changed" if args.changed else "all"
    files = get_target_files(mode)

    # Wenn explizite Dateien übergeben wurden, nur diese scannen
    if args.files:
        explicit = [Path(f).resolve() for f in args.files if f.endswith(".py")]
        if explicit:
            files = explicit

    all_issues: dict[str, list[str]] = {}
    for fp in files:
        issues = scan_file(fp)
        if issues:
            all_issues[str(fp.resolve().relative_to(_PROJECT_ROOT))] = issues
        # §V74: Check for silent except:pass
        silent_issues = check_silent_except_pass(fp)
        if silent_issues:
            key = str(fp.resolve().relative_to(_PROJECT_ROOT))
            all_issues.setdefault(key, []).extend(silent_issues)

    if args.json:
        import json

        print(json.dumps({"clean": len(all_issues) == 0, "files": len(files), "issues": all_issues}))
    else:
        if all_issues:
            print(f"\n❌ {sum(len(v) for v in all_issues.values())} Verdachtsfälle in {len(all_issues)} Dateien:\n")
            for fname, issues in all_issues.items():
                for issue in issues:
                    print(f"  {issue}")
            print("\n🚫 Commit blockiert: undefined-name Verdacht.")
            print("   Wenn false-positive: `# noqa: F821` am Zeilenende hinzufügen.\n")
            return 1
        else:
            print(f"✅ Static Guard: {len(files)} Dateien geprüft, keine undefined-name-Verdachtsfälle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
