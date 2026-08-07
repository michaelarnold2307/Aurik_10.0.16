#!/usr/bin/env python3
"""Audit: findet `try: from X import Y ... except ImportError/Exception`-Blöcke,
bei denen `Y` in Modul `X` (repo-intern) gar nicht existiert.

Hintergrund: `PANNSPlugin` (falsche Schreibweise, tatsächlich `PANNsPlugin`) in
backend/core/optimization/perceptual_loss.py führte dazu, dass ein kompletter
Loss-Zweig PERMANENT über den Exception-Fallback lief, ohne dass Tests/mypy
das je bemerkt hätten (ImportError wird immer gefangen, Fallback sieht wie
regulärer Betrieb aus). Dieses Skript sucht repo-weit nach demselben Muster.

Beschränkung: prüft nur *repo-interne* Module (auflösbar über Dateipfad).
Externe Pakete (torch, numpy, ...) werden übersprungen, da deren Attribute
nicht statisch über den Dateibaum ermittelbar sind.

Nutzung:
    python scripts/audit_silent_dead_imports.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIR_PARTS = {".venv_aurik", "node_modules", ".git", "__pycache__", ".mypy_cache", ".ruff_cache"}

_CATCH_NAMES = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _module_file(module_name: str) -> Path | None:
    """Löst einen absoluten Modulnamen (z. B. 'plugins.panns_plugin') auf eine Datei im Repo auf."""
    parts = module_name.split(".")
    as_file = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    if as_file.is_file():
        return as_file
    as_pkg = REPO_ROOT.joinpath(*parts, "__init__.py")
    if as_pkg.is_file():
        return as_pkg
    return None


def _top_level_names(path: Path) -> set[str] | None:
    """Sammelt alle auf Modul-Ebene importierbaren Namen (Klassen, Funktionen, Zuweisungen, Re-Exports)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None

    names: set[str] = set()

    def _collect(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Tuple):
                        for e in t.elts:
                            if isinstance(e, ast.Name):
                                names.add(e.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.If):
                # z. B. `if TYPE_CHECKING:` / `try/except`-Ersatzpfade auf Modulebene
                _collect(node.body)
                _collect(node.orelse)
            elif isinstance(node, ast.Try):
                _collect(node.body)
                for h in node.handlers:
                    _collect(h.body)
                _collect(node.orelse)
                _collect(node.finalbody)

    _collect(tree.body)
    return names


def _catches_import_error(handlers: list[ast.ExceptHandler]) -> bool:
    for h in handlers:
        if h.type is None:
            return True  # bare except
        if isinstance(h.type, ast.Name) and h.type.id in _CATCH_NAMES:
            return True
        if isinstance(h.type, ast.Tuple) and any(isinstance(e, ast.Name) and e.id in _CATCH_NAMES for e in h.type.elts):
            return True
    return False


def scan_file(path: Path, findings: list[tuple[Path, int, str, str]]) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not _catches_import_error(node.handlers):
            continue

        for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
                mod_path = _module_file(stmt.module)
                if mod_path is None:
                    continue  # externes Paket — nicht statisch pruefbar
                target_names = _top_level_names(mod_path)
                if target_names is None:
                    continue
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    if alias.name not in target_names:
                        findings.append((path, stmt.lineno, stmt.module, alias.name))


def main() -> int:
    findings: list[tuple[Path, int, str, str]] = []
    for py_file in REPO_ROOT.rglob("*.py"):
        if any(part in SKIP_DIR_PARTS for part in py_file.parts):
            continue
        scan_file(py_file, findings)

    if not findings:
        print(
            "Keine Treffer — keine repo-internen Import-Namen-Mismatches in try/except-ImportError-Bloecken gefunden."
        )
        return 0

    print(f"{len(findings)} verdaechtige Fundstelle(n):\n")
    for path, lineno, module, name in sorted(findings, key=lambda f: str(f[0])):
        rel = path.relative_to(REPO_ROOT)
        print(f"  {rel}:{lineno}: `{name}` nicht als Top-Level-Name in Modul `{module}` gefunden")
    return 1


if __name__ == "__main__":
    sys.exit(main())
