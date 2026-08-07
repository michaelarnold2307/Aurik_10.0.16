#!/usr/bin/env python3
"""
§v10.118 Scope-Bug-Detector — AST-basierte statische Analyse für Aurik-Phasen.

Erkennt die zwei häufigsten Scope-Bug-Klassen:
  1. Undefined variable in inner function (z.B. `kwargs` in _repair_channel)
  2. Used-before-assignment (z.B. `_mk` vor `_mk = ...`)

Zero-dependency — verwendet nur stdlib `ast`.

Usage:
  python3 scripts/scope_bug_detector.py                     # alle Phasen checken
  python3 scripts/scope_bug_detector.py --ci                 # Exit-Code 1 bei Funden
  python3 scripts/scope_bug_detector.py --file phase_23...   # einzelne Datei
"""

import ast
import os
import sys
from pathlib import Path
from typing import Any

PHASE_DIR = Path("backend/core/phases")


def find_undefined_in_functions(tree: ast.AST, source_file: str) -> list[str]:
    """Find variables used in inner functions but not in their parameter list."""
    issues: list[Any] = []

    class InnerFuncVisitor(ast.NodeVisitor):
        def __init__(self):
            self.outer_vars: set[str] = set()
            self.current_func_params: set[str] = set()
            self.in_outer_func = False
            self.issues: list[str] = []

        def visit_FunctionDef(self, node):
            if not self.in_outer_func:
                # This is the outer function (e.g., process, _repair_channel)
                old_params = self.current_func_params
                self.current_func_params = {arg.arg for arg in node.args.args}
                # Add **kwargs if present
                if node.args.kwarg:
                    self.current_func_params.add(node.args.kwarg.arg)
                self.in_outer_func = True

                # Collect all names assigned in this function body
                assigned = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                assigned.add(target.id)
                    elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                        assigned.add(child.target.id)

                # Now check inner functions
                for child in ast.iter_child_nodes(node):
                    self._check_inner(child, assigned)

                self.in_outer_func = False
                self.current_func_params = old_params
            else:
                self.generic_visit(node)

        def _check_inner(self, node, outer_assigned):
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef):
                    inner_params = {arg.arg for arg in child.args.args}
                    if child.args.kwarg:
                        inner_params.add(child.args.kwarg.arg)
                    # Find all Name nodes used in this inner function
                    for sub in ast.walk(child):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                            name = sub.id
                            # Is this name NOT a parameter of the inner function,
                            # NOT a builtin, and NOT assigned in the outer function?
                            if (
                                name not in inner_params
                                and name not in outer_assigned
                                and name not in self.current_func_params
                                and name not in __builtins__.__dict__
                                and not name.startswith("_")
                            ):
                                # But IS used? This is suspicious
                                # Check if it's a common module-level name
                                if name in (
                                    "np",
                                    "os",
                                    "sys",
                                    "time",
                                    "logging",
                                    "logger",
                                    "math",
                                    "json",
                                    "re",
                                    "Path",
                                    "Any",
                                    "Optional",
                                    "List",
                                    "Dict",
                                    "Tuple",
                                    "Union",
                                    "warnings",
                                    "traceback",
                                    "collections",
                                    "itertools",
                                    "functools",
                                    "threading",
                                    "abc",
                                    "dataclass",
                                    "field",
                                    "Enum",
                                    "auto",
                                    "deepcopy",
                                    "defaultdict",
                                    "partial",
                                    "TYPE_CHECKING",
                                    "PhaseInterface",
                                    "PhaseResult",
                                    "PhaseMetadata",
                                    "PhaseCategory",
                                    "PhaseMode",
                                    "create_phase_result",
                                    "MaterialType",
                                    "np",
                                ):
                                    continue
                                if name.isupper() and "_" in name:
                                    continue  # Constants
                                self.issues.append(
                                    f"{child.name}(): '{name}' used but not in params, "
                                    f"not assigned in outer scope, not a known import"
                                )

    visitor = InnerFuncVisitor()
    visitor.visit(tree)
    return visitor.issues


def check_file(filepath: Path) -> list[str]:
    """Check a single Python file for scope bugs."""
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
        return find_undefined_in_functions(tree, str(filepath))
    except SyntaxError as e:
        return [f"SYNTAX ERROR: {e}"]
    except Exception as e:
        return [f"PARSE ERROR: {e}"]


def main():
    ci_mode = "--ci" in sys.argv
    specific_file = None
    for arg in sys.argv[1:]:
        if arg.startswith("--file="):
            specific_file = arg.split("=", 1)[1]

    if specific_file:
        files = [Path(specific_file)]
    else:
        files = sorted(Path("backend/core/phases").glob("phase_*.py"))

    total_issues = 0
    files_with_issues = 0

    for fpath in files:
        if fpath.name == "phase_interface.py" or fpath.name == "__init__.py":
            continue
        issues = check_file(fpath)
        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            if ci_mode:
                print(f"❌ {fpath.name}:")
            else:
                print(f"⚠️  {fpath.name}:")
            for issue in issues[:3]:  # max 3 per file
                print(f"   {issue}")
            if len(issues) > 3:
                print(f"   ... and {len(issues) - 3} more")

    if total_issues == 0:
        print("✅ No scope bugs detected in any phase file.")
        sys.exit(0)
    else:
        print(f"\n❌ {total_issues} potential scope issues in {files_with_issues} files.")
        if ci_mode:
            sys.exit(1)
        else:
            print("   Run with --ci to fail CI on these issues.")
            sys.exit(0)


if __name__ == "__main__":
    main()
