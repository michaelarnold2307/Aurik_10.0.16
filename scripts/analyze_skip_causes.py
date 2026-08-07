#!/usr/bin/env python3
"""Static skip-cause analyzer for pytest suites.

Scans tests and conftest files without executing the test suite and reports
systematic skip causes (skip/skipif/importorskip + heavy collection guards).
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "skip_causes_report.json"


class SkipVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.pytest_skip_calls = 0
        self.pytest_importorskip_calls = 0
        self.skipif_decorators = 0
        self.skipif_true_decorators = 0

    def visit_Call(self, node: ast.Call) -> None:
        # pytest.skip(...), pytest.importorskip(...)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "pytest" and node.func.attr == "skip":
                self.pytest_skip_calls += 1
            if node.func.value.id == "pytest" and node.func.attr == "importorskip":
                self.pytest_importorskip_calls += 1

        # @pytest.mark.skipif(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "skipif":
            self.skipif_decorators += 1
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value is True:
                self.skipif_true_decorators += 1

        self.generic_visit(node)


def _scan_python_file(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return {
            "pytest_skip_calls": 0,
            "pytest_importorskip_calls": 0,
            "skipif_decorators": 0,
            "skipif_true_decorators": 0,
        }

    v = SkipVisitor()
    v.visit(tree)
    return {
        "pytest_skip_calls": v.pytest_skip_calls,
        "pytest_importorskip_calls": v.pytest_importorskip_calls,
        "skipif_decorators": v.skipif_decorators,
        "skipif_true_decorators": v.skipif_true_decorators,
    }


def main() -> int:
    files = sorted(ROOT.glob("tests/**/*.py"))
    files += [ROOT / "conftest.py", ROOT / "tests" / "conftest.py"]

    totals: Any = Counter()
    per_file: dict[str, dict[str, int]] = {}

    for path in files:
        if not path.exists():
            continue
        metrics = _scan_python_file(path)
        if any(metrics.values()):
            rel = str(path.relative_to(ROOT))
            per_file[rel] = metrics
            totals.update(metrics)

    report = {
        "summary": dict(totals),
        "files_with_skip_mechanisms": per_file,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Skip analysis written to", OUT.relative_to(ROOT))
    print("Summary:", dict(totals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
