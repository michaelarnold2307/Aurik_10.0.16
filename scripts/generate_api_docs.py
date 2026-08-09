#!/usr/bin/env python3
"""API-Docs-Generator. Spec 15 paragraph 7.5.
Extrahiert Docstrings aus backend/api/bridge.py und generiert Markdown.

Autor: Aurik 10
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


def extract_docstrings(filepath: str) -> list[dict]:
    """Extrahiert alle Funktions-/Klassen-Docstrings aus einer Python-Datei."""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    items = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                items.append({"name": node.name, "type": type(node).__name__, "doc": doc})
    return items


def generate_markdown(items: list[dict], title: str = "Aurik API Documentation") -> str:
    """Generiert Markdown aus Docstring-Liste."""
    md = [f"# {title}", "", "> Auto-generated. Do not edit.", ""]
    for item in items:
        md.append(f"## {item['name']}")
        md.append(f"_{item['type']}_")
        md.append(item["doc"])
        md.append("")
    return "\n".join(md)


def main():
    bridge = Path(__file__).parent.parent / "backend" / "api" / "bridge.py"
    if not bridge.exists():
        print(f"ERROR: {bridge} not found")
        sys.exit(1)
    items = extract_docstrings(str(bridge))
    md = generate_markdown(items)
    out = Path(__file__).parent.parent / "docs" / "api" / "bridge_api.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Generated: {out} ({len(items)} items)")


if __name__ == "__main__":
    main()
