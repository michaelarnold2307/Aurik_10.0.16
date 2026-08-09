#!/usr/bin/env python3
"""Parameter-Interaktions-Graph. Spec 22 C1.

AST-basierte Analyse: Extrahiert Parameter-Definitionen, verfolgt Datenfluss
bis zur Anwendung, generiert docs/parameter_graph.json.

Usage:
    python scripts/parameter_impact_trace.py [--check] [--output json|md]
"""
from __future__ import annotations
import ast, json, hashlib, os, sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
TARGET_MODULE = PROJECT_ROOT / "backend" / "core" / "excellence_optimizer.py"
OUTPUT_JSON = PROJECT_ROOT / "docs" / "parameter_graph.json"
OUTPUT_MD = PROJECT_ROOT / "docs" / "parameter_graph.md"


def extract_parameters(filepath: str) -> list[dict]:
    """Extrahiert alle Parameter-Definitionen aus einer Python-Datei."""
    with open(filepath) as f:
        tree = ast.parse(f.read(), filename=filepath)
    params = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets if isinstance(node.targets, list) else [node.targets]:
                if isinstance(target, ast.Name) and target.id.startswith("_"):
                    if isinstance(node.value, (ast.Constant, ast.Num)):
                        val = node.value.value if isinstance(node.value, ast.Constant) else node.value.n
                        params.append({
                            "name": target.id,
                            "value": val,
                            "line": node.lineno,
                            "type": type(node.value).__name__,
                        })
    return params


def trace_usage(filepath: str, param_name: str) -> list[dict]:
    """Verfolgt wo ein Parameter im Code verwendet wird."""
    with open(filepath) as f:
        lines = f.readlines()
        content = f.read()
    tree = ast.parse(content, filename=filepath)
    usages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == param_name:
            if isinstance(node.ctx, ast.Load):
                usages.append({"line": node.lineno, "col": node.col_offset,
                              "context": lines[node.lineno - 1].strip()[:120]})
    return usages


def generate_graph(target: str, output_json: str, output_md: str) -> dict:
    """Generiert vollständigen Parameter-Interaktions-Graphen."""
    params = extract_parameters(target)
    graph: dict[str, Any] = {"source_file": target, "parameters": [], "generated_by": "parameter_impact_trace.py"}
    for p in params:
        usages = trace_usage(target, p["name"])
        p["usages"] = usages
        p["impact_count"] = len(usages)
        graph["parameters"].append(p)
    graph["hash"] = hashlib.sha256(open(target, "rb").read()).hexdigest()[:16]
    with open(output_json, "w") as f:
        json.dump(graph, f, indent=2)
    lines = [f"# Parameter-Interaktions-Graph", f"", f"> Auto-generated. Source: `{target}`", f""]
    for p in sorted(graph["parameters"], key=lambda x: -x["impact_count"]):
        lines.append(f"## {p['name']} (line {p['line']})")
        lines.append(f"- Default: `{p['value']}`")
        lines.append(f"- Used in {p['impact_count']} locations")
        if p['usages']:
            lines.append(f"- First usage: line {p['usages'][0]['line']}")
        lines.append("")
    with open(output_md, "w") as f:
        f.write("\n".join(lines))
    return graph


def check_consistency(output_json: str, target: str) -> bool:
    """Prüft ob der gespeicherte Graph aktuell ist (Hash-Vergleich)."""
    if not os.path.exists(output_json):
        print("No cached graph found. Run without --check to generate.")
        return False
    with open(output_json) as f:
        cached = json.load(f)
    current_hash = hashlib.sha256(open(target, "rb").read()).hexdigest()[:16]
    if cached.get("hash") != current_hash:
        print(f"Graph outdated: cached={cached.get('hash')} current={current_hash}")
        return False
    print(f"Graph current (hash={current_hash})")
    return True


def main():
    if "--check" in sys.argv:
        ok = check_consistency(str(OUTPUT_JSON), str(TARGET_MODULE))
        sys.exit(0 if ok else 1)
    graph = generate_graph(str(TARGET_MODULE), str(OUTPUT_JSON), str(OUTPUT_MD))
    print(f"Generated: {OUTPUT_JSON} ({len(graph['parameters'])} params)")
    print(f"Generated: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
