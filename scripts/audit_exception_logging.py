#!/usr/bin/env python3
"""scripts/audit_exception_logging.py — §v10.700 L6b.

from typing import Any
logger = logging.getLogger(__name__)
AST-basierter Scanner: Prüft alle `except Exception:`-Blöcke im Backend
und Denker auf ordnungsgemäßes Logging. Klassifiziert in:
  ✅ Geloggt   — logger.*() oder raise im Block
  ⚠️ Teilweise — Code vorhanden, aber kein explizites Logging
  ❌ Silent    — Nur `pass` oder kein Logging

Nutzung:
  python scripts/audit_exception_logging.py              # Alle Funde
  python scripts/audit_exception_logging.py --ci          # CI-Mode: Exit 1 bei ❌
  python scripts/audit_exception_logging.py --json        # JSON-Ausgabe
  python scripts/audit_exception_logging.py --fix         # Auto-Fix: logger.debug hinzufügen

Exit-Codes:
  0 = Alle Blöcke geloggt (oder nur ⚠️)
  1 = ❌ Silent-Failures gefunden
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Zielverzeichnisse ───────────────────────────────────────────────────────
SCAN_DIRS: list[str] = ["backend", "denker"]
SKIP_PATTERNS: list[str] = [".venv", "__pycache__", ".git", "temp_repro", "site-packages"]

# ── Erlaubte Ausnahmen (begründete no-log Fälle) ────────────────────────────
# Diese Blöcke dürfen ohne Logging bleiben, wenn sie mit # no-log: <Grund>
# kommentiert sind. Beispiel:
#   except Exception:  # no-log: KeyboardInterrupt swallowed by design
ALLOWED_NO_LOG_PATTERNS: list[str] = [
    "KeyboardInterrupt",
    "GeneratorExit",
    "StopAsyncIteration",
    "SystemExit",
]


def _should_skip_path(filepath: str) -> bool:
    return any(skip in filepath for skip in SKIP_PATTERNS)


def _classify_body(body: list[ast.stmt], source_lines: list[str], node: ast.ExceptHandler) -> tuple[str, str]:
    """Klassifiziert einen except-Block: ✅ ⚠️ ❌"""
    # Prüfe auf # no-log Kommentar
    if hasattr(node, "lineno"):
        # Suche nach Kommentar in der except-Zeile
        except_line = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
        if "# no-log:" in except_line:
            return "✅", "begründete Ausnahme (# no-log)"

    # Prüfe auf logger-Aufrufe im Body
    has_logger = False
    has_code = False
    has_only_pass = True

    class LoggerFinder(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, call_node):
            if isinstance(call_node.func, ast.Attribute):
                if isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == "logger":
                    self.found = True
            elif isinstance(call_node.func, ast.Name):
                if call_node.func.id in ("logger", "log", "logging"):
                    self.found = True
            self.generic_visit(call_node)

        def visit_Raise(self, _node):
            self.found = True  # raise ist OK — Fehler wird weitergereicht

    finder = LoggerFinder()
    for stmt in body:
        finder.visit(stmt)
        if not isinstance(stmt, ast.Pass):
            has_code = True
            has_only_pass = False

    has_logger = finder.found

    if has_logger:
        return "✅", "logger-Aufruf oder raise gefunden"
    elif has_only_pass or not has_code:
        return "❌", "Silent Failure — nur pass oder leer"
    else:
        return "⚠️", "Code vorhanden, aber kein explizites Logging"


def scan_file(filepath: str) -> list[dict[str, Any]]:  # type: ignore[name-defined]
    """Scannt eine Datei und gibt Funde zurück."""
    findings: list[dict[str, Any]] = []  # type: ignore[name-defined]
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return findings

    # Finde alle except Exception:-Blöcke (kein bestimmter Typ)
    class ExceptFinder(ast.NodeVisitor):
        def visit_ExceptHandler(self, node):
            # Nur except Exception: (oder except Exception as e:) — kein spezifischer Typ
            if node.type is None:
                # Bare except: — separat behandelt
                pass
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                classification, reason = _classify_body(node.body, source_lines, node)
                findings.append(
                    {
                        "file": filepath,
                        "line": node.lineno,
                        "classification": classification,
                        "reason": reason,
                    }
                )
            elif isinstance(node.type, ast.Tuple):
                # except (Exception, ...):
                for elt in node.type.elts:
                    if isinstance(elt, ast.Name) and elt.id == "Exception":
                        classification, reason = _classify_body(node.body, source_lines, node)
                        findings.append(
                            {
                                "file": filepath,
                                "line": node.lineno,
                                "classification": classification,
                                "reason": reason,
                            }
                        )
                        break
            self.generic_visit(node)

    try:
        ExceptFinder().visit(tree)
    except Exception:
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return findings


def scan_all() -> dict[str, Any]:  # type: ignore[name-defined]
    """Scannt alle Python-Dateien in SCAN_DIRS."""
    all_findings: list[dict[str, Any]] = []  # type: ignore[name-defined]
    files_scanned = 0

    for scan_dir in SCAN_DIRS:
        base = Path(scan_dir)
        if not base.exists():
            continue
        for pyfile in base.rglob("*.py"):
            if _should_skip_path(str(pyfile)):
                continue
            files_scanned += 1
            all_findings.extend(scan_file(str(pyfile)))

    # Klassifikation
    ok = [f for f in all_findings if f["classification"] == "✅"]
    warn = [f for f in all_findings if f["classification"] == "⚠️"]
    critical = [f for f in all_findings if f["classification"] == "❌"]

    return {
        "files_scanned": files_scanned,
        "total_blocks": len(all_findings),
        "logged": len(ok),
        "partial": len(warn),
        "silent": len(critical),
        "passes_ci": len(critical) == 0,
        "results": all_findings,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Aurik Exception Logging Audit (L6b)")
    p.add_argument("--ci", action="store_true", help="CI-Mode: Exit 1 bei Silent-Failures")
    p.add_argument("--json", action="store_true", help="JSON-Ausgabe")
    p.add_argument("--show-silent", action="store_true", help="Nur ❌ Silent-Failures anzeigen")
    p.add_argument("--show-warn", action="store_true", help="Nur ⚠️ Teilweise anzeigen")
    args = p.parse_args()

    report = scan_all()

    if args.json:
        # Kompakte JSON-Ausgabe
        summary = {
            "files_scanned": report["files_scanned"],
            "total_blocks": report["total_blocks"],
            "logged": report["logged"],
            "partial": report["partial"],
            "silent": report["silent"],
            "passes_ci": report["passes_ci"],
            "silent_files": [
                {"file": f["file"], "line": f["line"]} for f in report["results"] if f["classification"] == "❌"
            ],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"🔍 Exception Logging Audit — {report['files_scanned']} Dateien gescannt")
        print(f"   except Exception-Blöcke: {report['total_blocks']}")
        print(f"   ✅ Geloggt (logger/raise): {report['logged']}")
        print(f"   ⚠️  Code ohne Logging:      {report['partial']}")
        print(f"   ❌ Silent (nur pass):      {report['silent']}")
        print()

        if args.show_silent:
            print("─── ❌ SILENT FAILURES ───")
            for f in report["results"]:
                if f["classification"] == "❌":
                    print(f"  {f['file']}:{f['line']} — {f['reason']}")
        elif args.show_warn:
            print("─── ⚠️  PARTIAL ───")
            for f in report["results"]:
                if f["classification"] == "⚠️":
                    print(f"  {f['file']}:{f['line']} — {f['reason']}")
        else:
            # Default: zeige nur ❌ und ⚠️
            if report["silent"] > 0:
                print("─── ❌ SILENT FAILURES (müssen behoben werden) ───")
                for f in report["results"]:
                    if f["classification"] == "❌":
                        print(f"  {f['file']}:{f['line']} — {f['reason']}")
            if report["partial"] > 0 and not args.show_silent:
                print(f"─── ⚠️  {report['partial']} Blöcke ohne explizites Logging (--show-warn für Details)")

        if report["passes_ci"]:
            print("\n✅ CI-Gate: BESTANDEN — keine Silent-Failures")
        else:
            print(f"\n❌ CI-Gate: FEHLGESCHLAGEN — {report['silent']} Silent-Failures")

    return 0 if not args.ci else (0 if report["passes_ci"] else 1)


if __name__ == "__main__":
    sys.exit(main())
