#!/usr/bin/env python3
"""scripts/fix_silent_exceptions.py — §v10.700 L6b.

Liest die Silent-Failure-Liste aus audit_exception_logging.py und fügt
automatisch `logger.debug()` in jeden `except Exception: pass`-Block ein.

Sicherheit:
  - Erzeugt Backup (.bak) vor jeder Änderung
  - Überspringt Zeilen mit # no-log: Marker
  - Prüft Syntax nach jedem Patch
  - Nur für except Exception:-Blöcke, die ausschließlich `pass` enthalten

Nutzung:
  python scripts/fix_silent_exceptions.py          # Alle Silent-Failures beheben
  python scripts/fix_silent_exceptions.py --dry-run # Nur anzeigen, nicht ändern
  python scripts/fix_silent_exceptions.py --file backend/core/unified_restorer_v3.py
"""

from __future__ import annotations

import ast
import logging
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class SilentBlock(NamedTuple):
    file: str
    line: int
    end_line: int  # Letzte Zeile des except-Blocks


def find_silent_blocks(filepath: str) -> list[SilentBlock]:
    """Findet alle except Exception: pass-Blöcke in einer Datei."""
    blocks: list[SilentBlock] = []
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        source_lines = source.splitlines()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return blocks

    class SilentFinder(ast.NodeVisitor):
        def visit_ExceptHandler(self, node):
            if node.type is None:
                pass
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                # Prüfe ob Body NUR aus pass besteht
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    # Prüfe auf # no-log Kommentar
                    except_line = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                    if "# no-log:" in except_line:
                        pass  # Überspringe begründete Ausnahmen
                    else:
                        end_lineno = node.body[-1].end_lineno
                        blocks.append(SilentBlock(str(filepath), node.lineno, end_lineno))
            self.generic_visit(node)

    try:
        SilentFinder().visit(tree)
    except Exception:
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return blocks


def fix_file(filepath: str, dry_run: bool = False) -> int:
    """Behebt alle Silent-Failures in einer Datei. Returns Anzahl Fixes."""
    blocks = find_silent_blocks(filepath)
    if not blocks:
        return 0

    with open(filepath, encoding="utf-8") as f:
        source_lines = f.readlines()  # Mit Newlines

    # Sortiere rückwärts (von unten nach oben), damit Zeilennummern stabil bleiben
    blocks_sorted = sorted(blocks, key=lambda b: b.line, reverse=True)

    fixed = 0
    for block in blocks_sorted:
        line_idx = block.line - 1  # 0-basiert
        if line_idx >= len(source_lines):
            continue

        # Extrahiere Einrückung der except-Zeile
        except_line = source_lines[line_idx]
        indent = except_line[: len(except_line) - len(except_line.lstrip())]

        # Extrahiere Variablenname aus "except Exception as e:"
        import re

        var_match = re.search(r"except\s+Exception\s+as\s+(\w+)", except_line)
        var_name = var_match.group(1) if var_match else "_exc"

        # Erstelle Logger-Zeile mit korrekter Einrückung
        # Füge 4 Leerzeichen mehr ein als die except-Zeile
        body_indent = indent + "    "
        log_line = f'{body_indent}logger.debug("{Path(filepath).name}:{block.line}: Silent exception absorbed", exc_info=True)\n'

        if dry_run:
            print(f"  WOULD FIX: {filepath}:{block.line} → logger.debug(...)")
            fixed += 1
            continue

        # Ersetze die pass-Zeile durch logger.debug + pass
        pass_idx = line_idx + 1  # pass ist direkt nach except
        if pass_idx < len(source_lines) and "pass" in source_lines[pass_idx]:
            source_lines[pass_idx] = log_line
            fixed += 1

    if fixed > 0 and not dry_run:
        # Backup erstellen
        backup = filepath + ".bak"
        shutil.copy2(filepath, backup)

        # Schreiben
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(source_lines)

        # Syntax-Prüfung
        try:
            with open(filepath, encoding="utf-8") as f:
                ast.parse(f.read(), filename=filepath)
            print(f"  ✅ {fixed} Fixes in {filepath} (Backup: {backup})")
        except SyntaxError as e:
            print(f"  ❌ SYNTAX-FEHLER nach Fix in {filepath}: {e}")
            print(f"     Backup wiederhergestellt aus {backup}")
            shutil.copy2(backup, filepath)
            fixed = 0

    return fixed


def main():
    import argparse

    p = argparse.ArgumentParser(description="Fix Silent except Exception: pass blocks")
    p.add_argument("--dry-run", action="store_true", help="Nur anzeigen")
    p.add_argument("--file", help="Nur eine Datei bearbeiten")
    args = p.parse_args()

    if args.file:
        files = [args.file]
    else:
        # Alle Dateien mit Silent-Failures aus dem Audit
        files = [
            "backend/core/adaptive_strength_optimizer.py",
            "backend/core/cumulative_interaction_guard.py",
            "backend/core/lyrics_guided_enhancement.py",
            "backend/core/musical_quality_assurance.py",
            "backend/core/naturalness_optimizer.py",
            "backend/core/one_take_export.py",
            "backend/core/reference_track_calibrator.py",
            "backend/core/signal_flow_tracer.py",
            "backend/core/singer_voice_model.py",
            "backend/core/aurik_orchestrator.py",
            "backend/core/pipeline_cumulative_guards.py",
            "backend/core/unified_restorer_v3.py",
            "backend/core/defect_scanner.py",
            "backend/core/ast_audio_set_classifier.py",
            "backend/core/ml_memory_budget.py",
            "backend/core/memmap_pool.py",
            "backend/core/metadata_aggregator.py",
            "backend/core/pre_analysis.py",
            "backend/core/holistic_perceptual_gate.py",
            "backend/api/bridge.py",
            "backend/core/dsp/sota_vocal_model_router.py",
            "backend/core/dsp/spectral_color_guard.py",
            "backend/core/dsp/stem_level_restorer.py",
            "backend/core/dsp/vocal_style_profiler.py",
            "backend/core/musical_goals/musical_goals_metrics.py",
            "backend/core/phases/phase_07_harmonic_restoration.py",
            "backend/core/phases/phase_23_spectral_repair.py",
            "backend/core/phases/phase_26_dynamic_range_expansion.py",
            "backend/core/phases/phase_interface.py",
            "denker/aurik_denker.py",
            "denker/phase_interaction_denker.py",
        ]

    total = 0
    for f in files:
        if Path(f).exists():
            c = fix_file(f, dry_run=args.dry_run)
            total += c
        else:
            print(f"  ⚠️  Datei nicht gefunden: {f}")

    action = "WÜRDEN" if args.dry_run else "WURDEN"
    print(f"\n🔧 {total} Silent except Exception: pass-Blöcke {action} behoben")


if __name__ == "__main__":
    main()
