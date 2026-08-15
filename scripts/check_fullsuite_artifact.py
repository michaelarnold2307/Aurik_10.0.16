#!/usr/bin/env python3
"""Fullsuite-Artefakt-Prüfer — §CI-FULLSUITE-ARTEFAKT (Vorschlag 07).

Prüft das Vollsuite-Artefakt-Paar
  logs/fullsuite_latest.log        (vollständiges pytest-Protokoll)
  reports/fullsuite_summary.md     (maschineller Kurzreport)

Kriterien:
  1. beide Dateien existieren,
  2. beide ≤ MAX_ARTIFACT_AGE_DAYS (7) Kalendertage alt,
  3. der Kurzreport weist `failed = 0` aus (belegtes Grün).

Modi:
  Standard            → Report-Modus (Exit 0, Verstöße als Warnung)
  AURIK_FULLSUITE_GATE=1 → fail-closed (Exit 1 bei Verstößen) — für das
                          Merge-Gate nach Maintainer-Sign-off.

Zusätzlich: `--generate-from <LOG>` erzeugt das Artefakt-Paar aus einem
vorhandenen pytest-Protokoll (Nightly-Ersatz bis zur Workflow-Verdrahtung).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
_LOG_ARTIFACT = _PROJECT / "logs" / "fullsuite_latest.log"
_SUMMARY_ARTIFACT = _PROJECT / "reports" / "fullsuite_summary.md"
MAX_ARTIFACT_AGE_DAYS = 7

_SUMMARY_LINE_RE = re.compile(r"^\s*(?:-\s*)?(?:[A-Za-zÄÖÜäöü]+\s*:\s*)?(\d+)\s+failed,\s+(\d+)\s+passed(?:,\s+(\d+)\s+skipped)?(?:,\s+(\d+)\s+deselected)?", re.MULTILINE)
_EXIT_LINE_RE = re.compile(r"VOLLSUITE(?:\d)?\s+ENDE.*Exit\s+(\d+)", re.IGNORECASE)
_HEAD_LINE_RE = re.compile(r"HEAD\s+([0-9a-f]{7,40})", re.IGNORECASE)


def _rel(p: Path) -> str:
    """Relativer Pfad, robust auch für Pfade außerhalb des Repos (Tests)."""
    try:
        return str(p.relative_to(_PROJECT))
    except ValueError:
        return str(p)


def evaluate_artifact(log_path: Path, summary_path: Path, now: float | None = None) -> tuple[bool, list[str]]:
    """(ok, violations) — pure Funktion, unit-testbar."""
    now = time.time() if now is None else now
    violations: list[str] = []

    if not log_path.exists():
        return False, [f"Artefakt fehlt: {_rel(log_path)}"]
    if not summary_path.exists():
        return False, [f"Artefakt fehlt: {_rel(summary_path)}"]

    _age_s = MAX_ARTIFACT_AGE_DAYS * 86400
    for p in (log_path, summary_path):
        _age = now - p.stat().st_mtime
        if _age > _age_s:
            violations.append(f"Artefakt zu alt ({_age / 86400:.1f} Tage): {_rel(p)}")

    _summary = summary_path.read_text(encoding="utf-8", errors="replace")
    _m = _SUMMARY_LINE_RE.search(_summary)
    if not _m:
        violations.append("Kurzreport ohne Ergebniszeile (N failed, M passed)")
    elif int(_m.group(1)) != 0:
        violations.append(f"Kurzreport weist {_m.group(1)} failed aus — kein belegtes Grün")

    _log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    _e = _EXIT_LINE_RE.search(_log_tail)
    if _e and int(_e.group(1)) != 0:
        violations.append(f"Protokoll-Exit {_e.group(1)} != 0")

    return (not violations), violations


def generate_from_log(log_path: Path) -> None:
    """Erzeugt logs/fullsuite_latest.log + reports/fullsuite_summary.md aus einem Protokoll."""
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    _m = _SUMMARY_LINE_RE.search(txt)
    _e = _EXIT_LINE_RE.search(txt[-4000:])
    _h = _HEAD_LINE_RE.search(txt[:2000])
    if not _m:
        sys.exit(f"Keine Ergebniszeile in {log_path} gefunden")

    _LOG_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    _LOG_ARTIFACT.write_text(txt, encoding="utf-8")
    _failed, _passed = int(_m.group(1)), int(_m.group(2))
    _skipped, _deselected = int(_m.group(3) or 0), int(_m.group(4) or 0)
    _exit = int(_e.group(1)) if _e else (-1 if _failed else 0)
    _SUMMARY_ARTIFACT.write_text(
        "# Fullsuite-Summary (maschinell generiert)\n\n"
        f"- Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- HEAD: {(_h.group(1) if _h else 'unbekannt')}\n"
        f"- Ergebnis: {_failed} failed, {_passed} passed, {_skipped} skipped, {_deselected} deselected\n"
        f"- Exit: {_exit}\n"
        f"- Quelle: {_rel(log_path)}\n",
        encoding="utf-8",
    )
    print(f"Artefakt erzeugt: {_LOG_ARTIFACT.relative_to(_PROJECT)} + {_SUMMARY_ARTIFACT.relative_to(_PROJECT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-from", type=Path, help="Artefakt-Paar aus vorhandenem Protokoll erzeugen")
    args = parser.parse_args(argv)

    if args.generate_from:
        generate_from_log(args.generate_from)
        return 0

    _ok, _violations = evaluate_artifact(_LOG_ARTIFACT, _SUMMARY_ARTIFACT)
    _fail_closed = __import__("os").environ.get("AURIK_FULLSUITE_GATE", "0") == "1"

    if _ok:
        print("Fullsuite-Artefakt: OK (belegtes Grün)")
        return 0
    for v in _violations:
        print(f"[{'FAIL' if _fail_closed else 'WARN'}] {v}")
    if _fail_closed:
        return 1
    print("Report-Modus: Verstöße dokumentiert, kein Block.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
