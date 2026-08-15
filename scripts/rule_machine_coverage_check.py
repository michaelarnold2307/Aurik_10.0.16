#!/usr/bin/env python3
"""Rule-Machine-Coverage-Check — Meta-Regel 0 (Vorschlag 00).

REPORT-MODUS (noch nicht fail-closed): Prüft, dass jede normative Regel
der Kette einen maschinellen Prüfpunkt besitzt oder als `advisory`
markiert ist. Schlägt nach Stabilisierung + Maintainer-Sign-off in den
fail-closed-Modus um (dann Exit-Code 1 bei Verstößen).

Prüfungen:
  1. `[RELEASE_MUST]`-Header in der normativen Kette → Test-Abdeckung
     (RELEASE_MUST-Marker in tests/ gesucht; vorhandene
     release_must_coverage_check.py prüft nur copilot-instructions —
     hier zusätzlich VERBOTEN/instructions/specs).
  2. V01–V52 aus VERBOTEN.md → hartkodierte Regeln in
     scripts/aurik_verboten_linter.py (fail-closed-Katalog).
  3. §G-Gebote aus GEBOTE.md → hartkodierte Teilmenge in
     scripts/gebote_verifier.py.
  4. Absätze ohne Prüfpunkt und ohne `advisory`-Marker → Report-Warnung.

Ausgabe: reports/rule_machine_coverage_report.md (+ .json)
Exit: 0 immer im Report-Modus (AURIK_RULE_MACHINE_FAIL=1 aktiviert
den fail-closed-Modus).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
_REPORT_MD = _PROJECT / "reports" / "rule_machine_coverage_report.md"
_REPORT_JSON = _PROJECT / "reports" / "rule_machine_coverage_report.json"
_FAIL_CLOSED = os.environ.get("AURIK_RULE_MACHINE_FAIL", "0") == "1"

_NORMATIVE_CHAIN = [
    ".github/copilot-instructions.md",
    ".github/VERBOTEN.md",
    ".github/instructions/pipeline.instructions.md",
    ".github/instructions/phases.instructions.md",
    ".github/instructions/dsp.instructions.md",
    ".github/instructions/musical_goals.instructions.md",
    ".github/instructions/tests.instructions.md",
]
_SPEC_DIR = _PROJECT / ".github" / "specs"

_RELEASE_MUST_RE = re.compile(r"\[RELEASE_MUST\]\s*(?P<header>[^\n]*)")
_ADVISORY_RE = re.compile(r"\badvisory\b", re.IGNORECASE)
_VERBOTEN_ID_RE = re.compile(r"V(0[1-9]|[1-4][0-9]|5[0-2])\b")
_GEBOTE_ID_RE = re.compile(r"§G(\d+)\b")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _scan_tests_for_release_must() -> set[str]:
    markers: set[str] = set()
    for p in (_PROJECT / "tests").rglob("*.py"):
        try:
            txt = _read(p)
        except OSError:
            continue
        for m in re.finditer(r"RELEASE_MUST[:\-\s]*([^\n\"']{0,80})", txt):
            markers.add(m.group(1).strip()[:60].lower())
    return markers


def _hardcoded_verboten_rules() -> set[str]:
    linter = _PROJECT / "scripts" / "aurik_verboten_linter.py"
    if not linter.exists():
        return set()
    txt = _read(linter)
    rules = set()
    for m in re.finditer(r'"(V\d{2})"', txt):
        rules.add(m.group(1))
    for m in re.finditer(r"\b(V\d{2})\b", txt):
        rules.add(m.group(1))
    return rules


def _hardcoded_gebote_rules() -> set[int]:
    verifier = _PROJECT / "scripts" / "gebote_verifier.py"
    if not verifier.exists():
        return set()
    txt = _read(verifier)
    return {int(m.group(1)) for m in re.finditer(r"\bG(\d{1,3})\b", txt)}


def main() -> int:
    findings: list[dict[str, str]] = []

    # ── 1. RELEASE_MUST-Abdeckung ──────────────────────────────────────
    test_markers = _scan_tests_for_release_must()
    release_must_total = 0
    release_must_covered = 0
    for rel in _NORMATIVE_CHAIN:
        p = _PROJECT / rel
        if not p.exists():
            findings.append({"level": "info", "where": rel, "what": "Datei fehlt"})
            continue
        for m in _RELEASE_MUST_RE.finditer(_read(p)):
            release_must_total += 1
            key = m.group("header").strip()[:60].lower()
            covered = any(key[:40] in t or t in key[:40] for t in test_markers)
            if covered:
                release_must_covered += 1
            else:
                findings.append(
                    {"level": "warn", "where": rel, "what": f"RELEASE_MUST ohne Test-Marker: {key}"}
                )
    for spec in sorted(_SPEC_DIR.glob("*.md")):
        for m in _RELEASE_MUST_RE.finditer(_read(spec)):
            release_must_total += 1
            key = m.group("header").strip()[:60].lower()
            if any(key[:40] in t or t in key[:40] for t in test_markers):
                release_must_covered += 1
            else:
                findings.append(
                    {"level": "warn", "where": str(spec.relative_to(_PROJECT)), "what": f"RELEASE_MUST ohne Test-Marker: {key}"}
                )

    # ── 2. VERBOTEN V01–V52 ↔ Linter ──────────────────────────────────
    verboten_txt = _read(_PROJECT / ".github" / "VERBOTEN.md") if (_PROJECT / ".github" / "VERBOTEN.md").exists() else ""
    documented_v = set(_VERBOTEN_ID_RE.findall(verboten_txt))
    hardcoded_v = _hardcoded_verboten_rules()
    missing_in_linter = sorted(documented_v - hardcoded_v)
    for v in missing_in_linter:
        findings.append({"level": "warn", "where": "VERBOTEN.md", "what": f"{v} dokumentiert, aber nicht im Linter-Katalog"})

    # ── 3. GEBOTE ↔ Verifier ──────────────────────────────────────────
    gebote_txt = _read(_PROJECT / ".github" / "GEBOTE.md") if (_PROJECT / ".github" / "GEBOTE.md").exists() else ""
    documented_g = set(_GEBOTE_ID_RE.findall(gebote_txt))
    hardcoded_g = _hardcoded_gebote_rules()
    missing_in_verifier = sorted({int(g) for g in documented_g if int(g) > 9} - hardcoded_g)
    for g in missing_in_verifier:
        findings.append(
            {"level": "info", "where": "GEBOTE.md", "what": f"§G{g} dokumentiert, aber nicht in der hartkodierten Verifier-Teilmenge (laut AGENTS.md zulässig — Referenzcharakter)"}
        )

    # ── 4. Advisory-Marker-Statistik ──────────────────────────────────
    advisory_count = 0
    for rel in _NORMATIVE_CHAIN:
        p = _PROJECT / rel
        if p.exists():
            advisory_count += len(_ADVISORY_RE.findall(_read(p)))

    report = {
        "mode": "fail-closed" if _FAIL_CLOSED else "report",
        "release_must_total": release_must_total,
        "release_must_covered": release_must_covered,
        "documented_verboten_v": sorted(documented_v),
        "hardcoded_verboten_v": sorted(hardcoded_v),
        "missing_in_linter": missing_in_linter,
        "gebote_missing_in_verifier_subset": missing_in_verifier,
        "advisory_marker_count": advisory_count,
        "findings": findings,
    }

    _REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_MD.write_text(
        "# Rule-Machine-Coverage-Report\n\n"
        f"- Modus: **{report['mode']}**\n"
        f"- RELEASE_MUST gesamt/abgedeckt: {release_must_total}/{release_must_covered}\n"
        f"- VERBOTEN V dokumentiert: {len(documented_v)} | im Linter-Katalog: {len(hardcoded_v)} | fehlend: {missing_in_linter}\n"
        f"- GEBOTE außerhalb Verifier-Teilmenge (Info): {missing_in_verifier}\n"
        f"- advisory-Marker: {advisory_count}\n\n"
        "## Findings\n\n" + "".join(f"- [{f['level']}] {f['where']}: {f['what']}\n" for f in findings)
        or "## Findings\n\n_(keine)_\n",
        encoding="utf-8",
    )
    _REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    warnings = [f for f in findings if f["level"] == "warn"]
    print(f"Rule-Machine-Coverage: {len(warnings)} Warnungen, {len(findings)} Findings — Report: {_REPORT_MD.relative_to(_PROJECT)}")
    if _FAIL_CLOSED and warnings:
        print("FAIL-CLOSED: Warnungen vorhanden → Exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
