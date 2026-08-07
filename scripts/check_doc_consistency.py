#!/usr/bin/env python3
"""scripts/check_doc_consistency.py — §v10.700 F7.

Prüft Konsistenz zwischen Dokumentation und Code:
  1. Versionsnummern-Konsistenz (Docs vs version.py)
  2. Phasen-Zahl-Konsistenz (Docs vs tatsächliche Dateien)
  3. Tote Links im docs/INDEX.md
  4. Veraltete Versionsreferenzen (v8.0, v10.0.8)
  5. Pre-Commit-Script-Existenz (alle referenzierten Scripts vorhanden)
  6. Requirements-Konsistenz (importierte Pakete in requirements.txt)

Nutzung:
  python scripts/check_doc_consistency.py          # Vollständiger Check
  python scripts/check_doc_consistency.py --ci      # CI-Mode: Exit 1 bei Fehlern
  python scripts/check_doc_consistency.py --json    # JSON-Ausgabe

Exit-Codes:
  0 = Alles konsistent
  1 = Inkonsistenzen gefunden
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_actual_version() -> str:
    """Liest die tatsächliche Version aus backend/core/version.py."""
    try:
        version_file = REPO_ROOT / "backend" / "core" / "version.py"
        content = version_file.read_text()
        match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
        return match.group(1) if match else "unknown"
    except Exception:
        return "unknown"


def get_actual_phase_count() -> int:
    """Zählt die tatsächliche Anzahl Phasen-Dateien."""
    phases_dir = REPO_ROOT / "backend" / "core" / "phases"
    if phases_dir.exists():
        return len(list(phases_dir.glob("phase_*.py")))
    return 0


def check_version_consistency() -> list[dict]:
    """Prüft ob Dokumente dieselbe Version wie version.py referenzieren."""
    issues: list[dict] = []
    actual = get_actual_version()
    if actual == "unknown":
        return issues

    # Alte Versionen, die nicht mehr vorkommen sollten (nur als Versionsangabe, nicht in Funktionsnamen)
    old_versions = [
        "Aurik 8.",
        "Aurik 9.",  # Produktnamen mit alter Major-Version
        "Version: 10.0.8",
        "Version: 10.0.17",
        "Version: 10.4.0",
        "Version: 10.5.0",
    ]

    for md_file in REPO_ROOT.rglob("*.md"):
        if ".venv" in str(md_file) or ".git" in str(md_file):
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue
        for old in old_versions:
            if old in content:
                # Skip roadmap itself (it documents the gaps)
                if "v10.700_weltspitze_roadmap" in str(md_file):
                    continue
                issues.append(
                    {
                        "file": str(md_file.relative_to(REPO_ROOT)),
                        "type": "old_version",
                        "detail": f"Enthält veraltete Version: {old} (aktuell: {actual})",
                    }
                )
                break  # Ein Issue pro Datei reicht
    return issues


def check_phase_count_consistency() -> list[dict]:
    """Prüft ob Dokumente die korrekte Phasen-Zahl referenzieren."""
    issues: list[dict] = []
    actual = get_actual_phase_count()
    if actual == 0:
        return issues

    old_counts = ["68 Phasen", "68 phases", "68-Phasen", "68 Spezialwerkzeuge"]

    for md_file in REPO_ROOT.rglob("*.md"):
        if ".venv" in str(md_file) or ".git" in str(md_file):
            continue
        if "v10.700_weltspitze_roadmap" in str(md_file):
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue
        for old in old_counts:
            if old in content and "71 Phasen (68 + 3 Phase-0)" not in content:
                issues.append(
                    {
                        "file": str(md_file.relative_to(REPO_ROOT)),
                        "type": "old_phase_count",
                        "detail": f"Enthält '{old}' (tatsächlich: {actual} Phasen-Dateien)",
                    }
                )
                break
    return issues


def check_dead_links() -> list[dict]:
    """Prüft tote Links im docs/INDEX.md."""
    issues: list[dict] = []
    index = REPO_ROOT / "docs" / "INDEX.md"
    if not index.exists():
        return issues

    content = index.read_text()
    # Finde Markdown-Links: [text](pfad)
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
    for text, path in links:
        if path.startswith("http"):
            continue  # Externe Links nicht prüfen
        if path.startswith("../"):
            target = (index.parent / path).resolve()
        else:
            target = (index.parent / path).resolve()
        if not target.exists():
            issues.append(
                {
                    "file": "docs/INDEX.md",
                    "type": "dead_link",
                    "detail": f"Toter Link: [{text}]({path}) → {target} existiert nicht",
                }
            )
    return issues


def check_precommit_scripts() -> list[dict]:
    """Prüft ob alle in .pre-commit-config.yaml referenzierten Scripts existieren."""
    issues: list[dict] = []
    precommit = REPO_ROOT / ".pre-commit-config.yaml"
    if not precommit.exists():
        return issues
    content = precommit.read_text()
    # Finde alle Script-Referenzen in entry:-Zeilen
    script_pattern = re.compile(r"scripts/[\w/_-]+\.(?:py|sh)")
    for match in script_pattern.finditer(content):
        script_path = REPO_ROOT / match.group(0)
        if not script_path.exists():
            issues.append(
                {
                    "file": ".pre-commit-config.yaml",
                    "type": "precommit_missing_script",
                    "detail": f"Pre-Commit referenziert nicht existentes Script: {match.group(0)}",
                }
            )
    return issues


def check_requirements_consistency() -> list[dict]:
    """Prüft ob kritische Pakete in requirements_aurik.txt gelistet sind."""
    issues: list[dict] = []
    req_file = REPO_ROOT / "requirements" / "requirements_aurik.txt"
    if not req_file.exists():
        return issues
    req_content = req_file.read_text().lower()
    # Kritische Pakete, die in requirements sein MÜSSEN
    critical_packages = [
        "numpy",
        "scipy",
        "soundfile",
        "rich",
        "onnxruntime",
        "torch",
        "librosa",
        "pyyaml",
        "requests",
    ]
    for pkg in critical_packages:
        if pkg not in req_content:
            issues.append(
                {
                    "file": str(req_file.relative_to(REPO_ROOT)),
                    "type": "requirements_missing",
                    "detail": f"Kritisches Paket '{pkg}' nicht in requirements_aurik.txt gefunden",
                }
            )
    return issues


def main():
    p = argparse.ArgumentParser(description="Aurik Document Consistency Check (F7)")
    p.add_argument("--ci", action="store_true", help="CI-Mode: Exit 1 bei Fehlern")
    p.add_argument("--json", action="store_true", help="JSON-Ausgabe")
    args = p.parse_args()

    all_issues = []
    all_issues.extend(check_version_consistency())
    all_issues.extend(check_phase_count_consistency())
    all_issues.extend(check_dead_links())
    all_issues.extend(check_precommit_scripts())
    all_issues.extend(check_requirements_consistency())

    if args.json:
        print(
            json.dumps(
                {
                    "issues": len(all_issues),
                    "details": all_issues,
                    "passes": len(all_issues) == 0,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(f"📋 Document Consistency Check — {get_actual_version()}, {get_actual_phase_count()} Phasen")
        if all_issues:
            print(f"\n❌ {len(all_issues)} Inkonsistenzen gefunden:")
            for i in all_issues:
                print(f"  [{i['type']}] {i['file']}: {i['detail']}")
        else:
            print("\n✅ Alle Dokumente konsistent")
        print(f"\n{'❌ CI-Gate: FEHLGESCHLAGEN' if all_issues else '✅ CI-Gate: BESTANDEN'}")

    return 1 if (args.ci and all_issues) else 0


if __name__ == "__main__":
    sys.exit(main())
