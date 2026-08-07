#!/usr/bin/env python3
"""
Spec-Compliance-Scan für alle 64 Phasen.
Prüft die 10 kritischen Anti-Patterns aus copilot-instructions.md
"""

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tabulate import tabulate

logger = logging.getLogger(__name__)

# Anti-Pattern-Definitionen
PATTERNS = {
    "np.max_abs": {
        "pattern": r"np\.max\s*\(\s*np\.abs\s*\(",
        "description": "np.max(np.abs()) statt np.percentile(..., 99.9)",
        "severity": "HIGH",
    },
    "audio_slice_1d": {
        "pattern": r"audio\s*\[\s*0\s*\](?!\s*[:=])",
        "description": "audio[0] bei 2D-Audio statt audio[:, 0]",
        "severity": "HIGH",
    },
    "len_audio_division": {
        "pattern": r"len\s*\(\s*audio\s*\)\s*[/][\s]*sr",
        "description": "len(audio)/sr bei 2D-Audio - sollte robust sein",
        "severity": "MEDIUM",
    },
    "boundary_reflect": {
        "pattern": r"boundary\s*=\s*['\"]reflect['\"]",
        "description": "boundary='reflect' sollte 'even' sein",
        "severity": "MEDIUM",
    },
    "session_run_no_chunking": {
        "pattern": r"session\.run\s*\(\s*(?![^)]*chunk|[^)]*\[:[^)]*\])",
        "description": "Session.run() ohne Chunking bei variable length input",
        "severity": "HIGH",
    },
    "rms_mean": {
        "pattern": r"np\.mean\s*\(\s*audio\s*\*\*\s*2\s*\)",
        "description": "np.mean(audio**2) als RMS statt gated",
        "severity": "MEDIUM",
    },
    "no_goal_exclusions": {
        "pattern": None,  # Special handling needed
        "description": "Phase fehlt in PHASE_GOAL_EXCLUSIONS",
        "severity": "MEDIUM",
    },
    "allocate_no_release": {
        "pattern": r"(?:try_allocate|plm\.try_allocate)\s*\([^)]*\)(?![\s\S]*?(?:release|set_active))",
        "description": "try_allocate ohne entsprechendes release()",
        "severity": "HIGH",
    },
    "stft_no_boundary": {
        "pattern": r"\.stft\s*\(\s*(?!.*boundary)",
        "description": "STFT ohne explizitem boundary parameter",
        "severity": "MEDIUM",
    },
    "lpc_order_low": {
        "pattern": r"(?:AR_ORDER|lpc_order)\s*[=<]\s*(?:[0-9]|1[0-5])(?!\d)",
        "description": "LPC-Ordnung < 16 (sollte >= 30 sein)",
        "severity": "MEDIUM",
    },
}


def scan_file(filepath):
    """Scannt eine Phase-Datei auf alle Anti-Patterns."""
    violations: Any = defaultdict(int)

    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"FEHLER beim Lesen {filepath}: {e}")
        return violations

    # Pattern-Scans
    for pattern_key, pattern_info in PATTERNS.items():
        if pattern_key == "no_goal_exclusions":
            # Special: Prüfe ob Phase in per_phase_musical_goals_gate.py existiert
            continue

        if pattern_info["pattern"] is None:  # type: ignore[index]
            continue

        try:
            matches = re.finditer(pattern_info["pattern"], content, re.IGNORECASE | re.MULTILINE)  # type: ignore[index]
            count = len(list(matches))
            if count > 0:
                violations[pattern_key] = count
        except Exception as e:
            violations[f"regex_error_{pattern_key}"] = str(e)

    return violations


def get_phase_id(filepath):
    """Extrahiert Phase-ID aus Dateipfad."""
    basename = os.path.basename(filepath)
    match = re.match(r"(phase_\d+)", basename)
    return match.group(1) if match else None


def check_goal_exclusions(phase_id):
    """Prüft ob Phase in per_phase_musical_goals_gate.py definiert ist."""
    try:
        gate_file = Path("/media/michael/Software 4TB/Aurik_Standalone/backend/core/per_phase_musical_goals_gate.py")
        with open(gate_file, encoding="utf-8") as f:
            content = f.read()
            # Prüfe ob phase_id in PHASE_GOAL_EXCLUSIONS vorkommt
            if f'"{phase_id}"' in content or f"'{phase_id}'" in content:
                return True
    except OSError:
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return False


def main():
    phases_dir = Path("/media/michael/Software 4TB/Aurik_Standalone/backend/core/phases")
    phase_files = sorted(phases_dir.glob("phase_*.py"))

    results = []
    violation_counts: Any = defaultdict(int)

    print(f"\n🔍 Spec-Compliance-Scan über {len(phase_files)} Phasen...\n")

    for phase_file in phase_files:
        phase_id = get_phase_id(str(phase_file))
        if not phase_id:
            continue

        violations = scan_file(str(phase_file))

        # Prüfe Goal-Exclusions
        has_goal_exclusions = check_goal_exclusions(phase_id)
        if not has_goal_exclusions:
            violations["no_goal_exclusions"] = 1

        total_violations = sum(violations.values())

        # Berechne Compliance-Score (0-10, max 10 = 0 Violations)
        violation_count = len(violations)
        compliance_score = max(0, 10 - (violation_count * 1.5))
        compliance_score = round(compliance_score, 1)

        # Status bestimmen
        if total_violations == 0:
            status = "✅ PASS"
        elif violation_count <= 2:
            status = "⚠️  WARNING"
        else:
            status = "❌ CRITICAL"

        # Violations als String formatieren
        violation_details = []
        for key, count in sorted(violations.items()):
            if key in PATTERNS:
                violation_details.append(f"{key}({count})")
            else:
                violation_details.append(key)

        violations_str = ", ".join(violation_details) if violation_details else "none"

        results.append(
            {
                "Phase": phase_id,
                "Violations": violation_count,
                "Details": violations_str[:50] + ("..." if len(violations_str) > 50 else ""),
                "Score": compliance_score,
                "Status": status,
                "Total": total_violations,
            }
        )

        # Zähle Violations pro Pattern
        for key in violations.keys():
            violation_counts[key] += 1

    # Sortiere nach Violation-Count (absteigend)
    results_sorted = sorted(results, key=lambda x: x["Total"], reverse=True)

    # Erstelle Ausgabe-Tabelle
    print("=" * 120)
    print("SPEC-COMPLIANCE-SCAN ERGEBNISSE (alle 64 Phasen)")
    print("=" * 120)

    table_data = []
    for r in results_sorted:
        table_data.append([r["Phase"], r["Violations"], r["Score"], r["Status"], r["Details"]])

    headers = ["Phase", "Violation Count", "Compliance Score", "Status", "Violation Details"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", maxcolwidths=[12, 15, 17, 12, 50]))

    print("\n" + "=" * 120)
    print("TOP 10 PHASEN MIT HÖCHSTEM VIOLATIONS-COUNT")
    print("=" * 120)

    top_10 = results_sorted[:10]
    top_10_data = []
    for r in top_10:
        top_10_data.append([r["Phase"], r["Total"], r["Score"], r["Status"]])

    print(tabulate(top_10_data, headers=["Phase", "Total Violations", "Score", "Status"], tablefmt="grid"))

    print("\n" + "=" * 120)
    print("ANTI-PATTERN-HÄUFIGKEIT (über alle Phasen)")
    print("=" * 120)

    pattern_data = []
    for pattern_key in sorted(PATTERNS.keys()):
        count = violation_counts.get(pattern_key, 0)
        if count > 0:
            pattern_data.append(
                [pattern_key, PATTERNS[pattern_key]["description"], count, PATTERNS[pattern_key]["severity"]]  # type: ignore[index]
            )

    print(tabulate(pattern_data, headers=["Pattern", "Description", "Count", "Severity"], tablefmt="grid"))

    # Zusammenfassung
    total_phases = len(results)
    pass_count = sum(1 for r in results if r["Total"] == 0)
    warning_count = sum(1 for r in results if 0 < r["Total"] <= 2)
    critical_count = sum(1 for r in results if r["Total"] > 2)

    print("\n" + "=" * 120)
    print("ZUSAMMENFASSUNG")
    print("=" * 120)
    print(f"✅ PASS (0 Violations):     {pass_count}/{total_phases} ({100 * pass_count / total_phases:.1f}%)")
    print(f"⚠️  WARNING (1-2):          {warning_count}/{total_phases} ({100 * warning_count / total_phases:.1f}%)")
    print(f"❌ CRITICAL (>2):          {critical_count}/{total_phases} ({100 * critical_count / total_phases:.1f}%)")
    print(f"\nDurchschn. Compliance-Score: {sum(r['Score'] for r in results) / total_phases:.2f}/10")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
