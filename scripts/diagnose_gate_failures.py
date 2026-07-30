#!/usr/bin/env python3
"""Fall-für-Fall-Quality-Gate-Diagnose — §19.

Analysiert den Real-Audio-Quality-Gate-Report und erstellt eine
detaillierte, priorisierte Diagnose pro fehlgeschlagenem Fall.

Nutzung:
  python scripts/diagnose_gate_failures.py
  python scripts/diagnose_gate_failures.py --case jazz_1950s_scratched
  python scripts/diagnose_gate_failures.py --output reports/gate_diagnosis.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_report(report_path: Path | None = None) -> dict:
    """Lädt den Quality-Gate-Report."""
    path = report_path or REPO_ROOT / "audit" / "real_audio_restoration_quality_report.json"
    if not path.exists():
        print(f"Fehler: Report nicht gefunden: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_case(case: dict) -> dict:
    """Analysiert einen einzelnen Fall und gibt eine priorisierte Diagnose zurück."""
    fail_reasons = case.get("fail_reasons", [])
    hpi = case.get("hpi")
    quality = case.get("quality_estimate")
    vqi = case.get("vqi")

    # Kategorisiere die Fehler nach Schweregrad
    severity: dict[str, list[str]] = defaultdict(list)
    for reason in fail_reasons:
        if reason in ("MUSICAL_GOALS_VIOLATION",):
            severity["critical"].append(reason)
        elif reason in ("NOISE_TEXTURE_INCOHERENT",):
            severity["high"].append(reason)
        elif reason in ("GOOSEBUMPS_LOW", "VQI_BELOW_THRESHOLD"):
            severity["medium"].append(reason)
        else:
            severity["low"].append(reason)

    # Vorschläge basierend auf den Fehler-Ursachen
    suggestions = []
    if "MUSICAL_GOALS_VIOLATION" in fail_reasons:
        suggestions.append(
            "Musical Goals verletzt → Goal-Directed Candidate Recovery: "
            "Prüfe welche der 14 Goals unter Threshold liegen und justiere "
            "die betroffenen Phasen."
        )
    if "NOISE_TEXTURE_INCOHERENT" in fail_reasons:
        suggestions.append(
            "Noise-Texture inkohärent → Noise-Texture-Repair: Das Restsignal "
            "nach Defektentfernung klingt unnatürlich. Prüfe Phase 03 (ML-Denoising), "
            "Phase 24 (Spectral Noise Reduction) und Phase 50 (Codec Repair)."
        )
    if "GOOSEBUMPS_LOW" in fail_reasons:
        suggestions.append(
            "Emotionale Wirkung reduziert → Frisson/Goosebumps-Protection: "
            "Prüfe ob EmotionalArcPreservation die Dynamik zu stark glättet. "
            "Phase 17 (Mastering Polish) kann emotionale Peaks abschneiden."
        )
    if "VQI_BELOW_THRESHOLD" in fail_reasons:
        suggestions.append(
            f"Vocal Quality {vqi:.3f} < {case.get('vqi_floor', 0.72):.3f} → "
            f"Vocal VQI Recovery: Prüfe Phase 42 (Vocal Enhancement) und "
            f"Phase 58 (Lyrics-Guided Enhancement)."
        )

    return {
        "case_id": case.get("case_id", "unknown"),
        "material": case.get("metadata", {}).get("material", "unknown"),
        "hpi": hpi,
        "quality_estimate": quality,
        "vqi": vqi,
        "failure_severity": dict(severity),
        "suggestions": suggestions,
    }


def analyze_all(report: dict) -> list[dict]:
    """Analysiert alle Fälle und sortiert nach Schweregrad."""
    cases = report.get("cases", [])
    analyzed = [analyze_case(c) for c in cases]

    # Sortiere: critical > high > medium > low, dann nach HPI aufsteigend
    def sort_key(a: dict) -> tuple[int, float]:
        sev = a["failure_severity"]
        level = 0
        if sev.get("critical"):
            level = 0
        elif sev.get("high"):
            level = 1
        elif sev.get("medium"):
            level = 2
        else:
            level = 3
        return (level, -(a["hpi"] or 0))

    analyzed.sort(key=sort_key)
    return analyzed


def generate_markdown(report: dict) -> str:
    """Generiert einen detaillierten Markdown-Diagnosebericht."""
    gate = report.get("gate", {})
    cases = analyze_all(report)

    lines = [
        "# Aurik Quality Gate — Fall-für-Fall-Diagnose",
        "",
        f"**Generiert:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Fälle:** {len(cases)} ({gate.get('real_audio_cases', '?')} total)",
        f"**Gate:** {'✅ BESTANDEN' if gate.get('passed') else '❌ NICHT BESTANDEN'}",
        "",
        "---",
        "",
        "## Zusammenfassung",
        "",
        f"| Metrik | Wert | Ziel | Status |",
        f"|--------|------|------|--------|",
        f"| HPI Ø | {gate.get('hpi_average', '?'):.3f} | ≥ 0.78 | {'✅' if (gate.get('hpi_average') or 0) >= 0.78 else '❌'} |",
        f"| Quality Ø | {gate.get('quality_estimate_average', '?'):.3f} | ≥ 0.84 | {'✅' if (gate.get('quality_estimate_average') or 0) >= 0.84 else '❌'} |",
        f"| Musical Goals | {gate.get('musical_goal_case_pass_rate', 0):.1%} | ≥ 90% | {'✅' if gate.get('musical_goal_case_pass_rate', 0) >= 0.90 else '❌'} |",
        f"| Noise Texture | {gate.get('noise_texture_case_pass_rate', 0):.1%} | ≥ 94% | {'✅' if gate.get('noise_texture_case_pass_rate', 0) >= 0.94 else '❌'} |",
        f"| Goosebumps | {gate.get('goosebumps_case_pass_rate', 0):.1%} | ≥ 90% | {'✅' if gate.get('goosebumps_case_pass_rate', 0) >= 0.90 else '❌'} |",
        f"| Vocal Floor | {gate.get('vocal_floor_pass_rate', 0):.1%} | 100% | {'✅' if gate.get('vocal_floor_pass_rate', 0) >= 1.0 else '❌'} |",
        f"| RT-Faktor | {gate.get('runtime_factor', 0):.1f}× | ≤ 8.0× | {'✅' if gate.get('runtime_factor', 0) <= 8.0 else '❌'} |",
        f"| Real-Audio Fälle | {gate.get('real_audio_cases', 0)} | ≥ 80 | {'✅' if gate.get('real_audio_cases', 0) >= 80 else '❌'} |",
        "",
        "---",
        "",
        "## Priorisierte Fall-Analyse",
        "",
    ]

    for case in cases:
        severity_icon = "🔴" if case["failure_severity"].get("critical") else \
                       "🟠" if case["failure_severity"].get("high") else \
                       "🟡" if case["failure_severity"].get("medium") else "🟢"
        lines.append(f"### {severity_icon} {case['case_id']}")
        lines.append("")
        lines.append(f"- **Material:** {case['material']}")
        lines.append(f"- **HPI:** {case['hpi']:.4f}" if case["hpi"] else "- **HPI:** —")
        lines.append(f"- **Quality:** {case['quality_estimate']:.4f}" if case["quality_estimate"] else "- **Quality:** —")
        lines.append(f"- **VQI:** {case['vqi']:.4f}" if case["vqi"] else "- **VQI:** —")

        all_reasons = []
        for sev, reasons in sorted(case["failure_severity"].items()):
            sev_label = {"critical": "KRITISCH", "high": "HOCH", "medium": "MITTEL", "low": "NIEDRIG"}.get(sev, sev)
            all_reasons.append(f"**{sev_label}:** {', '.join(reasons)}")
        if all_reasons:
            lines.append("- **Fehler:** " + " | ".join(all_reasons))

        if case["suggestions"]:
            lines.append("- **Aktionen:**")
            for s in case["suggestions"]:
                lines.append(f"  1. {s}")
        lines.append("")

    # Priorisierte globale Aktionen
    gate_actions = gate.get("prioritized_actions", [])
    if gate_actions:
        lines.append("## Globale priorisierte Aktionen")
        lines.append("")
        for i, action in enumerate(gate_actions, 1):
            lines.append(f"{i}. `{action}`")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Quality-Gate-Diagnose")
    parser.add_argument("--report", help="Pfad zum Quality-Gate-Report")
    parser.add_argument("--case", help="Nur einen spezifischen Fall analysieren")
    parser.add_argument("--output", help="Ausgabedatei (Markdown)")
    parser.add_argument("--json", action="store_true", help="JSON-Ausgabe")
    args = parser.parse_args()

    report = load_report(Path(args.report) if args.report else None)

    if args.case:
        # Einzelner Fall
        cases = report.get("cases", [])
        matched = [c for c in cases if c.get("case_id") == args.case]
        if not matched:
            print(f"Fall '{args.case}' nicht gefunden.", file=sys.stderr)
            sys.exit(1)
        analysis = analyze_case(matched[0])
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        return 0

    if args.json:
        print(json.dumps({
            "analyzed_cases": analyze_all(report),
            "gate": report.get("gate"),
        }, indent=2, ensure_ascii=False))
    else:
        md = generate_markdown(report)
        print(md)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
            print(f"\nDiagnose gespeichert: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
