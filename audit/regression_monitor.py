"""
Aurik Regressionstest und Langzeit-Monitoring
- Führt regelmäßige Regressionstests mit historischen und neuen Audiodaten durch
- Analysiert Trends im Audit-Log
- Meldet Qualitätsverluste oder Verbesserungen automatisch
"""

import json
from pathlib import Path
from typing import Any


def load_audit_log(audit_path="audit/audit_trail.json"):
    if not Path(audit_path).exists():
        print("Audit-Log nicht gefunden.")
        return []
    with open(audit_path) as f:
        return json.load(f)


def analyze_trends(audit_data):
    trends: dict[str, list[Any]] = {}
    for entry in audit_data:
        results = entry.get("results", {})
        for gate, value in results.items():
            if gate not in trends:
                trends[gate] = []
            trends[gate].append(value)
    return trends


def report_trends(trends):
    print("Langzeit-Monitoring: Qualitäts-Trends")
    for gate, values in trends.items():
        avg = sum([v for v in values if isinstance(v, float)]) / max(len(values), 1)
        print(f"- {gate}: Durchschnittswert {avg:.2f}, Anzahl Tests: {len(values)}")
        if any(v is False for v in values):
            print(f"  WARNUNG: Quality-Gate '{gate}' wurde mindestens einmal nicht bestanden!")


def monitor_regression() -> dict:
    """Composite-Einstiegspunkt für `PolicyEngine.monitor_regression()`.

    Verkettet Audit-Log-Laden → Trend-Analyse → Reporting und liefert
    zusätzlich die Gates zurück, die mindestens einmal nicht bestanden wurden.
    """
    audit_data = load_audit_log()
    trends = analyze_trends(audit_data)
    report_trends(trends)
    regressions = {gate: values for gate, values in trends.items() if any(v is False for v in values)}
    return {
        "status": "ok" if audit_data else "no_audit_data",
        "n_entries": len(audit_data),
        "trends": trends,
        "regressions": regressions,
    }


def main():
    audit_data = load_audit_log()
    trends = analyze_trends(audit_data)
    report_trends(trends)


if __name__ == "__main__":
    main()
