# Typing-Imports für Typannotationen
import json
import os
from typing import Any


def collect_results(result_dir="results_sota"):
    summary: Any = {"jobs": [], "scores": [], "errors": [], "policy_events": []}
    for fname in os.listdir(result_dir):
        if fname.startswith("result_") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                job = json.load(f)
                summary["jobs"].append(job)
        if fname.startswith("qalog_") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                qas = json.load(f)
                for entry in qas:
                    if entry.get("event") == "score":
                        summary["scores"].append(entry["score"])
                    if entry.get("event") == "error":
                        summary["errors"].append(entry["msg"])
        if fname.startswith("policylog_") and fname.endswith(".json"):
            with open(os.path.join(result_dir, fname)) as f:
                events = json.load(f)
                summary["policy_events"].extend(events)
    # Statistiken
    if summary["scores"]:
        summary["score_mean"] = sum(summary["scores"]) / len(summary["scores"])
        summary["score_min"] = min(summary["scores"])
        summary["score_max"] = max(summary["scores"])
    else:
        summary["score_mean"] = summary["score_min"] = summary["score_max"] = None
    summary["error_count"] = len(summary["errors"])
    summary["policy_event_count"] = len(summary["policy_events"])
    with open(os.path.join(result_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SOTA] Report geschrieben: {os.path.join(result_dir, 'report.json')}")


if __name__ == "__main__":
    collect_results()
