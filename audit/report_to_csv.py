# Typing-Imports für Typannotationen
import csv
import json
import os


def report_to_csv(result_dir="results_sota"):
    report_path = os.path.join(result_dir, "report.json")
    csv_path = os.path.join(result_dir, "report.csv")
    if not os.path.exists(report_path):
        print("[SOTA] Kein report.json gefunden.")
        return
    with open(report_path) as f:
        report = json.load(f)
    # Schreibe Scores und Fehler als CSV
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "job_id",
                "file",
                "score",
                "error",
                "policy_event_count",
                "primary_media",
                "transfer_chain",
                "quality_gates",
            ]
        )
        jobs = report.get("jobs", [])
        scores = report.get("scores", [])
        errors = report.get("errors", [])
        forensic_reports = report.get("forensic_reports", [])
        policy_event_count = report.get("policy_event_count", 0)
        for i, job in enumerate(jobs):
            score = scores[i] if i < len(scores) else ""
            error = errors[i] if i < len(errors) else ""
            forensic_report = None
            if i < len(forensic_reports):
                forensic_report = forensic_reports[i]
            primary_media = ""
            transfer_chain = ""
            if forensic_report:
                primary_media = (
                    getattr(forensic_report, "primary_media", "")
                    if hasattr(forensic_report, "primary_media")
                    else forensic_report.get("primary_media", "")
                )
                transfer_chain = (
                    getattr(forensic_report, "transfer_chain", [])  # type: ignore[assignment]
                    if hasattr(forensic_report, "transfer_chain")
                    else forensic_report.get("transfer_chain", [])
                )
                if isinstance(transfer_chain, list):
                    transfer_chain = " → ".join(str(m) for m in transfer_chain)
            # Quality-Gate-Details aus job extrahieren
            quality_gates = job.get("quality_gates", job.get("quality_details", ""))
            writer.writerow(
                [
                    job.get("job_id", ""),
                    job.get("file", ""),
                    score,
                    error,
                    policy_event_count,
                    primary_media,
                    transfer_chain,
                    str(quality_gates),
                ]
            )
    print(f"[SOTA] Report als CSV exportiert: {csv_path}")


if __name__ == "__main__":
    report_to_csv()
