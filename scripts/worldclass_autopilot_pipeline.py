#!/usr/bin/env python3
"""End-to-End-Orchestrierung fuer den Worldclass-Autopilot-Workflow.

Ablauf:
1) Revalidierungsplan erstellen (optional)
2) WP1 laufen lassen (dry-run oder execute)
3) Resultate zusammenfassen
4) KPI-Dashboard erzeugen
5) Trusted Vocal Restoration Report erzeugen
6) Harten Release-Gate ausfuehren
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _latest_run_dir(base: Path) -> Path:
    runs = sorted([p for p in base.glob("class_c_reval_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"Kein Revalidierungs-Run unter {base} gefunden.")
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Worldclass-Autopilot E2E-Workflow")
    parser.add_argument("--repo-root", default=".", help="Repo-Root")
    parser.add_argument("--manifest", default="config/class_c_revalidation_manifest.example.json")
    parser.add_argument("--revalidation-out", default="reports/revalidation")
    parser.add_argument("--run-dir", default="", help="Expliziter Run-Ordner. Wenn leer: neu erstellen/letzter Run")
    parser.add_argument("--create-plan", action="store_true", help="Erstellt einen neuen Klasse-C-Revalidierungsplan")
    parser.add_argument("--execute-wp1", action="store_true", help="Fuehrt WP1 im Execute-Modus aus")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--ml-runtime-budget-s", type=float, default=20.0)
    parser.add_argument("--threshold-config", default="config/worldclass_kpi_thresholds.json")
    parser.add_argument("--worldclass-out", default="reports/worldclass")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    py = sys.executable  # type: ignore[name-defined]
    if not Path(py).exists():
        raise SystemExit(f"Python-Interpreter fehlt: {py}")

    reval_base = (repo_root / args.revalidation_out).resolve()

    if args.create_plan:
        _run(
            [
                str(py),
                "scripts/run_class_c_revalidation_plan.py",
                "--manifest",
                args.manifest,
                "--out-dir",
                args.revalidation_out,
            ],
            cwd=repo_root,
        )

    if args.run_dir:
        run_dir = (repo_root / args.run_dir).resolve()
    else:
        run_dir = _latest_run_dir(reval_base)

    wp1_cmd = [
        str(py),
        "scripts/run_wp1_material_vqi_revalidation.py",
        "--run-dir",
        str(run_dir),
        "--max-cases",
        str(max(0, args.max_cases)),
        "--max-seconds",
        str(max(0.0, args.max_seconds)),
        "--ml-runtime-budget-s",
        str(max(1.0, args.ml_runtime_budget_s)),
    ]
    if args.execute_wp1:
        wp1_cmd.append("--execute")
    _run(wp1_cmd, cwd=repo_root)

    _run(
        [
            str(py),
            "scripts/summarize_class_c_revalidation_results.py",
            "--input-csv",
            str(run_dir / "result_template.csv"),
        ],
        cwd=repo_root,
    )

    _run(
        [
            str(py),
            "scripts/worldclass_kpi_dashboard.py",
            "--repo-root",
            ".",
            "--revalidation-run-dir",
            str(run_dir),
            "--threshold-config",
            args.threshold_config,
            "--out-dir",
            args.worldclass_out,
        ],
        cwd=repo_root,
    )

    _run(
        [
            str(py),
            "scripts/trusted_vocal_restoration_report.py",
            "--result-csv",
            str(run_dir / "result_template.csv"),
            "--dashboard-json",
            str((repo_root / args.worldclass_out / "worldclass_kpi_dashboard.json").resolve()),
            "--threshold-config",
            args.threshold_config,
            "--out-dir",
            args.worldclass_out,
        ],
        cwd=repo_root,
    )

    # Release-Gate ist absichtlich der letzte harte Schritt.
    _run(
        [
            str(py),
            "scripts/worldclass_release_gate.py",
            "--dashboard-json",
            str((repo_root / args.worldclass_out / "worldclass_kpi_dashboard.json").resolve()),
            "--trusted-report-json",
            str((repo_root / args.worldclass_out / "trusted_vocal_restoration_report.json").resolve()),
        ],
        cwd=repo_root,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Pipeline-Schritt fehlgeschlagen mit Exit-Code {exc.returncode}")
        raise SystemExit(exc.returncode) from exc
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(2) from exc
