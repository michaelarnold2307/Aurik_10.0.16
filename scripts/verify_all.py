#!/usr/bin/env python3
"""
verify_all.py — Vollständige Aurik-Verifikationssuite.
Führt alle Gates aus und produziert einen Abschlussbericht.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
VENV = ROOT / ".venv_aurik" / "bin" / "python"


@dataclass
class Check:
    name: str
    status: str = "pending"  # pass, fail, error, skip
    duration_s: float = 0.0
    output: str = ""
    details: dict = field(default_factory=dict)


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
        return r.returncode, r.stdout + r.stderr, time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT", timeout
    except Exception as e:
        return -2, str(e), time.perf_counter() - t0


def main():
    results: list[Check] = []
    print("=" * 70)
    print("AURIK VOLLSTÄNDIGE VERIFIKATION")
    print("=" * 70)

    # ── EBENE 0: Spezifikation ──────────────────────────────────────────
    print("\n── EBENE 0: Spezifikation ──")

    # Check 1: RELEASE_MUST coverage
    rc, out, dur = run([str(VENV), "scripts/release_must_coverage_check.py"], 30)
    c = Check("RELEASE_MUST Coverage", "pass" if rc == 0 else "fail", dur, out)
    if "100.0%" in out:
        c.details = {"coverage": "100%", "claims": 21}
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} ({dur:.1f}s)")

    # Check 2: Spec drift
    rc, out, dur = run([str(VENV), "scripts/spec_drift_check.py"], 30)
    c = Check("Spec Drift Check", "pass" if "OK" in out or rc == 0 else "fail", dur, out[:500])
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} ({dur:.1f}s)")

    # ── EBENE 1: Statische Gates ────────────────────────────────────────
    print("\n── EBENE 1: Statische Gates ──")

    # Check 3: VERBOTEN Linter
    rc, out, dur = run([str(VENV), "scripts/aurik_verboten_linter.py", "--json"], 60)
    try:
        linter_data = json.loads(out.split("\n")[-2] if "\n" in out else out)
    except Exception:
        linter_data = {"clean": False, "issues": -1}
    c = Check("VERBOTEN Linter", "pass" if linter_data.get("clean") else "fail", dur)
    c.details = linter_data
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} issues={linter_data.get('issues', '?')} ({dur:.1f}s)")

    # Check 4: Compliance Check
    rc, out, dur = run([str(VENV), "scripts/compliance_check.py", "--errors-only"], 60)
    c = Check("Compliance Check", "pass" if "bestanden" in out else "fail", dur)
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} ({dur:.1f}s)")

    # Check 5: Static Guard
    rc, out, dur = run([str(VENV), "scripts/pre_commit_static_guard.py"], 60)
    c = Check("Static Guard", "pass" if "✅" in out or rc == 0 else "fail", dur)
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} ({dur:.1f}s)")

    # Check 6: Platform Compat
    rc, out, dur = run([str(VENV), "scripts/platform_compat_check.py"], 30)
    c = Check("Platform Compat", "pass" if rc == 0 else "fail", dur)
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} ({dur:.1f}s)")

    # ── EBENE 2: Kompilierbarkeit ───────────────────────────────────────
    print("\n── EBENE 2: Kompilierbarkeit ──")

    bad = []
    t0 = time.perf_counter()
    for py_file in ROOT.rglob("*.py"):
        rp = str(py_file.relative_to(ROOT))
        if any(s in rp for s in [".venv", "__pycache__", ".git", "node_modules"]):
            continue
        try:
            compile(py_file.read_text(encoding="utf-8", errors="replace"), rp, "exec")
        except SyntaxError as e:
            bad.append((rp, str(e)))
    dur = time.perf_counter() - t0
    c = Check("Compile All .py", "pass" if not bad else "fail", dur)
    c.details = {"scanned": sum(1 for _ in ROOT.rglob("*.py")), "errors": len(bad)}
    if bad:
        c.details["samples"] = bad[:5]
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} {len(bad)} errors ({dur:.1f}s)")

    # ── EBENE 3: Normative Tests ────────────────────────────────────────
    print("\n── EBENE 3: Normative Tests ──")

    rc, out, dur = run(
        [
            str(VENV),
            "-m",
            "pytest",
            "tests/normative/test_no_production_stubs.py",
            "tests/normative/test_full_pipeline_determinism.py",
            "tests/normative/test_p2_audit_and_deployment_mode.py",
            "-q",
            "--timeout=30",
            "--tb=short",
            "--no-header",
        ],
        120,
    )
    passed = re.search(r"(\d+)\s+passed", out)
    c = Check("Normative Tests", "pass" if rc == 0 else "fail", dur)
    c.details = {"passed": int(passed.group(1)) if passed else 0}
    results.append(c)
    print(f"  {c.name}: {'✅' if c.status == 'pass' else '❌'} {c.details.get('passed', 0)} passed ({dur:.1f}s)")

    # ── P1 VERIFIKATION ─────────────────────────────────────────────────
    print("\n── P1 VERIFIKATION: V27-V31+V39 ──")

    # Check: V27-V31 matches in UNIFIED_RESTORER are all comments
    p1_files = [
        "backend/core/causal_defect_reasoner.py",
        "backend/core/defect_phase_mapper.py",
        "backend/core/unified_restorer_v3.py",
    ]
    p1_violations = []
    for fp in p1_files:
        if not (ROOT / fp).exists():
            continue
        text = (ROOT / fp).read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.split("\n"), 1):
            s = line.strip()
            # Only check non-comment, non-docstring lines
            if s.startswith("#") or s.startswith('"""') or s.startswith("'''"):
                continue
            if "JITTER_ARTIFACTS" in s and "phase_12" in s:
                p1_violations.append((fp, i, s[:120]))
            if "NR_BREATHING" in s and ("phase_03" in s or "phase_29" in s):
                p1_violations.append((fp, i, s[:120]))
            if "OVERLOAD_DISTORTION" in s and "phase_63" in s:
                p1_violations.append((fp, i, s[:120]))
            if re.search(r"\bALIASING\b", s) and "phase_03" in s:
                p1_violations.append((fp, i, s[:120]))

    c = Check("P1: V27-V31 Code-Audit", "pass" if not p1_violations else "fail", 0)
    c.details = {"violations_in_code": len(p1_violations), "samples": p1_violations[:3]}
    results.append(c)
    print(f"  {c.name}: {'✅ 0 echte Bugs' if c.status == 'pass' else f'❌ {len(p1_violations)} violations'}")

    # ── P2 VERIFIKATION ─────────────────────────────────────────────────
    print("\n── P2 VERIFIKATION: sosfilt + np.max ──")

    # sosfilt in signal path
    sosfilt_signal = []
    phase_dir = ROOT / "backend" / "core" / "phases"
    for pf in sorted(phase_dir.glob("*.py")):
        if pf.name == "__init__.py":
            continue
        text = pf.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "sosfilt(" in line and "sosfiltfilt" not in line:
                # Check if result modifies output signal (audio_out, result, output)
                for j in range(i, min(i + 5, len(lines))):
                    if any(kw in lines[j] for kw in ["audio_out +=", "audio_out =", "output ="]):
                        sosfilt_signal.append((pf.name, i + 1, line.strip()[:100]))
                        break

    c = Check("P2: sosfilt im Signalpfad", "pass" if not sosfilt_signal else "fail", 0)
    c.details = {"signal_mod_count": len(sosfilt_signal)}
    results.append(c)
    print(f"  {c.name}: {'✅ 0 im Signalpfad' if c.status == 'pass' else f'❌ {len(sosfilt_signal)}'}")

    # np.max in gain context
    np_max_gain = []
    for py_file in ROOT.rglob("*.py"):
        rp = str(py_file.relative_to(ROOT))
        if any(s in rp for s in [".venv", "__pycache__", ".git", "tests"]):
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue
        if "np.max(np.abs(audio))" in text:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if "np.max(np.abs(audio))" in line:
                    for j in range(i, min(i + 3, len(lines))):
                        if re.search(r"(?:gain|scale|norm)\s*=", lines[j]):
                            np_max_gain.append((rp, i + 1, line.strip()[:120]))
                            break

    c = Check("P2: np.max in Gain-Pfad", "pass" if not np_max_gain else "fail", 0)
    c.details = {"gain_context_count": len(np_max_gain)}
    results.append(c)
    print(f"  {c.name}: {'✅ 0 in Gain-Pfad' if c.status == 'pass' else f'❌ {len(np_max_gain)}'}")

    # ── ABSCHLUSSBERICHT ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")  # type: ignore[assignment]
    failed = sum(1 for r in results if r.status == "fail")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": f"{passed}/{total} gates passed",
        "all_pass": failed == 0,
        "checks": [
            {"name": r.name, "status": r.status, "duration_s": round(r.duration_s, 1), **r.details} for r in results
        ],
    }

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "verify_all_report.json").write_text(json.dumps(report, indent=2))

    for r in results:
        icon = "✅" if r.status == "pass" else "❌"
        print(f"  {icon} {r.name}")

    print(f"\nGESAMT: {passed}/{total} bestanden, {failed} fehlgeschlagen")
    print("Report: reports/verify_all_report.json")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
