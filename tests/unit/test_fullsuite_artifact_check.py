"""test_fullsuite_artifact_check — §CI-FULLSUITE-ARTEFAKT (Vorschlag 07).

Unit-Tests der puren evaluate_artifact-Funktion:
  frisch + grün ⇒ OK | alt ⇒ Verstoß | failed>0 ⇒ Verstoß | fehlend ⇒ Verstoß
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from scripts.check_fullsuite_artifact import MAX_ARTIFACT_AGE_DAYS, evaluate_artifact

_NOW = time.time()
_GREEN_SUMMARY = (
    "# Fullsuite-Summary\n\n"
    "- Datum: 2026-08-15\n- HEAD: abc1234\n"
    "- Ergebnis: 0 failed, 19000 passed, 31 skipped, 580 deselected\n- Exit: 0\n"
)
_RED_SUMMARY = _GREEN_SUMMARY.replace("0 failed", "151 failed")


def _write(tmp: Path, log_text: str, summary_text: str, age_days: float = 0.0) -> tuple[Path, Path]:
    log_p = tmp / "fullsuite_latest.log"
    sum_p = tmp / "fullsuite_summary.md"
    log_p.write_text(log_text, encoding="utf-8")
    sum_p.write_text(summary_text, encoding="utf-8")
    _mtime = _NOW - age_days * 86400
    os.utime(log_p, (_mtime, _mtime))
    os.utime(sum_p, (_mtime, _mtime))
    return log_p, sum_p


def test_fresh_green_artifact_is_ok(tmp_path: Path):
    log_p, sum_p = _write(tmp_path, "=== VOLLSUITE ENDE — Exit 0 ===\n", _GREEN_SUMMARY)
    ok, violations = evaluate_artifact(log_p, sum_p, now=_NOW)
    assert ok is True
    assert violations == []


def test_failed_summary_is_violation(tmp_path: Path):
    log_p, sum_p = _write(tmp_path, "=== VOLLSUITE ENDE — Exit 1 ===\n", _RED_SUMMARY)
    ok, violations = evaluate_artifact(log_p, sum_p, now=_NOW)
    assert ok is False
    assert any("151 failed" in v for v in violations)


def test_stale_artifact_is_violation(tmp_path: Path):
    log_p, sum_p = _write(
        tmp_path, "=== VOLLSUITE ENDE — Exit 0 ===\n", _GREEN_SUMMARY, age_days=MAX_ARTIFACT_AGE_DAYS + 1
    )
    ok, violations = evaluate_artifact(log_p, sum_p, now=_NOW)
    assert ok is False
    assert any("zu alt" in v for v in violations)


def test_missing_artifacts_are_violations(tmp_path: Path):
    ok, violations = evaluate_artifact(tmp_path / "fehlt.log", tmp_path / "fehlt.md", now=_NOW)
    assert ok is False
    assert len(violations) == 1
    assert "fehlt" in violations[0]
