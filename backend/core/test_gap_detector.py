"""
test_gap_detector.py — v10 Test-Gap-Detector
=============================================
Finds spec paragraphs without tests and material coverage gaps.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CRITICAL_SPECS = ["0h", "0i", "0p", "0a", "2.44", "2.49"]


@dataclass
class GapResult:
    spec_ref: str
    description: str
    severity: str = "warning"


@dataclass
class TestGapReport:
    untested_specs: list[GapResult] = field(default_factory=list)
    critical_gaps: list[GapResult] = field(default_factory=list)
    material_gaps: list[str] = field(default_factory=list)
    total_specs_found: int = 0
    total_tests_found: int = 0
    coverage_pct: float = 0.0


class TestGapDetector:
    def __init__(self, repo_root: str = ".") -> None:
        self._root = Path(repo_root)

    def scan_all(self) -> TestGapReport:
        report = TestGapReport()
        spec_refs = self._extract_refs(exclude_tests=True)
        test_refs = self._extract_refs(exclude_tests=False)
        report.total_specs_found = len(spec_refs)
        report.total_tests_found = len(test_refs)
        untested = spec_refs - test_refs
        if spec_refs:
            report.coverage_pct = (len(spec_refs) - len(untested)) / max(len(spec_refs), 1) * 100
        for ref in sorted(untested):
            sev = "critical" if ref in CRITICAL_SPECS else "warning"
            g = GapResult(ref, f"Spec {ref} has no test", sev)
            report.untested_specs.append(g)
            if sev == "critical":
                report.critical_gaps.append(g)
        return report

    def _extract_refs(self, exclude_tests: bool) -> set[str]:
        refs: set[str] = set()
        pattern = re.compile(r"([0-9]+[a-z]?(?:\.[0-9]+)?[a-z]?)")
        for py_file in self._root.rglob("*.py"):
            p = str(py_file)
            if "__pycache__" in p:
                continue
            if exclude_tests and "/tests/" in p:
                continue
            if not exclude_tests and "/tests/" not in p:
                continue
            try:
                for m in pattern.finditer(py_file.read_text(errors="ignore")):
                    refs.add(m.group(1))
            except Exception as _e:
                logger.debug("test_gap_detector: unkritisch exception: %s", _e)
        return refs


_detector: TestGapDetector | None = None


def get_detector() -> TestGapDetector:
    global _detector
    if _detector is None:
        _detector = TestGapDetector()
    return _detector
