"""Competitive-Gate-Results — Ergebnis-Dataclasses.
Spec 15 paragraph 1.3. OQS-Delta, Timbre-Fidelity, artifact_freedom, Laufzeit.

Autor: Aurik 10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    tool_name: str
    scenario: str
    oqs_delta: float = 0.0
    timbre_fidelity: float = 0.0
    artifact_freedom: float = 0.0
    runtime_s: float = 0.0
    passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)
    total_passed: int = 0
    total_failed: int = 0
    aurik_version: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "aurik_version": self.aurik_version,
            "timestamp": self.timestamp,
            "results": [{k: v for k, v in r.__dict__.items() if not k.startswith("_")} for r in self.results],
        }


def create_empty_report() -> GateReport:
    from datetime import datetime

    return GateReport(timestamp=datetime.now().isoformat())
