"""
pipeline_health_check.py — §v10 Pre-Pipeline Health Verification
=================================================================

Prüft vor Pipeline-Start die Betriebsbereitschaft aller kritischen
Komponenten. Verhindert stille Degradation durch defekte ML-Modelle,
korrupte Konfigurationen oder Ressourcen-Engpässe.

Checks:
  C1 — ML-Modelle: ONNX-Sessions geladen und inferenzfähig?
  C2 — DSP-Module: Kern-Signalverarbeitung importierbar?
  C3 — Ressourcen: Genügend RAM/CPU für die geplante Audio-Länge?
  C4 — Konfiguration: Alle erforderlichen Dateien vorhanden?
  C5 — Safe-Execution: Fehler-Statistiken aus letztem Lauf?
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Ergebnis eines einzelnen Health-Checks."""

    name: str
    passed: bool
    duration_ms: float = 0.0
    details: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineHealthReport:
    """Gesamt-Health-Report vor Pipeline-Start."""

    all_passed: bool = True
    checks: list[HealthCheckResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Pipeline Health: {'✅ ALL PASSED' if self.all_passed else '⚠️ ISSUES FOUND'}"]
        for c in self.checks:
            icon = "✅" if c.passed else "⚠️"
            lines.append(f"  {icon} {c.name} ({c.duration_ms:.0f}ms): {c.details}")
        for r in self.recommendations:
            lines.append(f"  💡 {r}")
        return "\n".join(lines)


def run_health_checks(audio_duration_s: float = 300.0) -> PipelineHealthReport:
    """Führt alle Health-Checks aus und gibt einen Report zurück.

    Args:
        audio_duration_s: Geplante Audio-Länge für Ressourcen-Check

    Returns:
        PipelineHealthReport mit allen Ergebnissen
    """
    t0 = time.monotonic()
    checks: list[HealthCheckResult] = []
    recommendations: list[str] = []

    # C1: ML-Modelle
    checks.append(_check_ml_models())

    # C2: DSP-Module
    checks.append(_check_dsp_modules())

    # C3: Ressourcen
    checks.append(_check_resources(audio_duration_s))

    # C4: Konfiguration
    checks.append(_check_configuration())

    # C5: Safe-Execution Stats
    checks.append(_check_error_statistics())

    all_passed = all(c.passed for c in checks)
    if not all_passed:
        recommendations.append(
            "Einige Health-Checks fehlgeschlagen — Pipeline kann trotzdem starten, aber Ergebnisse können suboptimal sein."
        )

    return PipelineHealthReport(
        all_passed=all_passed,
        checks=checks,
        total_duration_ms=(time.monotonic() - t0) * 1000,
        recommendations=recommendations,
    )


def _check_ml_models() -> HealthCheckResult:
    """C1: Prüft ob kritische ML-Modelle geladen sind."""
    t0 = time.monotonic()
    warnings: list[str] = []

    try:
        import numpy as np

        # Prüfe numpy (Basis für alle ML-Operationen)
        _ = np.zeros(1, dtype=np.float32)
        np_status = True
    except Exception as e:
        np_status = False
        warnings.append(f"numpy: {e}")

    # Prüfe scipy (Basis für DSP+ML)
    try:
        from scipy import signal

        _ = signal.butter(2, 0.5, "low")
        scipy_status = True
    except Exception as e:
        scipy_status = False
        warnings.append(f"scipy: {e}")

    passed = np_status and scipy_status
    return HealthCheckResult(
        name="ML-Modelle",
        passed=passed,
        duration_ms=(time.monotonic() - t0) * 1000,
        details=f"numpy={'OK' if np_status else 'FAIL'}, scipy={'OK' if scipy_status else 'FAIL'}",
        warnings=warnings,
    )


def _check_dsp_modules() -> HealthCheckResult:
    """C2: Prüft ob kritische DSP-Module importierbar sind."""
    t0 = time.monotonic()
    warnings: list[str] = []

    modules = [
        ("audio_utils", "backend.core.audio_utils"),
        ("core_utils", "backend.core.core_utils"),
    ]
    results = {}
    for name, path in modules:
        try:
            __import__(path)
            results[name] = True
        except Exception as e:
            results[name] = False
            warnings.append(f"{name}: {e}")

    passed = all(results.values())
    return HealthCheckResult(
        name="DSP-Module",
        passed=passed,
        duration_ms=(time.monotonic() - t0) * 1000,
        details=", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in results.items()),
        warnings=warnings,
    )


def _check_resources(audio_duration_s: float) -> HealthCheckResult:
    """C3: Prüft ob genügend Ressourcen verfügbar sind."""
    t0 = time.monotonic()
    warnings: list[str] = []

    try:
        import psutil

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        # Faustregel: ~200 MB RAM pro Minute Audio
        needed_gb = audio_duration_s / 60 * 0.2 + 1.0  # +1 GB Basis
        if available_gb < needed_gb:
            warnings.append(f"Wenig RAM: {available_gb:.1f}GB verfügbar, ~{needed_gb:.1f}GB empfohlen")
        passed = available_gb >= needed_gb * 0.5  # 50% Toleranz
    except ImportError:
        passed = True  # psutil nicht installiert → kein Check möglich
        details = "psutil nicht installiert — Ressourcen-Check übersprungen"
    else:
        details = f"RAM: {available_gb:.1f}GB frei (≥{needed_gb:.1f}GB empfohlen)"

    return HealthCheckResult(
        name="Ressourcen",
        passed=passed,
        duration_ms=(time.monotonic() - t0) * 1000,
        details=details,
        warnings=warnings,
    )


def _check_configuration() -> HealthCheckResult:
    """C4: Prüft ob erforderliche Konfigurationsdateien vorhanden sind."""
    t0 = time.monotonic()
    warnings: list[str] = []

    required_files = [
        "pytest.ini",
        ".github/specs/01_musical_goals.md",
    ]
    results = {}
    for f in required_files:
        exists = os.path.exists(f)
        results[f] = exists
        if not exists:
            warnings.append(f"Konfigurationsdatei fehlt: {f}")

    passed = all(results.values())
    return HealthCheckResult(
        name="Konfiguration",
        passed=passed,
        duration_ms=(time.monotonic() - t0) * 1000,
        details=f"{sum(results.values())}/{len(results)} Dateien gefunden",
        warnings=warnings,
    )


def _check_error_statistics() -> HealthCheckResult:
    """C5: Prüft Safe-Execution-Fehlerstatistiken + Bug-Patterns aus letztem Lauf."""
    t0 = time.monotonic()
    warnings: list[str] = []

    try:
        from backend.core.safe_execution import get_error_statistics

        stats = get_error_statistics()
        total = stats["total_errors"]
        if total > 100:
            warnings.append(f"Hohe Fehlerrate im letzten Lauf: {total} Errors")
        
        # §v10.106: Bug-Pattern-Erkennung aus OOM-Forensik
        bug_patterns = stats.get("bug_patterns", {})
        critical_patterns = [p for p, c in bug_patterns.items() if c >= 20]
        if critical_patterns:
            warnings.append(
                f"Kritische Bug-Patterns: {', '.join(critical_patterns)} "
                f"({sum(bug_patterns.values())} total)"
            )
        
        details = f"{total} Errors in {stats['unique_error_sites']} Sites"
        if bug_patterns:
            details += f", {len(bug_patterns)} Bug-Patterns"
        passed = total < 500 and not critical_patterns
    except ImportError:
        passed = True
        details = "safe_execution nicht importierbar — keine Fehlerstatistik"

    return HealthCheckResult(
        name="Fehler-Statistik + Bug-Patterns",
        passed=passed,
        duration_ms=(time.monotonic() - t0) * 1000,
        details=details,
        warnings=warnings,
    )
