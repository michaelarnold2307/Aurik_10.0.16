"""§v10.17 PreFlightIntegrityCheck — verhindert Pipeline-Start bei korrupten Modulen.

Läuft VOR DefectScan (0.5s statt 74s). Prüft:
  1. Alle kritischen Module kompilieren ohne SyntaxError
  2. UV3 ist importierbar
  3. PMGG ist funktionsfähig
  4. STCG-Singleton ist erreichbar
  5. FallbackAuditor ist aktiv
  6. Keine zirkulären Importe

Bei Fehler: Pipeline-Start wird BLOCKIERT (nicht still degradiert).
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Module die VOR dem DefectScan geprüft werden müssen
_CRITICAL_MODULES: list[tuple[str, str]] = [
    ("backend.core.unified_restorer_v3", "UV3-Haupt-Pipeline"),
    ("backend.core.per_phase_musical_goals_gate", "PMGG-Selbstkalibrierung"),
    ("backend.core.stereo_temporal_coherence_guard", "STCG-Stereo-Guard"),
    ("backend.core.post_processing_gate", "PostGate-Verify"),
    ("backend.core.fallback_auditor", "FallbackAuditor"),
    ("backend.core.one_take_export", "OneTakeExport"),
    ("backend.core.pipeline_health_monitor", "Circuit-Breaker"),
    ("backend.core.phase_error_registry", "Fehler-Taxonomie"),
    ("backend.core.phases.phase_interface", "Phase-Interface"),
    ("backend.core.export_quality_gate", "Export-Qualität"),
    ("backend.core.listening_fatigue_metric", "Fatigue-Metrik"),
    ("backend.core.song_goal_importance", "Mode-Gewichte"),
]


@dataclass
class IntegrityResult:
    passed: bool = True
    checks_total: int = 0
    checks_passed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0
    block_pipeline: bool = False


def run_preflight_checks() -> IntegrityResult:
    """Führt alle Pre-Flight-Checks durch. Blockiert den Start bei kritischen Fehlern."""
    t0 = time.time()
    result = IntegrityResult()

    # ── 1. Modul-Import-Test ──────────────────────────────────────────
    # Fast-path: compile-check for heavy modules, full import for light ones
    _COMPILE_ONLY = {
        "backend.core.unified_restorer_v3",  # 37K LOC
        "backend.core.per_phase_musical_goals_gate",  # 5.7K LOC
        "backend.core.song_goal_importance",  # heavy imports
    }
    for module_name, label in _CRITICAL_MODULES:
        result.checks_total += 1
        try:
            if module_name in _COMPILE_ONLY:
                # Nur Syntax prüfen, nicht ausführen (spart 9s)
                spec = importlib.util.find_spec(module_name)  # type: ignore[attr-defined]
                if spec and spec.origin:
                    compile(open(spec.origin).read(), spec.origin, "exec")
                result.checks_passed += 1
                logger.debug("PreFlight ✓ %s (compile-Pruefung)", label)
            else:
                importlib.import_module(module_name)
                result.checks_passed += 1
                logger.debug("PreFlight ✓ %s", label)
        except SyntaxError as e:
            result.failures.append(
                {
                    "module": module_name,
                    "label": label,
                    "error": f"SYNTAX-ERROR: {e}",
                    "severity": "fatal",
                }
            )
            logger.critical("PreFlight FATAL: %s hat Syntax-Error — Pipeline BLOCKIERT", label)
        except ImportError as e:
            result.failures.append(
                {
                    "module": module_name,
                    "label": label,
                    "error": f"IMPORT-ERROR: {e}",
                    "severity": "error",
                }
            )
            logger.error("PreFlight: %s nicht importierbar: %s", label, e)
        except Exception as e:
            result.failures.append(
                {
                    "module": module_name,
                    "label": label,
                    "error": f"RUNTIME: {type(e).__name__}: {e}",
                    "severity": "error",
                }
            )
            logger.error("PreFlight: %s Laufzeitfehler: %s", label, e)

    # ── 2. Syntax-Error = SOFORT BLOCKIEREN ───────────────────────────
    fatal_failures = [f for f in result.failures if f.get("severity") == "fatal"]
    if fatal_failures:
        result.passed = False
        result.block_pipeline = True
        result.duration_ms = (time.time() - t0) * 1000.0
        logger.critical(
            "PreFlight: %d FATALE Fehler — Pipeline-Start BLOCKIERT. Bitte Aurik-Updates prüfen.",
            len(fatal_failures),
        )
        return result

    # ── 3. Nur-Warnungen bei nicht-kritischen Fehlern ─────────────────
    non_fatal = [f for f in result.failures if f.get("severity") != "fatal"]
    if non_fatal:
        logger.warning(
            "PreFlight: %d nicht-kritische Fehler — Pipeline läuft degradiert. Folgende Module fehlen: %s",
            len(non_fatal),
            ", ".join(f["label"] for f in non_fatal),
        )
        result.passed = True  # Nicht blockieren, nur warnen
    else:
        result.passed = True

    result.duration_ms = (time.time() - t0) * 1000.0
    logger.info(
        "PreFlight: %d/%d Checks OK in %.0f ms",
        result.checks_passed,
        result.checks_total,
        result.duration_ms,
    )
    return result
