"""
Guard Effectiveness Auditor — Aurik §v10.5

Der Meta-Guard: Audit aller Schutzmechanismen auf Paralysis-Ereignisse.

PROBLEM:
  PMGG, CIG, ArtifactFreedomGate und ContentIntegrityGuard reduzieren die
  Strength von Phasen im Retry auf bis zu 6% — effektive Deaktivierung.
  Fünf Kernphasen (03, 01, 09, 24, 29) sind dokumentiert betroffen.

ARCHITEKTUR:
  Phase läuft → PMGG tracked strength + retry action
  → Auditor prüft: war final_strength < 25%?
  → Wenn ja: False-Positive-Check mit MediaDefectVerifier Alternativ-Proxies
  → Wenn false positive: Recovery-Empfehlung (re-run mit voller Strength)

DATENMODELL:
  ParalysisEvent: Phase + Ziel-Goal + Original-Regression + Alternativ-Proxy
  AuditorReport:  Alle Paralysis-Ereignisse + Recovery-Empfehlungen

INTEGRATION:
  - PMGG wrap_phase: nach jedem Retry/best_effort tracken
  - AdaptivePipeline: vor dem Finalisieren Auditor-Report abrufen
  - Auto-Recovery: paralysierte Phasen mit voller Strength + proxy-skip re-runnen

Author: Aurik v10.5 Development
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────
PARALYSIS_STRENGTH_THRESHOLD: float = 0.25  # Unterhalb = paralysiert
PARALYSIS_RETRY_COUNT_THRESHOLD: int = 3  # Retries bevor best_effort
FALSE_POSITIVE_RATIO: float = 0.65  # Alt-Regression < Original * 0.65 = false positive


@dataclass
class ParalysisEvent:
    """Ein einzelnes Paralysis-Ereignis: Phase wurde durch Guard deaktiviert."""

    phase_id: str
    initial_strength: float
    final_strength: float
    retries_exhausted: int
    pmgg_action: str  # "best_effort", "best_effort_r1", etc.
    goal_triggered: str = ""  # Welches Goal hat die Regression ausgelöst
    original_regression: float = 0.0  # Originale PMGG-Regression
    alternative_regression: float = 0.0  # MediaDefectVerifier Alternativ-Regression
    is_false_positive: bool = False
    audio_before: np.ndarray | None = None  # Für Recovery-Re-Run
    audio_after: np.ndarray | None = None
    scores_before: dict[str, float] | None = None
    scores_after: dict[str, float] | None = None


@dataclass
class AuditorReport:
    """Vollständiger Audit-Report nach Pipeline-Durchlauf."""

    total_phases_tracked: int = 0
    paralysis_events: list[ParalysisEvent] = field(default_factory=list)
    false_positives: int = 0
    confirmed_degradations: int = 0
    recovery_recommendations: list[dict[str, Any]] = field(default_factory=list)
    phases_to_recover: list[str] = field(default_factory=list)
    summary: str = ""


class GuardEffectivenessAuditor:
    """§v10.5 Meta-Guard: Audit aller Schutzmechanismen auf Paralysis.

    Thread-safe singleton. Wird vom PMGG nach jedem Retry/best_effort
    informiert und produziert am Pipeline-Ende einen vollständigen Report.

    Usage:
        auditor = get_effectiveness_auditor()
        # Während der Pipeline (nach jedem PMGG wrap_phase):
        auditor.track_phase_decision(phase_id, initial_strength, final_strength,
                                      retries, action, ...)
        # Am Pipeline-Ende:
        report = auditor.audit()
        auditor.recover_paralyzed_phases(pipeline_context)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._events: list[ParalysisEvent] = []
        self._phase_count: int = 0

    # ── Tracking ──

    def track_phase_decision(
        self,
        phase_id: str,
        initial_strength: float,
        final_strength: float,
        retries_exhausted: int,
        pmgg_action: str,
        goal_triggered: str = "",
        original_regression: float = 0.0,
        audio_before: np.ndarray | None = None,
        audio_after: np.ndarray | None = None,
        scores_before: dict[str, float] | None = None,
        scores_after: dict[str, float] | None = None,
    ) -> None:
        """§v10.5 Trackt eine PMGG-Entscheidung.

        Wird nach JEDEM wrap_phase-Aufruf vom PMGG aufgerufen.
        Nur best_effort-Aktionen werden als potenzielle Paralysis gespeichert.
        """
        with self._lock:
            self._phase_count += 1

            # Nur best_effort-Ereignisse sind Paralysis-verdächtig
            if not pmgg_action.startswith("best_effort"):
                return

            if final_strength >= PARALYSIS_STRENGTH_THRESHOLD:
                return  # Keine Paralysis — Strength ist noch akzeptabel

            event = ParalysisEvent(
                phase_id=phase_id,
                initial_strength=initial_strength,
                final_strength=final_strength,
                retries_exhausted=retries_exhausted,
                pmgg_action=pmgg_action,
                goal_triggered=goal_triggered,
                original_regression=original_regression,
                audio_before=np.asarray(audio_before).copy() if audio_before is not None else None,
                audio_after=np.asarray(audio_after).copy() if audio_after is not None else None,
                scores_before=dict(scores_before) if scores_before else None,
                scores_after=dict(scores_after) if scores_after else None,
            )
            self._events.append(event)

    # ── Audit ──

    def audit(self) -> AuditorReport:
        """§v10.5 Führt den vollständigen Audit durch.

        Prüft jedes Paralysis-Ereignis mit alternativen MediaDefectVerifier-Proxies
        und klassifiziert als false positive oder echte Degradation.

        Returns:
            AuditorReport mit Recovery-Empfehlungen
        """
        with self._lock:
            report = AuditorReport(
                total_phases_tracked=self._phase_count,
                paralysis_events=self._events,
            )

            for event in self._events:
                # Prüfe mit alternativen Proxies
                is_fp, alt_regression = self._check_false_positive(event)
                event.is_false_positive = is_fp
                event.alternative_regression = alt_regression

                if is_fp:
                    report.false_positives += 1
                    report.phases_to_recover.append(event.phase_id)
                    report.recovery_recommendations.append(
                        {
                            "phase": event.phase_id,
                            "paralyzed_at_strength": event.final_strength,
                            "original_strength": event.initial_strength,
                            "pmgg_action": event.pmgg_action,
                            "goal_triggered": event.goal_triggered,
                            "original_regression": event.original_regression,
                            "alternative_regression": alt_regression,
                            "action": "RE-RUN_AT_FULL_STRENGTH",
                            "reason": (
                                f"PMGG best_effort bei strength={event.final_strength:.0%} "
                                f"durch false positive in '{event.goal_triggered}' "
                                f"(Original Δ={event.original_regression:.3f}, "
                                f"Alternativ={alt_regression:.3f})"
                            ),
                        }
                    )
                else:
                    report.confirmed_degradations += 1

            # Summary
            if report.false_positives > 0:
                report.summary = (
                    f"GUARD PARALYSIS DETECTED: {report.false_positives} Phasen "
                    "durch false-positive Guards deaktiviert. "
                    f"Empfohlen: {len(report.phases_to_recover)} Phasen mit voller "
                    "Strength + alternativen Proxies re-runnen."
                )
                logger.warning("§v10.5 Auditor: %s", report.summary)
            elif self._events:
                report.summary = (
                    f"GUARD OK: {len(self._events)} best_effort-Phasen, "
                    f"aber {report.confirmed_degradations} echte Degradationen — "
                    "keine false positives. Guards arbeiten korrekt."
                )
                logger.info("§v10.5 Auditor: %s", report.summary)
            else:
                report.summary = f"GUARD CLEAN: Keine Paralysis-Ereignisse in {self._phase_count} Phasen."

            return report

    def _check_false_positive(self, event: ParalysisEvent) -> tuple[bool, float]:
        """§v10.5 Prüft ob die PMGG-Regression ein false positive war.

        Verwendet den MediaDefectVerifier für alternative Proxy-Metriken.
        Wenn die alternative Regression < FALSE_POSITIVE_RATIO * original,
        ist es ein false positive.
        """
        if event.audio_before is None or event.audio_after is None:
            # Kein Audio zum Nachprüfen → konservativ: kein false positive
            return False, event.original_regression

        sr = 48000  # Default; wird im echten Einsatz vom PMGG übergeben
        try:
            from backend.core.cassette_defect_verifier import (
                compute_phase_proxy_for_pmgg as _cv_proxy,
            )

            alt_scores = _cv_proxy(
                event.phase_id,
                event.audio_before,
                event.audio_after,
                sr,
            )

            if not alt_scores:
                return False, event.original_regression

            # Berechne alternative Regression für das triggernde Goal
            goal = event.goal_triggered
            if goal and goal in alt_scores:
                before_score = (event.scores_before or {}).get(goal, 0.5)
                after_score = alt_scores[goal]
                alt_reg = max(0.0, before_score - after_score)
            else:
                # Kein spezifischer Proxy → prüfe alle verfügbaren
                alt_regs = []
                for g in alt_scores:
                    b = (event.scores_before or {}).get(g, 0.5)
                    a = alt_scores[g]
                    if a < b:
                        alt_regs.append(b - a)
                alt_reg = float(np.mean(alt_regs)) if alt_regs else event.original_regression

            is_fp = alt_reg < event.original_regression * FALSE_POSITIVE_RATIO

            if is_fp:
                logger.info(
                    "§v10.5 Auditor: %s false positive bestätigt — "
                    "Originalsignal Δ=%.4f (goal=%s), alternativ Δ=%.4f (Verhaeltnis=%.1f%%)",
                    event.phase_id,
                    event.original_regression,
                    goal,
                    alt_reg,
                    (alt_reg / max(event.original_regression, 1e-6) * 100),
                )

            return is_fp, alt_reg

        except Exception as e:
            logger.debug("§v10.5 Auditor: Proxy-Pruefung fehlgeschlagen: %s", e)
            return False, event.original_regression

    # ── Recovery ──

    def get_recovery_phases(self) -> list[dict[str, Any]]:
        """§v10.5 Liste aller Phasen die mit voller Strength re-gerunned werden sollen."""
        with self._lock:
            result = []
            for event in self._events:
                if event.is_false_positive:
                    result.append(
                        {
                            "phase_id": event.phase_id,
                            "initial_strength": 1.0,  # Volle Strength
                            "skip_pmgg_goals": [event.goal_triggered] if event.goal_triggered else [],
                            "reason": f"Guard-induced paralysis at {event.final_strength:.0%} — "
                            f"false positive in '{event.goal_triggered}'",
                        }
                    )
            return result

    def has_paralysis(self) -> bool:
        """Schnell-Check: Gibt es Paralysis-Ereignisse?"""
        with self._lock:
            return any(e.is_false_positive for e in self._events)

    def reset(self) -> None:
        """Reset für neuen Pipeline-Durchlauf."""
        with self._lock:
            self._events.clear()
            self._phase_count = 0


# ── Singleton ──────────────────────────────────────────────────
_instance: GuardEffectivenessAuditor | None = None
_lock = threading.Lock()


def get_effectiveness_auditor() -> GuardEffectivenessAuditor:
    """Thread-safe Singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GuardEffectivenessAuditor()
    return _instance


def reset_effectiveness_auditor() -> None:
    """Reset für Tests."""
    global _instance
    with _lock:
        if _instance is not None:
            _instance.reset()
        _instance = None
