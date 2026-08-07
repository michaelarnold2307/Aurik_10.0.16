"""
Pipeline Provenance Tracker — Aurik §v10.4

Das fehlende Fundament: Verfolgt den kumulativen Fortschritt jedes Musical Goals
über ALLE Phasen — nicht nur Phase-zu-Phase wie der PMGG.

ARCHITEKTUR:
  Golden-Reference-Tracker  → best_per_goal + best_audio + contributor
  Per-Phase-Contribution     → welche Phase hat welches Goal wie stark verbessert
  Undo-Detection             → Phase N verschlechtert Goal das Phase M verbessert hatte
  Net-Delta-Tracking         → kumulativer Fortschritt pro Goal über ALLE Phasen

PROBLEM DAS GELÖST WIRD:
  phase_03 verbessert natuerlichkeit um +0.04
  phase_16 verschlechtert natuerlichkeit um -0.03 (durch HF-EQ)
  phase_29 verschlechtert natuerlichkeit um -0.02 (durch Hiss-Reduktion)
  PMGG sagt bei jeder Phase: "Regression < threshold -> pass"
  Netto: -0.01 — die +0.04 von phase_03 sind für immer verloren
  → Der Provenance Tracker erkennt: phase_16 hat phase_03's Arbeit an natuerlichkeit
    rückgängig gemacht. UNDO empfohlen.

INTEGRATION:
  - PMGG: nach jeder Phase Provenance-Check zusätzlich zu Regression-Check
  - Steering: best_audio-Recovery bei UNDO-Ereignis
  - Phase-Konflikt-Registry: automatische Konflikt-Erkennung

Author: Aurik v10.4 Development
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
UNDO_THRESHOLD: float = 0.015  # Delta unter diesem Wert = signifikantes Undo
NET_DELTA_WARNING: float = -0.010  # Netto-Verschlechterung über alle Phasen
NET_DELTA_CRITICAL: float = -0.025  # Kritische Netto-Verschlechterung


@dataclass
class GoalContribution:
    """Tracking eines einzelnen Goal-Beitrags einer Phase."""

    goal: str
    phase_id: str
    before_score: float
    after_score: float
    delta: float  # after - before (positiv = Verbesserung)
    is_improvement: bool
    timestamp: float = 0.0


@dataclass
class UndoEvent:
    """Phase N hat Goal G verschlechtert, das Phase M zuvor verbessert hatte."""

    goal: str
    undoing_phase: str  # Phase die verschlechtert
    original_contributor: str  # Phase die verbessert hatte
    original_delta: float  # Ursprüngliche Verbesserung
    undo_delta: float  # Verschlechterung (negativ)
    best_score: float  # Bester je erreichter Score
    current_score: float  # Aktueller Score nach Undo
    severity: str = "warning"  # "warning", "critical"

    @property
    def recovery_gap(self) -> float:
        """Wie viel muss wiederhergestellt werden."""
        return self.best_score - self.current_score


@dataclass
class GoalProvenance:
    """Vollständige Provenance eines einzelnen Musical Goals."""

    goal: str
    baseline_score: float  # Score vor Pipeline-Start
    best_ever_score: float  # Höchster je erreichter Score
    best_ever_phase: str = ""  # Phase die den Bestwert erreicht hat
    current_score: float = 0.0  # Aktueller Score
    net_delta: float = 0.0  # Kumulativ: current - baseline
    contributions: list[GoalContribution] = field(default_factory=list)
    undo_events: list[UndoEvent] = field(default_factory=list)
    undo_count: int = 0


@dataclass
class PipelineProvenance:
    """Gesamte Pipeline-Provenance über alle Goals und Phasen."""

    goals: dict[str, GoalProvenance] = field(default_factory=dict)
    total_phases_run: int = 0
    total_undos_detected: int = 0
    pipeline_start_audio_hash: str = ""
    best_audio: np.ndarray | None = None
    best_audio_phase: str = ""
    best_audio_score: float = 0.0  # Durchschnitt aller Goals zum Bestzeitpunkt


class PipelineProvenanceTracker:
    """§v10.4 Kernel des Pipeline Provenance Trackers.

    Thread-safe singleton. Wird vom PMGG nach JEDER Phase aufgerufen.

    Usage:
        tracker = get_provenance_tracker()
        tracker.start_pipeline(original_audio, baseline_scores)
        # ... nach jeder Phase:
        result = tracker.track_phase(phase_id, scores_before, scores_after,
                                     audio_after, effective_goals)
        if result.undo_detected:
            pass  # Trigger Recovery
        # ... am Ende:
        report = tracker.finalize()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._provenance = PipelineProvenance()
        self._active = False
        self._pipeline_audio: np.ndarray | None = None

    # ── Pipeline Lifecycle ──

    def start_pipeline(
        self,
        original_audio: np.ndarray,
        baseline_scores: dict[str, float],
        goals: list[str] | None = None,
    ) -> None:
        """Initialisiert die Provenance vor Pipeline-Start."""
        with self._lock:
            self._provenance = PipelineProvenance()
            self._active = True
            self._pipeline_audio = np.asarray(original_audio).copy()

            check_goals = goals if goals else list(baseline_scores.keys())
            for goal in check_goals:
                score = baseline_scores.get(goal, 0.5)
                self._provenance.goals[goal] = GoalProvenance(
                    goal=goal,
                    baseline_score=score,
                    best_ever_score=score,
                    current_score=score,
                )

    def track_phase(
        self,
        phase_id: str,
        scores_before: dict[str, float],
        scores_after: dict[str, float],
        audio_after: np.ndarray,
        effective_goals: list[str] | None = None,
    ) -> dict[str, Any]:
        """§v10.4 Trackt eine Phase und erkennt Undo-Ereignisse.

        Wird vom PMGG nach JEDER Phase aufgerufen.

        Returns dict mit:
          - undo_detected: bool
          - undo_events: list[dict]
          - net_delta_warning: bool
          - recovery_recommended: bool
          - net_deltas: dict[goal -> net_delta]
        """
        if not self._active:
            return {
                "undo_detected": False,
                "undo_events": [],
                "net_delta_warning": False,
                "recovery_recommended": False,
            }

        with self._lock:
            goals_to_check = (
                effective_goals if effective_goals else list(set(scores_before.keys()) & set(scores_after.keys()))
            )
            undo_events: list[UndoEvent] = []
            total_net_delta = 0.0

            for goal in goals_to_check:
                if goal not in self._provenance.goals:
                    continue

                prov = self._provenance.goals[goal]
                before = scores_before.get(goal, 0.5)
                after = scores_after.get(goal, 0.5)
                delta = after - before

                # Contribution tracken
                contrib = GoalContribution(
                    goal=goal,
                    phase_id=phase_id,
                    before_score=before,
                    after_score=after,
                    delta=delta,
                    is_improvement=delta > 0.005,
                )
                prov.contributions.append(contrib)

                # Best-Ever updaten
                if after > prov.best_ever_score:
                    prov.best_ever_score = after
                    prov.best_ever_phase = phase_id

                # Aktuellen Score updaten
                prov.current_score = after

                # Netto-Delta (kumulativ)
                prov.net_delta = after - prov.baseline_score
                total_net_delta += prov.net_delta

                # ── UNDO-DETECTION ──
                # Phase hat Goal verschlechtert, das zuvor von einer ANDEREN
                # Phase verbessert wurde, UND der aktuelle Score ist signifikant
                # unter dem Bestwert.
                if (
                    delta < -UNDO_THRESHOLD
                    and prov.best_ever_phase
                    and prov.best_ever_phase != phase_id
                    and prov.best_ever_score - after > UNDO_THRESHOLD
                ):
                    severity = "critical" if prov.best_ever_score - after > NET_DELTA_CRITICAL else "warning"
                    undo = UndoEvent(
                        goal=goal,
                        undoing_phase=phase_id,
                        original_contributor=prov.best_ever_phase,
                        original_delta=prov.best_ever_score - prov.baseline_score,
                        undo_delta=delta,
                        best_score=prov.best_ever_score,
                        current_score=after,
                        severity=severity,
                    )
                    undo_events.append(undo)
                    prov.undo_events.append(undo)
                    prov.undo_count += 1

            self._provenance.total_phases_run += 1
            if undo_events:
                self._provenance.total_undos_detected += 1

            # Best-Audio-Tracking: speichere Audio wenn der DURCHSCHNITT
            # aller Goal-Scores besser ist als zuvor
            current_avg = float(np.mean([v.current_score for v in self._provenance.goals.values()]))
            if current_avg > self._provenance.best_audio_score:
                self._provenance.best_audio_score = current_avg
                self._provenance.best_audio = np.asarray(audio_after).copy()
                self._provenance.best_audio_phase = phase_id

            # Net-Delta-Warnung: kumulativ
            avg_net = total_net_delta / max(len(goals_to_check), 1)
            net_delta_warning = avg_net < NET_DELTA_WARNING
            recovery_recommended = avg_net < NET_DELTA_CRITICAL

            if recovery_recommended:
                logger.warning(
                    "§v10.4 Provenance: CRITICAL net delta %.4f nach Verarbeitungsschritt %s (%d undo events). Wiederherstellung empfohlen.",
                    avg_net,
                    phase_id,
                    len(undo_events),
                )
            elif undo_events:
                logger.info(
                    "§v10.4 Provenance: %d undo(s) detektiert in Verarbeitungsschritt %s: %s",
                    len(undo_events),
                    phase_id,
                    [(u.goal, u.original_contributor) for u in undo_events],
                )

            return {
                "undo_detected": len(undo_events) > 0,
                "undo_events": [
                    {
                        "goal": u.goal,
                        "undoing_phase": u.undoing_phase,
                        "original_contributor": u.original_contributor,
                        "best_score": u.best_score,
                        "current_score": u.current_score,
                        "recovery_gap": u.recovery_gap,
                        "severity": u.severity,
                    }
                    for u in undo_events
                ],
                "net_delta_warning": net_delta_warning,
                "recovery_recommended": recovery_recommended,
                "net_deltas": {g: p.net_delta for g, p in self._provenance.goals.items()},
                "avg_net_delta": avg_net,
                "best_audio_phase": self._provenance.best_audio_phase,
            }

    def get_best_audio(self) -> np.ndarray | None:
        """Gibt das Audio zum besten Goal-Durchschnitt zurück."""
        with self._lock:
            if self._provenance.best_audio is not None:
                return self._provenance.best_audio.copy()
            return self._pipeline_audio.copy() if self._pipeline_audio is not None else None

    def get_goal_report(self, goal: str) -> dict[str, Any]:
        """Detaillierter Report für ein einzelnes Goal."""
        with self._lock:
            prov = self._provenance.goals.get(goal)
            if prov is None:
                return {}
            return {
                "goal": goal,
                "baseline": prov.baseline_score,
                "best": prov.best_ever_score,
                "best_phase": prov.best_ever_phase,
                "current": prov.current_score,
                "net_delta": prov.net_delta,
                "contributions": [{"phase": c.phase_id, "delta": c.delta} for c in prov.contributions],
                "undo_events": [
                    {"undoing": u.undoing_phase, "contributor": u.original_contributor, "severity": u.severity}
                    for u in prov.undo_events
                ],
                "undo_count": prov.undo_count,
            }

    def get_conflict_phases(self) -> list[dict[str, Any]]:
        """Extrahiert alle UNDO-Konflikte als Phase-Paare.

        Diese können in die CONFLICT_REGISTRY der phase_ontology
        übernommen werden (automatische Konflikt-Erkennung).
        """
        with self._lock:
            conflicts: dict[tuple[str, str], list[str]] = {}
            for goal, prov in self._provenance.goals.items():
                for undo in prov.undo_events:
                    pair = (undo.original_contributor, undo.undoing_phase)
                    if pair not in conflicts:
                        conflicts[pair] = []
                    conflicts[pair].append(goal)

            return [
                {
                    "contributor": orig,
                    "undoing": undoer,
                    "goals_affected": goals_list,
                    "count": len(goals_list),
                }
                for (orig, undoer), goals_list in sorted(conflicts.items(), key=lambda x: -len(x[1]))
            ]

    def finalize(self) -> dict[str, Any]:
        """Schließt die Pipeline ab und produziert den finalen Report."""
        with self._lock:
            self._active = False
            goals = self._provenance.goals

            # Netto-Erfolg pro Goal
            improved = sum(1 for g in goals.values() if g.net_delta > 0.005)
            degraded = sum(1 for g in goals.values() if g.net_delta < -0.005)
            neutral = len(goals) - improved - degraded

            # Phasen die am meisten verbessert haben
            phase_contributions: dict[str, float] = {}
            for g in goals.values():
                for c in g.contributions:
                    if c.is_improvement:
                        phase_contributions[c.phase_id] = phase_contributions.get(c.phase_id, 0.0) + c.delta

            top_contributors = sorted(phase_contributions.items(), key=lambda x: -x[1])[:5]

            # Phasen die am meisten Undos verursacht haben
            phase_undo_counts: dict[str, int] = {}
            for g in goals.values():
                for u in g.undo_events:
                    phase_undo_counts[u.undoing_phase] = phase_undo_counts.get(u.undoing_phase, 0) + 1

            top_undoers = sorted(phase_undo_counts.items(), key=lambda x: -x[1])[:5]

            # Durchschnittlicher Net-Delta
            avg_net = float(np.mean([g.net_delta for g in goals.values()]))

            report = {
                "total_phases": self._provenance.total_phases_run,
                "total_undos": self._provenance.total_undos_detected,
                "total_goals": len(goals),
                "improved_goals": improved,
                "degraded_goals": degraded,
                "neutral_goals": neutral,
                "avg_net_delta": avg_net,
                "best_audio_phase": self._provenance.best_audio_phase,
                "best_audio_score": self._provenance.best_audio_score,
                "top_contributors": [{"phase": ph, "total_delta": d} for ph, d in top_contributors],
                "top_undoers": [{"phase": ph, "undo_count": c} for ph, c in top_undoers],
                "detected_conflicts": self.get_conflict_phases(),
                "per_goal": {
                    g: {
                        "baseline": p.baseline_score,
                        "best": p.best_ever_score,
                        "current": p.current_score,
                        "net_delta": p.net_delta,
                        "best_phase": p.best_ever_phase,
                        "undo_count": p.undo_count,
                    }
                    for g, p in goals.items()
                },
            }

            logger.info(
                "§v10.4 Pipeline Provenance: %d Phasen, %d Goals, Avg-Net=%.4f, %d improved, %d degraded, %d undos",
                report["total_phases"],
                report["total_goals"],
                avg_net,
                improved,
                degraded,
                report["total_undos"],
            )

            return report

    def reset(self) -> None:
        """Reset für Test/Neustart."""
        with self._lock:
            self._provenance = PipelineProvenance()
            self._active = False
            self._pipeline_audio = None


# ── Singleton ──────────────────────────────────────────────────
_instance: PipelineProvenanceTracker | None = None
_lock = threading.Lock()


def get_provenance_tracker() -> PipelineProvenanceTracker:
    """Thread-safe Singleton accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PipelineProvenanceTracker()
    return _instance


def reset_provenance_tracker() -> None:
    """Reset für Tests."""
    global _instance
    with _lock:
        if _instance is not None:
            _instance.reset()
        _instance = None
