"""§AF-MAX: DynamicsGuardIntegration — Brücke zu allen Denker-Modulen.
Verbindet den RepairDynamicsGuard mit:
- PMGG-Scoring       → Dynamics-Metriken fließen in Bewertung
- GoalBudget         → Dynamics-Budget-Tracking (waerme/brillanz/punch)
- GuardWisdom        → Lernfähigkeit über Reparaturen hinweg
- CrossGuardCoordinator → dynamics_arc Kategorie
- EmotionalArcPreserver → Arousal/Valence-Erhalt
- restoration_context → dynamics_report für alle Denker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DynamicsIntegrationReport:
    """Gesammeltes Feedback aller integrierten Module."""

    pmgg_dynamics_score: float = 0.0
    budget_remaining: dict[str, float] = field(default_factory=dict)
    wisdom_strength_mod: float = 1.0
    cross_guard_verdict: str = "ok"
    emotional_arc_preserved: bool = True
    warnings: list[str] = field(default_factory=list)


class DynamicsGuardIntegration:
    """Brücke zwischen RepairDynamicsGuard und allen Denker-Modulen.

    Wird vom UV3-Phase-Loop nach jeder Defekt-Reparatur aufgerufen.
    Verteilt die Dynamics-Metriken an alle registrierten Denker.
    """

    def __init__(self) -> None:
        self._reports: list[DynamicsIntegrationReport] = []

    def integrate_phase_result(
        self,
        *,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        phase_id: str,
        restoration_context: dict[str, Any],
        dynamics_guard,  # RepairDynamicsGuard instance
        pmgg_instance=None,
        goal_budget=None,
        guard_wisdom=None,
        cross_guard_coordinator=None,
        emotional_arc_preserver=None,
    ) -> DynamicsIntegrationReport:
        """Führt eine vollständige Integration nach einer Reparatur-Phase durch.

        Args:
            audio_before/after: Audio vor/nach der Reparatur
            sr: Sample-Rate
            phase_id: Phasen-ID für Logging
            restoration_context: UV3 restoration_context dict
            dynamics_guard: RepairDynamicsGuard Instanz
            pmgg_instance: PMGG-Scorer (optional)
            goal_budget: GoalBudget Instanz (optional)
            guard_wisdom: GuardWisdom Instanz (optional)
            cross_guard_coordinator: CrossGuardCoordinator (optional)
            emotional_arc_preserver: EmotionalArcPreserver (optional)
        """
        report = DynamicsIntegrationReport()

        # 1. RepairDynamicsGuard Verifikation
        ct = dynamics_guard.verify_continuity(
            audio_after, sr, [0, len(audio_after.shape) if audio_after.ndim == 1 else audio_after.shape[-1]]
        )
        sb = dynamics_guard.verify_stereo_balance(audio_before, audio_after)
        pc = dynamics_guard.verify_phase_coherence(audio_before, audio_after)
        gd = dynamics_guard.verify_global_dynamics(audio_before, audio_after, sr)

        # 2. GuardWisdom: Lerne aus Ergebnissen
        if guard_wisdom is not None:
            wisdom_metrics = {
                "max_env_dev_db": ct.max_envelope_deviation_db,
                "stereo_drift_db": sb.max_stereo_drift_db,
                "phase_corr": pc.min_phase_correlation,
                "crest_change_pct": abs(gd.crest_factor_after - gd.crest_factor_before)
                / max(gd.crest_factor_before, 1e-10)
                * 100,
            }
            verdict = "ok"
            if not ct.continuity_ok or not sb.stereo_balance_ok or not gd.global_dynamics_ok:
                verdict = "violation"
                report.warnings.append(f"GuardWisdom registered violation in {phase_id}")
            try:
                guard_wisdom.record(phase_id, "dynamics_guard", wisdom_metrics, verdict)
            except Exception:
                logger.debug("integrate_Verarbeitungsschritt_Ergebnis: silent except suppressed", exc_info=True)
            report.wisdom_strength_mod = getattr(guard_wisdom, "_strength_mod", 1.0)

        # 3. GoalBudget: Dynamics-Budget abbuchen
        if goal_budget is not None:
            try:
                budget_map = {
                    "brillanz": "brillanz",
                    "punch": "punch",
                }
                for goal_key, budget_key in budget_map.items():
                    if hasattr(goal_budget, "fraction_left") and goal_budget.fraction_left(budget_key) > 0.001:
                        # Ein Reparatur-Delta basierend auf Envelope-Verbesserung
                        delta = min(0.05, ct.max_envelope_deviation_db / 50.0)
                        if delta > 0.001:
                            goal_budget.record_delta(budget_key, delta)
                report.budget_remaining = {g: goal_budget.fraction_left(g) for g in ["waerme", "brillanz", "punch"]}
            except Exception:
                logger.debug("integrate_Verarbeitungsschritt_Ergebnis: silent except suppressed", exc_info=True)

        # 4. CrossGuardCoordinator: dynamics_arc Kategorie
        if cross_guard_coordinator is not None:
            dynamics_metrics = {
                "lufs_drift": abs(gd.lufs_integrated_before - gd.lufs_integrated_after),
                "crest_factor_before": gd.crest_factor_before,
                "crest_factor_after": gd.crest_factor_after,
                "envelope_deviation_db": ct.max_envelope_deviation_db,
            }
            try:
                if hasattr(cross_guard_coordinator, "record"):
                    cross_guard_coordinator.record("dynamics_arc", phase_id, dynamics_metrics)
                if hasattr(cross_guard_coordinator, "evaluate"):
                    evaluation = cross_guard_coordinator.evaluate()
                    report.cross_guard_verdict = evaluation.get("verdict", "ok")
            except Exception:
                logger.debug("integrate_Verarbeitungsschritt_Ergebnis: silent except suppressed", exc_info=True)

        # 5. EmotionalArcPreserver: Arousal/Valence prüfen
        if emotional_arc_preserver is not None:
            try:
                if hasattr(emotional_arc_preserver, "_measure"):
                    before_arc = emotional_arc_preserver._measure(audio_before, sr)
                    after_arc = emotional_arc_preserver._measure(audio_after, sr)
                    if before_arc is not None and after_arc is not None:
                        arousal_corr = float(np.corrcoef(before_arc[0], after_arc[0])[0, 1])
                        valence_corr = float(np.corrcoef(before_arc[1], after_arc[1])[0, 1])
                        report.emotional_arc_preserved = arousal_corr > 0.9 and valence_corr > 0.9
                        if not report.emotional_arc_preserved:
                            report.warnings.append(
                                f"EmotionalArc degraded: arousal_corr={arousal_corr:.3f}, valence_corr={valence_corr:.3f}"
                            )
            except Exception:
                logger.debug("integrate_Verarbeitungsschritt_Ergebnis: silent except suppressed", exc_info=True)

        # 6. PMGG: Dynamics-Score aktualisieren
        if pmgg_instance is not None:
            try:
                pmgg_score = self._compute_dynamics_pmgg_score(ct, sb, pc, gd)
                report.pmgg_dynamics_score = pmgg_score
                if hasattr(pmgg_instance, "_set_dynamics_score"):
                    pmgg_instance._set_dynamics_score(pmgg_score)
            except Exception:
                logger.debug("integrate_Verarbeitungsschritt_Ergebnis: silent except suppressed", exc_info=True)

        # 7. restoration_context injizieren
        if isinstance(restoration_context, dict):
            restoration_context["_dynamics_guard_report"] = {
                "continuity_ok": ct.continuity_ok,
                "stereo_balance_ok": sb.stereo_balance_ok,
                "phase_coherence_ok": pc.phase_coherence_ok,
                "global_dynamics_ok": gd.global_dynamics_ok,
                "max_envelope_deviation_db": ct.max_envelope_deviation_db,
                "max_stereo_drift_db": sb.max_stereo_drift_db,
                "min_phase_correlation": pc.min_phase_correlation,
                "lufs_before": gd.lufs_integrated_before,
                "lufs_after": gd.lufs_integrated_after,
                "integration_verdict": report.cross_guard_verdict,
            }

        self._reports.append(report)
        return report

    def integrate_post_pipeline(
        self,
        *,
        audio_original: np.ndarray,
        audio_restored: np.ndarray,
        sr: int,
        restoration_context: dict[str, Any],
        dynamics_guard,
        goal_budget=None,
        guard_wisdom=None,
        cross_guard_coordinator=None,
    ) -> DynamicsIntegrationReport:
        """Gesamt-Integration nach der vollständigen Pipeline.

        Sammelt finale Metriken und schreibt den Abschlussbericht.
        """
        report = DynamicsIntegrationReport()

        # Vollständige Verifikation
        gd = dynamics_guard.verify_global_dynamics(audio_original, audio_restored, sr)

        # Budget-Finalisierung
        if goal_budget is not None:
            try:
                report.budget_remaining = {g: goal_budget.fraction_left(g) for g in ["waerme", "brillanz", "punch"]}
            except Exception:
                logger.debug("integrate_post_pipeline: silent except suppressed", exc_info=True)

        # Wisdom-Feedback
        if guard_wisdom is not None:
            try:
                snapshot = guard_wisdom.snapshot()
                report.wisdom_strength_mod = snapshot.get("strength_mod", 1.0)
                report.warnings.extend([f"GuardWisdom: {snapshot.get('rollbacks', 0)} rollbacks total"])
            except Exception:
                logger.debug("integrate_post_pipeline: silent except suppressed", exc_info=True)

        # CrossGuard finale Auswertung
        if cross_guard_coordinator is not None:
            try:
                if hasattr(cross_guard_coordinator, "evaluate"):
                    evaluation = cross_guard_coordinator.evaluate()
                    report.cross_guard_verdict = evaluation.get("verdict", "ok")
            except Exception:
                logger.debug("integrate_post_pipeline: silent except suppressed", exc_info=True)

        # restoration_context abschließend
        if isinstance(restoration_context, dict):
            restoration_context["_dynamics_guard_final"] = {
                "global_dynamics_ok": gd.global_dynamics_ok,
                "crest_factor_original": gd.crest_factor_before,
                "crest_factor_restored": gd.crest_factor_after,
                "lufs_original": gd.lufs_integrated_before,
                "lufs_restored": gd.lufs_integrated_after,
                "wisdom_strength_mod": report.wisdom_strength_mod,
                "budget_remaining": report.budget_remaining,
            }

        self._reports.append(report)
        return report

    def _compute_dynamics_pmgg_score(self, ct, sb, pc, gd) -> float:
        """Berechnet einen Dynamics-Score für PMGG (0.0–1.0)."""
        score = 1.0

        # Envelope-Kontinuität (max 0.3 Abzug)
        if not ct.continuity_ok:
            score -= min(0.3, ct.max_envelope_deviation_db / 20.0)

        # Stereo-Balance (max 0.2 Abzug)
        if not sb.stereo_balance_ok:
            score -= min(0.2, sb.max_stereo_drift_db / 5.0)

        # Phase (max 0.2 Abzug)
        if not pc.phase_coherence_ok:
            score -= min(0.2, (0.90 - pc.min_phase_correlation) * 2.0)

        # Global (max 0.3 Abzug)
        if not gd.global_dynamics_ok:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def get_all_reports(self) -> list[DynamicsIntegrationReport]:
        return list(self._reports)

    def clear_reports(self) -> None:
        self._reports.clear()
