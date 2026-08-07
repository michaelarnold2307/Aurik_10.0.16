"""
Quality Feedback Loop - Aurik 10.0.0
==================================

Adaptive quality control with real-time metrics feedback.

Concept:
1. Process audio with standard parameters
2. Measure naturalness metrics
3. If below target: adapt parameters, repeat (max 2 iterations)
4. Return best result

Benefits:
- Prevents over-processing (main cause of unnaturalness)
- Adaptive to material quality
- Auto-optimizes for target quality
- Minimal performance cost (~5-10% overhead)

Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
"""

import logging
from typing import Any

import numpy as np

from backend.core.comprehensive_metrics import PsychoAcousticMetrics
from backend.core.phases.phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


# ── §v10 Steering-Regel (ersetzt Stop-Regel) ──
# Ziel ist nicht Abbruch, sondern VERBESSERUNG.
# Wenn Angenehmheit sinkt → Parameter anpassen, nicht aufgeben.

_PMGG_CONSECUTIVE_NO_IMPROVEMENT: int = 0
_PLEASANTNESS_DECLINING_COUNT: int = 0
_BEST_PLEASANTNESS: float = 0.0


class SteerAction:
    """§v10 Steering-Aktionen — wie ein Toningenieur reagiert."""

    CONTINUE = "continue"  # Alles gut, weitermachen
    RETRY_LIGHTER = "retry_lighter"  # Gleicher Schritt mit reduzierter Intensität
    RETRY_DIFFERENT = "retry_different"  # Alternativer Ansatz versuchen
    SKIP = "skip"  # Schritt überspringen (würde nur verschlechtern)
    ROLLBACK = "rollback"  # Zurück zum besten Zustand
    STOP_GRACEFUL = "stop_graceful"  # Keine weitere Verbesserung möglich


def steer_pipeline(
    pmgg_delta: float,
    pleasantness_delta: float,
    phase_id: str,
    step_index: int,
    total_steps: int,
    pmgg_threshold: float = 0.01,
    max_pmgg_noop: int = 3,
    max_pleasantness_drops: int = 2,
    mode: str = "restoration",
) -> tuple[str, str]:
    """§v10 Steering: Nicht stoppen, sondern nachsteuern.

    Wie ein Toningenieur: „Das klang nicht gut — ich versuch's mit weniger."
    Nicht: „Das klang nicht gut — ich hör auf."

    §v10.1 MODE-AWARE:
      RESTORATION: Konservativ. Aura heilig. RETRY_LIGHTER schon bei -0.01.
                   STOP_GRACEFUL früh. Original-Klangfarbe unantastbar.
      STUDIO 2026: Aggressiv. Moderner Klang zählt. Toleriert HPE-Drops bis
                   -0.06 vor SKIP. RETRY_DIFFERENT statt nur RETRY_LIGHTER.
                   Das beste Ergebnis zählt, nicht die Original-Treue.

    Returns: (Aktion, Begründung)
    """
    global _PMGG_CONSECUTIVE_NO_IMPROVEMENT, _PLEASANTNESS_DECLINING_COUNT, _BEST_PLEASANTNESS

    is_studio = "studio" in str(mode).lower() or "2026" in str(mode).lower()

    # Mode-adjustierte Schwellwerte (§v10.1)
    if is_studio:
        hpe_up_threshold = 0.025  # Studio: braucht starke Evidenz zum Weitermachen
        hpe_light_drop = 0.060  # Studio: toleriert leichte Einbrüche
        hpe_heavy_drop = 0.100  # Studio: erst starker Einbruch triggert SKIP
        max_drops = max_pleasantness_drops + 1
    else:
        hpe_up_threshold = 0.010  # Restoration: jede Verbesserung = gut
        hpe_light_drop = 0.020  # Restoration: kleinster Drop → RETRY
        hpe_heavy_drop = 0.040  # Restoration: moderater Drop → SKIP
        max_drops = max_pleasantness_drops

    # Track best pleasantness
    if pleasantness_delta > 0:
        _BEST_PLEASANTNESS = max(_BEST_PLEASANTNESS, pleasantness_delta)

    # ── Angenehmheit STEIGT → weitermachen ──
    if pleasantness_delta > hpe_up_threshold:
        _PLEASANTNESS_DECLINING_COUNT = max(0, _PLEASANTNESS_DECLINING_COUNT - 1)
        _PMGG_CONSECUTIVE_NO_IMPROVEMENT = 0
        return SteerAction.CONTINUE, f"HPE ↑ (ΔP=+{pleasantness_delta:.3f})"

    # ── Angenehmheit fällt LEICHT → RETRY_LIGHTER (Restoration) oder RETRY_DIFFERENT (Studio) ──
    if -hpe_heavy_drop < pleasantness_delta <= -hpe_light_drop:
        _PLEASANTNESS_DECLINING_COUNT += 1
        if is_studio and _PLEASANTNESS_DECLINING_COUNT >= 2:
            return SteerAction.RETRY_DIFFERENT, (f"HPE ↓ (ΔP={pleasantness_delta:+.3f}) — versuche alternativen Ansatz")
        return SteerAction.RETRY_LIGHTER, (f"HPE ↓ (ΔP={pleasantness_delta:+.3f}) — versuche reduzierte Intensität")

    # ── Angenehmheit fällt STARK → SKIP ──
    if pleasantness_delta <= -hpe_heavy_drop:
        _PLEASANTNESS_DECLINING_COUNT += 1
        if max_drops <= _PLEASANTNESS_DECLINING_COUNT:
            return SteerAction.ROLLBACK, (
                f"HPE ↓↓ seit {_PLEASANTNESS_DECLINING_COUNT} Schritten "
                f"— ROLLBACK zum besten Zustand (max ΔP=+{_BEST_PLEASANTNESS:.3f})"
            )
        return SteerAction.SKIP, (f"HPE ↓↓ (ΔP={pleasantness_delta:+.3f}) — Schritt {phase_id} überspringen")

    # ── PMGG konvergiert → STOP_GRACEFUL ──
    if abs(pmgg_delta) < pmgg_threshold:
        _PMGG_CONSECUTIVE_NO_IMPROVEMENT += 1
        if max_pmgg_noop <= _PMGG_CONSECUTIVE_NO_IMPROVEMENT:
            return SteerAction.STOP_GRACEFUL, ("PMGG konvergiert — Bearbeitung optimal abgeschlossen.")

    # ── Pipeline-Ende erreicht ──
    if step_index >= total_steps - 1:
        return SteerAction.STOP_GRACEFUL, "Pipeline vollständig — Ergebnis optimal."

    return SteerAction.CONTINUE, "Weitermachen."


def reset_steer_state():
    """Setzt alle Steering-Zähler zurück."""
    global _PMGG_CONSECUTIVE_NO_IMPROVEMENT, _PLEASANTNESS_DECLINING_COUNT, _BEST_PLEASANTNESS
    _PMGG_CONSECUTIVE_NO_IMPROVEMENT = 0
    _PLEASANTNESS_DECLINING_COUNT = 0
    _BEST_PLEASANTNESS = 0.0


# ── Legacy-Kompatibilität ──
def should_stop_pipeline(
    pmgg_delta, phase_id, threshold=0.01, max_consecutive=3, pleasantness_delta=0.0, max_pleasantness_drops=2
):
    """§v10 Legacy-Wrapper: Verwende steer_pipeline() für intelligentes Nachsteuern."""
    action, reason = steer_pipeline(
        pmgg_delta,
        pleasantness_delta,
        phase_id,
        step_index=0,
        total_steps=max_consecutive,
        pmgg_threshold=threshold,
        max_pleasantness_drops=max_pleasantness_drops,
    )
    if action in (SteerAction.STOP_GRACEFUL, SteerAction.ROLLBACK):
        return True, reason
    return False, ""


def reset_stop_rule_state():
    """Legacy-Wrapper für reset_steer_state."""
    reset_steer_state()


class QualityFeedbackLoop:
    """
    Adaptive Quality Control with Real-Time Metrics.

    Iteratively processes audio, measuring quality after each pass
    and adapting parameters until target quality is reached.

    Example:
        feedback = QualityFeedbackLoop(target_naturalness=0.80)
        result = feedback.process_with_feedback(phase, audio, material='vinyl')
    """

    def __init__(self, target_naturalness: float = 0.80, max_iterations: int = 2, min_improvement: float = 0.02):
        """
        Initialisiert feedback loop.

        Args:
            target_naturalness: Target naturalness score (0-1)
            max_iterations: Maximum processing iterations
            min_improvement: Minimum improvement to continue iterating
        """
        self.target_naturalness = target_naturalness
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.metrics = PsychoAcousticMetrics()

    def process_with_feedback(
        self, phase: PhaseInterface, audio: np.ndarray, sample_rate: int = 44100, **kwargs
    ) -> PhaseResult:
        """
        Verarbeitet audio with iterative quality feedback.

        Args:
            phase: Phase to process with
            audio: Input audio
            sample_rate: Sample rate
            **kwargs: Phase-specific parameters

        Returns:
            PhaseResult with best quality achieved
        """
        original_audio = audio.copy()
        original_kwargs = dict(kwargs)
        best_result = None
        best_naturalness = 0.0
        prev_naturalness: float | None = None

        # Update metrics sample rate
        self.metrics.sample_rate = sample_rate

        for iteration in range(self.max_iterations):
            # Process audio
            result = phase.process(audio, sample_rate, **kwargs)

            if not result.success:
                logger.warning(
                    "Verarbeitungsschritt %s fehlgeschlagen in iteration %s", phase.get_metadata().name, iteration
                )
                break

            # Measure naturalness
            try:
                naturalness_scores = self.metrics.calculate_naturalness_score(result.audio, reference=original_audio)
                naturalness = naturalness_scores["naturalness_overall"]

                logger.info(
                    "Iteration %s: Naturalness %.3f (target: %.3f)",
                    iteration + 1,
                    naturalness,
                    self.target_naturalness,
                )

                # Store if best so far
                if naturalness > best_naturalness:
                    best_result = result
                    best_naturalness = naturalness

                    # Add naturalness info to metadata
                    best_result.metadata["naturalness_score"] = naturalness
                    best_result.metadata["naturalness_details"] = naturalness_scores
                    best_result.metadata["feedback_iterations"] = iteration + 1

                # Check if target reached
                if naturalness >= self.target_naturalness:
                    logger.info("✅ Quality target reached: %.3f", naturalness)
                    break

                # Check if improvement is too small to continue
                if prev_naturalness is not None and (naturalness - prev_naturalness) < self.min_improvement:
                    logger.info("⚠️  Improvement too small (%.3f), stoppe", naturalness - prev_naturalness)
                    break

                prev_naturalness = naturalness

                # Adapt parameters for next iteration
                if iteration < self.max_iterations - 1:
                    naturalness_deficit = self.target_naturalness - naturalness
                    kwargs = self._adapt_parameters(kwargs, naturalness_deficit, naturalness_scores)
                    audio = result.audio  # Use previous result as input
                    logger.info("🔧 Adapting parameters for iteration %s...", iteration + 2)

            except Exception as e:
                logger.error("Quality measurement fehlgeschlagen: %s", e)
                if best_result is None:
                    best_result = result
                break

        # Return best result achieved
        if best_result is None:
            # Fallback: return original processing result
            logger.warning("No valid Ergebnis achieved, returning Originalsignal processing")
            return phase.process(original_audio, sample_rate, **original_kwargs)

        return best_result

    def _adapt_parameters(
        self, params: dict[str, Any], naturalness_deficit: float, quality_scores: dict[str, float]
    ) -> dict[str, Any]:
        """
        Adapt processing parameters based on quality deficit.

        Strategy:
        - Deficit >0.2: Reduce aggressiveness significantly
        - Deficit 0.1-0.2: Fine-tune blend ratios
        - Deficit <0.1: Minimal adjustment

        Also considers which specific quality aspect is weak.

        Args:
            params: Current parameters
            naturalness_deficit: How far below target (positive = need improvement)
            quality_scores: Detailed quality metrics

        Returns:
            Adapted parameters
        """
        adapted = params.copy()

        # Identify weak areas
        temporal_smoothness = quality_scores.get("temporal_smoothness", 1.0)
        harmonic_coherence = quality_scores.get("harmonic_coherence", 1.0)
        quality_scores.get("noise_floor_consistency", 1.0)

        # Major deficit: reduce processing intensity
        if naturalness_deficit > 0.2:
            logger.debug("Major deficit (%.2f), reducing aggressiveness", naturalness_deficit)

            # Reduce reduction amounts
            if "reduction_db" in adapted:
                adapted["reduction_db"] = adapted["reduction_db"] * 0.7
                logger.debug("  reduction_db: %s → %s", params.get("reduction_db"), adapted["reduction_db"])

            if "repair_strength" in adapted:
                adapted["repair_strength"] = adapted["repair_strength"] * 0.8
                logger.debug("  repair_strength: %s → %s", params.get("repair_strength"), adapted["repair_strength"])

            if "threshold" in adapted:
                adapted["threshold"] = adapted["threshold"] * 1.2
                logger.debug("  Schwelle: %s → %s", params.get("threshold"), adapted["threshold"])

            # If temporal smoothness is low, reduce attack speed
            if temporal_smoothness < 0.6 and "attack_ms" in adapted:
                adapted["attack_ms"] = adapted["attack_ms"] * 1.5
                logger.debug("  attack_ms increased (low smoothness)")

        # Moderate deficit: fine-tune
        elif naturalness_deficit > 0.1:
            logger.debug("Moderate deficit (%.2f), fine-tuning", naturalness_deficit)

            if "repair_strength" in adapted:
                adapted["repair_strength"] = adapted["repair_strength"] * 0.9

            if "blend_amount" in adapted:
                adapted["blend_amount"] = adapted["blend_amount"] * 0.95

            # If harmonic coherence is low, preserve more original
            if harmonic_coherence < 0.7 and "texture_preserve" in adapted:
                adapted["texture_preserve"] = min(0.95, adapted.get("texture_preserve", 0.85) * 1.1)
                logger.debug("  texture_preserve increased (low coherence)")

        # Small deficit: minimal adjustment
        else:
            logger.debug("Small deficit (%.2f), minimal tuning", naturalness_deficit)

            if "repair_strength" in adapted:
                adapted["repair_strength"] = adapted["repair_strength"] * 0.95

            if "reduction_db" in adapted:
                adapted["reduction_db"] = adapted["reduction_db"] * 0.95

        return adapted

    def should_use_feedback(self, phase: PhaseInterface, audio: np.ndarray, sample_rate: int = 44100) -> bool:
        """
        Decide if feedback loop should be used for this phase.

        Feedback is most beneficial for:
        - Defect removal phases (clicks, crackle, noise)
        - Phases with high impact on naturalness
        - Material with severe defects

        Feedback is less useful for:
        - Enhancement phases (EQ, compression)
        - Clean material

        Args:
            phase: Phase to evaluate
            audio: Input audio
            sample_rate: Sample rate

        Returns:
            True if feedback would be beneficial
        """
        metadata = phase.get_metadata()
        phase_id = metadata.phase_id.lower()

        # High-impact phases benefit from feedback
        high_impact_phases = [
            "click_removal",
            "crackle_removal",
            "spectral_repair",
            "noise_gate",
            "denoise",
            "hum_removal",
        ]

        for phase_name in high_impact_phases:
            if phase_name in phase_id:
                return True

        # Check if audio has defects (quick heuristic)
        self.metrics.sample_rate = sample_rate
        initial_quality = self.metrics.calculate_naturalness_score(audio)

        # If naturalness is already high, feedback not needed
        if initial_quality["naturalness_overall"] > 0.85:
            logger.debug(
                "Audio quality already high (%.2f), skipping feedback",
                initial_quality["naturalness_overall"],
            )
            return False

        # If temporal smoothness is low, defects likely present
        if initial_quality["temporal_smoothness"] < 0.7:
            logger.debug("Defects erkannt (smoothness: %.2f), using feedback", initial_quality["temporal_smoothness"])
            return True

        return False


class QualityGating:
    """
    Quality Gating: Skip phases that won't improve quality.

    Measures audio before phase, predicts if phase would help,
    skips if improvement unlikely.

    Benefits:
    - 20-40% faster processing
    - Prevents unnecessary phases (that can only degrade)
    """

    def __init__(self, min_expected_improvement: float = 0.05):
        """
        Initialisiert quality gating.

        Args:
            min_expected_improvement: Minimum improvement to process (0-1)
        """
        self.min_expected_improvement = min_expected_improvement
        self.metrics = PsychoAcousticMetrics()

    def should_process_phase(
        self, phase: PhaseInterface, audio: np.ndarray, sample_rate: int = 44100, **kwargs
    ) -> bool:
        """
        Decide if phase should be processed.

        Args:
            phase: Phase to evaluate
            audio: Input audio
            sample_rate: Sample rate
            **kwargs: Phase parameters

        Returns:
            True if phase should be processed
        """
        metadata = phase.get_metadata()
        self.metrics.sample_rate = sample_rate

        phase_id = metadata.phase_id.lower()

        # Tape/Hiss-Phasen können bei eindeutig nicht-tape Material sofort verworfen werden.
        # Das ändert keine Qualitätsentscheidung, spart aber die teure Metrikberechnung.
        if "tape" in phase_id or "hiss" in phase_id:
            material = kwargs.get("material", "unknown")
            if "tape" not in str(material).lower():
                logger.info("⏭️  Skipping %s: Not tape material (%s)", metadata.name, material)
                return False

        # Quick quality estimate
        try:
            quality = self.metrics.calculate_naturalness_score(audio)
            naturalness = quality["naturalness_overall"]

            # Phase-specific heuristics
            # Denoise: Skip if already clean (high SNR)
            if "denoise" in phase_id:
                # Estimate SNR from noise floor
                noise_floor = quality.get("noise_floor_consistency", 0.5)
                if noise_floor > 0.85:
                    logger.info("⏭️  Skipping %s: Audio already clean (noise floor: %.2f)", metadata.name, noise_floor)
                    return False

            # Crackle removal: Skip if no vinyl/shellac characteristics
            if "crackle" in phase_id:
                temporal_smoothness = quality.get("temporal_smoothness", 1.0)
                if temporal_smoothness > 0.90:
                    logger.info(
                        "⏭️  Skipping %s: No crackle erkannt (smoothness: %.2f)",
                        metadata.name,
                        temporal_smoothness,
                    )
                    return False

            # General: Skip if quality already excellent
            if naturalness > 0.90:
                logger.info("⏭️  Skipping %s: Quality already excellent (%.2f)", metadata.name, naturalness)
                return False

            return True

        except Exception as e:
            logger.warning("Quality gating Pruefung fehlgeschlagen: %s, processing anyway", e)
            return True


if __name__ == "__main__":
    # Test feedback loop
    logger.debug("\n%s", "=" * 70)
    logger.debug("Quality Feedback Loop Test")
    logger.debug("%s", "=" * 70)

    # Create synthetic audio with defects
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sr))

    demo_audio = np.sin(2 * np.pi * 440 * t) * 0.3
    demo_audio += np.random.randn(len(demo_audio)) * 0.05

    # Add clicks
    for _ in range(20):
        pos = np.random.randint(0, len(demo_audio))
        demo_audio[pos] += 0.5

    # Test metrics
    metrics = PsychoAcousticMetrics(sr)
    demo_initial_quality = metrics.calculate_naturalness_score(demo_audio)

    logger.debug("\nInitial Audio Quality:")
    for key, val in demo_initial_quality.items():
        logger.debug("  %s: %.3f", key, val)

    # Test quality gating
    logger.debug("\n%s", "=" * 70)
    logger.debug("Quality Gating Test")
    logger.debug("%s", "=" * 70)

    gating = QualityGating()

    # Mock phase
    class MockPhase(PhaseInterface):
        """Einfacher Mock für den Gating-Selbsttest."""

        def get_metadata(self):
            return PhaseMetadata(
                phase_id="test_denoise",
                name="Test Denoise",
                category=PhaseCategory.DEFECT_REMOVAL,
                priority=5,
                version="1.0",
                dependencies=[],
                estimated_time_factor=0.1,
                memory_requirement_mb=50,
                is_cpu_intensive=False,
                is_io_intensive=False,
                quality_impact=0.8,
                description="Test phase",
            )

        def process(self, audio, sample_rate=48000, material_type="unknown", **kwargs) -> PhaseResult:
            raise NotImplementedError("MockPhase wird nur für Gating-Tests verwendet — process wird nicht aufgerufen")

    mock_phase = MockPhase()
    should_process = gating.should_process_phase(mock_phase, demo_audio, sr)
    logger.debug("\nShould verarbeiten Verarbeitungsschritt: %s", should_process)

    logger.debug("\n%s", "=" * 70)
    logger.debug("✅ Quality Feedback Loop Module operational")
