"""
Ethics Engine – Epistemic Gate & Conduct Regulator für AURIK v8.0

Diese Komponente fügt eine philosophisch-ethische Meta-Ebene über die
technische Policy Engine hinzu und beantwortet die fundamentale Frage:

    "DÜRFEN wir in dieses Audio eingreifen?"

Architektur:
    PHASE 4: Feature Analysis
         ↓
    PHASE 4.5: Ethics Engine ← NEU!
         ↓ (Epistemic Decision)
    PHASE 5: Policy Engine (technisch)
         ↓
    PHASE 6: Planning

Autor: AURIK Team
Version: 8.0 (Ethics Extension)
Datum: 7. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Models
# ==============================================================================


class EpistemicDecision(Enum):
    """
    Fundamentale ethische Entscheidungen über Eingriffe.

    PRESERVE: Keine Veränderung erlaubt (pure preservation, forensische Integrität)
    MODE_A: Reparatur/Rekonstruktion erlaubt (bewahrender Charakter)
    MODE_B: Gestaltung/Enhancement erlaubt (moderne Reproduktion 2026)
    HARD_STOP: Zone C - Confidence zu niedrig, Eingriff nicht vertretbar
    """

    PRESERVE = "preserve"  # Pure Preservation (keine DSP)
    MODE_A = "mode_a"  # Bewahren (Reparatur + minimal Rekonstruktion)
    MODE_B = "mode_b"  # Gestalten (moderne Reproduktion)
    HARD_STOP = "hard_stop"  # Zone C - zu unsicher


class ProcessingMode(Enum):
    """Processing Modes für Mode A vs Mode B"""

    REPAIR = "repair"  # Korrektur vorhandener Information
    RECONSTRUCTION_MINIMAL = "reconstruction_minimal"  # Minimal, modelliert, reversibel
    RECONSTRUCTION_CREATIVE = "reconstruction_creative"  # Gestaltend (Mode B)


@dataclass
class AuthenticityConstraints:
    """
    Ethische Grenzen für Audio-Processing.

    Basiert auf den 5 Authenticity Safeguards aus AURIK v8.0:
    1. Parallel Processing (≥70% Original)
    2. Voice Identity Preservation (≥95%)
    3. Mix Balance Preservation (<1dB drift)
    4. Spectral Fingerprint Preservation (≥95%)
    5. Formant Stability (<10Hz drift)
    """

    min_original_ratio: float = 0.70  # Minimum 70% Original im Final Mix
    min_voice_identity_score: float = 0.95  # ≥95% Voice Embedding Similarity
    max_mix_balance_drift_db: float = 1.0  # <1dB Mix-Balance Veränderung
    min_spectral_fingerprint_match: float = 0.95  # ≥95% Spektraler Fingerprint
    max_formant_drift_hz: float = 10.0  # <10Hz Formant Drift (F1-F4)


@dataclass
class EthicsReport:
    """Report über ethische Prüfung"""

    decision: EpistemicDecision
    mode: ProcessingMode | None
    reasoning: str
    constraints_passed: bool
    violated_constraints: list[str]
    confidence_score: float
    recommendation: str


# ==============================================================================
# Ethics Engine
# ==============================================================================


class EthicsEngine:
    """
    Epistemic Gate & Conduct Regulator.

    Diese Komponente entscheidet auf philosophisch-ethischer Ebene,
    ob und wie Audio-Material verändert werden darf.

    Usage:
        engine = EthicsEngine()
        decision = engine.epistemic_gate(context)

        if decision == EpistemicDecision.MODE_A:
            plan = create_processing_plan(...)
            approved = engine.conduct_regulator(plan, context)
    """

    def __init__(self, constraints: AuthenticityConstraints | None = None):
        self.constraints = constraints or AuthenticityConstraints()
        self.logger = logger

    def epistemic_gate(self, context: dict[str, Any]) -> EthicsReport:
        """
        Fundamentale Frage: DÜRFEN wir eingreifen?

        Diese Methode prüft auf philosophischer Ebene, ob ein Eingriff
        in das Audio-Material ethisch vertretbar ist.

        Args:
            context: Dict mit:
                - confidence: float (0-1) - Zone-Konfidenz
                - defect_severity: str - "none", "low", "medium", "high"
                - defect_type: str - "click", "noise", "dropout", etc.
                - user_mode: str - "forensic", "archival", "restoration", "modern"
                - medium: str - "vinyl", "tape", "cd", etc.
                - era: str - "1920s", "1950s", "1980s", etc.
                - has_cultural_significance: bool

        Returns:
            EthicsReport mit Decision + Reasoning
        """
        confidence = context.get("confidence", 0.0)
        defect_severity = context.get("defect_severity", "none")
        defect_type = context.get("defect_type", "unknown")
        user_mode = context.get("user_mode", "restoration")
        has_significance = context.get("has_cultural_significance", False)

        # ==============================================
        # DECISION LOGIC
        # ==============================================

        # 1. HARD STOP: Zone C (Confidence < 85%)
        if confidence < 0.85:
            return EthicsReport(
                decision=EpistemicDecision.HARD_STOP,
                mode=None,
                reasoning=f"Confidence {confidence:.2%} < 85% → Zone C. "
                "Eingriff nicht vertretbar bei hoher Unsicherheit.",
                constraints_passed=False,
                violated_constraints=["confidence_threshold"],
                confidence_score=confidence,
                recommendation="Audio unverändert belassen (Pure Preservation).",
            )

        # 2. PRESERVE: Keine Defekte erkannt
        if defect_severity == "none":
            return EthicsReport(
                decision=EpistemicDecision.PRESERVE,
                mode=None,
                reasoning="Keine Defekte erkannt. Audio ist bereits in gutem Zustand.",
                constraints_passed=True,
                violated_constraints=[],
                confidence_score=confidence,
                recommendation="Original Export ohne Processing.",
            )

        # 3. PRESERVE: Forensische Integrität gefordert
        if user_mode == "forensic":
            return EthicsReport(
                decision=EpistemicDecision.PRESERVE,
                mode=None,
                reasoning="Forensic Mode: Absolute Integrität erforderlich. Kein DSP-Eingriff erlaubt.",
                constraints_passed=True,
                violated_constraints=[],
                confidence_score=confidence,
                recommendation="Forensic Export mit vollständiger Chain-of-Custody.",
            )

        # 4. PRESERVE: Cultural Significance + Archival Mode
        if has_significance and user_mode == "archival":
            return EthicsReport(
                decision=EpistemicDecision.PRESERVE,
                mode=None,
                reasoning="Kulturell bedeutsames Material + Archival Mode: Konservative Bewahrung hat Vorrang.",
                constraints_passed=True,
                violated_constraints=[],
                confidence_score=confidence,
                recommendation="Archival Export mit Metadaten-Dokumentation.",
            )

        # 5. MODE A: Klare Reparatur (bekannte Defekte)
        if defect_type in ["click", "pop", "noise", "hum", "clipping"]:
            return EthicsReport(
                decision=EpistemicDecision.MODE_A,
                mode=ProcessingMode.REPAIR,
                reasoning=f"Klarer Defekt erkannt: {defect_type}. Reparatur ethisch vertretbar und technisch machbar.",
                constraints_passed=True,
                violated_constraints=[],
                confidence_score=confidence,
                recommendation="Mode A: Reparatur mit konservativen Parametern.",
            )

        # 6. MODE A: Rekonstruktion (Dropouts, Lücken)
        if defect_type in ["dropout", "gap", "missing_harmonics"]:
            # Nur bei hoher Confidence (>90%)
            if confidence >= 0.90:
                return EthicsReport(
                    decision=EpistemicDecision.MODE_A,
                    mode=ProcessingMode.RECONSTRUCTION_MINIMAL,
                    reasoning=f"Fehlende Information erkannt: {defect_type}. "
                    f"Confidence {confidence:.2%} ≥90% → Minimale Rekonstruktion vertretbar.",
                    constraints_passed=True,
                    violated_constraints=[],
                    confidence_score=confidence,
                    recommendation="Mode A: Minimal-invasive Rekonstruktion mit Rollback-Option.",
                )
            else:
                return EthicsReport(
                    decision=EpistemicDecision.PRESERVE,
                    mode=None,
                    reasoning=f"Fehlende Information erkannt, aber Confidence {confidence:.2%} < 90%. "
                    "Rekonstruktion zu riskant.",
                    constraints_passed=False,
                    violated_constraints=["confidence_threshold_reconstruction"],
                    confidence_score=confidence,
                    recommendation="Preserve Original oder nur Reparatur von klaren Defekten.",
                )

        # 7. MODE B: Moderne Reproduktion (User explizit gewünscht)
        if user_mode == "modern_reproduction":
            return EthicsReport(
                decision=EpistemicDecision.MODE_B,
                mode=ProcessingMode.RECONSTRUCTION_CREATIVE,
                reasoning="User Mode: Modern Reproduction. "
                "Gestalterische Enhancement erlaubt (Re-Voicing, Transients, Balance).",
                constraints_passed=True,
                violated_constraints=[],
                confidence_score=confidence,
                recommendation="Mode B: Kreative Verbesserung mit Authenticity Safeguards.",
            )

        # 8. DEFAULT: Mode A (Conservative Restoration)
        return EthicsReport(
            decision=EpistemicDecision.MODE_A,
            mode=ProcessingMode.REPAIR,
            reasoning=f"Standard Restoration Mode. Defekt: {defect_type}, "
            f"Severity: {defect_severity}, Confidence: {confidence:.2%}.",
            constraints_passed=True,
            violated_constraints=[],
            confidence_score=confidence,
            recommendation="Mode A: Conservative Restoration mit Quality Gates.",
        )

    def conduct_regulator(self, processing_plan: dict[str, Any], context: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Reguliert Verarbeitungsplan nach ethischen Prinzipien.

        GRENZEN (basierend auf 5 Authenticity Safeguards):
        1. Max 30% Veränderung (min 70% Original)
        2. Voice Identity ≥95%
        3. Mix Balance <1dB Drift
        4. Spektraler Fingerprint ≥95%
        5. Formant Drift <10Hz

        Args:
            processing_plan: Dict mit geplanten Processing-Parametern:
                - enhancement_strength: float (0-1)
                - planned_voice_identity_score: float (0-1)
                - planned_mix_balance_drift_db: float
                - planned_spectral_match: float (0-1)
                - planned_formant_drift_hz: float

            context: Zusätzlicher Kontext (Medium, Era, etc.)

        Returns:
            (approved: bool, violations: List[str])
        """
        violations = []

        # 1. Check: Original Ratio (≥70% Original)
        strength = processing_plan.get("enhancement_strength", 0.3)
        if strength > 0.7:
            violations.append(f"Enhancement Strength {strength:.2f} > 0.7 (verletzt 70% Original Minimum)")

        # 2. Check: Voice Identity (≥95%)
        voice_id = processing_plan.get("planned_voice_identity_score", 1.0)
        if voice_id < self.constraints.min_voice_identity_score:
            violations.append(f"Voice Identity {voice_id:.2%} < {self.constraints.min_voice_identity_score:.2%}")

        # 3. Check: Mix Balance (<1dB Drift)
        mix_drift = processing_plan.get("planned_mix_balance_drift_db", 0.0)
        if mix_drift > self.constraints.max_mix_balance_drift_db:
            violations.append(
                f"Mix Balance Drift {mix_drift:.2f}dB > {self.constraints.max_mix_balance_drift_db:.2f}dB"
            )

        # 4. Check: Spectral Fingerprint (≥95%)
        spectral = processing_plan.get("planned_spectral_match", 1.0)
        if spectral < self.constraints.min_spectral_fingerprint_match:
            violations.append(f"Spectral Match {spectral:.2%} < {self.constraints.min_spectral_fingerprint_match:.2%}")

        # 5. Check: Formant Drift (<10Hz)
        formant_drift = processing_plan.get("planned_formant_drift_hz", 0.0)
        if formant_drift > self.constraints.max_formant_drift_hz:
            violations.append(f"Formant Drift {formant_drift:.1f}Hz > {self.constraints.max_formant_drift_hz:.1f}Hz")

        # 6. Special: Cultural Significance → Extra Conservative
        if context.get("has_cultural_significance", False) and strength > 0.5:
            violations.append(f"Kulturell bedeutsames Material: Max Enhancement 0.5 (aktuell: {strength:.2f})")

        # FINAL DECISION
        approved = len(violations) == 0

        if approved:
            self.logger.info("Conduct Regulator: Plan APPROVED ✅")
        else:
            logger.warning("Conduct Regulator: Plan REJECTED ❌ (%s violations)", len(violations))
            for violation in violations:
                logger.warning("  - %s", violation)

        return approved, violations

    def get_mode_description(self, mode: ProcessingMode) -> str:
        """Helpful description für Processing Mode"""
        descriptions = {
            ProcessingMode.REPAIR: "Reparatur: Korrektur vorhandener Information (Click, Noise, Clipping, Sibilanz)",
            ProcessingMode.RECONSTRUCTION_MINIMAL: "Rekonstruktion (minimal): Lokale Ersetzung fehlender Information (Dropouts, Lücken)",
            ProcessingMode.RECONSTRUCTION_CREATIVE: "Rekonstruktion (gestaltend): Moderne Reproduktion (Re-Voicing, neue Präsenz, Transients)",
        }
        return descriptions.get(mode, "Unknown Mode")


# ==============================================================================
# Integration Helper
# ==============================================================================


def integrate_ethics_into_pipeline(pipeline, ethics_engine: EthicsEngine):
    """
    Integriert Ethics Engine in bestehende Processing Pipeline.

    Fügt Phase 4.5 (Ethics Check) zwischen Feature Analysis und Policy Engine ein.

    Args:
        pipeline: Bestehende AdaptiveProcessingPipelineV2
        ethics_engine: EthicsEngine Instanz
    """
    # Phase-4.5-Hook: Ethics Engine zwischen Feature-Analysis und Policy-Engine einschleifen.
    # Unterstützte Pipeline-Schnittstellen werden in absteigender Priorität geprüft.
    try:
        if hasattr(pipeline, "register_phase_hook"):
            # Primäre Schnittstelle: dedizierter Hook-Mechanismus
            pipeline.register_phase_hook("phase_4_5_ethics", ethics_engine.epistemic_gate)
            logger.info(
                "Ethics Engine: Verarbeitungsschritt-4.5-Hook via register_Verarbeitungsschritt_hook() registriert."
            )
        elif hasattr(pipeline, "ethics_engine"):
            # Sekundäre Schnittstelle: direktes Attribut
            pipeline.ethics_engine = ethics_engine
            logger.info("Ethics Engine: direkt in Pipeline-Attribut 'ethics_engine' eingesetzt.")
        elif hasattr(pipeline, "pre_processing_hooks") and isinstance(pipeline.pre_processing_hooks, list):
            # Tertiäre Schnittstelle: generische Pre-Processing-Hook-Liste
            pipeline.pre_processing_hooks.append(ethics_engine.epistemic_gate)
            logger.info("Ethics Engine: als Pre-Processing-Hook hinzugefügt.")
        else:
            logger.warning(
                "Ethics Engine: Pipeline bietet keine bekannte Hook-Schnittstelle. "
                "Manueller Aufruf: engine.epistemic_gate(context) vor jeder Verarbeitung."
            )
    except Exception as exc:
        logger.error("integrate_ethics_into_pipeline fehlgeschlagen: %s", exc)


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Setup
    logging.basicConfig(level=logging.INFO)
    engine = EthicsEngine()

    # Example 1: Vinyl mit Clicks (Mode A - Repair)
    context_vinyl = {
        "confidence": 0.95,
        "defect_severity": "medium",
        "defect_type": "click",
        "user_mode": "restoration",
        "medium": "vinyl",
        "era": "1970s",
        "has_cultural_significance": False,
    }

    report = engine.epistemic_gate(context_vinyl)
    logger.debug("\n=== Example 1: Vinyl mit Clicks ===")
    logger.debug("Decision: %s", report.decision.value)
    logger.debug("Betriebsart: %s", report.mode.value if report.mode else "N/A")
    logger.debug("Reasoning: %s", report.reasoning)
    logger.debug("Recommendation: %s", report.recommendation)

    # Example 2: Forensic Material (Preserve)
    context_forensic = {
        "confidence": 0.92,
        "defect_severity": "low",
        "defect_type": "noise",
        "user_mode": "forensic",
        "medium": "tape",
        "era": "1990s",
        "has_cultural_significance": True,
    }

    report2 = engine.epistemic_gate(context_forensic)
    logger.debug("\n=== Example 2: Forensic Material ===")
    logger.debug("Decision: %s", report2.decision.value)
    logger.debug("Reasoning: %s", report2.reasoning)

    # Example 3: Modern Reproduction (Mode B)
    context_modern = {
        "confidence": 0.98,
        "defect_severity": "low",
        "defect_type": "noise",
        "user_mode": "modern_reproduction",
        "medium": "cd",
        "era": "2000s",
        "has_cultural_significance": False,
    }

    report3 = engine.epistemic_gate(context_modern)
    logger.debug("\n=== Example 3: Modern Reproduction ===")
    logger.debug("Decision: %s", report3.decision.value)
    logger.debug("Betriebsart: %s", report3.mode.value if report3.mode else "N/A")
    logger.debug("Recommendation: %s", report3.recommendation)

    # Example 4: Conduct Regulator Check
    logger.debug("\n=== Example 4: Conduct Regulator ===")
    processing_plan_good = {
        "enhancement_strength": 0.3,  # 70% orig + 30% enhanced
        "planned_voice_identity_score": 0.97,
        "planned_mix_balance_drift_db": 0.5,
        "planned_spectral_match": 0.96,
        "planned_formant_drift_hz": 8.0,
    }

    approved, violations = engine.conduct_regulator(processing_plan_good, context_vinyl)
    logger.debug("Plan Approved: %s", approved)
    if violations:
        logger.debug("Violations: %s", violations)

    # Example 5: Conduct Regulator Rejection
    processing_plan_bad = {
        "enhancement_strength": 0.8,  # TOO AGGRESSIVE (>70%)
        "planned_voice_identity_score": 0.92,  # TOO LOW (<95%)
        "planned_mix_balance_drift_db": 1.5,  # TOO HIGH (>1dB)
    }

    approved2, violations2 = engine.conduct_regulator(processing_plan_bad, context_vinyl)
    logger.debug("\nPlan 2 Approved: %s", approved2)
    logger.debug("Violations: %s", violations2)
