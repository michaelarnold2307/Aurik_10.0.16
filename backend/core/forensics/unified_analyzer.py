"""
forensics/unified_analyzer.py
Unified Signal Forensics Analyzer
===================================

Integriert alle ML-basierten Detektoren:
- ML Medium Detector (6 Kategorien, 99%+ Ziel)
- ML Era Detector (8 Epochen, 95%+ Ziel)
- ML Defect Detector (5 Defekt-Typen, 98%+ Recall)

Features:
- Hierarchische Analyse (Medium → Era → Defects)
- Confidence Aggregation
- Cross-Detector Validation
- Unified Report Generation
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.core.forensics.ml_defect_detector import DefectDetectionResult, MLDefectDetector
from backend.core.forensics.ml_era_detector import EraDetectionResult, MLEraDetector
from backend.core.forensics.ml_medium_detector import DetectionResult as MediumResult
from backend.core.forensics.ml_medium_detector import MLMediumDetector

logger = logging.getLogger(__name__)


@dataclass
class UnifiedForensicAnalysis:
    """
    Comprehensive forensic analysis result.

    Combines results from all detectors:
    - Medium: Analog/Digital classification
    - Era: Decade detection
    - Defects: Audio quality issues
    """

    # Medium Detection
    medium_type: str  # VINYL, TAPE, CASSETTE, CD, DIGITAL, LOSSY
    medium_confidence: float
    medium_probabilities: dict[str, float]

    # Era Detection
    era: str  # 1950s-2020s
    era_confidence: float
    era_probabilities: dict[str, float]
    era_characteristics: dict[str, Any]

    # Defect Detection
    defects_detected: dict[str, bool]  # Defect type → detected
    defect_confidences: dict[str, float]
    defect_severities: dict[str, str]  # LOW/MEDIUM/HIGH

    # Overall Analysis
    overall_confidence: float  # Aggregated confidence
    analysis_quality: str  # EXCELLENT/GOOD/FAIR/POOR

    # Recommendations
    recommended_processing_chain: list[str]  # Processing modules
    restoration_priority: str  # HIGH/MEDIUM/LOW

    # Summary
    summary: str
    detailed_report: str

    # Metadata
    features_used: int
    model_versions: dict[str, str]


class UnifiedForensicAnalyzer:
    """
    Einheitlicher Signal-Forensik-Analysator.

    Orchestrates ML-based detectors for comprehensive analysis:
    1. Medium Detection → Material classification
    2. Era Detection → Temporal classification
    3. Defect Detection → Quality assessment
    4. Cross-validation → Consistency checks
    5. Report Generation → Unified insights
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        medium_detector: MLMediumDetector | None = None,
        era_detector: MLEraDetector | None = None,
        defect_detector: MLDefectDetector | None = None,
    ) -> None:
        """
        Initialisiert unified analyzer.

        Args:
            medium_detector: Trained ML Medium Detector
            era_detector: Trained ML Era Detector
            defect_detector: Trained ML Defect Detector
        """
        self.medium_detector = medium_detector
        self.era_detector = era_detector
        self.defect_detector = defect_detector

        # Analysis state
        self.last_analysis: UnifiedForensicAnalysis | None = None

    def analyze(self, audio: np.ndarray, sample_rate: int, verbose: bool = True) -> UnifiedForensicAnalysis:
        """
        Führt durch: unified forensic analysis.

        Args:
            audio: Audio signal (mono or stereo)
            sample_rate: Sample rate in Hz
            verbose: Print analysis progress

        Returns:
            UnifiedForensicAnalysis with comprehensive results
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("   Unified Forensic Analyse")
            logger.info("=" * 60)

        results = {}
        model_versions = {}
        features_used = 0

        # 1. Medium Detection
        if self.medium_detector is not None and self.medium_detector.is_trained:
            if verbose:
                logger.info("   [1/3] Medium Detection...")

            medium_result = self.medium_detector.predict(audio, sample_rate)
            results["medium"] = medium_result
            model_versions["medium"] = medium_result.model_version
            features_used += medium_result.features_used

            if verbose:
                logger.info("         Medium: %s (%.1f)", medium_result.category, medium_result.confidence)  # type: ignore[attr-defined]
        else:
            # Default values if detector not available
            results["medium"] = MediumResult(
                medium="UNKNOWN", confidence=0.0, probabilities={}, features_used=0, model_version="N/A"
            )

        # 2. Era Detection
        if self.era_detector is not None and self.era_detector.is_trained:
            if verbose:
                logger.info("   [2/3] Era Detection...")

            era_result = self.era_detector.predict(audio, sample_rate)
            results["era"] = era_result  # type: ignore[assignment]
            model_versions["era"] = era_result.model_version
            features_used += era_result.features_used

            if verbose:
                logger.info("         Era: %s (%.1f)", era_result.era, era_result.confidence)
        else:
            # Default values
            results["era"] = EraDetectionResult(  # type: ignore[assignment]
                era="UNKNOWN",
                confidence=0.0,
                probabilities={},
                features_used=0,
                model_version="N/A",
                era_characteristics={},
            )

        # 3. Defect Detection
        if self.defect_detector is not None and any(self.defect_detector.is_trained.values()):
            if verbose:
                logger.info("   [3/3] Defect Detection...")

            defect_result = self.defect_detector.predict(audio, sample_rate)
            results["defects"] = defect_result  # type: ignore[assignment]
            model_versions["defects"] = defect_result.model_version
            features_used += defect_result.features_used

            if verbose:
                detected = [d for d, v in defect_result.defects_detected.items() if v]
                if detected:
                    logger.info("         Defects: %s", ", ".join(detected))
                else:
                    logger.info("         Defects: None erkannt")
        else:
            # Default values
            results["defects"] = DefectDetectionResult(  # type: ignore[assignment]
                defects_detected={},
                defect_confidences={},
                defect_severities={},
                features_used=0,
                model_version="N/A",
                summary="Defect detection not available",
            )

        # 4. Cross-Validation & Consistency Checks
        if verbose:
            logger.info("\n   Cross-Validierung:")

        consistency_score = self._check_consistency(results, verbose)

        # 5. Aggregate Confidence
        overall_confidence = self._aggregate_confidence(results, consistency_score)

        # 6. Analysis Quality Assessment
        analysis_quality = self._assess_quality(overall_confidence, results)

        # 7. Generate Recommendations
        processing_chain = self._recommend_processing_chain(results)
        restoration_priority = self._assess_restoration_priority(results)

        # 8. Generate Reports
        summary = self._generate_summary(results)
        detailed_report = self._generate_detailed_report(results, consistency_score)

        if verbose:
            logger.info("\n   Overall Confidence: %.1f", overall_confidence)
            logger.info("   Analyse Quality: %s", analysis_quality)
            logger.info("=" * 60)

        # Build unified result
        analysis = UnifiedForensicAnalysis(
            medium_type=results["medium"].medium,
            medium_confidence=results["medium"].confidence,
            medium_probabilities=results["medium"].probabilities,
            era=results["era"].era,  # type: ignore[attr-defined]
            era_confidence=results["era"].confidence,
            era_probabilities=results["era"].probabilities,
            era_characteristics=results["era"].era_characteristics,  # type: ignore[attr-defined]
            defects_detected=results["defects"].defects_detected,  # type: ignore[attr-defined]
            defect_confidences=results["defects"].defect_confidences,  # type: ignore[attr-defined]
            defect_severities=results["defects"].defect_severities,  # type: ignore[attr-defined]
            overall_confidence=overall_confidence,
            analysis_quality=analysis_quality,
            recommended_processing_chain=processing_chain,
            restoration_priority=restoration_priority,
            summary=summary,
            detailed_report=detailed_report,
            features_used=features_used,
            model_versions=model_versions,
        )

        self.last_analysis = analysis
        return analysis

    def _check_consistency(self, results: dict[str, Any], verbose: bool = False) -> float:
        """
        Prüft consistency between detectors.

        Returns:
            Consistency score (0.0-1.0)
        """
        consistency_checks = []

        # Check 1: Medium-Era Consistency
        # Analog media (VINYL, TAPE, CASSETTE) → Older eras (1950s-1990s)
        # Digital media (CD, DIGITAL, LOSSY) → Newer eras (1990s-2020s)

        medium = results["medium"].medium
        era = results["era"].era

        if medium in ["VINYL", "TAPE", "CASSETTE"]:
            # Expect older eras
            if era in ["1950s", "1960s", "1970s", "1980s", "1990s"]:
                consistency_checks.append(1.0)
                if verbose:
                    logger.info("         ✓ Medium-Era consistent (Analog → %s)", era)
            else:
                consistency_checks.append(0.5)
                if verbose:
                    logger.info("         ⚠ Medium-Era inconsistency (Analog → %s)", era)

        elif medium in ["CD", "DIGITAL", "LOSSY"]:
            # Expect newer eras
            if era in ["1990s", "2000s", "2010s", "2020s"]:
                consistency_checks.append(1.0)
                if verbose:
                    logger.info("         ✓ Medium-Era consistent (Digital → %s)", era)
            else:
                consistency_checks.append(0.5)
                if verbose:
                    logger.info("         ⚠ Medium-Era inconsistency (Digital → %s)", era)

        # Check 2: Medium-Defect Consistency
        # Analog media → More defects expected
        # Digital media → Fewer analog defects

        defects = results["defects"]
        analog_defects = ["CLICKS", "HUM", "DROPOUT"]
        digital_defects = ["DISTORTION", "NOISE_BURST"]

        analog_detected = sum(defects.defects_detected.get(d, False) for d in analog_defects)
        digital_detected = sum(defects.defects_detected.get(d, False) for d in digital_defects)

        if medium in ["VINYL", "TAPE", "CASSETTE"]:
            if analog_detected >= digital_detected:
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.7)

        elif medium in ["CD", "DIGITAL", "LOSSY"]:
            if digital_detected >= analog_detected:
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.7)

        # Average consistency
        if consistency_checks:
            return np.mean(consistency_checks)  # type: ignore[return-value]
        return 1.0

    def _aggregate_confidence(self, results: dict[str, Any], consistency_score: float) -> float:
        """
        Aggregate confidence from all detectors.

        Weighted average with consistency bonus.
        """
        confidences = []
        weights = []

        # Medium detection (weight: 0.4)
        if results["medium"].confidence > 0:
            confidences.append(results["medium"].confidence)
            weights.append(0.4)

        # Era detection (weight: 0.3)
        if results["era"].confidence > 0:
            confidences.append(results["era"].confidence)
            weights.append(0.3)

        # Defect detection (weight: 0.3)
        # Average confidence of detected defects
        defect_confs = [c for c in results["defects"].defect_confidences.values() if c > 0]
        if defect_confs:
            confidences.append(np.mean(defect_confs))
            weights.append(0.3)

        if confidences:
            # Weighted average
            weights_norm = np.array(weights) / np.sum(weights)
            base_confidence: float = float(np.sum(np.array(confidences) * weights_norm))

            # Consistency bonus (+5% if consistent, -10% if inconsistent)
            consistency_bonus = (consistency_score - 0.5) * 0.15

            final_confidence = np.clip(base_confidence + consistency_bonus, 0.0, 1.0)
            return final_confidence  # type: ignore[no-any-return]

        return 0.5

    def _assess_quality(self, confidence: float, results: dict[str, Any]) -> str:
        """
        Assess overall analysis quality.
        """
        if confidence >= 0.9:
            return "EXCELLENT"
        elif confidence >= 0.7:
            return "GOOD"
        elif confidence >= 0.5:
            return "FAIR"
        else:
            return "POOR"

    def _recommend_processing_chain(self, results: dict[str, Any]) -> list[str]:
        """
        Recommend processing chain based on analysis.
        """
        chain = []

        medium = results["medium"].medium
        defects = results["defects"]

        # Always start with DC blocker
        chain.append("DCBlocker")

        # Medium-specific modules
        if medium == "VINYL":
            chain.append("RumbleFilter")
            if defects.defects_detected.get("CLICKS", False):
                chain.append("ClickRemover")
            if defects.defects_detected.get("HUM", False):
                chain.append("HumRemover")

        elif medium == "TAPE":
            chain.append("TapeCorrector")
            if defects.defects_detected.get("DROPOUT", False):
                chain.append("DropoutCorrector")
            if defects.defects_detected.get("HUM", False):
                chain.append("HumRemover")

        elif medium == "CASSETTE":
            chain.append("TapeCorrector")
            chain.append("NoiseReducer")
            if defects.defects_detected.get("DROPOUT", False):
                chain.append("DropoutCorrector")

        elif medium == "CD":
            chain.append("DigitalCorrector")
            if defects.defects_detected.get("DISTORTION", False):
                chain.append("DistortionReducer")

        elif medium in ["DIGITAL", "LOSSY"]:
            chain.append("CodecArtifactRemover")
            if defects.defects_detected.get("DISTORTION", False):
                chain.append("DistortionReducer")

        # Universal enhancement
        chain.append("Enhancement")

        return chain

    def _assess_restoration_priority(self, results: dict[str, Any]) -> str:
        """
        Assess restoration priority based on defects.
        """
        defects = results["defects"]

        # Count high-severity defects
        high_severity = sum(1 for severity in defects.defect_severities.values() if severity == "HIGH")

        # Count total detected defects
        total_defects = sum(defects.defects_detected.values())

        if high_severity >= 2 or total_defects >= 4:
            return "HIGH"
        elif high_severity >= 1 or total_defects >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_summary(self, results: dict[str, Any]) -> str:
        """Generiert concise summary."""
        medium = results["medium"].medium
        era = results["era"].era

        defects_list = [d for d, detected in results["defects"].defects_detected.items() if detected]

        defects_str = f", Defects: {', '.join(defects_list)}" if defects_list else ", No defects"

        return f"{medium} ({era}){defects_str}"

    def _generate_detailed_report(self, results: dict[str, Any], consistency_score: float) -> str:
        """Generiert detailed analysis report."""
        lines = []

        lines.append("DETAILED FORENSIC ANALYSIS REPORT")
        lines.append("=" * 60)

        # Medium Detection
        lines.append("\n1. MEDIUM DETECTION:")
        medium_result = results["medium"]
        lines.append(f"   Type: {medium_result.medium}")
        lines.append(f"   Confidence: {medium_result.confidence:.1%}")
        lines.append("   Probabilities:")
        for cat, prob in sorted(medium_result.probabilities.items(), key=lambda x: -x[1])[:3]:
            lines.append(f"     - {cat}: {prob:.1%}")

        # Era Detection
        lines.append("\n2. ERA DETECTION:")
        era_result = results["era"]
        lines.append(f"   Era: {era_result.era}")
        lines.append(f"   Confidence: {era_result.confidence:.1%}")
        lines.append("   Characteristics:")
        for key, value in era_result.era_characteristics.items():
            lines.append(f"     - {key}: {value}")

        # Defect Detection
        lines.append("\n3. DEFECT DETECTION:")
        defect_result = results["defects"]
        detected_defects = [
            (d, defect_result.defect_confidences[d], defect_result.defect_severities[d])
            for d, detected in defect_result.defects_detected.items()
            if detected
        ]

        if detected_defects:
            for defect, conf, severity in detected_defects:
                lines.append(f"   - {defect}: {conf:.1%} ({severity})")
        else:
            lines.append("   No defects detected")

        # Consistency
        lines.append(f"\n4. CONSISTENCY SCORE: {consistency_score:.1%}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def load_models(  # type: ignore[return]
        self,
        medium_model_path: str | None = None,
        era_model_path: str | None = None,
        defect_model_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Lädt trained models from files.

        Args:
            medium_model_path: Path to Medium Detector model
            era_model_path: Path to Era Detector model
            defect_model_path: Path to Defect Detector model
        """
        if medium_model_path:
            self.medium_detector = MLMediumDetector()
            self.medium_detector.load(medium_model_path)  # type: ignore[arg-type]
            logger.info("geladen Medium Detector from %s", medium_model_path)

        if era_model_path:
            self.era_detector = MLEraDetector()
            self.era_detector.load(era_model_path)  # type: ignore[arg-type]
            logger.info("geladen Era Detector from %s", era_model_path)

        if defect_model_path:
            self.defect_detector = MLDefectDetector()
            self.defect_detector.load(defect_model_path)
            logger.info("geladen Defect Detector from %s", defect_model_path)

    def is_ready(self) -> bool:
        """Prüft if analyzer is ready (at least one detector loaded)."""
        return (
            (self.medium_detector is not None and self.medium_detector.is_trained)
            or (self.era_detector is not None and self.era_detector.is_trained)
            or (self.defect_detector is not None and any(self.defect_detector.is_trained.values()))
        )

    def get_status(self) -> dict[str, bool]:
        """Gibt zurück: status of all detectors."""
        return {
            "medium_detector": self.medium_detector is not None and self.medium_detector.is_trained,
            "era_detector": self.era_detector is not None and self.era_detector.is_trained,
            "defect_detector": self.defect_detector is not None and any(self.defect_detector.is_trained.values()),
        }
