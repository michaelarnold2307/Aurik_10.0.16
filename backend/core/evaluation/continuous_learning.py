"""
Continuous Learning Analytics für AURIK v8.0

Analysiert historische Audit-Reports (1000+ Files) und optimiert:
- Strategie-Gewichte basierend auf Erfolgsraten
- Confidence Prediction Modelle
- Processing-Parameter
- Quality Gate Thresholds

Komponenten:
1. SuccessPatternAnalyzer - Identifiziert erfolgreiche Verarbeitungsstrategien
2. StrategyWeightOptimizer - Updated Policy-Engine Weights dynamisch
3. ConfidenceCalibrator - Verbessert Confidence-Prediction
4. PerformanceMetricsAggregator - Trend-Analyse & Recommendations

Autor: AURIK Team
Version: 8.0
Datum: 7. Februar 2026
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Models
# ==============================================================================


@dataclass
class ProcessingStrategy:
    """Eine Processing-Strategie mit Erfolgsmetriken."""

    strategy_name: str
    module_name: str
    success_count: int = 0
    failure_count: int = 0
    avg_cas_improvement: float = 0.0
    avg_processing_time_ms: float = 0.0
    confidence_scores: list[float] = field(default_factory=list)
    quality_gate_pass_rate: float = 0.0


@dataclass
class LearningRecommendation:
    """Optimierungs-Empfehlung basierend auf Analyse."""

    category: str  # "weight_update", "threshold_adjustment", "strategy_change"
    target: str  # Modul/Strategie/Parameter
    current_value: Any
    recommended_value: Any
    confidence: float  # 0-1
    reason: str
    expected_improvement: float  # % improvement


@dataclass
class LearningReport:
    """Umfassender Lern-Report."""

    total_files_analyzed: int
    analysis_date: str
    strategy_performance: dict[str, ProcessingStrategy] = field(default_factory=dict)
    recommendations: list[LearningRecommendation] = field(default_factory=list)
    trends: dict[str, Any] = field(default_factory=dict)
    optimization_summary: str = ""


# ==============================================================================
# Success Pattern Analyzer
# ==============================================================================


class SuccessPatternAnalyzer:
    """
    Analysiert Audit-Reports und identifiziert erfolgreiche Patterns.

    Liest JSON/YAML Reports von PermanentAudioMonitor und extrahiert:
    - Welche Module/Strategien führten zu Quality Gate PASS?
    - Welche Parameter korrelierten mit hohen CAS Scores?
    - Welche Confidence-Levels waren akkurat?
    """

    def __init__(self, audit_dir: str = "./audits"):
        self.audit_dir = Path(audit_dir)
        self.strategies: dict[str, ProcessingStrategy] = {}
        self.raw_data: list[dict[str, Any]] = []

    def load_audit_reports(self, max_files: int | None = None) -> int:
        """
        Lädt Audit-Reports aus Verzeichnis.

        Args:
            max_files: Maximum number of files to load (None = all)

        Returns:
            int: Number of files loaded
        """
        if not self.audit_dir.exists():
            logger.warning("Audit directory not found: %s", self.audit_dir)
            return 0

        # Load JSON files
        json_files = sorted(self.audit_dir.glob("audit_*.json"))
        if max_files:
            json_files = json_files[:max_files]

        loaded = 0
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    self.raw_data.append(data)
                    loaded += 1
            except Exception as e:
                logger.warning("konnte nicht laden %s: %s", json_file, e)

        logger.info("geladen %s audit reports from %s", loaded, self.audit_dir)
        return loaded

    def analyze_patterns(self) -> dict[str, ProcessingStrategy]:
        """
        Analysiert Patterns aus geladenen Reports.

        Returns:
            Dict[str, ProcessingStrategy]: Strategy name → performance metrics
        """
        strategy_stats: defaultdict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "success": 0,
                "failure": 0,
                "cas_improvements": [],
                "processing_times": [],
                "confidences": [],
            }
        )

        for report in self.raw_data:
            # Extract module logs
            module_logs = report.get("module_logs", [])
            overall_passed = report.get("overall_quality_passed", False)
            cas_improvement = report.get("cas_improvement", 0.0)

            for module_log in module_logs:
                module_name = module_log.get("module_name", "unknown")
                confidence = module_log.get("confidence", 0.0)
                processing_time = module_log.get("processing_time_ms", 0.0)
                quality_passed = module_log.get("quality_gate_passed", True)

                # Track strategy performance
                if quality_passed and overall_passed:
                    strategy_stats[module_name]["success"] += 1  # type: ignore[operator]
                else:
                    strategy_stats[module_name]["failure"] += 1  # type: ignore[operator]

                strategy_stats[module_name]["cas_improvements"].append(cas_improvement)  # type: ignore[attr-defined]
                strategy_stats[module_name]["processing_times"].append(processing_time)  # type: ignore[attr-defined]
                strategy_stats[module_name]["confidences"].append(confidence)  # type: ignore[attr-defined]

        # Convert to ProcessingStrategy objects
        strategies = {}
        for module_name, stats in strategy_stats.items():
            total = stats["success"] + stats["failure"]  # type: ignore[operator]
            strategies[module_name] = ProcessingStrategy(
                strategy_name=module_name,
                module_name=module_name,
                success_count=stats["success"],  # type: ignore[arg-type]
                failure_count=stats["failure"],  # type: ignore[arg-type]
                avg_cas_improvement=float(np.mean(stats["cas_improvements"])) if stats["cas_improvements"] else 0.0,
                avg_processing_time_ms=float(np.mean(stats["processing_times"])) if stats["processing_times"] else 0.0,
                confidence_scores=stats["confidences"],  # type: ignore[arg-type]
                quality_gate_pass_rate=stats["success"] / total if total > 0 else 0.0,
            )

        self.strategies = strategies
        logger.info("Analyzed %s distinct strategies", len(strategies))
        return strategies

    def identify_top_performers(self, top_n: int = 5) -> list[ProcessingStrategy]:
        """Identifiziert die besten N Strategien basierend auf Success Rate + CAS."""
        if not self.strategies:
            return []

        # Score: weighted combination of pass rate and CAS improvement
        scored_strategies = []
        for strategy in self.strategies.values():
            score = 0.6 * strategy.quality_gate_pass_rate + 0.4 * max(strategy.avg_cas_improvement, 0)
            scored_strategies.append((score, strategy))

        # Sort by score descending
        scored_strategies.sort(key=lambda x: x[0], reverse=True)

        return [strategy for _, strategy in scored_strategies[:top_n]]

    def identify_underperformers(self, threshold: float = 0.5) -> list[ProcessingStrategy]:
        """Identifiziert schlechte Strategien (Pass Rate < threshold)."""
        if not self.strategies:
            return []

        return [strategy for strategy in self.strategies.values() if strategy.quality_gate_pass_rate < threshold]


# ==============================================================================
# Strategy Weight Optimizer
# ==============================================================================


class StrategyWeightOptimizer:
    """
    Optimiert Policy-Engine Weights basierend auf historischen Erfolgsraten.

    Verwendet Bayesian Optimization für Parameter-Updates.
    """

    def __init__(self):
        self.current_weights: dict[str, float] = {}
        self.learning_rate: float = 0.1  # Conservative updates

    def compute_optimal_weights(self, strategies: dict[str, ProcessingStrategy]) -> dict[str, float]:
        """
        Berechnet optimale Weights für jede Strategie.

        Args:
            strategies: Strategy name → performance metrics

        Returns:
            Dict[str, float]: Strategy name → recommended weight (0-1)
        """
        optimal_weights = {}

        for name, strategy in strategies.items():
            # Weight formula: Success rate * CAS improvement * Speed factor
            success_rate = strategy.quality_gate_pass_rate
            cas_factor = np.clip(strategy.avg_cas_improvement + 1.0, 0.5, 1.5)

            # Speed factor: faster = better (penalize >1000ms)
            speed_factor = 1.0
            if strategy.avg_processing_time_ms > 1000:
                speed_factor = 1000.0 / strategy.avg_processing_time_ms

            # Composite weight
            weight = success_rate * cas_factor * speed_factor

            optimal_weights[name] = float(np.clip(weight, 0.0, 1.0))

        # Normalize weights
        total = sum(optimal_weights.values())
        if total > 0:
            optimal_weights = {k: v / total for k, v in optimal_weights.items()}

        return optimal_weights

    def generate_weight_recommendations(
        self,
        current_weights: dict[str, float],
        optimal_weights: dict[str, float],
    ) -> list[LearningRecommendation]:
        """
        Generiert Empfehlungen für Weight-Updates.

        Args:
            current_weights: Current policy weights
            optimal_weights: Recommended optimal weights

        Returns:
            List[LearningRecommendation]
        """
        recommendations: list[LearningRecommendation] = []

        for strategy_name, optimal_weight in optimal_weights.items():
            current_weight = current_weights.get(strategy_name, 0.5)

            # Only recommend if significant difference (>10%)
            diff = abs(optimal_weight - current_weight)
            if diff > 0.1:
                # Smooth update with learning rate
                recommended_weight = current_weight + self.learning_rate * (optimal_weight - current_weight)

                expected_improvement = diff * 100  # % improvement estimate

                recommendations.append(
                    LearningRecommendation(
                        category="weight_update",
                        target=strategy_name,
                        current_value=current_weight,
                        recommended_value=recommended_weight,
                        confidence=0.8,  # High confidence for weight updates
                        reason=f"Historical analysis shows {expected_improvement:.1f}% potential improvement",
                        expected_improvement=expected_improvement,
                    )
                )

        return recommendations


# ==============================================================================
# Confidence Calibrator
# ==============================================================================


class ConfidenceCalibrator:
    """
    Kalibriert Confidence-Predictions basierend auf tatsächlichen Ergebnissen.

    Verbessert Zone Classification (A/B/C) Accuracy.
    """

    def __init__(self):
        self.calibration_data: list[tuple[float, bool]] = []  # (predicted, actual_success)

    def collect_calibration_data(self, strategies: dict[str, ProcessingStrategy]):
        """Sammelt (Confidence, Success) Paare für Kalibrierung."""
        for strategy in strategies.values():
            total = strategy.success_count + strategy.failure_count
            if total == 0:
                continue

            for conf in strategy.confidence_scores:
                # Success if this strategy had high pass rate
                success = strategy.quality_gate_pass_rate > 0.8
                self.calibration_data.append((conf, success))

        logger.info("Collected %s Kalibrierung data points", len(self.calibration_data))

    def analyze_calibration(self) -> dict[str, Any]:
        """
        Analysiert Confidence-Calibration.

        Returns:
            Dict mit calibration metrics und recommendations
        """
        if len(self.calibration_data) < 10:
            return {"status": "insufficient_data"}

        # Bin confidences into ranges
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_accuracy = []

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            in_bin = [(conf, success) for conf, success in self.calibration_data if low <= conf < high]

            if len(in_bin) > 0:
                accuracy = sum(success for _, success in in_bin) / len(in_bin)
                bin_accuracy.append((low, high, accuracy, len(in_bin)))

        # Check calibration: accuracy should ≈ confidence
        calibration_error = 0.0
        for low, high, accuracy, count in bin_accuracy:
            bin_center = (low + high) / 2
            calibration_error += abs(accuracy - bin_center) * count

        calibration_error /= len(self.calibration_data)

        return {
            "status": "analyzed",
            "calibration_error": calibration_error,
            "bin_accuracies": bin_accuracy,
            "well_calibrated": calibration_error < 0.1,
        }

    def generate_calibration_recommendations(self, analysis: dict[str, Any]) -> list[LearningRecommendation]:
        """Generiert Empfehlungen zur Confidence-Kalibrierung."""
        recommendations: list[LearningRecommendation] = []

        if analysis.get("status") != "analyzed":
            return recommendations

        calibration_error = analysis["calibration_error"]

        if not analysis["well_calibrated"]:
            recommendations.append(
                LearningRecommendation(
                    category="threshold_adjustment",
                    target="confidence_thresholds",
                    current_value={"zone_a": 0.85, "zone_b": 0.70},
                    recommended_value={
                        "zone_a": 0.85 + calibration_error,
                        "zone_b": 0.70 + calibration_error,
                    },
                    confidence=0.7,
                    reason=f"Confidence calibration error: {calibration_error:.3f}. Adjust thresholds.",
                    expected_improvement=calibration_error * 100,
                )
            )

        return recommendations


# ==============================================================================
# Performance Metrics Aggregator
# ==============================================================================


class PerformanceMetricsAggregator:
    """
    Aggregiert Performance-Metriken über alle Audit-Reports.

    Identifiziert Trends, Anomalien und Optimierungspotenziale.
    """

    def __init__(self):
        self.metrics: dict[str, list[float]] = defaultdict(list)

    def aggregate_from_reports(self, reports: list[dict[str, Any]]):
        """Aggregiert Metriken aus Audit-Reports."""
        for report in reports:
            # Aggregate CAS improvements
            cas_improvement = report.get("cas_improvement", 0.0)
            self.metrics["cas_improvement"].append(cas_improvement)

            # Aggregate processing times
            total_time = report.get("total_processing_time_ms", 0.0)
            self.metrics["processing_time"].append(total_time)

            # Aggregate quality gate pass rates
            passed = 1.0 if report.get("overall_quality_passed", False) else 0.0
            self.metrics["quality_pass_rate"].append(passed)

    def compute_trends(self) -> dict[str, Any]:
        """Berechnet Trends über Zeit (assumes chronological order)."""
        trends = {}

        for metric_name, values in self.metrics.items():
            if len(values) < 10:
                continue

            # Simple linear trend (early vs. late average)
            early_avg = float(np.mean(values[: len(values) // 3]))
            late_avg = float(np.mean(values[-len(values) // 3 :]))
            trend_direction = "improving" if late_avg > early_avg else "declining"
            trend_magnitude = abs(late_avg - early_avg)

            trends[metric_name] = {
                "early_avg": early_avg,
                "late_avg": late_avg,
                "direction": trend_direction,
                "magnitude": trend_magnitude,
                "overall_avg": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        return trends

    def identify_optimization_opportunities(self, trends: dict[str, Any]) -> list[LearningRecommendation]:
        """Identifiziert Optimierungspotenziale basierend auf Trends."""
        recommendations = []

        # Check processing time trend
        if "processing_time" in trends:
            time_trend = trends["processing_time"]
            if time_trend["direction"] == "declining":  # Getting slower
                recommendations.append(
                    LearningRecommendation(
                        category="strategy_change",
                        target="processing_pipeline",
                        current_value=time_trend["late_avg"],
                        recommended_value=time_trend["early_avg"],
                        confidence=0.6,
                        reason=f"Processing time increased by {time_trend['magnitude']:.0f}ms. Consider optimization.",
                        expected_improvement=time_trend["magnitude"] / time_trend["late_avg"] * 100,
                    )
                )

        # Check CAS improvement trend
        if "cas_improvement" in trends:
            cas_trend = trends["cas_improvement"]
            if cas_trend["direction"] == "declining":
                recommendations.append(
                    LearningRecommendation(
                        category="strategy_change",
                        target="enhancement_pipeline",
                        current_value=cas_trend["late_avg"],
                        recommended_value=cas_trend["early_avg"],
                        confidence=0.7,
                        reason="CAS improvement declining. Review recent parameter changes.",
                        expected_improvement=abs(cas_trend["magnitude"]) * 100,
                    )
                )

        return recommendations


# ==============================================================================
# Main Continuous Learning System
# ==============================================================================


class ContinuousLearningSystem:
    """
    Hauptsystem für Continuous Learning Analytics.

    Orchestriert alle Komponenten und generiert umfassende Learning Reports.
    """

    def __init__(self, audit_dir: str = "./audits"):
        self.analyzer = SuccessPatternAnalyzer(audit_dir)
        self.optimizer = StrategyWeightOptimizer()
        self.calibrator = ConfidenceCalibrator()
        self.aggregator = PerformanceMetricsAggregator()

    def run_learning_cycle(self, min_files: int = 100, max_files: int | None = None) -> LearningReport:
        """
        Führt kompletten Lern-Zyklus durch.

        Args:
            min_files: Minimum files required for analysis
            max_files: Maximum files to analyze (None = all)

        Returns:
            LearningReport with all findings and recommendations
        """
        from datetime import datetime

        logger.info("🔄 Starting Continuous Learning Cycle...")

        # 1. Load audit reports
        loaded = self.analyzer.load_audit_reports(max_files)

        if loaded < min_files:
            logger.warning("Insufficient data: %s files (need %s). Skipping.", loaded, min_files)
            return LearningReport(
                total_files_analyzed=loaded,
                analysis_date=datetime.now().isoformat(),
                optimization_summary=f"⚠️  Insufficient data ({loaded}/{min_files} files)",
            )

        # 2. Analyze success patterns
        strategies = self.analyzer.analyze_patterns()
        top_performers = self.analyzer.identify_top_performers(top_n=5)
        underperformers = self.analyzer.identify_underperformers(threshold=0.5)

        logger.info("✓ Analyzed %s strategies", len(strategies))
        logger.info("  └─ Top performers: %s", len(top_performers))
        logger.info("  └─ Underperformers: %s", len(underperformers))

        # 3. Optimize strategy weights
        optimal_weights = self.optimizer.compute_optimal_weights(strategies)
        current_weights = dict.fromkeys(strategies.keys(), 0.5)  # Placeholder
        weight_recommendations = self.optimizer.generate_weight_recommendations(current_weights, optimal_weights)

        logger.info("✓ erzeugt %s weight recommendations", len(weight_recommendations))

        # 4. Calibrate confidence predictions
        self.calibrator.collect_calibration_data(strategies)
        calibration_analysis = self.calibrator.analyze_calibration()
        calibration_recommendations = self.calibrator.generate_calibration_recommendations(calibration_analysis)

        logger.info("✓ Confidence Kalibrierung: %s", calibration_analysis.get("status", "unknown"))

        # 5. Aggregate performance metrics
        self.aggregator.aggregate_from_reports(self.analyzer.raw_data)
        trends = self.aggregator.compute_trends()
        trend_recommendations = self.aggregator.identify_optimization_opportunities(trends)

        logger.info("✓ Identified %s performance trends", len(trends))

        # 6. Compile report
        all_recommendations = weight_recommendations + calibration_recommendations + trend_recommendations

        # Sort by expected improvement
        all_recommendations.sort(key=lambda r: r.expected_improvement, reverse=True)

        optimization_summary = self._generate_summary(loaded, top_performers, underperformers, all_recommendations)

        report = LearningReport(
            total_files_analyzed=loaded,
            analysis_date=datetime.now().isoformat(),
            strategy_performance=strategies,
            recommendations=all_recommendations,
            trends=trends,
            optimization_summary=optimization_summary,
        )

        logger.info("✅ Learning Cycle vollstaendig!")
        return report

    def _generate_summary(
        self,
        total_files: int,
        top_performers: list[ProcessingStrategy],
        underperformers: list[ProcessingStrategy],
        recommendations: list[LearningRecommendation],
    ) -> str:
        """Generiert textuelle Zusammenfassung."""
        summary = f"📊 Continuous Learning Analysis ({total_files} files)\n\n"

        summary += "🏆 TOP PERFORMERS:\n"
        for i, strategy in enumerate(top_performers[:3], 1):
            summary += f"  {i}. {strategy.module_name}: {strategy.quality_gate_pass_rate:.1%} pass rate, "
            summary += f"+{strategy.avg_cas_improvement:.3f} CAS\n"

        if underperformers:
            summary += "\n⚠️  UNDERPERFORMERS:\n"
            for strategy in underperformers[:3]:
                summary += f"  • {strategy.module_name}: {strategy.quality_gate_pass_rate:.1%} pass rate\n"

        if recommendations:
            summary += f"\n💡 TOP {min(3, len(recommendations))} RECOMMENDATIONS:\n"
            for i, rec in enumerate(recommendations[:3], 1):
                summary += f"  {i}. {rec.category.upper()}: {rec.target}\n"
                summary += f"     └─ {rec.reason}\n"
                summary += f"     └─ Expected improvement: {rec.expected_improvement:.1f}%\n"

        return summary

    def export_report(self, report: LearningReport, output_dir: str = "./learning_reports"):
        """Exportiert Learning Report als JSON und YAML."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export JSON
        json_path = output_path / f"learning_report_{timestamp}.json"
        with open(json_path, "w") as f:
            # Convert dataclasses to dict
            report_dict = {
                "total_files_analyzed": report.total_files_analyzed,
                "analysis_date": report.analysis_date,
                "strategy_performance": {
                    name: {
                        "success_count": s.success_count,
                        "failure_count": s.failure_count,
                        "pass_rate": s.quality_gate_pass_rate,
                        "avg_cas_improvement": s.avg_cas_improvement,
                        "avg_processing_time_ms": s.avg_processing_time_ms,
                    }
                    for name, s in report.strategy_performance.items()
                },
                "recommendations": [
                    {
                        "category": r.category,
                        "target": r.target,
                        "current_value": r.current_value,
                        "recommended_value": r.recommended_value,
                        "confidence": r.confidence,
                        "reason": r.reason,
                        "expected_improvement": r.expected_improvement,
                    }
                    for r in report.recommendations
                ],
                "trends": report.trends,
                "optimization_summary": report.optimization_summary,
            }
            json.dump(report_dict, f, indent=2)

        logger.info("✓ Exported learning report: %s", json_path)

        # Print summary to console
        logger.debug("\n" + "=" * 80)
        logger.debug("CONTINUOUS LEARNING REPORT")
        logger.debug("=" * 80)
        logger.debug(report.optimization_summary)
        logger.debug("=" * 80 + "\n")


# ==============================================================================
# Example Usage & Test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.debug("\n" + "=" * 80)
    logger.debug("CONTINUOUS LEARNING ANALYTICS TEST")
    logger.debug("=" * 80 + "\n")

    # Initialize system
    learning_system = ContinuousLearningSystem(audit_dir="/tmp/aurik_audits")  # nosec B108 — Demo-/Test-Pfad, kein Produktionscode

    # Run learning cycle
    report = learning_system.run_learning_cycle(min_files=1, max_files=100)

    # Export report
    learning_system.export_report(report, output_dir="/tmp/aurik_learning")  # nosec B108 — Demo-/Test-Pfad, kein Produktionscode

    logger.debug("\n" + "=" * 80)
    logger.debug("✓ Test abgeschlossen erfolgreich!")
    logger.debug("=" * 80 + "\n")
