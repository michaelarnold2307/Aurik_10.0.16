"""
Results Analyzer - Statistical Analysis for Blind Test Results

Analyzes blind test results and generates statistical reports:
- A/B test preference statistics
- A/B/X identification accuracy
- Rating distributions and averages
- Comparison with objective metrics
- Statistical significance tests

Usage:
    python results_analyzer.py --results results.json --protocol test_protocol.json
    python results_analyzer.py --validation validation_report.json --blind blind_results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze blind test results with statistical rigor."""

    def __init__(self):
        self.results = {}
        self.protocol = {}
        self.analysis = {}

    def load_blind_results(self, results_file: Path, protocol_file: Path):
        """
        Load blind test results and protocol.

        Args:
            results_file: JSON file with evaluator results
            protocol_file: JSON file with test protocol (ground truth)
        """
        logger.info(f"Loading blind test results from {results_file}")

        with open(results_file) as f:
            self.results = json.load(f)

        with open(protocol_file) as f:
            self.protocol = json.load(f)

        logger.info(f"✓ Loaded {len(self.protocol['tests'])} test protocols")

    def analyze_ab_tests(self) -> dict:
        """
        Analyze A/B comparison test results.

        Statistics:
        - Preference percentage (A vs B)
        - AURIK preference vs baseline
        - Confidence correlation with accuracy
        """
        logger.info("Analyzing A/B tests...")

        ab_results = self.results.get("ab_results", [])
        ab_protocol = [t for t in self.protocol["tests"] if t["type"] == "ab"]

        if not ab_results:
            logger.warning("No A/B results found")
            return {}

        # Count preferences
        aurik_preferred = 0
        baseline_preferred = 0
        confidences = []

        for result in ab_results:
            test_id = result["test_id"]
            preference = result["preference"]
            confidence = int(result.get("confidence", 3))

            # Find protocol entry
            protocol_entry = next((p for p in ab_protocol if p["test_id"] == test_id), None)

            if not protocol_entry:
                logger.warning(f"Protocol not found for {test_id}")
                continue

            # Determine what was preferred
            if preference == "A":
                preferred_type = protocol_entry["a_is"]
            elif preference == "B":
                preferred_type = protocol_entry["b_is"]
            else:
                logger.warning(f"Invalid preference '{preference}' for {test_id}")
                continue

            if preferred_type == "test":
                aurik_preferred += 1
            elif preferred_type == "baseline":
                baseline_preferred += 1

            confidences.append(confidence)

        total_valid = aurik_preferred + baseline_preferred

        if total_valid == 0:
            return {}

        # Calculate statistics
        aurik_preference_pct = (aurik_preferred / total_valid) * 100
        baseline_preference_pct = (baseline_preferred / total_valid) * 100
        avg_confidence = np.mean(confidences)

        # Binomial test: Is preference significantly different from 50/50?
        p_value = stats.binom_test(aurik_preferred, total_valid, p=0.5, alternative="two-sided")
        significant = p_value < 0.05

        analysis = {
            "total_tests": len(ab_results),
            "valid_tests": total_valid,
            "aurik_preferred": aurik_preferred,
            "baseline_preferred": baseline_preferred,
            "aurik_preference_pct": round(aurik_preference_pct, 1),
            "baseline_preference_pct": round(baseline_preference_pct, 1),
            "avg_confidence": round(avg_confidence, 2),
            "p_value": round(p_value, 4),
            "statistically_significant": significant,
        }

        logger.info(f"✓ A/B Analysis: AURIK preferred {aurik_preference_pct:.1f}% (p={p_value:.4f})")

        return analysis

    def analyze_abx_tests(self) -> dict:
        """
        Analyze A/B/X identification test results.

        Statistics:
        - Identification accuracy
        - Confusion matrix
        - Random chance comparison (50%)
        """
        logger.info("Analyzing A/B/X tests...")

        abx_results = self.results.get("abx_results", [])
        abx_protocol = [t for t in self.protocol["tests"] if t["type"] == "abx"]

        if not abx_results:
            logger.warning("No A/B/X results found")
            return {}

        # Count correct identifications
        correct = 0
        total = 0

        for result in abx_results:
            test_id = result["test_id"]
            answer = result["x_matches"]

            # Find protocol entry
            protocol_entry = next((p for p in abx_protocol if p["test_id"] == test_id), None)

            if not protocol_entry:
                logger.warning(f"Protocol not found for {test_id}")
                continue

            correct_answer = protocol_entry["x_matches"]

            if answer == correct_answer:
                correct += 1

            total += 1

        if total == 0:
            return {}

        # Calculate statistics
        accuracy_pct = (correct / total) * 100

        # Binomial test: Is accuracy significantly better than random (50%)?
        p_value = stats.binom_test(correct, total, p=0.5, alternative="greater")
        significantly_better_than_random = p_value < 0.05

        analysis = {
            "total_tests": len(abx_results),
            "valid_tests": total,
            "correct_identifications": correct,
            "accuracy_pct": round(accuracy_pct, 1),
            "random_chance_pct": 50.0,
            "p_value": round(p_value, 4),
            "significantly_better_than_random": significantly_better_than_random,
        }

        logger.info(f"✓ A/B/X Analysis: {accuracy_pct:.1f}% accuracy (p={p_value:.4f})")

        return analysis

    def analyze_rating_tests(self) -> dict:
        """
        Analyze rating test results.

        Statistics:
        - Mean ratings per criterion
        - Standard deviations
        - Distribution histograms
        """
        logger.info("Analyzing rating tests...")

        rating_results = self.results.get("rating_results", [])

        if not rating_results:
            logger.warning("No rating results found")
            return {}

        # Collect ratings by criterion
        criteria = ["overall_quality", "naturalness", "clarity", "character_preservation"]
        ratings_by_criterion: Any = {criterion: [] for criterion in criteria}

        for result in rating_results:
            for criterion in criteria:
                rating_str = result.get(criterion, "0")
                try:
                    rating = float(rating_str)
                    if 1 <= rating <= 5:
                        ratings_by_criterion[criterion].append(rating)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid rating '{rating_str}' for {criterion}")

        # Calculate statistics per criterion
        stats_by_criterion = {}

        for criterion, ratings in ratings_by_criterion.items():
            if not ratings:
                continue

            stats_by_criterion[criterion] = {
                "mean": round(float(np.mean(ratings)), 2),
                "std": round(float(np.std(ratings)), 2),
                "median": round(float(np.median(ratings)), 2),
                "min": float(np.min(ratings)),
                "max": float(np.max(ratings)),
                "count": len(ratings),
            }

        # Overall average across all criteria
        all_ratings = []
        for ratings in ratings_by_criterion.values():
            all_ratings.extend(ratings)

        overall_mean = np.mean(all_ratings) if all_ratings else 0

        analysis = {
            "total_tests": len(rating_results),
            "criteria": stats_by_criterion,
            "overall_mean": round(float(overall_mean), 2),
        }

        logger.info(f"✓ Rating Analysis: Overall mean = {overall_mean:.2f}/5.0")

        return analysis

    def combine_with_objective_metrics(self, validation_file: Path) -> dict:
        """
        Combine subjective (blind test) with objective (validation) metrics.

        Args:
            validation_file: JSON file with objective validation results

        Returns:
            Combined analysis with correlations
        """
        logger.info(f"Loading objective metrics from {validation_file}")

        with open(validation_file) as f:
            validation = json.load(f)

        # Extract objective metrics
        categories = validation.get("categories", {})

        objective_summary = {}
        for category, data in categories.items():
            stats = data.get("statistics", {})
            objective_summary[category] = {
                "avg_snr_db": stats.get("avg_snr_db", 0),
                "avg_thd_percent": stats.get("avg_thd_percent", 0),
                "file_count": data.get("file_count", 0),
            }

        combined = {
            "objective_metrics": objective_summary,
            "subjective_metrics": {
                "ab_preference": self.analysis.get("ab_tests", {}),
                "abx_accuracy": self.analysis.get("abx_tests", {}),
                "ratings": self.analysis.get("rating_tests", {}),
            },
        }

        logger.info("✓ Combined objective and subjective metrics")

        return combined

    def generate_report(self, output_file: Path):
        """
        Generate comprehensive analysis report.

        Args:
            output_file: Output JSON file
        """
        logger.info("Generating analysis report...")

        # Run all analyses
        self.analysis["ab_tests"] = self.analyze_ab_tests()
        self.analysis["abx_tests"] = self.analyze_abx_tests()
        self.analysis["rating_tests"] = self.analyze_rating_tests()

        # Add metadata
        self.analysis["evaluator"] = self.results.get("evaluator_name", "Unknown")
        self.analysis["evaluation_date"] = self.results.get("evaluation_date", "Unknown")
        self.analysis["listening_environment"] = self.results.get("listening_environment", "Unknown")

        # Save report
        with open(output_file, "w") as f:
            json.dump(self.analysis, f, indent=2)

        logger.info(f"✓ Analysis report saved to {output_file}")

        # Print summary
        self._print_summary()

        return self.analysis

    def _print_summary(self):
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print("BLIND TEST ANALYSIS SUMMARY")
        print("=" * 60)

        # A/B Tests
        ab = self.analysis.get("ab_tests", {})
        if ab:
            print(f"\n📊 A/B Comparison Tests ({ab['total_tests']} tests)")
            print(f"   AURIK preferred: {ab['aurik_preference_pct']}%")
            print(f"   Baseline preferred: {ab['baseline_preference_pct']}%")
            print(f"   Average confidence: {ab['avg_confidence']}/5.0")
            print(
                f"   Statistical significance: {'YES' if ab['statistically_significant'] else 'NO'} (p={ab['p_value']})"
            )

        # A/B/X Tests
        abx = self.analysis.get("abx_tests", {})
        if abx:
            print(f"\n🎯 A/B/X Identification Tests ({abx['total_tests']} tests)")
            print(f"   Accuracy: {abx['accuracy_pct']}%")
            print(f"   Random chance: {abx['random_chance_pct']}%")
            print(
                f"   Better than random: {'YES' if abx['significantly_better_than_random'] else 'NO'} (p={abx['p_value']})"
            )

        # Rating Tests
        rating = self.analysis.get("rating_tests", {})
        if rating:
            print(f"\n⭐ Rating Tests ({rating['total_tests']} tests)")
            print(f"   Overall mean: {rating['overall_mean']}/5.0")

            criteria = rating.get("criteria", {})
            for criterion, crit_stats in criteria.items():
                print(f"   {criterion}: {crit_stats['mean']} ± {crit_stats['std']}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Results Analyzer")
    parser.add_argument("--results", type=str, required=True, help="Blind test results JSON file")
    parser.add_argument("--protocol", type=str, required=True, help="Test protocol JSON file (ground truth)")
    parser.add_argument("--output", type=str, default="analysis_report.json", help="Output analysis report file")
    parser.add_argument("--validation", type=str, help="Optional: Objective validation report for correlation")

    args = parser.parse_args()

    analyzer = ResultsAnalyzer()

    # Load blind test data
    analyzer.load_blind_results(Path(args.results), Path(args.protocol))

    # Generate analysis report
    analyzer.generate_report(Path(args.output))

    # Optionally combine with objective metrics
    if args.validation:
        combined = analyzer.combine_with_objective_metrics(Path(args.validation))

        combined_file = Path(args.output).parent / "combined_analysis.json"
        with open(combined_file, "w") as f:
            json.dump(combined, f, indent=2)

        logger.info(f"✓ Combined analysis saved to {combined_file}")


if __name__ == "__main__":
    main()
