"""
Golden Sample Benchmark Runner für AURIK v8
===========================================

Automatisches Benchmarking auf Golden Sample Library.

Purpose:
- Load all golden samples
- Process with AURIK restoration/enhancement
- Measure quality metrics (Musical Goals + Perceptual)
- Compare against baseline
- Generate comprehensive benchmark report

Excellence Strategy #5: Golden Sample Library
- Automated quality validation
- Regression detection
- Performance tracking over time

Autor: AI Team
Datum: 11. Februar 2026
"""

import argparse
import importlib
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# Import AURIK components
from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
from backend.core.musical_goals.processing_modes import ProcessingMode
from backend.core.musical_goals.quality_gate import EnhancedQualityGate

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single golden sample."""
    filename: str
    category: str
    baseline_scores: Dict[str, float]
    achieved_scores: Dict[str, float]
    improvements: Dict[str, float]
    degradations: Dict[str, float]
    processing_time_s: float
    quality_gate_decision: str
    perceptual_metrics: Optional[Dict[str, float]]
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results."""
    total_samples: int
    passed: int
    failed: int
    avg_improvement: float
    avg_processing_time_s: float
    category_results: Dict[str, Dict[str, float]]
    baseline_avg: Dict[str, float]
    achieved_avg: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GoldenSampleBenchmarkRunner:
    """
    Benchmark runner for golden sample library.

    Workflow:
    1. Load golden samples from golden_samples/
    2. Load baselines from metadata.json
    3. Process each sample through AURIK
    4. Measure Musical Goals + Perceptual Quality
    5. Compare against baseline
    6. Generate benchmark report

    Features:
    - Parallel processing (optional)
    - Progress tracking
    - Error handling (skip failed samples)
    - Comprehensive reporting
    """

    def __init__(
        self,
        golden_samples_dir: Path,
        processing_mode: ProcessingMode = ProcessingMode.STUDIO_2026,
        enable_perceptual_metrics: bool = True,
        enable_quality_gates: bool = True
    ):
        """
        Initialize benchmark runner.

        Args:
            golden_samples_dir: Path to golden_samples/ directory
            processing_mode: AURIK processing mode
            enable_perceptual_metrics: If True, measure NISQA/DNSMOS/ViSQOL/CDPAM
            enable_quality_gates: If True, use EnhancedQualityGate
        """
        self.golden_samples_dir = Path(golden_samples_dir)
        self.processing_mode = processing_mode
        self.enable_perceptual_metrics = enable_perceptual_metrics
        self.enable_quality_gates = enable_quality_gates

        # Load metadata
        metadata_path = self.golden_samples_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Initialize quality checker
        self.musical_goals_checker = MusicalGoalsChecker()

        # Initialize quality gate (if enabled)
        if self.enable_quality_gates:
            self.quality_gate = EnhancedQualityGate(
                enable_perceptual_metrics=enable_perceptual_metrics,
                enable_auto_reprocessing=False  # Benchmark = no reprocessing
            )
        else:
            self.quality_gate = None  # type: ignore[assignment]

        logger.info(
            "BenchmarkRunner initialisiert: %d samples, Betriebsart=%s, perceptual=%s",
            len(self.metadata["golden_samples"]),
            processing_mode.value,
            enable_perceptual_metrics,
        )

    def run_benchmark(
        self,
        categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        processing_function: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
    ) -> Tuple[List[BenchmarkResult], BenchmarkSummary]:
        """
        Run benchmark on golden samples.

        Args:
            categories: List of categories to benchmark (None = all)
            max_samples: Maximum samples to process (None = all)
            processing_function: Custom processing function (audio, sr) -> processed_audio
                                If None, uses identity (baseline measurement)

        Returns:
            (results, summary)
        """
        samples = self.metadata["golden_samples"]

        # Filter by category
        if categories:
            samples = [s for s in samples if s["category"] in categories]

        # Limit samples
        if max_samples:
            samples = samples[:max_samples]

        logger.info("laeuft benchmark on %d samples...", len(samples))

        results = []

        for i, sample_meta in enumerate(samples, 1):
            logger.info("[%d/%d] Processing %s...", i, len(samples), sample_meta["filename"])

            try:
                result = self._benchmark_sample(sample_meta, processing_function)
                results.append(result)
                avg_improvement = np.mean(list(result.improvements.values())) if result.improvements else 0.0

                logger.info(
                    "  Ergebnis: passed=%s, avg_improvement=%.3f, time=%.2fs",
                    result.passed,
                    avg_improvement,
                    result.processing_time_s,
                )

            except Exception as e:
                logger.error("  konnte nicht benchmark %s: %s", sample_meta["filename"], e)
                continue

        # Generate summary
        summary = self._generate_summary(results)

        logger.info("\n✓ Benchmark vollstaendig: %d/%d passed", summary.passed, summary.total_samples)

        return results, summary

    def _benchmark_sample(
        self,
        sample_meta: Dict,
        processing_function: Optional[Callable[[np.ndarray, int], np.ndarray]]
    ) -> BenchmarkResult:
        """Benchmark a single sample."""
        # Load audio
        audio_path = self.golden_samples_dir / sample_meta["category"] / sample_meta["filename"]
        audio, sr = sf.read(audio_path)
        # sf.read returns (N, C) for multi-channel — UV3 and MusicalGoalsChecker handle stereo natively.

        # Get baseline from metadata
        baseline_scores = sample_meta.get("quality_baseline", {})

        # Process audio (or identity for baseline measurement)
        start_time = time.time()

        if processing_function:
            processed = processing_function(audio, sr)
        else:
            # No processing = baseline measurement
            processed = audio.copy()

        processing_time = time.time() - start_time

        # Measure Musical Goals
        achieved_scores = self.musical_goals_checker.measure_all(processed, sr)

        # Calculate improvements/degradations
        improvements = {}
        degradations = {}

        for goal in achieved_scores:
            baseline = baseline_scores.get(goal, 0.0)
            achieved = achieved_scores[goal]
            delta = achieved - baseline

            if delta > 0.01:
                improvements[goal] = delta
            elif delta < -0.01:
                degradations[goal] = delta

        # Quality gate validation (if enabled)
        passed = True
        quality_gate_decision = "not_checked"
        perceptual_metrics = None

        if self.quality_gate:
            # Use quality gate for validation
            post_check = self.quality_gate.enhanced_post_check(
                original=audio,
                processed=processed,
                sr=sr,
                mode=self.processing_mode,
                baseline_musical=baseline_scores,
                baseline_perceptual=None  # No perceptual baseline for synthetic samples
            )

            passed = post_check.passed
            quality_gate_decision = post_check.decision.value

            if post_check.achieved_perceptual:
                perceptual_metrics = {
                    "nisqa": post_check.achieved_perceptual.nisqa_mos,
                    "dnsmos": post_check.achieved_perceptual.dnsmos_ovrl,
                    "visqol": post_check.achieved_perceptual.visqol_mos_lqo,
                    "cdpam": post_check.achieved_perceptual.versa_score
                }

        return BenchmarkResult(
            filename=sample_meta["filename"],
            category=sample_meta["category"],
            baseline_scores=baseline_scores,
            achieved_scores=achieved_scores,
            improvements=improvements,
            degradations=degradations,
            processing_time_s=processing_time,
            quality_gate_decision=quality_gate_decision,
            perceptual_metrics=perceptual_metrics,
            passed=passed
        )

    def _generate_summary(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Generate summary statistics from results."""
        if not results:
            return BenchmarkSummary(
                total_samples=0,
                passed=0,
                failed=0,
                avg_improvement=0.0,
                avg_processing_time_s=0.0,
                category_results={},
                baseline_avg={},
                achieved_avg={}
            )

        total_samples = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total_samples - passed

        # Average improvement across all goals
        all_improvements: list[Any] = []
        for r in results:
            all_improvements.extend(r.improvements.values())

        avg_improvement = np.mean(all_improvements) if all_improvements else 0.0

        # Average processing time
        avg_processing_time = np.mean([r.processing_time_s for r in results])

        # Category-wise results
        category_results = {}
        categories = set(r.category for r in results)

        for category in categories:
            cat_results = [r for r in results if r.category == category]
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_improvements: list[Any] = []

            for r in cat_results:
                cat_improvements.extend(r.improvements.values())

            category_results[category] = {
                "total": len(cat_results),
                "passed": cat_passed,
                "pass_rate": cat_passed / len(cat_results) if cat_results else 0.0,
                "avg_improvement": np.mean(cat_improvements) if cat_improvements else 0.0
            }

        # Average baseline and achieved scores
        baseline_avg = {}
        achieved_avg = {}

        # Get all goal names
        all_goals: set[Any] = set()
        for r in results:
            all_goals.update(r.baseline_scores.keys())
            all_goals.update(r.achieved_scores.keys())

        for goal in all_goals:
            baseline_values = [r.baseline_scores.get(goal, 0.0) for r in results]
            achieved_values = [r.achieved_scores.get(goal, 0.0) for r in results]

            baseline_avg[goal] = np.mean(baseline_values)
            achieved_avg[goal] = np.mean(achieved_values)

        return BenchmarkSummary(
            total_samples=total_samples,
            passed=passed,
            failed=failed,
            avg_improvement=avg_improvement,
            avg_processing_time_s=avg_processing_time,  # type: ignore[arg-type]
            category_results=category_results,
            baseline_avg=baseline_avg,  # type: ignore[arg-type]
            achieved_avg=achieved_avg  # type: ignore[arg-type]
        )

    def export_report(
        self,
        results: List[BenchmarkResult],
        summary: BenchmarkSummary,
        output_path: Path
    ) -> None:
        """
        Export benchmark report to JSON.

        Args:
            results: List of benchmark results
            summary: Benchmark summary
            output_path: Path to output JSON file
        """
        report = {
            "summary": {
                "total_samples": summary.total_samples,
                "passed": summary.passed,
                "failed": summary.failed,
                "pass_rate": summary.passed / summary.total_samples if summary.total_samples else 0.0,
                "avg_improvement": summary.avg_improvement,
                "avg_processing_time_s": summary.avg_processing_time_s,
                "category_results": summary.category_results,
                "baseline_avg": summary.baseline_avg,
                "achieved_avg": summary.achieved_avg,
                "timestamp": summary.timestamp
            },
            "results": [
                {
                    "filename": r.filename,
                    "category": r.category,
                    "baseline_scores": r.baseline_scores,
                    "achieved_scores": r.achieved_scores,
                    "improvements": r.improvements,
                    "degradations": r.degradations,
                    "processing_time_s": r.processing_time_s,
                    "quality_gate_decision": r.quality_gate_decision,
                    "perceptual_metrics": r.perceptual_metrics,
                    "passed": r.passed,
                    "timestamp": r.timestamp
                }
                for r in results
            ],
            "configuration": {
                "processing_mode": self.processing_mode.value,
                "enable_perceptual_metrics": self.enable_perceptual_metrics,
                "enable_quality_gates": self.enable_quality_gates,
                "golden_samples_dir": str(self.golden_samples_dir)
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("✓ Benchmark report exported: %s", output_path)

    def print_summary(self, summary: BenchmarkSummary) -> None:
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("GOLDEN SAMPLE BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"\nTotal Samples: {summary.total_samples}")
        print(f"Passed: {summary.passed} ({summary.passed / summary.total_samples * 100:.1f}%)")
        print(f"Failed: {summary.failed} ({summary.failed / summary.total_samples * 100:.1f}%)")
        print(f"Avg Improvement: {summary.avg_improvement:+.3f}")
        print(f"Avg Processing Time: {summary.avg_processing_time_s:.2f}s")

        print("\nCategory Results:")
        for category, stats in summary.category_results.items():
            print(f"  {category}:")
            print(f"    Total: {stats['total']}")
            print(f"    Passed: {stats['passed']} ({stats['pass_rate'] * 100:.1f}%)")
            print(f"    Avg Improvement: {stats['avg_improvement']:+.3f}")

        print("\nMusical Goals (Baseline → Achieved):")
        for goal in summary.baseline_avg.keys():
            baseline = summary.baseline_avg[goal]
            achieved = summary.achieved_avg[goal]
            delta = achieved - baseline
            print(f"  {goal}: {baseline:.3f} → {achieved:.3f} ({delta:+.3f})")

        print("\n" + "=" * 60)


def main():
    """Run golden sample benchmark."""
    parser = argparse.ArgumentParser(description="Run golden sample benchmark")
    parser.add_argument(
        "--golden-samples",
        type=str,
        default="golden_samples",
        help="Golden samples directory (default: golden_samples)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="STUDIO_2026",
        choices=["RESTORATION", "STUDIO_2026", "FORENSIC"],
        help="Processing mode (default: STUDIO_2026)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Categories to benchmark (default: all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_report.json",
        help="Output report JSON (default: benchmark_report.json)"
    )
    parser.add_argument(
        "--no-perceptual",
        action="store_true",
        help="Disable perceptual metrics (NISQA, DNSMOS, etc.)"
    )
    parser.add_argument(
        "--no-quality-gates",
        action="store_true",
        help="Disable quality gates validation"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Aurik-Restaurierungs-Pipeline vor Benchmarking anwenden (denker.restauriere)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Parse processing mode
    mode_map = {
        "RESTORATION": ProcessingMode.RESTORATION,
        "STUDIO_2026": ProcessingMode.STUDIO_2026,
        "FORENSIC": ProcessingMode.FORENSIC  # type: ignore[attr-defined]
    }
    processing_mode = mode_map[args.mode]

    # Initialize runner
    runner = GoldenSampleBenchmarkRunner(
        golden_samples_dir=Path(args.golden_samples),
        processing_mode=processing_mode,
        enable_perceptual_metrics=not args.no_perceptual,
        enable_quality_gates=not args.no_quality_gates
    )

    # Processing-Funktion: echte Aurik-Restaurierung (optional via --process)
    processing_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
    if args.process:
        try:
            _restauriere = getattr(importlib.import_module("denker"), "restauriere")

            def _aurik_processing_fn(audio: np.ndarray, sr: int) -> np.ndarray:
                """Aurik-Restaurierung als Benchmark-Processing-Funktion."""
                try:
                    result = _restauriere(audio, sr=sr)
                    return result.audio  # type: ignore[no-any-return]
                except Exception as exc:
                    logging.warning("Restaurierung fehlgeschlagen (%s) – Passthrough", exc)
                    return audio

            processing_fn = _aurik_processing_fn
        except ImportError:
            logging.warning("denker-Modul nicht verfügbar – Benchmark ohne Processing (Baseline)")

    # Run benchmark
    results, summary = runner.run_benchmark(
        categories=args.categories,
        max_samples=args.max_samples,
        processing_function=processing_fn,
    )

    # Print summary
    runner.print_summary(summary)

    # Export report
    runner.export_report(results, summary, Path(args.output))

    print(f"\n✓ Report saved: {args.output}")


if __name__ == "__main__":
    main()
