"""
AURIK v8 Musical Goals Empirical Calibration Script
====================================================

Calibrates musical goals thresholds using golden samples.

Process:
1. Collect 30+ golden samples (high-quality reference audio)
   - 10 vinyl samples
   - 10 tape samples
   - 10 digital samples
2. Measure all 7 musical goals for each sample
3. Compute 5th percentile for each goal (95% confidence)
4. Set thresholds = 5th percentile - safety margin
5. Generate calibration report

Quelle: Finalisierungs_Roadmap.md - Component 0.7
Autor: AI Team
Datum: 8. Februar 2026

Usage:
    python scripts/calibrate_musical_goals.py \
        --golden-samples-dir /path/to/golden_samples \
        --output-report calibration_report.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_golden_samples(directory: Path, medium_filter: str = None) -> list[tuple[np.ndarray, int, str, str]]:  # type: ignore[assignment]
    """
    Load golden samples from directory.

    Expected directory structure:
    golden_samples/
        vinyl/
            sample1.wav
            sample2.wav
        tape/
            sample1.wav
            sample2.wav
        digital/
            sample1.wav
            sample2.wav

    Args:
        directory: Path to golden samples directory
        medium_filter: Optional medium filter ('vinyl', 'tape', 'digital')

    Returns:
        List of (audio, sr, filename, medium)
    """
    samples = []

    for medium_dir in directory.iterdir():
        if not medium_dir.is_dir():
            continue

        medium = medium_dir.name.lower()

        if medium_filter and medium != medium_filter:
            continue

        logger.info(f"lade samples from medium: {medium}")

        for audio_file in medium_dir.glob("*.wav"):
            try:
                audio, sr = sf.read(str(audio_file))

                # Convert stereo to mono if needed
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)

                samples.append((audio, sr, audio_file.name, medium))
                logger.info(f"  geladen: {audio_file.name} ({len(audio) / sr:.2f}s)")

            except Exception as e:
                logger.error(f"  konnte nicht laden {audio_file.name}: {e}")

    logger.info(f"Total samples geladen: {len(samples)}")
    return samples


def measure_all_samples(samples: list[tuple[np.ndarray, int, str, str]]) -> list[dict]:
    """
    Measure musical goals for all samples.

    Args:
        samples: List of (audio, sr, filename, medium)

    Returns:
        List of measurement results
    """
    from backend.core.musical_goals import MusicalGoalsChecker

    checker = MusicalGoalsChecker()
    results = []

    logger.info("Measuring musical goals for all samples...")

    for i, (audio, sr, filename, medium) in enumerate(samples, 1):
        logger.info(f"[{i}/{len(samples)}] Measuring {filename}...")

        try:
            goals = checker.measure_all(audio, sr)

            results.append(
                {"filename": filename, "medium": medium, "sr": sr, "duration": len(audio) / sr, "musical_goals": goals}
            )

            # Log scores
            for goal_name, score in goals.items():
                logger.debug(f"  {goal_name}: {score:.3f}")

        except Exception as e:
            logger.error(f"  konnte nicht measure {filename}: {e}")

    logger.info(f"Measurement vollstaendig: {len(results)} samples")
    return results


def compute_calibrated_thresholds(
    results: list[dict], percentile: float = 5.0, safety_margin: float = 0.02
) -> dict[str, dict]:
    """
    Compute calibrated thresholds using percentile method.

    Args:
        results: List of measurement results
        percentile: Percentile to use (default: 5.0 = 5th percentile)
        safety_margin: Safety margin to subtract from percentile (default: 0.02)

    Returns:
        Dict with calibrated thresholds per medium and overall
    """
    logger.info(f"berechne kalibriert thresholds (percentile={percentile}, margin={safety_margin})...")

    # Group by medium
    results_by_medium: dict[Any, Any] = {}
    for result in results:
        medium = result["medium"]
        if medium not in results_by_medium:
            results_by_medium[medium] = []
        results_by_medium[medium].append(result)

    # Compute thresholds per medium
    calibrated = {}

    for medium, medium_results in results_by_medium.items():
        logger.info(f"  Calibrating for medium: {medium} ({len(medium_results)} samples)")

        # Collect scores for each goal
        goal_scores = {}
        for goal_name in [
            "bass_kraft",
            "brillanz",
            "waerme",
            "natuerlichkeit",
            "authentizitaet",
            "emotionalitaet",
            "transparenz",
        ]:
            scores = [r["musical_goals"][goal_name] for r in medium_results]
            goal_scores[goal_name] = scores

        # Compute percentile thresholds
        medium_thresholds = {}
        for goal_name, scores in goal_scores.items():
            percentile_value = np.percentile(scores, percentile)
            threshold = max(0.0, percentile_value - safety_margin)
            medium_thresholds[goal_name] = {
                "threshold": round(threshold, 3),
                "percentile_value": round(percentile_value, 3),
                "min": round(np.min(scores), 3),
                "max": round(np.max(scores), 3),
                "mean": round(np.mean(scores), 3),
                "std": round(np.std(scores), 3),
                "n_samples": len(scores),
            }
            logger.info(
                f"    {goal_name:20s}: Schwelle={threshold:.3f} "
                f"(percentile={percentile_value:.3f}, mean={np.mean(scores):.3f})"
            )

        calibrated[medium] = medium_thresholds

    # Compute overall thresholds (all mediums combined)
    logger.info("  Calibrating overall thresholds (all mediums)")
    overall_goal_scores = {}
    for goal_name in [
        "bass_kraft",
        "brillanz",
        "waerme",
        "natuerlichkeit",
        "authentizitaet",
        "emotionalitaet",
        "transparenz",
    ]:
        overall_goal_scores[goal_name] = [r["musical_goals"][goal_name] for r in results]

    overall_thresholds = {}
    for goal_name, scores in overall_goal_scores.items():
        percentile_value = np.percentile(scores, percentile)
        threshold = max(0.0, percentile_value - safety_margin)
        overall_thresholds[goal_name] = {
            "threshold": round(threshold, 3),
            "percentile_value": round(percentile_value, 3),
            "min": round(np.min(scores), 3),
            "max": round(np.max(scores), 3),
            "mean": round(np.mean(scores), 3),
            "std": round(np.std(scores), 3),
            "n_samples": len(scores),
        }
        logger.info(
            f"    {goal_name:20s}: Schwelle={threshold:.3f} "
            f"(percentile={percentile_value:.3f}, mean={np.mean(scores):.3f})"
        )

    calibrated["overall"] = overall_thresholds

    return calibrated


def generate_calibration_report(calibrated_thresholds: dict, results: list[dict], output_file: Path):
    """
    Generate calibration report in YAML format.

    Args:
        calibrated_thresholds: Calibrated thresholds
        results: Measurement results
        output_file: Output file path
    """
    logger.info(f"Generating Kalibrierung report: {output_file}")

    report = {
        "calibration_info": {
            "date": str(np.datetime64("now")),
            "n_samples": len(results),
            "mediums": list({r["medium"] for r in results}),
            "percentile": 5.0,
            "safety_margin": 0.02,
            "method": "5th percentile - safety margin",
        },
        "calibrated_thresholds": calibrated_thresholds,
        "samples": [
            {
                "filename": r["filename"],
                "medium": r["medium"],
                "duration": round(r["duration"], 2),
                "musical_goals": {k: round(v, 3) for k, v in r["musical_goals"].items()},
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Kalibrierung report gespeichert: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(results)}")
    print(f"Mediums: {', '.join(report['calibration_info']['mediums'])}")  # type: ignore[index]
    print(f"Method: {report['calibration_info']['method']}")  # type: ignore[index]
    print("\nCALIBRATED THRESHOLDS (Overall):")
    for goal_name, values in calibrated_thresholds["overall"].items():
        print(f"  {goal_name:20s}: {values['threshold']:.3f} (mean={values['mean']:.3f}, std={values['std']:.3f})")
    print("=" * 70)
    print(f"\nFull report saved to: {output_file}")


def main():
    """Main calibration function."""
    parser = argparse.ArgumentParser(description="Calibrate musical goals thresholds using golden samples")
    parser.add_argument("--golden-samples-dir", type=Path, required=True, help="Path to golden samples directory")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("calibration_report.yaml"),
        help="Output calibration report file (default: calibration_report.yaml)",
    )
    parser.add_argument(
        "--medium-filter", type=str, choices=["vinyl", "tape", "digital"], help="Filter by medium (optional)"
    )
    parser.add_argument(
        "--percentile", type=float, default=5.0, help="Percentile for threshold calibration (default: 5.0)"
    )
    parser.add_argument(
        "--safety-margin", type=float, default=0.02, help="Safety margin to subtract from percentile (default: 0.02)"
    )

    args = parser.parse_args()

    # Validate directory
    if not args.golden_samples_dir.exists():
        logger.error(f"Golden samples directory not found: {args.golden_samples_dir}")
        return 1

    # Load golden samples
    samples = load_golden_samples(args.golden_samples_dir, medium_filter=args.medium_filter)

    if len(samples) < 10:
        logger.warning(
            f"Only {len(samples)} samples found. "
            f"Recommend at least 30 samples (10 per medium) for accurate Kalibrierung."
        )

    # Measure all samples
    results = measure_all_samples(samples)

    if len(results) < 5:
        logger.error("Too few valid measurements. Aborting Kalibrierung.")
        return 1

    # Compute calibrated thresholds
    calibrated_thresholds = compute_calibrated_thresholds(
        results, percentile=args.percentile, safety_margin=args.safety_margin
    )

    # Generate report
    generate_calibration_report(calibrated_thresholds, results, args.output_report)

    logger.info("Kalibrierung vollstaendig!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
