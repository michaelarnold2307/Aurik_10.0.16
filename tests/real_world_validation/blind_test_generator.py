"""
Blind Test Generator - A/B/X Comparison for Subjective Validation

Generates randomized blind test files for human evaluation:
- A/B Testing: Compare two versions
- A/B/X Testing: Identify which of A or B matches X
- Rating Tests: Score individual files

Usage:
    python blind_test_generator.py --input test_library/ --output blind_tests/
    python blind_test_generator.py --mode abx --baseline unprocessed/ --test aurik/
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlindTestGenerator:
    """Generate blind test files for subjective evaluation."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_protocol: list[Any] = []

    def generate_ab_tests(self, baseline_dir: Path, test_dir: Path, count: int = 10):
        """
        Generate A/B comparison tests.

        Randomly select pairs and randomize A/B order.
        Evaluators rate which sounds better.

        Args:
            baseline_dir: Directory with unprocessed files
            test_dir: Directory with AURIK-processed files
            count: Number of test pairs to generate
        """
        logger.info(f"Generating {count} A/B comparison tests...")

        # Find matching files
        baseline_files = list(baseline_dir.rglob("*.wav"))
        random.shuffle(baseline_files)

        ab_dir = self.output_dir / "ab_tests"
        ab_dir.mkdir(exist_ok=True)

        generated = 0

        for baseline_file in baseline_files:
            if generated >= count:
                break

            # Find corresponding test file
            relative_path = baseline_file.relative_to(baseline_dir)
            test_file = test_dir / relative_path

            if not test_file.exists():
                continue

            # Randomize A/B order
            if random.random() < 0.5:
                a_file, b_file = baseline_file, test_file
                a_label, b_label = "baseline", "test"
            else:
                a_file, b_file = test_file, baseline_file
                a_label, b_label = "test", "baseline"

            # Create test pair
            test_id = f"ab_{generated + 1:03d}"

            # Copy files with neutral names
            a_out = ab_dir / f"{test_id}_A.wav"
            b_out = ab_dir / f"{test_id}_B.wav"

            self._copy_audio(a_file, a_out)
            self._copy_audio(b_file, b_out)

            # Store protocol
            self.test_protocol.append(
                {
                    "test_id": test_id,
                    "type": "ab",
                    "source_file": baseline_file.name,
                    "a_is": a_label,
                    "b_is": b_label,
                    "files": {
                        "A": str(a_out.relative_to(self.output_dir)),
                        "B": str(b_out.relative_to(self.output_dir)),
                    },
                }
            )

            generated += 1
            logger.info(f"Generated A/B test {test_id}")

        logger.info(f"✓ Generated {generated} A/B tests")
        return generated

    def generate_abx_tests(self, baseline_dir: Path, test_dir: Path, count: int = 10):
        """
        Generate A/B/X comparison tests.

        X is identical to either A or B, evaluator must identify which.
        This tests whether listeners can reliably distinguish processing.

        Args:
            baseline_dir: Directory with unprocessed files
            test_dir: Directory with AURIK-processed files
            count: Number of test triples to generate
        """
        logger.info(f"Generating {count} A/B/X comparison tests...")

        # Find matching files
        baseline_files = list(baseline_dir.rglob("*.wav"))
        random.shuffle(baseline_files)

        abx_dir = self.output_dir / "abx_tests"
        abx_dir.mkdir(exist_ok=True)

        generated = 0

        for baseline_file in baseline_files:
            if generated >= count:
                break

            # Find corresponding test file
            relative_path = baseline_file.relative_to(baseline_dir)
            test_file = test_dir / relative_path

            if not test_file.exists():
                continue

            # Randomize A/B order
            if random.random() < 0.5:
                a_file, b_file = baseline_file, test_file
                a_label, b_label = "baseline", "test"
            else:
                a_file, b_file = test_file, baseline_file
                a_label, b_label = "test", "baseline"

            # X is randomly A or B
            x_is_a = random.random() < 0.5
            x_file = a_file if x_is_a else b_file
            x_matches = "A" if x_is_a else "B"

            # Create test triple
            test_id = f"abx_{generated + 1:03d}"

            # Copy files with neutral names
            a_out = abx_dir / f"{test_id}_A.wav"
            b_out = abx_dir / f"{test_id}_B.wav"
            x_out = abx_dir / f"{test_id}_X.wav"

            self._copy_audio(a_file, a_out)
            self._copy_audio(b_file, b_out)
            self._copy_audio(x_file, x_out)

            # Store protocol
            self.test_protocol.append(
                {
                    "test_id": test_id,
                    "type": "abx",
                    "source_file": baseline_file.name,
                    "a_is": a_label,
                    "b_is": b_label,
                    "x_matches": x_matches,
                    "files": {
                        "A": str(a_out.relative_to(self.output_dir)),
                        "B": str(b_out.relative_to(self.output_dir)),
                        "X": str(x_out.relative_to(self.output_dir)),
                    },
                }
            )

            generated += 1
            logger.info(f"Generated A/B/X test {test_id}")

        logger.info(f"✓ Generated {generated} A/B/X tests")
        return generated

    def generate_rating_tests(self, test_dir: Path, count: int = 20):
        """
        Generate rating tests for individual files.

        Evaluators rate files on 1-5 scale for:
        - Overall quality
        - Naturalness (no artifacts)
        - Clarity
        - Preservation of original character

        Args:
            test_dir: Directory with files to rate
            count: Number of files to include
        """
        logger.info(f"Generating {count} rating tests...")

        # Find files
        test_files = list(test_dir.rglob("*.wav"))
        random.shuffle(test_files)

        rating_dir = self.output_dir / "rating_tests"
        rating_dir.mkdir(exist_ok=True)

        generated = 0

        for test_file in test_files[:count]:
            test_id = f"rating_{generated + 1:03d}"

            # Copy file with neutral name
            out_file = rating_dir / f"{test_id}.wav"
            self._copy_audio(test_file, out_file)

            # Store protocol
            self.test_protocol.append(
                {
                    "test_id": test_id,
                    "type": "rating",
                    "source_file": test_file.name,
                    "file": str(out_file.relative_to(self.output_dir)),
                    "rating_criteria": ["overall_quality", "naturalness", "clarity", "character_preservation"],
                }
            )

            generated += 1

        logger.info(f"✓ Generated {generated} rating tests")
        return generated

    def _copy_audio(self, src: Path, dst: Path):
        """Copy audio file, ensuring consistent format."""
        audio, sr = librosa.load(src, sr=44100, mono=False)
        sf.write(dst, audio.T if audio.ndim == 2 else audio, sr)

    def save_protocol(self):
        """Save test protocol (answers) to JSON."""
        protocol_file = self.output_dir / "test_protocol.json"

        protocol = {
            "generation_date": "2026-02-09",
            "total_tests": len(self.test_protocol),
            "test_types": {
                "ab": len([t for t in self.test_protocol if t["type"] == "ab"]),
                "abx": len([t for t in self.test_protocol if t["type"] == "abx"]),
                "rating": len([t for t in self.test_protocol if t["type"] == "rating"]),
            },
            "tests": self.test_protocol,
        }

        with open(protocol_file, "w") as f:
            json.dump(protocol, f, indent=2)

        logger.info(f"✓ Saved test protocol to {protocol_file}")

        # Also create instructions for evaluators
        self._create_instructions()

    def _create_instructions(self):
        """Create instruction document for evaluators."""
        instructions = """
# Blind Test Instructions

Thank you for participating in this blind listening test!

## Test Types

### A/B Comparison Tests (ab_tests/)
- Listen to file A, then file B
- Which sounds better overall? (A or B)
- Rate your confidence (1-5, where 5 = very confident)

### A/B/X Identification Tests (abx_tests/)
- Listen to file A, then file B, then file X
- Does X match A or B? (A or B)
- X is identical to either A or B (not a third option)

### Rating Tests (rating_tests/)
- Listen to each file
- Rate on scale of 1-5 for each criterion:
  - Overall Quality (1=poor, 5=excellent)
  - Naturalness (1=many artifacts, 5=completely natural)
  - Clarity (1=muddy/unclear, 5=crystal clear)
  - Character Preservation (1=character lost, 5=perfectly preserved)

## Tips

1. Use good headphones or studio monitors in a quiet environment
2. Take breaks to avoid ear fatigue
3. You can listen to each file multiple times
4. Be honest - there are no "wrong" answers
5. Don't overthink it - trust your ears

## Recording Your Results

Fill out the results form (provided separately) with:
- Test ID (e.g., "ab_001")
- Your answer (A, B, or rating values)
- Confidence level (for A/B tests)
- Any notes or observations

## Important

- Do NOT look at the test_protocol.json file (it contains answers)
- Tests are randomized - there's no pattern to discover
- Take your time - quality over speed

Thank you for your participation!
"""

        instructions_file = self.output_dir / "INSTRUCTIONS.md"
        with open(instructions_file, "w") as f:
            f.write(instructions)

        logger.info(f"✓ Created instructions file: {instructions_file}")

        # Create results template
        self._create_results_template()

    def _create_results_template(self):
        """Create template for recording results."""
        results = {
            "evaluator_name": "YOUR_NAME",
            "evaluation_date": "YYYY-MM-DD",
            "listening_environment": "headphones/monitors/speakers",
            "ab_results": [],
            "abx_results": [],
            "rating_results": [],
        }

        # Add placeholders for each test
        for test in self.test_protocol:
            if test["type"] == "ab":
                results["ab_results"].append(  # type: ignore[attr-defined]
                    {"test_id": test["test_id"], "preference": "A or B", "confidence": "1-5", "notes": ""}
                )
            elif test["type"] == "abx":
                results["abx_results"].append({"test_id": test["test_id"], "x_matches": "A or B", "notes": ""})  # type: ignore[attr-defined]
            elif test["type"] == "rating":
                results["rating_results"].append(  # type: ignore[attr-defined]
                    {
                        "test_id": test["test_id"],
                        "overall_quality": "1-5",
                        "naturalness": "1-5",
                        "clarity": "1-5",
                        "character_preservation": "1-5",
                        "notes": "",
                    }
                )

        template_file = self.output_dir / "results_template.json"
        with open(template_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Created results template: {template_file}")


def main():
    parser = argparse.ArgumentParser(description="Blind Test Generator")
    parser.add_argument("--output", type=str, required=True, help="Output directory for blind tests")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline (unprocessed) directory")
    parser.add_argument("--test", type=str, required=True, help="Test (processed) directory")
    parser.add_argument("--ab-count", type=int, default=10, help="Number of A/B tests to generate")
    parser.add_argument("--abx-count", type=int, default=10, help="Number of A/B/X tests to generate")
    parser.add_argument("--rating-count", type=int, default=20, help="Number of rating tests to generate")

    args = parser.parse_args()

    generator = BlindTestGenerator(output_dir=Path(args.output))

    # Generate all test types
    generator.generate_ab_tests(Path(args.baseline), Path(args.test), count=args.ab_count)

    generator.generate_abx_tests(Path(args.baseline), Path(args.test), count=args.abx_count)

    generator.generate_rating_tests(Path(args.test), count=args.rating_count)

    # Save protocol and instructions
    generator.save_protocol()

    logger.info("✓ Blind test generation complete!")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Total tests: {len(generator.test_protocol)}")


if __name__ == "__main__":
    main()
