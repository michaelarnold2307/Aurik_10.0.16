#!/usr/bin/env python3
"""
Competitive Benchmarking Suite for AURIK v9 (Legacy Suite — eingeschränkte CI-Nutzung)
========================================================================================

⚠️  WICHTIGER HINWEIS — VERBOTENE METRIKEN (spec §3.1, §4.4):

    PESQ, SI-SDR, STOI, VISQOL (Speech-Mode), DNSMOS und NISQA sind für
    Musikqualitäts-Bewertung in Aurik 10.0.0 ABSOLUT VERBOTEN. Sie sind auf
    Telefonsprach-Korpora (16 kHz, 300–3400 Hz) trainiert und bewerten
    Musik systematisch falsch.

    Zulässige Musik-Metriken:
        MUSHRA (OQS), PQS-MOS, ViSQOL v3 (--audio Mode), PEAQ, FAD,
        Musical Goals (14 Ziele, §1.2).

    Das Modul enthält noch Methoden calculate_pesq(), calculate_si_sdr()
    etc. — diese sind als LEGACY markiert und dürfen in keiner
    Produktions- oder CI-Auswertung für Musik verwendet werden.
    Vgl. FORBIDDEN_METRICS-Konstante weiter unten.

Zweck (Legacy):
    Manueller Vergleich von Aurik 10.0.0 gegen:
    - iZotope RX 11 (Industry Standard)
    - Accusonus ERA (AI-Powered)
    - Waves Clarity (Professional Grade)

Für CI-Einsatz (norm. Competitive-Gate):
    tests/normative/test_competitive_ci_gate.py  ← korrekte Datei
    pytest tests/normative/test_competitive_ci_gate.py -m competitive

Erlaubte Metriken im CI (§8.2 Punkt 11):
    MUSHRA-Score aus run_benchmark() (AMRB, synthetische Signale)
    Keine LibriSpeech-Abhängigkeit!

Manuelle Nutzung (nicht für CI):
    python benchmarks/competitive/benchmark_suite.py --quick
    python benchmarks/competitive/benchmark_suite.py --report --output report.html
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verbotene Metriken (spec §3.1, §4.4) — dürfen NICHT für Musik-CI verwendet werden.
# test_competitive_ci_gate.py prüft, ob diese Konstante korrekt gesetzt ist.
# ---------------------------------------------------------------------------
FORBIDDEN_METRICS: frozenset[str] = frozenset(
    {
        "pesq",  # ITU-T P.862 — Telefonband 300–3400 Hz, Sprachkorpus
        "stoi",  # Sprachverständlichkeit 150–5000 Hz — sinnlos für Instrumentalmusik
        "si_sdr",  # Scale-Invariant SDR — für Stem-Separation-Sprachforschung entwickelt
        "visqol",  # ViSQOL im Speech-Default-Mode — nur --audio Mode zulässig
        "dnsmos",  # DNS-Challenge-Sprachkorpus (16 kHz) — bewertet Musik falsch
        "nisqa",  # Deep-CNN auf Sprachkorpora — keine Musik-Trainingsdaten
    }
)


class AudioQualityMetrics:
    """
    Objective audio quality metrics for benchmarking.
    """

    @staticmethod
    def calculate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        signal_power = np.mean(clean**2)
        noise_power = np.mean((clean - noisy) ** 2)

        if noise_power < 1e-10:
            return np.inf

        snr = 10 * np.log10(signal_power / noise_power)
        return snr  # type: ignore[no-any-return]

    @staticmethod
    def calculate_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
        """Legacy stub: SI-SDR is forbidden for music benchmarking (§4.4)."""
        logger.warning("SI-SDR is forbidden for music benchmarking (§4.4). Returning None.")
        return None

    @staticmethod
    def calculate_pesq(reference: np.ndarray, degraded: np.ndarray, sr: int = 16000) -> float | None:
        """
        Calculate PESQ score (requires pesq library).

        Note: Install with: pip install pesq
        """
        logger.warning("PESQ is forbidden for music benchmarking (§4.4). Returning None.")
        return None

    @staticmethod
    def calculate_stoi(clean: np.ndarray, processed: np.ndarray, sr: int = 16000) -> float | None:
        """
        Calculate STOI (Short-Time Objective Intelligibility).

        Note: Install with: pip install pystoi
        """
        logger.warning("STOI is forbidden for music benchmarking (§4.4). Returning None.")
        return None

    @staticmethod
    def calculate_all_metrics(reference: np.ndarray, processed: np.ndarray, sr: int = 16000) -> dict[str, float]:
        """Calculate all available metrics."""
        metrics = {}

        # Always available
        metrics["snr"] = AudioQualityMetrics.calculate_snr(reference, processed)
        # Forbidden speech/separation metrics are intentionally not computed (§4.4).

        return metrics


class PerformanceMetrics:
    """
    Performance benchmarking metrics.
    """

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_cpu = None
        self.start_memory = None

    def start(self):
        """Start performance measurement."""
        self.start_time = time.perf_counter()
        self.start_cpu = self.process.cpu_percent()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def stop(self, audio_duration: float) -> dict[str, float]:
        """Stop measurement and return metrics."""
        end_time = time.perf_counter()
        end_cpu = self.process.cpu_percent()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        start_time = self.start_time if self.start_time is not None else end_time
        start_cpu = self.start_cpu if self.start_cpu is not None else end_cpu
        start_memory = self.start_memory if self.start_memory is not None else end_memory

        processing_time = end_time - start_time
        real_time_factor = processing_time / audio_duration if audio_duration > 0 else np.inf

        return {
            "processing_time": processing_time,
            "audio_duration": audio_duration,
            "real_time_factor": real_time_factor,
            "cpu_usage": (start_cpu + end_cpu) / 2,
            "memory_usage": end_memory,
            "memory_delta": end_memory - start_memory,
        }


class CompetitiveBenchmark:
    """
    Benchmark AURIK against competitors.
    """

    def __init__(self, test_audio_dir: str, output_dir: str = "benchmarks/competitive/results"):
        self.test_audio_dir = Path(test_audio_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Any = {
            "aurik": [],
            "izotope_rx11": [],
            "accusonus_era": [],
            "waves_clarity": [],
        }

    def benchmark_aurik(self, audio_path: str, reference_path: str) -> dict:
        """Benchmark AURIK v8."""
        import soundfile as sf

        from cli.aurik_cli import process_audio

        logger.info(f"Benchmarking AURIK on {audio_path}")

        # Load audio
        audio, sr = sf.read(audio_path)
        reference, _ = sf.read(reference_path)

        # Start performance measurement
        perf = PerformanceMetrics()
        perf.start()

        # Process audio
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                tmp_out_path = tmp_out.name

            try:
                result = process_audio(audio_path, tmp_out_path, verbose=False, mode="Restoration")
                processed = np.asarray(result.audio, dtype=np.float32)  # type: ignore[attr-defined]
            finally:
                try:
                    Path(tmp_out_path).unlink(missing_ok=True)
                except Exception:
                    logger.warning("benchmark_suite.py::benchmark_aurik fallback", exc_info=True)
        except Exception as e:
            logger.error(f"AURIK processing failed: {e}")
            return None  # type: ignore[return-value]

        # Stop measurement
        audio_duration = len(audio) / sr
        perf_metrics = perf.stop(audio_duration)

        # Calculate audio quality metrics
        quality_metrics = AudioQualityMetrics.calculate_all_metrics(reference, processed, sr)

        return {
            "tool": "aurik",
            "audio_path": str(audio_path),
            "performance": perf_metrics,
            "quality": quality_metrics,
        }

    def benchmark_izotope(self, audio_path: str, reference_path: str) -> dict:
        """
        Benchmark iZotope RX 11.

        Note: Requires iZotope RX 11 installed with CLI access.
        This is a placeholder - actual integration requires iZotope license.
        """
        logger.info(f"Benchmarking iZotope RX 11 on {audio_path}")

        # Real: Open-Source-Benchmark als Proxy (Spec 15 §1.1)
        from benchmarks.competitive.open_source_benchmark import run as _oss_run

        logger.info("iZotope RX 11 requires license — using Open-Source-Benchmark proxy")
        try:
            results, summary = _oss_run(tools=["scipy_wiener", "librosa_hpss"], dur=1.0)
            return {
                "tool": "izotope_rx11",
                "audio_path": str(audio_path),
                "proxy": "open_source_benchmark",
                "performance": {"real_time_factor": summary.get("mean_rt_a", 0)},
                "quality": {"pqs_delta": summary.get("mean_delta", 0)},
                "summary": summary,
            }
        except Exception as e:
            return {
                "tool": "izotope_rx11",
                "audio_path": str(audio_path),
                "error": str(e),
            }

    def benchmark_accusonus(self, audio_path: str, reference_path: str) -> dict:
        """
        Benchmark Accusonus ERA.

        Note: Requires Accusonus ERA installed.
        This is a placeholder - actual integration requires Accusonus license.
        """
        logger.info(f"Benchmarking Accusonus ERA on {audio_path}")

        # Placeholder
        logger.warning("Accusonus ERA benchmarking not implemented (requires license)")

        return {
            "tool": "accusonus_era",
            "audio_path": str(audio_path),
            "performance": {"real_time_factor": None},
            "quality": {},
            "note": "Requires Accusonus ERA license",
        }

    def benchmark_waves(self, audio_path: str, reference_path: str) -> dict:
        """
        Benchmark Waves Clarity.

        Note: Requires Waves Clarity installed.
        This is a placeholder - actual integration requires Waves license.
        """
        logger.info(f"Benchmarking Waves Clarity on {audio_path}")

        # Placeholder
        logger.warning("Waves Clarity benchmarking not implemented (requires license)")

        return {
            "tool": "waves_clarity",
            "audio_path": str(audio_path),
            "performance": {"real_time_factor": None},
            "quality": {},
            "note": "Requires Waves Clarity license",
        }

    def run_full_benchmark(self, competitors: list[str] | None = None) -> dict:
        """Run benchmark on all test files."""
        if competitors is None:
            competitors = ["aurik"]
        test_files = list(self.test_audio_dir.glob("*.wav"))

        if len(test_files) == 0:
            logger.error(f"No test files found in {self.test_audio_dir}")
            return {}

        logger.info(f"Found {len(test_files)} test files")

        for test_file in test_files:
            # Assume reference is in 'reference' subdirectory
            reference_file = self.test_audio_dir / "reference" / test_file.name

            if not reference_file.exists():
                logger.warning(f"No reference file for {test_file.name}, skipping")
                continue

            # Benchmark each competitor
            if "aurik" in competitors:
                result = self.benchmark_aurik(str(test_file), str(reference_file))
                if result:
                    self.results["aurik"].append(result)

            if "izotope" in competitors:
                result = self.benchmark_izotope(str(test_file), str(reference_file))
                if result:
                    self.results["izotope_rx11"].append(result)

            if "accusonus" in competitors:
                result = self.benchmark_accusonus(str(test_file), str(reference_file))
                if result:
                    self.results["accusonus_era"].append(result)

            if "waves" in competitors:
                result = self.benchmark_waves(str(test_file), str(reference_file))
                if result:
                    self.results["waves_clarity"].append(result)

        return self.results  # type: ignore[no-any-return]

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON."""
        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    def generate_report(self) -> str:
        """Generate benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("AURIK v8 Competitive Benchmark Report")
        report.append("=" * 80)
        report.append("")

        for tool, results in self.results.items():
            if len(results) == 0:
                continue

            report.append(f"\n{tool.upper().replace('_', ' ')}")
            report.append("-" * 80)

            # Aggregate metrics
            rt_factors = [
                r["performance"].get("real_time_factor")
                for r in results
                if r["performance"].get("real_time_factor") is not None
            ]
            snrs = [r["quality"].get("snr") for r in results if r["quality"].get("snr") is not None]
            si_sdrs = [r["quality"].get("si_sdr") for r in results if r["quality"].get("si_sdr") is not None]

            if rt_factors:
                report.append(f"Average Real-Time Factor: {np.mean(rt_factors):.3f}× (lower is better)")
                report.append(f"  Min: {np.min(rt_factors):.3f}×, Max: {np.max(rt_factors):.3f}×")

            if snrs:
                report.append(f"Average SNR: {np.mean(snrs):.2f} dB (higher is better)")
                report.append(f"  Min: {np.min(snrs):.2f} dB, Max: {np.max(snrs):.2f} dB")

            if si_sdrs:
                report.append(f"Average SI-SDR: {np.mean(si_sdrs):.2f} dB (higher is better)")
                report.append(f"  Min: {np.min(si_sdrs):.2f} dB, Max: {np.max(si_sdrs):.2f} dB")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AURIK Competitive Benchmarking")
    parser.add_argument("--test-set", default="test_audio", help="Test audio directory")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (subset)")
    parser.add_argument(
        "--competitor",
        choices=["aurik", "izotope", "accusonus", "waves"],
        nargs="+",
        default=["aurik"],
        help="Competitors to benchmark",
    )
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--output", default="benchmark_results", help="Output file prefix")

    args = parser.parse_args()

    # Create benchmark
    benchmark = CompetitiveBenchmark(args.test_set)

    # Run benchmark
    logger.info("Starting competitive benchmark...")
    benchmark.run_full_benchmark(competitors=args.competitor)

    # Save results
    benchmark.save_results(f"{args.output}.json")

    # Generate report
    if args.report:
        report = benchmark.generate_report()
        print(report)

        report_file = Path("benchmarks/competitive/results") / f"{args.output}.txt"
        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
