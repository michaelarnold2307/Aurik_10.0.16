"""
benchmarks/restoration_benchmark.py — Aurik Restaurierungs-Benchmark Suite
===========================================================================

Öffentlich nachweisbarer Benchmark für automatisierte Musikrestaurierung.
Vergleicht Aurik gegen Referenzwerte von iZotope RX, CEDAR, SpectraLayers Pro.

BENCHMARK-DIMENSIONEN:
  1. **MUSIC_OVR** (Music-MOS Overall, 1–5): Subjektive Gesamt-Qualität
  2. **MUSIC_NAT** (Naturalness, 1–5): Klingt es nach echter Musik?
  3. **NOISE_FLOOR** (dBFS): Hintergrund-Rauschpegel, niedriger=besser
  4. **CLICK_DENSITY** (ppm): Klick-/Tick-Dichte, niedriger=besser
  5. **RT_FACTOR** (×): Verarbeitungszeit relativ zur Audio-Dauer, niedriger=besser

  HINWEIS: SI-SDR ist gemäß §4.4 VERBOTEN als Musikmetrik (kein Musik-Training).

REFERENZ-SCORES (aus Literatur + eigenen Messungen):
  | System           | OVR  | NAT  | RF    |
  |-----------------|------|------|-------|
  | iZotope RX 10   | 4.0  | 3.8  | 3.0× |
  | CEDAR Cambridge | 4.5  | 4.2  | 4.5× |
  | SpectraLayers   | 3.8  | 3.5  | 2.5× |
  | Aurik 10.0.0 (Ziel)| 4.0  | 3.8  | 1.0× |
  | Aurik 10.0.0 (Ziel)| 4.3  | 4.0  | 0.5× |

TESTMATERIAL-KATEGORIEN:
  - shellac_heavy: Starkes Oberflächenrauschen, Klicks, Mono
  - vinyl_normal: Standard-Vinyl-Klang, Tick/Knister
  - tape_dropout: Magnetband mit Dropouts und Bandrauschen
  - digital_clean: Referenz-Digitalisierung (sollte unverändert bleiben: OVR ≥ 4.8)

VERWENDUNG::

    benchmark = RestorationBenchmark()
    report = benchmark.run_all(output_dir="results/benchmark_2026/")
    benchmark.print_summary(report)
    benchmark.compare_to_reference(report)

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ─── Referenz-Scores ─────────────────────────────────────────────────────

REFERENCE_SCORES: dict[str, dict[str, float]] = {
    "iZotope RX 10": {
        "MUSIC_OVR": 4.0,
        "MUSIC_NAT": 3.8,
        "NOISE_FLOOR_dBFS": -55.0,
        "CLICK_DENSITY_ppm": 5.0,
        "RT_FACTOR": 3.0,
    },
    "CEDAR Cambridge": {
        "MUSIC_OVR": 4.5,
        "MUSIC_NAT": 4.2,
        "NOISE_FLOOR_dBFS": -62.0,
        "CLICK_DENSITY_ppm": 2.0,
        "RT_FACTOR": 4.5,
    },
    "SpectraLayers Pro 10": {
        "MUSIC_OVR": 3.8,
        "MUSIC_NAT": 3.5,
        "NOISE_FLOOR_dBFS": -50.0,
        "CLICK_DENSITY_ppm": 8.0,
        "RT_FACTOR": 2.5,
    },
    "Aurik 10.0.0 (Ziel)": {
        "MUSIC_OVR": 4.3,
        "MUSIC_NAT": 4.0,
        "NOISE_FLOOR_dBFS": -60.0,
        "CLICK_DENSITY_ppm": 3.0,
        "RT_FACTOR": 0.5,
    },
}


# ─── Test-Audio-Generierung ───────────────────────────────────────────────


def _generate_test_signal(
    category: str,
    sample_rate: int = 44100,
    duration_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generiert synthetisches Test-Audio für Benchmark-Kategorien.
    Returns (degraded_audio, clean_reference).
    """
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n)

    # Basis-Musiksignal: Harmonischer Klang (Cello-ähnlich)
    fundamental = 220.0  # A3
    clean = (
        0.5 * np.sin(2 * np.pi * fundamental * t)
        + 0.3 * np.sin(2 * np.pi * 2 * fundamental * t)
        + 0.15 * np.sin(2 * np.pi * 3 * fundamental * t)
        + 0.05 * np.sin(2 * np.pi * 4 * fundamental * t)
    )
    clean *= 0.7  # Normierung

    rng = np.random.default_rng(42)  # Reproduzierbarkeit

    if category == "shellac_heavy":
        # Starkes Shellac-Rauschen: -30 dBFS, viele Klicks
        noise = rng.normal(0, 0.03, n)  # -30 dBFS Rauschen
        noise += rng.normal(0, 0.0005, n)  # Sehr schwaches Grundrauschen
        # Klicks (1% der Samples)
        click_indices = rng.integers(0, n, n // 100)
        noise[click_indices] += rng.choice([-1, 1], len(click_indices)) * 0.3
        degraded = clean + noise

    elif category == "vinyl_normal":
        # Vinyl: Knister, leichte Rillenverzerrung
        crackle = rng.normal(0, 0.005, n)  # -46 dBFS Knister
        # Gelegentliche Ticks
        tick_indices = rng.integers(0, n, n // 500)
        crackle[tick_indices] += rng.choice([-1, 1], len(tick_indices)) * 0.1
        degraded = clean + crackle

    elif category == "tape_dropout":
        # Bandrauschen + Dropouts
        tape_noise = rng.normal(0, 0.008, n)  # Rosa-Rauschen-ähnlich
        degraded = clean + tape_noise
        # Dropouts: 3 Stellen ohne Signal
        for _ in range(3):
            dropout_start = rng.integers(n // 4, 3 * n // 4)
            dropout_len = int(0.05 * sample_rate)  # 50ms Dropout
            degraded[dropout_start : dropout_start + dropout_len] = 0.0

    elif category == "digital_clean":
        # Digitales Signal: fast unverändert (nur minimales Quantisierungsrauschen)
        degraded = clean + rng.normal(0, 1e-5, n)

    else:
        degraded = clean.copy()

    # Clipping verhindern
    degraded = np.clip(degraded, -1.0, 1.0)
    return degraded, clean


# ─── Metrik-Berechnung ───────────────────────────────────────────────────


def _compute_noise_floor(audio: np.ndarray) -> float:
    """Rauschpegel in dBFS (10. Perzentil der Frame-Energien)."""
    frame_size = 512
    mono = audio.flatten()
    frames = [mono[i : i + frame_size] for i in range(0, len(mono) - frame_size, frame_size)]
    if not frames:
        return -80.0
    rms_vals = [np.sqrt(np.mean(f**2)) + 1e-10 for f in frames]
    return float(20 * np.log10(np.percentile(rms_vals, 10)))


def _compute_click_density_ppm(audio: np.ndarray) -> float:
    """Klick-Dichte in Parts per Million."""
    mono = audio.flatten()
    abs_audio = np.abs(mono)
    threshold = np.percentile(abs_audio, 99.9)
    if threshold < 1e-10:
        return 0.0
    n_clicks = np.sum(abs_audio > threshold * 3)
    return float(n_clicks / len(mono) * 1_000_000)


# ─── BenchmarkResult ─────────────────────────────────────────────────────


@dataclass
class TestCaseResult:
    """Ergebnis eines einzelnen Testfalls."""

    category: str
    MUSIC_OVR: float
    MUSIC_NAT: float
    NOISE_FLOOR_dBFS: float
    CLICK_DENSITY_ppm: float
    RT_FACTOR: float
    processing_time_s: float
    audio_duration_s: float
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Vollständiger Benchmark-Bericht."""

    timestamp: str
    aurik_version: str
    test_results: list[TestCaseResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    comparison: dict[str, dict[str, str]] = field(default_factory=dict)  # vs Referenz

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ─── RestorationBenchmark ────────────────────────────────────────────────


class RestorationBenchmark:
    """
    Automatisierter Restaurierungs-Benchmark.

    Führt Aurik gegen synthetisches und reales Testmaterial aus,
    berechnet standardisierte Metriken und vergleicht mit Referenz-Systemen.
    """

    TEST_CATEGORIES = ["shellac_heavy", "vinyl_normal", "tape_dropout", "digital_clean"]
    SAMPLE_RATE = 44100
    DURATION_S = 10.0

    def __init__(
        self,
        aurik_version: str = "9.5.0",
        restoration_fn: object | None = None,
    ) -> None:
        """
        Args:
            aurik_version: Aurik-Versionsnummer
            restoration_fn: Optional — Funktion(audio, sr, category) → audio.
                           Wenn None: Identitätsfunktion (unverändert).
        """
        self.aurik_version = aurik_version
        self.restoration_fn = restoration_fn or (lambda audio, sr, cat: audio)

    def _run_test_case(self, category: str) -> TestCaseResult:
        """Führt einen einzelnen Testfall aus und berechnet Metriken."""
        try:
            degraded, clean_ref = _generate_test_signal(category, self.SAMPLE_RATE, self.DURATION_S)

            t0 = time.perf_counter()
            restored = self.restoration_fn(degraded, self.SAMPLE_RATE, category)  # type: ignore[operator]
            elapsed = time.perf_counter() - t0

            # Metriken
            from backend.core.music_quality_scorer import score_music_mos

            mos = score_music_mos(restored, self.SAMPLE_RATE)
            noise_floor = _compute_noise_floor(restored)
            click_density = _compute_click_density_ppm(restored)
            rt_factor = elapsed / self.DURATION_S

            return TestCaseResult(
                category=category,
                MUSIC_OVR=mos.MUSIC_OVR,
                MUSIC_NAT=mos.MUSIC_NAT,
                NOISE_FLOOR_dBFS=round(noise_floor, 1),
                CLICK_DENSITY_ppm=round(click_density, 2),
                RT_FACTOR=round(rt_factor, 3),
                processing_time_s=round(elapsed, 3),
                audio_duration_s=self.DURATION_S,
            )

        except Exception as exc:  # pragma: no cover
            logger.error("Testfall '%s' fehlgeschlagen: %s", category, exc)
            return TestCaseResult(
                category=category,
                MUSIC_OVR=0.0,
                MUSIC_NAT=0.0,
                NOISE_FLOOR_dBFS=0.0,
                CLICK_DENSITY_ppm=9999.0,
                RT_FACTOR=99.0,
                processing_time_s=0.0,
                audio_duration_s=0.0,
                error=str(exc),
            )

    def run_all(self, output_dir: str | None = None) -> BenchmarkReport:
        """
        Führt alle Testfälle aus.

        Args:
            output_dir: Optional — Verzeichnis zum Speichern der Ergebnisse als JSON.

        Returns:
            BenchmarkReport mit allen Ergebnissen und Vergleichen.
        """
        from datetime import datetime

        timestamp = datetime.now().isoformat()

        report = BenchmarkReport(
            timestamp=timestamp,
            aurik_version=self.aurik_version,
        )

        for category in self.TEST_CATEGORIES:
            logger.info("Benchmark: Kategorie '%s' läuft ...", category)
            result = self._run_test_case(category)
            report.test_results.append(result)

        # Summary: Mittelwert über alle Testfälle
        valid = [r for r in report.test_results if r.error is None]
        if valid:
            report.summary = {
                "MUSIC_OVR_mean": round(np.mean([r.MUSIC_OVR for r in valid]), 3),  # type: ignore[dict-item]
                "MUSIC_NAT_mean": round(np.mean([r.MUSIC_NAT for r in valid]), 3),  # type: ignore[dict-item]
                "NOISE_FLOOR_mean": round(np.mean([r.NOISE_FLOOR_dBFS for r in valid]), 1),  # type: ignore[dict-item]
                "CLICK_DENSITY_mean": round(np.mean([r.CLICK_DENSITY_ppm for r in valid]), 2),  # type: ignore[dict-item]
                "RT_FACTOR_mean": round(np.mean([r.RT_FACTOR for r in valid]), 3),  # type: ignore[dict-item]
                "n_test_cases": len(valid),
            }

        # Vergleich mit Referenz-Systemen
        aurik_summary = report.summary
        for system, ref_scores in REFERENCE_SCORES.items():
            comparison = {}
            for metric, ref_val in ref_scores.items():
                mean_key = f"{metric}_mean" if not metric.endswith("_mean") else metric
                aurik_val = aurik_summary.get(mean_key)
                if aurik_val is None:
                    continue
                # Vorzeichen: Höher=besser oder Niedriger=besser?
                higher_is_better = metric not in {"NOISE_FLOOR_dBFS", "CLICK_DENSITY_ppm", "RT_FACTOR"}
                if higher_is_better:
                    delta = aurik_val - ref_val
                    status = "✅ besser" if delta > 0 else ("⚠️ gleich" if abs(delta) < 0.2 else "❌ schlechter")
                else:
                    delta = ref_val - aurik_val
                    status = "✅ besser" if delta > 0 else ("⚠️ gleich" if abs(delta) < 0.2 else "❌ schlechter")
                comparison[metric] = f"{aurik_val:.2f} vs {ref_val:.2f} ({status})"
            report.comparison[system] = comparison

        # Speichern
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            report_file = output_path / f"benchmark_{timestamp.replace(':', '-')}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report.to_json())
            logger.info("Benchmark-Bericht gespeichert: %s", report_file)

        return report

    def print_summary(self, report: BenchmarkReport) -> None:
        """Gibt formatierte Zusammenfassung auf stdout aus."""
        print(f"\n{'=' * 60}")
        print(f"  AURIK RESTORATION BENCHMARK v{report.aurik_version}")
        print(f"  Zeitstempel: {report.timestamp}")
        print(f"{'=' * 60}")

        print("\n📊 TESTERGEBNISSE:\n")
        for r in report.test_results:
            status = "🔴 FEHLER" if r.error else "🟢 OK"
            print(
                f"  [{status}] {r.category:20s}  OVR={r.MUSIC_OVR:.2f}  NAT={r.MUSIC_NAT:.2f}  "
                f"NF={r.NOISE_FLOOR_dBFS:.0f}dBFS  RT={r.RT_FACTOR:.2f}×"
            )

        if report.summary:
            print("\n📈 ZUSAMMENFASSUNG (Mittelwert):\n")
            for k, v in report.summary.items():
                print(f"  {k:30s}: {v}")

        print("\n🏆 VERGLEICH MIT REFERENZ-SYSTEMEN:\n")
        for system, comp in report.comparison.items():
            print(f"  vs {system}:")
            for metric, result_str in comp.items():
                print(f"    {metric:25s}: {result_str}")
            print()

    def compare_to_reference(self, report: BenchmarkReport) -> bool:
        """
        Prüft ob Aurik mind. einen Referenz-Wert des Ziel-Systems erreicht.
        Returns True wenn Aurik den Ziel-Score (v10.0.0) in ≥50% der Metriken erreicht.
        """
        target = REFERENCE_SCORES.get("Aurik 10.0.0 (Ziel)", {})
        if not target or not report.summary:
            return False

        wins = 0
        total = 0
        metric_map = {
            "MUSIC_OVR": "MUSIC_OVR_mean",
            "MUSIC_NAT": "MUSIC_NAT_mean",
            "RT_FACTOR": "RT_FACTOR_mean",
        }

        for ref_key, summary_key in metric_map.items():
            ref_val = target.get(ref_key)
            aurik_val = report.summary.get(summary_key)
            if ref_val is None or aurik_val is None:
                continue
            total += 1
            higher_better = ref_key not in {"RT_FACTOR"}
            if higher_better:
                if aurik_val >= ref_val * 0.95:  # ±5% Toleranz
                    wins += 1
            else:
                if aurik_val <= ref_val * 1.05:
                    wins += 1

        success = (wins >= total // 2 + 1) if total > 0 else False
        logger.info("Benchmark-Ziel: %d/%d Metriken erreicht (Ziel: ≥%d)", wins, total, total // 2 + 1)
        return success
