#!/usr/bin/env python3
"""
§v10.800: Echter-Corpus-Benchmark — die ehrliche Wahrheitsmessung.

Misst die SOTA-Kette gegen ECHTE beschädigte Aufnahmen aus corpus/.
Jede Datei hat ein clean-Pendant. Metriken:

  1. SNR-Verbesserung (dB)  — wie viel Rauschen wurde entfernt?
  2. MSE-Reduktion (%)      — wie viel näher am Clean-Referenz?
  3. UTMOS-MOS-Delta        — klingt es nach der Restauration besser?
  4. Verdict pro Datei      — verbessert / verschlechtert / neutral

Ehrlichkeits-Regel: Keine Filterung, keine Rosinenpickerei.
Alle Ergebnisse werden berichtet — auch Verschlechterungen.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

SR = 48000
CHUNK_SEC = 4.0        # Erste 4s pro Datei (Benchmark-Geschwindigkeit)
FILES_PER_MEDIUM = 2   # Stichprobe: 2 Dateien pro Medium


@dataclass
class FileResult:
    medium: str
    file: str
    snr_before_db: float
    snr_after_db: float
    snr_improvement_db: float
    mse_before: float
    mse_after: float
    mos_damaged: float
    mos_restored: float
    mos_clean: float
    verdict: str          # improved / degraded / neutral
    processing_time: float


@dataclass
class BenchmarkReport:
    results: list[FileResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def improved(self) -> int:
        return sum(1 for r in self.results if r.verdict == "improved")

    @property
    def degraded(self) -> int:
        return sum(1 for r in self.results if r.verdict == "degraded")

    @property
    def avg_snr_improvement(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.snr_improvement_db for r in self.results]))


def _pair_files(medium: str) -> list[tuple[Path, Path]]:
    """Paart damaged-Dateien mit ihren clean-Pendants."""
    clean_dir = _PROJECT / "corpus" / medium / "clean"
    damaged_dir = _PROJECT / "corpus" / medium / "damaged"
    pairs = []
    for damaged in sorted(damaged_dir.glob("*.wav"))[:FILES_PER_MEDIUM]:
        # clean-Pendant: gleicher Name ohne Defekt-Suffix
        stem = damaged.stem
        # z.B. cassette_hiphop_1980s_hiss → cassette_hiphop_1980s_clean
        base = stem.rsplit("_", 1)[0] if "_" in stem else stem
        clean = clean_dir / f"{base}_clean.wav"
        if clean.exists():
            pairs.append((damaged, clean))
    return pairs


def _load_chunk(path: Path, sr: int = SR) -> np.ndarray:
    audio, file_sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = min(int(CHUNK_SEC * file_sr), len(audio))
    return audio[:n].astype(np.float32)


def _snr_db(reference: np.ndarray, signal: np.ndarray) -> float:
    noise = signal - reference
    ref_power = float(np.mean(reference ** 2)) + 1e-10
    noise_power = float(np.mean(noise ** 2)) + 1e-10
    return float(10 * np.log10(ref_power / noise_power))


def run_benchmark() -> BenchmarkReport:
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline
    from backend.core.coordinated_repair import RepairPlanner, CoordinatedRepair
    from backend.core.perceptual_closed_loop import PerceptualClosedLoop

    consensus = DefectConsensusPipeline()
    loop = PerceptualClosedLoop()

    report = BenchmarkReport()
    t0 = time.time()

    for medium in ["cassette", "digital", "reel_tape", "shellac", "tape", "vinyl"]:
        pairs = _pair_files(medium)
        if not pairs:
            print(f"\n{medium}: keine Paare gefunden — übersprungen")
            continue
        print(f"\n{'=' * 60}")
        print(f"Medium: {medium} ({len(pairs)} Paare)")
        print(f"{'=' * 60}")

        for damaged_path, clean_path in pairs:
            damaged = _load_chunk(damaged_path)
            clean = _load_chunk(clean_path)
            min_len = min(len(damaged), len(clean))
            damaged, clean = damaged[:min_len], clean[:min_len]

            # Metriken VOR der Restauration
            snr_before = _snr_db(clean, damaged)
            mse_before = float(np.mean((damaged - clean) ** 2))
            mos_damaged = loop.estimate_mos(damaged, SR)
            mos_clean = loop.estimate_mos(clean, SR)

            # SOTA-Kette
            t_step = time.time()
            try:
                manifest = consensus.analyze(damaged, SR)
                planner = RepairPlanner()
                plan = planner.plan(manifest, len(damaged))
                executor = CoordinatedRepair()
                restored, _ = executor.execute(damaged, plan, manifest, SR)
                restored = np.asarray(restored)
                if restored.ndim > 1:
                    restored = restored.mean(axis=0)
                restored = restored[:min_len]
            except Exception as exc:
                print(f"  ❌ {damaged_path.name}: Kette fehlgeschlagen ({exc})")
                continue
            proc_time = time.time() - t_step

            # Metriken NACH der Restauration
            snr_after = _snr_db(clean, restored)
            mse_after = float(np.mean((restored - clean) ** 2))
            mos_restored = loop.estimate_mos(restored, SR)

            snr_improvement = snr_after - snr_before

            # Verdict
            if snr_improvement > 0.5:
                verdict = "improved"
            elif snr_improvement < -0.5:
                verdict = "degraded"
            else:
                verdict = "neutral"

            result = FileResult(
                medium=medium,
                file=damaged_path.name,
                snr_before_db=snr_before,
                snr_after_db=snr_after,
                snr_improvement_db=snr_improvement,
                mse_before=mse_before,
                mse_after=mse_after,
                mos_damaged=mos_damaged,
                mos_restored=mos_restored,
                mos_clean=mos_clean,
                verdict=verdict,
                processing_time=proc_time,
            )
            report.results.append(result)

            print(
                f"  {result.file[:45]:45s} SNR {snr_before:+6.1f}→{snr_after:+6.1f} dB "
                f"({snr_improvement:+5.1f}) | MOS {mos_damaged:.2f}→{mos_restored:.2f} "
                f"(clean {mos_clean:.2f}) | {verdict}"
            )

    report.total_time = time.time() - t0
    return report


def main() -> int:
    print("§v10.800 Echter-Corpus-Benchmark")
    print(f"Stichprobe: {FILES_PER_MEDIUM} Dateien × 6 Medien, je {CHUNK_SEC:.0f}s")
    report = run_benchmark()

    print(f"\n{'=' * 60}")
    print("GESAMTBILANZ")
    print(f"{'=' * 60}")
    print(f"Verbessert:   {report.improved}/{len(report.results)}")
    print(f"Neutral:      {sum(1 for r in report.results if r.verdict == 'neutral')}/{len(report.results)}")
    print(f"Verschlechtert: {report.degraded}/{len(report.results)}")
    print(f"Ø SNR-Verbesserung: {report.avg_snr_improvement:+.2f} dB")
    print(f"Gesamtzeit: {report.total_time:.0f}s")

    # Per-Medium-Bilanz
    print(f"\nPer Medium:")
    by_medium: dict[str, list[FileResult]] = {}
    for r in report.results:
        by_medium.setdefault(r.medium, []).append(r)
    for medium, results in sorted(by_medium.items()):
        avg = float(np.mean([r.snr_improvement_db for r in results]))
        improved = sum(1 for r in results if r.verdict == "improved")
        print(f"  {medium:12s}: Ø {avg:+5.2f} dB, {improved}/{len(results)} verbessert")

    # Speichern als JSON
    out = _PROJECT / "benchmarks" / f"corpus_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v10.800",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "improved": report.improved,
            "degraded": report.degraded,
            "avg_snr_improvement_db": round(report.avg_snr_improvement, 2),
            "total_files": len(report.results),
        },
        "results": [
            {
                "medium": r.medium,
                "file": r.file,
                "snr_before_db": round(r.snr_before_db, 2),
                "snr_after_db": round(r.snr_after_db, 2),
                "snr_improvement_db": round(r.snr_improvement_db, 2),
                "verdict": r.verdict,
            }
            for r in report.results
        ],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nBericht: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
