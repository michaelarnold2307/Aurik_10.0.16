#!/usr/bin/env python3
"""
§v10.930: Stereo-Benchmark — die bisher ungemessene Dimension.

Der bisherige Corpus-Benchmark mischte Stereo zu Mono. Dieser Benchmark
misst die STEREO-Integrität der Repair-Kette:

  1. Channel-Korrelation  — bleibt das Stereobild erhalten?
  2. Phasenkohärenz       — bleibt die inter-kanalige Phase intakt?
  3. SNR pro Kanal        — verbessert sich L UND R (nicht nur Mono-Mittel)?
  4. Side-Energie-Anteil  — bleibt die M/S-Balance erhalten?

Verdict: Ein Schritt darf die Stereo-Integrität NICHT zerstören.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

SR = 48000
CHUNK_SEC = 4.0


def _load_stereo(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    n = min(int(CHUNK_SEC * sr), audio.shape[1])
    return audio[:, :n].astype(np.float32), sr


def _snr_db(reference: np.ndarray, signal: np.ndarray) -> float:
    n = np.asarray(signal) - np.asarray(reference)
    ref = np.asarray(reference)
    return float(10 * np.log10((np.mean(ref ** 2) + 1e-10) / (np.mean(n ** 2) + 1e-10)))


def _channel_correlation(audio: np.ndarray) -> float:
    """Korrelation zwischen L und R — 1.0 = identisch, ~0 = unkorreliert."""
    if audio.ndim < 2 or audio.shape[0] < 2:
        return 1.0
    l, r = audio[0], audio[1]
    if l.std() < 1e-8 or r.std() < 1e-8:
        return 1.0
    return float(np.corrcoef(l, r)[0, 1])


def _phase_coherence(audio: np.ndarray, sr: int = SR) -> float:
    """Inter-kanalige Phasenkohärenz im Mittenband (300 Hz - 3 kHz)."""
    if audio.ndim < 2 or audio.shape[0] < 2:
        return 1.0
    n_fft = 2048
    hop = 512
    l, r = audio[0], audio[1]
    n_frames = max(1, (len(l) - n_fft) // hop)
    spec_l = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.complex64)
    spec_r = np.zeros_like(spec_l)
    window = np.hanning(n_fft)
    for i in range(n_frames):
        s = i * hop
        if s + n_fft > len(l):
            break
        spec_l[i] = np.fft.rfft(l[s:s + n_fft] * window)
        spec_r[i] = np.fft.rfft(r[s:s + n_fft] * window)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= 300) & (freqs <= 3000)
    # Phase-Differenz im Band
    phase_diff = np.angle(spec_l[:, band] * np.conj(spec_r[:, band]))
    # Kohärenz = |mean(e^{iΔφ})| — 1.0 = perfekt phasengleich, 0 = zufällig
    coherence = float(np.abs(np.exp(1j * phase_diff).mean()))
    return coherence


def _side_energy_ratio(audio: np.ndarray) -> float:
    """Side-Energie-Anteil = E[(L-R)/2] / (E[L] + E[R]) — Stereobreite-Maß."""
    if audio.ndim < 2 or audio.shape[0] < 2:
        return 0.0
    l, r = audio[0], audio[1]
    side = (l - r) / 2
    mid = (l + r) / 2
    side_e = float(np.mean(side ** 2))
    mid_e = float(np.mean(mid ** 2))
    return side_e / (side_e + mid_e + 1e-10)


@dataclass
class StereoResult:
    file: str
    corr_before: float
    corr_after: float
    phase_before: float
    phase_after: float
    side_before: float
    side_after: float
    snr_l_before: float
    snr_l_after: float
    snr_r_before: float
    snr_r_after: float
    stereo_ok: bool


def run_stereo_benchmark() -> list[StereoResult]:
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline
    from backend.core.coordinated_repair import RepairPlanner, CoordinatedRepair

    results: list[StereoResult] = []

    for medium in ["cassette", "digital", "reel_tape", "shellac", "tape", "vinyl"]:
        damaged_dir = _PROJECT / "corpus" / medium / "damaged"
        clean_dir = _PROJECT / "corpus" / medium / "clean"
        if not damaged_dir.is_dir():
            continue

        damaged_files = sorted(damaged_dir.glob("*.wav"))[:1]
        for damaged_path in damaged_files:
            base = damaged_path.stem.rsplit("_", 1)[0]
            clean_path = clean_dir / f"{base}_clean.wav"
            if not clean_path.exists():
                continue

            damaged, sr = _load_stereo(damaged_path)
            clean, _ = _load_stereo(clean_path)
            min_len = min(damaged.shape[1], clean.shape[1])
            damaged, clean = damaged[:, :min_len], clean[:, :min_len]

            # Metriken VORHER
            corr_b = _channel_correlation(damaged)
            phase_b = _phase_coherence(damaged, sr)
            side_b = _side_energy_ratio(damaged)
            snr_l_b = _snr_db(clean[0], damaged[0])
            snr_r_b = _snr_db(clean[1], damaged[1])

            # Repair-Kette auf STEREO
            consensus = DefectConsensusPipeline()
            # Konsens auf Mono-Mittel (Detektion ist Mono-basiert)
            mono = damaged.mean(axis=0)
            manifest = consensus.analyze(
                mono, sr, metadata={"material": medium, "is_digital": medium == "digital"},
            )
            plan = RepairPlanner().plan(manifest, min_len)
            executor = CoordinatedRepair()
            restored, _ = executor.execute(damaged, plan, manifest, sr, material=medium)
            restored = np.asarray(restored)
            if restored.ndim == 1:
                restored = restored[np.newaxis, :]
            restored = restored[:, :min_len]

            # Metriken NACHHER
            corr_a = _channel_correlation(restored)
            phase_a = _phase_coherence(restored, sr)
            side_a = _side_energy_ratio(restored)
            snr_l_a = _snr_db(clean[0], restored[0])
            snr_r_a = _snr_db(clean[1], restored[1])

            # Stereo-Verdict: Korrelation und Side-Energie dürfen nicht stark abweichen
            corr_drop = abs(corr_a - corr_b)
            side_drop = abs(side_a - side_b)
            stereo_ok = corr_drop < 0.2 and side_drop < 0.2

            results.append(StereoResult(
                file=damaged_path.name,
                corr_before=corr_b, corr_after=corr_a,
                phase_before=phase_b, phase_after=phase_a,
                side_before=side_b, side_after=side_a,
                snr_l_before=snr_l_b, snr_l_after=snr_l_a,
                snr_r_before=snr_r_b, snr_r_after=snr_r_a,
                stereo_ok=stereo_ok,
            ))

            print(
                f"{medium:10s} {damaged_path.name[:38]:38s} "
                f"Corr {corr_b:.3f}→{corr_a:.3f} | Phase {phase_b:.2f}→{phase_a:.2f} | "
                f"Side {side_b:.3f}→{side_a:.3f} | SNR-L {snr_l_b:+.1f}→{snr_l_a:+.1f} | "
                f"SNR-R {snr_r_b:+.1f}→{snr_r_a:+.1f} | {'✅' if stereo_ok else '❌'}"
            )

    return results


def main() -> int:
    print("§v10.930 Stereo-Benchmark — die unbekannte Dimension")
    print("=" * 70)
    t0 = time.time()
    results = run_stereo_benchmark()

    print("\n" + "=" * 70)
    ok = sum(1 for r in results if r.stereo_ok)
    print(f"Stereo-Integrität: {ok}/{len(results)} Dateien intakt")
    if results:
        corr_drops = [abs(r.corr_after - r.corr_before) for r in results]
        print(f"Ø Korrelations-Drift: {np.mean(corr_drops):.3f}")
        snr_deltas = [
            ((r.snr_l_after + r.snr_r_after) / 2) - ((r.snr_l_before + r.snr_r_before) / 2)
            for r in results
        ]
        print(f"Ø SNR-Änderung (L+R gemittelt): {np.mean(snr_deltas):+.2f} dB")
    print(f"Dauer: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
