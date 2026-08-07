#!/usr/bin/env python3
"""Aurik MAX vs iZotope RX 11 vs CEDAR — Competitive Benchmark v11.

Reproduzierbarer Wettbewerbsvergleich mit objektiven Metriken.
Benötigt vorverarbeitete Referenzdateien von iZotope RX 11 und CEDAR.

Usage:
  python benchmarks/competitive_benchmark_v11.py --test-dir test_audio --output reports/

Die iZotope- und CEDAR-Dateien werden in benchmarks/competitive/izotope_rx/
und benchmarks/competitive/cedar/ erwartet (manuell vorverarbeitet).
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """Lädt Audio (48kHz float32 mono/stereo)."""
    try:
        import soundfile as sf

        audio, sr = sf.read(filepath, dtype="float32")
        if sr != 48000:
            from scipy.signal import resample_poly

            48000 / sr
            audio = resample_poly(audio, up=48000, down=sr)
        return audio.astype(np.float32), 48000
    except Exception as e:
        logger.error("Failed to load %s: %s", filepath, e)
        raise


def compute_hpe(audio: np.ndarray, sr: int) -> float:
    """Human Pleasantness Estimator Score."""
    try:
        from backend.core.human_pleasantness_estimator import compute_pleasantness

        return float(compute_pleasantness(audio, sr).score)
    except Exception:
        logger.warning("competitive_benchmark_v11.py::compute_hpe fallback", exc_info=True)
        return 0.5


def compute_snr_improvement(original: np.ndarray, processed: np.ndarray) -> float:
    """SNR-Verbesserung in dB."""
    noise = original - processed[: len(original)]
    signal_rms = float(np.sqrt(np.mean(original**2)) + 1e-12)
    noise_rms = float(np.sqrt(np.mean(noise**2)) + 1e-12)
    return 20.0 * np.log10(signal_rms / (noise_rms + 1e-12))  # type: ignore[no-any-return]


def compute_stereo_preservation(original: np.ndarray, processed: np.ndarray) -> float:
    """Stereo-Breiten-Erhalt (Korrelation)."""
    if original.ndim < 2 or original.shape[1] < 2:
        return 1.0
    orig_ms = np.corrcoef(original[:, 0], original[:, 1])[0, 1]
    proc_ms = np.corrcoef(processed[:, 0], processed[:, 1])[0, 1]
    return 1.0 - abs(orig_ms - proc_ms)  # type: ignore[no-any-return]


def compute_spectral_flatness(audio: np.ndarray) -> float:
    """Spektrale Flachheit (niedriger = tonaler, höher = rauschhafter)."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    spec = np.abs(np.fft.rfft(mono[:4096] * np.hanning(4096))) + 1e-12
    geo_mean = np.exp(np.mean(np.log(spec)))
    arith_mean = np.mean(spec)
    return float(geo_mean / arith_mean)


def compute_mushra_proxy(original: np.ndarray, processed: np.ndarray, sr: int) -> float:
    """MERT-basierter MUSHRA-Proxy-Score (0-100)."""
    try:
        from backend.core.mert_mushra_proxy import estimate_mushra_proxy

        return float(estimate_mushra_proxy(original, processed, sr).proxy_score)
    except Exception:
        logger.warning("competitive_benchmark_v11.py::compute_mushra_proxy fallback", exc_info=True)
        return 50.0


def compute_all_metrics(original: np.ndarray, processed: np.ndarray, sr: int) -> dict:
    """Berechnet alle Metriken für eine Datei."""
    proc = processed[: len(original)]
    return {
        "hpe": round(compute_hpe(proc, sr), 3),
        "snr_improvement_db": round(compute_snr_improvement(original, proc), 1),
        "stereo_preservation": round(compute_stereo_preservation(original, proc), 3),
        "spectral_flatness": round(compute_spectral_flatness(proc), 4),
        "mushra_proxy": round(compute_mushra_proxy(original, proc, sr), 1),
    }


def compare_tools(
    test_files: list[str],
    aurik_dir: str,
    izotope_dir: str | None,
    cedar_dir: str | None,
) -> dict:
    """Führt den Vergleich durch."""
    results: Any = {"per_file": {}, "summary": {}}
    aurik_path = Path(aurik_dir)
    izotope_path = Path(izotope_dir) if izotope_dir else None
    cedar_path = Path(cedar_dir) if cedar_dir else None

    for tf in test_files:
        tf_path = Path(tf)
        name = tf_path.stem
        logger.info("Processing: %s", name)

        try:
            original, sr = load_audio(str(tf_path))
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue

        file_result = {}

        # Aurik
        aurik_file = aurik_path / f"{name}_restored.wav"
        if aurik_file.exists():
            aurik_audio, _ = load_audio(str(aurik_file))
            file_result["aurik"] = compute_all_metrics(original, aurik_audio, sr)
        else:
            file_result["aurik"] = {"error": "file not found"}

        # iZotope RX 11
        if izotope_path:
            iz_file = izotope_path / f"{name}_rx11.wav"
            if iz_file.exists():
                iz_audio, _ = load_audio(str(iz_file))
                file_result["izotope_rx11"] = compute_all_metrics(original, iz_audio, sr)
            else:
                file_result["izotope_rx11"] = {"error": "file not found"}

        # CEDAR
        if cedar_path:
            cd_file = cedar_path / f"{name}_cedar.wav"
            if cd_file.exists():
                cd_audio, _ = load_audio(str(cd_file))
                file_result["cedar"] = compute_all_metrics(original, cd_audio, sr)
            else:
                file_result["cedar"] = {"error": "file not found"}

        results["per_file"][name] = file_result
        time.sleep(0.5)

    # Summary
    tools = ["aurik"]
    if izotope_path:
        tools.append("izotope_rx11")
    if cedar_path:
        tools.append("cedar")

    for metric in ["hpe", "snr_improvement_db", "stereo_preservation", "mushra_proxy"]:
        avg = {}
        for tool in tools:
            values = []
            for name, fr in results["per_file"].items():
                if tool in fr and "error" not in fr[tool] and metric in fr[tool]:
                    values.append(fr[tool][metric])
            avg[tool] = round(sum(values) / len(values), 2) if values else 0
        results["summary"][metric] = avg

    # Find winners
    results["winners"] = {}
    for metric, avgs in results["summary"].items():
        if avgs:
            higher_better = metric != "spectral_flatness"
            results["winners"][metric] = max(avgs, key=avgs.get) if higher_better else min(avgs, key=avgs.get)

    return results  # type: ignore[no-any-return]


def generate_report(results: dict, output_path: str):
    """Generiert Markdown-Vergleichsbericht."""
    lines = [
        "# Aurik MAX vs iZotope RX 11 vs CEDAR — Competitive Benchmark",
        "",
        f"**Datum:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Dateien:** {len(results['per_file'])}",
        "",
        "## Zusammenfassung",
        "",
        "| Metrik | Aurik MAX | iZotope RX 11 | CEDAR | Gewinner |",
        "|--------|-----------|---------------|-------|----------|",
    ]

    metrics = ["hpe", "snr_improvement_db", "stereo_preservation", "mushra_proxy"]
    labels = {
        "hpe": "HPE Naturalness",
        "snr_improvement_db": "SNR Δ (dB)",
        "stereo_preservation": "Stereo-Erhalt",
        "mushra_proxy": "MUSHRA Proxy",
    }
    for metric in metrics:
        avgs = results["summary"].get(metric, {})
        if not avgs:
            continue
        aurik = avgs.get("aurik", "-")
        izo = avgs.get("izotope_rx11", "-")
        cedar = avgs.get("cedar", "-")
        winner = results["winners"].get(metric, "?")
        lines.append(f"| {labels[metric]} | {aurik} | {izo} | {cedar} | **{winner}** |")

    lines += [
        "",
        "## Pro Datei",
        "",
    ]
    for name, fr in results["per_file"].items():
        lines.append(f"### {name}")
        lines.append("| Tool | HPE | SNR Δ | Stereo | MUSHRA |")
        lines.append("|------|-----|-------|--------|--------|")
        for tool in ["aurik", "izotope_rx11", "cedar"]:
            if tool not in fr:
                continue
            m = fr[tool]
            if "error" in m:
                lines.append(f"| {tool} | ❌ | ❌ | ❌ | ❌ |")
            else:
                lines.append(
                    f"| {tool} | {m.get('hpe', '-')} | {m.get('snr_improvement_db', '-')} | {m.get('stereo_preservation', '-')} | {m.get('mushra_proxy', '-')} |"
                )
        lines.append("")

    # Fazit
    lines += [
        "## Fazit",
        "",
    ]
    aurik_wins = sum(1 for w in results["winners"].values() if w == "aurik")
    total = len(results["winners"])
    lines.append(f"Aurik gewinnt **{aurik_wins}/{total}** Metriken.")

    if aurik_wins >= total * 0.5:
        lines.append("Aurik MAX ist wettbewerbsfähig mit kommerziellen Tools — zu 0 €.")
    else:
        lines.append("Kommerzielle Tools haben noch Vorteile in einzelnen Metriken.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report saved: %s", path)


def main():
    parser = argparse.ArgumentParser(description="Aurik vs iZotope RX 11 vs CEDAR")
    parser.add_argument("--test-dir", default="test_audio", help="Original test files directory")
    parser.add_argument("--aurik-output", default="output", help="Aurik processed files directory")
    parser.add_argument("--izotope-dir", default="benchmarks/competitive/izotope_rx")
    parser.add_argument("--cedar-dir", default="benchmarks/competitive/cedar")
    parser.add_argument("--output", default="reports/competitive_v11_report.md")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        sys.exit(1)

    test_files = sorted(test_dir.glob("*.wav")) + sorted(test_dir.glob("*.flac"))
    if not test_files:
        logger.error("No audio files found in %s", test_dir)
        sys.exit(1)

    results = compare_tools(
        [str(f) for f in test_files],
        args.aurik_output,
        args.izotope_dir if Path(args.izotope_dir).exists() else None,
        args.cedar_dir if Path(args.cedar_dir).exists() else None,
    )
    generate_report(results, args.output)
    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
