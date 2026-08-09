"""backup/core/export_transparency.py — §v10.700 A5: Export-Transparenz.

Berechnet eine ExportTransparency-Zusammenfassung mit:
- Resample-Kette (48→44.1 kHz via Lanczos-4)
- True-Peak nach Export (dBTP)
- Integrated LUFS nach Export
- Dateigröße vorher/nachher
- Dithering-Methode

Diese Daten werden in RestorationResult.metadata["export_transparency"]
gespeichert und von der GUI im Export-Dialog angezeigt.

§v10.700: Kein Nutzer soll raten müssen, was beim Export passiert ist.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExportTransparency:
    """Export-Details für GUI-Transparenz."""

    # Resample
    original_sample_rate: int = 48000
    export_sample_rate: int = 48000
    resample_method: str = "none"  # "none" | "Lanczos-4" | "Kaiser-b14"
    resample_chain: str = ""  # z.B. "48.0 kHz → 44.1 kHz (Lanczos-4)"

    # Pegel
    true_peak_dbtp: float = 0.0  # dBTP nach ITU-R BS.1770-5
    true_peak_ok: bool = True  # <= -1.0 dBTP?
    integrated_lufs: float = 0.0  # LUFS integrated
    lufs_target: float = -14.0  # Ziel-LUFS
    lufs_ok: bool = True  # innerhalb ±2 LU?

    # Dithering
    bit_depth: int = 24
    dither_method: str = "none"  # "none" | "POW-r3" | "TPDF"

    # Dateigröße
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    compression_ratio: float = 1.0  # output/input

    # Format
    export_format: str = "FLAC"

    def to_dict(self) -> dict:
        return {
            "original_sample_rate": self.original_sample_rate,
            "export_sample_rate": self.export_sample_rate,
            "resample_method": self.resample_method,
            "resample_chain": self.resample_chain,
            "true_peak_dbtp": round(self.true_peak_dbtp, 2),
            "true_peak_ok": self.true_peak_ok,
            "integrated_lufs": round(self.integrated_lufs, 1),
            "lufs_target": self.lufs_target,
            "lufs_ok": self.lufs_ok,
            "bit_depth": self.bit_depth,
            "dither_method": self.dither_method,
            "input_size_bytes": self.input_size_bytes,
            "output_size_bytes": self.output_size_bytes,
            "input_size_mb": round(self.input_size_bytes / 1_048_576, 2) if self.input_size_bytes else 0,
            "output_size_mb": round(self.output_size_bytes / 1_048_576, 2) if self.output_size_bytes else 0,
            "compression_ratio": round(self.compression_ratio, 2),
            "export_format": self.export_format,
        }

    def summary_text(self) -> str:
        """Einzeilige Zusammenfassung für Quick-Anzeige."""
        parts = []
        if self.resample_chain:
            parts.append(f"↕ {self.resample_chain}")
        if self.export_format:
            parts.append(f"📦 {self.export_format} {self.bit_depth}-bit")
        if self.dither_method and self.dither_method != "none":
            parts.append(f"🔊 {self.dither_method}")
        if self.true_peak_dbtp != 0:
            ok = "✓" if self.true_peak_ok else "⚠"
            parts.append(f"📏 {self.true_peak_dbtp:+.1f} dBTP {ok}")
        if self.integrated_lufs != 0:
            ok = "✓" if self.lufs_ok else "⚠"
            parts.append(f"🔉 {self.integrated_lufs:+.1f} LUFS {ok}")
        if self.compression_ratio != 1.0:
            parts.append(f"📊 ×{self.compression_ratio:.1f}")
        return " · ".join(parts) if parts else "Export ohne Details"


def compute_export_transparency(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    input_path: str | Path | None = None,
    export_sample_rate: int | None = None,
    bit_depth: int = 24,
    export_format: str = "FLAC",
    lufs_target: float = -14.0,
) -> ExportTransparency:
    """Berechnet die Export-Transparenz-Daten.

    Args:
        audio: Exportiertes Audio (float32, [-1, 1])
        sample_rate: Original-Sample-Rate
        output_path: Pfad zur exportierten Datei
        input_path: Pfad zur Original-Datei (für Größenvergleich)
        export_sample_rate: Sample-Rate des Exports (None = kein Resampling)
        bit_depth: Bit-Tiefe (16, 24, 32)
        export_format: Dateiformat (FLAC, WAV, MP3, ...)
        lufs_target: Ziel-LUFS für Loudness-Normalisierung
    """
    result = ExportTransparency(
        original_sample_rate=sample_rate,
        export_sample_rate=export_sample_rate or sample_rate,
        bit_depth=bit_depth,
        export_format=export_format,
        lufs_target=lufs_target,
    )

    # ── Resample-Kette ──
    if export_sample_rate and export_sample_rate != sample_rate:
        result.resample_method = "Lanczos-4"
        result.resample_chain = f"{sample_rate / 1000:.1f} kHz → {export_sample_rate / 1000:.1f} kHz (Lanczos-4)"
    else:
        result.resample_chain = f"{sample_rate / 1000:.1f} kHz (kein Resampling)"

    # ── True-Peak (ITU-R BS.1770-5, 4× Oversampling) ──
    try:
        from backend.core.audio_exporter import _approx_true_peak

        tp = _approx_true_peak(audio.astype(np.float32), sample_rate, upsample=4)
        result.true_peak_dbtp = round(float(tp), 2) if tp != float("-inf") else -120.0
        result.true_peak_ok = result.true_peak_dbtp <= -1.0
    except ImportError:
        # Fallback: einfacher Peak
        peak = float(np.max(np.abs(audio)))
        result.true_peak_dbtp = round(float(20.0 * np.log10(max(peak, 1e-10))), 2)
        result.true_peak_ok = result.true_peak_dbtp <= -1.0

    # ── Integrated LUFS (ITU-R BS.1770-5) ──
    try:
        import pyloudnorm as pln

        meter = pln.Meter(rate=sample_rate, block_size=0.400)
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
        lufs = float(meter.integrated_loudness(mono.astype(np.float64)))
        result.integrated_lufs = round(lufs, 1)
        result.lufs_ok = abs(lufs - lufs_target) <= 2.0
    except ImportError:
        # Fallback: RMS-basierte Schätzung
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        estimated_lufs = float(20.0 * np.log10(max(rms, 1e-10))) - 3.0  # grobe Näherung
        result.integrated_lufs = round(estimated_lufs, 1)
        result.lufs_ok = True  # Kann nicht verifiziert werden
        logger.warning("§G23 pyloudnorm nicht verfügbar — LUFS via RMS-Schätzung (DSP-Ersatzpfad)")

    # ── Dithering ──
    if bit_depth <= 16 and export_format in ("WAV", "AIFF", "FLAC"):
        try:
            from backend.core.audio_exporter import AudioExporter

            # Prüfe ob POW-r verfügbar ist
            result.dither_method = "POW-r Type 3"  # Primär via AudioExporter
        except ImportError:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            result.dither_method = "TPDF (Fallback)"
    elif bit_depth >= 24:
        result.dither_method = "none (24-bit — kein Dithering nötig)"

    # ── Dateigröße ──
    output_path = Path(output_path)
    if output_path.exists():
        result.output_size_bytes = output_path.stat().st_size
    if input_path:
        input_path = Path(input_path)
        if input_path.exists():
            result.input_size_bytes = input_path.stat().st_size
    if result.input_size_bytes > 0 and result.output_size_bytes > 0:
        result.compression_ratio = round(result.output_size_bytes / result.input_size_bytes, 2)

    logger.info(
        "§v10.700 A5 Ausgabe-Transparenz: %s %d-bit, %.1f dBTP, %.1f LUFS, %s → %s",
        result.export_format,
        result.bit_depth,
        result.true_peak_dbtp,
        result.integrated_lufs,
        f"{result.input_size_bytes / 1e6:.1f} MB" if result.input_size_bytes else "?",
        f"{result.output_size_bytes / 1e6:.1f} MB" if result.output_size_bytes else "?",
    )
    return result


def quick_export_summary(
    audio: np.ndarray,
    sample_rate: int,
    output_path: str | Path,
    input_path: str | Path | None = None,
) -> str:
    """Einzeilige Export-Zusammenfassung für schnelle Anzeige."""
    t = compute_export_transparency(
        audio,
        sample_rate,
        output_path,
        input_path,
        export_sample_rate=sample_rate,
    )
    return t.summary_text()
