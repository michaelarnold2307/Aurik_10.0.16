"""Result-Enrichment — RT-Faktor, Phase-Report, Export-Chain. Spec v10.206 §1-5.

Extrahiert aus RestorationResult die Daten für:
- Performance-Transparenz (RT-Faktor, Phase-Timings)
- Phase-Report (Deltas, Übersprungene, Timeline)
- Export-Chain (Resample→Dither→Format)
- Technische Metriken (LUFS-Delta, Chroma, VQI, Goosebumps)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceInfo:
    """RT-Faktor und Phase-Timings."""
    rt_factor: float = 0.0
    total_duration_s: float = 0.0
    processing_time_s: float = 0.0
    phase_timings: dict[str, float] = field(default_factory=dict)  # phase_id → seconds
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "rt_factor": f"{self.rt_factor:.1f}×",
            "duration": f"{self.total_duration_s:.1f}s",
            "processing": f"{self.processing_time_s:.1f}s",
            "top_phases": sorted(self.phase_timings.items(), key=lambda x: -x[1])[:5],
        }


@dataclass
class PhaseReport:
    """Phase-für-Phase-Report."""
    phases_total: int = 0
    phases_executed: int = 0
    phases_skipped: list[str] = field(default_factory=list)
    phase_deltas: dict[str, float] = field(default_factory=dict)  # phase_id → quality_delta
    skip_reasons: dict[str, str] = field(default_factory=dict)  # phase_id → reason
    timeline: list[tuple[str, float, float]] = field(default_factory=list)  # (phase_id, start_s, end_s)

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "total": self.phases_total,
            "executed": self.phases_executed,
            "skipped": len(self.phases_skipped),
            "skipped_list": self.phases_skipped[:10],
            "improvements": sorted(self.phase_deltas.items(), key=lambda x: -x[1])[:5],
            "degradations": sorted(self.phase_deltas.items(), key=lambda x: x[1])[:3],
        }


@dataclass
class ExportChainInfo:
    """Export-Chain-Transparenz."""
    input_format: str = ""
    output_format: str = ""
    sample_rate_in: int = 0
    sample_rate_out: int = 0
    bit_depth: int = 24
    dither_type: str = "POW-r Type 3"
    resample_method: str = "soxr_hq"
    true_peak_db: float = -1.0
    lufs_integrated: float = -14.0
    file_size_before_mb: float = 0.0
    file_size_after_mb: float = 0.0

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "chain": f"{self.sample_rate_in}Hz → Resample({self.resample_method}) → {self.sample_rate_out}Hz → Dither({self.dither_type}) → {self.output_format.upper()} {self.bit_depth}-bit",
            "true_peak": f"{self.true_peak_db:.1f} dBTP",
            "lufs": f"{self.lufs_integrated:.1f} LUFS",
            "size_change": f"{self.file_size_before_mb:.1f} MB → {self.file_size_after_mb:.1f} MB",
        }


@dataclass
class TechnicalMetrics:
    """Technische Metriken (LUFS-Delta, Chroma, VQI, Goosebumps)."""
    lufs_delta: float = 0.0
    chroma_correlation: float = 0.0
    warmth_band_loss_db: float = 0.0
    vqi: float = 0.0
    goosebumps_score: float = 0.0
    presence_score: float = 0.0

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "lufs_delta": f"{self.lufs_delta:+.1f} LUFS",
            "chroma": f"{self.chroma_correlation:.3f}",
            "warmth_loss": f"{self.warmth_band_loss_db:.1f} dB",
            "vqi": f"{self.vqi:.2f}",
            "goosebumps": f"{self.goosebumps_score:.2f}",
            "presence": f"{self.presence_score:.2f}",
        }


class ResultEnricher:
    """Extrahiert erweiterte Metriken aus RestorationResult + Pipeline-Context."""

    @staticmethod
    def extract_performance(result: Any) -> PerformanceInfo:
        info = PerformanceInfo()
        try:
            info.processing_time_s = float(getattr(result, "total_time_seconds", 0) or 0)
            info.total_duration_s = float(getattr(result, "audio_duration_s", 0) or 0)
            if info.processing_time_s > 0 and info.total_duration_s > 0:
                info.rt_factor = info.processing_time_s / info.total_duration_s
            meta = getattr(result, "metadata", None) or {}
            info.phase_timings = meta.get("phase_timings", {})
        except Exception:
            pass
        return info

    @staticmethod
    def extract_phase_report(result: Any) -> PhaseReport:
        report = PhaseReport()
        try:
            meta = getattr(result, "metadata", None) or {}
            report.phases_skipped = meta.get("phases_skipped", [])
            report.phase_deltas = meta.get("phase_deltas", {})
            report.skip_reasons = meta.get("skip_reasons", {})
            report.phases_total = meta.get("phases_total", 0)
            report.phases_executed = report.phases_total - len(report.phases_skipped)
            report.timeline = meta.get("phase_timeline", [])
        except Exception:
            pass
        return report

    @staticmethod
    def extract_export_chain(result: Any) -> ExportChainInfo:
        info = ExportChainInfo()
        try:
            meta = getattr(result, "metadata", None) or {}
            info.true_peak_db = float(meta.get("true_peak_db", -1.0))
            info.lufs_integrated = float(meta.get("lufs_integrated", -14.0))
            info.file_size_before_mb = float(meta.get("file_size_before_mb", 0))
            info.file_size_after_mb = float(meta.get("file_size_after_mb", 0))
        except Exception:
            pass
        return info

    @staticmethod
    def extract_technical_metrics(result: Any) -> TechnicalMetrics:
        metrics = TechnicalMetrics()
        try:
            meta = getattr(result, "metadata", None) or {}
            metrics.lufs_delta = float(meta.get("lufs_delta", 0))
            metrics.chroma_correlation = float(meta.get("chroma_correlation", 0))
            metrics.warmth_band_loss_db = float(meta.get("warmth_band_loss_cumulative_db", 0))
            metrics.vqi = float(meta.get("vqi", 0))
            metrics.goosebumps_score = float(meta.get("goosebumps_result", {}).get("score", 0) if isinstance(meta.get("goosebumps_result"), dict) else 0)
        except Exception:
            pass
        return metrics

    @classmethod
    def enrich(cls, result: Any) -> dict[str, Any]:
        """Alle Metriken extrahieren."""
        return {
            "performance": cls.extract_performance(result).to_display_dict(),
            "phase_report": cls.extract_phase_report(result).to_display_dict(),
            "export_chain": cls.extract_export_chain(result).to_display_dict(),
            "technical": cls.extract_technical_metrics(result).to_display_dict(),
        }
