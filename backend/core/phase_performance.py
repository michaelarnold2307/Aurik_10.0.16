"""Phase-Performance-Instrumentierung. 10.0.20 Upgrade #4.

Misst pro Phase:
- Ausführungszeit (Wall-Clock)
- RT-Faktor (Verarbeitungszeit / Audio-Dauer)
- Peak-Memory (via tracemalloc)
- Phase-Status (OK / Skipped / Failed / Rollback)

Integration: unified_restorer_v3._execute_pipeline → PhasePerformanceTracker.
GUI: Aurik10/core/result_enrichment.py → PerformanceInfo nutzt diese Daten.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhaseTiming:
    """Timing-Daten einer einzelnen Phase."""

    phase_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0
    audio_duration_s: float = 0.0
    rt_factor: float = 0.0
    status: str = "pending"  # ok, skipped, failed, rollback
    memory_delta_mb: float = 0.0
    input_samples: int = 0
    output_samples: int = 0


@dataclass
class PipelinePerformance:
    """Gesamt-Performance-Daten des Pipeline-Laufs."""

    phases: list[PhaseTiming] = field(default_factory=list)
    total_audio_duration_s: float = 0.0
    total_processing_s: float = 0.0
    overall_rt_factor: float = 0.0
    phases_executed: int = 0
    phases_skipped: int = 0
    phases_failed: int = 0
    peak_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_audio_s": round(self.total_audio_duration_s, 1),
            "total_processing_s": round(self.total_processing_s, 1),
            "overall_rt_factor": round(self.overall_rt_factor, 1),
            "phases_executed": self.phases_executed,
            "phases_skipped": self.phases_skipped,
            "phases_failed": self.phases_failed,
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "per_phase": [
                {
                    "phase": t.phase_id,
                    "duration_s": round(t.duration_s, 2),
                    "rt": round(t.rt_factor, 1),
                    "status": t.status,
                    "memory_delta_mb": round(t.memory_delta_mb, 1),
                }
                for t in self.phases
            ],
            "top5_slowest": [
                {"phase": t.phase_id, "duration_s": round(t.duration_s, 1)}
                for t in sorted(self.phases, key=lambda x: -x.duration_s)[:5]
            ],
        }


class PhasePerformanceTracker:
    """Hängt sich in _execute_pipeline ein, misst pro Phase.

    Usage:
        tracker = PhasePerformanceTracker()
        tracker.start_pipeline(audio_duration_s=180.0)
        for phase in phases:
            tracker.start_phase("phase_01")
            result = phase.process(audio)
            tracker.end_phase("phase_01", status="ok")
        perf = tracker.finish_pipeline()
    """

    def __init__(self) -> None:
        self._pipeline_start: float = 0.0
        self._phase_start: float = 0.0
        self._phases: list[PhaseTiming] = []
        self._audio_duration_s: float = 0.0
        self._peak_memory_mb: float = 0.0
        try:
            import tracemalloc

            tracemalloc.start()
            self._has_tracemalloc = True
        except Exception:
            self._has_tracemalloc = False

    def start_pipeline(self, audio_duration_s: float) -> None:
        """Pipeline-Start markieren."""
        self._pipeline_start = time.monotonic()
        self._audio_duration_s = audio_duration_s
        self._phases = []
        self._peak_memory_mb = 0.0
        if self._has_tracemalloc:
            import tracemalloc

            tracemalloc.reset_peak()

    def start_phase(self, phase_id: str) -> None:
        """Phasen-Start markieren."""
        self._phase_start = time.monotonic()

    def end_phase(
        self,
        phase_id: str,
        *,
        status: str = "ok",
        input_samples: int = 0,
        output_samples: int = 0,
    ) -> None:
        """Phasen-Ende markieren und Timing speichern."""
        end = time.monotonic()
        duration = end - self._phase_start
        rt = duration / max(self._audio_duration_s, 0.001)
        timing = PhaseTiming(
            phase_id=phase_id,
            start_time=self._phase_start,
            end_time=end,
            duration_s=duration,
            audio_duration_s=self._audio_duration_s,
            rt_factor=rt,
            status=status,
            input_samples=input_samples,
            output_samples=output_samples,
        )
        self._phases.append(timing)

        if status == "ok":
            pass
        elif status == "skipped":
            logger.debug("Phase %s übersprungen", phase_id)
        elif status == "failed":
            logger.warning("Phase %s fehlgeschlagen (%.2fs)", phase_id, duration)
        elif status == "rollback":
            logger.warning("Phase %s Rollback (%.2fs)", phase_id, duration)

    def finish_pipeline(self) -> PipelinePerformance:
        """Pipeline beenden und Gesamt-Performance berechnen."""
        total = time.monotonic() - self._pipeline_start

        if self._has_tracemalloc:
            import tracemalloc

            _, peak = tracemalloc.get_traced_memory()
            self._peak_memory_mb = peak / (1024 * 1024)

        return PipelinePerformance(
            phases=self._phases,
            total_audio_duration_s=self._audio_duration_s,
            total_processing_s=total,
            overall_rt_factor=total / max(self._audio_duration_s, 0.001),
            phases_executed=sum(1 for p in self._phases if p.status == "ok"),
            phases_skipped=sum(1 for p in self._phases if p.status == "skipped"),
            phases_failed=sum(1 for p in self._phases if p.status in ("failed", "rollback")),
            peak_memory_mb=self._peak_memory_mb,
        )
