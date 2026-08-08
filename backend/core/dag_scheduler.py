"""DAG-Scheduler — Unabhängige Phasen parallel ausführen.

Analysiert Phasen-Abhängigkeiten via Frequenzbereichs-Überlappung.
Phasen ohne gemeinsame Frequenzbänder können parallel laufen.
Nutzt ThreadPoolExecutor für CPU-Phasen.

Spec 11 §ROADMAP-4 Erweiterung.
"""
from __future__ import annotations
import logging, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Frequenzprofile pro Phase (Hz-Bereiche)
PHASE_FREQ_BANDS: dict[str, tuple[float, float]] = {
    "phase_01": (20, 20000), "phase_03": (20, 20000), "phase_04": (20, 20000),
    "phase_07": (500, 8000), "phase_18": (20, 20000), "phase_19": (2000, 12000),
    "phase_20": (200, 8000), "phase_23": (500, 16000), "phase_35": (20, 20000),
    "phase_38": (2000, 8000), "phase_39": (8000, 20000), "phase_40": (20, 20000),
    "phase_42": (100, 8000), "phase_47": (20, 20000), "phase_48": (1000, 15000),
}


class DAGScheduler:
    """Plant und führt Phasen nach DAG-Abhängigkeiten parallel aus."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._freq_bands = dict(PHASE_FREQ_BANDS)

    def are_independent(self, phase_a: str, phase_b: str) -> bool:
        """True wenn zwei Phasen keine Frequenz-Überlappung haben."""
        if phase_a not in self._freq_bands or phase_b not in self._freq_bands:
            return True  # Unbekannt → sicherheitshalber parallel erlauben
        lo_a, hi_a = self._freq_bands[phase_a]
        lo_b, hi_b = self._freq_bands[phase_b]
        return hi_a < lo_b or hi_b < lo_a

    def build_dag(self, phase_ids: list[str]) -> list[list[str]]:
        """Gruppiert Phasen in parallel ausführbare Stufen."""
        levels: list[list[str]] = []
        remaining = list(phase_ids)
        while remaining:
            level = [remaining[0]]
            rest = []
            for pid in remaining[1:]:
                independent = all(self.are_independent(pid, l) for l in level)
                if independent:
                    level.append(pid)
                else:
                    rest.append(pid)
            levels.append(level)
            remaining = rest
        return levels

    def execute_parallel(self, phases: list[tuple[str, Callable]], audio: Any, sample_rate: int,
                         progress_callback: Callable | None = None) -> dict[str, Any]:
        """Führt Phasen gemäß DAG parallel aus."""
        phase_ids = [p[0] for p in phases]
        levels = self.build_dag(phase_ids)
        results: dict[str, Any] = {}
        t0 = time.monotonic()

        for level_idx, level in enumerate(levels):
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(level))) as executor:
                futures = {}
                for pid in level:
                    fn = next((p[1] for p in phases if p[0] == pid), None)
                    if fn:
                        futures[executor.submit(fn, audio, sample_rate)] = pid

                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        results[pid] = future.result(timeout=300)
                        results[pid]["_dag_level"] = level_idx
                    except Exception as e:
                        logger.warning("DAG-Phase %s fehlgeschlagen: %s", pid, e)
                        results[pid] = {"error": str(e), "_dag_level": level_idx}

            if progress_callback:
                progress_callback(level_idx / max(1, len(levels)))

        results["_dag_stats"] = {"levels": len(levels), "parallel_phases": len(phases),
                                  "wall_time_s": round(time.monotonic() - t0, 2),
                                  "speedup_estimate": f"{len(phases)/len(levels):.1f}x"}
        return results

    def get_independent_groups(self, phase_ids: list[str]) -> list[list[str]]:
        """Gibt Gruppen unabhängiger Phasen zurück (für manuelle Parallelisierung)."""
        return self.build_dag(phase_ids)
