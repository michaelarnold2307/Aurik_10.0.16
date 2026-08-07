"""§v10.15 Pipeline Budget Controller
===================================
Three workflow improvements for UV3 phase execution:

1. Phase progress tracking — for Watchdog dialog enrichment
2. Per-phase time caps — prevents individual ML-phase hangs
3. Budget-adaptive quality scaling — gradual strength reduction

Usage in UV3:
    from backend.core.pipeline_budget_controller import PipelineBudgetController
    _pbc = PipelineBudgetController(pipeline_wall_budget=4800.0)

    # At phase start:
    _pbc.on_phase_start(phase_id, phase_idx, total_phases)

    # During phase setup:
    budget_scalar = _pbc.get_budget_scalar(elapsed_non_exempt_s)
    effective_strength *= budget_scalar

    # Phase time cap check:
    if _pbc.is_phase_timed_out(phase_id, phase_start_ts):
        logger.warning("Phase skipped due to time cap")
        continue

    # For Watchdog dialog:
    progress = _pbc.get_progress()
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Per-Phase Time Caps (prevents individual ML-phase hangs) ──
PHASE_TIME_CAP_S: dict[str, float] = {
    "phase_06_frequency_restoration": 2700.0,  # NVSR max 45 min
    "phase_03_denoise": 1800.0,  # MelBandRoformer+Resemble 30 min
    "phase_23_spectral_repair": 1800.0,  # FlashSR 30 min
    "phase_12_wow_flutter_fix": 1200.0,  # pYIN+CREPE 20 min
    "phase_20_reverb_reduction": 1200.0,  # SGMSE+ 20 min
    "phase_29_tape_hiss_reduction": 900.0,  # OMLSA/IMCRA 15 min
}
FALLBACK_TIME_CAP_S: float = 3600.0  # 60 min default


class PipelineBudgetController:
    """Controls budget pressure, phase timeouts, and progress reporting."""

    def __init__(self, pipeline_wall_budget: float = 4800.0) -> None:
        self._wall_budget = max(1.0, pipeline_wall_budget)
        self._phase_current: int = 0
        self._phase_total: int = 0
        self._phase_name: str = ""
        self._phase_start_ts: float = 0.0
        self._elapsed_non_exempt: float = 0.0

    # ── Phase progress (for Watchdog dialog) ────────────────────────

    def on_phase_start(self, phase_id: str, phase_idx: int, total_phases: int) -> None:
        """Call at the start of each phase."""
        self._phase_current = phase_idx + 1
        self._phase_total = total_phases
        self._phase_name = phase_id
        self._phase_start_ts = time.monotonic()

    def update_elapsed(self, elapsed_non_exempt_s: float) -> None:
        """Update accumulated non-exempt elapsed time."""
        self._elapsed_non_exempt = elapsed_non_exempt_s

    def get_progress(self) -> dict[str, Any]:
        """Return progress dict for Watchdog dialog enrichment."""
        return {
            "current": self._phase_current,
            "total": self._phase_total,
            "name": self._phase_name,
            "elapsed_non_exempt_s": self._elapsed_non_exempt,
        }

    # ── Per-phase time cap ──────────────────────────────────────────

    def get_time_cap(self, phase_id: str) -> float:
        """Return the time cap in seconds for a given phase."""
        return PHASE_TIME_CAP_S.get(phase_id, FALLBACK_TIME_CAP_S)

    def is_phase_timed_out(self, phase_id: str, phase_start_ts: float) -> bool:
        """Check if a phase has exceeded its time cap."""
        cap = self.get_time_cap(phase_id)
        elapsed = time.monotonic() - phase_start_ts
        if elapsed > cap:
            logger.warning(
                "§v10.15 Verarbeitungsschritt-Time-Cap: %s elapsed %.0fs > cap %.0fs — recommend ueberspringen",
                phase_id,
                elapsed,
                cap,
            )
            return True
        return False

    # ── Budget-adaptive quality scaling ─────────────────────────────

    def get_budget_scalar(self, elapsed_non_exempt_s: float | None = None) -> float:
        """Return quality scalar based on budget pressure.

        < 80% budget: 1.0 (full quality)
        80–100%:     1.0 → 0.50 linear (gradual reduction)
        > 100%:       0.50 (floor)

        Gradual instead of binary skip/no-skip.
        """
        if elapsed_non_exempt_s is not None:
            self._elapsed_non_exempt = elapsed_non_exempt_s
        if self._wall_budget <= 0:
            return 1.0
        pressure = min(1.0, self._elapsed_non_exempt / self._wall_budget)
        if pressure < 0.80:
            return 1.0
        # Linear ramp: 0.80→1.0 maps to 1.0→0.50
        scalar = max(0.50, 1.0 - (pressure - 0.80) * 2.5)
        if pressure > 0.80:
            logger.info(
                "§v10.15 Grenze-Pressure: %.0f%% → quality scalar=%.2f",
                pressure * 100,
                scalar,
            )
        return float(scalar)

    def get_chunk_factor(self, elapsed_non_exempt_s: float | None = None) -> float:
        """Return chunk size factor based on budget pressure.

        < 80% budget: 1.0 (normal chunk size)
        80–100%:      1.0 → 2.0 (larger chunks = less overhead)
        """
        if elapsed_non_exempt_s is not None:
            self._elapsed_non_exempt = elapsed_non_exempt_s
        if self._wall_budget <= 0:
            return 1.0
        pressure = min(1.0, self._elapsed_non_exempt / self._wall_budget)
        if pressure < 0.80:
            return 1.0
        return float(min(2.0, 1.0 + (pressure - 0.80) * 5.0))
