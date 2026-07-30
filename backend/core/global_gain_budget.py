"""§GGB-1 Global Gain Budget (v10.13).

Cross-phase gain accumulator that prevents cumulative loudness inflation.
Tracks makeup gains across all phases and caps the total at a configurable
limit. Individual phases request gain budget; the coordinator approves or
caps based on global remaining budget.

Design:
  - Singleton pattern (thread-safe)
  - Per-phase request: budget.request(phase_id, gain_db, priority)
  - Returns approved gain (≤ requested)
  - Caps: 6 dB total pipeline, 2 dB per phase (except loudness norm)
  - best_effort phases get 0 dB
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class GlobalGainBudget:
    """Thread-safe singleton managing cumulative gain across all phases."""

    _TOTAL_BUDGET_DB: float = 6.0
    _MAX_PER_PHASE_DB: float = 2.0
    _LOUDNESS_NORM_PHASES: frozenset[str] = frozenset({"phase_40_loudness_normalization"})

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cumulative_db: float = 0.0
        self._phase_gains: dict[str, float] = {}
        # §v10.18: Chain-depth-adaptive scaling. Multi-generation recordings
        # (e.g. reel_tape→vinyl→cassette→mp3_low, depth=4) accumulate
        # frequency-dependent gain loss at each generation. A 6 dB cap is
        # sufficient for single-generation sources but too restrictive for
        # 4-generation chains where cumulative headroom loss exceeds 12 dB.
        self._total_budget_db: float = self._TOTAL_BUDGET_DB

    def configure_for_chain_depth(self, transfer_depth: int, snr_db: float = 30.0, material: str = "unknown") -> None:
        """Scale the total gain budget based on transfer chain depth and SNR.

        Each generation adds ~3 dB of cumulative gain loss:
          depth=1 (single source):  6.0 dB (default)
          depth=2:                   8.0 dB
          depth=3:                  10.0 dB
          depth≥4:                  12.0 dB

        §v10.101 SNR-adaptive scaling: noisy material (SNR < 20 dB) needs
        more headroom because aggressive NR phases request more makeup gain.
        Each 5 dB below 30 dB adds 2 dB to the budget.
          SNR ≥ 30 dB:  ×1.00  (no adjustment)
          SNR = 25 dB:  ×1.33
          SNR = 20 dB:  ×1.67
          SNR = 14 dB:  ×2.13  (cassette with heavy noise)

        Material-specific floor: cassette/tape gets +2 dB extra.

        Called once per pipeline run from UV3/Denker after chain detection.
        """
        # §v10.101: Reset before configuring — singleton accumulates across runs!
        self.reset()

        depth = max(1, int(transfer_depth))
        if depth >= 5:
            self._total_budget_db = 14.0  # §v10.200: 12→14 — extreme chains brauchen mehr
        elif depth >= 4:
            self._total_budget_db = 12.0  # §v10.200: 10→12 — 4-stufige Kassetten brauchen mehr
        elif depth >= 2:
            self._total_budget_db = 8.0
        else:
            self._total_budget_db = 6.0

        # §v10.101 SNR-adaptive scaling
        snr = float(max(1.0, snr_db))
        if snr < 30.0:
            snr_factor = 1.0 + (30.0 - snr) / 15.0
            if str(material).lower() in ("cassette", "reel_tape", "tape"):
                snr_factor += 0.30
            self._total_budget_db = float(max(self._total_budget_db, min(self._total_budget_db * snr_factor, 24.0)))

        logger.info(
            "§GGB-1: chain-depth=%d snr=%.1fdB mat=%s → total budget = %.1f dB",
            depth,
            snr,
            material,
            self._total_budget_db,
        )

    def configure_snr(self, snr_db: float, material: str = "unknown") -> None:
        """§v10.101 Adjust total budget for SNR and material type.

        Noisy material (SNR < 20 dB) needs more headroom because aggressive
        NR phases request more makeup gain. Call AFTER configure_for_chain_depth().
        """
        snr = float(max(1.0, snr_db))
        if snr < 30.0:
            snr_factor = 1.0 + (30.0 - snr) / 15.0
            if str(material).lower() in ("cassette", "reel_tape", "tape"):
                snr_factor += 0.30
            new_budget = float(max(self._total_budget_db, min(self._total_budget_db * snr_factor, 24.0)))
            if new_budget > self._total_budget_db:
                logger.info(
                    "§GGB-1 SNR-adapt: snr=%.1fdB mat=%s → budget %.1f→%.1f dB (×%.2f)",
                    snr,
                    material,
                    self._total_budget_db,
                    new_budget,
                    snr_factor,
                )
                self._total_budget_db = new_budget

    def request(self, phase_id: str, requested_db: float, priority: str = "normal") -> float:
        """Request gain budget for a phase. Returns approved gain in dB.

        Args:
            phase_id: Phase identifier (e.g. "phase_12_wow_flutter_fix").
            requested_db: Requested makeup gain in dB (positive values only).
            priority: "normal", "high", or "best_effort".

        Returns:
            Approved gain in dB (0.0 ≤ returned ≤ requested).
        """
        requested = float(max(0.0, requested_db))
        if requested <= 0.0:
            return 0.0

        with self._lock:
            # best_effort phases get nothing
            if priority == "best_effort":
                logger.debug("§GGB-1: %s best_effort → 0 dB", phase_id)
                return 0.0

            # Per-phase cap (except loudness normalization)
            if phase_id not in self._LOUDNESS_NORM_PHASES:
                requested = min(requested, self._MAX_PER_PHASE_DB)

            # Global cap
            remaining = max(0.0, self._total_budget_db - self._cumulative_db)
            # §v10.303: Bei wiederholten Requests derselben Phase (z.B. Phase 28
            # pro Chunk) Budget gleichmäßig verteilen. Letzte Chunks bekommen
            # sonst 0.15 dB während erste 2.00 dB kriegen → Amplitudenmodulation.
            _prev = self._phase_gains.get(phase_id, 0.0)
            if _prev > 0.0 and remaining < 5.0:
                # Wiederholter Request + knappes Budget → nicht mehr als remaining/4
                requested = min(requested, max(0.1, remaining / 4.0))
            approved = min(requested, remaining)

            self._cumulative_db += approved
            self._phase_gains[phase_id] = approved

            if approved < requested:
                logger.info(
                    "§GGB-1: %s requested %.2f dB → approved %.2f dB (cap: total %.2f/%.2f dB, remaining %.2f dB)",
                    phase_id,
                    requested_db,
                    approved,
                    self._cumulative_db,
                    self._total_budget_db,  # §v10.200 FIX: instance variable, not class constant
                    remaining - approved,
                )

            return approved

    def reset(self) -> None:
        """Reset budget for a new pipeline run."""
        with self._lock:
            self._cumulative_db = 0.0
            self._phase_gains.clear()

    @property
    def cumulative_db(self) -> float:
        with self._lock:
            return self._cumulative_db

    @property
    def remaining_db(self) -> float:
        with self._lock:
            return max(0.0, self._TOTAL_BUDGET_DB - self._cumulative_db)


# Thread-safe singleton
_instance: GlobalGainBudget | None = None
_lock: threading.Lock = threading.Lock()


def get_global_gain_budget() -> GlobalGainBudget:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GlobalGainBudget()
    return _instance
