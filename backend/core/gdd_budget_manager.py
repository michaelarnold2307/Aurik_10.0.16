"""
backend/core/gdd_budget_manager.py — §v10.650 GDD-Budget-Manager (§18.2)

Präventives Gruppenlaufzeit-Budget: Verteilt das kumulative GDD-Budget
(STFT-Group-Delay-Deviation) pro Phase, statt reaktiv nach Überschreitung
die gesamte Pipeline zu reverten.

Jede STFT-Phase fordert VOR ihrer Ausführung ein Budget an. Wenn das Budget
erschöpft ist, wird die Stärke reduziert statt dass die Phase normal läuft
und dann der CIG (CumulativeInteractionGuard) einen Rollback auslöst.

Architektur:
  1. Bei Pipeline-Init: total_budget aus Material + Chain-Depth
  2. Vor jeder STFT-Phase: allocate(phase_id) → erlaubte GDD (ms)
  3. Nach jeder STFT-Phase: consume(phase_id, actual_gdd_ms)
  4. Bei Budget-Erschöpfung: Stärke dämpfen statt Phase zu reverten

§v10.650 integriert mit §18.2 (Non-Plus-Ultra Perceptual Fidelity).

Status: ✅ Implementiert §v10.650 GDD-Budget-Lücke
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)


# ── Material-adaptive Gesamtbudgets (§18.2 Tabelle) ──
_MATERIAL_GDD_BUDGETS: dict[str, dict[str, float]] = {
    "shellac": {"total_ms": 8.0, "per_phase_cap_ms": 4.0},
    "wax_cylinder": {"total_ms": 8.0, "per_phase_cap_ms": 4.0},
    "lacquer_disc": {"total_ms": 10.0, "per_phase_cap_ms": 5.0},
    "vinyl": {"total_ms": 10.0, "per_phase_cap_ms": 5.0},
    "lp": {"total_ms": 10.0, "per_phase_cap_ms": 5.0},
    "cassette": {"total_ms": 13.2, "per_phase_cap_ms": 6.0},
    "kassette": {"total_ms": 13.2, "per_phase_cap_ms": 6.0},
    "reel_tape": {"total_ms": 15.0, "per_phase_cap_ms": 7.0},
    "tape": {"total_ms": 13.2, "per_phase_cap_ms": 6.0},
    "wire_recording": {"total_ms": 12.0, "per_phase_cap_ms": 5.5},
    "minidisc": {"total_ms": 10.0, "per_phase_cap_ms": 5.0},
    "dat": {"total_ms": 5.0, "per_phase_cap_ms": 3.0},
    "cd": {"total_ms": 5.0, "per_phase_cap_ms": 3.0},
    "cd_digital": {"total_ms": 5.0, "per_phase_cap_ms": 3.0},
    "digital": {"total_ms": 5.0, "per_phase_cap_ms": 3.0},
    "streaming": {"total_ms": 7.0, "per_phase_cap_ms": 4.0},
    "mp3_low": {"total_ms": 10.0, "per_phase_cap_ms": 5.0},
    "mp3_high": {"total_ms": 8.0, "per_phase_cap_ms": 4.0},
    "aac": {"total_ms": 8.0, "per_phase_cap_ms": 4.0},
}

# Chain-Depth-Multiplikator: tiefere Ketten vertragen mehr GDD
_DEPTH_BUDGET_MULTIPLIER: dict[int, float] = {
    1: 1.0,
    2: 1.2,
    3: 1.4,
    4: 1.7,
    5: 2.0,
}

_DEFAULT_BUDGET = {"total_ms": 10.0, "per_phase_cap_ms": 5.0}


class GddBudgetManager:
    """Verteilt und überwacht das kumulative GDD-Budget.

    Nutzung:
        gdd = GddBudgetManager(material="cassette", chain_depth=4)
        # Vor Phase:
        budget_ms = gdd.allocate("phase_03")
        if budget_ms < 1.0:
            strength *= 0.25  # Near-passthrough
        # Nach Phase:
        gdd.consume("phase_03", actual_gdd_ms=4.2)
    """

    _STFT_PHASE_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "stft",
            "istft",
            "denoise",
            "declick",
            "decrackle",
            "spectral",
            "azimuth",
            "deharsh",
            "wow",
            "flutter",
        }
    )

    def __init__(
        self,
        material: str = "vinyl",
        chain_depth: int = 1,
    ) -> None:
        _mat = str(material or "vinyl").lower()
        _cfg = _MATERIAL_GDD_BUDGETS.get(_mat, _DEFAULT_BUDGET)
        _depth = max(1, min(5, int(chain_depth or 1)))
        _mult = _DEPTH_BUDGET_MULTIPLIER.get(_depth, 1.0)

        self.material = _mat
        self.chain_depth = _depth
        self.total_budget_ms = float(_cfg["total_ms"]) * _mult
        self.per_phase_cap_ms = float(_cfg["per_phase_cap_ms"])
        self.remaining_ms = self.total_budget_ms
        self.consumed_ms: float = 0.0
        self.phase_history: list[dict] = []

        logger.info(
            "§v10.650 GDD-Grenze-Manager: material=%s depth=%d Grenze=%.1f ms per_Verarbeitungsschritt_cap=%.1f ms",
            _mat,
            _depth,
            self.total_budget_ms,
            self.per_phase_cap_ms,
        )

    def allocate(self, phase_id: str) -> float:
        """Gibt das verbleibende GDD-Budget für diese Phase zurück (ms).

        Returns 0.0 wenn kein Budget mehr verfügbar → Phase sollte
        mit minimaler Stärke laufen oder übersprungen werden.
        """
        if self.remaining_ms <= 0.0:
            logger.debug(
                "§GDD-Grenze %s: Grenze erschöpft (%.1f/%.1f ms) → ueberspringen",
                phase_id,
                self.consumed_ms,
                self.total_budget_ms,
            )
            return 0.0

        # Pro-Phase-Cap: max(self.per_phase_cap_ms, remaining)
        _allocated = min(self.per_phase_cap_ms, self.remaining_ms)

        logger.debug(
            "§GDD-Grenze %s: allocate %.1f ms (remaining=%.1f/%.1f ms)",
            phase_id,
            _allocated,
            self.remaining_ms,
            self.total_budget_ms,
        )
        return _allocated

    def consume(self, phase_id: str, actual_gdd_ms: float) -> bool:
        """Verbraucht GDD-Budget nach der Phase.

        Returns:
            True wenn Budget ausreicht, False wenn Budget überschritten —
            die AKTUELLE Phase sollte ihre Stärke reduzieren.
        """
        _actual = max(0.0, float(actual_gdd_ms))
        _before = self.remaining_ms
        self.remaining_ms = max(0.0, self.remaining_ms - _actual)
        self.consumed_ms += min(_actual, _before)  # Nur was wirklich vom Budget abgeht

        _ok = _actual <= _before
        self.phase_history.append(
            {
                "phase": phase_id,
                "gdd_ms": round(_actual, 2),
                "budget_ok": _ok,
                "remaining_ms": round(self.remaining_ms, 2),
            }
        )

        if not _ok:
            logger.warning(
                "§GDD-Grenze %s: Grenze überschritten! %.1f ms verbraucht, nur %.1f ms verfügbar → Stärke dämpfen",
                phase_id,
                _actual,
                _before,
            )
        else:
            logger.debug(
                "§GDD-Grenze %s: consumed %.1f ms (%.1f/%.1f ms remaining)",
                phase_id,
                _actual,
                self.remaining_ms,
                self.total_budget_ms,
            )
        return _ok

    @staticmethod
    def is_stft_phase(phase_id: str) -> bool:
        """Prüft ob eine Phase STFT-basierte Verarbeitung nutzt."""
        _lower = str(phase_id or "").lower()
        return any(kw in _lower for kw in GddBudgetManager._STFT_PHASE_KEYWORDS)

    def remaining(self) -> float:
        """Verbleibendes Gesamtbudget in ms."""
        return self.remaining_ms

    def summary(self) -> dict:
        """GDD-Budget-Zusammenfassung für Logs/Metadaten."""
        return {
            "material": self.material,
            "chain_depth": self.chain_depth,
            "total_budget_ms": round(self.total_budget_ms, 1),
            "consumed_ms": round(self.consumed_ms, 1),
            "remaining_ms": round(self.remaining_ms, 1),
            "phases_tracked": len(self.phase_history),
            "budget_exceeded": any(not p["budget_ok"] for p in self.phase_history),
        }
