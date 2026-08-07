"""
ChainOptimizer: Optimiert Reihenfolge und Parameter der Processing-Module für das Material.

Algorithmus: Kostenbasierte Greedy-Optimierung
  - Jedes Modul hat Kosten (compute_cost) und einen geschätzten Qualitätsgewinn (quality_gain)
  - Ziel: Maximale Qualität bei gegebenen Budget-Constraints
  - Strategie: Sortierung nach Quality/Cost-Ratio (Greedy-Knapsack-Näherung)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


_instance: ChainOptimizer | None = None
_lock = threading.Lock()


def get_chain_optimizer(compute_budget: float = 1.0) -> ChainOptimizer:
    """Gibt zurück: or create ChainOptimizer singleton.

    Args:
        compute_budget: Maximum compute budget (only used on first call)

    Returns:
        ChainOptimizer singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ChainOptimizer(compute_budget)
    return _instance


# Kostenmodell: Modulname → (compute_cost, quality_gain, empfohlene_Reihenfolge_Priorität)
_MODULE_PROFILE: dict[str, dict[str, float]] = {
    "noise_reduction": {"cost": 0.30, "gain": 0.80, "priority": 1.0},
    "declip": {"cost": 0.25, "gain": 0.75, "priority": 0.9},
    "declick": {"cost": 0.15, "gain": 0.60, "priority": 0.85},
    "eq": {"cost": 0.10, "gain": 0.50, "priority": 0.7},
    "dynamic_eq": {"cost": 0.20, "gain": 0.65, "priority": 0.75},
    "compressor": {"cost": 0.10, "gain": 0.45, "priority": 0.6},
    "limiter": {"cost": 0.05, "gain": 0.40, "priority": 0.55},
    "enhancer": {"cost": 0.15, "gain": 0.35, "priority": 0.5},
    "stereo_widener": {"cost": 0.10, "gain": 0.30, "priority": 0.4},
    "reverb_reduction": {"cost": 0.35, "gain": 0.70, "priority": 0.8},
    "decrackle": {"cost": 0.20, "gain": 0.65, "priority": 0.88},
    "dehiss": {"cost": 0.25, "gain": 0.70, "priority": 0.87},
    "dehum": {"cost": 0.10, "gain": 0.55, "priority": 0.82},
}

# Kanonische Verarbeitungsreihenfolge (Signal-Flow-Konvention)
_CANONICAL_ORDER = [
    "declip",
    "declick",
    "decrackle",
    "noise_reduction",
    "dehiss",
    "dehum",
    "reverb_reduction",
    "eq",
    "dynamic_eq",
    "compressor",
    "enhancer",
    "stereo_widener",
    "limiter",
]


class ChainOptimizer:
    """
    Optimiert die Reihenfolge der DSP-Module in einem Chain-Template.

    Strategie:
    1. Bekannte Module werden nach kanonischer Signalflussreihenfolge sortiert.
    2. Unbekannte Module werden ans Ende angehängt (Konservativprinzip).
    3. Budget-Constraint: Module mit schlechter Quality/Cost-Ratio werden
       bei Überschreitung des compute_budget entfernt.
    """

    def __init__(self, compute_budget: float = 1.0) -> None:
        """
        Parameters
        ----------
        compute_budget : maximale Gesamtkosten (Summe der compute_cost aller Module).
                         1.0 = unbegrenzt (keine Module entfernen).
        """
        self.compute_budget = compute_budget
        logger.info("ChainOptimizer initialisiert with Grenze=%s", compute_budget)

    def optimize_chain(
        self,
        chain_template: list[Any],
        audio_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Optimiert das Chain-Template.

        Parameters
        ----------
        chain_template  : Liste von Modul-Deskriptoren (str oder dict mit 'name'-Key).
        audio_metadata  : Optionale Audio-Metadaten (material, rms, clipping_ratio …)
                          – für adaptive Parameter-Anpassung.
        Returns
        -------
        Sortiertes (und ggf. gefiltertes) Chain-Template.
        """
        if not chain_template:
            return []

        metadata = audio_metadata or {}

        # Module normalisieren: alle als dict mit 'name' Key
        modules: list[dict[str, Any]] = []
        for item in chain_template:
            if isinstance(item, str):
                modules.append({"name": item})
            elif isinstance(item, dict):
                modules.append(dict(item))
            else:
                modules.append({"name": str(item)})

        # --- Schritt 1: Kanonische Sortierung ---
        def sort_key(mod: dict[str, Any]) -> float:
            name = mod.get("name", "").lower()
            # Exakter Treffer
            if name in _CANONICAL_ORDER:
                return float(_CANONICAL_ORDER.index(name))
            # Teilstring-Treffer
            for idx, canon in enumerate(_CANONICAL_ORDER):
                if canon in name or name in canon:
                    return float(idx) + 0.5
            return float(len(_CANONICAL_ORDER))  # Unbekannt → Ende

        modules.sort(key=sort_key)

        # --- Schritt 2: Budget-Constraint ---
        if self.compute_budget < 1.0:
            # Quality/Cost-Ratio berechnen, niedrige Ratio erst herausnehmen
            def qc_ratio(mod: dict[str, Any]) -> float:
                name = mod.get("name", "").lower()
                prof = _MODULE_PROFILE.get(name, {"cost": 0.1, "gain": 0.3})
                return prof["gain"] / max(prof["cost"], 1e-6)

            modules.sort(key=lambda m: qc_ratio(m), reverse=True)
            total_cost = 0.0
            selected = []
            for mod in modules:
                name = mod.get("name", "").lower()
                cost = _MODULE_PROFILE.get(name, {"cost": 0.1})["cost"]
                if total_cost + cost <= self.compute_budget:
                    selected.append(mod)
                    total_cost += cost
            # Re-sortieren nach kanonischer Reihenfolge
            selected.sort(key=sort_key)
            modules = selected

        # --- Schritt 3: Material-spezifische Parametervariationen ---
        material = str(metadata.get("material", "")).lower()
        clipping_ratio = float(metadata.get("clipping_ratio", 0.0))

        for mod in modules:
            name = mod.get("name", "").lower()
            params = mod.setdefault("params", {})
            if material == "vinyl":
                if "noise" in name or "dehiss" in name:
                    params.setdefault("strength", 0.7)
                if "decrackle" in name or "declick" in name:
                    params.setdefault("threshold", 0.015)
            elif material == "tape":
                if "noise" in name or "dehiss" in name:
                    params.setdefault("strength", 0.5)
                if "eq" in name:
                    params.setdefault("tape_eq_correction", True)
            elif material == "shellac" or material == "78rpm":
                if "noise" in name:
                    params.setdefault("strength", 0.9)
                if "eq" in name:
                    params.setdefault("riaa_correction", "shellac")
            if clipping_ratio > 0.01 and "declip" in name:
                params.setdefault("n_iter", max(10, int(clipping_ratio * 100)))

        # Originale Struktur wiederherstellen (str → str, dict → dict)
        result = []
        for orig, opt in zip(chain_template, modules):
            if isinstance(orig, str):
                result.append(opt.get("name", orig))
            else:
                result.append(opt)

        # Falls Budget Module entfernt hat: result kürzen
        if len(modules) < len(chain_template):
            result = result[: len(modules)]

        return result
