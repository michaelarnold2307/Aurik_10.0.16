"""Expert Mode — Toggle für erweiterte Metriken & Funktionen. Spec v10.206 §10.

Steuert Sichtbarkeit von:
- RT-Faktor, Phase-Timings, CPU-History (Performance)
- LUFS-Delta, Chroma, VQI, Goosebumps (Technische Metriken)
- Phase-Report (Deltas, Übersprungene)
- Export-Chain-Details
- Session-History
- Batch-Übersicht
"""

from __future__ import annotations

import json
import os
from pathlib import Path

EXPERT_CONFIG_PATH = Path.home() / ".aurik" / "expert_mode.json"


class ExpertMode:
    """Singleton: Experten-Modus-Konfiguration."""

    _instance: ExpertMode | None = None

    def __new__(cls) -> ExpertMode:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        self._enabled: bool = False
        self._features: dict[str, bool] = {
            "performance": False,  # RT-Faktor, Phase-Timings
            "technical_metrics": False,  # LUFS, Chroma, VQI, Goosebumps
            "phase_report": False,  # Phase-Deltas, Übersprungene
            "export_chain": False,  # Export-Chain-Details
            "session_history": False,  # Frühere Ergebnisse
            "batch_overview": False,  # Batch-Tabelle
            "spectrum_compare": False,  # Spektrum-Vergleich
        }
        if EXPERT_CONFIG_PATH.is_file():
            try:
                with open(EXPERT_CONFIG_PATH) as f:
                    data = json.load(f)
                self._enabled = data.get("enabled", False)
                self._features.update(data.get("features", {}))
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self) -> None:
        EXPERT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPERT_CONFIG_PATH, "w") as f:
            json.dump({"enabled": self._enabled, "features": self._features}, f, indent=2)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if value:
            for k in self._features:
                self._features[k] = True
        self._save()

    def is_visible(self, feature: str) -> bool:
        """True wenn Feature im Experten-Modus sichtbar."""
        if not self._enabled:
            return False
        return self._features.get(feature, False)

    def toggle_feature(self, feature: str) -> bool:
        """Toggle einzelnes Feature. Gibt neuen Zustand zurück."""
        self._features[feature] = not self._features.get(feature, False)
        self._save()
        return self._features[feature]

    def get_visible_features(self) -> list[str]:
        return [k for k, v in self._features.items() if v]

    def reset(self) -> None:
        self._enabled = False
        for k in self._features:
            self._features[k] = False
        self._save()


def get_expert_mode() -> ExpertMode:
    return ExpertMode()
