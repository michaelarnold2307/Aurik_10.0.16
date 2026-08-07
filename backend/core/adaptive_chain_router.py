"""
AdaptiveChainRouter: Wählt und konfiguriert Verarbeitungsketten basierend auf Forensik-Analyse und Confidence.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


_instance: AdaptiveChainRouter | None = None
_lock = threading.Lock()


def get_adaptive_chain_router(templates: dict[str, list[str]] | None = None) -> AdaptiveChainRouter:
    """Gibt zurück: or create AdaptiveChainRouter singleton.

    Args:
        templates: Chain templates dict (only used on first call)

    Returns:
        AdaptiveChainRouter singleton instance
    """
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = AdaptiveChainRouter(templates or CHAIN_TEMPLATES)
    return _instance


class AdaptiveChainRouter:
    """Wählt und konfiguriert Verarbeitungsketten basierend auf Forensik-Analyse."""

    def __init__(self, templates: dict[str, list[str]]) -> None:
        """Initialisiert den Router mit den gegebenen Ketten-Templates."""
        self.templates = templates
        logger.info("AdaptiveChainRouter initialisiert with %s templates", len(templates))

    def select_chain(self, forensic_report: dict[str, str], confidence: float) -> list[str]:
        """Wählt die optimale Verarbeitungskette basierend auf Forensik-Report und Confidence."""
        material = forensic_report.get("medium_type", "GENERIC").upper()
        chain = self.templates.get(material, self.templates.get("GENERIC", []))
        # Optional: Anpassung der Kette je nach Confidence
        if confidence < 0.6:
            chain = [m for m in chain if m != "Enhancement"]
        return chain

    def configure_modules(
        self, chain: list[str], forensic_report: dict[str, str]
    ) -> dict[str, dict[str, float | str | bool]]:
        """Gibt modul-spezifische Parameter für jede Phase der Kette zurück."""
        config = {}
        for module in chain:
            # Beispiel: Material- und Defekt-spezifische Parameter
            params: dict[str, float | str | bool] = {}
            if module == "Denoiser" and forensic_report.get("defects_detected") == "NOISE_BURST":
                params["strength"] = 0.9
            config[module] = params
        return config


# Beispiel-Templates
CHAIN_TEMPLATES = {
    "VINYL": [
        "DCBlocker",
        "ClickRemover",
        "RumbleFilter",
        "HumRemover",
        "WowFlutterCorrection",
        "Denoiser",
        "Enhancement",
    ],
    "TAPE": [
        "DCBlocker",
        "TapeAzimuthCorrector",
        "PrintThroughRemover",
        "WowFlutterCorrection",
        "HFRestoration",
        "Denoiser",
    ],
    "CD": ["DCBlocker", "CodecArtifactRemover", "PreEchoSuppressor", "SpectralHoleFiller", "Dynamics", "Enhancement"],
    "DIGITAL": ["DCBlocker", "CodecArtifactRemover", "SpectralHoleFiller", "Dynamics", "Enhancement"],
    "GENERIC": ["DCBlocker", "Denoiser", "Enhancement"],
}
