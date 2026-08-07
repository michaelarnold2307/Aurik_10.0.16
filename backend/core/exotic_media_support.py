"""
ExoticMediaSupport: Erweiterte Unterstützung für seltene Medien und Defekte.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

EXOTIC_MEDIA_TEMPLATES = {
    "SCHELLACK": ["DCBlocker", "CrackleRemover", "RumbleFilter", "HumRemover", "DeClicker", "Denoiser", "Enhancement"],
    "VINYL": ["DCBlocker", "CrackleRemover", "RumbleFilter", "HumRemover", "DeClicker", "Denoiser", "Enhancement"],
    "MINIDISC": ["ATRACArtifactRemover", "SpectralHoleFiller", "DCBlocker", "Dynamics", "Enhancement"],
    "DAT": ["DCBlocker", "DropoutRemover", "HFRestoration", "Denoiser", "Enhancement"],
    "WIRE": ["DCBlocker", "WowFlutterCorrection", "NoiseGate", "Denoiser", "Enhancement"],
    "CYLINDER": ["DCBlocker", "CrackleRemover", "RumbleFilter", "NoiseGate", "Denoiser", "Enhancement"],
}

EXOTIC_DEFECTS = [
    "CRACKLE",
    "DROP_OUT",
    "PRE_ECHO",
    "ATRAC_ARTIFACT",
    "CYLINDER_WOW",
    "WIRE_NOISE",
    "SCHELLACK_DISTORTION",
    "VINYL_WARP",
]


class ExoticMediaHandler:
    def __init__(self, templates: dict[str, list[str]], defects: list[str]) -> None:
        self.templates = templates
        self.defects = defects
        logger.info("ExoticMediaHandler initialisiert")

    def get_chain_for_media(self, media_type: str) -> list[str]:
        """Gibt zurück: processing chain for exotic media type.

        Args:
            media_type: Media type name

        Returns:
            List of processing module names
        """
        return self.templates.get(media_type.upper(), [])

    def detect_exotic_defects(self, analysis: dict[str, Any]) -> list[str]:
        """Erkennt exotic defects from analysis.

        Args:
            analysis: Defect analysis dict

        Returns:
            List of detected exotic defect names
        """
        found = []
        for defect in self.defects:
            if defect in analysis.get("defects_detected", ""):
                found.append(defect)
        return found
