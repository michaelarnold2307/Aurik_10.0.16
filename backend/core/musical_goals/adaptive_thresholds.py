"""
Adaptive Thresholds für Genre/Medium-Specific Musical Goals.

Passt Musical Goals Thresholds basierend auf Genre, Medium und musikalischem
Kontext an für kontextbewusste Musical Goals Guarantee.

Component 0.9.4: Genre & Medium-Specific Calibration
Impact: +1 Punkt - Kontextbewusste Garantie

Beispiele:
- Jazz Vocals: Natürlichkeit=0.95, Wärme=0.90 (hohe Standards)
- Rock Drums: Bass-Kraft=0.90, Transparenz=0.80 (punch wichtig)
- Classical: Alle Goals=0.92+ (höchste Ansprüche)
- Vinyl: Wärme=0.92, Bass-Kraft=0.88 (Charakter-Erhalt)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThresholdProfile:
    """
    Threshold profile für spezifischen context.

    Attributes:
        name: Profile name
        thresholds: Dict von goal -> threshold value
        description: Beschreibung des Profiles
        applies_to: Welche contexts apply (genre, medium, etc.)
    """

    name: str
    thresholds: dict[str, float]
    description: str
    applies_to: dict[str, Any]

    def __repr__(self) -> str:
        return f"ThresholdProfile({self.name})"


class AdaptiveThresholdsManager:
    """
    Genre/Medium-spezifische Musical Goals Thresholds.

    Features:
    - Base Thresholds (Default)
    - Genre-Specific Rules (10+ Genres)
    - Medium-Specific Rules (Vinyl/Tape/Shellac/Digital)
    - Instrument-Specific Rules (Vocals/Drums/etc.)
    - Empirical Validation

    Workflow:
    1. Start mit Base Thresholds
    2. Apply Medium Adjustments
    3. Apply Genre Adjustments
    4. Apply Instrument Focus Adjustments
    5. Validate & Clamp to valid range
    6. Return Adjusted Thresholds
    """

    # Base thresholds (default für alle contexts)
    BASE_THRESHOLDS = {
        "bass-kraft": 0.85,
        "brillanz": 0.85,
        "waerme": 0.80,
        "natuerlichkeit": 0.90,
        "authentizitaet": 0.88,
        "emotionalitaet": 0.87,
        "transparenz": 0.89,
    }

    # Medium-specific adjustments (additive)
    MEDIUM_ADJUSTMENTS = {
        "vinyl": {
            "waerme": +0.07,
            "authentizitaet": +0.05,
            "bass-kraft": +0.03,
            "brillanz": -0.03,
            "natuerlichkeit": +0.02,
        },
        "tape": {
            "waerme": +0.10,
            "natuerlichkeit": +0.03,
            "authentizitaet": +0.05,
            "brillanz": -0.05,
            "transparenz": -0.02,
        },
        "shellac": {
            "authentizitaet": +0.10,
            "natuerlichkeit": +0.08,
            "waerme": +0.05,
            "brillanz": -0.10,
            "transparenz": -0.05,
            "emotionalitaet": +0.03,
        },
        "digital": {"transparenz": +0.05, "brillanz": +0.03, "natuerlichkeit": +0.02},
        "cd": {"transparenz": +0.04, "brillanz": +0.02, "natuerlichkeit": +0.02},
    }

    # Genre-specific adjustments
    GENRE_ADJUSTMENTS = {
        "classical": {
            "natuerlichkeit": +0.10,
            "authentizitaet": +0.10,
            "transparenz": +0.08,
            "emotionalitaet": +0.05,
            "bass-kraft": -0.05,
            "brillanz": -0.03,
        },
        "jazz": {
            "natuerlichkeit": +0.08,
            "waerme": +0.10,
            "transparenz": +0.05,
            "emotionalitaet": +0.07,
            "authentizitaet": +0.07,
        },
        "rock": {
            "bass-kraft": +0.10,
            "emotionalitaet": +0.08,
            "brillanz": +0.05,
            "natuerlichkeit": -0.05,
            "transparenz": +0.03,
        },
        "pop": {"brillanz": +0.08, "emotionalitaet": +0.05, "transparenz": +0.05, "bass-kraft": +0.03},
        "electronic": {
            "bass-kraft": +0.12,
            "brillanz": +0.08,
            "transparenz": +0.07,
            "natuerlichkeit": -0.10,
            "authentizitaet": -0.08,
        },
        "folk": {"natuerlichkeit": +0.12, "waerme": +0.10, "authentizitaet": +0.10, "emotionalitaet": +0.08},
        "blues": {"waerme": +0.12, "emotionalitaet": +0.10, "natuerlichkeit": +0.05, "bass-kraft": +0.08},
        "country": {"natuerlichkeit": +0.10, "waerme": +0.08, "authentizitaet": +0.08, "transparenz": +0.05},
        "metal": {
            "bass-kraft": +0.15,
            "brillanz": +0.10,
            "emotionalitaet": +0.12,
            "transparenz": +0.05,
            "waerme": -0.05,
        },
        "hip-hop": {"bass-kraft": +0.15, "brillanz": +0.08, "transparenz": +0.08, "natuerlichkeit": -0.05},
    }

    # Instrument focus adjustments
    INSTRUMENT_ADJUSTMENTS = {
        "vocals": {
            "authentizitaet": +0.05,
            "natuerlichkeit": +0.05,
            "waerme": +0.03,
            "transparenz": +0.05,
            "bass-kraft": -0.05,
        },
        "drums": {"bass-kraft": +0.10, "transparenz": +0.08, "emotionalitaet": +0.05, "waerme": -0.05},
        "bass": {"bass-kraft": +0.12, "waerme": +0.05, "transparenz": +0.05, "natuerlichkeit": +0.03},
        "strings": {"waerme": +0.10, "emotionalitaet": +0.08, "natuerlichkeit": +0.08, "transparenz": +0.05},
        "brass": {"brillanz": +0.08, "emotionalitaet": +0.08, "transparenz": +0.05, "natuerlichkeit": +0.05},
        "piano": {"transparenz": +0.10, "natuerlichkeit": +0.08, "brillanz": +0.05, "emotionalitaet": +0.05},
        "guitar": {"waerme": +0.08, "natuerlichkeit": +0.08, "transparenz": +0.05, "emotionalitaet": +0.05},
    }

    def __init__(self, profiles_path: Path | None = None) -> None:
        """
        Initialisiert Adaptive Thresholds Manager.

        Args:
            profiles_path: Optional path to custom profiles JSON
        """
        self.profiles_path = profiles_path
        self.custom_profiles: list[ThresholdProfile] = []

        if profiles_path and profiles_path.exists():
            self._load_custom_profiles()

        self.calibration_count = 0

    def get_thresholds(
        self,
        genre: str | None = None,
        medium_type: str | None = None,
        instrument_focus: str | None = None,
        custom_adjustments: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Gibt context-specific thresholds zurück.

        Args:
            genre: Genre name ('classical', 'jazz', 'rock', etc.)
            medium_type: Medium type ('vinyl', 'tape', 'shellac', 'digital')
            instrument_focus: Primary instrument ('vocals', 'drums', etc.)
            custom_adjustments: Additional custom adjustments

        Returns:
            Dict[goal_name, threshold] - Context-specific thresholds
        """
        self.calibration_count += 1

        # Start with base thresholds
        thresholds = self.BASE_THRESHOLDS.copy()

        # Apply medium adjustments
        if medium_type:
            self._apply_adjustments(
                thresholds, self.MEDIUM_ADJUSTMENTS.get(medium_type.lower(), {}), f"medium={medium_type}"
            )

        # Apply genre adjustments
        if genre:
            self._apply_adjustments(thresholds, self.GENRE_ADJUSTMENTS.get(genre.lower(), {}), f"genre={genre}")

        # Apply instrument focus adjustments
        if instrument_focus:
            self._apply_adjustments(
                thresholds,
                self.INSTRUMENT_ADJUSTMENTS.get(instrument_focus.lower(), {}),
                f"instrument={instrument_focus}",
            )

        # Apply custom adjustments
        if custom_adjustments:
            self._apply_adjustments(thresholds, custom_adjustments, "custom")

        # Validate and clamp to valid range [0.0, 1.0]
        for goal in thresholds:
            thresholds[goal] = np.clip(thresholds[goal], 0.0, 1.0)

        logger.info("kalibriert thresholds: genre=%s, medium=%s, instrument=%s", genre, medium_type, instrument_focus)

        return thresholds

    def _apply_adjustments(self, thresholds: dict[str, float], adjustments: dict[str, float], context: str):
        """Wendet Schwellwert-Anpassungen an."""
        for goal, adjustment in adjustments.items():
            if goal in thresholds:
                thresholds[goal] += adjustment
                logger.debug("angewendet %s adjustment: %s %+.3f", context, goal, adjustment)

    def get_profile(
        self, genre: str | None = None, medium_type: str | None = None, instrument_focus: str | None = None
    ) -> ThresholdProfile:
        """
        Get complete threshold profile für context.

        Returns:
            ThresholdProfile with thresholds and metadata
        """
        thresholds = self.get_thresholds(genre, medium_type, instrument_focus)

        name_parts = []
        if genre:
            name_parts.append(genre.title())
        if medium_type:
            name_parts.append(medium_type.title())
        if instrument_focus:
            name_parts.append(instrument_focus.title())

        name = " / ".join(name_parts) if name_parts else "Default"

        description = f"Thresholds für {name}"

        return ThresholdProfile(
            name=name,
            thresholds=thresholds,
            description=description,
            applies_to={"genre": genre, "medium_type": medium_type, "instrument_focus": instrument_focus},
        )

    def create_custom_profile(
        self,
        name: str,
        thresholds: dict[str, float],
        description: str = "",
        applies_to: dict[str, Any] | None = None,
    ) -> ThresholdProfile:
        """
        Erstellt custom threshold profile.

        Args:
            name: Profile name
            thresholds: Complete thresholds dict
            description: Profile description
            applies_to: Context where profile applies

        Returns:
            ThresholdProfile
        """
        # Validate thresholds
        for goal, value in thresholds.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Threshold {goal}={value} out of range [0, 1]")

        profile = ThresholdProfile(
            name=name,
            thresholds=thresholds,
            description=description or f"Custom profile: {name}",
            applies_to=applies_to or {},
        )

        self.custom_profiles.append(profile)

        logger.info("erstellt custom Profil: %s", name)

        return profile

    def save_custom_profiles(self, path: Path | None = None):
        """Speichert custom profiles to JSON file."""
        save_path = path or self.profiles_path or Path("data/threshold_profiles.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        profiles_data = []
        for profile in self.custom_profiles:
            profiles_data.append(
                {
                    "name": profile.name,
                    "thresholds": profile.thresholds,
                    "description": profile.description,
                    "applies_to": profile.applies_to,
                }
            )

        with open(save_path, "w") as f:
            json.dump(profiles_data, f, indent=2)

        logger.info("gespeichert %s custom profiles to %s", len(profiles_data), save_path)

    def _load_custom_profiles(self):
        """Lädt custom profiles from JSON file."""
        path = self.profiles_path
        if path is None or not path.exists():
            return

        with open(path) as f:
            profiles_data = json.load(f)

        for data in profiles_data:
            profile = ThresholdProfile(
                name=data["name"],
                thresholds=data["thresholds"],
                description=data.get("description", ""),
                applies_to=data.get("applies_to", {}),
            )
            self.custom_profiles.append(profile)

        logger.info("geladen %s custom profiles", len(profiles_data))

    def get_all_genres(self) -> list[str]:
        """Gibt zurück: list of all supported genres."""
        return sorted(self.GENRE_ADJUSTMENTS.keys())

    def get_all_medium_types(self) -> list[str]:
        """Gibt zurück: list of all supported medium types."""
        return sorted(self.MEDIUM_ADJUSTMENTS.keys())

    def get_all_instruments(self) -> list[str]:
        """Gibt zurück: list of all supported instrument focuses."""
        return sorted(self.INSTRUMENT_ADJUSTMENTS.keys())

    def compare_profiles(self, profile1: dict[str, Any], profile2: dict[str, Any]) -> dict[str, float]:
        """
        Compare two threshold profiles.

        Args:
            profile1: First profile context dict
            profile2: Second profile context dict

        Returns:
            Dict[goal, difference] - Threshold differences
        """
        thresholds1 = self.get_thresholds(**profile1)
        thresholds2 = self.get_thresholds(**profile2)

        differences = {}
        for goal in thresholds1:
            differences[goal] = thresholds2[goal] - thresholds1[goal]

        return differences

    def get_statistics(self) -> dict[str, Any]:
        """Gibt zurück: calibration statistics."""
        return {
            "calibration_count": self.calibration_count,
            "custom_profiles": len(self.custom_profiles),
            "supported_genres": len(self.GENRE_ADJUSTMENTS),
            "supported_mediums": len(self.MEDIUM_ADJUSTMENTS),
            "supported_instruments": len(self.INSTRUMENT_ADJUSTMENTS),
            "base_thresholds": self.BASE_THRESHOLDS.copy(),
        }
