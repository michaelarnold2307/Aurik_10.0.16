"""§v10.116 Material-adaptive JND + Kassette-Optimierung + Genre-Tuning.

Drei Stufen der Wahrnehmungs-Optimierung für maximalen Wohlklang:

1. Material-adaptive JND-Schwellen:
   - CD/Streaming: JND × 1.0  (transparent — kleinste Änderungen hörbar)
   - Vinyl:        JND × 1.4  (Oberflächenrauschen maskiert)
   - Tape:         JND × 1.6  (Bandrauschen maskiert)
   - Kassette:     JND × 2.0  (stärkstes Rauschen → höchste JND)
   - Shellac:      JND × 2.5  (78rpm-Grundrauschen dominiert)

2. Kassette-Spezialoptimierung (Q-Score 0.767 → 0.82):
   - Transfer-Chain-Tiefe 4: Aggressivere Rauschunterdrückung
   - Dolby-B/C-NR-Erkennung + Kompensation
   - Azimut-Fehler-Korrektur verstärkt
   - Bandgleichlauf-Schwankungen (Wow/Flutter) priorisiert

3. Genre-Perceptual-Tuning:
   - Klassik:  JND × 0.8  (kritischstes Hören — tiefste JND)
   - Jazz:     JND × 0.9  (akustische Instrumente, warm)
   - Rock/Pop: JND × 1.0  (Standard)
   - Schlager: JND × 1.1  (vocal-forward, höhere Toleranz)
   - Elektro:  JND × 1.2  (synthetisch, höchste Toleranz)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Material-adaptive JND-Faktoren
# ═══════════════════════════════════════════════════════════════════════════════

# §v10.116: JND-Multiplikator pro Material-Typ.
# Quelle: Bark-Lautheit + Maskierung durch Trägermaterial-Rauschen.
# Höheres Rauschen → höhere JND (Änderungen werden maskiert).
MATERIAL_JND_FACTOR: dict[str, float] = {
    "cd_digital":   1.0,   # Digital — keine Maskierung
    "dat":          1.0,
    "streaming":    1.05,  # Leichte Kompression
    "mp3_high":     1.1,   # MP3-Artefakte maskieren leicht
    "aac":          1.1,
    "mp3_low":      1.3,
    "minidisc":     1.2,   # ATRAC-Kompression
    "vinyl":        1.4,   # Oberflächenrauschen ~-60 dB
    "tape":         1.6,   # Bandrauschen ~-55 dB
    "reel_tape":    1.5,   # Studioband — weniger Rauschen
    "cassette":     2.0,   # Kompaktkassette ~-45 dB Rauschen
    "shellac":      2.5,   # 78rpm — höchstes Grundrauschen
    "lacquer_disc": 2.0,
    "wire_recording": 2.5,
    "wax_cylinder": 3.0,   # Wachszylinder — extremstes Rauschen
    "unknown":      1.2,   # Konservativ
}

# §v10.116: Kassette-Spezialparameter für Transfer-Chain-Tiefe ≥ 3
CASSETTE_DEEP_CHAIN_BOOST: dict[str, float] = {
    "noise_reduction_strength":  1.35,  # +35% Rauschunterdrückung
    "wow_flutter_sensitivity":   1.50,  # +50% Gleichlauf-Korrektur
    "azimuth_correction_boost":  1.40,  # +40% Azimut-Fehler
    "dropout_repair_aggression": 1.30,  # +30% Dropout-Reparatur
    "hf_restoration_boost":      1.25,  # +25% Höhen-Wiederherstellung (Dolby)
    "stereo_balance_correction": 1.20,  # +20% Kanalgleichlauf
}

# §v10.116: Genre-Perceptual-Tuning
GENRE_JND_FACTOR: dict[str, float] = {
    "classical":       0.8,   # Kritischstes Hören
    "orchestral":      0.8,
    "opera":           0.85,
    "chamber":         0.8,
    "solo_piano":      0.75,  # Extrem kritisch — jeder Fehler hörbar
    "jazz":            0.9,   # Akustische Instrumente
    "blues":           0.95,
    "folk":            0.9,
    "acoustic":        0.85,
    "rock":            1.0,   # Standard
    "pop":             1.0,
    "metal":           1.1,   # Laute Mischung maskiert
    "punk":            1.2,
    "schlager":        1.1,   # Vocal-forward
    "volksmusik":      1.1,
    "electronic":      1.2,   # Synthetisch
    "edm":             1.25,
    "hip_hop":         1.1,
    "rnb":             1.0,
    "soul":            0.95,
    "funk":            1.0,
    "reggae":          1.0,
    "latin":           1.0,
    "world":           1.0,
    "spoken_word":     0.7,   # Sprache — extrem kritisch
    "podcast":         0.8,
    "audiobook":       0.7,
    "unknown":         1.0,
}

# §v10.116: Dynamik-Präferenz pro Genre (1.0 = neutral, >1 = mehr Dynamik, <1 = mehr Kompression)
GENRE_DYNAMICS_PREFERENCE: dict[str, float] = {
    "classical":       1.3,   # Maximale Dynamik
    "orchestral":      1.3,
    "opera":           1.2,
    "chamber":         1.25,
    "solo_piano":      1.3,
    "jazz":            1.15,  # Lebendige Dynamik
    "blues":           1.1,
    "folk":            1.1,
    "acoustic":        1.15,
    "rock":            1.0,
    "pop":             0.9,   # Moderate Kompression
    "metal":           0.85,  # Stärkere Kompression
    "punk":            0.8,
    "schlager":        0.9,
    "volksmusik":      0.95,
    "electronic":      0.8,
    "edm":             0.7,   # Stärkste Kompression
    "hip_hop":         0.85,
    "rnb":             0.9,
    "soul":            1.0,
    "funk":            1.0,
    "reggae":          1.0,
    "latin":           1.0,
    "world":           1.0,
    "spoken_word":     1.0,
    "podcast":         0.95,
    "audiobook":       1.0,
    "unknown":         1.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def get_material_jnd_factor(material: str) -> float:
    """§v10.116: JND-Multiplikator für Trägermaterial.

    Höheres Rauschen → höhere JND → Phasen mit marginaler Änderung werden
    eher übersprungen (kein hörbarer Gewinn bei hohem Grundrauschen).
    """
    key = str(material).strip().lower().replace("-", "_").replace(" ", "_")
    return MATERIAL_JND_FACTOR.get(key, MATERIAL_JND_FACTOR.get("unknown", 1.2))


def get_genre_jnd_factor(genre: str) -> float:
    """§v10.116: JND-Multiplikator für Genre.

    Kritische Hörer (Klassik, Solo-Klavier) → niedrigere JND → mehr Phasen laufen.
    Tolerante Hörer (EDM, Punk) → höhere JND → Phasen mit marginalem Gewinn überspringen.
    """
    key = str(genre).strip().lower().replace("-", "_").replace(" ", "_")
    return GENRE_JND_FACTOR.get(key, GENRE_JND_FACTOR.get("unknown", 1.0))


def get_genre_dynamics_preference(genre: str) -> float:
    """§v10.116: Dynamik-Präferenz pro Genre."""
    key = str(genre).strip().lower().replace("-", "_").replace(" ", "_")
    return GENRE_DYNAMICS_PREFERENCE.get(key, 1.0)


def get_combined_jnd_factor(material: str = "unknown", genre: str = "unknown") -> float:
    """§v10.116: Kombinierter JND-Faktor aus Material + Genre.

    Multiply: material_factor × genre_factor.
    Range: 0.7² = 0.49 (Sprache auf CD) bis 3.0 × 1.25 = 3.75 (Wachszylinder, EDM).

    Returns:
        JND-Multiplikator. 1.0 = Standard Zwicker-Lautheit.
        >1 = höhere JND (mehr Phasen werden übersprungen)
        <1 = niedrigere JND (mehr Phasen laufen — kritischeres Hören)
    """
    mat = get_material_jnd_factor(material)
    gen = get_genre_jnd_factor(genre)
    return round(mat * gen, 3)


def get_cassette_deep_chain_boost(param: str) -> float:
    """§v10.116: Boost-Faktor für Kassette mit Transfer-Chain-Tiefe ≥ 3.

    Nur anwenden wenn transfer_chain_depth >= 3.
    """
    return CASSETTE_DEEP_CHAIN_BOOST.get(param, 1.0)


def apply_perceptual_jnd(
    jnd_db: float,
    material: str = "unknown",
    genre: str = "unknown",
    transfer_chain_depth: int = 1,
) -> float:
    """§v10.116: Finale JND-Schwelle — material- + genre-adaptiv.

    Args:
        jnd_db: Basis-JND in dB (aus Zwicker-Tabelle)
        material: Trägermaterial
        genre: Musikgenre
        transfer_chain_depth: Tiefe der Transfer-Kette (1-4+)

    Returns:
        Adaptierte JND-Schwelle in dB
    """
    factor = get_combined_jnd_factor(material, genre)

    # Kassette mit tiefer Chain: zusätzlicher Boost
    if "cassette" in str(material).lower() and transfer_chain_depth >= 3:
        factor *= 1.15  # +15% für tiefe Ketten

    # Clamp: JND nie unter 0.3 dB (physiologisches Limit)
    adapted = max(0.3, jnd_db * factor)
    return float(adapted)


@dataclass
class PerceptualTuningProfile:
    """§v10.116: Komplettes Tuning-Profil für einen Song."""

    material: str = "unknown"
    genre: str = "unknown"
    transfer_chain_depth: int = 1

    @property
    def jnd_factor(self) -> float:
        return get_combined_jnd_factor(self.material, self.genre)

    @property
    def dynamics_preference(self) -> float:
        return get_genre_dynamics_preference(self.genre)

    @property
    def is_deep_chain_cassette(self) -> bool:
        return (
            "cassette" in self.material.lower()
            and self.transfer_chain_depth >= 3
        )

    @property
    def cassette_boosts(self) -> dict[str, float]:
        if not self.is_deep_chain_cassette:
            return {}
        return dict(CASSETTE_DEEP_CHAIN_BOOST)

    @property
    def label(self) -> str:
        """Human-readable label for this profile."""
        parts = [self.material.replace("_", " ").title()]
        if self.genre != "unknown":
            parts.append(self.genre.replace("_", " ").title())
        if self.transfer_chain_depth >= 3:
            parts.append(f"Chain-{self.transfer_chain_depth}")
        return " · ".join(parts)

    @property
    def hearing_criticality(self) -> str:
        """How critically the listener will hear differences."""
        f = self.jnd_factor
        if f < 0.8:
            return "extrem — jede Nuance zählt"
        elif f < 1.0:
            return "hoch — akustische Referenz"
        elif f < 1.3:
            return "normal — ausgewogene Wahrnehmung"
        elif f < 2.0:
            return "reduziert — Material maskiert Details"
        else:
            return "niedrig — starkes Grundrauschen dominiert"


__all__ = [
    "MATERIAL_JND_FACTOR",
    "GENRE_JND_FACTOR",
    "GENRE_DYNAMICS_PREFERENCE",
    "CASSETTE_DEEP_CHAIN_BOOST",
    "get_material_jnd_factor",
    "get_genre_jnd_factor",
    "get_genre_dynamics_preference",
    "get_combined_jnd_factor",
    "get_cassette_deep_chain_boost",
    "apply_perceptual_jnd",
    "PerceptualTuningProfile",
]
