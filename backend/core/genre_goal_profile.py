"""§H: Genre-abhängige Musical-Goal-Prioritäten.

Jedes Genre priorisiert die 15 Musical Goals unterschiedlich, basierend auf
dem, was das menschliche Ohr in diesem Genre als „gut" empfindet.

Profile werden aus GenreClassifier-Ergebnissen geladen und in PMGG sowie
ExzellenzDenker eingespielt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── 15 Musical Goals ─────────────────────────────────────────────────────────
_ALL_GOALS: tuple[str, ...] = (
    "waerme",
    "brillanz",
    "emotionalitaet",
    "groove",
    "artikulation",
    "authentizitaet",
    "transparenz",
    "punch",
    "raeumlichkeit",
    "mikrodynamik",
    "makrodynamik",
    "natuerlichkeit",
    "textverstaendlichkeit",
    "bass_praesenz",
    "hoehen_luft",
)

# ── Genre → Goal-Gewichtung (0.5 = weniger wichtig, 2.0 = doppelt so wichtig) ─
# Basierend auf perzeptiver Musikpsychologie und Hörertests.
_GENRE_PROFILES: dict[str, dict[str, float]] = {
    # Deutscher Schlager: Wärme, Emotionalität, Textverständlichkeit
    "schlager": {
        "waerme": 2.0,
        "emotionalitaet": 2.0,
        "textverstaendlichkeit": 2.0,
        "artikulation": 1.5,
        "authentizitaet": 1.5,
        "groove": 1.3,
        "makrodynamik": 1.0,
        "natuerlichkeit": 1.2,
        "brillanz": 0.8,
        "transparenz": 0.8,
        "punch": 0.7,
        "raeumlichkeit": 0.8,
        "mikrodynamik": 0.9,
        "bass_praesenz": 0.9,
        "hoehen_luft": 0.7,
    },
    # Rock/Punk: Punch, Energie, Durchschlagskraft
    "rock": {
        "punch": 2.0,
        "groove": 1.8,
        "artikulation": 1.7,
        "mikrodynamik": 1.5,
        "makrodynamik": 1.5,
        "transparenz": 1.3,
        "brillanz": 1.4,
        "raeumlichkeit": 1.2,
        "waerme": 1.0,
        "emotionalitaet": 1.0,
        "authentizitaet": 1.0,
        "natuerlichkeit": 0.9,
        "textverstaendlichkeit": 0.9,
        "bass_praesenz": 1.5,
        "hoehen_luft": 1.1,
    },
    # Pop: Brillanz, Transparenz, Textverständlichkeit
    "pop": {
        "brillanz": 1.8,
        "transparenz": 1.8,
        "textverstaendlichkeit": 1.8,
        "punch": 1.5,
        "artikulation": 1.4,
        "groove": 1.4,
        "bass_praesenz": 1.6,
        "hoehen_luft": 1.4,
        "waerme": 1.2,
        "emotionalitaet": 1.1,
        "raeumlichkeit": 1.2,
        "authentizitaet": 1.0,
        "mikrodynamik": 0.9,
        "makrodynamik": 0.9,
        "natuerlichkeit": 1.0,
    },
    # Jazz: Authentizität, Räumlichkeit, Mikrodynamik
    "jazz": {
        "authentizitaet": 2.0,
        "raeumlichkeit": 2.0,
        "mikrodynamik": 2.0,
        "natuerlichkeit": 1.8,
        "transparenz": 1.6,
        "artikulation": 1.5,
        "waerme": 1.4,
        "emotionalitaet": 1.3,
        "groove": 1.3,
        "brillanz": 1.1,
        "makrodynamik": 1.2,
        "punch": 0.7,
        "textverstaendlichkeit": 0.7,
        "bass_praesenz": 1.2,
        "hoehen_luft": 1.0,
    },
    # Klassik: Makrodynamik, Authentizität, Natürlichkeit
    "classical": {
        "makrodynamik": 2.0,
        "authentizitaet": 2.0,
        "natuerlichkeit": 2.0,
        "raeumlichkeit": 1.8,
        "mikrodynamik": 1.8,
        "transparenz": 1.5,
        "waerme": 1.2,
        "hoehen_luft": 1.2,
        "brillanz": 1.0,
        "emotionalitaet": 1.0,
        "artikulation": 1.0,
        "groove": 0.6,
        "punch": 0.5,
        "textverstaendlichkeit": 0.7,
        "bass_praesenz": 1.0,
    },
    # Electronic/Dance: Bass-Präsenz, Punch, Groove
    "electronic": {
        "bass_praesenz": 2.0,
        "punch": 2.0,
        "groove": 2.0,
        "hoehen_luft": 1.7,
        "brillanz": 1.6,
        "transparenz": 1.4,
        "raeumlichkeit": 1.5,
        "mikrodynamik": 1.2,
        "waerme": 0.9,
        "emotionalitaet": 0.8,
        "artikulation": 1.0,
        "authentizitaet": 0.7,
        "makrodynamik": 1.0,
        "natuerlichkeit": 0.7,
        "textverstaendlichkeit": 0.7,
    },
    # Folk/Acoustic: Natürlichkeit, Authentizität, Wärme
    "folk": {
        "natuerlichkeit": 2.0,
        "authentizitaet": 2.0,
        "waerme": 1.8,
        "artikulation": 1.6,
        "textverstaendlichkeit": 1.6,
        "emotionalitaet": 1.4,
        "raeumlichkeit": 1.3,
        "transparenz": 1.2,
        "brillanz": 1.0,
        "groove": 1.0,
        "mikrodynamik": 1.1,
        "makrodynamik": 1.0,
        "punch": 0.6,
        "bass_praesenz": 0.8,
        "hoehen_luft": 0.8,
    },
    # Metal: Punch, Aggression, Bass-Präsenz
    "metal": {
        "punch": 2.0,
        "bass_praesenz": 1.9,
        "artikulation": 1.7,
        "groove": 1.6,
        "mikrodynamik": 1.4,
        "makrodynamik": 1.4,
        "transparenz": 1.3,
        "brillanz": 1.2,
        "waerme": 0.8,
        "emotionalitaet": 0.9,
        "authentizitaet": 0.9,
        "natuerlichkeit": 0.7,
        "textverstaendlichkeit": 0.7,
        "raeumlichkeit": 1.0,
        "hoehen_luft": 1.1,
    },
    # Hip-Hop: Bass-Präsenz, Textverständlichkeit, Groove
    "hiphop": {
        "bass_praesenz": 2.0,
        "textverstaendlichkeit": 2.0,
        "groove": 2.0,
        "punch": 1.8,
        "artikulation": 1.4,
        "transparenz": 1.2,
        "raeumlichkeit": 1.1,
        "hoehen_luft": 1.2,
        "waerme": 1.0,
        "brillanz": 1.0,
        "emotionalitaet": 0.9,
        "authentizitaet": 0.8,
        "mikrodynamik": 0.8,
        "makrodynamik": 0.8,
        "natuerlichkeit": 0.8,
    },
    # R&B/Soul: Wärme, Emotionalität, Groove
    "rnb": {
        "waerme": 2.0,
        "emotionalitaet": 2.0,
        "groove": 1.8,
        "bass_praesenz": 1.6,
        "artikulation": 1.5,
        "textverstaendlichkeit": 1.5,
        "mikrodynamik": 1.3,
        "transparenz": 1.2,
        "brillanz": 1.1,
        "punch": 1.1,
        "raeumlichkeit": 1.1,
        "authentizitaet": 1.0,
        "natuerlichkeit": 1.0,
        "makrodynamik": 1.0,
        "hoehen_luft": 1.0,
    },
}

# Fallback: gleichmäßige Gewichtung
_FALLBACK_PROFILE: dict[str, float] = dict.fromkeys(_ALL_GOALS, 1.0)


@dataclass
class GenreGoalProfile:
    """Gewichtetes Goal-Profil für ein erkanntes Genre."""

    genre: str
    weights: dict[str, float] = field(default_factory=dict)

    def weight_for(self, goal: str) -> float:
        """Goal-Gewicht (0.5–2.0) für dieses Genre, 1.0 bei unbekanntem Goal."""
        return self.weights.get(goal, 1.0)

    def apply_to_thresholds(self, base_thresholds: dict[str, float]) -> dict[str, float]:
        """Passt PMGG-Schwellwerte an das Genre an.

        Höhere Gewichtung → strengerer Schwellwert (muss besser sein).
        Niedrigere Gewichtung → toleranterer Schwellwert.
        """
        adjusted: dict[str, float] = {}
        for goal, base in base_thresholds.items():
            w = self.weight_for(goal)
            # Formel: threshold = base × (1.0 + (1.0 - w) × 0.3)
            # Bei w=2.0: threshold = base × 0.7 (strenger, muss 30% besser sein)
            # Bei w=0.5: threshold = base × 1.15 (toleranter, 15% mehr erlaubt)
            adjusted[goal] = base * (1.0 + (1.0 - w) * 0.3)
        return adjusted

    def apply_to_strength(self, goal: str, base_strength: float) -> float:
        """Passt Phasen-Stärke an Genre-Priorität an.

        Höhere Gewichtung → mehr Stärke (genauer arbeiten).
        Niedrigere Gewichtung → weniger Stärke (nicht übertreiben).
        """
        w = self.weight_for(goal)
        # Formel: strength = base × (0.7 + 0.3 × w)
        # Bei w=2.0: strength = base × 1.3
        # Bei w=0.5: strength = base × 0.85
        return base_strength * (0.7 + 0.3 * w)


def get_genre_profile(genre_label: str) -> GenreGoalProfile:
    """Lädt das Goal-Profil für ein erkanntes Genre."""
    genre_lower = genre_label.lower().replace(" ", "_").replace("-", "_")

    # Fuzzy-Matching: prüfe Teilstrings
    for profile_key, weights in _GENRE_PROFILES.items():
        if profile_key in genre_lower or genre_lower in profile_key:
            return GenreGoalProfile(genre=profile_key, weights=dict(weights))

    # Kein Match → Fallback (gleichmäßig)
    logger.info("§H: Kein Genre-Profil für '%s', verwende Ersatzpfad", genre_label)
    return GenreGoalProfile(genre="unknown", weights=dict(_FALLBACK_PROFILE))


def get_priority_weight(genre_label: str, goal: str) -> float:
    """Einzelne Goal-Gewichtung abfragen – leichter Einstiegspunkt."""
    return get_genre_profile(genre_label).weight_for(goal)


def get_all_genre_profiles() -> dict[str, dict[str, float]]:
    """Alle Profile für Diagnose/Audit."""
    return dict(_GENRE_PROFILES)
