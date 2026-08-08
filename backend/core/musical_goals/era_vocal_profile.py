"""EraVocalProfile — Ära-adaptive VQI-Kalibrierung (§EraVocalProfile, v10.0.0).

Historische Vokalstile haben abweichende Vibrato-Charakteristika, Nasalität und
Formant-Toleranzen. Feste ±2 dB-Grenzen (wie in VQI-Basisversion) erzeugen
falsch-negative Scores für prä-1960-Material und unnötige Recovery-Kaskaden.

Kanonischer Aufruf:
    from backend.core.musical_goals.era_vocal_profile import get_era_vocal_profile
    era_profile = get_era_vocal_profile(era_decade)
    result = compute_vqi(audio_orig, audio_restored, sr, era_profile=era_profile)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EraVocalProfile:
    """Ära-spezifische Toleranzwerte für VQI-Kalibrierung.

    Attributes:
        vibrato_hz_range:        Typischer Vibrato-Frequenz-Bereich dieser Ära (Hz).
        f1_tolerance_db:         F1-Abweichungstoleranz in dB (historisch höher = lockerer).
        f2_f4_tolerance_db:      F2–F4-Abweichungstoleranz in dB.
        nasality_expected:       True wenn Nasalität Stilmittel der Ära ist (kein VQI-Abzug).
        dynamic_range_typical_lu: Typisches Dynamikfenster in LU (für emotional_arc-Proxy).
    """

    vibrato_hz_range: tuple[float, float]
    f1_tolerance_db: float
    f2_f4_tolerance_db: float
    nasality_expected: bool
    dynamic_range_typical_lu: float


#: Kanonische Ären-Profile — Immutable-Konstante, nicht zur Laufzeit ändern.
ERA_VOCAL_PROFILES: dict[str, EraVocalProfile] = {
    # Akustische Trichteraufnahmen — stark gefärbt, hohe Nasalität Stilmittel,
    # Vibrato deutlich über modernem Standard.
    "1900_1925": EraVocalProfile(
        vibrato_hz_range=(5.0, 10.0),
        f1_tolerance_db=4.0,
        f2_f4_tolerance_db=3.5,
        nasality_expected=True,
        dynamic_range_typical_lu=8.0,
    ),
    # Elektrische Aufnahmen + Shellac — verbesserte Übertragung, aber
    # Belcanto/Bariton-Stil mit ausgeprägtem Vibrato noch üblich.
    "1925_1945": EraVocalProfile(
        vibrato_hz_range=(5.5, 9.0),
        f1_tolerance_db=3.5,
        f2_f4_tolerance_db=2.5,
        nasality_expected=True,
        dynamic_range_typical_lu=10.0,
    ),
    # Vinylära — Mikrofon-Nahaufnahmetechnik senkt Nasalität, Vibrato
    # nimmt modernen 5–7 Hz-Standard an.
    "1945_1960": EraVocalProfile(
        vibrato_hz_range=(5.0, 7.5),
        f1_tolerance_db=3.0,
        f2_f4_tolerance_db=2.0,
        nasality_expected=False,
        dynamic_range_typical_lu=12.0,
    ),
    # Rock/Pop-Ära — Mikrofonierungsstil konsolidiert sich auf
    # nah-mikrofonierten Klang mit modernem Vibrato-Profil.
    "1960_1975": EraVocalProfile(
        vibrato_hz_range=(4.5, 7.0),
        f1_tolerance_db=2.5,
        f2_f4_tolerance_db=2.0,
        nasality_expected=False,
        dynamic_range_typical_lu=14.0,
    ),
    # Modern — enge Toleranzen, Vibrato 4–7 Hz wie in bisherigem VQI.
    "1975_plus": EraVocalProfile(
        vibrato_hz_range=(4.0, 7.0),
        f1_tolerance_db=2.0,
        f2_f4_tolerance_db=2.0,
        nasality_expected=False,
        dynamic_range_typical_lu=16.0,
    ),
}


def get_era_vocal_profile(era_decade: int | None) -> EraVocalProfile:
    """Gibt das passende EraVocalProfile für ein Jahrzehnt zurück.

    Args:
        era_decade: Aufnahme-Jahrzehnt (z.B. 1930, 1965). None → modernes Profil.

    Returns:
        EraVocalProfile-Instanz (immer ein gültiges Objekt, nie None).
    """
    if era_decade is None:
        return ERA_VOCAL_PROFILES["1975_plus"]
    d = int(era_decade)
    if d < 1925:
        return ERA_VOCAL_PROFILES["1900_1925"]
    if d < 1945:
        return ERA_VOCAL_PROFILES["1925_1945"]
    if d < 1960:
        return ERA_VOCAL_PROFILES["1945_1960"]
    if d < 1975:
        return ERA_VOCAL_PROFILES["1960_1975"]
    return ERA_VOCAL_PROFILES["1975_plus"]


def resolve_formant_tolerance_db(
    era_decade: int | None = None,
    era_profile: EraVocalProfile | None = None,
    fallback_db: float = 2.0,
) -> float:
    """Gibt die kanonische F1-F4-Rollback-Toleranz für Vocal-Guards zurück.

    Historisches Material darf nicht an einem modernen Fixwert falsch-negativ
    scheitern. Der Guard nutzt deshalb den strengeren modernen Wert nur dann,
    wenn kein EraVocalProfile verfügbar ist.
    """
    profile = era_profile or get_era_vocal_profile(era_decade)
    try:
        tol = max(float(profile.f1_tolerance_db), float(profile.f2_f4_tolerance_db))
    except Exception:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
        tol = float(fallback_db)
    return float(max(1.5, min(4.0, tol)))
