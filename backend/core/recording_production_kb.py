"""
RecordingProductionKB — Aufnahme-Produktions-Wissensbasis (v10.0.0.x)
====================================================================

Schließt Lücke 2 der Provenienz-Analyse: Ohne Kenntnis des spezifischen
Studios lassen sich aus Audio-Merkmalen (Raumcharakter, Kompressionscharakter,
Mikrofon-Wärme) deutlich spezifischere Produktionsprofile ableiten als
nur aus Ära/Genre/Material allein.

Methodisches Vorgehen:
    1. ``detect_production_signature(audio, sr)`` → ``ProductionSignature``
       - Raumcharakter via RT60-Schätzung (Schroeder-Integration)
       - Kompressionscharakter via Crest-Factor-Varianz + Pumping-Detektion
       - Mikrofon-Wärme via Spektralneigung im 2–6-kHz-Bereich

    2. ``get_production_profile(era, genre, material, sig)`` → ``ProductionProfile``
       - Lookup in 3-stufiger KB:
         - Stufe 1: Exakter Match (era_bucket × genre × room_char × compression)
         - Stufe 2: Partieller Match (era_bucket × genre)
         - Stufe 3: Era-Material-Fallback

    3. ``ProductionProfile.goal_adjustments`` werden in
       ``calibration_matrix.estimate_song_goal_targets()`` als 4. Bias-Schicht
       oben aufgelegt (nach era/material/genre).

Historische Basis:
    - Eargle, J. (2004). The Microphone Book. Focal Press.
    - Cunningham, M. (1996). Good Vibrations: A History of Record Production. Sanctuary.
    - Moorefield, V. (2005). The Producer as Composer. MIT Press.
    - Massey, H. (2009). Behind the Glass, Vol. 2. Backbeat Books.
    - Chanan, M. (1995). Repeated Takes. Verso.
    - Nielsen, S. (1987). The Leq(m) weighting filter. JAES 35(3).

Autor: Aurik Development Team
Datum: Mai 2026
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ProductionSignature — gemessene Audio-Eigenschaften
# ---------------------------------------------------------------------------


@dataclass
class ProductionSignature:
    """Messergebnis der Produktionssignatur-Erkennung."""

    room_character: str = "unknown"
    """Raumcharakter:
    - "dry_studio"       RT60 < 0.25 s  — Toter Raum, Nahfeld-Aufnahme
    - "intimate_studio"  0.25–0.60 s   — Kleines Studio (Rudy Van Gelder, Atlantic)
    - "large_studio"     0.60–1.20 s   — Großes Studio (Capitol Tower, Abbey Road A)
    - "echo_chamber"     nicht-lineare Abkling-Kurve + Präsenz-Resonanzen
    - "live_venue"       RT60 > 1.20 s — Konzerthalle, Kirche
    - "unknown"          Schätzung nicht möglich (zu kurzes Audio, extremes Rauschen)
    """

    compression_character: str = "natural"
    """Kompressionscharakter:
    - "dry"              Crest-Factor-Varianz hoch, kein Pumping — kein Kompressor
    - "natural"          Leichter Kompressor, keine Artefakte (LA-2A leicht)
    - "heavy"            Stark komprimiert, geringe Crest-Factor-Varianz (UA 1176 all-in)
    - "limited"          Hard-Limiting sichtbar, Crest-Factor-Spitzen beschnitten
    """

    mic_warmth: str = "neutral"
    """Mikrofon-Wärme (Spektralneigung 2–6 kHz):
    - "warm"       Absenkung > 3 dB in 4–6 kHz (Ribbon-Mikrofon: Coles 4038, RCA 44)
    - "neutral"    Flache Kurve ±2 dB (Neumann U47 moderat, AKG C12)
    - "presence"   Anhebung > 3 dB in 3–5 kHz (Neumann U87 "Presence-Peak", SM7B)
    """

    rt60_s: float = 0.0
    """Geschätzter RT60 (s). 0.0 = nicht schätzbar."""

    crest_factor_mean: float = 12.0
    """Mittlerer Crest-Factor (dB). Hoch = wenig Kompression."""

    mic_type: str = "unknown"
    """§Lücke6 Erkannter Mikrofon-Typ (aus `microphone_character.py`).
    Werte: "Neumann U47", "RCA 44-BX", "Crystal Mic", "Shure SM7B", "unknown", ...
    Schutz-EQ-Parameter über `mic_signature`-Feld verfügbar.
    """

    mic_signature: object = None
    """§Lücke6 Vollständiges `MicrophoneSignature`-Objekt (aus microphone_character.py).
    Enthält protect_proximity_below_hz, protect_presence_band, protect_rolloff_above_hz.
    Für phase_04 als EQ-Schutzbereich verwendbar.
    """


# ---------------------------------------------------------------------------
# ProductionProfile — abgeleitetes Profil mit Restaurierungsanpassungen
# ---------------------------------------------------------------------------


@dataclass
class ProductionProfile:
    """Produktionsprofil: Ergebnis der KB-Abfrage.

    Alle ``goal_adjustments`` werden ADDITIV zu den bestehenden
    era/material/genre-Biases in ``estimate_song_goal_targets()`` aufgelegt.
    """

    profile_name: str = "generic"
    """Identifikationsname des Profils (für Logging)."""

    goal_adjustments: dict[str, float] = field(default_factory=dict)
    """Additive Zielwert-Korrekturen je Ziel. Beispiel:
    {"raumtiefe": +0.06, "natuerlichkeit": +0.04}
    Werden durch kappa_provenance (0.30) skaliert (konservativ).
    """

    vocal_protection_level: str = "standard"
    """Vokal-Schutz-Stufe:
    - "strict"   (Klassik/Oper): F1–F4 Rollback-Toleranz ±1 dB (statt ±2)
    - "standard" (Jazz/Schlager): Spec-Standard ±2 dB
    - "relaxed"  (Rock/Pop):     ±3 dB, weniger Vibrato-Schutz
    """

    preserve_room: bool = False
    """True → Phase_49 (Dereverb) deaktiviert; Raum ist künstlerische Absicht."""

    compression_artifact_type: str = "none"
    """Typ der Kompressions-Artefakte für phase_47:
    - "none"           kein Handling nötig
    - "tape_saturation" Tape-Sättigung (H2/H4 bewahren)
    - "pumping"        LRA-Wiederherstellung erforderlich
    - "limiting"       Clipping-Repair (phase_01 bevorzugen)
    """

    era_note: str = ""
    """Menschenlesbare Notiz zum historischen Kontext (Logging only)."""


# ---------------------------------------------------------------------------
# Produktions-KB — Kern-Wissensbasis
# ---------------------------------------------------------------------------
# Struktur: dict[(era_bucket, genre_key, room_char, compression_char)] → ProductionProfile
#
# era_bucket: "1920s", "1930s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s+", "any"
# genre_key:  "jazz", "klassik", "schlager", "pop", "rock", "soul", "folk", "any"
# room_char:  "dry_studio", "intimate_studio", "large_studio", "echo_chamber", "live_venue", "any"
# compression_char: "dry", "natural", "heavy", "limited", "any"
# ---------------------------------------------------------------------------

_PRODUCTION_KB: dict[tuple[str, str, str, str], ProductionProfile] = {
    # -----------------------------------------------------------------------
    # 1920s–1930s Akustisch/Elektrisch — sehr frühe Aufnahmen
    # -----------------------------------------------------------------------
    ("1920s", "any", "any", "dry"): ProductionProfile(
        profile_name="acoustic_era_no_compression",
        goal_adjustments={
            "natuerlichkeit": +0.08,  # Akustische Aufnahme ohne Elektronik = natürlicher Klang
            "authentizitaet": +0.10,  # Historische Authentizität hat höchste Priorität
            "raumtiefe": -0.12,  # Trichteraufnahme: kein Raumgefühl
            "transparenz": -0.16,  # Hornresonanz färbt das Spektrum stark
        },
        vocal_protection_level="strict",
        preserve_room=False,
        compression_artifact_type="none",
        era_note="Akustische Trichteraufnahme 1920–1925: Hornresonanz bei 600–900 Hz ist authentisch",
    ),
    ("1930s", "jazz", "intimate_studio", "dry"): ProductionProfile(
        profile_name="1930s_jazz_close_mic",
        goal_adjustments={
            "natuerlichkeit": +0.06,
            "waerme": +0.06,  # RCA/Columbia Early-Electric: nahes Mikrofon = Nähe-Effekt
            "raumtiefe": -0.08,
        },
        vocal_protection_level="strict",
        preserve_room=False,
        compression_artifact_type="none",
        era_note="Early Electric Jazz (Brunswick/Columbia): RCA 44 Ribbon, kein Kompressor",
    ),
    # -----------------------------------------------------------------------
    # 1940s — Krieg/Nachkrieg
    # -----------------------------------------------------------------------
    ("1940s", "jazz", "intimate_studio", "natural"): ProductionProfile(
        profile_name="1940s_jazz_studio",
        goal_adjustments={
            "natuerlichkeit": +0.06,
            "waerme": +0.08,  # Tube-Preamps (UTC/Pultec) färben warm
            "raumtiefe": -0.04,
            "transparenz": -0.06,
        },
        vocal_protection_level="strict",
        preserve_room=False,
        compression_artifact_type="tape_saturation",
        era_note="Jazz 1940er: Ahuja/Altec 21/639 Ribbon, Tube-Preamps, frühe Bandaufzeichnung",
    ),
    ("1940s", "klassik", "large_studio", "dry"): ProductionProfile(
        profile_name="1940s_classical_large_studio",
        goal_adjustments={
            "raumtiefe": +0.14,  # Großer Raum ist Teil der Aufführung
            "natuerlichkeit": +0.10,
            "mikrodynamik": +0.08,  # Klassik: Dynamik ist expressiv
            "transparenz": -0.04,
        },
        vocal_protection_level="strict",
        preserve_room=True,  # Raumklang ist Absicht
        compression_artifact_type="none",
        era_note="Klassik 1940er: NBC/CBS Studio, Neumann CMV 3/CMV 5 Kondensator",
    ),
    # -----------------------------------------------------------------------
    # 1950s — Goldenes Zeitalter der Vinyl-Aufnahme
    # -----------------------------------------------------------------------
    ("1950s", "jazz", "intimate_studio", "natural"): ProductionProfile(
        profile_name="1950s_jazz_van_gelder",
        goal_adjustments={
            "raumtiefe": +0.12,  # RVG Hackensack/Englewood Cliffs: tiefer Holzboden-Raum
            "natuerlichkeit": +0.08,
            "waerme": +0.06,  # Neumann M49/Telefunken ELA M 251: warme Röhre
            "mikrodynamik": +0.06,
            "brillanz": -0.04,  # Blue Note 1950s: HF-Rolloff durch Tube-Chain
        },
        vocal_protection_level="strict",
        preserve_room=True,  # Van Gelder Raumklang ist Markenzeichen
        compression_artifact_type="none",
        era_note="Blue Note/Prestige 1950er (Rudy Van Gelder): M49 + Fairchild 660, Hackensack Studio",
    ),
    ("1950s", "jazz", "dry_studio", "heavy"): ProductionProfile(
        profile_name="1950s_jazz_west_coast",
        goal_adjustments={
            "natuerlichkeit": +0.04,
            "transparenz": +0.06,  # West Coast: trockenerer, klarerer Sound
            "waerme": +0.04,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="natural",
        era_note="West Coast Jazz 1950er (Capitol/Contemporary): trocken, klar — Capitol Tower Echo Chamber",
    ),
    ("1950s", "schlager", "large_studio", "natural"): ProductionProfile(
        profile_name="1950s_schlager_electrola",
        goal_adjustments={
            "natuerlichkeit": +0.04,
            "transparenz": -0.06,  # Electrola/Polydor: Hall-Kammer (EMT 140 Platte)
            "raumtiefe": +0.08,  # Plattenreverb ist authentisch
            "waerme": +0.06,
        },
        vocal_protection_level="standard",
        preserve_room=True,  # EMT 140 Platten-Hall: Teil der Ästhetik
        compression_artifact_type="none",
        era_note="Schlager 1950er (Electrola/Polydor): Studioraum Berlin/Köln, EMT 140 Platten-Hall",
    ),
    ("1950s", "klassik", "large_studio", "dry"): ProductionProfile(
        profile_name="1950s_classical_abbey_road",
        goal_adjustments={
            "raumtiefe": +0.16,  # Abbey Road Studio 1: RT60 ~2.5s (Kirche-ähnlich)
            "natuerlichkeit": +0.12,
            "mikrodynamik": +0.10,
            "waerme": +0.04,
            "brillanz": -0.06,
        },
        vocal_protection_level="strict",
        preserve_room=True,
        compression_artifact_type="none",
        era_note="Klassik 1950er (EMI/HMV Abbey Road): Neumann M49, Studer A80, Room ~2.5s RT60",
    ),
    ("1950s", "soul", "intimate_studio", "heavy"): ProductionProfile(
        profile_name="1950s_rbs_chess_records",
        goal_adjustments={
            "waerme": +0.10,  # Chess Records Chicago: naher Mikrofon, warmer Sound
            "natuerlichkeit": +0.06,
            "groove": +0.06,
            "transparenz": -0.08,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="tape_saturation",
        era_note="R&B/Blues 1950er (Chess Records Chicago): Altec 21D, Ampex 350, Tube-Kompressor",
    ),
    # -----------------------------------------------------------------------
    # 1960s — Stereo-Ära
    # -----------------------------------------------------------------------
    ("1960s", "jazz", "intimate_studio", "natural"): ProductionProfile(
        profile_name="1960s_jazz_impulse",
        goal_adjustments={
            "raumtiefe": +0.10,
            "natuerlichkeit": +0.06,
            "transparenz": +0.04,
            "waerme": +0.04,
        },
        vocal_protection_level="strict",
        preserve_room=True,
        compression_artifact_type="none",
        era_note="Impulse!/Columbia Jazz 1960er: Rupert Neve Konsole, Neumann U67, frühe Stereomikrofonie",
    ),
    ("1960s", "schlager", "large_studio", "natural"): ProductionProfile(
        profile_name="1960s_schlager_polydor",
        goal_adjustments={
            "raumtiefe": +0.10,  # Großorchester + EMT 140 Reverb
            "transparenz": -0.04,
            "waerme": +0.08,  # Röhren-Mischpulte noch vorhanden
            "natuerlichkeit": +0.04,
        },
        vocal_protection_level="standard",
        preserve_room=True,
        compression_artifact_type="none",
        era_note="Schlager 1960er (Polydor/Ariola): Großorchester, EMT 140, REMS-Studio Hamburg",
    ),
    ("1960s", "pop", "large_studio", "natural"): ProductionProfile(
        profile_name="1960s_pop_motown",
        goal_adjustments={
            "groove": +0.08,  # Motown Rhythm Section: präzise, eng
            "natuerlichkeit": +0.04,
            "waerme": +0.06,
            "transparenz": -0.04,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="tape_saturation",
        era_note="Motown 1960er (Hitsville USA): Neve 8014, close-mic'd Rhythm Section, Snare-Cardboard",
    ),
    ("1960s", "pop", "echo_chamber", "heavy"): ProductionProfile(
        profile_name="1960s_pop_capitol_tower",
        goal_adjustments={
            "raumtiefe": +0.12,  # Capitol Tower Echo Chamber: charakteristischer Sound
            "waerme": +0.04,
            "natuerlichkeit": +0.04,
        },
        vocal_protection_level="standard",
        preserve_room=True,  # Echo Chamber ist Markenzeichen
        compression_artifact_type="limiting",
        era_note="Pop 1960er (Capitol Records): Capitol Tower Echo Chamber, Fairchild 670 Limiting",
    ),
    ("1960s", "klassik", "live_venue", "dry"): ProductionProfile(
        profile_name="1960s_classical_concert_hall",
        goal_adjustments={
            "raumtiefe": +0.18,  # Konzerthalle: größtes Raumsignal in der KB
            "natuerlichkeit": +0.14,
            "mikrodynamik": +0.12,
        },
        vocal_protection_level="strict",
        preserve_room=True,
        compression_artifact_type="none",
        era_note="Klassik 1960er Live: DG/Decca Kirchenaufnahmen, Neumann SM 23, AB-Mikrofoniertechnik",
    ),
    # -----------------------------------------------------------------------
    # 1970s — Analogband-Höhepunkt
    # -----------------------------------------------------------------------
    ("1970s", "jazz", "intimate_studio", "natural"): ProductionProfile(
        profile_name="1970s_jazz_ecm",
        goal_adjustments={
            "raumtiefe": +0.14,  # ECM Records: Raumklang ist Identität (Manfred Eicher)
            "natuerlichkeit": +0.08,
            "transparenz": +0.06,
            "brillanz": +0.04,
        },
        vocal_protection_level="strict",
        preserve_room=True,
        compression_artifact_type="none",
        era_note="ECM Jazz 1970er: Oslo Tonstudio, Studer A80, Neumann U87, sehr langer RT60 (~1.5s)",
    ),
    ("1970s", "soul", "large_studio", "heavy"): ProductionProfile(
        profile_name="1970s_soul_philadelphia",
        goal_adjustments={
            "waerme": +0.06,
            "groove": +0.08,
            "natuerlichkeit": +0.04,
            "transparenz": -0.04,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="tape_saturation",
        era_note="Philadelphia Soul 1970er (Sigma Sound): Neve 8028, lush String-Arrangements, Studer A80",
    ),
    ("1970s", "rock", "large_studio", "heavy"): ProductionProfile(
        profile_name="1970s_rock_studio",
        goal_adjustments={
            "groove": +0.06,
            "bass_kraft": +0.08,
            "natuerlichkeit": +0.02,
        },
        vocal_protection_level="relaxed",
        preserve_room=False,
        compression_artifact_type="tape_saturation",
        era_note="Rock 1970er: Record Plant/AIR London, API 1604, SSL 4000 ab 1975, Tube-Saturation",
    ),
    ("1970s", "schlager", "large_studio", "heavy"): ProductionProfile(
        profile_name="1970s_schlager_germany",
        goal_adjustments={
            "raumtiefe": +0.08,
            "waerme": +0.04,
            "natuerlichkeit": +0.04,
        },
        vocal_protection_level="standard",
        preserve_room=True,
        compression_artifact_type="tape_saturation",
        era_note="Schlager 1970er (Hansa/Intercord Berlin): Vollmer-Konsole, EMT 250 Hall, Studer A80",
    ),
    # -----------------------------------------------------------------------
    # 1980s — Digital/Analog-Übergang
    # -----------------------------------------------------------------------
    ("1980s", "pop", "large_studio", "limited"): ProductionProfile(
        profile_name="1980s_pop_ssl_era",
        goal_adjustments={
            "transparenz": +0.06,
            "brillanz": +0.06,
            "natuerlichkeit": -0.04,  # SSL G-Bus-Kompressor: charakteristischer Klang
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="limiting",
        era_note="Pop 1980er (SSL G-Series): Power Station/Criteria, SSL 4000 G-Bus, Lexicon 480L Hall",
    ),
    ("1980s", "schlager", "large_studio", "heavy"): ProductionProfile(
        profile_name="1980s_schlager_synthesizer",
        goal_adjustments={
            "transparenz": +0.04,
            "natuerlichkeit": -0.04,  # Synthesizer-Orchestrierung klingt weniger natürlich
            "raumtiefe": +0.04,  # Lexicon 224/480L sehr präsent
        },
        vocal_protection_level="standard",
        preserve_room=True,
        compression_artifact_type="limiting",
        era_note="Schlager 1980er: Synthesizer-Arrangements, Lexicon 224 Hall, Neve VR Series",
    ),
    ("1980s", "jazz", "intimate_studio", "natural"): ProductionProfile(
        profile_name="1980s_jazz_contemporary",
        goal_adjustments={
            "natuerlichkeit": +0.06,
            "transparenz": +0.04,
            "raumtiefe": +0.06,
        },
        vocal_protection_level="strict",
        preserve_room=True,
        compression_artifact_type="none",
        era_note="Jazz 1980er (GRP/Verve): Digital Recording, sauberer Klang, Neve 8078/Focusrite ISA",
    ),
    # -----------------------------------------------------------------------
    # 1990s+ — Digital-Ära
    # -----------------------------------------------------------------------
    ("1990s+", "pop", "any", "heavy"): ProductionProfile(
        profile_name="1990s_pop_loudness_war_early",
        goal_adjustments={
            "natuerlichkeit": -0.04,
            "mikrodynamik": -0.06,  # Loudness War beginnt: DR-Verlust
            "brillanz": +0.04,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="limiting",
        era_note="Pop 1990er: Pro Tools DAW, multiband-Limiting, frühe Loudness-War-Phase",
    ),
    ("1990s+", "jazz", "intimate_studio", "natural"): ProductionProfile(
        profile_name="1990s_jazz_acoustic",
        goal_adjustments={
            "natuerlichkeit": +0.06,
            "transparenz": +0.06,
            "raumtiefe": +0.06,
        },
        vocal_protection_level="strict",
        preserve_room=True,
        compression_artifact_type="none",
        era_note="Jazz 1990er (Nonesuch/ECM): sorgfältige Mikrofonierung, Pro Tools, geringer Kompressor",
    ),
    # -----------------------------------------------------------------------
    # Era-Material-Fallbacks (wenn kein spezifischer Match)
    # -----------------------------------------------------------------------
    ("1920s", "any", "any", "any"): ProductionProfile(
        profile_name="acoustic_era_fallback",
        goal_adjustments={
            "authentizitaet": +0.08,
            "natuerlichkeit": +0.06,
            "raumtiefe": -0.10,
            "transparenz": -0.12,
        },
        vocal_protection_level="strict",
        preserve_room=False,
        compression_artifact_type="none",
        era_note="Akustische Ära 1920er: kein elektronischer Kompressor, Hornresonanz dominant",
    ),
    ("1930s", "any", "any", "any"): ProductionProfile(
        profile_name="early_electric_fallback",
        goal_adjustments={
            "authentizitaet": +0.06,
            "waerme": +0.06,
            "raumtiefe": -0.06,
        },
        vocal_protection_level="strict",
        preserve_room=False,
        compression_artifact_type="none",
        era_note="Frühe Elektrische Aufnahme 1930er: Ribbon-Mikrofone, Tube-Preamps",
    ),
    ("1940s", "any", "any", "any"): ProductionProfile(
        profile_name="wartime_postwar_fallback",
        goal_adjustments={
            "waerme": +0.06,
            "authentizitaet": +0.06,
        },
        vocal_protection_level="strict",
        preserve_room=False,
        compression_artifact_type="tape_saturation",
        era_note="Kriegs-/Nachkriegszeit 1940er: frühe Bandaufzeichnung, Tube-Sättigung",
    ),
    ("1950s", "any", "any", "any"): ProductionProfile(
        profile_name="golden_age_vinyl_fallback",
        goal_adjustments={
            "waerme": +0.06,
            "natuerlichkeit": +0.04,
            "authentizitaet": +0.06,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="none",
        era_note="Goldenes Zeitalter Vinyl 1950er: Neumann M49/U47, Fairchild-Kompressor",
    ),
    ("1960s", "any", "any", "any"): ProductionProfile(
        profile_name="stereo_era_fallback",
        goal_adjustments={
            "natuerlichkeit": +0.04,
            "waerme": +0.04,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="none",
        era_note="Stereo-Ära 1960er: Neumann U67/U87, Neve Konsolen",
    ),
    ("1970s", "any", "any", "any"): ProductionProfile(
        profile_name="analog_peak_fallback",
        goal_adjustments={
            "waerme": +0.04,
            "groove": +0.04,
        },
        vocal_protection_level="standard",
        preserve_room=False,
        compression_artifact_type="tape_saturation",
        era_note="Analog-Höhepunkt 1970er: SSL/Neve Konsolen, Studer A80 Bandmaschine",
    ),
}

# ---------------------------------------------------------------------------
# Lookup-Prioritätsliste (spezifisch → generisch)
# ---------------------------------------------------------------------------
# Tuple: (era_bucket, genre_key, room_char, compression_char)
# "any" matcht alles; Priorität: mehr spezifische Keys gewinnen.


def _score_key(
    key: tuple[str, str, str, str],
    era: str,
    genre: str,
    room: str,
    compression: str,
) -> int:
    """Berechnet Spezifizitäts-Score (höher = besser)."""
    score = 0
    if key[0] == era:
        score += 8
    elif key[0] != "any":
        return -1  # kein Match
    if key[1] == genre:
        score += 4
    elif key[1] != "any":
        return -1  # kein Match
    if key[2] == room:
        score += 2
    elif key[2] != "any":
        return -1  # kein Match
    if key[3] == compression:
        score += 1
    elif key[3] != "any":
        return -1  # kein Match
    return score


# ---------------------------------------------------------------------------
# Produktionssignatur-Detektion
# ---------------------------------------------------------------------------


def detect_production_signature(audio: np.ndarray, sr: int) -> ProductionSignature:
    """Erkennt Produktionssignatur aus Audio via DSP.

    Nicht-blockierend: alle Ausnahmen → sichere Defaults.
    Zeitbudget: ≤ 1.5 s für 30 s Stereo (48 kHz).

    Args:
        audio: Mono oder Stereo float32, normiert auf [-1, 1].
        sr:    Abtastrate (Hz).

    Returns:
        ProductionSignature mit room_character, compression_character, mic_warmth.
    """
    sig = ProductionSignature()
    try:
        # Mono-Konvertierung
        if audio.ndim > 1:
            mono: np.ndarray = audio.mean(axis=-1 if audio.shape[-1] <= 2 else 0).astype(np.float32)
        else:
            mono = audio.astype(np.float32)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        # Mindest-Signallänge: 2 s
        if len(mono) < 2 * sr:
            return sig

        # --- 1. Raumcharakter via RT60-Schätzung ---
        sig.room_character, sig.rt60_s = _estimate_room_character(mono, sr)

        # --- 2. Kompressionscharakter ---
        sig.compression_character, sig.crest_factor_mean = _estimate_compression_character(mono, sr)

        # --- 3. Mikrofon-Wärme via Spektralneigung 2–6 kHz ---
        sig.mic_warmth = _estimate_mic_warmth(mono, sr)

        # --- 4. §Lücke6 Mikrofon-Charakter-Erkennung ---
        try:
            from backend.core.dsp.microphone_character import (  # pylint: disable=import-outside-toplevel
                detect_microphone_character,
            )

            _mic_sig = detect_microphone_character(mono, sr)
            sig.mic_type = _mic_sig.detected_mic
            sig.mic_signature = _mic_sig
        except Exception as _mc_exc:
            logger.debug("RecordingProductionKB §Lücke6 MicChar: Ersatzpfad — %s", _mc_exc)

        logger.info(
            "RecordingProductionKB: room=%s (RT60=%.2fs) compression=%s mic=%s",
            sig.room_character,
            sig.rt60_s,
            sig.compression_character,
            sig.mic_warmth,
        )
    except Exception as exc:
        logger.debug("erkennen_production_signature Ersatzpfad: %s", exc)
    return sig


def _estimate_room_character(mono: np.ndarray, sr: int) -> tuple[str, float]:
    """Schätzt Raumcharakter via vereinfachter Schroeder-Integration.

    Methodik: Energie-Abklingkurve aus letztem Drittel des Signals
    (nach Direktsignal). RT60 = Zeit von 0 dB auf -60 dB auf der Kurve.
    Bei Shellac/Vinyl: Carrier-Rauschen überdeckt Raumhall — RT60 oft 0.

    Returns:
        (room_character_str, rt60_seconds)
    """
    # Analyse-Segment: letztes Drittel (ab 2/3 der Dauer) — reduziert Direktsignal-Einfluss
    _n = len(mono)
    _seg = mono[int(_n * 0.67) :]
    if len(_seg) < int(0.5 * sr):
        return "unknown", 0.0

    # Schroeder-Integration via backward-kumulative Quadratsumme
    sq = _seg**2
    cum = np.cumsum(sq[::-1])[::-1]
    cum = np.clip(cum, 1e-12, None)
    decay_db = 10.0 * np.log10(cum / (cum[0] + 1e-12))

    # RT60: Zeit von -5 dB auf -35 dB (EDT/T20-Methode) extrapoliert auf -60 dB
    times = np.arange(len(decay_db), dtype=np.float32) / sr
    idx_5 = int(np.searchsorted(-decay_db, 5.0))
    idx_35 = int(np.searchsorted(-decay_db, 35.0))
    if idx_35 >= idx_5 + 5 and idx_35 < len(times):
        t20 = float(times[idx_35] - times[idx_5])
        rt60 = t20 * 3.0  # T20 × 3 = RT60-Extrapolation
    else:
        rt60 = 0.0

    if rt60 <= 0.01:
        room_char = "unknown"
    elif rt60 < 0.25:
        room_char = "dry_studio"
    elif rt60 < 0.60:
        room_char = "intimate_studio"
    elif rt60 < 1.20:
        room_char = "large_studio"
    else:
        room_char = "live_venue"

    # Echo-Chamber-Erkennung: nicht-lineare Abklingkurve + Resonanz-Peaks
    # (vereinfacht: hohe Varianz in der Abklingkurve + RT60 in Echo-Bereich)
    if 0.3 < rt60 < 0.8:
        _decay_diff = np.diff(decay_db[: min(int(0.3 * sr), len(decay_db) - 1)])
        _varianz = float(np.std(_decay_diff))
        if _varianz > 1.5:  # Nicht-lineares Abklingen = Echo-Chamber-Hinweis
            room_char = "echo_chamber"

    return room_char, float(np.clip(rt60, 0.0, 5.0))


def _estimate_compression_character(mono: np.ndarray, sr: int) -> tuple[str, float]:
    """Schätzt Kompressionscharakter via Crest-Factor-Analyse.

    Crest-Factor (CF) = Peak / RMS (dB). Hoch = wenig Kompression.
    Pumping-Detektion: Korrelierte Amplitudenmodulation im 2–8 Hz-Bereich.

    Returns:
        (compression_character_str, crest_factor_mean_db)
    """
    hop = int(sr * 0.1)  # 100 ms Frames
    n_frames = len(mono) // hop
    if n_frames < 5:
        return "natural", 12.0

    crest_vals: list[float] = []
    rms_vals: list[float] = []
    for i in range(n_frames):
        frame = mono[i * hop : (i + 1) * hop]
        peak = float(np.percentile(np.abs(frame), 99.9))
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        if rms > 1e-4:
            cf_db = float(20.0 * np.log10(peak / rms + 1e-12))
            crest_vals.append(cf_db)
            rms_vals.append(rms)

    if not crest_vals:
        return "natural", 12.0

    cf_mean = float(np.mean(crest_vals))
    cf_std = float(np.std(crest_vals))

    # Pumping-Detektion: spektrale Energie der RMS-Kurve im 2–8 Hz-Bereich
    rms_arr = np.array(rms_vals, dtype=np.float32)
    _pumping = False
    if len(rms_arr) >= 16:
        # RMS-Kurve ist 10 Hz-Abtastrate; 2–8 Hz = Indizes 20% bis 80%
        fft_rms = np.abs(np.fft.rfft(rms_arr - np.mean(rms_arr)))
        n_rms_bins = len(fft_rms)
        _low = max(1, int(0.20 * n_rms_bins))
        _high = max(_low + 1, int(0.80 * n_rms_bins))
        pumping_energy = float(np.sum(fft_rms[_low:_high]))
        total_energy = float(np.sum(fft_rms[1:])) + 1e-12
        _pumping = (pumping_energy / total_energy) > 0.45

    if cf_mean > 15.0 and cf_std > 3.0:
        return "dry", cf_mean
    if cf_mean < 8.0:
        return "limited", cf_mean
    if cf_mean < 11.0 or _pumping:
        return "heavy", cf_mean
    return "natural", cf_mean


def _estimate_mic_warmth(mono: np.ndarray, sr: int) -> str:
    """Schätzt Mikrofon-Wärme via Spektralneigung im 2–6 kHz-Bereich.

    Methodik: Vergleich der Energie im Presence-Bereich (3–5 kHz)
    mit der Energie im Grund-Bereich (300 Hz–2 kHz).

    Returns:
        "warm" | "neutral" | "presence"
    """
    n_fft = 2048
    seg = mono[: min(len(mono), int(sr * 5))]
    if len(seg) < n_fft:
        return "neutral"
    spec = np.abs(np.fft.rfft(seg, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    def _band_energy(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            return 0.0
        return float(np.mean(spec[mask] ** 2))

    e_presence = _band_energy(3000.0, 6000.0)
    e_base = _band_energy(300.0, 2000.0)
    if e_base < 1e-12:
        return "neutral"
    ratio_db = float(10.0 * np.log10(e_presence / e_base + 1e-12))

    if ratio_db < -5.0:  # Presence stark gedämpft → Ribbon-Mikrofon-Charakter
        return "warm"
    if ratio_db > -1.0:  # Presence-Peak → Kondensator (U87, SM7B)
        return "presence"
    return "neutral"


# ---------------------------------------------------------------------------
# Hauptfunktion: Profil-Lookup
# ---------------------------------------------------------------------------


def get_production_profile(
    era_decade: int | None,
    genre_label: str | None,
    _material_type: str | None,  # reserved for future BW-ceiling integration
    signature: ProductionSignature | None = None,
) -> ProductionProfile:
    """Gibt das spezifischste Produktionsprofil für die gegebenen Parameter zurück.

    Sucht in 3 Stufen:
        1. Volles Quad-Key-Match (era × genre × room × compression)
        2. Partieller Match mit "any" für room/compression
        3. Generischer Era-Fallback

    Args:
        era_decade:   Jahrzehnt (z.B. 1950). None → generischer Fallback.
        genre_label:  Genre-String (z.B. "jazz"). None oder "" → "any".
        material_type: Materialtyp (für Logging). None → ignoriert.
        signature:    Produktionssignatur aus ``detect_production_signature()``.
                      None → room_char="any", compression="any" im Lookup.

    Returns:
        ProductionProfile (niemals None — immer sicherer Fallback).
    """
    era_bucket = _decade_to_era_bucket(era_decade)
    genre_key = str(genre_label or "").strip().lower()
    room_char = signature.room_character if signature is not None else "any"
    compression = signature.compression_character if signature is not None else "any"

    # Normalisiere room_char und compression auf KB-Keys
    if room_char not in {"dry_studio", "intimate_studio", "large_studio", "echo_chamber", "live_venue"}:
        room_char = "any"
    if compression not in {"dry", "natural", "heavy", "limited"}:
        compression = "any"

    # Genre-Normalisierung: Genre-Aliase auf KB-Keys mappen
    genre_key = _normalize_genre(genre_key)

    # Greedy-Lookup: bestes Match gewinnt
    best_profile: ProductionProfile | None = None
    best_score: int = -1
    for key, profile in _PRODUCTION_KB.items():
        score = _score_key(key, era_bucket, genre_key, room_char, compression)
        if score > best_score:
            best_score = score
            best_profile = profile

    if best_profile is None or best_score < 0:
        logger.debug(
            "RecordingProductionKB: kein Match für era=%s genre=%s room=%s comp=%s → generic",
            era_bucket,
            genre_key,
            room_char,
            compression,
        )
        return ProductionProfile(profile_name="generic")

    logger.info(
        "RecordingProductionKB: Profil '%s' (era=%s genre=%s room=%s comp=%s Wert=%d) — %s",
        best_profile.profile_name,
        era_bucket,
        genre_key,
        room_char,
        compression,
        best_score,
        best_profile.era_note,
    )
    return best_profile


def _decade_to_era_bucket(decade: int | None) -> str:
    """Konvertiert Jahrzehnt-Integer in KB-Bucket-String."""
    if decade is None:
        return "any"
    d = int(decade)
    if d < 1930:
        return "1920s"
    if d < 1940:
        return "1930s"
    if d < 1950:
        return "1940s"
    if d < 1960:
        return "1950s"
    if d < 1970:
        return "1960s"
    if d < 1980:
        return "1970s"
    if d < 1990:
        return "1980s"
    return "1990s+"


def _normalize_genre(genre_key: str) -> str:
    """Normalisiert Genre-Strings auf KB-Keys."""
    _ALIASES: dict[str, str] = {
        "schlager": "schlager",
        "deutscher_schlager": "schlager",
        "german_schlager": "schlager",
        "german schlager": "schlager",
        "jazz": "jazz",
        "jazz_standard": "jazz",
        "bebop": "jazz",
        "swing": "jazz",
        "cool_jazz": "jazz",
        "blues": "jazz",
        "klassik": "klassik",
        "classical": "klassik",
        "oper": "klassik",
        "opera": "klassik",
        "kammermusik": "klassik",
        "soul": "soul",
        "rnb": "soul",
        "r&b": "soul",
        "gospel": "soul",
        "soul/r&b": "soul",
        "pop": "pop",
        "dance_pop": "pop",
        "synth_pop": "pop",
        "rock": "rock",
        "classic_rock": "rock",
        "hard_rock": "rock",
        "folk": "folk",
        "country": "folk",
        "folk_country": "folk",
        "volksmusik": "folk",
    }
    return _ALIASES.get(genre_key, "any")


# ---------------------------------------------------------------------------
# Singleton (thread-safe, double-checked locking)
# ---------------------------------------------------------------------------

_initialized = False
_init_lock = threading.Lock()


def initialize_production_kb() -> None:
    """Initialisiert die KB (Singleton, thread-safe). Kein I/O, kein ML."""
    global _initialized  # pylint: disable=global-statement
    if not _initialized:
        with _init_lock:
            if not _initialized:
                _n = len(_PRODUCTION_KB)
                logger.info("RecordingProductionKB: %d Produktionsprofile geladen", _n)
                _initialized = True
