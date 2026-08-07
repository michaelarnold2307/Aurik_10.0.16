"""backend/core/venue_acoustic_db.py — Aufnahme-Venue und Studio-Raumakustik Wissensbasis

Provides room acoustic fingerprints for 30+ classic recording venues.
Used by:
  - Phase 49 (Dereverb): constrains wet_mix cap for venues where early reflections
    are part of the authentic studio sound
  - Phase 20 / Phase 55 (Stereo imaging): width reference for venue-appropriate processing
  - Phase 07 (Harmonic Restoration): room coloration as a "don't remove" constraint

The key principle (§2.46f §0h): if the room sound was captured on the original recording,
it is NOT a defect to be removed — it is the authentic artistic environment.

Scientific references:
  Beranek (2004) «Concert Halls and Opera Houses: Music, Acoustics, and Architecture» 2nd ed.
  Ando (1998) «Architectural Acoustics — Blending Sound Sources, Sound Fields, and Listeners»
  Copeland (2008) «Manual of Analogue Sound Restoration» — studio acoustic characterisation
  Pätynen & Lokki (2010) «Directivity of Symphony Orchestra Instruments» (JASA)
  Kahle Acoustics (2018) — concert hall measurement database
  ISO 3382-1:2009 — Measurement of room acoustic parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VenueAcousticProfile:
    """Room acoustic fingerprint for a recording venue.

    All values represent measurements at the microphone position(s) used
    for the typical recording setup of this venue, as reported in the
    scientific literature.

    Attributes:
        label:              Human-readable venue name.
        rt60_mid_s:         Reverberation time T60 at 500–1000 Hz [seconds].
        rt60_low_s:         T60 at 125 Hz [seconds] — often longer than RT60_mid.
        rt60_high_s:        T60 at 4 kHz [seconds] — usually shorter.
        early_reflection_ms: Arrival time of first strong room reflection [ms].
                            Below 80 ms: integrated into direct sound (haas effect).
        drr_db:             Direct-to-Reverberant Ratio [dB]. High = dry.
        width_deg:          Apparent Source Width (ASW) indicator [0–100].
                            Based on Inter-Aural Cross-Correlation (IACC) measures.
        room_type:          "studio" | "concert_hall" | "opera_house" | "church"
                            | "outdoor" | "broadcast"
        dereverb_wet_cap:   Maximum safe wet_mix for phase_49 Dereverb [0..1].
                            Lower = preserve more room ambience.
        stereo_width_ref:   Reference stereo width [0..1] for phase_20 imaging.
        bass_warmth_db:     LF coloration at 100 Hz relative to 1 kHz [dB].
                            Positive = room adds bass warmth.
        era_min, era_max:   Years this characterisation is valid for.
        notes:              Scientific source / free text.
    """

    label: str
    rt60_mid_s: float
    rt60_low_s: float
    rt60_high_s: float
    early_reflection_ms: float
    drr_db: float
    width_deg: float  # ASW indicator 0–100
    room_type: str
    dereverb_wet_cap: float  # [0..1]
    stereo_width_ref: float  # [0..1]
    bass_warmth_db: float  # [dB] LF coloration
    era_min: int = 1900
    era_max: int = 2030
    notes: str = ""


# ---------------------------------------------------------------------------
# Venue database
# ---------------------------------------------------------------------------

_VENUE_DB: dict[str, VenueAcousticProfile] = {
    # ===================================================================
    # ICONIC RECORDING STUDIOS
    # ===================================================================
    "abbey_road_studio_2": VenueAcousticProfile(
        label="Abbey Road Studio 2 (EMI London)",
        rt60_mid_s=0.38,
        rt60_low_s=0.55,
        rt60_high_s=0.22,
        early_reflection_ms=12.0,
        drr_db=6.0,
        width_deg=55.0,
        room_type="studio",
        dereverb_wet_cap=0.15,  # early reflections are part of The Beatles sound
        stereo_width_ref=0.65,
        bass_warmth_db=+1.5,
        era_min=1958,
        era_max=2030,
        notes="Doyle (2004); ~175 m² live room. Room sound essential to Beatle recordings. "
        "Echo chamber used 1963–1968. Do NOT dereverb early reflections.",
    ),
    "abbey_road_studio_1": VenueAcousticProfile(
        label="Abbey Road Studio 1 (EMI London, Orchestral)",
        rt60_mid_s=1.85,
        rt60_low_s=2.50,
        rt60_high_s=1.00,
        early_reflection_ms=25.0,
        drr_db=0.0,
        width_deg=80.0,
        room_type="studio",
        dereverb_wet_cap=0.25,
        stereo_width_ref=0.85,
        bass_warmth_db=+2.5,
        era_min=1930,
        era_max=2030,
        notes="Large orchestral studio 330 m²; used for orchestral overdubs. "
        "Extended RT60 typical of British orchestral recordings.",
    ),
    "columbia_30th_street": VenueAcousticProfile(
        label="Columbia 30th Street Studio NYC ('The Church')",
        rt60_mid_s=1.70,
        rt60_low_s=2.30,
        rt60_high_s=0.95,
        early_reflection_ms=22.0,
        drr_db=1.0,
        width_deg=82.0,
        room_type="studio",
        dereverb_wet_cap=0.28,
        stereo_width_ref=0.88,
        bass_warmth_db=+3.0,
        era_min=1948,
        era_max=1981,
        notes="Katz (2007) §5; former Calvary Baptist Church 650 m². "
        "Massive warm bass bloom from church modes. Iconic Miles Davis/Glenn Gould.",
    ),
    "capitol_studios_b": VenueAcousticProfile(
        label="Capitol Studios B (Hollywood)",
        rt60_mid_s=0.45,
        rt60_low_s=0.70,
        rt60_high_s=0.28,
        early_reflection_ms=8.0,
        drr_db=8.0,
        width_deg=45.0,
        room_type="studio",
        dereverb_wet_cap=0.10,
        stereo_width_ref=0.55,
        bass_warmth_db=+1.0,
        era_min=1956,
        era_max=2030,
        notes="Capitol B: compact live room. Echo chambers used for Sinatra plates. "
        "High DRR → dry studio sound; echo chambers add controlled reverb.",
    ),
    "electric_lady_a": VenueAcousticProfile(
        label="Electric Lady Studios A (NYC, Hendrix)",
        rt60_mid_s=0.55,
        rt60_low_s=0.80,
        rt60_high_s=0.32,
        early_reflection_ms=15.0,
        drr_db=5.0,
        width_deg=60.0,
        room_type="studio",
        dereverb_wet_cap=0.18,
        stereo_width_ref=0.70,
        bass_warmth_db=+2.0,
        era_min=1970,
        era_max=2030,
        notes="Designed by Eddie Kramer; known for warm LF bloom and slight diffuse field. "
        "Curved walls reduce flutter echoes.",
    ),
    "ndr_studio_10_hamburg": VenueAcousticProfile(
        label="NDR Großer Sendesaal Hamburg (Studio 10)",
        rt60_mid_s=1.55,
        rt60_low_s=2.10,
        rt60_high_s=0.90,
        early_reflection_ms=20.0,
        drr_db=1.5,
        width_deg=78.0,
        room_type="broadcast",
        dereverb_wet_cap=0.22,
        stereo_width_ref=0.82,
        bass_warmth_db=+2.0,
        era_min=1945,
        era_max=2020,
        notes="German ARD broadcast standard room. Warm and spacious. "
        "Essential for NDR/DG symphonic archive recordings.",
    ),
    "decca_kingsway_hall": VenueAcousticProfile(
        label="Kingsway Hall London (Decca Records)",
        rt60_mid_s=2.10,
        rt60_low_s=3.00,
        rt60_high_s=1.20,
        early_reflection_ms=30.0,
        drr_db=-1.5,
        width_deg=90.0,
        room_type="studio",
        dereverb_wet_cap=0.30,
        stereo_width_ref=0.92,
        bass_warmth_db=+4.5,
        era_min=1950,
        era_max=1981,
        notes="Beranek (2004) equivalent class A. Decca Tree recordings (1950–1981). "
        "Exceptionally warm bass. Demolished 1984. Very long RT60 at LF — "
        "dereverbing this is musically wrong.",
    ),
    "rbb_haus_des_rundfunks": VenueAcousticProfile(
        label="RBB / RIAS Haus des Rundfunks Berlin",
        rt60_mid_s=1.40,
        rt60_low_s=2.00,
        rt60_high_s=0.80,
        early_reflection_ms=18.0,
        drr_db=2.0,
        width_deg=72.0,
        room_type="broadcast",
        dereverb_wet_cap=0.20,
        stereo_width_ref=0.78,
        bass_warmth_db=+1.8,
        era_min=1930,
        era_max=2030,
        notes="Designed 1929; Hans Poelzig. ARD reference broadcast hall. Classic German radio orchestral sound.",
    ),
    "konzerthaus_wien": VenueAcousticProfile(
        label="Konzerthaus Wien (Großer Saal)",
        rt60_mid_s=2.00,
        rt60_low_s=2.85,
        rt60_high_s=1.15,
        early_reflection_ms=28.0,
        drr_db=-0.5,
        width_deg=88.0,
        room_type="concert_hall",
        dereverb_wet_cap=0.35,
        stereo_width_ref=0.90,
        bass_warmth_db=+3.5,
        era_min=1913,
        era_max=2030,
        notes="Beranek (2004) Class A; 1840 seats. Key venue for Vienna Philharmonic. "
        "Warm 'Viennese' bass bloom fundamental to classical recordings.",
    ),
    "carnegie_hall_main": VenueAcousticProfile(
        label="Carnegie Hall Isaac Stern Auditorium (NYC)",
        rt60_mid_s=1.80,
        rt60_low_s=2.50,
        rt60_high_s=1.05,
        early_reflection_ms=25.0,
        drr_db=0.5,
        width_deg=85.0,
        room_type="concert_hall",
        dereverb_wet_cap=0.30,
        stereo_width_ref=0.88,
        bass_warmth_db=+2.8,
        era_min=1891,
        era_max=2030,
        notes="Beranek (2004); renovated 1986. Classic American orchestral venue. "
        "Balanced warmth; slightly drier than European halls.",
    ),
    "musikverein_wien": VenueAcousticProfile(
        label="Musikverein Wien (Goldener Saal)",
        rt60_mid_s=2.05,
        rt60_low_s=2.90,
        rt60_high_s=1.20,
        early_reflection_ms=27.0,
        drr_db=-0.8,
        width_deg=92.0,
        room_type="concert_hall",
        dereverb_wet_cap=0.35,
        stereo_width_ref=0.93,
        bass_warmth_db=+4.0,
        era_min=1870,
        era_max=2030,
        notes="Beranek 'the world's greatest concert hall'. 1744 seats. "
        "Extreme bass warmth from shallow balcony design. DG/Philips primary recording site. "
        "NEVER aggressively dereverb — the room IS the sound.",
    ),
    "concertgebouw_amsterdam": VenueAcousticProfile(
        label="Royal Concertgebouw Amsterdam",
        rt60_mid_s=2.10,
        rt60_low_s=3.10,
        rt60_high_s=1.25,
        early_reflection_ms=25.0,
        drr_db=-1.0,
        width_deg=91.0,
        room_type="concert_hall",
        dereverb_wet_cap=0.33,
        stereo_width_ref=0.92,
        bass_warmth_db=+3.8,
        era_min=1888,
        era_max=2030,
        notes="Beranek Class A. Acoustically critical Philips/DG recordings. "
        "Warm sustained reverb tail; RT60 drops slowly from 2.1 s at 500 Hz.",
    ),
    "boston_symphony_hall": VenueAcousticProfile(
        label="Boston Symphony Hall",
        rt60_mid_s=1.95,
        rt60_low_s=2.75,
        rt60_high_s=1.10,
        early_reflection_ms=22.0,
        drr_db=0.2,
        width_deg=87.0,
        room_type="concert_hall",
        dereverb_wet_cap=0.32,
        stereo_width_ref=0.90,
        bass_warmth_db=+3.2,
        era_min=1900,
        era_max=2030,
        notes="Beranek (2004). RCA Living Stereo primary venue (Boston SO/Munch). "
        "Slightly dryer than European halls at mid-frequency.",
    ),
    "philharmonie_berlin": VenueAcousticProfile(
        label="Berliner Philharmonie (Großer Saal)",
        rt60_mid_s=2.00,
        rt60_low_s=2.80,
        rt60_high_s=1.15,
        early_reflection_ms=22.0,
        drr_db=0.0,
        width_deg=89.0,
        room_type="concert_hall",
        dereverb_wet_cap=0.30,
        stereo_width_ref=0.91,
        bass_warmth_db=+3.0,
        era_min=1963,
        era_max=2030,
        notes="Hans Scharoun vineyard terrace design 1963. DG BPO/Karajan recordings. "
        "Excellent early lateral reflections from side walls; very enveloping.",
    ),
    "snape_maltings": VenueAcousticProfile(
        label="Snape Maltings Concert Hall (Aldeburgh)",
        rt60_mid_s=1.60,
        rt60_low_s=2.20,
        rt60_high_s=0.95,
        early_reflection_ms=18.0,
        drr_db=2.0,
        width_deg=75.0,
        room_type="concert_hall",
        dereverb_wet_cap=0.22,
        stereo_width_ref=0.80,
        bass_warmth_db=+2.0,
        era_min=1967,
        era_max=2030,
        notes="Decca chamber music recordings; Britten festival home. "
        "Slightly dryer than major concert halls; intimate.",
    ),
    "deutsche_grammophon_studio": VenueAcousticProfile(
        label="DG Hannover Studio (Christuskirche)",
        rt60_mid_s=1.90,
        rt60_low_s=2.60,
        rt60_high_s=1.10,
        early_reflection_ms=28.0,
        drr_db=-0.5,
        width_deg=85.0,
        room_type="studio",  # repurposed church
        dereverb_wet_cap=0.30,
        stereo_width_ref=0.88,
        bass_warmth_db=+3.5,
        era_min=1955,
        era_max=1990,
        notes="DG primary recording church; Karajan Berlin recordings 1960s–70s. "
        "Church acoustic prominent in Decca-tree recordings.",
    ),
    # ===================================================================
    # OPERA HOUSES
    # ===================================================================
    "scala_milan": VenueAcousticProfile(
        label="Teatro alla Scala Milano",
        rt60_mid_s=1.20,
        rt60_low_s=1.80,
        rt60_high_s=0.75,
        early_reflection_ms=20.0,
        drr_db=3.0,
        width_deg=70.0,
        room_type="opera_house",
        dereverb_wet_cap=0.18,
        stereo_width_ref=0.72,
        bass_warmth_db=+1.5,
        era_min=1778,
        era_max=2030,
        notes="Beranek (2004); 2800 seats. Drier than northern European halls — "
        "Italian operatic tradition favours clarity over reverb.",
    ),
    "vienna_staatsoper": VenueAcousticProfile(
        label="Wiener Staatsoper",
        rt60_mid_s=1.45,
        rt60_low_s=2.05,
        rt60_high_s=0.88,
        early_reflection_ms=22.0,
        drr_db=2.0,
        width_deg=74.0,
        room_type="opera_house",
        dereverb_wet_cap=0.22,
        stereo_width_ref=0.76,
        bass_warmth_db=+2.0,
        era_min=1869,
        era_max=2030,
        notes="Decca/DG opera recordings primary. Warmer than La Scala; horseshoe shape with multiple balconies.",
    ),
    "metropolitan_opera": VenueAcousticProfile(
        label="Metropolitan Opera New York",
        rt60_mid_s=1.80,
        rt60_low_s=2.40,
        rt60_high_s=1.05,
        early_reflection_ms=24.0,
        drr_db=0.5,
        width_deg=80.0,
        room_type="opera_house",
        dereverb_wet_cap=0.28,
        stereo_width_ref=0.82,
        bass_warmth_db=+2.5,
        era_min=1966,
        era_max=2030,
        notes="New Met (Lincoln Center) 1966; 3800 seats. DG/Sony classic opera recordings.",
    ),
    # ===================================================================
    # CHURCHES AND ECCLESIASTICAL
    # ===================================================================
    "thomaskirche_leipzig": VenueAcousticProfile(
        label="Thomaskirche Leipzig",
        rt60_mid_s=2.80,
        rt60_low_s=4.00,
        rt60_high_s=1.60,
        early_reflection_ms=40.0,
        drr_db=-3.0,
        width_deg=95.0,
        room_type="church",
        dereverb_wet_cap=0.40,
        stereo_width_ref=0.95,
        bass_warmth_db=+5.0,
        era_min=1600,
        era_max=2030,
        notes="Bach's own church; vast stone acoustic. Organ and choral recordings. "
        "RT60 > 2.5 s — the reverb IS the music. Maximum protection.",
    ),
    "notre_dame_paris": VenueAcousticProfile(
        label="Notre-Dame de Paris (Cathedral)",
        rt60_mid_s=4.50,
        rt60_low_s=6.00,
        rt60_high_s=2.50,
        early_reflection_ms=60.0,
        drr_db=-5.0,
        width_deg=98.0,
        room_type="church",
        dereverb_wet_cap=0.50,
        stereo_width_ref=0.98,
        bass_warmth_db=+7.0,
        era_min=1163,
        era_max=2019,
        notes="Gothic cathedral; enormous RT60. Organ recordings iconic. "
        "Dereverbing is musically meaningless — the music was composed FOR this space.",
    ),
    "liverpool_metropolitan": VenueAcousticProfile(
        label="Liverpool Metropolitan Cathedral",
        rt60_mid_s=3.80,
        rt60_low_s=5.20,
        rt60_high_s=2.10,
        early_reflection_ms=50.0,
        drr_db=-4.0,
        width_deg=96.0,
        room_type="church",
        dereverb_wet_cap=0.45,
        stereo_width_ref=0.96,
        bass_warmth_db=+6.0,
        era_min=1967,
        era_max=2030,
        notes="Circular concrete structure; peak RT60 at 63 Hz > 8 s. "
        "Harrison/Britten recordings. Maximum reverb protection.",
    ),
    # ===================================================================
    # BROADCAST STUDIOS
    # ===================================================================
    "bbc_maida_vale": VenueAcousticProfile(
        label="BBC Maida Vale Studios (London)",
        rt60_mid_s=0.80,
        rt60_low_s=1.20,
        rt60_high_s=0.50,
        early_reflection_ms=12.0,
        drr_db=5.0,
        width_deg=55.0,
        room_type="broadcast",
        dereverb_wet_cap=0.12,
        stereo_width_ref=0.60,
        bass_warmth_db=+1.0,
        era_min=1934,
        era_max=2020,
        notes="Copeland (2008); BBC BTR machines. Sessions studio: tighter, dryer. "
        "Natural sound with moderate BBC plate reverb added post-recording.",
    ),
    "wdr_koeln_funkhaus": VenueAcousticProfile(
        label="WDR Köln Funkhaus (Großer Sendesaal)",
        rt60_mid_s=1.60,
        rt60_low_s=2.20,
        rt60_high_s=0.95,
        early_reflection_ms=20.0,
        drr_db=1.5,
        width_deg=76.0,
        room_type="broadcast",
        dereverb_wet_cap=0.22,
        stereo_width_ref=0.80,
        bass_warmth_db=+2.2,
        era_min=1952,
        era_max=2030,
        notes="Designed by Franz Schuster; excellent LF bass bloom. WDR/DG symphonic archive recordings 1952–2000.",
    ),
    "orf_radiokulturhaus_wien": VenueAcousticProfile(
        label="ORF RadioKulturhaus Wien",
        rt60_mid_s=1.30,
        rt60_low_s=1.90,
        rt60_high_s=0.80,
        early_reflection_ms=16.0,
        drr_db=3.0,
        width_deg=68.0,
        room_type="broadcast",
        dereverb_wet_cap=0.18,
        stereo_width_ref=0.72,
        bass_warmth_db=+1.5,
        era_min=1945,
        era_max=2030,
        notes="Austrian broadcast standard. ORF editions primary site.",
    ),
    "ddr_berliner_rundfunk": VenueAcousticProfile(
        label="DDR Berliner Rundfunk (Studio Nalepastraße)",
        rt60_mid_s=1.45,
        rt60_low_s=2.00,
        rt60_high_s=0.85,
        early_reflection_ms=18.0,
        drr_db=2.5,
        width_deg=70.0,
        room_type="broadcast",
        dereverb_wet_cap=0.20,
        stereo_width_ref=0.74,
        bass_warmth_db=+1.8,
        era_min=1952,
        era_max=1991,
        notes="DDR state broadcast studio; Nalepastraße Berlin. "
        "EELA console + ORWO tape; slightly warm LF character. "
        "Many classic DDR popular music recordings (Puhdys, City, Karat).",
    ),
    # ===================================================================
    # OUTDOOR / AMBIENT
    # ===================================================================
    "outdoor_open": VenueAcousticProfile(
        label="Open Air / Outdoor Recording",
        rt60_mid_s=0.08,
        rt60_low_s=0.10,
        rt60_high_s=0.05,
        early_reflection_ms=5.0,
        drr_db=15.0,
        width_deg=20.0,
        room_type="outdoor",
        dereverb_wet_cap=0.05,
        stereo_width_ref=0.30,
        bass_warmth_db=0.0,
        era_min=1900,
        era_max=2030,
        notes="Essentially anechoic with ground reflection. Minimal RT60. "
        "Any reverb heard is added artificially or from proximity.",
    ),
    "small_club": VenueAcousticProfile(
        label="Small Jazz Club / Club Venue",
        rt60_mid_s=0.45,
        rt60_low_s=0.65,
        rt60_high_s=0.28,
        early_reflection_ms=8.0,
        drr_db=8.0,
        width_deg=38.0,
        room_type="studio",
        dereverb_wet_cap=0.10,
        stereo_width_ref=0.45,
        bass_warmth_db=+1.5,
        era_min=1940,
        era_max=2030,
        notes="Intimate live venue. Short RT60 typical; audience absorption heavy. Jazz and blues live recordings.",
    ),
}

# Aliases
_ALIASES: dict[str, str] = {
    "abbey_road": "abbey_road_studio_2",
    "emi": "abbey_road_studio_2",
    "columbia": "columbia_30th_street",
    "columbia_church": "columbia_30th_street",
    "the_church": "columbia_30th_street",
    "capitol": "capitol_studios_b",
    "electric_lady": "electric_lady_a",
    "ndr": "ndr_studio_10_hamburg",
    "kingsway": "decca_kingsway_hall",
    "decca": "decca_kingsway_hall",
    "musikverein": "musikverein_wien",
    "goldener_saal": "musikverein_wien",
    "concertgebouw": "concertgebouw_amsterdam",
    "boston_symphony": "boston_symphony_hall",
    "bso": "boston_symphony_hall",
    "philharmonie": "philharmonie_berlin",
    "berliner_philharmonie": "philharmonie_berlin",
    "bpo": "philharmonie_berlin",
    "scala": "scala_milan",
    "staatsoper": "vienna_staatsoper",
    "wiener_staatsoper": "vienna_staatsoper",
    "met": "metropolitan_opera",
    "thomaskirche": "thomaskirche_leipzig",
    "notre_dame": "notre_dame_paris",
    "bbc": "bbc_maida_vale",
    "maida_vale": "bbc_maida_vale",
    "wdr": "wdr_koeln_funkhaus",
    "orf": "orf_radiokulturhaus_wien",
    "ddr_studio": "ddr_berliner_rundfunk",
    "nalepastrasse": "ddr_berliner_rundfunk",
    "rias": "rbb_haus_des_rundfunks",
    "rbb": "rbb_haus_des_rundfunks",
    "outdoor": "outdoor_open",
    "jazz_club": "small_club",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _era_relevance(profile: VenueAcousticProfile, era_decade: int | None) -> float:
    """[0..1] relevance weight based on era overlap."""
    if era_decade is None:
        return 0.7
    d = int(era_decade)
    if profile.era_min <= d <= profile.era_max:
        return 1.0
    gap = max(0, profile.era_min - d, d - profile.era_max)
    return max(0.0, 1.0 - gap / 40.0)


def get_venue_profile(
    venue_hint: str,
    era_decade: int | None = None,
) -> VenueAcousticProfile | None:
    """Look up a venue profile by hint string (case-insensitive partial match).

    Args:
        venue_hint:  Free-form venue/studio name string.
        era_decade:  Recording decade for relevance check.

    Returns:
        VenueAcousticProfile or None.
    """
    if not venue_hint:
        return None

    hint = venue_hint.strip().lower().replace(" ", "_").replace("-", "_")
    canonical = _ALIASES.get(hint, hint)
    profile = _VENUE_DB.get(canonical)

    if profile is None:
        for key, val in _VENUE_DB.items():
            if hint in key or key in hint:
                profile = val
                break

    if profile is None:
        logger.debug("venue_acoustic_db: no match for %r", venue_hint)
        return None

    rel = _era_relevance(profile, era_decade)
    if rel < 0.10:
        logger.debug("venue_acoustic_db: %s era_relevance=%.2f < 0.10 — uebersprungen", profile.label, rel)
        return None

    logger.debug("venue_acoustic_db: %r → %s (era_rel=%.2f)", venue_hint, profile.label, rel)
    return profile


def get_dereverb_wet_cap(
    venue_hint: str,
    era_decade: int | None = None,
    *,
    default: float = 0.35,
) -> float:
    """Gibt the maximum safe Dereverb wet_mix for a venue zurück.

    Lower values protect more of the authentic room sound.

    Args:
        venue_hint:  Venue name string.
        era_decade:  Recording decade.
        default:     Fallback value if venue unknown [0..1].

    Returns:
        wet_mix cap [0..1].
    """
    profile = get_venue_profile(venue_hint, era_decade)
    if profile is None:
        return default
    return profile.dereverb_wet_cap


def get_rt60_profile(
    venue_hint: str,
    era_decade: int | None = None,
) -> dict[str, float] | None:
    """Gibt RT60 values at three frequency bands zurück.

    Returns:
        dict with keys "low_s", "mid_s", "high_s" or None.
    """
    profile = get_venue_profile(venue_hint, era_decade)
    if profile is None:
        return None
    return {
        "low_s": profile.rt60_low_s,
        "mid_s": profile.rt60_mid_s,
        "high_s": profile.rt60_high_s,
        "early_reflection_ms": profile.early_reflection_ms,
        "drr_db": profile.drr_db,
    }


def estimate_room_type(
    rt60_mid_s: float,
    drr_db: float,
    *,
    era_decade: int | None = None,
    material_hint: str | None = None,
) -> str:
    """Heuristic room type from measured RT60 and DRR with era plausibility.

    Args:
        rt60_mid_s:    Measured RT60 at 500–1000 Hz [s].
        drr_db:        Measured Direct-to-Reverberant Ratio [dB].
        era_decade:    Recording decade for plausibility check.
        material_hint: Material type (vinyl/tape → studio recording).

    Returns:
        One of: "studio" | "broadcast" | "concert_hall" | "church" | "outdoor".
    """
    if drr_db > 12.0 or rt60_mid_s < 0.15:
        return "outdoor"
    if rt60_mid_s < 0.60:
        return "studio"
    if rt60_mid_s < 1.20:
        return "broadcast"
    if rt60_mid_s < 2.50:
        room = "concert_hall"
    else:
        room = "church"

    # §2.59.16 Era-Plausibilität: Studio-Aufnahmen post-1955 sind
    # nie in Kirchen entstanden. Der Hall kommt vom Plattenhall/Chamber.
    # Caps: concert_hall→broadcast, church→broadcast.
    _is_studio_era = (
        era_decade is not None and era_decade >= 1960 and material_hint in ("vinyl", "tape", "reel_tape", "cassette")
    )
    if _is_studio_era and room in ("concert_hall", "church"):
        import logging

        _log = logging.getLogger(__name__)
        _log.info(
            "§2.59.16 Room-Era-Plausibilität: RT60=%.1fs → '%s', "
            "aber era=%d + material=%s → auf 'broadcast' korrigiert "
            "(kein Kirchenhall in Studio-Ära)",
            rt60_mid_s,
            room,
            era_decade,
            material_hint,
        )
        return "broadcast"

    return room


def list_venue_keys() -> list[str]:
    """Gibt all canonical venue keys zurück."""
    return sorted(_VENUE_DB.keys())


__all__ = [
    "VenueAcousticProfile",
    "get_venue_profile",
    "get_dereverb_wet_cap",
    "get_rt60_profile",
    "estimate_room_type",
    "list_venue_keys",
]
