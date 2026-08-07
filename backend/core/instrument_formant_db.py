"""backend/core/instrument_formant_db.py — Instrumenten-Formant- und Timbral-Wissensbasis

Provides spectral fingerprints for 70+ instruments and voice types used by:
- Phase 42 (Carrier Formant Enhancement): guides recovery of lost formant structure
- Phase 07 (Harmonic Restoration): bounds harmonics to instrument-authentic range
- Phase 03 (Denoise): prevents removal of instrument-specific partial energy
- LyricsGuidedEnhancement §2.36: phoneme-sensitive processing for vocal types

Each entry contains:
  - Formant frequencies F1–F4 [Hz] (vocal-tract resonances for voice;
    body/air resonances for acoustic instruments)
  - Fundamental range [Hz]: physically possible F0 range
  - Harmonic rolloff [dB/octave]: spectral tilt of partials above fundamental
  - Brightness centre [Hz]: spectral centroid typical for the instrument
  - Attack character: "impulsive" | "gradual" | "bowed"

Scientific references:
  Sundberg (1987) «The Science of the Singing Voice»
  Martin (1994) «Musical Instrument Acoustics» (JASA)
  Rossing, Moore & Wheeler (2002) «The Science of Sound» 3rd ed.
  Fletcher & Rossing (1998) «The Physics of Musical Instruments» 2nd ed.
  Wolfe (2021) UNSW Acoustics Group formant measurements (formant database)
  Levitin (2006) «This Is Your Brain on Music» — perceptual timbre model
  Peeters et al. (2011) «The Timbre Toolbox» (JASA)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormantProfile:
    """Spectral fingerprint for one instrument or voice type.

    All Hz values are *centre frequencies* of formant resonances.
    The formant list length varies by instrument type:
    - Voice: 4 formants (classical vocal-tract model, Sundberg 1987)
    - Bowed strings: 2–3 body resonances + air resonance
    - Wind instruments: 1–3 tube resonances
    - Percussion/plucked: 1–2 body resonances

    Attributes:
        label:          Human-readable name ("Soprano", "Violin", "Oboe").
        formant_hz:     Formant centre frequencies [Hz], ascending order.
        formant_bw_hz:  Formant bandwidth [Hz] (−3 dB), one per formant.
        f0_min_hz:      Lowest expected fundamental frequency [Hz].
        f0_max_hz:      Highest expected fundamental frequency [Hz].
        harmonic_rolloff_db_oct: Spectral tilt above fundamental [dB/octave].
                        Negative = falls with frequency (most instruments).
        brightness_hz:  Typical spectral centroid [Hz] (Peeters 2011).
        attack_char:    Attack envelope type for NR bypass decisions.
        category:       Instrument family string for grouping.
        notes:          Reference notes for logging.
    """

    label: str
    formant_hz: tuple[float, ...]
    formant_bw_hz: tuple[float, ...]
    f0_min_hz: float
    f0_max_hz: float
    harmonic_rolloff_db_oct: float  # dB/octave (negative = roll off)
    brightness_hz: float  # spectral centroid [Hz]
    attack_char: str = "gradual"  # "impulsive" | "gradual" | "bowed" | "breath"
    category: str = "unknown"
    notes: str = ""


# ---------------------------------------------------------------------------
# Voice types (Sundberg 1987 + Wolfe 2021 measurements)
# ---------------------------------------------------------------------------

_VOICE_DB: dict[str, FormantProfile] = {
    "soprano": FormantProfile(
        label="Soprano (classical)",
        formant_hz=(800.0, 1150.0, 2900.0, 3400.0),
        formant_bw_hz=(80.0, 90.0, 120.0, 180.0),
        f0_min_hz=262.0,
        f0_max_hz=1175.0,
        harmonic_rolloff_db_oct=-9.0,
        brightness_hz=2800.0,
        attack_char="breath",
        category="voice",
        notes="Sundberg 1987 §3.2; singers formant 2900–3200 Hz",
    ),
    "mezzo_soprano": FormantProfile(
        label="Mezzo-Soprano",
        formant_hz=(700.0, 1100.0, 2750.0, 3200.0),
        formant_bw_hz=(80.0, 90.0, 120.0, 170.0),
        f0_min_hz=220.0,
        f0_max_hz=880.0,
        harmonic_rolloff_db_oct=-9.5,
        brightness_hz=2600.0,
        attack_char="breath",
        category="voice",
    ),
    "alto": FormantProfile(
        label="Alto (classical)",
        formant_hz=(600.0, 1000.0, 2600.0, 3000.0),
        formant_bw_hz=(80.0, 100.0, 130.0, 180.0),
        f0_min_hz=196.0,
        f0_max_hz=698.0,
        harmonic_rolloff_db_oct=-10.0,
        brightness_hz=2400.0,
        attack_char="breath",
        category="voice",
    ),
    "tenor": FormantProfile(
        label="Tenor (classical)",
        formant_hz=(650.0, 1080.0, 2650.0, 3000.0),
        formant_bw_hz=(80.0, 100.0, 120.0, 180.0),
        f0_min_hz=130.0,
        f0_max_hz=523.0,
        harmonic_rolloff_db_oct=-10.0,
        brightness_hz=2500.0,
        attack_char="breath",
        category="voice",
        notes="Singers formant 2500–3000 Hz for operatic tenor",
    ),
    "baritone": FormantProfile(
        label="Baritone",
        formant_hz=(600.0, 1000.0, 2500.0, 2900.0),
        formant_bw_hz=(85.0, 105.0, 130.0, 190.0),
        f0_min_hz=110.0,
        f0_max_hz=392.0,
        harmonic_rolloff_db_oct=-10.5,
        brightness_hz=2300.0,
        attack_char="breath",
        category="voice",
    ),
    "bass": FormantProfile(
        label="Bass (classical)",
        formant_hz=(450.0, 900.0, 2300.0, 2700.0),
        formant_bw_hz=(90.0, 110.0, 140.0, 200.0),
        f0_min_hz=82.0,
        f0_max_hz=294.0,
        harmonic_rolloff_db_oct=-11.0,
        brightness_hz=2000.0,
        attack_char="breath",
        category="voice",
    ),
    "pop_female_vocal": FormantProfile(
        label="Pop/Schlager Female Vocal",
        formant_hz=(700.0, 1200.0, 2600.0, 3500.0),
        formant_bw_hz=(100.0, 120.0, 150.0, 220.0),
        f0_min_hz=175.0,
        f0_max_hz=1000.0,
        harmonic_rolloff_db_oct=-8.0,
        brightness_hz=3000.0,
        attack_char="breath",
        category="voice",
        notes="Non-classical production; presence boost 2.5–4 kHz common",
    ),
    "pop_male_vocal": FormantProfile(
        label="Pop/Schlager Male Vocal",
        formant_hz=(550.0, 1100.0, 2400.0, 3200.0),
        formant_bw_hz=(100.0, 120.0, 150.0, 210.0),
        f0_min_hz=100.0,
        f0_max_hz=500.0,
        harmonic_rolloff_db_oct=-9.0,
        brightness_hz=2700.0,
        attack_char="breath",
        category="voice",
    ),
    "jazz_vocalist": FormantProfile(
        label="Jazz Vocalist",
        formant_hz=(580.0, 1050.0, 2500.0, 3100.0),
        formant_bw_hz=(110.0, 130.0, 160.0, 220.0),
        f0_min_hz=110.0,
        f0_max_hz=880.0,
        harmonic_rolloff_db_oct=-9.5,
        brightness_hz=2600.0,
        attack_char="breath",
        category="voice",
        notes="Microphone technique proximity effect boosts LF (F1 region)",
    ),
    "child_voice": FormantProfile(
        label="Child Voice (< 12 years)",
        formant_hz=(800.0, 1600.0, 3000.0, 3800.0),
        formant_bw_hz=(100.0, 130.0, 160.0, 230.0),
        f0_min_hz=200.0,
        f0_max_hz=1200.0,
        harmonic_rolloff_db_oct=-8.0,
        brightness_hz=3500.0,
        attack_char="breath",
        category="voice",
        notes="Shorter vocal tract → all formants shifted up ~20% vs adult",
    ),
}

# ---------------------------------------------------------------------------
# String instruments (Fletcher & Rossing 1998; Martin 1994)
# ---------------------------------------------------------------------------

_STRING_DB: dict[str, FormantProfile] = {
    "violin": FormantProfile(
        label="Violin",
        formant_hz=(290.0, 500.0, 3200.0),
        formant_bw_hz=(60.0, 100.0, 400.0),
        f0_min_hz=196.0,
        f0_max_hz=3520.0,
        harmonic_rolloff_db_oct=-4.0,
        brightness_hz=4000.0,
        attack_char="bowed",
        category="bowed_string",
        notes="F1=290 Hz (main air resonance B0), F2=500 Hz (main wood resonance A0), "
        "Helmholtz: 290 Hz. Wolf: 470 Hz typical. Brightness peak 3–5 kHz.",
    ),
    "viola": FormantProfile(
        label="Viola",
        formant_hz=(220.0, 430.0, 2500.0),
        formant_bw_hz=(65.0, 110.0, 430.0),
        f0_min_hz=131.0,
        f0_max_hz=1760.0,
        harmonic_rolloff_db_oct=-4.5,
        brightness_hz=3000.0,
        attack_char="bowed",
        category="bowed_string",
        notes="Similar to violin but ~50% lower body resonances; darker timbre",
    ),
    "cello": FormantProfile(
        label="Cello",
        formant_hz=(105.0, 220.0, 1400.0),
        formant_bw_hz=(40.0, 80.0, 300.0),
        f0_min_hz=65.0,
        f0_max_hz=880.0,
        harmonic_rolloff_db_oct=-5.5,
        brightness_hz=1800.0,
        attack_char="bowed",
        category="bowed_string",
        notes="B0=105 Hz, A0=220 Hz; rich warmth region 200–600 Hz",
    ),
    "double_bass": FormantProfile(
        label="Double Bass",
        formant_hz=(60.0, 110.0, 800.0),
        formant_bw_hz=(30.0, 60.0, 200.0),
        f0_min_hz=41.0,
        f0_max_hz=294.0,
        harmonic_rolloff_db_oct=-6.0,
        brightness_hz=900.0,
        attack_char="bowed",
        category="bowed_string",
    ),
    "acoustic_guitar": FormantProfile(
        label="Acoustic Guitar",
        formant_hz=(100.0, 185.0, 800.0),
        formant_bw_hz=(20.0, 35.0, 120.0),
        f0_min_hz=82.0,
        f0_max_hz=1175.0,
        harmonic_rolloff_db_oct=-5.0,
        brightness_hz=2500.0,
        attack_char="impulsive",
        category="plucked_string",
        notes="Helmholtz air: 105 Hz; main top plate: 185 Hz; bright transient attack",
    ),
    "classical_guitar": FormantProfile(
        label="Classical Guitar",
        formant_hz=(100.0, 190.0, 750.0),
        formant_bw_hz=(18.0, 30.0, 100.0),
        f0_min_hz=82.0,
        f0_max_hz=1175.0,
        harmonic_rolloff_db_oct=-6.0,
        brightness_hz=2200.0,
        attack_char="impulsive",
        category="plucked_string",
    ),
    "electric_guitar": FormantProfile(
        label="Electric Guitar",
        formant_hz=(200.0, 500.0),
        formant_bw_hz=(80.0, 150.0),
        f0_min_hz=82.0,
        f0_max_hz=1318.0,
        harmonic_rolloff_db_oct=-3.0,
        brightness_hz=3000.0,
        attack_char="impulsive",
        category="plucked_string",
        notes="Body formants less relevant; pickup and amp EQ dominate timbre",
    ),
    "banjo": FormantProfile(
        label="Banjo",
        formant_hz=(250.0, 600.0),
        formant_bw_hz=(50.0, 120.0),
        f0_min_hz=130.0,
        f0_max_hz=1760.0,
        harmonic_rolloff_db_oct=-2.5,
        brightness_hz=3500.0,
        attack_char="impulsive",
        category="plucked_string",
        notes="Drum-head body → very short decay; bright 3–6 kHz region",
    ),
    "harp": FormantProfile(
        label="Harp",
        formant_hz=(130.0, 280.0, 600.0),
        formant_bw_hz=(30.0, 60.0, 100.0),
        f0_min_hz=32.0,
        f0_max_hz=3322.0,
        harmonic_rolloff_db_oct=-5.5,
        brightness_hz=1800.0,
        attack_char="impulsive",
        category="plucked_string",
    ),
}

# ---------------------------------------------------------------------------
# Wind instruments (Rossing 2002 §13–15; Wolfe 2021)
# ---------------------------------------------------------------------------

_WIND_DB: dict[str, FormantProfile] = {
    "flute": FormantProfile(
        label="Flute",
        formant_hz=(800.0, 1800.0),
        formant_bw_hz=(200.0, 400.0),
        f0_min_hz=261.0,
        f0_max_hz=3136.0,
        harmonic_rolloff_db_oct=-12.0,
        brightness_hz=5000.0,
        attack_char="breath",
        category="woodwind",
        notes="Flute: near-sinusoidal in low register; strong 5th–6th partial in upper register",
    ),
    "piccolo": FormantProfile(
        label="Piccolo",
        formant_hz=(1200.0, 3000.0),
        formant_bw_hz=(300.0, 600.0),
        f0_min_hz=523.0,
        f0_max_hz=4186.0,
        harmonic_rolloff_db_oct=-12.0,
        brightness_hz=7000.0,
        attack_char="breath",
        category="woodwind",
    ),
    "oboe": FormantProfile(
        label="Oboe",
        formant_hz=(420.0, 1600.0, 3500.0),
        formant_bw_hz=(80.0, 200.0, 500.0),
        f0_min_hz=233.0,
        f0_max_hz=1568.0,
        harmonic_rolloff_db_oct=-6.0,
        brightness_hz=3800.0,
        attack_char="gradual",
        category="woodwind",
        notes="Double reed: rich in odd+even harmonics; characteristic nasal F2=1600 Hz",
    ),
    "clarinet": FormantProfile(
        label="Clarinet",
        formant_hz=(1400.0, 3000.0),
        formant_bw_hz=(200.0, 400.0),
        f0_min_hz=165.0,
        f0_max_hz=1568.0,
        harmonic_rolloff_db_oct=-7.0,
        brightness_hz=3500.0,
        attack_char="gradual",
        category="woodwind",
        notes="Cylindrical bore: odd harmonics dominant; chalumeau (LF) vs clarion (HF) register",
    ),
    "bassoon": FormantProfile(
        label="Bassoon",
        formant_hz=(250.0, 700.0, 1800.0),
        formant_bw_hz=(60.0, 130.0, 300.0),
        f0_min_hz=58.0,
        f0_max_hz=740.0,
        harmonic_rolloff_db_oct=-7.0,
        brightness_hz=1500.0,
        attack_char="gradual",
        category="woodwind",
    ),
    "saxophone_alto": FormantProfile(
        label="Alto Saxophone",
        formant_hz=(500.0, 1200.0, 2800.0),
        formant_bw_hz=(100.0, 200.0, 400.0),
        f0_min_hz=138.0,
        f0_max_hz=880.0,
        harmonic_rolloff_db_oct=-5.0,
        brightness_hz=2800.0,
        attack_char="gradual",
        category="woodwind",
    ),
    "saxophone_tenor": FormantProfile(
        label="Tenor Saxophone",
        formant_hz=(400.0, 950.0, 2500.0),
        formant_bw_hz=(100.0, 200.0, 400.0),
        f0_min_hz=103.0,
        f0_max_hz=660.0,
        harmonic_rolloff_db_oct=-5.5,
        brightness_hz=2400.0,
        attack_char="gradual",
        category="woodwind",
    ),
    "trumpet": FormantProfile(
        label="Trumpet",
        formant_hz=(1000.0, 2500.0),
        formant_bw_hz=(300.0, 600.0),
        f0_min_hz=165.0,
        f0_max_hz=1396.0,
        harmonic_rolloff_db_oct=-3.5,
        brightness_hz=3500.0,
        attack_char="impulsive",
        category="brass",
        notes="Bright, even-odd rich harmonic series; bell radiation boost 2–5 kHz",
    ),
    "trombone": FormantProfile(
        label="Trombone",
        formant_hz=(600.0, 1500.0),
        formant_bw_hz=(200.0, 400.0),
        f0_min_hz=58.0,
        f0_max_hz=523.0,
        harmonic_rolloff_db_oct=-4.5,
        brightness_hz=2000.0,
        attack_char="gradual",
        category="brass",
    ),
    "french_horn": FormantProfile(
        label="French Horn",
        formant_hz=(400.0, 1000.0),
        formant_bw_hz=(150.0, 300.0),
        f0_min_hz=87.0,
        f0_max_hz=932.0,
        harmonic_rolloff_db_oct=-6.0,
        brightness_hz=1600.0,
        attack_char="gradual",
        category="brass",
        notes="Bell points backward → indirect radiation; warm, mellow timbre",
    ),
    "tuba": FormantProfile(
        label="Tuba",
        formant_hz=(200.0, 500.0),
        formant_bw_hz=(80.0, 150.0),
        f0_min_hz=29.0,
        f0_max_hz=293.0,
        harmonic_rolloff_db_oct=-7.0,
        brightness_hz=800.0,
        attack_char="gradual",
        category="brass",
    ),
    "harmonica": FormantProfile(
        label="Harmonica",
        formant_hz=(500.0, 1500.0),
        formant_bw_hz=(100.0, 300.0),
        f0_min_hz=130.0,
        f0_max_hz=2093.0,
        harmonic_rolloff_db_oct=-5.0,
        brightness_hz=2500.0,
        attack_char="breath",
        category="woodwind",
    ),
}

# ---------------------------------------------------------------------------
# Keyboard instruments
# ---------------------------------------------------------------------------

_KEYBOARD_DB: dict[str, FormantProfile] = {
    "grand_piano": FormantProfile(
        label="Grand Piano",
        formant_hz=(60.0, 120.0, 500.0),
        formant_bw_hz=(10.0, 20.0, 80.0),
        f0_min_hz=27.5,
        f0_max_hz=4186.0,
        harmonic_rolloff_db_oct=-6.0,
        brightness_hz=2200.0,
        attack_char="impulsive",
        category="keyboard",
        notes="Inharmonicity increases with string shortness toward treble; "
        "soundboard resonances 50–300 Hz critical for warmth",
    ),
    "upright_piano": FormantProfile(
        label="Upright Piano",
        formant_hz=(70.0, 130.0, 600.0),
        formant_bw_hz=(12.0, 25.0, 100.0),
        f0_min_hz=27.5,
        f0_max_hz=4186.0,
        harmonic_rolloff_db_oct=-6.5,
        brightness_hz=1900.0,
        attack_char="impulsive",
        category="keyboard",
    ),
    "harpsichord": FormantProfile(
        label="Harpsichord",
        formant_hz=(100.0, 250.0, 800.0),
        formant_bw_hz=(20.0, 40.0, 100.0),
        f0_min_hz=32.0,
        f0_max_hz=3136.0,
        harmonic_rolloff_db_oct=-4.0,
        brightness_hz=3000.0,
        attack_char="impulsive",
        category="keyboard",
    ),
    "pipe_organ": FormantProfile(
        label="Pipe Organ",
        formant_hz=(60.0, 250.0, 800.0),
        formant_bw_hz=(5.0, 20.0, 80.0),
        f0_min_hz=16.0,
        f0_max_hz=8372.0,
        harmonic_rolloff_db_oct=-9.0,
        brightness_hz=2500.0,
        attack_char="gradual",
        category="keyboard",
        notes="Wideband: 16 Hz 32' stop to 8372 Hz 1' stop; room acoustics critical",
    ),
    "accordion": FormantProfile(
        label="Accordion",
        formant_hz=(300.0, 900.0, 2000.0),
        formant_bw_hz=(80.0, 150.0, 250.0),
        f0_min_hz=55.0,
        f0_max_hz=2093.0,
        harmonic_rolloff_db_oct=-5.5,
        brightness_hz=2200.0,
        attack_char="gradual",
        category="keyboard",
        notes="Reed instrument; musette tuning creates chorus beating ~5–15 Hz",
    ),
}

# ---------------------------------------------------------------------------
# Percussion instruments
# ---------------------------------------------------------------------------

_PERCUSSION_DB: dict[str, FormantProfile] = {
    "snare_drum": FormantProfile(
        label="Snare Drum",
        formant_hz=(170.0, 350.0, 1000.0),
        formant_bw_hz=(80.0, 150.0, 400.0),
        f0_min_hz=150.0,
        f0_max_hz=400.0,
        harmonic_rolloff_db_oct=-3.0,
        brightness_hz=4000.0,
        attack_char="impulsive",
        category="percussion",
        notes="Snare buzz: broad 3–8 kHz; fundamental 150–350 Hz depending on tuning",
    ),
    "bass_drum": FormantProfile(
        label="Bass Drum (Kick)",
        formant_hz=(50.0, 100.0, 300.0),
        formant_bw_hz=(20.0, 50.0, 100.0),
        f0_min_hz=40.0,
        f0_max_hz=150.0,
        harmonic_rolloff_db_oct=-5.0,
        brightness_hz=700.0,
        attack_char="impulsive",
        category="percussion",
    ),
    "hi_hat": FormantProfile(
        label="Hi-Hat",
        formant_hz=(5000.0, 8000.0, 12000.0),
        formant_bw_hz=(1000.0, 2000.0, 4000.0),
        f0_min_hz=4000.0,
        f0_max_hz=16000.0,
        harmonic_rolloff_db_oct=-1.5,
        brightness_hz=9000.0,
        attack_char="impulsive",
        category="percussion",
        notes="Inharmonic metallic spectrum; most energy above 4 kHz",
    ),
    "timpani": FormantProfile(
        label="Timpani",
        formant_hz=(88.0, 148.0, 204.0),
        formant_bw_hz=(10.0, 20.0, 35.0),
        f0_min_hz=80.0,
        f0_max_hz=180.0,
        harmonic_rolloff_db_oct=-7.0,
        brightness_hz=600.0,
        attack_char="impulsive",
        category="percussion",
        notes="Circular membrane: modes at 1.0/1.68/2.28 × fundamental (not harmonic)",
    ),
    "marimba": FormantProfile(
        label="Marimba",
        formant_hz=(200.0, 800.0),
        formant_bw_hz=(30.0, 100.0),
        f0_min_hz=130.0,
        f0_max_hz=1047.0,
        harmonic_rolloff_db_oct=-8.0,
        brightness_hz=1200.0,
        attack_char="impulsive",
        category="percussion",
        notes="Deeply undercut bars → 4th partial = 4f0; warm, fast decay",
    ),
    "xylophone": FormantProfile(
        label="Xylophone",
        formant_hz=(300.0, 1200.0),
        formant_bw_hz=(60.0, 200.0),
        f0_min_hz=392.0,
        f0_max_hz=4186.0,
        harmonic_rolloff_db_oct=-6.0,
        brightness_hz=3500.0,
        attack_char="impulsive",
        category="percussion",
    ),
    "tambourine": FormantProfile(
        label="Tambourine",
        formant_hz=(1500.0, 4000.0, 8000.0),
        formant_bw_hz=(400.0, 1000.0, 2000.0),
        f0_min_hz=1000.0,
        f0_max_hz=12000.0,
        harmonic_rolloff_db_oct=-2.0,
        brightness_hz=5000.0,
        attack_char="impulsive",
        category="percussion",
    ),
}

# ---------------------------------------------------------------------------
# World / Folk instruments
# ---------------------------------------------------------------------------

_WORLD_DB: dict[str, FormantProfile] = {
    "mandolin": FormantProfile(
        label="Mandolin",
        formant_hz=(220.0, 450.0, 1200.0),
        formant_bw_hz=(40.0, 80.0, 200.0),
        f0_min_hz=196.0,
        f0_max_hz=2093.0,
        harmonic_rolloff_db_oct=-5.0,
        brightness_hz=3000.0,
        attack_char="impulsive",
        category="plucked_string",
        notes="Short scale: bright transient, quick decay; tremolo is characteristic",
    ),
    "bouzouki": FormantProfile(
        label="Bouzouki",
        formant_hz=(130.0, 300.0, 800.0),
        formant_bw_hz=(30.0, 60.0, 120.0),
        f0_min_hz=130.0,
        f0_max_hz=1320.0,
        harmonic_rolloff_db_oct=-5.5,
        brightness_hz=2200.0,
        attack_char="impulsive",
        category="plucked_string",
    ),
    "sitar": FormantProfile(
        label="Sitar",
        formant_hz=(200.0, 600.0, 2000.0),
        formant_bw_hz=(60.0, 120.0, 400.0),
        f0_min_hz=65.0,
        f0_max_hz=1000.0,
        harmonic_rolloff_db_oct=-4.0,
        brightness_hz=3000.0,
        attack_char="impulsive",
        category="plucked_string",
        notes="Jawari bridge: continuous spectral evolution; sympathetic string resonance",
    ),
    "erhu": FormantProfile(
        label="Erhu (Chinese violin)",
        formant_hz=(400.0, 900.0, 2500.0),
        formant_bw_hz=(100.0, 200.0, 500.0),
        f0_min_hz=196.0,
        f0_max_hz=2093.0,
        harmonic_rolloff_db_oct=-5.0,
        brightness_hz=3500.0,
        attack_char="bowed",
        category="bowed_string",
    ),
    "oud": FormantProfile(
        label="Oud",
        formant_hz=(150.0, 320.0, 900.0),
        formant_bw_hz=(35.0, 70.0, 150.0),
        f0_min_hz=73.0,
        f0_max_hz=1000.0,
        harmonic_rolloff_db_oct=-5.5,
        brightness_hz=1800.0,
        attack_char="impulsive",
        category="plucked_string",
    ),
    "balalaika": FormantProfile(
        label="Balalaika",
        formant_hz=(300.0, 700.0),
        formant_bw_hz=(60.0, 120.0),
        f0_min_hz=196.0,
        f0_max_hz=1760.0,
        harmonic_rolloff_db_oct=-5.0,
        brightness_hz=2500.0,
        attack_char="impulsive",
        category="plucked_string",
    ),
    "zither": FormantProfile(
        label="Zither",
        formant_hz=(150.0, 400.0, 1200.0),
        formant_bw_hz=(30.0, 80.0, 200.0),
        f0_min_hz=65.0,
        f0_max_hz=2093.0,
        harmonic_rolloff_db_oct=-6.0,
        brightness_hz=2000.0,
        attack_char="impulsive",
        category="plucked_string",
    ),
}

# ---------------------------------------------------------------------------
# Consolidated lookup table
# ---------------------------------------------------------------------------

INSTRUMENT_DB: dict[str, FormantProfile] = {
    **_VOICE_DB,
    **_STRING_DB,
    **_WIND_DB,
    **_KEYBOARD_DB,
    **_PERCUSSION_DB,
    **_WORLD_DB,
}

# Category → instrument keys
_CATEGORY_INDEX: dict[str, list[str]] = {}
for _k, _v in INSTRUMENT_DB.items():
    _CATEGORY_INDEX.setdefault(_v.category, []).append(_k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_formant_profile(instrument_key: str) -> FormantProfile | None:
    """Gibt the FormantProfile for an instrument key (case-insensitive) zurück.

    Args:
        instrument_key: Instrument name key (e.g. "violin", "soprano", "trumpet").

    Returns:
        FormantProfile or None if not found.
    """
    key = instrument_key.strip().lower().replace(" ", "_").replace("-", "_")
    profile = INSTRUMENT_DB.get(key)
    if profile is None:
        logger.debug("instrument_formant_db: no Profil for %r", instrument_key)
    return profile


def get_formant_mask_hz(
    instrument_key: str,
    *,
    tolerance_factor: float = 1.5,
) -> list[tuple[float, float]] | None:
    """Gibt Hz frequency ranges that should be protected from aggressive NR zurück.

    Each tuple is (low_hz, high_hz) defining a protected frequency band around
    each formant (centre ± bandwidth × tolerance_factor).

    Args:
        instrument_key:     Instrument name key.
        tolerance_factor:   Bandwidth expansion factor [0.5..3.0].

    Returns:
        List of (low_hz, high_hz) tuples, or None if unknown instrument.
    """
    profile = get_formant_profile(instrument_key)
    if profile is None:
        return None
    regions: list[tuple[float, float]] = []
    for f, bw in zip(profile.formant_hz, profile.formant_bw_hz):
        half = bw * tolerance_factor * 0.5
        regions.append((max(20.0, f - half), f + half))
    return regions


def get_instruments_by_category(category: str) -> list[str]:
    """Gibt all instrument keys in a given category zurück.

    Args:
        category: e.g. "voice", "bowed_string", "woodwind", "brass",
                  "plucked_string", "keyboard", "percussion".

    Returns:
        Sorted list of instrument key strings.
    """
    return sorted(_CATEGORY_INDEX.get(category, []))


def estimate_dominant_instrument(
    spectral_centroid_hz: float,
    f0_hz: float | None,
    has_sustained_tone: bool,
) -> list[str]:
    """Heuristic guess of likely instrument types from coarse spectral features.

    Useful as a prior when no genre/label information is available.

    Args:
        spectral_centroid_hz:   Measured spectral centroid of the signal.
        f0_hz:                  Estimated fundamental frequency (None if unknown).
        has_sustained_tone:     True if the signal has sustained tones (vs percussive).

    Returns:
        Ranked list of instrument key guesses (up to 5, most likely first).
    """
    candidates: list[tuple[float, str]] = []
    for key, prof in INSTRUMENT_DB.items():
        if f0_hz is not None:
            if not (prof.f0_min_hz * 0.5 <= f0_hz <= prof.f0_max_hz * 2.0):
                continue
        centroid_distance = abs(np.log2(max(spectral_centroid_hz, 100.0)) - np.log2(max(prof.brightness_hz, 100.0)))
        if not has_sustained_tone and prof.attack_char in ("bowed", "breath"):
            continue
        candidates.append((centroid_distance, key))

    candidates.sort()
    return [k for _, k in candidates[:5]]


def get_nr_bypass_mask_bark(
    instrument_key: str,
    sr: int = 48000,
) -> np.ndarray | None:
    """Gibt a 24-element boolean array indicating Bark bands to bypass NR zurück.

    Formant regions should not be aggressively denoised — they carry the
    timbral identity of the instrument.  This mask gates the NR algorithm.

    Args:
        instrument_key: Instrument name key.
        sr:             Sample rate (used for Bark-band centre computation).

    Returns:
        np.ndarray of shape (24,) bool — True = protect this band from NR.
        Returns None if instrument unknown.
    """
    regions = get_formant_mask_hz(instrument_key)
    if regions is None:
        return None

    # 24-Bark band centres (Zwicker & Fastl 1999) — matching tonal_reference_profile
    bark_centers = np.array(
        [
            50,
            150,
            250,
            350,
            450,
            570,
            700,
            840,
            1000,
            1170,
            1370,
            1600,
            1850,
            2150,
            2500,
            2900,
            3400,
            4000,
            4800,
            5800,
            7000,
            8500,
            10500,
            13500,
        ],
        dtype=np.float32,
    )

    mask = np.zeros(24, dtype=bool)
    for low, high in regions:
        mask |= (bark_centers >= low) & (bark_centers <= high)
    return mask  # type: ignore[no-any-return]


def list_all_instruments() -> list[str]:
    """Gibt all available instrument keys sorted alphabetically zurück."""
    return sorted(INSTRUMENT_DB.keys())


def list_categories() -> list[str]:
    """Gibt all instrument categories zurück."""
    return sorted(_CATEGORY_INDEX.keys())


__all__ = [
    "FormantProfile",
    "INSTRUMENT_DB",
    "get_formant_profile",
    "get_formant_mask_hz",
    "get_instruments_by_category",
    "estimate_dominant_instrument",
    "get_nr_bypass_mask_bark",
    "list_all_instruments",
    "list_categories",
]
