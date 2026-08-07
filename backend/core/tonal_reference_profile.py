"""
backend/core/tonal_reference_profile.py
Aurik 10.0.0 — Tonal Reference Profile: per-song spectral ceiling for phase steering.

Derives a 24-Bark-band gain ceiling curve from Era × Genre × Material context.
Used by Phase 06 (frequency restoration) and Phase 07 (harmonic restoration) as a
hard stop: phases must not boost any Bark band beyond its per-band ceiling.

Two complementary arrays per song context:
  band_ceiling_db  — hard stop (§0h §2.46e §6.2c): no boost beyond this level.
  band_target_db   — studio-day reconstruction target (§2.46): phases steer
                     TOWARD this curve when restoring lost frequency components;
                     never exceeds the ceiling.

The target is the combined transfer function of the original recording chain:
  Microphone FR + Console/Preamp EQ + Tape Machine response.
This gives phases fidelity-first steering: not just "up to here" but
"toward this specific spectral shape as it sounded in the studio that day".

Key invariants (§0h + §2.46e + §0a):
  Ceiling is absolute — no boost beyond it under any circumstance.
  Target is only a steering guide — phases must never be forced to reach it.
  Both arrays are 0 dB = flat reference; negative = original chain was attenuated.

Scientific references:
  Eargle (2004): «The Microphone Book» — era-specific mic frequency responses
  Zwicker & Fastl (1999): «Psychoacoustics» — 24-band Bark scale
  Copeland (2008): «Manual of Analogue Sound Restoration Techniques» — chain losses
  Kefauver & Patschke (2007): «Fundamentals of Digital Audio» — tape BW per IPS
  Voss & Clarke (1975): «1/f noise in music» — genre spectral balance
  Blauert & Braasch (2011): «Auditory Virtual Environments» — room imprint
  Neve (1970s schematics): 1073 transformer resonance +3 dB shelf at 12 kHz
  EMI (1950s–70s): REDD.37/TG12345 console measurements (Doyle 2004)
  Yeh et al. (2008) IEEE: vacuum tube harmonic distortion modeling
  Välimäki (2011): harmonic era classification

Singleton: get_tonal_reference_profiler()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import istft as _istft
from scipy.signal import stft as _stft

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bark scale (24 critical bands, Zwicker & Fastl 1999)
# ---------------------------------------------------------------------------

_BARK_EDGES_HZ: list[float] = [
    20.0,
    100.0,
    200.0,
    300.0,
    400.0,
    510.0,
    630.0,
    770.0,
    920.0,
    1080.0,
    1270.0,
    1480.0,
    1720.0,
    2000.0,
    2320.0,
    2700.0,
    3150.0,
    3700.0,
    4400.0,
    5300.0,
    6400.0,
    7700.0,
    9500.0,
    12000.0,
    15500.0,
]
_BARK_CENTERS_HZ: list[float] = [float(np.sqrt(_BARK_EDGES_HZ[i] * _BARK_EDGES_HZ[i + 1])) for i in range(24)]

# ---------------------------------------------------------------------------
# Era bandwidth table (Eargle 2004 / Copeland 2008, same as source_fidelity_reconstructor)
# ---------------------------------------------------------------------------

_ERA_BW_HZ: dict[int, float] = {
    1900: 3500.0,
    1910: 4000.0,
    1920: 6000.0,
    1930: 9000.0,
    1940: 12000.0,
    1950: 14500.0,
    1960: 16500.0,
    1970: 18500.0,
    1980: 20000.0,
    1990: 20000.0,
    2000: 20000.0,
    2010: 20000.0,
    2020: 20000.0,
    2025: 20000.0,
}

# Material hard BW ceilings (§0a §6.2c — same as Phase 06/07 BW_CEILING)
_MATERIAL_BW_CEILING_HZ: dict[str, float] = {
    "wax_cylinder": 5000.0,
    "wire_recording": 5000.0,
    "shellac": 8000.0,
    "lacquer_disc": 10000.0,
    "vinyl": 16000.0,
    "cassette": 14000.0,  # central definition default; Type II Chrome ≤ 15 kHz (unknown type → conservative)
    "reel_tape": 18000.0,
    "tape": 16000.0,
    "dat": 22000.0,
    "minidisc": 18000.0,
    "cd_digital": 22000.0,
    "mp3_low": 16000.0,
    "mp3_high": 20000.0,
    "aac": 20000.0,
    "streaming": 20000.0,
    "flac": 22000.0,
}

# Maximum allowed reconstruction boost (dB) per Bark band
_MAX_BOOST_DB: float = 4.0
# Rolloff steepness above era bandwidth: dB per octave
_ROLLOFF_DB_PER_OCT: float = 8.0


# ---------------------------------------------------------------------------
# Helper: breakpoint curve interpolation
# ---------------------------------------------------------------------------


def _interp_db_curve(freq_hz: float, breakpoints: list[tuple[float, float]]) -> float:
    """Linear interpolation on a (Hz, dB) breakpoint list. Clamps at ends."""
    if not breakpoints:
        return 0.0
    if freq_hz <= breakpoints[0][0]:
        return float(breakpoints[0][1])
    if freq_hz >= breakpoints[-1][0]:
        return float(breakpoints[-1][1])
    for i in range(len(breakpoints) - 1):
        f0, d0 = breakpoints[i]
        f1, d1 = breakpoints[i + 1]
        if f0 <= freq_hz < f1:
            t = (freq_hz - f0) / (f1 - f0 + 1e-12)
            return float(d0 + t * (d1 - d0))
    return float(breakpoints[-1][1])


# ===========================================================================
# MICROPHONE FREQUENCY RESPONSE PER ERA
# (Hz, dB) breakpoints — relative to 1 kHz = 0 dB; Eargle 2004.
#
# 1900: Carbon capsule/acoustic horn — extreme bandpass, 1–2 kHz peak
# 1920: Western Electric 394A condenser — usable to ~6 kHz
# 1930: RCA 44A/44B Ribbon — smooth, warm, flat 100 Hz – 8 kHz
# 1950: Neumann U47/M49 — golden-era presence peak 5–9 kHz
# 1960: Neumann U67/AKG C12 — extended presence 4–10 kHz
# 1970: Neumann U87/AKG C414 — modern, flat, +2 dB at 8 kHz
# 1980+: Near ruler-flat large condensers (DPA, Schoeps, TLM103)
# ===========================================================================

_ERA_MIC_RESPONSE: dict[int, list[tuple[float, float]]] = {
    1900: [
        (20.0, -24.0),
        (100.0, -12.0),
        (300.0, -4.0),
        (
            600.0,
            -1.0,
        ),  # (1000.0, 0.0), (1500.0, +1.5), (2000.0, +1.0), (3000.0, -2.0),
        # (4000.0, -8.0), (6000.0, -18.0), (10000.0, -36.0),
    ],
    1910: [
        (20.0, -20.0),
        (80.0, -10.0),
        (200.0, -3.0),
        (
            500.0,
            -0.5,
        ),  # (1000.0, 0.0), (1500.0, +1.0), (2500.0, -1.0), (4000.0, -6.0),
        # (6000.0, -14.0), (10000.0, -30.0),
    ],
    # Western Electric 394A condenser (1925). Usable to ~6 kHz.
    1920: [
        (20.0, -16.0),
        (60.0, -6.0),
        (200.0, -1.5),
        (
            500.0,
            -0.5,
        ),  # (1000.0, 0.0), (2000.0, +0.5), (3000.0, +1.0), (5000.0, -1.0),
        # (6000.0, -3.0), (8000.0, -10.0), (12000.0, -24.0),
    ],
    # RCA 44A/44B Ribbon (1932). Smooth, warm. Almost flat 100 Hz – 8 kHz.
    1930: [
        (20.0, -10.0),
        (80.0, -3.0),
        (150.0, -1.0),
        (
            300.0,
            0.0,
        ),  # (1000.0, 0.0), (3000.0, -0.3), (5000.0, -0.5), (7000.0, -1.5),
        # (8000.0, -3.0), (10000.0, -7.0), (14000.0, -18.0),
    ],
    # Neumann CMV3; RCA 44B dominant; some U47 proto.
    1940: [
        (20.0, -8.0),
        (60.0, -2.0),
        (100.0, -0.5),
        (
            300.0,
            0.0,
        ),  # (1000.0, 0.0), (3000.0, +0.3), (5000.0, +0.8), (7000.0, +0.3),
        # (8000.0, -0.5), (10000.0, -2.5), (12000.0, -6.0), (16000.0, -18.0),
    ],
    # Neumann U47 (1947) / M49 — golden era. Presence peak 5–9 kHz.
    1950: [
        (20.0, -6.0),
        (50.0, -1.5),
        (100.0, -0.3),
        (
            300.0,
            0.0,
        ),  # (1000.0, 0.0), (3000.0, +0.5), (5000.0, +1.5), (7000.0, +2.0),
        # (9000.0, +1.5), (11000.0, 0.0), (13000.0, -2.0), (16000.0, -8.0),  # (20000.0, -20.0),
    ],
    # Neumann U67 (1960) / AKG C12. Presence 4–10 kHz + LF warmth.
    1960: [
        (20.0, -5.0),
        (50.0, -1.0),
        (100.0, +0.3),
        (
            200.0,
            +0.5,
        ),  # (400.0, 0.0), (1000.0, 0.0), (3000.0, +0.8), (5000.0, +2.5),
        # (8000.0, +3.0), (10000.0, +2.5), (12000.0, +1.0), (15000.0, -1.0),  # (18000.0, -6.0), (20000.0, -12.0),
    ],
    # Neumann U87 (1967) / AKG C414 (1971). Linear, +2 dB @ 8 kHz.
    1970: [
        (20.0, -4.0),
        (50.0, -0.5),
        (100.0, 0.0),
        (
            500.0,
            0.0,
        ),  # (1000.0, 0.0), (3000.0, +0.5), (5000.0, +1.5), (8000.0, +2.0),
        # (12000.0, +1.5), (15000.0, +0.5), (18000.0, -1.5), (20000.0, -4.0),
    ],
    # Modern condensers (DPA 4006, Schoeps MK4, TLM103). Ruler-flat.
    1980: [
        (20.0, -2.0),
        (50.0, 0.0),
        (200.0, 0.0),
        (1000.0, 0.0),  # (5000.0, +0.5), (10000.0, +0.5), (15000.0, 0.0), (20000.0, -1.0),
    ],
    1990: [(20.0, -1.0), (100.0, 0.0), (1000.0, 0.0), (10000.0, 0.0), (20000.0, -0.5)],
    2000: [(20.0, -1.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, -0.5)],
    2010: [(20.0, -1.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2020: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2025: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
}


# ===========================================================================
# CONSOLE / PREAMP EQ SIGNATURE PER ERA
# (Hz, dB) — relative to flat = 0 dB at 1 kHz.
#
# 1930–1945: Western Electric/RCA early tube consoles. Transformer core warmth.
# 1945–1965: EMI REDD.17/REDD.37. Clean, slight mid-scoop (Doyle 2004).
# 1965–1975: Neve 1073. +3 dB shelf at 12 kHz (output transformer resonance).
#   Reference: Neve 1073 published measurements (Senior, Sound On Sound 2011).
# 1975–1985: SSL 4000/4056. Clinically flat, fast transient response.
# 1985+: Near-ideal flat (digital inline desks).
# ===========================================================================

_ERA_CONSOLE_EQ: dict[int, list[tuple[float, float]]] = {
    1900: [
        (20.0, -8.0),
        (60.0, -4.0),
        (200.0, -1.0),
        (1000.0, 0.0),  # (3000.0, +1.0), (5000.0, 0.0), (8000.0, -2.0), (12000.0, -8.0),
    ],
    1910: [
        (20.0, -6.0),
        (80.0, -3.0),
        (200.0, -1.0),
        (1000.0, 0.0),  # (4000.0, +0.5), (8000.0, -1.5), (12000.0, -6.0),
    ],
    1920: [
        (20.0, -5.0),
        (80.0, -2.0),
        (300.0, 0.0),
        (1000.0, 0.0),  # (5000.0, 0.0), (8000.0, -1.0), (12000.0, -4.0),
    ],
    # Early tube console: warm transformer core, mild upper-mid bump
    1930: [
        (20.0, -4.0),
        (60.0, -2.0),
        (200.0, +0.5),
        (
            600.0,
            +0.8,
        ),  # (1000.0, 0.0), (2000.0, +0.5), (4000.0, 0.0),  # (8000.0, -0.5), (12000.0, -2.0), (16000.0, -5.0),
    ],
    1940: [
        (20.0, -3.0),
        (60.0, -1.5),
        (200.0, +0.5),
        (1000.0, 0.0),  # (3000.0, +0.3), (6000.0, 0.0), (10000.0, -1.0), (14000.0, -3.5),
    ],
    # EMI REDD.37 (1950s–60s): clean, slight mid-scoop
    1950: [
        (20.0, -2.5),
        (60.0, -1.0),
        (200.0, 0.0),
        (1000.0, 0.0),  # (3000.0, -0.3), (6000.0, 0.0), (10000.0, -0.5), (16000.0, -2.0),
    ],
    # EMI TG12345 (1968+): cleaner, extends to 16+ kHz
    1960: [
        (20.0, -2.0),
        (60.0, -0.8),
        (200.0, 0.0),
        (1000.0, 0.0),  # (3000.0, 0.0), (6000.0, +0.3), (10000.0, 0.0), (16000.0, -1.0),
    ],
    # Neve 1073 (1970): +3 dB shelf at 12 kHz, 3 kHz presence knee, warm LF.
    1970: [
        (20.0, -1.5),
        (60.0, +0.5),
        (100.0, +1.0),
        (
            200.0,
            +0.8,
        ),  # (400.0, 0.0), (1000.0, 0.0), (3000.0, +0.5), (6000.0, +0.8),
        # (9000.0, +1.5), (12000.0, +2.5), (16000.0, +3.0), (20000.0, +2.5),
    ],
    # SSL 4000 (1977): flat, slight 3–5 kHz from circuit topology
    1980: [
        (20.0, -1.0),
        (50.0, 0.0),
        (200.0, 0.0),
        (1000.0, 0.0),  # (3000.0, +0.3), (5000.0, +0.5), (10000.0, 0.0), (20000.0, -0.5),
    ],
    1990: [(20.0, -0.5), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2000: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2010: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2020: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2025: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
}


# ===========================================================================
# TAPE MACHINE FREQUENCY RESPONSE PER ERA
# (Hz, dB) at standard playback EQ, relative to 1 kHz = 0 dB.
#
# Based on tape speed + formulation (Copeland 2008, Kefauver & Patschke 2007):
# 1930–1945: Early oxide (AEG Magnetophon K4). HF limit ~7 kHz, LF bump.
# 1945–1955: Ampex 200/350. NAB EQ 1953. Up to ~10 kHz.
# 1955–1965: Studer A62/Ampex 300. Agfa PE36 "Agfa sound" HF elevation.
# 1965–1975: Ampex 456/Studer A80. Famous warmth + sweet HF at 30 ips.
# 1975–1985: BASF 900/3M 996 at 30 ips. Near-flat to 20 kHz.
# 1985+: DAT/digital. No tape response.
# ===========================================================================

_ERA_TAPE_RESPONSE: dict[int, list[tuple[float, float]]] = {
    1900: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (5000.0, 0.0)],
    1910: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (5000.0, 0.0)],
    1920: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (5000.0, 0.0)],
    # AEG Magnetophon K4 (1935+). 15 ips. Thick oxide → LF bump, rapid HF rolloff.
    1930: [
        (20.0, +1.0),
        (80.0, +1.5),
        (200.0, +1.0),
        (
            400.0,
            +0.5,
        ),  # (1000.0, 0.0), (3000.0, -0.5), (5000.0, -1.5), (7000.0, -4.0),  # (9000.0, -10.0), (12000.0, -20.0),
    ],
    # Ampex 200/Magnecord 15 ips. NAB EQ emerging. Up to ~10 kHz.
    1940: [
        (20.0, +0.8),
        (60.0, +1.0),
        (150.0, +0.5),
        (
            400.0,
            0.0,
        ),  # (1000.0, 0.0), (4000.0, -0.5), (6000.0, -1.5), (8000.0, -3.5),  # (10000.0, -8.0), (12000.0, -18.0),
    ],
    # Ampex 350/351 at 15 ips (1950s). NAB EQ standardised 1953.
    1950: [
        (20.0, +0.5),
        (40.0, +0.8),
        (100.0, +0.3),
        (
            400.0,
            0.0,
        ),  # (1000.0, 0.0), (5000.0, -0.3), (8000.0, -1.0), (10000.0, -2.5),
        # (12000.0, -5.0), (14000.0, -10.0), (16000.0, -20.0),
    ],
    # Studer A62 / Ampex 300 (1960). 15–30 ips. Agfa PE36: subtle HF elevation.
    1960: [
        (20.0, +0.3),
        (60.0, +0.5),
        (200.0, +0.2),
        (
            600.0,
            0.0,
        ),  # (1000.0, 0.0), (4000.0, +0.3), (6000.0, +0.5), (8000.0, +0.5),
        # (10000.0, 0.0), (12000.0, -0.8), (14000.0, -2.5), (16000.0, -6.0),  # (18000.0, -14.0),
    ],
    # Ampex 456 / Studer A80 at 30 ips (1970s). Warm, sweet HF (Eargle 2004).
    1970: [
        (20.0, +0.5),
        (50.0, +0.8),
        (100.0, +0.5),
        (
            400.0,
            +0.2,
        ),  # (1000.0, 0.0), (4000.0, +0.2), (8000.0, +0.5), (12000.0, +0.3),
        # (15000.0, 0.0), (18000.0, -1.5), (20000.0, -4.0),
    ],
    # BASF 900 / 3M 996 at 30 ips (1980s). Near flat.
    1980: [
        (20.0, +0.2),
        (100.0, 0.0),
        (1000.0, 0.0),  # (10000.0, 0.0), (18000.0, -0.5), (20000.0, -1.0),
    ],
    1990: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2000: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2010: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2020: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
    2025: [(20.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (20000.0, 0.0)],
}


# ===========================================================================
# HARMONIC DENSITY PROFILES PER ERA
# Tube era (pre-1965): H2 dominant, even-harmonic distortion (Yeh et al. 2008).
# Transistor era (1965–1985): H3 increases, odd harmonics, lower THD.
# Digital era (1985+): Very low THD, odd harmonics only.
# ===========================================================================


@dataclass(frozen=True)
class HarmonicProfile:
    """Expected harmonic distortion profile for an era."""

    h2_ratio: float  # Amplitude ratio H2/H1 (0.0 – 0.05)
    h3_ratio: float  # Amplitude ratio H3/H1
    h4_ratio: float  # Amplitude ratio H4/H1
    thd_pct: float  # Approximate total harmonic distortion %
    era_label: str


_ERA_HARMONIC_PROFILE: dict[int, HarmonicProfile] = {
    1900: HarmonicProfile(0.040, 0.015, 0.008, 4.5, "Acoustic/Carbon"),
    1910: HarmonicProfile(0.035, 0.012, 0.006, 3.8, "Early Electronic"),
    1920: HarmonicProfile(0.030, 0.010, 0.005, 3.2, "Early Valve"),
    1930: HarmonicProfile(0.025, 0.008, 0.004, 2.7, "Ribbon/Valve"),
    1940: HarmonicProfile(0.020, 0.006, 0.003, 2.1, "Golden Tube"),
    1950: HarmonicProfile(0.018, 0.005, 0.002, 1.9, "Classic Tube"),
    1960: HarmonicProfile(0.014, 0.006, 0.002, 1.5, "Tube/Transistor"),
    1970: HarmonicProfile(0.006, 0.008, 0.001, 0.9, "Transistor Era"),
    1980: HarmonicProfile(0.002, 0.004, 0.001, 0.4, "Op-Amp Era"),
    1990: HarmonicProfile(0.001, 0.002, 0.0005, 0.2, "Digital Era"),
    2000: HarmonicProfile(0.0005, 0.001, 0.0002, 0.1, "Modern Digital"),
    2010: HarmonicProfile(0.0003, 0.0008, 0.0001, 0.08, "Contemporary"),
    2020: HarmonicProfile(0.0002, 0.0005, 0.0001, 0.05, "Contemporary"),
    2025: HarmonicProfile(0.0002, 0.0005, 0.0001, 0.05, "Contemporary"),
}


# ===========================================================================
# NOISE FLOOR SPECTRAL TEXTURE PER MATERIAL
# §0a rauschtextur: spectral form of residual noise must match carrier profile.
# Vinyl: pink (-3 dB/oct) dominated by tribonoise.
# Tape: brownish LF + white HF hiss above 4 kHz.
# CD/Digital: white (flat) quantisation floor.
# ===========================================================================


@dataclass(frozen=True)
class NoiseTextureProfile:
    """Per-material noise floor spectral texture target."""

    spectral_slope_db_oct: float  # dB per octave (negative = HF rolloff)
    hf_hiss_shelf_hz: float  # Frequency above which hiss shelf activates
    hf_hiss_gain_db: float  # HF hiss shelf gain (0 = none)
    lf_rumble_hz: float  # Low-frequency rumble cutoff
    lf_rumble_gain_db: float  # Rumble boost below cutoff (0 = none)
    texture_label: str


_MATERIAL_NOISE_TEXTURE: dict[str, NoiseTextureProfile] = {
    "wax_cylinder": NoiseTextureProfile(-6.0, 2000.0, -2.0, 200.0, +2.0, "wax_scratch"),
    "wire_recording": NoiseTextureProfile(-5.0, 3000.0, -1.0, 150.0, +1.5, "wire_hiss"),
    "shellac": NoiseTextureProfile(-4.0, 3000.0, -1.5, 100.0, +2.0, "shellac_fry"),
    "lacquer_disc": NoiseTextureProfile(-3.5, 4000.0, -0.5, 80.0, +1.5, "lacquer_hiss"),
    "vinyl": NoiseTextureProfile(-3.0, 6000.0, 0.0, 60.0, +1.0, "vinyl_pink"),
    "cassette": NoiseTextureProfile(-2.5, 4000.0, +2.5, 80.0, +0.5, "cassette_hiss"),
    "reel_tape": NoiseTextureProfile(-2.0, 4000.0, +1.5, 40.0, +0.3, "tape_brown"),
    "tape": NoiseTextureProfile(-2.5, 4000.0, +2.0, 60.0, +0.5, "tape_hiss"),
    "dat": NoiseTextureProfile(0.0, 20000.0, 0.0, 20.0, 0.0, "digital_flat"),
    "cd_digital": NoiseTextureProfile(0.0, 20000.0, 0.0, 20.0, 0.0, "digital_flat"),
    "mp3_low": NoiseTextureProfile(0.0, 16000.0, -3.0, 20.0, 0.0, "codec_noise"),
    "mp3_high": NoiseTextureProfile(0.0, 18000.0, -1.0, 20.0, 0.0, "codec_noise"),
    "aac": NoiseTextureProfile(0.0, 18000.0, -0.5, 20.0, 0.0, "codec_noise"),
    "streaming": NoiseTextureProfile(0.0, 18000.0, -0.5, 20.0, 0.0, "codec_noise"),
    "minidisc": NoiseTextureProfile(-0.5, 17000.0, -1.5, 20.0, 0.0, "atrac_noise"),
    "flac": NoiseTextureProfile(0.0, 22000.0, 0.0, 20.0, 0.0, "digital_flat"),
}


# ---------------------------------------------------------------------------
# Genre spectral balance deltas per Bark band region (dB)
# Each entry: (freq_lo_hz, freq_hi_hz, delta_db)
# Reference: Voss & Clarke (1975); Moylan (2002) «The Art of Recording»
# ---------------------------------------------------------------------------

_GENRE_DELTAS: dict[str, list[tuple[float, float, float]]] = {
    # Jazz family --------------------------------------------------------
    "jazz": [
        (40.0, 80.0, +0.8),
        (80.0, 300.0, +1.2),
        (300.0, 800.0, +0.8),
        (800.0, 2500.0, +1.5),
        (2500.0, 5000.0, +0.5),
        (5000.0, 10000.0, -0.3),
        (10000.0, 20000.0, -0.8),
    ],
    "bebop": [
        (80.0, 300.0, +1.0),
        (800.0, 3000.0, +1.8),
        (3000.0, 6000.0, +0.5),
        (6000.0, 20000.0, -0.5),
    ],
    "big_band": [
        (40.0, 150.0, +1.5),
        (150.0, 500.0, +1.0),
        (500.0, 2000.0, +1.2),
        (2000.0, 5000.0, +0.8),
        (5000.0, 10000.0, -0.2),
        (10000.0, 20000.0, -1.0),
    ],
    # Classical / orchestral ---------------------------------------------
    "klassik": [
        (30.0, 60.0, +0.3),
        (60.0, 200.0, +0.6),
        (200.0, 800.0, +0.3),
        (800.0, 2500.0, 0.0),
        (2500.0, 5000.0, -0.4),
        (5000.0, 10000.0, -0.6),
        (10000.0, 20000.0, -1.2),
    ],
    "kammermusik": [
        (80.0, 400.0, +0.4),
        (400.0, 2000.0, +0.2),
        (2000.0, 5000.0, -0.3),
        (5000.0, 20000.0, -0.8),
    ],
    "oper": [
        (80.0, 300.0, +0.5),
        (300.0, 1000.0, +0.8),
        (1000.0, 3000.0, +1.2),
        (3000.0, 5000.0, +0.5),
        (5000.0, 10000.0, -0.4),
        (10000.0, 20000.0, -1.0),
    ],
    "chor": [
        (150.0, 500.0, +0.6),
        (500.0, 2000.0, +0.8),
        (2000.0, 4000.0, +0.5),
        (4000.0, 8000.0, -0.2),
        (8000.0, 20000.0, -0.8),
    ],
    # Blues / Americana --------------------------------------------------
    "blues": [
        (60.0, 150.0, +1.0),
        (150.0, 500.0, +1.5),
        (500.0, 1200.0, +0.8),
        (1200.0, 3000.0, +0.5),
        (3000.0, 6000.0, -0.2),
        (6000.0, 12000.0, -0.5),
        (12000.0, 20000.0, -1.5),
    ],
    "country": [
        (80.0, 250.0, +0.8),
        (250.0, 800.0, +0.5),
        (800.0, 2500.0, +1.0),
        (2500.0, 5000.0, +0.8),
        (5000.0, 10000.0, +0.3),
        (10000.0, 20000.0, -0.3),
    ],
    "bluegrass": [
        (80.0, 250.0, +0.5),
        (250.0, 600.0, +0.8),
        (600.0, 2000.0, +1.0),
        (2000.0, 5000.0, +1.5),
        (5000.0, 10000.0, +0.8),
        (10000.0, 20000.0, -0.2),
    ],
    # Gospel / Soul / R&B ------------------------------------------------
    "gospel": [
        (80.0, 300.0, +1.0),
        (300.0, 1000.0, +1.2),
        (1000.0, 3000.0, +1.5),
        (3000.0, 6000.0, +0.8),
        (6000.0, 12000.0, +0.3),
        (12000.0, 20000.0, -0.3),
    ],
    "soul": [
        (60.0, 200.0, +1.2),
        (200.0, 600.0, +0.8),
        (600.0, 2000.0, +1.0),
        (2000.0, 5000.0, +0.8),
        (5000.0, 10000.0, +0.3),
        (10000.0, 20000.0, -0.5),
    ],
    "soul/r&b": [
        (60.0, 200.0, +1.2),
        (200.0, 600.0, +0.8),
        (600.0, 2000.0, +1.0),
        (2000.0, 5000.0, +0.8),
        (5000.0, 10000.0, +0.3),
        (10000.0, 20000.0, -0.5),
    ],
    # Folk / Singer-songwriter -------------------------------------------
    "folk": [
        (80.0, 250.0, +0.5),
        (250.0, 800.0, +1.2),
        (800.0, 2500.0, +1.5),
        (2500.0, 5000.0, +0.8),
        (5000.0, 10000.0, +0.2),
        (10000.0, 20000.0, -0.8),
    ],
    "singer_songwriter": [
        (100.0, 300.0, +0.5),
        (300.0, 1000.0, +1.0),
        (1000.0, 3000.0, +1.2),
        (3000.0, 6000.0, +0.5),
        (6000.0, 16000.0, 0.0),
    ],
    # Pop / Schlager / European vocal ------------------------------------
    "pop": [
        (50.0, 120.0, +1.0),
        (120.0, 300.0, +0.3),
        (300.0, 800.0, -0.2),
        (800.0, 2500.0, +0.5),
        (2500.0, 6000.0, +1.2),
        (6000.0, 12000.0, +0.8),
        (12000.0, 20000.0, +0.5),
    ],
    "schlager": [
        (80.0, 250.0, +0.5),
        (250.0, 800.0, +0.8),
        (800.0, 2500.0, +1.0),
        (2500.0, 5000.0, +0.5),
        (5000.0, 10000.0, +0.3),
        (10000.0, 20000.0, -0.5),
    ],
    "chanson": [
        (80.0, 300.0, +0.8),
        (300.0, 1000.0, +0.5),
        (1000.0, 3000.0, +1.5),
        (3000.0, 6000.0, +0.5),
        (6000.0, 12000.0, -0.3),
        (12000.0, 20000.0, -1.0),
    ],
    "tango": [
        (80.0, 300.0, +1.0),
        (300.0, 800.0, +0.8),
        (800.0, 2500.0, +1.2),
        (2500.0, 5000.0, +0.5),
        (5000.0, 12000.0, -0.2),
        (12000.0, 20000.0, -1.0),
    ],
    "fado": [
        (80.0, 300.0, +1.0),
        (300.0, 800.0, +0.8),
        (800.0, 2500.0, +1.5),
        (2500.0, 5000.0, +0.8),
        (5000.0, 12000.0, -0.3),
        (12000.0, 20000.0, -1.0),
    ],
    "bossa_nova": [
        (80.0, 250.0, +0.8),
        (250.0, 800.0, +0.5),
        (800.0, 2500.0, +1.0),
        (2500.0, 5000.0, +0.5),
        (5000.0, 10000.0, +0.2),
        (10000.0, 20000.0, -0.5),
    ],
    "samba": [
        (60.0, 200.0, +1.5),
        (200.0, 600.0, +1.0),
        (600.0, 2000.0, +1.2),
        (2000.0, 5000.0, +0.8),
        (5000.0, 12000.0, +0.5),
        (12000.0, 20000.0, -0.3),
    ],
    "latin": [
        (60.0, 200.0, +1.5),
        (200.0, 800.0, +0.8),
        (800.0, 3000.0, +1.0),
        (3000.0, 6000.0, +0.8),
        (6000.0, 12000.0, +0.5),
        (12000.0, 20000.0, -0.3),
    ],
    "flamenco": [
        (80.0, 250.0, +0.5),
        (250.0, 800.0, +1.2),
        (800.0, 2500.0, +1.5),
        (2500.0, 6000.0, +1.0),
        (6000.0, 12000.0, +0.5),
        (12000.0, 20000.0, -0.5),
    ],
    "klezmer": [
        (80.0, 300.0, +0.8),
        (300.0, 800.0, +0.5),
        (800.0, 2500.0, +1.5),
        (2500.0, 5000.0, +1.0),
        (5000.0, 10000.0, +0.3),
        (10000.0, 20000.0, -0.5),
    ],
    "walzer": [
        (80.0, 300.0, +0.8),
        (300.0, 800.0, +0.5),
        (800.0, 3000.0, +0.5),
        (3000.0, 6000.0, -0.2),
        (6000.0, 20000.0, -0.5),
    ],
    "marsch": [
        (60.0, 200.0, +1.5),
        (200.0, 600.0, +1.0),
        (600.0, 2000.0, +1.2),
        (2000.0, 5000.0, +0.8),
        (5000.0, 12000.0, +0.3),
        (12000.0, 20000.0, -0.5),
    ],
    # Rock / Metal -------------------------------------------------------
    "rock": [
        (60.0, 150.0, +1.2),
        (150.0, 500.0, +1.0),
        (500.0, 1200.0, +0.3),
        (1200.0, 3000.0, +0.5),
        (3000.0, 8000.0, +0.8),
        (8000.0, 16000.0, +0.3),
        (16000.0, 20000.0, -0.3),
    ],
    "hard_rock": [
        (50.0, 120.0, +1.5),
        (120.0, 400.0, +1.2),
        (400.0, 1000.0, +0.5),
        (1000.0, 3000.0, +1.0),
        (3000.0, 8000.0, +1.5),
        (8000.0, 14000.0, +0.5),
        (14000.0, 20000.0, -0.5),
    ],
    "heavy_metal": [
        (40.0, 100.0, +2.0),
        (100.0, 300.0, +1.5),
        (300.0, 800.0, -0.5),
        (800.0, 2000.0, +0.5),
        (2000.0, 5000.0, +2.0),
        (5000.0, 10000.0, +1.5),
        (10000.0, 20000.0, +0.5),
    ],
    "punk": [
        (60.0, 200.0, +1.5),
        (200.0, 600.0, +0.5),
        (600.0, 2000.0, +0.8),
        (2000.0, 6000.0, +1.5),
        (6000.0, 16000.0, +0.5),
        (16000.0, 20000.0, -0.5),
    ],
    # Electronic / Hip-Hop / Reggae -------------------------------------
    "electronic": [
        (20.0, 60.0, +2.5),
        (60.0, 150.0, +2.0),
        (150.0, 400.0, +0.5),
        (400.0, 1500.0, -0.2),
        (1500.0, 5000.0, +0.5),
        (5000.0, 12000.0, +0.8),
        (12000.0, 20000.0, +1.0),
    ],
    "hip-hop": [
        (20.0, 60.0, +3.0),
        (60.0, 150.0, +2.0),
        (150.0, 400.0, +0.5),
        (400.0, 1200.0, -0.3),
        (1200.0, 3500.0, +0.8),
        (3500.0, 8000.0, +0.8),
        (8000.0, 20000.0, +0.3),
    ],
    "trap": [
        (20.0, 50.0, +4.0),
        (50.0, 100.0, +2.5),
        (100.0, 300.0, +0.5),
        (300.0, 1500.0, -0.5),
        (1500.0, 4000.0, +1.0),
        (4000.0, 10000.0, +1.5),
        (10000.0, 20000.0, +0.5),
    ],
    "reggae": [
        (40.0, 120.0, +2.5),
        (120.0, 300.0, +1.5),
        (300.0, 800.0, +0.5),
        (800.0, 2500.0, +0.3),
        (2500.0, 5000.0, -0.3),
        (5000.0, 12000.0, -0.5),
        (12000.0, 20000.0, -1.5),
    ],
    "dub": [
        (30.0, 100.0, +3.0),
        (100.0, 250.0, +2.0),
        (250.0, 800.0, +0.5),
        (800.0, 3000.0, -0.5),
        (3000.0, 8000.0, +0.2),
        (8000.0, 20000.0, -1.5),
    ],
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TonalCurve:
    """24-Bark-band spectral knowledge for phase steering.

    Two complementary arrays:
      band_ceiling_db  — hard stop per band (§0h §2.46e §6.2c).
      band_target_db   — studio-day reconstruction target (§2.46).
                         Phases steer TOWARD this shape; never exceed ceiling.
    """

    band_ceiling_db: np.ndarray  # shape (24,)
    band_target_db: np.ndarray  # shape (24,) — recording chain reconstruction
    era_decade: int | None
    genre_label: str
    material_type: str
    confidence: float
    harmonic_profile: HarmonicProfile | None = None
    noise_texture: NoiseTextureProfile | None = None
    _era_bw_hz: float = field(default=20000.0, repr=False)
    _mat_bw_hz: float = field(default=22050.0, repr=False)

    def ceiling_for_hz(self, freq_hz: float) -> float:
        """Interpolierte Decke in dB für eine beliebige Frequenz."""
        freq = float(freq_hz)
        for i in range(24):
            if _BARK_EDGES_HZ[i] <= freq < _BARK_EDGES_HZ[i + 1]:
                return float(self.band_ceiling_db[i])
        return float(self.band_ceiling_db[-1 if freq >= _BARK_EDGES_HZ[-1] else 0])

    def target_for_hz(self, freq_hz: float) -> float:
        """Interpoliertes Studio-Tag-Rekonstruktionsziel in dB."""
        freq = float(freq_hz)
        for i in range(24):
            if _BARK_EDGES_HZ[i] <= freq < _BARK_EDGES_HZ[i + 1]:
                return float(self.band_target_db[i])
        return float(self.band_target_db[-1 if freq >= _BARK_EDGES_HZ[-1] else 0])

    def apply_ceiling(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Wendet an: per-Bark-band ceiling. Only attenuates, never boosts. Non-blocking."""
        try:
            return _apply_bark_ceiling(audio_pre, audio_post, sr, self.band_ceiling_db, self.confidence)
        except Exception as exc:
            logger.debug("TonalCurve.anwenden_ceiling nicht blockierend: %s", exc)
            return audio_post

    def apply_snr_adaptive_ceiling(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sr: int,
        *,
        n_noise_frames: int = 10,
    ) -> np.ndarray:
        """Signal-adaptive ceiling: per-Bark-band SNR gates the expansion headroom.

        Bands with high SNR (≥ 25 dB) receive full ceiling headroom — the signal
        is clean enough that expansion is safe.  Bands with low SNR (< 8 dB) receive
        zero headroom — any boost would amplify noise, not restore signal.

        Falls back to apply_ceiling on any exception (non-blocking).

        Args:
            audio_pre:      Pre-phase audio (reference for boost measurement AND SNR estimate).
            audio_post:     Post-phase audio (potentially over-boosted).
            sr:             Sample rate (must be 48 000 Hz).
            n_noise_frames: Quietest frame count for per-band noise estimate.

        Returns:
            audio_post with SNR-adaptive per-band ceiling applied.
        """
        try:
            band_snr = _estimate_bark_band_snr(audio_pre, sr, n_noise_frames)
            snr_ceilings = _snr_scaled_ceilings(self.band_ceiling_db, band_snr)
            result = _apply_bark_ceiling(audio_pre, audio_post, sr, snr_ceilings, self.confidence)
            logger.debug(
                "TonalCurve SNR-adaptive ceiling: min_snr=%.1f dB max_snr=%.1f dB",
                float(np.min(band_snr)),
                float(np.max(band_snr)),
            )
            return result
        except Exception as exc:
            logger.debug("TonalCurve.anwenden_snr_adaptive_ceiling Ersatzpfad: %s", exc)
            return self.apply_ceiling(audio_pre, audio_post, sr)

    def apply_target_steering(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sr: int,
        *,
        steering_strength: float = 0.40,
        min_gap_db: float = 0.5,
        max_boost_per_band_db: float = 3.0,
        n_noise_frames: int = 10,
    ) -> np.ndarray:
        """Active target steering: lift under-represented Bark bands toward band_target_db.

        This is the active half of the Dual-Array-Konzept (§2.46 §0a):
          - apply_snr_adaptive_ceiling()  → caps over-boost
          - apply_target_steering()        → lifts under-represented bands

        For each Bark band where the post-phase output is BELOW the recording-chain
        target (band_target_db), applies a gentle boost toward the target.

        Physics rationale: band_target_db encodes the weighted transfer function of
        mic (Eargle 2004), console EQ (Neve/EMI/SSL), and tape machine (Copeland 2008)
        for the detected era. A band that is below this target means the restoration
        phase did not fully recover the historically expected spectral content —
        e.g. a 1970s Neve 1073 recording where the +3 dB HF shelf is not yet present.

        Guards (§0 Primum non nocere — only boost, never harm):
          - Only boosts, never cuts (ceiling handles over-boost)
          - Never boosts above band_ceiling_db (§0h)
          - SNR gate: bands with SNR < 8 dB are not steered (would amplify noise)
          - Gap threshold: min_gap_db avoids sub-0.5 dB micro-adjustments
          - Max per-band boost: max_boost_per_band_db (default 3.0 dB)
          - Confidence blend: low confidence → weak steering
          - Non-blocking: any exception falls back to audio_post unchanged

        Args:
            audio_pre:              Pre-phase reference audio.
            audio_post:             Post-phase audio (to be steered).
            sr:                     Sample rate (48000 Hz).
            steering_strength:      [0.0, 1.0] — how strongly to steer.
            min_gap_db:             Minimum gap to trigger steering (dB).
            max_boost_per_band_db:  Per-band boost cap (dB).
            n_noise_frames:         Quietest frames for SNR estimate.

        Returns:
            audio_post with target-steered band energy, or audio_post unchanged.
        """
        if self.confidence < 0.50:
            return audio_post  # not enough confidence to steer

        try:
            rms_pre = _bark_band_rms(audio_pre, sr)  # shape (24,)
            rms_post = _bark_band_rms(audio_post, sr)  # shape (24,)
            band_snr = _estimate_bark_band_snr(audio_pre, sr, n_noise_frames)

            # Actual boost per band in dB (what phase achieved)
            with np.errstate(divide="ignore", invalid="ignore"):
                actual_boost_db = 20.0 * np.log10((rms_post + 1e-20) / (rms_pre + 1e-20))
            actual_boost_db = np.nan_to_num(actual_boost_db, nan=0.0, posinf=0.0, neginf=-60.0)

            # Gap: how far below target is the actual boost?
            # Positive gap → under-target band → candidate for steering
            gap_db = self.band_target_db.astype(np.float64) - actual_boost_db

            # Effective steering boost per band
            steer_db = np.zeros(24, dtype=np.float64)
            eff_strength = float(np.clip(steering_strength * self.confidence, 0.0, 1.0))

            for i in range(24):
                if gap_db[i] < min_gap_db:
                    continue  # at or above target
                if self.band_ceiling_db[i] <= -90.0:
                    continue  # material BW ceiling: forbidden zone
                if band_snr[i] < 8.0:
                    continue  # noisy band: steering would amplify noise

                # Target boost: partial fill of gap (steering_strength controls aggressiveness)
                raw_steer = float(gap_db[i]) * eff_strength
                # Cap to per-band maximum and to remaining headroom below ceiling
                headroom = float(self.band_ceiling_db[i]) - float(actual_boost_db[i])
                steer_db[i] = float(np.clip(raw_steer, 0.0, min(max_boost_per_band_db, max(0.0, headroom))))

            if float(np.max(steer_db)) < min_gap_db:
                return audio_post  # nothing meaningful to steer

            logger.debug(
                "TonalCurve target steering: %d band(s) steered, max=%.1f dB (conf=%.2f str=%.2f)",
                int(np.sum(steer_db > 0.1)),
                float(np.max(steer_db)),
                self.confidence,
                steering_strength,
            )

            # Build per-frequency-bin gain mask (linear)
            n_fft = 2048
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            gain_mask = np.ones(len(freqs), dtype=np.float64)
            for i in range(24):
                if steer_db[i] < min_gap_db:
                    continue
                f_lo = _BARK_EDGES_HZ[i]
                f_hi = _BARK_EDGES_HZ[i + 1]
                bin_mask = (freqs >= f_lo) & (freqs < f_hi)
                if bin_mask.any():
                    gain_mask[bin_mask] = float(10.0 ** (steer_db[i] / 20.0))

            # Apply via STFT/ISTFT on each channel
            hop = 512
            window = "hann"
            _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103

            def _apply_mono(ch: np.ndarray) -> np.ndarray:
                n_orig = len(ch)
                _, _, Z = _stft(ch.astype(np.float64), fs=sr, nperseg=n_fft, noverlap=_noverlap, window=window)
                Z_steered = Z * gain_mask[:, np.newaxis]
                _, out = _istft(Z_steered, fs=sr, nperseg=n_fft, noverlap=_noverlap, window=window)
                out = np.real(out)
                # §2.61 Output-Length-Guard
                if len(out) >= n_orig:
                    out = out[:n_orig]
                else:
                    out = np.pad(out, (0, n_orig - len(out)))
                return out.astype(np.float32)  # type: ignore[no-any-return,return-value]

            if audio_post.ndim == 1:
                result = _apply_mono(audio_post)
            elif audio_post.ndim == 2:
                is_ch_first = audio_post.shape[0] == 2 and audio_post.shape[1] > 2
                if is_ch_first:
                    # M/S-linked: apply identical gain to both channels (§2.51)
                    result = np.stack([_apply_mono(audio_post[c]) for c in range(2)], axis=0)
                else:
                    n_ch = audio_post.shape[1]
                    result = np.stack([_apply_mono(audio_post[:, c]) for c in range(n_ch)], axis=1)
            else:
                return audio_post

            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

        except Exception as exc:
            logger.debug("TonalCurve.anwenden_target_steering nicht blockierend: %s", exc)
            return audio_post


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bark_band_rms(audio: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet per-Bark-band RMS power via STFT.  Returns shape (24,)."""
    n_fft = 2048
    hop = 512
    # Mono mix
    if audio.ndim == 2:
        mono = audio.mean(axis=0) if (audio.shape[0] == 2 and audio.shape[1] > 2) else audio.mean(axis=1)
    else:
        mono = audio
    mono = mono.astype(np.float64)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Collect short-time power spectrum
    n_samples = len(mono)
    n_frames = max(1, (n_samples - n_fft) // hop)
    psd_sum = np.zeros(len(freqs), dtype=np.float64)
    for k in range(n_frames):
        start = k * hop
        frame = mono[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        window = np.hanning(n_fft)
        psd_sum += np.abs(np.fft.rfft(frame * window)) ** 2
    psd = psd_sum / max(1, n_frames)

    # Accumulate per Bark band
    rms = np.zeros(24, dtype=np.float64)
    for i in range(24):
        f_lo = _BARK_EDGES_HZ[i]
        f_hi = _BARK_EDGES_HZ[i + 1]
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if mask.any():
            rms[i] = float(np.sqrt(np.mean(psd[mask]) + 1e-20))
        else:
            rms[i] = 1e-10
    return rms  # type: ignore[no-any-return]


def _estimate_bark_band_snr(audio: np.ndarray, sr: int, n_noise_frames: int = 10) -> np.ndarray:
    """Per-Bark-band SNR estimation from signal audio alone.

    Approach (Ephraim & Malah 1984 spirit): the signal estimate is the median
    per-frame band energy; the noise estimate is the mean of the N quietest
    frames in that band.  Ratio → dB, clipped [0, 60].

    Args:
        audio:          Input audio (any layout; converted to mono internally).
        sr:             Sample rate in Hz.
        n_noise_frames: Number of quietest frames averaged as noise estimate.

    Returns:
        np.ndarray shape (24,): SNR in dB per Bark band.
    """
    if audio.ndim == 2:
        mono = audio.mean(axis=0) if (audio.shape[0] == 2 and audio.shape[1] > 2) else audio.mean(axis=1)
    else:
        mono = audio
    mono = mono.astype(np.float64)

    n_fft = 2048
    hop = 512
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    n_samples = len(mono)
    n_frames = max(2, (n_samples - n_fft) // hop)
    window = np.hanning(n_fft)

    # Collect per-frame, per-freq-bin power
    frames_power = np.zeros((len(freqs), n_frames), dtype=np.float64)
    for k in range(n_frames):
        start = k * hop
        frame = mono[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frames_power[:, k] = np.abs(np.fft.rfft(frame * window)) ** 2

    snr_db = np.zeros(24, dtype=np.float32)
    n_noise = max(1, min(n_noise_frames, n_frames))
    for i in range(24):
        mask = (freqs >= _BARK_EDGES_HZ[i]) & (freqs < _BARK_EDGES_HZ[i + 1])
        if not mask.any():
            snr_db[i] = 20.0
            continue
        band_power = frames_power[mask, :].mean(axis=0)  # (n_frames,)
        sig = float(np.median(band_power) + 1e-20)
        noise = float(np.mean(np.sort(band_power)[:n_noise]) + 1e-20)
        snr_db[i] = float(np.clip(10.0 * np.log10(sig / noise), 0.0, 60.0))

    return snr_db  # type: ignore[no-any-return]


def _snr_scaled_ceilings(band_ceiling_db: np.ndarray, band_snr_db: np.ndarray) -> np.ndarray:
    """Skaliert per-Bark-band ceiling headroom by measured SNR.

    Rationale: a band with high SNR (clean signal) can be expanded safely up to
    its full ceiling. A noisy band (low SNR) should not be expanded — boosting
    would amplify noise, not recover signal (§0h Primum non nocere).

    Scaling (linear ramp):
      SNR ≥ 25 dB  →  scale = 1.0   (full headroom)
      SNR  8–25 dB →  scale = (snr − 8) / 17   (partial headroom)
      SNR <  8 dB  →  scale = 0.0   (no expansion — ceiling clamped to 0 dB)

    Only positive ceiling values (headroom bands) are scaled.
    Negative ceiling values (absolute rolloff bands §6.2c) remain unchanged.

    Returns:
        np.ndarray shape (24,): SNR-scaled ceiling values.
    """
    snr_clip = np.clip(band_snr_db, 0.0, 25.0)
    snr_scale = np.clip((snr_clip - 8.0) / 17.0, 0.0, 1.0)  # 0.0 … 1.0
    scaled = np.where(
        band_ceiling_db > 0.0,
        band_ceiling_db * snr_scale,
        band_ceiling_db,  # keep rolloff bands (< 0) unchanged
    )
    return scaled.astype(np.float32)  # type: ignore[no-any-return]


def _apply_bark_ceiling(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    sr: int,
    band_ceiling_db: np.ndarray,
    confidence: float,
) -> np.ndarray:
    """STFT-based per-Bark-band gain ceiling enforcement."""
    rms_pre = _bark_band_rms(audio_pre, sr)
    rms_post = _bark_band_rms(audio_post, sr)

    # Actual boost per band in dB
    boost_db = 20.0 * np.log10((rms_post + 1e-20) / (rms_pre + 1e-20))

    # Excess beyond ceiling (only bands where actual boost > ceiling)
    excess_db = np.maximum(0.0, boost_db - band_ceiling_db)

    if float(np.max(excess_db)) < 0.15:
        # No band exceeds its ceiling → passthrough
        return audio_post

    logger.debug(
        "TonalCurve ceiling: %d band(s) over limit, max excess=%.1f dB",
        int(np.sum(excess_db > 0.15)),
        float(np.max(excess_db)),
    )

    # Build per-frequency-bin gain mask
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    gain_mask = np.ones(len(freqs), dtype=np.float32)
    for i in range(24):
        if excess_db[i] < 0.15:
            continue
        f_lo = _BARK_EDGES_HZ[i]
        f_hi = _BARK_EDGES_HZ[i + 1]
        bin_mask = (freqs >= f_lo) & (freqs < f_hi)
        if bin_mask.any():
            gain_linear = float(10.0 ** (-excess_db[i] / 20.0))
            gain_mask[bin_mask] = gain_linear

    # Blend: low confidence → conservative (weaker correction)
    if confidence < 0.70:
        blend = float(np.clip(confidence / 0.70, 0.0, 1.0))
        gain_mask = 1.0 + blend * (gain_mask - 1.0)

    # Apply via STFT/ISTFT on each channel
    hop = 512
    window = "hann"
    _noverlap = min(n_fft - hop, max(0, n_fft - 1))  # §v10.103

    def _apply_mono(ch: np.ndarray) -> np.ndarray:
        n_orig = len(ch)
        _, _, Z = _stft(ch.astype(np.float64), fs=sr, nperseg=n_fft, noverlap=_noverlap, window=window)
        Z_masked = Z * gain_mask[:, np.newaxis]
        _, out = _istft(Z_masked, fs=sr, nperseg=n_fft, noverlap=_noverlap, window=window)
        out = np.real(out)
        # Trim/pad to original length (§2.61 Output-Length-Guard)
        if len(out) >= n_orig:
            out = out[:n_orig]
        else:
            out = np.pad(out, (0, n_orig - len(out)))
        return out.astype(np.float32)  # type: ignore[no-any-return,return-value]

    if audio_post.ndim == 1:
        result = _apply_mono(audio_post)
    elif audio_post.ndim == 2:
        is_ch_first = audio_post.shape[0] == 2 and audio_post.shape[1] > 2
        if is_ch_first:
            result = np.stack([_apply_mono(audio_post[c]) for c in range(2)], axis=0)
        else:
            n_ch = audio_post.shape[1]
            result = np.stack([_apply_mono(audio_post[:, c]) for c in range(n_ch)], axis=1)
    else:
        return audio_post

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _era_bw_hz_for_decade(decade: int | None) -> float:
    """Nearest-decade lookup in _ERA_BW_HZ."""
    if decade is None:
        return 20000.0
    decades = sorted(_ERA_BW_HZ.keys())
    return float(_ERA_BW_HZ[min(decades, key=lambda d: abs(d - int(decade)))])


def _genre_delta_for_band(genre_label: str, band_idx: int) -> float:
    """Sum genre spectral deltas for a given Bark band index."""
    _ALIASES = {
        "german pop": "schlager",
        "volkstümlich": "schlager",
        "r&b": "soul/r&b",
        "soul": "soul/r&b",
        "rnb": "soul/r&b",
    }
    g = _ALIASES.get(str(genre_label or "").strip().lower(), str(genre_label or "").strip().lower())
    center = _BARK_CENTERS_HZ[band_idx]
    return sum(d for f_lo, f_hi, d in _GENRE_DELTAS.get(g, []) if f_lo <= center < f_hi)


def _nearest_era_key(decade: int | None, table: dict) -> int:
    """Nearest-decade lookup in any era-keyed table."""
    if decade is None:
        return 1980
    return int(min(table.keys(), key=lambda d: abs(d - int(decade))))


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_tonal_reference_curve(
    *,
    era_decade: int | None = None,
    genre_label: str = "",
    material_type: str = "unknown",
    restorability: float = 50.0,
    is_studio_2026: bool = False,
    chain_weight_mic: float = 0.50,
    chain_weight_console: float = 0.25,
    chain_weight_tape: float = 0.25,
    _mic_bp_override: list[tuple[float, float]] | None = None,
    _con_bp_override: list[tuple[float, float]] | None = None,
    _tap_bp_override: list[tuple[float, float]] | None = None,
) -> TonalCurve:
    """Berechnet 24-Bark-band spectral ceilings and targets for a song context.

    The ceiling prevents over-processing (§0h §2.46e).
    The target guides restoration toward the original studio-day spectral
    profile: Microphone FR + Console EQ + Tape response (§2.46).

    Args:
        era_decade:          Recording decade (e.g. 1960, 1970, …).
        genre_label:         Genre string (any case; normalised internally).
        material_type:       Canonical material key (vinyl, shellac, tape, …).
        restorability:       Estimated restorability score [0..100].
        is_studio_2026:      If True, slightly more headroom.
        chain_weight_mic:    Relative weight of microphone response.
        chain_weight_console: Relative weight of console/preamp EQ.
        chain_weight_tape:   Relative weight of tape machine response.

    Returns:
        TonalCurve with ceilings, targets, harmonic profile and noise texture.
    """
    mat_key = str(material_type or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    era_bw_hz = _era_bw_hz_for_decade(era_decade)
    mat_bw_hz = float(_MATERIAL_BW_CEILING_HZ.get(mat_key, 22050.0))
    effective_bw_hz = min(era_bw_hz, mat_bw_hz)
    max_boost = _MAX_BOOST_DB if not is_studio_2026 else _MAX_BOOST_DB + 1.0

    # Confidence
    rest = float(np.clip(restorability, 0.0, 100.0))
    era_conf = 0.82 if era_decade is not None else 0.55
    mat_conf = 0.85 if mat_key in _MATERIAL_BW_CEILING_HZ else 0.60
    genre_conf = 0.80 if str(genre_label or "").strip().lower() in _GENRE_DELTAS else 0.55
    confidence = float(np.clip(0.40 * era_conf + 0.35 * mat_conf + 0.25 * genre_conf, 0.30, 0.92))
    confidence = float(np.clip(confidence + 0.05 * (rest / 100.0 - 0.5), 0.30, 0.92))

    # Recording chain tables for the detected era (provenance override wins if given)
    era_key = _nearest_era_key(era_decade, _ERA_MIC_RESPONSE)
    mic_bp = _mic_bp_override if _mic_bp_override is not None else _ERA_MIC_RESPONSE[era_key]
    console_bp = _con_bp_override if _con_bp_override is not None else _ERA_CONSOLE_EQ[era_key]
    tape_bp = _tap_bp_override if _tap_bp_override is not None else _ERA_TAPE_RESPONSE[era_key]
    harmonic_prof = _ERA_HARMONIC_PROFILE.get(era_key)
    noise_tex = _MATERIAL_NOISE_TEXTURE.get(mat_key)

    # Normalise chain weights
    _wsum = chain_weight_mic + chain_weight_console + chain_weight_tape + 1e-12
    wm = chain_weight_mic / _wsum
    wc = chain_weight_console / _wsum
    wt = chain_weight_tape / _wsum

    ceilings = np.zeros(24, dtype=np.float32)
    targets = np.zeros(24, dtype=np.float32)

    for i, center in enumerate(_BARK_CENTERS_HZ):
        # Hard material BW ceiling (§6.2c §0a)
        if center > mat_bw_hz:
            ceilings[i] = -99.0
            targets[i] = -99.0
            continue

        # --- Ceiling (era BW rolloff) ---
        if center <= 0.5 * effective_bw_hz:
            ceil_db = max_boost
        elif center <= effective_bw_hz:
            t = (center - 0.5 * effective_bw_hz) / (0.5 * effective_bw_hz + 1e-12)
            ceil_db = max_boost * (1.0 - float(t))
        else:
            octaves = float(np.log2(max(center, effective_bw_hz + 1.0) / max(effective_bw_hz, 1.0)))
            ceil_db = -_ROLLOFF_DB_PER_OCT * octaves

        ceil_db += _genre_delta_for_band(genre_label, i)
        ceilings[i] = float(np.clip(ceil_db, -30.0, max_boost + 2.0))

        # --- Target: recording chain transfer function (Eargle 2004, Copeland 2008) ---
        chain_db = (
            wm * _interp_db_curve(center, mic_bp)
            + wc * _interp_db_curve(center, console_bp)
            + wt * _interp_db_curve(center, tape_bp)
        )
        # Genre shapes the target as well (at 60 % weight vs ceiling)
        chain_db += 0.6 * _genre_delta_for_band(genre_label, i)
        # Target must never exceed ceiling
        targets[i] = float(np.clip(chain_db, -30.0, float(ceilings[i])))

    return TonalCurve(
        band_ceiling_db=ceilings,
        band_target_db=targets,
        era_decade=era_decade,
        genre_label=str(genre_label or "").strip().lower(),
        material_type=mat_key,
        confidence=confidence,
        harmonic_profile=harmonic_prof,
        noise_texture=noise_tex,
        _era_bw_hz=era_bw_hz,
        _mat_bw_hz=mat_bw_hz,
    )


# ---------------------------------------------------------------------------
# Singleton profiler
# ---------------------------------------------------------------------------


class TonalReferenceProfiler:
    """Thin singleton wrapper: caches last curve per (era, genre, material) triple."""

    def __init__(self) -> None:
        self._cache: dict[tuple, TonalCurve] = {}
        self._lock = threading.Lock()

    def get_curve(
        self,
        *,
        era_decade: int | None = None,
        genre_label: str = "",
        material_type: str = "unknown",
        restorability: float = 50.0,
        is_studio_2026: bool = False,
    ) -> TonalCurve:
        """Gibt cached TonalCurve for the given era/genre/material/mode combination zurück."""
        key = (era_decade, str(genre_label).strip().lower(), str(material_type).strip().lower(), is_studio_2026)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = compute_tonal_reference_curve(
                    era_decade=era_decade,
                    genre_label=genre_label,
                    material_type=material_type,
                    restorability=restorability,
                    is_studio_2026=is_studio_2026,
                )
            return self._cache[key]

    def get_studio_console_curve(
        self,
        console_type: str = "neve_1073",
    ) -> list[tuple[float, float]]:
        """Gibt EQ breakpoints for a classic Studio-2026 console fingerprint zurück.

        Returns frequency-gain pairs (Hz, dB) from :data:`_STUDIO_CONSOLE_CURVES`.
        Designed for use in Studio 2026 mode as a subtle coloration pass
        (§Gap5 Console Character).

        Parameters
        ----------
        console_type : str
            Console profile name: ``"neve_1073"``, ``"ssl_4000"``, ``"api_2500"``,
            ``"neutral"``. Falls back to ``"neve_1073"`` if unknown.
        """
        return list(_STUDIO_CONSOLE_CURVES.get(str(console_type).lower(), _STUDIO_CONSOLE_CURVES["neve_1073"]))

    def get_curve_with_provenance(
        self,
        *,
        era_decade: int | None = None,
        genre_label: str = "",
        material_type: str = "unknown",
        restorability: float = 50.0,
        is_studio_2026: bool = False,
        provenance_hint: str = "",
    ) -> TonalCurve:
        """Like :meth:`get_curve` but with optional provenance overlay.

        When ``provenance_hint`` is non-empty, queries
        :func:`~backend.core.broadcast_archive_db.get_provenance_adjustment`
        and merges the studio-specific EQ deltas into the chain weights before
        computing the TonalCurve.  Falls back to plain :meth:`get_curve` if the
        provenance database cannot be imported or no match is found.

        Not cached separately per provenance (provenance is expected to be
        stable for a given import session).
        """
        if not provenance_hint:
            return self.get_curve(
                era_decade=era_decade,
                genre_label=genre_label,
                material_type=material_type,
                restorability=restorability,
                is_studio_2026=is_studio_2026,
            )
        try:
            from backend.core.broadcast_archive_db import (  # pylint: disable=import-outside-toplevel
                apply_provenance_to_chain,
                get_provenance_adjustment,
            )

            adj = get_provenance_adjustment(provenance_hint, era_decade)
            if adj is None:
                return self.get_curve(
                    era_decade=era_decade,
                    genre_label=genre_label,
                    material_type=material_type,
                    restorability=restorability,
                    is_studio_2026=is_studio_2026,
                )
            # Build base chain breakpoints from era tables, then overlay provenance
            ed = era_decade
            mic_bp = list(_ERA_MIC_RESPONSE.get(_nearest_era_key(ed, _ERA_MIC_RESPONSE), [(20, 0), (20000, 0)]))
            con_bp = list(_ERA_CONSOLE_EQ.get(_nearest_era_key(ed, _ERA_CONSOLE_EQ), [(20, 0), (20000, 0)]))
            tap_bp = list(_ERA_TAPE_RESPONSE.get(_nearest_era_key(ed, _ERA_TAPE_RESPONSE), [(20, 0), (20000, 0)]))
            mic_bp, con_bp, tap_bp = apply_provenance_to_chain(mic_bp, con_bp, tap_bp, adj, era_decade)

            # Rebuild TonalCurve with merged breakpoints injected via
            # compute_tonal_reference_curve using equal weights
            # (the merge has already blended the three sources).
            merged_curve = compute_tonal_reference_curve(
                era_decade=era_decade,
                genre_label=genre_label,
                material_type=material_type,
                restorability=restorability,
                is_studio_2026=is_studio_2026,
                _mic_bp_override=mic_bp,
                _con_bp_override=con_bp,
                _tap_bp_override=tap_bp,
            )
            logger.debug(
                "TonalReferenceProfiler: provenance=%r merged → %s",
                provenance_hint,
                adj.provenance_label,
            )
            return merged_curve
        except Exception as exc:
            logger.debug("get_curve_with_provenance Ersatzpfad: %s", exc)
            return self.get_curve(
                era_decade=era_decade,
                genre_label=genre_label,
                material_type=material_type,
                restorability=restorability,
                is_studio_2026=is_studio_2026,
            )


_profiler_instance: TonalReferenceProfiler | None = None
_profiler_lock = threading.Lock()


# ===========================================================================
# §Gap5 Studio-Console-Character-Profile (Studio 2026 only)
# Neve 1073 / SSL 4000 / API 2500 subtle EQ fingerprints for studio-mode
# coloration. Values are frequency (Hz) → gain (dB), applied as a steering
# suggestion (never exceed hallucination-guard or BW ceiling).
# References: Neve 1073 datasheet; SSL 4000G service manual; Maselec MEA-2.
# ===========================================================================
_STUDIO_CONSOLE_CURVES: dict[str, list[tuple[float, float]]] = {
    # Neve 1073: 1073 mic-pre / EQ — iconic warm transformer core.
    # Characteristics: low-end weight (+2 dB shelf@80Hz), upper-mid presence
    # bump (+1 dB@3 kHz), gentle HF air roll (~-0.5 dB@18 kHz).
    "neve_1073": [
        (20.0, 0.5),
        (80.0, 2.0),  # Low shelf
        (200.0, 0.5),
        (1000.0, 0.0),
        (3000.0, 1.0),  # Presence peak
        (6000.0, 0.5),
        (10000.0, 0.0),
        (18000.0, -0.5),
        (20000.0, -0.8),
    ],
    # SSL 4000G: solid-state console — tighter, more transient, extended air.
    # Characteristics: tight bass (+1 dB@100Hz), presence (+1.5 dB@5 kHz),
    # extended HF air (+1 dB@16 kHz).
    "ssl_4000": [
        (20.0, 0.0),
        (100.0, 1.0),  # Low-end tightness
        (300.0, 0.0),
        (1000.0, 0.0),
        (5000.0, 1.5),  # Presence/clarity
        (10000.0, 0.8),
        (16000.0, 1.0),  # Air
        (20000.0, 0.5),
    ],
    # API 2500: VCA bus compressor character — punchy midrange.
    # Characteristics: midrange focus (+1.2 dB@1 kHz), slight low cut at 60Hz.
    "api_2500": [
        (60.0, -0.5),
        (200.0, 0.3),
        (1000.0, 1.2),  # Punch focus
        (3000.0, 0.5),
        (8000.0, 0.0),
        (16000.0, -0.2),
        (20000.0, -0.3),
    ],
    # Flat (passthrough) — no console coloration
    "neutral": [
        (20.0, 0.0),
        (20000.0, 0.0),
    ],
}


def get_tonal_reference_profiler() -> TonalReferenceProfiler:
    """Thread-safe singleton access."""
    global _profiler_instance  # pylint: disable=global-statement
    if _profiler_instance is None:
        with _profiler_lock:
            if _profiler_instance is None:
                _profiler_instance = TonalReferenceProfiler()
    return _profiler_instance


def get_era_harmonic_profile(era_decade: int | None) -> HarmonicProfile:
    """Gibt the nearest HarmonicProfile for the given decade (rounds down) zurück.

    Looks up ``_ERA_HARMONIC_PROFILE`` for the largest available key that is
    <= *era_decade*.  Falls back to the 1970 Transistor-Era entry when
    *era_decade* is ``None`` or below the earliest available decade.

    Args:
        era_decade: Four-digit decade integer (e.g. 1940, 1960, 2000) or None.

    Returns:
        Matching :class:`HarmonicProfile` instance.
    """
    if era_decade is None:
        return _ERA_HARMONIC_PROFILE[1970]
    d = int(era_decade)
    available = sorted(_ERA_HARMONIC_PROFILE.keys())
    key = max((k for k in available if k <= d), default=available[0])
    return _ERA_HARMONIC_PROFILE[key]


__all__ = [
    "TonalCurve",
    "TonalReferenceProfiler",
    "HarmonicProfile",
    "NoiseTextureProfile",
    "compute_tonal_reference_curve",
    "get_tonal_reference_profiler",
    "get_era_harmonic_profile",
]
