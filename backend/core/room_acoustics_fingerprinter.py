"""backend/core/room_acoustics_fingerprinter.py — Room Acoustics Fingerprinting

Estimates RT60, DRR, and early reflection time from audio signal (blind estimation),
maps to a dereverb protection cap, and optionally refines via the venue_acoustic_db
when a venue hint is available.

Used by UV3 to inject ``room_acoustics_fingerprint`` into ``_restoration_context``,
which phase_20 and phase_49 use to protect authentic room character (§2.46f §0h).

Principle (§0 Primum non nocere):
    If the room sound was captured on the original recording, it is NOT a defect —
    it is the authentic artistic environment. The fingerprinter sets a conservative
    strength cap so that dereverb phases cannot over-process the authentic space.

Scientific references:
    Schroeder (1965) JASA 37:409 — backward integration RT60.
    Vesa & Harma (2005) ICASSP — blind DRR estimation.
    ISO 3382-1:2009 — measurement of room acoustic parameters.
    Beranek (2016) JASA 139:1548 — EDT relationship.
"""

from __future__ import annotations

import logging

import numpy as np

from backend.core.venue_acoustic_db import estimate_room_type, get_dereverb_wet_cap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Room-type-based protection caps (fallback when no venue profile available)
# Lower = more protection of authentic room character.
# studio:       modest room — dereverb is helpful
# broadcast:    controlled room — moderate protection
# concert_hall: hall IS the sound — strong protection
# church:       reverb is the identity — maximum protection
# outdoor:      no enclosed reverb — minimal protection
# ---------------------------------------------------------------------------
_ROOM_TYPE_CAPS: dict[str, float] = {
    "studio": 0.60,
    "broadcast": 0.50,
    "concert_hall": 0.30,
    "church": 0.20,
    "outdoor": 0.80,
}

# Effective RT60 above which the room character is dominant (protect more)
_RT60_HIGH_THRESHOLD_S = 1.2  # concert-hall territory
_RT60_LOW_THRESHOLD_S = 0.30  # dry studio — dereverb safe


def _estimate_rt60_schroeder(audio: np.ndarray, sample_rate: int) -> float:
    """Schroeder (1965) backward integration — blind T60 estimation.

    Returns T60 in seconds, clamped to [0.05, 3.0]. Fallback: 0.4 s.
    """
    x = np.asarray(audio, dtype=np.float64).ravel()
    x = x - float(np.mean(x))  # DC removal
    energy = x**2
    edc = np.cumsum(energy[::-1])[::-1]
    peak = float(edc.max())
    if peak < 1e-14:
        return 0.4
    edc_db = 10.0 * np.log10(edc / peak + 1e-14)
    below5 = np.where(edc_db <= -5.0)[0]
    below35 = np.where(edc_db <= -35.0)[0]
    if len(below5) == 0 or len(below35) == 0:
        return 0.4
    idx5, idx35 = int(below5[0]), int(below35[0])
    if idx35 <= idx5:
        return 0.4
    t30 = (idx35 - idx5) / float(sample_rate)
    return float(np.clip(2.0 * t30, 0.05, 3.0))


def _estimate_drr(audio: np.ndarray, sample_rate: int) -> float:
    """Blind DRR proxy: first 5 ms RMS vs full signal RMS (Vesa & Harma 2005).

    Returns DRR in dB, clamped to [-20, 30].
    """
    x = np.asarray(audio, dtype=np.float64).ravel()
    direct_window = max(1, int(0.005 * sample_rate))
    rms_direct = float(np.sqrt(np.mean(x[:direct_window] ** 2)) + 1e-14)
    rms_total = float(np.sqrt(np.mean(x**2)) + 1e-14)
    drr = 20.0 * np.log10(rms_direct / rms_total)
    return float(np.clip(drr, -20.0, 30.0))


def _early_reflection_estimate_ms(rt60_s: float) -> float:
    """Heuristic early reflection time from RT60 (Beranek 2004).

    Typical: ER = RT60 / 10 for concert halls; shorter for studios.
    Returns value in ms, clamped to [5, 80].
    """
    return float(np.clip(rt60_s * 100.0, 5.0, 80.0))


def compute_room_acoustics_fingerprint(
    audio: np.ndarray,
    sr: int,
    era_decade: int | None = None,
    venue_hint: str | None = None,
) -> dict[str, object]:
    """Berechnet a room acoustics fingerprint from audio signal.

    Args:
        audio:       Input audio (any shape — mono channel used for estimation).
        sr:          Sample rate (Hz).
        era_decade:  Recording decade (for venue_acoustic_db era relevance).
        venue_hint:  Optional venue/studio name (if known — improves precision).

    Returns:
        Dict with keys:
            rt60_s               (float) — estimated RT60 at mid frequency [s]
            drr_db               (float) — estimated direct-to-reverberant ratio [dB]
            room_type            (str)   — "studio"|"broadcast"|"concert_hall"|"church"|"outdoor"
            dereverb_strength_cap (float) — max safe dereverb strength [0..1]
            early_reflection_ms  (float) — estimated first reflection arrival [ms]
            protection_note      (str)   — human-readable reason for cap
    """
    result: dict[str, object] = {
        "rt60_s": 0.4,
        "drr_db": 6.0,
        "room_type": "studio",
        "dereverb_strength_cap": 0.60,
        "early_reflection_ms": 15.0,
        "protection_note": "default",
    }
    try:
        # Use mono channel for estimation (§2.51: stereo-safe — analysis only)
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono[0]
        # Limit to first 30 s for speed (Schroeder integration is O(N))
        max_samples = min(len(mono), int(30.0 * sr))
        mono = mono[:max_samples]

        rt60 = _estimate_rt60_schroeder(mono, sr)
        drr = _estimate_drr(mono, sr)
        result["rt60_s"] = float(rt60)
        result["drr_db"] = float(drr)

        room_type = estimate_room_type(rt60, drr, era_decade=era_decade, material_hint=venue_hint)
        result["room_type"] = room_type
        result["early_reflection_ms"] = _early_reflection_estimate_ms(rt60)

        # Prefer venue-specific cap if hint available (more precise)
        if venue_hint:
            cap = get_dereverb_wet_cap(venue_hint, era_decade, default=-1.0)
            if cap >= 0.0:
                result["dereverb_strength_cap"] = float(cap)
                result["protection_note"] = f"venue:{venue_hint} rt60={rt60:.2f}s"
                logger.info(
                    "§2.46f RoomAcoustics: venue=%s rt60=%.2fs drr=%.1fdB type=%s cap=%.2f",
                    venue_hint,
                    rt60,
                    drr,
                    room_type,
                    cap,
                )
                return result

        # Blind estimation: room_type-based cap
        cap = _ROOM_TYPE_CAPS.get(room_type, 0.60)

        # Additional RT60-based refinement: very long RT60 → tighten cap
        if rt60 >= _RT60_HIGH_THRESHOLD_S:
            cap = min(cap, 0.25)
            result["protection_note"] = f"rt60_high:{rt60:.2f}s type:{room_type}"
        elif rt60 <= _RT60_LOW_THRESHOLD_S:
            cap = min(cap, 0.70)  # dry studio — dereverb can be stronger
            result["protection_note"] = f"rt60_low:{rt60:.2f}s type:{room_type}"
        else:
            result["protection_note"] = f"rt60:{rt60:.2f}s type:{room_type}"

        result["dereverb_strength_cap"] = float(np.clip(cap, 0.10, 1.0))

        logger.info(
            "§2.46f RoomAcoustics fingerprint: rt60=%.2fs drr=%.1fdB room=%s cap=%.2f era=%s",
            rt60,
            drr,
            room_type,
            result["dereverb_strength_cap"],
            era_decade,
        )

    except Exception as _exc:
        logger.debug("RoomAcousticsFingerprinter nicht blockierend: %s", _exc)

    return result
