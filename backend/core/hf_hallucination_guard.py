"""
§0 / §2.46 System-Wide HF Hallucination Prevention Guard.

Prevents additive/reconstruction phases from synthesising spectral content above
the credible bandwidth ceiling of the source material.

Rationale (§0 Primum non nocere):
    Additive phases such as frequency restoration (phase_06), harmonic reconstruction
    (phase_07), air-band enhancement (phase_39) and spectral inpainting phases can
    introduce HF energy that was never present in the source material.  For archival
    materials the credible maximum bandwidth is a hard physical constraint of the
    recording medium, not an aesthetic preference.  Synthesising content above that
    ceiling is hallucination — a §0 violation regardless of perceptual goal scores.

    The per-phase BW cap that exists in phase_55 (diffusion inpainting) is here
    extended to all ADDITIVE and RECONSTRUCTION phases system-wide.

Literature anchors:
    Casey, "Sound Directions" (2007): wax cylinder usable BW ≤ 4.5 kHz.
    IASA TC-04 (2009): archival transfer bandwidth norms by medium.
    Fastl & Zwicker, "Psychoacoustics" (2007): §8 spectral brightness.
    Morton, "Off the Record" (2006): wire recorder HF characteristics.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Material → maximum credible restored bandwidth (Hz).
# All caps are conservative upper safety bounds anchored in archival literature.
# Digital materials receive the full Nyquist budget; no cap is applied.
# ---------------------------------------------------------------------------
_MATERIAL_HF_CAP_HZ: dict[str, float] = {
    # Pre-electrical / early electrical mechanical recording
    "wax_cylinder": 5_000.0,  # Acoustic horn 1900-1930; Casey 2007 / IASA TC-04
    "wire_recording": 6_000.0,  # Steel-wire Magnetophon variants 1940-1955; Morton 2006
    # Groove-cut analogue media
    "shellac": 7_000.0,  # Shellac / 78 rpm 1898-1960; §0 Vintage Aesthetics ≤7 kHz
    "lacquer_disc": 8_500.0,  # Acetate lacquer instantaneous discs 1930-1950
    "vinyl": 22_000.0,  # High-quality vinyl; no meaningful cap
    # Magnetic tape
    "reel_tape": 18_000.0,  # Biased tape at pro speeds; conservative to prevent ultra-HF
    "cassette": 14_000.0,  # Compact cassette; head geometry + speed limit
    "minidisc": 16_000.0,  # ATRAC-1/3 codec HF rolloff
    "dat": 22_000.0,  # Digital Archive Tape — essentially uncapped
    # Digital / lossy
    "cd_digital": 22_000.0,  # Nyquist-limited at 44.1 kHz; no hallucination risk
    "mp3_low": 16_000.0,  # ≤128 kbps: psychoacoustic model cuts HF aggressively
    "mp3_high": 20_000.0,  # 256–320 kbps: near-transparent
    "aac_low": 16_000.0,
    "aac_high": 20_000.0,
}

# Safe fallback for materials not in the table above
_DEFAULT_HF_CAP_HZ: float = 20_000.0

# ----------------------------------------------------------------------
# Additive/reconstruction phase prefixes that could synthesise HF content.
# phase_37 (bass) excluded: operates in low-frequency domain only.
# phase_55 already has its own internal BW cap; dual-check is safe.
# ----------------------------------------------------------------------
ADDITIVE_PHASE_PREFIXES: frozenset[str] = frozenset(
    {
        "phase_06",  # frequency / bandwidth restoration
        "phase_07",  # harmonic reconstruction
        "phase_21",  # harmonic exciter
        "phase_22",  # tape saturation (adds harmonics)
        "phase_23",  # spectral repair (can synthesise HF)
        "phase_37",  # bass enhancement (adds subharmonics, shifts spectral tilt)
        "phase_38",  # presence boost (2–8 kHz range)
        "phase_39",  # air band / HF enhancement
        "phase_44",  # guitar enhancement (additive, instrument-specific)
        "phase_50",  # spectral inpainting follow-up
        "phase_51",  # drums enhancement (additive, instrument-specific)
        "phase_55",  # diffusion inpainting (has own guard — second check is benign)
        "phase_56",  # spectral band-gap repair
    }
)

# HF energy-ratio increase (fraction of total energy) that triggers intervention.
_HF_ENERGY_DELTA_THRESHOLD: float = 0.035  # 3.5 % of broadband energy

# Minimum wet ratio kept even during rescue (preserves partial phase benefit)
_MIN_WET_RATIO: float = 0.35


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_hf_energy_ratio(audio: np.ndarray, sr: int, cutoff_hz: float) -> float:
    """Fast estimate of fraction of signal energy above *cutoff_hz*.

    Uses a single 1024-sample STFT frame from the middle of the signal.
    Returns 0.0 on any error (non-blocking, advisory-only).
    """
    try:
        mono: np.ndarray = (
            audio[0]
            if (audio.ndim == 2 and audio.shape[0] == 2)
            else (np.mean(audio, axis=0) if audio.ndim == 2 else audio)
        )
        n_fft = 1024
        center = max(0, len(mono) // 2 - n_fft // 2)
        frame = mono[center : center + n_fft]
        if len(frame) < 64:
            return 0.0
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        window = np.hanning(len(frame)).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
        total = float(np.sum(spectrum)) + 1e-12
        hf_energy = float(np.sum(spectrum[freqs >= cutoff_hz]))
        return float(np.clip(hf_energy / total, 0.0, 1.0))
    except Exception as e:
        logger.warning("hf_hallucination_guard.py::_estimate_hf_energy_Verhaeltnis Ersatzpfad: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_material_hf_cap(material: str) -> float:
    """Gibt the material BW cap in Hz (lookup with fallback) zurück."""
    return _MATERIAL_HF_CAP_HZ.get(str(material).lower(), _DEFAULT_HF_CAP_HZ)


def check_hf_hallucination(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int,
    material: str,
    recovery_certainty_scalar: float = 1.0,
) -> tuple[bool, float, float, float]:
    """Test whether an additive phase hallucinated HF content above the material BW cap.

    Args:
        audio_before:             Audio entering the phase (float32, mono or stereo).
        audio_after:              Audio leaving the phase.
        sr:                       Sample rate (48 000 Hz expected).
        material:                 Material key, e.g. ``"shellac"``, ``"vinyl"``.
        recovery_certainty_scalar: From ``_compute_recovery_certainty_profile()``
                                  [0.78, 1.0].  Lower value → tighter threshold.

    Returns:
        Tuple ``(ok, wet_cap, cap_hz, delta_hf_ratio)``:

        * ``ok``            – ``True`` when no problematic HF synthesis was detected.
        * ``wet_cap``       – Maximum wet ratio to apply (1.0 when *ok* is ``True``).
        * ``cap_hz``        – Material BW cap that was applied.
        * ``delta_hf_ratio``– Increase in HF energy fraction above cap.
    """
    cap_hz = get_material_hf_cap(material)

    # Digital materials (≥20 kHz cap) are effectively uncapped — skip measurement.
    nyquist = sr / 2.0
    if cap_hz >= nyquist - 500.0:
        return True, 1.0, cap_hz, 0.0

    hf_before = _estimate_hf_energy_ratio(audio_before, sr, cap_hz)
    hf_after = _estimate_hf_energy_ratio(audio_after, sr, cap_hz)
    delta = hf_after - hf_before

    # Threshold scales with certainty: lower certainty → stricter enforcement.
    cert = float(np.clip(recovery_certainty_scalar, 0.78, 1.0))
    threshold = _HF_ENERGY_DELTA_THRESHOLD * cert  # certainty 0.78 → 78 % of nominal

    if delta <= threshold:
        return True, 1.0, cap_hz, delta

    # Hallucination detected: compute safe wet ratio.
    # Higher delta → lower wet; clamped to [_MIN_WET_RATIO, 0.90].
    excess = delta - threshold
    scale = float(np.clip(excess / max(threshold, 1e-6), 0.0, 2.0))
    wet_cap = float(np.clip(1.0 - scale * 0.30, _MIN_WET_RATIO, 0.90))
    return False, wet_cap, cap_hz, delta
