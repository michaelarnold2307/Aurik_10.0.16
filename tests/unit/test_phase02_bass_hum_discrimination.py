"""§v10.998: Bass vs. Hum — Detektor-Diskriminierung auf SOTA-Niveau.

Die Kassetten-Diagnose bewies: Phase 02 verwechselte einen 50/60-Hz-Bass
mit Netzbrummen und zerstörte das Signal (−62 dB). Diese Tests pinnen den
physikalischen Kern der Unterscheidung:
  - Hum ist STATIONÄR (konstante Hüllkurve) → wird entfernt
  - Bass ist DYNAMISCH (modulierte Hüllkurve) → bleibt unberührt
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from backend.core.phases.phase_02_hum_removal import HumRemovalPhase
from backend.core.phases.phase_57_print_through_reduction import _lms_bilateral_subtraction

SR = 48000


def _make_phase() -> HumRemovalPhase:
    phase = HumRemovalPhase()
    phase.sample_rate = SR
    return phase


def _tone(freq: float, seconds: float = 3.0, mod_hz: float = 0.0, mod_depth: float = 0.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    if mod_hz > 0:
        # Sinusförmige AM (keine Gleichrichtung — die würde Oberschwingungen
        # erzeugen und den Test verfälschen)
        tone *= 1.0 - mod_depth * np.sin(2 * np.pi * mod_hz * t)
    return tone.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 02: Der Diskriminator
# ═══════════════════════════════════════════════════════════════════════════════


def test_stationary_hum_is_detected():
    """Echtes Netzbrummen (stationär) MUSS weiterhin erkannt werden."""
    phase = _make_phase()
    params = dict(phase.MATERIAL_PARAMS["unknown"])
    audio = _tone(50.0, seconds=4.0) * 0.3
    audio += np.random.default_rng(1).standard_normal(len(audio)).astype(np.float32) * 0.01
    detected = phase._detect_multi_fundamental(audio, params)
    assert 50 in detected, f"Stationäres Brummen wurde nicht erkannt: {detected}"


def test_am_modulated_bass_is_not_detected():
    """Dynamischer 50-Hz-Bass (Hip-Hop-Kick/Bass) darf NICHT als Hum gelten."""
    phase = _make_phase()
    params = dict(phase.MATERIAL_PARAMS["unknown"])
    audio = _tone(50.0, seconds=4.0, mod_hz=3.0, mod_depth=0.8) * 0.3
    detected = phase._detect_multi_fundamental(audio, params)
    assert 50 not in detected, f"Musikalischer Bass wurde als Hum fehlklassifiziert: {detected}"


def test_60hz_stationary_hum_is_detected():
    """60-Hz-Netze (US) bleiben abgedeckt."""
    phase = _make_phase()
    params = dict(phase.MATERIAL_PARAMS["unknown"])
    audio = _tone(60.0, seconds=4.0) * 0.3
    audio += np.random.default_rng(2).standard_normal(len(audio)).astype(np.float32) * 0.01
    detected = phase._detect_multi_fundamental(audio, params)
    assert 60 in detected


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 57: Block-LMS — korrekt UND schnell
# ═══════════════════════════════════════════════════════════════════════════════


def test_block_lms_removes_echo():
    """Synthetisches Echo wird entfernt (Alpha konvergiert Richtung alpha_max)."""
    rng = np.random.default_rng(7)
    clean = rng.standard_normal(96000) * 0.2
    delay = 200
    alpha_true = 0.5
    x = clean.copy()
    x[delay:] += alpha_true * clean[:-delay]

    out = _lms_bilateral_subtraction(
        x=x,
        delay_pre=0,
        delay_post=delay,
        alpha_pre_max=0.1,
        alpha_post_max=0.9,
    )
    # Echo-Energie im Ausgang deutlich reduziert (Korrelation mit verschobenem Signal sinkt)
    residual_echo = np.corrcoef(out[delay:], clean[:-delay])[0, 1]
    input_echo = np.corrcoef(x[delay:], clean[:-delay])[0, 1]
    assert abs(residual_echo) < abs(input_echo) * 0.7, (
        f"Echo nicht reduziert: corr in={input_echo:.3f} out={residual_echo:.3f}"
    )


def test_block_lms_is_fast():
    """720k Samples (15 s) in deutlich unter 2 s — vorher 5.5 s im Sample-Loop."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(720_000) * 0.1
    t0 = time.time()
    _lms_bilateral_subtraction(x=x, delay_pre=100, delay_post=200, alpha_pre_max=0.05, alpha_post_max=0.1)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"Block-LMS zu langsam: {elapsed:.2f}s"
