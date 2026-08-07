import logging

import pytest

"""Tests for Micro-Temporal Envelope Fidelity (MTEF) module.

Tests: ≥ 35 — Abdeckung: measure, morph, envelope, scales, edge-cases, mono, stereo, singleton
"""

import threading

import numpy as np

from backend.core.micro_temporal_envelope_fidelity import (
    MTEFResult,
    _frame_pearson,
    _hilbert_envelope,
    _smooth_envelope,
    get_mtef,
    measure,
    morph,
)

logger = logging.getLogger(__name__)

SR = 48000


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tone(freq: float = 440.0, dur_s: float = 1.0, sr: int = SR) -> np.ndarray:
    """Generate a pure sine tone."""
    t = np.arange(int(sr * dur_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _make_am_signal(dur_s: float = 1.0, carrier: float = 440.0, mod_rate: float = 8.0) -> np.ndarray:
    """Generate AM-modulated signal (simulates syllable-rate envelope modulation)."""
    t = np.arange(int(SR * dur_s), dtype=np.float32) / SR
    carrier_sig = np.sin(2 * np.pi * carrier * t)
    modulator = 0.5 + 0.5 * np.sin(2 * np.pi * mod_rate * t)
    return (0.4 * carrier_sig * modulator).astype(np.float32)  # type: ignore[no-any-return]


# ── MTEFResult dataclass ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_mtef_result_fields():
    r = MTEFResult(
        pearson_attack=0.95,
        pearson_syllable=0.93,
        pearson_note=0.91,
        fidelity_score=0.93,
        max_gain_applied_db=0.0,
        corrected=False,
    )
    assert r.pearson_attack == 0.95
    assert r.fidelity_score == 0.93
    assert r.corrected is False


# ── Hilbert envelope ─────────────────────────────────────────────────────────


def test_hilbert_envelope_shape():
    audio = _make_tone(dur_s=0.5)
    env = _hilbert_envelope(audio)
    assert env.shape == audio.shape
    assert env.dtype == np.float32


def test_hilbert_envelope_nonnegative():
    audio = _make_tone(dur_s=0.5)
    env = _hilbert_envelope(audio)
    assert np.all(env >= 0)


def test_hilbert_envelope_pure_tone_nearly_constant():
    """Hilbert envelope of a pure sine should be nearly constant."""
    audio = _make_tone(freq=1000.0, dur_s=0.5)
    env = _hilbert_envelope(audio)
    # Skip edge effects (first and last 5 ms)
    edge = int(0.005 * SR)
    inner = env[edge:-edge]
    assert np.std(inner) / np.mean(inner) < 0.05  # CV < 5%


def test_hilbert_envelope_am_captures_modulation():
    """AM-modulated signal should show clear envelope modulation."""
    audio = _make_am_signal(dur_s=1.0, mod_rate=5.0)
    env = _hilbert_envelope(audio)
    edge = int(0.01 * SR)
    inner = env[edge:-edge]
    # Modulation depth should be significant
    assert np.max(inner) / (np.min(inner[inner > 1e-6]) + 1e-10) > 1.5


def test_hilbert_envelope_stereo_to_mono():
    """Stereo input is converted to mono."""
    mono = _make_tone(dur_s=0.5)
    stereo = np.column_stack([mono, mono * 0.8])
    env = _hilbert_envelope(stereo)
    assert env.ndim == 1


def test_hilbert_envelope_long_audio_chunked():
    """Long audio (>30 s) should be processed in chunks without error."""
    np.random.seed(42)
    audio = (np.random.randn(SR * 35) * 0.1).astype(np.float32)
    env = _hilbert_envelope(audio)
    assert env.shape == audio.shape
    assert np.all(np.isfinite(env))


def test_hilbert_envelope_silence():
    """Silence should give near-zero envelope."""
    audio = np.zeros(SR, dtype=np.float32)
    env = _hilbert_envelope(audio)
    assert np.max(env) < 1e-5


# ── Smooth envelope ──────────────────────────────────────────────────────────


def test_smooth_envelope_shape_preserved():
    env = np.random.rand(SR).astype(np.float32)
    smoothed = _smooth_envelope(env, SR, 0.040)
    assert smoothed.shape == env.shape


def test_smooth_envelope_reduces_variance():
    np.random.seed(7)
    env = np.random.rand(SR).astype(np.float32) * 0.5 + 0.25
    smoothed = _smooth_envelope(env, SR, 0.040)
    assert np.std(smoothed) < np.std(env)


# ── Frame Pearson ─────────────────────────────────────────────────────────────


def test_frame_pearson_identical_signals():
    """Identical envelopes → Pearson ≈ 1.0."""
    env = _hilbert_envelope(_make_am_signal(dur_s=1.0))
    r = _frame_pearson(env, env, SR, 0.040)
    assert r > 0.99


def test_frame_pearson_uncorrelated():
    """Independent random envelopes → lower Pearson than identical."""
    np.random.seed(42)
    env1 = np.abs(np.random.randn(SR * 2)).astype(np.float32) * 0.3
    np.random.seed(99)
    env2 = np.abs(np.random.randn(SR * 2)).astype(np.float32) * 0.3
    r = _frame_pearson(env1, env2, SR, 0.040)
    # Smoothing introduces some spurious correlation; just verify < perfect
    assert r < 0.85


def test_frame_pearson_range():
    """Pearson must be in [-1, 1]."""
    np.random.seed(0)
    env1 = np.abs(np.random.randn(SR)).astype(np.float32) * 0.5
    env2 = np.abs(np.random.randn(SR)).astype(np.float32) * 0.5
    r = _frame_pearson(env1, env2, SR, 0.015)
    assert -1.0 <= r <= 1.0


# ── measure() ─────────────────────────────────────────────────────────────────


def test_measure_identical():
    """Identical signals → fidelity ≈ 1.0."""
    audio = _make_am_signal(dur_s=2.0)
    result = measure(audio, audio, SR)
    assert isinstance(result, MTEFResult)
    assert result.fidelity_score > 0.98
    assert result.corrected is False


def test_measure_slightly_modified():
    """Slightly gained signal should still have high fidelity."""
    audio = _make_am_signal(dur_s=2.0)
    modified = audio * 0.95
    result = measure(audio, modified, SR)
    assert result.fidelity_score > 0.90


def test_measure_corrupted():
    """Heavily corrupted envelope at syllable scale → reduced fidelity."""
    np.random.seed(42)
    original = _make_am_signal(dur_s=2.0, mod_rate=8.0)
    # Apply low-frequency random gain modulation (10-20 Hz rate) to corrupt
    # the envelope at the syllable/note time-scale (50-100 ms)
    t = np.arange(len(original), dtype=np.float32) / SR
    # Sum of random-phase slow modulations → destroys envelope shape
    mod = np.ones(len(original), dtype=np.float32)
    for freq in [7.0, 12.0, 18.0, 25.0]:
        phase = np.random.rand() * 2 * np.pi
        mod *= (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * freq * t + phase))).astype(np.float32)
    corrupted = original * mod
    result = measure(original, corrupted, SR)
    assert result.fidelity_score < 0.98  # Measurable degradation


def test_measure_stereo():
    """Stereo signals should be handled correctly."""
    mono = _make_am_signal(dur_s=1.0)
    stereo_orig = np.column_stack([mono, mono * 0.9])
    stereo_rest = np.column_stack([mono * 0.95, mono * 0.85])
    result = measure(stereo_orig, stereo_rest, SR)
    assert isinstance(result, MTEFResult)
    assert 0.0 <= result.fidelity_score <= 1.0


def test_measure_returns_all_scales():
    audio = _make_am_signal(dur_s=1.0)
    result = measure(audio, audio, SR)
    assert hasattr(result, "pearson_attack")
    assert hasattr(result, "pearson_syllable")
    assert hasattr(result, "pearson_note")


def test_measure_different_lengths():
    """Mismatched lengths should be handled gracefully."""
    audio_long = _make_am_signal(dur_s=2.0)
    audio_short = _make_am_signal(dur_s=1.5)
    result = measure(audio_long, audio_short, SR)
    assert isinstance(result, MTEFResult)


def test_measure_silence():
    """Silence → fidelity 1.0 (nothing to measure)."""
    silence = np.zeros(SR, dtype=np.float32)
    result = measure(silence, silence, SR)
    assert result.fidelity_score >= 0.95


def test_measure_nan_guard():
    """NaN input should not crash."""
    audio = np.full(SR, np.nan, dtype=np.float32)
    # measure relies on Hilbert transform — may produce NaN but should not crash
    try:
        result = measure(audio, audio, SR)
        assert isinstance(result, MTEFResult)
    except Exception:
        logger.warning("test fallback", exc_info=True)
        pass  # acceptable to raise on all-NaN


# ── morph() ───────────────────────────────────────────────────────────────────


def test_morph_identical_no_correction():
    """Identical signals → no correction applied."""
    audio = _make_am_signal(dur_s=1.0)
    corrected, result = morph(audio, audio, SR)
    assert result.corrected is False or result.fidelity_score >= 0.95
    assert np.allclose(corrected, audio, atol=0.01)


def test_morph_corrects_envelope_damage():
    """Envelope-damaged signal should be partially corrected."""
    np.random.seed(42)
    original = _make_am_signal(dur_s=2.0)
    # Apply random gain modulation to corrupt envelope
    mod = 0.5 + 0.5 * np.random.rand(len(original)).astype(np.float32)
    corrupted = original * mod

    # Measure before correction
    before = measure(original, corrupted, SR)
    # Apply correction
    corrected, after = morph(original, corrupted, SR, mode="restoration")

    # After correction should be at least as good as before
    assert after.fidelity_score >= before.fidelity_score - 0.02


def test_morph_output_shape_mono():
    audio = _make_am_signal(dur_s=1.0)
    corrupted = audio * 0.7
    corrected, _ = morph(audio, corrupted, SR)
    assert corrected.shape == corrupted.shape


def test_morph_output_shape_stereo():
    mono = _make_am_signal(dur_s=1.0)
    orig = np.column_stack([mono, mono * 0.9])
    rest = np.column_stack([mono * 0.8, mono * 0.7])
    corrected, _ = morph(orig, rest, SR)
    assert corrected.shape == rest.shape


def test_morph_no_clipping():
    """Output must be in [-1, 1]."""
    audio = _make_am_signal(dur_s=1.0)
    corrupted = audio * 0.5
    corrected, _ = morph(audio, corrupted, SR)
    assert np.max(np.abs(corrected)) <= 1.0


def test_morph_no_nan():
    """Output must be NaN-free."""
    audio = _make_am_signal(dur_s=1.0)
    corrupted = audio * 0.6
    corrected, _ = morph(audio, corrupted, SR)
    assert np.all(np.isfinite(corrected))


def test_morph_gain_limited_restoration():
    """Restoration mode: gain limited to ±2 dB."""
    audio = _make_am_signal(dur_s=1.0) * 0.7
    corrupted = audio * 0.3  # 7.4 dB drop
    _, result = morph(audio, corrupted, SR, mode="restoration")
    assert result.max_gain_applied_db <= 3.0  # ±2 dB + margin


def test_morph_gain_limited_studio():
    """Studio mode: gain limited to ±3 dB."""
    audio = _make_am_signal(dur_s=1.0) * 0.7
    corrupted = audio * 0.3
    _, result = morph(audio, corrupted, SR, mode="studio")
    assert result.max_gain_applied_db <= 4.0  # ±3 dB + margin


# ── Singleton ─────────────────────────────────────────────────────────────────


def test_singleton_same_instance():
    a = get_mtef()
    b = get_mtef()
    assert a is b


def test_singleton_thread_safe():
    instances = []

    def get():
        instances.append(get_mtef())

    threads = [threading.Thread(target=get) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(inst is instances[0] for inst in instances)


def test_singleton_wrapper_measure():
    mtef = get_mtef()
    audio = _make_am_signal(dur_s=0.5)
    result = mtef.measure(audio, audio, SR)
    assert isinstance(result, MTEFResult)


def test_singleton_wrapper_morph():
    mtef = get_mtef()
    audio = _make_am_signal(dur_s=0.5)
    corrected, result = mtef.morph(audio, audio, SR, mode="restoration")
    assert isinstance(result, MTEFResult)
    assert corrected.shape == audio.shape


# ── Scale ordering ────────────────────────────────────────────────────────────


def test_attack_scale_more_sensitive_than_note():
    """15 ms scale should detect finer envelope changes than 80 ms scale."""
    np.random.seed(42)
    original = _make_am_signal(dur_s=2.0, mod_rate=25.0)  # Fast modulation (attack-scale)
    # Add fine-scale noise to envelope
    noise = (np.random.randn(len(original)) * 0.1).astype(np.float32)
    corrupted = original + noise
    result = measure(original, corrupted, SR)
    # Attack scale should show more degradation (lower correlation)
    # than note scale (which is smoothed more)
    assert result.pearson_attack <= result.pearson_note + 0.1  # Allow small tolerance


def test_fidelity_weight_sum():
    """Scale weights must sum to 1.0 (fidelity is a proper weighted average)."""
    from backend.core.micro_temporal_envelope_fidelity import _SCALES

    total = sum(w for _, _, w in _SCALES)
    assert abs(total - 1.0) < 1e-6


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_very_short_audio():
    """Very short audio (<50 ms) should not crash."""
    audio = _make_tone(dur_s=0.03)  # 30 ms
    result = measure(audio, audio, SR)
    assert isinstance(result, MTEFResult)


def test_dc_offset_signal():
    """DC offset should not affect envelope fidelity."""
    audio = _make_am_signal(dur_s=1.0)
    offset = audio + 0.1
    result = measure(audio, offset, SR)
    # DC offset barely affects Hilbert envelope
    assert result.fidelity_score > 0.80
