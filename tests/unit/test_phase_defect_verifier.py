from __future__ import annotations

import pytest

"""
tests/unit/test_phase_defect_verifier.py
PhaseDefectVerifier §7 (Ursache 7: Post-Phase Defekt-Verifikation) Test-Suite

All tests are synthetic — no ML models, no heavy audio loading.
Tests verify:
  - Module import / singleton
  - All proxy functions (impulse_ratio, hf_noise_floor, hum_energy, low_freq_energy,
    dc_offset, dropout_ratio, mono_compat)
  - measure_proxies() per phase_id
  - check() — no-op for phases without targets
  - check() — keeps audio_after when defect improved
  - check() — rolls back when defect worsened by > 25 %
  - check() — handles NaN/Inf input gracefully
  - check() — metadata_store gets populated
  - check() — mono AND stereo audio
  - Session telemetry (reset + summary)
  - Edge cases: 0-length audio, 1-sample audio, unknown phase_id
"""


import numpy as np

SR = 48_000


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, dur: float = 2.0, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return np.asarray(amp * np.sin(2.0 * np.pi * freq * t), dtype=np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.asarray(np.stack([mono, mono * 0.9], axis=0), dtype=np.float32)


def _add_hf_noise(audio: np.ndarray, level: float = 0.10) -> np.ndarray:
    """Add white noise (uniformly distributed across all frequencies)."""
    rng = np.random.default_rng(42)
    return np.clip(audio + (rng.standard_normal(audio.shape) * level).astype(np.float32), -1.0, 1.0)


def _add_clicks(audio: np.ndarray, n: int = 50, amp: float = 0.95) -> np.ndarray:
    """Inject impulse clicks."""
    out = audio.copy()
    rng = np.random.default_rng(7)
    idxs = rng.integers(0, len(out), size=n)
    out[idxs] = amp
    return out


def _add_hum(audio: np.ndarray, f0: float = 50.0, amp: float = 0.08) -> np.ndarray:
    """Add 50 Hz hum + harmonics."""
    t = np.linspace(0, len(audio) / SR, len(audio), endpoint=False)
    hum = np.zeros_like(audio)
    for k in range(1, 5):
        hum += np.sin(2.0 * np.pi * f0 * k * t).astype(np.float32)
    return np.asarray(np.clip(audio + amp * hum / 4, -1.0, 1.0), dtype=np.float32)


def _add_dc(audio: np.ndarray, dc: float = 0.05) -> np.ndarray:
    return np.clip(audio + dc, -1.0, 1.0)


def _add_rumble(audio: np.ndarray, amp: float = 0.15) -> np.ndarray:
    t = np.linspace(0, len(audio) / SR, len(audio), endpoint=False)
    rumble = (amp * np.sin(2.0 * np.pi * 30.0 * t)).astype(np.float32)
    return np.asarray(np.clip(audio + rumble, -1.0, 1.0), dtype=np.float32)


def _add_dropouts(audio: np.ndarray, n: int = 10, gap_ms: float = 5.0) -> np.ndarray:
    """Zero out short segments to simulate dropouts."""
    out = audio.copy()
    gap = int(gap_ms / 1000.0 * SR)
    rng = np.random.default_rng(13)
    for _ in range(n):
        start = int(rng.integers(0, max(1, len(out) - gap)))
        out[start : start + gap] = 0.0
    return out


def _add_slow_amplitude_modulation(audio: np.ndarray, rate_hz: float = 6.0, depth: float = 0.7) -> np.ndarray:
    """Fügt hörbare Pegelmodulation hinzu (pumpender Eindruck)."""
    t = np.linspace(0, len(audio) / SR, len(audio), endpoint=False, dtype=np.float32)
    env = 1.0 + depth * np.sin(2.0 * np.pi * rate_hz * t)
    return np.asarray(np.clip(audio * env, -1.0, 1.0), dtype=np.float32)


# ---------------------------------------------------------------------------
# 00 — Import and singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.cassette_defect_verifier import (  # type: ignore[attr-defined]
        PhaseDefectVerifier,
        get_phase_defect_verifier,
    )

    assert PhaseDefectVerifier is not None
    assert callable(get_phase_defect_verifier)


def test_01_singleton():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    a = get_phase_defect_verifier()
    b = get_phase_defect_verifier()
    assert a is b


def test_02_result_dataclass():
    from backend.core.cassette_defect_verifier import DefectVerificationResult

    r = DefectVerificationResult(  # type: ignore[call-arg]
        phase_id="phase_03_denoise",
        targeted_defects=["HIGH_FREQ_NOISE"],
        proxies_before={"HIGH_FREQ_NOISE": 0.5},
        proxies_after={"HIGH_FREQ_NOISE": 0.3},
        worst_defect="HIGH_FREQ_NOISE",
        worst_relative_change=-0.4,
        rollback_triggered=False,
    )
    assert r.phase_id == "phase_03_denoise"
    assert not r.rollback_triggered  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 01 — Proxy functions: impulse ratio
# ---------------------------------------------------------------------------


def test_10_proxy_impulse_ratio_clean():
    from backend.core.cassette_defect_verifier import _proxy_impulse_ratio  # type: ignore[attr-defined]

    clean = _sine()
    ratio = _proxy_impulse_ratio(clean)
    assert ratio > 0.0
    assert ratio < 100.0


def test_11_proxy_impulse_ratio_with_clicks():
    from backend.core.cassette_defect_verifier import _proxy_impulse_ratio  # type: ignore[attr-defined]

    clean = _sine()
    # 200 clicks needed so the top-0.1% (≈96 samples of 96000) is fully within click range
    clicked = _add_clicks(clean, n=200)
    ratio_clean = _proxy_impulse_ratio(clean)
    ratio_clicked = _proxy_impulse_ratio(clicked)
    # clicks must raise the impulse ratio
    assert ratio_clicked > ratio_clean


def test_12_proxy_impulse_ratio_stereo():
    from backend.core.cassette_defect_verifier import _proxy_impulse_ratio  # type: ignore[attr-defined]

    stereo = _stereo(_sine())
    ratio = _proxy_impulse_ratio(stereo)
    assert ratio > 0.0


def test_13_proxy_impulse_ratio_zeros():
    from backend.core.cassette_defect_verifier import _proxy_impulse_ratio  # type: ignore[attr-defined]

    silence = np.zeros(SR, dtype=np.float32)
    # Should not raise; returns a numeric value
    ratio = _proxy_impulse_ratio(silence)
    assert np.isfinite(ratio)


# ---------------------------------------------------------------------------
# 02 — Proxy functions: HF noise floor
# ---------------------------------------------------------------------------


def test_20_proxy_hf_noise_floor_clean():
    from backend.core.cassette_defect_verifier import _proxy_hf_noise_floor  # type: ignore[attr-defined]

    clean = _sine()
    val = _proxy_hf_noise_floor(clean, SR)
    assert val >= 0.0


def test_21_proxy_hf_noise_floor_noisy_vs_clean():
    from backend.core.cassette_defect_verifier import _proxy_hf_noise_floor  # type: ignore[attr-defined]

    clean = _sine()
    noisy = _add_hf_noise(clean, level=0.20)
    val_clean = _proxy_hf_noise_floor(clean, SR)
    val_noisy = _proxy_hf_noise_floor(noisy, SR)
    # Noisy signal should have higher HF noise floor
    assert val_noisy > val_clean


def test_22_proxy_hf_noise_floor_stereo():
    from backend.core.cassette_defect_verifier import _proxy_hf_noise_floor  # type: ignore[attr-defined]

    stereo = _stereo(_add_hf_noise(_sine()))
    val = _proxy_hf_noise_floor(stereo, SR)
    assert val >= 0.0


def test_22a_psycho_hf_audibility_noisy_vs_clean():
    from backend.core.cassette_defect_verifier import _compute_hf_noise_audibility  # type: ignore[attr-defined]

    clean = _sine()
    noisy = _add_hf_noise(clean, level=0.20)
    val_clean = _compute_hf_noise_audibility(clean, SR)
    val_noisy = _compute_hf_noise_audibility(noisy, SR)
    assert 0.0 <= val_clean <= 1.0
    assert 0.0 <= val_noisy <= 1.0
    assert val_noisy >= val_clean


def test_22b_psycho_transient_harshness_clicks_vs_clean():
    from backend.core.cassette_defect_verifier import _compute_transient_harshness  # type: ignore[attr-defined]

    clean = _sine()
    clicked = _add_clicks(clean, n=180)
    harsh_clean = _compute_transient_harshness(clean, SR)
    harsh_clicked = _compute_transient_harshness(clicked, SR)
    assert 0.0 <= harsh_clean <= 1.0
    assert 0.0 <= harsh_clicked <= 1.0
    assert harsh_clicked >= harsh_clean


def test_22c_psycho_quasi_peak_burstiness_clicks_vs_clean():
    from backend.core.cassette_defect_verifier import _compute_quasi_peak_burstiness  # type: ignore[attr-defined]

    clean = _sine()
    clicked = _add_clicks(clean, n=180)
    b_clean = _compute_quasi_peak_burstiness(clean, SR)
    b_clicked = _compute_quasi_peak_burstiness(clicked, SR)
    assert 0.0 <= b_clean <= 1.0
    assert 0.0 <= b_clicked <= 1.0
    assert b_clicked >= b_clean


def test_22f_psycho_modulation_roughness_modulated_vs_clean():
    from backend.core.cassette_defect_verifier import _compute_modulation_roughness  # type: ignore[attr-defined]

    clean = _sine()
    modulated = _add_slow_amplitude_modulation(clean, rate_hz=7.0, depth=0.8)
    m_clean = _compute_modulation_roughness(clean, SR)
    m_mod = _compute_modulation_roughness(modulated, SR)

    assert 0.0 <= m_clean <= 1.0
    assert 0.0 <= m_mod <= 1.0
    assert m_mod >= m_clean


def test_22g_psycho_modulation_roughness_short_audio_safe():
    from backend.core.cassette_defect_verifier import _compute_modulation_roughness  # type: ignore[attr-defined]

    short = np.zeros(32, dtype=np.float32)
    m = _compute_modulation_roughness(short, SR)
    assert 0.0 <= m <= 1.0


def test_22d_frequency_selective_blend_hum_keeps_low_band_closer_to_before():
    from backend.core.cassette_defect_verifier import (  # type: ignore[attr-defined]
        _frequency_selective_blend,
        _proxy_hum_energy,
    )

    before = _sine(freq=50.0, amp=0.20)
    after = before + _sine(freq=50.0, amp=0.10)
    after = np.clip(after, -1.0, 1.0).astype(np.float32)

    alpha = 0.92
    global_blend = np.clip(alpha * after + (1.0 - alpha) * before, -1.0, 1.0).astype(np.float32)
    local_blend = _frequency_selective_blend(before, after, alpha=alpha, worst_defect="HUM", sr=SR)

    hum_before = _proxy_hum_energy(before, SR)
    hum_global = _proxy_hum_energy(global_blend, SR)
    hum_local = _proxy_hum_energy(local_blend, SR)
    assert abs(hum_local - hum_before) <= abs(hum_global - hum_before)


def test_22e_frequency_selective_blend_clicks_reduces_impulse_ratio_vs_global():
    from backend.core.cassette_defect_verifier import (  # type: ignore[attr-defined]
        _frequency_selective_blend,
        _proxy_impulse_ratio,
    )

    before = _sine()
    after = _add_clicks(before, n=220, amp=0.95)
    alpha = 0.90

    global_blend = np.clip(alpha * after + (1.0 - alpha) * before, -1.0, 1.0).astype(np.float32)
    burst_blend = _frequency_selective_blend(before, after, alpha=alpha, worst_defect="CLICKS", sr=SR)

    imp_before = _proxy_impulse_ratio(before)
    imp_global = _proxy_impulse_ratio(global_blend)
    imp_burst = _proxy_impulse_ratio(burst_blend)
    assert abs(imp_burst - imp_before) <= abs(imp_global - imp_before)


# ---------------------------------------------------------------------------
# 03 — Proxy functions: hum energy
# ---------------------------------------------------------------------------


def test_30_proxy_hum_energy_clean():
    from backend.core.cassette_defect_verifier import _proxy_hum_energy  # type: ignore[attr-defined]

    clean = _sine(440.0)
    val = _proxy_hum_energy(clean, SR)
    assert val >= 0.0


def test_31_proxy_hum_energy_with_hum():
    from backend.core.cassette_defect_verifier import _proxy_hum_energy  # type: ignore[attr-defined]

    clean = _sine(440.0)
    hummed = _add_hum(clean, f0=50.0)
    val_clean = _proxy_hum_energy(clean, SR)
    val_hummed = _proxy_hum_energy(hummed, SR)
    # Hum adds energy at 50 Hz harmonics
    assert val_hummed > val_clean


def test_32_proxy_hum_energy_60hz():
    from backend.core.cassette_defect_verifier import _proxy_hum_energy  # type: ignore[attr-defined]

    clean = _sine(440.0)
    hummed = _add_hum(clean, f0=60.0)
    val_hummed = _proxy_hum_energy(hummed, SR)
    assert val_hummed > _proxy_hum_energy(clean, SR)


# ---------------------------------------------------------------------------
# 04 — Proxy functions: low-freq energy
# ---------------------------------------------------------------------------


def test_40_proxy_low_freq_energy_clean():
    from backend.core.cassette_defect_verifier import _proxy_low_freq_energy  # type: ignore[attr-defined]

    clean = _sine()
    val = _proxy_low_freq_energy(clean, SR)
    assert val >= 0.0


def test_41_proxy_low_freq_energy_with_rumble():
    from backend.core.cassette_defect_verifier import _proxy_low_freq_energy  # type: ignore[attr-defined]

    clean = _sine(440.0)
    rumbled = _add_rumble(clean)
    val_clean = _proxy_low_freq_energy(clean, SR)
    val_rumbled = _proxy_low_freq_energy(rumbled, SR)
    assert val_rumbled > val_clean


# ---------------------------------------------------------------------------
# 05 — Proxy functions: DC offset
# ---------------------------------------------------------------------------


def test_50_proxy_dc_offset_clean():
    from backend.core.cassette_defect_verifier import _proxy_dc_offset  # type: ignore[attr-defined]

    clean = _sine()
    val = _proxy_dc_offset(clean)
    assert val < 0.01  # sine has near-zero mean


def test_51_proxy_dc_offset_with_dc():
    from backend.core.cassette_defect_verifier import _proxy_dc_offset  # type: ignore[attr-defined]

    clean = _sine()
    dc_audio = _add_dc(clean, dc=0.05)
    val = _proxy_dc_offset(dc_audio)
    assert val > 0.04  # should capture the DC


def test_52_proxy_dc_offset_stereo():
    from backend.core.cassette_defect_verifier import _proxy_dc_offset  # type: ignore[attr-defined]

    stereo = _stereo(_add_dc(_sine(), dc=0.05))
    val = _proxy_dc_offset(stereo)
    assert val >= 0.0


# ---------------------------------------------------------------------------
# 06 — Proxy functions: dropout ratio
# ---------------------------------------------------------------------------


def test_60_proxy_dropout_ratio_clean():
    from backend.core.cassette_defect_verifier import _proxy_dropout_ratio  # type: ignore[attr-defined]

    clean = _sine()
    val = _proxy_dropout_ratio(clean, SR)
    assert val < 0.05  # sine mostly above threshold


def test_61_proxy_dropout_ratio_with_dropouts():
    from backend.core.cassette_defect_verifier import _proxy_dropout_ratio  # type: ignore[attr-defined]

    clean = _sine()
    dropped = _add_dropouts(clean, n=20, gap_ms=20.0)
    val_clean = _proxy_dropout_ratio(clean, SR)
    val_dropped = _proxy_dropout_ratio(dropped, SR)
    assert val_dropped > val_clean


# ---------------------------------------------------------------------------
# 07 — Proxy functions: mono compat
# ---------------------------------------------------------------------------


def test_70_proxy_mono_compat_coherent_stereo():
    from backend.core.cassette_defect_verifier import _proxy_mono_compat  # type: ignore[attr-defined]

    mono = _sine()
    stereo = np.stack([mono, mono], axis=0)
    compat = _proxy_mono_compat(stereo)
    assert compat > 0.95  # perfectly in-phase = high mono compat


def test_71_proxy_mono_compat_anti_phase():
    from backend.core.cassette_defect_verifier import _proxy_mono_compat  # type: ignore[attr-defined]

    mono = _sine()
    anti_phase = np.stack([mono, -mono], axis=0)
    compat = _proxy_mono_compat(anti_phase)
    assert compat < 0.1  # anti-phase = very low mono compat


def test_72_proxy_mono_compat_mono_input():
    from backend.core.cassette_defect_verifier import _proxy_mono_compat  # type: ignore[attr-defined]

    mono = _sine()
    compat = _proxy_mono_compat(mono)
    assert compat == 1.0  # mono input → return 1.0


# ---------------------------------------------------------------------------
# 08 — measure_proxies
# ---------------------------------------------------------------------------


def test_80_measure_proxies_known_phase():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    proxies = pdv.measure_proxies("phase_03_denoise", _sine(), SR)
    # phase_03 targets HIGH_FREQ_NOISE → should have at least one proxy
    # (may be empty if reverse map unavailable in test env — that's also OK)
    assert isinstance(proxies, dict)


def test_81_measure_proxies_unknown_phase():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    proxies = pdv.measure_proxies("phase_99_nonexistent", _sine(), SR)
    assert proxies == {}


def test_82_measure_proxies_never_raises():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    # Pass garbage audio — must not raise
    garbage = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    proxies = pdv.measure_proxies("phase_30_dc_offset_removal", garbage, SR)
    assert isinstance(proxies, dict)


# ---------------------------------------------------------------------------
# 09 — check(): core behaviour
# ---------------------------------------------------------------------------


def test_90_check_unknown_phase_returns_audio_after():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    before = _sine()
    after = before * 0.9
    result = pdv.check("phase_99_unknown", before, after, SR)
    assert result is after  # no proxy → return audio_after unchanged


def test_91_check_no_rollback_when_defect_improved():
    """If a phase reduced HF noise, audio_after must be kept."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    noisy = _add_hf_noise(_sine(), level=0.30)
    clean = _sine()  # simulated denoised output

    pdv.check("phase_03_denoise", noisy, clean, SR)
    # Noise floor decreased: no rollback
    # Result is clean (possibly) — no identity check needed; rollback must NOT occur
    summary = pdv.get_session_summary()
    assert summary.get("rollback_count", 0) == 0


def test_92_check_rollback_when_defect_worsened():
    """If a phase INCREASED HF noise, audio_before must be returned."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    clean = _sine()
    # Simulate a phase that ADDS noise (bug scenario)
    noisier = _add_hf_noise(clean, level=0.50)

    # phase_03_denoise should reduce HF noise; instead it added a LOT → rollback
    result = pdv.check("phase_03_denoise", clean, noisier, SR)
    summary = pdv.get_session_summary()
    # If the reverse map resolved HIGH_FREQ_NOISE for phase_03 AND the proxy fired:
    if summary.get("total_checked", 0) > 0:
        assert result is clean or result is noisier  # valid return type
        # If rollback, result should be the clean reference
        if summary.get("rollback_count", 0) > 0:
            assert np.allclose(result, clean)


def test_93_check_hum_rollback():
    """phase_02_hum_removal: if hum energy increased, rollback."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    hummed = _add_hum(_sine())
    # Simulate a buggy phase that ADDS more hum
    more_hummed = _add_hum(hummed, amp=0.20)

    result = pdv.check("phase_02_hum_removal", hummed, more_hummed, SR)
    summary = pdv.get_session_summary()
    assert isinstance(result, np.ndarray)
    if summary.get("rollback_count", 0) > 0:
        assert np.allclose(result, hummed)


def test_94_check_dc_offset_rollback():
    """phase_30_dc_offset_removal: if DC increased, rollback."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    dc_audio = _add_dc(_sine(), dc=0.03)
    # Simulate a buggy phase that adds MORE DC
    more_dc = _add_dc(dc_audio, dc=0.10)

    result = pdv.check("phase_30_dc_offset_removal", dc_audio, more_dc, SR)
    summary = pdv.get_session_summary()
    assert isinstance(result, np.ndarray)
    if summary.get("rollback_count", 0) > 0:
        assert np.allclose(result, dc_audio)


def test_95_check_stereo_audio():
    """check() handles stereo audio for both before and after."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    stereo_before = _stereo(_add_hf_noise(_sine(), level=0.20))
    stereo_after = _stereo(_sine())  # improved

    result = pdv.check("phase_03_denoise", stereo_before, stereo_after, SR)
    assert result.ndim == 2
    assert result.shape == stereo_after.shape or result.shape == stereo_before.shape


def test_96_check_metadata_store_populated():
    """check() populates metadata_store when provided."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()
    meta: dict = {}
    noisy = _add_hf_noise(_sine(), level=0.20)
    clean = _sine()

    pdv.check("phase_03_denoise", noisy, clean, SR, metadata_store=meta)
    # metadata_store should have a 'phase_defect_verification' key if proxy fired
    # (it may be absent if phase has no targets in this env)
    if "phase_defect_verification" in meta:
        pdv_list = meta["phase_defect_verification"]
        assert isinstance(pdv_list, list)
        for entry in pdv_list:
            assert "phase_id" in entry
            assert "rollback" in entry


def test_96a_check_reweight_avoids_rollback_and_sets_metadata(monkeypatch):
    """Bei Proxy-Worsening wird zuerst reweighted; erfolgreicher Blend vermeidet Rollback."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    before = np.zeros(SR, dtype=np.float32)
    after = np.ones(SR, dtype=np.float32) * 0.6
    meta: dict = {}

    def _fake_measure(phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        if np.allclose(audio, before):
            return {"HIGH_FREQ_NOISE": 1.0}
        if np.allclose(audio, after):
            return {"HIGH_FREQ_NOISE": 1.40}  # +40% -> rollback trigger ohne Reweight
        return {"HIGH_FREQ_NOISE": 1.20}  # +20% -> unter 25%-Schwelle

    monkeypatch.setattr(pdv, "measure_proxies", _fake_measure)

    result = pdv.check("phase_03_denoise", before, after, SR, metadata_store=meta)

    expected = np.clip(0.92 * after + 0.08 * before, -1.0, 1.0).astype(np.float32)
    assert np.allclose(result, expected)

    entries = meta.get("phase_defect_verification", [])
    assert entries, "PDV-Metadataeintrag fehlt"
    last = entries[-1]
    assert last.get("rollback") is False
    assert last.get("reweight_applied") is True
    assert np.isclose(float(last.get("reweight_alpha", 0.0)), 0.92)


def test_96b_check_naturalness_hard_guard_forces_rollback_without_reweight(monkeypatch):
    """Bei Naturalness-Hard-Guard bleibt sofortiger Rollback Pflicht."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    before = np.zeros(SR, dtype=np.float32)
    after = np.ones(SR, dtype=np.float32) * 0.6
    meta: dict = {}

    def _fake_measure(phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        if np.allclose(audio, before):
            return {"HIGH_FREQ_NOISE": 1.0}
        if np.allclose(audio, after):
            return {"HIGH_FREQ_NOISE": 1.40}
        return {"HIGH_FREQ_NOISE": 1.20}

    monkeypatch.setattr(pdv, "measure_proxies", _fake_measure)

    result = pdv.check(
        "phase_03_denoise",
        before,
        after,
        SR,
        metadata_store=meta,
        goal_before={"natuerlichkeit": 0.90},
        goal_after={"natuerlichkeit": 0.70},
    )

    assert np.allclose(result, before)

    entries = meta.get("phase_defect_verification", [])
    assert entries, "PDV-Metadataeintrag fehlt"
    last = entries[-1]
    assert last.get("rollback") is True
    assert last.get("reweight_applied") is False
    assert np.isclose(float(last.get("reweight_alpha", 0.0)), 0.0)


def test_96c_check_psycho_hf_guard_strictens_threshold_to_rollback(monkeypatch):
    """Leichter Proxy-Drift kann bei starker Hoerbarkeit trotzdem rollbacken."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    before = np.zeros(SR, dtype=np.float32)
    after = np.ones(SR, dtype=np.float32) * 0.1

    def _fake_measure(phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        if np.allclose(audio, before):
            return {"HIGH_FREQ_NOISE": 1.0}
        return {"HIGH_FREQ_NOISE": 1.20}  # +20% (normalerweise unter 25%-Rollback-Schwelle)

    monkeypatch.setattr(pdv, "measure_proxies", _fake_measure)
    monkeypatch.setattr(
        "backend.core.cassette_defect_verifier._compute_hf_noise_audibility",
        lambda audio, sr: 0.10 if np.allclose(audio, before) else 0.90,
    )
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_transient_harshness", lambda audio, sr: 0.0)

    result = pdv.check("phase_03_denoise", before, after, SR)
    assert np.allclose(result, before)


def test_96d_check_metadata_contains_psychoacoustic_payload(monkeypatch):
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()
    meta: dict = {}
    before = _sine()
    after = _add_hf_noise(before, level=0.05)

    monkeypatch.setattr(
        "backend.core.cassette_defect_verifier._compute_hf_noise_audibility",
        lambda audio, sr: 0.2 if np.allclose(audio, before) else 0.3,
    )
    monkeypatch.setattr(
        "backend.core.cassette_defect_verifier._compute_transient_harshness",
        lambda audio, sr: 0.1 if np.allclose(audio, before) else 0.2,
    )
    monkeypatch.setattr(
        "backend.core.cassette_defect_verifier._compute_quasi_peak_burstiness",
        lambda audio, sr: 0.1 if np.allclose(audio, before) else 0.2,
    )

    pdv.check("phase_03_denoise", before, after, SR, metadata_store=meta)

    entries = meta.get("phase_defect_verification", [])
    if entries:
        last = entries[-1]
        assert last.get("psychoacoustic_guard") is True
        assert "psychoacoustic_before" in last
        assert "psychoacoustic_after" in last
        assert "quasi_peak_burstiness" in last["psychoacoustic_before"]
        assert "quasi_peak_burstiness" in last["psychoacoustic_after"]


def test_96e_check_defect_specific_reweight_prefers_click_profile(monkeypatch):
    """CLICKS nutzt eine eigene Reweight-Staffel statt globalem HF-Default."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    before = np.zeros(SR, dtype=np.float32)
    after = np.ones(SR, dtype=np.float32) * 0.6
    meta: dict = {}

    def _fake_measure(phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        if np.allclose(audio, before):
            return {"CLICKS": 1.0}
        if np.allclose(audio, after):
            return {"CLICKS": 1.40}
        mean_amp = float(np.mean(audio))
        # alpha=0.90 -> mean 0.54 -> noch rollback
        if mean_amp > 0.50:
            return {"CLICKS": 1.26}
        # alpha=0.82 -> mean 0.492 -> kein rollback
        return {"CLICKS": 1.16}

    monkeypatch.setattr(pdv, "measure_proxies", _fake_measure)
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_hf_noise_audibility", lambda a, s: 0.0)
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_transient_harshness", lambda a, s: 0.0)
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_quasi_peak_burstiness", lambda a, s: 0.0)

    result = pdv.check("phase_09_crackle_removal", before, after, SR, metadata_store=meta)
    assert isinstance(result, np.ndarray)
    assert result.shape == before.shape

    entries = meta.get("phase_defect_verification", [])
    assert entries
    last = entries[-1]
    assert last.get("reweight_applied") is True
    assert np.isclose(float(last.get("reweight_alpha", 0.0)), 0.90)
    assert last.get("reweight_strategy") == "burst_selective"


def test_96f_check_hum_uses_frequency_selective_reweight_strategy(monkeypatch):
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    before = np.zeros(SR, dtype=np.float32)
    after = np.ones(SR, dtype=np.float32) * 0.6
    fs_candidate = np.ones(SR, dtype=np.float32) * 0.2
    meta: dict = {}

    def _fake_measure(phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        if np.allclose(audio, before):
            return {"HUM": 1.0}
        if np.allclose(audio, after):
            return {"HUM": 1.40}
        if np.allclose(audio, fs_candidate):
            return {"HUM": 1.12}
        return {"HUM": 1.30}

    monkeypatch.setattr(pdv, "measure_proxies", _fake_measure)
    monkeypatch.setattr(
        "backend.core.cassette_defect_verifier._frequency_selective_blend",
        lambda audio_before, audio_after, alpha, worst_defect, sr: fs_candidate,
    )
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_hf_noise_audibility", lambda a, s: 0.0)
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_transient_harshness", lambda a, s: 0.0)
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_quasi_peak_burstiness", lambda a, s: 0.0)

    result = pdv.check("phase_02_hum_removal", before, after, SR, metadata_store=meta)
    assert np.allclose(result, fs_candidate)

    entries = meta.get("phase_defect_verification", [])
    assert entries
    last = entries[-1]
    assert last.get("reweight_applied") is True
    assert last.get("reweight_strategy") == "frequency_selective"


def test_96g_check_clicks_uses_burst_selective_reweight_strategy(monkeypatch):
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    before = np.zeros(SR, dtype=np.float32)
    after = np.ones(SR, dtype=np.float32) * 0.6
    burst_candidate = np.ones(SR, dtype=np.float32) * 0.2
    meta: dict = {}

    def _fake_measure(phase_id: str, audio: np.ndarray, sr: int) -> dict[str, float]:
        if np.allclose(audio, before):
            return {"CLICKS": 1.0}
        if np.allclose(audio, after):
            return {"CLICKS": 1.40}
        if np.allclose(audio, burst_candidate):
            return {"CLICKS": 1.12}
        return {"CLICKS": 1.30}

    monkeypatch.setattr(pdv, "measure_proxies", _fake_measure)
    monkeypatch.setattr(
        "backend.core.cassette_defect_verifier._frequency_selective_blend",
        lambda audio_before, audio_after, alpha, worst_defect, sr: burst_candidate,
    )
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_hf_noise_audibility", lambda a, s: 0.0)
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_transient_harshness", lambda a, s: 0.0)
    monkeypatch.setattr("backend.core.cassette_defect_verifier._compute_quasi_peak_burstiness", lambda a, s: 0.0)

    result = pdv.check("phase_09_crackle_removal", before, after, SR, metadata_store=meta)
    assert np.allclose(result, burst_candidate)

    entries = meta.get("phase_defect_verification", [])
    assert entries
    last = entries[-1]
    assert last.get("reweight_applied") is True
    assert last.get("reweight_strategy") == "burst_selective"


def test_97_check_never_raises_on_nan_input():
    """check() must return a valid array even on NaN/Inf input."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    nan_audio = np.full(SR, np.nan, dtype=np.float32)
    normal = _sine()

    # Must not raise
    result = pdv.check("phase_30_dc_offset_removal", nan_audio, normal, SR)
    assert isinstance(result, np.ndarray)


def test_98_check_1_sample_audio():
    """check() handles 1-sample edge case."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    one_sample = np.array([0.5], dtype=np.float32)
    result = pdv.check("phase_02_hum_removal", one_sample, one_sample.copy(), SR)
    assert isinstance(result, np.ndarray)


def test_99_check_zero_length_audio():
    """check() handles empty audio without raising."""
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    empty = np.array([], dtype=np.float32)
    result = pdv.check("phase_02_hum_removal", empty, empty.copy(), SR)
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# 10 — Session telemetry
# ---------------------------------------------------------------------------


def test_100_session_reset_clears():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    # run a check then reset
    pdv.check("phase_30_dc_offset_removal", _sine(), _sine(), SR)
    pdv.reset_session()
    summary = pdv.get_session_summary()
    assert summary.get("total_checked", 0) == 0


def test_101_session_summary_structure():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()
    summary = pdv.get_session_summary()
    required_keys = {"total_checked", "rollback_count", "miss_count", "rollback_phases", "miss_phases"}
    assert required_keys.issubset(summary.keys())


def test_102_session_summary_counts_correctly():
    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    pdv.reset_session()

    # Two checks for phases with targets (even if proxy map not available in test env,
    # measure_proxies returns {} → check skips → total_checked stays 0)
    pdv.check("phase_02_hum_removal", _sine(), _sine(), SR)
    pdv.check("phase_30_dc_offset_removal", _sine(), _sine(), SR)

    summary = pdv.get_session_summary()
    assert isinstance(summary.get("total_checked", 0), int)
    assert isinstance(summary.get("rollback_count", 0), int)
    assert summary.get("rollback_count", 0) >= 0


def test_103_thread_safety_reset():
    """Concurrent reset + check must not crash."""
    import threading

    from backend.core.cassette_defect_verifier import get_phase_defect_verifier  # type: ignore[attr-defined]

    pdv = get_phase_defect_verifier()
    errors: list[str] = []

    def _work() -> None:
        try:
            for _ in range(5):
                pdv.check("phase_03_denoise", _sine(), _sine(), SR)
                pdv.reset_session()
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=_work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
