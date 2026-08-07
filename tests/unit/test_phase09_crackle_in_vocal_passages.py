"""
Regression test: phase_09 must detect and remove crackle even in harmonic/vocal passages.

Before the fix, _classify_crackle_regions() required harmonic_ratio < 0.4 on the full
1-second window. Vocal content makes harmonic_ratio >= 0.4, so crackle in vocal
passages was never repaired (crackle_regions_found == 0).

Fix: removed harmonic_ratio < 0.4 from is_crackle; ZCR > 0.3 + centroid > 3000
is sufficient to identify impulsive broadband noise even in harmonic contexts.
"""

import sys
import types

import numpy as np
import pytest

import backend.core.phases.phase_09_crackle_removal as p09mod

SR = 48_000


def _make_vocal_with_crackle(duration_s: float = 1.2) -> np.ndarray:
    """Synthesise a harmonic vocal-like tone (440 Hz + harmonics) with superimposed clicks.

    The signal has harmonic_ratio >> 0.4 so the old code would never classify
    it as a crackle region.  The clicks add broadband energy and raise ZCR.

    Clicks are modelled as exponential-decaying broadband noise bursts — matching
    real vinyl crackle (short-duration, HF-dominant impulse response of stylus+groove
    damage).  A smooth Hanning envelope would keep ZCR < 0.1 (too low for the zcr>0.3
    criterion); broadband noise bursts yield ZCR ≈ 0.45 in the crackle windows.
    """
    rng = np.random.default_rng(42)
    t = np.arange(int(duration_s * SR), dtype=np.float32) / SR

    # Harmonic vocal: f0=440 Hz plus overtones
    vocal = (
        0.40 * np.sin(2 * np.pi * 440 * t)
        + 0.20 * np.sin(2 * np.pi * 880 * t)
        + 0.12 * np.sin(2 * np.pi * 1320 * t)
        + 0.07 * np.sin(2 * np.pi * 1760 * t)
    ).astype(np.float32)

    # Superimpose vinyl-like impulse crackle (25 clicks randomly distributed).
    # Each click = exponential-decaying broadband noise burst (2 ms): high ZCR + high centroid.
    audio = vocal.copy()
    n_clicks = 25
    click_positions = rng.integers(SR // 4, int(duration_s * SR) - SR // 4, size=n_clicks)
    click_len = int(0.002 * SR)  # 2 ms broadband burst
    for pos in click_positions:
        click_sign = rng.choice([-1.0, 1.0])
        noise_burst = rng.standard_normal(click_len).astype(np.float32)
        decay = np.exp(-np.arange(click_len, dtype=np.float32) * 3.0 / click_len)
        click = click_sign * 0.55 * noise_burst * decay
        s, e = pos, pos + click_len
        if s < 0 or e > len(audio):
            continue
        audio[s:e] += click

    audio = np.clip(audio, -1.0, 1.0)
    return audio  # type: ignore[no-any-return]


@pytest.fixture
def phase09():
    from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase

    return CrackleRemovalPhase(sample_rate=SR)


@pytest.mark.unit
def test_crackle_regions_detected_in_vocal_passage(phase09):
    """crackle_regions must be non-empty when clicks overlay a harmonic signal."""
    audio = _make_vocal_with_crackle()
    params = {
        # 3.5 σ: detects synthesized clicks (adaptive_threshold ≈ 0.37 < click peak ≈ 0.55).
        # Production uses 0.15 (more sensitive). 5.0 would yield adaptive_threshold ≈ 0.53
        # — too close to click amplitude for reliable detection on short windows.
        "transient_threshold": 3.5,
        "min_density": 2,
        "interpolation": "spectral",
        "background_model": False,
        "texture_preserve": 0.0,
    }
    t_short, t_medium, t_long = phase09._detect_transients_multiscale(audio, params)
    regions = phase09._classify_crackle_regions(audio, t_short, t_medium, t_long, params)
    assert len(regions) > 0, (
        "No crackle regions detected in vocal+crackle passage. "
        "harmonic_ratio guard likely still blocking vocal regions."
    )


def test_crackle_removed_in_vocal_passage(phase09):
    """Restored signal must have lower HF impulsive energy than input in vocal+crackle region."""
    audio = _make_vocal_with_crackle()

    result = phase09.process(
        audio,
        material_type="vinyl",
        strength=1.0,
        mode="restoration",
        context=None,
    )
    assert result.success, "process() must succeed"
    restored = result.audio
    assert restored.shape == audio.shape

    # Crackle regions must have been detected (guard was removed for vocal passages).
    n_regions = result.modifications.get("crackle_regions_found", 0)
    assert n_regions > 0, (
        "No crackle regions found — centroid/ZCR guard may have been re-introduced, "
        "blocking detection in harmonic-heavy vocal passages."
    )

    # The phase must report measurable crackle reduction (HF impulsive energy).
    # Even the DSP fallback (texture_preserve blend) attenuates click HF energy ≥ 1 dB.
    reduction_db = result.modifications.get("crackle_reduction_db", 0.0)
    assert reduction_db > 0.5, (
        f"Expected >0.5 dB crackle reduction, got {reduction_db:.1f} dB. "
        f"Phase 09 may have processed vocal passage as passthrough."
    )

    # The output must differ from the input (processing did happen).
    diff_energy = float(np.mean((audio.astype(np.float64) - restored.astype(np.float64)) ** 2))
    assert diff_energy > 1e-8, (
        f"Output identical to input (diff_energy={diff_energy:.2e}): "
        f"phase_09 skipped the vocal+crackle passage entirely."
    )


def test_pure_harmonic_not_over_processed(phase09):
    """A pure harmonic signal without crackle must not be silenced or distorted."""
    np.random.default_rng(7)
    t = np.arange(SR, dtype=np.float32) / SR
    # Clean vocal tone, no crackle
    audio = (
        0.40 * np.sin(2 * np.pi * 440 * t) + 0.20 * np.sin(2 * np.pi * 880 * t) + 0.08 * np.sin(2 * np.pi * 1320 * t)
    ).astype(np.float32)

    params = {
        "transient_threshold": 3.5,
        "min_density": 2,
        "interpolation": "spectral",
        "background_model": False,
        "texture_preserve": 0.0,
    }
    t_short, t_medium, t_long = phase09._detect_transients_multiscale(audio, params)
    regions = phase09._classify_crackle_regions(audio, t_short, t_medium, t_long, params)

    # Pure sine should generate very few or no crackle regions
    total_crackle_samples = sum(e - s for s, e in regions)
    assert total_crackle_samples <= SR * 0.10, (
        f"Pure harmonic signal falsely classified as crackle: {total_crackle_samples / SR:.2f}s of {1.0:.2f}s total"
    )


# ---------------------------------------------------------------------------
# §0p Vokal-Konservatismus-Tests
# ---------------------------------------------------------------------------


def test_vocal_cap_limits_strength_when_singing_high(phase09):
    """§0p: Bei panns_singing >= 0.35 darf effective_strength 0.70 nicht überschreiten."""
    np.random.default_rng(42)
    t = np.arange(SR, dtype=np.float32) / SR
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # panns_singing=0.80, strength=1.0 → effective_strength sollte auf 0.70 gecapped werden
    result = phase09.process(
        audio,
        material_type="vinyl",
        sample_rate=SR,
        strength=1.0,
        panns_singing=0.80,
    )
    eff = result.metadata.get("effective_strength", 1.0)
    assert eff <= 0.70 + 1e-6, f"§0p Vokal-Cap verletzt: effective_strength={eff:.3f} > 0.70 bei panns_singing=0.80"


def test_vocal_cap_does_not_activate_when_singing_low(phase09):
    """§0p: Bei panns_singing < 0.35 bleibt effective_strength unbegrenzt."""
    np.random.default_rng(42)
    t = np.arange(SR, dtype=np.float32) / SR
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    result = phase09.process(
        audio,
        material_type="vinyl",
        sample_rate=SR,
        strength=1.0,
        panns_singing=0.10,
    )
    eff = result.metadata.get("effective_strength", 0.0)
    assert eff > 0.70, f"§0p Vokal-Cap fälschlicherweise aktiv: effective_strength={eff:.3f} bei panns_singing=0.10"


def test_region_selective_blend_is_conservative_outside_crackle_regions(phase09):
    """Außerhalb Crackle-Regionen muss der Blend konservativer als globales Wet sein."""
    n = SR
    dry = np.zeros(n, dtype=np.float32)
    wet = np.ones(n, dtype=np.float32)
    regions = [(n // 2 - 200, n // 2 + 200)]
    eff = 0.8

    out = phase09._apply_region_selective_strength_blend(dry, wet, regions, eff, SR)
    assert out.shape == dry.shape

    # Außerhalb Region deutlich konservativer als globales alpha=0.8
    outside = np.r_[out[: n // 2 - 400], out[n // 2 + 400 :]]
    assert float(np.mean(outside)) < 0.6


def test_region_selective_blend_without_regions_returns_dry(phase09):
    """Wenn keine Crackle-Region erkannt wurde, darf kein globales Wet angewandt werden."""
    dry = (0.2 * np.sin(2 * np.pi * 440 * (np.arange(SR, dtype=np.float32) / SR))).astype(np.float32)
    wet = np.clip(dry * 0.1, -1.0, 1.0).astype(np.float32)

    out = phase09._apply_region_selective_strength_blend(dry, wet, [], 0.7, SR)
    assert np.allclose(out, dry)


def test_region_selective_blend_without_regions_can_fallback_to_global(phase09):
    """ML-Pfad: Bei leeren Regionen darf optional globaler Blend genutzt werden."""
    dry = np.zeros(SR, dtype=np.float32)
    wet = np.ones(SR, dtype=np.float32)
    eff = 0.7

    out = phase09._apply_region_selective_strength_blend(
        dry,
        wet,
        [],
        eff,
        SR,
        fallback_to_global_when_no_regions=True,
    )
    assert np.allclose(out, np.full(SR, eff, dtype=np.float32), atol=1e-5)


def test_compute_crackle_regions_with_protection_applies_phoneme_mask(phase09, monkeypatch):
    """Gemeinsame Regionenquelle muss §2.36-Phonemschutz auch fuer ML/DSP erzwingen."""
    audio = np.zeros(SR, dtype=np.float32)
    params = {
        "transient_threshold": 0.1,
        "min_density": 1,
        "interpolation": "spectral",
        "background_model": False,
        "texture_preserve": 0.0,
    }

    # Erzwinge eine einzige Crackle-Region ueber den ganzen Track.
    monkeypatch.setattr(phase09, "_detect_transients_multiscale", lambda a, p: ([10], [20], [30]))
    monkeypatch.setattr(
        phase09,
        "_classify_crackle_regions",
        lambda a, ts, tm, tl, p: [(0, len(a))],
    )

    # Fake LGE-Modul mit aktiver Phonemmaske (alle Frames True) -> Region muss entfernt werden.
    fake_lge = types.SimpleNamespace(get_phoneme_mask=lambda mono, sr, hop_length=512: np.ones(8, dtype=bool))
    monkeypatch.setitem(sys.modules, "backend.core.lyrics_guided_enhancement", fake_lge)

    _, _, _, regions = phase09._compute_crackle_regions_with_protection(audio, params)
    assert regions == []


def test_ml_localization_with_phoneme_mask_keeps_outside_conservative(phase09, monkeypatch):
    """Aktive Phonemmaske soll Regionen filtern, verbleibende Regionen aber lokal stark bearbeiten."""
    n = SR
    audio = np.zeros(n, dtype=np.float32)
    params = {
        "transient_threshold": 0.1,
        "min_density": 1,
        "interpolation": "spectral",
        "background_model": False,
        "texture_preserve": 0.0,
    }

    # Zwei Regionen: erste wird durch Phonemmaske entfernt, zweite bleibt erhalten.
    monkeypatch.setattr(phase09, "_detect_transients_multiscale", lambda a, p: ([10], [20], [30]))
    monkeypatch.setattr(
        phase09,
        "_classify_crackle_regions",
        lambda a, ts, tm, tl, p: [(200, 1200), (3000, 4200)],
    )

    def _mask(_mono, _sr, hop_length=512):
        m = np.zeros(10, dtype=bool)
        m[0:3] = True  # schützt ca. Samples 0..1536 -> entfernt nur die erste Region
        return m

    fake_lge = types.SimpleNamespace(get_phoneme_mask=_mask)
    monkeypatch.setitem(sys.modules, "backend.core.lyrics_guided_enhancement", fake_lge)

    _, _, _, regions = phase09._compute_crackle_regions_with_protection(audio, params)
    assert regions == [(3000, 4200)]

    dry = np.zeros(n, dtype=np.float32)
    wet = np.ones(n, dtype=np.float32)
    out = phase09._apply_region_selective_strength_blend(
        dry,
        wet,
        regions,
        0.8,
        SR,
        fallback_to_global_when_no_regions=True,
    )

    # In verbleibender Region nahe voller Stärke, außerhalb deutlich konservativer als global 0.8.
    inside_mean = float(np.mean(out[3200:3800]))
    outside = np.r_[out[:1500], out[5000:8000]]
    outside_mean = float(np.mean(outside))
    assert inside_mean > 0.70
    assert outside_mean < 0.60


def test_process_onnx_branch_uses_region_selective_blend(phase09, monkeypatch):
    """ONNX-Branch muss lokale Regionsstaerke nutzen (kein globales Wetting)."""
    n = SR
    audio = np.zeros(n, dtype=np.float32)

    monkeypatch.setattr(p09mod, "QUALITY_MODE_AVAILABLE", True)
    monkeypatch.setattr(p09mod, "is_phase_ml_enabled", lambda phase_id: phase_id == 9)
    monkeypatch.setattr(p09mod, "log_mode_decision", lambda *args, **kwargs: None)

    monkeypatch.setattr(phase09, "_remove_crackle_onnx_direct", lambda a, sr, p: np.ones_like(a, dtype=np.float32))
    monkeypatch.setattr(phase09, "_measure_crackle_reduction", lambda a, b: 12.0)
    monkeypatch.setattr(
        phase09,
        "_compute_crackle_regions_with_protection",
        lambda a, p: ([], [], [], [(3000, 4200)]),
    )

    result = phase09.process(audio, sample_rate=SR, material_type="vinyl", strength=0.8)
    assert result.success
    restored = result.audio

    inside_mean = float(np.mean(restored[3200:3800]))
    outside = np.r_[restored[:1500], restored[5000:8000]]
    outside_mean = float(np.mean(outside))
    assert inside_mean > 0.70
    assert outside_mean < 0.60


def test_process_onnx_branch_without_regions_falls_back_global_blend(phase09, monkeypatch):
    """ONNX-Branch mit leeren Regionen soll globalen Sicherheitsmix verwenden."""
    n = SR
    audio = np.zeros(n, dtype=np.float32)

    monkeypatch.setattr(p09mod, "QUALITY_MODE_AVAILABLE", True)
    monkeypatch.setattr(p09mod, "is_phase_ml_enabled", lambda phase_id: phase_id == 9)
    monkeypatch.setattr(p09mod, "log_mode_decision", lambda *args, **kwargs: None)

    monkeypatch.setattr(phase09, "_remove_crackle_onnx_direct", lambda a, sr, p: np.ones_like(a, dtype=np.float32))
    monkeypatch.setattr(phase09, "_measure_crackle_reduction", lambda a, b: 12.0)
    monkeypatch.setattr(
        phase09,
        "_compute_crackle_regions_with_protection",
        lambda a, p: ([], [], [], []),
    )

    result = phase09.process(audio, sample_rate=SR, material_type="vinyl", strength=0.8)
    assert result.success
    restored = result.audio
    assert np.allclose(restored, np.full(n, 0.8, dtype=np.float32), atol=1e-5)


def test_process_docker_fallback_uses_region_selective_blend(phase09, monkeypatch):
    """Docker-Fallback muss wie ONNX lokal statt global blenden."""
    n = SR
    audio = np.zeros(n, dtype=np.float32)

    monkeypatch.setattr(p09mod, "QUALITY_MODE_AVAILABLE", True)
    monkeypatch.setattr(p09mod, "is_phase_ml_enabled", lambda phase_id: phase_id == 9)
    monkeypatch.setattr(p09mod, "log_mode_decision", lambda *args, **kwargs: None)

    # Erzwinge ONNX-Fehler, damit Docker-Fallback genommen wird.
    monkeypatch.setattr(
        phase09, "_remove_crackle_onnx_direct", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("onnx_fail"))
    )
    monkeypatch.setattr(phase09, "_get_banquet_plugin", lambda: object())
    monkeypatch.setattr(phase09, "_remove_crackle_ml", lambda a, plugin, p: np.ones_like(a, dtype=np.float32))
    monkeypatch.setattr(phase09, "_measure_crackle_reduction", lambda a, b: 11.0)
    monkeypatch.setattr(
        phase09,
        "_compute_crackle_regions_with_protection",
        lambda a, p: ([], [], [], [(3000, 4200)]),
    )

    result = phase09.process(audio, sample_rate=SR, material_type="vinyl", strength=0.8)
    assert result.success
    restored = result.audio

    inside_mean = float(np.mean(restored[3200:3800]))
    outside = np.r_[restored[:1500], restored[5000:8000]]
    outside_mean = float(np.mean(outside))
    assert inside_mean > 0.70
    assert outside_mean < 0.60


def test_process_docker_fallback_without_regions_uses_global_blend(phase09, monkeypatch):
    """Docker-Fallback mit leeren Regionen soll globalen Sicherheitsmix nutzen."""
    n = SR
    audio = np.zeros(n, dtype=np.float32)

    monkeypatch.setattr(p09mod, "QUALITY_MODE_AVAILABLE", True)
    monkeypatch.setattr(p09mod, "is_phase_ml_enabled", lambda phase_id: phase_id == 9)
    monkeypatch.setattr(p09mod, "log_mode_decision", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        phase09, "_remove_crackle_onnx_direct", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("onnx_fail"))
    )
    monkeypatch.setattr(phase09, "_get_banquet_plugin", lambda: object())
    monkeypatch.setattr(phase09, "_remove_crackle_ml", lambda a, plugin, p: np.ones_like(a, dtype=np.float32))
    monkeypatch.setattr(phase09, "_measure_crackle_reduction", lambda a, b: 11.0)
    monkeypatch.setattr(
        phase09,
        "_compute_crackle_regions_with_protection",
        lambda a, p: ([], [], [], []),
    )

    result = phase09.process(audio, sample_rate=SR, material_type="vinyl", strength=0.8)
    assert result.success
    restored = result.audio
    assert np.allclose(restored, np.full(n, 0.8, dtype=np.float32), atol=1e-5)
