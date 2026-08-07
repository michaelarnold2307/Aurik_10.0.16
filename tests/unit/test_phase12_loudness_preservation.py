import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix


def _rms_db(x: np.ndarray) -> float:
    xr = np.asarray(x, dtype=np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean(xr**2) + 1e-12)))


@pytest.mark.unit
def test_phase12_loudness_guard_limits_large_drop() -> None:
    phase = WowFlutterFix()
    n = 48000
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    original = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    processed = (original * 0.20).astype(np.float32)  # ~ -14 dB drop

    corrected, delta_db, makeup_db = phase._preserve_phase_loudness(original, processed, MaterialType.VINYL)

    assert corrected.shape == original.shape
    assert np.isfinite(corrected).all()
    assert delta_db >= -1.7  # vinyl cap + small numeric tolerance
    assert makeup_db > 0.0


def test_phase12_loudness_guard_limits_over_lift() -> None:
    phase = WowFlutterFix()
    n = 48000
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    original = (0.20 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    processed = (original * 2.0).astype(np.float32)  # strong lift

    corrected, delta_db, makeup_db = phase._preserve_phase_loudness(original, processed, MaterialType.TAPE)

    assert corrected.shape == original.shape
    assert np.isfinite(corrected).all()
    assert delta_db <= 1.1
    assert makeup_db < 0.0
    # Ensure guard actually changed output level relative to input processed signal.
    assert _rms_db(corrected) < _rms_db(processed)


def test_phase12_safe_timing_profile_dampens_vocal_vinyl() -> None:
    phase = WowFlutterFix()

    strength_scale, max_stretch_delta = phase._derive_safe_timing_profile(
        MaterialType.VINYL,
        mean_confidence=0.55,
        vocals_confidence=0.75,
    )

    assert strength_scale < 1.0
    assert max_stretch_delta <= 0.03


def test_phase12_stretch_factors_respect_safe_delta() -> None:
    phase = WowFlutterFix()
    pitch = np.array([220.0, 228.0, 212.0, 224.0, 216.0, 230.0, 210.0, 222.0, 218.0, 226.0, 214.0], dtype=np.float64)
    conf = np.ones_like(pitch, dtype=np.float64) * 0.95

    stretch = phase._calculate_stretch_factors(pitch, conf, 1.0, max_stretch_delta=0.03)

    assert np.isfinite(stretch).all()
    assert float(np.min(stretch)) >= 0.97 - 1e-6
    assert float(np.max(stretch)) <= 1.03 + 1e-6


def test_phase12_polyphonic_fallback_tightens_timing_profile() -> None:
    phase = WowFlutterFix()

    strength_scale, max_stretch_delta = phase._derive_safe_timing_profile(
        MaterialType.VINYL,
        mean_confidence=0.72,
        vocals_confidence=0.65,
        polyphonic_fallback=True,
    )

    assert strength_scale < 0.82
    assert max_stretch_delta <= 0.02


def test_phase12_unsafe_polyphonic_fallback_is_bypassed_for_vocal_vinyl() -> None:
    phase = WowFlutterFix()

    assert phase._should_bypass_unsafe_polyphonic_fallback(
        MaterialType.VINYL,
        mean_confidence=0.72,
        vocals_confidence=0.65,
        polyphonic_fallback=True,
    )


def test_phase12_unsafe_polyphonic_fallback_not_bypassed_for_nonvocal() -> None:
    phase = WowFlutterFix()

    assert not phase._should_bypass_unsafe_polyphonic_fallback(
        MaterialType.VINYL,
        mean_confidence=0.72,
        vocals_confidence=0.10,
        polyphonic_fallback=True,
    )


def test_phase12_loudness_guard_realigns_stereo_delay() -> None:
    phase = WowFlutterFix()
    sr = 48000
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float64)
    left = (0.25 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    delay = 220
    right = np.pad(left, (delay, 0))[: len(left)].astype(np.float32)
    original = np.column_stack([left, right])
    processed = original.copy()

    corrected, _delta_db, _makeup_db = phase._preserve_phase_loudness(original, processed, MaterialType.VINYL)

    corr = np.correlate(corrected[:, 0], corrected[:, 1], mode="full")
    lag = int(np.argmax(corr) - (len(corrected[:, 0]) - 1))
    assert abs(lag) <= 2, f"L/R-Zeitversatz nicht korrigiert: lag={lag} samples"


def test_phase12_loudness_guard_caps_percentile_peak() -> None:
    phase = WowFlutterFix()
    n = 48000
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    original = (0.03 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    processed = (original * 30.0).astype(np.float32)

    corrected, _delta_db, _makeup_db = phase._preserve_phase_loudness(original, processed, MaterialType.TAPE)
    peak99 = float(np.percentile(np.abs(corrected), 99.9))
    assert peak99 <= 0.986, f"p99.9-Peak zu hoch nach Guard: {peak99:.4f}"


def test_phase12_stretch_locality_mask_neutralizes_outside_segments() -> None:
    stretch = np.array([0.97, 1.03, 1.02, 0.98, 1.04, 0.96, 1.01, 0.99], dtype=np.float32)

    masked, coverage = WowFlutterFix._apply_defect_locality_to_stretch_factors(
        stretch,
        audio_length_samples=48_000,
        sample_rate=48_000,
        defect_locations={"wow": [(0.20, 0.40)]},
    )

    assert np.isfinite(masked).all()
    assert 0.0 < coverage < 1.0
    assert np.allclose(masked[:1], 1.0)
    assert np.allclose(masked[-2:], 1.0)
    assert np.any(np.abs(masked - 1.0) > 1e-4)


def test_phase12_process_applies_defect_locality_before_timestretch(monkeypatch) -> None:
    phase = WowFlutterFix()
    sr = 48_000
    t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float64)
    audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    monkeypatch.setattr(phase, "_estimate_pitch_yin", lambda _a, _sr: (np.full(32, 220.0), np.full(32, 0.9)))
    monkeypatch.setattr(phase, "_separate_wow_flutter", lambda _p, _sr: (np.zeros_like(_p), np.zeros_like(_p)))
    monkeypatch.setattr(phase, "_detect_wow_flutter", lambda _p, _c, _thr: (True, 0.8))
    monkeypatch.setattr(
        phase,
        "_calculate_stretch_factors",
        lambda _p, _c, _s, max_stretch_delta=0.05: np.linspace(0.97, 1.03, 32, dtype=np.float32),
    )
    monkeypatch.setattr(phase, "_calculate_max_deviation", lambda _p, _c: 0.2)
    monkeypatch.setattr(phase, "_preserve_phase_loudness", lambda orig, proc, material: (proc, 0.0, 0.0))
    monkeypatch.setattr(phase, "_apply_neural_phase_coherence", lambda audio_in, _sr, reference=None: audio_in)

    captured: dict[str, np.ndarray] = {}

    def _fake_stretch(x: np.ndarray, stretch_factors: np.ndarray, _sample_rate: int) -> np.ndarray:
        captured["stretch_factors"] = np.asarray(stretch_factors, dtype=np.float32).copy()
        return np.asarray(x, dtype=np.float32).copy()

    monkeypatch.setattr(phase, "_phase_vocoder_timestretch", _fake_stretch)

    result = phase.process(
        audio,
        sr,
        MaterialType.VINYL,  # type: ignore[arg-type]
        strength=1.0,
        defect_locations={"wow": [(0.20, 0.35)]},
    )

    stretch_used = captured["stretch_factors"]
    assert np.isfinite(stretch_used).all()
    assert np.allclose(stretch_used[:4], 1.0)
    assert np.any(np.abs(stretch_used[7:13] - 1.0) > 1e-4)
    assert np.allclose(stretch_used[-6:], 1.0)
    assert float(result.metadata.get("stretch_locality_coverage", 0.0)) > 0.0
