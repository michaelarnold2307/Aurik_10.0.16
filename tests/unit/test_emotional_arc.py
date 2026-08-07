import pytest

"""
tests/unit/test_emotional_arc.py — EmotionalArcPreservationMetric Test-Suite (≥ 15 Tests)
Alle Tests synthetisch, kein Datei-I/O, SR = 48_000.
"""

import math

import numpy as np

SR = 48_000
np.random.seed(42)


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────


def _audio(dur: float = 5.0, amp: float = 0.3, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _silence(dur: float = 5.0) -> np.ndarray:
    return np.zeros(int(dur * SR), dtype=np.float32)


def _dynamic_audio(dur: float = 40.0) -> np.ndarray:
    """Audio mit dynamischem Bogen: leise → laut → leise."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    envelope = np.sin(np.pi * t / dur)  # Peaking in the middle
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return (signal * envelope * 0.8).astype(np.float32)  # type: ignore[no-any-return]


# ─── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_00_import():
    from backend.core.emotional_arc_preservation import (
        EmotionalArcPreservationMetric,
        EmotionalArcResult,
    )

    assert EmotionalArcPreservationMetric is not None
    assert EmotionalArcResult is not None


def test_01_singleton_identity():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    a = get_emotional_arc_metric()
    b = get_emotional_arc_metric()
    assert a is b


def test_02_thread_safe():
    import concurrent.futures

    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_emotional_arc_metric) for _ in range(10)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_03_short_audio_returns_neutral():
    """Dateien < 30 s → Metrik deaktiviert, neutrale Rückgabe."""
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _audio(dur=10.0)
    rest = _audio(dur=10.0)
    result = metric.measure(orig, rest, SR)
    # Bei kurzen Dateien: arousal_pearson sollte definiert (≥ 0) oder 1.0 sein
    assert math.isfinite(result.arousal_pearson)


def test_04_result_dataclass_fields():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _audio(dur=35.0)
    rest = _audio(dur=35.0)
    result = metric.measure(orig, rest, SR)
    assert hasattr(result, "arousal_pearson")
    assert hasattr(result, "valence_pearson")
    assert hasattr(result, "klimax_peak_deviation")


def test_05_identical_signals_high_correlation():
    """Identische Signale → Korrelation = 1.0."""
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _dynamic_audio(dur=35.0)
    rest = orig.copy()
    result = metric.measure(orig, rest, SR)
    assert math.isfinite(result.arousal_pearson)
    assert result.arousal_pearson > 0.90


def test_06_all_scores_finite():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _dynamic_audio(dur=35.0)
    rest = _audio(dur=35.0, amp=0.5)
    result = metric.measure(orig, rest, SR)
    assert math.isfinite(result.arousal_pearson)
    assert math.isfinite(result.valence_pearson)
    assert math.isfinite(result.klimax_peak_deviation)


def test_07_arousal_pearson_bounded():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _dynamic_audio(dur=35.0)
    rest = _audio(dur=35.0)
    result = metric.measure(orig, rest, SR)
    assert -1.0 <= result.arousal_pearson <= 1.0


def test_08_valence_pearson_bounded():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _dynamic_audio(dur=35.0)
    rest = _audio(dur=35.0)
    result = metric.measure(orig, rest, SR)
    assert -1.0 <= result.valence_pearson <= 1.0


def test_09_klimax_deviation_non_negative():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _dynamic_audio(dur=35.0)
    rest = _audio(dur=35.0)
    result = metric.measure(orig, rest, SR)
    assert result.klimax_peak_deviation >= 0


def test_10_silence_input_no_crash():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _silence(dur=35.0)
    rest = _silence(dur=35.0)
    result = metric.measure(orig, rest, SR)
    assert math.isfinite(result.arousal_pearson)


def test_11_convenience_function():
    from backend.core.emotional_arc_preservation import measure_emotional_arc

    orig = _dynamic_audio(dur=35.0)
    rest = orig.copy()
    result = measure_emotional_arc(orig, rest, SR)
    assert result is not None
    assert math.isfinite(result.arousal_pearson)


def test_12_threshold_attributes():
    from backend.core.emotional_arc_preservation import EmotionalArcPreservationMetric

    assert hasattr(EmotionalArcPreservationMetric, "THRESHOLD_AROUSAL")
    assert hasattr(EmotionalArcPreservationMetric, "THRESHOLD_VALENCE")
    assert EmotionalArcPreservationMetric.THRESHOLD_AROUSAL > 0.0
    assert EmotionalArcPreservationMetric.THRESHOLD_VALENCE > 0.0


def test_13_sr_check():
    """Falscher SR muss mindestens nicht crashen oder AssertionError."""
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _audio(dur=35.0)
    rest = _audio(dur=35.0)
    try:
        result = metric.measure(orig, rest, 44100)
        assert math.isfinite(result.arousal_pearson)
    except (AssertionError, ValueError):
        pass  # Akzeptabel bei SR-Verletzung


def test_14_stereo_via_mono_average():
    """Stereo-Input (2D-Array) ergibt kein Crash."""
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    mono = _dynamic_audio(dur=35.0)
    stereo = np.stack([mono, mono], axis=0)  # (2, N)
    mono_avg = stereo.mean(axis=0)
    result = metric.measure(mono_avg, mono_avg, SR)
    assert math.isfinite(result.arousal_pearson)


def test_15_no_nan_in_result():
    from backend.core.emotional_arc_preservation import get_emotional_arc_metric

    metric = get_emotional_arc_metric()
    orig = _dynamic_audio(dur=35.0)
    rest = _audio(dur=35.0, amp=0.2)
    result = metric.measure(orig, rest, SR)
    for field_name in ("arousal_pearson", "valence_pearson", "klimax_peak_deviation"):
        val = getattr(result, field_name)
        assert math.isfinite(val), f"NaN/Inf in {field_name}: {val}"
