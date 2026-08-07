import pytest

"""
tests/unit/test_progressive_quality_mode.py — ProgressiveQualityMode Test-Suite (≥ 15 Tests)
Alle Tests synthetisch, kein Datei-I/O, SR = 48_000.
"""

import math

import numpy as np

SR = 48_000
np.random.seed(42)


# ─── Hilfsfunktionen ──────────────────────────────────────────────────────────


def _audio(dur: float = 3.0, amp: float = 0.3, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _silence(dur: float = 3.0) -> np.ndarray:
    return np.zeros(int(dur * SR), dtype=np.float32)


# ─── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_00_import():
    from backend.core.progressive_quality_mode import (
        PreviewResult,
        ProgressiveQualityMode,
    )

    assert ProgressiveQualityMode is not None
    assert PreviewResult is not None


def test_01_singleton_identity():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    a = get_progressive_quality_mode()
    b = get_progressive_quality_mode()
    assert a is b


def test_02_thread_safe():
    import concurrent.futures

    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_progressive_quality_mode) for _ in range(10)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_03_run_preview_returns_result():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert result is not None


def test_04_preview_result_has_audio():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert hasattr(result, "preview_audio")
    assert isinstance(result.preview_audio, np.ndarray)


def test_05_preview_audio_no_nan():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert np.isfinite(result.preview_audio).all()


def test_06_preview_audio_not_clipped():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert np.max(np.abs(result.preview_audio)) <= 1.0


def test_07_preview_mos_finite():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert hasattr(result, "preview_mos")
    assert math.isfinite(result.preview_mos)


def test_08_preview_mos_range():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert 1.0 <= result.preview_mos <= 5.0


def test_09_preview_defects_is_list():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert hasattr(result, "detected_defects")
    assert isinstance(result.detected_defects, list)


def test_10_silence_no_crash():
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _silence(dur=6.0)
    result = pqm.run_preview(audio, SR)
    assert result is not None
    assert math.isfinite(result.preview_mos)


def test_11_attributes():
    from backend.core.progressive_quality_mode import ProgressiveQualityMode

    assert hasattr(ProgressiveQualityMode, "PREVIEW_DURATION_S")
    assert hasattr(ProgressiveQualityMode, "MAX_STAGE1_COMPUTE_S")
    assert ProgressiveQualityMode.PREVIEW_DURATION_S > 0.0


def test_12_convenience_function():
    from backend.core.progressive_quality_mode import run_preview

    audio = _audio(dur=6.0)
    result = run_preview(audio, SR)
    assert result is not None
    assert math.isfinite(result.preview_mos)


def test_13_sr_check():
    """Falscher SR sollte AssertionError oder graceful sein."""
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=6.0)
    try:
        result = pqm.run_preview(audio, 44100)
        assert result is not None
    except (AssertionError, ValueError):
        pass  # Akzeptabel


def test_14_different_durations():
    """Verschiedene Dateilängen verarbeiten ohne Absturz."""
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    for dur in (3.0, 5.0, 10.0, 30.0):
        audio = _audio(dur=dur)
        result = pqm.run_preview(audio, SR)
        assert math.isfinite(result.preview_mos), f"NaN bei dur={dur}"


def test_15_preview_audio_shape_plausible():
    """Vorschau-Audio sollte ≤ Eingabe-Länge sein."""
    from backend.core.progressive_quality_mode import get_progressive_quality_mode

    pqm = get_progressive_quality_mode()
    audio = _audio(dur=10.0)
    result = pqm.run_preview(audio, SR)
    # Vorschau-Audio darf nicht länger als das Original sein
    assert len(result.preview_audio) <= len(audio) + SR  # +1s Puffer erlaubt
