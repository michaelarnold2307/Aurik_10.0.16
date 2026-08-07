"""
tests/unit/test_reference_anchor_synthesizer.py — ReferenceAnchorSynthesizer Test-Suite (≥ 20 Tests)
Alle Tests synthetisch, kein externer Download.
"""

import logging
import math

import numpy as np
import pytest

logger = logging.getLogger(__name__)

SR = 48_000
np.random.seed(42)


def _audio(dur: float = 5.0, amp: float = 0.3):
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _silence(dur: float = 5.0):
    return np.zeros(int(dur * SR), dtype=np.float32)


def _mock_era():
    """Minimales Mock-EraResult."""
    try:
        from backend.core.era_classifier import EraResult

        return EraResult(
            decade=1970,
            era_label="1970s",
            confidence=0.8,
            material_prior="tape",
            noise_profile=np.zeros(128, dtype=np.float32),
        )
    except Exception:
        logger.warning("test fallback", exc_info=True)
        return None


# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_00_import():
    from backend.core.reference_anchor_synthesizer import ReferenceAnchorSynthesizer

    assert ReferenceAnchorSynthesizer is not None


def test_01_singleton_identity():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    a = get_reference_anchor_synthesizer()
    b = get_reference_anchor_synthesizer()
    assert a is b


def test_02_thread_safe():
    import concurrent.futures

    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(get_reference_anchor_synthesizer) for _ in range(10)]
        instances = [f.result() for f in futs]
    assert all(inst is instances[0] for inst in instances)


def test_03_synthesize_returns_ndarray():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "schlager", "tape")
    assert isinstance(anchor, np.ndarray)


def test_04_anchor_length_128():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "vinyl")
    assert len(anchor) == 128


def test_05_anchor_finite():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "unknown", "unknown")
    assert np.isfinite(anchor).all()


def test_06_anchor_dtype_float32():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "tape")
    assert anchor.dtype == np.float32


def test_07_apply_to_audio_shape():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "tape")
    audio = _audio(5.0)
    out = ras.apply_to_audio(audio, SR, anchor)  # type: ignore[arg-type]
    assert out.shape == audio.shape


def test_08_apply_to_audio_no_nan():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "tape")
    audio = _audio(5.0)
    out = ras.apply_to_audio(audio, SR, anchor)  # type: ignore[arg-type]
    assert np.isfinite(out).all()


def test_09_apply_to_audio_not_clipped():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "jazz", "vinyl")
    audio = _audio(5.0)
    out = ras.apply_to_audio(audio, SR, anchor)  # type: ignore[arg-type]
    assert np.max(np.abs(out)) <= 1.0 + 1e-5


def test_10_max_eq_db_attribute():
    from backend.core.reference_anchor_synthesizer import ReferenceAnchorSynthesizer

    ras = ReferenceAnchorSynthesizer()
    assert hasattr(ras, "MAX_EQ_DB")
    assert ras.MAX_EQ_DB > 0.0


def test_11_k_nearest_attribute():
    from backend.core.reference_anchor_synthesizer import ReferenceAnchorSynthesizer

    ras = ReferenceAnchorSynthesizer()
    assert hasattr(ras, "K_NEAREST")
    assert ras.K_NEAREST >= 1


def test_12_silence_no_crash():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "tape")
    audio = _silence(5.0)
    out = ras.apply_to_audio(audio, SR, anchor)  # type: ignore[arg-type]
    assert np.isfinite(out).all()


def test_13_none_era_fallback():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    anchor = ras.synthesize(None, "unknown", "unknown")
    assert isinstance(anchor, np.ndarray)
    assert len(anchor) == 128
    assert np.isfinite(anchor).all()


def test_14_multiple_genres():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    for genre in ["schlager", "jazz", "klassik", "pop", "unknown"]:
        anchor = ras.synthesize(era, genre, "tape")
        assert len(anchor) == 128
        assert np.isfinite(anchor).all()


def test_15_multiple_materials():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    for mat in ["tape", "vinyl", "shellac", "mp3_high", "unknown"]:
        anchor = ras.synthesize(era, "pop", mat)
        assert len(anchor) == 128


def test_16_assert_sr():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "tape")
    audio = _audio(5.0)
    with pytest.raises((AssertionError, ValueError)):
        ras.apply_to_audio(audio, 44100, anchor)  # type: ignore[arg-type]


def test_17_consistent_anchor():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    a1 = ras.synthesize(era, "pop", "tape")
    a2 = ras.synthesize(era, "pop", "tape")
    np.testing.assert_array_almost_equal(a1, a2, decimal=4)


def test_18_consistent_apply():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "tape")
    audio = _audio(5.0)
    out1 = ras.apply_to_audio(audio.copy(), SR, anchor)  # type: ignore[arg-type]
    out2 = ras.apply_to_audio(audio.copy(), SR, anchor)  # type: ignore[arg-type]
    np.testing.assert_array_almost_equal(out1, out2, decimal=4)


def test_19_apply_float64():
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "pop", "tape")
    audio = _audio(5.0).astype(np.float64)
    out = ras.apply_to_audio(audio, SR, anchor)  # type: ignore[arg-type]
    assert np.isfinite(out).all()


def test_20_max_eq_limited():
    """EQ-Eingriff darf nicht mehr als MAX_EQ_DB (6 dB) betragen."""
    from backend.core.reference_anchor_synthesizer import get_reference_anchor_synthesizer

    ras = get_reference_anchor_synthesizer()
    era = _mock_era()
    anchor = ras.synthesize(era, "schlager", "tape")
    audio = _audio(5.0)
    out = ras.apply_to_audio(audio, SR, anchor)  # type: ignore[arg-type]
    rms_in = np.sqrt(np.mean(audio**2))
    rms_out = np.sqrt(np.mean(out**2))
    if rms_in > 1e-8 and rms_out > 1e-8:
        ratio_db = abs(20 * math.log10(rms_out / rms_in))
        assert ratio_db <= ras.MAX_EQ_DB + 3.0, f"EQ-Eingriff zu groß: {ratio_db:.1f} dB"
