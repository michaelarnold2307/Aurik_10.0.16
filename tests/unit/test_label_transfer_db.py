"""Unit tests for §LTD-1 LabelTransferDB (backend/core/label_transfer_db.py).

Tests: singleton, lookup, EQ curve, apply_label_eq, edge cases.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq: float = 1000.0, sr: int = 48000, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_label_transfer_db_singleton():
    from backend.core.label_transfer_db import get_label_transfer_db

    db1 = get_label_transfer_db()
    db2 = get_label_transfer_db()
    assert db1 is db2


def test_label_transfer_db_list_all_not_empty():
    from backend.core.label_transfer_db import get_label_transfer_db

    db = get_label_transfer_db()
    profiles = db.list_all()
    assert len(profiles) >= 10


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_lookup_vinyl_1963():
    from backend.core.label_transfer_db import get_label_transfer_db

    r = get_label_transfer_db().lookup(1963, "vinyl")
    assert r.profile is not None
    assert r.confidence > 0.5
    assert r.label_id != ""


def test_lookup_shellac_1930():
    from backend.core.label_transfer_db import get_label_transfer_db

    r = get_label_transfer_db().lookup(1930, "shellac")
    assert r.profile is not None
    assert r.confidence > 0.5


def test_lookup_no_match_far_future():
    from backend.core.label_transfer_db import get_label_transfer_db

    r = get_label_transfer_db().lookup(2050, "cd_digital")
    # No profiles cover 2050 → confidence 0.0
    assert r.confidence == 0.0
    assert r.profile is None


def test_lookup_label_hint_boost():
    from backend.core.label_transfer_db import get_label_transfer_db

    db = get_label_transfer_db()
    r_no_hint = db.lookup(1960, "vinyl")
    r_with_hint = db.lookup(1960, "vinyl", label_hint="decca uk")
    # With matching hint, confidence should be ≥ without hint
    assert r_with_hint.confidence >= r_no_hint.confidence - 0.01


def test_lookup_acoustic_era():
    from backend.core.label_transfer_db import get_label_transfer_db

    r = get_label_transfer_db().lookup(1910, "shellac")
    assert r.profile is not None
    assert "acoustic" in r.label_id.lower()


def test_lookup_preemphasis_field_set():
    from backend.core.label_transfer_db import get_label_transfer_db

    r = get_label_transfer_db().lookup(1960, "vinyl")
    assert r.profile is not None
    assert r.profile.preemphasis_correction in {
        "riaa_1954",
        "nab_1948",
        "columbia_eq",
        "decca_eq",
        "none",
    }


# ---------------------------------------------------------------------------
# EQ curve
# ---------------------------------------------------------------------------


def test_get_eq_curve_shape():
    from backend.core.label_transfer_db import get_label_transfer_db

    freq_bins = np.linspace(20, 22050, 512, dtype=np.float32)
    eq = get_label_transfer_db().get_eq_curve(1960, "vinyl", freq_bins)
    assert eq.shape == (512,)
    assert eq.dtype == np.float32


def test_get_eq_curve_all_positive():
    from backend.core.label_transfer_db import get_label_transfer_db

    freq_bins = np.array([100, 500, 1000, 5000, 10000], dtype=np.float32)
    eq = get_label_transfer_db().get_eq_curve(1960, "vinyl", freq_bins)
    assert np.all(eq > 0.0)


def test_get_eq_curve_no_match_returns_ones():
    from backend.core.label_transfer_db import get_label_transfer_db

    freq_bins = np.array([100, 1000, 10000], dtype=np.float32)
    eq = get_label_transfer_db().get_eq_curve(2050, "cd_digital", freq_bins)
    np.testing.assert_allclose(eq, np.ones(3, dtype=np.float32), rtol=1e-5)


# ---------------------------------------------------------------------------
# apply_label_eq
# ---------------------------------------------------------------------------


def test_apply_label_eq_mono_shape_preserved():
    from backend.core.label_transfer_db import get_label_transfer_db

    audio = _sine()
    out = get_label_transfer_db().apply_label_eq(audio, 48000, 1960, "vinyl")
    assert out.shape == audio.shape
    assert out.dtype == np.float32


def test_apply_label_eq_stereo_shape_preserved():
    from backend.core.label_transfer_db import get_label_transfer_db

    audio = np.stack([_sine(), _sine(500)], axis=-1)
    out = get_label_transfer_db().apply_label_eq(audio, 48000, 1960, "vinyl")
    assert out.shape == audio.shape


def test_apply_label_eq_strength_zero_passthrough():
    from backend.core.label_transfer_db import get_label_transfer_db

    audio = _sine()
    out = get_label_transfer_db().apply_label_eq(audio, 48000, 1960, "vinyl", strength=0.0)
    np.testing.assert_array_equal(out, audio)


def test_apply_label_eq_clip_guard():
    from backend.core.label_transfer_db import get_label_transfer_db

    audio = np.ones(4800, dtype=np.float32)  # DC = 1.0
    out = get_label_transfer_db().apply_label_eq(audio, 48000, 1960, "vinyl", strength=1.0)
    assert float(np.max(np.abs(out))) <= 1.0


def test_apply_label_eq_nan_safe():
    from backend.core.label_transfer_db import get_label_transfer_db

    audio = _sine()
    audio[100:110] = np.nan
    out = get_label_transfer_db().apply_label_eq(audio, 48000, 1960, "vinyl")
    assert not np.any(np.isnan(out))


def test_apply_label_eq_assert_sr():
    from backend.core.label_transfer_db import get_label_transfer_db

    audio = _sine()
    with pytest.raises(AssertionError):
        get_label_transfer_db().apply_label_eq(audio, 44100, 1960, "vinyl")


# ---------------------------------------------------------------------------
# LabelProfile helpers
# ---------------------------------------------------------------------------


def test_label_profile_interpolate_eq():
    from backend.core.label_transfer_db import LabelProfile

    profile = LabelProfile(
        label_id="test",
        display_name="Test",
        era_range=(1950, 1970),
        materials=("vinyl",),
        eq_breakpoints_hz_db=[(100, -3.0), (1000, 0.0), (10000, -6.0)],
    )
    # At breakpoints
    assert profile.interpolate_eq_gain_db(100.0) == pytest.approx(-3.0, abs=1e-5)
    assert profile.interpolate_eq_gain_db(1000.0) == pytest.approx(0.0, abs=1e-5)
    # Midpoint (linear interp)
    mid = profile.interpolate_eq_gain_db(550.0)
    assert -3.0 <= mid <= 0.0


def test_label_profile_eq_curve_output_shape():
    from backend.core.label_transfer_db import LabelProfile

    profile = LabelProfile(
        label_id="test",
        display_name="Test",
        era_range=(1950, 1970),
        materials=("vinyl",),
        eq_breakpoints_hz_db=[(100, -3.0), (10000, 0.0)],
    )
    freqs = np.linspace(100, 10000, 256, dtype=np.float32)
    curve = profile.get_eq_curve(freqs)
    assert curve.shape == (256,)
    assert np.all(curve > 0.0)
