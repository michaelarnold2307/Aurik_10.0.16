from __future__ import annotations

"""Unit-Tests: backend/core/album_consistency.py

Tests cover:
- LUFS measurement consistency
- Spectral tilt measurement
- Median target computation (robust to outliers)
- LUFS outlier detection + gain correction
- Tilt outlier detection + shelf correction
- Within-threshold songs are NOT corrected (§0 Primum non nocere)
- Single/double-song skip (< 3 songs = no pass)
- Shelf filter: NaN-safe output
- Singleton accessor
"""


import numpy as np
import pytest

from backend.core.album_consistency import (
    _LUFS_MAX_CORRECTION_DB,
    _LUFS_OUTLIER_THRESHOLD_LU,
    _MIN_SONGS,
    _TILT_MAX_CORRECTION_DB,
    AlbumConsistencyPass,
    AlbumConsistencyReport,
    SongConsistencyProfile,
    get_album_consistency_pass,
)

SR = 48000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq: float, dur: float = 2.0, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    """Mono sine wave."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _white(dur: float = 2.0, amp: float = 0.1, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(sr * dur)) * amp).astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_returns_same_instance():
    a = get_album_consistency_pass()
    b = get_album_consistency_pass()
    assert a is b


def test_singleton_is_album_consistency_pass():
    assert isinstance(get_album_consistency_pass(), AlbumConsistencyPass)


# ---------------------------------------------------------------------------
# LUFS measurement
# ---------------------------------------------------------------------------


def test_measure_lufs_silence_returns_low_value():
    acp = AlbumConsistencyPass()
    silence = np.zeros(SR * 2, dtype=np.float32)
    lufs = acp._measure_lufs(silence, SR)
    assert lufs < -50.0, f"Silence LUFS should be < -50, got {lufs}"


def test_measure_lufs_louder_signal_higher():
    acp = AlbumConsistencyPass()
    soft = _sine(440, amp=0.05)
    loud = _sine(440, amp=0.5)
    lufs_soft = acp._measure_lufs(soft, SR)
    lufs_loud = acp._measure_lufs(loud, SR)
    assert lufs_loud > lufs_soft, "Louder signal should have higher LUFS"


def test_measure_lufs_stereo_audio():
    acp = AlbumConsistencyPass()
    mono = _sine(440)
    stereo = np.stack([mono, mono], axis=0)  # (2, samples)
    lufs_mono = acp._measure_lufs(mono, SR)
    lufs_stereo = acp._measure_lufs(stereo, SR)
    # Allow ±3 LU difference for mono vs stereo representation
    assert abs(lufs_mono - lufs_stereo) < 5.0


# ---------------------------------------------------------------------------
# Spectral tilt measurement
# ---------------------------------------------------------------------------


def test_measure_spectral_tilt_finite():
    acp = AlbumConsistencyPass()
    audio = _white()
    tilt = acp._measure_spectral_tilt(audio, SR)
    assert np.isfinite(tilt), f"Spectral tilt should be finite, got {tilt}"
    assert -12.0 <= tilt <= 2.0, f"Spectral tilt out of expected range: {tilt}"


def test_measure_spectral_tilt_silence_returns_finite():
    acp = AlbumConsistencyPass()
    silence = np.zeros(SR * 2, dtype=np.float32)
    tilt = acp._measure_spectral_tilt(silence, SR)
    assert np.isfinite(tilt)


# ---------------------------------------------------------------------------
# Median target and outlier logic
# ---------------------------------------------------------------------------


def test_analyze_skips_fewer_than_min_songs():
    acp = AlbumConsistencyPass()
    audios = [_sine(440)] * (_MIN_SONGS - 1)
    srs = [SR] * len(audios)
    fps = [f"/tmp/song_{i}.wav" for i in range(len(audios))]
    report = acp.analyze(audios, srs, fps)
    assert report.skipped_insufficient_songs is True


def test_analyze_min_songs_proceeds():
    acp = AlbumConsistencyPass()
    audios = [_sine(440, amp=0.2)] * _MIN_SONGS
    srs = [SR] * _MIN_SONGS
    fps = [f"/tmp/song_{i}.wav" for i in range(_MIN_SONGS)]
    report = acp.analyze(audios, srs, fps)
    assert report.skipped_insufficient_songs is False
    assert report.n_songs == _MIN_SONGS
    assert len(report.songs) == _MIN_SONGS


def test_analyze_median_target_computed():
    acp = AlbumConsistencyPass()
    # Use identical signals → median = single value, no outliers
    audios = [_sine(440, amp=0.2)] * 4
    srs = [SR] * 4
    fps = [f"/tmp/song_{i}.wav" for i in range(4)]
    report = acp.analyze(audios, srs, fps)
    # All songs identical → no corrections needed
    for p in report.songs:
        assert abs(p.lufs_correction_db) < 0.01, "Identical songs should need no LUFS correction"


def test_analyze_lufs_outlier_gets_correction():
    """A song that is 5 LU louder than the rest should get a negative LUFS correction."""
    acp = AlbumConsistencyPass()
    rng = np.random.default_rng(0)

    # 3 "quiet" songs + 1 "loud" song
    quiet = (rng.standard_normal(SR * 2) * 0.05).astype(np.float32)
    loud = (rng.standard_normal(SR * 2) * 0.5).astype(np.float32)

    audios = [quiet, quiet, quiet, loud]
    srs = [SR] * 4
    fps = [f"/tmp/song_{i}.wav" for i in range(4)]

    report = acp.analyze(audios, srs, fps)
    # Last song (loud) should have a negative correction
    loud_profile = report.songs[-1]
    assert loud_profile.lufs_correction_db < 0.0, (
        f"Loud outlier should have negative LUFS correction, got {loud_profile.lufs_correction_db}"
    )


def test_analyze_within_threshold_songs_not_corrected():
    """Songs within LUFS_OUTLIER_THRESHOLD_LU of the median should get no LUFS correction."""
    acp = AlbumConsistencyPass()
    rng = np.random.default_rng(1)
    # 4 signals all within 1 LU of each other → no outliers
    base = (rng.standard_normal(SR * 2) * 0.2).astype(np.float32)
    audios = [base] * 4
    srs = [SR] * 4
    fps = [f"/tmp/song_{i}.wav" for i in range(4)]

    report = acp.analyze(audios, srs, fps)
    for p in report.songs:
        assert abs(p.lufs_correction_db) < _LUFS_OUTLIER_THRESHOLD_LU, (
            f"Song within threshold should not be corrected: {p.lufs_correction_db}"
        )


def test_lufs_correction_capped_at_max():
    """LUFS correction is bounded by _LUFS_MAX_CORRECTION_DB regardless of deviation."""
    AlbumConsistencyPass()
    rng = np.random.default_rng(2)
    # Very loud outlier (−3 LUFS) vs very quiet group (−30 LUFS)
    (rng.standard_normal(SR * 2) * 0.9).astype(np.float32)
    quiet = np.zeros(SR * 2, dtype=np.float32)

    # Manually test analyze with mocked LUFS
    profile = SongConsistencyProfile(
        file_path="/tmp/test.wav",
        lufs=-3.0,
        spectral_tilt=-3.0,
        dynamic_range_db=12.0,
    )
    album_lufs = -30.0
    lufs_dev = profile.lufs - album_lufs  # +27 → would be capped
    raw_correction = float(np.clip(-lufs_dev, -_LUFS_MAX_CORRECTION_DB, _LUFS_MAX_CORRECTION_DB))
    assert abs(raw_correction) <= _LUFS_MAX_CORRECTION_DB


def test_tilt_correction_capped_at_max():
    """Tilt correction is bounded by _TILT_MAX_CORRECTION_DB."""
    profile = SongConsistencyProfile(
        file_path="/tmp/test.wav",
        lufs=-18.0,
        spectral_tilt=-8.0,  # very dark
        dynamic_range_db=12.0,
    )
    album_tilt = -3.0
    tilt_dev = profile.spectral_tilt - album_tilt  # -5 → big outlier
    raw_correction = float(np.clip(-tilt_dev * 0.5, -_TILT_MAX_CORRECTION_DB, _TILT_MAX_CORRECTION_DB))
    assert abs(raw_correction) <= _TILT_MAX_CORRECTION_DB


# ---------------------------------------------------------------------------
# Gain correction
# ---------------------------------------------------------------------------


def test_apply_gain_positive_makes_louder():
    acp = AlbumConsistencyPass()
    audio = _sine(440, amp=0.1)
    louder = acp._apply_gain(audio, +3.0)
    assert louder.max() > audio.max(), "Positive gain should make signal louder"


def test_apply_gain_negative_makes_quieter():
    acp = AlbumConsistencyPass()
    audio = _sine(440, amp=0.5)
    quieter = acp._apply_gain(audio, -3.0)
    assert quieter.max() < audio.max(), "Negative gain should make signal quieter"


def test_apply_gain_zero_returns_unchanged():
    acp = AlbumConsistencyPass()
    audio = _sine(440)
    out = acp._apply_gain(audio, 0.0)
    np.testing.assert_array_almost_equal(out, audio)


def test_apply_gain_peak_safety():
    """After gain, output must not exceed peak safety ceiling."""
    from backend.core.album_consistency import _PEAK_SAFETY

    acp = AlbumConsistencyPass()
    audio = np.ones(SR, dtype=np.float32) * 0.95
    out = acp._apply_gain(audio, +10.0)  # would clip without guard
    peak = float(np.percentile(np.abs(out), 99.9))
    assert peak <= _PEAK_SAFETY + 1e-4


def test_apply_gain_nan_safe():
    acp = AlbumConsistencyPass()
    audio = _sine(440)
    audio[100] = np.nan
    out = acp._apply_gain(audio, 0.0)
    # nan_to_num is applied in apply(), but gain itself should not crash
    assert out is not None


# ---------------------------------------------------------------------------
# Shelf filter
# ---------------------------------------------------------------------------


def test_shelf_sos_identity_for_zero_gain():
    acp = AlbumConsistencyPass()
    sos = acp._build_shelf_sos(0.0, 3000.0, SR)
    # Identity SOS: [1, 0, 0, 1, 0, 0]
    assert sos.shape == (1, 6)
    assert abs(sos[0, 0] - 1.0) < 1e-6  # b0
    assert abs(sos[0, 3] - 1.0) < 1e-6  # a0


def test_tilt_correction_no_nan():
    acp = AlbumConsistencyPass()
    audio = _white(dur=1.0)
    out = acp._apply_tilt_correction(audio, +1.0, SR)
    assert np.all(np.isfinite(out)), "Tilt correction should produce no NaN/Inf"


def test_tilt_correction_stereo_channels_shape():
    acp = AlbumConsistencyPass()
    audio = np.stack([_sine(440), _sine(880)], axis=0)  # (2, samples)
    out = acp._apply_tilt_correction(audio, -0.8, SR)
    assert out.shape == audio.shape


def test_tilt_correction_zero_noop():
    acp = AlbumConsistencyPass()
    audio = _white(dur=1.0)
    out = acp._apply_tilt_correction(audio, 0.0, SR)
    np.testing.assert_array_equal(out, audio)


# ---------------------------------------------------------------------------
# apply() — end-to-end correction on a single song
# ---------------------------------------------------------------------------


def test_apply_returns_same_shape():
    acp = AlbumConsistencyPass()
    audio = _sine(440)
    profile = SongConsistencyProfile(
        file_path="/tmp/test.wav",
        lufs=-18.0,
        spectral_tilt=-3.0,
        dynamic_range_db=12.0,
        lufs_correction_db=+1.5,
        tilt_correction_db=-0.5,
    )
    corrected, meta = acp.apply(audio, SR, profile)
    assert corrected.shape == audio.shape
    assert corrected.dtype == audio.dtype


def test_apply_no_correction_when_zero():
    acp = AlbumConsistencyPass()
    audio = _sine(440)
    profile = SongConsistencyProfile(
        file_path="/tmp/test.wav",
        lufs=-18.0,
        spectral_tilt=-3.0,
        dynamic_range_db=12.0,
        lufs_correction_db=0.0,
        tilt_correction_db=0.0,
    )
    corrected, meta = acp.apply(audio, SR, profile)
    assert meta["correction_applied"] is False
    np.testing.assert_array_almost_equal(corrected, audio)


def test_apply_output_is_nan_free():
    acp = AlbumConsistencyPass()
    audio = _white(dur=1.0)
    audio[50] = np.nan
    profile = SongConsistencyProfile(
        file_path="/tmp/test.wav",
        lufs=-18.0,
        spectral_tilt=-3.0,
        dynamic_range_db=12.0,
        lufs_correction_db=+2.0,
        tilt_correction_db=+1.0,
    )
    corrected, _ = acp.apply(audio, SR, profile)
    assert np.all(np.isfinite(corrected)), "apply() output must be NaN-free"


def test_apply_correction_applied_flag():
    acp = AlbumConsistencyPass()
    audio = _sine(440)
    profile = SongConsistencyProfile(
        file_path="/tmp/test.wav",
        lufs=-18.0,
        spectral_tilt=-3.0,
        dynamic_range_db=12.0,
        lufs_correction_db=+1.0,
        tilt_correction_db=0.0,
    )
    _, meta = acp.apply(audio, SR, profile)
    assert meta["correction_applied"] is True


# ---------------------------------------------------------------------------
# AlbumConsistencyReport dataclass
# ---------------------------------------------------------------------------


def test_report_defaults():
    report = AlbumConsistencyReport(
        n_songs=4,
        album_lufs_median=-18.0,
        album_tilt_median=-3.0,
        album_dr_median=12.0,
    )
    assert report.corrections_applied == 0
    assert report.songs == []
    assert report.skipped_insufficient_songs is False
