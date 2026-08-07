from __future__ import annotations

"""Unit-Tests für §SCK (V24) spectral_color_guard.py.

Testet check_spectral_color_preservation() und SpectralColorResult.
"""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48000
_N = 48000  # 1 s bei 48 kHz


def _make_sine(freq_hz: float, n: int = _N, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0.0, n / SR, n, endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]


def _make_noise(n: int = _N, amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(n)).astype(np.float32)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpectralColorGuardImport:
    def test_import_function(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        assert callable(check_spectral_color_preservation)

    def test_import_result_class(self):
        from backend.core.dsp.spectral_color_guard import SpectralColorResult

        assert SpectralColorResult is not None

    def test_import_threshold(self):
        from backend.core.dsp.spectral_color_guard import SPECTRAL_COLOR_THRESHOLD

        assert SPECTRAL_COLOR_THRESHOLD == 0.97


# ---------------------------------------------------------------------------
# SpectralColorResult Dataclass
# ---------------------------------------------------------------------------


class TestSpectralColorResult:
    def test_fields_exist(self):
        from backend.core.dsp.spectral_color_guard import SpectralColorResult

        r = SpectralColorResult(
            correlation=0.99,
            ok=True,
            pre_profile_db=[0.0] * 18,
            post_profile_db=[0.0] * 18,
        )
        assert r.correlation == 0.99
        assert r.ok is True
        assert len(r.pre_profile_db) == 18
        assert len(r.post_profile_db) == 18

    def test_ok_false_possible(self):
        from backend.core.dsp.spectral_color_guard import SpectralColorResult

        r = SpectralColorResult(
            correlation=0.50,
            ok=False,
            pre_profile_db=[],
            post_profile_db=[],
        )
        assert r.ok is False
        assert r.correlation < 0.97


# ---------------------------------------------------------------------------
# Hauptfunktion: identisches Signal → ok=True
# ---------------------------------------------------------------------------


class TestCheckSpectralColorIdentical:
    def test_identical_mono_ok(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_sine(440.0)
        result = check_spectral_color_preservation(audio, audio.copy(), SR)
        assert result.ok is True
        assert result.correlation >= 0.97

    def test_identical_stereo_ok(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        stereo = np.stack([_make_sine(440.0), _make_sine(880.0)], axis=0)
        result = check_spectral_color_preservation(stereo, stereo.copy(), SR)
        assert result.ok is True

    def test_correlation_near_one(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_noise()
        result = check_spectral_color_preservation(audio, audio, SR)
        assert result.correlation > 0.97


# ---------------------------------------------------------------------------
# Kleine Gain-Änderung → weiterhin ok (Spektralfarbe bleibt gleich)
# ---------------------------------------------------------------------------


class TestCheckSpectralColorSmallGain:
    def test_gain_095_still_ok(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_noise()
        post = (audio * 0.95).astype(np.float32)
        result = check_spectral_color_preservation(audio, post, SR)
        # Globaler Gain ändert nicht die Spektralfarbe (Shape identisch)
        assert result.ok is True

    def test_slight_tilt_still_ok(self):
        """Kleine Hochfrequenz-Absenkung sollte gerade noch ok sein."""
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_noise()
        # Sehr kleiner Spektralfarbe-Unterschied
        post = audio * 0.98 + _make_noise(amp=0.001)
        result = check_spectral_color_preservation(audio, post, SR)
        # Hohe Ähnlichkeit → ok
        assert result.ok is True


# ---------------------------------------------------------------------------
# Drastische Spektralfarbe-Änderung → ok=False
# ---------------------------------------------------------------------------


class TestCheckSpectralColorDrastisch:
    def test_bass_cut_fails(self):
        """Heavy Low-Cut → Spektralfarbe deutlich verschoben → ok=False."""
        from scipy.signal import butter, sosfiltfilt

        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_noise(amp=0.3)
        # High-Pass bei 4 kHz — entfernt fast alles unter 4 kHz
        sos = butter(8, 4000.0 / (SR / 2.0), btype="high", output="sos")
        post = sosfiltfilt(sos, audio).astype(np.float32)
        result = check_spectral_color_preservation(audio, post, SR)
        assert result.ok is False, "Drastische Spektralfarbverschiebung sollte ok=False liefern"
        assert result.correlation < 0.97


# ---------------------------------------------------------------------------
# Rückgabe-Felder sind vorhanden
# ---------------------------------------------------------------------------


class TestCheckSpectralColorReturnFields:
    def test_profile_lists_not_empty(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_noise()
        result = check_spectral_color_preservation(audio, audio.copy(), SR)
        assert isinstance(result.pre_profile_db, list)
        assert isinstance(result.post_profile_db, list)
        assert len(result.pre_profile_db) > 0
        assert len(result.post_profile_db) > 0

    def test_correlation_in_range(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_noise()
        result = check_spectral_color_preservation(audio, audio.copy(), SR)
        assert 0.0 <= result.correlation <= 1.001

    def test_sr_assert_48000(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        audio = _make_noise(n=22050)
        with pytest.raises(AssertionError):
            check_spectral_color_preservation(audio, audio.copy(), 44100)


# ---------------------------------------------------------------------------
# Randfall: Stille
# ---------------------------------------------------------------------------


class TestCheckSpectralColorSilence:
    def test_silence_fallback_ok(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        silence = np.zeros(_N, dtype=np.float32)
        result = check_spectral_color_preservation(silence, silence, SR)
        # Stille → Fallback → ok=True (kein Artefakt)
        assert result.ok is True

    def test_different_shapes_fallback(self):
        from backend.core.dsp.spectral_color_guard import check_spectral_color_preservation

        pre = _make_noise(n=_N)
        post = _make_noise(n=_N // 2)
        # Unterschiedliche Shapes → Fallback
        result = check_spectral_color_preservation(pre, post, SR)
        assert result.ok is True
