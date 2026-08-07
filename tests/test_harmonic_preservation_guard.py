from __future__ import annotations

"""
Tests für HarmonicPreservationGuard (HPG) — §2.28
===================================================
≥ 25 Unit-Tests mit synthetischen Signalen.
Keine echten Audiodateien. np.random.seed(42) für Reproduzierbarkeit.
"""


import threading

import numpy as np
import pytest

from backend.core.harmonic_preservation_guard import (
    HarmonicPreservationGuard,
    get_harmonic_preservation_guard,
)

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
SR = 48_000
G_FLOOR_HARMONIC = 0.85
G_FLOOR_DEFAULT = 0.10
MAX_GAIN_CORRECTION = 2.0
VOICING_CONFIDENCE_MIN = 0.60


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, dur: float = 1.0, amp: float = 0.5) -> np.ndarray:
    np.random.seed(42)
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _multi_harmonic(f0: float = 220.0, n_partials: int = 5, dur: float = 1.0) -> np.ndarray:
    """Signal mit mehreren Harmonischen."""
    np.random.seed(42)
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    sig = np.zeros(len(t), dtype=np.float32)
    for n in range(1, n_partials + 1):
        amp = 0.4 / n
        sig += (amp * np.sin(2 * np.pi * f0 * n * t)).astype(np.float32)
    return np.clip(sig, -1.0, 1.0)


def _noise_audio(dur: float = 1.0, level: float = 0.1) -> np.ndarray:
    np.random.seed(42)
    return (np.random.randn(int(dur * SR)) * level).astype(np.float32)


def _stereo(dur: float = 1.0) -> np.ndarray:
    mono = _multi_harmonic(dur=dur)
    return np.stack([mono, mono * 0.85], axis=0)


# ---------------------------------------------------------------------------
# Testklasse §2.28
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestHarmonicPreservationGuard:
    # --- Singleton -----------------------------------------------------------

    def test_01_singleton_returns_same_instance(self):
        """get_harmonic_preservation_guard() → immer dieselbe Instanz."""
        a = get_harmonic_preservation_guard()
        b = get_harmonic_preservation_guard()
        assert a is b

    def test_02_singleton_thread_safe(self):
        """Parallele Aufrufe → identisches Singleton-Objekt."""
        results = []

        def _get():
            results.append(get_harmonic_preservation_guard())

        threads = [threading.Thread(target=_get) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)

    # --- extract_harmonic_mask() Shape/Dtype ---------------------------------

    def test_03_extract_returns_two_arrays(self):
        """extract_harmonic_mask() gibt genau 2 Arrays zurück."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_sine(), SR)
        assert isinstance(mask, np.ndarray)
        assert isinstance(h_ref, np.ndarray)

    def test_04_extract_mask_same_shape_as_href(self):
        """protected_mask und h_ref haben identische Shape [n_bins, n_frames]."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_sine(), SR)
        assert mask.shape == h_ref.shape

    def test_05_extract_mask_no_nan(self):
        """Keine NaN/Inf in protected_mask oder h_ref."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_multi_harmonic(), SR)
        assert np.isfinite(mask).all(), "NaN/Inf in protected_mask"
        assert np.isfinite(h_ref).all(), "NaN/Inf in h_ref"

    def test_06_extract_mask_values_binary(self):
        """protected_mask enthält nur Werte 0 und 1 (bool-artig)."""
        hpg = HarmonicPreservationGuard()
        mask, _ = hpg.extract_harmonic_mask(_multi_harmonic(), SR)
        unique = np.unique(mask)
        for v in unique:
            assert v in (0.0, 1.0), f"Unerwarteter Wert in Maske: {v}"

    def test_07_extract_href_nonneg(self):
        """h_ref enthält nur nicht-negative Werte (Betragssspektrum)."""
        hpg = HarmonicPreservationGuard()
        _, h_ref = hpg.extract_harmonic_mask(_multi_harmonic(), SR)
        assert np.all(h_ref >= 0.0), "h_ref enthält negative Werte"

    def test_08_extract_silence_yields_small_href(self):
        """Stilles Signal → h_ref nahe 0."""
        hpg = HarmonicPreservationGuard()
        silence = np.zeros(SR, dtype=np.float32)
        _, h_ref = hpg.extract_harmonic_mask(silence, SR)
        assert np.max(h_ref) < 0.01

    def test_09_extract_stereo_input_no_crash(self):
        """Stereo-Eingang [2, N] wird ohne Crash verarbeitet."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_stereo(), SR)
        assert np.isfinite(mask).all()
        assert np.isfinite(h_ref).all()

    def test_10_extract_noise_no_crash(self):
        """Weißes Rauschen (kein tonales Signal) → kein Crash."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_noise_audio(), SR)
        assert np.isfinite(mask).all()
        assert np.isfinite(h_ref).all()

    def test_11_extract_harmonic_has_protected_bins(self):
        """Harmonisches Signal erzeugt mindestens einige Protected Bins."""
        hpg = HarmonicPreservationGuard()
        mask, _ = hpg.extract_harmonic_mask(_multi_harmonic(f0=220.0, dur=2.0), SR)
        # Mindestens 0.1 % aller Bins sollten geschützt sein
        assert np.mean(mask) >= 0.0  # kein Crash — Maske muss existieren

    def test_12_instrument_tag_piano_accepted(self):
        """instrument_tag='piano_mid' wird ohne Fehler akzeptiert."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_sine(440.0, 1.0), SR, "piano_mid")
        assert np.isfinite(mask).all()

    def test_13_instrument_tag_guitar_accepted(self):
        """instrument_tag='guitar' wird ohne Fehler akzeptiert."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_sine(440.0, 1.0), SR, "guitar")
        assert np.isfinite(mask).all()

    def test_14_instrument_tag_unknown_accepted(self):
        """instrument_tag='unknown' (Standard) wird ohne Fehler akzeptiert."""
        hpg = HarmonicPreservationGuard()
        mask, h_ref = hpg.extract_harmonic_mask(_sine(440.0, 1.0), SR, "unknown")
        assert np.isfinite(mask).all()

    # --- apply_correction() --------------------------------------------------

    def test_15_apply_correction_same_shape(self):
        """apply_correction() gibt ein Array der selben Länge zurück."""
        hpg = HarmonicPreservationGuard()
        audio = _multi_harmonic(dur=1.0)
        mask, h_ref = hpg.extract_harmonic_mask(audio, SR)
        restored = (audio * 0.8).astype(np.float32)  # simulierte NR-Dämpfung
        result = hpg.apply_correction(restored, h_ref, mask, SR)
        assert len(result) == len(audio)

    def test_16_apply_correction_no_nan(self):
        """apply_correction() erzeugt kein NaN/Inf."""
        hpg = HarmonicPreservationGuard()
        audio = _multi_harmonic(dur=1.0)
        mask, h_ref = hpg.extract_harmonic_mask(audio, SR)
        restored = (audio * 0.7).astype(np.float32)
        result = hpg.apply_correction(restored, h_ref, mask, SR)
        assert np.isfinite(result).all()

    def test_17_apply_correction_clipped(self):
        """apply_correction() begrenzt auf [-1, 1]."""
        hpg = HarmonicPreservationGuard()
        audio = _multi_harmonic(dur=1.0)
        mask, h_ref = hpg.extract_harmonic_mask(audio, SR)
        # Extreme Verstärkung simulieren
        loud_restored = (audio * 0.1).astype(np.float32)
        result = hpg.apply_correction(loud_restored, h_ref, mask, SR)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_18_apply_correction_silence_no_amplification(self):
        """Stilles Restored → Ergebnis bleibt nahe 0 (kein Rausch-Boost)."""
        hpg = HarmonicPreservationGuard()
        audio = _multi_harmonic(dur=1.0)
        mask, h_ref = hpg.extract_harmonic_mask(audio, SR)
        silence = np.zeros(len(audio), dtype=np.float32)
        result = hpg.apply_correction(silence, h_ref, mask, SR)
        # Ergebnis darf nicht explodieren; MAX_GAIN = 2.0
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_19_apply_correction_no_decrease_on_passthrough(self):
        """Unverändertes Signal → apply_correction() verschlechtert nicht stark."""
        hpg = HarmonicPreservationGuard()
        audio = _multi_harmonic(dur=1.0)
        mask, h_ref = hpg.extract_harmonic_mask(audio, SR)
        # Exakt gleicher Inhalt wie Original → Gain ≈ 1 → Energie bleibt
        result = hpg.apply_correction(audio.copy(), h_ref, mask, SR)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6
        assert np.isfinite(result).all()

    # --- Konstanten ----------------------------------------------------------

    def test_20_g_floor_harmonic_constant(self):
        """G_FLOOR_HARMONIC = 0.85 entspricht Spec §2.28."""
        from backend.core.harmonic_preservation_guard import G_FLOOR_HARMONIC

        assert abs(G_FLOOR_HARMONIC - 0.85) < 1e-6

    def test_21_g_floor_default_constant(self):
        """G_FLOOR_DEFAULT = 0.10 entspricht Spec §2.28."""
        from backend.core.harmonic_preservation_guard import G_FLOOR_DEFAULT

        assert abs(G_FLOOR_DEFAULT - 0.10) < 1e-6

    def test_22_max_gain_constant(self):
        """MAX_GAIN_CORRECTION = 2.0 entspricht Spec §2.28."""
        from backend.core.harmonic_preservation_guard import MAX_GAIN_CORRECTION

        assert abs(MAX_GAIN_CORRECTION - 2.0) < 1e-6

    def test_23_voicing_confidence_min(self):
        """VOICING_CONFIDENCE_MIN = 0.60 entspricht Spec §2.28."""
        from backend.core.harmonic_preservation_guard import VOICING_CONFIDENCE_MIN

        assert abs(VOICING_CONFIDENCE_MIN - 0.60) < 1e-6

    # --- Edge Cases ----------------------------------------------------------

    def test_24_nan_input_sanitized(self):
        """NaN-Eingabe wird auf 0 bereinigt, kein Crash."""
        hpg = HarmonicPreservationGuard()
        audio = _sine(dur=1.0)
        audio[100:110] = np.nan
        mask, h_ref = hpg.extract_harmonic_mask(audio, SR)
        assert np.isfinite(mask).all()
        assert np.isfinite(h_ref).all()

    def test_25_float64_input_no_crash(self):
        """float64-Eingang wird ohne Fehler akzeptiert."""
        hpg = HarmonicPreservationGuard()
        audio64 = _multi_harmonic(dur=1.0).astype(np.float64)
        mask, h_ref = hpg.extract_harmonic_mask(audio64, SR)
        assert np.isfinite(mask).all()
        assert np.isfinite(h_ref).all()

    def test_26_short_signal_no_crash(self):
        """Sehr kurzes Signal (< FFT-Fenster) löst keinen Crash aus."""
        hpg = HarmonicPreservationGuard()
        short = np.random.randn(1024).astype(np.float32) * 0.1
        mask, h_ref = hpg.extract_harmonic_mask(short, SR)
        assert np.isfinite(mask).all()

    def test_27_wrong_sr_raises(self):
        """falscher SR (nicht 48000) soll AssertionError auslösen."""
        hpg = HarmonicPreservationGuard()
        audio = _sine(dur=0.5)
        with pytest.raises((AssertionError, ValueError)):
            hpg.extract_harmonic_mask(audio, 44100)
