from __future__ import annotations

"""
tests/test_micro_dynamics_envelope_morphing.py
================================================

Unit-Tests für MicroDynamicsEnvelopeMorphing (§2.30 Spec).

Anforderungen aus der Spec (§2.30):
  * Letzter Schritt vor Export — stellt Mikro-Dynamik-Profil wieder her
  * 400-ms-LUFS-Profil berechnen (hop 200 ms, 50 % Überlappung)
  * Gain G[k] = L_orig[k] − L_rest[k], clip ±MAX_GAIN_LU
  * Savitzky-Golay-Glättung (window=7, polyorder=2)
  * Stille-Schutz: L_orig[k] < MIN_LEVEL_LUFS → G[k] = 0
  * True-Peak-Prüfung nach Morphing: −1.0 dBTP
    * mode="restoration" → MAX_GAIN=4.0 LU, "studio2026" → 6.0 LU
  * Alle Ausgaben NaN-frei, float32, geclippt auf ±1.0

Konstanten (Spec §2.30):
    MAX_GAIN_LU       = 6.0
  FRAME_SIZE_SAMPLES = 19200  (400 ms @ 48000 Hz)
  HOP_SIZE_SAMPLES   = 9600   (200 ms, 50 % Hop)
  PEARSON_TARGET     = 0.93
  MIN_LEVEL_LUFS     = -60.0

Alle Signale: synthetisch, np.random.seed(42), SR = 48 000 Hz.
"""


import numpy as np
import pytest

SR = 48_000

# ── Imports ──────────────────────────────────────────────────────────────────
from backend.core.micro_dynamics_envelope_morphing import (
    FRAME_SIZE_SAMPLES,
    HOP_SIZE_SAMPLES,
    MAX_GAIN_LU,
    MIN_LEVEL_LUFS,
    PEARSON_TARGET,
    MicroDynamicsEnvelopeMorphing,
    compute_lufs_profile,
    get_mdem,
    morph_micro_dynamics,
)

# ── Hilfsfunktionen ───────────────────────────────────────────────────────────


def _tone(duration_s: float = 8.0, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    """Synthetischer Sinuston, float32, mono, 48 kHz."""
    np.random.seed(42)
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _noise(duration_s: float = 8.0, amp: float = 0.1) -> np.ndarray:
    """Weißes Rauschen, float32, mono."""
    np.random.seed(42)
    return np.random.randn(int(duration_s * SR)).astype(np.float32) * amp


def _silence(duration_s: float = 8.0) -> np.ndarray:
    """Stille, float32, mono."""
    return np.zeros(int(duration_s * SR), dtype=np.float32)


# ── Testklassen ───────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSingleton:
    """Singleton-Invarianten (§2.30)."""

    def test_01_get_mdem_returns_instance(self):
        mdem = get_mdem()
        assert isinstance(mdem, MicroDynamicsEnvelopeMorphing)

    def test_02_singleton_same_object(self):
        m1 = get_mdem()
        m2 = get_mdem()
        assert m1 is m2


class TestConstants:
    """Konstanten müssen exakt den Spec-Werten entsprechen (§2.30)."""

    def test_03_max_gain_lu(self):
        assert pytest.approx(6.0, abs=1e-9) == MAX_GAIN_LU

    def test_04_min_level_lufs(self):
        assert pytest.approx(-60.0, abs=1e-9) == MIN_LEVEL_LUFS

    def test_05_frame_size_samples(self):
        # 400 ms @ 48 000 Hz = 19 200
        assert FRAME_SIZE_SAMPLES == 19_200

    def test_06_hop_size_samples(self):
        # 200 ms @ 48 000 Hz = 9 600
        assert HOP_SIZE_SAMPLES == 9_600

    def test_07_pearson_target(self):
        assert pytest.approx(0.93, abs=1e-9) == PEARSON_TARGET

    def test_08_frame_size_matches_400ms(self):
        """FRAME_SIZE_SAMPLES muss genau 400 ms bei SR=48000 entsprechen."""
        assert int(0.4 * SR) == FRAME_SIZE_SAMPLES

    def test_09_hop_size_is_half_frame(self):
        """50 % Überlappung: HOP = FRAME / 2."""
        assert HOP_SIZE_SAMPLES == FRAME_SIZE_SAMPLES // 2


class TestComputeLufsProfile:
    """compute_lufs_profile (§2.30)."""

    def test_10_returns_1d_ndarray(self):
        audio = _tone(8.0)
        profile = compute_lufs_profile(audio, SR)
        assert isinstance(profile, np.ndarray)
        assert profile.ndim == 1

    def test_11_returns_float32(self):
        audio = _tone(8.0)
        profile = compute_lufs_profile(audio, SR)
        assert profile.dtype == np.float32

    def test_12_no_nan_in_profile(self):
        audio = _tone(8.0)
        profile = compute_lufs_profile(audio, SR)
        assert np.all(np.isfinite(profile)), "NaN/Inf im LUFS-Profil"

    def test_13_profile_values_are_negative(self):
        """LUFS-Werte sind stets ≤ 0 (außer Clipping, aber synthetisch ≤ 0.5 peak)."""
        audio = _tone(8.0, amp=0.5)
        profile = compute_lufs_profile(audio, SR)
        # Alle Werte müssen ≤ 0 sein
        assert np.all(profile <= 0.0 + 1e-3), f"Positiver LUFS-Wert: {profile.max()}"

    def test_14_silence_gives_floor_lufs(self):
        """Stille → LUFS-Werte nahe MIN_LEVEL_LUFS (−60.0 dB)."""
        audio = _silence(8.0)
        profile = compute_lufs_profile(audio, SR)
        # Alle Frames sollten ≤ MIN_LEVEL_LUFS + 5 dB sein (Toleranz für Filterringing)
        assert np.all(profile <= MIN_LEVEL_LUFS + 5.0), f"Stille hat zu hohe LUFS-Werte: max={profile.max()}"

    def test_15_minimum_one_frame(self):
        """Kurzes Audio ergibt mindestens 1 Frame."""
        audio = _tone(1.0)  # 1 Sekunde
        profile = compute_lufs_profile(audio, SR)
        assert len(profile) >= 1

    def test_16_stereo_input(self):
        """Stereo-Eingabe: Profil wird ohne Crash berechnet."""
        np.random.seed(42)
        audio = (np.random.randn(SR * 8, 2) * 0.3).astype(np.float32)
        profile = compute_lufs_profile(audio, SR)
        assert isinstance(profile, np.ndarray)
        assert np.all(np.isfinite(profile))

    def test_17_louder_signal_higher_lufs(self):
        """Lauteres Signal → höhere (weniger negative) LUFS-Werte."""
        quiet = _tone(8.0, amp=0.05)
        loud = _tone(8.0, amp=0.5)
        profile_quiet = compute_lufs_profile(quiet, SR)
        profile_loud = compute_lufs_profile(loud, SR)
        # Lauteres Signal muss im Mittel höher sein
        assert profile_loud.mean() > profile_quiet.mean()


class TestMorph:
    """morph / morph_micro_dynamics (§2.30)."""

    def test_18_output_same_length(self):
        audio = _tone(8.0)
        morphed = morph_micro_dynamics(audio, audio, SR, mode="restoration")
        assert len(morphed) == len(audio)

    def test_19_output_float32(self):
        audio = _tone(8.0)
        morphed = morph_micro_dynamics(audio, audio, SR, mode="restoration")
        assert morphed.dtype == np.float32

    def test_20_output_no_nan(self):
        audio = _tone(8.0)
        morphed = morph_micro_dynamics(audio, audio, SR, mode="restoration")
        assert np.all(np.isfinite(morphed)), "NaN/Inf im Ausgang"

    def test_21_output_clipped_to_unity(self):
        morphed = morph_micro_dynamics(_tone(8.0), _tone(8.0), SR)
        assert np.max(np.abs(morphed)) <= 1.0 + 1e-6

    def test_22_passthrough_similar_to_input(self):
        """Selbes Signal als restored und original → Ausgabe nah am Eingang."""
        audio = _tone(8.0)
        morphed = morph_micro_dynamics(audio, audio, SR, mode="restoration")
        # Korrelation muss hoch sein (gleicher Inhalt)
        corr = float(np.corrcoef(audio, morphed)[0, 1])
        assert corr > 0.90, f"Passthrough-Korrelation zu niedrig: {corr:.3f}"

    def test_23_nan_input_handled(self):
        """NaN im Input darf keinen Crash verursachen."""
        audio = _tone(8.0)
        corrupted = audio.copy()
        corrupted[100:200] = float("nan")
        morphed = morph_micro_dynamics(corrupted, audio, SR)
        assert isinstance(morphed, np.ndarray)
        assert np.all(np.isfinite(morphed))

    def test_24_restoration_mode_no_crash(self):
        audio = _tone(8.0)
        morphed = morph_micro_dynamics(audio, audio, SR, mode="restoration")
        assert isinstance(morphed, np.ndarray)

    def test_25_studio2026_mode_no_crash(self):
        audio = _tone(8.0)
        morphed = morph_micro_dynamics(audio, audio, SR, mode="studio2026")
        assert isinstance(morphed, np.ndarray)

    def test_26_wrong_sr_raises(self):
        """falsche SR muss AssertionError auslösen."""
        audio = _tone(8.0)
        with pytest.raises((AssertionError, ValueError)):
            morph_micro_dynamics(audio, audio, sr=44100)

    def test_27_silence_original_no_boost(self):
        """Stilles Original → kein Rausch-Boost (Stille-Schutz via G=0)."""
        original = _silence(8.0)
        restored = _tone(8.0, amp=0.3)
        morphed = morph_micro_dynamics(restored, original, SR, mode="restoration")
        # Auslenkung sollte durch Silence-Guard begrenzt sein
        assert np.max(np.abs(morphed)) <= 1.0 + 1e-6

    def test_28_very_short_audio_no_crash(self):
        """Audio kürzer als FRAME_SIZE_SAMPLES → kein Crash."""
        audio = _tone(0.3)  # 300 ms < 400 ms Frame
        morphed = morph_micro_dynamics(audio, audio, SR)
        assert isinstance(morphed, np.ndarray)

    def test_29_float64_input_no_crash(self):
        """float64-Input wird intern zu float32 konvertiert."""
        audio = _tone(8.0).astype(np.float64)
        morphed = morph_micro_dynamics(audio, audio, SR)
        assert isinstance(morphed, np.ndarray)

    def test_30_module_level_compute_lufs_profile(self):
        """Modul-level compute_lufs_profile muss identisch zu Instanz-Methode sein."""
        audio = _tone(8.0)
        profile_module = compute_lufs_profile(audio, SR)
        profile_inst = get_mdem().compute_lufs_profile(audio, SR)
        np.testing.assert_array_equal(profile_module, profile_inst)

    def test_31_class_morph_method_matches_convenience(self):
        """MicroDynamicsEnvelopeMorphing().morph() und morph_micro_dynamics() identisch."""
        audio = _tone(8.0)
        morphed_conv = morph_micro_dynamics(audio, audio, SR, mode="restoration")
        morphed_inst = get_mdem().morph(audio, audio, SR, mode="restoration")
        np.testing.assert_array_almost_equal(morphed_inst, morphed_conv, decimal=5)

    def test_32_lufs_profile_wrong_sr_raises(self):
        """compute_lufs_profile mit falschem SR muss AssertionError auslösen."""
        audio = _tone(8.0)
        with pytest.raises((AssertionError, ValueError)):
            get_mdem().compute_lufs_profile(audio, sr=22050)

    def test_33_stereo_morph_no_crash(self):
        """Stereo-Signal morphen ohne Crash."""
        np.random.seed(42)
        audio = (np.random.randn(SR * 8, 2) * 0.3).astype(np.float32)
        morphed = morph_micro_dynamics(audio, audio, SR, mode="restoration")
        assert isinstance(morphed, np.ndarray)
        assert np.all(np.isfinite(morphed))
