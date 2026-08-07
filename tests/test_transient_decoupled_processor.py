from __future__ import annotations

import pytest

"""
Tests für TransientDecoupledProcessing (TDP) — §2.27
=====================================================
≥ 20 Unit-Tests mit synthetischen Signalen.
Keine echten Audiodateien. np.random.seed(42) für Reproduzierbarkeit.
"""


import threading

import numpy as np

# ---------------------------------------------------------------------------
# Modul importieren
# ---------------------------------------------------------------------------
from backend.core.transient_decoupled_processor import (
    TransientDecoupledProcessor,
    get_transient_decoupled_processor,
    recombine_transients,
    separate_transients,
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen für synthetische Signale
# ---------------------------------------------------------------------------
SR = 48_000


def _sine(freq: float = 440.0, dur: float = 2.0, amp: float = 0.5) -> np.ndarray:
    """Reiner Sinuston (harmonischer Anteil)."""
    np.random.seed(42)
    t = np.linspace(0, dur, int(dur * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _impulse_train(dur: float = 2.0, rate_hz: float = 4.0, amp: float = 0.8) -> np.ndarray:
    """Reguläre Dirac-Impulse (perkussiver Anteil)."""
    sig = np.zeros(int(dur * SR), dtype=np.float32)
    period = int(SR / rate_hz)
    for i in range(0, len(sig), period):
        sig[i] = amp
    return sig


def _mixed(dur: float = 2.0) -> np.ndarray:
    """Gemischtes Signal: Sinus + Impulse + leichtes Rauschen."""
    np.random.seed(42)
    s = _sine(440.0, dur) + _impulse_train(dur) * 0.3
    s += np.random.randn(len(s)).astype(np.float32) * 0.02
    return np.clip(s, -1.0, 1.0)  # type: ignore[no-any-return]


def _stereo(dur: float = 2.0) -> np.ndarray:
    """Stereo-Signal [2, N]."""
    np.random.seed(42)
    mono = _mixed(dur)
    return np.stack([mono, mono * 0.9], axis=0)


# ---------------------------------------------------------------------------
# Testklasse §2.27
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestTransientDecoupledProcessor:
    # --- Singleton -----------------------------------------------------------

    def test_01_singleton_returns_same_instance(self):
        """get_transient_decoupled_processor() → immer dieselbe Instanz."""
        a = get_transient_decoupled_processor()
        b = get_transient_decoupled_processor()
        assert a is b

    def test_02_singleton_thread_safe(self):
        """Parallele Aufrufe liefern identisches Singleton-Objekt."""
        results = []

        def _get():
            results.append(get_transient_decoupled_processor())

        threads = [threading.Thread(target=_get) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)

    # --- separate() Outputs --------------------------------------------------

    def test_03_separate_returns_two_arrays(self):
        """separate() gibt genau 2 float32-Arrays zurück."""
        tdp = TransientDecoupledProcessor()
        mono = _sine(dur=1.0)
        perc, harm = tdp.separate(mono, SR)
        assert isinstance(perc, np.ndarray)
        assert isinstance(harm, np.ndarray)

    def test_04_separate_same_length_as_input(self):
        """|audio_p| und |audio_h| haben dieselbe Länge wie Eingang."""
        tdp = TransientDecoupledProcessor()
        mono = _mixed(dur=1.5)
        perc, harm = tdp.separate(mono, SR)
        assert len(perc) == len(mono)
        assert len(harm) == len(mono)

    def test_05_separate_output_no_nan(self):
        """Kein NaN/Inf in perkussivem oder harmonischem Anteil."""
        tdp = TransientDecoupledProcessor()
        mono = _mixed(dur=1.0)
        perc, harm = tdp.separate(mono, SR)
        assert np.isfinite(perc).all(), "NaN/Inf in audio_percussive"
        assert np.isfinite(harm).all(), "NaN/Inf in audio_harmonic"

    def test_06_separate_output_clipped(self):
        """Ausgaben sind auf [-1, 1] begrenzt."""
        tdp = TransientDecoupledProcessor()
        loud = (_mixed(dur=1.0) * 3.0).astype(np.float32)
        perc, harm = tdp.separate(loud, SR)
        assert np.max(np.abs(perc)) <= 1.0 + 1e-6
        assert np.max(np.abs(harm)) <= 1.0 + 1e-6

    def test_07_separate_stereo_input(self):
        """Stereo-Eingang wird ohne Crash verarbeitet; Ausgaben sind finite."""
        tdp = TransientDecoupledProcessor()
        stereo = _stereo(dur=1.0)
        perc, harm = tdp.separate(stereo, SR)
        # Modul kann intern zu Mono konvertieren — nur Crash-Freiheit und
        # Finite-Invariante prüfen (Shape-Transformation ist implementierungsdef.)
        assert isinstance(perc, np.ndarray) and np.all(np.isfinite(perc))
        assert isinstance(harm, np.ndarray) and np.all(np.isfinite(harm))

    def test_08_separate_silence_yields_small_output(self):
        """Stilles Signal → beide Ausgaben nahe 0."""
        tdp = TransientDecoupledProcessor()
        silence = np.zeros(SR, dtype=np.float32)
        perc, harm = tdp.separate(silence, SR)
        assert np.max(np.abs(perc)) < 0.01
        assert np.max(np.abs(harm)) < 0.01

    def test_09_separate_dc_offset_no_crash(self):
        """DC-Offset-Signal löst keinen Absturz aus."""
        tdp = TransientDecoupledProcessor()
        dc = np.full(SR, 0.5, dtype=np.float32)
        perc, harm = tdp.separate(dc, SR)
        assert np.isfinite(perc).all()
        assert np.isfinite(harm).all()

    def test_10_separate_impulse_no_crash(self):
        """Impuls-Zug trennen läuft fehlerfrei; beide Ausgaben finite."""
        tdp = TransientDecoupledProcessor()
        sig = _impulse_train(dur=1.0, rate_hz=4.0)
        perc, harm = tdp.separate(sig, SR)
        # HPSS-Verhalten bei kurzen Impulsen ist implementierungsabhängig;
        # Mindest-Anforderung ist NaN-Freiheit und Finite-Ausgabe.
        assert np.all(np.isfinite(perc)), "NaN/Inf im perkussiven Anteil"
        assert np.all(np.isfinite(harm)), "NaN/Inf im harmonischen Anteil"
        # Gesamtenergie der Trennung bleibt endlich
        assert np.sum(perc**2) + np.sum(harm**2) < 1e6

    # --- recombine() Outputs --------------------------------------------------

    def test_11_recombine_no_nan(self):
        """`recombine()` erzeugt kein NaN/Inf."""
        tdp = TransientDecoupledProcessor()
        mono = _mixed(dur=1.0)
        perc, harm = tdp.separate(mono, SR)
        result = tdp.recombine(perc, harm, SR)
        assert np.isfinite(result).all()

    def test_12_recombine_clipped(self):
        """`recombine()` begrenzt auf [-1, 1]."""
        tdp = TransientDecoupledProcessor()
        perc = np.ones(SR, dtype=np.float32)
        harm = np.ones(SR, dtype=np.float32)
        result = tdp.recombine(perc, harm, SR)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_13_recombine_same_length(self):
        """`recombine()` hat dieselbe Länge wie Eingang."""
        tdp = TransientDecoupledProcessor()
        mono = _mixed(dur=1.5)
        perc, harm = tdp.separate(mono, SR)
        result = tdp.recombine(perc, harm, SR)
        assert len(result) == len(mono)

    def test_14_recombine_passthrough_energy(self):
        """Roundtrip-Energie liegt in vernünftigem Rahmen (nicht auf 0)."""
        tdp = TransientDecoupledProcessor()
        mono = _mixed(dur=1.0)
        in_energy = float(np.sum(mono**2))
        perc, harm = tdp.separate(mono, SR)
        result = tdp.recombine(perc, harm, SR)
        out_energy = float(np.sum(result**2))
        assert out_energy > in_energy * 0.01, "Energie zu stark reduziert"

    # --- Convenience-Funktionen -----------------------------------------------

    def test_15_separate_transients_convenience(self):
        """`separate_transients()` ist ein funktionierender Wrapper."""
        mono = _sine(dur=1.0)
        perc, harm = separate_transients(mono, SR)
        assert isinstance(perc, np.ndarray)
        assert isinstance(harm, np.ndarray)
        assert np.isfinite(perc).all()
        assert np.isfinite(harm).all()

    def test_16_recombine_transients_convenience(self):
        """`recombine_transients()` ist ein funktionierender Wrapper."""
        mono = _mixed(dur=1.0)
        perc, harm = separate_transients(mono, SR)
        result = recombine_transients(perc, harm, SR)
        assert isinstance(result, np.ndarray)
        assert np.isfinite(result).all()

    # --- Konsistenz -----------------------------------------------------------

    def test_17_deterministic_same_input(self):
        """Gleicher Eingang → gleicher Ausgang (deterministisch)."""
        tdp = TransientDecoupledProcessor()
        mono = _mixed(dur=1.0)
        p1, h1 = tdp.separate(mono, SR)
        p2, h2 = tdp.separate(mono.copy(), SR)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(h1, h2)

    def test_18_short_signal_no_crash(self):
        """Sehr kurzes Signal (< Fensterbreite) löst keinen Crash aus."""
        tdp = TransientDecoupledProcessor()
        short = np.random.randn(512).astype(np.float32) * 0.1
        perc, harm = tdp.separate(short, SR)
        assert np.isfinite(perc).all()
        assert np.isfinite(harm).all()

    def test_19_white_noise_no_crash(self):
        """Weißes Rauschen löst keinen Crash aus."""
        np.random.seed(42)
        tdp = TransientDecoupledProcessor()
        noise = np.random.randn(SR * 2).astype(np.float32) * 0.3
        perc, harm = tdp.separate(noise, SR)
        result = tdp.recombine(perc, harm, SR)
        assert np.isfinite(result).all()

    def test_20_float64_input_accepted(self):
        """float64-Eingang wird ohne Fehler akzeptiert."""
        tdp = TransientDecoupledProcessor()
        mono64 = _mixed(dur=1.0).astype(np.float64)
        perc, harm = tdp.separate(mono64, SR)
        assert np.isfinite(perc).all()
        assert np.isfinite(harm).all()

    def test_21_percussive_only_phases_constant(self):
        """PERCUSSIVE_ONLY_PHASES enthält phase_01 und phase_27."""
        from backend.core.transient_decoupled_processor import PERCUSSIVE_ONLY_PHASES

        assert "phase_01_click_removal" in PERCUSSIVE_ONLY_PHASES
        assert "phase_27_click_pop_removal" in PERCUSSIVE_ONLY_PHASES

    def test_22_hpss_kernel_constants(self):
        """HPSS-Kernel-Konstanten entsprechen Spec (v10.0.0: Harmonic=17, Percussive=13)."""
        from backend.core.transient_decoupled_processor import (
            HPSS_HARMONIC_KERNEL,
            HPSS_PERCUSSIVE_KERNEL,
        )

        assert HPSS_HARMONIC_KERNEL == 17
        assert HPSS_PERCUSSIVE_KERNEL == 13
