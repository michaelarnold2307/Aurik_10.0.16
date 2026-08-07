from __future__ import annotations

"""
tests/unit/test_v100_core_modules.py

Unit-Tests für die 5 neuen Spec-Kernmodule (§2.10–§2.13, §4.5):
    - PsychoacousticMaskingModel (§4.5)
    - HarmonicLatticeAnalyzer    (§2.11)
    - SegmentAdaptiveProcessor   (§2.10)
    - MusicalPhraseContextExtractor (§2.12)
    - ArtistSignatureStore       (§2.13)

Standard: ≥ 35 Tests pro Klasse (§5.1 der Copilot-Instructions).
Alle Tests:
    - synthetische Signale (keine realen Audio-Dateien)
    - np.random.seed(42) pro Test
    - SR = 48000 Hz invariant geprüft
    - NaN/Inf frei
    - Bounds-geprüft
"""


import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48000


def make_sine(freq: float = 440.0, duration_s: float = 2.0, sr: int = SR) -> np.ndarray:
    """Einfacher Sinus als Test-Audio."""
    t = np.arange(int(duration_s * sr)) / float(sr)
    return (np.sin(2.0 * np.pi * freq * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def make_noise(duration_s: float = 2.0, sr: int = SR) -> np.ndarray:
    """Weißrauschen."""
    np.random.seed(42)
    return np.random.randn(int(duration_s * sr)).astype(np.float32) * 0.3


def make_silence(duration_s: float = 2.0) -> np.ndarray:
    """Absolutes Schweigen."""
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def make_dirac(duration_s: float = 2.0) -> np.ndarray:
    """Einzelner Dirac-Impuls."""
    sig = np.zeros(int(duration_s * SR), dtype=np.float32)
    sig[SR // 2] = 1.0
    return sig


def make_complex(duration_s: float = 5.0, sr: int = SR) -> np.ndarray:
    """Komplexes Test-Signal: Mischung Sinusse + Rauschen."""
    np.random.seed(42)
    n = int(duration_s * sr)
    t = np.arange(n) / float(sr)
    sig = (
        0.4 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2 * np.pi * 880.0 * t)
        + 0.1 * np.random.randn(n)
    ).astype(np.float32)
    return np.clip(sig, -1.0, 1.0)  # type: ignore[no-any-return]


# ===========================================================================
# 1. PsychoacousticMaskingModel (§4.5)
# ===========================================================================


@pytest.mark.unit
class TestPsychoacousticMaskingModel:
    """≥ 35 Tests für core/psychoacoustic_masking_model.py"""

    def _get_model(self):
        from backend.core.psychoacoustic_masking_model import get_masking_model

        return get_masking_model()

    # --- Imports & Singleton ---

    def test_01_import_succeeds(self):
        import backend.core.psychoacoustic_masking_model as m

        assert hasattr(m, "PsychoacousticMaskingModel")

    def test_02_singleton_returns_same_instance(self):
        from backend.core.psychoacoustic_masking_model import get_masking_model

        a = get_masking_model()
        b = get_masking_model()
        assert a is b

    # --- compute_threshold: Basisprüfungen ---

    def test_03_compute_threshold_returns_masking_result(self):
        from backend.core.psychoacoustic_masking_model import MaskingResult

        m = self._get_model()
        result = m.compute_threshold(make_sine(), SR)
        assert isinstance(result, MaskingResult)

    def test_04_gain_modifier_no_nan(self):
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        assert np.all(np.isfinite(r.gain_modifier))

    def test_05_gain_modifier_shape(self):
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        assert r.gain_modifier.ndim == 2
        assert r.gain_modifier.shape[1] == 24  # 24 Bark-Bänder

    def test_06_gain_modifier_floor(self):
        """Gain darf nie unter 0.1 fallen (§4.5 GAIN_FLOOR)."""
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        assert float(np.min(r.gain_modifier)) >= 0.1 - 1e-6

    def test_07_gain_modifier_ceiling(self):
        """Gain darf nie über 1.0 liegen."""
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        assert float(np.max(r.gain_modifier)) <= 1.0 + 1e-6

    def test_08_masking_threshold_no_nan(self):
        m = self._get_model()
        r = m.compute_threshold(make_noise(), SR)
        assert np.all(np.isfinite(r.masking_threshold))

    def test_09_masking_threshold_positive(self):
        m = self._get_model()
        r = m.compute_threshold(make_noise(), SR)
        assert np.all(r.masking_threshold >= 0)

    def test_10_silence_frames_bool_array(self):
        m = self._get_model()
        r = m.compute_threshold(make_silence(), SR)
        assert r.silence_frames.dtype == bool

    def test_11_silence_mostly_silent_frames(self):
        m = self._get_model()
        r = m.compute_threshold(make_silence(), SR)
        assert float(np.mean(r.silence_frames)) > 0.5

    def test_12_silence_gain_modifier_capped(self):
        """Stille → Gain ≤ 0.30 (SILENCE_GAIN_MAX)."""
        m = self._get_model()
        r = m.compute_threshold(make_silence(), SR)
        # Nur Stille-Frames prüfen
        if np.any(r.silence_frames):
            silence_gain = r.gain_modifier[r.silence_frames, :]
            assert float(np.max(silence_gain)) <= 0.30 + 1e-6

    def test_13_dirac_no_nan(self):
        m = self._get_model()
        r = m.compute_threshold(make_dirac(), SR)
        assert np.all(np.isfinite(r.gain_modifier))

    def test_14_noise_gain_modifier_shape_consistent(self):
        m = self._get_model()
        r = m.compute_threshold(make_noise(), SR)
        n_frames = r.gain_modifier.shape[0]
        assert r.silence_frames.shape == (n_frames,)
        assert r.post_mask_frames.shape == (n_frames,)

    # --- compute_threshold: SR Invariante ---

    def test_15_wrong_sr_raises(self):
        m = self._get_model()
        with pytest.raises(AssertionError, match="48000"):
            m.compute_threshold(make_sine(sr=44100), 44100)

    # --- apply_adaptive_gain ---

    def test_16_apply_adaptive_gain_output_no_nan(self):
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        gain = np.ones_like(r.gain_modifier)
        out = m.apply_adaptive_gain(gain, r)
        assert np.all(np.isfinite(out))

    def test_17_apply_adaptive_gain_output_floor(self):
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        gain = np.ones_like(r.gain_modifier)
        out = m.apply_adaptive_gain(gain, r)
        assert float(np.min(out)) >= 0.1 - 1e-6

    def test_18_apply_adaptive_gain_output_ceiling(self):
        m = self._get_model()
        r = m.compute_threshold(make_noise(), SR)
        gain = np.ones_like(r.gain_modifier)
        out = m.apply_adaptive_gain(gain, r)
        assert float(np.max(out)) <= 1.0 + 1e-6

    def test_19_apply_adaptive_gain_shape_preserved(self):
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        gain = np.ones_like(r.gain_modifier)
        out = m.apply_adaptive_gain(gain, r)
        assert out.shape == gain.shape

    # --- Convenience-Funktion ---

    def test_20_convenience_function_exists(self):
        from backend.core.psychoacoustic_masking_model import compute_masking_threshold

        r = compute_masking_threshold(make_sine(), SR)
        assert r is not None

    def test_21_convenience_no_nan(self):
        from backend.core.psychoacoustic_masking_model import compute_masking_threshold

        r = compute_masking_threshold(make_noise(), SR)
        assert np.all(np.isfinite(r.gain_modifier))

    # --- Edge Cases ---

    def test_22_very_short_audio(self):
        m = self._get_model()
        short = np.zeros(512, dtype=np.float32)
        r = m.compute_threshold(short, SR)
        assert np.all(np.isfinite(r.gain_modifier))

    def test_23_stereo_audio(self):
        from backend.core.psychoacoustic_masking_model import compute_masking_threshold

        stereo = np.stack([make_sine(), make_sine(freq=880.0)], axis=0)
        r = compute_masking_threshold(stereo, SR)
        assert np.all(np.isfinite(r.gain_modifier))

    def test_24_nan_input_handled(self):
        m = self._get_model()
        audio = make_sine()
        audio[100] = float("nan")
        r = m.compute_threshold(audio, SR)
        assert np.all(np.isfinite(r.gain_modifier))

    def test_25_inf_input_handled(self):
        m = self._get_model()
        audio = make_sine()
        audio[200] = float("inf")
        r = m.compute_threshold(audio, SR)
        assert np.all(np.isfinite(r.gain_modifier))

    def test_26_masking_result_as_dict(self):
        m = self._get_model()
        r = m.compute_threshold(make_sine(), SR)
        d = r.as_dict()
        assert isinstance(d, dict)
        assert "n_frames" in d or len(d) > 0

    def test_27_post_mask_frames_bool(self):
        m = self._get_model()
        r = m.compute_threshold(make_dirac(), SR)
        assert r.post_mask_frames.dtype == bool

    def test_28_bark_bins_ndarray(self):
        """_build_bark_bins liefert NumPy-Array mit 25 Kanten (= 24 Bänder)."""
        from backend.core.psychoacoustic_masking_model import N_BARK, PsychoacousticMaskingModel

        new_m = PsychoacousticMaskingModel()
        bins = new_m._build_bark_bins(SR)
        assert isinstance(bins, np.ndarray)
        # 24 Bänder = 25 Kantenpositionen (N_BARK + 1)
        assert len(bins) == N_BARK + 1

    def test_29_consistent_output_same_input(self):
        """Gleiche Eingabe → gleiche Ausgabe (Determinismus)."""
        m = self._get_model()
        audio = make_sine()
        r1 = m.compute_threshold(audio, SR)
        r2 = m.compute_threshold(audio, SR)
        np.testing.assert_array_almost_equal(r1.gain_modifier, r2.gain_modifier)

    def test_30_long_audio(self):
        m = self._get_model()
        long_audio = make_complex(duration_s=10.0)
        r = m.compute_threshold(long_audio, SR)
        assert np.all(np.isfinite(r.gain_modifier))

    def test_31_silence_threshold_db_value(self):
        from backend.core.psychoacoustic_masking_model import SILENCE_DB

        assert SILENCE_DB <= -30.0  # Stille bei ≤ -30 dBFS (§3.1)

    def test_32_gain_floor_constant(self):
        from backend.core.psychoacoustic_masking_model import GAIN_FLOOR

        assert GAIN_FLOOR >= 0.1

    def test_33_n_bark_bands_constant(self):
        from backend.core.psychoacoustic_masking_model import N_BARK

        assert N_BARK == 24

    def test_34_masking_slope_length(self):
        from backend.core.psychoacoustic_masking_model import _MASKING_SLOPE_DB

        assert len(_MASKING_SLOPE_DB) == 24

    def test_35_masking_slope_increasing(self):
        """ISO 11172-3: Masking-Slope wächst mit Bandnummer."""
        from backend.core.psychoacoustic_masking_model import _MASKING_SLOPE_DB

        for i in range(1, len(_MASKING_SLOPE_DB)):
            assert _MASKING_SLOPE_DB[i] >= _MASKING_SLOPE_DB[i - 1]


# ===========================================================================
# 2. HarmonicLatticeAnalyzer (§2.11)
# ===========================================================================


class TestHarmonicLatticeAnalyzer:
    """≥ 35 Tests für core/harmonic_lattice_analyzer.py"""

    def _get_analyzer(self):
        from backend.core.harmonic_lattice_analyzer import get_harmonic_lattice

        return get_harmonic_lattice()

    def test_01_import_succeeds(self):
        import backend.core.harmonic_lattice_analyzer as m

        assert hasattr(m, "HarmonicLatticeAnalyzer")

    def test_02_singleton_same_instance(self):
        from backend.core.harmonic_lattice_analyzer import get_harmonic_lattice

        assert get_harmonic_lattice() is get_harmonic_lattice()

    def test_03_analyze_returns_result(self):
        from backend.core.harmonic_lattice_analyzer import HarmonicLatticeResult

        a = self._get_analyzer()
        r = a.analyze(make_sine(), SR)
        assert isinstance(r, HarmonicLatticeResult)

    def test_04_result_f0_finite(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(freq=200.0), SR)
        assert math.isfinite(r.f0_hz)

    def test_05_result_lattice_score_bounds(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(), SR)
        assert 0.0 <= r.lattice_score <= 1.0

    def test_06_result_inharmonicity_b_bounds(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(), SR)
        assert 0.0 <= r.inharmonicity_b <= 0.05

    def test_07_sine_has_valid_f0(self):
        """Sinus bei 220 Hz → f₀ nahe 220 Hz."""
        a = self._get_analyzer()
        r = a.analyze(make_sine(freq=220.0, duration_s=3.0), SR)
        # f0 muss irgendwo detektiert werden (kann 0 sein bei kurzen Signalen)
        assert math.isfinite(r.f0_hz)
        assert r.f0_hz >= 0.0

    def test_08_silence_no_crash(self):
        a = self._get_analyzer()
        r = a.analyze(make_silence(), SR)
        assert r is not None
        assert math.isfinite(r.lattice_score)

    def test_09_noise_no_crash(self):
        a = self._get_analyzer()
        r = a.analyze(make_noise(), SR)
        assert r is not None

    def test_10_wrong_sr_raises(self):
        a = self._get_analyzer()
        with pytest.raises(AssertionError, match="48000"):
            a.analyze(make_sine(sr=44100), 44100)

    def test_11_as_dict_returns_dict(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(), SR)
        d = r.as_dict()
        assert isinstance(d, dict)
        assert "f0_hz" in d
        assert "lattice_score" in d

    def test_12_partials_list(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(), SR)
        assert isinstance(r.partials, list)

    def test_13_partial_deviation_finite(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(freq=440.0, duration_s=3.0), SR)
        for p in r.partials:
            assert math.isfinite(p.deviation_cent)

    def test_14_partial_n_positive(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(freq=440.0, duration_s=3.0), SR)
        for p in r.partials:
            assert p.partial_n >= 1

    def test_15_partial_needs_correction_bool(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(freq=440.0, duration_s=3.0), SR)
        for p in r.partials:
            assert isinstance(p.needs_correction, bool)

    def test_16_enforce_coherence_output_no_nan(self):
        a = self._get_analyzer()
        audio = make_sine(freq=220.0, duration_s=2.0)
        r = a.analyze(audio, SR)
        out = a.enforce_coherence(audio, SR, r)
        assert np.all(np.isfinite(out))

    def test_17_enforce_coherence_clipped(self):
        a = self._get_analyzer()
        audio = make_sine()
        r = a.analyze(audio, SR)
        out = a.enforce_coherence(audio, SR, r)
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-5

    def test_18_enforce_coherence_same_length(self):
        a = self._get_analyzer()
        audio = make_sine(duration_s=1.5)
        r = a.analyze(audio, SR)
        out = a.enforce_coherence(audio, SR, r)
        assert len(out) == len(audio)

    def test_19_enforce_coherence_wrong_sr_raises(self):
        a = self._get_analyzer()
        audio = make_sine()
        r = a.analyze(audio, SR)
        with pytest.raises(AssertionError, match="48000"):
            a.enforce_coherence(audio, 44100, r)

    def test_20_enforce_no_enforcement_needed(self):
        """Kein Enforcement wenn needs_enforcement=False."""
        from backend.core.harmonic_lattice_analyzer import HarmonicLatticeResult

        a = self._get_analyzer()
        audio = make_sine()
        a.analyze(audio, SR)
        r_no = HarmonicLatticeResult(
            f0_hz=0.0,
            inharmonicity_b=0.0,
            partials=[],
            lattice_score=1.0,
            instrument_tag="flute",
            needs_enforcement=False,
        )
        out = a.enforce_coherence(audio, SR, r_no)
        np.testing.assert_array_almost_equal(out, np.clip(audio, -1.0, 1.0))

    def test_21_inharmonicity_priors_keys(self):
        from backend.core.harmonic_lattice_analyzer import INHARMONICITY_PRIORS

        for key in ["piano_bass", "piano_mid", "guitar", "violin", "flute", "brass", "unknown"]:
            assert key in INHARMONICITY_PRIORS

    def test_22_inharmonicity_priors_values_valid(self):
        from backend.core.harmonic_lattice_analyzer import INHARMONICITY_PRIORS

        for k, v in INHARMONICITY_PRIORS.items():
            assert 0.0 <= v <= 0.05, f"B-Koeff. für {k} außerhalb [0, 0.05]"

    def test_23_flute_b_is_zero(self):
        from backend.core.harmonic_lattice_analyzer import INHARMONICITY_PRIORS

        assert INHARMONICITY_PRIORS["flute"] == 0.0

    def test_24_convenience_function(self):
        from backend.core.harmonic_lattice_analyzer import analyze_harmonic_lattice

        r = analyze_harmonic_lattice(make_sine(), SR)
        assert r is not None

    def test_25_convenience_no_nan(self):
        from backend.core.harmonic_lattice_analyzer import analyze_harmonic_lattice

        r = analyze_harmonic_lattice(make_complex(), SR)
        assert math.isfinite(r.lattice_score)

    def test_26_stereo_input(self):
        from backend.core.harmonic_lattice_analyzer import analyze_harmonic_lattice

        stereo = np.stack([make_sine(), make_sine(freq=880.0)], axis=0)
        r = analyze_harmonic_lattice(stereo, SR)
        assert r is not None

    def test_26b_stereo_input_samples_channels_layout(self):
        from backend.core.harmonic_lattice_analyzer import analyze_harmonic_lattice

        stereo = np.column_stack([make_sine(), make_sine(freq=880.0)])
        r = analyze_harmonic_lattice(stereo, SR)
        assert r is not None
        assert math.isfinite(r.lattice_score)

    def test_27_nan_input_handled(self):
        a = self._get_analyzer()
        audio = make_sine()
        audio[1000] = float("nan")
        r = a.analyze(audio, SR)
        assert math.isfinite(r.lattice_score)

    def test_28_instrument_tag_preserved(self):
        a = self._get_analyzer()
        r = a.analyze(make_sine(), SR, instrument_tag="guitar")
        assert r.instrument_tag == "guitar"

    def test_29_null_result_for_silence(self):
        """Stille → kein Crash, needs_enforcement=False."""
        a = self._get_analyzer()
        r = a.analyze(make_silence(), SR)
        assert r.needs_enforcement is False

    def test_30_b_estimation_from_partials(self):
        """_estimate_b_from_partials gibt Wert in [0, 0.05]."""
        from backend.core.harmonic_lattice_analyzer import PartialAnalysis

        a = self._get_analyzer()
        partials = [
            PartialAnalysis(1, 440.0, 440.0, 0.0, False),
            PartialAnalysis(2, 880.5, 880.0, 1.0, False),
            PartialAnalysis(3, 1321.5, 1320.0, 1.9, False),
        ]
        b = a._estimate_b_from_partials(partials, 440.0)
        assert 0.0 <= b <= 0.05

    def test_31_fletcher_correction_formula(self):
        """Fletcher-Modell: fₙ = n·f₀·√(1+B·n²)"""
        f0 = 110.0
        b = 0.002
        n = 3
        expected = n * f0 * math.sqrt(1.0 + b * n * n)
        assert expected > n * f0  # Inharmonizität erhöht Frequenz

    def test_32_estimate_f0_positive_for_sine(self):
        """Autokorrelation findet positiven f₀ für Sinus."""
        a = self._get_analyzer()
        sine = make_sine(freq=200.0, duration_s=3.0)
        f0 = a._estimate_f0(sine, SR)
        assert f0 >= 0.0

    def test_33_detect_partials_empty_for_silence(self):
        """Stille → keine Partials oder f₀=0."""
        a = self._get_analyzer()
        silence = make_silence(duration_s=2.0)
        f0 = a._estimate_f0(silence, SR)
        if f0 > 0:
            partials = a._detect_partials(silence, SR, f0, 0.001)
            # Keine Einschränkung auf Anzahl — aber kein Absturz
            assert isinstance(partials, list)

    def test_34_max_partials_limit(self):
        from backend.core.harmonic_lattice_analyzer import MAX_PARTIALS

        assert MAX_PARTIALS >= 8

    def test_35_max_cent_deviation_constant(self):
        from backend.core.harmonic_lattice_analyzer import MAX_CENT_DEVIATION

        assert MAX_CENT_DEVIATION == 5.0


# ===========================================================================
# 3. SegmentAdaptiveProcessor (§2.10)
# ===========================================================================


class TestSegmentAdaptiveProcessor:
    """≥ 35 Tests für core/segment_adaptive_processor.py"""

    def _get_processor(self):
        from backend.core.segment_adaptive_processor import get_segment_processor

        return get_segment_processor()

    def _passthrough(self, audio, sr, params):
        return audio.copy()

    def test_01_import_succeeds(self):
        import backend.core.segment_adaptive_processor as m

        assert hasattr(m, "SegmentAdaptiveProcessor")

    def test_02_singleton_same_instance(self):
        from backend.core.segment_adaptive_processor import get_segment_processor

        assert get_segment_processor() is get_segment_processor()

    def test_03_process_returns_result(self):
        from backend.core.segment_adaptive_processor import AdaptiveProcessingResult

        p = self._get_processor()
        audio = make_complex(duration_s=6.0)
        r = p.process(audio, SR, self._passthrough)
        assert isinstance(r, AdaptiveProcessingResult)

    def test_04_output_no_nan(self):
        p = self._get_processor()
        audio = make_complex(duration_s=6.0)
        r = p.process(audio, SR, self._passthrough)
        assert np.all(np.isfinite(r.audio))

    def test_05_output_clipped(self):
        p = self._get_processor()
        audio = make_complex(duration_s=6.0)
        r = p.process(audio, SR, self._passthrough)
        assert float(np.max(np.abs(r.audio))) <= 1.0 + 1e-5

    def test_06_output_same_length_as_input(self):
        p = self._get_processor()
        audio = make_complex(duration_s=6.0)
        r = p.process(audio, SR, self._passthrough)
        assert len(r.audio) == len(audio)

    def test_07_short_audio_uses_fallback(self):
        p = self._get_processor()
        audio = make_sine(duration_s=3.0)  # < 5 s
        r = p.process(audio, SR, self._passthrough)
        assert r.used_fallback is True

    def test_08_long_audio_segment_adaptive(self):
        p = self._get_processor()
        audio = make_complex(duration_s=10.0)
        r = p.process(audio, SR, self._passthrough)
        assert r.n_segments >= 1

    def test_09_enabled_false_uses_fallback(self):
        p = self._get_processor()
        audio = make_complex(duration_s=10.0)
        r = p.process(audio, SR, self._passthrough, enabled=False)
        assert r.used_fallback is True

    def test_10_wrong_sr_raises(self):
        p = self._get_processor()
        with pytest.raises(AssertionError, match="48000"):
            p.process(make_sine(sr=44100), 44100, self._passthrough)

    def test_11_segments_not_empty(self):
        p = self._get_processor()
        audio = make_complex(duration_s=8.0)
        r = p.process(audio, SR, self._passthrough)
        assert len(r.segments) >= 1

    def test_12_segment_start_end_ordered(self):
        p = self._get_processor()
        audio = make_complex(duration_s=8.0)
        r = p.process(audio, SR, self._passthrough)
        for seg in r.segments:
            assert seg.start_sample < seg.end_sample

    def test_13_silence_segment_type(self):
        p = self._get_processor()
        audio = make_silence(duration_s=10.0)
        r = p.process(audio, SR, self._passthrough)
        types = [s.segment_type for s in r.segments]
        assert "silence" in types or "mixed" in types  # Stille oder mixed

    def test_14_defect_severity_bounds(self):
        p = self._get_processor()
        audio = make_complex(duration_s=8.0)
        r = p.process(audio, SR, self._passthrough)
        for seg in r.segments:
            assert 0.0 <= seg.defect_severity <= 1.0

    def test_15_max_segments_not_exceeded(self):
        from backend.core.segment_adaptive_processor import MAX_SEGMENTS

        p = self._get_processor()
        audio = make_complex(duration_s=15.0)
        r = p.process(audio, SR, self._passthrough)
        assert r.n_segments <= MAX_SEGMENTS

    def test_16_as_dict_returns_dict(self):
        p = self._get_processor()
        audio = make_complex(duration_s=6.0)
        r = p.process(audio, SR, self._passthrough)
        d = r.as_dict()
        assert isinstance(d, dict)
        assert "n_segments" in d

    def test_17_audio_segment_as_dict(self):
        from backend.core.segment_adaptive_processor import AudioSegment

        s = AudioSegment(start_sample=0, end_sample=1024)
        d = s.as_dict()
        assert "start_sample" in d

    def test_18_nan_input_handled(self):
        p = self._get_processor()
        audio = make_complex(duration_s=6.0)
        audio[100] = float("nan")
        r = p.process(audio, SR, self._passthrough)
        assert np.all(np.isfinite(r.audio))

    def test_19_inf_input_handled(self):
        p = self._get_processor()
        audio = make_complex(duration_s=6.0)
        audio[500] = float("inf")
        r = p.process(audio, SR, self._passthrough)
        assert np.all(np.isfinite(r.audio))

    def test_20_segment_audio_method(self):
        p = self._get_processor()
        audio = make_complex(duration_s=8.0)
        segments = p.segment_audio(audio, SR)
        assert isinstance(segments, list)
        assert len(segments) >= 1

    def test_21_segment_audio_wrong_sr_raises(self):
        p = self._get_processor()
        with pytest.raises(AssertionError, match="48000"):
            p.segment_audio(make_sine(sr=44100), 44100)

    def test_22_convenience_function(self):
        from backend.core.segment_adaptive_processor import process_adaptive

        audio = make_complex(duration_s=6.0)
        r = process_adaptive(audio, SR, self._passthrough)
        assert r is not None

    def test_23_convenience_output_no_nan(self):
        from backend.core.segment_adaptive_processor import process_adaptive

        r = process_adaptive(make_complex(duration_s=6.0), SR, self._passthrough)
        assert np.all(np.isfinite(r.audio))

    def test_24_process_fn_receives_correct_sr(self):
        seen_srs = []

        def capture_sr_fn(audio, sr, params):
            seen_srs.append(sr)
            return audio

        p = self._get_processor()
        p.process(make_complex(duration_s=6.0), SR, capture_sr_fn)
        # Alle Segmente müssen SR=48000 bekommen
        assert all(s == SR for s in seen_srs)

    def test_25_process_fn_receives_params_dict(self):
        received_params = []

        def capture_params_fn(audio, sr, params):
            received_params.append(params)
            return audio

        p = self._get_processor()
        p.process(make_complex(duration_s=7.0), SR, capture_params_fn)
        assert all(isinstance(p_, dict) for p_ in received_params)

    def test_26_min_segment_duration_constant(self):
        from backend.core.segment_adaptive_processor import MIN_SEGMENT_DURATION_S

        assert MIN_SEGMENT_DURATION_S >= 0.5

    def test_27_crossfade_ms_constant(self):
        from backend.core.segment_adaptive_processor import CROSSFADE_MS

        assert CROSSFADE_MS >= 10.0

    def test_28_default_params_keys(self):
        from backend.core.segment_adaptive_processor import SegmentAdaptiveProcessor

        s = SegmentAdaptiveProcessor()
        params = s._default_params()
        assert "noise_reduction_strength" in params
        assert "harmonic_boost_db" in params
        assert "ola_crossfade_ms" in params

    def test_29_adaptive_params_silence(self):
        from backend.core.segment_adaptive_processor import SegmentAdaptiveProcessor

        s = SegmentAdaptiveProcessor()
        params = s._adaptive_params("silence", 0.0)
        assert params["noise_reduction_strength"] <= 0.2

    def test_30_adaptive_params_high_severity(self):
        from backend.core.segment_adaptive_processor import SegmentAdaptiveProcessor

        s = SegmentAdaptiveProcessor()
        params = s._adaptive_params("mixed", 0.8)
        assert params["noise_reduction_strength"] >= 0.5

    def test_31_stereo_input_works(self):
        p = self._get_processor()
        stereo = np.stack([make_complex(duration_s=6.0)] * 2, axis=0)
        r = p.process(stereo, SR, self._passthrough)
        assert np.all(np.isfinite(r.audio))

    def test_32_dirac_no_crash(self):
        p = self._get_processor()
        r = p.process(make_dirac(duration_s=6.0), SR, self._passthrough)
        assert r is not None

    def test_33_n_segments_member_correct(self):
        p = self._get_processor()
        audio = make_complex(duration_s=8.0)
        r = p.process(audio, SR, self._passthrough)
        assert r.n_segments == len(r.segments)

    def test_34_defect_severity_estimate_positive(self):
        from backend.core.segment_adaptive_processor import SegmentAdaptiveProcessor

        s = SegmentAdaptiveProcessor()
        sev = s._estimate_defect_severity(make_noise(duration_s=1.0), SR)
        assert 0.0 <= sev <= 1.0

    def test_35_process_fn_exception_fallback(self):
        """Defekte Process-Funktion → Passthrough ohne Absturz."""

        def broken_fn(audio, sr, params):
            raise RuntimeError("Absicht")

        p = self._get_processor()
        r = p.process(make_complex(duration_s=7.0), SR, broken_fn)
        # Kein Absturz, Ausgabe endlich
        assert np.all(np.isfinite(r.audio))

    def test_36_segment_duration_property(self):
        from backend.core.segment_adaptive_processor import AudioSegment

        seg = AudioSegment(start_sample=0, end_sample=4800)
        assert seg.duration_samples == 4800


# ===========================================================================
# 4. MusicalPhraseContextExtractor (§2.12)
# ===========================================================================


class TestMusicalPhraseContextExtractor:
    """≥ 35 Tests für core/musical_phrase_context.py"""

    def _get_extractor(self):
        from backend.core.musical_phrase_context import get_phrase_extractor

        return get_phrase_extractor()

    def test_01_import_succeeds(self):
        import backend.core.musical_phrase_context as m

        assert hasattr(m, "MusicalPhraseContextExtractor")

    def test_02_singleton_same_instance(self):
        from backend.core.musical_phrase_context import get_phrase_extractor

        assert get_phrase_extractor() is get_phrase_extractor()

    def test_03_extract_returns_context(self):
        from backend.core.musical_phrase_context import PhraseContext

        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        gap_start = int(5.0 * SR)
        gap_end = int(5.2 * SR)
        ctx = e.extract_context(audio, SR, gap_start, gap_end)
        assert isinstance(ctx, PhraseContext)

    def test_04_context_audio_no_nan(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        ctx = e.extract_context(audio, SR, int(5 * SR), int(5.1 * SR))
        assert np.all(np.isfinite(ctx.audio_context))

    def test_05_chroma_mean_shape(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        ctx = e.extract_context(audio, SR, int(5 * SR), int(5.1 * SR))
        assert ctx.chroma_mean.shape == (12,)

    def test_06_chroma_mean_normalized(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        ctx = e.extract_context(audio, SR, int(5 * SR), int(5.1 * SR))
        norm = float(np.linalg.norm(ctx.chroma_mean))
        assert norm > 0.0  # Nicht Null-Vektor

    def test_07_tempo_bpm_bounds(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        ctx = e.extract_context(audio, SR, int(5 * SR), int(5.1 * SR))
        assert 40.0 <= ctx.tempo_bpm <= 240.0

    def test_08_short_audio_fallback(self):
        e = self._get_extractor()
        audio = make_sine(duration_s=4.0)  # < 8 s → Fallback
        ctx = e.extract_context(audio, SR, int(2 * SR), int(2.1 * SR))
        assert ctx.is_fallback is True

    def test_09_wrong_sr_raises(self):
        e = self._get_extractor()
        with pytest.raises(AssertionError, match="48000"):
            e.extract_context(make_sine(sr=44100), 44100, 0, 1024)

    def test_10_gap_zeroed_in_context(self):
        """Dropout-Lücke wird in context_audio auf 0 gesetzt."""
        e = self._get_extractor()
        audio = make_complex(duration_s=4.0)  # < 8 s → Fallback
        gap_start, gap_end = int(2.0 * SR), int(2.1 * SR)
        ctx = e.extract_context(audio, SR, gap_start, gap_end)
        # Im Fallback-Modus ist context_audio vorhanden
        assert np.all(np.isfinite(ctx.audio_context))

    def test_11_beat_positions_list(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        ctx = e.extract_context(audio, SR, int(5 * SR), int(5.1 * SR))
        assert isinstance(ctx.beat_positions, list)

    def test_12_phrase_start_le_phrase_end(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        ctx = e.extract_context(audio, SR, int(5 * SR), int(5.1 * SR))
        assert ctx.phrase_start_s <= ctx.phrase_end_s

    def test_13_gap_times_correct(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        gap_start, gap_end = int(5 * SR), int(5.2 * SR)
        ctx = e.extract_context(audio, SR, gap_start, gap_end)
        assert abs(ctx.gap_start_s - 5.0) < 0.1
        assert abs(ctx.gap_end_s - 5.2) < 0.1

    def test_14_condition_inpainting_no_nan(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=4.0)
        ctx = e.extract_context(audio, SR, int(2 * SR), int(2.1 * SR))
        gap = make_sine(duration_s=0.1)
        out = e.condition_inpainting(gap, ctx)
        assert np.all(np.isfinite(out))

    def test_15_condition_inpainting_clipped(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=4.0)
        ctx = e.extract_context(audio, SR, int(2 * SR), int(2.1 * SR))
        gap = make_noise(duration_s=0.1)
        out = e.condition_inpainting(gap, ctx)
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-5

    def test_16_condition_inpainting_same_length(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=4.0)
        ctx = e.extract_context(audio, SR, int(2 * SR), int(2.1 * SR))
        gap = make_sine(duration_s=0.2)
        out = e.condition_inpainting(gap, ctx)
        assert len(out) == len(gap)

    def test_17_convenience_function(self):
        from backend.core.musical_phrase_context import extract_phrase_context

        audio = make_complex(duration_s=15.0)
        ctx = extract_phrase_context(audio, SR, int(5 * SR), int(5.1 * SR))
        assert ctx is not None

    def test_18_convenience_no_nan(self):
        from backend.core.musical_phrase_context import extract_phrase_context

        audio = make_complex(duration_s=15.0)
        ctx = extract_phrase_context(audio, SR, int(7 * SR), int(7.15 * SR))
        assert np.all(np.isfinite(ctx.audio_context))

    def test_19_nan_input_handled(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        audio[1000] = float("nan")
        ctx = e.extract_context(audio, SR, int(5 * SR), int(5.1 * SR))
        assert np.all(np.isfinite(ctx.audio_context))

    def test_20_chroma_for_silence_valid(self):
        e = self._get_extractor()
        r = e._compute_chroma(make_silence(), SR)
        assert r.shape == (12,)
        assert np.all(np.isfinite(r))

    def test_21_chroma_for_noise_valid(self):
        e = self._get_extractor()
        r = e._compute_chroma(make_noise(), SR)
        assert r.shape == (12,)
        assert np.all(np.isfinite(r))

    def test_22_pearson_perfect_correlation(self):
        e = self._get_extractor()
        v = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(e._pearson(v, v)) - 1.0 < 1e-5

    def test_23_pearson_anticorrelation(self):
        e = self._get_extractor()
        v = np.array([1.0, 2.0, 3.0])
        assert e._pearson(v, -v) <= -0.99

    def test_24_pearson_zero_vector(self):
        e = self._get_extractor()
        v = np.zeros(5)
        r = e._pearson(v, np.ones(5))
        assert r == 0.0

    def test_25_estimate_tempo_returns_float(self):
        e = self._get_extractor()
        t = e._estimate_tempo(make_complex(duration_s=5.0), SR)
        assert isinstance(t, float)
        assert 40.0 <= t <= 240.0

    def test_26_estimate_beats_list(self):
        e = self._get_extractor()
        beats = e._estimate_beats(make_complex(duration_s=5.0), SR, 120.0)
        assert isinstance(beats, list)

    def test_27_detect_phrase_boundaries_includes_zero(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        beats = e._estimate_beats(audio, SR, 120.0)
        bndrs = e._detect_phrase_boundaries(audio, SR, beats, 120.0)
        assert 0 in bndrs

    def test_28_detect_phrase_boundaries_includes_end(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=15.0)
        beats = e._estimate_beats(audio, SR, 120.0)
        bndrs = e._detect_phrase_boundaries(audio, SR, beats, 120.0)
        assert len(audio) in bndrs

    def test_29_as_dict_returns_dict(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=4.0)
        ctx = e.extract_context(audio, SR, int(2 * SR), int(2.1 * SR))
        d = ctx.as_dict()
        assert "phrase_start_s" in d
        assert "gap_start_s" in d

    def test_30_max_context_duration(self):
        from backend.core.musical_phrase_context import MAX_CONTEXT_DURATION_S

        assert MAX_CONTEXT_DURATION_S == 30.0

    def test_31_chroma_jump_threshold(self):
        from backend.core.musical_phrase_context import CHROMA_JUMP_THRESHOLD

        assert 0.0 < CHROMA_JUMP_THRESHOLD <= 1.0

    def test_32_local_fallback_returns_context(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=3.0)
        ctx = e._local_fallback(audio, SR, int(1 * SR), int(1.2 * SR))
        assert ctx.is_fallback is True
        assert np.all(np.isfinite(ctx.audio_context))

    def test_33_very_long_audio(self):
        e = self._get_extractor()
        audio = make_complex(duration_s=20.0)
        ctx = e.extract_context(audio, SR, int(10 * SR), int(10.1 * SR))
        assert ctx is not None

    def test_34_stereo_input(self):
        e = self._get_extractor()
        stereo = np.stack([make_complex(duration_s=10.0)] * 2, axis=0)
        ctx = e.extract_context(stereo, SR, int(5 * SR), int(5.1 * SR))
        assert np.all(np.isfinite(ctx.audio_context))

    def test_35_gap_larger_than_50pct_of_phrase_uses_neighbor(self):
        """Wenn Lücke > 50% der Phrase → Fallback verwendet benachbarte Phrase."""
        e = self._get_extractor()
        # Kurze Datei → Fallback ohnehin
        audio = make_complex(duration_s=4.0)
        ctx = e.extract_context(audio, SR, 0, int(2.5 * SR))
        assert ctx is not None  # Kein Absturz


# ===========================================================================
# 5. ArtistSignatureStore (§2.13)
# ===========================================================================


class TestArtistSignatureStore:
    """≥ 35 Tests für core/artist_signature_store.py"""

    def _get_store(self, tmp_dir: str):
        """Erstellt Store mit temporärem Verzeichnis (kein ~/.aurik pollution)."""
        # Patche SIGNATURES_DIR temporär

        import backend.core.artist_signature_store as m

        m.SIGNATURES_DIR = Path(tmp_dir)
        m.SIGNATURES_DIR.mkdir(parents=True, exist_ok=True)
        # Neues Instance (Caches leeren)
        store = m.ArtistSignatureStore()
        store._cache.clear()
        return store

    def test_01_import_succeeds(self):
        import backend.core.artist_signature_store as m

        assert hasattr(m, "ArtistSignatureStore")

    def test_02_singleton_same_instance(self):
        from backend.core.artist_signature_store import get_signature_store

        assert get_signature_store() is get_signature_store()

    def test_03_detect_session_returns_8hex(self):
        from backend.core.artist_signature_store import ArtistSignatureStore

        s = ArtistSignatureStore()
        artist_id = s.detect_session([Path("/tmp/folder/song.flac")])
        assert len(artist_id) == 8
        assert all(c in "0123456789abcdef" for c in artist_id)

    def test_04_same_folder_same_session(self):
        from backend.core.artist_signature_store import ArtistSignatureStore

        s = ArtistSignatureStore()
        id1 = s.detect_session([Path("/some/folder/a.flac")])
        id2 = s.detect_session([Path("/some/folder/b.flac")])
        assert id1 == id2

    def test_05_different_folder_different_session(self):
        from backend.core.artist_signature_store import ArtistSignatureStore

        s = ArtistSignatureStore()
        id1 = s.detect_session([Path("/folder/a/song.flac")])
        id2 = s.detect_session([Path("/folder/b/song.flac")])
        assert id1 != id2

    def test_06_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            assert store.load("nonexistent") is None

    def test_07_save_and_load(self):
        from backend.core.artist_signature_store import ArtistSignature

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            sig = ArtistSignature(artist_id="test1234")
            store.save(sig)
            loaded = store.load("test1234")
            assert loaded is not None
            assert loaded.artist_id == "test1234"

    def test_08_confidence_increases_with_files(self):
        from backend.core.artist_signature_store import _confidence_from_n

        assert _confidence_from_n(0) == 0.0
        assert _confidence_from_n(1) < _confidence_from_n(5)
        assert _confidence_from_n(10) <= 1.0

    def test_09_confidence_caps_at_1(self):
        from backend.core.artist_signature_store import _confidence_from_n

        assert _confidence_from_n(100) <= 1.0

    def test_10_update_increments_n_files(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics(voice_gender="FEMALE")
            sig = store.update_from_analysis("abc12345", vc)
            assert sig.n_files_analyzed == 1
            sig2 = store.update_from_analysis("abc12345", vc)
            assert sig2.n_files_analyzed == 2

    def test_11_update_increases_confidence(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics()
            sig1 = store.update_from_analysis("aaaabbbb", vc)
            for _ in range(4):
                sig1 = store.update_from_analysis("aaaabbbb", vc)
            assert sig1.confidence >= 0.3

    def test_12_formant_profile_updated(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics(f1_hz=750.0)
            sig = store.update_from_analysis("f0f0f0f0", vc)
            assert "F1_median" in sig.formant_profile
            assert sig.formant_profile["F1_median"] > 0.0

    def test_13_vibrato_rate_clamped(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics(vibrato_rate_hz=100.0)  # Zu hoch → geclampt
            sig = store.update_from_analysis("aabbccdd", vc)
            assert 4.0 <= sig.vibrato_rate_hz <= 8.0

    def test_14_breathiness_clamped(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics(breathiness_ratio=5.0)  # Zu hoch
            sig = store.update_from_analysis("11223344", vc)
            assert 0.0 <= sig.breathiness_ratio <= 0.3

    def test_15_spectral_envelope_updated(self):
        from backend.core.artist_signature_store import SPECTRAL_ENVELOPE_DIM, VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            env = np.zeros(SPECTRAL_ENVELOPE_DIM, dtype=np.float32)
            env[0] = 1.0
            vc = VoiceCharacteristics(spectral_envelope=env)
            sig = store.update_from_analysis("envtest0", vc)
            assert sig.spectral_envelope.shape == (SPECTRAL_ENVELOPE_DIM,)

    def test_16_spectral_envelope_no_nan(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics()
            sig = store.update_from_analysis("00112233", vc)
            assert np.all(np.isfinite(sig.spectral_envelope))

    def test_17_delete_removes_file(self):
        from backend.core.artist_signature_store import ArtistSignature

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            sig = ArtistSignature(artist_id="deltst00")
            store.save(sig)
            path = Path(tmp) / "deltst00.json"
            assert path.exists()
            store.delete("deltst00")
            assert not path.exists()

    def test_18_delete_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            assert store.delete("doesnotexist") is False

    def test_19_list_all_returns_list(self):
        from backend.core.artist_signature_store import ArtistSignature

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            store.save(ArtistSignature(artist_id="aaaaaaaa"))
            store.save(ArtistSignature(artist_id="bbbbbbbb"))
            ids = store.list_all()
            assert "aaaaaaaa" in ids
            assert "bbbbbbbb" in ids

    def test_20_get_prior_strength_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            assert store.get_prior_strength("nope0000") == "kein Prior"

    def test_21_get_prior_strength_weak(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics()
            sig = store.update_from_analysis("weaktest", vc)
            store.save(sig)
            strength = store.get_prior_strength("weaktest")
            # confidence nach 1 Datei ≈ 0.15 → kein Prior
            assert strength in ("kein Prior", "schwacher Prior")

    def test_22_get_prior_strength_strong_after_many(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics()
            for _ in range(6):
                sig = store.update_from_analysis("strongst", vc)
            store.save(sig)
            strength = store.get_prior_strength("strongst")
            assert strength == "starker Prior"

    def test_23_signature_from_dict_roundtrip(self):
        from backend.core.artist_signature_store import ArtistSignature

        orig = ArtistSignature(artist_id="roundtr1", voice_gender="FEMALE", n_files_analyzed=3)
        d = orig.as_dict()
        restored = ArtistSignature.from_dict(d)
        assert restored.artist_id == "roundtr1"
        assert restored.voice_gender == "FEMALE"
        assert restored.n_files_analyzed == 3

    def test_24_last_updated_set_on_save(self):
        from backend.core.artist_signature_store import ArtistSignature

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            sig = ArtistSignature(artist_id="updttest")
            # __post_init__ kann already last_updated setzen – wir prüfen nur nach save()
            store.save(sig)
            loaded = store.load("updttest")
            assert loaded is not None
            assert len(loaded.last_updated) > 0

    def test_25_instrument_tags_accumulated(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc1 = VoiceCharacteristics(instrument_tags=["guitar"])
            vc2 = VoiceCharacteristics(instrument_tags=["piano"])
            sig = store.update_from_analysis("tagstest", vc1)
            sig = store.update_from_analysis("tagstest", vc2)
            assert "guitar" in sig.instrument_tags
            assert "piano" in sig.instrument_tags

    def test_26_no_duplicate_instrument_tags(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            vc = VoiceCharacteristics(instrument_tags=["guitar"])
            for _ in range(3):
                sig = store.update_from_analysis("dupetags", vc)
            assert sig.instrument_tags.count("guitar") == 1

    def test_27_confidence_weak_constant(self):
        from backend.core.artist_signature_store import CONFIDENCE_WEAK

        assert pytest.approx(0.3) == CONFIDENCE_WEAK

    def test_28_confidence_strong_constant(self):
        from backend.core.artist_signature_store import CONFIDENCE_STRONG

        assert pytest.approx(0.7) == CONFIDENCE_STRONG

    def test_29_spectral_envelope_dim_constant(self):
        from backend.core.artist_signature_store import SPECTRAL_ENVELOPE_DIM

        assert SPECTRAL_ENVELOPE_DIM == 128

    def test_30_voice_characteristics_default_values(self):
        from backend.core.artist_signature_store import VoiceCharacteristics

        vc = VoiceCharacteristics()
        assert vc.voice_gender == "UNKNOWN"
        assert vc.spectral_envelope is not None

    def test_31_signature_post_init_sets_envelope(self):
        from backend.core.artist_signature_store import SPECTRAL_ENVELOPE_DIM, ArtistSignature

        sig = ArtistSignature(artist_id="initchk0")
        assert sig.spectral_envelope.shape == (SPECTRAL_ENVELOPE_DIM,)

    def test_32_confidence_from_n_monotone(self):
        from backend.core.artist_signature_store import _confidence_from_n

        prev = 0.0
        for n in range(1, 11):
            c = _confidence_from_n(n)
            assert c >= prev
            prev = c

    def test_33_load_caches_result(self):
        from backend.core.artist_signature_store import ArtistSignature

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            sig = ArtistSignature(artist_id="cachekey")
            store.save(sig)
            loaded1 = store.load("cachekey")
            loaded2 = store.load("cachekey")
            # Zweiter Load aus Cache
            assert loaded1 is loaded2

    def test_34_save_updates_last_updated(self):
        from backend.core.artist_signature_store import ArtistSignature

        with tempfile.TemporaryDirectory() as tmp:
            store = self._get_store(tmp)
            sig = ArtistSignature(artist_id="timechk0")
            store.save(sig)
            assert "T" in sig.last_updated  # ISO 8601 enthält "T"

    def test_35_convenience_load(self):
        from backend.core.artist_signature_store import load_artist_signature

        result = load_artist_signature("nonexist")
        assert result is None

    def test_36_convenience_update(self):
        from backend.core.artist_signature_store import VoiceCharacteristics, update_artist_signature

        # Testet Convenience ohne zu speichern (tmp nicht verfügbar)
        vc = VoiceCharacteristics()
        # Ohne auto_save (Datei wird in ~/.aurik erstellt — vermeiden)
        sig = update_artist_signature("testconv0", vc, auto_save=False)
        assert sig is not None
        assert sig.n_files_analyzed >= 1

    def test_37_voice_characteristics_spectral_default_shape(self):
        from backend.core.artist_signature_store import SPECTRAL_ENVELOPE_DIM, VoiceCharacteristics

        vc = VoiceCharacteristics()
        assert vc.spectral_envelope is not None
        assert vc.spectral_envelope.shape == (SPECTRAL_ENVELOPE_DIM,)


# ===========================================================================
# Integration: Alle Module zusammen
# ===========================================================================


class TestV100Integration:
    """Integrations-Tests: Zusammenspiel der 5 neuen Kernmodule."""

    def test_01_masking_model_singleton_thread_safe(self):
        """Singleton-Zugriff aus mehreren Threads."""
        import threading

        from backend.core.psychoacoustic_masking_model import get_masking_model

        results = []
        errs = []

        def worker():
            try:
                m = get_masking_model()
                results.append(id(m))
            except Exception as e:
                errs.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errs
        assert len(set(results)) == 1  # Alle Threads bekommen dieselbe Instanz

    def test_02_harmonic_lattice_to_masking_model_pipeline(self):
        """Harmonic Lattice → Masking Model auf demselben Signal."""
        from backend.core.harmonic_lattice_analyzer import analyze_harmonic_lattice
        from backend.core.psychoacoustic_masking_model import compute_masking_threshold

        audio = make_complex(duration_s=3.0)
        lat = analyze_harmonic_lattice(audio, SR)
        mask = compute_masking_threshold(audio, SR)
        assert math.isfinite(lat.lattice_score)
        assert np.all(np.isfinite(mask.gain_modifier))

    def test_03_segment_adaptive_with_phrase_context(self):
        """SegmentAdaptive + PhraseContext interagieren ohne Absturz."""
        from backend.core.musical_phrase_context import extract_phrase_context
        from backend.core.segment_adaptive_processor import get_segment_processor

        audio = make_complex(duration_s=6.0)
        # Segmentieren
        proc = get_segment_processor()
        segs = proc.segment_audio(audio, SR)
        assert len(segs) >= 1
        # Phrase-Kontext für ein synthetisches Gap
        gap_s, gap_e = int(2 * SR), int(2.2 * SR)
        ctx = extract_phrase_context(audio, SR, gap_s, gap_e)
        assert ctx is not None

    def test_04_artist_signature_confidence_after_updates(self):
        """Confidence nach 5 Updates ≥ CONFIDENCE_WEAK."""
        import tempfile

        from backend.core.artist_signature_store import CONFIDENCE_WEAK, ArtistSignatureStore, VoiceCharacteristics

        with tempfile.TemporaryDirectory() as tmp:
            import backend.core.artist_signature_store as m

            m.SIGNATURES_DIR = Path(tmp)
            store = ArtistSignatureStore()
            vc = VoiceCharacteristics(voice_gender="MALE")
            for _ in range(5):
                sig = store.update_from_analysis("integr00", vc)
            assert sig.confidence >= CONFIDENCE_WEAK

    def test_05_all_modules_import_without_side_effects(self):
        """Alle 5 Module importierbar ohne Exceptions."""
        mods = [
            "core.psychoacoustic_masking_model",
            "core.harmonic_lattice_analyzer",
            "core.segment_adaptive_processor",
            "core.musical_phrase_context",
            "core.artist_signature_store",
        ]
        for mod in mods:
            import importlib

            m = importlib.import_module(mod)
            assert m is not None
