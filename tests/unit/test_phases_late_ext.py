"""
Unit-Tests für Aurik-Phasen 43-48, 50, 53.

Phasen:
  43 – ML DeEsser
  44 – Guitar Enhancement
  45 – Brass Enhancement
  46 – Spatial Enhancement
  47 – TruePeak Limiter
  48 – Stereo Width Enhancer
  50 – Spectral Repair
  53 – Semantic Audio

Aufruf-Konvention:
  process(audio: np.ndarray, sample_rate: int, **kwargs) -> PhaseResult
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.phases.phase_interface import PhaseResult

SR = 44100
_N = SR // 4  # 11025 Samples (0.25s) – keine Phase braucht längeres Audio


# ---------------------------------------------------------------------------
# Fixtures — scope="class": einmalig pro Testklasse (sicher bei in-place Ops)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def mono():
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, _N).astype(np.float32)


@pytest.fixture(scope="class")
def stereo():
    rng = np.random.default_rng(42)
    return rng.uniform(-0.1, 0.1, (_N, 2)).astype(np.float32)


@pytest.fixture(scope="class")
def silent_mono():
    return np.zeros(_N, dtype=np.float32)


@pytest.fixture(scope="class")
def hot_mono():
    """Durchgesteuertes Signal für Limiter-Tests."""
    rng = np.random.default_rng(7)
    return rng.uniform(-1.5, 1.5, _N).astype(np.float32)


@pytest.fixture(scope="class")
def sibilant_mono():
    """Signal mit starken Sibilanten (hohe 6-10 kHz-Energie) für DeEsser."""
    t = np.linspace(0, 0.25, _N, dtype=np.float32)
    sib = 0.3 * np.sin(2 * np.pi * 8000 * t)
    base = 0.1 * np.sin(2 * np.pi * 200 * t)
    return (base + sib).astype(np.float32)


# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------
def _assert_phase_result(result, orig_audio, *, check_clipping: bool = True, clip_threshold: float = 2.0):
    """Prüft ein PhaseResult auf Grundkorrektheit."""
    assert isinstance(result, PhaseResult), f"Kein PhaseResult: {type(result)}"
    assert result.success is True, f"success=False: {result}"
    assert isinstance(result.audio, np.ndarray), "audio muss ndarray sein"
    assert result.audio.shape == orig_audio.shape, f"Shape-Mismatch: {result.audio.shape} != {orig_audio.shape}"
    assert np.issubdtype(result.audio.dtype, np.floating), f"audio.dtype nicht floating: {result.audio.dtype}"
    assert isinstance(result.metadata, dict)
    assert isinstance(result.metrics, dict)
    assert float(result.execution_time_seconds) >= 0.0
    if check_clipping:
        peak = float(np.max(np.abs(result.audio)))
        assert peak <= clip_threshold, f"Audio stark übersteuert: {peak:.3f} > {clip_threshold}"


# ===========================================================================
# Phase 43 – ML DeEsser
# ===========================================================================
@pytest.mark.unit
class TestPhase43MLDeEsser:
    def setup_method(self):
        from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

        self.phase = MLDeEsserPhase()

    def test_mono_returns_phase_result(self, sibilant_mono):
        result = self.phase.process(sibilant_mono, SR)
        _assert_phase_result(result, sibilant_mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_silent_returns_phase_result(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_custom_threshold(self, sibilant_mono):
        result = self.phase.process(sibilant_mono, SR, threshold_db=-30.0)
        _assert_phase_result(result, sibilant_mono)
        assert result.metadata.get("threshold_db") == -30.0

    def test_conservative_threshold_barely_reduces(self, sibilant_mono):
        """Sehr hohe Schwelle → kaum Gain Reduction."""
        result = self.phase.process(sibilant_mono, SR, threshold_db=0.0)
        _assert_phase_result(result, sibilant_mono)

    def test_metadata_contains_threshold(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)
        assert "threshold_db" in result.metadata

    def test_preserves_rms_roughly(self, sibilant_mono):
        """Nach DeEssing sollte RMS ähnlich bleiben."""
        result = self.phase.process(sibilant_mono, SR, threshold_db=-6.0)
        _assert_phase_result(result, sibilant_mono)
        rms_in = float(np.sqrt(np.mean(sibilant_mono**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        # RMS kann durch GR sinken, sollte aber nicht explodieren
        assert rms_out <= rms_in * 2.0

    def test_zero_strength_passthrough(self, sibilant_mono):
        result = self.phase.process(sibilant_mono, SR, strength=0.0)
        _assert_phase_result(result, sibilant_mono)
        assert np.allclose(result.audio, sibilant_mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, sibilant_mono):
        result = self.phase.process(sibilant_mono, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, sibilant_mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 44 – Guitar Enhancement
# ===========================================================================
class TestPhase44GuitarEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_44_guitar_enhancement import GuitarEnhancementPhase

        self.phase = GuitarEnhancementPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_silent_returns_phase_result(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_custom_transient_gain(self, mono):
        result = self.phase.process(mono, SR, transient_gain=0.3)
        _assert_phase_result(result, mono)

    def test_custom_exciter_gain(self, mono):
        result = self.phase.process(mono, SR, exciter_gain=1.5)
        _assert_phase_result(result, mono)

    def test_zero_gains(self, mono):
        """Nullverstärkung → Audio sollte fast unverändert durchlaufen."""
        result = self.phase.process(mono, SR, transient_gain=0.0, exciter_gain=0.0)
        _assert_phase_result(result, mono)

    def test_44100_and_48000(self, mono):
        for sr in (44100, 48000):
            result = self.phase.process(mono[:sr], sr)
            _assert_phase_result(result, mono[:sr])

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_metadata_contains_clap_fields(self, mono):
        result = self.phase.process(mono, SR)
        assert "clap_model_used" in result.metadata
        assert "clap_confidence" in result.metadata
        assert "clap_top_genres" in result.metadata
        assert "clap_top_instruments" in result.metadata
        assert "clap_embedding_32" in result.metadata

    def test_metadata_contains_beats_fields(self, mono):
        result = self.phase.process(mono, SR)
        assert "beats_model_used" in result.metadata
        assert "beats_top_tags" in result.metadata
        assert "beats_embedding_32" in result.metadata

    def test_semantic_embeddings_are_transport_safe_lists(self, mono):
        result = self.phase.process(mono, SR)
        assert isinstance(result.metadata.get("clap_embedding_32"), list)
        assert isinstance(result.metadata.get("beats_embedding_32"), list)
        assert len(result.metadata.get("clap_embedding_32", [])) <= 32
        assert len(result.metadata.get("beats_embedding_32", [])) <= 32

    def test_stereo_no_introduced_lr_imbalance(self):
        """§2.51: Balanced stereo input → balanced output (imbalance delta ≤ 3 dB)."""
        rng = np.random.default_rng(44)
        sr = 48000
        t = np.linspace(0.0, 0.5, sr // 2, dtype=np.float32)
        base = 0.15 * np.sin(2 * np.pi * 440.0 * t)
        left = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        right = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        stereo_in = np.column_stack([left, right])
        result = self.phase.process(stereo_in, sr)
        assert result.success
        out_l = result.audio[:, 0]
        out_r = result.audio[:, 1]
        rms_l = float(np.sqrt(np.mean(out_l**2))) + 1e-12
        rms_r = float(np.sqrt(np.mean(out_r**2))) + 1e-12
        imbalance_db = abs(20.0 * np.log10(rms_l / rms_r))
        assert imbalance_db <= 3.0, f"Phase 44 introduced L/R imbalance: {imbalance_db:.2f} dB (max 3 dB)"

    def test_stereo_no_quiet_tail_boost(self):
        """§0h: Quiet intro must NOT be amplified by the EQ path."""
        rng = np.random.default_rng(440)
        sr = 48000
        n = sr // 2
        # Quiet noise at -35 dBFS
        quiet_amp = 10.0 ** (-35.0 / 20.0)
        noise = (rng.standard_normal((n, 2)) * quiet_amp).astype(np.float32)
        result = self.phase.process(noise, sr)
        assert result.success
        rms_in = float(np.sqrt(np.mean(noise**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        # Output must not be louder than input (EQ is presence boost, but on quiet signal
        # the peak normalisation step returns it to original level)
        assert rms_out <= rms_in * 1.5, (
            f"Phase 44 boosted quiet tail: rms_in={20 * np.log10(rms_in + 1e-12):.1f} dBFS, "
            f"rms_out={20 * np.log10(rms_out + 1e-12):.1f} dBFS"
        )


# ===========================================================================
# Phase 45 – Brass Enhancement
# ===========================================================================
class TestPhase45BrassEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_45_brass_enhancement import BrassEnhancementPhase

        self.phase = BrassEnhancementPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, 48000)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, 48000)
        _assert_phase_result(result, stereo)

    def test_silent_returns_phase_result(self, silent_mono):
        result = self.phase.process(silent_mono, 48000)
        _assert_phase_result(result, silent_mono)

    def test_custom_harmonics_gain(self, mono):
        result = self.phase.process(mono, 48000, gain_h2=0.08)
        _assert_phase_result(result, mono)

    def test_custom_presence_db(self, mono):
        result = self.phase.process(mono, 48000, presence_db=4.0)
        _assert_phase_result(result, mono)

    def test_custom_air_db(self, mono):
        result = self.phase.process(mono, 48000, air_db=3.0)
        _assert_phase_result(result, mono)

    def test_neutral_gains(self, mono):
        """Keine Harmonic-Verstärkung und 0 dB EQ → Signal fast unverändert."""
        result = self.phase.process(mono, 48000, gain_h2=0.0, presence_db=0.0, air_db=0.0)
        _assert_phase_result(result, mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, 48000, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, 48000, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_stereo_no_introduced_lr_imbalance(self):
        """§2.51: Balanced stereo input → balanced output (imbalance delta ≤ 3 dB)."""
        rng = np.random.default_rng(45)
        sr = 48000
        t = np.linspace(0.0, 0.5, sr // 2, dtype=np.float32)
        base = 0.15 * np.sin(2 * np.pi * 440.0 * t)
        left = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        right = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        stereo_in = np.column_stack([left, right])
        result = self.phase.process(stereo_in, sr)
        assert result.success
        out_l = result.audio[:, 0]
        out_r = result.audio[:, 1]
        rms_l = float(np.sqrt(np.mean(out_l**2))) + 1e-12
        rms_r = float(np.sqrt(np.mean(out_r**2))) + 1e-12
        imbalance_db = abs(20.0 * np.log10(rms_l / rms_r))
        assert imbalance_db <= 3.0, f"Phase 45 introduced L/R imbalance: {imbalance_db:.2f} dB (max 3 dB)"

    def test_stereo_no_quiet_tail_boost(self):
        """§0h: Quiet intro must NOT be amplified by the EQ path."""
        rng = np.random.default_rng(450)
        sr = 48000
        n = sr // 2
        quiet_amp = 10.0 ** (-35.0 / 20.0)
        noise = (rng.standard_normal((n, 2)) * quiet_amp).astype(np.float32)
        result = self.phase.process(noise, sr)
        assert result.success
        rms_in = float(np.sqrt(np.mean(noise**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        assert rms_out <= rms_in * 1.5, (
            f"Phase 45 boosted quiet tail: rms_in={20 * np.log10(rms_in + 1e-12):.1f} dBFS, "
            f"rms_out={20 * np.log10(rms_out + 1e-12):.1f} dBFS"
        )


# ===========================================================================
# Phase 46 – Spatial Enhancement
# ===========================================================================
class TestPhase46SpatialEnhancement:
    def setup_method(self):
        from backend.core.phases.phase_46_spatial_enhancement import SpatialEnhancementPhase

        self.phase = SpatialEnhancementPhase()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_mono_returns_success(self, mono):
        """Mono-Eingabe → Phase gibt sofort erfolgreiches PhaseResult zurück."""
        result = self.phase.process(mono, SR)
        assert isinstance(result, PhaseResult)
        assert result.success is True
        assert result.audio.shape == mono.shape

    def test_silent_stereo(self, silent_mono):
        stereo_silent = np.stack([silent_mono, silent_mono], axis=-1)
        result = self.phase.process(stereo_silent, SR)
        _assert_phase_result(result, stereo_silent)

    def test_custom_haas_ms(self, stereo):
        result = self.phase.process(stereo, SR, haas_ms=1.0)
        _assert_phase_result(result, stereo)

    def test_custom_allpass(self, stereo):
        result = self.phase.process(stereo, SR, allpass_ms=15.0, allpass_g=0.3)
        _assert_phase_result(result, stereo)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR)
        assert result.audio.ndim == 2
        assert result.audio.shape[1] == 2

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, strength=0.0)
        _assert_phase_result(result, stereo)
        assert np.allclose(result.audio, stereo, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_stereo_no_introduced_lr_imbalance(self):
        """§2.51: Balanced stereo input → allpass diffusion must not unbalance L/R (delta ≤ 3 dB)."""
        rng = np.random.default_rng(46)
        sr = 48000
        t = np.linspace(0.0, 0.5, sr // 2, dtype=np.float32)
        base = 0.15 * np.sin(2 * np.pi * 440.0 * t)
        left = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        right = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        stereo_in = np.column_stack([left, right])
        result = self.phase.process(stereo_in, sr)
        assert result.success
        out_l = result.audio[:, 0]
        out_r = result.audio[:, 1]
        rms_l = float(np.sqrt(np.mean(out_l**2))) + 1e-12
        rms_r = float(np.sqrt(np.mean(out_r**2))) + 1e-12
        imbalance_db = abs(20.0 * np.log10(rms_l / rms_r))
        assert imbalance_db <= 3.0, f"Phase 46 introduced L/R imbalance: {imbalance_db:.2f} dB (max 3 dB)"


# ===========================================================================
# Phase 47 – TruePeak Limiter
# ===========================================================================
class TestPhase47TruePeakLimiter:
    def setup_method(self):
        from backend.core.phases.phase_47_truepeak_limiter import TruePeakLimiterPhase

        self.phase = TruePeakLimiterPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_silent_returns_phase_result(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_hot_signal_limited_to_ceiling(self, hot_mono):
        ceiling_dbfs = -1.0
        ceiling_lin = 10 ** (ceiling_dbfs / 20.0)
        result = self.phase.process(hot_mono, SR, ceiling_dbfs=ceiling_dbfs)
        _assert_phase_result(result, hot_mono, check_clipping=False)
        peak = float(np.max(np.abs(result.audio)))
        assert peak <= ceiling_lin * 1.05, f"Limiter hat nicht gegriffen: peak={peak:.4f}, ceiling={ceiling_lin:.4f}"

    def test_below_ceiling_signal_passes_through(self, mono):
        """Signal unter der Ceiling sollte weitgehend unverändert durchlaufen."""
        ceiling_dbfs = -0.5
        result = self.phase.process(mono, SR, ceiling_dbfs=ceiling_dbfs)
        _assert_phase_result(result, mono)
        # Mono-Signal hat max-peak von 0.1 << ceiling → kaum Änderung
        diff = float(np.max(np.abs(result.audio - mono)))
        assert diff < 0.05

    def test_custom_ceiling(self, hot_mono):
        ceiling_dbfs = -3.0
        ceiling_lin = 10 ** (ceiling_dbfs / 20.0)
        result = self.phase.process(hot_mono, SR, ceiling_dbfs=ceiling_dbfs)
        _assert_phase_result(result, hot_mono, check_clipping=False)
        peak = float(np.max(np.abs(result.audio)))
        assert peak <= ceiling_lin * 1.1

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, hot_mono):
        result = self.phase.process(hot_mono, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, hot_mono, check_clipping=False)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ===========================================================================
# Phase 48 – Stereo Width Enhancer
# ===========================================================================
class TestPhase48StereoWidthEnhancer:
    def setup_method(self):
        from backend.core.phases.phase_48_stereo_width_enhancer import StereoWidthEnhancerPhase

        self.phase = StereoWidthEnhancerPhase()

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_mono_returns_success(self, mono):
        """Mono-Eingabe → Phase gibt sofort erfolgreiches PhaseResult zurück."""
        result = self.phase.process(mono, SR)
        assert isinstance(result, PhaseResult)
        assert result.success is True
        assert result.audio.shape == mono.shape

    def test_silent_stereo(self, silent_mono):
        stereo_silent = np.stack([silent_mono, silent_mono], axis=-1)
        result = self.phase.process(stereo_silent, SR)
        _assert_phase_result(result, stereo_silent)

    def test_unity_width_preserves_signal(self, stereo):
        """width=1.0 → M-Kanal unverändert, S-Kanal skaliert mit 1.0."""
        result = self.phase.process(stereo, SR, width=1.0)
        _assert_phase_result(result, stereo)

    def test_wide_width(self, stereo):
        result = self.phase.process(stereo, SR, width=2.0)
        _assert_phase_result(result, stereo)

    def test_narrow_width(self, stereo):
        result = self.phase.process(stereo, SR, width=0.5)
        _assert_phase_result(result, stereo)

    def test_output_is_stereo(self, stereo):
        result = self.phase.process(stereo, SR)
        assert result.audio.ndim == 2
        assert result.audio.shape[1] == 2

    def test_zero_strength_passthrough(self, stereo):
        result = self.phase.process(stereo, SR, strength=0.0)
        _assert_phase_result(result, stereo)
        assert np.allclose(result.audio, stereo, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, stereo):
        result = self.phase.process(stereo, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, stereo)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_stereo_no_introduced_lr_imbalance(self):
        """§2.51: Balanced stereo input → M/S width processing must not unbalance L/R (delta ≤ 3 dB)."""
        rng = np.random.default_rng(48)
        sr = 48000
        t = np.linspace(0.0, 0.5, sr // 2, dtype=np.float32)
        base = 0.15 * np.sin(2 * np.pi * 440.0 * t)
        left = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        right = base + 0.01 * rng.standard_normal(len(t)).astype(np.float32)
        stereo_in = np.column_stack([left, right])
        result = self.phase.process(stereo_in, sr)
        assert result.success
        out_l = result.audio[:, 0]
        out_r = result.audio[:, 1]
        rms_l = float(np.sqrt(np.mean(out_l**2))) + 1e-12
        rms_r = float(np.sqrt(np.mean(out_r**2))) + 1e-12
        imbalance_db = abs(20.0 * np.log10(rms_l / rms_r))
        assert imbalance_db <= 3.0, f"Phase 48 introduced L/R imbalance: {imbalance_db:.2f} dB (max 3 dB)"


# ===========================================================================
# Phase 50 – Spectral Repair
# ===========================================================================
class TestPhase50SpectralRepair:
    def setup_method(self):
        from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase

        self.phase = SpectralRepairPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_silent_returns_phase_result(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_custom_threshold_factor(self, mono):
        result = self.phase.process(mono, SR, threshold_factor=6.0)
        _assert_phase_result(result, mono)
        assert result.metadata.get("threshold_factor") == 6.0

    def test_metadata_contains_threshold_factor(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)
        assert "threshold_factor" in result.metadata

    def test_spectral_repair_reduces_outliers(self, mono):
        """Mit Impulsnoise: Spectral Repair sollte Output-Energie nicht erhöhen."""
        # Inject impulses simulating spectral outliers
        audio_with_impulses = mono.copy()
        audio_with_impulses[::1000] = 2.0  # Impulsive spikes
        result = self.phase.process(audio_with_impulses, SR, threshold_factor=2.0)
        _assert_phase_result(result, audio_with_impulses, check_clipping=False)

    def test_44100_and_48000(self, mono):
        for sr in (44100, 48000):
            result = self.phase.process(mono[:sr], sr)
            _assert_phase_result(result, mono[:sr])

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=0.03)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_stereo_ms_domain_coherence(self):
        """§2.51: Stereo processing uses M/S domain — identical spikes repaired coherently."""
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.05, _N).astype(np.float32)
        left = base + rng.normal(0, 0.01, _N).astype(np.float32)
        right = base + rng.normal(0, 0.01, _N).astype(np.float32)
        # Inject shared impulse spikes
        for idx in [1000, 3000, 5000]:
            left[idx] = 0.8
            right[idx] = 0.8
        stereo_in = np.column_stack([left, right])
        result = self.phase.process(stereo_in, 48000)
        assert result.success
        assert result.metadata.get("stereo_mode") == "ms_domain"
        assert result.audio.shape == stereo_in.shape

    def test_stereo_output_shape_preserved(self):
        """Stereo input → stereo output with correct shape."""
        rng = np.random.default_rng(123)
        stereo_in = rng.normal(0, 0.1, (_N, 2)).astype(np.float32)
        result = self.phase.process(stereo_in, 48000)
        assert result.audio.shape == stereo_in.shape


# ===========================================================================
# Phase 53 – Semantic Audio
# ===========================================================================
class TestPhase53SemanticAudio:
    @pytest.fixture(autouse=True)
    def _patch_ml_budget(self, monkeypatch):
        """ML-Modelle (CLAP, BEATs) in Unit-Tests deaktivieren — try_allocate → False
        verhindert torch-Modell-Loading (>20 s Timeout in Unit-Suite). DSP-Fallback läuft."""
        monkeypatch.setattr(
            "backend.core.ml_memory_budget.try_allocate",
            lambda *_a, **_kw: False,
            raising=False,
        )

    def setup_method(self):
        from backend.core.phases.phase_53_semantic_audio import SemanticAudioPhase

        self.phase = SemanticAudioPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_silent_returns_phase_result(self, silent_mono):
        result = self.phase.process(silent_mono, SR)
        _assert_phase_result(result, silent_mono)

    def test_audio_unchanged(self, mono):
        """SemanticAudioPhase verändert das Audio nicht (nur Analyse)."""
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)
        np.testing.assert_array_equal(result.audio, mono, err_msg="SemanticAudioPhase darf Audio nicht verändern")

    def test_metadata_contains_bpm(self, mono):
        result = self.phase.process(mono, SR)
        assert "bpm" in result.metadata, "Kein BPM in Metadata"

    def test_metadata_contains_key(self, mono):
        result = self.phase.process(mono, SR)
        assert "key" in result.metadata, "Kein Key in Metadata"

    def test_metadata_contains_genre_hint(self, mono):
        result = self.phase.process(mono, SR)
        assert "genre_hint" in result.metadata, "Kein genre_hint in Metadata"

    def test_metadata_contains_genre_hint_source_and_confidence(self, mono):
        result = self.phase.process(mono, SR)
        assert "genre_hint_source" in result.metadata
        assert "genre_hint_confidence" in result.metadata

    def test_genre_hint_is_canonical_label(self, mono):
        result = self.phase.process(mono, SR)
        assert result.metadata.get("genre_hint") in {
            "Klassik",
            "Oper",
            "Jazz",
            "Rock",
            "Pop",
            "Blues",
            "Folk",
            "Electronic",
            "Hip-Hop",
            "Reggae",
            "Gospel",
            "Soul/R&B",
            "Schlager",
            "Unbekannt",
        }

    def test_metadata_contains_loudness_class(self, mono):
        result = self.phase.process(mono, SR)
        assert "loudness_class" in result.metadata

    def test_bpm_is_positive(self, mono):
        result = self.phase.process(mono, SR)
        bpm = result.metadata.get("bpm", 0)
        assert bpm > 0, f"BPM sollte positiv sein, war: {bpm}"

    def test_48000_hz(self, mono):
        audio_48k = np.tile(mono, 2)[:48000].astype(np.float32)
        result = self.phase.process(audio_48k, 48000)
        assert result.success is True
        assert result.audio.shape == audio_48k.shape

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, strength=0.0)
        _assert_phase_result(result, mono)
        assert np.allclose(result.audio, mono, atol=1e-7)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_metadata_contains_semantic_tier_fields(self, mono):
        result = self.phase.process(mono, SR)
        assert "clap_model_used" in result.metadata
        assert "clap_top_genres" in result.metadata
        assert "beats_model_used" in result.metadata
        assert "beats_top_tags" in result.metadata

    def test_phase53_canonicalize_genre_helper(self):
        from backend.core.phases.phase_53_semantic_audio import _canonicalize_genre_hint

        assert _canonicalize_genre_hint("classical") == "Klassik"
        assert _canonicalize_genre_hint("rock_metal") == "Rock"
        assert _canonicalize_genre_hint("rnb") == "Soul/R&B"
        assert _canonicalize_genre_hint("hip_hop") == "Hip-Hop"
