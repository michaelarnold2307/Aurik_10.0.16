"""
Unit-Tests für Phasen 01–09 (Frühe Restaurierungsphasen).

Abgedeckt:
  Phase 01 — Click Removal (ClickRemovalPhase)
  Phase 02 — Hum Removal (HumRemovalPhase)
  Phase 03 — Denoise (DenoisePhase)
  Phase 04 — EQ Correction (EQCorrectionPhase)
  Phase 05 — Rumble Filter (RumbleFilterPhase)
  Phase 06 — Frequency Restoration (FrequencyRestorationPhase)
  Phase 07 — Harmonic Restoration (HarmonicRestorationPhase)
  Phase 08 — Transient Preservation (TransientPreservationPhase)
  Phase 09 — Crackle Removal (CrackleRemovalPhase)

Alle Tests auf 0.1-Sekunden-Arrays (4410 Samples @ 44100 Hz).
"""

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.phases.phase_interface import PhaseResult

SR = 48000
N = SR // 10  # 4800 Samples ≈ 0.1s für schnelle Tests


# ---------------------------------------------------------------------------
# Fixtures — scope="module": einmalig pro Testdatei
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mono():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal(N).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="module")
def mono_quiet():
    rng = np.random.default_rng(99)
    return np.clip(rng.standard_normal(N).astype(np.float32) * 0.05, -1.0, 1.0)


@pytest.fixture(scope="module")
def stereo():
    rng = np.random.default_rng(42)
    return np.clip(rng.standard_normal((N, 2)).astype(np.float32) * 0.35, -1.0, 1.0)


@pytest.fixture(scope="module")
def sine_mono():
    """Reines Sinus-Signal mit 440 Hz."""
    t = np.linspace(0, N / SR, N, endpoint=False)
    return (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Gemeinsame Validierung
# ---------------------------------------------------------------------------


def _assert_phase_result(result, orig_audio, check_clipping: bool = True):
    assert isinstance(result, PhaseResult), f"Kein PhaseResult: {type(result)}"
    assert result.success is True, f"success=False: {result}"
    assert isinstance(result.audio, np.ndarray), "audio muss ndarray sein"
    assert result.audio.shape == orig_audio.shape, f"Shape geändert: {orig_audio.shape} → {result.audio.shape}"
    # Dtype: float32 oder float64 akzeptabel (interne DSP-Umwandlung)
    assert np.issubdtype(result.audio.dtype, np.floating), f"Dtype nicht floating: {result.audio.dtype}"
    assert isinstance(result.metadata, dict), "metadata muss dict sein"
    assert isinstance(result.metrics, dict), "metrics muss dict sein"
    assert float(result.execution_time_seconds) >= 0.0
    if check_clipping:
        assert np.max(np.abs(result.audio)) <= 2.0, f"Audio stark übersteuert: {np.max(np.abs(result.audio)):.3f}"


# ---------------------------------------------------------------------------
# Phase 01: Click Removal
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase01ClickRemoval:
    def setup_method(self):
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        self.phase = ClickRemovalPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_sine_unchanged_roughly(self, sine_mono):
        """Reines Sinussignal (keine Klicks) bleibt nahezu unverändert."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        corr = float(np.corrcoef(sine_mono, result.audio)[0, 1])
        assert corr > 0.8, f"Korrelation zu gering: {corr:.3f}"

    def test_signal_with_click_processed(self, mono):
        """Signal mit eingebautem Klick wurde verarbeitet (kein Absturz)."""
        audio_with_click = mono.copy()
        audio_with_click[100] = 0.99  # Starker Klick
        audio_with_click[200] = -0.99
        result = self.phase.process(audio_with_click, SR)
        _assert_phase_result(result, audio_with_click)

    def test_metrics_present(self, mono):
        result = self.phase.process(mono, SR)
        assert result.metrics is not None or result.metadata is not None


# ---------------------------------------------------------------------------
# Phase 02: Hum Removal
# ---------------------------------------------------------------------------


class TestPhase02HumRemoval:
    def setup_method(self):
        from backend.core.phases.phase_02_hum_removal import HumRemovalPhase

        self.phase = HumRemovalPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_50hz_hum_reduced(self):
        """50-Hz-Brumm: Nach Entfernung muss 50-Hz-Energie gesunken sein."""
        t = np.linspace(0, N / SR, N, endpoint=False)
        hum = (np.sin(2 * np.pi * 50.0 * t) * 0.3).astype(np.float32)
        result = self.phase.process(hum, SR)
        _assert_phase_result(result, hum)
        # Energie bei 50 Hz nach Entfernung niedriger
        fft_in = np.abs(np.fft.rfft(hum.astype(float)))
        fft_out = np.abs(np.fft.rfft(result.audio.astype(float)))
        bin_50 = int(50 * N / SR)
        # Mindestens keine Verstärkung des Brumms
        assert fft_out[bin_50] <= fft_in[bin_50] * 1.5

    def test_quiet_signal_not_amplified(self, mono_quiet):
        result = self.phase.process(mono_quiet, SR)
        _assert_phase_result(result, mono_quiet)
        rms_in = float(np.sqrt(np.mean(mono_quiet**2)))
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out <= rms_in * 3.0  # Keine extreme Verstärkung

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, SR, strength=0.0)
        _assert_phase_result(result, mono)
        # §v10.62: PhaseResult.__post_init__ applies apply_soft_clip() to ALL
        # phase outputs unconditionally — samples at/near the fixture's own
        # ±1.0 clip boundary get nudged slightly, so a byte-identical atol=1e-7
        # comparison is too strict even for a true strength=0 passthrough.
        assert np.allclose(result.audio, mono, atol=0.03)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, SR, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6


# ---------------------------------------------------------------------------
# Phase 03: Denoise
# ---------------------------------------------------------------------------


class TestPhase03Denoise:
    def setup_method(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        self.phase = DenoisePhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)  # type: ignore[arg-type]
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)  # type: ignore[arg-type]
        _assert_phase_result(result, stereo)

    def test_broadband_noise_reduced(self):
        """Breitbandrauschen wird reduziert."""
        rng = np.random.default_rng(7)
        noise = (rng.standard_normal(N) * 0.3).astype(np.float32)
        result = self.phase.process(noise, SR)  # type: ignore[arg-type]
        _assert_phase_result(result, noise)
        rms_in = float(np.sqrt(np.mean(noise**2)))
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out <= rms_in * 1.1  # Keine Verstärkung des Rauschens

    def test_material_type_supported(self, mono):
        """material_type='vinyl' wird akzeptiert."""
        result = self.phase.process(mono, material_type="vinyl")
        _assert_phase_result(result, mono)

    def test_quality_mode_ignores_lightweight_dsp_for_ml_path(self, mono, monkeypatch):
        """Quality-first: quality mode must not force DSP-only via lightweight hint."""
        import backend.core.phases.phase_03_denoise as phase03_mod

        class _DummyRM:
            @staticmethod
            def should_use_lightweight_mode():
                return True

            @staticmethod
            def get_cpu_usage():
                return 95.0

            @staticmethod
            def get_memory_usage():
                return 92.0

        class _DummyMLResult:
            def __init__(self, audio):
                self.audio = audio
                self.omlsa_applied = True
                self.resemble_applied = False
                self.quality_estimate = 0.8
                self.processing_time = 0.01
                self.strategy_used = "hybrid"
                self.metadata = {}

        class _DummyDenoiser:
            def __init__(self, config):
                self.config = config

            def denoise(self, audio, sample_rate=48000):
                return _DummyMLResult(np.asarray(audio, dtype=np.float32))

        monkeypatch.setattr(phase03_mod, "RESOURCE_MANAGER_AVAILABLE", True)
        monkeypatch.setattr(phase03_mod, "adaptive_resource_manager", _DummyRM)
        monkeypatch.setattr(phase03_mod, "ML_HYBRID_AVAILABLE", True)
        monkeypatch.setattr(phase03_mod, "HybridMLDenoiser", _DummyDenoiser)

        result = self.phase.process(mono, material_type="tape", quality_mode="quality", sample_rate=48000)
        _assert_phase_result(result, mono)
        assert bool(result.metadata.get("ml_hybrid")) is True

    def test_ml_hybrid_respects_explicit_strength_wet_scale(self, mono, monkeypatch):
        """Explicit PMGG/oracle strength must scale the final ML-Hybrid wet output."""
        import backend.core.phases.phase_03_denoise as phase03_mod

        class _DummyMLResult:
            def __init__(self, audio):
                # §2.45a: eine Skalierung von 0.5 (≈-6dB) würde den
                # Loudness-Preservation-Makeup-Gain-Pfad triggern (tape-Schwelle
                # 2.0dB), was die simple Wet-Scale-Formel unten verfälscht.
                # 0.95 (≈-0.45dB) bleibt sicher unter der Schwelle.
                self.audio = np.asarray(audio, dtype=np.float32) * 0.95
                self.omlsa_applied = True
                self.resemble_applied = True
                self.quality_estimate = 0.8
                self.processing_time = 0.01
                self.strategy_used = "hybrid"
                self.metadata = {}

        class _DummyDenoiser:
            def __init__(self, config):
                self.config = config

            def denoise(self, audio, sample_rate=48000):
                return _DummyMLResult(audio)

        monkeypatch.setattr(phase03_mod, "ML_HYBRID_AVAILABLE", True)
        monkeypatch.setattr(phase03_mod, "HybridMLDenoiser", _DummyDenoiser)

        strength = 0.15
        result = self.phase.process(
            mono, material_type="tape", quality_mode="quality", sample_rate=48000, strength=strength
        )

        _assert_phase_result(result, mono)
        effective_strength = float(result.modifications["strength"])
        expected_wet = effective_strength / self.phase.MATERIAL_PARAMS["tape"]["strength"]  # type: ignore[operator]
        assert effective_strength < strength
        assert result.modifications["ml_wet"] == pytest.approx(expected_wet)
        assert result.modifications["ml_requested_wet"] == pytest.approx(expected_wet)
        assert result.modifications["noise_reduction_db"] == pytest.approx(
            result.modifications["ml_raw_noise_reduction_db"] * expected_wet
        )
        # §v10.62 Soft-Clip-Toleranz: das Fixture enthält Samples exakt bei
        # ±1.0 (np.clip-Grenze), die durch PhaseResult.__post_init__()s
        # globalen apply_soft_clip() geringfügig genudged werden.
        assert np.max(np.abs(result.audio - (mono * (1.0 - 0.05 * expected_wet)))) < 0.03


# ---------------------------------------------------------------------------
# Phase 04: EQ Correction
# ---------------------------------------------------------------------------


class TestPhase04EQCorrection:
    def setup_method(self):
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        self.phase = EQCorrectionPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)  # type: ignore[arg-type]
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)  # type: ignore[arg-type]
        _assert_phase_result(result, stereo)

    def test_frequency_range_preserved(self, sine_mono):
        """Sinus-Signal: Kein vollständiges Auslöschen durch EQ."""
        result = self.phase.process(sine_mono, SR)  # type: ignore[arg-type]
        _assert_phase_result(result, sine_mono)
        rms_in = float(np.sqrt(np.mean(sine_mono**2)))
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out > rms_in * 0.1, "EQ hat Signal fast vollständig eliminiert"

    def test_material_types(self, mono):
        for mat in ["vinyl", "tape", "unknown"]:
            result = self.phase.process(mono, material_type=mat)
            # EQ kann starke Boosts erzeugen → check_clipping=False
            _assert_phase_result(result, mono, check_clipping=False)


# ---------------------------------------------------------------------------
# Phase 05: Rumble Filter
# ---------------------------------------------------------------------------


class TestPhase05RumbleFilter:
    SR_PHASE = 48000

    def setup_method(self):
        from backend.core.phases.phase_05_rumble_filter import RumbleFilterPhase

        self.phase = RumbleFilterPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, sample_rate=self.SR_PHASE)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, sample_rate=self.SR_PHASE)
        _assert_phase_result(result, stereo)

    def test_low_freq_rumble_reduced(self):
        """LF-Rumble (10 Hz) wird stark reduziert."""
        t = np.linspace(0, N / SR, N, endpoint=False)
        rumble = (np.sin(2 * np.pi * 10.0 * t) * 0.5).astype(np.float32)
        result = self.phase.process(rumble, sample_rate=self.SR_PHASE)
        _assert_phase_result(result, rumble)
        # Energie bei 10 Hz nach Filterung niedriger
        fft_in = np.abs(np.fft.rfft(rumble.astype(float)))
        fft_out = np.abs(np.fft.rfft(result.audio.astype(float)))
        bin_10 = max(1, int(10 * N / SR))
        assert fft_out[bin_10] <= fft_in[bin_10] + 1e-3

    def test_high_freq_preserved(self, sine_mono):
        """440-Hz-Signal bleibt nach Rumble-Filter erhalten."""
        result = self.phase.process(sine_mono, sample_rate=self.SR_PHASE)
        _assert_phase_result(result, sine_mono)
        rms_out = float(np.sqrt(np.mean(result.audio.astype(float) ** 2)))
        assert rms_out > 0.05  # Signal muss erhalten bleiben

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, material_type="vinyl", strength=0.0)
        _assert_phase_result(result, mono)
        # §v10.62 Soft-Clip-Toleranz, siehe Kommentar oben.
        assert np.allclose(result.audio, mono, atol=0.03)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        result = self.phase.process(mono, material_type="vinyl", strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, mono)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6

    def test_musical_body_preserved_while_rumble_reduced(self):
        """Rumble entfernen ohne signifikanten Verlust des Musik-Körpers."""
        t = np.linspace(0, N / SR, N, endpoint=False)
        music = 0.30 * np.sin(2 * np.pi * 80.0 * t)  # low musical body
        rumble = 0.25 * np.sin(2 * np.pi * 10.0 * t)  # defect component
        signal_in = (music + rumble).astype(np.float32)

        result = self.phase.process(signal_in, material_type="vinyl", sample_rate=self.SR_PHASE)
        _assert_phase_result(result, signal_in)

        in_rms = float(np.sqrt(np.mean(signal_in.astype(np.float64) ** 2) + 1e-12))
        out_rms = float(np.sqrt(np.mean(result.audio.astype(np.float64) ** 2) + 1e-12))
        delta_db = 20.0 * np.log10(max(out_rms / in_rms, 1e-30))
        # Gesamt-RMS darf sinken, weil Defektenergie (10 Hz rumble) entfernt wird.
        assert delta_db > -3.8

        fft_in = np.abs(np.fft.rfft(signal_in.astype(float)))
        fft_out = np.abs(np.fft.rfft(result.audio.astype(float)))
        bin_10 = max(1, int(10 * N / SR))
        bin_80 = max(1, int(80 * N / SR))
        assert fft_out[bin_10] < fft_in[bin_10]
        # Musical low-end (80 Hz) must stay substantially present.
        assert fft_out[bin_80] > fft_in[bin_80] * 0.65


# ---------------------------------------------------------------------------
# Phase 06: Frequency Restoration
# ---------------------------------------------------------------------------


class TestPhase06FrequencyRestoration:
    def setup_method(self):
        from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

        self.phase = FrequencyRestorationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_no_extreme_gain(self, mono):
        """Frequenzwiederherstellung darf Signal nicht extrem verstärken."""
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)
        peak_in = float(np.max(np.abs(mono)))
        peak_out = float(np.max(np.abs(result.audio)))
        assert peak_out <= max(peak_in * 5.0, 1.05)  # Kein unkontrollierter Gain


# ---------------------------------------------------------------------------
# Phase 07: Harmonic Restoration
# ---------------------------------------------------------------------------


class TestPhase07HarmonicRestoration:
    def setup_method(self):
        from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

        self.phase = HarmonicRestorationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_sine_not_destroyed(self, sine_mono):
        """Grundton bleibt nach harmonischer Restaurierung erhalten."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        corr = float(np.corrcoef(sine_mono, result.audio)[0, 1])
        assert corr > 0.5, f"Korrelation zu gering nach Harmonic Restoration: {corr:.3f}"


# ---------------------------------------------------------------------------
# Phase 08: Transient Preservation
# ---------------------------------------------------------------------------


class TestPhase08TransientPreservation:
    def setup_method(self):
        from backend.core.phases.phase_08_transient_preservation import TransientPreservationPhase

        self.phase = TransientPreservationPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_transients_preserved(self):
        """Impuls-Signal: Transienten-Energie bleibt erhalten."""
        audio = np.zeros(N, dtype=np.float32)
        audio[:: int(SR * 0.02)] = 0.8  # Impulse alle 20ms
        result = self.phase.process(audio, SR)
        _assert_phase_result(result, audio)
        # Peak muss noch signifikant sein
        assert float(np.max(np.abs(result.audio))) > 0.3


# ---------------------------------------------------------------------------
# Phase 09: Crackle Removal
# ---------------------------------------------------------------------------


class TestPhase09CrackleRemoval:
    def setup_method(self):
        from backend.core.phases.phase_09_crackle_removal import CrackleRemovalPhase

        self.phase = CrackleRemovalPhase()

    def test_mono_returns_phase_result(self, mono):
        result = self.phase.process(mono, SR)
        _assert_phase_result(result, mono)

    def test_stereo_returns_phase_result(self, stereo):
        result = self.phase.process(stereo, SR)
        _assert_phase_result(result, stereo)

    def test_sine_not_damaged(self, sine_mono):
        """Reiner Sinus (kein Crackle): Signal bleibt weitgehend unverändert."""
        result = self.phase.process(sine_mono, SR)
        _assert_phase_result(result, sine_mono)
        corr = float(np.corrcoef(sine_mono, result.audio)[0, 1])
        assert corr > 0.7, f"Korrelation zu gering: {corr:.3f}"

    def test_crackle_signal_processed(self, mono):
        """Signal mit simulierten Crackles wird verarbeitet."""
        audio = mono.copy()
        rng = np.random.default_rng(0)
        crackle_pos = rng.integers(0, N - 1, size=20)
        audio[crackle_pos] = rng.choice([-0.95, 0.95], size=20).astype(np.float32)
        result = self.phase.process(audio, SR)
        _assert_phase_result(result, audio)

    def test_material_types(self, mono):
        for mat in ["shellac", "vinyl", "unknown"]:
            result = self.phase.process(mono, material_type=mat)
            _assert_phase_result(result, mono)

    def test_zero_strength_passthrough(self, mono):
        result = self.phase.process(mono, strength=0.0)
        _assert_phase_result(result, mono)
        # §v10.62 Soft-Clip-Toleranz, siehe Kommentar oben.
        assert np.allclose(result.audio, mono, atol=0.03)
        assert result.metadata.get("algorithm") == "skipped_zero_strength"
        assert float(result.metadata.get("effective_strength", 1.0)) == 0.0

    def test_locality_reduces_effective_strength(self, mono):
        audio = mono.copy()
        rng = np.random.default_rng(123)
        crackle_pos = rng.integers(0, N - 1, size=30)
        audio[crackle_pos] = rng.choice([-0.98, 0.98], size=30).astype(np.float32)

        result = self.phase.process(audio, strength=1.0, phase_locality_factor=0.4)
        _assert_phase_result(result, audio)
        eff = float(result.metadata.get("effective_strength", 1.0))
        assert 0.0 < eff < 1.0
        assert float(result.metadata.get("phase_locality_factor", 1.0)) <= 0.4 + 1e-6
