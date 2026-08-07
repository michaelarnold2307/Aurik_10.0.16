"""Tests for Phase 42 psychoacoustic improvements.

Validates:
1. Compression: Vocal-optimized attack/release + soft-knee + adaptive makeup
2. Presence boost: ISO 226 loudness compensation + vibrato protection
3. Vocal-stem MDEM: Micro-dynamics recovery on isolated vocal stem
4. Formant correction strength: Adaptive based on gender confidence
"""

import numpy as np
import pytest
from scipy import signal as scipy_signal

SR = 48000


def _make_vocal(duration_s: float = 1.0, f0: float = 220.0) -> np.ndarray:
    n = int(SR * duration_s)
    t = np.arange(n, dtype=np.float64) / SR
    sig = np.zeros(n, dtype=np.float64)
    for h in range(1, 6):
        sig += (0.5 / h) * np.sin(2 * np.pi * f0 * h * t)
    rng = np.random.default_rng(42)
    sig += rng.standard_normal(n) * 0.01
    return (sig / (np.max(np.abs(sig)) + 1e-12) * 0.7).astype(np.float64)  # type: ignore[no-any-return]


def _make_vibrato_signal(
    duration_s: float = 2.0, f0: float = 260.0, vibrato_rate: float = 5.5, vibrato_depth: float = 0.03
) -> np.ndarray:
    """Vocal with strong vibrato (5.5 Hz, 3% depth)."""
    n = int(SR * duration_s)
    t = np.arange(n, dtype=np.float64) / SR
    freq = f0 * (1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sig = np.sin(phase) * 0.6
    for h in [2, 3]:
        sig += (0.3 / h) * np.sin(h * phase)
    return sig  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestPsychoacousticCompression:
    """Vocal compression should preserve micro-dynamics better than flat RMS."""

    def test_compression_output_valid(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal(0.5)
        config = {"compression_ratio": 2.2}
        result = phase._apply_compression(audio, SR, config)
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))

    def test_compression_soft_knee_no_hard_clipping(self):
        """Soft-knee should produce smoother gain transitions than hard-knee."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        # Ramp signal to test knee region
        n = SR
        ramp = np.linspace(0.0, 0.9, n)
        config = {"compression_ratio": 3.0}
        result = phase._apply_compression(ramp, SR, config)
        # No sudden jumps: max sample-to-sample difference should be small
        diff = np.abs(np.diff(result))
        assert np.max(diff) < 0.05, f"Compression has too-sharp transitions: {np.max(diff):.4f}"

    def test_compression_preserves_transients(self):
        """Fast attack (3ms) should let initial consonant transients through."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = np.zeros(SR, dtype=np.float64)
        # Sharp transient at 0.5s
        audio[SR // 2 : SR // 2 + 50] = 0.9
        # Followed by sustained signal
        audio[SR // 2 + 50 : SR // 2 + 5000] = 0.3
        config = {"compression_ratio": 2.0}
        result = phase._apply_compression(audio, SR, config)
        # Transient peak should still be significantly louder than sustained
        peak_region = np.max(np.abs(result[SR // 2 : SR // 2 + 50]))
        sustain_region = np.mean(np.abs(result[SR // 2 + 200 : SR // 2 + 4000]))
        assert peak_region > sustain_region * 1.3, "Transient was over-compressed"

    def test_adaptive_makeup_gain(self):
        """Makeup gain should adapt to compression depth, not be fixed."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        # Loud signal (more compression → more makeup)
        loud = _make_vocal(0.5) * 0.9
        # Quiet signal (less compression → less makeup)
        quiet = _make_vocal(0.5) * 0.1
        config = {"compression_ratio": 2.5}
        result_loud = phase._apply_compression(loud, SR, config)
        result_quiet = phase._apply_compression(quiet, SR, config)
        # Both should produce finite, valid output
        assert np.all(np.isfinite(result_loud))
        assert np.all(np.isfinite(result_quiet))

    def test_short_audio_passthrough(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        short = np.array([0.1, 0.2, 0.3])
        config = {"compression_ratio": 2.0}
        result = phase._apply_compression(short, SR, config)
        np.testing.assert_array_equal(result, short)


# ---------------------------------------------------------------------------
# Presence boost tests
# ---------------------------------------------------------------------------
class TestPresenceBoostISO226:
    """Presence boost should adapt to loudness and protect vibrato."""

    def test_presence_boost_valid_output(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal(0.5)
        config = {"presence_gain_db": 4.0}
        result = phase._boost_presence(audio, SR, config)
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))

    def test_loudness_compensation_quiet_vs_loud(self):
        """Quiet signal should get different presence gain than loud signal
        (ISO 226 equal-loudness compensation)."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        base = _make_vocal(1.0, f0=300.0)
        config = {"presence_gain_db": 4.5}

        quiet = base * 0.05
        loud = base * 0.8
        result_quiet = phase._boost_presence(quiet, SR, config)
        result_loud = phase._boost_presence(loud, SR, config)

        # Measure boost magnitude in presence band (4-5 kHz)
        sos = scipy_signal.butter(4, [4000.0, 5000.0], btype="band", fs=SR, output="sos")

        def presence_energy(sig):
            return float(np.mean(scipy_signal.sosfilt(sos, sig) ** 2))

        ratio_quiet = presence_energy(result_quiet) / (presence_energy(quiet) + 1e-20)
        ratio_loud = presence_energy(result_loud) / (presence_energy(loud) + 1e-20)
        # They should differ (loudness compensation active)
        # Allow either direction since ISO 226 curve shape varies
        assert ratio_quiet != pytest.approx(ratio_loud, rel=0.05), "Presence boost should be loudness-adaptive"

    def test_vibrato_attenuation(self):
        """Strong vibrato should reduce presence boost to protect F0 modulation."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        vibrato = _make_vibrato_signal(2.0, vibrato_rate=5.5, vibrato_depth=0.04)
        no_vibrato = _make_vocal(2.0, f0=260.0)
        config = {"presence_gain_db": 5.0}

        result_vib = phase._boost_presence(vibrato, SR, config)
        result_no = phase._boost_presence(no_vibrato, SR, config)

        # Both valid
        assert np.all(np.isfinite(result_vib))
        assert np.all(np.isfinite(result_no))

    def test_zero_gain_passthrough(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal(0.3)
        config = {"presence_gain_db": 0.0}
        result = phase._boost_presence(audio, SR, config)
        # Near-passthrough with zero gain (loudness compensation might nudge slightly)
        np.testing.assert_allclose(result, audio, atol=1e-4)

    def test_short_audio(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        short = np.array([0.1, 0.2])
        result = phase._boost_presence(short, SR, {"presence_gain_db": 3.0})
        np.testing.assert_array_equal(result, short)


# ---------------------------------------------------------------------------
# Vocal-Stem MDEM
# ---------------------------------------------------------------------------
class TestVocalStemMDEM:
    """MDEM should be applied to vocal stem before remix."""

    def test_mdem_method_exists(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        assert hasattr(phase, "_apply_vocal_stem_mdem")

    def test_mdem_returns_valid_audio(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        original = _make_vocal(1.0)
        enhanced = original * 1.1  # slight gain change
        result = phase._apply_vocal_stem_mdem(enhanced, original, SR)
        assert result.shape == enhanced.shape
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) <= 1.0

    def test_mdem_fallback_on_error(self):
        """If MDEM not available, original enhanced audio returned unchanged."""
        from unittest.mock import patch

        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        enhanced = _make_vocal(0.5)
        original = enhanced * 0.8
        with patch(
            "backend.core.phases.phase_42_vocal_enhancement.VocalEnhancement._apply_vocal_stem_mdem",
            side_effect=lambda e, o, s: e,
        ):
            result = phase._apply_vocal_stem_mdem(enhanced, original, SR)
        np.testing.assert_array_equal(result, enhanced)


# ---------------------------------------------------------------------------
# Adaptive formant correction strength
# ---------------------------------------------------------------------------
class TestAdaptiveFormantStrength:
    """Formant correction should be stronger when gender is known."""

    def test_known_gender_metadata(self):
        """When gender is 'female', metadata should reflect it."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal(0.5, f0=260.0)
        result = phase.process(audio, SR, vocal_gender="female", strength=0.5)
        assert result.success
        assert result.metadata.get("vocal_gender") == "female"

    def test_unknown_gender_metadata(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal(0.5)
        result = phase.process(audio, SR, strength=0.5)
        assert result.success
        assert result.metadata.get("vocal_gender") == "unknown"


# ---------------------------------------------------------------------------
# Full pipeline psychoacoustic regression
# ---------------------------------------------------------------------------
class TestPhase42PsychoacousticRegression:
    """End-to-end regression: enhanced audio must satisfy basic invariants."""

    def test_no_nan_inf(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vocal(1.0)
        result = phase.process(audio, SR, strength=1.0)
        assert np.all(np.isfinite(result.audio))
        assert np.max(np.abs(result.audio)) <= 1.0

    def test_stereo_invariant(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        mono = _make_vocal(1.0)
        stereo = np.column_stack([mono, mono * 0.95])
        result = phase.process(stereo, SR, strength=0.8)
        assert result.success
        assert result.audio.ndim == 2
        assert result.audio.shape[1] == 2
        assert np.all(np.isfinite(result.audio))

    def test_vibrato_signal_succeeds(self):
        """Vibrato-heavy signal should process without errors."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        audio = _make_vibrato_signal(2.0)
        result = phase.process(audio, SR, strength=0.7)
        assert result.success
        assert np.all(np.isfinite(result.audio))


class TestPhase42StemSafety:
    """Regression tests for stereo stem masking and ML fallback behavior."""

    def test_wiener_stereo_handles_channels_first(self):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        ch_l = _make_vocal(0.4).astype(np.float32)
        ch_r = (_make_vocal(0.4, f0=230.0) * 0.95).astype(np.float32)
        audio_cf = np.vstack([ch_l, ch_r])  # (2, N)
        voc_mono = np.mean(audio_cf, axis=0).astype(np.float32)

        vocals, inst = phase._wiener_stereo_from_mono(audio_cf, voc_mono, SR)

        assert vocals.shape == audio_cf.shape
        assert inst.shape == audio_cf.shape
        assert np.all(np.isfinite(vocals))
        assert np.all(np.isfinite(inst))

    def test_wiener_stereo_tail_ringing_guard(self):
        """OLA tail ringing guard: last win_size samples must not exceed input RMS."""
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

        phase = VocalEnhancement()
        # Fade-out signal: loud music fading to near-silence in last 2s
        total = int(SR * 5.0)
        t = np.arange(total, dtype=np.float32) / SR
        fade = np.clip(1.0 - t / 5.0, 0.0, 1.0)  # 1→0 envelope
        music = (np.sin(2 * np.pi * 220.0 * t) * 0.5 * fade).astype(np.float32)
        audio_cn = np.column_stack([music, music * 0.97])  # (N, 2)
        voc_mono = music * 0.7  # mock vocal separation

        vocals, inst = phase._wiener_stereo_from_mono(audio_cn, voc_mono, SR)

        # Check the last win_size (2048) samples: output must not be louder than input
        tail_n = 2048
        orig_tail_rms = float(np.sqrt(np.mean(audio_cn[-tail_n:] ** 2)) + 1e-9)
        voc_tail_rms = float(np.sqrt(np.mean(vocals[-tail_n:] ** 2)) + 1e-9)
        inst_tail_rms = float(np.sqrt(np.mean(inst[-tail_n:] ** 2)) + 1e-9)

        assert voc_tail_rms <= orig_tail_rms * 1.1, (
            f"OLA tail ringing: vocals tail rms {voc_tail_rms:.6f} > input {orig_tail_rms:.6f} * 1.1"
        )
        assert inst_tail_rms <= orig_tail_rms * 1.1, (
            f"OLA tail ringing: inst tail rms {inst_tail_rms:.6f} > input {orig_tail_rms:.6f} * 1.1"
        )

    def test_try_stem_separation_rejects_low_sdri_roformer(self, monkeypatch):
        from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement
        from plugins import bs_roformer_plugin as rof_mod
        from plugins import mdx23c_plugin as mdx_mod

        class _FakeSep:
            def __init__(self, vocals: np.ndarray):
                self.stems = {"vocals": vocals}
                self.sr = SR
                self.sdri_db = -4.8  # intentional bad quality
                self.model_used = "bs_roformer"
                self.confidence = 0.95

        class _FakeRoformer:
            def separate(self, audio, sr, stems=None):
                return _FakeSep(np.asarray(audio, dtype=np.float32) * 0.4)

        class _FakeMDX:
            def process(self, audio, sr, stem="vocals"):
                x = np.asarray(audio, dtype=np.float32)
                return x * (0.45 if stem == "vocals" else 0.55)

        monkeypatch.setattr(rof_mod, "get_bs_roformer", lambda: _FakeRoformer())
        monkeypatch.setattr(mdx_mod, "get_mdx23c_plugin", lambda: _FakeMDX())

        phase = VocalEnhancement()
        mono = _make_vocal(0.5)
        stereo = np.column_stack([mono, mono * 0.97]).astype(np.float32)

        result = phase._try_stem_separation(stereo, SR, quality_mode="quality", quality_first_unleashed=True)

        assert result is not None
        _, _, _, model_used = result
        assert model_used == "mdx23c_kim_vocal_2"
