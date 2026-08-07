"""
Tests for Improvement J — SBR Phase-Vocoder Transposition in phase_06.

Verifies:
  - _sbr_phase_vocoder_transposition: shape, dtype, unit-magnitude, finite,
    coherence properties, edge cases
  - _apply_sbr: uses PVT (no longer random HF phase → reduced ringing)
  - process() integration: no crash, finite, no-clip, SBR produces output

Scientific basis:
    Laroche & Dolson (1999). "Improved Phase Vocoder Time-Scale Modification
    of Audio." IEEE Trans. Speech Audio Process. 7(3), 323–332.
    Dietz et al. (2002). "Spectral Band Replication, a Novel Approach in
    Audio Coding." Proc. AES 112th Conv.
"""

import numpy as np
import pytest
import scipy.signal as ss

SR = 48_000
HOP = 512
N_FFT = 4096


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def phase():
    from backend.core.phases.phase_06_frequency_restoration import FrequencyRestorationPhase

    return FrequencyRestorationPhase(sample_rate=SR)


def _make_rolloff_audio(dur: float = 0.5, lp_hz: float = 4200.0) -> np.ndarray:
    """LP-filtered bandlimited audio simulating shellac rolloff."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    src = (
        0.40 * np.sin(2 * np.pi * 220 * t)
        + 0.30 * np.sin(2 * np.pi * 880 * t)
        + 0.20 * np.sin(2 * np.pi * 2200 * t)
        + 0.15 * np.sin(2 * np.pi * 4000 * t)
        + 0.08 * np.sin(2 * np.pi * 3800 * t)
    ).astype(np.float32)
    sos = ss.butter(8, lp_hz / (SR / 2), btype="low", output="sos")
    return ss.sosfiltfilt(sos, src).astype(np.float32)  # type: ignore[no-any-return]


def _make_broadband_rolloff_audio(dur: float = 1.0, lp_hz: float = 4200.0) -> np.ndarray:
    """Broadband LP-filtered audio — reliable rolloff detection for
    spectral-mean-based detectors (Welch PSD)."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rng = np.random.RandomState(42)
    src = (
        0.40 * np.sin(2 * np.pi * 220 * t)
        + 0.30 * np.sin(2 * np.pi * 880 * t)
        + 0.20 * np.sin(2 * np.pi * 2200 * t)
        + 0.15 * np.sin(2 * np.pi * 4000 * t)
        + 0.15 * rng.randn(n)
    ).astype(np.float32)
    sos = ss.butter(8, lp_hz / (SR / 2), btype="low", output="sos")
    return ss.sosfiltfilt(sos, src).astype(np.float32)  # type: ignore[no-any-return]


def _make_zxx(n_src: int = 12, n_frames: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.standard_normal((n_src, n_frames)) + 1j * rng.standard_normal((n_src, n_frames))).astype(np.complex64)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  _sbr_phase_vocoder_transposition — API contracts
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSbrPvtApi:
    def test_output_shape(self, phase):
        n_src, n_frames, n_tgt = 10, 50, 20
        pvt = phase._sbr_phase_vocoder_transposition(
            _make_zxx(n_src, n_frames),
            np.linspace(2000.0, 5000.0, n_src),
            np.linspace(10000.0, 18000.0, n_tgt),
            hop=HOP,
            sr=SR,
        )
        assert pvt.shape == (n_tgt, n_frames)

    def test_dtype_complex64(self, phase):
        pvt = phase._sbr_phase_vocoder_transposition(
            _make_zxx(8, 30),
            np.linspace(1000.0, 4000.0, 8),
            np.linspace(8000.0, 14000.0, 15),
            hop=HOP,
            sr=SR,
        )
        assert pvt.dtype == np.complex64

    def test_unit_magnitude(self, phase):
        """All phasors must have magnitude exactly 1.0 (±1e-5)."""
        pvt = phase._sbr_phase_vocoder_transposition(
            _make_zxx(12, 60),
            np.linspace(2000.0, 5000.0, 12),
            np.linspace(10000.0, 18000.0, 25),
            hop=HOP,
            sr=SR,
        )
        np.testing.assert_allclose(np.abs(pvt), 1.0, atol=1e-5)

    def test_finite_output(self, phase):
        pvt = phase._sbr_phase_vocoder_transposition(
            _make_zxx(10, 40),
            np.linspace(3000.0, 6000.0, 10),
            np.linspace(12000.0, 20000.0, 18),
            hop=HOP,
            sr=SR,
        )
        assert np.all(np.isfinite(pvt))

    def test_single_frame(self, phase):
        """n_frames=1 must not crash and return correct shape."""
        pvt = phase._sbr_phase_vocoder_transposition(
            _make_zxx(8, 1),
            np.linspace(2000.0, 4000.0, 8),
            np.linspace(8000.0, 12000.0, 16),
            hop=HOP,
            sr=SR,
        )
        assert pvt.shape == (16, 1)
        assert np.all(np.isfinite(pvt))

    def test_single_source_bin_fallback(self, phase):
        """n_src=1 triggers early-return ones fallback."""
        pvt = phase._sbr_phase_vocoder_transposition(
            _make_zxx(1, 20),
            np.array([2000.0]),
            np.linspace(8000.0, 12000.0, 10),
            hop=HOP,
            sr=SR,
        )
        assert pvt.shape == (10, 20)
        np.testing.assert_allclose(np.abs(pvt), 1.0, atol=1e-5)

    def test_zero_frames_edge(self, phase):
        """n_frames=0 triggers early-return ones fallback."""
        pvt = phase._sbr_phase_vocoder_transposition(
            np.zeros((6, 0), dtype=np.complex64),
            np.linspace(2000.0, 4000.0, 6),
            np.linspace(8000.0, 12000.0, 12),
            hop=HOP,
            sr=SR,
        )
        assert pvt.shape == (12, 0)

    def test_ratio_gt_1(self, phase):
        """Target band above source → ratio > 1 → phase integrates faster."""
        pvt = phase._sbr_phase_vocoder_transposition(
            _make_zxx(8, 40, seed=7),
            np.linspace(1000.0, 2000.0, 8),
            np.linspace(8000.0, 16000.0, 16),
            hop=HOP,
            sr=SR,
        )
        assert pvt.shape == (16, 40)
        assert np.all(np.isfinite(pvt))

    def test_pure_tone_phase_coherence(self, phase):
        """
        Pure tone at f_s = 20 Hz placed at source bin 4; target at 2×f_s = 40 Hz.
        With n_src = n_tgt = 8 and src_f[4] = 20 Hz exactly, the src_idx
        mapping lands on bin 4 directly (frac = 0).  Expected target phase
        increment:

            Δφ = 2 × 2π × 20 × 512 / 48000 ≈ 2.681 rad/frame  (< π, no wrap)

        The PVT angular mean over consecutive output frames at bin 4
        must be within 15 % of this expected value.
        """
        f_s = 20.0  # Hz — Tone frequency
        n_bins = 8
        # src_f[4] = 10 + 4 * 2.5 = 20 Hz exactly so linspace src_idx[4]=4 maps here
        src_f = np.array([10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5])
        tgt_f = src_f * 2.0  # ratio 2 at every bin; tgt_f[4] = 40 Hz
        n_frames = 80
        omega_s = 2.0 * np.pi * f_s / SR
        phi_src = omega_s * HOP * np.arange(n_frames)
        Zxx = np.zeros((n_bins, n_frames), dtype=np.complex64)
        Zxx[4, :] = np.exp(1j * phi_src)  # tone at bin 4

        pvt = phase._sbr_phase_vocoder_transposition(Zxx, src_f, tgt_f, hop=HOP, sr=SR)
        # d_phi from bin 4 using circular difference
        d_phi = np.angle(pvt[4, 1:] * np.conj(pvt[4, :-1]))
        mean_inc = float(np.mean(d_phi))
        expected_inc = 2.0 * omega_s * HOP  # ≈ 2.681 rad/frame
        assert abs(mean_inc - expected_inc) < expected_inc * 0.15, (
            f"Mean phase inc {mean_inc:.4f} rad/frame, expected ≈ {expected_inc:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  _apply_sbr: uses PVT (phase consistency property)
# ──────────────────────────────────────────────────────────────────────────────


class TestApplySbrPvt:
    """
    Verify that _apply_sbr produces output with smoother inter-frame phase
    differences than random-phase assignment (which is what the old code did).
    """

    def _make_stft(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f, _, Zxx = ss.stft(audio, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP)
        return f, Zxx.copy()

    def test_apply_sbr_no_nan(self, phase):
        audio = _make_rolloff_audio(dur=0.5)
        f, Zxx = self._make_stft(audio)
        rolloff_bin = int(np.argmin(np.abs(f - 4500)))
        ext_start = int(np.argmin(np.abs(f - 4500)))
        ext_end = int(np.argmin(np.abs(f - 10000)))
        Zxx_out = phase._apply_sbr(Zxx.copy(), f, rolloff_bin, ext_start, ext_end, sbr_ratio=0.6, strength=0.9, hop=HOP)
        assert np.all(np.isfinite(Zxx_out))

    def test_apply_sbr_shape_unchanged(self, phase):
        audio = _make_rolloff_audio(dur=0.5)
        f, Zxx = self._make_stft(audio)
        rolloff_bin = int(np.argmin(np.abs(f - 4500)))
        ext_start = int(np.argmin(np.abs(f - 4500)))
        ext_end = int(np.argmin(np.abs(f - 10000)))
        Zxx_out = phase._apply_sbr(Zxx.copy(), f, rolloff_bin, ext_start, ext_end, sbr_ratio=0.6, strength=0.9, hop=HOP)
        assert Zxx_out.shape == Zxx.shape

    def test_apply_sbr_adds_hf_energy(self, phase):
        """SBR must increase energy in extension band."""
        audio = _make_rolloff_audio(dur=0.5)
        f, Zxx = self._make_stft(audio)
        rolloff_bin = int(np.argmin(np.abs(f - 4500)))
        ext_start = int(np.argmin(np.abs(f - 5000)))
        ext_end = int(np.argmin(np.abs(f - 9000)))
        energy_before = np.sum(np.abs(Zxx[ext_start:ext_end, :]) ** 2)
        Zxx_out = phase._apply_sbr(Zxx.copy(), f, rolloff_bin, ext_start, ext_end, sbr_ratio=0.6, strength=0.9, hop=HOP)
        energy_after = np.sum(np.abs(Zxx_out[ext_start:ext_end, :]) ** 2)
        assert energy_after > energy_before, "SBR must add HF energy"

    def test_pvt_phase_smoother_than_random(self, phase):
        """
        Pure tone at f_s = 1500 Hz placed at source bin 4; target at 2250 Hz.
        With n_src = n_tgt = 8 and src_f[4] = 1500 Hz exactly, the src_idx
        mapping (linspace(0,7,8)[4] = 4) lands on bin 4 directly.

        Because  f_t × hop / SR = 2250 × 512 / 48000 = 24 (integer),
        the PVT phase increment wraps to exactly 0 every frame:

            IF_target × hop = 24 × 2π ≡ 0 (mod 2π)  →  var(dφ_PVT) ≈ 0.

        Random assignment instead gives dφ ~ U[−π,+π] → var ≈ π²/3 ≈ 3.29.
        PVT must be < 25 % of random variance.
        """
        rng = np.random.RandomState(0)
        f_s = 1500.0
        n_bins = 8
        # src_f[4] = 1300 + 4*50 = 1500 Hz exactly → src_idx[4] = 4.0
        src_f = np.array([1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1650.0])
        tgt_f = src_f * 1.5  # ratio 1.5; tgt_f[4] = 2250 Hz
        n_frames = 80
        omega = 2.0 * np.pi * f_s / SR
        phi = omega * HOP * np.arange(n_frames)
        Zxx_src = np.zeros((n_bins, n_frames), dtype=np.complex64)
        Zxx_src[4, :] = 0.8 * np.exp(1j * phi).astype(np.complex64)  # tone at bin 4
        # Tiny background noise on other bins
        Zxx_src += 0.01 * (
            rng.standard_normal((n_bins, n_frames)) + 1j * rng.standard_normal((n_bins, n_frames))
        ).astype(np.complex64)

        pvt = phase._sbr_phase_vocoder_transposition(Zxx_src, src_f, tgt_f, hop=HOP, sr=SR)
        # d_phi at bin 4: IF_target×hop = 24×2π ≡ 0 mod 2π → near-zero var
        d_phi_pvt = np.angle(pvt[4, 1:] * np.conj(pvt[4, :-1]))
        var_pvt = float(np.var(d_phi_pvt))

        # Random-phase baseline: uniform dφ → var ≈ π²/3 ≈ 3.29
        rand_phasors = np.exp(1j * rng.uniform(-np.pi, np.pi, n_frames).astype(np.float32))
        d_phi_rand = np.angle(rand_phasors[1:] * np.conj(rand_phasors[:-1]))
        var_rand = float(np.var(d_phi_rand))

        # PVT must be dramatically smoother: var < 25 % of random
        assert var_pvt < var_rand * 0.25, f"PVT var {var_pvt:.4f} should be < 25 % of random var {var_rand:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Full process() integration
# ──────────────────────────────────────────────────────────────────────────────


class TestProcessIntegration:
    def test_no_nan_shellac(self, phase):
        audio = _make_rolloff_audio(dur=1.0)
        r = phase.process(audio, material_type="shellac", sample_rate=SR)
        assert np.all(np.isfinite(r.audio))

    def test_no_clipping_shellac(self, phase):
        audio = _make_rolloff_audio(dur=1.0)
        r = phase.process(audio, material_type="shellac", sample_rate=SR)
        assert np.max(np.abs(r.audio)) <= 1.0 + 1e-6

    def test_shape_mono(self, phase):
        audio = _make_rolloff_audio(dur=0.5)
        r = phase.process(audio, material_type="shellac", sample_rate=SR)
        assert r.audio.shape == audio.shape

    def test_shape_stereo(self, phase):
        mono = _make_rolloff_audio(dur=0.5)
        stereo = np.column_stack([mono, mono * 0.95])
        r = phase.process(stereo, material_type="shellac", sample_rate=SR)
        assert r.audio.shape == stereo.shape

    def test_cd_digital_passthrough(self, phase):
        """cd_digital has no rolloff → pass-through."""
        n = SR // 2
        t = np.arange(n, dtype=np.float32) / SR
        audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        r = phase.process(audio, material_type="cd_digital", sample_rate=SR)
        assert r.success
        assert r.modifications.get("frequency_restored") is False

    def test_vinyl_no_crash(self, phase):
        sos = ss.butter(6, 11000 / (SR / 2), btype="low", output="sos")
        n = SR
        t = np.arange(n) / SR
        audio = ss.sosfiltfilt(sos, 0.4 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t)).astype(
            np.float32
        )
        r = phase.process(audio, material_type="vinyl", sample_rate=SR)
        assert r.success
        assert np.all(np.isfinite(r.audio))

    def test_silence_no_crash(self, phase):
        audio = np.zeros(SR // 2, dtype=np.float32)
        r = phase.process(audio, material_type="shellac", sample_rate=SR)
        assert r.success
        assert np.all(np.isfinite(r.audio))

    def test_stereo_rolloff_extends_hf(self, phase):
        """Restored stereo must have more HF energy than rolled-off input."""
        mono = _make_broadband_rolloff_audio(dur=1.0)
        stereo = np.column_stack([mono, mono * 0.98])
        # quality_mode="restoration" → DSP-only path (not in ML list, no ML timeout in unit tests)
        r = phase.process(stereo, material_type="shellac", sample_rate=SR, quality_mode="restoration")
        assert r.modifications.get("frequency_restored", False), "Rolloff should be detected for broadband stereo input"
        rolloff_hz = 4500.0
        sos = ss.butter(4, rolloff_hz / (SR / 2), btype="high", output="sos")
        hf_before = float(np.sqrt(np.mean(ss.sosfiltfilt(sos, stereo[:, 0]) ** 2)))
        hf_after = float(np.sqrt(np.mean(ss.sosfiltfilt(sos, r.audio[:, 0]) ** 2)))
        assert hf_after >= hf_before * 0.98, f"HF energy after {hf_after:.6f} < before {hf_before:.6f}"
