from __future__ import annotations

"""Tests for Phase 50 §PriorPhase-Guard — HF-bin protection from Pass-1 spike detection.

Root-cause: Phase_50 uses a per-frequency local-average spike detector (11-bin window).
When Phase_07 (harmonic restoration) or Phase_06 (SBR) adds a restored harmonic at a
frequency that was previously near the noise floor, the smooth envelope at that bin is
~H/11 (only 1 bin elevated).  The spike ratio is 11 > threshold_factor (3.0–4.5), so
Phase_50 flags and inpaints the harmonic — reverting Phase_07/06's restoration.

Fix: `hf_protected_bin_start` parameter in `_repair_channel` excludes bins above the
material's natural rolloff from Pass 1.  Pass 2 (frame energy dropout) is NOT affected.
"""


import numpy as np
import pytest

SR = 48000
_FFT_SIZE = 2048
_BIN_HZ = SR / _FFT_SIZE  # ≈ 23.4 Hz per STFT bin


def _sine(freq: float = 440.0, seconds: float = 2.0, amp: float = 0.3) -> np.ndarray:
    """Mono sine wave."""
    t = np.linspace(0.0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(seconds: float = 2.0) -> np.ndarray:
    """White-noise stereo audio."""
    rng = np.random.default_rng(0)
    n = int(SR * seconds)
    return rng.standard_normal((n, 2)).astype(np.float32) * 0.2


# ---------------------------------------------------------------------------
# _repair_channel unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRepairChannelHFProtection:
    """Direct tests for the low-level _repair_channel HF-spike guard."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.core.phases.phase_50_spectral_repair import _repair_channel

        self._repair_channel = _repair_channel

    def _isolated_harmonic_channel(self, harmonic_hz: float = 13_000.0, seconds: float = 2.0) -> np.ndarray:
        """Broadband noise + single isolated sine at harmonic_hz (simulating Phase_07 output)."""
        rng = np.random.default_rng(1)
        n = int(SR * seconds)
        noise = rng.standard_normal(n).astype(np.float64) * 0.002  # near noise floor
        t = np.linspace(0, seconds, n, endpoint=False)
        harmonic = (0.3 * np.sin(2.0 * np.pi * harmonic_hz * t)).astype(np.float64)
        return (noise + harmonic).astype(np.float64)  # type: ignore[no-any-return]

    def test_isolated_harmonic_preserved_with_hf_guard(self):
        """Isolated HF harmonic must pass through unchanged when HF protection is active."""
        harmonic_hz = 13_000.0
        channel = self._isolated_harmonic_channel(harmonic_hz)

        # Protected bin start at 11 kHz (85 % of 13 kHz material rolloff)
        protected_bin = int(11_000.0 / _BIN_HZ)
        repaired, n_rep = self._repair_channel(channel, SR, 4.0, hf_protected_bin_start=protected_bin)

        # Compute energy ratio of the harmonic frequency in input vs output
        from scipy.signal import stft

        _, _, Zxx_in = stft(channel, fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        _, _, Zxx_out = stft(repaired, fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        harm_bin = int(harmonic_hz / _BIN_HZ)
        energy_in = float(np.mean(np.abs(Zxx_in[harm_bin, :])))
        energy_out = float(np.mean(np.abs(Zxx_out[harm_bin, :])))

        # With HF guard: harmonic energy must be within 10 % of input (not inpainted away)
        assert energy_out >= energy_in * 0.90, (
            f"HF harmonic at {harmonic_hz} Hz was inpainted despite hf_protected_bin_start. "
            f"energy_in={energy_in:.4f}, energy_out={energy_out:.4f}"
        )

    def test_isolated_harmonic_removed_without_hf_guard(self):
        """Without protection, the same harmonic IS flagged as a spike and partially removed."""
        harmonic_hz = 13_000.0
        channel = self._isolated_harmonic_channel(harmonic_hz)

        # No HF protection (default behaviour before the fix)
        repaired, n_rep = self._repair_channel(channel, SR, 4.0, hf_protected_bin_start=0)

        from scipy.signal import stft

        _, _, Zxx_in = stft(channel, fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        _, _, Zxx_out = stft(repaired, fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        harm_bin = int(harmonic_hz / _BIN_HZ)
        energy_in = float(np.mean(np.abs(Zxx_in[harm_bin, :])))
        energy_out = float(np.mean(np.abs(Zxx_out[harm_bin, :])))

        # Without guard: the 13 kHz harmonic is flagged as spike → energy reduced
        # (Documents the original bug so the guard's value is testable)
        assert n_rep > 0, "Expected some bins to be flagged as spikes without HF protection"
        assert energy_out < energy_in * 0.99, (
            "Without HF guard the isolated harmonic should have been partially reduced by Pass 1"
        )

    def test_lf_codec_spike_still_detected_with_hf_guard(self):
        """Low-frequency codec spikes (within natural bandwidth) must still be caught."""
        rng = np.random.default_rng(2)
        n = int(SR * 2.0)
        channel = (rng.standard_normal(n) * 0.1).astype(np.float64)

        # Plant an artificial spike at 2 kHz (within any material's natural bandwidth)
        from scipy.signal import istft, stft

        spike_hz = 2_000.0
        spike_bin = int(spike_hz / _BIN_HZ)
        _, _, Zxx = stft(channel, fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        Zxx[spike_bin, :] *= 20.0  # 20× spike above neighbours
        _, ch_spiked = istft(Zxx, fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        ch_spiked = ch_spiked[:n].astype(np.float64)

        # HF protection at 11 kHz: does NOT protect the 2 kHz spike
        protected_bin = int(11_000.0 / _BIN_HZ)
        repaired, n_rep = self._repair_channel(ch_spiked, SR, 4.0, hf_protected_bin_start=protected_bin)

        assert n_rep > 0, "LF codec spike below hf_protected_bin_start must still be detected"

    def test_no_hf_protection_on_digital_material(self):
        """For digital materials (no rolloff protection) all bins are processed normally."""
        channel = self._isolated_harmonic_channel(13_000.0)
        # hf_protected_bin_start = 0 → no protection
        _, n_rep_no_prot = self._repair_channel(channel, SR, 4.0, hf_protected_bin_start=0)
        _, n_rep_with_prot = self._repair_channel(channel, SR, 4.0, hf_protected_bin_start=int(11_000 / _BIN_HZ))
        # Without protection more bins should be flagged (HF harmonic counted as spike)
        assert n_rep_no_prot >= n_rep_with_prot, (
            f"Expected more spikes without HF protection: {n_rep_no_prot} vs {n_rep_with_prot}"
        )


# ---------------------------------------------------------------------------
# process() integration tests
# ---------------------------------------------------------------------------


class TestPhase50Process:
    """Integration tests via SpectralRepairPhase.process()."""

    @pytest.fixture(autouse=True)
    def _phase(self):
        from backend.core.phases.phase_50_spectral_repair import SpectralRepairPhase

        self.phase = SpectralRepairPhase()

    def test_output_shape_mono(self):
        audio = _sine()
        result = self.phase.process(audio, SR)
        assert result.audio.shape == audio.shape

    def test_output_shape_stereo(self):
        audio = _stereo()
        result = self.phase.process(audio, SR)
        assert result.audio.shape == audio.shape

    def test_no_nan_inf(self):
        audio = _sine()
        result = self.phase.process(audio, SR)
        assert np.all(np.isfinite(result.audio))

    def test_clipped(self):
        audio = np.ones(SR * 2, dtype=np.float32) * 2.0
        result = self.phase.process(audio, SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_vinyl_material_populates_hf_protection_metadata(self):
        """For vinyl, metadata must report non-zero hf_protected_bin_start."""
        audio = _sine()
        result = self.phase.process(audio, SR, material_type="vinyl")
        assert result.metadata.get("hf_protected_bin_start", 0) > 0, (
            "Vinyl material should produce non-zero hf_protected_bin_start in metadata"
        )

    def test_shellac_material_populates_hf_protection_metadata(self):
        """For shellac (7 kHz rolloff), hf_protected_bin_start should be around 255 bins."""
        audio = _sine()
        result = self.phase.process(audio, SR, material_type="shellac")
        expected_bin = int(7_000.0 * 0.85 / _BIN_HZ)  # ≈ 254
        reported_bin = result.metadata.get("hf_protected_bin_start", 0)
        assert reported_bin > 0, "Shellac should produce non-zero HF protection"
        assert abs(reported_bin - expected_bin) <= 2, (
            f"Expected hf_protected_bin_start ≈ {expected_bin}, got {reported_bin}"
        )

    def test_cd_digital_material_no_hf_protection(self):
        """For CD digital material, no HF protection should be applied."""
        audio = _sine()
        result = self.phase.process(audio, SR, material_type="cd_digital")
        assert result.metadata.get("hf_protected_bin_start", 0) == 0, (
            "cd_digital should not trigger HF protection (no analog rolloff)"
        )

    def test_mp3_low_material_no_hf_protection(self):
        """MP3 low has no analog rolloff — no HF protection needed."""
        audio = _sine()
        result = self.phase.process(audio, SR, material_type="mp3_low")
        assert result.metadata.get("hf_protected_bin_start", 0) == 0

    def test_rms_drop_bounded(self):
        """Phase 50 must not drop RMS by more than 6 dB on broadband audio (music-like)."""
        # Pure sine is a single-bin isolated spike → naturally flagged by Pass-1 spike detection.
        # Use broadband pink-ish noise instead (more representative of music spectrum).
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(SR * 2).astype(np.float32) * 0.2
        result = self.phase.process(audio, SR)
        rms_in = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        rms_out = float(np.sqrt(np.mean(result.audio.astype(np.float64) ** 2)))
        drop_db = 20.0 * np.log10(max(rms_out, 1e-12) / max(rms_in, 1e-12))
        assert drop_db > -6.0, f"Phase 50 dropped RMS by {-drop_db:.1f} dB on broadband audio"

    def test_zero_strength_passthrough(self):
        """At strength=0 the output must match the input exactly."""
        audio = _sine()
        result = self.phase.process(audio, SR, strength=0.0)
        np.testing.assert_array_equal(result.audio, np.nan_to_num(np.clip(audio, -1.0, 1.0)))

    def test_stereo_ms_domain(self):
        """Stereo output must be in M/S domain (metadata stereo_mode = ms_domain)."""
        audio = _stereo()
        result = self.phase.process(audio, SR)
        assert result.metadata.get("stereo_mode") == "ms_domain"

    def test_vinyl_hf_harmonic_preservation(self):
        """Restored 13 kHz harmonic (Phase_07 proxy) must not be inpainted by Phase_50."""
        rng = np.random.default_rng(3)
        n = int(SR * 2.0)
        noise = rng.standard_normal(n).astype(np.float32) * 0.002
        t = np.linspace(0, 2.0, n, endpoint=False)
        harmonic = (0.3 * np.sin(2.0 * np.pi * 13_000.0 * t)).astype(np.float32)
        audio = (noise + harmonic).astype(np.float32)

        result = self.phase.process(audio, SR, material_type="vinyl")

        from scipy.signal import stft

        _, _, Zxx_in = stft(audio.astype(np.float64), fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        _, _, Zxx_out = stft(result.audio.astype(np.float64), fs=SR, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - 512)
        harm_bin = int(13_000.0 / _BIN_HZ)
        energy_in = float(np.mean(np.abs(Zxx_in[harm_bin, :])))
        energy_out = float(np.mean(np.abs(Zxx_out[harm_bin, :])))

        assert energy_out >= energy_in * 0.88, (
            f"Phase_50 should preserve Phase_07 restored HF harmonic at 13 kHz for vinyl. "
            f"energy_in={energy_in:.4f}, energy_out={energy_out:.4f}, ratio={energy_out / energy_in:.3f}"
        )
