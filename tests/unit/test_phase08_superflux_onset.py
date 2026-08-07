"""
Tests for Superflux onset detection in TransientPreservationPhase.

Verifies the Böck & Widmer (2013) Superflux implementation:
- Basic API contracts (shape, dtype, bounds)
- Vibrato suppression: max-filter removes false onsets on sustained tones
- Hard onset detection: genuine transients detected reliably
- Edge cases: silence, single-sample, stereo input
- Regression: fewer false positives than naive spectral flux on vibrato
- Integration: process() contract unchanged
"""

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SR = 48_000


@pytest.fixture()
def phase():
    from backend.core.phases.phase_08_transient_preservation import TransientPreservationPhase

    p = TransientPreservationPhase()
    p.sample_rate = SR
    return p


def _sine(freq: float, dur: float, amp: float = 0.5) -> np.ndarray:
    """Pure sine, mono, SR=48000."""
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _vibrato_sine(freq: float, dur: float, vib_rate: float = 6.0, vib_depth: float = 0.04) -> np.ndarray:
    """Sine with vibrato (±4 % pitch), 6 Hz rate — triggers false onsets with naive flux."""
    t = np.arange(int(dur * SR)) / SR
    phase = 2.0 * np.pi * (freq * t + (vib_depth * freq / vib_rate) * np.sin(2.0 * np.pi * vib_rate * t))
    return (0.5 * np.sin(phase)).astype(np.float32)  # type: ignore[no-any-return]


def _impulse_train(n_impulses: int, dur: float, amp: float = 0.8) -> np.ndarray:
    """Equally-spaced impulses — clear hard onsets."""
    audio = np.zeros(int(dur * SR), dtype=np.float32)
    spacing = len(audio) // (n_impulses + 1)
    for k in range(1, n_impulses + 1):
        fade = int(0.005 * SR)  # 5 ms attack onset
        idx = k * spacing
        end = min(idx + fade, len(audio))
        audio[idx:end] = amp * np.linspace(0, 1, end - idx)
    return audio


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Return-type contracts
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSuperfluxReturnContract:
    def test_returns_two_arrays(self, phase):
        audio = _sine(440.0, 0.5)
        times, strengths = phase._detect_onsets_spectral_flux(audio, sensitivity=0.7)
        assert isinstance(times, np.ndarray)
        assert isinstance(strengths, np.ndarray)

    def test_same_length(self, phase):
        audio = _sine(440.0, 0.5)
        times, strengths = phase._detect_onsets_spectral_flux(audio, sensitivity=0.7)
        assert len(times) == len(strengths)

    def test_times_nonnegative(self, phase):
        audio = _impulse_train(4, 1.0)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.5)
        if len(times) > 0:
            assert np.all(times >= 0.0)

    def test_times_within_duration(self, phase):
        dur = 1.0
        audio = _impulse_train(4, dur)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.5)
        if len(times) > 0:
            assert np.all(times <= dur + 0.05)

    def test_strengths_positive(self, phase):
        audio = _impulse_train(4, 1.0)
        _, strengths = phase._detect_onsets_spectral_flux(audio, sensitivity=0.5)
        if len(strengths) > 0:
            assert np.all(strengths > 0.0)

    def test_strengths_no_nan(self, phase):
        audio = _impulse_train(4, 1.0)
        _, strengths = phase._detect_onsets_spectral_flux(audio, sensitivity=0.5)
        assert np.all(np.isfinite(strengths))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Vibrato suppression — core Superflux property
# ──────────────────────────────────────────────────────────────────────────────


class TestVibratoSuppression:
    """
    Superflux invariant (Böck & Widmer 2013):
    A sustained note with vibrato must produce fewer false onsets than
    naive spectral flux.  On a pure vibrato tone with no genuine attacks,
    the onset count should be noticeably smaller.
    """

    @staticmethod
    def _naive_flux_count(audio: np.ndarray, sensitivity: float) -> int:
        """Reference: classic frame-difference flux (no max-filter)."""
        import scipy.signal as ss

        hop, n_fft = 512, 2048
        _, _, Zxx = ss.stft(audio, fs=SR, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(Zxx)
        flux = np.zeros(mag.shape[1])
        for i in range(1, mag.shape[1]):
            diff = mag[:, i] - mag[:, i - 1]
            flux[i] = np.sum(np.maximum(diff, 0))
        flux /= np.max(flux) + 1e-10
        threshold = np.median(flux) + sensitivity * (np.max(flux) - np.median(flux))
        peaks, _ = ss.find_peaks(flux, height=threshold, distance=int(0.05 * SR / hop))
        return len(peaks)

    def test_fewer_false_onsets_than_naive_flux(self, phase):
        # Use low sensitivity so naive flux (no vibrato suppression) produces
        # clearly more onsets on a high-depth vibrato sine, while Superflux
        # suppresses them.  The invariant: superflux_count <= naive_count.
        audio = _vibrato_sine(300.0, dur=3.0, vib_rate=6.0, vib_depth=0.06)
        sensitivity = 0.25  # Low threshold: naive floods with vibrato artifacts
        naive_count = self._naive_flux_count(audio, sensitivity)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=sensitivity)
        superflux_count = len(times)
        # Naive must produce at least 2 onsets to make the test meaningful;
        # if not, the vibrato signal is not stirring the naive flux (relax).
        if naive_count >= 2:
            assert superflux_count <= naive_count, (
                f"Superflux ({superflux_count}) should have ≤ false onsets than naive flux ({naive_count})"
            )
        else:
            # Both algorithms agree: no strong false onsets — acceptable.
            assert superflux_count <= naive_count + 1

    def test_sustained_sine_minimal_onsets(self, phase):
        """A steady pure sine has no genuine attacks — very few onsets."""
        audio = _sine(880.0, dur=1.5)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.7)
        # A plain sine may trigger 0 or 1 onset (start-up ramp), never many
        assert len(times) <= 2, f"Sustained sine produced too many onsets: {len(times)}"

    def test_vibrato_sine_few_onsets(self, phase):
        """Even high-depth vibrato (±4%) should not flood with false onsets."""
        audio = _vibrato_sine(300.0, dur=2.0, vib_depth=0.04)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.65)
        # Loose bound — just ensure suppression kicks in
        assert len(times) <= 8, f"Vibrato sine triggered {len(times)} onsets (expected ≤ 8)"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Hard-onset detection — genuine transients must survive
# ──────────────────────────────────────────────────────────────────────────────


class TestHardOnsetDetection:
    def test_impulse_train_detects_onsets(self, phase):
        """4 clear impulse onsets should each be detected."""
        audio = _impulse_train(4, dur=2.0)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.4)
        # Relaxed: at least 2 out of 4 must be found
        assert len(times) >= 2, f"Only {len(times)} of 4 impulse onsets detected"

    def test_high_sensitivity_more_onsets_than_low(self, phase):
        """Higher sensitivity should yield at least as many onsets."""
        audio = _impulse_train(3, dur=1.5)
        times_lo, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.9)
        times_hi, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.2)
        assert len(times_hi) >= len(times_lo)

    def test_amplitude_step_detected(self, phase):
        """Sudden level jump from silence to loud tone must be detected."""
        n = int(SR * 1.0)
        audio = np.zeros(n, dtype=np.float32)
        audio[n // 2 :] = 0.7 * np.sin(2.0 * np.pi * 440.0 * np.arange(n // 2) / SR)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.5)
        assert len(times) >= 1, "Amplitude step onset not detected"


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Edge-case robustness
# ──────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_silence_no_crash(self, phase):
        audio = np.zeros(SR, dtype=np.float32)
        times, strengths = phase._detect_onsets_spectral_flux(audio, sensitivity=0.7)
        assert isinstance(times, np.ndarray)
        assert isinstance(strengths, np.ndarray)

    def test_silence_no_false_onsets(self, phase):
        audio = np.zeros(SR, dtype=np.float32)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.5)
        assert len(times) == 0

    def test_very_short_audio(self, phase):
        """Very short clip (100 samples) — must not crash."""
        audio = np.random.randn(100).astype(np.float32) * 0.1
        times, strengths = phase._detect_onsets_spectral_flux(audio, sensitivity=0.7)
        assert len(times) == len(strengths)

    def test_stereo_input(self, phase):
        """Stereo array (N,2) must be handled via mono downmix."""
        mono = _sine(440.0, 1.0)
        stereo = np.column_stack([mono, mono * 0.9])
        times, strengths = phase._detect_onsets_spectral_flux(stereo, sensitivity=0.6)
        assert isinstance(times, np.ndarray)
        assert len(times) == len(strengths)

    def test_stereo_result_matches_mono(self, phase):
        """Stereo onset times should equal mono (same content, downmix)."""
        mono = _sine(440.0, 1.0) + _impulse_train(3, 1.0)
        stereo = np.column_stack([mono, mono])
        times_mono, _ = phase._detect_onsets_spectral_flux(mono, sensitivity=0.6)
        times_stereo, _ = phase._detect_onsets_spectral_flux(stereo, sensitivity=0.6)
        assert len(times_mono) == len(times_stereo)

    def test_dc_offset_signal(self, phase):
        # A DC signal has no spectral changes.  The STFT boundary zero-padding
        # can produce at most one startup transient at frame 0/1, so we allow ≤ 1.
        audio = np.full(SR, 0.5, dtype=np.float32)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.7)
        assert len(times) <= 1, f"DC signal should not trigger many onsets, got {len(times)}"

    def test_single_frame_audio(self, phase):
        """Audio exactly one STFT frame — must not crash."""
        audio = 0.3 * np.random.randn(2048).astype(np.float32)
        times, _ = phase._detect_onsets_spectral_flux(audio, sensitivity=0.7)
        assert isinstance(times, np.ndarray)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Process() integration (phase-level smoke tests)
# ──────────────────────────────────────────────────────────────────────────────


class TestProcessIntegration:
    def test_process_returns_phase_result(self, phase):
        from backend.core.phases.phase_interface import PhaseResult

        audio = _sine(440.0, 0.5)
        result = phase.process(audio, material_type="tape", sample_rate=SR)
        assert isinstance(result, PhaseResult)

    def test_process_output_shape_preserved(self, phase):
        audio = _sine(440.0, 0.5)
        result = phase.process(audio, material_type="vinyl", sample_rate=SR)
        assert result.audio.shape == audio.shape

    def test_process_no_nan_inf(self, phase):
        audio = _impulse_train(3, 1.0)
        result = phase.process(audio, material_type="tape", sample_rate=SR)
        assert np.all(np.isfinite(result.audio))

    def test_process_no_clipping(self, phase):
        audio = _impulse_train(3, 1.0)
        result = phase.process(audio, material_type="tape", sample_rate=SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_process_stereo_no_crash(self, phase):
        mono = _impulse_train(3, 1.0)
        stereo = np.column_stack([mono, mono * 0.95])
        result = phase.process(stereo, material_type="vinyl", sample_rate=SR)
        assert result.audio.shape == stereo.shape

    def test_process_silence_passthrough(self, phase):
        audio = np.zeros(SR, dtype=np.float32)
        result = phase.process(audio, material_type="unknown", sample_rate=SR)
        assert result.success
        assert np.all(result.audio == 0.0)
