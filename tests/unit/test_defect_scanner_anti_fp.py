from __future__ import annotations

import pytest

"""Anti-False-Positive-Tests für DefectScanner (§6.3 — Aurik v10.0.0).

Validates that the three hardened detectors (_detect_clicks,
_detect_crackle, _detect_compression_artifacts) do NOT produce false
positives on clean/musical signals while still detecting real defects.
"""


import numpy as np

SR = 48_000


def _scanner(sr: int = SR):
    from backend.core.defect_scanner import DefectScanner

    return DefectScanner(sample_rate=sr)


def _sine(freq: float = 440.0, amp: float = 0.5, duration: float = 3.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)  # type: ignore[no-any-return]


def _complex_tone(duration: float = 3.0) -> np.ndarray:
    """Harmonically rich tone (fundamental + 5 harmonics) — no defects."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    sig = np.zeros_like(t, dtype=np.float32)
    for k in range(1, 7):
        sig += (0.3 / k) * np.sin(2 * np.pi * 440 * k * t).astype(np.float32)
    return sig


# ============================================================
# CLICKS – Anti-False-Positive
# ============================================================


@pytest.mark.unit
class TestClicksAntiFP:
    """Clean signals must NOT trigger click detection."""

    def test_pure_sine_no_clicks(self):
        """Pure 440 Hz sine → severity 0, no locations."""
        sc = _scanner()
        score = sc._detect_clicks(_sine(440, 0.5))
        assert score.severity == 0.0
        assert len(score.locations) == 0

    def test_complex_tone_no_clicks(self):
        """Harmonically rich tone → negligible click severity."""
        sc = _scanner()
        score = sc._detect_clicks(_complex_tone())
        # Allow marginal severity from harmonic peak transitions
        assert score.severity < 0.05

    def test_loud_sine_no_clicks(self):
        """Full-scale sine → no clicks (high diff values, but periodic)."""
        sc = _scanner()
        score = sc._detect_clicks(_sine(440, 0.95))
        assert score.severity == 0.0

    def test_low_freq_sine_no_clicks(self):
        """50 Hz sine → steeper diff per sample, still no clicks."""
        sc = _scanner()
        score = sc._detect_clicks(_sine(50, 0.5))
        assert score.severity == 0.0

    def test_real_clicks_still_detected(self):
        """Injected clicks on sine must still be found."""
        sc = _scanner()
        audio = _sine(440, 0.3)
        for pos in [0.5, 1.5, 2.5]:
            idx = int(pos * SR)
            audio[idx] = 0.99
        score = sc._detect_clicks(audio)
        assert score.severity > 0.0
        assert len(score.locations) >= 3

    def test_many_clicks_not_capped_at_50(self):
        """200 injected clicks should not be truncated to 50 locations."""
        sc = _scanner()
        audio = _sine(440, 0.3)
        rng = np.random.default_rng(42)
        positions = rng.integers(1000, len(audio) - 1000, size=200)
        for p in positions:
            audio[int(p)] = 0.99
        score = sc._detect_clicks(audio)
        assert len(score.locations) > 50
        assert score.metadata["total_clicks"] > 50  # severity uses full count

    def test_drum_transient_not_click(self):
        """Realistic drum transient (~2 ms attack) must not be detected as click."""
        sc = _scanner()
        audio = _sine(440, 0.2)
        # Simulate a realistic drum hit: smooth 2 ms attack + 20 ms decay
        # Added on TOP of existing signal (like real drums) to avoid
        # boundary discontinuities that create artificial click-like edges.
        idx = int(1.0 * SR)
        attack_samples = int(0.002 * SR)  # 2 ms attack (96 samples @ 48 kHz)
        decay_samples = int(0.020 * SR)  # 20 ms decay
        total = attack_samples + decay_samples
        envelope = np.concatenate(
            [
                np.linspace(0.0, 0.6, attack_samples),
                np.linspace(0.6, 0.0, decay_samples),
            ]
        ).astype(np.float32)
        audio[idx : idx + total] += envelope
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        score = sc._detect_clicks(audio)
        # Wide musical transient should not be flagged as click
        click_at_1s = [loc for loc in score.locations if abs(loc[0] - 1.0) < 0.03]
        assert len(click_at_1s) == 0


# ============================================================
# CRACKLE – Anti-False-Positive
# ============================================================


class TestCrackleAntiFP:
    """Brilliant / HF-rich signals must NOT trigger crackle detection."""

    def test_bright_harmonic_no_crackle(self):
        """Harmonically rich signal (cymbals-like) → no crackle FP."""
        sc = _scanner()
        score = sc._detect_crackle(_complex_tone())
        assert score.severity < 0.15  # tolerance for edge cases

    def test_pure_hf_sine_no_crackle(self):
        """12 kHz sine → high HP energy, but tonal → no crackle."""
        sc = _scanner()
        score = sc._detect_crackle(_sine(12000, 0.3))
        assert score.severity < 0.15

    def test_real_crackle_detected(self):
        """Injected impulsive noise (crackle) must still be found."""
        sc = _scanner()
        rng = np.random.default_rng(7)
        audio = _sine(440, 0.2)
        # Inject sparse high-frequency impulses (crackle)
        n_crackle = 500
        positions = rng.integers(0, len(audio), size=n_crackle)
        amplitudes = rng.uniform(0.05, 0.2, size=n_crackle).astype(np.float32)
        for p, a in zip(positions, amplitudes):
            audio[int(p)] += a * (1 if rng.random() > 0.5 else -1)
        score = sc._detect_crackle(audio)
        assert score.severity > 0.0


# ============================================================
# COMPRESSION ARTIFACTS – Anti-False-Positive
# ============================================================


class TestCompressionArtifactsAntiFP:
    """Tonal / full-bandwidth signals must NOT trigger codec artifact detection."""

    def test_pure_sine_no_compression(self):
        """440 Hz sine → no compression artifacts."""
        sc = _scanner()
        score = sc._detect_compression_artifacts(_sine(440, 0.5))
        assert score.severity < 0.15

    def test_complex_fullband_no_compression(self):
        """Harmonically rich tone with HF content → no codec FP."""
        sc = _scanner()
        score = sc._detect_compression_artifacts(_complex_tone())
        assert score.severity < 0.15

    def test_white_noise_no_compression(self):
        """Broadband white noise has low SFM-variance → could be edge case."""
        sc = _scanner()
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(3 * SR) * 0.1).astype(np.float32)
        score = sc._detect_compression_artifacts(audio)
        # White noise has uniform SFM → low temporal variance → not compression
        assert score.severity < 0.3

    def test_bandwidth_limited_detected(self):
        """Signal hard-cut at 16 kHz (like low-bitrate MP3) → detected."""
        sc = _scanner()
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(3 * SR) * 0.1).astype(np.float32)
        # Hard low-pass at 16 kHz via FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1 / SR)
        fft[freqs > 16000] = 0
        audio = np.fft.irfft(fft, n=len(audio)).astype(np.float32)
        score = sc._detect_compression_artifacts(audio)
        # Should detect the bandwidth limitation
        assert score.severity > 0.0


class TestReverbExcessSilenceHandling:
    """Reverb detector must not produce false positives on silence layouts."""

    def test_leading_silence_does_not_inflate_reverb(self):
        """Leading silence must not inflate score vs. same signal without leading silence."""
        sc = _scanner()
        lead = np.zeros(int(0.5 * SR), dtype=np.float32)
        prog = _sine(440.0, 0.3, duration=2.5)
        score_ref = sc._detect_reverb_excess(prog)
        score_with_lead = sc._detect_reverb_excess(np.concatenate([lead, prog]).astype(np.float32))
        assert score_with_lead.severity <= score_ref.severity + 0.05

    def test_near_silence_returns_zero_reverb(self):
        """Near-silent tails must be guarded and return zero severity."""
        sc = _scanner()
        audio = np.zeros(int(3.0 * SR), dtype=np.float32)
        score = sc._detect_reverb_excess(audio)
        assert score.severity == 0.0
        assert bool(score.metadata.get("silence_guarded", False))


# ============================================================
# RIAA_CURVE_ERROR — Medium-Gate (§6.3 Medium-Filter)
# ============================================================


class TestRiaaMediumGate:
    """RIAA_CURVE_ERROR must be suppressed on non-disc media.

    RIAA equalisation is physically relevant ONLY for disc media
    (vinyl, shellac, lacquer disc, wax cylinder).  A bass-heavy tape
    or MP3 signal must NEVER report RIAA_CURVE_ERROR ≠ 0.
    """

    @staticmethod
    def _bass_heavy_signal(duration: float = 3.0) -> np.ndarray:
        """Synthesise a bass-heavy signal whose bass/mid ratio > 5.

        This would trigger RIAA_CURVE_ERROR on a purely spectral check.
        """
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        bass = 0.8 * np.sin(2 * np.pi * 80 * t)  # dominant bass
        mid = 0.02 * np.sin(2 * np.pi * 2000 * t)  # very quiet mid
        return (bass + mid).astype(np.float32)  # type: ignore[no-any-return]

    def test_tape_no_riaa_detection(self):
        """Tape material + bass-heavy signal → RIAA severity must be 0."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        result = sc.scan(self._bass_heavy_signal(), material_type=MaterialType.TAPE, sample_rate=SR)
        riaa = result.scores[DefectType.RIAA_CURVE_ERROR]
        assert riaa.severity == 0.0, f"RIAA_CURVE_ERROR must be 0 on TAPE, got {riaa.severity:.3f}"

    def test_mp3_low_no_riaa_detection(self):
        """MP3_LOW material + bass-heavy signal → RIAA severity must be 0."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        result = sc.scan(self._bass_heavy_signal(), material_type=MaterialType.MP3_LOW, sample_rate=SR)
        riaa = result.scores[DefectType.RIAA_CURVE_ERROR]
        assert riaa.severity == 0.0, f"RIAA_CURVE_ERROR must be 0 on MP3_LOW, got {riaa.severity:.3f}"

    def test_cd_digital_no_riaa_detection(self):
        """CD_DIGITAL material + bass-heavy signal → RIAA severity must be 0."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        result = sc.scan(self._bass_heavy_signal(), material_type=MaterialType.CD_DIGITAL, sample_rate=SR)
        riaa = result.scores[DefectType.RIAA_CURVE_ERROR]
        assert riaa.severity == 0.0, f"RIAA_CURVE_ERROR must be 0 on CD_DIGITAL, got {riaa.severity:.3f}"

    def test_reel_tape_no_riaa_detection(self):
        """REEL_TAPE material + bass-heavy signal → RIAA severity must be 0."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        result = sc.scan(self._bass_heavy_signal(), material_type=MaterialType.REEL_TAPE, sample_rate=SR)
        riaa = result.scores[DefectType.RIAA_CURVE_ERROR]
        assert riaa.severity == 0.0, f"RIAA_CURVE_ERROR must be 0 on REEL_TAPE, got {riaa.severity:.3f}"

    def test_vinyl_allows_riaa_detection(self):
        """Vinyl material + bass-heavy signal → RIAA detection ALLOWED."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        result = sc.scan(self._bass_heavy_signal(), material_type=MaterialType.VINYL, sample_rate=SR)
        riaa = result.scores[DefectType.RIAA_CURVE_ERROR]
        # On vinyl, the spectral check should detect the imbalance
        assert riaa.severity > 0.0, "RIAA_CURVE_ERROR should be non-zero on VINYL with bass-heavy signal"

    def test_shellac_allows_riaa_detection(self):
        """Shellac material + bass-heavy signal → RIAA detection ALLOWED."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        result = sc.scan(self._bass_heavy_signal(), material_type=MaterialType.SHELLAC, sample_rate=SR)
        riaa = result.scores[DefectType.RIAA_CURVE_ERROR]
        assert riaa.severity > 0.0, "RIAA_CURVE_ERROR should be non-zero on SHELLAC with bass-heavy signal"

    def test_medium_gated_metadata_present(self):
        """When RIAA is suppressed, metadata must contain medium_gated=True."""
        from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

        sc = DefectScanner(sample_rate=SR)
        result = sc.scan(self._bass_heavy_signal(), material_type=MaterialType.TAPE, sample_rate=SR)
        riaa = result.scores[DefectType.RIAA_CURVE_ERROR]
        assert riaa.metadata.get("medium_gated") is True, (
            "medium_gated flag must be set when RIAA is suppressed on tape"
        )
        assert riaa.metadata.get("original_severity", 0.0) > 0.0, "original_severity must be preserved in metadata"
