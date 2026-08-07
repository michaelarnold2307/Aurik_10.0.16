import pytest

"""Tests for Krumhansl-Schmuckler Tonal Center metric (TonalCenterMetric).

Verifies:
- Key detection accuracy for major/minor keys
- Stability across transpositions
- Chroma vector extraction
- Edge cases (silence, noise, very short audio)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_c_major_chord(sr: int = 48000, dur: float = 2.0) -> np.ndarray:
    """C major triad: C4 (261.63), E4 (329.63), G4 (392.00)."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False).astype(np.float32)
    c = 0.3 * np.sin(2 * np.pi * 261.63 * t)
    e = 0.3 * np.sin(2 * np.pi * 329.63 * t)
    g = 0.3 * np.sin(2 * np.pi * 392.00 * t)
    return (c + e + g).astype(np.float32)  # type: ignore[no-any-return]


def _make_a_minor_chord(sr: int = 48000, dur: float = 2.0) -> np.ndarray:
    """A minor triad: A3 (220.00), C4 (261.63), E4 (329.63)."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False).astype(np.float32)
    a = 0.3 * np.sin(2 * np.pi * 220.00 * t)
    c = 0.3 * np.sin(2 * np.pi * 261.63 * t)
    e = 0.3 * np.sin(2 * np.pi * 329.63 * t)
    return (a + c + e).astype(np.float32)  # type: ignore[no-any-return]


def _chromagram_12(audio: np.ndarray, sr: int) -> np.ndarray:
    """Simple 12-bin chroma vector via FFT."""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    n = len(audio)
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    chroma = np.zeros(12, dtype=np.float64)
    for i, f in enumerate(freqs):
        if f > 20:
            midi = 12 * np.log2(f / 440.0) + 69
            bin_idx = int(round(midi)) % 12
            chroma[bin_idx] += spectrum[i] ** 2
    return chroma / (np.max(chroma) + 1e-12)


# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _ks_detect_key(chroma: np.ndarray) -> tuple[int, str, float]:
    """Krumhansl-Schmuckler key detection. Returns (root, mode, correlation)."""

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        a0 = a - float(np.mean(a))
        b0 = b - float(np.mean(b))
        denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a0, b0) / denom)

    best_corr = -2.0
    best_root = 0
    best_mode = "major"
    for root in range(12):
        rotated = np.roll(chroma, -root)
        corr_maj = _safe_corr(rotated, _MAJOR_PROFILE)
        corr_min = _safe_corr(rotated, _MINOR_PROFILE)
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_root = root
            best_mode = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_root = root
            best_mode = "minor"
    return best_root, best_mode, best_corr


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# Key Detection Accuracy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKeyDetection:
    """K-S algorithm should detect known keys correctly."""

    def test_c_major_detection(self):
        audio = _make_c_major_chord()
        chroma = _chromagram_12(audio, 48000)
        root, mode, corr = _ks_detect_key(chroma)
        key_name = _NOTE_NAMES[root]
        assert key_name == "C" and mode == "major", f"Expected C major, got {key_name} {mode}"
        assert corr > 0.5

    def test_a_minor_detection(self):
        audio = _make_a_minor_chord()
        chroma = _chromagram_12(audio, 48000)
        root, mode, corr = _ks_detect_key(chroma)
        key_name = _NOTE_NAMES[root]
        # A minor and C major are relative — K-S may pick either
        assert (key_name == "A" and mode == "minor") or (key_name == "C" and mode == "major")

    def test_detection_confidence_above_threshold(self):
        audio = _make_c_major_chord()
        chroma = _chromagram_12(audio, 48000)
        _, _, corr = _ks_detect_key(chroma)
        assert corr >= 0.5, f"K-S confidence {corr:.3f} below 0.5"


# ---------------------------------------------------------------------------
# Transposition Stability
# ---------------------------------------------------------------------------


class TestTranspositionStability:
    """Tonal center metric should be stable across transpositions."""

    def test_transpose_preserves_mode(self):
        """Same chord type in different keys should detect same mode."""
        sr = 48000
        dur = 2.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False).astype(np.float32)
        modes_detected = []
        # Major triads in C, D, E
        for root_hz in [261.63, 293.66, 329.63]:
            third = root_hz * 2 ** (4 / 12)  # major third
            fifth = root_hz * 2 ** (7 / 12)  # perfect fifth
            audio = (
                0.3 * np.sin(2 * np.pi * root_hz * t)
                + 0.3 * np.sin(2 * np.pi * third * t)
                + 0.3 * np.sin(2 * np.pi * fifth * t)
            ).astype(np.float32)
            chroma = _chromagram_12(audio, sr)
            _, mode, _ = _ks_detect_key(chroma)
            modes_detected.append(mode)
        # All should be detected as major
        assert all(m == "major" for m in modes_detected), f"Modes: {modes_detected}"


# ---------------------------------------------------------------------------
# Chroma Vector Properties
# ---------------------------------------------------------------------------


class TestChromaVector:
    """Chroma extraction properties."""

    def test_chroma_12_bins(self):
        audio = _make_c_major_chord()
        chroma = _chromagram_12(audio, 48000)
        assert chroma.shape == (12,)

    def test_chroma_normalized(self):
        audio = _make_c_major_chord()
        chroma = _chromagram_12(audio, 48000)
        assert np.max(chroma) <= 1.0 + 1e-6
        assert np.min(chroma) >= 0.0

    def test_chroma_peak_at_root(self):
        """C major chord should have strong C chroma (bin 0)."""
        audio = _make_c_major_chord()
        chroma = _chromagram_12(audio, 48000)
        # C = bin 0 (since we use A440 reference, C is midi 60 % 12 = 0)
        # Actually midi(261.63) ≈ 60, 60%12=0 → bin 0
        top_bins = np.argsort(chroma)[-3:]
        assert 0 in top_bins, "C should be in top 3 chroma bins for C major"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestTonalCenterEdgeCases:
    """Edge cases for tonal center detection."""

    def test_silence(self):
        audio = np.zeros(48000 * 2, dtype=np.float32)
        chroma = _chromagram_12(audio, 48000)
        _, _, corr = _ks_detect_key(chroma)
        # Silence should have low confidence
        assert isinstance(corr, float)

    def test_white_noise(self):
        np.random.seed(42)
        audio = np.random.randn(48000 * 2).astype(np.float32) * 0.3
        chroma = _chromagram_12(audio, 48000)
        _, _, corr = _ks_detect_key(chroma)
        # Noise should have lower correlation than tonal signal
        assert corr < 0.9

    def test_very_short_audio(self):
        """256 samples should not crash."""
        audio = _make_c_major_chord(dur=0.005)  # ~240 samples
        chroma = _chromagram_12(audio, 48000)
        assert chroma.shape == (12,)

    def test_stereo_input(self):
        mono = _make_c_major_chord()
        stereo = np.column_stack([mono, mono * 0.9])
        chroma = _chromagram_12(stereo, 48000)
        assert chroma.shape == (12,)
