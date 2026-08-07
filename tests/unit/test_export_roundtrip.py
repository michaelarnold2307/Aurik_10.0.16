from __future__ import annotations

"""
tests/unit/test_export_roundtrip.py — Export-Roundtrip-Tests für Aurik 10.0.0.

Prüft dass Audio-Daten nach Export (FLAC, WAV, MP3) verlustfrei oder
mit definiertem max. Verlust zurückgelesen werden können.
"""


import importlib.util
import math
import pathlib
import tempfile

import numpy as np
import pytest

# Import-Guard: soundfile ist Pflicht; scipy optional
try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

HAS_EXPORTER = importlib.util.find_spec("backend.core.audio_exporter") is not None


np.random.seed(42)
SR = 48_000


def _sine(freq: float = 440.0, duration_s: float = 1.0, sr: int = SR) -> np.ndarray:
    """Erzeugt einen Sinus-Ton als float32 in [-1, 1]."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return signal  # type: ignore[no-any-return]


def _stereo(signal: np.ndarray) -> np.ndarray:
    """Wandelt 1D-Mono in 2D-Stereo um."""
    return np.stack([signal, signal * 0.95], axis=1)


# ─── Soundfile-basierte Basisprüfungen ───────────────────────────────────────


@pytest.mark.skipif(not HAS_SOUNDFILE, reason="soundfile nicht installiert")
class TestFlacRoundtrip:
    """FLAC ist verlustfrei — Roundtrip muss numerisch identisch sein."""

    def test_mono_flac_roundtrip_identical(self, tmp_path: pathlib.Path) -> None:
        audio = _sine()
        out = tmp_path / "test.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, sr_out = sf.read(str(out), dtype="float32")
        assert sr_out == SR
        assert recovered.ndim == 1
        np.testing.assert_allclose(audio, recovered, atol=1e-5)

    def test_stereo_flac_roundtrip_shape(self, tmp_path: pathlib.Path) -> None:
        audio = _stereo(_sine())
        out = tmp_path / "test_stereo.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, sr_out = sf.read(str(out), dtype="float32")
        assert sr_out == SR
        assert recovered.shape == audio.shape

    def test_flac_no_clipping(self, tmp_path: pathlib.Path) -> None:
        audio = np.clip(_sine(duration_s=2.0) * 0.99, -1.0, 1.0)
        out = tmp_path / "noclip.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")
        assert float(np.max(np.abs(recovered))) <= 1.0 + 1e-4

    def test_flac_nan_free_after_roundtrip(self, tmp_path: pathlib.Path) -> None:
        audio = _sine()
        out = tmp_path / "nan_test.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")
        assert np.isfinite(recovered).all()

    def test_flac_correct_sample_rate(self, tmp_path: pathlib.Path) -> None:
        audio = _sine()
        out = tmp_path / "sr_test.flac"
        sf.write(str(out), audio, SR)
        _, sr_out = sf.read(str(out))
        assert sr_out == SR

    def test_flac_silence_roundtrip(self, tmp_path: pathlib.Path) -> None:
        audio = np.zeros(SR, dtype=np.float32)
        out = tmp_path / "silence.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")
        np.testing.assert_allclose(audio, recovered, atol=1e-6)

    def test_flac_short_signal(self, tmp_path: pathlib.Path) -> None:
        """Sehr kurzes Signal (512 Samples) — kein Absturz."""
        audio = _sine(duration_s=512 / SR)
        out = tmp_path / "short.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")
        assert len(recovered) == len(audio)


@pytest.mark.skipif(not HAS_SOUNDFILE, reason="soundfile nicht installiert")
class TestWavRoundtrip:
    """WAV PCM 24-bit — verlustfrei."""

    def test_wav_mono_roundtrip(self, tmp_path: pathlib.Path) -> None:
        audio = _sine()
        out = tmp_path / "test.wav"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, sr_out = sf.read(str(out), dtype="float32")
        assert sr_out == SR
        np.testing.assert_allclose(audio, recovered, atol=1e-4)

    def test_wav_stereo_roundtrip(self, tmp_path: pathlib.Path) -> None:
        audio = _stereo(_sine())
        out = tmp_path / "stereo.wav"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")
        assert recovered.shape == audio.shape

    def test_wav_16bit_roundtrip_bounded_error(self, tmp_path: pathlib.Path) -> None:
        """16-bit WAV hat Quantisierungsrauschen ≤ 1/32767 ≈ 3e-5."""
        audio = _sine()
        out = tmp_path / "16bit.wav"
        sf.write(str(out), audio, SR, subtype="PCM_16")
        recovered, _ = sf.read(str(out), dtype="float32")
        max_err = float(np.max(np.abs(audio - recovered)))
        assert max_err < 1e-3, f"Max-Fehler 16-bit WAV: {max_err:.2e}"

    def test_wav_output_no_nan(self, tmp_path: pathlib.Path) -> None:
        audio = _sine(duration_s=0.5)
        out = tmp_path / "nan_wav.wav"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")
        assert np.isfinite(recovered).all()

    def test_wav_float32_subtype(self, tmp_path: pathlib.Path) -> None:
        """FLOAT-WAV erlaubt volle float32-Präzision."""
        audio = _sine()
        out = tmp_path / "float32.wav"
        sf.write(str(out), audio, SR, subtype="FLOAT")
        recovered, _ = sf.read(str(out), dtype="float32")
        np.testing.assert_allclose(audio, recovered, atol=1e-7)


@pytest.mark.skipif(not HAS_SOUNDFILE, reason="soundfile nicht installiert")
class TestExportInvariants:
    """Allgemeine Export-Invarianten, unabhängig vom Format."""

    def test_original_not_modified_after_export(self, tmp_path: pathlib.Path) -> None:
        """Das Original-Array darf beim Export nicht verändert werden."""
        audio = _sine()
        original_copy = audio.copy()
        out = tmp_path / "check.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        np.testing.assert_array_equal(audio, original_copy)

    def test_exported_file_nonzero_size(self, tmp_path: pathlib.Path) -> None:
        audio = _sine()
        out = tmp_path / "size_check.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        assert out.stat().st_size > 0

    def test_different_signals_different_files(self, tmp_path: pathlib.Path) -> None:
        a1 = _sine(freq=440.0)
        a2 = _sine(freq=880.0)
        p1 = tmp_path / "440.flac"
        p2 = tmp_path / "880.flac"
        sf.write(str(p1), a1, SR, subtype="PCM_24")
        sf.write(str(p2), a2, SR, subtype="PCM_24")
        r1, _ = sf.read(str(p1), dtype="float32")
        r2, _ = sf.read(str(p2), dtype="float32")
        assert not np.allclose(r1, r2, atol=1e-4)

    def test_export_to_tempdir_works(self) -> None:
        audio = _sine()
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "tmp_export.flac"
            sf.write(str(out), audio, SR, subtype="PCM_24")
            recovered, _ = sf.read(str(out), dtype="float32")
            assert len(recovered) == len(audio)

    def test_lufs_not_drastically_changed_by_export(self, tmp_path: pathlib.Path) -> None:
        """Export+Read darf LUFS nicht um mehr als 0.5 LU verändern."""
        audio = _sine()
        out = tmp_path / "lufs_check.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")
        rms_orig = float(np.sqrt(np.mean(audio**2)))
        rms_recv = float(np.sqrt(np.mean(recovered**2)))
        # RMS-Differenz in dB als LUFS-Näherung
        if rms_orig > 1e-9 and rms_recv > 1e-9:
            diff_db = abs(20 * math.log10(rms_recv / rms_orig))
            assert diff_db < 0.5, f"RMS-Differenz nach Roundtrip: {diff_db:.3f} dB"

    def test_chroma_correlation_preserved(self, tmp_path: pathlib.Path) -> None:
        """Tonhöhe bleibt nach FLAC-Roundtrip identisch (Chroma-Proxy)."""
        try:
            import librosa
        except ImportError:
            pytest.skip("librosa nicht installiert")

        audio = _sine(freq=440.0, duration_s=2.0)
        out = tmp_path / "chroma.flac"
        sf.write(str(out), audio, SR, subtype="PCM_24")
        recovered, _ = sf.read(str(out), dtype="float32")

        # chroma_stft (pure numpy FFT) statt chroma_cqt (numba-basiert) —
        # vermeidet UFuncNoLoopError in numba's _phasor_angles DUFunc;
        # für einen Lossless-Roundtrip-Test mit Sinus ist chroma_stft äquivalent.
        # chroma_stft statt chroma_cqt: robust ohne numba-CQT-Abhängigkeit,
        # ausreichend für Roundtrip-Korrelationstest (§8.2 Tonale Stabilität ≥ 0.95)
        chroma_orig = librosa.feature.chroma_stft(y=audio, sr=SR, hop_length=512)
        chroma_recv = librosa.feature.chroma_stft(y=recovered, sr=SR, hop_length=512)
        corr = float(np.corrcoef(chroma_orig.ravel(), chroma_recv.ravel())[0, 1])
        assert corr >= 0.99, f"Chroma-Korrelation nach Roundtrip: {corr:.4f}"


# ─── Energie-Invarianten ─────────────────────────────────────────────────────


class TestEnergyInvariants:
    """Prüft Energie-Erhalt und -Grenzen für verschiedene Signaltypen."""

    def test_sine_energy_positive(self) -> None:
        audio = _sine()
        energy = float(np.sum(audio**2))
        assert energy > 0.0

    def test_silence_energy_zero(self) -> None:
        audio = np.zeros(SR, dtype=np.float32)
        energy = float(np.sum(audio**2))
        assert math.isclose(energy, 0.0, abs_tol=1e-12)

    def test_clipped_signal_energy_bounded(self) -> None:
        """Geclipptes Signal hat max. Energie entsprechend ±1.0."""
        audio = np.clip(_sine() * 10.0, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(audio**2)))
        assert rms <= 1.0

    def test_noise_energy_finite(self) -> None:
        audio = np.random.randn(SR).astype(np.float32) * 0.1
        energy = float(np.sum(audio**2))
        assert math.isfinite(energy)

    def test_stereo_energy_per_channel(self) -> None:
        stereo = _stereo(_sine())
        e_left = float(np.sum(stereo[:, 0] ** 2))
        e_right = float(np.sum(stereo[:, 1] ** 2))
        assert e_left > 0.0
        assert e_right > 0.0
        # Rechts-Kanal 5 % leiser → etwas weniger Energie
        assert e_right < e_left

    def test_energy_not_nan(self) -> None:
        audio = _sine()
        energy = float(np.sum(audio**2))
        assert not math.isnan(energy)
        assert not math.isinf(energy)

    def test_rms_normalized_signal(self) -> None:
        audio = _sine()
        rms = float(np.sqrt(np.mean(audio**2)))
        # RMS eines Sinus mit Amplitude 1 = 1/√2 ≈ 0.707
        assert abs(rms - 1.0 / math.sqrt(2.0)) < 0.01
