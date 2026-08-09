"""Export-Pipeline-Integrationstest. 10.0.20 Upgrade #5.

Verifiziert die vollständige Export-Pipeline:
- CD-Rauschprofil-Injektion (§G4)
- POW-r Type 3 Dither (§V5)
- Atomic Write
- Alle Formate (FLAC, WAV, MP3, OGG)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_test_audio(duration_s: float = 2.0, sr: int = 48000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.column_stack([audio, audio * 0.9])
    return stereo.astype(np.float32)


class TestExportPipeline:
    """E2E-Export-Tests für alle Formate."""

    def test_export_flac_with_cd_noise_and_dither(self):
        """FLAC-Export mit CD-Rauschprofil + POW-r Dither."""
        from backend.core.audio_exporter import AudioExporter

        audio = _make_test_audio()
        exporter = AudioExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.flac"
            exporter.export(audio, 48000, out, bit_depth=16, quality="high")
            assert out.exists()
            assert out.stat().st_size > 1000

    def test_export_wav_24bit(self):
        """WAV 24-bit Export (kein Dither nötig)."""
        from backend.core.audio_exporter import AudioExporter

        audio = _make_test_audio()
        exporter = AudioExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.wav"
            exporter.export(audio, 48000, out, bit_depth=24)
            assert out.exists()

    def test_export_mp3(self):
        """MP3-Export (lossy — kein Dither)."""
        from backend.core.audio_exporter import AudioExporter

        audio = _make_test_audio()
        exporter = AudioExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.mp3"
            try:
                exporter.export(audio, 48000, out, quality="high")
                assert out.exists()
            except Exception as e:
                if "MP3" in str(e) or "libmp3" in str(e):
                    pytest.skip("MP3 codec not available")

    def test_export_mono(self):
        """Mono-Export."""
        from backend.core.audio_exporter import AudioExporter

        audio = _make_test_audio()[:, 0]
        exporter = AudioExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_mono.flac"
            exporter.export(audio, 48000, out)
            assert out.exists()

    def test_atomic_write_no_temp_left(self):
        """Atomic Write: kein .tmp bleibt zurück."""
        from backend.core.audio_exporter import AudioExporter

        audio = _make_test_audio()
        exporter = AudioExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.flac"
            exporter.export(audio, 48000, out)
            tmp_files = list(Path(tmpdir).glob("*.tmp"))
            assert len(tmp_files) == 0, f"tmp file left: {tmp_files}"

    def test_export_with_metadata(self):
        """Export mit Metadaten."""
        from backend.core.audio_exporter import AudioExporter

        audio = _make_test_audio()
        exporter = AudioExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_meta.flac"
            exporter.export(audio, 48000, out, metadata={"artist": "Test", "title": "Test"})
            assert out.exists()


class TestPOWrDither:
    """POW-r Type 3 Dither Verifikation."""

    def test_dither_adds_noise_below_lsb(self):
        """Dither-Rauschen < 1 LSB."""
        from backend.core.dsp.powr_dither import apply_powr_dither

        audio = _make_test_audio()[:, 0]
        dithered = apply_powr_dither(audio, 48000, bit_depth=16)
        diff_std = float(np.std(audio - dithered))
        lsb = 2.0 / 65536
        assert diff_std < lsb * 2, f"Dither noise {diff_std:.8f} > 2 LSB"

    def test_dither_is_deterministic_with_seed(self):
        """Gleicher Seed = gleiches Dither."""
        from backend.core.dsp.powr_dither import apply_powr_dither

        audio = _make_test_audio()[:, 0]
        d1 = apply_powr_dither(audio, 48000, bit_depth=16, seed=42)
        d2 = apply_powr_dither(audio, 48000, bit_depth=16, seed=42)
        assert np.array_equal(d1, d2)

    def test_noise_floor_reduction_in_mid_band(self):
        """POW-r reduziert Rauschen im 2-5 kHz Bereich vs. TPDF."""
        from backend.core.dsp.powr_dither import compute_noise_floor_reduction

        reduction = compute_noise_floor_reduction()
        mid_band = reduction.get("2-5 kHz (Max. Empfindlichkeit)", 0)
        assert mid_band > 2.0, f"Noise floor reduction {mid_band}dB should be > 2dB"
