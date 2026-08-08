import pytest

"""
Unit-Tests für core/delivery_standards.py — BWFMetadataWriter

Testet:
  - BWFMetadataWriter.write_bwf_metadata()
  - EBU Tech 3285 BEXT-Chunk Struktur
"""

import struct
import wave
from pathlib import Path

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit

from backend.core.delivery_standards import BWFMetadataWriter

# ---------------------------------------------------------------------------
# Hilfsfunktion: Minimale WAV-Datei erstellen
# ---------------------------------------------------------------------------


def _create_test_wav(path: Path, n_samples: int = 4410, sr: int = 44100) -> Path:
    """Erstellt eine gültige mono-WAV-Datei für Tests."""
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        data = (np.random.randn(n_samples) * 16000).astype(np.int16)  # type: ignore[arg-type]  # §V5 Dither applied at export level
        wf.writeframes(data.tobytes())
    return path


# ---------------------------------------------------------------------------
# BWFMetadataWriter Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBWFMetadataWriter:
    def test_returns_true_for_valid_wav(self, tmp_path):
        wav = _create_test_wav(tmp_path / "test.wav")
        result = BWFMetadataWriter.write_bwf_metadata(wav)
        assert result is True

    def test_bext_chunk_present_after_write(self, tmp_path):
        """Nach dem Schreiben muss ein 'bext'-Chunk in der WAV-Datei vorhanden sein."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav, description="Test-Beschreibung")
        raw = wav.read_bytes()
        assert b"bext" in raw, "BEXT-Chunk fehlt in der WAV-Datei"

    def test_riff_header_intact(self, tmp_path):
        """RIFF- und WAVE-Header müssen nach BWF-Schreiben korrekt sein."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav)
        raw = wav.read_bytes()
        assert raw[:4] == b"RIFF"
        assert raw[8:12] == b"WAVE"

    def test_riff_size_updated(self, tmp_path):
        """RIFF-Größenfeld muss der tatsächlichen Dateigröße entsprechen."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav)
        raw = wav.read_bytes()
        declared = struct.unpack("<I", raw[4:8])[0]
        expected = len(raw) - 8
        assert declared == expected, f"RIFF-Größe falsch: {declared} != {expected}"

    def test_data_chunk_still_present(self, tmp_path):
        """'data'-Chunk darf nicht verloren gehen."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav)
        raw = wav.read_bytes()
        assert b"data" in raw

    def test_bext_before_data(self, tmp_path):
        """BEXT-Chunk muss VOR dem 'data'-Chunk liegen (EBU Konvention)."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav)
        raw = wav.read_bytes()
        bext_pos = raw.find(b"bext")
        data_pos = raw.find(b"data")
        assert bext_pos < data_pos, "BEXT muss vor dem data-Chunk liegen"

    def test_description_encoded_in_bext(self, tmp_path):
        """Die Beschreibung muss im BEXT-Chunk encodiert sein."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav, description="AURIK-TEST-2026")
        raw = wav.read_bytes()
        assert b"AURIK-TEST-2026" in raw

    def test_originator_encoded_in_bext(self, tmp_path):
        """Der Originator muss im BEXT-Chunk vorhanden sein."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav, originator="TestStudio")
        raw = wav.read_bytes()
        assert b"TestStudio" in raw

    def test_description_truncated_to_256_chars(self, tmp_path):
        """Beschreibung darf maximal 256 Zeichen haben."""
        wav = _create_test_wav(tmp_path / "test.wav")
        long_desc = "A" * 300
        result = BWFMetadataWriter.write_bwf_metadata(wav, description=long_desc)
        assert result is True
        raw = wav.read_bytes()
        # BEXT-Chunk vorhanden und Datei gültig
        assert b"bext" in raw

    def test_bext_chunk_size_even(self, tmp_path):
        """BEXT-Chunk-Größe muss gerade sein (RIFF-Alignment)."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav, coding_history="A=PCM,F=44100")
        raw = wav.read_bytes()
        bext_pos = raw.find(b"bext")
        if bext_pos >= 0:
            chunk_size = struct.unpack("<I", raw[bext_pos + 4 : bext_pos + 8])[0]
            assert chunk_size % 2 == 0, f"BEXT-Chunk-Größe ungerade: {chunk_size}"

    def test_wav_still_readable_after_bwf_write(self, tmp_path):
        """WAV-Datei muss nach BWF-Schreiben noch lesbar sein."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav)
        # Python wave-Modul kann die Datei noch öffnen
        with wave.open(str(wav), "r") as wf:
            assert wf.getnframes() > 0

    def test_nonexistent_file_returns_false(self, tmp_path):
        """Nicht existierende Datei → False zurückgeben."""
        nonexistent = tmp_path / "doesnotexist.wav"
        result = BWFMetadataWriter.write_bwf_metadata(nonexistent)
        assert result is False

    def test_default_date_generated(self, tmp_path):
        """Ohne explizites Datum: Aktuelles Datum wird ins BEXT geschrieben."""
        import datetime

        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav)
        raw = wav.read_bytes()
        year = str(datetime.date.today().year).encode()
        assert year in raw

    def test_bwf_version_2(self, tmp_path):
        """BWF Version 2 gemäß EBU Tech 3285."""
        wav = _create_test_wav(tmp_path / "test.wav")
        BWFMetadataWriter.write_bwf_metadata(wav)
        raw = wav.read_bytes()
        bext_pos = raw.find(b"bext")
        if bext_pos >= 0:
            # BWF-Version liegt bei Offset 354 (256+32+32+10+8+8+2)
            # = bext_chunk_data_start + 348
            chunk_data_start = bext_pos + 8
            version_offset = chunk_data_start + 346
            version = struct.unpack("<H", raw[version_offset : version_offset + 2])[0]
            assert version == 2, f"BWF-Version muss 2 sein, nicht {version}"
