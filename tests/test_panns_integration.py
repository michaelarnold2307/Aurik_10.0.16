#!/usr/bin/env python3
"""
Test PANNS Integration in AnalysisEngineAdapter
===============================================

Tests whether PANNS successfully extracts Genre, Vocals, and Instruments
and integrates them into AnalysisProfile.

Usage:
    python test_panns_integration.py <test_audio.wav>
"""

import logging
import sys
from pathlib import Path

import pytest
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.mark.timeout(600)
def test_panns_integration():
    """Test PANNS integration with AnalysisEngineAdapter (pytest-automatisiert)."""
    candidates = [
        Path("audio_examples/Elke_Best_Freund.mp3"),
        Path("audio_examples/Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"),
        Path("test_audio/Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"),
    ]
    test_audio = next((p for p in candidates if p.exists()), None)
    if test_audio is None:
        pytest.skip(
            "Keine reale MP3-Testdatei gefunden. Erwartet eine der Kandidaten: " + ", ".join(str(p) for p in candidates)
        )

    logger.info(f"Testing PANNS integration with: {test_audio}")
    audio, sr = sf.read(str(test_audio))
    logger.info(f"Loaded audio: {audio.shape}, {sr}Hz")
    from backend.core.forensics.analysis_and_modules import AnalysisEngineAdapter

    adapter = AnalysisEngineAdapter()
    profile = adapter.analyze(audio, sr, test_audio)  # type: ignore[arg-type]
    # Genre
    genre = profile.musical_context.genre  # type: ignore[attr-defined]
    genre_conf = profile.musical_context.genre_confidence  # type: ignore[attr-defined]
    # Vocals
    has_vocals = profile.vocal_analysis.has_vocals  # type: ignore[attr-defined]
    _ = profile.vocal_analysis.vocal_confidence  # type: ignore[attr-defined]
    # Instruments
    instruments = profile.musical_context.dominant_instruments  # type: ignore[attr-defined]
    # Assertions für pytest
    assert genre is not None, "Genre wurde nicht erkannt."
    assert genre_conf > 0.1, "Genre Confidence zu niedrig."
    assert isinstance(has_vocals, (bool, int)), "Vocal Detection fehlgeschlagen."
    assert isinstance(instruments, (list, tuple)), "Instrumentenerkennung fehlgeschlagen."
    # Mindestens ein Instrument oder Genre muss erkannt werden
    assert len(instruments) > 0 or (genre_conf > 0.3 and genre != "UNKNOWN"), "Weder Instrumente noch Genre erkannt."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_panns_integration.py <test_audio.wav>")
        print("\nExample:")
        print("  python test_panns_integration.py test_files/classical_piano.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        logger.error(f"File not found: {audio_path}")
        sys.exit(1)

    try:
        test_panns_integration()
        logger.info("Test completed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
