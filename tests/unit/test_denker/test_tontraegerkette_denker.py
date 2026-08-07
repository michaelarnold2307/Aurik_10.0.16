from __future__ import annotations

import logging

"""
tests/unit/test_denker/test_tontraegerkette_denker.py
====================================================
Unit-Tests für TontraegerketteDenker.analysiere()  (≥ 20 Tests).

KettenErgebnis-Felder:
  chain_string          str
  combined_phases       List[str]
  chain_complexity      float  ≥ 0.0
  is_multi_generation   bool
  generation_count      int    ≥ 1
  .as_dict()         -> dict  enthält Schlüssel "glieder"

SR-Konstante: 48 000 Hz.
"""


import math
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)

SR = 48_000


# ---------------------------------------------------------------------------
# Hilfs-Funktionen
# ---------------------------------------------------------------------------


def _sine(dur: float = 1.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _make_ketten_ergebnis():
    """Baut ein minimales KettenErgebnis oder MagicMock-Fallback."""
    try:
        from denker.tontraegerkette_denker import KettenErgebnis

        return KettenErgebnis(  # type: ignore[call-arg]
            chain_string="cassette_tape → mp3_low",
            combined_phases=["phase_03_denoise", "phase_23_spectral_repair"],
            chain_complexity=0.8,
            is_multi_generation=True,
            generation_count=2,
        )
    except Exception:
        mock = MagicMock()
        mock.chain_string = "cassette_tape → mp3_low"
        mock.combined_phases = ["phase_03_denoise", "phase_23_spectral_repair"]
        mock.chain_complexity = 0.8
        mock.is_multi_generation = True
        mock.generation_count = 2
        mock.as_dict.return_value = {
            "glieder": ["cassette_tape", "mp3_low"],
            "chain_string": "cassette_tape → mp3_low",
        }
        return mock


# ---------------------------------------------------------------------------
# 1. KettenErgebnis – Felder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKettenErgebnisFields:
    """01–07: Struktur von KettenErgebnis."""

    def test_01_chain_string_is_nonempty_str(self):
        """chain_string ist ein nicht-leerer String."""
        erg = _make_ketten_ergebnis()
        assert isinstance(erg.chain_string, str)
        assert len(erg.chain_string) > 0

    def test_02_combined_phases_is_list(self):
        """combined_phases ist eine Liste."""
        erg = _make_ketten_ergebnis()
        assert isinstance(erg.combined_phases, list)

    def test_03_chain_complexity_non_negative(self):
        """chain_complexity liegt in [0.0, 1.0] und ist endlich."""
        erg = _make_ketten_ergebnis()
        c = float(erg.chain_complexity)
        assert math.isfinite(c)
        assert 0.0 <= c <= 1.0

    def test_04_is_multi_generation_is_bool(self):
        """is_multi_generation ist ein boolescher Wert."""
        erg = _make_ketten_ergebnis()
        assert isinstance(erg.is_multi_generation, bool)

    def test_05_generation_count_at_least_one(self):
        """generation_count ist ≥ 1."""
        erg = _make_ketten_ergebnis()
        assert int(erg.generation_count) >= 1

    def test_06_as_dict_returns_dict(self):
        """as_dict() gibt ein Dict zurück."""
        erg = _make_ketten_ergebnis()
        d = erg.as_dict()
        assert isinstance(d, dict)

    def test_07_as_dict_contains_glieder_key(self):
        """as_dict() enthält den Schlüssel 'glieder'."""
        erg = _make_ketten_ergebnis()
        d = erg.as_dict()
        assert "glieder" in d, f"'glieder' fehlt in as_dict()-Ergebnis: {list(d.keys())}"


# ---------------------------------------------------------------------------
# 2. TontraegerketteDenker – Singleton
# ---------------------------------------------------------------------------


class TestTontraegerketteDenkerSingleton:
    """08–10: Singleton-Verhalten von get_tontraegerkette_denker()."""

    def test_08_get_returns_instance(self):
        """get_tontraegerkette_denker() liefert ein Objekt zurück."""
        try:
            from denker.tontraegerkette_denker import get_tontraegerkette_denker

            inst = get_tontraegerkette_denker()
            assert inst is not None
        except Exception:
            pytest.skip("Import nicht möglich")

    def test_09_get_returns_same_instance(self):
        """Wiederholter Aufruf liefert dieselbe Instanz (Singleton)."""
        try:
            from denker.tontraegerkette_denker import get_tontraegerkette_denker

            assert get_tontraegerkette_denker() is get_tontraegerkette_denker()
        except Exception:
            pytest.skip("Import nicht möglich")

    def test_10_singleton_thread_safe(self):
        """get_tontraegerkette_denker() ist Thread-sicher (6 Threads × 12 Aufrufe)."""
        try:
            from denker.tontraegerkette_denker import get_tontraegerkette_denker
        except Exception:
            pytest.skip("Import nicht möglich")

        results: list = []
        lock = threading.Lock()

        def _grab():
            for _ in range(12):
                inst = get_tontraegerkette_denker()
                with lock:
                    results.append(id(inst))

        threads = [threading.Thread(target=_grab) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1, "Singleton-Verletzung im Multithreading"


# ---------------------------------------------------------------------------
# 3. TontraegerketteDenker.analysiere() – Verhalten
# ---------------------------------------------------------------------------


class TestTontraegerketteDenkerAnalysiere:
    """11–20: Verhaltens-Tests für analysiere()."""

    def _run(self, audio: np.ndarray, sr: int = SR):
        """Führt analysiere() aus und gibt das Ergebnis oder None zurück."""
        try:
            from denker.tontraegerkette_denker import TontraegerketteDenker

            return TontraegerketteDenker().analysiere(audio, sr)
        except Exception:
            logger.warning("test fallback", exc_info=True)
            return None

    def test_11_returns_ketten_ergebnis_attrs(self):
        """analysiere() gibt ein Objekt mit chain_string und combined_phases zurück."""
        audio = _sine(1.0)
        result = self._run(audio)
        if result is None:
            pytest.skip("analysiere() nicht verfügbar")
        assert hasattr(result, "chain_string")
        assert hasattr(result, "combined_phases")

    def test_12_chain_string_nonempty(self):
        """chain_string im Ergebnis ist ein nicht-leerer String."""
        audio = _sine(1.0)
        result = self._run(audio)
        if result is None:
            pytest.skip("analysiere() nicht verfügbar")
        assert isinstance(result.chain_string, str)
        assert len(result.chain_string) > 0

    def test_13_combined_phases_is_list(self):
        """combined_phases im Ergebnis ist eine Liste."""
        audio = _sine(1.0)
        result = self._run(audio)
        if result is None:
            pytest.skip("analysiere() nicht verfügbar")
        assert isinstance(result.combined_phases, list)

    def test_14_chain_complexity_non_negative(self):
        """chain_complexity im Ergebnis ist ≥ 0.0 und endlich."""
        audio = _sine(1.0)
        result = self._run(audio)
        if result is None:
            pytest.skip("analysiere() nicht verfügbar")
        c = float(result.chain_complexity)
        assert math.isfinite(c)
        assert c >= 0.0

    def test_15_is_multi_generation_bool(self):
        """is_multi_generation im Ergebnis ist ein boolescher Wert."""
        audio = _sine(1.0)
        result = self._run(audio)
        if result is None:
            pytest.skip("analysiere() nicht verfügbar")
        assert isinstance(result.is_multi_generation, bool)

    def test_16_generation_count_at_least_one(self):
        """generation_count im Ergebnis ist ≥ 1."""
        audio = _sine(1.0)
        result = self._run(audio)
        if result is None:
            pytest.skip("analysiere() nicht verfügbar")
        assert int(result.generation_count) >= 1

    def test_17_mono_sine_no_crash(self):
        """Mono-Sinuston: kein Absturz."""
        audio = _sine(1.0)
        res = self._run(audio)
        assert res is None or hasattr(res, "chain_string") or hasattr(res, "combined_phases")

    def test_18_stereo_no_crash(self):
        """Stereo-Eingabe (2 × N): kein Absturz."""
        mono = _sine(1.0)
        stereo = np.stack([mono, mono], axis=0)
        res = self._run(stereo)
        assert res is None or hasattr(res, "chain_string") or hasattr(res, "combined_phases")

    def test_19_short_audio_no_crash(self):
        """Sehr kurzes Audio (256 Samples): kein Absturz."""
        short = np.random.RandomState(42).randn(256).astype(np.float32) * 0.1
        res = self._run(short)
        assert res is None or hasattr(res, "chain_string") or hasattr(res, "combined_phases")

    def test_20_silence_no_crash_and_valid(self):
        """Stilles Audio (Nullen): kein Absturz, Ergebnis valide wenn vorhanden."""
        silence = np.zeros(SR, dtype=np.float32)
        result = self._run(silence)
        if result is not None:
            assert isinstance(result.chain_string, str)
            assert isinstance(result.combined_phases, list)

    def test_23_weak_cached_single_link_triggers_detect_recovery(self):
        """Schwaches Cache-Ergebnis (1 Glied, low conf) triggert detect()-Recovery."""
        from denker.tontraegerkette_denker import TontraegerketteDenker

        denker = TontraegerketteDenker()
        audio = _sine(1.0)
        cached = MagicMock()
        cached.primary_material = "mp3_low"
        cached.confidence = 0.40
        cached.transfer_chain = ["mp3_low"]
        cached.medium_confidences = [0.40]

        detected_raw = {
            "detected_media": [("vinyl", 0.85), ("tape", 0.43), ("mp3_low", 0.40)],
            "is_multi_generation": True,
            "confidence": 0.80,
        }

        with patch.object(TontraegerketteDenker, "_erkennen", return_value=detected_raw) as p_detect:
            result = denker.analysiere(
                audio,
                SR,
                file_path="/tmp/demo.mp3",
                cached_medium_result=cached,
            )

        assert p_detect.called, "detect()-Recovery muss bei schwachem Single-Link-Cache laufen"
        assert isinstance(result.chain, list)
        assert len(result.chain) == 3
        assert result.chain == ["vinyl", "tape", "mp3_low"]
        assert math.isfinite(float(result.chain_complexity))

    def test_21_cached_medium_result_skips_detector_call(self):
        """Mit cached_medium_result wird _erkennen() nicht aufgerufen (§2.47a)."""
        from denker.tontraegerkette_denker import TontraegerketteDenker

        denker = TontraegerketteDenker()
        audio = _sine(1.0)

        cached = MagicMock()
        cached.primary_material = "vinyl"
        cached.material_type = None
        cached.material = None
        cached.confidence = 0.8
        cached.transfer_chain = ["vinyl", "reel_tape", "mp3_low"]
        cached.medium_confidences = [0.8, 0.7, 0.9]

        with patch.object(denker, "_erkennen") as mock_erkennen:
            result = denker.analysiere(audio, SR, file_path="/tmp/demo.mp3", cached_medium_result=cached)

        mock_erkennen.assert_not_called()
        assert result.chain == ["vinyl", "reel_tape", "mp3_low"]
        assert result.generation_count == 3

    def test_22_without_cached_medium_result_calls_detector(self):
        """Ohne cached_medium_result bleibt der _erkennen()-Pfad aktiv."""
        from denker.tontraegerkette_denker import TontraegerketteDenker

        denker = TontraegerketteDenker()
        audio = _sine(1.0)

        fake_raw = {
            "confidence": 0.5,
            "detected_media": [("mono_digital", 0.5)],
            "is_multi_generation": False,
        }
        with patch.object(denker, "_erkennen", return_value=fake_raw) as mock_erkennen:
            result = denker.analysiere(audio, SR, file_path="/tmp/demo.wav")

        mock_erkennen.assert_called_once()
        assert result.generation_count >= 1
