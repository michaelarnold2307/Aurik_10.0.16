from __future__ import annotations

import pytest

"""
tests/unit/test_v99_ab_compare.py — ABCompareManager Test-Suite (≥ 35 Tests)

Testet:
  • ABDiff-Berechnung und -Serialisierung
  • ABSession-Struktur und as_dict()
  • ABCompareManager.store() / get() / list_sessions()
  • RMS, spektrale Ähnlichkeit, SNR-Schätzung
  • LRU-Cache-Limit
  • Thread-Safety (Singleton + concurrent store)
  • JSON-Sidecar (Existenz und Korrektheit)
  • compare_audio() Rückgabe
  • NaN/Inf-Robustheit
  • Convenience-Funktion store_ab_session()
  • human_verdict()-Texte
"""


import json
import math
import threading
import time

import numpy as np

from backend.core.ab_compare_manager import (
    _AB_SESSION_DIR,
    MAX_SESSIONS,
    ABCompareManager,
    ABDiff,
    ABSession,
    get_ab_manager,
    store_ab_session,
)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

SR = 48_000


def _white(n: int = SR, amp: float = 0.3) -> np.ndarray:
    """Weißes Rauschen als float32."""
    return (np.random.randn(n) * amp).astype(np.float32)


def _sine(freq: float = 440.0, n: int = SR) -> np.ndarray:
    """Sinussignal als float32."""
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _fresh_manager() -> ABCompareManager:
    """Erzeugt eine frische, unabhängige Instanz (kein Singleton)."""
    return ABCompareManager()


# ===========================================================================
# 1. ABDiff — Struktur & Serialisierung
# ===========================================================================


@pytest.mark.unit
class TestABDiff:
    def test_01_as_dict_all_keys_present(self):
        d = ABDiff(
            rms_original_db=-20.0,
            rms_restored_db=-30.0,
            rms_improvement_db=10.0,
            spectral_similarity=0.85,
            snr_estimate_db=18.0,
            peak_original=0.8,
            peak_restored=0.6,
            duration_seconds=3.0,
            n_channels=1,
        )
        keys = d.as_dict()
        required = {
            "rms_original_db",
            "rms_restored_db",
            "rms_improvement_db",
            "spectral_similarity",
            "snr_estimate_db",
            "peak_original",
            "peak_restored",
            "duration_seconds",
            "n_channels",
        }
        assert required.issubset(set(keys.keys()))

    def test_02_as_dict_no_nan(self):
        d = ABDiff(
            rms_original_db=-20.0,
            rms_restored_db=-30.0,
            rms_improvement_db=10.0,
            spectral_similarity=0.85,
            snr_estimate_db=18.0,
            peak_original=0.8,
            peak_restored=0.6,
            duration_seconds=3.0,
            n_channels=1,
        )
        for v in d.as_dict().values():
            if isinstance(v, float):
                assert math.isfinite(v)

    def test_03_human_verdict_drama(self):
        d = ABDiff(
            rms_original_db=-10.0,
            rms_restored_db=-25.0,
            rms_improvement_db=15.0,
            spectral_similarity=0.70,
            snr_estimate_db=15.0,
            peak_original=0.9,
            peak_restored=0.7,
            duration_seconds=5.0,
            n_channels=1,
        )
        verdict = d.human_verdict()
        assert isinstance(verdict, str)
        assert len(verdict) > 5

    def test_04_human_verdict_no_change(self):
        d = ABDiff(
            rms_original_db=-20.0,
            rms_restored_db=-19.5,
            rms_improvement_db=-0.5,
            spectral_similarity=0.99,
            snr_estimate_db=30.0,
            peak_original=0.8,
            peak_restored=0.8,
            duration_seconds=5.0,
            n_channels=1,
        )
        verdict = d.human_verdict()
        assert isinstance(verdict, str)

    def test_05_human_verdict_all_thresholds(self):
        """Alle Verbesserungsstufen liefern einen String."""
        for delta in [15.0, 8.0, 4.0, 1.0, 0.0, -2.0]:
            d = ABDiff(
                rms_original_db=-20.0,
                rms_restored_db=-20.0 - delta,
                rms_improvement_db=delta,
                spectral_similarity=0.80,
                snr_estimate_db=15.0,
                peak_original=0.7,
                peak_restored=0.7,
                duration_seconds=3.0,
                n_channels=1,
            )
            assert isinstance(d.human_verdict(), str)


# ===========================================================================
# 2. ABCompareManager — store & get
# ===========================================================================


class TestStoreGet:
    def test_06_store_returns_uuid_string(self):
        m = _fresh_manager()
        sid = m.store(_white(), _sine(), SR, "vinyl")
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID-Format

    def test_07_get_returns_ab_session(self):
        m = _fresh_manager()
        sid = m.store(_white(), _sine(), SR, "tape")
        s = m.get(sid)
        assert isinstance(s, ABSession)

    def test_08_get_unknown_returns_none(self):
        m = _fresh_manager()
        assert m.get("nonexistent-id") is None

    def test_09_session_material_matches(self):
        m = _fresh_manager()
        sid = m.store(_white(), _sine(), SR, "shellac")
        s = m.get(sid)
        assert s is not None
        assert s.material == "shellac"

    def test_10_session_sample_rate_matches(self):
        m = _fresh_manager()
        sid = m.store(_white(), _sine(), SR, "vinyl")
        s = m.get(sid)
        assert s is not None
        assert s.sample_rate == SR

    def test_11_session_has_sha256(self):
        m = _fresh_manager()
        orig = _white()
        sid = m.store(orig, _sine(), SR, "vinyl")
        s = m.get(sid)
        assert s is not None
        assert len(s.original_sha256) == 16  # 16 Hex-Zeichen

    def test_12_different_originals_different_sha256(self):
        m = _fresh_manager()
        sig_a = _white(n=SR)
        sig_b = _white(n=SR)
        sid_a = m.store(sig_a, _sine(), SR)
        sid_b = m.store(sig_b, _sine(), SR)
        # Unterschiedliche Zufallssignale → sehr wahrscheinlich unterschiedliche SHA256
        sa = m.get(sid_a)
        sb = m.get(sid_b)
        assert sa is not None
        assert sb is not None
        assert sa.original_sha256 != sb.original_sha256

    def test_13_as_dict_has_human_verdict(self):
        m = _fresh_manager()
        sid = m.store(_white(), _sine(), SR)
        s = m.get(sid)
        assert s is not None
        d = s.as_dict()
        assert "human_verdict" in d
        assert isinstance(d["human_verdict"], str)

    def test_14_created_at_is_recent(self):
        m = _fresh_manager()
        t_before = time.time()
        sid = m.store(_white(), _sine(), SR)
        t_after = time.time()
        s = m.get(sid)
        assert s is not None
        assert t_before <= s.created_at <= t_after


# ===========================================================================
# 3. compare_audio()
# ===========================================================================


class TestCompareAudio:
    def test_15_returns_tuple_of_arrays(self):
        m = _fresh_manager()
        orig = _white()
        rest = _sine()
        sid = m.store(orig, rest, SR)
        result = m.compare_audio(sid)
        assert result is not None
        assert len(result) == 2
        o, r = result
        assert isinstance(o, np.ndarray)
        assert isinstance(r, np.ndarray)

    def test_16_compare_audio_is_copy(self):
        """Rückgabe ist eine Kopie — Veränderungen beeinflussen nicht den Cache."""
        m = _fresh_manager()
        orig = _white()
        sid = m.store(orig.copy(), _sine(), SR)
        first = m.compare_audio(sid)
        assert first is not None
        o, _ = first
        o[:] = 0.0
        second = m.compare_audio(sid)
        assert second is not None
        o2, _ = second
        assert not np.allclose(o2, 0.0)

    def test_17_unknown_session_returns_none(self):
        m = _fresh_manager()
        assert m.compare_audio("bad-id") is None


# ===========================================================================
# 4. ABDiff-Berechnungen
# ===========================================================================


class TestDiffCalculations:
    M = _fresh_manager()

    def test_18_identical_audio_high_similarity(self):
        sig = _sine()
        sid = self.M.store(sig.copy(), sig.copy(), SR)
        s = self.M.get(sid)
        assert s is not None
        diff = s.diff
        assert diff.spectral_similarity > 0.85

    def test_19_completely_different_low_similarity(self):
        sig1 = _sine(440.0)
        sig2 = _white()
        sid = self.M.store(sig1, sig2, SR)
        s = self.M.get(sid)
        assert s is not None
        diff = s.diff
        # Sehr unterschiedliche Signale → niedrige Ähnlichkeit
        assert diff.spectral_similarity < 0.90  # lockere Schranke

    def test_20_rms_db_silence_is_minus120(self):
        silence = np.zeros(SR, dtype=np.float32)
        assert self.M._rms_db(silence) == -120.0

    def test_21_rms_db_unity_signal_near_zero(self):
        unity = np.ones(SR, dtype=np.float32)
        rms = self.M._rms_db(unity)
        assert abs(rms) < 1.0  # 0 dBFS ± 1 dB

    def test_22_snr_identical_is_small(self):
        sig = _sine()
        snr = self.M._snr_estimate(sig, sig)
        # noise = 0 → SNR → ±inf → clipped to 0.0 by our guard
        assert snr == 0.0

    def test_23_spectral_similarity_in_01(self):
        for _ in range(5):
            s1 = _white()
            s2 = _sine(np.random.uniform(200, 1000))
            sim = self.M._spectral_similarity(s1, s2, SR)
            assert 0.0 <= sim <= 1.0, f"Ähnlichkeit außerhalb [0,1]: {sim}"

    def test_24_duration_correct(self):
        n = SR * 3  # 3 Sekunden
        sig = _white(n)
        sid = self.M.store(sig, sig.copy(), SR)
        s = self.M.get(sid)
        assert s is not None
        diff = s.diff
        assert abs(diff.duration_seconds - 3.0) < 0.1

    def test_25_peak_clipped_to_01(self):
        loud = np.full(SR, 2.0, dtype=np.float32)
        sid = self.M.store(loud, loud.copy(), SR)
        s = self.M.get(sid)
        assert s is not None
        diff = s.diff
        assert 0.0 <= diff.peak_original <= 1.0
        assert 0.0 <= diff.peak_restored <= 1.0


# ===========================================================================
# 5. NaN / Inf Robustheit
# ===========================================================================


class TestNaNInfRobustness:
    M = _fresh_manager()

    def test_26_nan_original_handled(self):
        orig = np.full(SR, float("nan"), dtype=np.float32)
        rest = _sine()
        sid = self.M.store(orig, rest, SR)
        assert sid is not None
        s = self.M.get(sid)
        assert s is not None
        assert math.isfinite(s.diff.rms_original_db)

    def test_27_inf_restored_handled(self):
        orig = _sine()
        rest = np.full(SR, float("inf"), dtype=np.float32)
        sid = self.M.store(orig, rest, SR)
        assert sid is not None

    def test_28_zero_duration_audio(self):
        """Leeres Audio (0 Samples) → store() gibt Session-ID zurück, kein Absturz."""
        empty = np.zeros(0, dtype=np.float32)
        filled = _sine()
        sid = self.M.store(empty, filled, SR)
        assert sid is not None

    def test_29_short_audio_no_crash(self):
        """Audio kürzer als ein FFT-Fenster → kein Absturz."""
        short = _white(n=64)
        sid = self.M.store(short, short.copy(), SR)
        assert sid is not None


# ===========================================================================
# 6. list_sessions & LRU
# ===========================================================================


class TestListLRU:
    def test_30_list_sessions_empty(self):
        m = _fresh_manager()
        assert m.list_sessions() == []

    def test_31_list_sessions_contains_stored(self):
        m = _fresh_manager()
        m.store(_white(), _sine(), SR, "vinyl")
        sessions = m.list_sessions()
        assert len(sessions) == 1
        assert "session_id" in sessions[0]

    def test_32_lru_evicts_oldest(self):
        m = _fresh_manager()
        ids = []
        for _ in range(MAX_SESSIONS + 2):
            ids.append(m.store(_white(), _sine(), SR))
        # Die ersten Sessions müssen entfernt sein
        assert m.get(ids[0]) is None
        # Die letzten Sessions müssen noch da sein
        assert m.get(ids[-1]) is not None

    def test_33_clear_empties_cache(self):
        m = _fresh_manager()
        m.store(_white(), _sine(), SR)
        m.clear()
        assert m.list_sessions() == []

    def test_34_latest_session_id_correct(self):
        m = _fresh_manager()
        m.store(_white(), _sine(), SR)
        sid2 = m.store(_sine(), _white(), SR)
        assert m.latest_session_id() == sid2


# ===========================================================================
# 7. Singleton & Thread-Safety
# ===========================================================================


class TestSingletonThread:
    def test_35_singleton_same_object(self):
        m1 = get_ab_manager()
        m2 = get_ab_manager()
        assert m1 is m2

    def test_36_concurrent_store_no_errors(self):
        """20 Threads speichern gleichzeitig — kein Absturz, alle Sessions gültig."""
        m = get_ab_manager()
        results, errors = [], []

        def worker():
            try:
                sid = m.store(_white(), _sine(), SR, material="vinyl")
                s = m.get(sid)
                results.append(s is not None)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Thread-Fehler: {errors}"
        assert all(results)

    def test_37_convenience_function_works(self):
        sid = store_ab_session(_white(), _sine(), SR, "tape")
        assert isinstance(sid, str)
        assert len(sid) == 36

    def test_38_sidecar_json_created(self):
        """JSON-Sidecar wird in ~/.aurik/ab_sessions/ geschrieben."""
        m = _fresh_manager()
        sid = m.store(_white(), _sine(), SR, "vinyl")
        short_id = sid[:8]
        path = _AB_SESSION_DIR / f"{short_id}.json"
        assert path.exists(), f"Sidecar nicht gefunden: {path}"

    def test_39_sidecar_json_valid(self):
        """Sidecar-JSON ist valide und enthält die Session-ID."""
        m = _fresh_manager()
        sig = _white()
        sid = m.store(sig, _sine(), SR, "tape")
        path = _AB_SESSION_DIR / f"{sid[:8]}.json"
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["session_id"] == sid
        assert "diff" in data
        assert "human_verdict" in data

    def test_40_stereo_audio_handled(self):
        """Stereo-Audio (2D) wird korrekt verarbeitet."""
        stereo_orig = np.random.randn(SR, 2).astype(np.float32) * 0.3
        stereo_rest = np.random.randn(SR, 2).astype(np.float32) * 0.2
        m = _fresh_manager()
        sid = m.store(stereo_orig, stereo_rest, SR)
        s = m.get(sid)
        assert s is not None
        assert s.diff.n_channels == 2
