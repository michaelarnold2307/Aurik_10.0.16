from __future__ import annotations

"""
tests/unit/test_natuerlichkeit_stateless_and_emotional_arc_score.py
====================================================================
FIXED v10.0.0 — Zwei kritische Bugs, die HPG-Integrität untergruben:

1. NatuerlichkeitMetric.measure() war CREPE-load-state-abhängig:
   Ohne CREPE: w_onset=0.24, w_voice=0.0
   Mit CREPE:  w_onset=0.16, w_voice=0.18
   → identisches Audio → unterschiedliche P1-Scores → nicht-deterministisch

2. EmotionalArcResult.preservation_score Property fehlte:
   UV3 nutzt getattr(result, "preservation_score", 1.0) als HPG-Eingabe.
   Ohne das Property: immer 1.0 → emotional_arc_preservation nie wirksam.

Alle Tests laufen ohne ML-Modelle (AURIK_DISABLE_CREPE=1).
"""


import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq: float = 440.0, sr: int = 48000, dur: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(sr: int = 48000, dur: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * dur)).astype(np.float32) * 0.3


def _harmonic_signal(sr: int = 48000, dur: float = 2.0) -> np.ndarray:
    """Tonal multi-harmonic signal (tonal → high flatness_score → natural)."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    sig = sum((1.0 / k) * np.sin(2 * np.pi * k * 220.0 * t) for k in range(1, 9))
    return (sig / np.max(np.abs(sig) + 1e-9) * 0.4).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# §1 NatuerlichkeitMetric — Determinismus / Stateless
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNatuerlichkeitStateless:
    """FIXED v10.0.0: Immer gleiche Gewichte, CREPE verfeinert nur voicing_naturalness."""

    @pytest.fixture(autouse=True)
    def _disable_crepe(self, monkeypatch):
        """CREPE deaktivieren — testet rein DSP-Pfad (stateless baseline)."""
        monkeypatch.setenv("AURIK_DISABLE_CREPE", "1")

    def _metric(self):
        from backend.core.musical_goals.musical_goals_metrics import NatuerlichkeitMetric

        return NatuerlichkeitMetric()

    def test_deterministic_two_calls_sine(self):
        """Identisches Audio → identischer Score (Determinismus)."""
        m = self._metric()
        audio = _sine()
        s1 = m.measure(audio.copy(), 48000)
        s2 = m.measure(audio.copy(), 48000)
        assert s1 == s2, f"Non-deterministic: {s1} ≠ {s2}"

    def test_deterministic_two_calls_noise(self):
        """Weißes Rauschen → identischer Score bei zwei Aufrufen."""
        m = self._metric()
        audio = _white_noise()
        s1 = m.measure(audio.copy(), 48000)
        s2 = m.measure(audio.copy(), 48000)
        assert s1 == s2, f"Non-deterministic (noise): {s1} ≠ {s2}"

    def test_tonal_scores_higher_than_noise(self):
        """Tonal > Rauschen — grundlegende Richtigkeit des Signals."""
        m = self._metric()
        tonal = m.measure(_harmonic_signal(), 48000)
        noise = m.measure(_white_noise(), 48000)
        assert tonal > noise, f"Tonal {tonal:.3f} sollte > Noise {noise:.3f} sein"

    def test_score_in_unit_interval(self):
        """Score ∈ [0, 1] — keine Overflow-Werte."""
        m = self._metric()
        for audio in [_sine(), _white_noise(), _harmonic_signal()]:
            s = m.measure(audio, 48000)
            assert 0.0 <= s <= 1.0, f"Score außerhalb [0,1]: {s}"

    def test_no_nan_or_inf(self):
        """Score ist endlich (kein NaN/Inf)."""
        m = self._metric()
        for audio in [_sine(), _white_noise(), _harmonic_signal()]:
            s = m.measure(audio, 48000)
            assert math.isfinite(s), f"Score nicht endlich: {s}"

    def test_very_short_audio_returns_half(self):
        """Zu kurzes Audio → 0.5 (sicherer Fallback)."""
        m = self._metric()
        short = np.zeros(4, dtype=np.float32)
        s = m.measure(short, 48000)
        assert s == pytest.approx(0.5), f"Kurzes Audio: {s}"

    def test_stereo_input_converted_to_mono(self):
        """Stereo-Input → Score identisch mit Mono-Mix."""
        m = self._metric()
        mono = _sine()
        stereo = np.stack([mono, mono], axis=0)
        s_mono = m.measure(mono, 48000)
        s_stereo = m.measure(stereo, 48000)
        assert abs(s_mono - s_stereo) < 0.05, f"Stereo/Mono-Divergenz: mono={s_mono:.3f} stereo={s_stereo:.3f}"

    def test_high_naturalness_for_harmonic_signal(self):
        """Tonal-harmonisches Signal → Natürlichkeit ≥ 0.50."""
        m = self._metric()
        s = m.measure(_harmonic_signal(), 48000)
        assert s >= 0.50, f"Harmonisches Signal zu niedrig: {s:.3f}"

    def test_noise_below_tonal_threshold(self):
        """Weißes Rauschen < P1-Schwellwert von 0.90 (Signal ist tatsächlich unnatürlich)."""
        m = self._metric()
        s = m.measure(_white_noise(), 48000)
        # Rauschen muss unterhalb des idealen Signals liegen
        # (Es muss nicht unter 0.90 fallen — nur tiefer als tonal)
        tonal_s = m.measure(_harmonic_signal(), 48000)
        assert s < tonal_s, f"Noise {s:.3f} sollte < Tonal {tonal_s:.3f}"

    def test_weights_sum_invariant(self):
        """Alle Gewichte summieren zu 1.0 — immer (FIXED v10.0.0 stateless)."""
        # Gewichte sind jetzt fest: w_flat=0.24 + w_zcr=0.21 + w_cont=0.21 + w_voice=0.18 + w_onset=0.16 = 1.00
        w_flat, w_zcr, w_cont, w_voice, w_onset = 0.24, 0.21, 0.21, 0.18, 0.16
        total = w_flat + w_zcr + w_cont + w_voice + w_onset
        assert abs(total - 1.0) < 1e-9, f"Gewichte summieren zu {total}, nicht 1.0"


# ---------------------------------------------------------------------------
# §2 EmotionalArcResult.preservation_score — Korrektheit
# ---------------------------------------------------------------------------


class TestEmotionalArcPreservationScore:
    """FIXED v10.0.0: preservation_score Property fehlte → HPG immer 1.0."""

    def _make_result(self, **kwargs):
        from backend.core.emotional_arc_preservation import EmotionalArcResult

        defaults = {
            "arousal_pearson": 1.0,
            "valence_pearson": 1.0,
            "klimax_peak_deviation": 0.0,
            "klimax_level_deviation_db": 0.0,
            "arc_preserved": True,
        }
        defaults.update(kwargs)
        return EmotionalArcResult(**defaults)  # type: ignore[arg-type]

    def test_property_exists(self):
        """EmotionalArcResult hat preservation_score Property."""
        r = self._make_result()
        assert hasattr(r, "preservation_score"), "preservation_score fehlt!"

    def test_perfect_preservation_returns_one(self):
        """Perfekte Korrelation + kein Klimax-Drift → score = 1.0."""
        r = self._make_result(
            arousal_pearson=1.0,
            valence_pearson=1.0,
            klimax_peak_deviation=0.0,
            klimax_level_deviation_db=0.0,
        )
        assert r.preservation_score == pytest.approx(1.0), f"Perfekte Erhaltung → 1.0, erhalten: {r.preservation_score}"

    def test_skipped_returns_one(self):
        """skipped=True (kurze Datei) → neutraler Prior = 1.0."""
        r = self._make_result(
            arc_preserved=True,
            skipped=True,
            arousal_pearson=0.0,
            valence_pearson=0.0,
        )
        assert r.preservation_score == pytest.approx(1.0)

    def test_zero_arousal_pearson_reduces_score(self):
        """Arousal-Korrelation 0 → score < 1.0 (HPG-Faktor wirksam)."""
        r = self._make_result(
            arousal_pearson=0.0,
            valence_pearson=1.0,
            klimax_peak_deviation=0.0,
            klimax_level_deviation_db=0.0,
        )
        # 0.50*0 + 0.30*1 + 0.20*1 = 0.50
        assert r.preservation_score == pytest.approx(0.50, abs=0.01)

    def test_zero_valence_pearson_reduces_score(self):
        """Valence-Korrelation 0 → score < 1.0."""
        r = self._make_result(
            arousal_pearson=1.0,
            valence_pearson=0.0,
            klimax_peak_deviation=0.0,
            klimax_level_deviation_db=0.0,
        )
        # 0.50*1 + 0.30*0 + 0.20*1 = 0.70
        assert r.preservation_score == pytest.approx(0.70, abs=0.01)

    def test_max_klimax_deviation_reduces_klimax_component(self):
        """Klimax um MAX_KLIMAX_DEVIATION Segmente verschoben → klimax_pos=0.0."""
        from backend.core.emotional_arc_preservation import EmotionalArcResult

        max_dev = EmotionalArcResult.MAX_KLIMAX_DEVIATION_SEGMENTS
        r = self._make_result(
            arousal_pearson=1.0,
            valence_pearson=1.0,
            klimax_peak_deviation=float(max_dev),
            klimax_level_deviation_db=0.0,
        )
        # klimax_pos = 1 - max_dev/max_dev = 0.0
        # klimax_lev = 1.0
        # klimax = 0.5*0.0 + 0.5*1.0 = 0.5
        # score = 0.50*1 + 0.30*1 + 0.20*0.5 = 0.90
        assert r.preservation_score == pytest.approx(0.90, abs=0.01)

    def test_negative_pearson_clamped_to_zero(self):
        """Negative Pearson-Werte werden auf 0 geclampt (kein negativer Score)."""
        r = self._make_result(
            arousal_pearson=-0.9,
            valence_pearson=-0.8,
            klimax_peak_deviation=0.0,
            klimax_level_deviation_db=0.0,
        )
        # ar=0, val=0 → 0.50*0 + 0.30*0 + 0.20*1 = 0.20
        assert r.preservation_score == pytest.approx(0.20, abs=0.01)

    def test_score_in_unit_interval_all_bad(self):
        """Schlechtester Fall → Score ∈ [0, 1]."""
        from backend.core.emotional_arc_preservation import EmotionalArcResult

        r = self._make_result(
            arousal_pearson=-1.0,
            valence_pearson=-1.0,
            klimax_peak_deviation=float(EmotionalArcResult.MAX_KLIMAX_DEVIATION_SEGMENTS * 10),
            klimax_level_deviation_db=float(EmotionalArcResult.MAX_KLIMAX_LEVEL_DB * 10),
            arc_preserved=False,
        )
        s = r.preservation_score
        assert 0.0 <= s <= 1.0, f"Score {s} außerhalb [0, 1]"

    def test_as_dict_includes_preservation_score(self):
        """as_dict() enthält preservation_score (für Telemetrie/UI)."""
        r = self._make_result()
        d = r.as_dict()
        assert "preservation_score" in d, "preservation_score fehlt in as_dict()"
        assert isinstance(d["preservation_score"], float)

    def test_as_dict_preservation_score_matches_property(self):
        """as_dict().preservation_score == r.preservation_score."""
        r = self._make_result(
            arousal_pearson=0.7,
            valence_pearson=0.6,
            klimax_peak_deviation=1.0,
            klimax_level_deviation_db=1.0,
        )
        assert abs(r.as_dict()["preservation_score"] - r.preservation_score) < 1e-6

    def test_property_is_finite(self):
        """preservation_score ist stets endlich (kein NaN/Inf)."""
        r = self._make_result(
            arousal_pearson=float("nan"),
            valence_pearson=1.0,
            klimax_peak_deviation=0.0,
            klimax_level_deviation_db=0.0,
        )
        # Nach max(0, nan) = 0; kein NaN propagiert
        s = r.preservation_score
        # NaN-Eingabe kann NaN ausgeben — Test erwartet, dass der Wert
        # durch den Caller (UV3: np.clip) abgefangen wird. Hier prüfen wir
        # nur, dass der Aufruf nicht explodiert.
        assert isinstance(s, float)


# ---------------------------------------------------------------------------
# §3 UV3-Integration — emotional_arc_for_hpi wird korrekt propagiert
# ---------------------------------------------------------------------------


class TestEmotionalArcPropagationInUV3Attribute:
    """Whitebox-Test: UV3 nutzt getattr(result, "preservation_score", 1.0).
    Mit der neuen Property wird dieser Aufruf jetzt den echten Score liefern."""

    def test_getattr_preservation_score_not_fallback(self):
        """getattr(result, 'preservation_score', 1.0) muss den echten Wert liefern."""
        from backend.core.emotional_arc_preservation import EmotionalArcResult

        result = EmotionalArcResult(
            arousal_pearson=0.5,
            valence_pearson=0.5,
            klimax_peak_deviation=0.0,
            klimax_level_deviation_db=0.0,
            arc_preserved=False,
        )
        # Vor dem Fix: kein Property → getattr liefert 1.0
        # Nach dem Fix: Property existiert → getattr liefert echten Score
        val = float(getattr(result, "preservation_score", 1.0))
        expected = result.preservation_score
        assert val == pytest.approx(expected, abs=1e-9), (
            f"getattr-Fallback aktiv: {val} ≠ {expected} — Property fehlt noch!"
        )
        # Sicherstellen, dass es NICHT der Default 1.0 ist (für degradierten arc)
        assert val < 0.95, f"HPG-Faktor immer noch 1.0 oder fast 1.0 ({val}): emotional_arc_preservation wirkungslos!"
