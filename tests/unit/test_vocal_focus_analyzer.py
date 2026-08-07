from __future__ import annotations

"""
Tests für VocalFocusAnalyzer (§0p — v10.0.0).
"""


import numpy as np
import pytest

SR = 48000


def _make_sine(freq_hz: float, duration_s: float = 3.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]


def _make_vocal_signal(duration_s: float = 3.0) -> np.ndarray:
    """Einfaches Vokal-ähnliches Summensignal (F0 + Harmonische)."""
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    sig = (
        0.5 * np.sin(2 * np.pi * 220 * t)  # F0
        + 0.3 * np.sin(2 * np.pi * 440 * t)  # H2
        + 0.15 * np.sin(2 * np.pi * 660 * t)  # H3
        + 0.08 * np.sin(2 * np.pi * 880 * t)  # H4
    ).astype(np.float32)
    return np.clip(sig, -1.0, 1.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Import / Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVocalFocusAnalyzerImport:
    def test_import_ok(self):
        from backend.core.vocal_focus_analyzer import get_vocal_focus_analyzer

        assert callable(get_vocal_focus_analyzer)

    def test_singleton(self):
        from backend.core.vocal_focus_analyzer import get_vocal_focus_analyzer

        a = get_vocal_focus_analyzer()
        b = get_vocal_focus_analyzer()
        assert a is b

    def test_vfa_result_import(self):
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult()
        assert r.panns_singing == 0.0
        assert r.vocal_present is False
        assert r.dominant_register == "chest"


# ---------------------------------------------------------------------------
# VFAResult dataclass
# ---------------------------------------------------------------------------


class TestVFAResult:
    def test_to_dict_keys(self):
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult(panns_singing=0.7, vocal_present=True, vqi_gate_active=True)
        d = r.to_dict()
        expected_keys = {
            "panns_singing",
            "vocal_present",
            "dominant_register",
            "energy_bias_db",
            "frisson_zones",
            "formant_f1_mean",
            "formant_f2_mean",
            "formant_stable",
            "passaggio_zones",
            "vibrato_zones",
            "tension_zones",
            "release_zones",
            "whisper_zones",
            "climax_type",
            "vqi_gate_active",
            "style_intent_zones",
            "style_confidence",
            "singer_school",
            "phoneme_protection_level",
            "analysis_duration_s",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_serializable(self):
        """Alle Werte müssen JSON-serialisierbar sein (dict/list/float/bool)."""
        import json

        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult(
            panns_singing=0.8,
            frisson_zones=[(1.0, 2.5), (5.0, 6.0)],
            passaggio_zones=[(3.0, 3.5)],
        )
        # Darf keine Exception werfen
        json.dumps(r.to_dict())


# ---------------------------------------------------------------------------
# analyze() — Gesang aktiv
# ---------------------------------------------------------------------------


class TestVFAAnalyzeVocal:
    @pytest.fixture()
    def vfa(self):
        from backend.core.vocal_focus_analyzer import get_vocal_focus_analyzer

        return get_vocal_focus_analyzer()

    def test_vocal_present_above_threshold(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.6)
        assert result.vocal_present is True

    def test_vqi_gate_active_above_035(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.4)
        assert result.vqi_gate_active is True

    def test_vqi_gate_inactive_below_035(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.30)
        assert result.vqi_gate_active is False

    def test_energy_bias_is_negative(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.7)
        assert result.energy_bias_db < 0.0

    def test_energy_bias_range(self, vfa):
        """energy_bias_db muss in [-9, -3] dB liegen."""
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.6)
        assert -9.0 <= result.energy_bias_db <= -3.0

    def test_dominant_register_valid(self, vfa):
        """Rückgabe muss ein valides Register sein."""
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.6)
        assert result.dominant_register in {"head", "chest", "fry_whisper", "unknown"}

    def test_analysis_duration_positive(self, vfa):
        audio = _make_vocal_signal(5.0)
        result = vfa.analyze(audio, SR, panns_singing=0.5)
        assert result.analysis_duration_s > 0.0
        # Darf nie länger sein als _ANALYSIS_MAX_S
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        assert result.analysis_duration_s <= VocalFocusAnalyzer._ANALYSIS_MAX_S + 0.1

    def test_frisson_zones_is_list(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.7)
        assert isinstance(result.frisson_zones, list)

    def test_passaggio_zones_is_list(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.6)
        assert isinstance(result.passaggio_zones, list)

    def test_stereo_input(self, vfa):
        """Stereo-Input muss akzeptiert werden."""
        mono = _make_vocal_signal(3.0)
        stereo = np.stack([mono, mono * 0.9], axis=0)  # (2, N)
        result = vfa.analyze(stereo, SR, panns_singing=0.5)
        assert result.vocal_present is True

    def test_formant_stable_bool(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(audio, SR, panns_singing=0.6)
        assert isinstance(result.formant_stable, bool)

    def test_low_raw_panns_is_raised_by_vocal_tags_and_schlager_genre(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(
            audio,
            SR,
            panns_singing=0.05,
            panns_tags={"Vocals": 0.12, "Music": 0.75},
            genre_label="Deutscher Schlager",
        )
        assert result.panns_singing >= 0.35
        assert result.vocal_present is True
        assert result.vqi_gate_active is True

    def test_strict_vocal_genre_activates_vqi_gate_without_panns(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(
            audio,
            SR,
            panns_singing=0.0,
            panns_tags={},
            genre_label="opera",
        )
        assert result.panns_singing >= 0.35
        assert result.vocal_present is True
        assert result.vqi_gate_active is True

    def test_vocal_material_prior_activates_vqi_gate_without_tag_support(self, vfa):
        audio = _make_vocal_signal(3.0)
        result = vfa.analyze(
            audio,
            SR,
            panns_singing=0.0,
            panns_tags={},
            vocal_material_prior=True,
        )
        assert result.panns_singing >= 0.35
        assert result.vocal_present is True
        assert result.vqi_gate_active is True


# ---------------------------------------------------------------------------
# analyze() — Kein Gesang (panns_singing < 0.25)
# ---------------------------------------------------------------------------


class TestVFAAnalyzeNoVocal:
    @pytest.fixture()
    def vfa(self):
        from backend.core.vocal_focus_analyzer import get_vocal_focus_analyzer

        return get_vocal_focus_analyzer()

    def test_vocal_absent_below_threshold(self, vfa):
        audio = _make_sine(440.0)  # Instrumentaler Ton
        result = vfa.analyze(audio, SR, panns_singing=0.10)
        assert result.vocal_present is False

    def test_vqi_gate_inactive_when_no_vocal(self, vfa):
        audio = _make_sine(440.0)
        result = vfa.analyze(audio, SR, panns_singing=0.10)
        assert result.vqi_gate_active is False

    def test_defaults_when_no_vocal(self, vfa):
        """Bei fehlendem Gesang: energy_bias=-6.0 (Chest-Default)."""
        audio = _make_sine(440.0)
        result = vfa.analyze(audio, SR, panns_singing=0.0)
        assert result.energy_bias_db == pytest.approx(-6.0)

    def test_frisson_zones_empty_when_no_vocal(self, vfa):
        """Keine Analyse bei panns_singing < 0.25."""
        audio = _make_sine(440.0)
        result = vfa.analyze(audio, SR, panns_singing=0.0)
        assert result.frisson_zones == []

    def test_music_and_speech_without_vocal_evidence_stays_inactive(self, vfa):
        audio = _make_sine(440.0)
        result = vfa.analyze(
            audio,
            SR,
            panns_singing=0.05,
            panns_tags={"Music": 0.75, "Speech": 0.45},
            genre_label="Klassik",
        )
        assert result.panns_singing < 0.25
        assert result.vocal_present is False
        assert result.vqi_gate_active is False


# ---------------------------------------------------------------------------
# Robustheit — Edge Cases
# ---------------------------------------------------------------------------


class TestVFARobustness:
    @pytest.fixture()
    def vfa(self):
        from backend.core.vocal_focus_analyzer import get_vocal_focus_analyzer

        return get_vocal_focus_analyzer()

    def test_silence_input(self, vfa):
        """Stilles Signal → kein Crash."""
        audio = np.zeros(SR * 3, dtype=np.float32)
        result = vfa.analyze(audio, SR, panns_singing=0.7)
        assert isinstance(result.dominant_register, str)

    def test_very_short_audio(self, vfa):
        """Sehr kurzes Signal (0.5 s) → kein Crash."""
        audio = _make_vocal_signal(0.5)
        result = vfa.analyze(audio, SR, panns_singing=0.5)
        assert result.vocal_present is True

    def test_very_long_audio(self, vfa):
        """Langes Signal (35 s > max 30 s) → truncated, kein Crash."""
        audio = _make_vocal_signal(35.0)
        result = vfa.analyze(audio, SR, panns_singing=0.6)
        # Analysiert nur MAX_ANALYSIS_S Sekunden
        assert result.analysis_duration_s <= 30.5

    def test_clipped_input(self, vfa):
        """Vollständig geclipptes Signal → kein Crash."""
        audio = np.ones(SR * 2, dtype=np.float32)
        result = vfa.analyze(audio, SR, panns_singing=0.5)
        assert result.vocal_present is True

    def test_panns_singing_exactly_025(self, vfa):
        """Grenzfall: panns_singing=0.25 → vocal_present=True."""
        audio = _make_vocal_signal(2.0)
        result = vfa.analyze(audio, SR, panns_singing=0.25)
        assert result.vocal_present is True

    def test_panns_singing_exactly_035(self, vfa):
        """Grenzfall: panns_singing=0.35 → vqi_gate_active=True."""
        audio = _make_vocal_signal(2.0)
        result = vfa.analyze(audio, SR, panns_singing=0.35)
        assert result.vqi_gate_active is True

    def test_panns_singing_below_025(self, vfa):
        """Grenzfall: panns_singing=0.24 → vocal_present=False."""
        audio = _make_vocal_signal(2.0)
        result = vfa.analyze(audio, SR, panns_singing=0.24)
        assert result.vocal_present is False
