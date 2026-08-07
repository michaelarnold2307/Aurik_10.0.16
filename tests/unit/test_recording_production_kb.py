from __future__ import annotations

import pytest

"""
Tests für RecordingProductionKB und Singer-School-Klassifikation (v10.0.0.x)

Abdeckung:
  - RecordingProductionKB: Profil-Lookup (spezifisch + Fallback + generisch)
  - detect_production_signature: Raumcharakter, Kompression, Mikrofon-Wärme
  - calibration_matrix: production_profile als 4. Bias-Schicht
  - VFAResult: singer_school + phoneme_protection_level Felder
  - VocalFocusAnalyzer._classify_singer_school: alle Schulen + Fallback
"""


import gc

import numpy as np

# ---------------------------------------------------------------------------
# RecordingProductionKB
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProductionProfileLookup:
    """Testet get_production_profile() — Lookup-Logik."""

    def setup_method(self) -> None:
        from backend.core.recording_production_kb import ProductionSignature, get_production_profile

        self.get_profile = get_production_profile
        self.Sig = ProductionSignature

    def test_exact_match_1950s_jazz_intimate_natural(self) -> None:
        sig = self.Sig(room_character="intimate_studio", compression_character="natural")
        profile = self.get_profile(1955, "jazz", "shellac", sig)
        assert profile.profile_name == "1950s_jazz_van_gelder"
        assert profile.preserve_room is True
        assert profile.goal_adjustments.get("raumtiefe", 0.0) > 0.0

    def test_exact_match_1960s_schlager_large_natural(self) -> None:
        sig = self.Sig(room_character="large_studio", compression_character="natural")
        profile = self.get_profile(1963, "schlager", "vinyl", sig)
        assert profile.profile_name == "1960s_schlager_polydor"
        assert profile.preserve_room is True

    def test_era_fallback_1940s_unknown_room(self) -> None:
        sig = self.Sig(room_character="unknown", compression_character="natural")
        profile = self.get_profile(1942, "unbekannt", "shellac", sig)
        # Sollte auf 1940s-Fallback fallen
        assert "1940" in profile.profile_name or profile.profile_name == "wartime_postwar_fallback"
        assert profile.goal_adjustments.get("waerme", 0.0) > 0.0

    def test_no_signature_uses_any_keys(self) -> None:
        profile = self.get_profile(1950, "jazz", "shellac", None)
        # Kein Exception, kein None
        assert profile is not None
        assert isinstance(profile.goal_adjustments, dict)

    def test_unknown_era_returns_generic(self) -> None:
        profile = self.get_profile(None, None, None, None)
        assert profile is not None  # Fallback, kein Crash

    def test_profile_name_is_string(self) -> None:
        sig = self.Sig(room_character="large_studio", compression_character="heavy")
        profile = self.get_profile(1970, "rock", "vinyl", sig)
        assert isinstance(profile.profile_name, str)
        assert len(profile.profile_name) > 0

    def test_goal_adjustments_are_floats(self) -> None:
        sig = self.Sig(room_character="intimate_studio", compression_character="dry")
        profile = self.get_profile(1960, "jazz", "vinyl", sig)
        for k, v in profile.goal_adjustments.items():
            assert isinstance(k, str), f"Key {k!r} ist kein str"
            assert isinstance(v, float), f"Value {v!r} für Key {k!r} ist kein float"

    def test_vocal_protection_level_valid(self) -> None:
        valid = {"strict", "standard", "relaxed"}
        for decade, genre in [(1950, "jazz"), (1960, "klassik"), (1980, "pop")]:
            profile = self.get_profile(decade, genre, "vinyl", None)
            assert profile.vocal_protection_level in valid, (
                f"Ungültiger vocal_protection_level: {profile.vocal_protection_level!r}"
            )

    def test_1950s_classical_large_studio_strict(self) -> None:
        sig = self.Sig(room_character="large_studio", compression_character="dry")
        profile = self.get_profile(1952, "klassik", "vinyl", sig)
        assert profile.vocal_protection_level == "strict"
        assert profile.preserve_room is True

    def test_genre_alias_soul_rnb_matches(self) -> None:
        sig = self.Sig(room_character="intimate_studio", compression_character="heavy")
        profile = self.get_profile(1958, "soul/r&b", "shellac", sig)
        # Sollte Chess-Records-Profil oder 1950s-Fallback treffen
        assert profile is not None
        assert profile.goal_adjustments.get("waerme", 0.0) > 0.0

    def test_ecm_jazz_1970s_preserve_room(self) -> None:
        sig = self.Sig(room_character="intimate_studio", compression_character="natural")
        profile = self.get_profile(1972, "jazz", "vinyl", sig)
        assert profile.preserve_room is True

    def teardown_method(self) -> None:
        gc.collect(0)


class TestDetectProductionSignature:
    """Testet detect_production_signature() DSP-Erkennung."""

    def _make_sine(self, freq: float = 440.0, duration: float = 5.0, sr: int = 48000) -> np.ndarray:
        t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)
        return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)  # type: ignore[no-any-return]

    def test_returns_production_signature(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        audio = self._make_sine()
        sig = detect_production_signature(audio, 48000)
        assert sig is not None

    def test_room_character_valid_values(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        valid = {"dry_studio", "intimate_studio", "large_studio", "echo_chamber", "live_venue", "unknown"}
        audio = self._make_sine()
        sig = detect_production_signature(audio, 48000)
        assert sig.room_character in valid

    def test_compression_character_valid_values(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        valid = {"dry", "natural", "heavy", "limited"}
        audio = self._make_sine()
        sig = detect_production_signature(audio, 48000)
        assert sig.compression_character in valid

    def test_mic_warmth_valid_values(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        valid = {"warm", "neutral", "presence"}
        audio = self._make_sine()
        sig = detect_production_signature(audio, 48000)
        assert sig.mic_warmth in valid

    def test_rt60_nonnegative(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        audio = self._make_sine()
        sig = detect_production_signature(audio, 48000)
        assert sig.rt60_s >= 0.0

    def test_too_short_audio_returns_defaults(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        audio = np.zeros(100, dtype=np.float32)  # < 2s
        sig = detect_production_signature(audio, 48000)
        assert sig.room_character == "unknown"

    def test_stereo_input_accepted(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        audio = np.random.randn(2, 48000 * 5).astype(np.float32) * 0.2
        sig = detect_production_signature(audio, 48000)
        assert sig is not None

    def test_nan_input_safe(self) -> None:
        from backend.core.recording_production_kb import detect_production_signature

        audio = np.full(48000 * 4, np.nan, dtype=np.float32)
        sig = detect_production_signature(audio, 48000)  # Kein Crash
        assert sig is not None

    def test_crest_factor_sine_valid_compression_class(self) -> None:
        """Sinus-Signal: detect_production_signature liefert gültigen compression_character."""
        from backend.core.recording_production_kb import detect_production_signature

        valid = {"dry", "natural", "heavy", "limited"}
        audio = self._make_sine(duration=10.0)
        sig = detect_production_signature(audio, 48000)
        # Sinus hat CF ≈ 1.41 (3 dB) — wird als 'limited' klassifiziert (sehr geringe Dynamik)
        assert sig.compression_character in valid

    def teardown_method(self) -> None:
        gc.collect(0)


# ---------------------------------------------------------------------------
# calibration_matrix: production_profile als 4. Bias-Schicht
# ---------------------------------------------------------------------------


class TestEstimateSongGoalTargetsWithProductionProfile:
    """Testet estimate_song_goal_targets() mit ProductionProfile-Parameter."""

    def setup_method(self) -> None:
        from backend.core.calibration_matrix import estimate_song_goal_targets
        from backend.core.recording_production_kb import ProductionProfile

        self.estimate = estimate_song_goal_targets
        self.Profile = ProductionProfile

    def test_no_profile_still_works(self) -> None:
        targets = self.estimate(era_decade=1970, genre_label="jazz", material_type="vinyl")
        assert isinstance(targets, dict)
        assert len(targets) > 0

    def test_profile_adjusts_raumtiefe_upward(self) -> None:
        profile = self.Profile(
            profile_name="test",
            goal_adjustments={"raumtiefe": +0.30},  # Großes +, damit messbar
        )
        targets_with = self.estimate(
            era_decade=1970, genre_label="jazz", material_type="vinyl", production_profile=profile
        )
        targets_without = self.estimate(era_decade=1970, genre_label="jazz", material_type="vinyl")
        # raumtiefe sollte mit Profil höher sein
        r_with = targets_with.get("raumtiefe", targets_with.get("spatial_depth", 0.0))
        r_without = targets_without.get("raumtiefe", targets_without.get("spatial_depth", 0.0))
        assert r_with > r_without, f"Raumtiefe mit Profil ({r_with:.3f}) sollte > ohne ({r_without:.3f}) sein"

    def test_profile_adjusts_natuerlichkeit_downward(self) -> None:
        profile = self.Profile(
            profile_name="test_down",
            goal_adjustments={"natuerlichkeit": -0.30},
        )
        targets_with = self.estimate(era_decade=1980, genre_label="pop", material_type="cd", production_profile=profile)
        targets_without = self.estimate(era_decade=1980, genre_label="pop", material_type="cd")
        n_with = targets_with.get("natuerlichkeit", 0.0)
        n_without = targets_without.get("natuerlichkeit", 0.0)
        assert n_with < n_without

    def test_targets_always_clipped_within_bounds(self) -> None:
        """Auch mit extremen Provenance-Adjustments: Targets ∈ [0.30, 0.99]."""
        profile = self.Profile(
            profile_name="extreme",
            goal_adjustments=dict.fromkeys(["natuerlichkeit", "raumtiefe", "waerme"], +99.0),
        )
        targets = self.estimate(
            era_decade=1950, genre_label="jazz", material_type="shellac", production_profile=profile
        )
        for goal, val in targets.items():
            assert 0.30 <= val <= 0.99, f"{goal}={val:.3f} außerhalb [0.30, 0.99]"

    def test_empty_goal_adjustments_no_change(self) -> None:
        profile = self.Profile(profile_name="empty", goal_adjustments={})
        targets_with = self.estimate(
            era_decade=1960, genre_label="schlager", material_type="vinyl", production_profile=profile
        )
        targets_without = self.estimate(era_decade=1960, genre_label="schlager", material_type="vinyl")
        for goal in targets_without:
            assert abs(targets_with.get(goal, 0.0) - targets_without[goal]) < 1e-6, (
                f"{goal}: Leeres Profil sollte keinen Unterschied machen"
            )

    def teardown_method(self) -> None:
        gc.collect(0)


# ---------------------------------------------------------------------------
# VFAResult: singer_school + phoneme_protection_level
# ---------------------------------------------------------------------------


class TestVFAResultSingerSchoolFields:
    """Testet die neuen singer_school-Felder in VFAResult."""

    def test_vfaresult_has_singer_school(self) -> None:
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult()
        assert hasattr(r, "singer_school")
        assert r.singer_school == "unknown"

    def test_vfaresult_has_phoneme_protection_level(self) -> None:
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult()
        assert hasattr(r, "phoneme_protection_level")
        assert r.phoneme_protection_level == "standard"

    def test_to_dict_contains_singer_school(self) -> None:
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult(singer_school="jazz", phoneme_protection_level="standard")
        d = r.to_dict()
        assert "singer_school" in d
        assert d["singer_school"] == "jazz"

    def test_to_dict_contains_phoneme_protection_level(self) -> None:
        from backend.core.vocal_focus_analyzer import VFAResult

        r = VFAResult(singer_school="classical", phoneme_protection_level="strict")
        d = r.to_dict()
        assert "phoneme_protection_level" in d
        assert d["phoneme_protection_level"] == "strict"

    def teardown_method(self) -> None:
        gc.collect(0)


class TestClassifySingerSchool:
    """Testet VocalFocusAnalyzer._classify_singer_school() — alle Schulen."""

    def _make_vfa(self, **kwargs) -> object:
        from backend.core.vocal_focus_analyzer import VFAResult

        defaults = {
            "vocal_present": True,
            "dominant_register": "chest",
            "formant_f1_mean": 500.0,
            "formant_f2_mean": 1500.0,
            "formant_stable": True,
            "style_confidence": 0.05,
            "vibrato_zones": [],
        }
        defaults.update(kwargs)
        return VFAResult(**defaults)  # type: ignore[arg-type]

    def test_classical_profile(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(
            formant_f1_mean=520.0,
            formant_f2_mean=1350.0,
            formant_stable=True,
            style_confidence=0.05,
            vibrato_zones=[(2.0, 5.0)],
        )
        school, prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "classical"
        assert prot == "strict"

    def test_soul_profile(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(
            dominant_register="chest",
            formant_f1_mean=720.0,
            style_confidence=0.30,
        )
        school, prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "soul_rnb"
        assert prot == "standard"

    def test_jazz_profile(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(
            formant_stable=False,
            style_confidence=0.20,
            vibrato_zones=[(1.0, 3.0), (5.0, 7.0)],
        )
        school, _prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "jazz"

    def test_folk_country_profile(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(
            formant_f1_mean=780.0,
            formant_stable=False,
            style_confidence=0.04,
            vibrato_zones=[],
        )
        school, _prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "folk_country"

    def test_schlager_profile(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(
            formant_f1_mean=560.0,
            formant_f2_mean=1400.0,
            formant_stable=True,
            style_confidence=0.06,
            vibrato_zones=[],
        )
        school, _prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "schlager"

    def test_pop_default_fallback(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(
            formant_f1_mean=630.0,
            formant_stable=False,
            style_confidence=0.09,
            vibrato_zones=[],
        )
        school, prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "pop"
        assert prot == "relaxed"

    def test_no_vocal_returns_unknown(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(vocal_present=False)
        school, prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "unknown"
        assert prot == "standard"

    def test_no_formants_returns_pop(self) -> None:
        from backend.core.vocal_focus_analyzer import VocalFocusAnalyzer

        vfa = self._make_vfa(formant_f1_mean=0.0)  # nicht messbar
        school, _prot = VocalFocusAnalyzer._classify_singer_school(vfa)  # type: ignore[arg-type]
        assert school == "pop"

    def test_all_schools_are_valid_strings(self) -> None:
        from backend.core.vocal_focus_analyzer import VFAResult, VocalFocusAnalyzer

        valid = {"classical", "jazz", "soul_rnb", "schlager", "folk_country", "pop", "unknown"}
        valid_prot = {"strict", "standard", "relaxed"}
        r = VFAResult(vocal_present=True, formant_f1_mean=500.0)
        school, prot = VocalFocusAnalyzer._classify_singer_school(r)
        assert school in valid
        assert prot in valid_prot

    def teardown_method(self) -> None:
        gc.collect(0)
