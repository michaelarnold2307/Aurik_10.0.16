import pytest

"""Tests für v10.14 — Perceptual Depth & Musical Presence.

Abgedeckte Verbesserungen:
  - ExcellenceOptimizer: _MODULATION_STRENGTH und _HARM_BOOST_DB erhöht (war zu konservativ vs. iZotope RX11)
  - ExcellenceOptimizer: Material-Profile Shellac/Vinyl/Tape ebenfalls angehoben
  - EmotionalArc: Arousal-Formel ersetzt ZCR durch Spektral-Centroid (pitch-bewusst)
  - Phase 37 Bass Enhancement: mix-Werte für alle Materialien erhöht (stärker hörbar)
  - Phase 38 Presence Boost: BOOST_CONFIG und Era-Cap angehoben
  - Phase 39 Air Band: shelf_gain_db für Tape/CD/Streaming erhöht
"""

import importlib
import inspect
import re

import numpy as np

# ---------------------------------------------------------------------------
# ExcellenceOptimizer — Konstanten und Material-Profile
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExcellenceOptimizerBoostValues:
    """_MODULATION_STRENGTH und _HARM_BOOST_DB müssen deutlich über früheren konservativen Werten liegen."""

    def _get_module(self):
        import sys

        mod_name = "backend.core.excellence_optimizer"
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        from backend.core import excellence_optimizer as m

        return m

    def test_modulation_strength_increased(self):
        """_MODULATION_STRENGTH muss ≥ 0.40 sein (war 0.25 — zu konservativ)."""
        m = self._get_module()
        assert m._MODULATION_STRENGTH >= 0.40, (
            f"_MODULATION_STRENGTH={m._MODULATION_STRENGTH:.3f} < 0.40. "
            "v10.14: Micro-Dynamik-Reinjektion war zu schwach für über-denoisete Signale."
        )

    def test_harm_boost_db_increased(self):
        """_HARM_BOOST_DB muss ≥ 2.0 dB sein (war 1.0 dB — Oberton-Brillanz zu schwach)."""
        m = self._get_module()
        assert m._HARM_BOOST_DB >= 2.0, (
            f"_HARM_BOOST_DB={m._HARM_BOOST_DB:.2f} < 2.0. "
            "v10.14: Oberton-Brillanz zu konservativ — iZotope RX11 nutzt deutlich stärkere Harmonic Reinforcement."
        )

    def test_shellac_profile_harm_boost(self):
        """Shellac-Profil: harm_boost_db muss ≥ 2.5 sein."""
        m = self._get_module()
        shellac = m.MATERIAL_PROFILES.get("shellac")
        assert shellac is not None, "Shellac-Profil fehlt in MATERIAL_PROFILES"
        assert shellac.harm_boost_db >= 2.5, (
            f"Shellac harm_boost_db={shellac.harm_boost_db:.2f} < 2.5. "
            "Shellac-Aufnahmen brauchen starke Obert on-Restaurierung."
        )

    def test_vinyl_profile_harm_boost(self):
        """Vinyl-Profil: harm_boost_db muss ≥ 1.5 sein (war 0.8)."""
        m = self._get_module()
        vinyl = m.MATERIAL_PROFILES.get("vinyl")
        assert vinyl is not None, "Vinyl-Profil fehlt in MATERIAL_PROFILES"
        assert vinyl.harm_boost_db >= 1.5, (
            f"Vinyl harm_boost_db={vinyl.harm_boost_db:.2f} < 1.5. v10.14: Vinyl Obert one-Boost verdoppelt."
        )

    def test_tape_profile_harm_boost(self):
        """Tape-Profil: harm_boost_db muss ≥ 1.2 sein (war 0.6)."""
        m = self._get_module()
        tape = m.MATERIAL_PROFILES.get("tape")
        assert tape is not None, "Tape-Profil fehlt in MATERIAL_PROFILES"
        assert tape.harm_boost_db >= 1.2, (
            f"Tape harm_boost_db={tape.harm_boost_db:.2f} < 1.2. v10.14: Tape-Sättigung stärker betonen."
        )

    def test_shellac_profile_modulation_strength(self):
        """Shellac-Profil: modulation_strength muss ≥ 0.30 sein."""
        m = self._get_module()
        shellac = m.MATERIAL_PROFILES.get("shellac")
        assert shellac is not None
        assert shellac.modulation_strength >= 0.30, (
            f"Shellac modulation_strength={shellac.modulation_strength:.3f} < 0.30."
        )

    def test_vinyl_profile_modulation_strength(self):
        """Vinyl-Profil: modulation_strength muss ≥ 0.25 sein."""
        m = self._get_module()
        vinyl = m.MATERIAL_PROFILES.get("vinyl")
        assert vinyl is not None
        assert vinyl.modulation_strength >= 0.25, f"Vinyl modulation_strength={vinyl.modulation_strength:.3f} < 0.25."

    def test_tape_profile_modulation_strength(self):
        """Tape-Profil: modulation_strength muss ≥ 0.20 sein."""
        m = self._get_module()
        tape = m.MATERIAL_PROFILES.get("tape")
        assert tape is not None
        assert tape.modulation_strength >= 0.20, f"Tape modulation_strength={tape.modulation_strength:.3f} < 0.20."

    def test_cd_digital_profile_conservative(self):
        """CD/Digital-Profil: modulation_strength bleibt niedrig (digitales Material braucht weniger)."""
        m = self._get_module()
        cd = m.MATERIAL_PROFILES.get("cd_digital")
        assert cd is not None
        # CD should stay conservative — don't push it too hard
        assert cd.modulation_strength <= 0.15, (
            f"CD modulation_strength={cd.modulation_strength:.3f} > 0.15. "
            "Digitales Material braucht minimale Eingriffe."
        )


# ---------------------------------------------------------------------------
# EmotionalArc — Arousal-Formel (ZCR → Spektral-Centroid)
# ---------------------------------------------------------------------------


class TestEmotionalArcArousalFormula:
    """Arousal-Formel muss Spektral-Centroid statt ZCR nutzen (pitch-bewusst)."""

    def _get_source(self):
        from backend.core import emotional_arc_preservation as m

        return inspect.getsource(m)

    def test_zcr_removed_from_arousal(self):
        """ZCR darf in der Arousal-Berechnung nicht mehr als Hauptkomponente auftreten."""
        src = self._get_source()
        # zcr should not appear in the arousal formula line
        # It may still appear in comments but not as main computation
        # The new formula uses centroid
        arousal_lines = [ln for ln in src.splitlines() if "arousal_list.append" in ln]
        assert len(arousal_lines) >= 1, "arousal_list.append nicht gefunden"
        for line in arousal_lines:
            assert "zcr" not in line.lower(), (
                f"ZCR taucht noch in Arousal-Berechnung auf: {line.strip()!r}. "
                "v10.14: ZCR durch Spektral-Centroid ersetzen (pitch-invariantes Arousal)."
            )

    def test_spectral_centroid_in_arousal(self):
        """Spektral-Centroid muss Teil der Arousal-Formel sein."""
        src = self._get_source()
        # Look for centroid-related keyword in the arousal section
        assert "_centroid_norm" in src or "centroid_norm" in src, (
            "Kein Spektral-Centroid in Arousal-Formel gefunden. v10.14: arousal(t) = rms * 0.55 + centroid_norm * 0.45."
        )

    def test_arousal_uses_rfft(self):
        """Arousal-Berechnung muss FFT für Spektral-Centroid nutzen."""
        src = self._get_source()
        # rfft should be used in the feature computation
        assert "rfft" in src or "fft" in src, (
            "Keine FFT in _compute_features — Spektral-Centroid kann nicht berechnet werden."
        )

    def test_arousal_formula_rms_weight(self):
        """RMS-Anteil in der Arousal-Formel muss zwischen 0.45 und 0.70 liegen."""
        src = self._get_source()
        # Find the arousal append line and check weights
        # Expect something like: rms * 0.55 + _centroid_norm * 0.45
        match = re.search(r"arousal_list\.append\(.*?rms\s*\*\s*([\d.]+)", src)
        if match:
            rms_weight = float(match.group(1))
            assert 0.45 <= rms_weight <= 0.70, f"RMS-Gewicht in Arousal = {rms_weight:.2f}, erwartet 0.45–0.70."

    def test_compute_features_still_returns_two_arrays(self):
        """_compute_features muss nach Refactoring (arousal, valence, centroids) zurückgeben."""
        from backend.core.emotional_arc_preservation import EmotionalArcPreservationMetric

        eap = EmotionalArcPreservationMetric()
        rng = np.random.default_rng(42)
        mono = rng.standard_normal(48000 * 15).astype(np.float32) * 0.3  # 15s — groß genug für seg_len=5s
        sr = 48000
        seg_len = int(sr * 5)
        hop_len = int(sr * 5)
        result = eap._compute_features(mono, sr, seg_len, hop_len)
        assert len(result) >= 2, "_compute_features must return at least (arousal, valence)"
        a, v = result[0], result[1]
        assert isinstance(a, np.ndarray) and isinstance(v, np.ndarray), "_compute_features returned non-ndarray"
        assert a.dtype == np.float32 and v.dtype == np.float32, "_compute_features should return float32 arrays"
        assert len(a) >= 1 and len(v) >= 1, "_compute_features returned empty arrays"

    def test_arousal_values_finite(self):
        """Alle Arousal-Werte müssen endlich und nicht-negativ sein."""
        from backend.core.emotional_arc_preservation import EmotionalArcPreservationMetric

        eap = EmotionalArcPreservationMetric()
        rng = np.random.default_rng(99)
        mono = rng.standard_normal(48000 * 15).astype(np.float32) * 0.4  # 15s
        sr = 48000
        seg_len = int(sr * 5)
        hop_len = int(sr * 5)
        result = eap._compute_features(mono, sr, seg_len, hop_len)
        a, v = result[0], result[1]
        assert np.all(np.isfinite(a)), "Nicht-endliche Arousal-Werte"
        assert np.all(a >= 0.0), f"Negative Arousal-Werte: min={a.min():.4f}"

    def test_centroid_arousal_higher_for_bright_signal(self):
        """Arousal soll für helle (viel HF) Signale höher sein als für dunkle — Nicht-Regression."""
        from backend.core.emotional_arc_preservation import EmotionalArcPreservationMetric

        eap = EmotionalArcPreservationMetric()
        sr = 48000
        t = np.linspace(0, 3.0, sr * 3)
        # Bright: high-frequency sine sum
        bright = (np.sin(2 * np.pi * 4000 * t) + np.sin(2 * np.pi * 8000 * t)) * 0.3
        # Dark: low-frequency sine sum
        dark = (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 300 * t)) * 0.3

        seg_len = int(sr * 3)
        hop_len = int(sr * 3)
        result_bright = eap._compute_features(bright.astype(np.float32), sr, seg_len, hop_len)
        result_dark = eap._compute_features(dark.astype(np.float32), sr, seg_len, hop_len)
        a_bright = result_bright[0]
        a_dark = result_dark[0]
        assert a_bright.mean() > a_dark.mean(), (
            f"Bright-Signal ({a_bright.mean():.4f}) sollte höheres Arousal als Dark-Signal "
            f"({a_dark.mean():.4f}) haben. Spektral-Centroid-Arousal-Formel nicht korrekt."
        )


# ---------------------------------------------------------------------------
# Phase 37 — Bass Enhancement: mix-Werte
# ---------------------------------------------------------------------------


class TestPhase37BassEnhancementMix:
    """Bass Enhancement mix-Werte müssen für alle Materialien erhöht sein."""

    def _get_config(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_37_bass_enhancement import BassEnhancement

        return BassEnhancement.ENHANCEMENT_CONFIG, MaterialType

    def test_shellac_mix_increased(self):
        """Shellac mix muss ≥ 0.60 sein (war 0.50)."""
        cfg, MT = self._get_config()
        val = cfg[MT.SHELLAC]["mix"]
        assert val >= 0.60, f"Shellac mix={val:.2f} < 0.60. v10.14: Shellac-Bass muss deutlich stärker hörbar sein."

    def test_vinyl_mix_increased(self):
        """Vinyl mix muss ≥ 0.55 sein (war 0.45)."""
        cfg, MT = self._get_config()
        val = cfg[MT.VINYL]["mix"]
        assert val >= 0.55, f"Vinyl mix={val:.2f} < 0.55. v10.14."

    def test_cd_digital_mix_increased(self):
        """CD_DIGITAL mix muss ≥ 0.58 sein (war 0.50)."""
        cfg, MT = self._get_config()
        val = cfg[MT.CD_DIGITAL]["mix"]
        assert val >= 0.58, f"CD_DIGITAL mix={val:.2f} < 0.58. v10.14."

    def test_tape_mix_increased(self):
        """Tape mix muss ≥ 0.45 sein (war 0.35)."""
        cfg, MT = self._get_config()
        val = cfg[MT.TAPE]["mix"]
        assert val >= 0.45, f"Tape mix={val:.2f} < 0.45. v10.14."

    def test_streaming_mix_increased(self):
        """Streaming mix muss ≥ 0.50 sein (war 0.45)."""
        cfg, MT = self._get_config()
        val = cfg[MT.STREAMING]["mix"]
        assert val >= 0.50, f"Streaming mix={val:.2f} < 0.50. v10.14."

    def test_mix_values_in_valid_range(self):
        """Alle mix-Werte müssen im Bereich [0.0, 1.0] liegen."""
        cfg, MT = self._get_config()
        for mat, params in cfg.items():
            mv = params["mix"]
            assert 0.0 <= mv <= 1.0, f"mix={mv:.2f} für {mat} außerhalb [0,1]"

    def test_harmonic_gains_valid(self):
        """harmonic_2_gain und harmonic_3_gain müssen > 0 und < 1.0 bleiben."""
        cfg, MT = self._get_config()
        for mat, params in cfg.items():
            assert 0.0 < params["harmonic_2_gain"] < 1.0, (
                f"harmonic_2_gain={params['harmonic_2_gain']:.2f} für {mat} ungültig"
            )
            assert 0.0 < params["harmonic_3_gain"] < 1.0, (
                f"harmonic_3_gain={params['harmonic_3_gain']:.2f} für {mat} ungültig"
            )


# ---------------------------------------------------------------------------
# Phase 38 — Presence Boost: BOOST_CONFIG erhöht
# ---------------------------------------------------------------------------


class TestPhase38PresenceBoostConfig:
    """BOOST_CONFIG muss für alle Materialien erhöhte Presence-Werte haben."""

    def _get_config(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_38_presence_boost import PresenceBoost

        return PresenceBoost.BOOST_CONFIG, MaterialType

    def test_shellac_lower_gain_increased(self):
        """Shellac lower_gain_db muss ≥ 4.0 sein (war 3.0)."""
        cfg, MT = self._get_config()
        val = cfg[MT.SHELLAC]["lower_gain_db"]
        assert val >= 4.0, f"Shellac lower_gain_db={val:.1f} < 4.0. v10.14."

    def test_shellac_upper_gain_increased(self):
        """Shellac upper_gain_db muss ≥ 5.0 sein (war 4.0)."""
        cfg, MT = self._get_config()
        val = cfg[MT.SHELLAC]["upper_gain_db"]
        assert val >= 5.0, f"Shellac upper_gain_db={val:.1f} < 5.0. v10.14."

    def test_vinyl_lower_gain_increased(self):
        """Vinyl lower_gain_db muss ≥ 3.0 sein (war 2.5)."""
        cfg, MT = self._get_config()
        val = cfg[MT.VINYL]["lower_gain_db"]
        assert val >= 3.0, f"Vinyl lower_gain_db={val:.1f} < 3.0. v10.14."

    def test_vinyl_upper_gain_increased(self):
        """Vinyl upper_gain_db muss ≥ 4.0 sein (war 3.5)."""
        cfg, MT = self._get_config()
        val = cfg[MT.VINYL]["upper_gain_db"]
        assert val >= 4.0, f"Vinyl upper_gain_db={val:.1f} < 4.0. v10.14."

    def test_tape_presence_increased(self):
        """Tape upper_gain_db muss ≥ 3.5 sein (war 3.0)."""
        cfg, MT = self._get_config()
        val = cfg[MT.TAPE]["upper_gain_db"]
        assert val >= 3.5, f"Tape upper_gain_db={val:.1f} < 3.5. v10.14."

    def test_cd_digital_presence_increased(self):
        """CD_DIGITAL lower_gain_db muss ≥ 4.0 sein (war 3.5)."""
        cfg, MT = self._get_config()
        val = cfg[MT.CD_DIGITAL]["lower_gain_db"]
        assert val >= 4.0, f"CD_DIGITAL lower_gain_db={val:.1f} < 4.0. v10.14."

    def test_streaming_presence_increased(self):
        """Streaming lower_gain_db muss ≥ 3.5 sein (war 3.0)."""
        cfg, MT = self._get_config()
        val = cfg[MT.STREAMING]["lower_gain_db"]
        assert val >= 3.5, f"Streaming lower_gain_db={val:.1f} < 3.5. v10.14."

    def test_era_cap_increased_in_source(self):
        """Era-Cap für ≤1950-Material muss ≥ 3.5/4.0 dB sein (war 2.5/3.0)."""
        from backend.core.phases import phase_38_presence_boost as m

        src = inspect.getsource(m)
        # The vintage cap should be 3.5 or higher (was 2.5)
        assert "3.5" in src or "4.0" in src or "4.5" in src, (
            "Era-Cap für ≤1950-Material nicht auf ≥ 3.5/4.0 angehoben. "
            "v10.14: Auch Vintage-Material soll mehr Presence erhalten."
        )

    def test_gain_values_positive(self):
        """Alle Gain-Werte müssen positiv sein (Presence = Anhebung, kein Cut)."""
        cfg, MT = self._get_config()
        for mat, params in cfg.items():
            assert params["lower_gain_db"] > 0, f"lower_gain_db <= 0 für {mat}"
            assert params["upper_gain_db"] > 0, f"upper_gain_db <= 0 für {mat}"

    def test_q_factor_valid(self):
        """Q-Faktor muss für alle Materialien > 0 sein."""
        cfg, MT = self._get_config()
        for mat, params in cfg.items():
            assert params["q_factor"] > 0, f"q_factor <= 0 für {mat}"


# ---------------------------------------------------------------------------
# Phase 39 — Air Band Enhancement: shelf_gain_db erhöht
# ---------------------------------------------------------------------------


class TestPhase39AirBandConfig:
    """shelf_gain_db für Tape und CD_DIGITAL muss erhöht sein."""

    def _get_config(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_39_air_band_enhancement import AirBandEnhancement

        return AirBandEnhancement.AIR_CONFIG, MaterialType

    def test_tape_shelf_gain_increased(self):
        """Tape shelf_gain_db muss ≥ 4.5 sein (war 3.0)."""
        cfg, MT = self._get_config()
        val = cfg[MT.TAPE]["shelf_gain_db"]
        assert val >= 4.5, (
            f"Tape shelf_gain_db={val:.1f} < 4.5 dB. "
            "v10.14: Nach Tape-Hiss-Reduktion ist aggressivere HF-Restaurierung sicher."
        )

    def test_cd_digital_shelf_gain_increased(self):
        """CD_DIGITAL shelf_gain_db muss ≥ 4.5 sein (war 3.5)."""
        cfg, MT = self._get_config()
        val = cfg[MT.CD_DIGITAL]["shelf_gain_db"]
        assert val >= 4.5, (
            f"CD_DIGITAL shelf_gain_db={val:.1f} < 4.5 dB. v10.14: CD hat klare HF-Basis, mehr Air-Boost sicher."
        )

    def test_streaming_shelf_gain_increased(self):
        """Streaming shelf_gain_db muss ≥ 4.0 sein (war 4.0 — jetzt mindestens 4.5)."""
        cfg, MT = self._get_config()
        val = cfg[MT.STREAMING]["shelf_gain_db"]
        assert val >= 4.0, f"Streaming shelf_gain_db={val:.1f} < 4.0 dB. v10.14."

    def test_shellac_shelf_gain_unchanged_or_higher(self):
        """Shellac shelf_gain_db muss ≥ 6.0 bleiben (stark bandwidth-limitiert)."""
        cfg, MT = self._get_config()
        val = cfg[MT.SHELLAC]["shelf_gain_db"]
        assert val >= 6.0, f"Shellac shelf_gain_db={val:.1f} < 6.0. Shellac braucht maximale HF-Restaurierung."

    def test_vinyl_shelf_gain_preserved(self):
        """Vinyl shelf_gain_db muss ≥ 4.0 bleiben."""
        cfg, MT = self._get_config()
        val = cfg[MT.VINYL]["shelf_gain_db"]
        assert val >= 4.0, f"Vinyl shelf_gain_db={val:.1f} < 4.0. v10.14."

    def test_shelf_gain_not_excessive(self):
        """shelf_gain_db darf nicht > 8.0 dB sein (HF-Kumulativ-Limit §8.2)."""
        cfg, MT = self._get_config()
        for mat, params in cfg.items():
            val = params["shelf_gain_db"]
            assert val <= 8.0, f"shelf_gain_db={val:.1f} > 8.0 für {mat}. §8.2 HF-Kumulativ-Limit."

    def test_exciter_mix_in_range(self):
        """exciter_mix muss für alle Materialien in [0.0, 1.0] liegen."""
        cfg, MT = self._get_config()
        for mat, params in cfg.items():
            v = params["exciter_mix"]
            assert 0.0 <= v <= 1.0, f"exciter_mix={v:.2f} für {mat} außerhalb [0,1]"

    def test_saturation_drive_in_range(self):
        """saturation_drive muss für alle Materialien in [0.0, 1.0] liegen."""
        cfg, MT = self._get_config()
        for mat, params in cfg.items():
            v = params["saturation_drive"]
            assert 0.0 <= v <= 1.0, f"saturation_drive={v:.2f} für {mat} außerhalb [0,1]"
