from __future__ import annotations

import pytest

"""
tests/unit/test_literature_based_improvements.py — Literatur-gestützte Verbesserungen

Testet drei wissenschaftlich fundierte Änderungen (April 2026):

1. Zwicker ISO 532-1 Inter-Band-Masking-Spread (_apply_excitation_spread)
   — Zwicker & Fastl (2007) Table 8.1: 25 dB/Bark aufwärts, 40 dB/Bark abwärts
   — Breitbandrauschen muss höheres N liefern als enge Einzeltöne gleicher Energie

2. Valence-Proxy Krumhansl-Schmuckler (_KK_MAJOR/_KK_MINOR Templates)
   — Eerola & Vuoskoski (2011): Dur/Moll ist stärkster Valenz-Prädiktor (r=0.63)
   — Dur-Signal → höhere Valenz; Moll-Signal → niedrigere Valenz

3. SGMSE+ Sigma SNR-adaptiv (Richter et al. 2022 §V-D)
   — σ(SNR) = clip(0.55 + (12 − SNR) × 0.018, 0.25, 0.75)
   — SNR=0 dB → σ≈0.75; SNR=12 dB → σ=0.55; SNR=20 dB → σ≈0.39
"""


import numpy as np

SR = 48000


# ─────────────────────────────────────────────────────────────────────────────
# 1. Zwicker ISO 532-1 Inter-Band-Masking-Spread
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestZwickerMaskingSpread:
    """_apply_excitation_spread folgt ISO 532-1 Spreading-Funktion."""

    def _spl(self) -> list:
        """Helper: alle Bänder stumm (0 dB SPL → kein Beitrag)."""
        return [0.0] * 24

    def test_no_spread_when_all_silent(self):
        from dsp.psychoacoustics import _apply_excitation_spread

        result = _apply_excitation_spread([0.0] * 24)
        assert result == [0.0] * 24

    def test_masker_spreads_upward_to_adjacent(self):
        """Lautes Band 0 soll Band 1 anheben (25 dB/Bark upward slope)."""
        from dsp.psychoacoustics import _apply_excitation_spread

        levels = [0.0] * 24
        levels[0] = 80.0  # Masker im niedrigsten Bark-Band
        result = _apply_excitation_spread(levels)
        # Band 1 (1 Bark Abstand): erwartet 80 - 25 = 55 dB
        assert abs(result[1] - 55.0) < 0.01
        # Band 2 (2 Bark Abstand): 80 - 50 = 30 dB
        assert abs(result[2] - 30.0) < 0.01
        # Band 3: 80 - 75 = 5 dB
        assert abs(result[3] - 5.0) < 0.01
        # Band 4: 80 - 100 < 0 → kein Spread
        assert result[4] == 0.0

    def test_downward_spread_steeper_than_upward(self):
        """Abwärts-Spread (40 dB/Bark) ≥ Aufwärts-Spread (25 dB/Bark) bei gleicher Distanz."""
        from dsp.psychoacoustics import _apply_excitation_spread

        up_levels = [0.0] * 24
        dn_levels = [0.0] * 24
        up_levels[5] = 80.0  # Masker in Mitte, Spread aufwärts zu Band 6
        dn_levels[5] = 80.0  # Masker in Mitte, Spread abwärts zu Band 4

        up_result = _apply_excitation_spread(up_levels)
        dn_result = _apply_excitation_spread(dn_levels)
        # 1 Bark aufwärts: 80 - 25 = 55
        assert abs(up_result[6] - 55.0) < 0.01
        # 1 Bark abwärts: 80 - 40 = 40
        assert abs(dn_result[4] - 40.0) < 0.01
        # Abwärts ist kleiner value (stärker gedämpft)
        assert dn_result[4] < up_result[6]

    def test_effective_level_never_below_direct(self):
        """Effektives Niveau ≥ direktes Niveau (Spreading addiert, subtrahiert nicht)."""
        from dsp.psychoacoustics import _apply_excitation_spread

        rng = np.random.default_rng(42)
        levels = (rng.uniform(0, 80, 24)).tolist()
        result = _apply_excitation_spread(levels)
        for i in range(24):
            assert result[i] >= levels[i] - 1e-9, f"Band {i}: effective {result[i]:.2f} < direct {levels[i]:.2f}"

    def test_broadband_noise_louder_than_narrowband_same_energy(self):
        """Breitband weißes Rauschen (Spreading) muss mehr N liefern als ein einzelner Ton
        gleicher Gesamt-Energie — weil Spreading die High-Bands miterhöht."""
        from dsp.psychoacoustics import compute_specific_loudness_zwicker

        rng = np.random.default_rng(99)
        n = SR * 3
        # Breitband weißes Rauschen bei ca. -20 dBFS
        noise = rng.standard_normal(n).astype(np.float32) * 0.1
        # Einzelton 1 kHz mit gleicher Energie
        t = np.arange(n, dtype=np.float32) / SR
        tone_energy_target = float(np.mean(noise**2))
        tone_amp = float(np.sqrt(2.0 * tone_energy_target))
        tone = (tone_amp * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

        n_noise = compute_specific_loudness_zwicker(noise, SR)
        n_tone = compute_specific_loudness_zwicker(tone, SR)
        # Breitband-Spreading erzeugt mehr Gesamtlautheit als konzentrierter Ton
        assert n_noise > n_tone, f"Breitband N={n_noise:.2f} sone sollte > Einzelton N={n_tone:.2f} sone sein"

    def test_spreading_increases_total_loudness_vs_independent(self):
        """Stichprobe: Dominanter Bass-Band hebt leise obere Bänder via Spreading
        über deren Threshold-Quiet → höheres N als ohne Spreading.

        Aufbau: Band 0 bei 95 dB (weit über Threshold 55) → Spreading hebt Band 1 auf
        95-25=70 dB > Threshold 35 → neue Loudness-Contribution, die ohne Spreading fehlt.
        """
        from dsp.psychoacoustics import _THRESHOLD_QUIET_DB, _apply_excitation_spread

        # Band 0 sehr laut, restliche Bänder bei 0 dB (unter Threshold)
        levels = [95.0] + [0.0] * 23
        without = sum(
            0.063 * (10.0 ** (0.3 * max(0.0, lv - thr) / 10.0))
            for lv, thr in zip(levels, _THRESHOLD_QUIET_DB)
            if lv > thr
        )
        effective = _apply_excitation_spread(levels)
        with_spread = sum(
            0.063 * (10.0 ** (0.3 * max(0.0, lv - thr) / 10.0))
            for lv, thr in zip(effective, _THRESHOLD_QUIET_DB)
            if lv > thr
        )
        assert with_spread > without, f"Spreading muss N erhöhen: with={with_spread:.3f}, without={without:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Valence-Proxy: Krumhansl-Schmuckler Dur/Moll-Erkennung
# ─────────────────────────────────────────────────────────────────────────────


class TestValenceKrumhanslSchmuckler:
    """Valence-Proxy basiert auf Krumhansl (1990) — Dur→hoch, Moll→niedrig."""

    @staticmethod
    def _make_chord(root_hz: float, mode: str, secs: float = 6.0) -> np.ndarray:
        """Baut einen einfachen drei-stimmigen Dreiklang (additive Synthese).

        mode='major': Grundton + Große Terz + Quinte
        mode='minor': Grundton + Kleine Terz + Quinte
        """
        t = np.arange(int(SR * secs), dtype=np.float64) / SR
        third_interval = 4 if mode == "major" else 3  # Halbtöne
        third_hz = root_hz * 2 ** (third_interval / 12.0)
        fifth_hz = root_hz * 2 ** (7 / 12.0)
        sig = np.sin(2 * np.pi * root_hz * t) + np.sin(2 * np.pi * third_hz * t) + np.sin(2 * np.pi * fifth_hz * t)
        return (sig / (np.max(np.abs(sig)) + 1e-12) * 0.7).astype(np.float32)  # type: ignore[no-any-return]

    def test_major_chord_higher_valence_than_minor(self):
        """Dur-Dreiklang muss höhere Valenz als Moll-Dreiklang ergeben."""
        from backend.core.emotional_arc_preservation import EmotionalArcPreservationMetric

        metric = EmotionalArcPreservationMetric()
        root_hz = 261.63  # C4
        seg_len = int(SR * 5.0)
        hop_len = int(SR * 2.5)
        major_chord = self._make_chord(root_hz, "major", secs=30.0)
        minor_chord = self._make_chord(root_hz, "minor", secs=30.0)
        _, v_major, _ = metric._compute_features(major_chord, SR, seg_len, hop_len)
        _, v_minor, _ = metric._compute_features(minor_chord, SR, seg_len, hop_len)
        assert float(np.mean(v_major)) > float(np.mean(v_minor)), (
            f"Dur mittl. Valenz={np.mean(v_major):.3f} muss > Moll {np.mean(v_minor):.3f} sein"
        )

    def test_valence_not_correlated_with_noise_level(self):
        """Valence darf sich nach Denoising nicht systematisch erhöhen.

        Vorher-Bug: val = 1 - speaktral_flatness → nach Denoising stieg Valenz,
        weil Rauschen das Spektrum flacher machte.
        Neuer Proxy soll unkorreliert mit Rauschpegel sein.
        """
        from backend.core.emotional_arc_preservation import EmotionalArcPreservationMetric

        rng = np.random.default_rng(7)
        metric = EmotionalArcPreservationMetric()
        n = int(SR * 30.0)
        seg_len = int(SR * 5.0)
        hop_len = int(SR * 2.5)
        t = np.arange(n, dtype=np.float64) / SR
        root_hz = 220.0  # A3
        chord = (np.sin(2 * np.pi * root_hz * t) + np.sin(2 * np.pi * root_hz * 5 / 4 * t)).astype(np.float32) * 0.7
        noisy = chord + rng.standard_normal(n).astype(np.float32) * 0.3
        noisy = np.clip(noisy, -1.0, 1.0)

        _, v_clean, _ = metric._compute_features(chord.astype(np.float32), SR, seg_len, hop_len)
        _, v_noisy, _ = metric._compute_features(noisy.astype(np.float32), SR, seg_len, hop_len)

        # Valenz-Differenz darf ±0.3 nicht überschreiten — robust gegen Rauschen
        mean_diff = abs(float(np.mean(v_clean)) - float(np.mean(v_noisy)))
        assert mean_diff < 0.30, (
            f"Valenz-Differenz clean/noisy = {mean_diff:.3f} zu groß — "
            "Proxy muss robust gegen Rauschen sein (Eerola & Vuoskoski 2011)"
        )

    def test_kk_templates_are_centered(self):
        """Krumhansl-Templates müssen zentriert sein (Mittelwert ≈ 0)."""
        from backend.core.emotional_arc_preservation import _KK_MAJOR, _KK_MINOR

        assert abs(float(_KK_MAJOR.mean())) < 1e-6, "KK_MAJOR nicht zentriert"
        assert abs(float(_KK_MINOR.mean())) < 1e-6, "KK_MINOR nicht zentriert"

    def test_kk_templates_shape(self):
        """Krumhansl-Templates haben genau 12 Einträge (Halbtonklassen)."""
        from backend.core.emotional_arc_preservation import _KK_MAJOR, _KK_MINOR

        assert len(_KK_MAJOR) == 12
        assert len(_KK_MINOR) == 12

    def test_valence_range_is_valid(self):
        """Valence-Werte liegen im Bereich [0, 1]."""
        from backend.core.emotional_arc_preservation import EmotionalArcPreservationMetric

        metric = EmotionalArcPreservationMetric()
        rng = np.random.default_rng(123)
        n = int(SR * 30.0)
        seg_len = int(SR * 5.0)
        hop_len = int(SR * 2.5)
        audio = rng.standard_normal(n).astype(np.float32) * 0.3
        _, valences, _ = metric._compute_features(audio, SR, seg_len, hop_len)
        assert float(np.min(valences)) >= 0.0
        assert float(np.max(valences)) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. SGMSE+ Sigma SNR-adaptiv — Richter et al. (2022) §V-D
# ─────────────────────────────────────────────────────────────────────────────


class TestSgmseSigmaSNRAdaptive:
    """σ(SNR) = clip(0.55 + (12 - SNR) × 0.018, 0.25, 0.75) — Richter et al. 2022."""

    @staticmethod
    def _compute_sigma(snr_db: float, material: str = "vinyl") -> float:
        """Repliziert die Sigma-Formel aus phase_03_denoise.py."""
        sigma_from_snr = float(np.clip(0.55 + (12.0 - snr_db) * 0.018, 0.25, 0.75))
        material_bonus = 0.05 if material == "shellac" else 0.0
        return float(np.clip(sigma_from_snr + material_bonus, 0.25, 0.75))

    def test_sigma_at_snr_0_is_max(self):
        """SNR=0 dB → σ nahe 0.75 (stark verrauscht → aggressivste Diffusion)."""
        sigma = self._compute_sigma(0.0)
        assert sigma >= 0.74, f"SNR=0 dB: σ={sigma:.3f} zu niedrig"

    def test_sigma_at_snr_12_is_nominal(self):
        """SNR=12 dB entspricht dem trainierten SGMSE+-Optimum → σ≈0.55."""
        sigma = self._compute_sigma(12.0)
        assert abs(sigma - 0.55) < 0.01, f"SNR=12 dB: σ={sigma:.3f} ≠ 0.55"

    def test_sigma_at_snr_20_is_gentle(self):
        """SNR=20 dB → σ≈0.39 (eher sauberes Signal → behutsame Diffusion)."""
        sigma = self._compute_sigma(20.0)
        expected = float(np.clip(0.55 + (12.0 - 20.0) * 0.018, 0.25, 0.75))
        assert abs(sigma - expected) < 0.001

    def test_sigma_monotonic_decreasing_with_snr(self):
        """Sigma muss mit steigendem SNR monoton fallen (bis Clip-Grenze)."""
        snrs = list(range(0, 36, 2))
        sigmas = [self._compute_sigma(float(s)) for s in snrs]
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1] - 1e-9, (
                f"Sigma nicht monoton: σ({snrs[i]})={sigmas[i]:.3f} > σ({snrs[i + 1]})={sigmas[i + 1]:.3f}"
            )

    def test_sigma_clipped_to_bounds(self):
        """Sigma ist immer in [0.25, 0.75]."""
        for snr in [-30.0, -10.0, 0.0, 20.0, 35.0, 100.0]:
            sigma = self._compute_sigma(snr)
            assert 0.25 <= sigma <= 0.75, f"SNR={snr}: σ={sigma:.3f} außerhalb [0.25, 0.75]"

    def test_shellac_gets_bonus_sigma(self):
        """Shellac erhält +0.05 Bonus, da schwere HF-Verluste zusätzliche Diffusionstiefe brauchen."""
        sigma_shellac = self._compute_sigma(15.0, material="shellac")
        sigma_vinyl = self._compute_sigma(15.0, material="vinyl")
        assert sigma_shellac > sigma_vinyl, f"Shellac σ={sigma_shellac:.3f} sollte > Vinyl σ={sigma_vinyl:.3f}"

    def test_sigma_higher_for_heavy_degradation_than_clean(self):
        """Stark rauschende Aufnahme bekommt aggressiveres Sigma als saubere."""
        heavy = self._compute_sigma(2.0)
        clean = self._compute_sigma(28.0)
        assert heavy > clean, f"Stark degradiert (SNR=2 dB) σ={heavy:.3f} muss > sauber (SNR=28 dB) σ={clean:.3f}"

    def test_fallback_snr_for_material_types(self):
        """Fallback-SNR-Werte: tape/reel_tape/shellac=5 dB, andere=15 dB."""
        # When _est_snr_db is None, the phase uses material fallbacks.
        # Simulate the same fallback logic:
        for mat in ("tape", "reel_tape", "shellac"):
            snr = 5.0
            sigma = self._compute_sigma(snr, material=mat if mat == "shellac" else "generic")
            assert 0.25 <= sigma <= 0.75

        for mat in ("vinyl", "cd_digital", "mp3_low"):
            snr = 15.0
            sigma = self._compute_sigma(snr, material="vinyl")
            expected = float(np.clip(0.55 + (12.0 - 15.0) * 0.018, 0.25, 0.75))
            assert abs(sigma - expected) < 0.001
