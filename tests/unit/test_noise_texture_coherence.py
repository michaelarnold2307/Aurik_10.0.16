from __future__ import annotations

import pytest

"""Unit-Tests für NoiseTextureCoherenceGuard (§4.7, v10.14)."""


import numpy as np

from backend.core.noise_texture_coherence import (
    NoiseTextureCoherenceGuard,
    NoiseTextureResult,
    compute_noise_texture_coherence,
    get_noise_texture_coherence_guard,
)


@pytest.mark.unit
class TestComputeNoiseTextureCoherence:
    """§4.7 — Kernfunktion: PSD-Korrelation zum Trägerprofil."""

    def test_white_noise_matches_cd_digital(self):
        """Weißes Rauschen passt zu cd_digital (flaches Profil)."""
        rng = np.random.default_rng(42)
        white = rng.normal(0, 0.01, 48000 * 2).astype(np.float32)
        result = compute_noise_texture_coherence(white, 48000, "cd_digital")
        assert isinstance(result, NoiseTextureResult)
        assert result.coherence >= 0.0  # Gültiger Bereich

    def test_pink_noise_matches_vinyl(self):
        """Rosa Rauschen hat abfallende Spectral Density → passt zu Vinyl."""
        rng = np.random.default_rng(123)
        # Approximiere rosa Rauschen via kumulative Summe von weißem Rauschen
        white = rng.normal(0, 0.01, 48000 * 2).astype(np.float64)
        pink_approx = np.cumsum(white)
        pink_approx = pink_approx / (np.max(np.abs(pink_approx)) + 1e-10) * 0.01
        result = compute_noise_texture_coherence(pink_approx, 48000, "vinyl")
        assert result.coherence >= 0.0  # Muss gültig sein
        assert result.coherence <= 1.0

    def test_short_signal_passes(self):
        """Zu kurzes Signal → coherence=1.0 (pass-through)."""
        short = np.zeros(512, dtype=np.float32)
        result = compute_noise_texture_coherence(short, 48000, "vinyl")
        assert result.coherence == 1.0
        assert result.is_compliant is True

    def test_stereo_input_handled(self):
        """Stereo-Input wird zu Mono gemixed."""
        rng = np.random.default_rng(77)
        stereo = rng.normal(0, 0.01, (48000 * 2, 2)).astype(np.float32)
        result = compute_noise_texture_coherence(stereo, 48000, "tape")
        assert 0.0 <= result.coherence <= 1.0

    def test_digital_materials_use_white_profile(self):
        """Alle digitalen Materialien nutzen weißes Profil."""
        rng = np.random.default_rng(55)
        noise = rng.normal(0, 0.01, 48000 * 2).astype(np.float32)
        for mat in ["dat", "minidisc", "mp3_low", "aac", "streaming"]:
            result = compute_noise_texture_coherence(noise, 48000, mat)
            assert 0.0 <= result.coherence <= 1.0

    def test_unknown_material_handled(self):
        """Unbekanntes Material → konservativer Fallback."""
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.01, 48000 * 2).astype(np.float32)
        result = compute_noise_texture_coherence(noise, 48000, "nonexistent_material")
        assert 0.0 <= result.coherence <= 1.0

    def test_result_fields(self):
        """Alle Felder korrekt befüllt."""
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, 48000 * 2).astype(np.float32)
        result = compute_noise_texture_coherence(noise, 48000, "shellac")
        assert result.material_type == "shellac"
        assert isinstance(result.reference_slope, float)
        assert isinstance(result.measured_slope, float)
        assert isinstance(result.is_compliant, bool)


class TestNoiseTextureCoherenceGuard:
    """§4.7 — Guard-Integration (per-phase + end-of-pipeline)."""

    def test_per_phase_high_coherence(self):
        """Hohe Kohärenz → wet_multiplier = 1.0."""
        guard = NoiseTextureCoherenceGuard()
        rng = np.random.default_rng(42)
        before = rng.normal(0, 0.1, 48000 * 2).astype(np.float32)
        # Leichte Veränderung → Residual ist ähnlich geformt
        after = before * 0.9
        coh, wet = guard.check_per_phase(before, after, 48000, "cd_digital")
        assert 0.0 <= coh <= 1.0
        assert wet <= 1.0

    def test_end_of_pipeline_returns_result(self):
        """End-of-Pipeline gibt NoiseTextureResult zurück."""
        guard = NoiseTextureCoherenceGuard()
        rng = np.random.default_rng(42)
        original = rng.normal(0, 0.1, 48000 * 2).astype(np.float32)
        restored = original * 0.8
        result = guard.check_end_of_pipeline(original, restored, 48000, "vinyl")
        assert isinstance(result, NoiseTextureResult)
        assert result.material_type == "vinyl"

    def test_end_of_pipeline_studio_mode_no_enforcement(self):
        """Studio 2026: Textur-Kohärenz nicht enforced."""
        guard = NoiseTextureCoherenceGuard()
        rng = np.random.default_rng(42)
        original = rng.normal(0, 0.1, 48000 * 2).astype(np.float32)
        restored = np.zeros_like(original)  # Extrem: alles entfernt
        result = guard.check_end_of_pipeline(original, restored, 48000, "vinyl", quality_mode="studio_2026")
        assert isinstance(result, NoiseTextureResult)


class TestSingleton:
    def test_get_noise_texture_coherence_guard_returns_same_instance(self):
        g1 = get_noise_texture_coherence_guard()
        g2 = get_noise_texture_coherence_guard()
        assert g1 is g2


class TestNoiseTextureWetReduction60To80Range:
    """§4.7 v10.14 — Per-Phase wet×0.85 im [0.60, 0.80)-Kohärenz-Bereich."""

    def _make_guard_with_patched_coherence(self, target_coherence: float, monkeypatch):
        """Erstellt Guard, dessen compute-Funktion eine bestimmte Kohärenz simuliert."""
        import backend.core.noise_texture_coherence as _ntc_mod

        def _fake_compute(residual, sr, material_type):
            return NoiseTextureResult(
                coherence=target_coherence,
                material_type=material_type,
                reference_slope=-3.0,
                measured_slope=-1.5,
                is_compliant=target_coherence >= 0.80,
            )

        monkeypatch.setattr(_ntc_mod, "compute_noise_texture_coherence", _fake_compute)
        return NoiseTextureCoherenceGuard()

    def test_coherence_070_gives_wet_085(self, monkeypatch):
        """Kohärenz 0.70 (in [0.60,0.80)) → wet_mult = 0.85 (nicht 1.0)."""
        guard = self._make_guard_with_patched_coherence(0.70, monkeypatch)
        rng = np.random.default_rng(10)
        before = rng.normal(0, 0.05, 48000).astype(np.float32)
        after = before * 0.85
        coh, wet = guard.check_per_phase(before, after, 48000, "vinyl")
        assert abs(coh - 0.70) < 1e-6, f"Erwartete Kohärenz 0.70, erhalten {coh}"
        assert abs(wet - 0.85) < 1e-6, (
            f"Kohärenz 0.70 muss wet_mult=0.85 liefern (§4.7 v10.14), erhalten wet_mult={wet}"
        )

    def test_coherence_065_gives_wet_085(self, monkeypatch):
        """Kohärenz 0.65 (in [0.60,0.80)) → wet_mult = 0.85."""
        guard = self._make_guard_with_patched_coherence(0.65, monkeypatch)
        rng = np.random.default_rng(11)
        before = rng.normal(0, 0.05, 48000).astype(np.float32)
        after = before * 0.85
        _, wet = guard.check_per_phase(before, after, 48000, "shellac")
        assert abs(wet - 0.85) < 1e-6, f"Erwartete wet 0.85, erhalten {wet}"

    def test_coherence_079_gives_wet_085(self, monkeypatch):
        """Kohärenz 0.79 (knapp unter 0.80) → wet_mult = 0.85."""
        guard = self._make_guard_with_patched_coherence(0.79, monkeypatch)
        rng = np.random.default_rng(12)
        before = rng.normal(0, 0.05, 48000).astype(np.float32)
        after = before * 0.90
        _, wet = guard.check_per_phase(before, after, 48000, "tape")
        assert abs(wet - 0.85) < 1e-6, f"Erwartete wet 0.85 bei coh=0.79, erhalten {wet}"

    def test_coherence_080_gives_wet_100(self, monkeypatch):
        """Kohärenz genau 0.80 → wet_mult = 1.0 (kein Eingriff)."""
        guard = self._make_guard_with_patched_coherence(0.80, monkeypatch)
        rng = np.random.default_rng(13)
        before = rng.normal(0, 0.05, 48000).astype(np.float32)
        after = before * 0.90
        _, wet = guard.check_per_phase(before, after, 48000, "vinyl")
        assert abs(wet - 1.0) < 1e-6, f"Erwartete wet 1.0 bei coh=0.80, erhalten {wet}"

    def test_coherence_below_060_gives_wet_070(self, monkeypatch):
        """Kohärenz < 0.60 → wet_mult = 0.70 (stärkere Dämpfung bleibt)."""
        guard = self._make_guard_with_patched_coherence(0.50, monkeypatch)
        rng = np.random.default_rng(14)
        before = rng.normal(0, 0.05, 48000).astype(np.float32)
        after = before * 0.85
        _, wet = guard.check_per_phase(before, after, 48000, "vinyl")
        assert abs(wet - 0.70) < 1e-6, f"Erwartete wet 0.70 bei coh=0.50, erhalten {wet}"


class TestNoiseTextureEndGateAdaptiveCompliance:
    """§4.7 v10.14.x — adaptive End-Gate-Schwelle (Dauer/Material/Mode)."""

    def _patch_compute_with_coherence(self, monkeypatch, coherence: float):
        import backend.core.noise_texture_coherence as _ntc_mod

        def _fake_compute(residual, sr, material_type):
            return NoiseTextureResult(
                coherence=coherence,
                material_type=material_type,
                reference_slope=-3.0,
                measured_slope=-1.5,
                is_compliant=coherence >= 0.80,
            )

        monkeypatch.setattr(_ntc_mod, "compute_noise_texture_coherence", _fake_compute)

    def test_short_analog_clip_relaxes_minimum(self, monkeypatch):
        """Kurzer Analog-Clip erhält robustere Schwelle statt starrem 0.80-Fail."""
        self._patch_compute_with_coherence(monkeypatch, coherence=0.39)
        guard = NoiseTextureCoherenceGuard()
        sr = 48000
        n = sr * 3
        original = np.zeros(n, dtype=np.float32)
        restored = np.zeros(n, dtype=np.float32)

        result = guard.check_end_of_pipeline(original, restored, sr, "vinyl", quality_mode="restoration")

        assert result.min_required_coherence < 0.80
        assert result.is_compliant is True

    def test_long_analog_clip_keeps_strict_minimum(self, monkeypatch):
        """Langer Analog-Clip bleibt bei strenger Schwelle nahe 0.80."""
        self._patch_compute_with_coherence(monkeypatch, coherence=0.39)
        guard = NoiseTextureCoherenceGuard()
        sr = 48000
        n = sr * 12
        original = np.zeros(n, dtype=np.float32)
        restored = np.zeros(n, dtype=np.float32)

        result = guard.check_end_of_pipeline(original, restored, sr, "vinyl", quality_mode="restoration")

        assert result.min_required_coherence >= 0.79
        assert result.is_compliant is False

    def test_studio_mode_disables_end_gate_enforcement(self, monkeypatch):
        """Studio 2026: End-Gate markiert Noise-Texture nie als hard fail."""
        self._patch_compute_with_coherence(monkeypatch, coherence=0.0)
        guard = NoiseTextureCoherenceGuard()
        sr = 48000
        n = sr * 10
        original = np.zeros(n, dtype=np.float32)
        restored = np.zeros(n, dtype=np.float32)

        result = guard.check_end_of_pipeline(original, restored, sr, "vinyl", quality_mode="studio_2026")

        assert result.min_required_coherence == 0.0
        assert result.is_compliant is True

    def test_digital_carrier_allows_zero_coherence_in_restoration(self, monkeypatch):
        """Digitale Carrier erzwingen keine analoge Noise-Texture-Kohärenz."""
        self._patch_compute_with_coherence(monkeypatch, coherence=0.0)
        guard = NoiseTextureCoherenceGuard()
        sr = 48000
        n = sr * 3
        original = np.zeros(n, dtype=np.float32)
        restored = np.zeros(n, dtype=np.float32)

        result = guard.check_end_of_pipeline(
            original,
            restored,
            sr,
            "mp3_low",
            quality_mode="restoration",
        )

        assert result.min_required_coherence == 0.0
        assert result.is_compliant is True
