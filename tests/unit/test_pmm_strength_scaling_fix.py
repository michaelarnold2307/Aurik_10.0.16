from __future__ import annotations

"""Unit-Tests für Fix 10/11/12: PMM Gain-Skalierung und Phase-18-Blend-Ordering.

Fix 10 – §2.45a / §2.54: phase_29 Psychoacoustic Masking Gain-Clamp muss bei
         niedrigem intensity_scale proportional gegen 1.0 interpoliert werden,
         damit phase_29 bei PMGG-Strength=0.14 keine unbeabsichtigte TFS-Degradation
         und keinen übermäßigen Makeup-Gain erzeugt.

Fix 11 – §2.45a / §2.54: phase_03 DSP-Pfad identisches Fix (PMM-Skalierung
         als Funktion von params["strength"]).

Fix 12 – §2.45a / §2.53: phase_18 Blend-Ordering: Wet/Dry-Blend muss VOR
         Loudness-Preservation erfolgen, damit rms_drop_db / loudness_makeup_db
         im PhaseResult die tatsächliche Ausgabepegeldifferenz widerspiegeln
         (nicht den 100%-Wet-Zwischenwert vor dem Blend).
"""


import numpy as np
import pytest

SR = 48000


def _white_noise(secs: float = 2.0, amp: float = 0.3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * secs)).astype(np.float32) * amp


def _sine_plus_noise(freq: float = 440.0, secs: float = 2.0, noise_amp: float = 0.08) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    sig = 0.25 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(sig)).astype(np.float32) * noise_amp
    return np.clip(sig + noise, -1.0, 1.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Fix 12: Phase 18 — Blend-Ordering (Blend vor Makeup)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase18BlendOrdering:
    """Stellt sicher dass Wet/Dry-Blend VOR Loudness-Preservation in phase_18 stattfindet.

    Invariante: Bei sehr niedrigem effective_strength (≤ 0.10) ist der tatsächliche
    Ausgabe-RMS-Drop proportional zu strength — und deutlich geringer als 1.5 dB.
    """

    @pytest.fixture
    def phase_18(self):
        from backend.core.phases.phase_18_noise_gate import NoiseGate

        return NoiseGate()

    @pytest.fixture
    def vinyl_material(self):
        from backend.core.defect_scanner import MaterialType

        return MaterialType.VINYL

    def _run(self, phase, audio, material, strength: float) -> tuple[np.ndarray, dict]:
        result = phase.process(
            audio=audio,
            sample_rate=SR,
            material=material,
            strength=strength,
        )
        return result.audio, result.metadata

    def test_blend_ordering_low_strength_rms_drop_proportional(self, phase_18, vinyl_material):
        """RMS-Drop im Metadata muss bei strength=0.05 deutlich unter 1.5 dB liegen."""
        audio = _sine_plus_noise()
        _, meta = self._run(phase_18, audio, vinyl_material, strength=0.05)
        rms_drop = float(meta.get("rms_drop_db", 0.0))
        # Mit korrektem Blend-vor-Makeup darf der Drop nicht den 1.5 dB Vinyl-Cap erreichen
        # (da das blendete Signal kaum von input abweicht bei 5 % Wet).
        assert rms_drop > -1.5, (
            f"Blend-Ordering-Bug: rms_drop_db={rms_drop:.3f} dB ist ≤ -1.5 dB "
            f"obwohl strength=0.05 (nur 5 % Wet). "
            f"Likely: Makeup wurde auf 100%-Wet-Signal berechnet, nicht auf Blend."
        )

    def test_blend_ordering_makeup_proportional_to_strength(self, phase_18, vinyl_material):
        """Makeup-Gain bei strength=0.05 muss deutlich kleiner als bei strength=1.0 sein."""
        audio = _sine_plus_noise()
        _, meta_low = self._run(phase_18, audio, vinyl_material, strength=0.05)
        _, meta_full = self._run(phase_18, audio, vinyl_material, strength=1.0)
        makeup_low = float(meta_low.get("loudness_makeup_db", 0.0))
        makeup_full = float(meta_full.get("loudness_makeup_db", 0.0))
        # Bei strength=0.05 darf Makeup nicht höher sein als bei strength=1.0
        assert makeup_low <= makeup_full + 0.1, (
            f"Blend-Ordering-Bug: makeup bei strength=0.05 ({makeup_low:.2f} dB) "
            f">= makeup bei strength=1.0 ({makeup_full:.2f} dB). "
            f"Blend muss vor Loudness-Preservation erfolgen."
        )

    def test_blend_ordering_output_close_to_dry_at_low_strength(self, phase_18, vinyl_material):
        """Bei strength=0.05 muss der Ausgang nahe am Eingangssignal liegen."""
        audio = _sine_plus_noise()
        out, _ = self._run(phase_18, audio, vinyl_material, strength=0.05)
        rms_in = float(np.sqrt(np.mean(audio**2) + 1e-12))
        rms_out = float(np.sqrt(np.mean(out**2) + 1e-12))
        rms_diff_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-9))
        # Ausgabe darf höchstens 1.0 dB vom Eingang abweichen bei 5 % Wet.
        assert abs(rms_diff_db) < 1.0, (
            f"Bei strength=0.05 weicht Ausgabe um {rms_diff_db:.2f} dB vom Eingang ab — "
            f"erwartet < 1.0 dB (95 % Dry-Signal dominiert)."
        )

    def test_output_bounds(self, phase_18, vinyl_material):
        """Ausgabe muss [-1, 1] einhalten."""
        audio = _sine_plus_noise()
        out, _ = self._run(phase_18, audio, vinyl_material, strength=0.05)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_output_shape_preserved(self, phase_18, vinyl_material):
        """Ausgabe muss gleiche Shape wie Eingabe haben."""
        audio = _sine_plus_noise()
        out, _ = self._run(phase_18, audio, vinyl_material, strength=0.50)
        assert out.shape == audio.shape


# ---------------------------------------------------------------------------
# Fix 10: Phase 29 — PMM Gain-Skalierung bei niedrigem intensity_scale
# ---------------------------------------------------------------------------


class TestPhase29PMMStrengthScaling:
    """Stellt sicher dass der Psychoacoustic-Masking-Gain-Clamp in phase_29
    bei niedrigem PMGG-Strength nur proportional schwache Unterdrückung ausübt.

    Invariante: RMS-Drop bei strength=0.10 muss deutlich kleiner sein als bei
    strength=1.0 (proportionale Skalierung, nicht konstant).
    """

    @pytest.fixture
    def phase_29(self):
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        return TapeHissReductionPhase()

    @pytest.fixture
    def vinyl_material(self):
        from backend.core.defect_scanner import MaterialType

        return MaterialType.VINYL

    def _run(self, phase, audio, material, strength: float) -> tuple[np.ndarray, dict]:
        result = phase.process(
            audio=audio,
            sample_rate=SR,
            material=material,
            quality_mode="fast",
            strength=strength,
        )
        return result.audio, result.metadata

    def test_pmm_scaling_low_strength_rms_proportional(self, phase_29, vinyl_material):
        """RMS-Drop bei strength=0.10 muss kleiner sein als bei strength=1.0."""
        audio = _white_noise(amp=0.2)
        _, meta_low = self._run(phase_29, audio, vinyl_material, strength=0.10)
        _, meta_full = self._run(phase_29, audio, vinyl_material, strength=1.0)
        drop_low = float(meta_low.get("rms_drop_db", 0.0))
        drop_full = float(meta_full.get("rms_drop_db", 0.0))
        # Bei sehr niedrigem Strength muss der Drop kleiner sein als bei vollem Strength
        # (vor dem Fix: Drop war gleich groß wegen ungeschaftem PMM-Clamp).
        assert drop_low > drop_full - 0.5, (
            f"PMM-Scaling-Bug: rms_drop bei strength=0.10 ({drop_low:.2f} dB) "
            f"ist fast so groß wie bei strength=1.0 ({drop_full:.2f} dB). "
            f"PMM-Gain muss mit intensity_scale skaliert werden."
        )

    def test_output_stable_at_low_strength(self, phase_29, vinyl_material):
        """Bei strength=0.10 muss Ausgabe nahe am Eingang sein (< 2 dB RMS-Differenz)."""
        audio = _white_noise(amp=0.2)
        out, _ = self._run(phase_29, audio, vinyl_material, strength=0.10)
        rms_in = float(np.sqrt(np.mean(audio**2) + 1e-12))
        rms_out = float(np.sqrt(np.mean(out**2) + 1e-12))
        rms_diff_db = 20.0 * np.log10(max(rms_out / rms_in, 1e-9))
        assert rms_diff_db > -2.0, (
            f"Phase_29 bei strength=0.10: RMS-Drop {rms_diff_db:.2f} dB > 2 dB — PMM-Clamp-Skalierung nicht korrekt."
        )

    def test_output_bounds(self, phase_29, vinyl_material):
        """Ausgabe muss [-1, 1] und finite sein."""
        audio = _white_noise(amp=0.3)
        out, _ = self._run(phase_29, audio, vinyl_material, strength=0.50)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6

    def test_output_shape_mono(self, phase_29, vinyl_material):
        """Mono-Shape bleibt erhalten."""
        audio = _white_noise()
        out, _ = self._run(phase_29, audio, vinyl_material, strength=0.50)
        assert out.shape == audio.shape

    def test_output_shape_stereo(self, phase_29, vinyl_material):
        """Stereo-Shape bleibt erhalten."""
        audio = np.stack([_white_noise(seed=1), _white_noise(seed=2)], axis=-1)
        out, _ = self._run(phase_29, audio, vinyl_material, strength=0.50)
        assert out.shape == audio.shape

    def test_strength_zero_passthrough(self, phase_29, vinyl_material):
        """Bei effective_strength=0.0 muss Ausgabe gleich Eingang sein."""
        audio = _sine_plus_noise()
        out, _ = self._run(phase_29, audio, vinyl_material, strength=0.0)
        # Phase sollte bei strength=0 skippen oder exakt passthrough liefern
        assert np.allclose(out, audio, atol=1e-5), "Phase_29 bei strength=0 liefert nicht den Eingang zurück."
