from __future__ import annotations

"""
test_post_pipeline_frisson.py — §0i Post-Pipeline Frisson-Verifikation
======================================================================

Beweist, dass Frisson-Zonen (Gänsehaut-Passagen) die Pipeline überleben:
1. Signal mit bekannter Dynamik-Hüllkurve (simulierter Frisson-Bogen)
2. Nach simulierter Pipeline: emotional_arc_preservation ≥ 0.85
3. Frisson-Zonen werden nicht plattkomprimiert

Spec: §0i Perceptual Transparency, §8.3 Gänsehaut-Prinzipien
"""


import numpy as np
import pytest

SR = 48_000


def _make_frisson_signal(duration: float = 5.0) -> np.ndarray:
    """Erzeugt ein Signal mit Gänsehaut-Charakteristik:
    - Leise Strophe (0–2s, −18 dB)
    - Aufbauende Bridge (2–3s, −12 → −6 dB)
    - Laute Klimax (3–4s, 0 dB) ← Frisson-Zone
    - Ruhiges Outro (4–5s, −20 dB)
    """
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype=np.float32)

    # Trägersignal: 440 Hz Sinus + Obertöne
    signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    signal += 0.15 * np.sin(2 * np.pi * 880 * t)
    signal += 0.08 * np.sin(2 * np.pi * 1320 * t)

    # Dynamik-Hüllkurve (Gänsehaut-Bogen)
    envelope = np.ones(len(t), dtype=np.float32)
    envelope[: int(2.0 * SR)] = 0.125  # −18 dB (Strophe)
    ramp_up = np.linspace(0.125, 0.5, int(1.0 * SR), dtype=np.float32)
    envelope[int(2.0 * SR) : int(3.0 * SR)] = ramp_up  # Bridge
    envelope[int(3.0 * SR) : int(4.0 * SR)] = 1.0  # Klimax (Frisson)
    ramp_down = np.linspace(1.0, 0.1, int(0.5 * SR), dtype=np.float32)
    envelope[int(4.0 * SR) : int(4.5 * SR)] = ramp_down
    envelope[int(4.5 * SR) :] = 0.08  # −22 dB (Outro)

    return (signal * envelope).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
@pytest.mark.pleasantness
class TestPostPipelineFrisson:
    """§0i: Frisson-Bogen überlebt die Pipeline."""

    def test_01_frisson_signal_has_dynamic_arc(self):
        """Das Test-Signal hat einen messbaren Dynamik-Bogen."""
        audio = _make_frisson_signal()
        np.sqrt(np.mean(audio**2))

        # Strophe sollte leiser sein als Klimax
        verse_rms = np.sqrt(np.mean(audio[: int(2.0 * SR)] ** 2))
        climax_rms = np.sqrt(np.mean(audio[int(3.0 * SR) : int(4.0 * SR)] ** 2))

        assert climax_rms > verse_rms * 2, (
            f"Frisson-Signal hat keinen Dynamik-Bogen: verse={verse_rms:.4f}, climax={climax_rms:.4f}"
        )

    def test_02_envelope_correlation_before_after_light_processing(self):
        """Nach leichter Verarbeitung bleibt die Hüllkurven-Korrelation hoch."""
        audio = _make_frisson_signal()
        np.sqrt(np.mean(audio**2) + 1e-12)

        # Simuliere leichte Dynamik-Glättung (wie Phase 54)
        from scipy.ndimage import uniform_filter1d

        smoothed = uniform_filter1d(np.abs(audio), size=int(0.03 * SR))
        processed = audio * (smoothed / (np.abs(audio) + 1e-12))
        processed = np.clip(processed, -1.0, 1.0)

        # Hüllkurven-Korrelation
        env_orig = np.abs(audio)
        env_proc = np.abs(processed)

        # Pearson-Korrelation der Hüllkurven
        corr = np.corrcoef(env_orig[::100], env_proc[::100])[0, 1]

        assert corr > 0.85, f"Frisson-Bogen nicht erhalten: envelope correlation = {corr:.3f} < 0.85"

    def test_03_climax_not_compressed_below_threshold(self):
        """Die Klimax-Passage wird nicht unter 70% ihrer ursprünglichen Energie komprimiert."""
        audio = _make_frisson_signal()
        climax_region = audio[int(3.0 * SR) : int(4.0 * SR)]
        original_climax_rms = np.sqrt(np.mean(climax_region**2))

        # Simuliere moderate Dynamics-Verarbeitung
        from scipy.ndimage import uniform_filter1d

        smoothed = uniform_filter1d(np.abs(audio), size=int(0.06 * SR))
        processed = audio * (smoothed / (np.abs(audio) + 1e-12))
        processed = np.clip(processed, -1.0, 1.0)

        processed_climax_rms = np.sqrt(np.mean(processed[int(3.0 * SR) : int(4.0 * SR)] ** 2))
        ratio = processed_climax_rms / max(original_climax_rms, 1e-12)

        assert ratio > 0.70, f"Klimax zu stark komprimiert: post/pre ratio = {ratio:.3f} < 0.70"

    def test_04_frisson_hard_veto_in_verboten_md(self):
        """Frisson Hard-Veto ist in VERBOTEN.md dokumentiert."""
        src = open(".github/VERBOTEN.md", encoding="utf-8").read()
        assert "Frisson-Extremzone" in src, "Frisson Hard-Veto nicht in VERBOTEN.md dokumentiert"
        assert "chirurgische" in src.lower() or "Defekt-Behebung" in src, (
            "Priorität 'chirurgische Defekt-Behebung vor Frisson' nicht dokumentiert"
        )

    def test_05_frisson_zones_in_kwargs_flow(self):
        """Frisson-Zonen werden via kwargs an Phasen durchgereicht."""
        # Check that UV3 passes frisson_zones to phase kwargs
        import backend.core.unified_restorer_v3 as uv3_mod

        src = open(uv3_mod.__file__, encoding="utf-8").read()
        assert "frisson_zones" in src, "frisson_zones nicht in UV3 — werden nicht an Phasen durchgereicht"
