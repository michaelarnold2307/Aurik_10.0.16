from __future__ import annotations

"""Unit-Tests für §4.5b-Instrumental g_floor-Schutz in phase_03_denoise.

Prüft, dass rein instrumentales Material (PANNs Singing < 0.10) einen um +0.05
erhöhten OMLSA g_floor erhält — Oberton-Schutz für Streicher/Bläser/Piano.

Spec: §4.5b-Instrumental (Spec 04 §4.5b), copilot-instructions.md VERBOTEN-Tabelle
"""


import numpy as np
import pytest

SR = 48_000


def _make_sine(f0: float = 440.0, duration_s: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    """Einfaches Sinus-Signal (instrumentaler Oberton-Stellvertreter)."""
    n = int(SR * duration_s)
    t = np.arange(n, dtype=np.float64) / SR
    return (amplitude * np.sin(2 * np.pi * f0 * t)).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
class TestPhase03InstrumentalGFloorBoost:
    """§4.5b: g_floor wird bei panns_singing < 0.10 um +0.05 erhöht."""

    @pytest.fixture
    def phase(self):
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    def _captured_params(self, phase, audio, panns_singing, material="vinyl", genre="Klassik"):
        """Führt phase.process() aus und fängt den params-Dict ab,
        der an _denoise_mono_professional übergeben wird."""
        captured = {}

        _original_denoise = phase._denoise_mono_professional

        def _capture(audio_in, params, *args, **kwargs):
            captured["g_floor"] = params.get("g_floor")
            return _original_denoise(audio_in, params, *args, **kwargs)

        phase._denoise_mono_professional = _capture

        try:
            phase.process(
                audio,
                material_type=material,
                panns_singing=panns_singing,
                genre_label=genre,
                quality_mode="fast",  # "fast" bypasses ML-Hybrid → DSP-Pfad garantiert
                sample_rate=SR,
            )
        except Exception:
            # Falls andere Teile der Phase fehlschlagen, captured["g_floor"] ist noch valide
            pass

        phase._denoise_mono_professional = _original_denoise
        return captured.get("g_floor")

    def test_instrumental_material_raises_gfloor(self, phase):
        """panns_singing=0.05 (rein instrumental) → g_floor = material_default + 0.05."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        # Erwarteter Basis-g_floor aus MATERIAL_PARAMS["vinyl"] = 0.12
        base_g_floor = float(DenoisePhase.MATERIAL_PARAMS["vinyl"].get("g_floor", 0.10))  # type: ignore[arg-type]
        expected = float(np.clip(base_g_floor + 0.05, 0.10, 0.45))

        audio = _make_sine()
        g_floor_actual = self._captured_params(phase, audio, panns_singing=0.05, material="vinyl", genre="Klassik")

        assert g_floor_actual is not None, "g_floor wurde nicht an _denoise_mono_professional übergeben"
        assert abs(g_floor_actual - expected) < 1e-6, (
            f"Instrumental (panns=0.05): g_floor sollte {expected:.4f} sein "
            f"(base={base_g_floor:.4f} + 0.05), erhalten {g_floor_actual:.4f}"
        )

    def test_vocal_material_preserves_original_gfloor(self, phase):
        """panns_singing=0.30 (vokal) → g_floor unverändert (kein Instrumental-Boost)."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        base_g_floor = float(DenoisePhase.MATERIAL_PARAMS["vinyl"].get("g_floor", 0.10))  # type: ignore[arg-type]

        audio = _make_sine()
        g_floor_actual = self._captured_params(phase, audio, panns_singing=0.30, material="vinyl", genre="Pop")

        if g_floor_actual is not None:
            # Beim Vokal-Pfad wird ggf. DeepFilterNet verwendet und OMLSA gar nicht aufgerufen.
            # Falls OMLSA aufgerufen wird, darf g_floor nicht durch Instrumental-Boost erhöht sein.
            assert g_floor_actual <= base_g_floor + 1e-6, (
                f"Vokal (panns=0.30): g_floor sollte ≤ {base_g_floor:.4f} sein, "
                f"erhalten {g_floor_actual:.4f} (Instrumental-Boost fälschlicherweise aktiv)"
            )

    def test_panns_singing_exactly_010_no_boost(self, phase):
        """panns_singing=0.10 (Schwelle, nicht < 0.10) → kein Instrumental-Boost."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        base_g_floor = float(DenoisePhase.MATERIAL_PARAMS["vinyl"].get("g_floor", 0.10))  # type: ignore[arg-type]

        audio = _make_sine()
        g_floor_actual = self._captured_params(phase, audio, panns_singing=0.10, material="vinyl", genre="Klassik")

        if g_floor_actual is not None:
            assert g_floor_actual <= base_g_floor + 1e-6, (
                f"panns=0.10 (Schwelle): kein Boost erwartet (g_floor ≤ {base_g_floor:.4f}), "
                f"erhalten {g_floor_actual:.4f}"
            )

    def test_params_dict_is_shallow_copy(self, phase):
        """Instrumental-Boost DARF das Klassen-Level-Dict MATERIAL_PARAMS nicht mutieren."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        original_g_floor = DenoisePhase.MATERIAL_PARAMS["vinyl"].get("g_floor")
        audio = _make_sine()
        self._captured_params(phase, audio, panns_singing=0.05, material="vinyl", genre="Klassik")

        after_g_floor = DenoisePhase.MATERIAL_PARAMS["vinyl"].get("g_floor")
        assert original_g_floor == after_g_floor, (
            f"MATERIAL_PARAMS['vinyl']['g_floor'] wurde mutiert: vorher={original_g_floor}, nachher={after_g_floor}"
        )
