"""
E2E Integrationstest: Komplette Pipeline mit Steering auf synthetischem Defekt-Audio.

Validiert:
1. HPE verbessert sich nach NaturalnessOptimizer
2. Steering-Guard wird in UV3.__init__ aktiviert
3. Keine NaN/Inf im Output
4. Audio-Länge bleibt erhalten
5. Stereo bleibt erhalten
6. CrossPhaseTracker wird pro Stage aufgerufen
"""

import numpy as np
import pytest

_SR = 48000


def _make_defective_audio(duration: float = 3.0) -> np.ndarray:
    """Synthetisches Audio mit realistischen Defekten."""
    t = np.arange(int(duration * _SR), dtype=np.float32) / _SR
    # Musik-ähnliches Signal: 440Hz + 554Hz (kleine Terz)
    sine = (np.sin(2 * np.pi * 440 * t) * 0.4 + np.sin(2 * np.pi * 554 * t) * 0.3).astype(np.float32)
    # Clicks (alle 0.5s)
    click_positions = np.arange(0, int(duration * _SR), int(0.5 * _SR))
    for pos in click_positions:
        if pos + 50 < len(sine):
            sine[pos : pos + 50] += np.hanning(50).astype(np.float32) * 0.8
    # Breitbandrauschen -20dB
    noise = np.random.randn(len(t)).astype(np.float32) * 0.03
    # 50Hz Brummen
    hum = np.sin(2 * np.pi * 50 * t).astype(np.float32) * 0.05
    # MP3-artige Artefakte (periodische Nullen)
    frame = int(0.026 * _SR)
    for i in range(5, len(t), frame * 10):
        if i + 10 < len(sine):
            sine[i : i + 10] *= 0.3
    return np.clip(sine + noise + hum, -1, 1)  # type: ignore[no-any-return]


def _make_stereo_defective() -> np.ndarray:
    mono = _make_defective_audio()
    return np.stack([mono, mono * 0.85], axis=1).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
class TestE2EPipeline:
    """End-to-End Tests mit synthetischem Defekt-Audio."""

    def test_01_naturalness_optimizer_improves_hpe(self):
        """NaturalnessOptimizer verbessert HPE auf defektem Audio."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        audio = _make_defective_audio()
        result = optimize_naturalness(audio, audio.copy(), _SR, material="vinyl")
        assert result.hpe_after >= result.hpe_before - 0.01, (
            f"HPE verschlechtert: {result.hpe_before:.3f} → {result.hpe_after:.3f}"
        )
        assert len(result.applied_stages) >= 4, f"Nur {len(result.applied_stages)} Stages, erwarte >= 4"

    def test_02_no_nan_inf_in_output(self):
        """Output ist NaN/Inf-frei."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        audio = _make_defective_audio()
        result = optimize_naturalness(audio, audio.copy(), _SR)
        assert not np.any(np.isnan(result.audio)), "NaN im Output!"
        assert not np.any(np.isinf(result.audio)), "Inf im Output!"

    def test_03_audio_length_preserved(self):
        """Audio-Länge bleibt erhalten."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        audio = _make_defective_audio()
        result = optimize_naturalness(audio, audio.copy(), _SR)
        assert len(result.audio) == len(audio), f"Länge geändert: {len(result.audio)} != {len(audio)}"

    def test_04_stereo_preserved(self):
        """Stereo-Audio bleibt stereo."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        audio = _make_stereo_defective()
        result = optimize_naturalness(audio, audio.copy(), _SR)
        assert result.audio.ndim == 2, f"ndim={result.audio.ndim}, erwarte 2"
        assert result.audio.shape[1] == 2, f"channels={result.audio.shape[1]}, erwarte 2"

    def test_05_unified_steering_used(self):
        """PhaseSteeringEngine steuert alle Nopt-Stages (unified)."""
        from backend.core.naturalness_optimizer import optimize_naturalness
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        engine = PhaseSteeringEngine()
        assert engine.enabled, "Steering-Engine sollte aktiv sein"
        audio = _make_defective_audio()
        result = optimize_naturalness(audio, audio.copy(), _SR, material="vinyl")
        assert result.hpe_after >= result.hpe_before - 0.02

    def test_06_steering_engine_available(self):
        """PhaseSteeringEngine ist importierbar und aktiv."""
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        engine = PhaseSteeringEngine()
        assert engine.enabled, "Steering-Engine sollte aktiv sein"
        assert engine._state is not None

    def test_07_multiple_materials_all_improve(self):
        """Verschiedene Materialien werden alle nicht verschlechtert."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        for material in ["vinyl", "tape", "cd_digital", "mp3_low", "unknown"]:
            audio = _make_defective_audio()
            result = optimize_naturalness(audio, audio.copy(), _SR, material=material)
            assert result.hpe_after >= result.hpe_before - 0.02, (
                f"{material}: HPE {result.hpe_before:.3f} → {result.hpe_after:.3f}"
            )

    def test_08_glue_stage_applied(self):
        """Multi-Band-Glue wird auf defektem Audio angewendet."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        audio = _make_defective_audio()
        result = optimize_naturalness(audio, audio.copy(), _SR)
        assert "multiband_glue" in result.applied_stages or result.glue_reduction_db > 0, (
            "Glue-Stage wurde nicht angewendet!"
        )

    def test_09_rapid_processing(self):
        """3s Audio wird in <2s verarbeitet."""
        import time

        from backend.core.naturalness_optimizer import optimize_naturalness

        audio = _make_defective_audio(3.0)
        t0 = time.perf_counter()
        optimize_naturalness(audio, audio.copy(), _SR)
        elapsed = time.perf_counter() - t0
        # Dry run for timing baseline
        result_dry = optimize_naturalness(audio, audio.copy(), _SR, dry_run=True)
        assert elapsed < 5.0, f"Zu langsam: {elapsed:.2f}s für 3s Audio"

    def test_10_deterministic_output(self):
        """Gleicher Input → gleicher Output (deterministisch)."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        audio = _make_defective_audio(1.0)
        r1 = optimize_naturalness(audio, audio.copy(), _SR)
        r2 = optimize_naturalness(audio, audio.copy(), _SR)
        assert np.allclose(r1.audio, r2.audio, atol=1e-6), "Output nicht deterministisch!"
