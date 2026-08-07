"""Tests für ExamplePlugin — Vorlage für eigene Plugin-Tests.

§15.6: Jedes Plugin MUSS Tests haben.
Kopiere diese Datei und passe sie an dein Plugin an.
"""

import numpy as np
import pytest

from plugins.sdk.testing_fixtures import (
    VirtualAurikPipeline,
    make_noisy_audio,
    make_test_audio,
)


class TestExamplePlugin:
    """Test-Suite für ExamplePlugin."""

    @pytest.fixture
    def plugin(self):
        from plugins.sdk.example_plugin.example_plugin import ExamplePlugin

        return ExamplePlugin()  # type: ignore[abstract]

    @pytest.fixture
    def pipeline(self):
        return VirtualAurikPipeline(material="vinyl", era=1972)

    def test_manifest_valid(self, plugin):
        """Manifest muss alle Pflichtfelder haben."""
        manifest = plugin.get_manifest()
        assert manifest.name == "example-plugin"
        assert manifest.version == "1.0.0"
        assert len(manifest.description) > 0

    def test_validate_passes(self, plugin):
        """validate() muss True zurückgeben."""
        ok, msg = plugin.validate()
        assert ok, f"validate() failed: {msg}"

    def test_process_audio_identity(self, plugin):
        """Gain=0 dB → Audio unverändert."""
        audio = make_test_audio(duration_s=1.0, sr=48000)
        result = plugin.process_audio(audio, sr=48000, gain_db=0.0)
        assert np.allclose(audio, result, atol=1e-6)
        assert result.dtype == np.float32

    def test_process_audio_gain(self, plugin):
        """+6 dB → Amplitude verdoppelt sich."""
        audio = make_test_audio(duration_s=1.0, sr=48000)
        result = plugin.process_audio(audio, sr=48000, gain_db=6.0)
        assert np.allclose(result, audio * 2.0, atol=1e-6)

    def test_process_audio_no_nan(self, plugin):
        """Keine NaN/Inf im Output."""
        audio = make_test_audio(duration_s=1.0, sr=48000)
        for gain in [-20, -6, 0, 6, 20]:
            result = plugin.process_audio(audio, sr=48000, gain_db=float(gain))
            assert np.all(np.isfinite(result)), f"NaN/Inf bei gain={gain}"

    def test_safe_process_handles_error(self, plugin):
        """safe_process() fängt Fehler und gibt PluginResult zurück."""
        result = plugin.safe_process(np.ones(100), sr=48000, gain_db=0)
        assert result.success
        assert result.audio is not None

    def test_virtual_pipeline_integration(self, plugin, pipeline):
        """Plugin läuft erfolgreich in VirtualAurikPipeline."""
        audio = make_noisy_audio(duration_s=2.0, sr=48000)
        result = pipeline.run_plugin(plugin, audio, sr=48000, gain_db=3.0)
        assert result.success, f"Pipeline failed: {result.error}"
        assert result.audio.shape == audio.shape
        assert result.total_time_s > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
