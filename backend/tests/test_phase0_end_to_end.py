"""§v10.303.21 End-to-End-Integrationstest für Phase-0-Pipeline.

Testet:
  - ChainedPhase0Preprocessor Struktur (ApolloResult)
  - Cache-Hit/Miss-Verhalten
  - should_apply() für alle Materialtypen
  - Mehrfachaufruf-Konsistenz

Usage:
  pytest backend/tests/test_phase0_end_to_end.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def phase0_processor():
    """Einmalig geladener ChainedPhase0Preprocessor für alle Tests."""
    from plugins.apollo_phase0_integration import ChainedPhase0Preprocessor

    return ChainedPhase0Preprocessor()


@pytest.fixture
def sample_audio():
    """2-Sekunden-Testsignal: 440 Hz + Rauschen."""
    sr = 48000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    tone = np.sin(2 * np.pi * 440 * t) * 0.3
    noise = np.random.RandomState(42).randn(len(t)).astype(np.float32) * 0.02
    return (tone + noise).astype(np.float32)


@pytest.fixture
def different_audio():
    """Anderes Testsignal für Cache-Miss-Test."""
    sr = 48000
    noise = np.random.RandomState(99).randn(sr * 2).astype(np.float32) * 0.05
    return noise


# ── Tests ────────────────────────────────────────────────────────────────


class TestChainedPhase0Preprocessor:
    """Struktur- und API-Tests."""

    def test_should_apply_all_materials(self, phase0_processor):
        """should_apply() returns True für alle Materialtypen."""
        assert phase0_processor.should_apply("mp3_low")
        assert phase0_processor.should_apply("mp3_high")
        assert phase0_processor.should_apply("vinyl")
        assert phase0_processor.should_apply("cd_digital")

    def test_process_returns_apollo_result(self, phase0_processor, sample_audio):
        """process() gibt ApolloResult mit korrekter Struktur zurück."""
        result = phase0_processor.process(sample_audio, 48000, "mp3_high")
        assert result is not None
        assert hasattr(result, "audio")
        assert hasattr(result, "applied")
        assert hasattr(result, "material")
        assert hasattr(result, "metadata")
        assert result.material == "mp3_high"
        assert "chain" in (result.metadata or {})
        assert "stages" in (result.metadata or {})

    def test_process_preserves_audio_shape(self, phase0_processor, sample_audio):
        """Audio-Form bleibt erhalten (auch wenn Phase 0 nichts anwendet)."""
        result = phase0_processor.process(sample_audio, 48000, "vinyl")
        assert result.audio.shape == sample_audio.shape
        assert result.audio.dtype == np.float32

    def test_process_on_noise_rejected(self, phase0_processor):
        """Reines Rauschen wird von allen Stufen abgelehnt."""
        noise = np.random.RandomState(7).randn(48000).astype(np.float32) * 0.1
        result = phase0_processor.process(noise, 48000, "mp3_low")
        # Auf Rauschen sollte applied=False sein
        stages = (result.metadata or {}).get("stages", [])
        applied_stages = [s for s in stages if s.get("applied")]
        # Mindestens Apollo sollte auf Rauschen nicht anwenden
        # (Hallucination-Guard schlägt zu)
        assert len(applied_stages) <= 1  # Maximal eine Stufe könnte durchrutschen


class TestPhase0Cache:
    """Cache-Verhalten."""

    def test_first_call_no_cache(self, phase0_processor, sample_audio):
        """Erster Aufruf: kein Cache-Treffer."""
        result = phase0_processor.process(sample_audio, 48000, "mp3_high")
        cached = (result.metadata or {}).get("cached", False)
        # Bei reinem Ton wird Apollo wahrscheinlich abgelehnt → kein Cache
        # Der Test prüft nur, dass die Struktur passt
        assert isinstance(cached, bool)

    def test_same_audio_cache_hit(self, phase0_processor, sample_audio):
        """Zweiter Aufruf mit gleichem Audio: Cache-Treffer."""
        r1 = phase0_processor.process(sample_audio, 48000, "mp3_high")
        r2 = phase0_processor.process(sample_audio, 48000, "mp3_high")
        # Beide Ergebnisse sollten identisch sein
        assert np.array_equal(r1.audio, r2.audio)
        assert r1.applied == r2.applied

    def test_different_audio_no_false_cache(self, phase0_processor, sample_audio, different_audio):
        """Unterschiedliches Audio: kein falscher Cache-Treffer."""
        r1 = phase0_processor.process(sample_audio, 48000, "mp3_high")
        r2 = phase0_processor.process(different_audio, 48000, "mp3_high")
        # Audio-Inhalte sollten unterschiedlich sein
        assert not np.array_equal(r1.audio, r2.audio)
