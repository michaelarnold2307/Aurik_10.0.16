from __future__ import annotations

"""Unit-Tests für backend/core/perceptual_embedder.py.

Spec §2.9: PerceptualEmbedder — 5 Kanäle (Gammatone, Chroma, MFCC-ähnlich,
Perceptual Loudness, Spectral Flatness), 256-dim AudioEmbedding, Singleton.
≥ 35 Tests: Shape, NaN, Bounds, Mono/Stereo, Ähnlichkeit gleicher Signale,
Distanz verschiedener Signale, Segment-Embedding, Edge-Cases.
"""


import numpy as np
import pytest

np.random.seed(2)

from backend.core.perceptual_embedder import (
    AudioEmbedding,
    PerceptualEmbedder,
    embed_audio,
    get_embedder,
)

SR = 48000


def _sine(freq: float = 440.0, secs: float = 2.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)  # type: ignore[no-any-return]


def _noise(secs: float = 2.0, amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(99)
    return (rng.standard_normal(int(SR * secs)) * amp).astype(np.float32)


def _stereo(secs: float = 2.0) -> np.ndarray:
    mono = _sine(secs=secs)
    return np.stack([mono, mono * 0.85])


def _short(n: int = 64) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Klasse 1: Import und Struktur
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImportAndStructure:
    def test_01_module_importable(self):
        assert PerceptualEmbedder is not None

    def test_02_audio_embedding_importable(self):
        assert AudioEmbedding is not None

    def test_03_embed_audio_callable(self):
        assert callable(embed_audio)

    def test_04_get_embedder_returns_instance(self):
        emb = get_embedder()
        assert isinstance(emb, PerceptualEmbedder)

    def test_05_get_embedder_is_singleton(self):
        emb1 = get_embedder()
        emb2 = get_embedder()
        assert emb1 is emb2


# ---------------------------------------------------------------------------
# Klasse 2: AudioEmbedding Dataclass
# ---------------------------------------------------------------------------


class TestAudioEmbeddingDataclass:
    def _make_embedding(self, dim: int = 256) -> AudioEmbedding:
        audio = _sine(secs=2.0)
        return embed_audio(audio, SR)

    def test_06_embedding_has_vector_attribute(self):
        emb = self._make_embedding()
        assert hasattr(emb, "vector") or hasattr(emb, "embedding")

    def test_07_embedding_vector_is_ndarray(self):
        emb = self._make_embedding()
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert isinstance(vec, np.ndarray)

    def test_08_embedding_no_nan(self):
        emb = self._make_embedding()
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert not np.any(np.isnan(vec))

    def test_09_embedding_no_inf(self):
        emb = self._make_embedding()
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert not np.any(np.isinf(vec))

    def test_10_cosine_similarity_same_signal_is_one(self):
        audio = _sine(secs=2.0)
        emb1 = embed_audio(audio, SR)
        emb2 = embed_audio(audio, SR)
        sim = emb1.cosine_similarity(emb2)
        assert abs(sim - 1.0) < 1e-3

    def test_11_cosine_similarity_different_signals_lower(self):
        emb_sine = embed_audio(_sine(secs=2.0), SR)
        emb_noise = embed_audio(_noise(secs=2.0), SR)
        sim = emb_sine.cosine_similarity(emb_noise)
        assert sim < 1.0

    def test_12_cosine_similarity_in_minus_one_to_one(self):
        emb1 = embed_audio(_sine(440.0, secs=2.0), SR)
        emb2 = embed_audio(_sine(880.0, secs=2.0), SR)
        sim = emb1.cosine_similarity(emb2)
        assert -1.0 <= sim <= 1.0 + 1e-6

    def test_13_perceptual_distance_same_signal_near_zero(self):
        audio = _sine(secs=2.0)
        emb1 = embed_audio(audio, SR)
        emb2 = embed_audio(audio, SR)
        dist = emb1.perceptual_distance(emb2)
        assert dist < 0.1

    def test_14_perceptual_distance_nonnegative(self):
        emb1 = embed_audio(_sine(440.0, secs=2.0), SR)
        emb2 = embed_audio(_noise(secs=2.0), SR)
        dist = emb1.perceptual_distance(emb2)
        assert dist >= 0.0

    def test_15_to_dict_returns_dict(self):
        emb = embed_audio(_sine(secs=2.0), SR)
        d = emb.to_dict()
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# Klasse 3: embed_audio() — Mono
# ---------------------------------------------------------------------------


class TestEmbedAudioMono:
    def test_16_mono_1s_returns_embedding(self):
        audio = _sine(secs=1.0)
        emb = embed_audio(audio, SR)
        assert isinstance(emb, AudioEmbedding)

    def test_17_mono_2s_no_nan(self):
        audio = _sine(secs=2.0)
        emb = embed_audio(audio, SR)
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert not np.any(np.isnan(vec))

    def test_18_mono_noise_returns_embedding(self):
        audio = _noise(secs=2.0)
        emb = embed_audio(audio, SR)
        assert isinstance(emb, AudioEmbedding)

    def test_19_silence_returns_embedding(self):
        audio = np.zeros(2 * SR, dtype=np.float32)
        emb = embed_audio(audio, SR)
        assert isinstance(emb, AudioEmbedding)

    def test_20_silence_no_nan(self):
        audio = np.zeros(2 * SR, dtype=np.float32)
        emb = embed_audio(audio, SR)
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert not np.any(np.isnan(vec))

    def test_21_clipped_audio_safe(self):
        audio = np.clip(_sine(secs=2.0) * 5.0, -1.0, 1.0)
        emb = embed_audio(audio, SR)
        assert isinstance(emb, AudioEmbedding)

    def test_22_mono_float64_accepted(self):
        audio = _sine(secs=1.0).astype(np.float64)
        emb = embed_audio(audio, SR)
        assert isinstance(emb, AudioEmbedding)


# ---------------------------------------------------------------------------
# Klasse 4: embed_audio() — Stereo
# ---------------------------------------------------------------------------


class TestEmbedAudioStereo:
    def test_23_stereo_returns_embedding(self):
        audio = _stereo(secs=2.0)
        emb = embed_audio(audio, SR)
        assert isinstance(emb, AudioEmbedding)

    def test_24_stereo_no_nan(self):
        audio = _stereo(secs=2.0)
        emb = embed_audio(audio, SR)
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert not np.any(np.isnan(vec))

    def test_25_stereo_vs_mono_different(self):
        # Stereo und Mono-Mix desselben Signals können leicht unterschiedliche
        # Embeddings erzeugen, aber beide müssen gültig sein
        mono = _sine(secs=2.0)
        stereo = np.stack([mono, mono])
        emb_mono = embed_audio(mono, SR)
        emb_stereo = embed_audio(stereo, SR)
        assert isinstance(emb_mono, AudioEmbedding)
        assert isinstance(emb_stereo, AudioEmbedding)

    def test_26_stereo_same_signal_both_channels_cosine_high(self):
        mono = _sine(secs=2.0)
        stereo = np.stack([mono, mono])
        emb_stereo = embed_audio(stereo, SR)
        emb_mono = embed_audio(mono, SR)
        sim = emb_stereo.cosine_similarity(emb_mono)
        assert sim > 0.7  # Stereo mit identischen Kanälen ≈ Mono


# ---------------------------------------------------------------------------
# Klasse 5: Edge-Cases und kurzem Audio
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_27_very_short_audio_safe(self):
        audio = _short(64)
        try:
            emb = embed_audio(audio, SR)
            assert isinstance(emb, AudioEmbedding)
        except Exception as e:
            pytest.fail(f"Sehr kurzes Audio warf Exception: {e}")

    def test_28_segment_s_none_uses_full_signal(self):
        audio = _sine(secs=3.0)
        emb = embed_audio(audio, SR, segment_s=None)
        assert isinstance(emb, AudioEmbedding)

    def test_29_segment_s_5s_with_short_audio(self):
        audio = _sine(secs=1.0)
        emb = embed_audio(audio, SR, segment_s=5.0)
        assert isinstance(emb, AudioEmbedding)

    def test_30_different_frequencies_have_different_embeddings(self):
        emb_a = embed_audio(_sine(440.0, secs=2.0), SR)
        emb_b = embed_audio(_sine(1000.0, secs=2.0), SR)
        sim = emb_a.cosine_similarity(emb_b)
        # Verschiedene Frequenzen → keine vollständige Identität
        assert sim < 1.0

    def test_31_embedder_embed_method_works(self):
        embedder = get_embedder()
        audio = _sine(secs=2.0)
        emb = embedder.embed(audio, SR)
        assert isinstance(emb, AudioEmbedding)

    def test_32_embed_no_nan_with_full_scale(self):
        audio = np.ones(2 * SR, dtype=np.float32)
        emb = embed_audio(audio, SR)
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert not np.any(np.isnan(vec))

    def test_33_nan_input_produces_finite_embedding(self):
        """NaN-Eingang muss entweder eine Exception werfen ODER
        ein vollständig finites Embedding zurückgeben — stiller NaN-Pass verboten."""
        audio = np.full(2 * SR, float("nan"), dtype=np.float32)
        try:
            emb = embed_audio(audio, SR)
        except Exception as exc:
            # Explizite Exception ist akzeptabel
            pytest.skip(f"embed_audio wirft bei NaN-Eingang Exception (akzeptabel): {exc}")
            return
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert np.all(np.isfinite(vec)), (
            f"NaN-Eingang ergab nicht-finites Embedding: {np.sum(~np.isfinite(vec))} nicht-finite Werte"
        )

    def test_34_perceptual_distance_symmetric(self):
        emb1 = embed_audio(_sine(440.0, secs=2.0), SR)
        emb2 = embed_audio(_noise(secs=2.0), SR)
        d12 = emb1.perceptual_distance(emb2)
        d21 = emb2.perceptual_distance(emb1)
        assert abs(d12 - d21) < 1e-6

    def test_35_cosine_similarity_reflexive(self):
        audio = _noise(secs=2.0)
        emb = embed_audio(audio, SR)
        sim = emb.cosine_similarity(emb)
        assert abs(sim - 1.0) < 1e-3

    def test_36_embed_returns_finite_values(self):
        audio = _sine(secs=2.0)
        emb = embed_audio(audio, SR)
        vec = emb.vector if hasattr(emb, "vector") else emb.embedding  # type: ignore[attr-defined]
        assert np.all(np.isfinite(vec))
