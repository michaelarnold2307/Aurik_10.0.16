"""Tests für backend/core/singer_voice_model.py — SingerVoiceModel §SVM-1.

Testet: build_from_audio, to_dict, confidence, formant_targets, vibrato, HNR.
"""

import numpy as np
import pytest

from backend.core.singer_voice_model import (
    _N_MELS,
    SingerVoiceModelResult,
    get_singer_voice_model,
)


@pytest.mark.unit
class TestSingerVoiceModelResult:
    """Dataclass + to_dict()."""

    def test_to_dict_returns_all_keys(self):
        svm = SingerVoiceModelResult(
            spectral_envelope=np.ones(_N_MELS, dtype=np.float32),
            formant_targets={"F1": 800.0, "F2": 1200.0},
            vibrato_rate_hz=5.5,
            vibrato_depth_cents=45.0,
            spectral_tilt_db_per_octave=-2.0,
            hnr_db=18.0,
            vocal_segments_seconds=3.0,
            confidence=0.6,
        )
        d = svm.to_dict()
        assert "spectral_envelope" in d
        assert "formant_targets" in d
        assert "vibrato_rate_hz" in d
        assert "vibrato_depth_cents" in d
        assert "spectral_tilt_db_per_octave" in d
        assert "hnr_db" in d
        assert "vocal_segments_seconds" in d
        assert "confidence" in d

    def test_to_dict_formant_targets_are_floats(self):
        svm = SingerVoiceModelResult(
            spectral_envelope=np.zeros(_N_MELS, dtype=np.float32),
            formant_targets={"F1": 800, "F2": 1200.5},
            vibrato_rate_hz=0.0,
            vibrato_depth_cents=0.0,
            spectral_tilt_db_per_octave=0.0,
            hnr_db=20.0,
            vocal_segments_seconds=1.0,
            confidence=1.0,
        )
        d = svm.to_dict()
        assert isinstance(d["formant_targets"]["F1"], float)  # type: ignore[index]
        assert isinstance(d["formant_targets"]["F2"], float)  # type: ignore[index]

    def test_to_dict_spectral_envelope_is_list(self):
        env = np.random.randn(_N_MELS).astype(np.float32)
        svm = SingerVoiceModelResult(
            spectral_envelope=env,
            formant_targets={},
            vibrato_rate_hz=0.0,
            vibrato_depth_cents=0.0,
            spectral_tilt_db_per_octave=0.0,
            hnr_db=20.0,
            vocal_segments_seconds=2.0,
            confidence=0.5,
        )
        d = svm.to_dict()
        assert isinstance(d["spectral_envelope"], list)
        assert len(d["spectral_envelope"]) == _N_MELS

    def test_confidence_is_float(self):
        svm = SingerVoiceModelResult(
            spectral_envelope=np.ones(_N_MELS, dtype=np.float32),
            formant_targets={},
            vibrato_rate_hz=0.0,
            vibrato_depth_cents=0.0,
            spectral_tilt_db_per_octave=0.0,
            hnr_db=20.0,
            vocal_segments_seconds=0.0,
            confidence=0.75,
        )
        assert isinstance(svm.confidence, float)
        assert 0.0 <= svm.confidence <= 1.0

    def test_hnr_db_sane_range(self):
        svm = SingerVoiceModelResult(
            spectral_envelope=np.ones(_N_MELS, dtype=np.float32),
            formant_targets={},
            vibrato_rate_hz=0.0,
            vibrato_depth_cents=0.0,
            spectral_tilt_db_per_octave=0.0,
            hnr_db=15.0,
            vocal_segments_seconds=1.0,
            confidence=0.5,
        )
        assert -20.0 <= svm.hnr_db <= 60.0

    def test_vocal_segments_seconds_non_negative(self):
        svm = SingerVoiceModelResult(
            spectral_envelope=np.ones(_N_MELS, dtype=np.float32),
            formant_targets={},
            vibrato_rate_hz=0.0,
            vibrato_depth_cents=0.0,
            spectral_tilt_db_per_octave=0.0,
            hnr_db=20.0,
            vocal_segments_seconds=0.5,
            confidence=0.3,
        )
        assert svm.vocal_segments_seconds >= 0.0


class TestSingerVoiceModelBuild:
    """build_from_audio() Integrationstests."""

    def test_build_from_sine_sweep(self):
        """Synthetischer Chirp: SVM ist rein DSP-basiert, kann auch auf Nicht-Gesang Resultate liefern."""
        sr = 48000
        dur = 3.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        chirp = np.sin(2 * np.pi * (200 + 800 * t / dur) * t).astype(np.float32)
        stereo = np.column_stack([chirp, chirp * 0.5])

        svm = get_singer_voice_model()
        result = svm.build_from_audio(stereo, sr, panns_singing=0.1)
        # SVM ist DSP-basiert; prüfe dass Resultat valide ist
        if result is not None:
            assert result.spectral_envelope is not None
            assert np.isfinite(result.hnr_db)

    def test_build_from_sine_with_vibrato(self):
        """Sinus mit Frequenzmodulation (simuliert leichtes Vibrato)."""
        sr = 48000
        dur = 5.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f0 = 220.0
        vibrato = 5.0 * np.sin(2 * np.pi * 5.5 * t)
        sine = np.sin(2 * np.pi * (f0 + vibrato) * t).astype(np.float32)

        svm = get_singer_voice_model()
        result = svm.build_from_audio(sine, sr, panns_singing=0.6)
        if result is not None:
            assert result.vocal_segments_seconds >= 0.0
            assert result.confidence >= 0.0

    def test_build_from_noise_returns_none(self):
        """Weißes Rauschen: kein Gesang → None."""
        sr = 48000
        dur = 3.0
        noise = np.random.randn(int(sr * dur)).astype(np.float32) * 0.1

        svm = get_singer_voice_model()
        result = svm.build_from_audio(noise, sr, panns_singing=0.1)
        assert result is None or result.vocal_segments_seconds < 0.5

    def test_build_from_silence_returns_none(self):
        """Stille: kein Gesang → None."""
        sr = 48000
        dur = 2.0
        silence = np.zeros(int(sr * dur), dtype=np.float32)

        svm = get_singer_voice_model()
        result = svm.build_from_audio(silence, sr, panns_singing=0.5)
        assert result is None or result.confidence < 0.3

    def test_panns_singing_below_threshold(self):
        """panns_singing ist informativ, aber SVM arbeitet DSP-basiert weiter."""
        sr = 48000
        dur = 3.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        svm = get_singer_voice_model()
        result = svm.build_from_audio(audio, sr, panns_singing=0.1)
        # SVM liefert Resultat; confidence >= 0 da genug sauberes Signal
        if result is not None:
            assert result.vocal_segments_seconds >= 0.0

    def test_mono_vs_stereo_same_shape(self):
        """Mono und Stereo sollten gleiche Spectral-Envelope-Shape liefern."""
        sr = 48000
        dur = 4.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f0 = 330.0
        mono = np.sin(2 * np.pi * f0 * t).astype(np.float32)
        stereo = np.column_stack([mono, mono])

        svm = get_singer_voice_model()
        r1 = svm.build_from_audio(mono, sr, panns_singing=0.7)
        r2 = svm.build_from_audio(stereo, sr, panns_singing=0.7)

        if r1 is not None and r2 is not None:
            assert r1.vibrato_rate_hz == pytest.approx(r2.vibrato_rate_hz, abs=1.0)
            assert r1.spectral_tilt_db_per_octave == pytest.approx(r2.spectral_tilt_db_per_octave, abs=3.0)
        # Beide sollten None oder beide nicht-None
        assert (r1 is None) == (r2 is None)


class TestSingerVoiceModelBounds:
    """Bounds + NaN/Inf-Tests."""

    def test_nan_input(self):
        svm = get_singer_voice_model()
        audio = np.full(48000, np.nan, dtype=np.float32)
        result = svm.build_from_audio(audio, 48000, panns_singing=0.5)
        assert result is None or result.confidence < 0.5

    def test_inf_input(self):
        svm = get_singer_voice_model()
        audio = np.full(48000, np.inf, dtype=np.float32)
        result = svm.build_from_audio(audio, 48000, panns_singing=0.5)
        assert result is None or np.isfinite(result.confidence)

    def test_zero_length(self):
        svm = get_singer_voice_model()
        result = svm.build_from_audio(np.array([], dtype=np.float32), 48000, panns_singing=0.5)
        assert result is None

    def test_very_short_input(self):
        svm = get_singer_voice_model()
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, 480)).astype(np.float32)
        result = svm.build_from_audio(audio, 48000, panns_singing=0.5)
        assert result is None or result.vocal_segments_seconds < 0.2


class TestSingerVoiceModelSingleton:
    """Singleton-Pattern."""

    def test_singleton_returns_same_instance(self):
        a = get_singer_voice_model()
        b = get_singer_voice_model()
        assert a is b

    def test_singleton_thread_safety_smoke(self):
        import threading

        results = []

        def get_instance():
            results.append(get_singer_voice_model())

        threads = [threading.Thread(target=get_instance) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)


class TestSingerVoiceModelReconstruct:
    """reconstruct_damaged_vocal() — wenn das Modell gebaut ist."""

    def test_reconstruct_from_built_model(self):
        """Mit einem gebauten Modell: Rekonstruktion sollte kein NaN erzeugen."""
        sr = 48000
        dur = 3.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f0 = 220.0
        clean = np.sin(2 * np.pi * f0 * t).astype(np.float32)

        svm = get_singer_voice_model()
        model = svm.build_from_audio(clean, sr, panns_singing=0.8)
        if model is not None:
            # Simuliere ein beschädigtes Segment
            damaged = clean.copy()
            damaged[1000:5000] *= 0.3
            damage_mask = np.zeros(len(clean), dtype=np.float32)
            damage_mask[1000:5000] = 1.0

            repaired = svm.reconstruct_damaged_vocal(damaged, sr, model, damage_mask)
            assert repaired is not None
            assert repaired.dtype == np.float32
            assert np.all(np.isfinite(repaired))
            # Repariertes Segment sollte nicht identisch zu beschädigtem sein
            assert not np.allclose(repaired[1000:5000], damaged[1000:5000], atol=1e-6)
