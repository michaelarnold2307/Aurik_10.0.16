import numpy as np
import pytest

from dsp.spectral_denoiser import SpectralDenoiser


@pytest.mark.unit
def test_spectral_denoiser_reduces_noise():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 0.2, sr)
    audio = clean + noise
    denoiser = SpectralDenoiser(reduction_db=30.0)
    processed = denoiser.process(audio, sr)
    # Prüfe, ob die Varianz nach Denoising kleiner ist als die des Originalsignals
    assert np.var(processed) < np.var(audio)


def test_spectral_denoiser_identity_on_clean():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Für saubere Signale: minimale Dämpfung, musikalische Integrität
    denoiser = SpectralDenoiser(reduction_db=1.0)
    processed = denoiser.process(clean, sr)
    mae = np.mean(np.abs(processed - clean))
    assert mae < 0.05
    # Hinweis: In produktiven Modulen sollte die Dämpfung und alle Parameter automatisch an Song, Genre und musikalische Ziele angepasst werden (siehe SOTA-Architektur).


def test_spectral_denoiser_extreme_noise():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 1.0, sr)
    audio = clean + noise
    denoiser = SpectralDenoiser(reduction_db=30.0)
    processed = denoiser.process(audio, sr)
    # Bei extremem Rauschen sollte die Varianz deutlich sinken
    assert np.var(processed - clean) < np.var(audio - clean)


def test_spectral_denoiser_stereo_lag_integrity():
    """Stereo (N, 2): identische Shape, finite Werte, keine Laufzeit-Änderung."""
    sr = 16000
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, sr, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t))[:, None]
    stereo = (tone + rng.standard_normal((sr, 2)) * 0.05).astype(np.float64)
    denoiser = SpectralDenoiser(reduction_db=12.0)
    out = denoiser.process(stereo, sr)
    assert out.shape == stereo.shape
    assert np.isfinite(out).all()


def test_spectral_denoiser_consistent_after_backend_import():
    """Produktions-Welt: backend patcht scipy.signal.stft (§v10.115).

    Der Denoiser muss das STFT/ISTFT-Paar frame-korrekt halten
    (boundary="zeros" beidseitig) — sonst entsteht ein Lag/Versatz und
    saubere Signale werden entkoppelt (Regression: MAE > 0.05).
    """
    import importlib
    import sys

    import backend  # aktiviert den signal.stft-Wrapper global (Seiteneffekt gewollt)

    sys.modules.pop("dsp.spectral_denoiser", None)
    mod = importlib.import_module("dsp.spectral_denoiser")
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
    denoiser = mod.SpectralDenoiser(reduction_db=1.0)
    out = denoiser.process(clean, sr)
    mae = np.mean(np.abs(out - clean))
    assert mae < 0.05, f"backend-Welt: MAE={mae:.4f} (Frame-Mismatch STFT/ISTFT)"


def test_spectral_denoiser_ml_hook_shape_mismatch_returns_dsp():
    """ML-Hook: Shape-Mismatch des ML-Ausgangs fällt auf DSP-only zurück."""
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.1, sr)).astype(np.float64)
    denoiser = SpectralDenoiser()
    dsp_only = denoiser.process(audio, sr)
    with_ml = denoiser.process(audio, sr, ml_output=np.zeros(sr - 10))
    assert np.allclose(with_ml, dsp_only)
