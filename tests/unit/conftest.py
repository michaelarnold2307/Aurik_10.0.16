"""
Aurik 10.0.0.x — Unit-Test Fixtures (leichtgewichtig)
==================================================

Nur numpy / stdlib — KEIN Backend-Import.
Session-scoped: einmalig pro xdist-Worker, nicht pro Test.
Ergänzt (überlagert nicht) tests/conftest.py.

Author: Aurik Testing Team
"""

import numpy as np
import pytest

# ─── Konstanten ───────────────────────────────────────────────────────────────
SR_44 = 44100
SR_48 = 48000
DUR = 0.25  # Sekunden — kurz genug für alle Unit-Tests


# ─── Mono / Stereo ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def sr44() -> int:
    """Samplerate 44100 Hz."""
    return SR_44


@pytest.fixture(scope="session")
def sr48() -> int:
    """Samplerate 48000 Hz."""
    return SR_48


@pytest.fixture(scope="session")
def mono_sine_44(sr44) -> np.ndarray:
    """440-Hz-Sinus, 0.25s@44100Hz, float32. Einmal pro Worker."""
    n = int(sr44 * DUR)
    t = np.linspace(0, DUR, n, endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def mono_sine_48(sr48) -> np.ndarray:
    """440-Hz-Sinus, 0.25s@48000Hz, float32. Einmal pro Worker."""
    n = int(sr48 * DUR)
    t = np.linspace(0, DUR, n, endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def stereo_sine_44(mono_sine_44) -> np.ndarray:
    """Stereo 2×N@44100Hz: L=440Hz, R=880Hz, float32."""
    n = mono_sine_44.shape[0]
    t = np.linspace(0, DUR, n, endpoint=False, dtype=np.float32)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    return np.stack([mono_sine_44, right])


@pytest.fixture(scope="session")
def stereo_sine_48(mono_sine_48) -> np.ndarray:
    """Stereo 2×N@48000Hz, float32."""
    n = mono_sine_48.shape[0]
    t = np.linspace(0, DUR, n, endpoint=False, dtype=np.float32)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    return np.stack([mono_sine_48, right])


# ─── Sonderfälle ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def silence_44(sr44) -> np.ndarray:
    """Stille, 0.25s@44100Hz, float32."""
    return np.zeros(int(sr44 * DUR), dtype=np.float32)


@pytest.fixture(scope="session")
def white_noise_44(sr44) -> np.ndarray:
    """Weißes Rauschen, 0.25s@44100Hz, seed=42, float32."""
    rng = np.random.default_rng(seed=42)
    return (rng.standard_normal(int(sr44 * DUR)) * 0.1).astype(np.float32)


@pytest.fixture(scope="session")
def impulse_44(sr44) -> np.ndarray:
    """Einzelimpuls bei Sample 0, 0.25s@44100Hz, float32."""
    sig = np.zeros(int(sr44 * DUR), dtype=np.float32)
    sig[0] = 1.0
    return sig


# ─── Parametrisierte Hilfstabellen ────────────────────────────────────────────

COMMON_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000, 96000]
COMMON_BIT_DEPTHS = [8, 16, 24, 32]
COMMON_CHANNELS = [1, 2]
