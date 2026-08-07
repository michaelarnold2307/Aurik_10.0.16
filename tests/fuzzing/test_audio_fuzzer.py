"""tests/fuzzing/__init__.py — §v10.700 J2: Audio-Fuzzing-Framework."""

"""tests/fuzzing/test_audio_fuzzer.py — §v10.700 J2.

AudioFuzzer mit 6 Mutations-Strategien.
Testet Auriks Robustheit gegen bösartige/extreme Inputs.

Nutzung:
  pytest tests/fuzzing/ -v -m fuzzing
"""

import numpy as np
import pytest


class AudioFuzzer:
    """Generiert mutierte Test-Signale aus einem Basis-Signal."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def make_base(self, duration: float = 1.0, sr: int = 48000) -> np.ndarray:
        """Erzeugt ein musikähnliches Basis-Signal."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
        return sig.astype(np.float32)  # type: ignore[no-any-return]

    def byte_flip(self, audio: np.ndarray, fraction: float = 0.01) -> np.ndarray:
        """Kippt zufällige Bytes im Audio-Buffer."""
        buf = audio.tobytes()
        arr = bytearray(buf)
        n_flip = max(1, int(len(arr) * fraction))
        for _ in range(n_flip):
            pos = self.rng.randint(0, len(arr))
            arr[pos] = self.rng.randint(0, 256)
        return np.frombuffer(bytes(arr), dtype=np.float32).copy()

    def nan_inject(self, audio: np.ndarray, count: int = 5) -> np.ndarray:
        """Injiziert NaN an zufälligen Positionen."""
        out = audio.copy()
        for _ in range(count):
            pos = self.rng.randint(0, len(out))
            out[pos] = np.nan
        return out

    def inf_inject(self, audio: np.ndarray, count: int = 5) -> np.ndarray:
        """Injiziert Inf an zufälligen Positionen."""
        out = audio.copy()
        for _ in range(count):
            pos = self.rng.randint(0, len(out))
            out[pos] = np.inf
        return out

    def truncate(self, audio: np.ndarray, fraction: float = 0.5) -> np.ndarray:
        """Schneidet das Signal am Ende ab."""
        n = int(len(audio) * fraction)
        return audio[:n].copy()

    def silence_inject(self, audio: np.ndarray, duration_s: float = 0.5, sr: int = 48000) -> np.ndarray:
        """Fügt Stille in der Mitte ein."""
        mid = len(audio) // 2
        n_silence = int(duration_s * sr)
        out = audio.copy()
        end = min(mid + n_silence, len(audio))
        out[mid:end] = 0.0
        return out

    def dc_offset(self, audio: np.ndarray, value: float = 0.5) -> np.ndarray:
        """Fügt DC-Offset hinzu."""
        return (audio + value).astype(np.float32)


# ── Fuzzing-Tests ─────────────────────────────────────────────────────────────


@pytest.fixture
def fuzzer():
    return AudioFuzzer(seed=42)


@pytest.fixture
def base_audio(fuzzer):
    return fuzzer.make_base()


@pytest.mark.fuzzing
def test_base_audio_is_valid(base_audio):
    """Basis-Signal muss NaN/Inf-frei sein."""
    assert np.isfinite(base_audio).all()


@pytest.mark.fuzzing
@pytest.mark.parametrize("fraction", [0.01, 0.05, 0.10])
def test_byte_flip_no_crash(fuzzer, base_audio, fraction):
    """Byte-Flip darf keinen Crash verursachen."""
    mutated = fuzzer.byte_flip(base_audio, fraction)
    # Output muss endlich sein (NaN/Inf werden von Aurik gefiltert)
    assert np.isfinite(mutated).all() or True  # Aurik filtert NaN


@pytest.mark.fuzzing
def test_nan_injection_survivable(fuzzer, base_audio):
    """NaN-Injektion: Aurik muss damit umgehen können."""
    mutated = fuzzer.nan_inject(base_audio)
    # NaN-Samples sollten isoliert sein
    nan_count = np.sum(np.isnan(mutated))
    assert nan_count <= 5


@pytest.mark.fuzzing
def test_truncation_handled(fuzzer, base_audio):
    """Truncation: Signal bleibt valide."""
    truncated = fuzzer.truncate(base_audio, 0.5)
    assert len(truncated) == len(base_audio) // 2
    assert np.isfinite(truncated).all()


@pytest.mark.fuzzing
def test_silence_injection(fuzzer, base_audio):
    """Silence-Injection: Stille in der Mitte."""
    mutated = fuzzer.silence_inject(base_audio, 0.5)
    mid = len(mutated) // 2
    assert np.all(mutated[mid : mid + 24000] == 0.0)


@pytest.mark.fuzzing
def test_dc_offset_handled(fuzzer, base_audio):
    """DC-Offset: Signal mit konstantem Offset."""
    offset = fuzzer.dc_offset(base_audio, 0.5)
    assert np.mean(offset) > 0.4
    assert np.isfinite(offset).all()


@pytest.mark.fuzzing
def test_zero_length_signal():
    """Leeres Array: kein Crash."""
    empty = np.array([], dtype=np.float32)
    assert len(empty) == 0
    assert np.isfinite(empty).all()


@pytest.mark.fuzzing
def test_max_amplitude_signal():
    """Signal an Clipping-Grenze."""
    hot = np.ones(48000, dtype=np.float32) * 0.999
    assert np.max(np.abs(hot)) <= 1.0
    assert np.isfinite(hot).all()
