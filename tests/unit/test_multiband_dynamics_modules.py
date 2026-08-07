import numpy as np
import pytest

from dsp.multiband_expander import MultibandExpander
from dsp.multiband_gate import MultibandGate
from dsp.multiband_limiter import MultibandLimiter


def _mixture(sr: int = 48000, dur_s: float = 1.0) -> np.ndarray:
    t = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False)
    x = 0.35 * np.sin(2.0 * np.pi * 110.0 * t)
    x += 0.25 * np.sin(2.0 * np.pi * 850.0 * t)
    x += 0.15 * np.sin(2.0 * np.pi * 4800.0 * t)
    return x.astype(np.float64)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_multiband_gate_supports_more_than_three_bands() -> None:
    audio = _mixture()
    proc = MultibandGate(bands=5, crossovers=(180.0, 700.0, 2200.0, 6500.0)).process(audio, 48000)  # type: ignore[arg-type]
    assert proc.shape == audio.shape
    assert np.isfinite(proc).all()
    assert np.max(np.abs(proc)) <= 1.0


def test_multiband_expander_supports_more_than_three_bands() -> None:
    audio = _mixture()
    proc = MultibandExpander(bands=4, crossovers=(160.0, 900.0, 3200.0)).process(audio, 48000)  # type: ignore[arg-type]
    assert proc.shape == audio.shape
    assert np.isfinite(proc).all()
    assert np.max(np.abs(proc)) <= 1.0


def test_multiband_limiter_lookahead_padding_no_artifact_spike() -> None:
    audio = np.zeros(48000, dtype=np.float64)
    audio[20000:20010] = 1.25
    proc = MultibandLimiter(bands=3, lookahead_ms=(3.0, 3.0, 3.0)).process(audio, 48000)
    assert proc.shape == audio.shape
    assert np.isfinite(proc).all()
    assert np.max(np.abs(proc)) <= 1.0


def test_multiband_modules_handle_stereo_input() -> None:
    mono = _mixture()
    stereo = np.stack([mono, mono * 0.8], axis=0)

    gated = MultibandGate().process(stereo, 48000)
    expanded = MultibandExpander().process(stereo, 48000)
    limited = MultibandLimiter().process(stereo, 48000)

    assert gated.shape == stereo.shape
    assert expanded.shape == stereo.shape
    assert limited.shape == stereo.shape
    assert np.isfinite(gated).all()
    assert np.isfinite(expanded).all()
    assert np.isfinite(limited).all()
