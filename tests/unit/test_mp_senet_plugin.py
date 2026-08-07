from typing import Any

import numpy as np
import pytest


def _build_fake_session(call_log):
    class _Input:
        def __init__(self, name):
            self.name = name

    class _Session:
        def get_inputs(self):
            return [_Input("noisy_amp"), _Input("noisy_pha")]

        def run(self, _out_names, feed):
            amp = np.asarray(feed["noisy_amp"], dtype=np.float32)
            pha = np.asarray(feed["noisy_pha"], dtype=np.float32)
            # Guard against the exact runtime failure reported in logs:
            # model must only receive fixed T=32 segments.
            assert amp.shape[2] == 32
            assert pha.shape[2] == 32
            call_log.append(int(amp.shape[2]))
            return [amp, pha]

    return _Session()


def _build_retry_layout_session(call_log):
    class _Input:
        def __init__(self, name):
            self.name = name

    class _Session:
        def get_inputs(self):
            return [_Input("noisy_amp"), _Input("noisy_pha")]

        def run(self, _out_names, feed):
            amp = np.asarray(feed["noisy_amp"], dtype=np.float32)
            pha = np.asarray(feed["noisy_pha"], dtype=np.float32)
            call_log.append(tuple(amp.shape))
            # First attempt [B,F,T] fails with the same signature as the real ORT error.
            if amp.shape == (1, 201, 32):
                raise RuntimeError("Reshape_4: input tensor cannot be reshaped")
            # Retry must use [B,T,F]
            assert amp.shape == (1, 32, 201)
            assert pha.shape == (1, 32, 201)
            # Return in [B,T,F] to verify output normalization in plugin.
            return [amp, pha]

    return _Session()


@pytest.mark.unit
def test_mp_senet_short_input_is_padded_to_fixed_time(monkeypatch):
    from plugins.mp_senet_plugin import MpSenetPlugin

    monkeypatch.setattr(MpSenetPlugin, "_try_load", lambda self: None)
    plugin = MpSenetPlugin()

    calls: list[Any] = []
    plugin._session = _build_fake_session(calls)

    audio = np.random.randn(144000).astype(np.float32) * 0.01
    enhanced, fail_reason = plugin._enhance_onnx(audio, 48000)

    assert fail_reason is None
    assert enhanced.shape == audio.shape
    assert len(calls) >= 1
    assert all(t == 32 for t in calls)


def test_mp_senet_long_input_uses_multiple_fixed_chunks(monkeypatch):
    from plugins.mp_senet_plugin import MpSenetPlugin

    monkeypatch.setattr(MpSenetPlugin, "_try_load", lambda self: None)
    plugin = MpSenetPlugin()

    calls: list[Any] = []
    plugin._session = _build_fake_session(calls)

    # Long enough to create >404 STFT frames and force chunked inference.
    audio = np.random.randn(300000).astype(np.float32) * 0.01
    enhanced, fail_reason = plugin._enhance_onnx(audio, 48000)

    assert fail_reason is None
    assert enhanced.shape == audio.shape
    assert len(calls) >= 2
    assert all(t == 32 for t in calls)


def test_mp_senet_retries_alternate_layout_on_reshape_error(monkeypatch):
    from plugins.mp_senet_plugin import MpSenetPlugin

    monkeypatch.setattr(MpSenetPlugin, "_try_load", lambda self: None)
    plugin = MpSenetPlugin()

    calls: list[Any] = []
    plugin._session = _build_retry_layout_session(calls)

    audio = np.random.randn(144000).astype(np.float32) * 0.01
    enhanced, fail_reason = plugin._enhance_onnx(audio, 48000)

    assert fail_reason is None
    assert enhanced.shape == audio.shape
    assert (1, 201, 32) in calls
    assert (1, 32, 201) in calls
