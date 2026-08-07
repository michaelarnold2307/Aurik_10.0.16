from __future__ import annotations

"""Regression tests for frontend audio playback normalization."""


import numpy as np
import pytest

pytestmark = pytest.mark.gui

pytest.importorskip("PyQt5")

from Aurik10.ui import modern_window


class _DummyButton:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _DummyTitleBar:
    def __init__(self) -> None:
        self.last_status: tuple[str, str] | None = None

    def set_status(self, text: str, color: str) -> None:
        self.last_status = (text, color)


class _DummyStatusText:
    def __init__(self) -> None:
        self.style = ""
        self.text = ""

    def setStyleSheet(self, style: str) -> None:
        self.style = style

    def setText(self, text: str) -> None:
        self.text = text


class _DummyWindow:
    def __init__(self) -> None:
        self.btn_stop_playback = _DummyButton()
        self.title_bar = _DummyTitleBar()
        self.status_text = _DummyStatusText()
        self._play_thread = None
        self._streaming_player = None
        self._playhead_timer = None

    def _dispatch_to_gui(self, fn) -> None:
        fn()

    def _update_playhead(self) -> None:
        return

    def _apply_status_text_style(self, style: str) -> None:
        pass

    def _stop_sd_playback_locked(self) -> None:
        pass

    def _ensure_playhead_timer(self):
        if self._playhead_timer is None:
            self._playhead_timer = _DummyTimer()  # type: ignore[assignment]
        return self._playhead_timer


class _ImmediateThread:
    def __init__(self, target=None, daemon: bool | None = None, name: str | None = None, **_: object) -> None:
        self._target = target
        self._alive = False

    def start(self) -> None:
        self._alive = True
        if self._target is not None:
            self._target()
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


class _DummySignal:
    def connect(self, _handler) -> None:
        return


class _DummyTimer:
    def __init__(self, *_args, **_kwargs) -> None:
        self.timeout = _DummySignal()

    def start(self, *_args, **_kwargs) -> None:
        return

    def stop(self) -> None:
        return


def test_normalize_audio_transposes_and_limits_channels() -> None:
    audio = np.array(
        [
            [0.25, np.nan, 2.0, -2.0],
            [0.5, np.inf, -np.inf, -0.5],
        ],
        dtype=np.float32,
    )

    normalized = modern_window._normalize_audio(audio)

    assert normalized.shape == (4, 2)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert np.max(np.abs(normalized)) <= 1.0


def test_play_audio_normalizes_before_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeStream:
        def __init__(self, **kwargs: object) -> None:
            self._chunks: list[np.ndarray] = []
            captured["samplerate"] = kwargs.get("samplerate")

        def start(self) -> None:
            pass

        def write(self, chunk: np.ndarray) -> None:
            self._chunks.append(np.array(chunk, copy=True))
            captured["data"] = np.concatenate(self._chunks, axis=0)

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeInactiveStream:
        active: bool = False

    class _FakeSoundDevice:
        def stop(self) -> None:
            captured["stopped"] = True

        def query_devices(self, kind: str | None = None) -> dict:  # type: ignore[override]
            return {}

        def play(self, data: np.ndarray, samplerate: int, blocking: bool = True, latency: str = "high") -> None:
            captured["data"] = np.array(data, copy=True)
            captured["samplerate"] = samplerate

        def get_stream(self) -> _FakeInactiveStream:
            return _FakeInactiveStream()

    monkeypatch.setattr(modern_window, "_SD_AVAILABLE", True)
    monkeypatch.setattr(modern_window, "_sd", _FakeSoundDevice())
    monkeypatch.setattr(modern_window.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(modern_window, "QTimer", _DummyTimer)

    window = _DummyWindow()
    audio = np.array(
        [
            [0.25, 0.5, 0.75],
            [np.nan, np.inf, -np.inf],
            [2.0, -2.0, 0.0],
            [0.1, 0.2, 0.3],
        ],
        dtype=np.float32,
    )

    modern_window.ModernMainWindow._play_audio(window, audio, 48_000)  # type: ignore[arg-type]

    played = captured.get("data")
    assert isinstance(played, np.ndarray)
    assert played.shape == (4, 2)
    assert np.isfinite(played).all()
    assert np.max(np.abs(played)) <= 1.0
    assert captured.get("samplerate") == 48_000
    assert window.btn_stop_playback.enabled is True
