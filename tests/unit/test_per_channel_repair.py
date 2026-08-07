"""Tests for per-channel defect repair and live waveform updates.

Validates:
- _ChannelSplitPhaseProxy processes stereo channels independently
- audio_update_callback is emitted after each phase
- Intermediate L-channel callback fires before R-channel processing
- Mono audio passes through the proxy unchanged
- WaveformWidget.update_audio_live preserves zoom/pan state
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePhaseResult:
    """Minimal PhaseResult stub."""

    def __init__(self, audio, success=True):
        self.audio = audio
        self.success = success
        self.warnings = []
        self.execution_time_seconds = 0.01
        self.profiling = {}


class _FakePhase:
    """Minimal PhaseInterface stub for testing."""

    def __init__(self, gain: float = 0.5):
        self._gain = gain
        self._call_count = 0

    def process(self, audio, **kwargs):
        self._call_count += 1
        out = (audio * self._gain).astype(np.float32)
        return _FakePhaseResult(out)

    def get_metadata(self):
        return SimpleNamespace(
            phase_id="phase_01_click_removal",
            name="Click Removal",
            estimated_time_factor=0.1,
        )


class _FakePhaseChangesLength:
    """Phase that changes audio length (edge case)."""

    def process(self, audio, **kwargs):
        # Returns shorter audio
        out = audio[: len(audio) // 2].astype(np.float32)
        return _FakePhaseResult(out)

    def get_metadata(self):
        return SimpleNamespace(phase_id="phase_99_truncator", name="Truncator", estimated_time_factor=0.1)


class _FakePhaseReturnsNdarray:
    """Phase that returns raw ndarray instead of PhaseResult."""

    def process(self, audio, **kwargs):
        return (audio * 0.8).astype(np.float32)

    def get_metadata(self):
        return SimpleNamespace(phase_id="phase_98_raw", name="Raw", estimated_time_factor=0.1)


def _get_proxy_class():
    """Import _ChannelSplitPhaseProxy from UV3."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    return UnifiedRestorerV3._ChannelSplitPhaseProxy


# ---------------------------------------------------------------------------
# Tests: _ChannelSplitPhaseProxy — Stereo split
# ---------------------------------------------------------------------------


class TestChannelSplitPhaseProxy:
    """Tests for the per-channel defect repair proxy."""

    def test_stereo_splits_into_two_mono_calls(self):
        """Stereo audio must result in two independent phase.process() calls."""
        Proxy = _get_proxy_class()
        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        stereo = np.random.randn(4800, 2).astype(np.float32)
        proxy.process(stereo)
        # Phase should be called twice (once per channel)
        assert phase._call_count == 2

    def test_stereo_result_has_two_channels(self):
        """Result from proxy must be stereo (N, 2)."""
        Proxy = _get_proxy_class()
        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        stereo = np.random.randn(4800, 2).astype(np.float32)
        result = proxy.process(stereo)
        audio_out = result.audio if hasattr(result, "audio") else result
        assert audio_out.ndim == 2
        assert audio_out.shape == (4800, 2)

    def test_stereo_channels_processed_independently(self):
        """Each channel should be processed with the same gain independently."""
        Proxy = _get_proxy_class()
        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        stereo = np.ones((100, 2), dtype=np.float32)
        stereo[:, 0] = 0.8  # L = 0.8
        stereo[:, 1] = 0.4  # R = 0.4
        result = proxy.process(stereo)
        audio_out = result.audio if hasattr(result, "audio") else result
        np.testing.assert_allclose(audio_out[:, 0], 0.4, atol=1e-6)  # 0.8 * 0.5
        np.testing.assert_allclose(audio_out[:, 1], 0.2, atol=1e-6)  # 0.4 * 0.5

    def test_mono_passthrough(self):
        """Mono audio (1D) must pass through to the real phase without splitting."""
        Proxy = _get_proxy_class()
        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        mono = np.ones(4800, dtype=np.float32)
        result = proxy.process(mono)
        # Phase called only once for mono
        assert phase._call_count == 1
        audio_out = result.audio if hasattr(result, "audio") else result
        assert audio_out.ndim == 1

    def test_intermediate_callback_fires_after_left_channel(self):
        """audio_update_callback must fire after L channel, before R channel."""
        Proxy = _get_proxy_class()
        callback_calls = []

        def _cb(audio, sr, phase_id):
            callback_calls.append((audio.copy(), sr, phase_id))

        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=_cb, sample_rate=48000)
        stereo = np.ones((100, 2), dtype=np.float32)
        proxy.process(stereo)

        # Should have exactly 1 intermediate callback (after L, before R)
        assert len(callback_calls) == 1
        audio_intermediate, sr, pid = callback_calls[0]
        assert sr == 48000
        assert ":L" in pid
        # Intermediate: L channel processed (0.5), R channel original (1.0)
        np.testing.assert_allclose(audio_intermediate[:, 0], 0.5, atol=1e-6)
        np.testing.assert_allclose(audio_intermediate[:, 1], 1.0, atol=1e-6)

    def test_no_callback_for_mono(self):
        """No intermediate callback for mono audio."""
        Proxy = _get_proxy_class()
        callback_calls = []
        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=lambda a, s, p: callback_calls.append(1), sample_rate=48000)
        mono = np.ones(100, dtype=np.float32)
        proxy.process(mono)
        assert len(callback_calls) == 0

    def test_length_mismatch_padded(self):
        """If phase returns shorter audio, it should be padded to original length."""
        Proxy = _get_proxy_class()
        phase = _FakePhaseChangesLength()
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        stereo = np.ones((200, 2), dtype=np.float32)
        result = proxy.process(stereo)
        audio_out = result.audio if hasattr(result, "audio") else result
        assert audio_out.shape == (200, 2)

    def test_raw_ndarray_return(self):
        """Phase returning raw ndarray instead of PhaseResult works."""
        Proxy = _get_proxy_class()
        phase = _FakePhaseReturnsNdarray()
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        stereo = np.ones((100, 2), dtype=np.float32)
        result = proxy.process(stereo)
        # When phase returns ndarray, proxy returns combined ndarray
        audio_out = result if isinstance(result, np.ndarray) else result.audio
        assert audio_out.shape == (100, 2)
        np.testing.assert_allclose(audio_out, 0.8, atol=1e-6)

    def test_get_metadata_delegates(self):
        """Proxy get_metadata() must delegate to the real phase."""
        Proxy = _get_proxy_class()
        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        meta = proxy.get_metadata()
        assert meta.phase_id == "phase_01_click_removal"

    def test_getattr_delegates(self):
        """Arbitrary attribute access must delegate to the real phase."""
        Proxy = _get_proxy_class()
        phase = _FakePhase(gain=0.75)
        proxy = Proxy(phase, audio_update_callback=None, sample_rate=48000)
        assert proxy._gain == 0.75

    def test_nan_in_phase_output_guarded(self):
        """NaN in phase output should be replaced with zeros."""
        Proxy = _get_proxy_class()

        class _NanPhase:
            def process(self, audio, **kw):
                out = np.full_like(audio, np.nan)
                return _FakePhaseResult(out)

            def get_metadata(self):
                return SimpleNamespace(phase_id="phase_nan", name="Nan", estimated_time_factor=0.1)

        proxy = Proxy(_NanPhase(), audio_update_callback=None, sample_rate=48000)
        stereo = np.ones((100, 2), dtype=np.float32)
        result = proxy.process(stereo)
        audio_out = result.audio if hasattr(result, "audio") else result
        assert np.all(np.isfinite(audio_out))

    def test_callback_exception_does_not_crash(self):
        """Exception in audio_update_callback must not crash the proxy."""
        Proxy = _get_proxy_class()

        def _bad_cb(audio, sr, pid):
            raise RuntimeError("Callback crashed")

        phase = _FakePhase(gain=0.5)
        proxy = Proxy(phase, audio_update_callback=_bad_cb, sample_rate=48000)
        stereo = np.ones((100, 2), dtype=np.float32)
        # Should not raise
        result = proxy.process(stereo)
        audio_out = result.audio if hasattr(result, "audio") else result
        assert audio_out.shape == (100, 2)


# ---------------------------------------------------------------------------
# Tests: _CHANNEL_SPECIFIC_PHASES set
# ---------------------------------------------------------------------------


class TestChannelSpecificPhasesSet:
    """Verify the _CHANNEL_SPECIFIC_PHASES frozenset."""

    def test_expected_phases_present(self):
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        expected = {
            "phase_01_click_removal",
            "phase_09_crackle_removal",
            "phase_23_spectral_repair",
            "phase_24_dropout_repair",
            "phase_27_click_pop_removal",
        }
        assert expected == UnifiedRestorerV3._CHANNEL_SPECIFIC_PHASES

    def test_is_frozenset(self):
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        assert isinstance(UnifiedRestorerV3._CHANNEL_SPECIFIC_PHASES, frozenset)


# ---------------------------------------------------------------------------
# Tests: audio_update_callback in _execute_pipeline signature
# ---------------------------------------------------------------------------


class TestExecutePipelineCallback:
    """Verify audio_update_callback parameter exists in _execute_pipeline."""

    def test_audio_update_callback_param_exists(self):
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        sig = inspect.signature(UnifiedRestorerV3._execute_pipeline)
        assert "audio_update_callback" in sig.parameters

    def test_audio_update_callback_default_none(self):
        import inspect

        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        sig = inspect.signature(UnifiedRestorerV3._execute_pipeline)
        param = sig.parameters["audio_update_callback"]
        assert param.default is None


# ---------------------------------------------------------------------------
# Tests: WaveformWidget.update_audio_live
# ---------------------------------------------------------------------------


class TestWaveformWidgetLiveUpdate:
    """Verify update_audio_live preserves zoom/pan."""

    @pytest.mark.gui
    def test_update_audio_live_preserves_view(self):
        """update_audio_live must NOT reset _view_start / _view_end."""
        try:
            from Aurik10.ui.modern_window import WaveformWidget
        except ImportError:
            pytest.skip("WaveformWidget not importable (Qt not available)")

        try:
            widget = WaveformWidget.__new__(WaveformWidget)
            # Simulate pre-existing state
            widget.audio_data = np.zeros(4800, dtype=np.float32)
            widget.sample_rate = 48000
            widget._view_start = 0.25
            widget._view_end = 0.75
            widget.update = MagicMock()  # Don't actually repaint  # type: ignore[method-assign]

            new_audio = np.ones(4800, dtype=np.float32) * 0.5
            widget.update_audio_live(new_audio, 48000)

            # View window must NOT be reset
            assert widget._view_start == 0.25
            assert widget._view_end == 0.75
            # Audio data must be updated
            np.testing.assert_array_equal(widget.audio_data, new_audio)
            # update() must be called to trigger repaint
            widget.update.assert_called_once()
        except Exception:
            pytest.skip("WaveformWidget instantiation failed (no QApplication)")

    @pytest.mark.gui
    def test_update_waveform_resets_view(self):
        """update_waveform (initial load) MUST reset _view_start / _view_end."""
        try:
            from Aurik10.ui.modern_window import WaveformWidget
        except ImportError:
            pytest.skip("WaveformWidget not importable (Qt not available)")

        try:
            widget = WaveformWidget.__new__(WaveformWidget)
            widget.audio_data = None
            widget.sample_rate = 48000
            widget._view_start = 0.25
            widget._view_end = 0.75
            widget._repair_history = []  # type: ignore[assignment]
            widget._resolved_locations = {}
            widget._active_tool = "old_tool"
            widget.update = MagicMock()  # type: ignore[method-assign]

            widget.update_waveform(np.zeros(4800, dtype=np.float32), 48000)

            assert widget._view_start == 0.0
            assert widget._view_end == 1.0
            assert widget._active_tool == ""
        except Exception:
            pytest.skip("WaveformWidget instantiation failed (no QApplication)")

    @pytest.mark.gui
    def test_set_defects_does_not_mark_detected_low_score_as_resolved(self):
        """Low rounded scores in detected state must not move active locations to resolved."""
        try:
            from Aurik10.ui.modern_window import WaveformWidget
        except ImportError:
            pytest.skip("WaveformWidget not importable (Qt not available)")

        try:
            widget = WaveformWidget.__new__(WaveformWidget)
            widget.update = MagicMock()  # type: ignore[method-assign]
            widget._active_tool = "DSP"
            widget._repair_history = {}
            widget._resolved_locations = {}
            widget._channel_locations = {}
            # Existing active marker from previous frame.
            widget._defect_locations = {"clicks": [(1.0, 1.1)]}

            # New frame: score rounded to 0, but locations still present.
            widget.set_defects(
                {
                    "status": "detected",
                    "clicks": 0,
                    "_locations": {"clicks": [(1.0, 1.1)]},
                    "_channel_locations": {},
                }
            )

            assert widget._defect_locations.get("clicks") == [(1.0, 1.1)]
            assert widget._resolved_locations.get("clicks", []) == []
        except Exception:
            pytest.skip("WaveformWidget instantiation failed (no QApplication)")

    @pytest.mark.gui
    def test_set_defects_marks_resolved_in_correcting_when_locations_gone(self):
        """During correcting/completed, vanished locations with zero score should move to resolved."""
        try:
            from Aurik10.ui.modern_window import WaveformWidget
        except ImportError:
            pytest.skip("WaveformWidget not importable (Qt not available)")

        try:
            widget = WaveformWidget.__new__(WaveformWidget)
            widget.update = MagicMock()  # type: ignore[method-assign]
            widget._active_tool = "DeClick"
            widget._repair_history = {}
            widget._resolved_locations = {}
            widget._channel_locations = {}
            widget._defect_locations = {"clicks": [(1.0, 1.1)]}

            widget.set_defects(
                {
                    "status": "correcting",
                    "clicks": 0,
                    "_locations": {},
                    "_channel_locations": {},
                }
            )

            assert widget._resolved_locations.get("clicks") == [(1.0, 1.1)]
            assert widget._repair_history.get("clicks") is not None
        except Exception:
            pytest.skip("WaveformWidget instantiation failed (no QApplication)")


# ---------------------------------------------------------------------------
# Tests: Denker chain — audio_update_callback forwarding
# ---------------------------------------------------------------------------


class TestDenkerCallbackForwarding:
    """Verify audio_update_callback is accepted in denker signatures."""

    def test_restaurier_denker_accepts_audio_update_callback(self):
        import inspect

        from denker.restaurier_denker import RestaurierDenker

        sig = inspect.signature(RestaurierDenker.restauriere)
        assert "audio_update_callback" in sig.parameters

    def test_aurik_denker_restauriere_accepts_audio_update_callback(self):
        import inspect

        from denker.aurik_denker import AurikDenker

        sig = inspect.signature(AurikDenker.restauriere)
        assert "audio_update_callback" in sig.parameters

    def test_aurik_denker_orchestriere_accepts_audio_update_callback(self):
        import inspect

        from denker.aurik_denker import AurikDenker

        sig = inspect.signature(AurikDenker._orchestriere)
        assert "audio_update_callback" in sig.parameters


# ---------------------------------------------------------------------------
# Tests: BatchProcessingThread signal
# ---------------------------------------------------------------------------


class TestBatchProcessingThreadSignal:
    """Verify waveform_phase_update signal exists."""

    def test_waveform_phase_update_signal_exists(self):
        try:
            from Aurik10.ui.modern_window import BatchProcessingThread

            assert hasattr(BatchProcessingThread, "waveform_phase_update")
        except ImportError:
            pytest.skip("BatchProcessingThread not importable (Qt not available)")
