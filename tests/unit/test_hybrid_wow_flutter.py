import numpy as np
import pytest


@pytest.mark.unit
def test_polyphonic_implausible_speed_curve_falls_back_to_pyin(monkeypatch):
    from backend.core.hybrid.hybrid_wow_flutter import PolyphonicSpeedCurveEstimator

    class _FakeBasicPitchResult:
        def __init__(self, t: int, k: int):
            # Create a huge pitch jump that leads to implausible cents deviation.
            # First half sets per-voice reference medians to ~100 Hz.
            # Second half pushes > +500 cents after clipping, thus >200 cents final range.
            self.pitches_hz = np.full((t, k), 100.0, dtype=np.float32)
            self.pitches_hz[t // 2 :, :] = 2000.0
            self.confidences = np.full((t, k), 0.95, dtype=np.float32)
            self.frame_times_s = np.arange(t, dtype=np.float32) * 0.01

    class _FakeBasicPitch:
        _model_loaded = True

        def analyze(self, _audio, _sr, max_polyphony=6):
            return _FakeBasicPitchResult(t=120, k=min(3, max_polyphony))

    est = PolyphonicSpeedCurveEstimator()
    est._bp = _FakeBasicPitch()  # type: ignore[assignment]

    fallback_pitch = np.full(120, 220.0, dtype=np.float32)
    fallback_conf = np.full(120, 0.42, dtype=np.float32)

    def _fake_pyin_fallback(_audio, _sr):
        return fallback_pitch, fallback_conf

    monkeypatch.setattr(est, "_pyin_fallback", _fake_pyin_fallback)

    audio = np.random.randn(48000).astype(np.float32) * 0.01
    pitch, conf = est.estimate(audio, 48000)

    assert np.allclose(pitch, fallback_pitch)
    assert np.allclose(conf, fallback_conf)
