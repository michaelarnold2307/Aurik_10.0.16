"""Regression-Tests fuer phase_31 iSTFT-Notfallpfad."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase

SR = 48_000


def _sine(freq: float = 440.0, dur: float = 1.2, sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, dur, int(dur * sr), endpoint=False, dtype=np.float64)
    return (0.45 * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)  # type: ignore[no-any-return]


@pytest.mark.unit
def test_phase31_istft_exception_uses_ola_fallback(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    audio = _sine()

    def _raise_istft(*_args, **_kwargs):
        raise RuntimeError("forced-istft-fail")

    monkeypatch.setattr("backend.core.phases.phase_31_speed_pitch_correction.signal.istft", _raise_istft)

    out = phase._phase_vocoder_mono(audio, ratio=1.04, nperseg=2048, noverlap=1024)

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    # Der Notfallpfad darf kein stiller Passthrough sein.
    assert float(np.mean(np.abs(out - audio))) > 1e-6


def test_phase31_double_failure_returns_original_not_silence(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    audio = _sine()

    def _raise_istft(*_args, **_kwargs):
        raise RuntimeError("forced-istft-fail")

    def _raise_irfft(*_args, **_kwargs):
        raise RuntimeError("forced-irfft-fail")

    monkeypatch.setattr("backend.core.phases.phase_31_speed_pitch_correction.signal.istft", _raise_istft)
    monkeypatch.setattr("backend.core.phases.phase_31_speed_pitch_correction.np.fft.irfft", _raise_irfft)

    out = phase._phase_vocoder_mono(audio, ratio=1.03, nperseg=2048, noverlap=1024)

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    # Bei doppeltem Fehler muss ein qualitaetssicherer Fallback greifen (kein Nullsignal).
    assert float(np.max(np.abs(out))) > 1e-4
    assert float(np.mean(np.abs(out - audio))) < 1e-4


def test_phase31_stereo_phase_vocoder_applies_combined_peak_guard(monkeypatch):
    phase = SpeedPitchCorrectionPhase()

    audio = np.column_stack([_sine(), 0.9 * _sine(freq=330.0)]).astype(np.float64)

    def _loud_mono(_audio, _ratio, _nperseg, _noverlap):
        return np.asarray(_audio, dtype=np.float64) * 1.8

    monkeypatch.setattr(phase, "_phase_vocoder_mono", _loud_mono)

    out = phase._correct_phase_vocoder(audio, ratio=1.02, _params={})

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert float(np.percentile(np.abs(out), 99.9)) <= 1.0 + 1e-6


def test_phase31_stereo_paths_keep_target_length(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    audio = np.column_stack([_sine(dur=1.0), 0.8 * _sine(freq=300.0, dur=1.0)]).astype(np.float64)
    n = audio.shape[0]

    def _wsola_bad_lengths(_audio, _window_size, _hop_analysis, _hop_synthesis):
        # Kanal A zu kurz, Kanal B zu lang.
        if np.mean(_audio) > 0:
            return np.ones(n - 123, dtype=np.float64) * 0.1
        return np.ones(n + 211, dtype=np.float64) * 0.1

    monkeypatch.setattr(phase, "_wsola_mono", _wsola_bad_lengths)
    out_wsola = phase._correct_wsola(audio, ratio=0.97, params={})
    assert out_wsola.shape == audio.shape
    assert np.all(np.isfinite(out_wsola))

    def _pv_bad_lengths(_audio, _ratio, _nperseg, _noverlap):
        if np.mean(_audio) > 0:
            return np.ones(n - 101, dtype=np.float64) * 0.2
        return np.ones(n + 173, dtype=np.float64) * 0.2

    monkeypatch.setattr(phase, "_phase_vocoder_mono", _pv_bad_lengths)
    out_pv = phase._correct_phase_vocoder(audio, ratio=1.03, _params={})
    assert out_pv.shape == audio.shape
    assert np.all(np.isfinite(out_pv))


def test_phase31_phase_vocoder_uses_75pct_overlap(monkeypatch):
    phase = SpeedPitchCorrectionPhase()
    audio = np.column_stack([_sine(dur=0.6), _sine(freq=330.0, dur=0.6)]).astype(np.float64)

    captured: Any = {"noverlap": []}

    def _capture(_audio, _ratio, _nperseg, _noverlap):
        captured["noverlap"].append(_noverlap)
        return np.asarray(_audio, dtype=np.float64)

    monkeypatch.setattr(phase, "_phase_vocoder_mono", _capture)
    _ = phase._correct_phase_vocoder(audio, ratio=1.01, _params={})

    assert captured["noverlap"]
    assert all(v == 1536 for v in captured["noverlap"])


def test_phase31_ola_empty_stft_returns_original_audio():
    phase = SpeedPitchCorrectionPhase()
    audio = _sine(dur=0.4)

    out = phase._istft_fallback_ola(
        zxx=np.zeros((1025, 0), dtype=np.complex64),
        nperseg=2048,
        noverlap=1024,
        original_audio=audio,
    )

    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))
    assert float(np.mean(np.abs(out - audio))) < 1e-10


def test_phase31_damage_shield_corrects_clear_stereo_onset_delay():
    phase = SpeedPitchCorrectionPhase()
    left = _sine(dur=0.5).astype(np.float32)
    shift = 96
    right = np.zeros_like(left)
    right[shift:] = left[:-shift]
    stereo = np.column_stack([left, right]).astype(np.float32)

    guarded, meta = phase._apply_preventive_damage_shield(
        original_audio=stereo,
        processed_audio=stereo,
        sample_rate=SR,
        material_type="tape",
    )

    assert guarded.shape == stereo.shape
    assert np.all(np.isfinite(guarded))
    assert meta["phase31_stereo_delay_corrected"] is True
    assert float(np.mean(np.abs(guarded[:, 0] - guarded[:, 1]))) < float(np.mean(np.abs(stereo[:, 0] - stereo[:, 1])))
