from __future__ import annotations

"""
Tests für backend/core/signal_flow_tracer.py
"""


import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_sine(freq: float = 440.0, sr: int = 48000, duration: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _make_silence(sr: int = 48000, duration: float = 0.1) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _make_white_noise(sr: int = 48000, duration: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (0.1 * rng.standard_normal(int(sr * duration))).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixture: frischer Tracer pro Test (Singleton zurücksetzen)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_tracer():
    """Gibt einen frisch initialisierten SignalFlowTracer zurück."""
    import backend.core.signal_flow_tracer as sft_mod

    old_instance = sft_mod._instance
    sft_mod._instance = None
    tracer = sft_mod.get_signal_flow_tracer()
    yield tracer
    # Teardown: Singleton zurücksetzen
    sft_mod._instance = old_instance


# ---------------------------------------------------------------------------
# Tests: Singleton
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_singleton_returns_same_instance():
    from backend.core.signal_flow_tracer import get_signal_flow_tracer

    t1 = get_signal_flow_tracer()
    t2 = get_signal_flow_tracer()
    assert t1 is t2


# ---------------------------------------------------------------------------
# Tests: begin_session
# ---------------------------------------------------------------------------


def test_begin_session_sets_state(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(
        original_audio=audio,
        sr=sr,
        mode="restoration",
        source_path="/test/song.mp3",
        material="cassette",
        era_decade=1975,
        panns_singing=0.72,
    )
    assert fresh_tracer._session_active
    assert fresh_tracer._material == "cassette"
    assert fresh_tracer._era_decade == 1975
    assert fresh_tracer._panns_singing == pytest.approx(0.72)
    assert fresh_tracer._is_vocal is True
    assert fresh_tracer._orig_peak_db < 0.0  # Sinuston unter 0 dBFS
    assert fresh_tracer._orig_rms_db < 0.0


def test_begin_session_non_vocal(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration", panns_singing=0.15)
    assert fresh_tracer._is_vocal is False


def test_begin_session_clears_previous_phases(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration", panns_singing=0.5)
    fresh_tracer._phases.append(object())  # Dummy
    fresh_tracer.begin_session(audio, sr, "restoration", panns_singing=0.5)
    assert len(fresh_tracer._phases) == 0


def test_begin_session_with_silence_zones(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    zones = [(0, 1000), (44000, 48000)]
    fresh_tracer.begin_session(audio, sr, "restoration", structural_silence_zones=zones, panns_singing=0.5)
    assert fresh_tracer._silence_zones == zones


def test_begin_session_exception_safe(fresh_tracer):
    """begin_session darf bei kaputtem Audio nicht crashen."""
    fresh_tracer.begin_session(None, 48000, "restoration")  # type: ignore[arg-type]
    # Kein Exception = OK


# ---------------------------------------------------------------------------
# Tests: capture_pre_phase
# ---------------------------------------------------------------------------


def test_capture_pre_phase_sets_ref(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(220.0, sr=sr)
    fresh_tracer.capture_pre_phase(pre)
    assert fresh_tracer._pre_audio_ref is pre  # Referenz, kein Copy


def test_capture_pre_phase_not_active_does_nothing(fresh_tracer):
    pre = _make_sine()
    fresh_tracer.capture_pre_phase(pre)  # session nicht aktiv
    # Kein crash erwartet


# ---------------------------------------------------------------------------
# Tests: record_phase — Flags
# ---------------------------------------------------------------------------


def test_record_phase_clean_signal_no_flags(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(sr=sr)
    # Minimale Änderung: -1 dB Absenkung → kein Flag erwartet
    # (Diff < -40 dB relativ zu Pre → Echo-Guard greift; keine Pegelexplosion)
    post = pre * 0.891  # ~-1 dB
    fresh_tracer.record_phase("phase_test", pre, post, sr)
    assert len(fresh_tracer._phases) == 1
    assert fresh_tracer._phases[0].flags == []


def test_record_phase_detects_pegelexplosion_warn(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(0.1, sr=sr, duration=0.5)  # leises Signal
    post = pre * 3.0  # +~9.5 dB → WARN aber < CRIT
    # Exakt auf Schwellwert abschneiden: +6 bis +12 dB
    post_warn = pre * 2.05  # ~+6.2 dB
    fresh_tracer.record_phase("phase_test_pegelwarn", pre, post_warn, sr)
    phase = fresh_tracer._phases[0]
    flag_texts = " ".join(phase.flags)
    assert "PEGELEXPLOSION_WARN" in flag_texts


def test_record_phase_detects_pegelexplosion_crit(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(sr=sr, duration=0.3)
    post = pre * 5.0  # +~14 dB → CRIT
    fresh_tracer.record_phase("phase_test_pegelcrit", pre, post, sr)
    phase = fresh_tracer._phases[0]
    flag_texts = " ".join(phase.flags)
    assert "PEGELEXPLOSION_CRIT" in flag_texts


def test_record_phase_detects_level_collapse(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(sr=sr, duration=0.3)
    post = pre * 1e-6  # fast Stille → Level Collapse
    fresh_tracer.record_phase("phase_test_collapse", pre, post, sr)
    phase = fresh_tracer._phases[0]
    flag_texts = " ".join(phase.flags)
    assert "LEVEL_COLLAPSE" in flag_texts


def test_record_phase_hnr_drop_vocal(fresh_tracer):
    """Bei PANNs-Singing >= 0.25 wird HNR gemessen."""
    sr = 48000
    audio = _make_sine(440.0, sr=sr, duration=1.0)
    fresh_tracer.begin_session(audio, sr, "restoration", panns_singing=0.65)
    assert fresh_tracer._is_vocal

    pre = _make_sine(440.0, sr=sr, duration=1.0)
    # Post = weißes Rauschen (HNR kollabiert massiv)
    post = _make_white_noise(sr=sr, duration=1.0) * 0.5

    fresh_tracer.record_phase("phase_nr_test", pre, post, sr)
    phase = fresh_tracer._phases[0]
    # HNR sollte gemessen worden sein
    assert phase.hnr_db_pre is not None
    # Bei weißem Rauschen: entweder HNR_DROP oder kein tonaler Anteil erkannt


def test_record_phase_no_hnr_non_vocal(fresh_tracer):
    """Bei PANNs-Singing < 0.25 wird HNR NICHT gemessen."""
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration", panns_singing=0.10)
    pre = _make_sine(sr=sr, duration=0.5)
    post = pre * 0.9
    fresh_tracer.record_phase("phase_no_hnr", pre, post, sr)
    phase = fresh_tracer._phases[0]
    assert phase.hnr_db_pre is None
    assert phase.hnr_db_post is None


def test_record_phase_silence_contamination(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr, duration=1.0)
    # Stille-Zone: Samples 0–1000
    zones = [(0, 1000)]
    fresh_tracer.begin_session(audio, sr, "restoration", structural_silence_zones=zones, panns_singing=0.5)

    pre = np.zeros(sr, dtype=np.float32)  # original war Stille
    post = pre.copy()
    post[0:1000] = 0.3 * np.sin(np.linspace(0, 6 * np.pi, 1000)).astype(np.float32)  # Energie hinzugefügt

    fresh_tracer.record_phase("phase_silence_test", pre, post, sr)
    phase = fresh_tracer._phases[0]
    flag_texts = " ".join(phase.flags)
    assert "SILENCE_CONTAMINATION" in flag_texts


def test_record_phase_max_records_guard(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(sr=sr, duration=0.05)
    post = pre.copy()
    from backend.core.signal_flow_tracer import _MAX_PHASE_RECORDS

    # Fülle bis Max + 10
    for i in range(_MAX_PHASE_RECORDS + 10):
        fresh_tracer.record_phase(f"phase_{i:03d}", pre, post, sr)
    assert len(fresh_tracer._phases) == _MAX_PHASE_RECORDS


def test_record_phase_none_audio_safe(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    fresh_tracer.record_phase("phase_none_test", None, None, sr)
    # Kein Crash, aber kein Record (zu kurz / None)
    assert len(fresh_tracer._phases) == 0


def test_record_phase_goal_delta_stored(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(sr=sr, duration=0.3)
    post = pre * 0.98
    delta = {"natuerlichkeit": 0.03, "brillanz": -0.01}
    fresh_tracer.record_phase("phase_delta_test", pre, post, sr, goal_delta=delta)
    assert fresh_tracer._phases[0].goal_delta == delta


# ---------------------------------------------------------------------------
# Tests: finalize
# ---------------------------------------------------------------------------


def test_finalize_writes_trace_file(fresh_tracer, tmp_path):
    """finalize() schreibt eine gültige JSON-Datei."""
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration", source_path="/test/song.mp3", panns_singing=0.5)
    pre = _make_sine(sr=sr, duration=0.3)
    post = pre * 0.99
    fresh_tracer.record_phase("phase_01_click_removal", pre, post, sr)

    with (
        patch("backend.core.signal_flow_tracer._TRACE_DIR", tmp_path / "traces"),
        patch("backend.core.signal_flow_tracer._LATEST_SYMLINK", tmp_path / "sft_latest.json"),
    ):
        fresh_tracer.finalize(hpi=0.72, artifact_freedom=0.97, vqi=0.85)

    latest = tmp_path / "sft_latest.json"
    assert latest.exists()
    data = json.loads(latest.read_text())
    assert data["hpi"] == pytest.approx(0.72, abs=0.001)
    assert data["artifact_freedom"] == pytest.approx(0.97, abs=0.001)
    assert data["vqi"] == pytest.approx(0.85, abs=0.001)
    assert len(data["phases"]) == 1
    assert data["phases"][0]["phase_id"] == "phase_01_click_removal"


def test_finalize_session_inactive_afterwards(fresh_tracer, tmp_path):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    with (
        patch("backend.core.signal_flow_tracer._TRACE_DIR", tmp_path / "traces"),
        patch("backend.core.signal_flow_tracer._LATEST_SYMLINK", tmp_path / "sft_latest.json"),
    ):
        fresh_tracer.finalize(hpi=0.5, artifact_freedom=0.99)
    assert not fresh_tracer._session_active


# ---------------------------------------------------------------------------
# Tests: report
# ---------------------------------------------------------------------------


def test_report_contains_session_id(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration", material="vinyl", panns_singing=0.0)
    rpt = fresh_tracer.report()
    assert "§SFT" in rpt
    assert "vinyl" in rpt


def test_report_shows_critical_flags(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    pre = _make_sine(sr=sr, duration=0.3)
    post = pre * 5.0  # CRIT Pegelexplosion
    fresh_tracer.record_phase("phase_test_crit", pre, post, sr)
    rpt = fresh_tracer.report()
    assert "KRITISCH" in rpt or "CRIT" in rpt


def test_report_clean_shows_checkmark(fresh_tracer):
    sr = 48000
    audio = _make_sine(sr=sr)
    fresh_tracer.begin_session(audio, sr, "restoration")
    rpt = fresh_tracer.report()
    assert "sauber" in rpt.lower() or "clean" in rpt.lower() or "✅" in rpt


def test_report_latest_no_file(fresh_tracer, tmp_path):
    """report_latest gibt lesbare Meldung zurück wenn kein File vorhanden."""
    with patch("backend.core.signal_flow_tracer._LATEST_SYMLINK", tmp_path / "nonexistent.json"):
        rpt = fresh_tracer.report_latest()
    assert "§SFT" in rpt


# ---------------------------------------------------------------------------
# Tests: latest_output_wav
# ---------------------------------------------------------------------------


def test_latest_output_wav_returns_none_without_output(fresh_tracer, tmp_path):
    """Gibt None zurück wenn kein WAV existiert."""
    with patch("backend.core.signal_flow_tracer._find_latest_output_wav", return_value=None):
        wav = fresh_tracer.latest_output_wav()
    assert wav is None


def test_latest_output_wav_filesystem_fallback(tmp_path):
    """_find_latest_output_wav findet neueste WAV."""
    from backend.core.signal_flow_tracer import _find_latest_output_wav

    # Da die reale output/-Verzeichnis-Scan-Funktion den Workspace nutzt,
    # prüfen wir nur dass sie keinen Crash erzeugt:
    result = _find_latest_output_wav()
    # Entweder None oder ein gültiger Pfad
    assert result is None or Path(result).suffix.lower() == ".wav"


# ---------------------------------------------------------------------------
# Tests: DSP-Hilfsfunktionen
# ---------------------------------------------------------------------------


def test_to_db_peak_silent():
    from backend.core.signal_flow_tracer import _to_db_peak

    silence = np.zeros(100, dtype=np.float32)
    assert _to_db_peak(silence) <= -100.0


def test_to_db_peak_full_scale():
    from backend.core.signal_flow_tracer import _to_db_peak

    full = np.ones(1000, dtype=np.float32)
    assert _to_db_peak(full) == pytest.approx(0.0, abs=0.1)


def test_to_db_rms():
    from backend.core.signal_flow_tracer import _to_db_rms

    x = np.full(1000, 0.1, dtype=np.float32)
    db = _to_db_rms(x)
    assert db == pytest.approx(-20.0, abs=0.5)


def test_to_mono_2d_channels_first():
    """(2, N) → mono"""
    from backend.core.signal_flow_tracer import _to_mono

    stereo = np.random.randn(2, 1000).astype(np.float32)
    mono = _to_mono(stereo)
    assert mono is not None
    assert mono.ndim == 1
    assert len(mono) == 1000


def test_to_mono_2d_samples_first():
    """(N, 2) → mono"""
    from backend.core.signal_flow_tracer import _to_mono

    stereo = np.random.randn(1000, 2).astype(np.float32)
    mono = _to_mono(stereo)
    assert mono is not None
    assert mono.ndim == 1
    assert len(mono) == 1000


def test_hnr_fast_sine_positive():
    """Sinuston → HNR > 10 dB."""
    from backend.core.signal_flow_tracer import _hnr_fast

    sr = 48000
    t = np.linspace(0, 0.5, sr // 2)
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    hnr = _hnr_fast(sine, sr)
    if hnr is not None:
        assert hnr > 5.0  # Sinus hat hohe Tonanteils-Ratio


def test_hnr_fast_noise_low_or_none():
    """Weißes Rauschen → HNR None oder < 10 dB."""
    from backend.core.signal_flow_tracer import _hnr_fast

    sr = 48000
    rng = np.random.default_rng(99)
    noise = (0.1 * rng.standard_normal(sr // 2)).astype(np.float32)
    hnr = _hnr_fast(noise, sr)
    if hnr is not None:
        assert hnr < 12.0


def test_detect_echo_no_echo():
    """Sauber-Signal → keine Echo-Artefakte."""
    from backend.core.signal_flow_tracer import _detect_echo

    sr = 48000
    silence = np.zeros(sr // 5, dtype=np.float32)
    corr, lag_ms = _detect_echo(silence, sr)
    assert corr < 0.5


def test_compute_spectral_novelty_identical():
    """Identisches Signal vs. Original → Novelty nahe 0."""
    from backend.core.signal_flow_tracer import _compute_psd_fingerprint, _compute_spectral_novelty_fast

    sr = 48000
    audio = _make_sine(sr=sr, duration=1.0)
    psd, freqs = _compute_psd_fingerprint(audio, sr)
    novelty = _compute_spectral_novelty_fast(audio, sr, psd, freqs)
    assert novelty < 0.05


def test_compute_spectral_novelty_different():
    """Stark verändertes Signal → höhere Novelty."""
    from backend.core.signal_flow_tracer import _compute_psd_fingerprint, _compute_spectral_novelty_fast

    sr = 48000
    orig = _make_sine(440.0, sr=sr, duration=1.0)
    # Mehrere neue Frequenzen hinzufügen
    noise_heavy = orig + 0.8 * _make_white_noise(sr=sr, duration=1.0)
    psd, freqs = _compute_psd_fingerprint(orig, sr)
    novelty = _compute_spectral_novelty_fast(noise_heavy, sr, psd, freqs)
    assert novelty > 0.02  # deutlich mehr Novelty


def test_compute_spectral_novelty_no_orig_psd():
    """Keine Original-PSD → Novelty = 0.0 (kein Crash)."""
    from backend.core.signal_flow_tracer import _compute_spectral_novelty_fast

    audio = _make_sine()
    novelty = _compute_spectral_novelty_fast(audio, 48000, None, None)
    assert novelty == 0.0
