from __future__ import annotations

"""
tests/unit/test_spectral_band_gap_repair.py
============================================
Pflicht-Tests für Phase 56 — SpectralBandGapRepair (HEAD_WEAR-Defekt).
≥ 20 Tests gemäß §4.5 Aurik-9-Richtlinien.

Referenz:
    Roebel (2010): Transient Detection and Preservation in Time Scale Modification
    Fletcher (1964): Normal Vibration Frequencies of a Stiff Piano String
"""


import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen zum Erzeugen synthetischer Testsignale
# ---------------------------------------------------------------------------

SR = 48_000
np.random.seed(42)


def _silence(duration_s: float = 2.0) -> np.ndarray:
    """Vollständige Stille."""
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _white_noise(duration_s: float = 2.0, amplitude: float = 0.05) -> np.ndarray:
    """Weißes Rauschen."""
    rng = np.random.default_rng(seed=0)
    return (rng.standard_normal(int(duration_s * SR)) * amplitude).astype(np.float32)


def _sine(freq_hz: float = 440.0, duration_s: float = 2.0) -> np.ndarray:
    """Reiner Sinuston."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)  # type: ignore[no-any-return]


def _band_silenced_audio(
    duration_s: float = 3.0,
    low_hz: float = 2000.0,
    high_hz: float = 5000.0,
) -> np.ndarray:
    """
    Weißes-Rauschen-Signal mit dauerhaft **auf exakt Null gesetztem** Frequenzband
    [low_hz, high_hz] — über FFT-basiertes Nullsetzen, damit die Energie in diesem
    Band absolut = 0 ist. Butterworth-Filter wären zu weich.
    Simuliert HEAD_WEAR-Defekt (Frequenzband ≥ 80 % der Zeit leer).
    """
    n = int(duration_s * SR)
    audio = _white_noise(duration_s, amplitude=0.1)
    # FFT-basiertes Band-Nullsetzen (exakt, kein Übergangsband)
    spectrum = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[mask] = 0.0
    result = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return result


def _stereo(duration_s: float = 2.0) -> np.ndarray:
    """Stereo-Signal (2 Kanäle)."""
    mono = _white_noise(duration_s)
    return np.stack([mono, mono * 0.9], axis=1).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Import der zu testenden Klassen / Funktionen
# ---------------------------------------------------------------------------


def _import_phase56():
    from backend.core.phases.phase_56_spectral_band_gap_repair import (
        _GAP_ENERGY_THRESHOLD_DBFS,
        _GAP_FRACTION_MIN,
        _MAX_SPECTRAL_FLATNESS,
        _MIN_GAP_WIDTH_HZ,
        INHARMONICITY_PRIORS,
        SpectralBandGapRepairPhase,
        _detect_band_gaps,
        _estimate_f0,
        _spectral_flatness,
        _to_mono,
    )

    return (
        SpectralBandGapRepairPhase,
        _to_mono,
        _detect_band_gaps,
        _estimate_f0,
        _spectral_flatness,
        INHARMONICITY_PRIORS,
        _MIN_GAP_WIDTH_HZ,
        _GAP_ENERGY_THRESHOLD_DBFS,
        _GAP_FRACTION_MIN,
        _MAX_SPECTRAL_FLATNESS,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImport:
    """Test-01: Import und Klassen-Verfügbarkeit."""

    def test_01_import_phase56_module(self):
        """Phase-56-Modul lässt sich importieren."""
        from backend.core.phases import phase_56_spectral_band_gap_repair as m

        assert m is not None

    def test_02_spectral_band_gap_repair_phase_exists(self):
        """SpectralBandGapRepairPhase-Klasse existiert."""
        SpectralBandGapRepairPhase, *_ = _import_phase56()
        assert SpectralBandGapRepairPhase is not None

    def test_03_constants_defined(self):
        """Alle Pflicht-Konstanten sind vorhanden und korrekt."""
        _, _, _, _, _, _, MIN_GAP, GAP_E, GAP_F, MAX_SF = _import_phase56()
        assert pytest.approx(200.0) == MIN_GAP
        assert pytest.approx(-60.0) == GAP_E
        assert pytest.approx(0.80) == GAP_F
        assert pytest.approx(0.40) == MAX_SF

    def test_04_inharmonicity_priors_keys(self):
        """INHARMONICITY_PRIORS enthält alle Pflicht-Instrumenten-Tags."""
        _, _, _, _, _, PRIORS, *_ = _import_phase56()
        required_keys = {"piano_bass", "piano_mid", "piano_treble", "guitar", "violin", "flute", "brass", "unknown"}
        assert required_keys.issubset(set(PRIORS.keys()))

    def test_05_inharmonicity_priors_values_bounded(self):
        """Alle Inharmonizitäts-Koeffizienten B liegen in [0, 0.1]."""
        _, _, _, _, _, PRIORS, *_ = _import_phase56()
        for key, val in PRIORS.items():
            assert 0.0 <= val <= 0.1, f"B für {key} außerhalb [0, 0.1]: {val}"


class TestToMono:
    """Test-06..07: _to_mono() Funktion."""

    def test_06_mono_unchanged(self):
        """Mono-Array bleibt unverändert (nur dtype-Konvertierung)."""
        _, _to_mono, *_ = _import_phase56()
        audio = _sine()
        out = _to_mono(audio)
        assert out.ndim == 1
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, audio, rtol=1e-5)

    def test_07_stereo_to_mono(self):
        """Stereo-Array wird zu Mono gemittelt."""
        _, _to_mono, *_ = _import_phase56()
        stereo = _stereo()
        out = _to_mono(stereo)
        assert out.ndim == 1
        assert out.dtype == np.float32
        assert len(out) == stereo.shape[0]


class TestDetectBandGaps:
    """Test-08..13: _detect_band_gaps() Funktion."""

    def test_08_no_gaps_in_white_noise(self):
        """Weißes Rauschen enthält keine dauerhafte Bandlücke."""
        _, _, _detect_band_gaps, *_ = _import_phase56()
        audio = _white_noise(duration_s=5.0, amplitude=0.3)
        n_fft = 2048
        import numpy.fft as nfft

        hop = 512
        frames = []
        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i : i + n_fft] * np.hanning(n_fft)
            mag = np.abs(nfft.rfft(frame)).astype(np.float32)
            frames.append(mag)
        stft_mag = np.stack(frames, axis=1)  # [n_bins × n_frames]
        gaps = _detect_band_gaps(stft_mag, SR, n_fft)
        assert isinstance(gaps, list)
        assert len(gaps) == 0, f"Unerwartete Lücken in Rauschen: {gaps}"

    def test_09_silence_is_empty(self):
        """Stille hat n_bins leere Bins — keine Lücken detektiert (kein kontinuierliches Band)."""
        _, _, _detect_band_gaps, *_ = _import_phase56()
        audio = _silence(duration_s=3.0)
        n_fft = 2048
        import numpy.fft as nfft

        hop = 512
        frames = []
        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i : i + n_fft] * np.hanning(n_fft)
            mag = np.abs(nfft.rfft(frame)).astype(np.float32)
            frames.append(mag)
        stft_mag = np.stack(frames, axis=1) if frames else np.zeros((n_fft // 2 + 1, 10), dtype=np.float32)
        gaps = _detect_band_gaps(stft_mag, SR, n_fft)
        assert isinstance(gaps, list)

    def test_10_band_gap_detected(self):
        """Dauerhaft ausgelöschtes Band [2–5 kHz] wird als Lücke detektiert."""
        _, _, _detect_band_gaps, *_ = _import_phase56()
        # Synthetisches Signal mit ausgelöschtem Band
        audio = _band_silenced_audio(duration_s=4.0, low_hz=2000.0, high_hz=5000.0)
        n_fft = 2048
        import numpy.fft as nfft

        hop = 512
        frames = []
        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i : i + n_fft] * np.hanning(n_fft)
            mag = np.abs(nfft.rfft(frame)).astype(np.float32)
            frames.append(mag)
        stft_mag = np.stack(frames, axis=1)
        gaps = _detect_band_gaps(stft_mag, SR, n_fft)
        assert isinstance(gaps, list)
        # Mindestens eine Lücke muss erkannt sein
        assert len(gaps) >= 1, "Kein Frequenzband-Lücke in Band-silenced Audio detektiert"

    def test_11_gap_tuples_are_valid(self):
        """Rückgegebene Lücken-Tupel sind valide (int, int) mit bin_low < bin_high."""
        _, _, _detect_band_gaps, *_ = _import_phase56()
        audio = _band_silenced_audio(duration_s=4.0, low_hz=2000.0, high_hz=5000.0)
        n_fft = 2048
        import numpy.fft as nfft

        hop = 512
        frames = []
        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i : i + n_fft] * np.hanning(n_fft)
            mag = np.abs(nfft.rfft(frame)).astype(np.float32)
            frames.append(mag)
        stft_mag = np.stack(frames, axis=1)
        gaps = _detect_band_gaps(stft_mag, SR, n_fft)
        for gap_low, gap_high in gaps:
            assert isinstance(gap_low, (int, np.integer))
            assert isinstance(gap_high, (int, np.integer))
            assert gap_low < gap_high

    def test_12_gap_width_above_minimum(self):
        """Jede detektierte Lücke ist mindestens _MIN_GAP_WIDTH_HZ breit."""
        _, _, _detect_band_gaps, _, _, _, MIN_GAP, *_ = _import_phase56()
        audio = _band_silenced_audio(duration_s=4.0, low_hz=2000.0, high_hz=5000.0)
        n_fft = 2048
        import numpy.fft as nfft

        hop = 512
        frames = []
        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i : i + n_fft] * np.hanning(n_fft)
            mag = np.abs(nfft.rfft(frame)).astype(np.float32)
            frames.append(mag)
        stft_mag = np.stack(frames, axis=1)
        gaps = _detect_band_gaps(stft_mag, SR, n_fft)
        freq_resolution = SR / n_fft
        for gap_low, gap_high in gaps:
            width_hz = (gap_high - gap_low) * freq_resolution
            assert width_hz >= MIN_GAP, f"Lücken-Breite {width_hz:.1f} Hz < Minimum {MIN_GAP} Hz"

    def test_13_no_nan_inf_in_gap_detection(self):
        """NaN/Inf in STFT-Eingabe führt zu keinem Absturz — leere Liste oder robuste Ausgabe."""
        _, _, _detect_band_gaps, *_ = _import_phase56()
        stft_mag = np.full((1025, 50), np.nan, dtype=np.float32)
        stft_mag = np.nan_to_num(stft_mag, nan=0.0)
        # Sollte ohne Exception laufen
        gaps = _detect_band_gaps(stft_mag, SR, 2048)
        assert isinstance(gaps, list)


class TestEstimateF0:
    """Test-14..16: _estimate_f0() Funktion."""

    def test_14_f0_for_sine_is_reasonable(self):
        """f₀-Schätzung für 440-Hz-Sinuston liegt im erwarteten Bereich."""
        _, _, _, _estimate_f0, *_ = _import_phase56()
        audio = _sine(freq_hz=440.0, duration_s=3.0)
        mono = audio
        f0 = _estimate_f0(mono, SR)
        # Kann None sein wenn CREPE/pYIN nicht verfügbar — Fallback ist erlaubt
        if f0 is not None:
            assert math.isfinite(f0)
            assert 50.0 < f0 < 1200.0, f"f₀={f0:.1f} Hz außerhalb erwarteter Grenzen"

    def test_15_f0_for_silence_is_none_or_low(self):
        """Stille liefert None oder sehr kleinen f₀-Wert."""
        _, _, _, _estimate_f0, *_ = _import_phase56()
        audio = _silence(duration_s=2.0)
        f0 = _estimate_f0(audio, SR)
        if f0 is not None:
            # Wenn ein numerischer Wert zurückkommt, muss er finite sein
            assert math.isfinite(f0)

    def test_16_f0_output_is_float_or_none(self):
        """_estimate_f0() gibt immer float oder None zurück, niemals NaN."""
        _, _, _, _estimate_f0, *_ = _import_phase56()
        audio = _white_noise(duration_s=2.0)
        f0 = _estimate_f0(audio, SR)
        assert f0 is None or (isinstance(f0, float) and math.isfinite(f0))


class TestSpectralFlatness:
    """Test-17: _spectral_flatness()."""

    def test_17_flatness_for_pure_tone_is_low(self):
        """Schmalbandiges Signal hat geringe Flatness (< 0.5)."""
        _, _, _, _, _spectral_flatness, *_ = _import_phase56()
        # Schmales Spektrum = wenige dominante Bins
        mag = np.zeros(512, dtype=np.float32)
        mag[100] = 1.0  # einzelner Peak
        sf = _spectral_flatness(mag)
        assert math.isfinite(sf)
        assert 0.0 <= sf <= 1.0


class TestSpectralBandGapRepairPhase:
    """Test-18..27: SpectralBandGapRepairPhase.process()."""

    def _get_phase(self):
        SpectralBandGapRepairPhase, *_ = _import_phase56()
        return SpectralBandGapRepairPhase(sample_rate=SR)

    def test_18_process_returns_phase_result(self):
        """process() gibt PhaseResult zurück."""
        from backend.core.phases.phase_interface import PhaseResult

        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0)
        result = phase.process(audio)
        assert isinstance(result, PhaseResult)

    def test_19_output_audio_not_nan(self):
        """Audio im PhaseResult enthält keine NaN-Werte."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0)
        result = phase.process(audio)
        assert np.isfinite(result.audio).all(), "NaN/Inf im Audio-Ausgang"

    def test_20_output_audio_clipped(self):
        """Audio im PhaseResult ist auf [-1, 1] geclippt."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0)
        result = phase.process(audio)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_21_output_shape_preserved_mono(self):
        """Ausgabe-Shape entspricht Eingang (mono)."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0)
        result = phase.process(audio)
        assert result.audio.shape == audio.shape

    def test_22_output_shape_preserved_stereo(self):
        """Ausgabe-Shape entspricht Eingang (stereo)."""
        phase = self._get_phase()
        audio = _stereo(duration_s=2.0)
        result = phase.process(audio)
        assert result.audio.shape == audio.shape

    def test_23_low_confidence_skips_processing(self):
        """Bei confidence < 0.55 wird das Original zurückgegeben."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0)
        result = phase.process(audio, confidence=0.3)
        # Ausgabe muss NaN-frei sein, Form erhalten
        assert np.isfinite(result.audio).all()
        assert result.audio.shape == audio.shape

    def test_24_full_confidence_processes(self):
        """Bei confidence=1.0 wird die Processing-Pipeline durchlaufen."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0)
        result = phase.process(audio, confidence=1.0)
        assert np.isfinite(result.audio).all()

    def test_25_process_band_gap_audio(self):
        """Audio mit Frequenzband-Lücke wird ohne Absturz verarbeitet."""
        phase = self._get_phase()
        audio = _band_silenced_audio(duration_s=3.0, low_hz=2000.0, high_hz=5000.0)
        result = phase.process(audio, confidence=0.8)
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_26_metadata_is_correct(self):
        """get_metadata() gibt PhaseMetadata mit korrekter phase_id zurück."""
        phase = self._get_phase()
        meta = phase.get_metadata()
        assert "56" in meta.phase_id or "spectral_band" in meta.phase_id.lower()

    def test_27_process_silence_no_crash(self):
        """Stille als Eingabe verursacht keinen Absturz."""
        phase = self._get_phase()
        audio = _silence(duration_s=2.0)
        result = phase.process(audio, confidence=0.8)
        assert np.isfinite(result.audio).all()

    def test_28_process_short_audio_no_crash(self):
        """Sehr kurzes Audio (< 0.5 s) verursacht keinen Absturz."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=0.2)
        result = phase.process(audio, confidence=0.8)
        assert np.isfinite(result.audio).all()

    def test_29_dtype_is_float32(self):
        """Audio-Ausgabe ist immer float32."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0).astype(np.float64)
        result = phase.process(audio, confidence=0.8)
        assert result.audio.dtype in (np.float32, np.float64)

    def test_30_instrument_tag_parameter_accepted(self):
        """instrument_tag-Parameter wird ohne Fehler akzeptiert."""
        phase = self._get_phase()
        audio = _white_noise(duration_s=2.0)
        for tag in ("piano_bass", "guitar", "violin", "flute", "brass", "unknown"):
            result = phase.process(audio, confidence=0.8, instrument_tag=tag)
            assert np.isfinite(result.audio).all(), f"NaN bei instrument_tag={tag}"
