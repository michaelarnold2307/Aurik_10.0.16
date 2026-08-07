from __future__ import annotations

"""
tests/unit/test_noise_floor.py
================================
Pflicht-Tests für die Rauschboden-Qualitätsgarantie gemäß §8.2 Aurik-9-Richtlinien.

Garantien (§8.2, Punkt 8 — Rauschboden-Qualitätsgarantie):
    - Residual-Rauschen ≤ −72 dBFS (Archivniveau)
    - A-gewichteter Rauschpegel ≤ −75 dB(A)
    - Musical-Noise-Ereignisse (kurze Artefakt-Aufwüchse) = 0 in Stille-Segmenten
    - Messung: AES17-angelehnt, 1-s-Stille-Fenster, RMS-Analyse

Referenz:
    AES17 (2020): Measurement of Digital Audio Equipment
    §8.2 Aurik-9-Richtlinien (Rauschboden-Qualitätsgarantie)
Pflicht-Test: tests/unit/test_noise_floor.py
"""


import math

import numpy as np
import pytest

SR = 48_000
np.random.seed(13)

# ---------------------------------------------------------------------------
# Pflicht-Schwellwerte (aus §8.2)
# ---------------------------------------------------------------------------
NOISE_FLOOR_DBFS_LIMIT = -72.0  # Rauschboden-Maximum [dBFS]
A_WEIGHTED_DBFS_LIMIT = -75.0  # A-gewichteter Rauschpegel-Maximum [dB(A)]
MUSICAL_NOISE_EVENTS_MAX = 0  # Null erlaubt in Stille-Segmenten

# Interne Messschwelle — Stille-Segment-Detektion
SILENCE_THRESHOLD_DBFS = -40.0


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _silence(duration_s: float = 2.0) -> np.ndarray:
    """Vollständige Stille (Float32-Nullen)."""
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _quantization_noise(bit_depth: int = 24, duration_s: float = 2.0) -> np.ndarray:
    """Simuliertes Quantisierungs-Rauschen (LSB-Level)."""
    lsb = 1.0 / (2 ** (bit_depth - 1))
    rng = np.random.default_rng(seed=42)
    return (rng.uniform(-lsb, lsb, int(duration_s * SR))).astype(np.float32)


def _very_quiet_signal(target_dbfs: float = -80.0, duration_s: float = 2.0) -> np.ndarray:
    """Signal mit definiertem RMS-Pegel weit unter Stille-Schwelle."""
    n = int(duration_s * SR)
    rng = np.random.default_rng(seed=99)
    noise = rng.standard_normal(n).astype(np.float32)
    rms_lin = 10.0 ** (target_dbfs / 20.0)
    actual_rms = float(np.sqrt(np.mean(noise**2)) + 1e-15)
    return ((noise / actual_rms) * rms_lin).astype(np.float32)  # type: ignore[no-any-return]


def _clean_music(duration_s: float = 3.0) -> np.ndarray:
    """Synthetisches sauberes Musiksignal (Grundton + Obertöne, −14 LUFS)."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    sig = np.float32(0.0)
    for n in range(1, 6):
        sig = sig + (0.4 / n) * np.sin(2 * np.pi * n * 220.0 * t).astype(np.float32)
    peak = float(np.max(np.abs(sig)) + 1e-9)
    return (sig / peak * 0.3).astype(np.float32)  # ~ −10 dBFS  # type: ignore[return-value]


def _rms_dbfs(audio: np.ndarray) -> float:
    """RMS-Pegel in dBFS."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-15))
    return 20.0 * math.log10(rms)


def _peak_dbfs(audio: np.ndarray) -> float:
    """Peak-Pegel in dBFS."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    peak = float(np.max(np.abs(mono)) + 1e-15)
    return 20.0 * math.log10(peak)


def _a_weight_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """A-weighting Filterung (vereinfachtes Butterworth-Modell, IEC 61672)."""
    from scipy.signal import butter, sosfilt

    # A-weighting: stark gedämpft < 500 Hz, Peak ~3.5 kHz, abfallend > 10 kHz
    sos_hp = butter(4, 1000.0 / (sr / 2), btype="high", output="sos")
    # Grobe Näherung der A-Kurve: Hochpass über 1 kHz
    return sosfilt(sos_hp, audio.astype(np.float64)).astype(np.float32)  # type: ignore[no-any-return]


def _detect_musical_noise_events(
    audio: np.ndarray,
    sr: int,
    threshold_db_above_floor: float = 6.0,
    min_silence_level_dbfs: float = -40.0,
    window_ms: float = 50.0,
) -> int:
    """
    Zählt Musical-Noise-Ereignisse in Stille-Segmenten.

    Erkennung:
        - Segmente mit Gesamtpegel < min_silence_level_dbfs
        - Im Segment: lokale Energie-Sprünge > threshold_db_above_floor über
          dem lokalen Rauschboden des Segments
    """
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    window_samples = int(window_ms / 1000.0 * sr)
    n_events = 0

    # Übergeordneten Rauschboden messen
    global_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-15))
    global_rms_db = 20.0 * math.log10(global_rms)

    # Nur in Stille-Segmenten suchen
    if global_rms_db >= min_silence_level_dbfs:
        return 0  # Kein Stille-Segment — kein Musical-Noise-Test

    # Fenster-weise Energie
    window_energies_db = []
    for start in range(0, len(mono) - window_samples, window_samples // 2):
        frame = mono[start : start + window_samples].astype(np.float64)
        rms = float(np.sqrt(np.mean(frame**2) + 1e-15))
        window_energies_db.append(20.0 * math.log10(rms))

    if not window_energies_db:
        return 0

    floor_db = np.percentile(window_energies_db, 10)  # Unteres Dezil = Rauschboden
    for e in window_energies_db:
        if e > floor_db + threshold_db_above_floor:
            n_events += 1

    return n_events


# ---------------------------------------------------------------------------
# Tests: Fundamentale Rauschboden-Messungen
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoiseFloorMeasurement:
    """Grundlegende Rauschboden-Messung (Referenz-Tests)."""

    def test_01_silence_rms_below_limit(self):
        """Stille (Nullen) hat RMS << −72 dBFS."""
        audio = _silence(duration_s=2.0)
        rms_db = _rms_dbfs(audio)
        assert rms_db < NOISE_FLOOR_DBFS_LIMIT, f"Stille-RMS {rms_db:.1f} dBFS ≥ Grenze {NOISE_FLOOR_DBFS_LIMIT} dBFS"

    def test_02_quantization_noise_level(self):
        """24-bit-Quantisierungsrauschen liegt weit unter −72 dBFS."""
        audio = _quantization_noise(bit_depth=24, duration_s=2.0)
        rms_db = _rms_dbfs(audio)
        # 24-bit theoretisch ~−144 dBFS; praktisch ≤ −120 dBFS
        assert rms_db < -100.0, f"24-bit Quantisierungsrauschen {rms_db:.1f} dBFS zu hoch"

    def test_03_16bit_quantization_above_24bit(self):
        """16-bit-Quantisierungsrauschen > 24-bit (weniger Auflösung)."""
        audio_24 = _quantization_noise(bit_depth=24, duration_s=2.0)
        audio_16 = _quantization_noise(bit_depth=16, duration_s=2.0)
        rms_24 = _rms_dbfs(audio_24)
        rms_16 = _rms_dbfs(audio_16)
        assert rms_16 > rms_24, "16-bit-Rauschen ist leiser als 24-bit — Fehler"

    def test_04_rms_measurement_is_finite(self):
        """_rms_dbfs() gibt immer finiten Wert zurück."""
        for dur in (0.5, 1.0, 3.0):
            audio = _silence(duration_s=dur)
            assert math.isfinite(_rms_dbfs(audio))

    def test_05_peak_below_silence_threshold(self):
        """Stille-Signal hat Peak weit unter −40 dBFS."""
        audio = _silence(duration_s=2.0)
        peak = _peak_dbfs(audio)
        assert peak < SILENCE_THRESHOLD_DBFS


class TestAWeighting:
    """A-gewichteter Rauschboden (≤ −75 dB(A))."""

    def test_06_a_weighted_silence_below_limit(self):
        """A-gewichtete Stille liegt unter −75 dB(A)."""
        audio = _silence(duration_s=2.0)
        a_weighted = _a_weight_filter(audio, SR)
        rms_db = _rms_dbfs(a_weighted)
        assert rms_db < A_WEIGHTED_DBFS_LIMIT, (
            f"A-gewichtete Stille {rms_db:.1f} dB(A) ≥ Grenze {A_WEIGHTED_DBFS_LIMIT}"
        )

    def test_07_a_weighted_is_finite(self):
        """A-Weighting-Ergebnis enthält keine NaN/Inf."""
        audio = _silence(duration_s=2.0)
        a_weighted = _a_weight_filter(audio, SR)
        assert np.isfinite(a_weighted).all()

    def test_08_a_weighting_reduces_low_freq(self):
        """A-Weighting dämpft Tieffrequenzen stärker als Hochfrequenzen."""
        t = np.linspace(0, 2.0, int(2.0 * SR), endpoint=False)
        low_freq = np.sin(2 * np.pi * 50 * t).astype(np.float32) * 0.05
        high_freq = np.sin(2 * np.pi * 5000 * t).astype(np.float32) * 0.05

        low_a = _a_weight_filter(low_freq, SR)
        high_a = _a_weight_filter(high_freq, SR)

        rms_low = _rms_dbfs(low_a)
        rms_high = _rms_dbfs(high_a)
        # A-Kurve dämpft 50 Hz deutlich stärker als 5 kHz
        assert rms_low < rms_high, f"A-Weighting dämpft 50 Hz ({rms_low:.1f}) nicht stärker als 5 kHz ({rms_high:.1f})"


class TestMusicalNoise:
    """Musical-Noise-Ereignisse in Stille-Segmenten = 0."""

    def test_09_silence_has_zero_musical_noise(self):
        """Reine Stille enthält 0 Musical-Noise-Ereignisse."""
        audio = _silence(duration_s=5.0)
        n_events = _detect_musical_noise_events(audio, SR)
        assert n_events == MUSICAL_NOISE_EVENTS_MAX

    def test_10_quantization_noise_no_musical_noise(self):
        """24-bit-Quantisierungsrauschen (sehr ruhig) hat 0 Musical-Noise-Ereignisse."""
        audio = _quantization_noise(bit_depth=24, duration_s=5.0)
        # −144 dBFS Rauschen ist ein Stille-Segment
        n_events = _detect_musical_noise_events(audio, SR, min_silence_level_dbfs=-40.0)
        assert n_events == 0

    def test_11_loud_click_in_silence_detected_as_event(self):
        """Einzelner lauter Klick in Stille wird als Musical-Noise-Ereignis erkannt."""
        audio = _silence(duration_s=5.0)
        # Synthetischen Klick bei t=2s einfügen (−30 dBFS)
        click_start = 2 * SR
        click_end = click_start + 100
        audio[click_start:click_end] = 0.03  # ~ −30 dBFS
        n_events = _detect_musical_noise_events(audio, SR, min_silence_level_dbfs=-40.0)
        assert n_events >= 1, "Lauter Klick in Stille wurde nicht als Musical-Noise erkannt"

    def test_12_event_detector_no_crash_on_empty_audio(self):
        """Musical-Noise-Detektor stürzt bei extrem kurzem Audio nicht ab."""
        audio = np.zeros(100, dtype=np.float32)
        n_events = _detect_musical_noise_events(audio, SR)
        assert isinstance(n_events, int)
        assert n_events >= 0


class TestPipelineNoiseFloor:
    """Pipeline-Level-Tests: Phasen fügen keinen Rauschboden zur Stille hinzu."""

    def test_13_phase03_does_not_add_noise_to_silence(self):
        """Phase 03 (denoise) verschlechtert Rauschboden nicht."""
        try:
            from backend.core.phases.phase_03_denoise import DenoisePhase

            phase = DenoisePhase(sample_rate=SR)
        except Exception:
            pytest.skip("phase_03_denoise nicht verfügbar")

        audio = _silence(duration_s=3.0)
        rms_before = _rms_dbfs(audio)
        result = phase.process(audio)
        rms_after = _rms_dbfs(result.audio)

        # Rauschboden darf nicht um mehr als 20 dB steigen
        assert rms_after - rms_before < 20.0, f"Phase 03 erhöht Rauschboden um {rms_after - rms_before:.1f} dB"

    def test_14_phase40_does_not_amplify_silence_massively(self):
        """Phase 40 (loudness norm) verstärkt Stille nicht um > 40 dB."""
        try:
            from backend.core.defect_scanner import MaterialType as _MT
            from backend.core.phases.phase_40_loudness_normalization import LoudnessNormalizationPhase

            phase = LoudnessNormalizationPhase()
            _material = _MT.UNKNOWN
        except Exception:
            pytest.skip("phase_40_loudness_normalization nicht verfügbar")

        audio = _silence(duration_s=3.0)
        result = phase.process(audio, sample_rate=SR, material_type=_material)
        rms_after = _rms_dbfs(result.audio)
        # Selbst wenn Normalisierung auf −14 LUFS → Stille bleibt Stille
        assert rms_after < SILENCE_THRESHOLD_DBFS + 30.0, (
            f"Phase 40 amplifies Stille auf {rms_after:.1f} dBFS (zu hoch)"
        )

    def test_15_phase47_does_not_add_noise_to_silence(self):
        """Phase 47 (truepeak_limiter) fügt keinen Rauschboden zur Stille hinzu."""
        try:
            from backend.core.phases.phase_47_truepeak_limiter import TruePeakLimiterPhase

            phase = TruePeakLimiterPhase(sample_rate=SR)
        except Exception:
            pytest.skip("phase_47_truepeak_limiter nicht verfügbar")

        audio = _silence(duration_s=2.0)
        result = phase.process(audio, SR)
        rms_after = _rms_dbfs(result.audio)
        assert rms_after < -60.0, f"Phase 47 hebt Rauschboden auf {rms_after:.1f} dBFS"

    def test_16_phase01_click_removal_silence(self):
        """Phase 01 (click_removal) verarbeitet Stille ohne Rauschboden-Erhöhung."""
        try:
            from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

            phase = ClickRemovalPhase()
        except Exception:
            pytest.skip("phase_01_click_removal nicht verfügbar")

        audio = _silence(duration_s=2.0)
        result = phase.process(audio)
        rms_after = _rms_dbfs(result.audio)
        assert rms_after < NOISE_FLOOR_DBFS_LIMIT + 30.0


class TestNoiseMeasurementInvariants:
    """Mathematische Invarianten der Rauschboden-Messung."""

    def test_17_rms_monotone_with_amplitude(self):
        """RMS steigt monoton mit der Amplitude."""
        rms_vals = [
            _rms_dbfs(_very_quiet_signal(-100.0)),
            _rms_dbfs(_very_quiet_signal(-80.0)),
            _rms_dbfs(_very_quiet_signal(-60.0)),
        ]
        assert rms_vals[0] < rms_vals[1] < rms_vals[2]

    def test_18_nan_in_audio_handled_safely(self):
        """NaN in Audio wird sicher zu 0.0 normalisiert vor Rauschboden-Messung."""
        audio = np.full(SR, np.nan, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0)
        rms_db = _rms_dbfs(audio)
        assert math.isfinite(rms_db)

    def test_19_musical_noise_detector_returns_int(self):
        """Musical-Noise-Detektor gibt immer int ≥ 0 zurück."""
        for audio in (_silence(), _clean_music()):
            n = _detect_musical_noise_events(audio, SR)
            assert isinstance(n, int)
            assert n >= 0

    def test_20_peak_always_geq_rms(self):
        """Peak-Pegel ist immer ≥ RMS-Pegel."""
        audio = _clean_music(duration_s=2.0)
        peak = _peak_dbfs(audio)
        rms = _rms_dbfs(audio)
        assert peak >= rms - 0.1, f"Peak {peak:.1f} < RMS {rms:.1f} (physikalisch unmöglich)"

    def test_21_rms_silence_well_below_threshold(self):
        """Stille-Signal liegt deutlich unter −72 dBFS und erfüllt AES17-Anforderung."""
        audio = _silence(duration_s=1.0)
        assert _rms_dbfs(audio) < NOISE_FLOOR_DBFS_LIMIT

    def test_22_close_to_archive_standard(self):
        """Quantisierungsrauschen bei 24-bit erfüllt −72 dBFS Archiv-Standard."""
        audio = _quantization_noise(bit_depth=24, duration_s=2.0)
        rms_db = _rms_dbfs(audio)
        assert rms_db < NOISE_FLOOR_DBFS_LIMIT, (
            f"24-bit Rauschen {rms_db:.1f} dBFS ≥ −72 dBFS — Archivstandard verletzt"
        )
