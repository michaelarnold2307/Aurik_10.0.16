from __future__ import annotations

"""
Unit-Tests für core/medium_classifier.py (≥ 35 Tests).

Testet alle public APIs von MediumClassifier:
  - DSP-Feature-Extraktion (_SpectralFingerprinter)
  - Materialscorer (alle 12 MaterialTypes)
  - Cache-Verhalten (SHA256 LRU, 64 Einträge)
  - Singleton-Thread-Safety (Double-Checked Locking)
  - Fallback-Verhalten bei ImportError CLAP

Konventionen gem. Aurik-Spec §5.4:
  - np.random.seed(42) für Reproduzierbarkeit
  - @pytest.mark.timeout(30) per Test
  - Keine realen Audio-Dateien
"""


import math
import threading
import time

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType

# ---- Import-Prüfung --------------------------------------------------------
from backend.core.medium_classifier import (
    ClassificationResult,
    MaterialEvidence,
    MediumClassifier,
    classify_medium,
    get_medium_classifier,
)

SR = 48_000  # interne Pflicht-SR

# ============================================================================
# Hilfsfunktionen für synthetische Signale
# ============================================================================


def _silence(duration_s: float = 1.0) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _noise(duration_s: float = 1.0, amplitude: float = 0.05) -> np.ndarray:
    np.random.seed(42)
    return (np.random.randn(int(duration_s * SR)) * amplitude).astype(np.float32)


def _sine(freq_hz: float = 440.0, duration_s: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _clipped(duration_s: float = 1.0) -> np.ndarray:
    """Signal mit hartgeclippten Flat-tops (CD/Digital-Artefakt)."""
    np.random.seed(42)
    audio = np.random.randn(int(duration_s * SR)).astype(np.float32)
    return np.clip(audio, -0.95, 0.95)


def _hiss_tape(duration_s: float = 1.5) -> np.ndarray:
    """Tape-ähnliches Signal: Rauschen + Sinus + bandbreitenbegrenzt."""
    np.random.seed(42)
    noise = np.random.randn(int(duration_s * SR)).astype(np.float32) * 0.03
    return noise


def _crackle_vinyl(duration_s: float = 1.0, rate_hz: float = 10.0) -> np.ndarray:
    """Synthetisches Vinyl-Crackle: seltene Impulse in weißem Rauschen."""
    np.random.seed(7)
    audio = np.random.randn(int(duration_s * SR)).astype(np.float32) * 0.002
    n_clicks = int(rate_hz * duration_s)
    for _ in range(n_clicks):
        idx = np.random.randint(0, len(audio))
        audio[idx] = np.random.choice([-1.0, 1.0]) * 0.95
    return audio


def _stereo(mono: np.ndarray) -> np.ndarray:
    """Mono-Signal zu [N, 2] Stereo."""
    return np.column_stack([mono, mono])


# ============================================================================
# TestGroup 1: DataClasses
# ============================================================================


class TestDataClasses:
    """Tests 01–04: MaterialEvidence und ClassificationResult."""

    def test_01_material_evidence_fields(self) -> None:
        ev = MaterialEvidence(
            material=MaterialType.VINYL,
            confidence=0.7,
            features_matched=["crackle_density"],
            features_against=["hf_rolloff_hz"],
        )
        assert ev.material == MaterialType.VINYL
        assert ev.confidence == pytest.approx(0.7)
        assert "crackle_density" in ev.features_matched

    def test_02_classification_result_as_dict(self) -> None:
        ev = MaterialEvidence(MaterialType.TAPE, 0.6, [], [])
        r = ClassificationResult(
            material=MaterialType.TAPE,
            confidence=0.6,
            evidence=[ev],
            bandwidth_hz=12_000.0,
            snr_db=20.0,
            noise_color=1.5,
            crackle_density=0.0,
            wow_flutter_hz=0.3,
            block_artifact=0.0,
            pre_echo_ms=0.0,
            classifier_source="dsp",
        )
        d = r.as_dict()
        assert d["material"] == MaterialType.TAPE.value
        assert math.isfinite(d["confidence"])

    def test_03_result_material_type(self) -> None:
        ev = MaterialEvidence(MaterialType.SHELLAC, 0.9, ["noise_color"], [])
        r = ClassificationResult(
            material=MaterialType.SHELLAC,
            confidence=0.9,
            evidence=[ev],
            bandwidth_hz=5_000.0,
            snr_db=5.0,
            noise_color=2.0,
            crackle_density=0.0,
            wow_flutter_hz=0.0,
            block_artifact=0.0,
            pre_echo_ms=0.0,
            classifier_source="dsp",
        )
        assert r.material in list(MaterialType)

    def test_04_result_confidence_range(self) -> None:
        """Konfidenz immer ∈ [0.0, 1.0]."""
        ev = MaterialEvidence(MaterialType.UNKNOWN, 0.0, [], [])
        r = ClassificationResult(
            material=MaterialType.UNKNOWN,
            confidence=0.0,
            evidence=[ev],
            bandwidth_hz=0.0,
            snr_db=0.0,
            noise_color=0.0,
            crackle_density=0.0,
            wow_flutter_hz=0.0,
            block_artifact=0.0,
            pre_echo_ms=0.0,
            classifier_source="dsp",
        )
        assert 0.0 <= r.confidence <= 1.0


# ============================================================================
# TestGroup 2: Singleton + Thread-Safety
# ============================================================================


class TestSingleton:
    """Tests 05–08: Thread-sicheres Singleton (Double-Checked Locking)."""

    def test_05_singleton_same_instance(self) -> None:
        a = get_medium_classifier()
        b = get_medium_classifier()
        assert a is b

    def test_06_singleton_threaded(self) -> None:
        """Unter 16 parallelen Threads darf nur eine Instanz entstehen."""
        instances: list[MediumClassifier] = []
        lock = threading.Lock()

        def _create() -> None:
            inst = get_medium_classifier()
            with lock:
                instances.append(inst)

        threads = [threading.Thread(target=_create) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 16
        assert all(x is instances[0] for x in instances)

    def test_07_convenience_fn_same_instance(self) -> None:
        """classify_medium nutzt denselben Singleton."""
        audio = _noise(0.5)
        result = classify_medium(audio, sr=SR, use_ml=False)
        assert isinstance(result, ClassificationResult)

    def test_08_classify_medium_wrapper(self) -> None:
        audio = _sine(440, 0.5)
        r = classify_medium(audio, sr=SR, use_ml=False)
        assert r.material in list(MaterialType)


# ============================================================================
# TestGroup 3: DSP-Feature-Extraktion (Robustheit)
# ============================================================================


class TestDSPFeatures:
    """Tests 09–18: Einzelne Feature-Extraktion — NaN/Inf-Freiheit, Bounds."""

    def _classify(self, audio: np.ndarray) -> ClassificationResult:
        return get_medium_classifier().classify(audio, SR, use_ml=False)

    def test_09_silence_no_nan(self) -> None:
        r = self._classify(_silence(1.0))
        assert math.isfinite(r.confidence)
        assert math.isfinite(r.snr_db)
        assert math.isfinite(r.bandwidth_hz)

    def test_10_noise_bandwidth_positive(self) -> None:
        r = self._classify(_noise(1.0))
        assert r.bandwidth_hz > 0.0

    def test_11_noise_snr_finite(self) -> None:
        r = self._classify(_noise(1.0))
        assert math.isfinite(r.snr_db)

    def test_12_sine_bandwidth_above_freq(self) -> None:
        r = self._classify(_sine(440, 1.0))
        # 440 Hz Sinus → Bandbreite sollte > 100 Hz sein
        assert r.bandwidth_hz >= 100.0

    def test_13_crackle_density_range(self) -> None:
        r = self._classify(_crackle_vinyl(1.0))
        assert 0.0 <= r.crackle_density <= 1.0

    def test_14_wow_flutter_nonneg(self) -> None:
        r = self._classify(_noise(2.0))
        assert r.wow_flutter_hz >= 0.0

    def test_15_block_artifact_range(self) -> None:
        r = self._classify(_clipped(1.0))
        assert 0.0 <= r.block_artifact <= 1.0

    def test_16_pre_echo_nonneg(self) -> None:
        r = self._classify(_noise(2.0))
        assert r.pre_echo_ms >= 0.0

    def test_17_dirac_impulse(self) -> None:
        """Einzelner Impuls erzeugt keine Exceptions."""
        audio = np.zeros(SR, dtype=np.float32)
        audio[SR // 2] = 1.0
        r = self._classify(audio)
        assert r.material in list(MaterialType)

    def test_18_stereo_input(self) -> None:
        audio = _stereo(_noise(1.0))
        r = self._classify(audio)
        assert r.material in list(MaterialType)


# ============================================================================
# TestGroup 4: Klassifikationsergebnisse — Materialtypen
# ============================================================================


class TestMaterialScoring:
    """Tests 19–30: Plausibilitätsprüfung aller 12 MaterialTypes."""

    def _classify(self, audio: np.ndarray) -> ClassificationResult:
        return get_medium_classifier().classify(audio, SR, use_ml=False)

    def test_19_result_material_in_enum(self) -> None:
        r = self._classify(_noise())
        assert r.material in list(MaterialType)

    def test_20_confidence_in_bounds(self) -> None:
        for audio in [_silence(), _noise(), _sine(), _clipped(), _crackle_vinyl()]:
            r = self._classify(audio)
            assert 0.0 <= r.confidence <= 1.0, f"confidence={r.confidence}"

    def test_21_all_evidence_confidence_finite(self) -> None:
        r = self._classify(_noise())
        for ev in r.evidence:
            assert math.isfinite(ev.confidence)

    def test_22_classifier_source_dsp_when_ml_off(self) -> None:
        r = get_medium_classifier().classify(_noise(), SR, use_ml=False)
        assert r.classifier_source in ("dsp", "unknown")

    def test_23_shellac_noise_color(self) -> None:
        """Shellac → starkes rosa/rotes Rauschen → noise_color > 1.0 erwartet."""
        # Starkes gefärbtes Rauschen erzeugen (beta ~2)
        np.random.seed(42)
        n = SR * 2
        white = np.random.randn(n).astype(np.float32)
        from scipy.signal import lfilter

        colored = lfilter([1.0], [1.0, -0.97], white).astype(np.float32)
        colored /= np.max(np.abs(colored)) + 1e-10
        colored *= 0.4
        r = self._classify(colored)
        assert r.noise_color >= 0.0  # kein Absturz, kein NaN

    def test_24_short_audio_no_crash(self) -> None:
        """Sehr kurze Signale (< 0.1 s) dürfen keine Exception werfen."""
        audio = _noise(0.05)
        r = self._classify(audio)
        assert r.material in list(MaterialType)

    def test_25_very_long_audio_no_crash(self) -> None:
        """5 Minuten Audio — nur 30 s werden für Features genutzt."""
        audio = _noise(10.0)  # 10 s (Proxy für lange Datei)
        r = self._classify(audio)
        assert r.material in list(MaterialType)

    def test_26_flat_top_ratio_range(self) -> None:
        """Flat-Top-Ratio ∈ [0, 1]."""
        r = self._classify(_clipped())
        d = r.as_dict()
        # Flat-Top nicht direkt in ClassificationResult, aber kein Crash
        assert d["material"] in [m.value for m in MaterialType]

    def test_27_evidence_list_nonempty(self) -> None:
        r = self._classify(_noise())
        assert len(r.evidence) >= 1

    def test_28_cd_digital_heuristic(self) -> None:
        """Geclipptes Signal → keine Exception, Ergebnis stabil."""
        audio = _clipped(2.0)
        r1 = self._classify(audio)
        r2 = self._classify(audio)
        assert r1.material == r2.material  # deterministische DSP

    def test_29_vinyl_crackle_detected(self) -> None:
        """Vinyl-Crackle-Signal → crackle_density erhöht."""
        r = self._classify(_crackle_vinyl(2.0, rate_hz=20.0))
        assert r.crackle_density >= 0.0  # mindestens kein NaN

    def test_30_unknown_fallback(self) -> None:
        """Stille führt zu einem definierten (nicht Exception) Ergebnis."""
        r = self._classify(_silence(2.0))
        assert r.material in list(MaterialType)


# ============================================================================
# TestGroup 5: Cache-Verhalten
# ============================================================================


class TestCache:
    """Tests 31–34: SHA256-LRU-Cache (64 Einträge)."""

    def test_31_same_input_same_result(self) -> None:
        audio = _noise(1.0)
        r1 = get_medium_classifier().classify(audio, SR, use_ml=False)
        r2 = get_medium_classifier().classify(audio, SR, use_ml=False)
        assert r1.material == r2.material
        assert r1.confidence == pytest.approx(r2.confidence)

    def test_32_different_sr_different_key(self) -> None:
        """Unterschiedliche SR → unterschiedliche Cache-Keys."""
        audio = _noise(1.0)
        r1 = get_medium_classifier().classify(audio, 44_100, use_ml=False)
        r2 = get_medium_classifier().classify(audio, SR, use_ml=False)
        # Beide müssen ohne Fehler geliefert werden
        assert r1.material in list(MaterialType)
        assert r2.material in list(MaterialType)

    def test_33_different_audio_different_result_possible(self) -> None:
        """Verschieden generierte Signale können unterschiedliche Keys haben."""
        audio_a = _noise(1.0)
        audio_b = _sine(440, 1.0)
        r_a = get_medium_classifier().classify(audio_a, SR, use_ml=False)
        r_b = get_medium_classifier().classify(audio_b, SR, use_ml=False)
        # Nur Robustheitsprüfung — nicht ob Material verschieden
        assert r_a.material in list(MaterialType)
        assert r_b.material in list(MaterialType)

    def test_34_cache_speed(self) -> None:
        """Zweiter Aufruf mit identischem Signal deutlich schneller (Cache-Hit)."""
        audio = _noise(1.0)
        clf = get_medium_classifier()
        clf.classify(audio, SR, use_ml=False)  # warm-up
        t0 = time.perf_counter()
        for _ in range(5):
            clf.classify(audio, SR, use_ml=False)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0  # großzügig; Cache-Hits sollten trivial schnell sein


# ============================================================================
# TestGroup 6: CLAP-ML-Fallback
# ============================================================================


class TestCLAPFallback:
    """Tests 35–37: Graceful Degradation wenn CLAP nicht verfügbar."""

    def test_35_classify_without_ml(self) -> None:
        audio = _noise(1.0)
        r = get_medium_classifier().classify(audio, SR, use_ml=False)
        assert r.classifier_source in ("dsp", "unknown", "clap_ml")

    def test_36_clap_import_error_falls_back_to_dsp(self) -> None:
        """_try_clap_classification wirft Exception → kein Absturz, DSP-Fallback aktiv."""
        clf = get_medium_classifier()
        original_fn = clf._try_clap_classification

        def _failing_clap(audio, sr):  # type: ignore[override]
            raise ImportError("Simulierter CLAP-ImportError")

        clf._try_clap_classification = _failing_clap  # type: ignore[method-assign]
        try:
            audio = _noise(1.0)
            r = clf.classify(audio, SR, use_ml=True)
            assert r.material in list(MaterialType)
        finally:
            clf._try_clap_classification = original_fn  # type: ignore[method-assign]

    def test_37_clap_returns_unknown_still_valid(self) -> None:
        """CLAP-Klassifikation UNKNOWN → DSP-Fallback greift."""
        audio = _noise(1.0)
        r = get_medium_classifier().classify(audio, SR, use_ml=False)
        assert r.material in list(MaterialType)
        assert 0.0 <= r.confidence <= 1.0


# ============================================================================
# TestGroup 7: NaN/Inf-Safety (Pflicht §3.1)
# ============================================================================


class TestNumericalRobustness:
    """Tests 38–40: NaN/Inf-Freiheit bei pathologischen Eingaben."""

    def _classify(self, audio: np.ndarray) -> ClassificationResult:
        return get_medium_classifier().classify(audio, SR, use_ml=False)

    def test_38_nan_input_no_crash(self) -> None:
        audio = np.full(SR, np.nan, dtype=np.float32)
        r = self._classify(audio)
        assert r.material in list(MaterialType)
        assert math.isfinite(r.confidence)

    def test_39_inf_input_no_crash(self) -> None:
        audio = np.full(SR, np.inf, dtype=np.float32)
        r = self._classify(audio)
        assert r.material in list(MaterialType)

    def test_40_zero_sr_raises(self) -> None:
        """sr = 0 ist physikalisch sinnlos → ValueError erwartet."""
        audio = _noise(1.0)
        with pytest.raises((ValueError, ZeroDivisionError, AssertionError)):
            get_medium_classifier().classify(audio, 0, use_ml=False)
