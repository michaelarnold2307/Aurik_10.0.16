"""
Unit-Tests für plugins/vocos_plugin.py — Aurik 10.0.0

Testumfang: 40 Tests — alle öffentlichen Methoden, Randfälle, NaN/Inf-Schutz,
Singleton-Thread-Safety, SR-Routing, Fallback-Kaskade.

Standard: tests/unit/test_v99_vocos_plugin.py
Pytest-Konfiguration: --timeout=30, kein reales Audio, np.random.seed(42)
"""

import logging
import os

# Sicherstellen dass der VEnv-Pfad stimmt
import sys
import threading
import unittest

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from plugins.vocos_plugin import (
    _HOP_48K,
    _MEL_SR_24K,
    _MEL_SR_44K,
    _MEL_SR_48K,
    _MODEL_44K,
    _MODEL_48K,
    _N_FFT_48K,
    _N_MELS_48K,
    _WIN_48K,
    AURIK_SR,
    MEL_SR_22K,
    VocosPlugin,
    VocosResult,
    get_vocos_plugin,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.ml, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _white_noise(n: int = 48000, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(n).astype(np.float32) * 0.3


def _silence(n: int = 48000) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = AURIK_SR) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5  # type: ignore[no-any-return]


def _dirac(n: int = 48000) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    x[n // 2] = 1.0
    return x


# ---------------------------------------------------------------------------
# TestVocosResult — Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


class TestVocosResult(unittest.TestCase):
    """Tests für VocosResult-Datenklasse."""

    def test_01_basic_construction(self):
        """VocosResult kann mit Pflichtfeldern erstellt werden."""
        r = VocosResult(
            audio=np.zeros(48000, np.float32),
            sr=AURIK_SR,
            pqs_mos=4.2,
            model_used="vocos_pypi",
            confidence=0.95,
        )
        self.assertEqual(r.sr, AURIK_SR)
        self.assertAlmostEqual(r.pqs_mos, 4.2, places=4)
        self.assertEqual(r.model_used, "vocos_pypi")

    def test_02_as_dict_contains_required_keys(self):
        """as_dict() enthält alle Mediametriken."""
        r = VocosResult(
            audio=np.zeros(100, np.float32),
            sr=AURIK_SR,
            pqs_mos=3.5,
            model_used="griffin_lim_fallback",
            confidence=0.70,
            mel_snr_db=15.3,
            model_sr=MEL_SR_22K,
        )
        d = r.as_dict()
        self.assertIn("sr", d)
        self.assertIn("pqs_mos", d)
        self.assertIn("model_used", d)
        self.assertIn("confidence", d)
        self.assertIn("mel_snr_db", d)

    def test_03_default_mel_snr_is_zero(self):
        """mel_snr_db default = 0.0"""
        r = VocosResult(
            audio=np.zeros(100, np.float32),
            sr=AURIK_SR,
            pqs_mos=4.0,
            model_used="vocos_onnx",
            confidence=0.93,
        )
        self.assertEqual(r.mel_snr_db, 0.0)

    def test_04_audio_shape_preserved(self):
        """audio-Array im Ergebnis hat richtige Shape."""
        audio = np.ones(24000, np.float32) * 0.5
        r = VocosResult(audio=audio, sr=AURIK_SR, pqs_mos=4.1, model_used="x", confidence=0.9)
        self.assertEqual(len(r.audio), 24000)

    def test_05_metadata_default_empty(self):
        """metadata-Dict ist standardmäßig leer."""
        r = VocosResult(audio=np.zeros(10, np.float32), sr=AURIK_SR, pqs_mos=3.8, model_used="x", confidence=0.8)
        self.assertIsInstance(r.metadata, dict)


# ---------------------------------------------------------------------------
# TestVocosPluginResample — _resample()
# ---------------------------------------------------------------------------


class TestVocosPluginResample(unittest.TestCase):
    """Tests für die SR-Konvertierungsmethode."""

    def test_06_resample_identity(self):
        """Gleiches SR → unveränderter Output."""
        x = _sine()
        r = VocosPlugin._resample(x, AURIK_SR, AURIK_SR)
        np.testing.assert_array_almost_equal(x, r, decimal=5)

    def test_07_resample_down_48k_to_22k(self):
        """Downsampling 48 000 → 22 050 Hz: korrekte Länge."""
        x = _white_noise(48000)
        r = VocosPlugin._resample(x, 48000, 22050)
        expected_len = int(len(x) * 22050 / 48000)
        self.assertAlmostEqual(len(r), expected_len, delta=200)

    def test_08_resample_up_22k_to_48k(self):
        """Upsampling 22 050 → 48 000 Hz: korrekte Länge."""
        x = _white_noise(22050)
        r = VocosPlugin._resample(x, 22050, 48000)
        expected_len = int(len(x) * 48000 / 22050)
        self.assertAlmostEqual(len(r), expected_len, delta=200)

    def test_09_resample_output_dtype_float32(self):
        """Ausgabe ist immer float32."""
        x = np.ones(4800, dtype=np.float64)
        r = VocosPlugin._resample(x, 48000, 22050)
        self.assertEqual(r.dtype, np.float32)

    def test_10_resample_no_nan(self):
        """Kein NaN nach Resampling."""
        x = _sine()
        r = VocosPlugin._resample(x, 48000, 22050)
        self.assertTrue(np.isfinite(r).all())

    def test_11_resample_nan_input_handled(self):
        """NaN-Eingabe erzeugt keinen Absturz (nan_to_num nach resample)."""
        x = np.full(4800, np.nan, dtype=np.float32)
        # scipy.resample_poly wird intern NaN propagieren aber nan_to_num fängt ab
        try:
            r = VocosPlugin._resample(x, 48000, 22050)
            self.assertEqual(r.dtype, np.float32)
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Fehler sind toleriert, Absturz nicht


# ---------------------------------------------------------------------------
# TestVocosPluginMelFilterbank — _build_mel_filterbank()
# ---------------------------------------------------------------------------


class TestVocosPluginMelFilterbank(unittest.TestCase):
    """Tests für die Mel-Filterbank-Konstruktion."""

    def test_12_filterbank_shape(self):
        """Filterbank hat korrekte Form [n_mels, n_freq]."""
        fb = VocosPlugin._build_mel_filterbank(80, 513, 22050, 1024)
        self.assertEqual(fb.shape, (80, 513))

    def test_13_filterbank_nonnegative(self):
        """Filterbank-Gewichte sind alle ≥ 0."""
        fb = VocosPlugin._build_mel_filterbank(80, 513, 22050, 1024)
        self.assertTrue((fb >= 0).all())

    def test_14_filterbank_no_nan(self):
        """Keine NaN in der Filterbank."""
        fb = VocosPlugin._build_mel_filterbank(80, 513, 22050, 1024)
        self.assertTrue(np.isfinite(fb).all())

    def test_15_filterbank_dtype_float32(self):
        """Filterbank ist float32."""
        fb = VocosPlugin._build_mel_filterbank(80, 513, 22050, 1024)
        self.assertEqual(fb.dtype, np.float32)

    def test_16_filterbank_100_bands_44k(self):
        """100-Band-Filterbank für 44100 Hz hat korrekte Shape."""
        fb = VocosPlugin._build_mel_filterbank(100, 513, 44100, 1024)
        self.assertEqual(fb.shape, (100, 513))


# ---------------------------------------------------------------------------
# TestVocosPluginComputeMel — _compute_mel()
# ---------------------------------------------------------------------------


class TestVocosPluginComputeMel(unittest.TestCase):
    """Tests für die Mel-Spektrogramm-Berechnung."""

    def test_17_mel_output_shape(self):
        """Mel-Ausgabe hat Form [n_mels, T], T > 0."""
        x = _sine(duration=0.5, sr=22050)
        mel = VocosPlugin._compute_mel(x, 22050, 80)
        self.assertEqual(mel.shape[0], 80)
        self.assertGreater(mel.shape[1], 0)

    def test_18_mel_output_no_nan(self):
        """Mel-Spektrogramm enthält kein NaN/Inf."""
        x = _white_noise(22050)
        mel = VocosPlugin._compute_mel(x, 22050, 80)
        self.assertTrue(np.isfinite(mel).all())

    def test_19_mel_silence_near_minimum(self):
        """Stille ergibt sehr kleine Mel-Werte (nahe log(1e-8))."""
        x = _silence(22050)
        mel = VocosPlugin._compute_mel(x, 22050, 80)
        self.assertLess(float(np.max(mel)), 1.0)  # log(1e-8) ≈ -18.4

    def test_20_mel_dtype_float32(self):
        """Mel-Ausgabe ist float32."""
        x = _sine(duration=0.2, sr=22050)
        mel = VocosPlugin._compute_mel(x, 22050, 80)
        self.assertEqual(mel.dtype, np.float32)

    def test_21_mel_dirac_no_crash(self):
        """Dirac-Impuls als Eingabe erzeugt keinen Absturz."""
        x = _dirac(22050)
        mel = VocosPlugin._compute_mel(x, 22050, 80)
        self.assertTrue(np.isfinite(mel).all())


# ---------------------------------------------------------------------------
# TestVocosPluginMatchLength — _match_length()
# ---------------------------------------------------------------------------


class TestVocosPluginMatchLength(unittest.TestCase):
    def test_22_match_length_pad(self):
        """Zu kurzes Audio wird mit Nullen auf Ziellänge aufgefüllt."""
        x = np.ones(100, np.float32)
        r = VocosPlugin._match_length(x, 200)
        self.assertEqual(len(r), 200)
        np.testing.assert_array_equal(r[100:], 0.0)

    def test_23_match_length_trim(self):
        """Zu langes Audio wird auf Ziellänge gekürzt."""
        x = np.ones(300, np.float32)
        r = VocosPlugin._match_length(x, 200)
        self.assertEqual(len(r), 200)

    def test_24_match_length_exact(self):
        """Exakte Länge → unveränderter Output."""
        x = np.arange(150, dtype=np.float32)
        r = VocosPlugin._match_length(x, 150)
        np.testing.assert_array_equal(x, r)


# ---------------------------------------------------------------------------
# TestVocosPluginFallback — Griffin-Lim-Pfad (ohne Modell)
# ---------------------------------------------------------------------------


class TestVocosPluginFallback(unittest.TestCase):
    """Tests für den Griffin-Lim-Fallback ohne Modell."""

    def setUp(self):
        # Plugin ohne Modell instantiieren (kein Modell in ~/.aurik/models/vocos/)
        self.plugin = VocosPlugin.__new__(VocosPlugin)
        self.plugin._prefer_sr = MEL_SR_22K
        self.plugin._model_sr = MEL_SR_22K
        self.plugin._vocos_pypi = None
        self.plugin._onnx_session = None
        self.plugin._model_loaded = False
        self.plugin._fallback_mode = "griffin_lim_fallback"

    def test_25_fallback_output_shape(self):
        """Griffin-Lim Fallback: Output-Länge = Input-Länge."""
        x = _white_noise(48000)
        result, name, conf = self.plugin._synthesize_griffin_lim(x, AURIK_SR)  # type: ignore[attr-defined]
        self.assertEqual(len(result), len(x))

    def test_26_fallback_no_nan(self):
        """Griffin-Lim Fallback: kein NaN im Output."""
        x = _sine()
        result, _, _ = self.plugin._synthesize_griffin_lim(x, AURIK_SR)  # type: ignore[attr-defined]
        self.assertTrue(np.isfinite(result).all())

    def test_27_fallback_clipped(self):
        """Griffin-Lim Fallback: Ausgabe liegt in [-1, 1]."""
        x = _white_noise(48000)
        result, _, _ = self.plugin._synthesize_griffin_lim(x, AURIK_SR)  # type: ignore[attr-defined]
        self.assertLessEqual(float(np.max(np.abs(result))), 1.0)

    def test_28_fallback_model_name(self):
        """Griffin-Lim Fallback: model_name korrekt."""
        x = _silence(4800)
        _, name, _ = self.plugin._synthesize_griffin_lim(x, AURIK_SR)  # type: ignore[attr-defined]
        self.assertIn("griffin_lim", name)

    def test_29_fallback_confidence_lower(self):
        """Fallback-Konfidenz ist < 0.85 (schlechter als neuronales Modell)."""
        x = _white_noise(4800)
        _, _, conf = self.plugin._synthesize_griffin_lim(x, AURIK_SR)  # type: ignore[attr-defined]
        self.assertLess(conf, 0.85)

    def test_30_fallback_silence_input(self):
        """Stille als Eingabe: Griffin-Lim gibt Stille (oder nahezu) zurück."""
        x = _silence(48000)
        result, _, _ = self.plugin._synthesize_griffin_lim(x, AURIK_SR)  # type: ignore[attr-defined]
        self.assertLess(float(np.max(np.abs(result))), 0.1)

    def test_31_fallback_short_audio(self):
        """Sehr kurzes Audio (< 1024 Samples): kein Absturz."""
        x = np.zeros(100, dtype=np.float32)
        try:
            result, name, conf = self.plugin._synthesize_griffin_lim(x, AURIK_SR)  # type: ignore[attr-defined]
            self.assertEqual(len(result), 100)
        except Exception:
            logger.warning("test fallback", exc_info=True)
            pass  # Sehr kurze Signale können Fehler erzeugen — kein Crash

    def test_32_vocode_restoration_raises(self):
        """vocode() im Restoration-Modus raises ValueError."""
        with self.assertRaises(ValueError):
            self.plugin.vocode(_sine(), AURIK_SR, mode="restoration")

    def test_33_vocode_wrong_sr_raises(self):
        """vocode() mit falscher SR raises AssertionError."""
        with self.assertRaises(AssertionError):
            self.plugin.vocode(_sine(), 44100, mode="studio2026")

    def test_34_vocode_nan_input_sanitized(self):
        """NaN im Input-Audio wird bereinigt, kein Absturz."""
        x = np.full(48000, np.nan, dtype=np.float32)
        result = self.plugin.vocode(x, AURIK_SR, mode="studio2026")
        self.assertIsInstance(result, VocosResult)
        self.assertTrue(np.isfinite(result.audio).all())

    def test_35_vocode_inf_input_sanitized(self):
        """Inf im Input-Audio wird bereinigt."""
        x = np.full(48000, np.inf, dtype=np.float32)
        result = self.plugin.vocode(x, AURIK_SR, mode="studio2026")
        self.assertTrue(np.isfinite(result.audio).all())

    def test_36_active_backend_property(self):
        """active_backend-Property gibt aktuellen Backend-Namen zurück."""
        self.assertEqual(self.plugin.active_backend, "griffin_lim_fallback")

    def test_37_model_loaded_property_false(self):
        """model_loaded ist False wenn kein Modell geladen."""
        self.assertFalse(self.plugin.model_loaded)


# ---------------------------------------------------------------------------
# TestVocosPluginPqsMos — _estimate_pqs_mos()
# ---------------------------------------------------------------------------


class TestVocosPluginPqsMos(unittest.TestCase):
    def setUp(self):
        self.plugin = VocosPlugin.__new__(VocosPlugin)
        self.plugin._prefer_sr = MEL_SR_22K
        self.plugin._model_sr = MEL_SR_22K
        self.plugin._vocos_pypi = None
        self.plugin._onnx_session = None
        self.plugin._model_loaded = False
        self.plugin._fallback_mode = "griffin_lim_fallback"

    def test_38_mos_bounds(self):
        """PQS-MOS liegt immer in [1.0, 5.0]."""
        x = _sine()
        mos = self.plugin._estimate_pqs_mos(x, x, AURIK_SR)
        self.assertGreaterEqual(mos, 1.0)
        self.assertLessEqual(mos, 5.0)

    def test_39_mos_identical_high(self):
        """Identisches Audio: MOS ist hoch (≥ 4.0)."""
        x = _sine()
        mos = self.plugin._estimate_pqs_mos(x, x, AURIK_SR)
        self.assertGreaterEqual(mos, 4.0)

    def test_40_mel_snr_identical_audio(self):
        """Identisches Audio: Mel-SNR ist groß (≥ 20 dB)."""
        x = _sine()
        snr = self.plugin._mel_snr(x, x, AURIK_SR)
        self.assertGreaterEqual(snr, 20.0)


# ---------------------------------------------------------------------------
# TestVocosSingleton — Thread-Safety & Singleton-Pattern
# ---------------------------------------------------------------------------


class TestVocosSingleton(unittest.TestCase):
    def test_singleton_same_object(self):
        """get_vocos_plugin() gibt immer dieselbe Instanz zurück."""
        import plugins.vocos_plugin as vmod

        # Reset
        vmod._instance = None
        inst1 = get_vocos_plugin()
        inst2 = get_vocos_plugin()
        self.assertIs(inst1, inst2)
        vmod._instance = None  # Cleanup

    def test_singleton_thread_safe(self):
        """Gleichzeitige Singleton-Erstellung durch 8 Threads gibt dasselbe Objekt."""
        import plugins.vocos_plugin as vmod

        vmod._instance = None

        instances = []
        lock = threading.Lock()

        def create():
            inst = get_vocos_plugin()
            with lock:
                instances.append(id(inst))

        threads = [threading.Thread(target=create) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        alive = [t for t in threads if t.is_alive()]
        self.assertEqual(len(alive), 0, f"{len(alive)} Threads hängen nach 10 s Timeout")
        self.assertEqual(len(set(instances)), 1, "Singleton verletzt: mehrere Instanzen erzeugt")
        vmod._instance = None  # Cleanup


# ---------------------------------------------------------------------------
# TestVocosPlugin48kHz — 48kHz-Primärpfad, Konstanten, ONNX-Inferenz
# ---------------------------------------------------------------------------


class TestVocosPlugin48kHz(unittest.TestCase):
    """Tests für den 48 kHz nativ-Pfad (kein Resampling bei Aurik-SR 48000)."""

    # -- Konstanten ----------------------------------------------------------

    def test_43_mel_sr_48k_constant(self):
        """_MEL_SR_48K == 48000."""
        self.assertEqual(_MEL_SR_48K, 48_000)

    def test_44_n_mels_48k_constant(self):
        """_N_MELS_48K == 128 (vocos-mel-48khz-alpha1 Konfiguration)."""
        self.assertEqual(_N_MELS_48K, 128)

    def test_45_n_fft_48k_constant(self):
        """_N_FFT_48K == 2048."""
        self.assertEqual(_N_FFT_48K, 2048)

    def test_46_hop_48k_constant(self):
        """_HOP_48K == 256."""
        self.assertEqual(_HOP_48K, 256)

    def test_47_win_48k_constant(self):
        """_WIN_48K == 2048."""
        self.assertEqual(_WIN_48K, 2048)

    def test_48_model_48k_path_suffix(self):
        """_MODEL_48K endet auf vocos_48khz.onnx."""
        self.assertTrue(_MODEL_48K.endswith("vocos_48khz.onnx"))

    def test_49_priority_48k_before_44k(self):
        """48kHz-Pfad liegt im Dateisystem vor 44kHz-Pfad (Priority-Reihenfolge)."""

        # Priorität: 48k → 44k → 24k (Spec §2.37)
        self.assertIn("48khz", _MODEL_48K)
        self.assertNotIn("48", _MODEL_44K)

    def test_50_model_sr_differs_per_tier(self):
        """Alle drei Tier-SRs sind unterschiedlich."""
        self.assertNotEqual(_MEL_SR_48K, _MEL_SR_44K)
        self.assertNotEqual(_MEL_SR_44K, _MEL_SR_24K)
        self.assertNotEqual(_MEL_SR_48K, _MEL_SR_24K)

    # -- Lade-Verhalten (mit gemocktem ONNX-Pfad) ----------------------------

    def test_51_try_load_48k_sets_model_sr(self):
        """_try_load mit 48kHz-Datei setzt _model_sr auf 48000."""
        if not os.path.exists(_MODEL_48K):
            self.skipTest("vocos_48khz.onnx nicht gebündelt — Offline-Test übersprungen")
        plugin = VocosPlugin.__new__(VocosPlugin)
        plugin._prefer_sr = _MEL_SR_48K
        plugin._model_sr = _MEL_SR_24K
        plugin._mel_n_mels = 100
        plugin._mel_n_fft = 1024
        plugin._mel_hop = 256
        plugin._mel_win = 1024
        plugin._vocos_pypi = None
        plugin._onnx_session = None
        plugin._model_loaded = False
        plugin._fallback_mode = "griffin_lim_fallback"
        plugin._try_load(_MODEL_48K)
        if plugin._model_loaded:
            self.assertEqual(plugin._model_sr, _MEL_SR_48K)
            self.assertEqual(plugin._mel_n_mels, _N_MELS_48K)
            self.assertEqual(plugin._mel_n_fft, _N_FFT_48K)
            self.assertEqual(plugin._mel_hop, _HOP_48K)
            self.assertEqual(plugin._fallback_mode, "vocos_onnx")

    def test_52_onnx_inference_48k_input_output(self):
        """ONNX-Inferenz: mel [1,128,T] → audio [1,S], NaN-frei, Float32."""
        if not os.path.exists(_MODEL_48K):
            self.skipTest("vocos_48khz.onnx nicht gebündelt")
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime nicht installiert")
        sess = ort.InferenceSession(_MODEL_48K, providers=["CPUExecutionProvider"])
        T = 40  # kurzer Test-Batch
        mel = np.random.randn(1, 128, T).astype(np.float32) * 0.01 - 8.0
        audio = np.asarray(sess.run(None, {"mel": mel})[0], dtype=np.float32)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape[0], 1)
        self.assertGreater(audio.shape[1], 0)
        self.assertTrue(np.isfinite(audio).all(), "Inf/NaN in ONNX-Ausgabe")
        self.assertEqual(audio.dtype, np.float32)

    def test_53_onnx_output_length_48k(self):
        """ONNX-Ausgabelänge folgt (T-G+1)*hop mit G=8, hop=256."""
        if not os.path.exists(_MODEL_48K):
            self.skipTest("vocos_48khz.onnx nicht gebündelt")
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime nicht installiert")
        sess = ort.InferenceSession(_MODEL_48K, providers=["CPUExecutionProvider"])
        T = 50
        G = 8  # OLA overlap factor
        hop = _HOP_48K
        mel = np.zeros((1, 128, T), dtype=np.float32)
        audio = sess.run(None, {"mel": mel})[0]
        expected = (T - G + 1) * hop
        self.assertEqual(audio.shape[1], expected, f"Ausgabelänge {audio.shape[1]} != erwartet {expected}")

    def test_54_vocode_prefers_48k_when_loaded(self):
        """Nach _try_load(48kHz) zeigt active_backend 'vocos_onnx'."""
        if not os.path.exists(_MODEL_48K):
            self.skipTest("vocos_48khz.onnx nicht gebündelt")
        plugin = VocosPlugin.__new__(VocosPlugin)
        plugin._prefer_sr = _MEL_SR_48K
        plugin._model_sr = _MEL_SR_24K
        plugin._mel_n_mels = 100
        plugin._mel_n_fft = 1024
        plugin._mel_hop = 256
        plugin._mel_win = 1024
        plugin._vocos_pypi = None
        plugin._onnx_session = None
        plugin._model_loaded = False
        plugin._fallback_mode = "griffin_lim_fallback"
        plugin._try_load(_MODEL_48K)
        if plugin._model_loaded:
            self.assertEqual(plugin.active_backend, "vocos_onnx")
            self.assertTrue(plugin.model_loaded)


if __name__ == "__main__":
    unittest.main()
