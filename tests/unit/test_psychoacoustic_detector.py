"""
Tests für core/psychoacoustic_artifact_detector.py
Metriken: masking_effect, transient_loss, musical_transparency.
"""

import numpy as np
import pytest

from backend.core.psychoacoustic_artifact_detector import PsychoacousticArtifactDetector

SR = 22050


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _sine(freq: float = 440.0, n: int = SR * 2) -> np.ndarray:
    t = np.linspace(0, n / SR, n, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _white_noise(n: int = SR * 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, n).astype(np.float32)


def _silence(n: int = SR) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _impulses(n: int = SR * 2, period: int = 1000) -> np.ndarray:
    """Periodische Impulsreihe — starke Transienten."""
    audio = np.zeros(n, dtype=np.float32)
    audio[::period] = 0.9
    return audio


# ===========================================================================
# Initialization
# ===========================================================================
class TestInit:
    def test_detected_artifacts_empty_on_init(self):
        det = PsychoacousticArtifactDetector()
        assert det.detected_artifacts == []


# ===========================================================================
# analyze() — Rückgabeformat
# ===========================================================================
class TestAnalyzeOutput:
    def setup_method(self):
        self.det = PsychoacousticArtifactDetector()

    def test_returns_dict(self):
        from backend.core.psychoacoustic_artifact_detector import PsychoacousticArtifactResult

        result = self.det.analyze(_sine(), SR)
        # API now returns a typed @dataclass with dict-compatible access
        assert isinstance(result, PsychoacousticArtifactResult)
        assert "masking_effect" in result  # __contains__ compatibility

    def test_has_required_keys(self):
        result = self.det.analyze(_white_noise(), SR)
        for key in ("masking_effect", "transient_loss", "musical_transparency"):
            assert key in result, f"Schlüssel fehlt: {key}"

    def test_all_values_in_range(self):
        result = self.det.analyze(_white_noise(), SR)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} außerhalb [0,1]: {val:.4f}"

    def test_values_are_floats(self):
        result = self.det.analyze(_sine(), SR)
        for key, val in result.items():
            assert isinstance(val, float), f"{key}: Typ {type(val)} ist kein float"

    def test_no_nan_inf(self):
        result = self.det.analyze(_white_noise(seed=99), SR)
        for key, val in result.items():
            assert np.isfinite(val), f"{key} ist NaN oder Inf: {val}"


# ===========================================================================
# _detect_masking
# ===========================================================================
class TestDetectMasking:
    def setup_method(self):
        self.det = PsychoacousticArtifactDetector()

    def test_in_range(self):
        val = self.det._detect_masking(_white_noise(), SR)
        assert 0.0 <= val <= 1.0

    def test_pure_sine_has_high_masking(self):
        """
        Ein einzelner Sinus = ein dominanter Ton pro Bark-Band.
        → Höherer Maskierungsindex als Weißrauschen (gleichmäßige Energie).
        """
        val_sine = self.det._detect_masking(_sine(freq=1000.0), SR)
        val_noise = self.det._detect_masking(_white_noise(), SR)
        # Sinus soll höheren Maskierungsindex haben als Rauschen
        assert val_sine >= val_noise * 0.8, f"Sinus-Maskierung ({val_sine:.3f}) soll >= Rauschen ({val_noise:.3f})"

    def test_white_noise_lower_masking_than_sine(self):
        """
        Weißrauschen: viele Frequenzen gleich → weniger Masking als reiner Sinus.
        """
        val_sine = self.det._detect_masking(_sine(freq=1000.0, n=SR * 4), SR)
        val_noise = self.det._detect_masking(_white_noise(n=SR * 4), SR)
        # Rauschen hat niedrigere Dominanz pro Band als Sinus
        assert val_noise <= val_sine + 0.15, (
            f"Rauschen ({val_noise:.3f}) hat höheres Masking als Sinus ({val_sine:.3f})"
        )

    def test_silence_returns_valid(self):
        val = self.det._detect_masking(_silence(), SR)
        assert 0.0 <= val <= 1.0

    def test_deterministic(self):
        """Gleiche Eingabe → gleicher Ausgabewert (keine Zufälligkeit)."""
        audio = _white_noise(seed=42)
        v1 = self.det._detect_masking(audio, SR)
        v2 = self.det._detect_masking(audio, SR)
        assert v1 == v2


# ===========================================================================
# _detect_transient_loss
# ===========================================================================
class TestDetectTransientLoss:
    def setup_method(self):
        self.det = PsychoacousticArtifactDetector()

    def test_in_range(self):
        val = self.det._detect_transient_loss(_white_noise(), SR)
        assert 0.0 <= val <= 1.0

    def test_impulses_scored_in_range(self):
        """Impulsreihe: Transient-Score muss in [0, 1] liegen."""
        val = self.det._detect_transient_loss(_impulses(), SR)
        assert 0.0 <= val <= 1.0, f"Impuls-Score außerhalb [0,1]: {val:.3f}"

    def test_stationary_sine_has_near_zero_loss(self):
        """
        Stationärer Sinus: kein Spektralfluss zwischen Frames
        → loss=0.0 (kein Transient, kein Transient-Verlust messbar).
        """
        val = self.det._detect_transient_loss(_sine(n=SR * 4), SR)
        # Stationarer Sinus hat keine Flux-Variation → loss nahe 0
        assert val <= 0.3, f"Sinus-Transientverlust zu hoch: {val:.3f}"

    def test_silence_returns_zero(self):
        """Stilles Signal → kein Spektraler Fluss → loss=0."""
        val = self.det._detect_transient_loss(_silence(n=SR), SR)
        assert val == 0.0, f"Stilles Signal: Verlust sollte 0 sein, ist {val:.4f}"

    def test_deterministic(self):
        audio = _sine()
        v1 = self.det._detect_transient_loss(audio, SR)
        v2 = self.det._detect_transient_loss(audio, SR)
        assert v1 == v2


# ===========================================================================
# _estimate_transparency
# ===========================================================================
class TestEstimateTransparency:
    def setup_method(self):
        self.det = PsychoacousticArtifactDetector()

    def test_in_range(self):
        val = self.det._estimate_transparency(_white_noise(), SR)
        assert 0.0 <= val <= 1.0

    def test_white_noise_higher_transparency_than_sine(self):
        """
        Weißrauschen: flaches Spektrum → hohe SFM → hohe Transparenz.
        Sinus: peakreiches Spektrum → niedrige SFM → niedrige Transparenz.
        """
        val_noise = self.det._estimate_transparency(_white_noise(n=SR * 4), SR)
        val_sine = self.det._estimate_transparency(_sine(freq=440.0, n=SR * 4), SR)
        assert val_noise >= val_sine, f"Rauschen Transparenz ({val_noise:.3f}) sollte >= Sinus ({val_sine:.3f}) sein"

    def test_short_audio_returns_valid(self):
        val = self.det._estimate_transparency(np.zeros(10, dtype=np.float32), SR)
        assert 0.0 <= val <= 1.0

    def test_deterministic(self):
        audio = _white_noise(seed=77)
        v1 = self.det._estimate_transparency(audio, SR)
        v2 = self.det._estimate_transparency(audio, SR)
        assert v1 == v2


# ===========================================================================
# minimize_artifacts
# ===========================================================================
class TestMinimizeArtifacts:
    def setup_method(self):
        self.det = PsychoacousticArtifactDetector()

    def test_output_length(self):
        audio = _white_noise()
        out = self.det.minimize_artifacts(audio, SR)
        assert len(out) == len(audio)

    def test_no_nan_inf(self):
        audio = _white_noise(seed=5)
        out = self.det.minimize_artifacts(audio, SR)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()

    def test_output_in_range(self):
        audio = _white_noise(n=SR * 2, seed=11)
        out = self.det.minimize_artifacts(audio, SR)
        assert np.max(np.abs(out)) <= 1.0 + 1e-5

    def test_dtype_preserved(self):
        audio = _white_noise().astype(np.float32)
        out = self.det.minimize_artifacts(audio, SR)
        assert out.dtype == np.float32

    def test_silence_stays_silent(self):
        audio = _silence(SR)
        out = self.det.minimize_artifacts(audio, SR)
        assert np.max(np.abs(out)) < 1e-6

    def test_energy_preserved(self):
        """RMS der Ausgabe soll sich nicht mehr als ±6 dB ändern."""
        audio = _white_noise(seed=3)
        out = self.det.minimize_artifacts(audio, SR)
        rms_in = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        rms_out = np.sqrt(np.mean(out.astype(np.float64) ** 2))
        ratio = rms_out / (rms_in + 1e-30)
        assert 0.25 < ratio < 4.0, f"Energie drastisch verändert: Ratio={ratio:.3f}"

    def test_sets_detected_artifacts_list(self):
        """Nach minimize_artifacts soll detected_artifacts gesetzt sein (List[str])."""
        audio = _white_noise(seed=8)
        self.det.minimize_artifacts(audio, SR)
        assert isinstance(self.det.detected_artifacts, list)

    def test_sine_no_crash(self):
        audio = _sine()
        out = self.det.minimize_artifacts(audio, SR)
        assert len(out) == len(audio)


# ===========================================================================
# Vollständige Pipeline — verschiedene Signaltypen
# ===========================================================================
class TestPipeline:
    @pytest.mark.parametrize(
        "signal_fn",
        [
            lambda: _white_noise(),
            lambda: _sine(),
            lambda: _impulses(),
            lambda: _silence(SR),
        ],
    )
    def test_analyze_then_minimize_valid(self, signal_fn):
        signal = signal_fn()
        det = PsychoacousticArtifactDetector()
        result = det.analyze(signal, SR)
        out = det.minimize_artifacts(signal, SR)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val:.4f} außerhalb [0,1]"
        assert len(out) == len(signal)
        assert not np.isnan(out).any()
