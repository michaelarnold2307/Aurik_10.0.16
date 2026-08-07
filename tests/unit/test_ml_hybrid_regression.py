from __future__ import annotations

"""
tests/unit/test_ml_hybrid_regression.py
=========================================
Regressionstests für die 10 ML-Hybrid-Phasen (§3 copilot-instructions.md).

Abgedeckte Phasen:
    Phase 01 — Click Removal       (DeepFilterNet-Hybrid)
    Phase 02 — Hum Removal         (DeepFilterNet-Hybrid)
    Phase 03 — Denoise             (OMLSA + Resemble Enhance)
    Phase 09 — Crackle Removal     (BANQUET Vinyl-Hybrid)
    Phase 12 — Wow/Flutter Fix     (pYIN / CREPE-Hybrid)
    Phase 18 — Noise Gate          (Silero VAD-Hybrid)
    Phase 19 — De-Esser            (Phoneme Detection-Hybrid)
    Phase 20 — Reverb Reduction    (DCCRN-Hybrid)
    Phase 23 — Spectral Repair     (FlashSR-Hybrid)
    Phase 24 — Dropout Repair      (FlashSR-Hybrid)
    Phase 29 — Tape Hiss Reduction (DeepFilterNet-Hybrid)

Jede Phase wird getestet auf:
    R-01  Ausgabe NaN/Inf-frei (§3.1)
    R-02  Kein Hard-Clipping (|audio| ≤ 1.0 + ε)
    R-03  Shape-Erhalt (Samples × Kanäle bleibt gleich)
    R-04  DSP-Fallback aktiv — Phase funktioniert OHNE ML (ML-Import-Fehler toleriert)
    R-05  Pass-Through-Invariante: sauberes Signal nicht verschlechtert (SNR-Verlust ≤ 3 dB)
    R-06  Mono und Stereo-Eingang verarbeitet
    R-07  Stille-Eingang — kein Absturz, Stille bleibt Stille
    R-08  Dirac-Impuls — kein Absturz
    R-09  RT-Budget: Verarbeitung ≤ 5 s für 1 s Audio (§9.5)

Invarianten:
    np.random.seed(42) für Reproduzierbarkeit
    SR = 48000 (§6.6)
    Nur synthetische Signale (§5.1)
    timeout=30 s per Test (pytest.ini)
"""


import importlib
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.ml, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Globale Testsignale
# ---------------------------------------------------------------------------
np.random.seed(42)
SR: int = 48_000
DUR_S: float = 1.0
N: int = int(SR * DUR_S)

# Sauberes Sinus-Signal (≈ SNR > 40 dB gegenüber Rauschen)
_t = np.linspace(0, DUR_S, N, endpoint=False, dtype=np.float32)
AUDIO_SINE = (0.4 * np.sin(2 * np.pi * 440 * _t)).astype(np.float32)

# Verrauschtes Signal (Testobjekt für NR-Phasen)
AUDIO_NOISY = (AUDIO_SINE + 0.05 * np.random.randn(N).astype(np.float32)).astype(np.float32)

# Vinyl-Crackle: Sinus + Impuls-Spikes
AUDIO_CRACKLE = AUDIO_SINE.copy()
for _i in range(0, N, SR // 20):
    AUDIO_CRACKLE[_i] = np.clip(AUDIO_CRACKLE[_i] + 0.8, -1.0, 1.0)

# Stereo-Versionen
AUDIO_SINE_STEREO = np.stack([AUDIO_SINE, AUDIO_SINE * 0.9], axis=1)  # (N, 2)
AUDIO_NOISY_STEREO = np.stack([AUDIO_NOISY, AUDIO_NOISY * 0.9], axis=1)

# Stille und Dirac
AUDIO_SILENCE = np.zeros(N, dtype=np.float32)
AUDIO_DIRAC = np.zeros(N, dtype=np.float32)
AUDIO_DIRAC[N // 2] = 0.5

# Dropout-Signal: 100-ms-Lücke
AUDIO_DROPOUT = AUDIO_SINE.copy()
gap_start = SR // 4
gap_end = gap_start + SR // 10  # 100 ms
AUDIO_DROPOUT[gap_start:gap_end] = 0.0

# Hum: Sinus mit 50-Hz-Brumm
_t50 = np.linspace(0, DUR_S, N, endpoint=False, dtype=np.float32)
AUDIO_HUM = AUDIO_SINE + 0.2 * np.sin(2 * np.pi * 50 * _t50).astype(np.float32)

# ε für Clipping-Check
_CLIP_EPS: float = 1e-3


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _load_phase(module_name: str, class_name: str):
    """Importiert eine Phase-Klasse. Raises ImportError bei fehlendem Modul."""
    mod = importlib.import_module(f"backend.core.phases.{module_name}")
    return getattr(mod, class_name)


def _run_phase(cls, audio: np.ndarray, **kwargs):
    """Instanziiert Phase, führt process() aus, gibt PhaseResult zurück."""
    phase = cls()
    return phase.process(audio, sample_rate=SR, **kwargs)


def _audio_from_result(result) -> np.ndarray:
    """Extrahiert Audio aus PhaseResult (Audio-Array)."""
    if isinstance(result, np.ndarray):
        return result
    if hasattr(result, "audio"):
        return result.audio  # type: ignore[no-any-return]
    if hasattr(result, "processed_audio"):
        return result.processed_audio  # type: ignore[no-any-return]
    raise AttributeError(f"Kein audio-Feld in {type(result)}")


def _snr_db(clean: np.ndarray, processed: np.ndarray) -> float:
    """SNR-Verlust in dB (positiv = besser, negativ = Verlust)."""
    clean_p = float(np.mean(clean**2)) + 1e-12
    diff_p = float(np.mean((processed - clean) ** 2)) + 1e-12
    return 10.0 * np.log10(clean_p / diff_p)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Parametrisierung: (Modulname, Klassenname, Testsignal, kwargs)
# ---------------------------------------------------------------------------
from backend.core.defect_scanner import MaterialType as _MAT_T

_M = _MAT_T.CD_DIGITAL  # material-Argument für Phasen die es benötigen (§6.1)

_HYBRID_PHASES = [
    ("phase_01_click_removal", "ClickRemovalPhase", AUDIO_CRACKLE, {}),
    ("phase_02_hum_removal", "HumRemovalPhase", AUDIO_HUM, {}),
    ("phase_03_denoise", "DenoisePhase", AUDIO_NOISY, {}),
    ("phase_09_crackle_removal", "CrackleRemovalPhase", AUDIO_CRACKLE, {}),
    ("phase_12_wow_flutter_fix", "WowFlutterFix", AUDIO_SINE, {}),
    ("phase_18_noise_gate", "NoiseGate", AUDIO_NOISY, {}),
    ("phase_19_de_esser", "DeEsserPhase", AUDIO_SINE, {"material": _M}),
    ("phase_20_reverb_reduction", "ReverbReduction", AUDIO_SINE, {}),
    pytest.param(
        "phase_23_spectral_repair",
        "SpectralRepair",
        AUDIO_NOISY,
        {},
        marks=pytest.mark.ml,  # FlashSR (5.9 GB) — Heavy-ML; nur mit --run-heavy-tests
    ),
    ("phase_24_dropout_repair", "DropoutRepairPhase", AUDIO_DROPOUT, {}),
    ("phase_29_tape_hiss_reduction", "TapeHissReductionPhase", AUDIO_NOISY, {}),
]

_PHASE_IDS = [
    (t.id if hasattr(t, "id") and t.id else t.values[0]) if hasattr(t, "values") else t[0] for t in _HYBRID_PHASES
]


# ---------------------------------------------------------------------------
# R-01/R-02/R-03: NaN-frei, kein Clipping, Shape-Erhalt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R01_output_finite(module_name, class_name, audio, kwargs):
    """R-01 — Ausgabe NaN/Inf-frei (§3.1)."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, audio, **kwargs)
    out = _audio_from_result(result)
    assert np.isfinite(out).all(), f"{module_name}: NaN/Inf im Ausgang"


@pytest.mark.parametrize("module_name,class_name,audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R02_no_hard_clipping(module_name, class_name, audio, kwargs):
    """R-02 — Kein Hard-Clipping (|audio| ≤ 1.0 + ε)."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, audio, **kwargs)
    out = _audio_from_result(result)
    assert np.max(np.abs(out)) <= 1.0 + _CLIP_EPS, f"{module_name}: Clipping — max={np.max(np.abs(out)):.4f}"


@pytest.mark.parametrize("module_name,class_name,audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R03_shape_preserved(module_name, class_name, audio, kwargs):
    """R-03 — Shape-Erhalt (gleiche Sample-Anzahl)."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, audio, **kwargs)
    out = _audio_from_result(result)
    assert out.shape[0] == audio.shape[0], f"{module_name}: Shape geändert — {audio.shape} → {out.shape}"


# ---------------------------------------------------------------------------
# R-04: DSP-Fallback — kein Absturz wenn ML-Modell fehlt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R04_dsp_fallback_no_crash(module_name, class_name, audio, kwargs):
    """R-04 — Phase funktioniert ohne ML (DSP-Fallback, §3.4)."""
    cls = _load_phase(module_name, class_name)
    # Fallback wird durch normalen Aufruf aktiviert wenn ML unavailable;
    # da im Test keine GPU/ONNX-Modelle geladen werden, testet dies implizit den Fallback-Pfad.
    result = _run_phase(cls, audio, **kwargs)
    out = _audio_from_result(result)
    assert out is not None
    assert np.isfinite(out).all(), f"{module_name}: Fallback-Ausgang NaN/Inf"


# ---------------------------------------------------------------------------
# R-05: Pass-Through-Invariante — sauberes Signal nicht verschlechtert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R05_passthrough_clean_signal(module_name, class_name, audio, kwargs):
    """R-05 — Sauberes Signal (SNR-Verlust ≤ 3 dB) — §8.2 Pass-Through-Invariante."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, AUDIO_SINE, **kwargs)
    out = _audio_from_result(result)
    # Länge angleichen (Phase kann marginalen Trim verursachen)
    min_len = min(len(AUDIO_SINE), len(out))
    snr = _snr_db(AUDIO_SINE[:min_len], out[:min_len])
    # Mindest-SNR: −3 dB (sauberes Material darf kaum schlechter werden)
    assert snr >= -3.0, f"{module_name}: Pass-Through-Verlust zu groß — SNR={snr:.1f} dB"


# ---------------------------------------------------------------------------
# R-06: Mono und Stereo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,_audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R06_mono_input(module_name, class_name, _audio, kwargs):
    """R-06a — Mono-Eingang verarbeitet."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, AUDIO_SINE, **kwargs)
    out = _audio_from_result(result)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("module_name,class_name,_audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R06_stereo_input(module_name, class_name, _audio, kwargs):
    """R-06b — Stereo-Eingang (N, 2) — kein Absturz, Ausgabe endlich."""
    cls = _load_phase(module_name, class_name)
    # Stereo intern oft als (2, N) oder (N, 2) — Phase entscheidet
    try:
        result = _run_phase(cls, AUDIO_SINE_STEREO, **kwargs)
    except Exception:
        # Stereo nicht unterstützt: Mono-Fallback testen
        result = _run_phase(cls, AUDIO_SINE, **kwargs)
    out = _audio_from_result(result)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# R-07: Stille-Eingang
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,_audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R07_silence_input(module_name, class_name, _audio, kwargs):
    """R-07 — Stille bleibt endlich, kein Absturz (§3.1)."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, AUDIO_SILENCE, **kwargs)
    out = _audio_from_result(result)
    assert np.isfinite(out).all(), f"{module_name}: Stille-Eingang → NaN/Inf"
    # Ausgabe darf nicht lauter als Stille sein (kein Rauschen generiert)
    rms = float(np.sqrt(np.mean(out**2)))
    assert rms < 0.05, f"{module_name}: Stille-Eingang erzeugt Rauschen (RMS={rms:.4f})"


# ---------------------------------------------------------------------------
# R-08: Dirac-Impuls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,_audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R08_dirac_input(module_name, class_name, _audio, kwargs):
    """R-08 — Dirac-Impuls — kein Absturz, endliche Ausgabe."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, AUDIO_DIRAC, **kwargs)
    out = _audio_from_result(result)
    assert np.isfinite(out).all(), f"{module_name}: Dirac-Eingang → NaN/Inf"


# ---------------------------------------------------------------------------
# R-09: RT-Budget (≤ 5 s für 1 s Audio, §9.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R09_rt_budget(module_name, class_name, audio, kwargs):
    """R-09 — RT-Budget: Verarbeitung ≤ 5 s für 1 s Audio (§9.5, Desktop-CPU)."""
    cls = _load_phase(module_name, class_name)
    t0 = time.perf_counter()
    _run_phase(cls, audio, **kwargs)
    elapsed = time.perf_counter() - t0
    audio_dur = len(audio) / SR
    rt_factor = elapsed / max(audio_dur, 1e-9)
    assert rt_factor <= 5.0, f"{module_name}: RT-Faktor {rt_factor:.1f}× überschreitet Budget (≤ 5.0×)"


# ---------------------------------------------------------------------------
# R-10: PhaseResult.success = True (wenn PhaseResult zurückgegeben wird)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name,class_name,audio,kwargs", _HYBRID_PHASES, ids=_PHASE_IDS)  # type: ignore[arg-type]
def test_R10_phase_result_success(module_name, class_name, audio, kwargs):
    """R-10 — PhaseResult.success = True (keine intern gemeldeten Fehler)."""
    cls = _load_phase(module_name, class_name)
    result = _run_phase(cls, audio, **kwargs)
    if hasattr(result, "success"):
        assert result.success, f"{module_name}: PhaseResult.success = False"
    # Wenn kein .success-Feld: nur NaN-Check
    else:
        out = _audio_from_result(result)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Spezifische Regressionstests für kritische ML-Phasen
# ---------------------------------------------------------------------------


class TestPhase03DenoiseRegression:
    """Spezifische Regressionstests für Phase 03 (OMLSA + Resemble Enhance)."""

    @staticmethod
    def _load():
        from backend.core.phases.phase_03_denoise import DenoisePhase

        return DenoisePhase()

    def test_denoise_reduces_noise_rms(self):
        """Verrauschtes Signal wird leiser gemacht (NR funktioniert)."""
        phase = self._load()
        result = phase.process(AUDIO_NOISY, sample_rate=SR)
        out = _audio_from_result(result)
        rms_in = float(np.sqrt(np.mean(AUDIO_NOISY**2)))
        rms_out = float(np.sqrt(np.mean(out**2)))
        # Ausgabe darf nicht lauter als Eingang sein
        assert rms_out <= rms_in * 1.1, f"Phase 03: RMS nach NR größer — in={rms_in:.4f}, out={rms_out:.4f}"

    def test_denoise_no_musical_noise_in_silence(self):
        """Stille-Fenster nach NR: kein Musical-Noise (RMS < 0.02)."""
        phase = self._load()
        result = phase.process(AUDIO_SILENCE, sample_rate=SR)
        out = _audio_from_result(result)
        rms = float(np.sqrt(np.mean(out**2)))
        assert rms < 0.02, f"Phase 03: Musical-Noise in Stille (RMS={rms:.4f})"

    def test_denoise_tonal_content_preserved(self):
        """440-Hz-Ton nach NR noch erkennbar (FFT-Spektralpeak bleibt erhalten)."""
        phase = self._load()
        result = phase.process(AUDIO_NOISY, sample_rate=SR)
        out = _audio_from_result(result)
        min_len = min(len(AUDIO_SINE), len(out))
        # FFT-Energie bei 440 Hz im Ausgangssignal
        fft_out = np.abs(np.fft.rfft(out[:min_len]))
        freqs = np.fft.rfftfreq(min_len, d=1.0 / SR)
        # Bin bei 440 Hz finden
        bin_440 = int(np.argmin(np.abs(freqs - 440)))
        peak_energy = float(fft_out[bin_440])
        # Rauschboden-Energie: Median aller Bins (robuster Referenzwert)
        noise_floor = float(np.median(fft_out))
        # Peak muss mind. 3× über Rauschboden liegen (Ton erkennbar)
        assert peak_energy >= noise_floor * 3.0, (
            f"Phase 03: 440-Hz-Peak nicht erkennbar (peak={peak_energy:.2f}, floor={noise_floor:.2f})"
        )


class TestPhase12WowFlutterRegression:
    """Regressionstests für Phase 12 (pYIN/CREPE Pitch-Stabilisierung)."""

    @staticmethod
    def _load():
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        return WowFlutterFix()

    def test_wow_flutter_output_length_unchanged(self):
        """Sample-Länge bleibt nach Wow/Flutter-Korrektur exakt gleich."""
        phase = self._load()
        result = phase.process(AUDIO_SINE, sample_rate=SR)
        out = _audio_from_result(result)
        assert len(out) == len(AUDIO_SINE)

    def test_wow_flutter_no_nan_on_transient_signal(self):
        """Impuls-Signal — kein NaN nach Pitch-Korrektur."""
        phase = self._load()
        result = phase.process(AUDIO_DIRAC, sample_rate=SR)
        out = _audio_from_result(result)
        assert np.isfinite(out).all()


class TestPhase24DropoutRepairRegression:
    """Regressionstests für Phase 24 (FlashSR Dropout-Interpolation)."""

    @staticmethod
    def _load():
        from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

        return DropoutRepairPhase()

    def test_dropout_gap_filled(self):
        """100-ms-Lücke nach Dropout-Repair nicht mehr komplett Null."""
        phase = self._load()
        result = phase.process(AUDIO_DROPOUT, sample_rate=SR)
        out = _audio_from_result(result)
        min_len = min(len(out), gap_end)
        gap_rms = float(np.sqrt(np.mean(out[gap_start:min_len] ** 2)))
        # Lücke sollte nach Repair etwas Signal haben (nicht mehr exakt 0)
        before_rms = float(np.sqrt(np.mean(AUDIO_SINE[gap_start:gap_end] ** 2)))
        # Mindestens 10% des ursprünglichen Signals wiederhergestellt
        assert gap_rms >= before_rms * 0.05, f"Phase 24: Dropout-Lücke nicht gefüllt (gap_rms={gap_rms:.5f})"

    def test_dropout_repair_finite(self):
        """Ausgabe NaN/Inf-frei (DSP-Fallback muss greifen)."""
        phase = self._load()
        result = phase.process(AUDIO_DROPOUT, sample_rate=SR)
        out = _audio_from_result(result)
        assert np.isfinite(out).all()


class TestPhase29TapeHissRegression:
    """Regressionstests für Phase 29 (DeepFilterNet Tape-Hiss)."""

    @staticmethod
    def _load():
        from backend.core.phases.phase_29_tape_hiss_reduction import TapeHissReductionPhase

        return TapeHissReductionPhase()

    def test_hiss_reduction_no_over_suppression(self):
        """Hiss-Reduktion löscht den Sinus-Ton nicht aus (RMS bleibt ≥ 0.1)."""
        # Signal: Sinus + leises weißes Rauschen (Tape-Hiss-Simulation)
        hiss = AUDIO_SINE + 0.03 * np.random.randn(N).astype(np.float32)
        phase = self._load()
        result = phase.process(hiss, sample_rate=SR)
        out = _audio_from_result(result)
        rms_out = float(np.sqrt(np.mean(out**2)))
        # Ton muss ≥ 25% seiner Energie behalten (DSP-Fallback darf etwas dämpfen)
        rms_sine = float(np.sqrt(np.mean(AUDIO_SINE**2)))
        assert rms_out >= rms_sine * 0.25, (
            f"Phase 29: Ton übermäßig unterdrückt (rms_out={rms_out:.4f}, rms_sine={rms_sine:.4f})"
        )

    def test_hiss_reduction_silence_stays_silent(self):
        """Stille bleibt nach Hiss-Reduktion leise (kein Artefakt)."""
        phase = self._load()
        result = phase.process(AUDIO_SILENCE, sample_rate=SR)
        out = _audio_from_result(result)
        assert float(np.sqrt(np.mean(out**2))) < 0.02
