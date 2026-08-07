"""
§2.46f Edge-Taper-Invariante — Regressionstest Intro/Outro-Artefakte

Sicherstellt:
1. Phase 03 (Denoise) und Phase 23 (SpectralRepair) dürfen am Intro/Outro (erste/letzte 0.5 s)
   keine zusätzlichen Artefakte im Vergleich zum Originaleingang einbringen.
2. nperseg-Guard in PsychoacousticMetrics verhindert UserWarning bei kurzen Segmenten.

Normative Grundlage: §2.46f Natural-Performance-Artifacts-Guard, §0 Primum non nocere.
"""

import numpy as np
import pytest


def _make_vinyl_audio(sr: int = 48000, duration: float = 5.0, seed: int = 42) -> np.ndarray:
    """Stereo Vinyl-ähnliches Testsignal mit leichtem Rauschen (channels-last, float32)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(duration * sr), endpoint=False, dtype=np.float32)
    # Stimmähnliches Signal bei 220 Hz + Oberton + leichtes Vinyl-Rauschen
    signal = (
        0.35 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 440 * t)
        + 0.04 * rng.standard_normal(len(t)).astype(np.float32)
    )
    stereo = np.stack([signal, signal * 0.98], axis=1)  # (N, 2)
    return np.clip(stereo, -1.0, 1.0)  # type: ignore[no-any-return]


def _edge_rms(audio: np.ndarray, sr: int, zone_s: float = 0.5) -> tuple[float, float]:
    """RMS-Energie in Intro- und Outro-Zone (channels-last oder mono)."""
    n = int(zone_s * sr)
    if audio.ndim == 2:
        intro = audio[:n, :].ravel()
        outro = audio[-n:, :].ravel()
    else:
        intro = audio[:n]
        outro = audio[-n:]
    rms_intro = float(np.sqrt(np.mean(intro.astype(np.float64) ** 2) + 1e-15))
    rms_outro = float(np.sqrt(np.mean(outro.astype(np.float64) ** 2) + 1e-15))
    return rms_intro, rms_outro


def _estimate_lr_delay_samples(audio: np.ndarray, max_lag: int = 256) -> int:
    """Estimate L/R delay (samples) via bounded cross-correlation on channels-last stereo."""
    assert audio.ndim == 2 and audio.shape[1] == 2
    l = audio[:, 0].astype(np.float64)
    r = audio[:, 1].astype(np.float64)
    l = l - np.mean(l)
    r = r - np.mean(r)
    float(np.linalg.norm(l) * np.linalg.norm(r) + 1e-12)
    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ls = l[-lag:]
            rs = r[: len(r) + lag]
        elif lag > 0:
            ls = l[: len(l) - lag]
            rs = r[lag:]
        else:
            ls = l
            rs = r
        if len(ls) < 32:
            continue
        corr = float(np.dot(ls, rs) / (np.linalg.norm(ls) * np.linalg.norm(rs) + 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return int(best_lag)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Phase 03 Edge-Taper
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPhase03EdgeTaper:
    """Phase 03 darf kein Artefakt im Intro/Outro (erste/letzte 0.5 s) einbringen."""

    def test_intro_outro_not_louder_than_original(self):
        """Verarbeitetes Intro/Outro darf nicht lauter als Original sein (kein Pegelanstieg)."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        sr = 48000
        audio = _make_vinyl_audio(sr=sr, duration=5.0)
        phase = DenoisePhase()

        result = phase.process(
            audio=audio,
            material_type="vinyl",
            sample_rate=sr,
            strength=0.5,
        )
        assert result.success, "Phase 03 execute fehlgeschlagen"

        out = result.audio
        # channels-first (2, N) → channels-last (N, 2) für Vergleich
        if out.ndim == 2 and out.shape[0] == 2 and out.shape[1] > 2:
            out = out.T

        rms_orig_intro, rms_orig_outro = _edge_rms(audio, sr)
        rms_out_intro, rms_out_outro = _edge_rms(out, sr)

        # Erlaubter Anstieg: ≤ +2 dB = Faktor ~1.26 (Makeup-Gain darf leicht anheben)
        _max_gain_factor = 1.26
        assert rms_out_intro <= rms_orig_intro * _max_gain_factor, (
            f"Phase03 Intro-Artefakt: out_rms={rms_out_intro:.6f} > orig_rms×{_max_gain_factor}={rms_orig_intro * _max_gain_factor:.6f}"
        )
        assert rms_out_outro <= rms_orig_outro * _max_gain_factor, (
            f"Phase03 Outro-Artefakt: out_rms={rms_out_outro:.6f} > orig_rms×{_max_gain_factor}={rms_orig_outro * _max_gain_factor:.6f}"
        )

    def test_intro_outro_similarity_to_original(self):
        """Intro/Outro des verarbeiteten Signals muss hohe Korrelation zum Original haben."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        sr = 48000
        audio = _make_vinyl_audio(sr=sr, duration=5.0)
        phase = DenoisePhase()

        result = phase.process(
            audio=audio,
            material_type="vinyl",
            sample_rate=sr,
            strength=0.6,
        )
        assert result.success

        out = result.audio
        if out.ndim == 2 and out.shape[0] == 2 and out.shape[1] > 2:
            out = out.T

        n_edge = int(0.5 * sr)
        # Mono-Mix für Korrelation
        orig_intro = audio[:n_edge, 0].astype(np.float64)
        out_intro = out[:n_edge, 0].astype(np.float64)

        # Pearson-Korrelation (Vektor-Form, NaN-safe)
        orig_std = np.std(orig_intro)
        out_std = np.std(out_intro)
        if orig_std < 1e-9 or out_std < 1e-9:
            corr = 1.0
        else:
            corr = float(
                np.dot(orig_intro - orig_intro.mean(), out_intro - out_intro.mean())
                / (orig_std * out_std * len(orig_intro) + 1e-12)
            )

        # Mindest-Korrelation 0.70 — Edge-Taper sorgt dafür, dass Intro ≈ Original
        assert corr >= 0.70, f"Phase03 Intro-Korrelation zu gering: corr={corr:.3f} < 0.70 (Edge-Taper fehlt?)"

    def test_no_new_lr_lag(self):
        """Phase 03 darf keine neue L/R-Zeitverschiebung einführen."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        sr = 48000
        audio = _make_vinyl_audio(sr=sr, duration=5.0)
        phase = DenoisePhase()

        result = phase.process(
            audio=audio,
            material_type="vinyl",
            sample_rate=sr,
            strength=0.5,
        )
        assert result.success

        out = result.audio
        if out.ndim == 2 and out.shape[0] == 2 and out.shape[1] > 2:
            out = out.T

        in_lag = _estimate_lr_delay_samples(audio)
        out_lag = _estimate_lr_delay_samples(out)
        assert abs(out_lag - in_lag) <= 1, (
            f"Phase03 L/R-Lag-Regressionsfehler: input={in_lag} samples, output={out_lag} samples"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Phase 23 Edge-Taper
# ─────────────────────────────────────────────────────────────────────────────


class TestPhase23EdgeTaper:
    """Phase 23 SpectralRepair darf kein Artefakt im Intro/Outro einbringen."""

    @staticmethod
    def _enable_mocked_phase23_ml(phase, monkeypatch):
        """Force deterministic ML path with fake FlashSR (no external model deps)."""
        monkeypatch.setenv("AURIK_RUN_HEAVY_TESTS", "1")
        monkeypatch.setattr(
            "backend.core.phases.phase_23_spectral_repair.is_phase_ml_enabled",
            lambda _phase_id: True,
        )
        monkeypatch.setattr(
            "backend.core.phases.phase_23_spectral_repair.QualityModeConfig.should_use_ml",
            lambda _phase_id, _severity: True,
        )
        monkeypatch.setattr(phase, "_has_sufficient_ml_headroom", lambda *_args, **_kwargs: True)

        class _FakeFlashSR:
            def process(self, x, sr, target_sr):
                return np.asarray(x, dtype=np.float32) * 0.7

        monkeypatch.setattr(phase, "_get_flashsr_plugin", lambda: _FakeFlashSR())

    def test_intro_outro_not_louder_than_original(self, monkeypatch):
        """Verarbeitetes Intro/Outro darf max. +2 dB lauter als Original sein."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        sr = 48000
        audio = _make_vinyl_audio(sr=sr, duration=5.0)
        phase = SpectralRepair()
        self._enable_mocked_phase23_ml(phase, monkeypatch)

        result = phase.process(
            audio=audio,
            material=MaterialType.VINYL,
            sample_rate=sr,
            mode="restoration",
        )
        assert result.success

        out = result.audio
        if out.ndim == 2 and out.shape[0] == 2 and out.shape[1] > 2:
            out = out.T

        rms_orig_intro, rms_orig_outro = _edge_rms(audio, sr)
        rms_out_intro, rms_out_outro = _edge_rms(out, sr)

        _max_gain_factor = 1.26  # ≤ +2 dB
        assert rms_out_intro <= rms_orig_intro * _max_gain_factor, (
            f"Phase23 Intro: out={rms_out_intro:.6f} > max={rms_orig_intro * _max_gain_factor:.6f}"
        )
        assert rms_out_outro <= rms_orig_outro * _max_gain_factor, (
            f"Phase23 Outro: out={rms_out_outro:.6f} > max={rms_orig_outro * _max_gain_factor:.6f}"
        )

    def test_no_new_lr_lag(self, monkeypatch):
        """Phase 23 darf keine neue L/R-Zeitverschiebung einführen."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        sr = 48000
        audio = _make_vinyl_audio(sr=sr, duration=5.0)
        phase = SpectralRepair()
        self._enable_mocked_phase23_ml(phase, monkeypatch)

        result = phase.process(
            audio=audio,
            material=MaterialType.VINYL,
            sample_rate=sr,
            mode="restoration",
        )
        assert result.success

        out = result.audio
        if out.ndim == 2 and out.shape[0] == 2 and out.shape[1] > 2:
            out = out.T

        in_lag = _estimate_lr_delay_samples(audio)
        out_lag = _estimate_lr_delay_samples(out)
        assert abs(out_lag - in_lag) <= 1, (
            f"Phase23 L/R-Lag-Regressionsfehler: input={in_lag} samples, output={out_lag} samples"
        )


class TestPhase23StereoMSInvariant:
    """Phase 23 FlashSR-Stereo-Pfad muss M/S-kohärent bleiben (Side-Erhalt)."""

    def test_flashsr_path_preserves_side_component_channels_first(self):
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        class _DummyFlashSR:
            def process(self, x, sr, target_sr):
                # Simulate audible Mid modification without changing length.
                return np.asarray(x, dtype=np.float32) * 0.65

        sr = 48000
        audio = _make_vinyl_audio(sr=sr, duration=3.0)
        audio_cf = audio.T.astype(np.float32, copy=False)  # (2, N)

        phase = SpectralRepair()
        phase._has_sufficient_ml_headroom = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
        phase._current_material = (
            "unknown"  # skip BW hard-cap branch for deterministic side check  # type: ignore[assignment]
        )

        out_cf = phase._repair_with_flashsr(
            audio=audio_cf,
            sample_rate=sr,
            defect_mask=np.zeros((8, 8), dtype=bool),
            repair_strength=0.8,
            flashsr=_DummyFlashSR(),
        )

        assert out_cf.shape == audio_cf.shape
        side_in = 0.5 * (audio_cf[0] - audio_cf[1])
        side_out = 0.5 * (out_cf[0] - out_cf[1])
        # Side should be preserved when Mid-only repair is used.
        assert np.max(np.abs(side_out - side_in)) <= 1e-5, "Phase23 M/S invariant violated: Side changed unexpectedly"

    def test_flashsr_path_preserves_side_component_samples_first(self):
        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        class _DummyFlashSR:
            def process(self, x, sr, target_sr):
                return np.asarray(x, dtype=np.float32) * 0.7

        sr = 48000
        audio = _make_vinyl_audio(sr=sr, duration=3.0).astype(np.float32, copy=False)  # (N, 2)

        phase = SpectralRepair()
        phase._has_sufficient_ml_headroom = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
        phase._current_material = "unknown"  # type: ignore[assignment]

        out = phase._repair_with_flashsr(
            audio=audio,
            sample_rate=sr,
            defect_mask=np.zeros((8, 8), dtype=bool),
            repair_strength=0.8,
            flashsr=_DummyFlashSR(),
        )

        assert out.shape == audio.shape
        side_in = 0.5 * (audio[:, 0] - audio[:, 1])
        side_out = 0.5 * (out[:, 0] - out[:, 1])
        assert np.max(np.abs(side_out - side_in)) <= 1e-5, "Phase23 M/S invariant violated for samples-first"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: nperseg-Guard PsychoacousticMetrics
# ─────────────────────────────────────────────────────────────────────────────
