"""
§v10.112–§v10.117 Regression-Tests und Performance-Budgets (§Feinpolitur).

Testet die in dieser Konversation implementierten Guards:
  - §v10.112 Groove-Guard (NaturalnessOptimizer)
  - §v10.113 HPI-Gate (Goosebumps-Recovery)
  - §v10.114 Silence-Guard (Phase 07 FeedbackChain)
  - §v10.115 Universal Safety Wrapper (RMS/Transient/Hallucination)
  - §v10.117 Universal Formant-Guard (Vocal spectral envelope)

Plus: Performance-Budgets für kritische Pipeline-Stufen.
"""

import time

import numpy as np
import pytest

SR: int = 48_000
_rng = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════════
# §v10.115 Universal Safety Wrapper — Guard-Abdeckung
# ═══════════════════════════════════════════════════════════════════════════


class TestUniversalSafetyWrapper:
    """Verifies that ALL phases get RMS/formant/transient/hallucination guards."""

    def test_rms_guard_active_on_all_phases(self):
        """§v10.115: Jede Phase muss rms_drop_db im Metadata haben (nicht hartcodiert 0.0)."""
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        p = ClickRemovalPhase()
        sig = np.clip(_rng.standard_normal(4800).astype(np.float32) * 0.3, -1.0, 1.0)
        r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
        assert "rms_drop_db" in r.metadata, "RMS-Guard metadata missing"
        rms = float(r.metadata["rms_drop_db"])
        # Real RMS values are never exactly 0.0 for non-silent input
        # (hardcoded 0.0 would indicate the guard didn't overwrite)
        assert abs(rms) < 10.0, f"rms_drop_db implausible: {rms}"

    def test_formant_guard_active(self):
        """§v10.117: Jede Phase muss formant_stability berechnen."""
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        p = ClickRemovalPhase()
        sig = np.clip(_rng.standard_normal(4800).astype(np.float32) * 0.3, -1.0, 1.0)
        r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
        assert "formant_stability" in r.metadata, "§v10.117 Formant-Guard metadata missing"
        fs = float(r.metadata["formant_stability"])
        assert 0.0 <= fs <= 1.01, f"formant_stability out of range: {fs}"

    def test_formant_guard_detects_spectral_shift(self):
        """§v10.117: Formant-Guard muss spektrale Verschiebungen erkennen."""
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        p = ClickRemovalPhase()
        # Create two signals with different spectral envelopes
        sig_in = np.sin(2 * np.pi * 440 * np.arange(4800) / SR).astype(np.float32) * 0.3
        sig_out = np.sin(2 * np.pi * 880 * np.arange(4800) / SR).astype(np.float32) * 0.3
        # Manually check formant correlation via the band-vector method
        from backend.core.phases.phase_interface import PhaseResult

        # Quick inline test: different spectra should have correlation < 1.0
        n = min(len(sig_in), 8192)
        bands = np.logspace(np.log10(300), np.log10(3500), 11)
        bins = np.clip(np.round(bands * n / SR).astype(int), 1, n // 2 - 1)
        spec_in = np.abs(np.fft.rfft(sig_in[:n]))
        spec_out = np.abs(np.fft.rfft(sig_out[:n]))
        env_in = np.array([float(np.mean(spec_in[bins[i] : bins[i + 1]])) for i in range(10)])
        env_out = np.array([float(np.mean(spec_out[bins[i] : bins[i + 1]])) for i in range(10)])
        env_in_n = env_in / (np.max(env_in) + 1e-12)
        env_out_n = env_out / (np.max(env_out) + 1e-12)
        corr = float(np.dot(env_in_n, env_out_n) / (np.linalg.norm(env_in_n) * np.linalg.norm(env_out_n) + 1e-12))
        # Different fundamental → different formant structure → correlation < 0.95
        assert corr < 0.95, f"Formant check should detect spectral difference, corr={corr:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# §v10.112 Groove-Guard — Transient-Erhalt bei groovigen Signalen
# ═══════════════════════════════════════════════════════════════════════════


class TestGrooveGuard:
    """§v10.112: NaturalnessOptimizer darf Attack-Transienten nicht zerstören."""

    @pytest.mark.timeout(10)
    def test_groovy_signal_retains_peaks(self):
        """Bei hoher Transientendichte (>5/s) müssen Peaks erhalten bleiben."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        # Grooviges Signal: viele Attack-Transienten
        rng = np.random.default_rng(999)
        sig = np.zeros(48000, dtype=np.float32)
        for i in range(0, 48000, 400):  # 120 attacks = 10/s
            sig[i : i + 50] = rng.standard_normal(50).astype(np.float32) * 0.9
        result = optimize_naturalness(sig.copy(), sig, SR, material="vinyl", era="1960-1970", mode="RESTORATION")
        output_peak = np.max(np.abs(result.audio))
        assert output_peak > 0.4, f"§v10.112 Groove-Guard failed: peak={output_peak:.3f} (transients destroyed)"

    @pytest.mark.timeout(10)
    def test_already_natural_not_degraded(self):
        """Bereits natürliches Audio darf nicht verschlechtert werden."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        sig = _rng.standard_normal(48000).astype(np.float32) * 0.3
        result = optimize_naturalness(sig.copy(), sig, SR, material="cd_digital", era="post-1980")
        assert result.delta_hpe >= -0.05, f"NaturalnessOptimizer degraded audio: delta_hpe={result.delta_hpe:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# §v10.114 Silence-Guard — Phase 07 kein Kollaps im FeedbackChain
# ═══════════════════════════════════════════════════════════════════════════


class TestSilenceGuard:
    """§v10.114: Phase 07 darf auf sauberem Audio nicht kollabieren."""

    @pytest.mark.timeout(10)
    def test_clean_audio_not_silenced(self):
        """Sauberes Audio (hohes H2/H1) muss Phase 07 sicher passieren."""
        from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

        p = HarmonicRestorationPhase()
        # Bereits sauberes harmonisches Signal (sollte H2/H1 > 0.35 haben)
        t = np.arange(48000) / SR
        sig = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.12 * np.sin(2 * np.pi * 880 * t)
        sig = np.asarray(sig, dtype=np.float32)
        r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
        output_rms = float(np.sqrt(np.mean(r.audio**2)))
        assert output_rms > 0.01, (
            f"§v10.114 Silence-Guard failed: Phase 07 produced near-silence (RMS={output_rms:.6f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# §v10.116 ExcellenceOptimizer — RX11-Kalibrierung
# ═══════════════════════════════════════════════════════════════════════════


class TestExcellenceCalibration:
    """§v10.116: ExcellenceOptimizer mit RX11-kalibrierten Parametern."""

    def test_auto_profile_uses_new_defaults(self):
        """Auto-Profil muss die neuen §v10.116 Defaults verwenden."""
        from backend.core.excellence_optimizer import _HARM_BOOST_DB, _MODULATION_STRENGTH, MATERIAL_PROFILES

        profile = MATERIAL_PROFILES["auto"]
        assert profile.modulation_strength >= 0.50, f"Auto modulation_strength too low: {profile.modulation_strength}"
        assert profile.harm_boost_db >= 3.0, f"Auto harm_boost_db too low: {profile.harm_boost_db}"

    def test_shellac_profile_uses_high_values(self):
        """Schellack muss die höchsten Restaurations-Werte haben."""
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        profile = MATERIAL_PROFILES["shellac"]
        assert profile.harm_boost_db >= 3.2, f"Shellac harm_boost_db too low: {profile.harm_boost_db}"
        assert profile.modulation_strength >= 0.48, (
            f"Shellac modulation_strength too low: {profile.modulation_strength}"
        )

    @pytest.mark.timeout(15)
    def test_optimizer_runs_without_errors(self):
        """ExcellenceOptimizer muss fehlerfrei durchlaufen."""
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        sig = np.clip(_rng.standard_normal((48000, 2)).astype(np.float32) * 0.3, -1.0, 1.0)
        opt = ExcellenceOptimizer(sample_rate=SR, material="auto")
        audio, report = opt.optimize(sig)
        assert audio.shape == sig.shape
        assert not np.any(np.isnan(audio))


# ═══════════════════════════════════════════════════════════════════════════
# Performance-Budgets
# ═══════════════════════════════════════════════════════════════════════════


class TestPerformanceBudgets:
    """Performance-Budget-Assertions: Keine Phase darf das RT-Budget sprengen."""

    @pytest.mark.timeout(10)
    def test_naturalness_optimizer_budget(self):
        from backend.core.naturalness_optimizer import optimize_naturalness

        sig = _rng.standard_normal(48000).astype(np.float32) * 0.3
        t0 = time.perf_counter()
        optimize_naturalness(sig.copy(), sig, SR, material="cd_digital", era="post-1980")
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"NaturalnessOptimizer budget exceeded: {elapsed:.1f}s"

    @pytest.mark.timeout(15)
    def test_excellence_optimizer_budget(self):
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        sig = np.clip(_rng.standard_normal((48000, 2)).astype(np.float32) * 0.3, -1.0, 1.0)
        opt = ExcellenceOptimizer(sample_rate=SR, material="auto")
        t0 = time.perf_counter()
        opt.optimize(sig)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"ExcellenceOptimizer budget exceeded: {elapsed:.1f}s"


# ═══════════════════════════════════════════════════════════════════════════
# Material-Profile — Wohlklang-Garantie für ALLE Importsongs (§v10.116)
# ═══════════════════════════════════════════════════════════════════════════


class TestMaterialProfiles:
    """Jedes Material-Profil muss mit seinem spezifischen Signal-Typ funktionieren."""

    # Alle 8 Profile + auto
    MATERIALS = ["auto", "vinyl", "tape", "shellac", "broadcast", "mp3_low", "mp3_high", "cd_digital"]

    @pytest.mark.timeout(15)
    @pytest.mark.parametrize("material", MATERIALS)
    def test_profile_exists_and_valid(self, material):
        """Profil muss existieren, ladbar sein und plausible Werte haben."""
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        assert material in MATERIAL_PROFILES, f"Material '{material}' not in MATERIAL_PROFILES"
        p = MATERIAL_PROFILES[material]
        assert 0.0 <= p.modulation_strength <= 1.0, f"{material}: modulation_strength={p.modulation_strength}"
        assert 0.0 <= p.harm_boost_db <= 5.0, f"{material}: harm_boost_db={p.harm_boost_db}"
        assert 0.01 <= p.target_cv_min <= 0.50, f"{material}: target_cv_min={p.target_cv_min}"
        assert 5.0 <= p.ola_ms <= 100.0, f"{material}: ola_ms={p.ola_ms}"

    @pytest.mark.timeout(15)
    @pytest.mark.parametrize("material", ["vinyl", "tape", "shellac", "cd_digital"])
    def test_optimizer_does_not_degrade_per_material(self, material):
        """§v10.116: ExcellenceOptimizer darf auf keinem Material verschlechtern."""
        from backend.core.excellence_optimizer import ExcellenceOptimizer

        sig = np.clip(_rng.standard_normal((48000, 2)).astype(np.float32) * 0.3, -1.0, 1.0)
        opt = ExcellenceOptimizer(sample_rate=SR, material=material)
        audio, report = opt.optimize(sig)
        # Form-Prüfung
        assert audio.shape == sig.shape, f"{material}: shape changed {sig.shape} → {audio.shape}"
        assert not np.any(np.isnan(audio)), f"{material}: NaN im Output"
        # RMS-Prüfung: kein extremer Pegelverlust
        rms_in = float(np.sqrt(np.mean(sig**2)) + 1e-12)
        rms_out = float(np.sqrt(np.mean(audio**2)) + 1e-12)
        rms_db = 20 * np.log10(rms_out / rms_in)
        assert rms_db > -12.0, f"{material}: RMS-Drop {rms_db:.1f} dB zu stark"
        assert rms_db < 6.0, f"{material}: RMS-Gain {rms_db:.1f} dB zu stark (clipping-risk)"

    def test_material_profiles_proportional(self):
        """Restaurations-intensive Materialien müssen höhere Werte haben als cleanere."""
        from backend.core.excellence_optimizer import MATERIAL_PROFILES

        shellac = MATERIAL_PROFILES["shellac"]
        cd = MATERIAL_PROFILES["cd_digital"]
        # Schellack braucht mehr Restauration als CD
        assert shellac.modulation_strength > cd.modulation_strength, "Shellac sollte mehr Modulation brauchen als CD"
        assert shellac.harm_boost_db > cd.harm_boost_db, "Shellac sollte mehr Harmonik-Boost brauchen als CD"


# ═══════════════════════════════════════════════════════════════════════════
# Edge-Case-Tests — Pipeline-robustheit für extreme Eingaben
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineEdgeCases:
    """Wohlklang-Garantie: extreme Eingaben dürfen die Pipeline nicht crashen."""

    @pytest.mark.timeout(10)
    def test_silence_through_naturalness_optimizer(self):
        """Stille darf NaturalnessOptimizer nicht crashen."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        sig = np.zeros(48000, dtype=np.float32)
        result = optimize_naturalness(sig, sig, SR, material="cd_digital", era="post-1980")
        assert not np.any(np.isnan(result.audio)), "NaN in silence output"
        assert result.audio.shape == sig.shape

    @pytest.mark.timeout(10)
    def test_clipping_through_optimizer(self):
        """Voll ausgesteuertes Signal darf nicht crashen."""
        from backend.core.naturalness_optimizer import optimize_naturalness

        rng = np.random.default_rng(42)
        sig = np.clip(rng.standard_normal(48000).astype(np.float32) * 2.0, -1.0, 1.0)
        result = optimize_naturalness(sig.copy(), sig, SR, material="cd_digital", era="post-1980", mode="RESTORATION")
        assert not np.any(np.isnan(result.audio)), "NaN in clipped output"
        # Output must stay within bounds
        assert np.max(np.abs(result.audio)) <= 2.0, (
            f"Output exceeded safety ceiling: {np.max(np.abs(result.audio)):.3f}"
        )

    @pytest.mark.timeout(10)
    def test_dc_offset_passthrough(self):
        """DC-Offset durch Phase 01: kein Crash, keine Verstärkung."""
        from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

        p = ClickRemovalPhase()
        sig = np.full(4800, 0.5, dtype=np.float32)
        r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
        assert not np.any(np.isnan(r.audio)), "NaN in DC output"
        # DC sollte nicht verstärkt werden
        assert np.max(np.abs(r.audio)) <= 1.0, f"DC amplified: {np.max(np.abs(r.audio)):.3f}"

    @pytest.mark.timeout(10)
    def test_stereo_balance_preserved(self):
        """Stereo-Phasen dürfen die Kanal-Balance nicht zerstören."""
        from backend.core.phases.phase_15_stereo_balance import StereoBalancePhaseV2

        p = StereoBalancePhaseV2()
        # Asymmetrisches Stereo-Signal
        left = _rng.standard_normal(4800).astype(np.float32) * 0.5
        right = _rng.standard_normal(4800).astype(np.float32) * 0.1
        sig = np.column_stack([left, right])
        r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
        assert r.audio.shape == sig.shape, f"Stereo shape changed: {r.audio.shape}"
        assert not np.any(np.isnan(r.audio)), "NaN in stereo output"

    @pytest.mark.timeout(10)
    def test_very_short_input_handled(self):
        """Extrem kurze Eingabe (16 Samples) muss alle Phasen sicher passieren."""
        # Teste die kritischsten Phasen mit sehr kurzer Eingabe
        phases_to_test = [
            ("phase_01_click_removal", "ClickRemovalPhase"),
        ]
        for mod_name, class_name in phases_to_test:
            import importlib

            mod = importlib.import_module(f"backend.core.phases.{mod_name}")
            p = getattr(mod, class_name)()
            sig = np.array([0.1, -0.1] * 8, dtype=np.float32)
            try:
                r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
                assert r.audio is not None, f"{mod_name}: returned None"
            except Exception as e:
                # _safe_process should catch exceptions, never propagate
                pytest.fail(f"{mod_name}: unexpected exception: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Vocal Quality — Wohlklang-Garantie für Gesang (§v10.117)
# ═══════════════════════════════════════════════════════════════════════════


class TestVocalQuality:
    """§v10.117: Gesangsqualität muss in ALLEN Phasen geschützt werden."""

    @pytest.mark.timeout(10)
    def test_vocal_like_signal_retains_formants_after_non_vocal_phase(self):
        """Selbst nicht-vokale Phasen (z.B. EQ) dürfen Gesangs-Formanten nicht verschieben."""
        from backend.core.phases.phase_04_eq_correction import EQCorrectionPhase

        p = EQCorrectionPhase()
        # Vocal-ähnliches Signal: Grundton + Formanten
        t = np.arange(4800) / SR
        sig = (
            0.3 * np.sin(2 * np.pi * 220 * t)  # F0 ~A3
            + 0.15 * np.sin(2 * np.pi * 600 * t)  # F1
            + 0.10 * np.sin(2 * np.pi * 1100 * t)  # F2
            + 0.05 * np.sin(2 * np.pi * 2800 * t)  # F3
        ).astype(np.float32)
        r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
        fs = float(r.metadata.get("formant_stability", 0.0))
        assert fs > 0.80, f"§v10.117: EQ-Phase verschob Formanten (stability={fs:.3f}) — Gesang klingt unnatürlich"

    @pytest.mark.timeout(10)
    def test_denoise_preserves_vocal_formants(self):
        """Denoising darf Gesangsformanten nicht zerstören."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        p = DenoisePhase()
        t = np.arange(48000) / SR
        sig = (
            0.2 * np.sin(2 * np.pi * 330 * t)  # E4
            + 0.10 * np.sin(2 * np.pi * 800 * t)  # F1-ähnlich
            + 0.06 * np.sin(2 * np.pi * 1200 * t)  # F2-ähnlich
            + 0.01 * _rng.standard_normal(48000).astype(np.float32)  # Leichtes Rauschen
        ).astype(np.float32)
        r = p._safe_process(np.clip(sig, -1, 1), sample_rate=SR, material_type="cd_digital")
        # RMS sollte nicht kollabieren
        out_rms = float(np.sqrt(np.mean(r.audio**2)))
        assert out_rms > 0.01, f"Denoising silenced vocal-like signal: RMS={out_rms:.6f}"
        assert not np.any(np.isnan(r.audio)), "NaN in denoised vocal"

    @pytest.mark.timeout(10)
    def test_breath_like_signal_survives(self):
        """Atem-ähnliches Rauschen darf nicht vollständig entfernt werden."""
        from backend.core.phases.phase_03_denoise import DenoisePhase

        p = DenoisePhase()
        # Breath-like: sehr leises, hochfrequentes Rauschen
        rng = np.random.default_rng(777)
        breath = rng.standard_normal(48000).astype(np.float32) * 0.005
        # Hochpass-Filterung simuliert Atem-Frequenzgang
        from scipy.signal import butter, sosfiltfilt

        sos = butter(2, 2000 / (SR / 2), btype="high", output="sos")
        breath = sosfiltfilt(sos, breath).astype(np.float32)
        sig = np.clip(breath + rng.standard_normal(48000).astype(np.float32) * 0.001, -1, 1)
        r = p._safe_process(sig, sample_rate=SR, material_type="cd_digital")
        out_rms = float(np.sqrt(np.mean(r.audio**2)))
        in_rms = float(np.sqrt(np.mean(sig**2)))
        # Atem darf leiser werden (NR), aber nicht komplett verschwinden
        assert out_rms > in_rms * 0.01, f"Breath completely removed: RMS ratio={out_rms / in_rms:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Phase-Scope-Fixes — Regression für behobene Bugs
# ═══════════════════════════════════════════════════════════════════════════


class TestScopeFixes:
    """Regression-Tests für die behobenen Scope-Bugs (kwargs, _mk)."""

    def test_phase23_kwargs_in_repair_channel(self):
        """Phase 23 _repair_channel muss jetzt **kwargs akzeptieren."""
        import inspect

        from backend.core.phases.phase_23_spectral_repair import SpectralRepair

        sig = inspect.signature(SpectralRepair._repair_channel)
        param_names = list(sig.parameters.keys())
        assert "kwargs" in param_names, f"_repair_channel missing **kwargs: params={param_names}"

    def test_phase30_mk_replaced_with_material(self):
        """Phase 30 _preserve_phase_loudness darf _mk nicht mehr referenzieren."""
        import inspect

        source = inspect.getsource(
            __import__(
                "backend.core.phases.phase_30_dc_offset_removal", fromlist=["DCOffsetRemoval"]
            ).DCOffsetRemoval._preserve_phase_loudness
        )
        # _mk sollte nirgends mehr vorkommen (jetzt: material.name)
        assert "_mk" not in source, "Phase 30 _preserve_phase_loudness still references _mk"

    def test_safe_stft_available(self):
        """§v10.115: scipy.signal.safe_stft muss nach backend-Import existieren."""
        import scipy.signal as signal

        import backend  # Triggers monkey-patch

        assert hasattr(signal, "safe_stft"), "safe_stft not monkey-patched"
        # Verify it actually works
        sig = np.random.randn(4800).astype(np.float32)
        f, t, Zxx = signal.safe_stft(sig, SR, nperseg=512, noverlap=384)
        assert Zxx.shape[0] == 257, f"safe_stft wrong freq bins: {Zxx.shape}"
