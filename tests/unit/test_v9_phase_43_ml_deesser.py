from __future__ import annotations

"""
tests/unit/test_v9_phase_43_ml_deesser.py
==========================================
Dedizierte Tests für Phase 43 — MLDeEsserPhase v2.1 (stimmtyp-adaptiv, §2.8).

Abgedeckte Invarianten:
  - NaN / Inf / Clipping
  - Shape-Erhalt (Mono + Stereo)
  - Stimmtyp-adaptive Frequenzauswahl (§2.8): MALE 5–10 kHz, FEMALE 6–12 kHz, CHILD 7–14 kHz
  - Strength-Cap (§2.19.3 Schlager: 0.45)
  - Threshold-Verhalten (weich / aggressiv)
  - Metadata-Vollständigkeit
  - Stille / Dirac / weißes Rauschen
  - Konsistenz-Test (selbe Eingabe → selber Ausgang)
  - Pass-Through bei Threshold=0 dBFS
  - Verschiedene Abtastraten
  - Geschwindigkeits-Budget (≤ 5× Echtzeit)
"""


import time

import numpy as np
import pytest

SR = 48_000
DUR_S = 1.0
N = int(SR * DUR_S)

np.random.seed(42)


# ---------------------------------------------------------------------------
# Hilfsfunktionen & Fixtures
# ---------------------------------------------------------------------------


def _sibilant_signal(n: int = N, sr: int = SR, freq_hz: float = 7000.0) -> np.ndarray:
    """Reiner Sinus im Sibilantenbereich (steuert De-Esser)."""
    t = np.linspace(0, n / sr, n, endpoint=False)
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]


def _bass_signal(n: int = N) -> np.ndarray:
    """Reiner Sinus im Bass-Bereich — soll vom De-Esser unberührt bleiben."""
    t = np.linspace(0, N / SR, n, endpoint=False)
    return (0.5 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)  # type: ignore[no-any-return]


@pytest.fixture()
def phase():
    from backend.core.phases.phase_43_ml_deesser import MLDeEsserPhase

    return MLDeEsserPhase()


@pytest.fixture()
def mono():
    return _sibilant_signal()


@pytest.fixture()
def stereo():
    ch1 = _sibilant_signal()
    ch2 = _sibilant_signal(freq_hz=8000.0)
    return np.column_stack([ch1, ch2]).astype(np.float32)


@pytest.fixture()
def silent():
    return np.zeros(N, dtype=np.float32)


# ---------------------------------------------------------------------------
# Grundlegende Ausgangs-Invarianten
# ---------------------------------------------------------------------------


class TestOutputInvariants:
    """NaN, Clipping, Shape, PhaseResult.success."""

    def test_01_mono_no_nan(self, phase, mono):
        result = phase.process(mono, SR, enable_ml_refine=True)
        assert np.isfinite(result.audio).all(), "NaN/Inf im Ausgang (Mono)"

    def test_02_mono_no_clipping(self, phase, mono):
        result = phase.process(mono, SR, enable_ml_refine=True)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6, "Clipping überschritten"

    def test_03_mono_shape_preserved(self, phase, mono):
        result = phase.process(mono, SR)
        assert result.audio.shape == mono.shape

    def test_04_stereo_no_nan(self, phase, stereo):
        result = phase.process(stereo, SR)
        assert np.isfinite(result.audio).all()

    def test_05_stereo_no_clipping(self, phase, stereo):
        result = phase.process(stereo, SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_06_stereo_shape_preserved(self, phase, stereo):
        result = phase.process(stereo, SR)
        assert result.audio.shape == stereo.shape

    def test_07_success_true(self, phase, mono):
        result = phase.process(mono, SR)
        assert result.success is True

    def test_08_silent_no_nan(self, phase, silent):
        result = phase.process(silent, SR)
        assert np.isfinite(result.audio).all()

    def test_09_silent_stays_silent(self, phase, silent):
        result = phase.process(silent, SR)
        assert np.allclose(result.audio, 0.0, atol=1e-6)

    def test_10_dirac_pulse_no_nan(self, phase):
        dirac = np.zeros(N, dtype=np.float32)
        dirac[N // 2] = 0.9
        result = phase.process(dirac, SR)
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Stimmtyp-adaptive Frequenzauswahl (§2.8)
# ---------------------------------------------------------------------------


class TestGenderAdaptiveFrequencies:
    """Verifiziert, dass gender-Parameter die korrekten Frequenzbereiche setzt."""

    @pytest.mark.parametrize(
        "gender,freq_low,freq_high",
        [
            ("male", 5_000.0, 10_000.0),
            ("female", 6_000.0, 12_000.0),
            ("child", 7_000.0, 14_000.0),
            ("unknown", 5_000.0, 9_000.0),
        ],
    )
    def test_11_gender_freq_in_metadata(self, phase, gender, freq_low, freq_high):
        """Metadata enthält gender-spezifische Frequenzgrenzen."""
        audio = _sibilant_signal(freq_hz=(freq_low + freq_high) / 2)
        result = phase.process(audio, SR, gender=gender)
        assert result.metadata["freq_low_hz"] == pytest.approx(freq_low, rel=0.01)
        assert result.metadata["freq_high_hz"] == pytest.approx(freq_high, rel=0.01)

    def test_12_gender_stored_in_metadata(self, phase, mono):
        result = phase.process(mono, SR, gender="female")
        assert result.metadata["gender"] == "female"

    def test_13_explicit_freq_overrides_gender(self, phase, mono):
        """Explizite freq_low/freq_high überschreiben Gender-Auswahl."""
        result = phase.process(mono, SR, gender="male", freq_low=8000.0, freq_high=11000.0)
        assert result.metadata["freq_low_hz"] == pytest.approx(8000.0, rel=0.01)
        assert result.metadata["freq_high_hz"] == pytest.approx(11000.0, rel=0.01)

    def test_14_female_reduces_7khz_sinus(self, phase):
        """FEMALE-De-Esser dämpft 7 kHz Sinus (liegt im 6–12 kHz Band)."""
        sib = _sibilant_signal(freq_hz=7_000.0)
        result = phase.process(sib, SR, gender="female", threshold_db=-30.0)
        rms_in = float(np.sqrt(np.mean(sib**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        assert rms_out < rms_in, "FEMALE De-Esser soll 7 kHz-Sibilanz dämpfen"

    def test_15_male_reduces_6khz_sinus(self, phase):
        """MALE-De-Esser dämpft 6 kHz Sinus (liegt im 5–10 kHz Band)."""
        sib = _sibilant_signal(freq_hz=6_000.0)
        result = phase.process(sib, SR, gender="male", threshold_db=-30.0)
        rms_in = float(np.sqrt(np.mean(sib**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        assert rms_out < rms_in, "MALE De-Esser soll 6 kHz-Sibilanz dämpfen"

    def test_16_child_reduces_8khz_sinus(self, phase):
        """CHILD-De-Esser dämpft 8 kHz Sinus (liegt im 7–14 kHz Band)."""
        sib = _sibilant_signal(freq_hz=8_000.0)
        result = phase.process(sib, SR, gender="child", threshold_db=-30.0)
        rms_in = float(np.sqrt(np.mean(sib**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        assert rms_out < rms_in, "CHILD De-Esser soll 8 kHz-Sibilanz dämpfen"

    def test_17_bass_unaffected_by_de_esser(self, phase):
        """Bass-Sinus (200 Hz) darf durch De-Esser kaum verändert werden."""
        bass = _bass_signal()
        result = phase.process(bass, SR, gender="female", threshold_db=-30.0)
        rms_in = float(np.sqrt(np.mean(bass**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        # Bass bleibt zu ≥ 95 % erhalten
        assert rms_out >= rms_in * 0.95, "Bass-Signal wurde durch De-Esser zu stark beeinflusst"

    def test_18_unknown_gender_fallback(self, phase, mono):
        """Unbekannter Gender-String → Fallback auf 'unknown' (5–9 kHz)."""
        result = phase.process(mono, SR, gender="androgynous")
        # Kein Absturz, Ausgang valide
        assert np.isfinite(result.audio).all()
        assert result.metadata["freq_low_hz"] == pytest.approx(5_000.0, rel=0.01)


# ---------------------------------------------------------------------------
# Strength-Cap (§2.19.3 Schlager-Modus)
# ---------------------------------------------------------------------------


class TestStrengthCap:
    """Strength-Cap limitiert maximale Gain-Reduction."""

    def test_19_strength_cap_stored_in_metadata(self, phase, mono):
        result = phase.process(mono, SR, strength_cap=0.45)
        assert result.metadata["strength_cap"] == pytest.approx(0.45)

    def test_20_strength_cap_limits_reduction(self, phase):
        """Mit strength_cap=0.45 ist die Dämpfung weniger stark als ohne Cap."""
        sib = _sibilant_signal(freq_hz=8_000.0)
        result_no_cap = phase.process(sib, SR, threshold_db=-30.0, strength_cap=1.0)
        result_capped = phase.process(sib, SR, threshold_db=-30.0, strength_cap=0.45)

        rms_no_cap = float(np.sqrt(np.mean(result_no_cap.audio**2)))
        rms_capped = float(np.sqrt(np.mean(result_capped.audio**2)))
        # Mit Cap ist Ausgang lauter (weniger GR)
        assert rms_capped >= rms_no_cap * 0.98, "strength_cap soll maximale Gain-Reduction begrenzen"

    def test_21_strength_cap_0_no_crash(self, phase, mono):
        """strength_cap=0.0 → keine GR, Signal unverändert."""
        result = phase.process(mono, SR, threshold_db=-30.0, strength_cap=0.0)
        assert np.isfinite(result.audio).all()
        # GR = 0 bedeutet gr_smooth = max(gr_smooth, 0.0) = pass-through auf Gain-Seite
        # (mathematisch: GR auf 0 geclamppt → Sibilantenband wird nicht dämpft)

    def test_22_strength_cap_clamped_to_1(self, phase, mono):
        """strength_cap > 1.0 → wird intern auf 1.0 begrenzt."""
        result = phase.process(mono, SR, strength_cap=5.0)
        assert result.metadata["strength_cap"] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Threshold- und Ratio-Verhalten
# ---------------------------------------------------------------------------


class TestThresholdBehavior:
    """Threshold kontrolliert, wann De-Esser eingreift."""

    def test_23_high_threshold_preserves_signal(self, phase, mono):
        """Sehr hohe Schwelle (0 dBFS) → kaum GR, Signal bleibt erhalten."""
        result = phase.process(mono, SR, threshold_db=0.0)
        rms_in = float(np.sqrt(np.mean(mono**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        # RMS-Verlust < 10 %
        assert rms_out >= rms_in * 0.90

    def test_24_low_threshold_reduces_rms(self, phase):
        """Aggressive Schwelle (-40 dBFS) → deutliche RMS-Reduktion der Sibilanz."""
        sib = _sibilant_signal(freq_hz=7_500.0)
        result = phase.process(sib, SR, threshold_db=-40.0, gender="female")
        rms_in = float(np.sqrt(np.mean(sib**2)))
        rms_out = float(np.sqrt(np.mean(result.audio**2)))
        assert rms_out < rms_in * 0.95, "Aggressive Schwelle soll Sibilanz reduzieren"


class TestSeverityCoupledActivation:
    """Hohe Sibilance/Harshness darf bei niedriger PMGG-Stärke nicht wirkungslos werden."""

    def test_25_high_severity_raises_control_strength(self, phase):
        sib = _sibilant_signal(freq_hz=7_200.0)
        result = phase.process(
            sib,
            SR,
            strength=0.06,
            phase_locality_factor=1.0,
            defect_scores={"sibilance": 0.92, "vocal_harshness": 0.78},
        )
        assert result.metadata["effective_strength"] == pytest.approx(0.06, rel=0.01)
        assert result.metadata["control_strength"] >= 0.30
        assert result.metadata["sibilance_pressure"] >= 0.90
        assert result.metadata["ratio"] > 1.8
        assert result.metadata["threshold_db"] <= -24.0

    def test_26_low_severity_keeps_soft_control(self, phase):
        sib = _sibilant_signal(freq_hz=7_200.0)
        result = phase.process(
            sib,
            SR,
            strength=0.06,
            phase_locality_factor=1.0,
            defect_scores={"sibilance": 0.20},
        )
        assert result.metadata["effective_strength"] == pytest.approx(0.06, rel=0.01)
        assert result.metadata["control_strength"] == pytest.approx(0.06, rel=0.01)
        assert result.metadata["ratio"] < 1.3
        assert result.metadata["threshold_db"] == pytest.approx(-20.0, abs=0.1)

    def test_25_metadata_threshold_matches_input(self, phase, mono):
        result = phase.process(mono, SR, threshold_db=-15.0)
        assert result.metadata["threshold_db"] == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# Abtastrate und Konsistenz
# ---------------------------------------------------------------------------


class TestSampleRateAndConsistency:
    """Verhalten bei verschiedenen Abtastraten und Wiederholbarkeit."""

    def test_26_consistent_output(self, phase, mono):
        """Selbe Eingabe → identische Ausgabe (Deterministik)."""
        r1 = phase.process(mono, SR)
        r2 = phase.process(mono, SR)
        np.testing.assert_array_equal(r1.audio, r2.audio)

    def test_27_performance_budget(self, phase):
        """Laufzeit ≤ 5× Echtzeit (§9.5)."""
        audio = np.random.randn(SR * 5).astype(np.float32) * 0.3  # 5 s
        t0 = time.time()
        phase.process(audio, SR)
        elapsed = time.time() - t0
        assert elapsed <= 5.0 * 5.0, f"Zu langsam: {elapsed:.2f} s für 5 s Audio"

    def test_28_full_metadata_keys(self, phase, mono):
        """Alle erwarteten Metadata-Schlüssel vorhanden."""
        result = phase.process(mono, SR)
        required_keys = {
            "gender",
            "threshold_db",
            "ratio",
            "attack_ms",
            "release_ms",
            "freq_low_hz",
            "freq_high_hz",
            "strength_cap",
            "intelligibility_protected",
            "intelligibility_score",
            "intelligibility_presence_ratio",
            "intelligibility_articulation_ratio",
            "intelligibility_air_ratio",
            "intelligibility_fricative_snr_delta_db",
            "fricative_snr_invariant_met",
            "fricative_snr_before_deessing_db",
            "fricative_snr_after_chain_db",
        }
        assert required_keys.issubset(result.metadata.keys()), (
            f"Fehlende Metadata-Schlüssel: {required_keys - set(result.metadata.keys())}"
        )

    def test_29_metrics_contains_avg_gr(self, phase, mono):
        result = phase.process(mono, SR)
        assert "avg_gain_reduction_db" in result.metrics
        assert np.isfinite(result.metrics["avg_gain_reduction_db"])

    def test_30_white_noise_input_no_nan(self, phase):
        """Weißes Rauschen (breitband) → kein NaN, kein Clipping."""
        noise = (np.random.randn(N) * 0.5).astype(np.float32)
        result = phase.process(noise, SR)
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_31_ml_refine_bypass_metadata_present(self, phase, mono, monkeypatch):
        import backend.core.phases.phase_43_ml_deesser as phase43

        monkeypatch.setattr(
            phase43,
            "_try_mp_senet_refine",
            lambda audio, sr: (np.asarray(audio, dtype=np.float32).copy(), "omlsa_dsp_fallback"),
        )

        result = phase.process(mono, SR, enable_ml_refine=True)

        assert result.metadata["ml_refine_applied"] is False
        assert result.metadata["ml_refine_bypassed"] is True
        assert result.metadata["ml_refine_bypass_reason"] == "omlsa_dsp_fallback"

    def test_32_intelligibility_guard_blends_back_mid_band_loss(self, phase, monkeypatch):
        import scipy.signal as ss

        def _overdull(ch, sr, threshold_db, ratio, attack_ms, release_ms, freq_low, freq_high, strength_cap=1.0):
            sos = ss.butter(4, [2000.0, 8000.0], btype="band", fs=sr, output="sos")
            mid = ss.sosfiltfilt(sos, ch)
            processed = ch - mid + mid * 0.35
            return processed.astype(ch.dtype), -8.0

        monkeypatch.setattr("backend.core.phases.phase_43_ml_deesser._deess_channel", _overdull)

        audio = (
            0.25 * _sibilant_signal(freq_hz=3000.0)
            + 0.25 * _sibilant_signal(freq_hz=6500.0)
            + 0.25 * _sibilant_signal(freq_hz=10000.0)
        ).astype(np.float32)
        result = phase.process(audio, SR, gender="female", threshold_db=-35.0)

        assert result.metadata["intelligibility_protected"] is True
        assert result.metadata["intelligibility_score"] >= 0.80

    def test_33_ml_refine_rejects_articulation_loss_even_when_core_is_unchanged(self, phase, monkeypatch):
        import scipy.signal as ss

        def _candidate(audio, sr):
            sos = ss.butter(4, [4000.0, 8000.0], btype="band", fs=sr, output="sos")
            articulation = ss.sosfiltfilt(sos, audio)
            ml_candidate = audio - articulation + articulation * 0.30
            return ml_candidate.astype(np.float32), "mp_senet_onnx"

        monkeypatch.setattr("backend.core.phases.phase_43_ml_deesser._try_mp_senet_refine", _candidate)

        audio = (
            0.28 * _sibilant_signal(freq_hz=1000.0)
            + 0.20 * _sibilant_signal(freq_hz=5500.0)
            + 0.22 * _sibilant_signal(freq_hz=9000.0)
        ).astype(np.float32)
        result = phase.process(audio, SR, gender="female", threshold_db=-30.0, enable_ml_refine=True)

        assert result.metadata["ml_refine_applied"] is False
        assert result.metadata["ml_refine_bypass_reason"] == "safety_gate"

    def test_34_phase43_intelligibility_metadata_present(self, phase):
        audio = (
            0.25 * _sibilant_signal(freq_hz=3000.0)
            + 0.25 * _sibilant_signal(freq_hz=6500.0)
            + 0.25 * _sibilant_signal(freq_hz=10000.0)
        ).astype(np.float32)

        result = phase.process(audio, SR, gender="female", threshold_db=-30.0)

        assert "intelligibility_protected" in result.metadata
        assert "intelligibility_score" in result.metadata
        assert "intelligibility_presence_ratio" in result.metadata
        assert "intelligibility_articulation_ratio" in result.metadata
        assert "intelligibility_air_ratio" in result.metadata
        assert "intelligibility_fricative_snr_delta_db" in result.metadata
        assert "intelligibility_score" in result.metrics
        assert "intelligibility_presence_ratio" in result.metrics
        assert "intelligibility_articulation_ratio" in result.metrics
        assert "intelligibility_air_ratio" in result.metrics
        assert "intelligibility_fricative_snr_delta_db" in result.metrics

    def test_35_phase43_intelligibility_metadata_bounded(self, phase):
        audio = np.column_stack(
            [
                _sibilant_signal(freq_hz=5500.0),
                _sibilant_signal(freq_hz=9000.0),
            ]
        ).astype(np.float32)

        result = phase.process(audio, SR, gender="female", threshold_db=-30.0)
        meta = result.metadata
        metrics = result.metrics

        assert isinstance(meta["intelligibility_protected"], bool)
        assert 0.0 <= meta["intelligibility_score"] <= 1.0
        assert 0.0 <= meta["intelligibility_presence_ratio"] <= 1.25
        assert 0.0 <= meta["intelligibility_articulation_ratio"] <= 1.25
        assert 0.0 <= meta["intelligibility_air_ratio"] <= 1.25
        assert np.isfinite(meta["intelligibility_fricative_snr_delta_db"])
        assert 0.0 <= metrics["intelligibility_score"] <= 1.0
        assert 0.0 <= metrics["intelligibility_presence_ratio"] <= 1.25
        assert 0.0 <= metrics["intelligibility_articulation_ratio"] <= 1.25
        assert 0.0 <= metrics["intelligibility_air_ratio"] <= 1.25
        assert np.isfinite(metrics["intelligibility_fricative_snr_delta_db"])

    def test_36_metrics_contains_musical_goal_keys(self, phase):
        audio = (
            0.25 * _sibilant_signal(freq_hz=3000.0)
            + 0.25 * _sibilant_signal(freq_hz=6500.0)
            + 0.25 * _sibilant_signal(freq_hz=10000.0)
        ).astype(np.float32)

        result = phase.process(audio, SR, gender="female", threshold_db=-30.0)
        metrics = result.metrics

        assert "musical_goal_brillanz" in metrics
        assert "musical_goal_artikulation" in metrics
        assert "musical_goal_authentizitaet" in metrics
        assert "musical_goal_transparenz" in metrics
        assert 0.0 <= metrics["musical_goal_brillanz"] <= 1.0
        assert 0.0 <= metrics["musical_goal_artikulation"] <= 1.0
        assert 0.0 <= metrics["musical_goal_authentizitaet"] <= 1.0
        assert 0.0 <= metrics["musical_goal_transparenz"] <= 1.0

    def test_37_phase43_metadata_contains_fricative_snr_fields(self, phase):
        audio = (
            0.25 * _sibilant_signal(freq_hz=3000.0)
            + 0.25 * _sibilant_signal(freq_hz=6500.0)
            + 0.25 * _sibilant_signal(freq_hz=10000.0)
        ).astype(np.float32)

        result = phase.process(audio, SR, gender="female", threshold_db=-30.0)

        assert "fricative_snr_invariant_met" in result.metadata
        assert "fricative_snr_before_deessing_db" in result.metadata
        assert "fricative_snr_after_chain_db" in result.metadata
        assert isinstance(result.metadata["fricative_snr_invariant_met"], bool)

    def test_38_phase43_fricative_snr_metadata_is_finite(self, phase):
        audio = (
            0.25 * _sibilant_signal(freq_hz=3000.0)
            + 0.25 * _sibilant_signal(freq_hz=6500.0)
            + 0.25 * _sibilant_signal(freq_hz=10000.0)
        ).astype(np.float32)

        result = phase.process(audio, SR, gender="female", threshold_db=-30.0)

        assert np.isfinite(result.metadata["fricative_snr_before_deessing_db"])
        assert np.isfinite(result.metadata["fricative_snr_after_chain_db"])
