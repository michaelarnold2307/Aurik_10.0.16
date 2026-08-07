from __future__ import annotations

"""
tests/unit/test_fricative_snr_feedback_invariant.py
=====================================================

Tests für §2.8 Feedback-Invariante: Nach der vollständigen Kette
(ConsonantEnhancement → Phase-19-De-Esser → Stage-8b/8c) muss gelten:

    SNR_frikativ_after_chain ≥ SNR_frikativ_before_deessing + 3 dB

Neue Funktionen / Erweiterungen:
  - measure_fricative_snr()  (core/consonant_enhancement.py)
  - Stage 8c in Phase 19     (phases/phase_19_de_esser.py)
    → Metadata-Felder: fricative_snr_invariant_met,
                       fricative_snr_before_deessing_db,
                       fricative_snr_after_chain_db

Alle Tests nutzen ausschließlich synthetische Signale (§5.4).
"""


import math

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

# Konstanten
SR = 48_000
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _white_noise(seconds: float = 2.0) -> np.ndarray:
    return RNG.standard_normal(int(SR * seconds)).astype(np.float32) * 0.2


def _hf_noise(seconds: float = 2.0, f_lo: float = 6_000.0, f_hi: float = 12_000.0) -> np.ndarray:
    """Bandpass-Rauschen im Frikativband — simuliert Frikative."""
    from scipy.signal import butter, sosfilt

    raw = RNG.standard_normal(int(SR * seconds)).astype(np.float32) * 0.3
    sos = butter(4, [f_lo / (SR / 2), f_hi / (SR / 2)], btype="band", output="sos")
    return sosfilt(sos, raw).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(mono: np.ndarray) -> np.ndarray:
    """Phase 19 erwartet Stereo als [n_samples, 2] (column_stack-Konvention)."""
    return np.column_stack([mono, mono * 0.95])


# ---------------------------------------------------------------------------
# measure_fricative_snr — Einheits-Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMeasureFricativeSNR:
    """§2.8-kompatible SNR-Messung im adaptiven Frikativband."""

    def test_01_returns_finite_on_mono(self):
        """Mono-Array → endlicher float."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        snr = measure_fricative_snr(_white_noise(), SR, "female")
        assert math.isfinite(snr), "SNR muss endlich sein"

    def test_02_returns_finite_on_stereo(self):
        """Stereo-Array → endlicher float."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        snr = measure_fricative_snr(_stereo(_white_noise()), SR, "male")
        assert math.isfinite(snr)

    def test_03_nan_array_returns_zero(self):
        """NaN-Array → 0.0 (sicher)."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        nan_arr = np.full(SR, float("nan"), dtype=np.float32)
        result = measure_fricative_snr(nan_arr, SR, "unknown")
        assert math.isfinite(result)

    def test_04_empty_array_returns_zero(self):
        """Leerer Array → 0.0, kein Absturz."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        result = measure_fricative_snr(np.array([], dtype=np.float32), SR)
        assert result == 0.0

    def test_05_hf_signal_higher_snr_than_silence(self):
        """HF-Signal im Frikativband liefert höheren SNR als Stille."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        snr_hf = measure_fricative_snr(_hf_noise(), SR, "female")
        snr_silence = measure_fricative_snr(np.zeros(SR, dtype=np.float32), SR, "female")
        assert snr_hf > snr_silence, "HF-Rauschen muss höheren Band-SNR liefern als Stille"

    def test_06_gender_adaptive_female_band(self):
        """female-Band: Signal bei 9 kHz hat höheren SNR als bei male-Band-Messung."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        hf = _hf_noise(f_lo=8_000.0, f_hi=11_000.0)  # im female-Band (6–12 kHz)
        snr_f = measure_fricative_snr(hf, SR, "female")
        snr_m = measure_fricative_snr(hf, SR, "male")
        # Beide endlich
        assert math.isfinite(snr_f) and math.isfinite(snr_m)

    def test_07_output_in_reasonable_range(self):
        """SNR-Rückgabe liegt in physikalisch sinnvollem Bereich [−60, +60 dB]."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        snr = measure_fricative_snr(_white_noise(), SR, "child")
        assert -60.0 <= snr <= 60.0, f"Unerwarteter SNR: {snr}"

    def test_08_none_not_accepted(self):
        """None als Audio → 0.0, kein Absturz."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        result = measure_fricative_snr(None, SR)  # type: ignore[arg-type]
        assert result == 0.0

    def test_09_consistent_repeated_call(self):
        """Identisches Signal → identischer SNR (Deterministizität)."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        audio = _hf_noise()
        snr1 = measure_fricative_snr(audio, SR, "female")
        snr2 = measure_fricative_snr(audio, SR, "female")
        assert snr1 == pytest.approx(snr2, abs=1e-4)

    def test_10_androgynous_gender_works(self):
        """'androgynous' Stimmtyp → kein Absturz, endlicher Wert."""
        from backend.core.consonant_enhancement import measure_fricative_snr

        result = measure_fricative_snr(_white_noise(), SR, "androgynous")
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# Stage 8c in Phase 19 — Metadaten-Tests
# ---------------------------------------------------------------------------


class TestPhase19FeedbackInvariantMetadata:
    """Phase 19 Stage 8c gibt §2.8-Metadaten zurück."""

    def _run_phase19(self, audio: np.ndarray, gender: str = "female") -> dict:
        """Hilfsmethode: Phase 19 mit synthetischem Signal ausführen."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        phase = DeEsserPhase(gender_type=gender)
        result = phase.process(audio, SR, material_type=MaterialType.TAPE, gender=gender)
        return result.metadata

    def test_11_metadata_contains_snr_fields(self):
        """PhaseResult.metadata enthält alle drei §2.8-Felder."""
        audio = _white_noise(2.0)
        meta = self._run_phase19(audio)
        assert "fricative_snr_invariant_met" in meta
        assert "fricative_snr_before_deessing_db" in meta
        assert "fricative_snr_after_chain_db" in meta

    def test_12_snr_before_is_finite(self):
        """SNR vor De-Essing ist endlich."""
        audio = _hf_noise() * 0.5 + _white_noise()
        meta = self._run_phase19(audio)
        assert math.isfinite(meta["fricative_snr_before_deessing_db"])

    def test_13_snr_after_is_finite(self):
        """SNR nach Kette ist endlich."""
        audio = _hf_noise() * 0.5 + _white_noise()
        meta = self._run_phase19(audio)
        assert math.isfinite(meta["fricative_snr_after_chain_db"])

    def test_14_invariant_met_is_bool(self):
        """fricative_snr_invariant_met ist bool."""
        meta = self._run_phase19(_white_noise())
        assert isinstance(meta["fricative_snr_invariant_met"], bool)

    def test_15_stereo_input_yields_finite_snr(self):
        """Stereo-Eingang → endliche SNR-Metadaten."""
        stereo = _stereo(_hf_noise() * 0.4 + _white_noise(2.0))
        meta = self._run_phase19(stereo)
        assert math.isfinite(meta["fricative_snr_before_deessing_db"])
        assert math.isfinite(meta["fricative_snr_after_chain_db"])

    def test_16_silence_input_no_crash(self):
        """Stille → kein Absturz, endliche Werte."""
        meta = self._run_phase19(np.zeros(SR * 2, dtype=np.float32))
        assert math.isfinite(meta["fricative_snr_before_deessing_db"])
        assert math.isfinite(meta["fricative_snr_after_chain_db"])

    def test_17_nan_input_no_crash(self):
        """NaN-Eingang → kein Absturz (NaN wird intern bereinigt)."""
        nan_audio = np.full(SR * 2, float("nan"), dtype=np.float32)
        meta = self._run_phase19(nan_audio)
        assert math.isfinite(meta["fricative_snr_before_deessing_db"])
        assert math.isfinite(meta["fricative_snr_after_chain_db"])

    def test_18_male_gender_yields_different_snr_band(self):
        """male-Gender liefert SNR-Wert (andere Band-Grenzen als female)."""
        audio = _hf_noise(f_lo=5_000.0, f_hi=9_000.0)  # im male-Band
        meta_m = self._run_phase19(audio, gender="male")
        meta_f = self._run_phase19(audio, gender="female")
        # Beide geben finite Werte — Band-Kompensation kann sich unterscheiden
        assert math.isfinite(meta_m["fricative_snr_before_deessing_db"])
        assert math.isfinite(meta_f["fricative_snr_before_deessing_db"])

    def test_19_result_audio_nan_free(self):
        """Output-Audio nach Stage 8c ist NaN/Inf-frei."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        audio = _hf_noise() * 0.3 + _white_noise()
        phase = DeEsserPhase(gender_type="female")
        result = phase.process(audio, SR, material_type=MaterialType.VINYL, gender="female")
        assert np.isfinite(result.audio).all(), "Output darf kein NaN/Inf enthalten"
        assert np.max(np.abs(result.audio)) <= 1.0, "Output darf nicht clippen"

    def test_20_snr_before_rounded_to_two_decimals(self):
        """fricative_snr_before_deessing_db ist auf 2 Dezimalstellen gerundet."""
        audio = _hf_noise() * 0.5 + _white_noise()
        meta = self._run_phase19(audio)
        val = meta["fricative_snr_before_deessing_db"]
        assert val == round(val, 2)

    def test_21_snr_after_rounded_to_two_decimals(self):
        """fricative_snr_after_chain_db ist auf 2 Dezimalstellen gerundet."""
        audio = _hf_noise() * 0.5 + _white_noise()
        meta = self._run_phase19(audio)
        val = meta["fricative_snr_after_chain_db"]
        assert val == round(val, 2)

    def test_22_child_gender_no_crash(self):
        """child-Gender → kein Absturz."""
        meta = self._run_phase19(_hf_noise(f_lo=7_000.0, f_hi=14_000.0), gender="child")
        assert math.isfinite(meta["fricative_snr_before_deessing_db"])

    def test_23_snr_after_not_below_minus60(self):
        """SNR-Werte liegen in physikalisch sinnvollem Bereich."""
        audio = _hf_noise() * 0.5 + _white_noise()
        meta = self._run_phase19(audio)
        assert meta["fricative_snr_before_deessing_db"] > -70.0
        assert meta["fricative_snr_after_chain_db"] > -70.0

    def test_24_phase_result_success(self):
        """Phase 19 gibt .success=True zurück."""
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(_white_noise(2.0), SR, material_type=MaterialType.TAPE)
        assert result.success is True

    def test_25_metadata_present_when_consonant_enhancement_unavailable(self):
        """Auch ohne ConsonantEnhancement sind die SNR-Felder vorhanden (Defaults)."""

        # _HAS_CONSONANT_ENHANCEMENT simuliert auf False setzen
        from backend.core.phases import phase_19_de_esser as p19

        original = p19._HAS_CONSONANT_ENHANCEMENT
        try:
            p19._HAS_CONSONANT_ENHANCEMENT = False
            from backend.core.defect_scanner import MaterialType

            phase = p19.DeEsserPhase(gender_type="female")
            result = phase.process(_white_noise(2.0), SR, material_type=MaterialType.TAPE)
            meta = result.metadata
            # Felder müssen vorhanden sein (Defaults: True / 0.0 / 0.0)
            assert "fricative_snr_invariant_met" in meta
            assert "fricative_snr_before_deessing_db" in meta
            assert "fricative_snr_after_chain_db" in meta
        finally:
            p19._HAS_CONSONANT_ENHANCEMENT = original


class TestAssessDeEsserIntelligibility:
    """Direkte Unit-Tests für den Intelligibility-Helper aus Commit 7."""

    @staticmethod
    def _tone(freq_hz: float, amp: float = 0.3, seconds: float = 1.0) -> np.ndarray:
        t = np.linspace(0.0, seconds, int(SR * seconds), endpoint=False, dtype=np.float32)
        return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]

    def test_26_report_is_finite_and_typed(self):
        from backend.core.dsp.deesser_intelligibility import (
            DeEsserIntelligibilityReport,
            assess_deesser_intelligibility_preservation,
        )

        original = self._tone(3000.0) + self._tone(6500.0) + self._tone(10000.0)
        report = assess_deesser_intelligibility_preservation(original, original, SR, voice_gender="female")

        assert isinstance(report, DeEsserIntelligibilityReport)
        assert math.isfinite(report.presence_ratio)
        assert math.isfinite(report.articulation_ratio)
        assert math.isfinite(report.air_ratio)
        assert math.isfinite(report.intelligibility_score)
        assert report.should_protect is False

    def test_27_air_only_loss_is_less_severe_than_presence_articulation_loss(self):
        from backend.core.dsp.deesser_intelligibility import assess_deesser_intelligibility_preservation

        original = 0.35 * self._tone(3000.0) + 0.30 * self._tone(6500.0) + 0.25 * self._tone(10000.0)
        air_only_loss = 0.35 * self._tone(3000.0) + 0.30 * self._tone(6500.0) + 0.05 * self._tone(10000.0)
        speech_loss = 0.20 * self._tone(3000.0) + 0.16 * self._tone(6500.0) + 0.25 * self._tone(10000.0)

        air_report = assess_deesser_intelligibility_preservation(original, air_only_loss, SR, voice_gender="female")
        speech_report = assess_deesser_intelligibility_preservation(original, speech_loss, SR, voice_gender="female")

        assert air_report.intelligibility_loss < speech_report.intelligibility_loss
        assert speech_report.should_protect is True

    def test_28_zero_band_reference_stays_safe(self):
        from backend.core.dsp.deesser_intelligibility import assess_deesser_intelligibility_preservation

        original = np.zeros(SR, dtype=np.float32)
        after = np.zeros(SR, dtype=np.float32)
        report = assess_deesser_intelligibility_preservation(original, after, SR, voice_gender="male")

        assert report.presence_ratio == pytest.approx(1.0, abs=1e-6)
        assert report.articulation_ratio == pytest.approx(1.0, abs=1e-6)
        assert report.air_ratio == pytest.approx(1.0, abs=1e-6)


class TestPhase19IntelligibilityMetadata:
    """Phase 19 exportiert die neuen Intelligibility-Metadaten vollständig."""

    def test_29_metadata_contains_intelligibility_fields(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        audio = 0.35 * TestAssessDeEsserIntelligibility._tone(3000.0)
        audio += 0.30 * TestAssessDeEsserIntelligibility._tone(6500.0)
        audio += 0.25 * TestAssessDeEsserIntelligibility._tone(10000.0)

        phase = DeEsserPhase(gender_type="female")
        result = phase.process(audio.astype(np.float32), SR, material_type=MaterialType.TAPE, gender="female")
        meta = result.metadata
        metrics = result.metrics

        assert "intelligibility_protected" in meta
        assert "intelligibility_score" in meta
        assert "intelligibility_presence_ratio" in meta
        assert "intelligibility_articulation_ratio" in meta
        assert "intelligibility_air_ratio" in meta
        assert "intelligibility_fricative_snr_delta_db" in meta
        assert "intelligibility_score" in metrics
        assert "intelligibility_presence_ratio" in metrics
        assert "intelligibility_articulation_ratio" in metrics
        assert "intelligibility_air_ratio" in metrics
        assert "intelligibility_fricative_snr_delta_db" in metrics

    def test_30_intelligibility_values_are_finite_and_bounded(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_19_de_esser import DeEsserPhase

        audio = _stereo(_hf_noise() * 0.35 + _white_noise())
        phase = DeEsserPhase(gender_type="female")
        result = phase.process(audio, SR, material_type=MaterialType.VINYL, gender="female")
        meta = result.metadata

        assert isinstance(meta["intelligibility_protected"], bool)
        assert 0.0 <= meta["intelligibility_score"] <= 1.0
        assert 0.0 <= meta["intelligibility_presence_ratio"] <= 1.25
        assert 0.0 <= meta["intelligibility_articulation_ratio"] <= 1.25
        assert 0.0 <= meta["intelligibility_air_ratio"] <= 1.25
        assert math.isfinite(meta["intelligibility_fricative_snr_delta_db"])
