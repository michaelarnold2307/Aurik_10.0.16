from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as signal

from backend.core.causal_defect_reasoner import CAUSE_PARAMS, CAUSE_TO_PHASES, CAUSES, LIKELIHOOD_FNS, MATERIAL_PRIORS
from backend.core.defect_phase_mapper import DefectPhaseMapper
from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

SR = 48_000


def _harmonic_stack(freq: float = 330.0, secs: float = 5.0) -> np.ndarray:
    t = np.linspace(0.0, secs, int(SR * secs), endpoint=False, dtype=np.float32)
    audio = np.zeros_like(t)
    for multiple, amp in [(1, 0.22), (2, 0.12), (3, 0.08), (4, 0.05), (5, 0.03)]:
        audio += amp * np.sin(2.0 * np.pi * (freq * multiple) * t)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _scrape_flutter_signal(secs: float = 5.0, carrier_hz: float = 880.0, scrape_rate_hz: float = 85.0) -> np.ndarray:
    t = np.linspace(0.0, secs, int(SR * secs), endpoint=False, dtype=np.float32)
    audio = 0.28 * np.sin(2.0 * np.pi * carrier_hz * t)
    audio += 0.10 * np.sin(2.0 * np.pi * (carrier_hz - scrape_rate_hz) * t)
    audio += 0.10 * np.sin(2.0 * np.pi * (carrier_hz + scrape_rate_hz) * t)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]


def _head_clog_signal(secs: float = 5.0) -> np.ndarray:
    audio = _harmonic_stack(freq=280.0, secs=secs).astype(np.float32)
    lowpass_sos = signal.butter(4, 1400.0, btype="lowpass", fs=SR, output="sos")
    n_event = int(0.24 * SR)
    fade = np.ones(max(8, n_event), dtype=np.float32)
    for start_s in [0.9, 2.1, 3.35]:
        start = int(start_s * SR)
        end = start + n_event
        if end > len(audio):
            continue
        clogged = signal.sosfiltfilt(lowpass_sos, audio[start:end]).astype(np.float32)
        mix = fade.astype(np.float32)
        audio[start:end] = audio[start:end] * (1.0 - mix) + clogged * mix
    return np.clip(audio, -1.0, 1.0)


@pytest.mark.unit
class TestTapeSpecialistDefectScanner:
    def test_material_sensitivity_contains_new_tape_specialists(self) -> None:
        scanner = DefectScanner(sample_rate=SR)

        for material in MaterialType:
            assert DefectType.SCRAPE_FLUTTER in scanner.MATERIAL_SENSITIVITY[material]
            assert DefectType.TAPE_HEAD_CLOG in scanner.MATERIAL_SENSITIVITY[material]

    def test_clean_tape_signal_keeps_new_scores_low(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        audio = _harmonic_stack(freq=220.0, secs=4.0)

        result = scanner.scan(audio, SR, material_type=MaterialType.TAPE)

        assert float(result.scores[DefectType.SCRAPE_FLUTTER].severity) < 0.20
        assert float(result.scores[DefectType.TAPE_HEAD_CLOG].severity) < 0.30

    def test_scrape_flutter_detects_high_rate_sidebands(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        result = scanner.scan(_scrape_flutter_signal(), SR, material_type=MaterialType.CASSETTE)
        score = result.scores[DefectType.SCRAPE_FLUTTER]

        assert float(score.severity) > 0.12
        assert int(score.metadata.get("n_scrape_sideband_pairs", 0)) >= 1
        assert 35.0 <= float(score.metadata.get("dominant_scrape_rate_hz", 0.0)) <= 120.0

    def test_tape_head_clog_detects_local_hf_dropouts(self) -> None:
        scanner = DefectScanner(sample_rate=SR)
        result = scanner.scan(_head_clog_signal(), SR, material_type=MaterialType.REEL_TAPE)
        score = result.scores[DefectType.TAPE_HEAD_CLOG]

        assert float(score.severity) > 0.10
        assert int(score.metadata.get("clog_event_count", 0)) >= 2
        assert float(score.metadata.get("mean_hf_drop_db", 0.0)) >= 6.0


class TestTapeSpecialistReasonerAndMapper:
    def test_new_causes_are_fully_wired(self) -> None:
        for cause in ["scrape_flutter", "tape_head_clog"]:
            assert cause in CAUSES
            assert cause in CAUSE_TO_PHASES
            assert cause in CAUSE_PARAMS
            assert cause in LIKELIHOOD_FNS
            assert "tape" in MATERIAL_PRIORS
            assert cause in MATERIAL_PRIORS["tape"]

    def test_mapper_routes_new_defects_to_expected_primary_phases(self) -> None:
        mapper = DefectPhaseMapper()

        assert "phase_12_wow_flutter_fix" in mapper.get_primary_phases(DefectType.SCRAPE_FLUTTER)
        assert "phase_56_spectral_band_gap_repair" in mapper.get_primary_phases(DefectType.TAPE_HEAD_CLOG)

    def test_cause_to_phase_order_is_transport_then_repair(self) -> None:
        assert CAUSE_TO_PHASES["scrape_flutter"][0] == "phase_12_wow_flutter_fix"
        assert CAUSE_TO_PHASES["tape_head_clog"][0] == "phase_56_spectral_band_gap_repair"
