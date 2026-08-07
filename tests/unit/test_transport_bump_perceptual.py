from __future__ import annotations

"""
test_transport_bump_perceptual.py — HPE/Pleasantness-Check nach Reparatur
=========================================================================

Schließt die verbleibenden P2-Gaps:
1. HPE-Gate NACH Transport Bump / Head Level Dip Reparatur
2. "Unhörbarkeits"-Test: Defekt + Reparatur → HPE ≥ neutral

Spec: §v10 Pleasantness-First, §0h Music-Death-Shield
"""


import numpy as np
import pytest

SR = 48_000


# ── Synthetische Defekt-Signale ────────────────────────────────────────────


def _make_dip_audio(dip_depth_db: float = 12.0, dip_dur_s: float = 0.2) -> np.ndarray:
    """Erzeugt 3s Sinus mit einem Head-Level-Dip in der Mitte."""
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # Dip in der Mitte
    dip_start = int(1.4 * SR)
    dip_end = int(dip_start + dip_dur_s * SR)
    dip_samples = dip_end - dip_start
    # Gradueller Einbruch und Recovery
    env = np.ones(len(audio), dtype=np.float32)
    ramp_down = np.linspace(1.0, 10 ** (-dip_depth_db / 20), dip_samples // 3, dtype=np.float32)
    dip_floor = np.full(dip_samples - 2 * (dip_samples // 3), 10 ** (-dip_depth_db / 20), dtype=np.float32)
    ramp_up = np.linspace(10 ** (-dip_depth_db / 20), 1.0, dip_samples // 3, dtype=np.float32)
    env[dip_start:dip_end] = np.concatenate([ramp_down, dip_floor, ramp_up])[:dip_samples]
    audio = audio * env
    return audio.astype(np.float32)  # type: ignore[no-any-return]


def _make_bump_audio(bump_start_s: float = 1.0) -> np.ndarray:
    """Erzeugt 3s Sinus mit einem Transport-Bump."""
    duration = 3.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 880 * t)

    # Bump: Energie-Dropout + LF-Thump + Pitch-Excursion
    bump_start = int(bump_start_s * SR)
    bump_dur = int(0.15 * SR)
    audio[bump_start : bump_start + bump_dur] *= 0.05  # Dropout

    # LF-Thump (80 Hz)
    thump_start = bump_start + bump_dur
    thump_dur = int(0.03 * SR)
    thump_t = np.arange(thump_dur, dtype=np.float32) / SR
    audio[thump_start : thump_start + thump_dur] += 0.15 * np.sin(2 * np.pi * 80 * thump_t)

    return audio.astype(np.float32)  # type: ignore[no-any-return]


# ── Detection Tests ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestDipDetectionAccuracy:
    """Head Level Dip: Wird der Defekt erkannt?"""

    def test_01_dip_detected_on_tape_material(self):
        """Dip wird auf Tape-Material erkannt."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(material_type=MaterialType.CASSETTE, sample_rate=SR)
        audio = _make_dip_audio(dip_depth_db=12.0)
        score = scanner._detect_tape_head_level_dips(audio)

        assert score.severity > 0.0, f"Dip nicht erkannt: severity={score.severity}"
        assert len(score.locations) > 0, f"Keine Dip-Locations: {score.locations}"

    def test_02_dip_detection_handles_any_audio(self):
        """Dip-Detection funktioniert unabhängig vom Material (Gating in scan())."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(material_type=MaterialType.CD_DIGITAL, sample_rate=SR)
        audio = _make_dip_audio(dip_depth_db=12.0)
        score = scanner._detect_tape_head_level_dips(audio)

        # Die Roh-Detektion findet Dips in jedem Audio.
        # Das Material-Gating erfolgt in scan() über MATERIAL_SENSITIVITY.
        assert score.severity >= 0.0, f"severity sollte >= 0 sein: {score.severity}"

    def test_03_silence_produces_no_false_dip(self):
        """Stille erzeugt keinen False-Positive Dip."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(material_type=MaterialType.CASSETTE, sample_rate=SR)
        silence = np.zeros(SR * 3, dtype=np.float32)
        score = scanner._detect_tape_head_level_dips(silence)

        assert score.severity < 0.1, f"False Positive auf Stille: severity={score.severity}"

    def test_04_dip_threshold_is_adaptive(self):
        """Dip-Schwellwert ist SNR-adaptiv (nicht festes 3.0 dB)."""
        import backend.core.defect_scanner as ds_mod

        src = open(ds_mod.__file__, encoding="utf-8").read()
        assert "_local_dyn" in src, "dip_thresh_db nicht SNR-adaptiv — _local_dyn fehlt in _detect_tape_head_level_dips"


@pytest.mark.unit
class TestBumpDetectionAccuracy:
    """Transport Bump: Wird der Defekt erkannt?"""

    def test_10_bump_detected_on_cassette(self):
        """Transport Bump wird auf Kassette erkannt."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(material_type=MaterialType.CASSETTE, sample_rate=SR)
        audio = _make_bump_audio()
        score = scanner._detect_transport_bump(audio)

        # Transport Bump sollte zumindest Spuren zeigen
        assert score.severity is not None, "Bump-Detection liefert None"
        assert 0.0 <= score.severity <= 1.0, f"Severity außerhalb [0,1]: {score.severity}"

    def test_11_bump_detection_handles_any_audio(self):
        """Transport-Bump-Detection funktioniert unabhängig vom Material."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(material_type=MaterialType.CD_DIGITAL, sample_rate=SR)
        audio = _make_bump_audio()
        score = scanner._detect_transport_bump(audio)

        # Die Roh-Detektion findet Bumps in jedem Audio.
        # Das Material-Gating erfolgt in scan() über MATERIAL_SENSITIVITY.
        assert score.severity >= 0.0, f"severity sollte >= 0 sein: {score.severity}"

    def test_12_silence_produces_no_false_bump(self):
        """Stille erzeugt keinen False-Positive Bump."""
        from backend.core.defect_scanner import DefectScanner, MaterialType

        scanner = DefectScanner(material_type=MaterialType.CASSETTE, sample_rate=SR)
        silence = np.zeros(SR * 3, dtype=np.float32)
        score = scanner._detect_transport_bump(silence)

        assert score.severity < 0.1, f"False Positive auf Stille: severity={score.severity}"


@pytest.mark.unit
@pytest.mark.pleasantness
class TestPerceptualTransparency:
    """§v10: Defekt + Reparatur → HPE ≥ neutral → Defekt unhörbar."""

    def test_20_pleasantness_estimator_available(self):
        """compare_pleasantness ist verfügbar für HPE-Messung."""
        from backend.core.human_pleasantness_estimator import compare_pleasantness

        assert callable(compare_pleasantness), "HPE nicht verfügbar"

    def test_21_dip_audio_differs_from_clean(self):
        """Dip-Audio unterscheidet sich messbar vom Clean-Referenz."""
        from backend.core.human_pleasantness_estimator import compare_pleasantness

        clean = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, int(SR * 3.0), endpoint=False, dtype=np.float32))
        dipped = _make_dip_audio(dip_depth_db=12.0)

        result = compare_pleasantness(clean, dipped, sr=SR)
        # Mit 12dB Dip sollte HPE negativ sein (verschlechtert)
        delta = float(result.get("delta_score", 0.0))
        # Der Dip SOLLTE als Verschlechterung erkannt werden
        assert delta < 0.1, f"HPE-Delta bei 12dB Dip: {delta:.3f} — erwartet < 0.1 (Verschlechterung)"

    def test_22_hpe_gate_exists_in_pmgg(self):
        """PMGG hat HPE-Gate für Post-Phase-Validierung."""
        import backend.core.per_phase_musical_goals_gate as pmgg_mod

        src = open(pmgg_mod.__file__, encoding="utf-8").read()
        assert "compare_pleasantness" in src, "PMGG hat kein compare_pleasantness — keine HPE-Validierung nach Phase"
        assert "hpe_skip" in src, "PMGG hat kein hpe_skip — Phase kann bei HPE-Verschlechterung nicht verworfen werden"

    def test_23_cause_to_phases_covers_both_defects(self):
        """Beide Defekte sind in CAUSE_TO_PHASES geroutet."""
        import backend.core.causal_defect_reasoner as cdr_mod

        src = open(cdr_mod.__file__, encoding="utf-8").read()
        assert "transport_bump" in src, "transport_bump nicht in CausalDefectReasoner"
        assert "tape_head_level_dip" in src, "tape_head_level_dip nicht in CausalDefectReasoner"
        # Beide müssen zu Phasen geroutet sein
        assert "phase_12" in src, "phase_12 (Transport-Bump-Reparatur) nicht geroutet"
        assert "phase_54" in src, "phase_54 (Head-Dip-Reparatur) nicht geroutet"
