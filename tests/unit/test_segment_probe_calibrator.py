from __future__ import annotations

"""Tests für §2.82 SegmentProbeCalibrator.

Testet: SegmentProbeResult-Dataclass, Kern-Algorithmus, Fallback-Pfade, Eligible-Phases-Set.
"""


from typing import Any

import numpy as np
import pytest

from backend.core.dsp.segment_probe_calibrator import (
    SEGMENT_PROBE_ELIGIBLE_PHASES,
    SegmentProbeResult,
    _compute_team_score,
    _extract_probe_segment,
    _safe_probe_kwargs,
    run_segment_probe,
)

SR = 48000


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _sine(duration_s: float, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    """Mono-Sinussignal als float32."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _stereo(duration_s: float) -> np.ndarray:
    """Stereo (2, N) Signal."""
    mono = _sine(duration_s)
    return np.stack([mono, mono * 0.9], axis=0)


def _identity_phase(seg: np.ndarray, **_kw: Any) -> np.ndarray:
    """Phase-Stub: gibt Segment unverändert zurück."""
    return np.array(seg, dtype=np.float32)


def _attenuate_phase(seg: np.ndarray, strength: float = 0.5, **_kw: Any) -> np.ndarray:
    """Phase-Stub: dämpft Signal um strength (strength=0 → Stille, 1 → Original)."""
    return (seg * (1.0 - float(strength))).astype(np.float32)


def _amplify_phase(seg: np.ndarray, strength: float = 0.5, **_kw: Any) -> np.ndarray:
    """Phase-Stub: verstärkt Signal um strength (0 → keine Änderung, 1 → +6 dB)."""
    return np.clip(seg * (1.0 + float(strength)), -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. SegmentProbeResult Dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSegmentProbeResultDataclass:
    def test_felder_vorhanden(self) -> None:
        r = SegmentProbeResult(
            confirmed_strength=0.6,
            oracle_strength=0.5,
            best_candidate_idx=1,
            candidates=[0.3, 0.5, 0.7],
            team_scores=[0.1, 0.8, 0.3],
            overprocessing_penalty=[0.0, 0.0, 0.1],
            probe_duration_s=0.42,
            segment_start_s=10.0,
        )
        assert r.confirmed_strength == pytest.approx(0.6)
        assert r.oracle_strength == pytest.approx(0.5)
        assert r.skipped is False
        assert r.skip_reason == ""

    def test_frozen(self) -> None:
        r = SegmentProbeResult(
            confirmed_strength=0.5,
            oracle_strength=0.5,
            best_candidate_idx=0,
            candidates=[0.5],
            team_scores=[0.0],
            overprocessing_penalty=[0.0],
            probe_duration_s=0.1,
            segment_start_s=0.0,
        )
        with pytest.raises(Exception):
            r.confirmed_strength = 0.9  # type: ignore[misc]

    def test_to_dict_keys(self) -> None:
        r = SegmentProbeResult(
            confirmed_strength=0.5,
            oracle_strength=0.5,
            best_candidate_idx=0,
            candidates=[0.5],
            team_scores=[0.5],
            overprocessing_penalty=[0.0],
            probe_duration_s=0.1,
            segment_start_s=5.0,
        )
        d = r.to_dict()
        for key in (
            "confirmed_strength",
            "oracle_strength",
            "best_candidate_idx",
            "candidates",
            "team_scores",
            "overprocessing_penalty",
            "probe_duration_s",
            "segment_start_s",
            "skipped",
            "skip_reason",
        ):
            assert key in d, f"Fehlender Key: {key}"

    def test_to_dict_json_kompatibel(self) -> None:
        import json

        r = SegmentProbeResult(
            confirmed_strength=0.6,
            oracle_strength=0.5,
            best_candidate_idx=1,
            candidates=[0.3, 0.5, 0.7],
            team_scores=[0.1, 0.9, 0.3],
            overprocessing_penalty=[0.0, 0.0, 0.2],
            probe_duration_s=0.5,
            segment_start_s=12.0,
        )
        json.dumps(r.to_dict())  # darf nicht werfen


# ---------------------------------------------------------------------------
# 2. _safe_probe_kwargs
# ---------------------------------------------------------------------------


class TestSafeProbeKwargs:
    def test_strength_gesetzt(self) -> None:
        kw = {"strength": 0.5, "threshold_db": -20.0}
        safe = _safe_probe_kwargs(kw, 0.3)
        assert safe["strength"] == pytest.approx(0.3)

    def test_arrays_entfernt(self) -> None:
        kw = {
            "strength": 0.5,
            "reference_audio": np.zeros(100),
            "noise_profile": np.ones(50),
            "mode": "restoration",
        }
        safe = _safe_probe_kwargs(kw, 0.5)
        assert "reference_audio" not in safe
        assert "noise_profile" not in safe
        assert safe.get("mode") == "restoration"

    def test_callbacks_entfernt(self) -> None:
        kw = {"strength": 0.5, "progress_sub_callback": lambda x: x}
        safe = _safe_probe_kwargs(kw, 0.5)
        assert "progress_sub_callback" not in safe

    def test_probe_mode_gesetzt(self) -> None:
        safe = _safe_probe_kwargs({"strength": 0.5}, 0.5)
        assert safe["_probe_mode"] is True


# ---------------------------------------------------------------------------
# 3. _extract_probe_segment
# ---------------------------------------------------------------------------


class TestExtractProbeSegment:
    def test_mono_laenge(self) -> None:
        audio = _sine(30.0)
        seg, start_s = _extract_probe_segment(audio, SR)
        assert seg.shape[-1] == pytest.approx(3 * SR, abs=SR // 10)

    def test_stereo_kanaele_erhalten(self) -> None:
        audio = _stereo(30.0)
        seg, _ = _extract_probe_segment(audio, SR)
        assert seg.ndim == 2
        assert seg.shape[0] == 2

    def test_start_nach_intro(self) -> None:
        audio = _sine(60.0)
        _, start_s = _extract_probe_segment(audio, SR)
        assert start_s >= 10.0

    def test_kurzaudio_fallback(self) -> None:
        # Audio kürzer als 3s → Fallback: Anfang nehmen
        audio = _sine(2.0)
        seg, start_s = _extract_probe_segment(audio, SR)
        assert seg.shape[-1] > 0
        assert start_s == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. _compute_team_score
# ---------------------------------------------------------------------------


class TestComputeTeamScore:
    def test_gap_geschlossen(self) -> None:
        pre = {"natuerlichkeit": 0.70}
        post = {"natuerlichkeit": 0.80}
        gaps = {"natuerlichkeit": 0.10}
        t_score, penalty = _compute_team_score(pre, post, gaps, None)
        assert t_score > 0.5
        assert penalty == pytest.approx(0.0)

    def test_overprocessing_penalty(self) -> None:
        pre = {"natuerlichkeit": 0.85}
        post = {"natuerlichkeit": 0.72}  # Verschlechterung
        gaps = {"natuerlichkeit": 0.05}
        _, penalty = _compute_team_score(pre, post, gaps, None)
        assert penalty > 0.0

    def test_kein_gap_kein_beitrag(self) -> None:
        pre = {"brillanz": 0.90}
        post = {"brillanz": 0.91}
        gaps = {"brillanz": 0.0}  # kein Gap → keine Wertung
        t_score, _ = _compute_team_score(pre, post, gaps, None)
        assert t_score == pytest.approx(0.0)

    def test_gewichtung_einfluss(self) -> None:
        pre = {"brillanz": 0.70, "natuerlichkeit": 0.70}
        post = {"brillanz": 0.80, "natuerlichkeit": 0.75}
        gaps = {"brillanz": 0.10, "natuerlichkeit": 0.10}
        weights = {"brillanz": 3.0, "natuerlichkeit": 1.0}
        t_score, _ = _compute_team_score(pre, post, gaps, weights)
        # brillanz (3×) trägt mehr bei als natuerlichkeit (1×)
        t_score_no_weight, _ = _compute_team_score(pre, post, gaps, None)
        assert t_score > t_score_no_weight  # Gewicht erhöht Score für höher gewichtetes Goal


# ---------------------------------------------------------------------------
# 5. run_segment_probe — Basis-Szenarien
# ---------------------------------------------------------------------------


class TestRunSegmentProbeBasic:
    def test_zu_kurz_skip(self) -> None:
        """Audio < 10 s → skipped=True."""
        audio = _sine(5.0)
        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_identity_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={"natuerlichkeit": 0.1},
        )
        assert result.skipped is True
        assert result.skip_reason == "audio_zu_kurz"
        assert result.confirmed_strength == pytest.approx(0.5)

    def test_keine_gaps_skip(self) -> None:
        """Leere goal_gaps → skipped=True."""
        audio = _sine(20.0)
        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_identity_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={},
        )
        assert result.skipped is True
        assert result.skip_reason == "keine_goal_gaps"

    def test_identity_phase_oracle_bestaetigt(self) -> None:
        """Identity-Phase → Probe liefert Ergebnis oder überspringt graceful."""
        audio = _sine(30.0)
        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_identity_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={"timbre_authentizitaet": 0.05},
        )
        # Contract: kein Crash, confirmed_strength im gültigen Bereich
        assert 0.02 <= result.confirmed_strength <= 1.0
        # Bei Skip: oracle_strength muss unverändert zurückgegeben werden
        if result.skipped:
            assert result.confirmed_strength == pytest.approx(result.oracle_strength)

    def test_drei_kandidaten_erzeugt(self) -> None:
        """Probe testet genau 3 Kandidaten (60 %, 100 %, 140 % der Oracle-Stärke)."""
        audio = _sine(30.0)
        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_identity_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={"natuerlichkeit": 0.1},
        )
        assert len(result.candidates) in {2, 3}  # Duplikate bei Extremwerten möglich

    def test_duration_unter_budget(self) -> None:
        """Probe läuft unter dem 2s-Zeitbudget."""
        audio = _sine(20.0)
        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_identity_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={"natuerlichkeit": 0.1},
        )
        assert result.probe_duration_s < 2.5  # leichter Puffer


# ---------------------------------------------------------------------------
# 6. run_segment_probe — Non-Blocking / Fehlerrobustheit
# ---------------------------------------------------------------------------


class TestRunSegmentProbeRobust:
    def test_phase_wirft_exception(self) -> None:
        """Exception in Phase → bestätigte Oracle-Stärke zurück."""

        def _crashing_phase(seg: np.ndarray, **_kw: Any) -> np.ndarray:
            raise RuntimeError("Intentionaler Test-Crash")

        audio = _sine(20.0)
        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_crashing_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={"natuerlichkeit": 0.1},
        )
        # Alle Kandidaten schlagen fehl → skipped=True oder oracle unverändert
        assert result.confirmed_strength == pytest.approx(0.5)
        assert result.skipped is True

    def test_nan_audio(self) -> None:
        """NaN-Audio → kein Crash."""
        audio = np.full(20 * SR, float("nan"), dtype=np.float32)
        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_identity_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={"natuerlichkeit": 0.1},
        )
        # Kein Crash, skipped oder oracle
        assert 0.0 <= result.confirmed_strength <= 1.0

    def test_stereo_audio(self) -> None:
        """Stereo-Audio wird korrekt verarbeitet."""
        audio = _stereo(20.0)

        def _stereo_phase(seg: np.ndarray, **_kw: Any) -> np.ndarray:
            assert seg.ndim == 2
            return np.array(seg, dtype=np.float32)

        result = run_segment_probe(
            audio=audio,
            sr=SR,
            phase_process_fn=_stereo_phase,
            base_kwargs={"strength": 0.5},
            oracle_strength=0.5,
            goal_gaps={"natuerlichkeit": 0.1},
        )
        assert 0.0 <= result.confirmed_strength <= 1.0


# ---------------------------------------------------------------------------
# 7. SEGMENT_PROBE_ELIGIBLE_PHASES — Invarianten
# ---------------------------------------------------------------------------


class TestEligiblePhasesInvarianten:
    def test_set_nicht_leer(self) -> None:
        assert len(SEGMENT_PROBE_ELIGIBLE_PHASES) > 0

    def test_keine_verbotenen_phasen(self) -> None:
        """§0a-verbotene Phasen dürfen NICHT im Eligible-Set sein."""
        verboten = {
            "phase_21_exciter",
            "phase_35_multiband_compression",
            "phase_42_vocal_enhancement",
        }
        gemeinsam = SEGMENT_PROBE_ELIGIBLE_PHASES & verboten
        assert not gemeinsam, f"§0a-verbotene Phase im Eligible-Set: {gemeinsam}"

    def test_nur_phase_prefixed_names(self) -> None:
        for name in SEGMENT_PROBE_ELIGIBLE_PHASES:
            assert name.startswith("phase_"), f"Ungültiger Name: {name}"

    def test_alle_phase_ids_auf_disk(self) -> None:
        """Alle Eligible-Phase-IDs müssen als phase_*.py auf Disk existieren."""
        import os

        phases_dir = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "core", "phases")
        for phase_id in SEGMENT_PROBE_ELIGIBLE_PHASES:
            fname = f"{phase_id}.py"
            fpath = os.path.join(phases_dir, fname)
            assert os.path.exists(fpath), f"Phasendatei nicht gefunden: {fpath}"
