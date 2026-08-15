from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import scripts.continuous_deep_analysis as cda_module
from scripts.continuous_deep_analysis import ContinuousDeepAnalyzer, PhaseCheckpoint


@pytest.mark.unit
def test_collect_checkpoints_prefers_phase_local_hpi_and_afg() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)

    metadata = {
        "holistic_perceptual_gate": {"hpi": 0.88, "artifact_freedom": 0.99},
        "artifact_freedom": {"score": 0.99},
        "carrier_chain_recovery_ratio": 0.12,
        "pmgg_log_entries": [
            {
                "phase_id": "phase_03_denoise",
                "timestamp": 1.0,
                "scores_before": {"natuerlichkeit": 0.92},
                "scores_after": {"natuerlichkeit": 0.91},
                "hpi_score": 0.41,
                "artifact_freedom_score": 0.96,
            },
            {
                "phase_id": "phase_04_eq_correction",
                "timestamp": 2.0,
                "scores_before": {"natuerlichkeit": 0.91},
                "scores_after": {"natuerlichkeit": 0.92},
                "holistic_perceptual_gate": {"hpi": 0.73, "artifact_freedom": 0.98},
            },
        ],
    }
    restoration_result = SimpleNamespace(metadata=metadata)

    analyzer._collect_checkpoints_from_restore_result(restoration_result, pre_result=None)

    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0].hpi_score == 0.41
    assert analyzer.checkpoints[0].artifact_freedom == 0.96
    assert analyzer.checkpoints[1].hpi_score == 0.73
    assert analyzer.checkpoints[1].artifact_freedom == 0.98


def test_collect_checkpoints_reads_phase_proxies_from_metadata() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)

    metadata = {
        "holistic_perceptual_gate": {"hpi": 0.90, "artifact_freedom": 0.99},
        "artifact_freedom": {"score": 0.99},
        "carrier_chain_recovery_ratio": 0.10,
        "pmgg_log_entries": [
            {
                "phase_id": "phase_01_click_removal",
                "timestamp": 1.0,
                "scores_before": {"natuerlichkeit": 0.90},
                "scores_after": {"natuerlichkeit": 0.91},
                "metadata": {
                    "phase_hpi_proxy": 0.64,
                    "phase_artifact_freedom_proxy": 0.97,
                },
            },
            {
                "phase_id": "phase_02_hum_removal",
                "timestamp": 2.0,
                "scores_before": {"natuerlichkeit": 0.91},
                "scores_after": {"natuerlichkeit": 0.89},
                "metadata": {
                    "phase_hpi_proxy": 0.58,
                    "phase_artifact_freedom_proxy": 0.94,
                },
            },
        ],
    }
    restoration_result = SimpleNamespace(metadata=metadata)

    analyzer._collect_checkpoints_from_restore_result(restoration_result, pre_result=None)

    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0].hpi_score == 0.64
    assert analyzer.checkpoints[0].artifact_freedom == 0.97
    assert analyzer.checkpoints[1].hpi_score == 0.58
    assert analyzer.checkpoints[1].artifact_freedom == 0.94


def test_generate_summary_requires_hpi_gate_for_excellent() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)
    analyzer._final_musical_goals = {
        "natuerlichkeit": 0.95,
        "authentizitaet": 0.93,
    }
    analyzer.checkpoints = [
        PhaseCheckpoint(
            phase_id="phase_65_vocal_naturalness_restoration",
            wall_time_s=0.0,
            musical_goals={"natuerlichkeit": 0.95, "authentizitaet": 0.93},
            hpi_score=0.46,
            artifact_freedom=0.99,
            carrier_recovery_ratio=0.0,
            noise_floor_db=None,
            defects_remaining=None,
            anomalies=[],
        )
    ]

    summary = analyzer._generate_summary()

    assert summary["quality_status"] == "NEEDS_REVIEW"
    assert "hpi<0.60" in summary["quality_gate_reasons"]


def test_extract_vocal_metrics_from_metadata_reads_vqi_and_identity() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)

    metrics = analyzer._extract_vocal_metrics_from_metadata(
        {
            "vqi": 0.83,
            "singer_identity_cosine": 0.94,
            "holistic_perceptual_gate": {"mert_similarity": 0.91},
        }
    )

    assert metrics["vqi"] == 0.83
    assert metrics["singer_identity_cosine"] == 0.94
    assert metrics["mert_similarity"] == 0.91


def test_generate_summary_includes_final_vocal_metrics() -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)
    analyzer._final_musical_goals = {
        "natuerlichkeit": 0.92,
        "authentizitaet": 0.91,
    }
    analyzer._final_vocal_metrics = {
        "vqi": 0.86,
        "singer_identity_cosine": 0.95,
    }
    analyzer.checkpoints = [
        PhaseCheckpoint(
            phase_id="phase_03_denoise",
            wall_time_s=0.0,
            musical_goals={"natuerlichkeit": 0.92, "authentizitaet": 0.91},
            hpi_score=0.72,
            artifact_freedom=0.97,
            carrier_recovery_ratio=0.0,
            noise_floor_db=None,
            defects_remaining=None,
            anomalies=[],
        )
    ]

    summary = analyzer._generate_summary()

    assert summary["final_vqi"] == 0.86
    assert summary["final_singer_identity_cosine"] == 0.95


def test_run_analysis_resets_state_between_runs(monkeypatch, tmp_path) -> None:
    analyzer = ContinuousDeepAnalyzer(realtime=False)
    analyzer.checkpoints = [
        PhaseCheckpoint(
            phase_id="phase_old",
            wall_time_s=0.0,
            musical_goals={"natuerlichkeit": 0.5},
            hpi_score=0.2,
            artifact_freedom=0.8,
            carrier_recovery_ratio=0.0,
            noise_floor_db=None,
            defects_remaining=None,
            anomalies=[],
        )
    ]
    analyzer._final_musical_goals = {"natuerlichkeit": 0.99}
    analyzer._final_vocal_metrics = {"vqi": 0.99}

    monkeypatch.setattr(
        cda_module,
        "load_audio_file",
        lambda _path: {"audio": np.zeros(480, dtype=np.float32), "sr": 48000},
    )
    monkeypatch.setattr(
        cda_module,
        "run_pre_analysis",
        lambda *_args, **_kwargs: SimpleNamespace(
            medium=SimpleNamespace(primary_material="vinyl"),
            era=SimpleNamespace(decade=1970),
            restorability=SimpleNamespace(restorability_score=60.0),
            defects=SimpleNamespace(scores={}),
        ),
    )

    class _FakeRestorer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.config = SimpleNamespace(mode=SimpleNamespace(value="quality"))

        def restore(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(
                musical_goals={},
                metadata={
                    "pmgg_log_entries": [],
                    "holistic_perceptual_gate": {"hpi": 0.75},
                    "artifact_freedom": {"score": 0.98},
                },
            )

        def is_studio_mode(self):
            return False

    monkeypatch.setattr(cda_module, "UnifiedRestorerV3", _FakeRestorer)

    result = analyzer.run_analysis(
        audio_path="dummy.wav",
        output_dir=str(tmp_path),
    )

    assert result["checkpoints"] == []
    assert result["final_musical_goals"] == {}
    assert result["final_vocal_metrics"] == {}
    assert result["summary"] == {"status": "no_checkpoints"}


def test_collect_checkpoints_hpi_components_in_final_checkpoint_only() -> None:
    """§2.44: MERT, timbral, VQI, EAP erscheinen nur im finalen Checkpoint."""
    analyzer = ContinuousDeepAnalyzer(realtime=False)

    metadata = {
        "holistic_perceptual_gate": {
            "hpi": 0.55,
            "artifact_freedom": 0.99,
            "mert_similarity": 0.84,
            "timbral_fidelity": 0.72,
            "emotional_arc_preservation": 0.93,
        },
        "vqi": 0.81,
        "artifact_freedom": {"score": 0.99},
        "carrier_chain_recovery_ratio": 0.20,
        "hpi_material_ceiling": 0.61,
        "pmgg_log_entries": [
            {
                "phase_id": "phase_03_denoise",
                "timestamp": 1.0,
                "scores_before": {"natuerlichkeit": 0.88},
                "scores_after": {"natuerlichkeit": 0.90},
            },
            {
                "phase_id": "phase_65_vocal_naturalness_restoration",
                "timestamp": 2.0,
                "scores_before": {"natuerlichkeit": 0.90},
                "scores_after": {"natuerlichkeit": 0.91},
            },
        ],
    }
    analyzer._collect_checkpoints_from_restore_result(SimpleNamespace(metadata=metadata), pre_result=None)

    assert len(analyzer.checkpoints) == 2
    # Intermediate-Checkpoint: keine HPI-Komponenten
    first = analyzer.checkpoints[0]
    assert first.mert_similarity is None
    assert first.timbral_fidelity is None
    assert first.vqi is None
    assert first.emotional_arc_preservation is None
    assert first.hpi_ceiling is None
    # Finaler Checkpoint: alle HPI-Komponenten gesetzt
    last = analyzer.checkpoints[-1]
    assert last.mert_similarity == 0.84
    assert last.timbral_fidelity == 0.72
    assert last.vqi == 0.81
    assert last.emotional_arc_preservation == 0.93
    assert last.hpi_ceiling == 0.61


def test_hpi_components_in_to_dict_serializable() -> None:
    """PhaseCheckpoint.to_dict() serialisiert HPI-Komponenten korrekt (JSON-safe)."""
    cp = PhaseCheckpoint(
        phase_id="phase_65_vocal_naturalness_restoration",
        wall_time_s=1_000_000.0,
        musical_goals={"natuerlichkeit": 0.91},
        hpi_score=0.55,
        artifact_freedom=0.99,
        carrier_recovery_ratio=0.20,
        noise_floor_db=None,
        defects_remaining=None,
        anomalies=[],
        vqi=0.81,
        mert_similarity=0.84,
        timbral_fidelity=0.72,
        emotional_arc_preservation=0.93,
        hpi_ceiling=0.61,
    )
    d = cp.to_dict()
    assert d["vqi"] == 0.81
    assert d["mert_similarity"] == 0.84
    assert d["timbral_fidelity"] == 0.72
    assert d["emotional_arc_preservation"] == 0.93
    assert d["hpi_ceiling"] == 0.61


def test_compute_hpi_material_ceiling_cassette_mp3_low_vocal() -> None:
    """§0k: HPI-Ceiling für Cassette+mp3_low mit Gesang (SGMSE+ Chain ohne MIIPHER)."""
    from backend.core.unified_restorer_v3 import _compute_hpi_material_ceiling_uv3

    ceiling = _compute_hpi_material_ceiling_uv3(
        material_type="cassette",
        transfer_chain=["mp3_low"],
        panns_singing=0.6,
        is_studio_mode=False,
    )
    # §G76/§G77 (GEBOTE.md) kontinuierliches Modell (ersetzt die alten diskreten
    # Per-Metrik-Buckets): Basis(cassette)=0.82, Depth=len(chain)=1 → depth_factor=1.0,
    # vocal=True (0.6 ≥ adaptive Schwelle 0.304) → vocal_factor=0.94
    # ceiling = 0.82 × 1.0 × 0.94 = 0.7708
    assert abs(ceiling - 0.7708) < 1e-6, f"Unerwartet: {ceiling}"


def test_compute_hpi_material_ceiling_cd_instrumental() -> None:
    """§0k: HPI-Ceiling für CD-Material ohne Gesang."""
    from backend.core.unified_restorer_v3 import _compute_hpi_material_ceiling_uv3

    ceiling = _compute_hpi_material_ceiling_uv3(
        material_type="cd",
        transfer_chain=[],
        panns_singing=0.1,
        is_studio_mode=False,
    )
    # Ohne VQI-Faktor, CD-timbral=0.97, MERT=0.97 → Ceiling sehr hoch
    assert ceiling > 0.85, f"CD-Instrumental-Ceiling sollte > 0.85 sein: {ceiling}"
