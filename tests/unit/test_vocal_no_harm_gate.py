from __future__ import annotations

"""Unit tests for the vocal no-harm gate."""


import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from backend.core.phases.phase_interface import PhaseCategory, PhaseMetadata, create_phase_result

SR = 48_000


class _DummyUv3Phase:
    def __init__(self) -> None:
        self._meta = PhaseMetadata(
            phase_id="phase_37_bass_enhancement",
            name="Dummy Vocal Harm Phase",
            category=PhaseCategory.METADATA,
            priority=5,
            version="1.0",
            dependencies=[],
            estimated_time_factor=0.0,
            memory_requirement_mb=1,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.0,
            description="dummy",
        )

    def get_metadata(self) -> PhaseMetadata:
        return self._meta

    @staticmethod
    def process(audio: np.ndarray, **kwargs: Any):  # pylint: disable=unused-argument
        return create_phase_result(audio=np.clip(audio * 0.25, -1.0, 1.0).astype(np.float32))


def _synthetic_vocal() -> np.ndarray:
    timeline = np.linspace(0.0, 1.0, SR, endpoint=False)
    audio = 0.2 * np.sin(2.0 * np.pi * 220.0 * timeline)
    audio += 0.08 * np.sin(2.0 * np.pi * 440.0 * timeline)
    return audio.astype(np.float32)  # type: ignore[no-any-return]


def _patch_measurements(
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_freedom: float = 1.0,
    vqi: float = 0.90,
    singer_identity: float = 0.95,
    delta_hnr: float = 0.0,
    formant_rollback: bool = False,
    formant_shift_db: float = 0.0,
) -> None:
    import backend.core.vocal_no_harm_gate as module

    class _FakeArtifactGate:
        @staticmethod
        def evaluate(*args: Any, **kwargs: Any) -> SimpleNamespace:  # pylint: disable=unused-argument
            return SimpleNamespace(artifact_freedom=artifact_freedom)

    def _fake_load_symbol(module_name: str, symbol_name: str) -> Any:
        if symbol_name == "get_artifact_freedom_gate":
            return lambda: _FakeArtifactGate()
        if symbol_name == "compute_vqi":
            return lambda *args, **kwargs: {  # pylint: disable=unused-argument
                "vqi": vqi,
                "singer_identity_cosine": singer_identity,
            }
        if symbol_name == "get_vqi_material_floor":
            return lambda material_type, is_studio_2026=False: 0.87 if is_studio_2026 else 0.72
        if symbol_name == "check_hnr_delta":
            return lambda *args, **kwargs: {"delta_hnr": delta_hnr}  # pylint: disable=unused-argument
        if symbol_name == "check_formant_shift_db":
            return lambda *args, **kwargs: (formant_rollback, formant_shift_db)  # pylint: disable=unused-argument
        raise AssertionError(f"Unexpected lazy symbol: {module_name}.{symbol_name}")

    monkeypatch.setattr(module, "_load_symbol", _fake_load_symbol)


def test_vocal_no_harm_gate_is_inactive_below_vocal_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.core.vocal_no_harm_gate as module

    monkeypatch.setattr(
        module,
        "_load_symbol",
        lambda module_name, symbol_name: pytest.fail(f"measurement loaded for inactive gate: {symbol_name}"),
    )
    audio = _synthetic_vocal()

    result = module.get_vocal_no_harm_gate().evaluate(audio, audio.copy(), SR, panns_singing=0.20)

    assert result.active is False
    assert result.passed is True
    assert result.requires_rollback is False
    assert result.reason == "not_vocal"


def test_vocal_no_harm_gate_passes_clean_vocal_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch)
    audio = _synthetic_vocal()

    result = get_vocal_no_harm_gate().evaluate(
        audio,
        audio.copy(),
        SR,
        panns_singing=0.80,
        material_type="vinyl",
        phase_id="phase_03_denoise",
    )

    assert result.active is True
    assert result.passed is True
    assert result.requires_rollback is False
    assert result.checks["artifact_freedom_ok"] is True
    assert result.checks["vqi_ok"] is True
    assert result.checks["singer_identity_ok"] is True
    assert result.checks["hnr_ok"] is True
    assert result.checks["formant_ok"] is True
    json.dumps(result.to_dict())


@pytest.mark.parametrize(
    ("measurement_kwargs", "expected_reason"),
    [
        ({"artifact_freedom": 0.90}, "artifact_freedom"),
        ({"vqi": 0.60}, "vqi"),
        ({"singer_identity": 0.80}, "singer_identity"),
        ({"delta_hnr": 4.0}, "hnr_overcleaned"),
        ({"formant_rollback": True, "formant_shift_db": 3.0}, "formant_shift"),
    ],
)
def test_vocal_no_harm_gate_requests_rollback_for_measured_harm(
    monkeypatch: pytest.MonkeyPatch,
    measurement_kwargs: dict[str, float | bool],
    expected_reason: str,
) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch, **measurement_kwargs)  # type: ignore[arg-type]
    audio = _synthetic_vocal()

    result = get_vocal_no_harm_gate().evaluate(
        audio,
        audio.copy(),
        SR,
        panns_singing=0.90,
        material_type="vinyl",
    )

    assert result.active is True
    assert result.passed is False
    assert result.requires_rollback is True
    assert expected_reason in result.reason


def test_vocal_no_harm_gate_uses_studio_vqi_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch, vqi=0.84)
    audio = _synthetic_vocal()

    result = get_vocal_no_harm_gate().evaluate(
        audio,
        audio.copy(),
        SR,
        panns_singing=0.90,
        material_type="cd_digital",
        mode="studio2026",
    )

    assert result.requires_rollback is True
    assert result.scores["vqi_floor"] == pytest.approx(0.87)
    assert "vqi" in result.reason


def test_vocal_no_harm_gate_rolls_back_when_below_maximum_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch, vqi=0.80)
    audio = _synthetic_vocal()

    result = get_vocal_no_harm_gate().evaluate(
        audio,
        audio.copy(),
        SR,
        panns_singing=0.90,
        material_type="vinyl",
        restorability_score=100.0,
    )

    assert result.requires_rollback is True
    assert "vocal_max_alignment" in result.reason
    assert result.scores["vocal_max_target"] == pytest.approx(0.88)
    assert result.scores["vocal_max_alignment_percent"] == pytest.approx(90.91, abs=0.01)
    assert result.scores["vocal_max_alignment_floor_percent"] == pytest.approx(94.0)
    assert result.scores["vocal_max_alignment_floor_percent_effective"] == pytest.approx(92.0)
    assert result.checks["vocal_max_alignment_ok"] is False


def test_vocal_no_harm_gate_allows_small_alignment_gap_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch, vqi=0.81)
    audio = _synthetic_vocal()

    result = get_vocal_no_harm_gate().evaluate(
        audio,
        audio.copy(),
        SR,
        panns_singing=0.90,
        material_type="vinyl",
        restorability_score=100.0,
    )

    assert result.requires_rollback is False
    assert result.scores["vocal_max_alignment_percent"] == pytest.approx(92.05, abs=0.02)
    assert result.scores["vocal_max_alignment_floor_percent"] == pytest.approx(94.0)
    assert result.scores["vocal_max_alignment_floor_percent_effective"] == pytest.approx(92.0)
    assert result.checks["vocal_max_alignment_ok"] is True


def test_vocal_no_harm_gate_relaxes_maximum_alignment_for_low_restorability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch, vqi=0.80)
    audio = _synthetic_vocal()

    result = get_vocal_no_harm_gate().evaluate(
        audio,
        audio.copy(),
        SR,
        panns_singing=0.90,
        material_type="vinyl",
        restorability_score=0.0,
    )

    assert result.requires_rollback is False
    assert result.scores["vocal_max_target"] == pytest.approx(0.86)
    assert result.scores["vocal_max_alignment_percent"] == pytest.approx(93.02, abs=0.01)
    assert result.scores["vocal_max_alignment_floor_percent"] == pytest.approx(90.0)
    assert result.checks["vocal_max_alignment_ok"] is True


def test_vocal_no_harm_gate_requests_rollback_when_protected_breath_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch)
    audio = _synthetic_vocal()
    start = int(0.20 * SR)
    end = int(0.34 * SR)
    rng = np.random.default_rng(1234)
    audio[start:end] += rng.normal(0.0, 0.015, end - start).astype(np.float32)
    post = audio.copy()
    post[start:end] *= 0.10

    result = get_vocal_no_harm_gate().evaluate(
        audio,
        post,
        SR,
        panns_singing=0.90,
        material_type="vinyl",
        breath_segments=[SimpleNamespace(start_s=0.20, end_s=0.34, category="emotional_tension")],
    )

    assert result.requires_rollback is True
    assert "breath_preservation" in result.reason
    assert result.checks["breath_preservation_ok"] is False
    assert result.scores["max_breath_attenuation_db"] > 3.0


def test_vocal_no_harm_gate_catches_cumulative_breath_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.core.vocal_no_harm_gate import get_vocal_no_harm_gate

    _patch_measurements(monkeypatch)
    reference = _synthetic_vocal()
    start = int(0.20 * SR)
    end = int(0.34 * SR)
    rng = np.random.default_rng(5678)
    reference[start:end] += rng.normal(0.0, 0.018, end - start).astype(np.float32)
    pre_phase = reference.copy()
    pre_phase[start:end] *= 0.64
    post_phase = pre_phase.copy()
    post_phase[start:end] *= 0.70

    result = get_vocal_no_harm_gate().evaluate(
        pre_phase,
        post_phase,
        SR,
        panns_singing=0.90,
        material_type="vinyl",
        reference_audio=reference,
        breath_segments=[SimpleNamespace(start_s=0.20, end_s=0.34, category="natural")],
    )

    assert result.requires_rollback is True
    assert "cumulative_breath_preservation" in result.reason
    assert result.checks["breath_preservation_ok"] is True
    assert result.checks["cumulative_breath_preservation_ok"] is False
    assert result.scores["max_breath_attenuation_db"] < 6.0
    assert result.scores["cumulative_max_breath_attenuation_db"] > 6.0


def test_vocal_no_harm_gate_catches_cumulative_hnr_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.core.vocal_no_harm_gate as module

    reference = _synthetic_vocal()
    pre_phase = (reference * 0.98).astype(np.float32)
    post_phase = (reference * 0.96).astype(np.float32)

    class _FakeArtifactGate:
        @staticmethod
        def evaluate(*args: Any, **kwargs: Any) -> SimpleNamespace:  # pylint: disable=unused-argument
            return SimpleNamespace(artifact_freedom=1.0)

    def _fake_load_symbol(module_name: str, symbol_name: str) -> Any:
        if symbol_name == "get_artifact_freedom_gate":
            return lambda: _FakeArtifactGate()
        if symbol_name == "compute_vqi":
            return lambda *args, **kwargs: {  # pylint: disable=unused-argument
                "vqi": 0.90,
                "singer_identity_cosine": 0.95,
            }
        if symbol_name == "get_vqi_material_floor":
            return lambda material_type, is_studio_2026=False: 0.72  # pylint: disable=unused-argument
        if symbol_name == "check_formant_shift_db":
            return lambda *args, **kwargs: (False, 0.0)  # pylint: disable=unused-argument
        if symbol_name == "check_hnr_delta":

            def _fake_check_hnr_delta(pre: np.ndarray, post: np.ndarray, sr: int) -> dict[str, float]:
                del post, sr
                is_cumulative_reference = np.allclose(pre, reference, atol=1e-7)
                return {"delta_hnr": 4.2 if is_cumulative_reference else 1.2}

            return _fake_check_hnr_delta
        raise AssertionError(f"Unexpected lazy symbol: {module_name}.{symbol_name}")

    monkeypatch.setattr(module, "_load_symbol", _fake_load_symbol)

    result = module.get_vocal_no_harm_gate().evaluate(
        pre_phase,
        post_phase,
        SR,
        panns_singing=0.90,
        material_type="vinyl",
        reference_audio=reference,
    )

    assert result.requires_rollback is True
    assert "cumulative_hnr_overcleaned" in result.reason
    assert result.checks["hnr_ok"] is True
    assert result.checks["cumulative_hnr_ok"] is False
    assert result.scores["delta_hnr_db"] == pytest.approx(1.2)
    assert result.scores["cumulative_delta_hnr_db"] == pytest.approx(4.2)


def test_uv3_profiled_phase_call_rolls_back_when_vocal_no_harm_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.core.vocal_no_harm_gate as gate_module
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    def _fake_evaluate_vocal_no_harm(*args: Any, **kwargs: Any):  # pylint: disable=unused-argument
        assert kwargs.get("breath_segments") == [breath_segment]
        assert kwargs.get("reference_audio") is reference_audio
        return gate_module.VocalNoHarmResult(
            active=True,
            passed=False,
            requires_rollback=True,
            reason="vqi;formant_shift",
            scores={"vqi": 0.50, "max_formant_shift_db": 3.1},
            checks={"vqi_ok": False, "formant_ok": False},
        )

    monkeypatch.setattr(gate_module, "evaluate_vocal_no_harm", _fake_evaluate_vocal_no_harm)
    monkeypatch.setattr(UnifiedRestorerV3, "_fast_goal_snapshot", staticmethod(lambda *args, **kwargs: {}))

    uv3 = UnifiedRestorerV3()
    breath_segment = SimpleNamespace(start_s=0.10, end_s=0.20, category="natural")
    uv3._restoration_context = {
        "panns_singing": 0.90,
        "breath_segments": [breath_segment],
    }
    uv3._song_calibration_profile = {
        "family_scalars": {"denoise": 1.0, "general": 1.0},
        "strict_conflict_policy": {
            "rollback_decay_per_family": {"denoise": 0.90, "general": 0.96},
            "rollback_decay_floor": 0.55,
            "phase_strength_caps": {},
        },
    }
    phase = _DummyUv3Phase()
    audio = _synthetic_vocal()[:4096]
    reference_audio = np.asarray(audio, dtype=np.float32).copy()
    uv3._restoration_context["vocal_no_harm_reference_audio"] = reference_audio

    result = uv3._profiled_phase_call(phase, audio, sample_rate=SR, panns_singing=0.90)

    assert np.allclose(result.audio, audio)
    assert result.metadata["vocal_no_harm_rollback"] is True
    assert result.metadata["vocal_no_harm_reason"] == "vqi;formant_shift"
    assert result.metadata["vocal_no_harm_gate"]["requires_rollback"] is True
    # §2.58a Vocal-Supremacy-Kopplung: panns_singing=0.90 → vocal_boost angewendet auf Decay
    # _vocal_boost = 1.0 + 0.20 * ((0.90 - 0.35) / 0.65) ≈ 1.16923
    # _decay_att = 1.0 - 0.90 = 0.10
    # effective_decay = 1.0 - (0.10 * 1.16923) ≈ 0.88308
    assert uv3._song_calibration_profile["family_scalars"]["denoise"] == pytest.approx(0.8830769230769231, abs=1e-6)
    assert uv3._phase_goal_conflict_runtime["events"][-1]["reason"] == "vocal_no_harm_rollback"
