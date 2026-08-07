import types

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

SR = 48_000


def _sine(secs: float = 0.3, freq: float = 440.0) -> np.ndarray:
    n = int(SR * secs)
    t = np.arange(n, dtype=np.float32) / SR
    return (0.1 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


class _PhaseStub:
    def __init__(self, phase_id: str, estimated_time_factor: float) -> None:
        self._phase_id = phase_id
        self._estimated_time_factor = estimated_time_factor

    def get_metadata(self) -> object:
        return types.SimpleNamespace(
            estimated_time_factor=self._estimated_time_factor,
            phase_id=self._phase_id,
            name=self._phase_id,
        )


def _build_restorer() -> UnifiedRestorerV3:
    restorer = UnifiedRestorerV3(
        RestorationConfig(
            enable_phase_gate=False,
            enable_phase_skipping=False,
            enable_performance_guard=False,
        )
    )
    restorer.phase_metadata = {
        "phase_03_denoise": {"name": "Denoise", "dependencies": [], "category": "cleanup"},
        "phase_17_mastering_polish": {"name": "Mastering Polish", "dependencies": [], "category": "mastering"},
        "phase_39_air_band_enhancement": {"name": "Air Band", "dependencies": [], "category": "enhancement"},
        "phase_42_vocal_enhancement": {"name": "Vocal Enhancement", "dependencies": [], "category": "vocal"},
    }

    phase_costs = {
        "phase_03_denoise": 10.0,
        "phase_17_mastering_polish": 200.0,
        "phase_39_air_band_enhancement": 80.0,
        "phase_42_vocal_enhancement": 10.0,
    }

    restorer._get_phase = lambda pid: _PhaseStub(pid, phase_costs[pid])  # type: ignore[method-assign]
    restorer._profiled_phase_call = (  # type: ignore[method-assign]
        lambda _phase, _audio, **_kwargs: types.SimpleNamespace(
            success=True,
            audio=np.clip(np.asarray(_audio, dtype=np.float32) * 1.001, -1.0, 1.0),
            execution_time_seconds=0.001,
            warnings=[],
        )
    )
    return restorer


@pytest.mark.unit
def test_budget_pressure_reason_reserves_budget_for_later_vocal_phase() -> None:
    restorer = _build_restorer()
    reason = restorer._budget_pressure_skip_reason(
        "phase_17_mastering_polish",
        material_key="vinyl",
        remaining_budget_s=23.0,
        estimated_time_s=60.0,
        future_phases=["phase_42_vocal_enhancement"],
    )

    assert reason is not None
    assert reason["reason"] == "reserve_for_priority_followup"
    assert reason["future_priority_phase"] == "phase_42_vocal_enhancement"


def test_budget_pressure_reason_uses_historically_weak_phase_for_same_material() -> None:
    restorer = _build_restorer()
    restorer._phase_budget_gain_history = {
        "vinyl": {
            "phase_39": {
                "samples": 3.0,
                "mean_net_gain": 0.001,
            }
        }
    }
    reason = restorer._budget_pressure_skip_reason(
        "phase_39_air_band_enhancement",
        material_key="vinyl",
        remaining_budget_s=12.0,
        estimated_time_s=24.0,
        future_phases=[],
    )

    assert reason is not None
    assert reason["reason"] == "historically_weak_delta"
    assert reason["history_samples"] == 3


def test_pmgg_retry_budget_hint_reserves_retries_for_later_priority_phase() -> None:
    restorer = _build_restorer()

    hint = restorer._pmgg_retry_budget_hint(
        "phase_42_vocal_enhancement",
        material_key="vinyl",
        remaining_budget_s=11.0,
        estimated_time_s=15.0,
        future_phases=["phase_58_lyrics_guided_enhancement"],
    )

    assert hint is not None
    assert hint["reason"] == "reserve_for_priority_followup"
    assert hint["max_retries_cap"] == 2
    assert hint["future_priority_phases"] == ["phase_58_lyrics_guided_enhancement"]


def test_execute_pipeline_skips_phase_when_budget_helper_requests_passthrough() -> None:
    restorer = _build_restorer()
    audio = _sine()

    def _forced_budget_reason(
        phase_id: str,
        *,
        material_key: str,
        remaining_budget_s: float,
        estimated_time_s: float,
        future_phases: list[str],
    ) -> dict[str, object] | None:
        if phase_id == "phase_17_mastering_polish":
            return {
                "reason": "reserve_for_priority_followup",
                "future_priority_phase": "phase_39_air_band_enhancement",
            }
        return None

    restorer._budget_pressure_skip_reason = _forced_budget_reason  # type: ignore[method-assign]

    out, executed, skipped, deferred = restorer._execute_pipeline(
        audio=audio,
        sample_rate=SR,
        material_type=MaterialType.VINYL,
        defect_result=types.SimpleNamespace(scores={}),
        selected_phases=[
            "phase_03_denoise",
            "phase_17_mastering_polish",
            "phase_39_air_band_enhancement",
        ],
        no_rt_limit=True,
    )

    assert isinstance(out, np.ndarray)
    assert executed == ["phase_03_denoise", "phase_39_air_band_enhancement"]
    assert "phase_17_mastering_polish" in skipped
    assert deferred == []


def test_phase50_wall_budget_protection_triggers_on_lossy_chain_goal_deficit() -> None:
    restorer = _build_restorer()
    protected = restorer._should_protect_phase50_from_wall_budget(
        phase_id="phase_50_spectral_repair",
        transfer_chain=["cassette", "mp3_low"],
        baseline_scores={"brillanz": 0.20},
        effective_targets={"brillanz": 0.45},
        applicable_goals={"brillanz", "transient_energie"},
    )
    assert protected is True


def test_phase50_wall_budget_protection_off_for_non_lossy_chain() -> None:
    restorer = _build_restorer()
    protected = restorer._should_protect_phase50_from_wall_budget(
        phase_id="phase_50_spectral_repair",
        transfer_chain=["vinyl", "tape"],
        baseline_scores={"brillanz": 0.20},
        effective_targets={"brillanz": 0.45},
        applicable_goals={"brillanz"},
    )
    assert protected is False


def test_general_wall_budget_protection_triggers_for_phase39_goal_deficit() -> None:
    restorer = _build_restorer()
    protected = restorer._should_protect_phase_from_wall_budget(
        phase_id="phase_39_air_band_enhancement",
        transfer_chain=["vinyl", "tape"],
        baseline_scores={"brillanz": 0.58, "transparenz": 0.61},
        effective_targets={"brillanz": 0.76, "transparenz": 0.74},
        applicable_goals={"brillanz", "transparenz", "artikulation"},
    )
    assert protected is True


def test_general_wall_budget_protection_ignores_phase39_without_goal_deficit() -> None:
    restorer = _build_restorer()
    protected = restorer._should_protect_phase_from_wall_budget(
        phase_id="phase_39_air_band_enhancement",
        transfer_chain=["vinyl", "tape"],
        baseline_scores={"brillanz": 0.81, "transparenz": 0.80},
        effective_targets={"brillanz": 0.76, "transparenz": 0.74},
        applicable_goals={"brillanz", "transparenz", "artikulation"},
    )
    assert protected is False
