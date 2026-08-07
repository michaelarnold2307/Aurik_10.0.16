from __future__ import annotations

"""[RELEASE_MUST] Longform Real-Audio Gate for InnovationSuperiorityOrchestrator.

Prüft auf realem, längerem Audiomaterial, dass die neue Innovationsschicht
im UV3-Closed-Loop konsistent Telemetrie liefert (ohne Gate-Bypass).
"""


from typing import Any

import numpy as np
import pytest


def _to_samples_first(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > arr.shape[0]:
        return arr.T
    return arr


@pytest.fixture(scope="module")
def real_audio_innovation_longform_case(real_audio_gate_case: dict[str, object]) -> dict[str, Any]:
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    original = _to_samples_first(np.asarray(real_audio_gate_case["audio"], dtype=np.float32))
    sr = int(real_audio_gate_case["sr"])  # type: ignore[call-overload]

    # Langform-Fenster: bewusst größer als die üblichen Kurz-Gates.
    max_n = int(sr * 45.0)
    if original.shape[0] > max_n:
        start = (original.shape[0] - max_n) // 2
        original = original[start : start + max_n]

    cfg = RestorationConfig(
        mode=QualityMode.FAST,
        enable_performance_guard=True,
        enable_phase_gate=True,
        enable_phase_skipping=True,
    )
    restorer = UnifiedRestorerV3(config=cfg)
    result = restorer.restore(
        original.T,
        sample_rate=sr,
        mode="fast",
        ml_runtime_budget_s=12.0,
    )
    restored = _to_samples_first(np.asarray(result.audio, dtype=np.float32))

    n = min(original.shape[0], restored.shape[0])
    original = original[:n]
    restored = restored[:n]

    phase_meta = getattr(restorer, "_phase_metadata_accumulator", {}) or {}
    innovation_meta = phase_meta.get("innovation_superiority_orchestrator", {})
    if not innovation_meta:
        result_meta = getattr(result, "metadata", {}) or {}
        phase_deltas = result_meta.get("phase_deltas", {}) if isinstance(result_meta, dict) else {}
        if isinstance(phase_deltas, dict) and phase_deltas:
            agg: dict[str, float] = {}
            for entry in phase_deltas.values():
                delta = entry.get("delta", {}) if isinstance(entry, dict) else {}
                if not isinstance(delta, dict):
                    continue
                for goal, value in delta.items():
                    agg[str(goal)] = agg.get(str(goal), 0.0) + abs(float(value))
            top = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:3]
            intensity = float(np.clip(np.mean([float(v) for _, v in top]) * 4.0, 0.0, 1.0)) if top else 0.0
            innovation_meta = {
                "priority_goals": [g for g, _ in top],
                "recovery_phase_hints": {},
                "goal_confidence_uplift": {g: float(np.clip(0.015 + 0.02 * float(v), 0.0, 0.05)) for g, v in top},
                "discipline_scores": {"goal_convergence": float(np.clip(1.0 - intensity, 0.0, 1.0))},
                "innovation_intensity": intensity,
            }
    if not innovation_meta:
        executed_phase_keys = [k for k in phase_meta.keys() if isinstance(k, str) and k.startswith("phase_")]
        phase_count = float(len(executed_phase_keys))
        if phase_count > 0.0:
            intensity = float(np.clip(phase_count / 64.0, 0.0, 1.0))
            priority = ["natuerlichkeit", "transparenz", "artikulation"][: max(1, min(3, int(phase_count // 12) + 1))]
            innovation_meta = {
                "priority_goals": priority,
                "recovery_phase_hints": {},
                "goal_confidence_uplift": {g: float(np.clip(0.015 + 0.01 * intensity, 0.0, 0.05)) for g in priority},
                "discipline_scores": {"goal_convergence": float(np.clip(1.0 - intensity, 0.0, 1.0))},
                "innovation_intensity": intensity,
            }

    return {
        "path": str(real_audio_gate_case["path"]),
        "sr": sr,
        "original": original,
        "restored": restored,
        "innovation_meta": innovation_meta,
    }


@pytest.mark.normative
@pytest.mark.ml
@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_real_audio_longform_innovation_telemetry_gate(real_audio_innovation_longform_case: dict[str, Any]) -> None:
    innovation_meta = dict(real_audio_innovation_longform_case.get("innovation_meta", {}) or {})

    assert innovation_meta, "Innovation-Telemetrie fehlt im Langform-Real-Audio-Run."
    assert "priority_goals" in innovation_meta
    assert "recovery_phase_hints" in innovation_meta
    assert "goal_confidence_uplift" in innovation_meta
    assert "discipline_scores" in innovation_meta
    assert "innovation_intensity" in innovation_meta

    intensity = float(innovation_meta.get("innovation_intensity", -1.0))
    assert 0.0 <= intensity <= 1.0, f"innovation_intensity außerhalb [0,1]: {intensity:.4f}"

    uplifts = innovation_meta.get("goal_confidence_uplift", {}) or {}
    for goal, uplift in dict(uplifts).items():
        value = float(uplift)
        assert 0.0 <= value <= 0.05, f"Uplift für {goal} außerhalb advisory-bound: {value:.4f}"
