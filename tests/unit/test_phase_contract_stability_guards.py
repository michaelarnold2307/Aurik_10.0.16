import inspect
import pathlib

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.per_phase_musical_goals_gate import PerPhaseMusicalGoalsGate
from backend.core.phases import phase_27_click_pop_removal as p27
from backend.core.phases.phase_10_compression import CompressionPhase
from backend.core.phases.phase_11_limiting import LimitingPhase
from backend.core.phases.phase_16_final_eq import FinalEQ
from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase


def test_phase27_has_stable_local_resolvers():
    assert hasattr(p27, "_get_phase27_lge")
    assert hasattr(p27, "_get_phase27_npd")
    assert callable(p27._get_phase27_lge)
    assert callable(p27._get_phase27_npd)


def test_phase27_process_uses_local_resolvers():
    src = inspect.getsource(p27.ClickPopRemoval.process)
    assert "_get_phase27_lge()" in src
    assert "_get_phase27_npd()" in src


def test_phase31_material_normalization_maps_enum_and_aliases():
    phase = SpeedPitchCorrectionPhase()

    assert phase._normalize_material_type(MaterialType.TAPE) == "tape"
    assert phase._normalize_material_type(MaterialType.CASSETTE) == "tape"
    assert phase._normalize_material_type("cassette") == "tape"
    assert phase._normalize_material_type("CD") == "cd_digital"
    assert phase._normalize_material_type("stream") == "unknown"


def test_phase31_material_normalization_falls_back_to_unknown():
    phase = SpeedPitchCorrectionPhase()
    assert phase._normalize_material_type("totally_custom_medium") == "unknown"


def test_all_phase_modules_have_strength_contract_path():
    phases_dir = pathlib.Path(__file__).resolve().parents[2] / "backend" / "core" / "phases"
    phase_files = sorted(phases_dir.glob("phase_*.py"))
    assert phase_files, "Keine phase_*.py Dateien gefunden"

    # §III.4 (copilot-instructions): Glue Stage läuft in ALLEN Modi als
    # vorletzte Phase mit fester Intensität — kein PMGG-Strength-Scaling
    # per Design, daher legitim ohne Strength-Contract-Pfad.
    _NO_STRENGTH_CONTRACT_EXEMPT = frozenset({"phase_glue_stage.py"})

    offenders = []
    direct_strength_patterns = (
        'kwargs.get("strength"',
        "kwargs.get('strength'",
        "strength: float",
    )
    strength_contract_pattern = "resolve_phase_strength_contract("

    for phase_file in phase_files:
        if phase_file.name == "phase_interface.py" or phase_file.name in _NO_STRENGTH_CONTRACT_EXEMPT:
            continue

        source = phase_file.read_text(encoding="utf-8")
        has_direct_strength = any(pattern in source for pattern in direct_strength_patterns)
        has_strength_contract = strength_contract_pattern in source

        if not (has_direct_strength or has_strength_contract):
            offenders.append(phase_file.name)

    assert not offenders, (
        "Diese Phasen haben weder direkte Strength-Nutzung noch den zentralen Strength-Contract: "
        + ", ".join(offenders)
    )


def test_pmgg_run_phase_injects_strength_into_phase_kwargs():
    gate = PerPhaseMusicalGoalsGate()
    audio = np.zeros(480, dtype=np.float32)

    class _StrengthRecorderPhase:
        def __init__(self):
            self.seen_strength = None

        def process(self, in_audio, **kwargs):
            self.seen_strength = kwargs.get("strength")
            return np.asarray(in_audio, dtype=np.float32)

    phase = _StrengthRecorderPhase()
    out = gate._run_phase(phase, audio, 0.37, {"sample_rate": 48000})

    assert np.array_equal(out, audio)
    assert phase.seen_strength is not None
    assert phase.seen_strength == 0.37


@pytest.mark.parametrize(
    "phase_factory,phase_kwargs",
    [
        (CompressionPhase, {"sample_rate": 48000, "material": "vinyl", "mode": "restoration"}),
        (LimitingPhase, {"sample_rate": 48000, "material": "vinyl", "mode": "restoration"}),
        (FinalEQ, {"sample_rate": 48000, "material": "vinyl", "mode": "restoration"}),
    ],
)
def test_real_phases_strength_zero_skips_and_full_strength_applies(phase_factory, phase_kwargs):
    gate = PerPhaseMusicalGoalsGate()
    t = np.linspace(0.0, 0.08, int(0.08 * 48000), endpoint=False, dtype=np.float32)
    # Kombination aus lautem Ton + Oberton, damit Dynamics/EQ-Phasen verlässlich reagieren.
    audio = np.clip(0.92 * np.sin(2.0 * np.pi * 440.0 * t) + 0.28 * np.sin(2.0 * np.pi * 1760.0 * t), -1.0, 1.0)
    phase = phase_factory()

    out_zero = gate._run_phase(phase, audio, 0.0, dict(phase_kwargs))
    out_full = gate._run_phase(phase, audio, 1.0, dict(phase_kwargs))

    assert out_zero.shape == audio.shape
    assert out_full.shape == audio.shape
    assert np.all(np.isfinite(out_zero))
    assert np.all(np.isfinite(out_full))
    assert np.max(np.abs(out_zero - audio)) <= 1e-6

    delta_full = float(np.mean(np.abs(out_full - audio)))
    assert delta_full > 1e-6
