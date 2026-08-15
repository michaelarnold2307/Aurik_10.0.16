import os

import pytest

from backend.core.unified_restorer_v3 import UnifiedRestorerV3


@pytest.mark.unit
def test_phase_intervention_registry_covers_all_phase_modules() -> None:
    phase_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "backend",
        "core",
        "phases",
    )
    phase_dir = os.path.abspath(phase_dir)

    module_phase_ids = {
        fname[:-3]
        for fname in os.listdir(phase_dir)
        if fname.startswith("phase_") and fname.endswith(".py") and fname not in {"phase_interface.py"}
    }
    registry = UnifiedRestorerV3.get_phase_intervention_registry()

    aliases = set(UnifiedRestorerV3._PHASE_ALIASES.keys())
    canonical_phase_ids = module_phase_ids | aliases

    missing = sorted(canonical_phase_ids - set(registry.keys()))
    assert not missing, f"Phases missing from intervention registry: {missing}"


def test_phase_intervention_registry_targets_64_phases() -> None:
    registry = UnifiedRestorerV3.get_phase_intervention_registry()
    # 66 nummerierte Phasen + phase_07_declipper + phase_glue_stage
    assert len(registry) == 68, f"Expected 68 registered phases, got {len(registry)}"
