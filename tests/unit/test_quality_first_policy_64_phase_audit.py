from __future__ import annotations

import pytest

"""Quality-first contract audit across all 64 phases.

This test enforces two invariants:
1) The phase set is complete (64 phase files).
2) Known time-gate patterns must be paired with a quality-first gate marker.
"""


import re
from pathlib import Path


@pytest.mark.unit
def test_phase_file_count_is_64() -> None:
    root = Path(__file__).resolve().parents[2]
    phases = sorted((root / "backend/core/phases").glob("phase_[0-9][0-9]_*.py"))
    # 66 nummerierte Phasen + phase_07_declipper (nummermuster-konform)
    assert len(phases) == 67, f"Expected 67 phase files, found {len(phases)}"


def test_time_gates_are_quality_gated_for_high_end_modes() -> None:
    root = Path(__file__).resolve().parents[2]
    phase_files = sorted((root / "backend/core/phases").glob("phase_[0-9][0-9]_*.py"))

    risk_patterns = [
        r"join\(timeout=",
        r"short_clip_guard",
        r"long_audio_",
        r"_PYIN_CAP_S",
        r"_STAGE_CAP_S",
        r"max_runtime_s\s*=",
    ]
    risk_re = re.compile("|".join(risk_patterns))

    quality_markers = [
        "quality_first_unleashed",
        'quality_mode in ("quality", "maximum")',
        'quality_mode not in ("quality", "maximum")',
        '_qm_hint in {"quality", "maximum"}',
        "_quality_mode_hint",
    ]

    offenders: list[str] = []
    for pf in phase_files:
        text = pf.read_text(encoding="utf-8")
        if risk_re.search(text):
            if not any(marker in text for marker in quality_markers):
                offenders.append(str(pf))

    assert not offenders, "Time-gated phase paths without quality-first gating found:\n" + "\n".join(offenders)
