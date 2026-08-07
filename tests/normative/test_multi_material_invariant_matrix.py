from __future__ import annotations

#!/usr/bin/env python3
"""Normative multi-material invariant matrix for critical late-pipeline phases.

Goal:
- Validate generalized robustness across material classes, quality modes,
  stereo layout variants and lightweight context signals (era/genre).
- Keep tests deterministic and fast (no heavy model dependency).

This gate is intentionally phase-focused and complements heavier e2e suites.
"""


import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_23_spectral_repair import SpectralRepair
from backend.core.phases.phase_41_output_format_optimization import OutputFormatOptimization
from backend.core.phases.phase_42_vocal_enhancement import VocalEnhancement

SR = 48_000
DUR_S = 1.0
N = int(SR * DUR_S)

MATERIAL_MATRIX = [
    MaterialType.VINYL,
    MaterialType.TAPE,
    MaterialType.CD_DIGITAL,
    MaterialType.MP3_LOW,
    MaterialType.STREAMING,
]

QUALITY_MODES = ["balanced", "quality", "maximum", "studio2026"]

# Lightweight contextual combinations to ensure code paths are context-robust.
CONTEXTS = [
    {"era_decade": 1930, "genre_label": "jazz_acoustic"},
    {"era_decade": 1970, "genre_label": "schlager"},
    {"era_decade": 2005, "genre_label": "rock"},
]


def _make_stereo_samples_first() -> np.ndarray:
    t = np.linspace(0.0, DUR_S, N, endpoint=False, dtype=np.float32)
    left = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.2 * np.sin(2 * np.pi * 660.0 * t)
    return np.column_stack([left, right]).astype(np.float32)


def _make_stereo_channels_first() -> np.ndarray:
    return _make_stereo_samples_first().T.copy()


@pytest.mark.parametrize("material", MATERIAL_MATRIX)
@pytest.mark.parametrize("quality_mode", QUALITY_MODES)
def test_phase41_pipeline_safety_material_mode_matrix(material: MaterialType, quality_mode: str):
    """Phase 41 must remain pipeline-safe independent of material/mode.

    Invariants:
    - Preserve in-pipeline sample rate (48 kHz)
    - Preserve floating pipeline output (bit depth metadata=32)
    - Preserve input shape
    - Keep delivery format as intended metadata only
    """
    phase = OutputFormatOptimization()
    audio = _make_stereo_samples_first()

    result = phase.process(audio, SR, material, quality_mode=quality_mode)  # type: ignore[arg-type]

    assert result.success is True
    assert result.audio.shape == audio.shape
    assert np.isfinite(result.audio).all()
    assert int(result.metrics["input_sample_rate"]) == SR
    assert int(result.metrics["output_sample_rate"]) == SR
    assert int(result.metrics["output_bit_depth"]) == 32
    assert "intended_output_sample_rate" in result.metrics
    assert "intended_output_bit_depth" in result.metrics
    assert bool(result.metadata.get("pipeline_safe_format_optimization", False)) is True


@pytest.mark.parametrize("layout", ["mono", "samples_first", "channels_first"])
def test_phase23_flashsr_helper_shape_invariant_matrix(layout: str):
    """Phase 23 FlashSR helper must preserve shape/orientation for all layouts."""

    class _FakeFlashSR:
        def process(self, audio, sr, target_sr):
            assert sr == SR
            assert target_sr == SR
            return np.asarray(audio, dtype=np.float32) * 0.5

    phase = SpectralRepair()
    phase._has_sufficient_ml_headroom = lambda *_args, **_kwargs: True  # type: ignore[method-assign]

    if layout == "mono":
        inp = _make_stereo_samples_first()[:, 0].copy()
    elif layout == "samples_first":
        inp = _make_stereo_samples_first()
    else:
        inp = _make_stereo_channels_first()

    out = phase._repair_with_flashsr(
        inp,
        SR,
        np.zeros((8, 8), dtype=bool),
        0.25,
        _FakeFlashSR(),
    )

    assert out.shape == inp.shape
    assert out.dtype == inp.dtype
    assert np.isfinite(out).all()


@pytest.mark.parametrize("material", MATERIAL_MATRIX)
@pytest.mark.parametrize("ctx", CONTEXTS)
def test_phase42_context_material_matrix_stable_with_stubbed_stem_sep(material: MaterialType, ctx: dict):
    """Phase 42 must stay stable across material/context combinations.

    Heavy stem-separation backends are stubbed to keep this matrix deterministic.
    """
    phase = VocalEnhancement()

    # Deterministic, lightweight stem split to exercise downstream vocal path.
    def _fake_stem_sep(audio, sr):
        audio = np.asarray(audio, dtype=np.float32)
        n = audio.shape[0] if audio.ndim == 2 else len(audio)
        if audio.ndim == 2:
            vocals = audio[:n] * 0.6
            instr = audio[:n] * 0.4
        else:
            vocals = audio[:n] * 0.6
            instr = audio[:n] * 0.4
        return vocals, instr, 0.7, "stubbed_stem_sep"

    phase._try_stem_separation = _fake_stem_sep  # type: ignore[assignment]

    audio = _make_stereo_samples_first()
    result = phase.process(audio, SR, material=material, **ctx)

    assert result.success is True
    assert result.audio.shape == audio.shape
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6
