from __future__ import annotations

"""Unit tests for phase_24_dropout_repair._compute_inpainting_runtime_profile (§2.54).

Tests
-----
- Bounds: all keys within declared ranges
- Directional: fast < maximum for quality-dependent keys
- Restorability: low restorability ≠ baseline values
- NaN safety: None/NaN inputs produce valid float output
- Integration: process() returns metadata["inpainting_runtime_profile"]
"""


import numpy as np
import pytest

from backend.core.phases.phase_24_dropout_repair import DropoutRepairPhase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(material: str = "vinyl", qm: str = "balanced", rest: float = 50.0) -> dict[str, float]:
    return DropoutRepairPhase._compute_inpainting_runtime_profile(material, qm, rest)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


class TestBounds:
    @pytest.mark.parametrize(
        "material", ["vinyl", "shellac", "tape", "reel_tape", "cassette", "cd_digital", "mp3_low", "unknown"]
    )
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    @pytest.mark.parametrize("rest", [0.0, 50.0, 100.0])
    def test_tonal_top_k_bounds(self, material, qm, rest):
        p = _profile(material, qm, rest)
        assert 8.0 <= p["tonal_top_k"] <= 56.0, f"tonal_top_k out of range: {p['tonal_top_k']}"

    @pytest.mark.parametrize("material", ["vinyl", "shellac", "cd_digital", "unknown"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    @pytest.mark.parametrize("rest", [10.0, 90.0])
    def test_atonal_nmf_rank_bounds(self, material, qm, rest):
        p = _profile(material, qm, rest)
        assert 4.0 <= p["atonal_nmf_rank"] <= 20.0, f"atonal_nmf_rank out of range: {p['atonal_nmf_rank']}"

    @pytest.mark.parametrize("material", ["tape", "shellac", "mp3_low", "unknown"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    @pytest.mark.parametrize("rest", [10.0, 90.0])
    def test_atonal_nmf_iterations_bounds(self, material, qm, rest):
        p = _profile(material, qm, rest)
        assert 10.0 <= p["atonal_nmf_iterations"] <= 90.0, (
            f"atonal_nmf_iterations out of range: {p['atonal_nmf_iterations']}"
        )

    @pytest.mark.parametrize("material", ["vinyl", "shellac", "tape", "cd_digital", "unknown"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    @pytest.mark.parametrize("rest", [0.0, 50.0, 100.0])
    def test_hybrid_tonal_blend_bounds(self, material, qm, rest):
        p = _profile(material, qm, rest)
        assert 0.35 <= p["hybrid_tonal_blend"] <= 0.70, f"hybrid_tonal_blend out of range: {p['hybrid_tonal_blend']}"


# ---------------------------------------------------------------------------
# Directional: fast < maximum
# ---------------------------------------------------------------------------


class TestDirectional:
    def test_tonal_top_k_fast_lt_maximum(self):
        p_fast = _profile("vinyl", "fast", 50.0)
        p_max = _profile("vinyl", "maximum", 50.0)
        assert p_fast["tonal_top_k"] < p_max["tonal_top_k"]

    def test_atonal_nmf_rank_fast_lt_maximum(self):
        p_fast = _profile("vinyl", "fast", 50.0)
        p_max = _profile("vinyl", "maximum", 50.0)
        assert p_fast["atonal_nmf_rank"] < p_max["atonal_nmf_rank"]

    def test_atonal_nmf_iterations_fast_lt_maximum(self):
        p_fast = _profile("vinyl", "fast", 50.0)
        p_max = _profile("vinyl", "maximum", 50.0)
        assert p_fast["atonal_nmf_iterations"] < p_max["atonal_nmf_iterations"]

    def test_hybrid_blend_fast_lt_maximum(self):
        # Higher quality → more tonal character preserved
        p_fast = _profile("vinyl", "fast", 50.0)
        p_max = _profile("vinyl", "maximum", 50.0)
        assert p_fast["hybrid_tonal_blend"] <= p_max["hybrid_tonal_blend"]


# ---------------------------------------------------------------------------
# Restorability effect
# ---------------------------------------------------------------------------


class TestRestorability:
    def test_low_restorability_top_k_ge_high(self):
        # Low restorability = complex signal → more or equal sinusoids
        p_low = _profile("vinyl", "balanced", 10.0)
        p_high = _profile("vinyl", "balanced", 90.0)
        assert p_low["tonal_top_k"] >= p_high["tonal_top_k"]

    def test_low_restorability_nmf_iterations_ge_high(self):
        p_low = _profile("shellac", "balanced", 10.0)
        p_high = _profile("shellac", "balanced", 90.0)
        assert p_low["atonal_nmf_iterations"] >= p_high["atonal_nmf_iterations"]


# ---------------------------------------------------------------------------
# Material: analog gets more sinusoids than digital
# ---------------------------------------------------------------------------


class TestMaterialEffect:
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    def test_shellac_top_k_ge_cd(self, qm):
        p_shellac = _profile("shellac", qm, 50.0)
        p_cd = _profile("cd_digital", qm, 50.0)
        assert p_shellac["tonal_top_k"] >= p_cd["tonal_top_k"]


# ---------------------------------------------------------------------------
# NaN / None safety
# ---------------------------------------------------------------------------


class TestNaNSafety:
    def test_none_quality_mode(self):
        p = _profile("vinyl", None, 50.0)  # type: ignore[arg-type]
        assert all(np.isfinite(v) for v in p.values())

    def test_none_restorability(self):
        p = DropoutRepairPhase._compute_inpainting_runtime_profile("vinyl", "balanced", None)  # type: ignore[arg-type]
        assert all(np.isfinite(v) for v in p.values())

    def test_garbage_quality_mode(self):
        p = _profile("vinyl", "unknown_mode_xyz", 50.0)
        assert all(np.isfinite(v) for v in p.values())

    def test_empty_material(self):
        p = _profile("", "balanced", 50.0)
        assert all(np.isfinite(v) for v in p.values())


# ---------------------------------------------------------------------------
# Required keys present
# ---------------------------------------------------------------------------


class TestKeys:
    def test_all_required_keys_present(self):
        p = _profile()
        required = {"tonal_top_k", "atonal_nmf_rank", "atonal_nmf_iterations", "hybrid_tonal_blend"}
        assert required <= set(p.keys())


# ---------------------------------------------------------------------------
# Integration: process() returns profile in metadata
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_process_returns_profile_in_metadata(self):
        phase = DropoutRepairPhase()
        sr = 48000
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # Introduce a small dropout
        dropout_start = sr // 4
        dropout_end = dropout_start + int(sr * 0.01)  # 10 ms
        audio[dropout_start:dropout_end] *= 0.0

        result = phase.process(audio, sr, material_type="vinyl", quality_mode="balanced", restorability_score=60.0)
        meta = result.metadata
        assert "inpainting_runtime_profile" in meta
        p = meta["inpainting_runtime_profile"]
        assert "tonal_top_k" in p
        assert "atonal_nmf_rank" in p
        assert "atonal_nmf_iterations" in p
        assert "hybrid_tonal_blend" in p

    def test_fast_vs_maximum_profile_differs_in_process(self):
        phase = DropoutRepairPhase()
        sr = 48000
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        ds = sr // 4
        audio[ds : ds + int(sr * 0.01)] = 0.0

        r_fast = phase.process(audio, sr, material_type="vinyl", quality_mode="fast")
        r_max = phase.process(audio, sr, material_type="vinyl", quality_mode="maximum")
        p_fast = r_fast.metadata["inpainting_runtime_profile"]
        p_max = r_max.metadata["inpainting_runtime_profile"]
        assert p_fast["tonal_top_k"] < p_max["tonal_top_k"]
        assert p_fast["atonal_nmf_iterations"] < p_max["atonal_nmf_iterations"]
