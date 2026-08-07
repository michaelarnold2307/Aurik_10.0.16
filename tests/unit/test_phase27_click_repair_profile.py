from __future__ import annotations

"""Unit tests for phase_27_click_pop_removal._compute_click_repair_profile (§2.54)."""


import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType
from backend.core.phases.phase_27_click_pop_removal import ClickPopRemoval


def _profile(material: str = "vinyl", qm: str = "balanced", rest: float = 50.0) -> dict[str, float]:
    return ClickPopRemoval._compute_click_repair_profile(material, qm, rest)


class TestBounds:
    @pytest.mark.parametrize("material", ["vinyl", "shellac", "tape", "cd_digital", "unknown"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    @pytest.mark.parametrize("rest", [0.0, 50.0, 100.0])
    def test_cubic_context_bounds(self, material, qm, rest):
        p = _profile(material, qm, rest)
        assert 3.0 <= p["cubic_context"] <= 12.0

    @pytest.mark.parametrize("material", ["vinyl", "shellac", "tape", "cd_digital", "unknown"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    @pytest.mark.parametrize("rest", [0.0, 50.0, 100.0])
    def test_ar_context_bounds(self, material, qm, rest):
        p = _profile(material, qm, rest)
        assert 64.0 <= p["ar_context"] <= 320.0

    @pytest.mark.parametrize("material", ["vinyl", "shellac", "tape", "cd_digital"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    def test_ar_order_never_below_16(self, material, qm):
        p = _profile(material, qm, 50.0)
        assert p["ar_order"] >= 16.0, "§VERBOTEN: ar_order must be >= 16"

    @pytest.mark.parametrize("material", ["vinyl", "shellac", "cd_digital"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    @pytest.mark.parametrize("rest", [10.0, 90.0])
    def test_ar_order_bounds(self, material, qm, rest):
        p = _profile(material, qm, rest)
        assert 16.0 <= p["ar_order"] <= 56.0

    @pytest.mark.parametrize("material", ["vinyl", "tape", "unknown"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    def test_crossfade_context_bounds(self, material, qm):
        p = _profile(material, qm, 50.0)
        assert 6.0 <= p["crossfade_context"] <= 24.0

    @pytest.mark.parametrize("material", ["vinyl", "shellac", "cd_digital", "unknown"])
    @pytest.mark.parametrize("qm", ["fast", "balanced", "quality", "maximum"])
    def test_taper_length_bounds(self, material, qm):
        p = _profile(material, qm, 50.0)
        assert 3.0 <= p["taper_length"] <= 12.0


class TestDirectional:
    def test_cubic_context_fast_lt_maximum(self):
        assert _profile("vinyl", "fast")["cubic_context"] < _profile("vinyl", "maximum")["cubic_context"]

    def test_ar_context_fast_lt_maximum(self):
        assert _profile("vinyl", "fast")["ar_context"] < _profile("vinyl", "maximum")["ar_context"]

    def test_ar_order_fast_lt_maximum(self):
        assert _profile("vinyl", "fast")["ar_order"] < _profile("vinyl", "maximum")["ar_order"]

    def test_crossfade_context_fast_lt_maximum(self):
        assert _profile("vinyl", "fast")["crossfade_context"] < _profile("vinyl", "maximum")["crossfade_context"]

    def test_taper_length_fast_lt_maximum(self):
        assert _profile("vinyl", "fast")["taper_length"] <= _profile("vinyl", "maximum")["taper_length"]


class TestMaterialEffect:
    def test_shellac_ar_context_ge_cd(self):
        for qm in ["fast", "balanced", "quality"]:
            assert _profile("shellac", qm)["ar_context"] >= _profile("cd_digital", qm)["ar_context"]

    def test_shellac_taper_ge_tape(self):
        assert _profile("shellac", "balanced")["taper_length"] >= _profile("tape", "balanced")["taper_length"]


class TestRestorability:
    def test_low_restorability_ar_context_ge_high(self):
        assert _profile("vinyl", "balanced", 10.0)["ar_context"] >= _profile("vinyl", "balanced", 90.0)["ar_context"]


class TestNaNSafety:
    def test_none_quality_mode(self):
        p = _profile("vinyl", None, 50.0)  # type: ignore[arg-type]
        assert all(np.isfinite(v) for v in p.values())

    def test_none_restorability(self):
        p = ClickPopRemoval._compute_click_repair_profile("vinyl", "balanced", None)  # type: ignore[arg-type]
        assert all(np.isfinite(v) for v in p.values())

    def test_garbage_mode(self):
        p = _profile("vinyl", "turbo_max_xyz", 50.0)
        assert all(np.isfinite(v) for v in p.values())

    def test_empty_material(self):
        p = _profile("", "balanced", 50.0)
        assert all(np.isfinite(v) for v in p.values())


class TestKeys:
    def test_all_keys_present(self):
        p = _profile()
        assert {"cubic_context", "ar_context", "ar_order", "crossfade_context", "taper_length"} <= set(p.keys())


class TestIntegration:
    def test_process_returns_profile_in_metadata(self):
        phase = ClickPopRemoval()
        sr = 48000
        t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
        audio = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        # inject a click
        audio[sr // 4] = 0.9

        result = phase.process(
            audio, sr, material=MaterialType.VINYL, quality_mode="balanced", restorability_score=60.0
        )
        assert "click_repair_profile" in result.metadata
        p = result.metadata["click_repair_profile"]
        assert "cubic_context" in p and "ar_order" in p

    def test_fast_vs_maximum_profile_differs(self):
        phase = ClickPopRemoval()
        sr = 48000
        audio = (0.4 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32))).astype(
            np.float32
        )
        audio[sr // 4] = 0.9

        r_fast = phase.process(audio, sr, material=MaterialType.VINYL, quality_mode="fast")
        r_max = phase.process(audio, sr, material=MaterialType.VINYL, quality_mode="maximum")
        assert r_fast.metadata["click_repair_profile"]["ar_order"] < r_max.metadata["click_repair_profile"]["ar_order"]

    def test_phase27_uses_lge_singleton_accessor(self, monkeypatch):
        phase = ClickPopRemoval()
        sr = 48000
        t = np.linspace(0, 0.2, int(sr * 0.2), dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        calls = {"count": 0}

        class _LgeStub:
            def get_phoneme_mask(self, audio, sr, hop_length=512):
                return np.zeros(max(1, len(audio) // hop_length), dtype=bool)

        def _get_lge():
            calls["count"] += 1
            return _LgeStub()

        monkeypatch.setattr("backend.core.phases.phase_27_click_pop_removal._get_phase27_lge", _get_lge)

        result = phase.process(
            audio, sr, material=MaterialType.VINYL, quality_mode="balanced", restorability_score=60.0
        )
        assert result.success is True
        assert calls["count"] >= 1

    def test_phase27_uses_npd_singleton_accessor(self, monkeypatch):
        phase = ClickPopRemoval()
        sr = 48000
        t = np.linspace(0, 0.2, int(sr * 0.2), dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        calls = {"count": 0}

        class _NpdMaskStub:
            def get_protected_mask(self, n_samples, _sr):
                return np.zeros(n_samples, dtype=bool)

        class _NpdStub:
            def detect(self, _audio, _sr):
                return _NpdMaskStub()

        def _get_npd():
            calls["count"] += 1
            return _NpdStub()

        monkeypatch.setattr("backend.core.phases.phase_27_click_pop_removal._get_phase27_npd", _get_npd)

        result = phase.process(
            audio, sr, material=MaterialType.VINYL, quality_mode="balanced", restorability_score=60.0
        )
        assert result.success is True
        assert calls["count"] >= 1
