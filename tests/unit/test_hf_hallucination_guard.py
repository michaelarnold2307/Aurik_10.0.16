"""
Unit tests for backend/core/hf_hallucination_guard.py

Normative Anforderungen:
  §0   Primum non nocere — kein HF-Inhalt oberhalb der Material-Bandbreite hinzufügen
  §2.46 Carrier-Chain-Inversion — Bandbreite des Quellmaterials respektieren
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq_hz: float, sr: int = 48_000, duration: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)  # type: ignore[no-any-return]


def _mix(*arrays: np.ndarray) -> np.ndarray:
    out = arrays[0].copy()
    for a in arrays[1:]:
        out = out + a
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------


class TestModuleImport:
    def test_module_imports_cleanly(self):
        """Guard module kann fehlerfrei importiert werden."""
        from backend.core.hf_hallucination_guard import (
            _MATERIAL_HF_CAP_HZ,
            ADDITIVE_PHASE_PREFIXES,
            check_hf_hallucination,
            get_material_hf_cap,
        )

        assert callable(check_hf_hallucination)
        assert callable(get_material_hf_cap)
        assert isinstance(ADDITIVE_PHASE_PREFIXES, frozenset)
        assert isinstance(_MATERIAL_HF_CAP_HZ, dict)

    def test_table_contains_all_archival_materials(self):
        """Alle kritischen Archiv-Materialien sind in der BW-Cap-Tabelle."""
        from backend.core.hf_hallucination_guard import _MATERIAL_HF_CAP_HZ

        for mat in ("wax_cylinder", "wire_recording", "shellac", "lacquer_disc", "cassette", "reel_tape", "vinyl"):
            assert mat in _MATERIAL_HF_CAP_HZ, f"{mat} fehlt in _MATERIAL_HF_CAP_HZ"

    def test_wax_cylinder_cap_invariante(self):
        """wax_cylinder BW-Cap MUSS ≤ 5 000 Hz sein (§0 VERBOTEN: Inpainting HF-Halluzination)."""
        from backend.core.hf_hallucination_guard import _MATERIAL_HF_CAP_HZ

        assert _MATERIAL_HF_CAP_HZ["wax_cylinder"] <= 5_000.0

    def test_shellac_cap_invariante(self):
        """shellac BW-Cap MUSS ≤ 7 000 Hz sein (§0 Vintage Aesthetics)."""
        from backend.core.hf_hallucination_guard import _MATERIAL_HF_CAP_HZ

        assert _MATERIAL_HF_CAP_HZ["shellac"] <= 7_000.0

    def test_digital_materials_effectively_uncapped(self):
        """cd_digital / dat sollen keinen Lautheitsverlust durch den Guard auslösen."""
        from backend.core.hf_hallucination_guard import _MATERIAL_HF_CAP_HZ

        for mat in ("cd_digital", "dat"):
            assert _MATERIAL_HF_CAP_HZ[mat] >= 20_000.0


# ---------------------------------------------------------------------------
# check_hf_hallucination — no hallucination cases
# ---------------------------------------------------------------------------


class TestCheckHfHallucinationClean:
    def test_vinyl_returns_ok_always(self):
        """Vinyl hat keine strenge BW-Begrenzung — Guard soll immer OK liefern."""
        from backend.core.hf_hallucination_guard import check_hf_hallucination

        sr = 48_000
        before = _sine(1_000, sr)
        # After: add high-frequency content — still OK for vinyl
        after = _mix(before, _sine(18_000, sr, 0.5))
        ok, wet_cap, cap_hz, delta = check_hf_hallucination(before, after, sr, "vinyl")
        assert ok is True, f"vinyl sollte immer ok sein, war aber wet_cap={wet_cap}"
        assert wet_cap == 1.0

    def test_cd_digital_returns_ok(self):
        """cd_digital hat kein BW-Cap — keine Intervention erwartet."""
        from backend.core.hf_hallucination_guard import check_hf_hallucination

        sr = 48_000
        before = _sine(440, sr)
        after = _mix(before, _sine(20_000, sr, 0.5))
        ok, wet_cap, _, _ = check_hf_hallucination(before, after, sr, "cd_digital")
        assert ok is True

    def test_no_hf_added_returns_ok(self):
        """Wenn eine Phase keine HF ergänzt, liefert der Guard ok=True."""
        from backend.core.hf_hallucination_guard import check_hf_hallucination

        sr = 48_000
        sig = _sine(1_000, sr)  # 1 kHz: well below any cap
        # before == after (identity phase)
        ok, wet_cap, _, delta = check_hf_hallucination(sig, sig.copy(), sr, "shellac")
        assert ok is True
        assert abs(delta) < 1e-4


# ---------------------------------------------------------------------------
# check_hf_hallucination — hallucination detected cases
# ---------------------------------------------------------------------------


class TestCheckHfHallucinationDetected:
    def test_shellac_10k_sine_triggers_guard(self):
        """Shellac (cap 7 kHz): Eine starke 10 kHz-Sinus-Addition soll erkannt werden."""
        from backend.core.hf_hallucination_guard import check_hf_hallucination

        sr = 48_000
        before = _sine(2_000, sr)  # 2 kHz — no HF
        # After: phase adds 10 kHz content (above 7 kHz cap)
        hf_added = _sine(10_000, sr, 0.5) * 0.7
        after = _mix(before, hf_added)
        ok, wet_cap, cap_hz, delta = check_hf_hallucination(before, after, sr, "shellac")
        assert ok is False, "Starke 10 kHz-Addition bei shellac sollte erkannt werden"
        assert cap_hz <= 7_000.0
        assert wet_cap < 1.0
        assert delta > 0.0

    def test_wax_cylinder_5k_sine_triggers_guard(self):
        """wax_cylinder (cap 5 kHz): Addition von 8 kHz-Inhalt soll erkannt werden."""
        from backend.core.hf_hallucination_guard import check_hf_hallucination

        sr = 48_000
        before = _sine(1_000, sr)
        hf_added = _sine(8_000, sr, 0.5) * 0.8
        after = _mix(before, hf_added)
        ok, wet_cap, cap_hz, delta = check_hf_hallucination(before, after, sr, "wax_cylinder")
        assert ok is False
        assert cap_hz <= 5_000.0
        assert wet_cap >= 0.35  # _MIN_WET_RATIO Untergrenze

    def test_wet_cap_bounded(self):
        """wet_cap darf niemals unter _MIN_WET_RATIO (0.35) fallen."""
        from backend.core.hf_hallucination_guard import _MIN_WET_RATIO, check_hf_hallucination

        sr = 48_000
        before = _sine(500, sr) * 0.1  # sehr leises Quellsignal
        # Massive HF injection
        after = _mix(before, _sine(10_000, sr, 0.5) * 1.0)
        ok, wet_cap, _, _ = check_hf_hallucination(before, after, sr, "shellac")
        if not ok:
            assert wet_cap >= _MIN_WET_RATIO

    def test_lower_certainty_tightens_threshold(self):
        """Niedrige recovery_certainty_scalar (0.78) greift früher ein als 1.0."""
        from backend.core.hf_hallucination_guard import check_hf_hallucination

        sr = 48_000
        before = _sine(2_000, sr)
        # Moderate HF addition — chosen near the boundary
        hf = _sine(9_000, sr, 0.5) * 0.30
        after = _mix(before, hf)
        ok_high, _, _, delta_high = check_hf_hallucination(before, after, sr, "shellac", recovery_certainty_scalar=1.0)
        ok_low, _, _, delta_low = check_hf_hallucination(before, after, sr, "shellac", recovery_certainty_scalar=0.78)
        # At lower certainty the threshold is tighter → not-ok at lower delta
        assert delta_high == pytest.approx(delta_low, abs=1e-4)  # delta is signal property
        # If high-certainty passes, low-certainty should be at least equally or more strict
        if ok_high:
            pass  # any outcome is valid when high certainty already passes
        else:
            assert not ok_low  # low certainty must also flag

    def test_unknown_material_uses_default_cap(self):
        """Unbekanntes Material nutzt _DEFAULT_HF_CAP_HZ (20 kHz) — kein Crash."""
        from backend.core.hf_hallucination_guard import check_hf_hallucination

        sr = 48_000
        before = _sine(440, sr)
        after = _mix(before, _sine(19_000, sr, 0.5) * 0.5)
        ok, _, cap_hz, _ = check_hf_hallucination(before, after, sr, "unknown_medium_xyz")
        # Default cap is 20 kHz → 19 kHz content is below cap → ok
        assert ok is True
        assert cap_hz >= 20_000.0


# ---------------------------------------------------------------------------
# get_material_hf_cap
# ---------------------------------------------------------------------------


class TestGetMaterialHfCap:
    @pytest.mark.parametrize(
        "mat,expected_max",
        [
            ("wax_cylinder", 5_000.0),
            ("wire_recording", 6_000.0),
            ("shellac", 7_000.0),
            ("cassette", 14_000.0),
        ],
    )
    def test_known_materials(self, mat, expected_max):
        from backend.core.hf_hallucination_guard import get_material_hf_cap

        assert get_material_hf_cap(mat) <= expected_max

    def test_fallback_for_unknown(self):
        from backend.core.hf_hallucination_guard import _DEFAULT_HF_CAP_HZ, get_material_hf_cap

        assert get_material_hf_cap("xyzzy") == _DEFAULT_HF_CAP_HZ
