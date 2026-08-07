import pytest

"""Unit-Tests: MidSideProcessing._compute_mid_side_profile() (§2.56)

Verifiziert dass der adaptive transient_preserve-Parameter
korrekt aus Material, Quality-Mode und Restorability berechnet wird.
"""

import numpy as np

from backend.core.phases.phase_34_mid_side_processing import MidSideProcessing

# ---------------------------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------------------------


def _profile(material: str, qm: str = "balanced", rest: float = 50.0) -> dict:
    return MidSideProcessing._compute_mid_side_profile(material, qm, rest)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMidSideProfileMaterial:
    """Material-adaptive Basiswerte."""

    def test_shellac_higher_than_cd(self):
        """Shellac benötigt stärkere Transientenbewahrung als CD."""
        val_shellac = _profile("shellac")["transient_preserve"]
        val_cd = _profile("cd_digital")["transient_preserve"]
        assert val_shellac > val_cd

    def test_vinyl_range(self):
        val = _profile("vinyl")["transient_preserve"]
        assert 0.70 <= val <= 0.90, f"vinyl transient_preserve out of range: {val}"

    def test_cd_digital_range(self):
        val = _profile("cd_digital")["transient_preserve"]
        assert 0.55 <= val <= 0.80, f"cd_digital transient_preserve out of range: {val}"

    def test_wax_cylinder_high(self):
        """Wax Cylinder: gleich hohe Bewahrung wie Shellac."""
        val = _profile("wax_cylinder")["transient_preserve"]
        assert val >= 0.78, f"wax_cylinder too low: {val}"

    def test_mp3_low_medium(self):
        val = _profile("mp3_low")["transient_preserve"]
        assert 0.60 <= val <= 0.85, f"mp3_low out of range: {val}"


class TestMidSideProfileQualityMode:
    """Quality-Mode-Anpassungen."""

    def test_quality_increases_preserve(self):
        base = _profile("vinyl", "balanced")["transient_preserve"]
        high = _profile("vinyl", "quality")["transient_preserve"]
        assert high > base

    def test_maximum_increases_preserve(self):
        base = _profile("vinyl", "balanced")["transient_preserve"]
        maxv = _profile("vinyl", "maximum")["transient_preserve"]
        assert maxv > base

    def test_fast_decreases_preserve(self):
        base = _profile("vinyl", "balanced")["transient_preserve"]
        fast = _profile("vinyl", "fast")["transient_preserve"]
        assert fast < base

    def test_none_quality_mode_uses_default(self):
        val = _profile("vinyl", None)["transient_preserve"]  # type: ignore[arg-type]
        assert 0.50 <= val <= 0.95


class TestMidSideProfileRestorability:
    """Restorability-Einfluss."""

    def test_low_restorability_increases_preserve(self):
        high_rest = _profile("vinyl", "balanced", 70.0)["transient_preserve"]
        low_rest = _profile("vinyl", "balanced", 25.0)["transient_preserve"]
        assert low_rest > high_rest

    def test_bounds_respected(self):
        """Kombination extremer Parameter bleibt in [0.50, 0.95]."""
        # Extremfall: Shellac + maximum + low-rest
        val_high = _profile("shellac", "maximum", 5.0)["transient_preserve"]
        # Extremfall: CD + fast + high-rest
        val_low = _profile("cd_digital", "fast", 95.0)["transient_preserve"]
        assert 0.50 <= val_high <= 0.95, f"upper extreme out of bounds: {val_high}"
        assert 0.50 <= val_low <= 0.95, f"lower extreme out of bounds: {val_low}"

    def test_unknown_material_fallback(self):
        val = _profile("unknown_material_xyz")["transient_preserve"]
        assert 0.50 <= val <= 0.95


class TestMidSideProfileIntegration:
    """End-to-End: Profil fließt in PhaseResult metadata."""

    def test_profile_key_in_metadata(self):
        phase = MidSideProcessing(sample_rate=48000)
        audio = np.random.uniform(-0.3, 0.3, (48000, 2)).astype(np.float32)
        result = phase.process(
            audio,
            48000,
            quality_mode="quality",
            restorability_score=60.0,
            strength=0.5,
        )
        assert result.success
        assert "mid_side_profile" in result.metadata
        assert "transient_preserve" in result.metadata
        # Wert muss im gültigen Bereich liegen
        tp = result.metadata["transient_preserve"]
        assert 0.50 <= tp <= 0.95
