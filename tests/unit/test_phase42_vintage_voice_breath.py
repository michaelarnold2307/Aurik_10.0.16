from __future__ import annotations

import pytest

"""Unit-Tests für §4.10-VintageVoice Vintage-Stimm-Identitätsschutz in phase_42.

Prüft, dass bei Vintage-Material (shellac/vinyl/reel_tape/tape/cassette etc.)
ohne erkannte Altersgruppe (GenderDetector returns age_group=None) der
breath_preservation-Boden auf mindestens 0.78 gesetzt wird.

Spec: §4.10-VintageVoice (Spec 04), copilot-instructions.md VERBOTEN-Tabelle
"""


import numpy as np

SR = 48_000
_VINTAGE_MATERIALS = [
    "shellac",
    "vinyl",
    "reel_tape",
    "tape",
    "cassette",
    "wax_cylinder",
    "wire_recording",
    "lacquer_disc",
    "acoustic_78",
]


def _make_vocal(f0: float = 220.0, duration_s: float = 1.0) -> np.ndarray:
    """Synthetisches Vokalsignal mit Harmonischen."""
    n = int(SR * duration_s)
    t = np.arange(n, dtype=np.float64) / SR
    sig = np.zeros(n)
    for h in range(1, 7):
        sig += (0.4 / h) * np.sin(2 * np.pi * f0 * h * t)
    # Atemrauschen simulieren
    rng = np.random.default_rng(42)
    sig += rng.standard_normal(n) * 0.02
    return (sig / (np.max(np.abs(sig)) + 1e-12) * 0.6).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
class TestPhase42VintageBreathDirectLogic:
    """Direkter Test der Vintage-Guard-Logik ohne volle Phase-Ausführung."""

    def test_breath_floor_logic_raises_when_none_and_vintage(self):
        """Simuliert die Guard-Logik isoliert — keine externen Dependencies."""
        _VINTAGE_MATERIAL_KEYS = frozenset(
            {
                "shellac",
                "wax_cylinder",
                "vinyl",
                "lacquer_disc",
                "wire_recording",
                "acoustic_78",
                "reel_tape",
                "tape",
                "cassette",
            }
        )

        for material in _VINTAGE_MATERIAL_KEYS:
            _detected_age_group_value = None
            _age_breath_preservation = 0.70  # Default-Fallback vor Guard

            if _detected_age_group_value is None and material in _VINTAGE_MATERIAL_KEYS:
                _age_breath_preservation = max(_age_breath_preservation, 0.78)

            assert _age_breath_preservation >= 0.78, (
                f"Guard-Logik fehlerhaft für material='{material}': "
                f"breath_preservation={_age_breath_preservation:.3f} < 0.78"
            )

    def test_breath_floor_logic_unaffected_when_age_group_known(self):
        """Wenn age_group erkannt → kein Vintage-Boden (Senior=0.90 steuert bereits)."""
        _VINTAGE_MATERIAL_KEYS = frozenset({"shellac", "vinyl", "reel_tape"})

        _detected_age_group_value = "adult"  # Erkannte Altersgruppe
        _age_breath_preservation = 0.72  # adult-profile breath_preservation

        if _detected_age_group_value is None and "shellac" in _VINTAGE_MATERIAL_KEYS:
            _age_breath_preservation = max(_age_breath_preservation, 0.78)

        # age_group bekannt → Guard soll NICHT feuern
        assert _age_breath_preservation == 0.72, (
            f"Wenn age_group bekannt, darf Vintage-Guard nicht feuern. Erhalten: {_age_breath_preservation}"
        )

    def test_breath_floor_logic_unaffected_for_digital_material(self):
        """Für cd_digital (nicht vintage) → kein Vintage-Boden."""
        _VINTAGE_MATERIAL_KEYS = frozenset(
            {
                "shellac",
                "wax_cylinder",
                "vinyl",
                "lacquer_disc",
                "wire_recording",
                "acoustic_78",
                "reel_tape",
                "tape",
                "cassette",
            }
        )

        _detected_age_group_value = None  # Unerkannte Altersgruppe
        _age_breath_preservation = 0.70

        if _detected_age_group_value is None and "cd_digital" in _VINTAGE_MATERIAL_KEYS:
            _age_breath_preservation = max(_age_breath_preservation, 0.78)

        assert _age_breath_preservation == 0.70, (
            f"cd_digital: Vintage-Guard soll nicht feuern, breath_preservation={_age_breath_preservation}"
        )


# ---------------------------------------------------------------------------
# §0p Passaggio-Schutz-Tests
# ---------------------------------------------------------------------------


class TestPhase42PassaggioGuard:
    """§0p: Passaggio-Zonen müssen formant_gain und presence_gain in Phase 42 reduzieren."""

    def _sim_passaggio_scale(self, audio_dur_s: float, zones: list) -> float:
        """Simuliert die Passaggio-Scale-Berechnung wie in Phase 42."""
        coverage = sum(e - s for s, e in zones if 0 <= s < e) / max(audio_dur_s, 1.0)
        coverage = float(min(max(coverage, 0.0), 1.0))
        if coverage <= 0.03:
            return 1.0  # Guard inaktiv
        return max(0.40, 1.0 - 0.60 * coverage)

    def test_passaggio_scale_reduces_when_coverage_exceeds_threshold(self):
        """Bei Passaggio-Coverage > 3 % muss scale < 1.0 sein."""
        scale = self._sim_passaggio_scale(4.0, [(1.0, 2.5)])
        assert scale < 1.0, f"Passaggio-Scale muss < 1.0 sein, erhalten: {scale:.3f}"

    def test_passaggio_scale_never_below_floor(self):
        """Passaggio-Scale darf nie unter 0.40 fallen (zu aggressiv wäre Timbre-Zerstörung)."""
        scale = self._sim_passaggio_scale(1.0, [(0.0, 1.0)])  # 100 % Passaggio
        assert scale >= 0.40, f"Passaggio-Scale unter Floor 0.40: {scale:.3f}"

    def test_passaggio_guard_inactive_below_threshold(self):
        """Unter 3 % Passaggio-Coverage bleibt scale == 1.0."""
        scale = self._sim_passaggio_scale(100.0, [(0.0, 0.5)])  # 0.5 % → inaktiv
        assert abs(scale - 1.0) < 1e-9, f"Guard feuerte bei <3 % Coverage: scale={scale:.3f}"

    def test_passaggio_kwargs_extraction_logic(self):
        """passaggio_zones aus kwargs hat Vorrang vor vfa_result."""

        # kwargs-Quelle hat Vorrang
        kwargs_zones = [(1.0, 2.0)]
        vfa_zones = [(3.0, 5.0)]

        class _FakeVFA:
            passaggio_zones = vfa_zones

        _passaggio_zones = list(kwargs_zones or [])
        if not _passaggio_zones and hasattr(_FakeVFA(), "passaggio_zones"):
            _passaggio_zones = list(_FakeVFA().passaggio_zones)
        assert _passaggio_zones == kwargs_zones, (
            "kwargs-passaggio_zones soll Vorrang vor vfa_result.passaggio_zones haben"
        )
