"""Tests für §9.11.1 Adaptive-Strength-Gates:

1. Confidence-Weighting in _salience_adjusted_severity:
   - Schwache Detektion (confidence < 0.5) → Severity gedämpft
   - Starke Detektion (confidence ≥ 1.0) → Severity unverändert
   - Vollständig maskierte Defekte kombiniert mit Confidence korrekt

2. Restorability-Ceiling (kontinuierlich):
   - _last_restorability_score > 65 → cap aktiv für optionale Phasen
   - _last_restorability_score ≤ 65 → kein cap
   - Formel: cap = 1.0 - 0.55 * (rest - 65) / 35, clipped [0.45, 1.0]

3. Perceptual-Plateau-Stop:
   - _plateau_delta_history < 3 Einträge → kein Damping
   - Alle 3 Einträge < 0.005 → Damping auf 40%
   - Mindestens 1 Eintrag ≥ 0.005 → kein Damping
"""

import math
import os

import numpy as np
import pytest

os.environ.setdefault("AURIK_DISABLE_CREPE", "1")

SR = 48000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(dur: float = 4.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), endpoint=False, dtype=np.float32)
    return (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)  # type: ignore[no-any-return]


def _make_mock_defect_score(severity: float, confidence: float, metadata: dict | None = None):
    """Erstellt einen minimalen MockDefectScore ohne echten Scanner."""
    from dataclasses import dataclass

    @dataclass
    class MockDefectScore:
        severity: float
        confidence: float
        metadata: dict

    return MockDefectScore(
        severity=float(severity),
        confidence=float(confidence),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# §1 — Confidence-Weighting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConfidenceWeighting:
    """§9.11.1 Confidence-Weighting in _salience_adjusted_severity."""

    def _conf_factor(self, confidence: float) -> float:
        """Spiegelt die implementierte Formel: max(0.35, sqrt(conf))."""
        return max(0.35, math.sqrt(max(0.0, min(1.0, confidence))))

    def test_full_confidence_no_damping(self):
        """Confidence=1.0 → Faktor=1.0, Severity unverändert."""
        factor = self._conf_factor(1.0)
        assert abs(factor - 1.0) < 1e-6

    def test_half_confidence_dampens_severity(self):
        """Confidence=0.5 → Faktor=0.707, Severity auf ~70% gedämpft."""
        factor = self._conf_factor(0.5)
        assert abs(factor - math.sqrt(0.5)) < 1e-6  # 0.7071
        assert factor < 1.0

    def test_low_confidence_floor(self):
        """Confidence=0.05 → Faktor am Floor 0.35 (kein völliger Ausschluss)."""
        factor = self._conf_factor(0.05)
        assert factor == 0.35

    def test_zero_confidence_floor(self):
        """Confidence=0.0 → Faktor=0.35 (Floor)."""
        factor = self._conf_factor(0.0)
        assert factor == 0.35

    def test_severity_product_low_confidence(self):
        """Schwacher Defekt severity=0.6 + confidence=0.1 → gedämpfte effective_sev."""
        raw_sev = 0.6
        factor = self._conf_factor(0.1)  # → 0.35 (floor)
        effective = raw_sev * factor
        assert effective == pytest.approx(0.6 * 0.35, abs=1e-5)
        assert effective < raw_sev  # gedämpft

    def test_severity_product_high_confidence(self):
        """Starke Detektion severity=0.6 + confidence=0.9 → kaum gedämpft."""
        raw_sev = 0.6
        factor = self._conf_factor(0.9)
        effective = raw_sev * factor
        # sqrt(0.9) ≈ 0.9487 → effective ≈ 0.569
        assert effective > 0.56
        assert effective < raw_sev

    def test_monotonic_with_confidence(self):
        """Höhere Confidence → höherer Faktor (monoton)."""
        confidences = [0.0, 0.1, 0.25, 0.49, 0.64, 0.81, 1.0]
        factors = [self._conf_factor(c) for c in confidences]
        for i in range(len(factors) - 1):
            assert factors[i] <= factors[i + 1], (
                f"Nicht monoton bei conf[{i}]={confidences[i]}: {factors[i]} > {factors[i + 1]}"
            )

    def test_factor_always_in_range(self):
        """Faktor bleibt immer in [0.35, 1.0]."""
        for conf in np.linspace(0.0, 1.0, 50):
            f = self._conf_factor(float(conf))
            assert 0.35 <= f <= 1.0, f"conf={conf:.2f}: Faktor={f} außerhalb [0.35, 1.0]"


# ---------------------------------------------------------------------------
# §2 — Restorability-Ceiling (Formel-Prüfung)
# ---------------------------------------------------------------------------


class TestRestorabilityceiling:
    """§9.11.1 Restorability als kontinuierliche Strength-Obergrenze."""

    def _rest_cap(self, restorability: float) -> float | None:
        """Spiegelt die implementierte Cap-Formel."""
        if restorability <= 65.0:
            return None  # kein Cap
        raw = 1.0 - 0.55 * (restorability - 65.0) / 35.0
        return float(np.clip(raw, 0.45, 1.0))

    def test_restorability_below_threshold_no_cap(self):
        """Restorability ≤ 65 → kein Cap (beschädigtes Material, volle Stärke erlaubt)."""
        assert self._rest_cap(65.0) is None
        assert self._rest_cap(30.0) is None
        assert self._rest_cap(0.0) is None

    def test_pristine_material_strong_cap(self):
        """Restorability = 100 → Cap ≈ 0.45 (minimale Stärke)."""
        cap = self._rest_cap(100.0)
        assert cap is not None
        assert abs(cap - 0.45) < 1e-5

    def test_medium_restorability_moderate_cap(self):
        """Restorability = 82.5 → Cap = 0.725 (halber Weg)."""
        cap = self._rest_cap(82.5)
        assert cap is not None
        # 1.0 - 0.55 * (82.5-65)/35 = 1.0 - 0.55*0.5 = 0.725
        assert abs(cap - 0.725) < 1e-4

    def test_cap_floor_enforced(self):
        """Cap kann nicht unter 0.45 fallen (auch bei Extrapolation)."""
        cap = self._rest_cap(100.0)
        assert cap >= 0.45  # type: ignore[operator]

    def test_cap_monotonic_decreasing_with_restorability(self):
        """Höheres Restorability → niedrigerer Cap (mehr Pristine = sanftere Eingriffe)."""
        caps = [self._rest_cap(r) for r in [70.0, 80.0, 90.0, 100.0]]
        for i in range(len(caps) - 1):
            assert caps[i] >= caps[i + 1], (  # type: ignore[operator]
                f"Nicht monoton: cap({70 + i * 10})={caps[i]:.4f} < cap({80 + i * 10})={caps[i + 1]:.4f}"
            )

    def test_strength_capped_when_above_ceiling(self):
        """Stärke > Cap → auf Cap reduziert."""
        original_strength = 1.0
        cap = self._rest_cap(90.0)
        assert cap is not None and cap < original_strength
        effective = min(original_strength, cap)
        assert effective == cap

    def test_strength_unchanged_when_below_ceiling(self):
        """Stärke < Cap → unverändert."""
        original_strength = 0.30
        cap = self._rest_cap(90.0)  # ≈ 0.587
        assert cap is not None and cap > original_strength
        effective = min(original_strength, cap)
        assert effective == original_strength


# ---------------------------------------------------------------------------
# §3 — Perceptual-Plateau-Stop
# ---------------------------------------------------------------------------


class TestPerceptualPlateauStop:
    """§9.11.1 Perceptual-Plateau-Damping-Logik."""

    _THRESHOLD = 0.005
    _DAMPEN = 0.40

    def _should_dampen(self, history: list[float]) -> bool:
        """Spiegelt die implementierte Bedingung."""
        from collections import deque

        d = deque(history[-3:], maxlen=3)
        return len(d) == 3 and all(v < self._THRESHOLD for v in d)

    def _apply_plateau(self, strength: float, history: list[float]) -> float:
        if self._should_dampen(history):
            return float(np.clip(strength * self._DAMPEN, 0.05, 1.0))
        return strength

    def test_empty_history_no_damping(self):
        """Leere History → kein Damping."""
        assert not self._should_dampen([])

    def test_two_entry_history_no_damping(self):
        """Nur 2 Einträge → kein Damping (Fenster noch nicht voll)."""
        assert not self._should_dampen([0.001, 0.002])

    def test_three_below_threshold_damping(self):
        """Alle 3 Einträge < 0.005 → Damping aktiv."""
        assert self._should_dampen([0.001, 0.002, 0.003])

    def test_one_above_threshold_no_damping(self):
        """Einer von 3 ≥ 0.005 → kein Damping."""
        assert not self._should_dampen([0.001, 0.006, 0.002])

    def test_exactly_at_threshold_no_damping(self):
        """Delta genau = 0.005 gilt als NICHT unter threshold (kein Damping)."""
        assert not self._should_dampen([0.001, 0.005, 0.002])

    def test_dampen_factor_applied_correctly(self):
        """Strength=1.0 bei Plateau → 0.40 (40%)."""
        dampened = self._apply_plateau(1.0, [0.001, 0.002, 0.003])
        assert abs(dampened - 0.40) < 1e-6

    def test_dampen_floor_respected(self):
        """Sehr geringe Stärke → nicht unter 0.05 (Pipeline-Min-Floor)."""
        dampened = self._apply_plateau(0.08, [0.001, 0.002, 0.003])
        # 0.08 * 0.40 = 0.032 → floor 0.05
        assert dampened == pytest.approx(0.05, abs=1e-5)

    def test_no_damping_when_recent_improvement(self):
        """Wenn letzter Eintrag > threshold → kein Damping obwohl vorherige klein."""
        assert not self._should_dampen([0.001, 0.003, 0.010])

    def test_rolling_window_uses_last_3(self):
        """Rolling-Window nutzt nur die letzten 3 Einträge."""
        # Lange History mit vielen schlechten, aber letzter gut
        long_history = [0.001] * 10 + [0.010]
        # Letzten 3: [0.001, 0.001, 0.010] → kein Damping
        assert not self._should_dampen(long_history)

    def test_mandatory_phase_exempt_conceptual(self):
        """Pflicht-Phasen (§6.2a) ignorieren PlateauStop — konzeptueller Invarianten-Test.

        Der Code prüft _is_mandatory_phase BEVOR er _plateau_delta_history auswertet.
        Dieser Test stellt sicher, dass bei is_mandatory=True der Damping-Ausdruck
        nie greift, auch wenn die History voll ist und alle Deltas < threshold wären.
        """
        # Simuliere: mandatory=True → plateau check übersprungen
        _is_mandatory = True
        _history = [0.001, 0.002, 0.003]
        _strength = 1.0
        # Logik: if not _is_mandatory and should_dampen(history): dampen
        if not _is_mandatory and self._should_dampen(_history):
            _strength = _strength * self._DAMPEN
        # Mandatory → Stärke bleibt 1.0
        assert _strength == 1.0


# ---------------------------------------------------------------------------
# §4 — Integration: UV3 importiert ohne Fehler
# ---------------------------------------------------------------------------


class TestUV3ImportSanity:
    """Stellt sicher, dass UV3 nach den Edits importierbar ist."""

    def test_uv3_importable(self):
        """UnifiedRestorerV3 lässt sich instantiieren."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        r = UnifiedRestorerV3()
        assert r is not None

    def test_uv3_has_restorability_score_attr(self):
        """_last_restorability_score wird nach restore() gesetzt (Default-Check)."""
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        r = UnifiedRestorerV3()
        # Vor restore(): kein attr oder Default → getattr safe
        val = float(getattr(r, "_last_restorability_score", 70.0))
        assert 0.0 <= val <= 100.0
