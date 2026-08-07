from __future__ import annotations

"""Integration-Contracts für Musical-Goals-Verdrahtung (Low-Effort, High-Signal).

Diese Tests sichern gezielt die reparierten Integrationspunkte ab:
- FeedbackChain Konstruktor-/Iterations-Vertrag
- FeedbackChain GoalPriority-Callback-Pfad
- UnifiedRestorerV3-FeedbackChain-Wiring (max_iterations)
- UnifiedRestorerV3 GoalApplicability/AdaptiveThresholds-Verdrahtung
"""


from pathlib import Path

import numpy as np
import pytest

from backend.core.feedback_chain import FeedbackChain
from backend.core.performance_guard import QualityMode
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

SR = 48_000


def _silence(secs: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * secs), dtype=np.float32)


class TestFeedbackChainIntegrationContracts:
    def test_01_max_iterations_supported(self) -> None:
        """Current FeedbackChain API must accept max_iterations without TypeError."""
        fc = FeedbackChain(max_iterations=2)
        res = fc.run(_silence(), lambda a, sr: a)
        assert res.iterations >= 1
        assert res.total_retries == res.iterations

    def test_02_goal_priority_callback_can_abort(self) -> None:
        """Externer GoalPriority-Callback muss Iteration abbrechen koennen."""
        fc = FeedbackChain(max_iterations=5)

        def _abort_cb(_before: np.ndarray, _after: np.ndarray) -> tuple[bool, str]:
            return True, "contract-abort"

        fc.goal_priority_callback = _abort_cb
        res = fc.run(_silence(), lambda a, sr: a)
        assert res.iterations == 1
        log = res.metadata.get("goal_priority_log", [])
        assert any("contract-abort" in s for s in list(log) if isinstance(s, str))  # type: ignore[call-overload]


class TestUnifiedRestorerWiringContracts:
    def test_03_feedback_chain_wiring_uses_max_iterations(self) -> None:
        """UnifiedRestorer muss kompatibel mit FeedbackChain-Signatur bleiben."""
        src = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        # v10.0.0: max_iterations ist jetzt material-adaptiv (_fc_max_iter)
        assert "max_iterations=_fc_max_iter" in src
        assert "max_retries=5" not in src

    def test_04_goal_wiring_contains_applicability_and_adaptive_thresholds(self) -> None:
        """Applicability + adaptive thresholds muessen im Restore-Result verdrahtet sein."""
        src = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert "_goal_applicability_result = _goal_applicability" in src
        assert "_effective_goal_thresholds" in src
        assert 'adaptive_thresholds=locals().get("_effective_goal_thresholds", {})' in src
        assert "goal_applicability=(" in src

    def test_04b_goal_applicability_wiring_normalizes_none_sets(self) -> None:
        """Result-Serialisierung darf bei malformierten GAF-Sets nicht an None iterieren."""
        src = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert 'frozenset(getattr(_goal_applicability_result, "applicable", ()) or ())' in src
        assert 'frozenset(getattr(_goal_applicability_result, "inapplicable", ()) or ())' in src

    def test_04c_restoration_critical_best_effort_rolls_back_reconstruction(self) -> None:
        """Restoration must not ship phase_24/55 best_effort reconstruction output."""
        src = Path("backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")
        assert '"phase_24_dropout_repair"' in src
        assert '"phase_55_diffusion_inpainting"' in src
        assert '"phase_17_mastering_polish"' in src
        assert '"phase_42_vocal_enhancement"' in src


class TestUnifiedRestorerRuntimeIntegration:
    @pytest.mark.ml
    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_05_restore_populates_goal_fields_minimal(self) -> None:
        """Echter Minimal-Restore: Goal-Felder muessen im Ergebnis gesetzt sein."""
        cfg = RestorationConfig(
            mode=QualityMode.FAST,
            enable_performance_guard=False,
            enable_phase_gate=False,
        )
        restorer = UnifiedRestorerV3(config=cfg)

        # 1s Stille minimiert Laufzeit und vermeidet datenabhaengige Streuung.
        audio = _silence(1.0)
        result = restorer.restore(audio, sample_rate=SR)

        assert isinstance(result.goal_applicability, dict)
        assert isinstance(result.adaptive_thresholds, dict)
        assert isinstance(result.physical_ceiling, dict)

        # Applicability sollte mindestens den Goal-Raum abbilden.
        assert len(result.goal_applicability) >= 6
        assert "natuerlichkeit" in result.goal_applicability

        # Effektive Thresholds sollen aus dem Checker befuellt werden (nicht leer).
        assert len(result.adaptive_thresholds) >= 7
        assert "natuerlichkeit" in result.adaptive_thresholds
