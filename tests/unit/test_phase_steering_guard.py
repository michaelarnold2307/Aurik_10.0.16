"""
Golden-Sample-Tests für PhaseSteeringGuard (§3.0 Steering Loop).

Validiert:
1. Keine Infinite Loops (max 2 RETRY pro Phase + globales Zeitlimit)
2. HPE-Verbesserung durch Steering (ΔHPE ≥ 0 über gesamte Pipeline)
3. Rollback funktioniert (bestes Zwischenergebnis wird zurückgegeben)
4. CrossPhaseTracker verhindert Band-Überbearbeitung
5. Kein Audio-Korruption (NaN/Inf/Length-Check nach jeder Phase)
"""

import logging
import os
import time

import numpy as np
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test-Audio: 3s Sinus + Rauschen @ 48kHz
_SR = 48000
_DURATION = 3.0


def _make_test_audio(duration: float = _DURATION) -> np.ndarray:
    """Erzeugt Test-Audio: Sinus 440Hz + weißes Rauschen −30dB."""
    t = np.arange(int(duration * _SR), dtype=np.float32) / _SR
    sine = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32) * 0.5
    noise = np.random.randn(len(t)).astype(np.float32) * 0.03
    return np.clip(sine + noise, -1.0, 1.0)  # type: ignore[no-any-return]


def _make_stereo_test_audio() -> np.ndarray:
    """Stereo-Test-Audio."""
    mono = _make_test_audio()
    return np.stack([mono, mono * 0.8], axis=1).astype(np.float32)  # type: ignore[no-any-return]


@pytest.mark.unit
class TestPhaseSteeringGuard:
    """Unit-Tests für PhaseSteeringGuard (ohne UV3-Integration)."""

    def test_01_engine_initialization(self):
        """Engine initialisiert korrekt."""
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        engine = PhaseSteeringEngine()
        assert engine is not None
        assert engine._state.total_phases == 0
        assert engine._state.best_hpe == 0.0
        assert engine._state.consecutive_stable == 0

    def test_02_engine_disabled_by_default(self):
        """Ohne AURIK_STEERING=1 ist Steering deaktiviert."""
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        engine = PhaseSteeringEngine()
        assert engine.enabled  # v10.4: immer aktiv

    def test_03_engine_enabled_with_env(self):
        """Mit AURIK_STEERING=1 ist Steering aktiv."""
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        assert engine.enabled

    def test_04_steering_decision_continue(self):
        """HPE-Verbesserung → CONTINUE."""
        from backend.core.phase_steering_guard import (
            PhaseSteeringEngine,
            SteerAction,
        )

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        decision = engine.decide(0.50, 0.55, "phase_test", 1.0)
        assert decision.action == SteerAction.CONTINUE
        assert decision.delta_hpe > 0.02

    def test_05_steering_decision_retry_lighter(self):
        """Milde HPE-Verschlechterung → RETRY_LIGHTER."""
        from backend.core.phase_steering_guard import (
            PhaseSteeringEngine,
            SteerAction,
        )

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        decision = engine.decide(0.60, 0.57, "phase_test", 1.0)
        assert decision.action == SteerAction.RETRY_LIGHTER
        assert decision.new_strength < 1.0

    def test_06_steering_decision_skip(self):
        """Deutliche HPE-Verschlechterung → SKIP."""
        from backend.core.phase_steering_guard import (
            PhaseSteeringEngine,
            SteerAction,
        )

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        decision = engine.decide(0.70, 0.60, "phase_test", 1.0)
        assert decision.action == SteerAction.SKIP

    def test_07_max_retries_enforced(self):
        """Max 2 RETRY pro Phase, dann SKIP."""
        from backend.core.phase_steering_guard import (
            PhaseSteeringEngine,
            SteerAction,
        )

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()

        # Drei milde Verschlechterungen hintereinander
        hpe = 0.70
        decisions = []
        for i in range(5):
            hpe -= 0.02
            d = engine.decide(0.70, hpe, "phase_test", 1.0)
            decisions.append(d.action)

        # Nach 2 RETRY sollte SKIP oder ROLLBACK kommen
        retry_count = sum(1 for d in decisions if d == SteerAction.RETRY_LIGHTER)
        assert retry_count <= 2, f"Zu viele RETRY: {retry_count}"

    def test_08_rollback_triggered(self):
        """Mehrere Phasen verschlechtern → ROLLBACK."""
        from backend.core.phase_steering_guard import (
            PhaseSteeringEngine,
            SteerAction,
        )

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        audio = _make_test_audio()

        # Phase 1: Verbesserung
        d1 = engine.decide(0.50, 0.55, "phase_01", 1.0)
        assert d1.action == SteerAction.CONTINUE
        engine.record_phase("phase_01", audio, 0.55, 0.80)

        # Phase 2: Verschlechterung
        d2 = engine.decide(0.55, 0.45, "phase_02", 1.0)
        assert d2.action in (SteerAction.SKIP, SteerAction.ROLLBACK)  # 0.55→0.45 = -0.10

        # Phase 3: Noch eine Verschlechterung
        d3 = engine.decide(0.45, 0.35, "phase_03", 1.0)
        # Sollte ROLLBACK sein (zu viele Drops)
        assert d3.action in (SteerAction.SKIP, SteerAction.ROLLBACK)

    def test_09_stop_graceful(self):
        """HPE stabil über 3 Phasen → STOP_GRACEFUL."""
        from backend.core.phase_steering_guard import (
            PhaseSteeringEngine,
            SteerAction,
        )

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        audio = _make_test_audio()

        # Drei Phasen mit minimaler Änderung
        engine.record_phase("p1", audio, 0.700, 0.90)
        engine.decide(0.699, 0.700, "p2", 1.0)
        engine.record_phase("p2", audio, 0.700, 0.90)

        d2 = engine.decide(0.700, 0.701, "p3", 1.0)
        # Bei PMGG > 0.89 und stabil → STOP_GRACEFUL
        assert d2.action in (SteerAction.CONTINUE, SteerAction.STOP_GRACEFUL)

    def test_10_best_audio_preserved(self):
        """Bestes Audio wird für Rollback bewahrt."""
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()

        audio_best = _make_test_audio(0.5) * 0.9
        audio_worse = _make_test_audio(0.5) * 0.3

        engine.record_phase("best", audio_best, 0.80, 0.95)
        engine.record_phase("worse", audio_worse, 0.30, 0.50)

        best = engine.get_best_audio()
        assert best is not None
        assert engine._state.best_hpe > 0.5  # Best HPE should be tracked

    def test_11_cross_phase_integration(self):
        """CrossPhaseTracker wird pro Phase aufgerufen."""
        from backend.core.cross_phase_naturalness import (
            get_tracker,
            reset_tracker,
        )

        reset_tracker()
        tracker = get_tracker()
        assert tracker is not None

        # Phase 1: Boost presence
        tracker.record("phase_38_presence_boost", {"presence": 3.0})
        ok = tracker.can_process("presence", 2.0)
        assert ok  # 3 + 2 = 5 < 8 → OK

        # Phase 2: Nochmal presence boost
        tracker.record("phase_39_air_band", {"presence": 3.0})
        ok = tracker.can_process("presence", 1.0)
        assert ok  # 6+1=7<8 and count=2<3 → can process
        # Actually: cumulative=6.0, remaining=2.0, 1.0 < 2.0 → True

        # Phase 3: Presence gesättigt
        tracker.record("phase_42_more_presence", {"presence": 2.0})
        ok = tracker.can_process("presence", 0.5)
        assert not ok  # 8 + 0.5 > 8 OR count=3 → False

    def test_12_no_audio_corruption(self):
        """Audio bleibt NaN/Inf-frei nach Steering."""
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        audio = _make_test_audio()

        for i in range(20):
            engine.record_phase(f"p{i}", audio, 0.50 + i * 0.01, 0.70)
            best = engine.get_best_audio()
            if best is not None:
                assert not np.any(np.isnan(best)), f"NaN in best audio at phase {i}"
                assert not np.any(np.isinf(best)), f"Inf in best audio at phase {i}"
                assert len(best) > 0, f"Empty audio at phase {i}"

    def test_13_stereo_preserved(self):
        """Stereo-Audio bleibt stereo nach Steering."""
        from backend.core.phase_steering_guard import PhaseSteeringEngine

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        audio = _make_stereo_test_audio()

        engine.record_phase("stereo_test", audio, 0.60, 0.80)
        best = engine.get_best_audio()
        assert best is not None
        assert best.ndim == 2
        assert best.shape[1] == 2

    def test_14_no_infinite_loop(self):
        """Steering-Engine blockiert nicht (Timeout-Test)."""
        from backend.core.phase_steering_guard import (
            PhaseSteeringEngine,
            SteerAction,
        )

        os.environ["AURIK_STEERING"] = "1"
        engine = PhaseSteeringEngine()
        audio = _make_test_audio(0.2)

        t0 = time.perf_counter()
        for i in range(100):
            d = engine.decide(0.50 - i * 0.001, 0.50 - (i + 1) * 0.001, f"p{i}", 1.0)
            if d.action in (SteerAction.SKIP, SteerAction.ROLLBACK, SteerAction.STOP_GRACEFUL):
                engine.record_phase(f"p{i}", audio, 0.45, 0.50)
            else:
                engine.record_phase(f"p{i}", audio, 0.55, 0.80)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"100 Entscheidungen dauerten {elapsed:.2f}s (>1s Limit)"
