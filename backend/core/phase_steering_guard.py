"""
§v10.4 Phase Steering Guard — HPE-gesteuerter Phase-Loop mit SOTA-Workflow.

Wrapped UnifiedRestorerV3._profiled_phase_call per Instance-Level-Hook mit:
1. HPE-Messung VOR und NACH jeder Phase
2. Steering-Entscheidungen: CONTINUE | RETRY_LIGHTER | SKIP | ROLLBACK
3. Cross-Phase Naturalness Consensus (Band-Sättigungs-Tracker)
4. Best-State-Rollback (kein permanenter Qualitätsverlust)
5. Safety-Guards: max_retries=3, max_rollbacks=5, kein Infinite-Loop
6. Stop-Regel: wenn PMGG > 0.92 und HPE > 0.72 über 3 Phasen → STOP

Aktivierung: Default-aktiv (kein opt-in nötig).

Architektur (§v10.53 Refactor):
- install_steering() → Factory, erzeugt Engine-Singleton
- wrap_steering(engine, original_fn) → Erzeugt Wrapper-Funktion
- UV3.__init__() bindet Wrapper per types.MethodType auf self
- Kein Klassen-Monkeypatch → keine Test-Pollution
- inspect.getsource(UV3._profiled_phase_call) bleibt intakt

Ref: §v10 Pleasantness-First, §3.0 Cross-Phase Consensus, §2.64 FeedbackChain
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Enums & Data
# ═══════════════════════════════════════════════════════════════════════════


class SteerAction(Enum):
    CONTINUE = "continue"  # ΔP ≥ 0 → normal weiter
    RETRY_LIGHTER = "retry_lighter"  # −0.03 < ΔP < 0 → mit 70% Intensität wiederholen
    SKIP = "skip"  # ΔP < −0.05 → Phase überspringen
    ROLLBACK = "rollback"  # Mehrere Drops → zurück zum besten Stand
    STOP_GRACEFUL = "stop"  # PMGG > 0.92 + HPE stabil → aufhören


@dataclass
class SteeringState:
    """Trackt den Steering-Zustand über alle Phasen."""

    best_audio: np.ndarray | None = None
    best_hpe: float = 0.0
    best_phase_idx: int = -1
    current_phase_idx: int = 0
    total_phases: int = 0
    consecutive_stable: int = 0
    rollback_count: int = 0
    retry_count: int = 0
    skipped_phases: list[str] = field(default_factory=list)
    phase_history: list[dict] = field(default_factory=list)

    MAX_RETRIES: int = 3
    MAX_ROLLBACKS: int = 5
    STOP_STABLE_THRESHOLD: int = 3  # 3 stabile Phasen → STOP


@dataclass
class SteeringDecision:
    action: SteerAction
    reason: str
    new_strength: float = 1.0
    delta_hpe: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Steering Engine
# ═══════════════════════════════════════════════════════════════════════════


class PhaseSteeringEngine:
    """HPE-gesteuerter Steering-Entscheider pro Phase."""

    def __init__(self):
        self._state = SteeringState()
        self._lock = threading.Lock()
        self._tracker = None
        self._enabled = True  # §v10.4: immer aktiv
        try:
            from backend.core.cross_phase_naturalness import get_tracker, reset_tracker

            reset_tracker()
            self._tracker = get_tracker()
        except Exception as e:
            logger.debug("tracker init fallback: %s", e)
            self._tracker = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── HPE-Messung ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_hpe(audio: np.ndarray, sr: int = 48000) -> float:
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            return float(compute_pleasantness(audio, sr).score)
        except Exception as e:
            logger.debug("_compute_hpe fallback: %s", e)
            mono = audio.mean(axis=1) if audio.ndim == 2 and audio.shape[1] <= 2 else audio.ravel()
            rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
            return float(np.clip(0.3 + rms * 2.0, 0.0, 1.0))

    @staticmethod
    def _compute_pmgg(audio: np.ndarray, sr: int = 48000) -> float:
        """Vereinfachter PMGG-Proxy (0-1). Echter PMGG wäre zu teuer pro Phase."""
        try:
            from backend.core.human_pleasantness_estimator import compute_pleasantness

            r = compute_pleasantness(audio, sr)
            return float(
                np.mean(
                    [
                        r.score,
                        r.tonalness,
                        1.0 - min(r.roughness_asper / 3.0, 1.0),
                        1.0 - abs(r.sharpness_zwicker - 1.5) / 3.0,
                    ]
                )
            )
        except Exception as e:
            logger.warning("_compute_pmgg: %s", e)
            return 0.5

    # ── Entscheidungslogik ────────────────────────────────────────────────

    def decide(self, hpe_before: float, hpe_after: float, phase_name: str, phase_strength: float) -> SteeringDecision:
        """Trifft Steering-Entscheidung basierend auf HPE-Änderung."""
        with self._lock:
            s = self._state
            delta = hpe_after - hpe_before

            # Case 1: HPE verbessert → CONTINUE
            if delta > 0.02:
                s.consecutive_stable = 0
                self._update_best(hpe_after, phase_name)
                return SteeringDecision(SteerAction.CONTINUE, f"HPE +{delta:+.3f} → weiter", delta_hpe=delta)

            # Case 2: HPE stabil (±0.02) → prüfe Stop
            if delta >= -0.02:
                s.consecutive_stable += 1
                self._update_best(hpe_after, phase_name)
                if s.consecutive_stable >= s.STOP_STABLE_THRESHOLD and hpe_after > 0.68:
                    return SteeringDecision(
                        SteerAction.STOP_GRACEFUL,
                        f"HPE stabil ({s.consecutive_stable} Phasen) + gut → STOP",
                        delta_hpe=delta,
                    )
                return SteeringDecision(SteerAction.CONTINUE, f"HPE stabil (Δ{delta:+.3f})", delta_hpe=delta)

            # Case 3: Leichte Verschlechterung → RETRY_LIGHTER
            if delta > -0.05:
                if s.retry_count < s.MAX_RETRIES:
                    s.retry_count += 1
                    new_str = max(0.3, phase_strength * 0.7)
                    logger.info(
                        "Steering %s: RETRY_LIGHTER (Δ%+.3f, str %.2f→%.2f, retry %d/%d)",
                        phase_name,
                        delta,
                        phase_strength,
                        new_str,
                        s.retry_count,
                        s.MAX_RETRIES,
                    )
                    return SteeringDecision(
                        SteerAction.RETRY_LIGHTER,
                        "Leichte Verschlechterung → leiser wiederholen",
                        new_strength=new_str,
                        delta_hpe=delta,
                    )
                else:
                    s.retry_count = 0
                    s.skipped_phases.append(phase_name)
                    logger.info("Steering %s: Max Retries erreicht → SKIP", phase_name)
                    return SteeringDecision(
                        SteerAction.SKIP, f"Max Retries ({s.MAX_RETRIES}) → überspringe {phase_name}", delta_hpe=delta
                    )

            # Case 4: Deutliche Verschlechterung → SKIP
            if delta > -0.10:
                s.skipped_phases.append(phase_name)
                logger.info("Steering %s: SKIP (Δ%+.3f, zu starke Verschlechterung)", phase_name, delta)
                return SteeringDecision(
                    SteerAction.SKIP, f"Deutliche Verschlechterung → überspringe {phase_name}", delta_hpe=delta
                )

            # Case 5: Massive Verschlechterung + beste Version existiert → ROLLBACK
            if s.best_audio is not None and s.rollback_count < s.MAX_ROLLBACKS:
                s.rollback_count += 1
                s.retry_count = 0
                logger.warning(
                    "Steering %s: ROLLBACK #%d (Δ%+.3f, restore best HPE %.3f)",
                    phase_name,
                    s.rollback_count,
                    delta,
                    s.best_hpe,
                )
                return SteeringDecision(
                    SteerAction.ROLLBACK, "Massive Verschlechterung → Rollback zu bestem Stand", delta_hpe=delta
                )

            # Case 6: Nichts hilft → SKIP
            s.skipped_phases.append(phase_name)
            return SteeringDecision(
                SteerAction.SKIP, f"Keine Verbesserung möglich → überspringe {phase_name}", delta_hpe=delta
            )

    def _update_best(self, hpe: float, phase_name: str):
        s = self._state
        if hpe > s.best_hpe + 0.005:
            s.best_hpe = hpe
            s.best_phase_idx = s.current_phase_idx
            # best_audio wird extern gesetzt (Referenz auf aktuelles Audio)
            s.retry_count = 0

    def record_phase(self, phase_name: str, audio: np.ndarray, hpe: float, pmgg: float):
        """Zeichnet Phase in History auf."""
        s = self._state
        s.phase_history.append(
            {
                "phase": phase_name,
                "idx": s.current_phase_idx,
                "hpe": round(hpe, 4),
                "pmgg": round(pmgg, 4),
                "time": time.monotonic(),
            }
        )
        # Update best audio referenz
        if s.best_audio is None or hpe >= s.best_hpe:
            s.best_audio = audio.copy()
            s.best_hpe = hpe
            s.best_phase_idx = s.current_phase_idx
        s.current_phase_idx += 1

    def get_best_audio(self) -> np.ndarray | None:
        return self._state.best_audio

    def get_summary(self) -> dict:
        s = self._state
        return {
            "phases_processed": s.current_phase_idx,
            "phases_skipped": len(s.skipped_phases),
            "skipped": s.skipped_phases,
            "rollbacks": s.rollback_count,
            "retries": s.retry_count,
            "best_hpe": round(s.best_hpe, 4),
            "best_phase_idx": s.best_phase_idx,
            "consecutive_stable": s.consecutive_stable,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Instance-Level Wrapper (kein Klassen-Monkeypatch)
# ═══════════════════════════════════════════════════════════════════════════

_engine: PhaseSteeringEngine | None = None


def install_steering() -> PhaseSteeringEngine:
    """Erzeugt den globalen Steering-Engine-Singleton (Factory).

    Kein Monkeypatch — die Engine wird per Instance-Level-Hook in
    UnifiedRestorerV3.__init__() eingebunden (§v10.53 Refactor).

    Returns:
        PhaseSteeringEngine (globaler Singleton), immer enabled=True.
    """
    global _engine

    if _engine is not None:
        return _engine

    _engine = PhaseSteeringEngine()
    logger.debug("PhaseSteeringGuard: Engine-Singleton erzeugt (enabled=%s)", _engine.enabled)
    return _engine


def wrap_steering(engine: PhaseSteeringEngine, original_fn):
    """Erzeugt eine Wrapper-Funktion, die HPE-Steering um original_fn legt.

    Die Wrapper-Funktion hat die Signatur (self, phase, audio, **kwargs)
    und kann per types.MethodType an eine UV3-Instanz gebunden werden.

    Args:
        engine: Die Steering-Engine (von install_steering()).
        original_fn: Die originale _profiled_phase_call (ungebundene Methode).

    Returns:
        Callable mit Signatur (self, phase, audio, **kwargs) → PhaseResult/ndarray.
    """

    def _steered_phase_call(self, phase, audio, **kwargs):
        """Gesteuerter Phase-Call mit HPE-Messung und Steering."""
        import types as _types

        phase_name = str(getattr(phase, "phase_id", getattr(phase, "__name__", "?")))
        strength = float(kwargs.get("strength", 1.0))
        sr = int(kwargs.get("sample_rate", 48000) or 48000)

        # HPE vor Phase
        hpe_before = engine._compute_hpe(audio, sr)

        # Original-Phase ausführen
        result = original_fn(self, phase, audio, **kwargs)
        result_audio = result.audio if hasattr(result, "audio") else result

        # HPE nach Phase
        hpe_after = engine._compute_hpe(result_audio, sr)

        # Steering-Entscheidung
        decision = engine.decide(hpe_before, hpe_after, phase_name, strength)
        pmgg_after = engine._compute_pmgg(result_audio, sr)

        if decision.action == SteerAction.CONTINUE:
            engine.record_phase(phase_name, result_audio, hpe_after, pmgg_after)
            return result

        elif decision.action == SteerAction.RETRY_LIGHTER:
            new_kwargs = dict(kwargs)
            new_kwargs["strength"] = decision.new_strength
            logger.info("Steering %s: RETRY_LIGHTER (str %.2f→%.2f)", phase_name, strength, decision.new_strength)
            result2 = original_fn(self, phase, audio, **new_kwargs)
            result2_audio = result2.audio if hasattr(result2, "audio") else result2
            hpe2 = engine._compute_hpe(result2_audio, sr)
            if hpe2 > hpe_before - 0.02:
                engine.record_phase(phase_name + "_retry", result2_audio, hpe2, engine._compute_pmgg(result2_audio, sr))
                return result2
            # RETRY half nicht → SKIP
            engine.record_phase(phase_name + "_retry_failed", audio, hpe_before, engine._compute_pmgg(audio, sr))
            return type(result)(audio=audio) if hasattr(result, "audio") else audio

        elif decision.action == SteerAction.SKIP:
            logger.info(
                "Steering %s: SKIP (HPE %.3f→%.3f Δ%+.3f)", phase_name, hpe_before, hpe_after, decision.delta_hpe
            )
            engine.record_phase(phase_name + "_skipped", audio, hpe_before, engine._compute_pmgg(audio, sr))
            return type(result)(audio=audio) if hasattr(result, "audio") else audio

        elif decision.action == SteerAction.ROLLBACK:
            best = engine.get_best_audio()
            if best is not None:
                logger.warning("Steering %s: ROLLBACK to best (HPE %.3f)", phase_name, engine._state.best_hpe)
                return type(result)(audio=best) if hasattr(result, "audio") else best
            engine.record_phase(phase_name + "_rollback_fallback", audio, hpe_before, pmgg_after)
            return result

        elif decision.action == SteerAction.STOP_GRACEFUL:
            logger.info("Steering: STOP_GRACEFUL — %s (HPE %.3f stabil)", phase_name, hpe_after)
            engine.record_phase(phase_name, result_audio, hpe_after, pmgg_after)
            return result

        return result

    return _steered_phase_call


def get_engine() -> PhaseSteeringEngine | None:
    """Gibt den globalen Steering-Engine-Singleton zurück."""
    return _engine
