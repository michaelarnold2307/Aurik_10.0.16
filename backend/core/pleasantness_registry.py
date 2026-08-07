"""
§v10 PleasantnessRegistry — Das zentrale Team-Bewusstsein von Aurik.

Bisher arbeitete jedes Modul für sich. Der StrategieDenker wusste nicht,
was der ExzellenzDenker gemessen hat. Der RestaurierDenker wusste nicht,
ob seine Arbeit das Ergebnis angenehmer oder anstrengender gemacht hat.

Die PleasantnessRegistry ändert das fundamental:

  JEDES Modul liest VOR der Arbeit: „Wie angenehm klingt es gerade?"
  JEDES Modul schreibt NACH der Arbeit: „Ich habe ΔP = +0.03 erreicht."
  JEDES Modul kennt das TEAM-ZIEL: P ≥ 0.75, nie unter 0.35 fallen.

Wie ein Orchester: Der Dirigent (Registry) gibt den Takt vor,
jeder Musiker (Denker) hört auf die anderen, und gemeinsam entsteht
ein Werk, das größer ist als die Summe seiner Teile.

Architektur:
  - Thread-safe Singleton mit Locking (§3.2)
  - Epoch-basierte Versionierung (jede Änderung → neue epoch)
  - Benachrichtigungs-Callbacks für Echtzeit-Monitoring
  - Audit-Trail: JEDE Pleasantness-Änderung wird protokolliert
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PleasantnessSnapshot:
    """Ein Snapshot der Angenehmheit zu einem Zeitpunkt."""

    epoch: int
    timestamp: float
    module_name: str
    phase: str  # "pre", "post", "intermediate"
    pleasantness: float
    delta: float = 0.0
    label: str = ""
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamStatus:
    """Der aktuelle Status des gesamten Aurik-Teams."""

    baseline_pleasantness: float = 0.5
    target_pleasantness: float = 0.75
    current_pleasantness: float = 0.5
    best_pleasantness: float = 0.5
    best_epoch: int = 0

    total_steps: int = 0
    steps_improved: int = 0
    steps_declined: int = 0
    steps_neutral: int = 0

    consecutive_declines: int = 0
    steering_active: bool = False
    current_steering_action: str = "continue"
    steering_reason: str = ""

    inviting_check_passed: bool = True
    inviting_issues: list[str] = field(default_factory=list)

    active_modules: list[str] = field(default_factory=list)
    completed_modules: list[str] = field(default_factory=list)

    global_verdict: str = "Initialisierung..."
    epoch: int = 0


class PleasantnessRegistry:
    """Das zentrale Team-Bewusstsein — Thread-safe Singleton.

    Nutzung (überall im Code):
        registry = get_pleasantness_registry()

        # Baseline setzen (StrategieDenker)
        registry.set_baseline(0.72, label="Angenehm")

        # Vor einem Schritt
        registry.report_pre("RestaurierDenker", 0.72)

        # Nach einem Schritt
        snapshot = registry.report_post("RestaurierDenker", 0.75,
                                        delta=0.03, label="Angenehm")

        # Status abfragen
        status = registry.get_status()

        # Auf Verschlechterungen reagieren
        if registry.should_steer():
            action = registry.get_steering_action()
    """

    _instance: PleasantnessRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> PleasantnessRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False  # type: ignore[has-type]
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[has-type]
            return
        self._initialized = True
        self._rw_lock = threading.RLock()

        self._baseline: float = 0.5
        self._target: float = 0.75
        self._current: float = 0.5
        self._best: float = 0.5
        self._best_epoch: int = 0
        self._epoch: int = 0

        self._history: list[PleasantnessSnapshot] = []
        self._active_modules: list[str] = []
        self._completed_modules: list[str] = []
        self._steering_actions: list[dict[str, Any]] = []

        self._total_steps: int = 0
        self._improved: int = 0
        self._declined: int = 0
        self._neutral: int = 0
        self._consecutive_declines: int = 0

        self._steering_active: bool = False
        self._current_steering_action: str = "continue"
        self._steering_reason: str = ""

        self._inviting_passed: bool = True
        self._inviting_issues: list[str] = []

        self._baseline_goosebumps: float = 0.5
        self._baseline_label: str = ""
        self._callbacks: list[Any] = []

        logger.info("PleasantnessRegistry initialisiert — Aurik hat jetzt ein Team-Bewusstsein.")

    # ── Öffentliche API ──────────────────────────────────────────────────

    def set_baseline(self, pleasantness: float, *, label: str = "", goosebumps: float = 0.5) -> None:
        """Setzt die Angenehmheits-Baseline (EINMAL vor der Pipeline)."""
        with self._rw_lock:
            self._baseline = pleasantness
            self._baseline_label = label
            self._baseline_goosebumps = goosebumps
            self._target = max(0.75, min(0.95, pleasantness + 0.10))
            self._current = pleasantness
            self._best = pleasantness
            self._epoch = 0
            self._record(0.0, "Baseline", "baseline", pleasantness, label, {"goosebumps": goosebumps})  # type: ignore[arg-type]
            logger.info(
                "Registry: Baseline P=%.3f (%s) | Target P=%.3f | Goosebumps=%.3f",
                pleasantness,
                label,
                self._target,
                goosebumps,
            )

    def report_pre(self, module_name: str, current_pleasantness: float) -> int:
        """Meldet: Modul BEGINNT mit der Arbeit. Gibt epoch zurück."""
        with self._rw_lock:
            self._current = current_pleasantness
            if module_name not in self._active_modules:
                self._active_modules.append(module_name)
            return self._epoch

    def report_post(
        self,
        module_name: str,
        pleasantness: float,
        *,
        delta: float = 0.0,
        label: str = "",
        issues: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PleasantnessSnapshot:
        """Meldet: Modul ist FERTIG. Gibt den Snapshot zurück."""
        with self._rw_lock:
            self._epoch += 1
            self._total_steps += 1
            self._current = pleasantness

            # Verbesserung/Verschlechterung tracken
            if delta > 0.015:
                self._improved += 1
                self._consecutive_declines = 0
            elif delta < -0.015:
                self._declined += 1
                self._consecutive_declines += 1
                logger.warning(
                    "Registry: %s hat P um %.3f VERSCHLECHTERT (von %.3f auf %.3f)",
                    module_name,
                    delta,
                    pleasantness - delta,
                    pleasantness,
                )
            else:
                self._neutral += 1

            # Bestwert aktualisieren
            if pleasantness > self._best:
                self._best = pleasantness
                self._best_epoch = self._epoch

            # Steering prüfen
            self._update_steering(module_name, delta)

            # Snapshot aufzeichnen und speichern
            snap = self._record(
                delta,
                module_name,
                "post",
                pleasantness,
                label,
                issues=issues or [],
                metadata=metadata or {},
            )

            # Modul als completed markieren
            if module_name not in self._completed_modules:
                self._completed_modules.append(module_name)

            # Callbacks auslösen
            for cb in self._callbacks:
                try:
                    cb(snap)
                except Exception as e:
                    logger.warning("pleasantness_registry.py::report_post Ersatzpfad: %s", e)

            return snap

    def report_intermediate(self, module_name: str, pleasantness: float, delta: float = 0.0) -> None:
        """Meldet einen Zwischenstand (kein kompletter Schritt)."""
        with self._rw_lock:
            self._current = pleasantness
            self._record(delta, module_name, "intermediate", pleasantness, "")

    def get_status(self) -> TeamStatus:
        """Gibt den aktuellen Team-Status zurück."""
        with self._rw_lock:
            return TeamStatus(
                baseline_pleasantness=self._baseline,
                target_pleasantness=self._target,
                current_pleasantness=self._current,
                best_pleasantness=self._best,
                best_epoch=self._best_epoch,
                total_steps=self._total_steps,
                steps_improved=self._improved,
                steps_declined=self._declined,
                steps_neutral=self._neutral,
                consecutive_declines=self._consecutive_declines,
                steering_active=self._steering_active,
                current_steering_action=self._current_steering_action,
                steering_reason=self._steering_reason,
                inviting_check_passed=self._inviting_passed,
                inviting_issues=list(self._inviting_issues),
                active_modules=list(self._active_modules),
                completed_modules=list(self._completed_modules),
                global_verdict=self._compute_verdict(),
                epoch=self._epoch,
            )

    def should_steer(self) -> bool:
        """Muss das Team nachsteuern?"""
        with self._rw_lock:
            return self._steering_active

    def get_steering_action(self) -> tuple[str, str]:
        """Gibt Steering-Aktion und Begründung zurück."""
        with self._rw_lock:
            return self._current_steering_action, self._steering_reason

    def set_inviting_check(self, passed: bool, issues: list[str]) -> None:
        """Setzt den Einladender-Klang-Check-Status."""
        with self._rw_lock:
            self._inviting_passed = passed
            self._inviting_issues = issues
            if not passed:
                logger.warning(
                    "Registry: Einladender-Klang-Pruefung NICHT BESTANDEN: %s",
                    "; ".join(issues),
                )

    def get_history(self) -> list[PleasantnessSnapshot]:
        """Gibt die gesamte Pleasantness-Historie zurück."""
        with self._rw_lock:
            return list(self._history)

    def subscribe(self, callback: Any) -> None:
        """Registriert einen Callback für Pleasantness-Änderungen."""
        with self._rw_lock:
            self._callbacks.append(callback)

    def reset(self) -> None:
        """Kompletter Reset für neue Pipeline."""
        with self._rw_lock:
            self._baseline = 0.5
            self._target = 0.75
            self._current = 0.5
            self._best = 0.5
            self._best_epoch = 0
            self._epoch = 0
            self._history.clear()
            self._active_modules.clear()
            self._completed_modules.clear()
            self._steering_actions.clear()
            self._total_steps = 0
            self._improved = 0
            self._declined = 0
            self._neutral = 0
            self._consecutive_declines = 0
            self._steering_active = False
            self._inviting_passed = True
            self._inviting_issues.clear()
            logger.info("Registry: zurueckgesetzt — bereit für neue Pipeline.")

    def current_pleasantness(self) -> float:
        """Gibt die aktuelle Pleasantness zurück (lock-frei für schnelle Reads)."""
        return self._current

    # ── Interne Methoden ─────────────────────────────────────────────────

    def _record(
        self,
        delta: float,
        module: str,
        phase: str,
        pleasantness: float,
        label: str,
        issues: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PleasantnessSnapshot:
        snap = PleasantnessSnapshot(
            epoch=self._epoch,
            timestamp=time.monotonic(),
            module_name=module,
            phase=phase,
            pleasantness=pleasantness,
            delta=delta,
            label=label,
            issues=issues or [],
            metadata=metadata or {},
        )
        self._history.append(snap)
        return snap

    def _update_steering(self, module_name: str, delta: float) -> None:
        """Aktualisiert den Steering-Status basierend auf HPE-Delta."""
        from backend.core.quality_feedback_loop import SteerAction, steer_pipeline

        action, reason = steer_pipeline(
            pmgg_delta=0.0,  # PMGG wird separat getrackt
            pleasantness_delta=delta,
            phase_id=module_name,
            step_index=self._total_steps,
            total_steps=99,  # Unbekannt — Registry kennt nicht die Gesamtzahl
            max_pleasantness_drops=3,
        )

        if action != SteerAction.CONTINUE:
            self._steering_active = True
            self._current_steering_action = action
            self._steering_reason = reason
            self._steering_actions.append(
                {
                    "epoch": self._epoch,
                    "module": module_name,
                    "delta": delta,
                    "action": action,
                    "reason": reason,
                }
            )
        else:
            self._steering_active = False

    def _compute_verdict(self) -> str:
        """Berechnet das globale Team-Urteil."""
        if self._best >= self._target:
            return f"Weltklasse: P={self._best:.3f} ≥ Ziel {self._target:.3f} — Ziel ÜBERTROFFEN!"
        elif self._best >= self._baseline + 0.05:
            return f"Deutlich verbessert: ΔP=+{self._best - self._baseline:.3f} — gutes Ergebnis."
        elif self._best >= self._baseline:
            return f"Leicht verbessert: ΔP=+{self._best - self._baseline:.3f} — akzeptabel."
        elif self._consecutive_declines >= 2:
            return f"WARNUNG: {self._consecutive_declines}x in Folge verschlechtert — Rollback empfohlen."
        else:
            return "In Bearbeitung..."


# ── Singleton-Zugriff ────────────────────────────────────────────────────


def get_pleasantness_registry() -> PleasantnessRegistry:
    """Gibt die globale PleasantnessRegistry-Instanz zurück."""
    return PleasantnessRegistry()


# ── Convenience: Team-Benachrichtigungen ─────────────────────────────────


def notify_team_improvement(module: str, delta: float) -> None:
    """Kurze Benachrichtigung: Modul hat Verbesserung erzielt."""
    reg = get_pleasantness_registry()
    status = reg.get_status()
    logger.info(
        "🎵 %s → Team: ΔP=+%.3f | Gesamtstatus: P=%.3f (Ziel=%.3f, Best=%.3f) | %d/%d Module fertig",
        module,
        delta,
        status.current_pleasantness,
        status.target_pleasantness,
        status.best_pleasantness,
        len(status.completed_modules),
        len(status.active_modules),
    )


def notify_team_decline(module: str, delta: float) -> None:
    """Kurze Warnung: Modul hat Verschlechterung verursacht."""
    reg = get_pleasantness_registry()
    logger.warning(
        "⚠️  %s → Team: ΔP=%.3f VERSCHLECHTERT | Rollback? %s",
        module,
        delta,
        "JA" if reg._consecutive_declines >= 2 else "noch nicht",
    )
