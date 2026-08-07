"""§Pipeline-Resilienz: Checkpoint, Phase-Timeout, Early-Exit.

Drei fundamentale Strategien gegen Pipeline-Hänger und Watchdog-Kills:

1. Checkpoint: Nach jeder Phase Audio+State auf Disk speichern.
   Bei Neustart wird der Checkpoint erkannt und die Pipeline fortgesetzt.

2. Phase-Timeout: Jede Einzelphase bekommt ein Zeitlimit.
   Überschreitung → Phase wird als passthrough markiert, Pipeline läuft weiter.

3. Early-Exit: Nach 60 % der Phasen PMGG-Scores prüfen.
   Sind alle P1-Ziele über dem Material-Schwellwert → Pipeline beenden.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path(tempfile.gettempdir()) / "aurik_checkpoints"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pipeline-Checkpoint
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineCheckpoint:
    """Gespeicherter Zustand nach einer Phase."""

    audio: np.ndarray
    phase_idx: int
    total_phases: int
    executed_phases: list[str]
    pmgg_scores: dict[str, float]
    sample_rate: int
    material: str
    timestamp: float = field(default_factory=time.time)
    checkpoint_file: str = ""

    @staticmethod
    def save(
        audio: np.ndarray,
        phase_idx: int,
        total_phases: int,
        executed: list[str],
        scores: dict[str, float],
        sr: int,
        material: str,
        input_path: str = "",
    ) -> str | None:
        """Speichert Checkpoint. Gibt Dateipfad zurück oder None bei Fehler."""
        try:
            _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = Path(input_path).stem if input_path else "unknown"
            cp_file = _CHECKPOINT_DIR / f"ckpt_{safe_name}_{int(time.time())}.pkl"

            cp = PipelineCheckpoint(
                audio=np.asarray(audio, dtype=np.float32),
                phase_idx=phase_idx,
                total_phases=total_phases,
                executed_phases=list(executed),
                pmgg_scores=dict(scores),
                sample_rate=sr,
                material=str(material),
                checkpoint_file=str(cp_file),
            )
            # Audio separat als .npy speichern (pickle ist ineffizient für große Arrays)
            audio_file = cp_file.with_suffix(".npy")
            np.save(audio_file, cp.audio)
            cp.audio = np.array([0.0], dtype=np.float32)  # Platzhalter für pickle
            cp.checkpoint_file = str(audio_file)

            with open(cp_file, "wb") as f:
                pickle.dump(cp, f)
            logger.info("§CKPT Gespeichert: Verarbeitungsschritt %d/%d → %s", phase_idx + 1, total_phases, cp_file.name)
            return str(cp_file)
        except Exception as e:
            logger.debug("§CKPT Fehler beim Speichern: %s", e)
            return None

    @staticmethod
    def load(filepath: str) -> PipelineCheckpoint | None:
        """Lädt Checkpoint von Disk."""
        try:
            path = Path(filepath)
            if not path.exists():
                return None
            with open(path, "rb") as f:
                cp = pickle.load(f)  # nosec B301 — lokaler Aurik-Checkpoint, nicht nutzerfremde Quelle
            # Audio aus .npy laden
            audio_file = Path(cp.checkpoint_file)
            if audio_file.exists():
                cp.audio = np.load(audio_file)
            else:
                return None
            logger.info(
                "§CKPT Geladen: Verarbeitungsschritt %d/%d, %d Phasen bereits ausgeführt",
                cp.phase_idx + 1,
                cp.total_phases,
                len(cp.executed_phases),
            )
            return cp  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("§CKPT Fehler beim Laden: %s", e)
            return None

    @staticmethod
    def find_latest(input_stem: str = "") -> str | None:
        """Findet den neuesten Checkpoint für eine Eingabedatei."""
        try:
            _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            pattern = f"ckpt_{input_stem}_*.pkl" if input_stem else "ckpt_*.pkl"
            files = sorted(_CHECKPOINT_DIR.glob(pattern), key=os.path.getmtime, reverse=True)
            for f in files:
                if f.suffix == ".pkl":
                    return str(f)
            return None
        except Exception as e:
            logger.warning("pipeline_resilience.py::find_latest Ersatzpfad: %s", e)
            return None

    @staticmethod
    def cleanup(input_stem: str = "") -> int:
        """Löscht alte Checkpoints. Gibt Anzahl gelöschter Dateien zurück."""
        try:
            pattern = f"ckpt_{input_stem}_*" if input_stem else "ckpt_*"
            count = 0
            for f in _CHECKPOINT_DIR.glob(pattern):
                f.unlink(missing_ok=True)
                count += 1
            return count
        except Exception as e:
            logger.warning("pipeline_resilience.py::cleanup Ersatzpfad: %s", e)
            return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Adaptiver Phase-Timeout
# ═══════════════════════════════════════════════════════════════════════════════

# Material-abhängige Basis-Zeitlimits (Sekunden pro Phase)
_PHASE_TIMEOUT_BASE: dict[str, float] = {
    "wax_cylinder": 600.0,
    "shellac": 480.0,
    "lacquer_disc": 480.0,
    "vinyl": 360.0,
    "tape": 300.0,
    "reel_tape": 300.0,
    "cassette": 240.0,
    "cd_digital": 180.0,
    "streaming": 120.0,
    "unknown": 300.0,
}

# ML-schwere Phasen bekommen mehr Zeit
_HEAVY_ML_PHASES: frozenset[str] = frozenset(
    {
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "phase_23_spectral_repair",
        "phase_24_dropout_repair",
        "phase_29_tape_hiss_reduction",
        "phase_49_advanced_dereverb",
        "phase_56_spectral_band_gap_repair",
    }
)


def get_phase_timeout(phase_id: str, material: str = "unknown", audio_duration_s: float = 0.0) -> float:
    """Berechnet das Zeitlimit für eine einzelne Phase.

    Basis = material-abhängig, ML-Phasen ×2, Audio-Dauer-Skalierung.
    """
    base = _PHASE_TIMEOUT_BASE.get(material, 300.0)
    if phase_id in _HEAVY_ML_PHASES:
        base *= 2.0
    # Skaliere mit Audio-Dauer: lange Dateien brauchen mehr Zeit
    if audio_duration_s > 60:
        scale = min(3.0, 1.0 + audio_duration_s / 180.0)
        base *= scale
    return max(30.0, min(1800.0, base))  # 30s–30min


class PhaseTimeoutGuard:
    """Überwacht eine einzelne Phase und bricht bei Überschreitung ab."""

    def __init__(self, timeout_s: float) -> None:
        self._timeout = timeout_s
        self._started = False
        self._start_time = 0.0
        self._timed_out = False
        self._event = threading.Event()

    def start(self) -> None:
        self._started = True
        self._start_time = time.perf_counter()
        self._timed_out = False
        self._event.clear()

    def check(self) -> bool:
        """True wenn OK, False wenn Timeout."""
        if not self._started:
            return True
        if time.perf_counter() - self._start_time > self._timeout:
            self._timed_out = True
            self._event.set()
            return False
        return True

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    @property
    def elapsed(self) -> float:
        if not self._started:
            return 0.0
        return time.perf_counter() - self._start_time


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Early-Exit Quality Gate
# ═══════════════════════════════════════════════════════════════════════════════

# Priorität-1-Ziele: MÜSSEN über Schwellwert sein
_P1_GOALS: frozenset[str] = frozenset(
    {
        "waerme",
        "brillanz",
        "emotionalitaet",
        "natuerlichkeit",
    }
)

_MATERIAL_QUALITY_FLOOR: dict[str, float] = {
    "wax_cylinder": 0.55,
    "shellac": 0.60,
    "lacquer_disc": 0.60,
    "vinyl": 0.65,
    "tape": 0.70,
    "reel_tape": 0.72,
    "cassette": 0.65,
    "cd_digital": 0.85,
    "streaming": 0.85,
    "unknown": 0.65,
}

_EARLY_EXIT_CHECKPOINT_RATIO = 0.60  # Nach 60 % der Phasen prüfen


class EarlyExitGate:
    """Prüft nach 60 % der Phasen, ob genug Qualität erreicht ist."""

    def __init__(self, material: str = "unknown") -> None:
        self._material = material
        self._floor = _MATERIAL_QUALITY_FLOOR.get(material, 0.65)
        self._checked = False

    def should_check(self, phase_idx: int, total_phases: int) -> bool:
        """True wenn jetzt ein Early-Exit-Check fällig ist."""
        if self._checked:
            return False
        return phase_idx >= int(total_phases * _EARLY_EXIT_CHECKPOINT_RATIO)

    def check(self, pmgg_scores: dict[str, float]) -> tuple[bool, str]:
        """(can_exit_early, reason)."""
        self._checked = True
        failed = []
        for goal in _P1_GOALS:
            score = pmgg_scores.get(goal, 0.0)
            if score < self._floor:
                failed.append(f"{goal}={score:.3f}<{self._floor:.2f}")
        if not failed:
            return True, "Alle P1-Goals über Material-Schwellwert → Early Exit"
        return False, f"{len(failed)} P1-Goals unter Schwellwert: {', '.join(failed[:3])}"
