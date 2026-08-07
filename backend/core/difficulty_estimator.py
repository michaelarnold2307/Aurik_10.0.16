"""DifficultyEstimator — §INCREMENTAL #4: Vorab-Schätzung.

Schätzt vor der Restaurierung: Schwierigkeit, Dauer, erwartetes ΔMOS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DifficultyEstimate:
    difficulty: int = 5  # 1-10
    estimated_duration_minutes: float = 10.0
    expected_improvement_mos: float = 1.5
    defect_count: int = 0
    defect_types: list[str] = None  # type: ignore[assignment]
    material: str = "unknown"
    era: int = 0
    recommendation: str = ""

    def __post_init__(self):
        if self.defect_types is None:
            self.defect_types = []


def estimate(
    audio: np.ndarray,
    sr: int,
    material: str = "unknown",
    era: int = 0,
    defects: dict[str, float] = None,  # type: ignore[assignment]
) -> DifficultyEstimate:
    """Schätzt Restaurierungs-Schwierigkeit aus Audio-Charakteristik."""
    mono = np.mean(audio, axis=-1) if audio.ndim > 1 else np.asarray(audio, dtype=np.float32)
    n = len(mono)
    duration_min = n / sr / 60.0

    # Faktoren für Schwierigkeit
    rms = float(np.sqrt(np.mean(mono**2))) + 1e-10
    rms_db = 20 * np.log10(rms)

    n_fft = min(4096, n)
    spec = np.abs(np.fft.rfft(mono[: n_fft * 8], n=n_fft))
    log_mean = np.exp(np.mean(np.log(spec + 1e-10)))
    arith_mean = np.mean(spec)
    # §v10.93: Epsilon-Differenzierung — 1e-8 im Nenner vs 1e-10 im spec verhindert
    # Kollision bei Stille (sonst noise_ratio ≈ 1.0, falsch-positiv als "noisy").
    noise_ratio = log_mean / max(arith_mean, 1e-8)  # >0.5 = noisy

    clipped_pct = float(np.mean(np.abs(mono) > 0.95)) * 100

    # Score-Berechnung
    score = 3  # Basis
    if material in ("shellac", "wax_cylinder"):
        score += 3
    elif material in ("vinyl", "reel_tape", "tape"):
        score += 2
    elif material in ("mp3_low", "cassette"):
        score += 1

    if noise_ratio > 0.7:
        score += 2
    elif noise_ratio > 0.5:
        score += 1

    if clipped_pct > 5:
        score += 2
    elif clipped_pct > 1:
        score += 1

    if rms_db < -30:
        score += 1  # Sehr leise

    if defects:
        score += min(len(defects) // 2, 3)

    difficulty = int(np.clip(score, 1, 10))

    # Dauer-Schätzung: ~15s pro Minute Audio, × Schwierigkeit
    est_duration = duration_min * 15 * (difficulty / 5.0)

    # Erwartete Verbesserung
    improvement = max(0.5, 5.0 - difficulty * 0.4)

    rec = "Empfohlen" if difficulty <= 4 else ("Lohnend" if difficulty <= 7 else "Anspruchsvoll — Ergebnis prüfen")

    return DifficultyEstimate(
        difficulty=difficulty,
        estimated_duration_minutes=round(est_duration, 1),
        expected_improvement_mos=round(improvement, 1),
        defect_count=len(defects) if defects else 0,
        defect_types=list(defects.keys())[:5] if defects else [],
        material=material,
        era=era,
        recommendation=rec,
    )
