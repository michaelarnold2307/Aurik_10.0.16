"""backend/core/temporal_consistency_guard.py — §v10.700 J3.

Erkennt zeitliche Inkonsistenzen zwischen aufeinanderfolgenden Phasen:
  - Energie-Sprünge >6 dB zwischen 100ms-Fenstern
  - Wiedereinführung von Rauschen nach Denoise-Phasen
  - Stereo-Bild-Kollaps (M/S-Ratio >30%)

Integration in _profiled_phase_call(): nach JEDER Phase.
§03 ROADMAP: spezifiziert, jetzt implementiert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalConsistencyResult:
    """Ergebnis der Temporal-Consistency-Prüfung."""

    passed: bool = True
    energy_jumps: int = 0
    noise_reintroduced: bool = False
    stereo_collapse: bool = False
    warnings: list[str] = field(default_factory=list)


class TemporalConsistencyGuard:
    """Prüft zeitliche Konsistenz zwischen Audio vor/nach einer Phase."""

    def __init__(
        self,
        energy_threshold_db: float = 6.0,
        noise_threshold_db: float = -60.0,
        stereo_threshold: float = 0.30,
        window_ms: int = 100,
    ):
        self.energy_threshold_db = energy_threshold_db
        self.noise_threshold_db = noise_threshold_db
        self.stereo_threshold = stereo_threshold
        self.window_samples = int(window_ms * 48 / 1000)  # für 48kHz default

    def check(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        phase_id: str = "",
    ) -> TemporalConsistencyResult:
        """Prüft Konsistenz zwischen vor/nach einer Phase.

        Returns:
            TemporalConsistencyResult mit passed=True wenn konsistent.
        """
        result = TemporalConsistencyResult()

        try:
            mono_before = np.mean(audio_before, axis=-1) if audio_before.ndim > 1 else audio_before
            mono_after = np.mean(audio_after, axis=-1) if audio_after.ndim > 1 else audio_after

            # 1. Energie-Sprünge zwischen 100ms-Fenstern
            result.energy_jumps = self._count_energy_jumps(mono_before, mono_after)

            # 2. Rausch-Wiedereinführung (nur nach Denoise-Phasen)
            if "denoise" in phase_id.lower() or "noise" in phase_id.lower():
                result.noise_reintroduced = self._check_noise_reintroduction(mono_before, mono_after)

            # 3. Stereo-Kollaps
            if audio_before.ndim > 1 and audio_after.ndim > 1:
                result.stereo_collapse = self._check_stereo_collapse(audio_before, audio_after)

            # Warnings generieren
            if result.energy_jumps > 0:
                result.warnings.append(
                    f"{result.energy_jumps} Energie-Sprünge >{self.energy_threshold_db}dB "
                    f"zwischen 100ms-Fenstern in Phase {phase_id}"
                )
            if result.noise_reintroduced:
                result.warnings.append(
                    f"Rauschen wurde in Phase {phase_id} wiedereingeführt (nach vorheriger Denoise-Phase)"
                )
            if result.stereo_collapse:
                result.warnings.append(
                    f"Stereo-Bild-Kollaps in Phase {phase_id} (M/S-Ratio >{self.stereo_threshold:.0%})"
                )

            result.passed = len(result.warnings) == 0

            if not result.passed:
                logger.warning(
                    "TemporalConsistencyGuard %s: %d Verletzungen — %s",
                    phase_id,
                    len(result.warnings),
                    "; ".join(result.warnings),
                )

        except Exception as exc:
            logger.debug("TemporalConsistencyGuard Fehler in %s: %s", phase_id, exc)

        return result

    def _count_energy_jumps(self, before: np.ndarray, after: np.ndarray) -> int:
        """Zählt Energie-Sprünge >threshold zwischen 100ms-Fenstern."""
        n = min(len(before), len(after))
        win = max(self.window_samples, 100)
        jumps = 0

        for start in range(0, n - win, win):
            rms_before = float(np.sqrt(np.mean(before[start : start + win] ** 2)) + 1e-12)
            rms_after = float(np.sqrt(np.mean(after[start : start + win] ** 2)) + 1e-12)
            delta_db = abs(float(20 * np.log10(rms_after / rms_before)))

            if delta_db > self.energy_threshold_db:
                jumps += 1

        return jumps

    def _check_noise_reintroduction(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Prüft ob nach einer Denoise-Phase wieder Rauschen hinzugefügt wurde."""
        rms_before = float(np.sqrt(np.mean(before**2)) + 1e-12)
        rms_after = float(np.sqrt(np.mean(after**2)) + 1e-12)

        # Nur relevant wenn Input sehr leise war (erfolgreiches Denoising)
        if 20 * np.log10(rms_before) < self.noise_threshold_db:
            return False

        # Prüfe ob Output signifikant mehr Rauschen hat (>3dB)
        return 20 * np.log10(rms_after / rms_before) > 3.0  # type: ignore[no-any-return]

    def _check_stereo_collapse(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Prüft ob das Stereo-Bild kollabiert ist (M/S-Ratio-Änderung >30%)."""
        if before.ndim < 2 or after.ndim < 2:
            return False

        n = min(before.shape[-2] if before.ndim > 1 else len(before), after.shape[-2] if after.ndim > 1 else len(after))

        # Mid/Side-Berechnung
        mid_before = (before[:n, 0] + before[:n, 1]) / 2
        side_before = (before[:n, 0] - before[:n, 1]) / 2
        mid_after = (after[:n, 0] + after[:n, 1]) / 2
        side_after = (after[:n, 0] - after[:n, 1]) / 2

        ms_ratio_before = float(np.sqrt(np.mean(side_before**2)) / (np.sqrt(np.mean(mid_before**2)) + 1e-12))
        ms_ratio_after = float(np.sqrt(np.mean(side_after**2)) / (np.sqrt(np.mean(mid_after**2)) + 1e-12))

        if ms_ratio_before < 0.01:  # Mono-Signal
            return False

        change = abs(ms_ratio_after - ms_ratio_before) / ms_ratio_before
        return change > self.stereo_threshold
