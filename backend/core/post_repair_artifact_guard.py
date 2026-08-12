#!/usr/bin/env python3
"""
§v10.610: Post-Repair Artifact Guard — Pumping- und Verzerrungsschutz für die SOTA-Kette.

Problem: Die SOTA-Pipelines (Denoise, Vocal, Repair, Inpainting) haben keine
eigenen Artefakt-Guards. Wenn der DiT zu aggressiv rekonstruiert oder der
Denoiser zu stark subtrahiert, entstehen Pumping/Verzerrung — und die globalen
Guards in unified_restorer_v3 laufen erst NACH der ganzen Kette.

Lösung: Der PostRepairArtifactGuard läuft NACH JEDEM Repair-Schritt:
  1. Formant-Drift-Check (vocal_overprocessing_detector) — erkennt
     Verzerrung durch zu aggressives Processing
  2. TruePeak-Check — erkennt Clipping/Übersteuerung
  3. Pumping-Check (Gain-Modulation im Zeitbereich) — erkennt Atmung
  4. Bei Verstoß: automatische Strength-Reduktion (Blend Richtung Original)

Integration: In CoordinatedRepair.execute() nach jedem Schritt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

SR = 48000

# Schwellwerte
TRUEPEAK_LIMIT_DBFS = 0.0      # 0 dBFS — darüber = Clipping
TRUEPEAK_WARN_DBFS = -0.5      # Warnschwelle
PUMPING_GAIN_MODULATION_MAX = 0.15  # max. 15% Gain-Modulation pro 100ms
FORMANT_DRIFT_MAX = 0.08       # max. 8% Formant-Drift


@dataclass
class GuardResult:
    """Ergebnis einer Artefakt-Prüfung."""
    passed: bool
    truepeak_dbfs: float
    pumping_index: float         # 0-1, 0 = kein Pumpen
    formant_drift: float         # 0-1, 0 = keine Drift
    violations: list[str] = field(default_factory=list)
    blended_back: bool = False   # Wurde Strength automatisch reduziert?


class PostRepairArtifactGuard:
    """
    Prüft nach jedem Repair-Schritt auf Pumping und Verzerrung.

    Nutzung:
        guard = PostRepairArtifactGuard()
        result = guard.check(audio_pre, audio_post, sr, phase_id)
        if not result.passed:
            audio_post = guard.blend_back(audio_pre, audio_post, 0.7)
    """

    def __init__(self):
        self._overprocessing = None
        self._init_detectors()

    def _init_detectors(self):
        try:
            from backend.core.vocal_overprocessing_detector import VocalOverprocessingDetector
            self._overprocessing = VocalOverprocessingDetector()
            log.debug("Artifact Guard: VocalOverprocessingDetector geladen")
        except Exception as exc:
            log.debug("Artifact Guard: Overprocessing-Detektor nicht verfügbar (%s)", exc)

    def check(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        sr: int = SR,
        phase_id: str = "unknown",
    ) -> GuardResult:
        """
        Führt alle drei Artefakt-Checks durch.

        Returns:
            GuardResult mit passed-Flag und Metriken.
        """
        violations: list[str] = []

        # ── Check 1: TruePeak (Clipping) ──
        truepeak = float(np.abs(audio_post).max())
        truepeak_dbfs = float(20 * np.log10(truepeak + 1e-10))
        if truepeak_dbfs > TRUEPEAK_LIMIT_DBFS:
            violations.append(f"truepeak_overflow_{truepeak_dbfs:+.1f}dBFS")
        elif truepeak_dbfs > TRUEPEAK_WARN_DBFS:
            violations.append(f"truepeak_warn_{truepeak_dbfs:+.1f}dBFS")

        # ── Check 2: Pumping (Gain-Modulation) ──
        pumping_index = self._measure_pumping(audio_pre, audio_post, sr)
        if pumping_index > PUMPING_GAIN_MODULATION_MAX:
            violations.append(f"pumping_{pumping_index:.2f}")

        # ── Check 3: Formant-Drift (Verzerrung) ──
        formant_drift = 0.0
        if self._overprocessing is not None:
            try:
                result = self._overprocessing.check_formant_drift(
                    vocals_pre=audio_pre,
                    vocals_post=audio_post,
                    sr=sr,
                    phase_id=phase_id,
                )
                if result is not None:
                    drift = getattr(result, "drift", None) or getattr(result, "score", None)
                    if drift is not None:
                        formant_drift = float(drift)
            except Exception:
                pass

        if formant_drift > FORMANT_DRIFT_MAX:
            violations.append(f"formant_drift_{formant_drift:.2f}")

        passed = len(violations) == 0

        return GuardResult(
            passed=passed,
            truepeak_dbfs=truepeak_dbfs,
            pumping_index=pumping_index,
            formant_drift=formant_drift,
            violations=violations,
        )

    def _measure_pumping(self, pre: np.ndarray, post: np.ndarray, sr: int) -> float:
        """
        Misst Gain-Modulation: wie stark schwankt die Verstärkung im Zeitverlauf?
        Pumping = periodisches An-/Abschwellen der Lautstärke.
        """
        if len(pre) == 0 or len(post) == 0:
            return 0.0

        # Frame-Energien (100 ms Fenster)
        frame_len = sr // 10  # 100 ms
        n_frames = max(1, min(len(pre), len(post)) // frame_len)

        pre_env = np.zeros(n_frames, dtype=np.float64)
        post_env = np.zeros(n_frames, dtype=np.float64)

        for i in range(n_frames):
            s = i * frame_len
            e = s + frame_len
            pre_env[i] = np.sqrt(np.mean(pre[s:e] ** 2) + 1e-10)
            post_env[i] = np.sqrt(np.mean(post[s:e] ** 2) + 1e-10)

        # Gain = post/pre pro Frame
        gain = post_env / (pre_env + 1e-10)

        # Pumping-Index = Variationskoeffizient des Gains
        if gain.std() > 0:
            pumping = float(gain.std() / (gain.mean() + 1e-10))
        else:
            pumping = 0.0

        return min(pumping, 1.0)

    def blend_back(
        self,
        audio_pre: np.ndarray,
        audio_post: np.ndarray,
        blend_ratio: float = 0.7,
    ) -> np.ndarray:
        """
        Reduziert die Strength automatisch: blend_ratio Anteil Original,
        (1 - blend_ratio) Anteil prozessiert.
        """
        return (blend_ratio * audio_pre + (1 - blend_ratio) * audio_post).astype(np.float32)

    def normalize_truepeak(self, audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
        """Begrenzt TruePeak auf target_dbfs."""
        peak = float(np.abs(audio).max())
        if peak <= 0:
            return audio
        target = 10 ** (target_dbfs / 20)
        if peak > target:
            return (audio * (target / peak)).astype(np.float32)
        return audio


# ═════════════════════════════════════════════════════════════════════════════
# Integration in Coordinated Repair
# ═════════════════════════════════════════════════════════════════════════════

def run_post_repair_guard(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    sr: int = SR,
    phase_id: str = "unknown",
) -> tuple[np.ndarray, GuardResult]:
    """
    Convenience-Funktion: Prüft und korrigiert automatisch.

    Returns:
        (korrigiertes_audio, GuardResult)
    """
    guard = PostRepairArtifactGuard()
    result = guard.check(audio_pre, audio_post, sr, phase_id)

    if not result.passed:
        # Automatische Korrektur
        corrected = guard.blend_back(audio_pre, audio_post, 0.7)
        corrected = guard.normalize_truepeak(corrected)
        result.blended_back = True
        log.warning(
            "Artifact Guard %s: Verstöße %s → Strength reduziert (Blend 70/30)",
            phase_id, result.violations,
        )
        return corrected, result

    return audio_post, result
