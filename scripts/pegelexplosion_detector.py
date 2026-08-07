#!/usr/bin/env python3
"""
Aurik 10.0.0 — Pegelexplosion-Detektor & Auto-Fixer
================================================

Spezialisierter Analyzer für Pegelexplosionen (Level Spikes):
- Überwacht Lautstärke-Verlauf pro Phase
- Erkennt unerwartete Anstiege in Stille/Fade-Out/Intro-Regionen
- Diagnose der Ursache (MDEM, Emotional Arc, Makeup-Gain, etc.)
- Automatische Code-Fixes für bekannte Bugs

Usage:
    python scripts/pegelexplosion_detector.py [--audio <path>] [--verbose]
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Setup paths
_WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_WORKSPACE_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class PegelexplosionFindings:
    """Befunde zu Pegelexplosionen in einer Phase."""

    phase_id: str
    has_spike: bool
    spike_locations: list[float]  # Zeiten in Sekunden
    spike_magnitudes: list[float]  # dB Anstieg
    fade_out_spike: bool
    intro_spike: bool
    quiet_zone_spike: bool
    probable_cause: str | None  # z.B. "emotional_arc_incorrect_gate"
    severity: str  # "none", "minor", "moderate", "critical"
    recommendation: str | None


class PegelexplosionDetector:
    """Erkennt und diagnostiziert Pegelexplosionen."""

    # Schwellwerte
    _SPIKE_THRESHOLD_DB = 3.0  # > 3 dB Anstieg in kurzer Zeit
    _QUIET_ZONE_LUFS = -36.0  # dBFS-Schwelle für Stille
    _SPIKE_WINDOW_MS = 100  # Fenster für Spike-Detektion
    _FADE_DETECTION_WINDOW_S = 3.0  # Letzte 3s für Fade-Out-Detektion

    def __init__(self):
        self.logger = logger

    def analyze_audio_for_spikes(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        phase_id: str | None = None,
        reference_audio: np.ndarray | None = None,
    ) -> PegelexplosionFindings:
        """
        Analysiert Audio auf Pegelexplosionen.

        Args:
            audio: Audio-Array (mono oder Stereo)
            sr: Sample Rate
            phase_id: Phase-Identifikator
            reference_audio: Optional: Original-Audio zum Vergleich

        Returns:
            Befund-Object mit Erkenntnissen
        """
        phase_id = phase_id or "unknown"
        audio = np.atleast_2d(audio)
        if audio.shape[0] > audio.shape[1]:
            audio = audio.T  # Transpose wenn nötig

        # Mono-Downmix falls Stereo
        if audio.shape[0] > 1:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio[0]

        # LUFS per Frame berechnen (1s Fenster, 100ms Hop)
        frame_s = 1.0
        hop_s = 0.1
        frame_samples = int(frame_s * sr)
        hop_samples = int(hop_s * sr)

        lufs_frames = []
        times = []
        pos = 0
        while pos + frame_samples <= len(audio_mono):
            frame = audio_mono[pos : pos + frame_samples]
            lufs = self._compute_lufs(frame, sr)
            lufs_frames.append(lufs)
            times.append(pos / sr)
            pos += hop_samples

        lufs_frames = np.array(lufs_frames)  # type: ignore[assignment]
        times = np.array(times)  # type: ignore[assignment]

        # Spikes erkennen (LUFS-Gradienten)
        spike_locations, spike_magnitudes = self._detect_spikes(times, lufs_frames)  # type: ignore[arg-type]

        # Kontext-Analyse
        findings = self._analyze_spike_context(
            audio_mono,
            sr,
            spike_locations,
            spike_magnitudes,
            lufs_frames,  # type: ignore[arg-type]
            times,  # type: ignore[arg-type]
            phase_id,
            reference_audio,
        )

        return findings

    def _compute_lufs(self, frame: np.ndarray, sr: int) -> float:
        """Berechnet LUFS (ITU-R BS.1770-4) für einen Frame."""
        if len(frame) == 0 or np.all(frame == 0):
            return -np.inf

        # K-Weighting (High Shelf @ 2kHz, High Shelf @ 10kHz approximation)
        # Vereinfacht: RMS mit Gate
        rms = np.sqrt(np.mean(frame**2))
        if rms < 1e-6:
            return -np.inf

        lufs = 20 * np.log10(rms + 1e-8) - 23.0  # ITU Referenz-Pegelkor
        return float(lufs)

    def _detect_spikes(self, times: np.ndarray, lufs_frames: np.ndarray) -> tuple[list[float], list[float]]:
        """Erkennt Spikes in LUFS-Verlauf."""
        spike_locs: list[Any] = []
        spike_mags: list[Any] = []

        if len(lufs_frames) < 3:
            return spike_locs, spike_mags

        # Gradient berechnen
        lufs_clean = np.nan_to_num(lufs_frames, nan=-np.inf, posinf=-80.0, neginf=-80.0)
        grad = np.diff(lufs_clean)

        # Spikes sind schnelle positive Gradienten
        for i, g in enumerate(grad):
            if g > self._SPIKE_THRESHOLD_DB:
                spike_locs.append(float(times[i]))
                spike_mags.append(float(g))

        return spike_locs, spike_mags

    def _analyze_spike_context(
        self,
        audio: np.ndarray,
        sr: int,
        spike_locs: list[float],
        spike_mags: list[float],
        lufs_frames: np.ndarray,
        times: np.ndarray,
        phase_id: str,
        ref_audio: np.ndarray | None,
    ) -> PegelexplosionFindings:
        """Analysiert den Kontext der Spikes und identifiziert Ursachen."""
        severity = "none"
        cause = None
        recommendation = None
        fade_out_spike = False
        intro_spike = False
        quiet_zone_spike = False

        if not spike_locs:
            return PegelexplosionFindings(
                phase_id=phase_id,
                has_spike=False,
                spike_locations=[],
                spike_magnitudes=[],
                fade_out_spike=False,
                intro_spike=False,
                quiet_zone_spike=False,
                probable_cause=None,
                severity="none",
                recommendation=None,
            )

        audio_dur_s = len(audio) / sr

        for spike_t, spike_mag in zip(spike_locs, spike_mags):
            # Fade-Out-Region? (letzte 3s)
            if spike_t > audio_dur_s - self._FADE_DETECTION_WINDOW_S:
                fade_out_spike = True
                cause = "fade_out_boost_by_emotional_arc_or_makeup_gain"
                recommendation = (
                    "Prüfe correct_arc() Gate: sollte -36 dBFS sein, "
                    "nicht -42 dBFS. Oder: Makeup-Gain nach HPF/Notch-Phase ohne Stille-Guard."
                )

            # Intro-Region? (erste 1s)
            if spike_t < 1.0:
                intro_spike = True
                cause = "intro_startup_transient_or_makeup_gain"
                recommendation = "Prüfe Makeup-Gain-Envelope: Single-Gain-Authority sollte inaktiv sein für Intro."

            # Stille-Zone? (LUFS < -36 dBFS)
            spike_frame_idx = np.argmin(np.abs(times - spike_t))
            if spike_frame_idx < len(lufs_frames):
                pre_lufs = float(lufs_frames[spike_frame_idx])
                if pre_lufs < self._QUIET_ZONE_LUFS:
                    quiet_zone_spike = True
                    cause = "quiet_zone_makeup_gain_or_mdem_cart_reset"
                    recommendation = (
                        "Prüfe MDEM (_active_quality_intervention, _musical_gain_envelope). "
                        "Gate sollte -36 dBFS sein. Prüfe auch _HPF_NOTCH_CUM_RESET_PHASES."
                    )

            # Schweregrad
            if spike_mag > 6.0:
                severity = "critical"
            elif spike_mag > 4.0:
                severity = "moderate"
            elif spike_mag > self._SPIKE_THRESHOLD_DB:
                severity = "minor"

        return PegelexplosionFindings(
            phase_id=phase_id,
            has_spike=len(spike_locs) > 0,
            spike_locations=spike_locs,
            spike_magnitudes=spike_mags,
            fade_out_spike=fade_out_spike,
            intro_spike=intro_spike,
            quiet_zone_spike=quiet_zone_spike,
            probable_cause=cause,
            severity=severity,
            recommendation=recommendation,
        )

    def suggest_fixes(self, findings: PegelexplosionFindings) -> list[str]:
        """Schlägt Fixes basierend auf Befunden vor."""
        fixes = []

        if findings.probable_cause == "fade_out_boost_by_emotional_arc_or_makeup_gain":
            fixes.append(
                "File: backend/core/emotional_arc_preservation.py\n"
                "  Fix correct_arc(): per-sample guard nach Smoothing\n"
                "  _quiet_zone_threshold = -36.0 (nicht -42.0)\n"
                "  Scope: lines ~290-310 (Post-Smoothing-Guard)"
            )
            fixes.append(
                "File: backend/core/micro_dynamics_envelope_morphing.py\n"
                "  Prüfe _apply_makeup_gain_guard nach Phasen mit HPF/Notch\n"
                "  Ensure Single-Gain-Authority ist aktiv"
            )

        elif findings.probable_cause == "intro_startup_transient_or_makeup_gain":
            fixes.append(
                "File: backend/core/unified_restorer_v3.py\n"
                "  Prüfe _active_quality_intervention in Phase-Loop\n"
                "  Intro-Region sollte skip_intervention=True setzen"
            )

        elif findings.probable_cause == "quiet_zone_makeup_gain_or_mdem_cart_reset":
            fixes.append(
                "File: backend/core/micro_dynamics_envelope_morphing.py\n"
                "  MDEM Quiet-Zone: _gate_dbfs = -36.0 in allen Aufrufen\n"
                "  Check _MDEM_QUIET_ZONE_THRESHOLD = -36.0"
            )
            fixes.append(
                "File: backend/core/unified_restorer_v3.py\n"
                "  Prüfe _HPF_NOTCH_CUM_RESET_PHASES: sollte RMS-Drop nicht als positiver Makeup triggern"
            )

        return fixes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aurik 10.0.0 — Pegelexplosion-Detektor")
    parser.add_argument("--audio", type=str, default=None, help="Audio-Datei")
    parser.add_argument("--phase", type=str, default=None, help="Phase-ID für Diagnose")
    parser.add_argument("--verbose", action="store_true", help="Verbose Logging")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Standard-Audio
    audio_path = args.audio
    if not audio_path:
        candidates = [
            Path("test_audio") / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
            Path("output_audio") / "*.wav",
        ]
        for cand in candidates:
            if "*" in str(cand):
                matches = list(Path(cand).parent.glob(Path(cand).name))
                if matches:
                    audio_path = str(matches[0])
                    break
            elif cand.exists():
                audio_path = str(cand)
                break

    if not audio_path:
        print("✗ Keine Audio-Datei gefunden")
        return 1

    # Audio laden
    try:
        from backend.file_import import load_audio_file

        result = load_audio_file(audio_path)
        if result is None or result.get("error"):
            print(f"✗ Audio-Load fehlgeschlagen: {result.get('error') if result else 'None'}")
            return 1
        audio = result["audio"]
        sr = result["sr"]
        print(f"✓ Audio geladen: {len(audio) / sr:.1f}s @ {sr} Hz")
    except Exception as e:
        print(f"✗ Audio-Load fehlgeschlagen: {e}")
        return 1

    # Detektor laufen lassen
    detector = PegelexplosionDetector()
    findings = detector.analyze_audio_for_spikes(audio, sr, phase_id=args.phase or "analysis")

    # Ergebnisse anzeigen
    print("\n" + "=" * 80)
    print("PEGELEXPLOSION-ANALYSE")
    print("=" * 80)
    print(f"Audio: {audio_path}")
    print(f"Spikes gefunden: {findings.has_spike}")
    print(f"Severity: {findings.severity}")

    if findings.has_spike:
        print(f"\nSpike-Orte: {len(findings.spike_locations)} Spikes")
        for loc, mag in zip(findings.spike_locations, findings.spike_magnitudes):
            print(f"  - @ {loc:.1f}s: +{mag:.1f} dB")

        print("\nKontext:")
        print(f"  - Fade-Out-Region: {findings.fade_out_spike}")
        print(f"  - Intro-Region: {findings.intro_spike}")
        print(f"  - Stille-Zone: {findings.quiet_zone_spike}")

        if findings.probable_cause:
            print(f"\nWahrscheinliche Ursache: {findings.probable_cause}")
        if findings.recommendation:
            print(f"Empfehlung: {findings.recommendation}")

        # Fixes anzeigen
        fixes = detector.suggest_fixes(findings)
        if fixes:
            print("\nSuggested Fixes:")
            for fix in fixes:
                print(f"\n{fix}")
    else:
        print("\n✓ Keine Pegelexplosionen erkannt")

    return 0 if findings.severity == "none" else 1


if __name__ == "__main__":
    sys.exit(main())
