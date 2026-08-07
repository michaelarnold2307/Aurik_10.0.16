"""§P2 Style-Intent-Detector — unterscheidet intentionale Pitch-Abweichungen von Schäden.

Erkennt konsistente Pitch-Deviationen (z.B. Blue Notes, Microtonal Bends, Culture-Specific
Tuning), die Ausdruck des künstlerischen Stils sind — und markiert diese Zonen als
geschützte Segmente. Phase_31 (Pitch-Korrektur) und Phase_42 (Vokal-Enhancement) lesen
`style_intent_zones` aus dem Restoration-Context und reduzieren dort ihre Stärke.

Algorithmus:
    1. F0-Extraktion: FCPE (primär) → librosa.pyin (Fallback) → Autokorrelation (2. Fallback)
    2. Pro Frame: Voiced-Detection (Konfidenz > 0.5, F0 ∈ [80, 900] Hz)
    3. F0 → nächste MIDI-Note (gerundet) → Deviation_cents = 1200 * log2(f0 / midi_hz)
    4. Gruppierung nach Pitch-Class (MIDI % 12) → (deviation_cents, time_sec) sammeln
    5. Per Pitch-Class: wenn n ≥ 5, std < 20 cents, |mean| > 10 cents → intentional
    6. `style_intent_zones`: Zeitfenster, in denen intentionale Pitch-Classes gesungen werden

Singleton-Pattern (thread-safe, §Codierregeln).
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import correlate as _scipy_correlate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanten (§P2)
# ---------------------------------------------------------------------------
_STYLE_INTENT_MIN_DEVIATION_CENTS: float = 10.0  # mind. 10 cents für Stil-Signal
_STYLE_INTENT_MAX_STD_CENTS: float = 20.0  # max. Streuung — konsistente Abweichung
_STYLE_INTENT_MIN_OCCURRENCES: int = 5  # mind. 5 Vorkommen gleicher Tonstufe
_F0_MIN_HZ: float = 80.0
_F0_MAX_HZ: float = 900.0
_F0_CONFIDENCE_MIN: float = 0.5
_HOP_LENGTH_FRAMES: int = 512  # bei 48 kHz ≈ 10.7 ms pro Frame


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class StyleIntentResult:
    """Ergebnis der Style-Intent-Analyse.

    Attributes:
        style_intent_zones:       Liste (start_s, end_s) intentionaler Pitch-Zonen.
        intentional_pitch_classes: pitch_class 0–11 → typische Abweichung in Cents.
        style_confidence:         Gesamt-Konfidenz [0, 1].
        n_intentional_events:     Anzahl intentionaler Voicing-Ereignisse.
        n_total_events:           Gesamt-Voiced-Frames.
    """

    style_intent_zones: list[tuple[float, float]] = field(default_factory=list)
    intentional_pitch_classes: dict[int, float] = field(default_factory=dict)
    style_confidence: float = 0.0
    n_intentional_events: int = 0
    n_total_events: int = 0


# ---------------------------------------------------------------------------
# Kernklasse
# ---------------------------------------------------------------------------


class StyleIntentDetector:
    """Erkennt intentionale Pitch-Abweichungen im Gesang.

    Singleton — verwende `get_style_intent_detector()`.
    """

    def __init__(self) -> None:
        self._fcpe_available: bool = importlib.util.find_spec("torchfcpe") is not None
        self._pyin_available: bool = importlib.util.find_spec("librosa") is not None
        self._fcpe_initialized: bool = False
        self._fcpe_model: Any = None  # lazy-init in _extract_f0_fcpe
        if self._fcpe_available:
            logger.debug("StyleIntentDetector: FCPE verfügbar (primär)")
        if self._pyin_available:
            logger.debug("StyleIntentDetector: librosa.pyin als Ersatzpfad verfügbar")

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def analyze(self, audio: np.ndarray, sr: int) -> StyleIntentResult:
        """Analysiert Gesang auf intentionale Pitch-Abweichungen.

        Args:
            audio: Mono-Float32-Array [-1, 1], beliebige SR.
            sr:    Sample-Rate in Hz.

        Returns:
            StyleIntentResult mit style_intent_zones und style_confidence.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0) if audio.shape[0] > audio.shape[1] else audio.mean(axis=1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        result = StyleIntentResult()
        if len(audio) < sr * 0.5:
            logger.debug("StyleIntentDetector: zu kurz (%.2f s) — übersprungen", len(audio) / sr)
            return result

        try:
            f0_hz, voiced_flag = self._extract_f0(audio, sr)
            result = self._analyze_pitch_classes(f0_hz, voiced_flag, sr)
        except Exception as exc:
            logger.debug("StyleIntentDetector.analyze fehlgeschlagen (nicht blockierend): %s", exc)

        return result

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _extract_f0(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Extrahiert F0-Kurve und Voiced-Maske.

        Returns:
            f0_hz:       Array[float] — F0 in Hz pro Frame (0 = unvoiced).
            voiced_flag: Array[bool]  — True wenn Voiced.
        """
        if self._fcpe_available:
            try:
                return self._extract_f0_fcpe(audio, sr)
            except Exception as exc:
                logger.debug("FCPE F0-Extraktion fehlgeschlagen: %s", exc)

        if self._pyin_available:
            try:
                return self._extract_f0_pyin(audio, sr)
            except Exception as exc:
                logger.debug("pyin F0-Extraktion fehlgeschlagen: %s", exc)

        return self._extract_f0_autocorr(audio, sr)

    def _extract_f0_fcpe(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """F0 via FCPE (torchfcpe)."""
        import torch  # type: ignore[import]  # pylint: disable=import-outside-toplevel
        import torchfcpe  # type: ignore[import]  # pylint: disable=import-outside-toplevel

        if not self._fcpe_initialized:
            self._fcpe_model = torchfcpe.spawn_bundled_infer_model(device="cpu")
            self._fcpe_initialized = True

        audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            f0 = self._fcpe_model.infer(
                audio_tensor,
                sr=sr,
                decoder_mode="local_argmax",
                threshold=_F0_CONFIDENCE_MIN,
            )
        f0_np = f0.squeeze().numpy().astype(np.float32)
        voiced = (f0_np >= _F0_MIN_HZ) & (f0_np <= _F0_MAX_HZ)
        f0_np[~voiced] = 0.0
        return f0_np, voiced

    def _extract_f0_pyin(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """F0 via librosa.pyin."""
        import librosa  # type: ignore[import]  # pylint: disable=import-outside-toplevel

        f0, voiced_flag, _ = librosa.pyin(
            audio.astype(np.float64),
            fmin=float(_F0_MIN_HZ),
            fmax=float(_F0_MAX_HZ),
            sr=sr,
            hop_length=_HOP_LENGTH_FRAMES,
            fill_na=0.0,
        )
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0)
        voiced = np.asarray(voiced_flag, dtype=bool)
        voiced &= (f0 >= _F0_MIN_HZ) & (f0 <= _F0_MAX_HZ)
        return f0, voiced

    def _extract_f0_autocorr(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Einfache Autokorrelations-basierte F0-Extraktion als letzter Fallback."""
        hop = _HOP_LENGTH_FRAMES
        n_frames = max(1, (len(audio) - hop) // hop)
        f0_out = np.zeros(n_frames, dtype=np.float32)
        voiced_out = np.zeros(n_frames, dtype=bool)

        min_lag = max(1, int(sr / _F0_MAX_HZ))
        max_lag = int(sr / _F0_MIN_HZ)

        for i in range(n_frames):
            frame = audio[i * hop : i * hop + hop]
            if len(frame) < hop // 2:
                continue
            # Energy check
            rms = float(np.sqrt(np.mean(frame**2)))
            if rms < 1e-4:
                continue
            # Autokorrelation
            corr = _scipy_correlate(frame, frame, mode="full", method="fft")
            corr = corr[len(corr) // 2 :]
            if max_lag >= len(corr):
                continue
            lag_range = corr[min_lag : max_lag + 1]
            if len(lag_range) == 0:
                continue
            peak_idx = int(np.argmax(lag_range)) + min_lag
            peak_val = corr[peak_idx] / (corr[0] + 1e-9)
            # Threshold 0.5: standard for voiced-frame detection in autocorr-based
            # F0 estimators (RAPT, AMDF). At 0.3 pure noise can exceed the threshold
            # by chance; 0.5 requires sigma*11 exceedance for a 512-sample uniform
            # noise window (P < 1e-28 per frame), virtually eliminating FP voicing.
            if peak_val > 0.5 and peak_idx > 0:
                f0 = float(sr) / peak_idx
                if _F0_MIN_HZ <= f0 <= _F0_MAX_HZ:
                    f0_out[i] = f0
                    voiced_out[i] = True

        return f0_out, voiced_out

    def _analyze_pitch_classes(self, f0_hz: np.ndarray, voiced: np.ndarray, sr: int) -> StyleIntentResult:
        """Analysiert Pitch-Classes auf konsistente Abweichungen."""
        result = StyleIntentResult()
        if not np.any(voiced):
            return result

        hop_sec = _HOP_LENGTH_FRAMES / max(sr, 1)
        n_frames = len(f0_hz)

        # Pro Pitch-Class: (deviation_cents, frame_idx) sammeln
        pitch_class_data: dict[int, list[tuple[float, int]]] = {i: [] for i in range(12)}

        for i in range(n_frames):
            if not voiced[i] or f0_hz[i] <= 0:
                continue
            # MIDI-Note und Pitch-Class
            midi_float = 12.0 * np.log2(float(f0_hz[i]) / 440.0) + 69.0
            midi_round = int(round(midi_float))
            midi_hz = 440.0 * (2.0 ** ((midi_round - 69) / 12.0))
            deviation = 1200.0 * float(np.log2(float(f0_hz[i]) / midi_hz))
            pc = midi_round % 12
            pitch_class_data[pc].append((deviation, i))

        result.n_total_events = int(np.sum(voiced))

        # Intentionale Pitch-Classes identifizieren
        intentional_pcs: dict[int, float] = {}
        # Adaptive minimum: a real intentional pitch class must have significantly
        # more events than a random/noise distribution would produce (12 equal PCs).
        # For n voiced frames: expected_per_pc = n/12; require at least 1.5× that.
        # This prevents pyin false-positive voiced frames in noise from being flagged.
        # E.g.: 67 voiced (noise) → threshold = max(5, int(5.58*1.5)+1) = 9 → 5 < 9 → blocked.
        _expected_per_pc = result.n_total_events / 12.0
        _adaptive_min = max(_STYLE_INTENT_MIN_OCCURRENCES, int(_expected_per_pc * 1.5) + 1)
        for pc, events in pitch_class_data.items():
            if len(events) < _adaptive_min:
                continue
            deviations = np.array([e[0] for e in events], dtype=np.float32)
            mean_dev = float(np.mean(deviations))
            std_dev = float(np.std(deviations))
            if abs(mean_dev) >= _STYLE_INTENT_MIN_DEVIATION_CENTS and std_dev <= _STYLE_INTENT_MAX_STD_CENTS:
                intentional_pcs[pc] = mean_dev
                result.n_intentional_events += len(events)

        result.intentional_pitch_classes = intentional_pcs
        if not intentional_pcs:
            return result

        # Style-Confidence: Anteil intentionaler Events / total voiced
        if result.n_total_events > 0:
            result.style_confidence = float(np.clip(result.n_intentional_events / result.n_total_events, 0.0, 1.0))

        # Style-Intent-Zonen: zusammenhängende Frame-Runs mit intentionalen PCs
        intentional_pc_set = set(intentional_pcs.keys())
        in_zone = False
        zone_start = 0.0
        zones: list[tuple[float, float]] = []

        for i in range(n_frames):
            is_intentional = False
            if voiced[i] and f0_hz[i] > 0:
                midi_round = int(round(12.0 * np.log2(float(f0_hz[i]) / 440.0) + 69.0))
                pc = midi_round % 12
                is_intentional = pc in intentional_pc_set

            if is_intentional and not in_zone:
                in_zone = True
                zone_start = i * hop_sec
            elif not is_intentional and in_zone:
                in_zone = False
                zone_end = i * hop_sec
                if zone_end - zone_start >= 0.05:  # mind. 50 ms
                    zones.append((zone_start, zone_end))

        if in_zone:
            zones.append((zone_start, n_frames * hop_sec))

        result.style_intent_zones = zones
        logger.debug(
            "StyleIntentDetector: %d intentionale PCs, %d Zonen, confidence=%.2f",
            len(intentional_pcs),
            len(zones),
            result.style_confidence,
        )
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: StyleIntentDetector | None = None
_lock = threading.Lock()


def get_style_intent_detector() -> StyleIntentDetector:
    """Gibt den thread-sicheren Singleton zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StyleIntentDetector()
    return _instance
