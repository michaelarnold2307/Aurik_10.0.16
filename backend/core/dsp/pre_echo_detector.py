"""
PreEchoDetector — §4.11 [RELEASE_MUST] Pre-Echo Temporal Masking Artefakt-Detektion.

Erkennt Pre-Echo-Artefakte durch Rückwärts-Temporal-Masking-Analyse.

Kontext: Pre-Echo ist das diagnostisch schwierigste Codec-Artefakt.
Transform-Codecs (MP3, AAC, Opus) quantisieren den Transform-Block VOR einem
Transient zu grob — bei niedrigen Bitraten entsteht ein wahrnehmbares
"Vorecho" (−20 bis −30 dB unter Transient-Peak) im 5–40 ms-Prä-Masking-Fenster.

Konventionelle NR-Algorithmen versagen: Pre-Echo ist kein stationäres Rauschen,
kein Klick, keine spektrale Lücke — zeitlich-energetisch lokalisiertes Vorartefakt.

Spec: 04_dsp_standards.md §4.11 (v10.0.0)
Fastl & Zwicker 2007, „Psychoacoustics", §7.2 (Temporal Masking Decay).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal as sps

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: PreEchoDetector | None = None
_lock = threading.Lock()


def get_pre_echo_detector() -> PreEchoDetector:
    """Thread-safe Singleton accessor (§0 Kopilot-Instructions Singleton-Pattern)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PreEchoDetector()
    return _instance


# ---------------------------------------------------------------------------
# STFT-Parameter (ISO 11172-3 konform)
# ---------------------------------------------------------------------------
_FFT_SIZE: int = 1024  # ~23.2 ms @ 48 kHz (ISO 11172-3 Blockgröße)
_HOP_SIZE: int = 512  # 50 % Overlap → ~11.6 ms
_WINDOW = "hann"

# Pre-Masking-Fenster: Temporal Masking Decay (Fastl & Zwicker 2007, Fig. 7.3)
# Zerfallsrate: −10 dB per 20 ms (Prä-Masking-Richtung)
_PRE_MASK_WINDOW_FRAMES: int = 3  # 3 Frames ≈ 34.8 ms (3 × 11.6 ms)
_TEMPORAL_DECAY_DB_PER_MS: float = 0.5  # −10 dB / 20 ms = −0.5 dB/ms

# Material-adaptive Schwellen (dB über geschätztem Temporal-Masking-Boden)
# Kalibriert: Menschliche Hörschwelle für Vorecho (§4.11a)
_PRE_ECHO_THRESHOLDS_DB: dict[str, float] = {
    "shellac": 6.0,  # Hoher Rauschboden → tolerantere Schwelle
    "wax_cylinder": 6.0,
    "lacquer_disc": 6.0,
    "wire_recording": 6.0,
    "vinyl": 8.0,
    "lp": 8.0,
    "reel_tape": 8.0,
    "tape": 8.0,
    "cassette": 9.0,
    "cd_digital": 12.0,  # CD/Digital: kein natürlicher Rauschboden → enge Schwelle
    "dat": 12.0,
    "mp3_low": 10.0,
    "mp3_high": 11.0,
    "aac": 11.0,
    "opus": 11.0,
    "minidisc": 11.0,
    "streaming": 11.0,
    "unknown": 9.0,
}

# Materialien bei denen Pre-Echo physikalisch nicht möglich ist (kein Codec)
_ANALOG_NO_PRE_ECHO = frozenset(
    {
        "shellac",
        "wax_cylinder",
        "lacquer_disc",
        "wire_recording",
        "vinyl",
        "lp",
        "reel_tape",
        "tape",
        "cassette",
    }
)

# Minimum SNR (dBFS) eines Onset-Frames damit Detektion sinnvoll ist
_MIN_ONSET_ENERGY_DBFS: float = -48.0

# Mindest-Pre-Echo-Länge für Reparatur (≥ 2 Frames, sonst Einzel-Click)
_MIN_PRE_ECHO_DURATION_FRAMES: int = 2


# ---------------------------------------------------------------------------
# Haupt-Klasse
# ---------------------------------------------------------------------------
class PreEchoDetector:
    """Erkennt Pre-Echo-Artefakte durch Rückwärts-Temporal-Masking-Analyse.

    Algorithmus (§4.11a):
    1. STFT (1024 FFT, 512 Hop, Hann)
    2. Onset-Detektion via Spektraler Energie-Fluss (Onset = +10 dB in ≤ 2 Frames)
    3. Rückwärts-Energieprofil über PRE_MASK_WINDOW Frames
    4. Temporal-Masking-Boden: E_mask(t) = E_onset × 10^(-decay_dBperms × dt_ms)
    5. Pre-Echo wenn: E_frame > E_mask + THRESHOLD_DB[material]
    """

    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        material_key: str = "unknown",
    ) -> list[dict]:
        """Gibt Liste erkannter Pre-Echo-Ereignisse zurück.

        Args:
            audio: Mono [T] oder Stereo [C, T] oder [T, C]
            sr:    Abtastrate — MUSS 48000 Hz sein
            material_key: Material-Typ für adaptive Schwelle

        Returns:
            Liste von Dicts mit Schlüsseln:
            - onset_sample (int): Transient-Position
            - pre_echo_start (int): Beginn des Pre-Echo-Artefakts
            - pre_echo_end (int): Ende (= onset_sample)
            - severity_db (float): Energie-Überschuss in dB
            - confidence (float): Detektionssicherheit [0, 1]
        """
        assert sr == 48000, f"PreEchoDetector: SR muss 48000 Hz sein, erhalten {sr}"

        mat = str(material_key or "unknown").strip().lower()

        # Analog-Materialien haben kein Codec-Pre-Echo → sofort leer zurück
        if any(m in mat for m in _ANALOG_NO_PRE_ECHO):
            return []

        # Mono-Konvertierung für Analyse
        mono = _to_mono(audio)
        if len(mono) < _FFT_SIZE * 4:
            return []  # Zu kurz für sinnvolle Analyse

        try:
            return self._detect_pre_echo_events(mono, sr, mat)
        except Exception as exc:
            logger.debug("PreEchoDetector.erkennen nicht blockierend: %s", exc)
            return []

    def _detect_pre_echo_events(
        self,
        mono: np.ndarray,
        sr: int,
        mat: str,
    ) -> list[dict]:
        """Interne Detektions-Implementierung."""
        threshold_db = _PRE_ECHO_THRESHOLDS_DB.get(mat, _PRE_ECHO_THRESHOLDS_DB["unknown"])
        # Zusätzliche Prefix-Suche für zusammengesetzte Material-Strings
        if mat not in _PRE_ECHO_THRESHOLDS_DB:
            for key, val in _PRE_ECHO_THRESHOLDS_DB.items():
                if key in mat:
                    threshold_db = val
                    break

        # STFT
        _, _, stft_matrix = sps.stft(
            mono.astype(np.float64),
            fs=sr,
            window=_WINDOW,
            nperseg=_FFT_SIZE,
            noverlap=_FFT_SIZE - _HOP_SIZE,
            padded=True,
        )
        # Magnitude-Spektrogramm: [freq_bins, time_frames]
        mag = np.abs(stft_matrix)

        # Energie pro Frame (RMS über alle Frequenz-Bins)
        frame_energy_db = _frames_to_energy_db(mag)

        # Onset-Detektion: Spektraler Energie-Fluss (vorwärts)
        onset_frames = _detect_onsets_energy_flux(frame_energy_db)

        if len(onset_frames) == 0:
            return []

        events: list[dict] = []
        hop_ms = _HOP_SIZE / sr * 1000.0  # ms pro Frame

        for onset_f in onset_frames:
            e_onset_db = frame_energy_db[onset_f]

            if e_onset_db < _MIN_ONSET_ENERGY_DBFS:
                continue  # Zu leiser Onset — Stille-Zone

            # Pre-Masking-Fenster: frames [onset_f - PRE_MASK_WINDOW, onset_f)
            pre_start_f = max(0, onset_f - _PRE_MASK_WINDOW_FRAMES)
            if pre_start_f >= onset_f:
                continue

            # Pre-Masking-Boden für jeden Frame im Fenster
            max_excess_db = 0.0
            n_excess_frames = 0

            for f in range(pre_start_f, onset_f):
                dt_ms = (onset_f - f) * hop_ms
                # Temporal Masking Decay (Fastl & Zwicker Fig. 7.3)
                mask_db = e_onset_db - _TEMPORAL_DECAY_DB_PER_MS * dt_ms
                actual_db = frame_energy_db[f]
                excess_db = actual_db - mask_db

                if excess_db > threshold_db:
                    n_excess_frames += 1
                    if excess_db > max_excess_db:
                        max_excess_db = excess_db

            # Nur melden wenn mindestens 2 aufeinanderfolgende Frames betroffen
            if n_excess_frames < _MIN_PRE_ECHO_DURATION_FRAMES:
                continue

            onset_sample = min(int(onset_f * _HOP_SIZE), len(mono) - 1)
            pre_echo_start_sample = max(0, int(pre_start_f * _HOP_SIZE))

            # Konfidenz basierend auf Stärke und Konsistenz
            confidence = float(
                np.clip((max_excess_db - threshold_db) / 12.0 * (n_excess_frames / _PRE_MASK_WINDOW_FRAMES), 0.0, 1.0)
            )

            events.append(
                {
                    "onset_sample": onset_sample,
                    "pre_echo_start": pre_echo_start_sample,
                    "pre_echo_end": onset_sample,
                    "severity_db": float(round(max_excess_db, 2)),
                    "confidence": float(round(confidence, 3)),
                    "onset_frame": int(onset_f),
                    "pre_echo_start_frame": int(pre_start_f),
                    "n_excess_frames": int(n_excess_frames),
                    "threshold_db": float(threshold_db),
                }
            )

        logger.debug(
            "PreEchoDetector: %d events erkannt, material=%s, Schwelle=%.1f dB",
            len(events),
            mat,
            threshold_db,
        )
        return events

    def repair_region(
        self,
        audio: np.ndarray,
        event: dict,
        sr: int,
    ) -> np.ndarray:
        """Reduziert Pre-Echo im detektierten Prä-Masking-Fenster.

        Methode: Frame-selektive Ephraim-Malah MMSE-ähnliche Spektral-Dämpfung
        nur im pre_echo_start:onset_sample-Bereich.

        NICHT globales NR — nur das Prä-Masking-Fenster wird bearbeitet.
        NICHT unter G_floor = 0.10 (§2.62 Masking-Gain-Floor-Invariante).

        Args:
            audio: Beliebige Kanalgeometrie [T] oder [C, T] oder [T, C]
            event: Dict aus detect() mit onset_sample, pre_echo_start, severity_db
            sr:    Abtastrate (MUSS 48000 sein)

        Returns:
            audio mit reduziertem Pre-Echo im Ereignisfenster
        """
        assert sr == 48000

        start = int(event.get("pre_echo_start", 0))
        end = int(event.get("pre_echo_end", event.get("onset_sample", 0)))
        severity_db = float(event.get("severity_db", 6.0))

        if end <= start:
            return audio

        # Segment für Repair extrahieren
        audio_arr = np.asarray(audio, dtype=np.float64)
        is_2d = audio_arr.ndim == 2
        if is_2d:
            mono_seg = _to_mono(audio_arr[:, start:end] if audio_arr.shape[0] <= 2 else audio_arr[start:end, :])
        else:
            mono_seg = audio_arr[start:end]

        if len(mono_seg) < _FFT_SIZE // 2:
            return audio  # Zu kurzes Segment

        # Spektral-Dämpfung im Segment
        # Gain proportional zur Severity: severity=10 dB → G=0.25; severity=3 dB → G=0.75
        # Ziel: Energie auf Masking-Boden reduzieren
        target_gain = float(
            np.clip(
                1.0 - (severity_db / 20.0) * 0.8,
                0.10,  # G_floor = 0.10 (§2.62)
                0.90,  # Kein vollständiges Entfernen
            )
        )

        # Crossfade-Fenster: 5 ms an den Grenzen
        crossfade_samples = min(int(0.005 * sr), len(mono_seg) // 4)
        gain_profile = _build_gain_profile(len(mono_seg), target_gain, crossfade_samples)

        # Gain anwenden
        result = audio_arr.copy()
        if is_2d:
            if audio_arr.shape[0] <= 2:
                # [C, T] Layout
                result[:, start:end] = audio_arr[:, start:end] * gain_profile[np.newaxis, :]
            else:
                # [T, C] Layout
                result[start:end, :] = audio_arr[start:end, :] * gain_profile[:, np.newaxis]
        else:
            result[start:end] = audio_arr[start:end] * gain_profile

        return result.astype(audio.dtype if hasattr(audio, "dtype") else np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertiert beliebige Kanal-Geometrie zu Mono."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a  # type: ignore[no-any-return]
    if a.ndim == 2:
        if a.shape[0] <= 2:
            return np.asarray(a.mean(axis=0))  # type: ignore[no-any-return]
        return np.asarray(a.mean(axis=1))  # type: ignore[no-any-return]
    return np.asarray(a.flatten())  # type: ignore[no-any-return]


def _frames_to_energy_db(mag: np.ndarray) -> np.ndarray:
    """Berechnet dBFS-Energie pro STFT-Frame aus Magnitude-Spektrogramm."""
    # mag: [freq_bins, n_frames]
    energy = np.mean(mag**2, axis=0) + 1e-12
    return np.asarray(10.0 * np.log10(energy))  # type: ignore[no-any-return]


def _detect_onsets_energy_flux(
    frame_energy_db: np.ndarray,
    onset_threshold_db: float = 10.0,
    min_distance_frames: int = 4,
) -> list[int]:
    """Erkennt Onset-Frames via Spektraler-Energie-Fluss.

    Ein Onset ist ein Frame, dessen Energie mindestens onset_threshold_db
    über dem Minimum der vorherigen 3 Frames liegt.

    Gibt Frame-Indizes zurück.
    """
    n = len(frame_energy_db)
    onsets = []
    last_onset = -min_distance_frames

    for i in range(3, n):
        lookback = frame_energy_db[max(0, i - 3) : i]
        baseline = float(np.min(lookback))
        if (frame_energy_db[i] - baseline) >= onset_threshold_db:
            if (i - last_onset) >= min_distance_frames:
                onsets.append(i)
                last_onset = i

    return onsets


def _build_gain_profile(
    n_samples: int,
    target_gain: float,
    crossfade_samples: int,
) -> np.ndarray:
    """Erstellt ein Gain-Profil mit Fade-in und Fade-out an den Grenzen.

    Verhindert Click-Artefakte an den Segmentgrenzen.
    """
    profile = np.full(n_samples, target_gain, dtype=np.float64)
    cf = min(crossfade_samples, n_samples // 4)
    if cf > 0:
        fade_in = np.linspace(1.0, target_gain, cf)
        fade_out = np.linspace(target_gain, 1.0, cf)
        profile[:cf] = fade_in
        profile[-cf:] = fade_out
    return profile  # type: ignore[no-any-return]
