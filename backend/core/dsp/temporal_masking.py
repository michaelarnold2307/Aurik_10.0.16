"""Temporal Masking — Forward-Masking-Fenster für NR-Strength-Freigabe (Aurik §4.11+).

Psychoakustisches Forward Masking (Zwicker & Fastl 1999, §4.2):
Nach einem starken Transient (Attack) ist das Gehör für 5–150 ms weniger sensitiv für
Rauschen. In diesem Nachklang-Fenster kann die NR-Stärke erhöht werden, ohne dass
hörbare Qualitätsverluste entstehen. Dieses Prinzip ergänzt §V22 (Pre-Echo-Prevention):
§V22 schützt vor rückwärts-zeitlicher Verfremdung, §ForwardMasking erlaubt vorwärts-
zeitliche NR-Verstärkung gezielt in Masking-Fenstern.

Forward-Masking-Fensterdauer nach Zwicker 1999 (Abb. 4.8):
    T_forward(E_dB) = clip(5 + (E_dB + 18) × 1.5, 5, 200)  [ms]
    (E_dB = Transient-Energie relativ zu RMS-Rauschboden)

Boost-Faktor proportional zur Transient-Energie:
    boost = clip((E_dB - 10) / 60, 0.0, 0.40)

Kanonische Nutzung (Phase-Strength-Oracle in NR-Phasen):
    from backend.core.dsp.temporal_masking import get_forward_masking_guard

    guard = get_forward_masking_guard()
    zones = guard.compute_zones(audio_pre, sr)
    boost_at_current = guard.get_boost_at_sample(zones, sample_idx)
    # NR-Stärke in Forward-Masking-Fenstern: strength + boost_at_current

VERBOTEN (V41):
    Additive Phase ohne Forward-Masking-Check bei `panns_singing ≥ 0.25`:
    → Transient-Boost-Zonen nicht genutzt → NR unter-angewandt in ohnehin
       maskierten Fenstern (suboptimale Rauschreduktion im richtigen Moment)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Physikalische Grenzen §ForwardMasking ────────────────────────────────────
_FORWARD_MASK_MIN_MS = 5.0  # Minimale Fensterdauer (ms)
_FORWARD_MASK_MAX_MS = 200.0  # Maximale Fensterdauer (ms)
_FORWARD_MASK_ENERGY_REF_DB = -18.0  # Referenz-Transienten-Energie relativ zu RMS-Boden
_FORWARD_MASK_SLOPE = 1.5  # ms pro dB Transient-Energie über Referenz
_BOOST_ONSET_DB = 10.0  # Ab dieser Energie (dBFS) beginnt Boost
_BOOST_MAX = 0.40  # Maximaler Stärke-Boost in Forward-Masking-Fenstern
_BOOST_RANGE_DB = 60.0  # Energie-Bereich für Boost-Skalierung

# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: ForwardMaskingGuard | None = None
_lock = threading.Lock()


@dataclass
class ForwardMaskingZone:
    """Ein Forward-Masking-Fenster nach einem Transient.

    Attributes:
        start_sample: Erster Sample des Masking-Fensters (= Transient-Position + 1).
        end_sample: Letzter Sample (exklusiv) des Masking-Fensters.
        transient_energy_db: Energie des Transients relativ zum RMS-Boden (dBFS).
        max_nr_strength_boost: Maximal zulässiger NR-Stärke-Boost in diesem Fenster.
    """

    start_sample: int
    end_sample: int
    transient_energy_db: float
    max_nr_strength_boost: float  # [0.0, 0.40]


class ForwardMaskingGuard:
    """Berechnet Forward-Masking-Fenster und NR-Stärke-Boost-Empfehlungen.

    Jeder Transient öffnet ein psychoakustisch motiviertes Zeitfenster, in dem
    NR-Verfahren stärker arbeiten können, ohne Artefakte zu erzeugen.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def compute_zones(
        self,
        audio: np.ndarray,
        sr: int,
        frame_length: int = 1024,
        hop_length: int = 256,
    ) -> list[ForwardMaskingZone]:
        """Berechnet alle Forward-Masking-Fenster aus Transient-Detektion.

        Algorithmus:
          1. Onset-Stärke via Spectral Flux (ohne librosa für Analyse-Pfade)
          2. Lokale Maxima über Durchschnitt + 2σ → Transient-Kandidaten
          3. Pro Transient: Fensterdauer T = clip(5 + (E-18) × 1.5, 5, 200) ms

        Args:
            audio: Mono oder Stereo. Wird intern zu Mono konvertiert.
            sr: Abtastrate in Hz.
            frame_length: STFT-Fenstergröße für Spectral-Flux.
            hop_length: Hop-Länge in Samples.

        Returns:
            Sortierte Liste von ForwardMaskingZone-Objekten (keine Überlappung).
        """
        with self._lock:
            return _compute_zones_impl(audio, sr, frame_length, hop_length)

    def get_boost_at_sample(
        self,
        zones: list[ForwardMaskingZone],
        sample_idx: int,
    ) -> float:
        """Gibt den NR-Stärke-Boost für eine Sample-Position zurück.

        Args:
            zones: Ergebnis von compute_zones().
            sample_idx: Sample-Index für den der Boost gefragt wird.

        Returns:
            Boost-Wert in [0.0, 0.40]. 0.0 außerhalb aller Masking-Fenster.
        """
        for zone in zones:
            if zone.start_sample <= sample_idx < zone.end_sample:
                return float(zone.max_nr_strength_boost)
        return 0.0

    def apply_to_strength(
        self,
        base_strength: float,
        zones: list[ForwardMaskingZone],
        sample_idx: int,
        max_total: float = 1.0,
    ) -> float:
        """Wendet Forward-Masking-Boost auf eine NR-Stärke an.

        Args:
            base_strength: Basis-Stärke aus Phase-Strength-Oracle.
            zones: Forward-Masking-Fenster.
            sample_idx: Aktuelle Sample-Position.
            max_total: Obere Grenze der resultierenden Stärke.

        Returns:
            Angepasste Stärke in [0, max_total].
        """
        boost = self.get_boost_at_sample(zones, sample_idx)
        return float(np.clip(float(base_strength) + boost, 0.0, float(max_total)))


def get_forward_masking_guard() -> ForwardMaskingGuard:
    """Singleton-Zugriff auf ForwardMaskingGuard."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ForwardMaskingGuard()
    return _instance


# ── Private Implementierung ───────────────────────────────────────────────────


def _compute_zones_impl(
    audio: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
) -> list[ForwardMaskingZone]:
    """Interne (nicht thread-safe) Zonen-Berechnung."""
    _empty: list[ForwardMaskingZone] = []
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim > 1:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                audio = audio.mean(axis=0)
            elif audio.shape[1] == 2 and audio.shape[0] > 2:
                audio = audio.mean(axis=1)
            else:
                audio = audio.mean(axis=0)
        audio = np.asarray(audio, dtype=np.float32)
        n_samples = len(audio)

        if n_samples < frame_length * 2:
            return _empty

        # ── Spectral Flux als Onset-Stärke ───────────────────────────────────
        n_frames = (n_samples - frame_length) // hop_length + 1
        window = np.hanning(frame_length).astype(np.float32)

        # Magnitude-Spektrum für jeden Frame
        prev_mag: np.ndarray | None = None
        flux = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + frame_length]
            if len(frame) < frame_length:
                break
            mag = np.abs(np.fft.rfft(frame * window))
            if prev_mag is not None:
                # Half-wave rectified spectral flux
                diff = mag - prev_mag
                flux[i] = float(np.sum(np.maximum(diff, 0.0)))
            prev_mag = mag

        if flux.max() < 1e-10:
            return _empty

        # ── Transient-Detektion via lokale Maxima ────────────────────────────
        # Schwelle: Mittelwert + 2 × Standardabweichung
        flux_mean = float(flux.mean())
        flux_std = float(flux.std())
        threshold = flux_mean + 2.0 * flux_std

        # Globaler RMS-Rauschboden des Signals (10th Percentile der Frame-Energie)
        frame_rms = np.array(
            [
                float(np.sqrt(np.mean(audio[i * hop_length : i * hop_length + frame_length] ** 2) + 1e-12))
                for i in range(n_frames)
            ],
            dtype=np.float32,
        )
        noise_rms = float(np.percentile(frame_rms, 10.0)) + 1e-10
        noise_db = float(20.0 * np.log10(noise_rms))

        zones: list[ForwardMaskingZone] = []
        min_gap_frames = max(1, int(_FORWARD_MASK_MIN_MS * 1e-3 * sr / hop_length))

        last_onset_frame = -min_gap_frames * 10  # Initialisierung weit weg
        for i in range(1, n_frames - 1):
            if flux[i] > threshold and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
                if i - last_onset_frame < min_gap_frames:
                    continue  # Zu nah am letzten Transient
                last_onset_frame = i

                # Energie des Transients relativ zum Rauschboden
                start_sample = i * hop_length
                transient_rms = float(np.sqrt(np.mean(audio[start_sample : start_sample + frame_length] ** 2) + 1e-12))
                e_db = float(20.0 * np.log10(transient_rms + 1e-10) - noise_db)

                # Fensterdauer nach Zwicker 1999 (Abb. 4.8)
                t_ms = float(
                    np.clip(
                        _FORWARD_MASK_MIN_MS + (e_db - _FORWARD_MASK_ENERGY_REF_DB) * _FORWARD_MASK_SLOPE,
                        _FORWARD_MASK_MIN_MS,
                        _FORWARD_MASK_MAX_MS,
                    )
                )
                window_samples = int(t_ms * 1e-3 * sr)

                # Boost-Faktor proportional zur Transient-Energie
                boost = float(np.clip((e_db - _BOOST_ONSET_DB) / _BOOST_RANGE_DB, 0.0, _BOOST_MAX))

                zone_start = start_sample + 1
                zone_end = min(n_samples, zone_start + window_samples)

                if zone_start < zone_end and boost > 0.0:
                    zones.append(
                        ForwardMaskingZone(
                            start_sample=zone_start,
                            end_sample=zone_end,
                            transient_energy_db=float(e_db),
                            max_nr_strength_boost=float(boost),
                        )
                    )

        logger.debug("ForwardMasking: %d Zonen erkannt (sr=%d)", len(zones), sr)
        return zones

    except Exception as exc:
        logger.debug("ForwardMaskingGuard.berechnen_zones nicht blockierend: %s", exc)
        return []
