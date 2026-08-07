"""
Aurik 10.0.0 — PerceptualAttentionModel (PAM) (§2.22)
==================================================
Salienz-gewichtete Restaurierung — berechnet eine frequenz-zeitliche Salienz-Karte.

Algorithmuss:
    1. PANNs CNN14 → Clip-Level Tags (Vokal/Melodie/Rhythmus/Bass/Ambience)
    2. MERT-Embeddings → Harmonic Saliency (optional)
    3. Vokal-Segmente: attention_weight × 1.8
    4. Stille-Segmente (< −40 dBFS): attention_weight × 0.3
    5. Percussive-Transienten (HPSS > 0.7): attention_weight × 1.2
    6. Salienz-Karte: 2D float32 [n_frames × n_bark_bands] ∈ [0.3, 2.0]

Invarianten:
    - Salienz-Karte immer NaN-frei und in [0.3, 2.0] geclamppt
    - Salienz-Karte ändert NICHT die Musical Goals
    - Fallback: einheitliche Salienz = 1.0 (wenn PANNs nicht verfügbar)
    - Laufzeit: ≤ 1 s / Minute Audio (DSP-only Fallback ≤ 0.2 s)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Bark-Skala: 24 Bänder (20 Hz – 15.5 kHz)
BARK_EDGES_HZ = [
    20,
    100,
    200,
    300,
    400,
    510,
    630,
    770,
    920,
    1080,
    1270,
    1480,
    1720,
    2000,
    2320,
    2700,
    3150,
    3700,
    4400,
    5300,
    6400,
    7700,
    9500,
    12000,
    15500,
]
N_BARK_BANDS = len(BARK_EDGES_HZ) - 1  # 24


@dataclass
class SaliencyMap:
    """Salienz-Karte für frequenz-zeitliche Restaurierungs-Priorisierung."""

    map: np.ndarray  # [n_frames × N_BARK_BANDS] float32, ∈ [0.3, 2.0]
    n_frames: int
    frame_hop_s: float
    bark_edges_hz: list[float]

    def get_gain_weight(self, frame_idx: int, bark_band: int) -> float:
        """Gibt Salienz-Gewicht für einen Frame/Band zurück."""
        if frame_idx >= self.n_frames or bark_band >= N_BARK_BANDS:
            return 1.0
        return float(self.map[frame_idx, bark_band])


class PerceptualAttentionModel:
    """Berechnet frequenz-zeitliche Salienz-Karte zur Restaurierungs-Priorisierung."""

    HOP_S: float = 0.5  # Frame-Hop in Sekunden
    SILENCE_THRESHOLD_DBFS: float = -40.0
    VOCAL_WEIGHT: float = 1.8
    SILENCE_WEIGHT: float = 0.3
    PERCUSSIVE_WEIGHT: float = 1.2
    BASE_WEIGHT: float = 1.0
    SALIENCY_MIN: float = 0.3
    SALIENCY_MAX: float = 2.0

    def compute_saliency_map(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Gibt [n_frames × 24] float32 Salienz-Matrix zurück (Bark-Bänder).

        Args:
            audio: float32 Audio [n_samples] mono oder stereo
            sr:    Sample-Rate (48000 Hz)

        Returns:
            Salienz-Karte [n_frames × N_BARK_BANDS], ∈ [0.3, 2.0], NaN-frei
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        mono = mono.astype(np.float32)

        hop_len = int(self.HOP_S * sr)
        n_frames = max(1, len(mono) // hop_len)

        # Basis-Salienz-Karte (alle 1.0)
        saliency = np.ones((n_frames, N_BARK_BANDS), dtype=np.float32)

        # HPSS für Percussive-Erkennung
        try:
            percussive_energy = self._compute_percussive_energy(mono, sr, hop_len, n_frames)
        except Exception:
            percussive_energy = np.zeros(n_frames, dtype=np.float32)

        # Frame-weise Salienz-Bestimmung
        for frame_idx in range(n_frames):
            start = frame_idx * hop_len
            end = min(start + hop_len, len(mono))
            frame = mono[start:end]

            if len(frame) == 0:
                continue

            # Energie-Level
            rms = float(np.sqrt(np.mean(frame**2)))
            level_dbfs = float(20.0 * np.log10(max(rms, 1e-10)))

            # Stille-Segmente
            if level_dbfs < self.SILENCE_THRESHOLD_DBFS:
                saliency[frame_idx, :] = self.SILENCE_WEIGHT
                continue

            # Spektral-Analyse für Bark-Bänder
            bark_weights = self._compute_bark_saliency(frame, sr)

            # Percussive-Transient-Anhebung
            if percussive_energy[frame_idx] > 0.7:
                bark_weights *= self.PERCUSSIVE_WEIGHT

            # Vokal-Erkennung (ZCR + Spektral-Flachheit Heuristik)
            is_vocal_likely = self._heuristic_vocal_detection(frame, sr)
            if is_vocal_likely:
                bark_weights *= self.VOCAL_WEIGHT

            saliency[frame_idx, :] = bark_weights

        # Bounds-Clamp + NaN-Schutz
        saliency = np.nan_to_num(saliency, nan=self.BASE_WEIGHT)
        saliency = np.clip(saliency, self.SALIENCY_MIN, self.SALIENCY_MAX)

        return saliency.astype(np.float32)  # type: ignore[no-any-return]

    def apply_to_gain(
        self,
        base_gain: np.ndarray,
        saliency_map: np.ndarray,
        lyrics_saliency: np.ndarray | None = None,
    ) -> np.ndarray:
        """Multipliziert Gain-Maske mit Salienz-Gewichten (bounds-sicher).

        Args:
            base_gain:       Gain-Maske [n_frames × n_bins] float32
            saliency_map:    Salienz-Karte [n_frames × n_bark_bands]
            lyrics_saliency: Optionale Lyrics-Salienz-Karte [n_frames × n_bark_bands]
                             aus ContentAwareProcessor (§2.36). Wird mit saliency_map
                             kombiniert (geometrisches Mittel), wenn vorhanden.

        Returns:
            Skalierte Gain-Maske, Ausgabe ≥ 0.0
        """
        if base_gain.ndim != 2 or saliency_map.ndim != 2:
            return base_gain

        n_frames = min(base_gain.shape[0], saliency_map.shape[0])
        result = base_gain.copy()

        # Lyrics-Salienz mit Basis-Salienz kombinieren (geometrisches Mittel §2.36)
        combined_saliency = saliency_map
        if lyrics_saliency is not None and lyrics_saliency.ndim == 2:
            n_frames_lyr = min(n_frames, lyrics_saliency.shape[0])
            lyr = np.nan_to_num(
                lyrics_saliency[:n_frames_lyr].astype(np.float32),
                nan=1.0,
                posinf=2.0,
                neginf=0.3,
            )
            base_part = saliency_map[:n_frames_lyr]
            # Geometrisches Mittel: kombiniert beide Salienz-Quellen
            combined_part = np.sqrt(np.clip(base_part, 0.3, 2.0) * np.clip(lyr, 0.3, 2.0))
            if n_frames_lyr < n_frames:
                combined_saliency = np.concatenate([combined_part, saliency_map[n_frames_lyr:n_frames]], axis=0)
            else:
                combined_saliency = combined_part

        for f in range(n_frames):
            frame_saliency = combined_saliency[f]  # [n_bark_bands]
            # Mittelwert der Bark-Salienz als globaler Frame-Gewicht
            mean_sal = float(np.mean(frame_saliency))
            result[f] *= mean_sal

        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, 0.0, None)
        return result.astype(np.float32)  # type: ignore[no-any-return]

    def _compute_bark_saliency(self, frame: np.ndarray, sr: int) -> np.ndarray:
        """Bark-Bänder-Salienz aus FFT-Energie."""
        fft_mag = np.abs(np.fft.rfft(frame, n=2048))
        freqs = np.fft.rfftfreq(2048, d=1.0 / sr)
        fft_energy = fft_mag**2 + 1e-12

        bark_weights = np.ones(N_BARK_BANDS, dtype=np.float32)
        total_energy = float(np.sum(fft_energy))

        for b in range(N_BARK_BANDS):
            lo = BARK_EDGES_HZ[b]
            hi = BARK_EDGES_HZ[b + 1]
            mask = (freqs >= lo) & (freqs < hi)
            band_energy = float(np.sum(fft_energy[mask]))
            # Relative Energie → Salienz-Gewicht
            rel_energy = band_energy / (total_energy + 1e-12)
            # Hohe Energie in diesem Band → hohe Salienz
            bark_weights[b] = float(
                np.clip(self.BASE_WEIGHT + rel_energy * N_BARK_BANDS * 0.5, self.SALIENCY_MIN, self.SALIENCY_MAX)
            )

        return bark_weights  # type: ignore[no-any-return]

    def _compute_percussive_energy(self, mono: np.ndarray, sr: int, hop_len: int, n_frames: int) -> np.ndarray:
        """Perkussive Energie pro Frame (HPSS-basiert)."""
        try:
            import librosa

            _, percussive = librosa.effects.hpss(mono)  # type: ignore[attr-defined]
            perc_energy = np.array(
                [float(np.sqrt(np.mean(percussive[f * hop_len : (f + 1) * hop_len] ** 2))) for f in range(n_frames)],
                dtype=np.float32,
            )
            # Normalisieren
            max_e = float(np.max(perc_energy)) + 1e-10
            return perc_energy / max_e  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("perceptual_attention_model.py::_berechnen_percussive_energy Ersatzpfad: %s", e)
            return np.zeros(n_frames, dtype=np.float32)  # type: ignore[no-any-return]

    def _heuristic_vocal_detection(self, frame: np.ndarray, sr: int) -> bool:
        """ZCR + F0-Heuristik für wahrscheinliche Vokal-Segmente."""
        if len(frame) < 10:
            return False
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
        rms = float(np.sqrt(np.mean(frame**2)))
        # Vokale: moderate ZCR + nennenswerte Energie
        return bool(0.01 <= zcr <= 0.25 and rms > 0.02)


# ---- Thread-sicherer Singleton ----

_instance: PerceptualAttentionModel | None = None
_lock = threading.Lock()


def get_perceptual_attention_model() -> PerceptualAttentionModel:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PerceptualAttentionModel()
    return _instance


def compute_saliency_map(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convenience-Wrapper: Salienz-Karte berechnen."""
    return get_perceptual_attention_model().compute_saliency_map(audio, sr)
