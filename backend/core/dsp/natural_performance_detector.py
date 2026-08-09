"""
§2.46f Natural-Performance-Artifacts-Guard — v10.0.0

Schützt performancebedingte Klangereignisse vor versehentlicher Entfernung
durch NR, Gate, Dereverb oder Pitch-Phasen. Diese Ereignisse sind KEIN Defekt —
sie gehören zur musikalischen Performance und zum emotionalen Ausdruck.

Drei geschützte Kategorien (§2.46f):
  1. Atemgeräusche   — zwischen Phrasen (−55 bis −40 dBFS, 50–500 ms, flatness > 0.4)
  2. Vibrato/Portamento — F0-Modulation 4–7 Hz, ≤ ±50 Cent
  3. Early Reflections — 0–50 ms nach Onset (Studio-Raumcharakter)

Non-blocking: Jede Exception führt zum sicheren Fallback (leere Schutzzonen).

VERBOTEN: Atemgeräusche via NR oder Gate entfernen; Vibrato glätten oder
quantisieren; Early Reflections via Dereverb entfernen (wet_mix cap gilt).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from scipy.signal import correlate as _sp_correlate
except Exception:  # pragma: no cover
    _sp_correlate = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurations-Konstanten
# ─────────────────────────────────────────────────────────────────────────────
# Atemgeräusche
_BREATH_ENERGY_MIN_DBFS: float = -55.0  # Mindestsignal (nicht Stille)
_BREATH_ENERGY_MAX_DBFS: float = -40.0  # Obergrenzen (kein Musiksignal)
_BREATH_MIN_DUR_S: float = 0.050  # 50 ms Mindestdauer
_BREATH_MAX_DUR_S: float = 0.500  # 500 ms Maximaldauer
_BREATH_FLATNESS_MIN: float = 0.40  # spectral flatness > 0.40 = rauschähnlich (kein Ton)

# Vibrato/Portamento
_VIBRATO_RATE_MIN_HZ: float = 4.0  # Mindest-Modulationsfrequenz
_VIBRATO_RATE_MAX_HZ: float = 7.0  # Maximal-Modulationsfrequenz
_VIBRATO_DEPTH_MAX_CENT: float = 50.0  # ±50 Cent Amplitude

# Early Reflections
_EARLY_REF_MAX_MS: float = 50.0  # Erste 50 ms nach Onset = Early Reflection

# Detektions-Zuverlässigkeits-Mindestlänge
_MIN_AUDIO_S: float = 0.100  # < 100 ms → kein sinnvoller Scan


# ─────────────────────────────────────────────────────────────────────────────
# Datenklassen
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BreathZone:
    """Zeitbereich eines Atemgeräuschs (Sample-Indizes, inklusive)."""

    start_sample: int
    end_sample: int
    energy_dbfs: float
    flatness: float


@dataclass(frozen=True)
class VibratoZone:
    """Zeitbereich mit Vibrato/Portamento (Sample-Indizes, inklusive)."""

    start_sample: int
    end_sample: int
    rate_hz: float
    depth_cent: float


@dataclass(frozen=True)
class EarlyReflectionZone:
    """Onset-gebundener Early-Reflection-Bereich (0–50 ms nach Onset)."""

    onset_sample: int
    end_sample: int  # = onset_sample + round(sr * 0.050)


@dataclass
class NaturalPerformanceProfile:
    """
    Vollständiges Profil natürlicher Performance-Artefakte für einen Song.

    Wird von :func:`detect_natural_performance` zurückgegeben.
    Nicht-blocking: Alle Zonen können leer sein wenn Detektion scheitert.
    """

    breath_zones: list[BreathZone] = field(default_factory=list)
    vibrato_zones: list[VibratoZone] = field(default_factory=list)
    early_reflection_zones: list[EarlyReflectionZone] = field(default_factory=list)

    # Meta
    detection_succeeded: bool = True
    error_message: str = ""

    def has_breath_at_sample(self, sample: int) -> bool:
        """Gibt True zurück wenn Sample in einer Atem-Schutzzone liegt."""
        return any(z.start_sample <= sample <= z.end_sample for z in self.breath_zones)

    def has_vibrato_at_sample(self, sample: int) -> bool:
        """Gibt True zurück wenn Sample in einer Vibrato-Schutzzone liegt."""
        return any(z.start_sample <= sample <= z.end_sample for z in self.vibrato_zones)

    def get_breath_mask(self, n_samples: int) -> npt.NDArray[np.bool_]:
        """
        Gibt eine Boolean-Maske (n_samples,) zurück — True = Atemschutzzone.

        Verwendung: NR-Bypass für Frames, in denen Atemgeräusche erkannt wurden.
        """
        mask = np.zeros(n_samples, dtype=np.bool_)
        for zone in self.breath_zones:
            s = max(0, zone.start_sample)
            e = min(n_samples, zone.end_sample + 1)
            mask[s:e] = True
        return mask

    def get_vibrato_mask(self, n_samples: int) -> npt.NDArray[np.bool_]:
        """
        Boolean-Maske (n_samples,) — True = Vibrato-Schutzzone.

        Verwendung: Pitch-Phasen überspringen Frames mit aktiver Vibrato-Detektion.
        """
        mask = np.zeros(n_samples, dtype=np.bool_)
        for zone in self.vibrato_zones:
            s = max(0, zone.start_sample)
            e = min(n_samples, zone.end_sample + 1)
            mask[s:e] = True
        return mask

    def __repr__(self) -> str:
        return (
            "NaturalPerformanceProfile("
            f"breaths={len(self.breath_zones)}, "
            f"vibratos={len(self.vibrato_zones)}, "
            f"early_refs={len(self.early_reflection_zones)}, "
            f"ok={self.detection_succeeded})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-API
# ─────────────────────────────────────────────────────────────────────────────
def detect_natural_performance(
    audio: npt.NDArray[np.float32 | np.float64],
    sr: int,
    panns_singing_confidence: float = 0.0,
) -> NaturalPerformanceProfile:
    """
    Erkenne natürliche Performance-Klangereignisse (§2.46f).

    Non-blocking: Jede interne Exception erzeugt ein leeres Profil mit
    ``detection_succeeded=False`` — nie einen Crash.

    Args:
        audio:                   Mono oder Stereo (2, N) / (N, 2) float.
        sr:                      Abtastrate in Hz.
        panns_singing_confidence: PANNs-Gesangswahrscheinlichkeit [0, 1].
                                  ≥ 0.25 → Vibrato-Scan wird aktiviert.

    Returns:
        :class:`NaturalPerformanceProfile` mit allen erkannten Schutzzonen.
    """
    try:
        return _detect_internal(audio, sr, panns_singing_confidence)
    except Exception as exc:
        logger.warning("§G23 natural_performance_detector: DSP-Ersatzpfad — %s", exc, exc_info=True)
        return NaturalPerformanceProfile(detection_succeeded=False, error_message=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Interne Implementierung
# ─────────────────────────────────────────────────────────────────────────────
def _to_mono(audio: npt.NDArray[Any], sr: int) -> npt.NDArray[np.float32]:  # pylint: disable=unused-argument
    """Downmix auf Mono, float32. Kein Resampling."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        # Detect channel dimension: channels ≤ 2 in first axis
        if a.shape[0] <= 2 and a.shape[0] < a.shape[1]:
            out: npt.NDArray[np.float32] = np.asarray(a.mean(axis=0), dtype=np.float32)
            return out
        out = np.asarray(a.mean(axis=1), dtype=np.float32)
        return out
    raise ValueError(f"Unsupported audio shape: {a.shape}")


def _rms_dbfs(segment: npt.NDArray[np.float32]) -> float:
    """Berechnet RMS level in dBFS. Returns -∞ for silent segments."""
    rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    if rms < 1e-12:
        return -120.0
    return float(20.0 * np.log10(rms + 1e-12))


def _spectral_flatness_segment(
    segment: npt.NDArray[np.float32],
    n_fft: int = 512,
) -> float:
    """Geometric-to-arithmetic mean ratio of power spectrum (Wiener entropy)."""
    if len(segment) < n_fft:
        n_fft = max(32, len(segment))
    win = np.hanning(n_fft).astype(np.float32)
    seg = segment[:n_fft] * win
    mag = np.abs(np.fft.rfft(seg.astype(np.float64))) ** 2
    mag = mag[1:]  # skip DC
    if len(mag) < 2:
        return 0.5
    log_mean = float(np.mean(np.log(mag + 1e-12)))
    mean_log = float(np.log(np.mean(mag) + 1e-12))
    flatness = float(np.exp(log_mean - mean_log))
    return float(np.clip(flatness, 0.0, 1.0))


def _detect_breath_zones(
    mono: npt.NDArray[np.float32],
    sr: int,
) -> list[BreathZone]:
    """
    Erkennt Atemgeräusche anhand von Energie, Dauer und Spectral Flatness.

    Atemgeräusche liegen zwischen Phrasen:
      - Energie: −55 dBFS bis −40 dBFS (kein Stille, kein Musiksignal)
      - Dauer: 50–500 ms
      - Spectral flatness > 0.40 (rauschähnlich, kein Ton)
    """
    hop = max(1, sr // 100)  # 10 ms Hop
    frame_len = max(16, sr // 50)  # 20 ms Frame
    min_samples = int(_BREATH_MIN_DUR_S * sr)
    max_samples = int(_BREATH_MAX_DUR_S * sr)

    zones: list[BreathZone] = []
    n = len(mono)

    i = 0
    while i + frame_len < n:
        seg = mono[i : i + frame_len]
        dbfs = _rms_dbfs(seg)

        if _BREATH_ENERGY_MIN_DBFS <= dbfs <= _BREATH_ENERGY_MAX_DBFS:
            # Kandidat gefunden — ausdehnen solange Bedingungen erfüllt
            j = i + frame_len
            while j + frame_len < n:
                seg_ext = mono[j : j + frame_len]
                dbfs_ext = _rms_dbfs(seg_ext)
                if not _BREATH_ENERGY_MIN_DBFS <= dbfs_ext <= _BREATH_ENERGY_MAX_DBFS:
                    break
                j += hop

            zone_len = j - i
            if min_samples <= zone_len <= max_samples:
                # Flatness-Check auf dem mittleren Abschnitt
                mid_start = i + zone_len // 4
                mid_end = i + 3 * zone_len // 4
                mid_seg = mono[mid_start:mid_end]
                flatness = _spectral_flatness_segment(mid_seg)
                if flatness >= _BREATH_FLATNESS_MIN:
                    mean_dbfs = _rms_dbfs(mono[i:j])
                    zones.append(
                        BreathZone(
                            start_sample=i,
                            end_sample=min(j - 1, n - 1),
                            energy_dbfs=mean_dbfs,
                            flatness=flatness,
                        )
                    )
            i = j  # weiter nach dem Kandidaten
        else:
            i += hop

    logger.debug("§2.46f breath_zones: %d erkannt (audio %.1fs)", len(zones), len(mono) / sr)
    return zones


def _detect_vibrato_zones(
    mono: npt.NDArray[np.float32],
    sr: int,
) -> list[VibratoZone]:
    """
    Erkennt Vibrato- und Portamento-Abschnitte via F0-Modulations-Analyse.

    Methode: PYIN (via librosa) Framewise F0 → Enveloppe der
    Kurzzeit-F0-Varianz → Passagen mit periodischer Modulation 4–7 Hz markieren.

    Non-blocking: Falls PYIN fehlschlägt → leere Liste.
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        # PYIN für F0-Schätzung (librosa ≥ 0.10)
        hop_length = 512
        f0, voiced_flag, _ = librosa.pyin(
            mono,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
            hop_length=hop_length,
            fill_na=None,
        )
    except Exception as exc:
        logger.debug("§2.46f vibrato_zones: pyin fehlgeschlagen (%s) — no protection", exc)
        return []

    if f0 is None or len(f0) < 10:
        return []

    frame_dur_s = hop_length / float(sr)
    hop_t = frame_dur_s

    zones: list[VibratoZone] = []
    n_frames = len(f0)

    # Schiebefen für Vibrato-Modul-Analyse: 0.5 s
    window_frames = max(8, int(0.5 / hop_t))
    step_frames = max(1, window_frames // 4)

    for start_f in range(0, n_frames - window_frames, step_frames):
        end_f = start_f + window_frames
        window_f0 = f0[start_f:end_f]
        voiced_w = voiced_flag[start_f:end_f] if voiced_flag is not None else np.ones(window_frames, dtype=np.bool_)

        # Nur Voiced-Frames
        valid = voiced_w & np.isfinite(window_f0)
        if np.sum(valid) < window_frames // 2:
            continue

        f0_valid = window_f0[valid].astype(np.float64)
        if len(f0_valid) < 4:
            continue

        # Modulationstiefe in Cent
        median_f0 = float(np.median(f0_valid))
        if median_f0 < 10.0:
            continue
        cents_deviation = 1200.0 * np.log2(f0_valid / median_f0 + 1e-12)
        depth_cent = float(np.std(cents_deviation))

        if depth_cent > _VIBRATO_DEPTH_MAX_CENT:
            # Zu viel Abweichung — kein sauberes Vibrato (Portamento oder Note-Sprung)
            continue
        if depth_cent < 2.0:
            # Zu wenig — stabile Pitch-Haltung (kein Vibrato)
            continue

        # Modulationsrate via Autocorrelation der Cent-Kurve
        cents_full = np.zeros(len(window_f0), dtype=np.float64)
        cents_full[valid] = cents_deviation
        if _sp_correlate is not None:
            corr_full = _sp_correlate(cents_full - cents_full.mean(), cents_full - cents_full.mean(), mode="full")
        else:
            # FFT-based autocorrelation O(N log N) — scipy unavailable fallback
            _x = cents_full - cents_full.mean()
            _n2 = 1
            while _n2 < 2 * len(_x):
                _n2 <<= 1
            _X = np.fft.rfft(_x, n=_n2)
            corr_full = np.fft.irfft(_X * np.conj(_X))[: len(_x)]
            corr_full = np.concatenate([corr_full[::-1][:-1], corr_full])
        corr = corr_full[len(corr_full) // 2 :]
        if len(corr) < 3:
            continue
        corr = corr / (corr[0] + 1e-12)

        # Suche erstes Lag-Maximum jenseits von 0 (Vibrato-Periode)
        lag_min = max(1, int(_VIBRATO_RATE_MIN_HZ * window_frames * hop_t))  # lag for max rate
        lag_max = max(lag_min + 1, int((1.0 / _VIBRATO_RATE_MIN_HZ) / hop_t))
        lag_min_hi = max(1, int((1.0 / _VIBRATO_RATE_MAX_HZ) / hop_t))
        lag_min_hi = min(lag_min_hi, lag_max)

        corr_window = corr[lag_min_hi : lag_max + 1] if lag_max < len(corr) else corr[lag_min_hi:]
        if len(corr_window) < 2:
            continue

        peak_lag = int(np.argmax(corr_window)) + lag_min_hi
        if peak_lag < 1:
            continue
        rate_hz = 1.0 / (peak_lag * hop_t + 1e-12)

        if _VIBRATO_RATE_MIN_HZ <= rate_hz <= _VIBRATO_RATE_MAX_HZ:
            start_sample = int(start_f * hop_length)
            end_sample = min(int(end_f * hop_length), len(mono) - 1)
            zones.append(
                VibratoZone(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    rate_hz=float(rate_hz),
                    depth_cent=float(depth_cent),
                )
            )

    # Merge überlappende Zonen
    zones = _merge_overlapping_zones_vibrato(zones)
    logger.debug("§2.46f vibrato_zones: %d erkannt", len(zones))
    return zones


def _merge_overlapping_zones_vibrato(zones: list[VibratoZone]) -> list[VibratoZone]:
    """Führt zusammen: adjacent/overlapping VibratoZones."""
    if not zones:
        return zones
    sorted_zones = sorted(zones, key=lambda z: z.start_sample)
    merged: list[VibratoZone] = [sorted_zones[0]]
    for z in sorted_zones[1:]:
        prev = merged[-1]
        if z.start_sample <= prev.end_sample + 1:
            # Overlap — merge
            merged[-1] = VibratoZone(
                start_sample=prev.start_sample,
                end_sample=max(prev.end_sample, z.end_sample),
                rate_hz=(prev.rate_hz + z.rate_hz) / 2.0,
                depth_cent=max(prev.depth_cent, z.depth_cent),
            )
        else:
            merged.append(z)
    return merged


def _detect_early_reflection_zones(
    mono: npt.NDArray[np.float32],
    sr: int,
) -> list[EarlyReflectionZone]:
    """
    Markiert die ersten 50 ms nach jedem signifikanten Onset als Early-Reflection-Zone.

    Early Reflections (0–50 ms nach Onset) definieren den Studio-Raumcharakter
    des Originalaufnahme-Raums und dürfen NICHT durch Dereverb entfernt werden.
    Der Dereverb-wet_mix-Cap (§2.46f: cap = 0.35 wenn C80 > 3 dB) wird extern gesetzt.
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        hop_length = 512
        onset_frames = librosa.onset.onset_detect(  # type: ignore[attr-defined]
            y=mono,
            sr=sr,
            hop_length=hop_length,
            units="samples",
        )
    except Exception as exc:
        logger.debug("§2.46f early_reflections: onset_erkennen fehlgeschlagen (%s)", exc)
        return []

    early_dur_samples = int(_EARLY_REF_MAX_MS / 1000.0 * sr)
    zones: list[EarlyReflectionZone] = []
    n = len(mono)

    for onset_s in onset_frames:
        onset_s = int(onset_s)
        end_s = min(onset_s + early_dur_samples, n - 1)
        zones.append(EarlyReflectionZone(onset_sample=onset_s, end_sample=end_s))

    logger.debug("§2.46f early_reflection_zones: %d onset-anchored", len(zones))
    return zones


def _detect_internal(
    audio: npt.NDArray[Any],
    sr: int,
    panns_singing_confidence: float,
) -> NaturalPerformanceProfile:
    """Interne Implementierung (non-blocking Wrapper in :func:`detect_natural_performance`)."""
    mono = _to_mono(audio, sr)

    if len(mono) / float(sr) < _MIN_AUDIO_S:
        logger.debug("§2.46f: audio too short (%.3fs) — empty Profil", len(mono) / float(sr))
        return NaturalPerformanceProfile()

    # 1. Atemgeräusche (immer aktiv — auch ohne Gesangs-Konfidenz)
    breath_zones = _detect_breath_zones(mono, sr)

    # 2. Vibrato/Portamento (nur bei erkanntem Gesang)
    vibrato_zones: list[VibratoZone] = []
    if panns_singing_confidence >= 0.25:
        vibrato_zones = _detect_vibrato_zones(mono, sr)

    # 3. Early Reflections (immer aktiv — Studio-Raumcharakter universell)
    early_reflection_zones = _detect_early_reflection_zones(mono, sr)

    return NaturalPerformanceProfile(
        breath_zones=breath_zones,
        vibrato_zones=vibrato_zones,
        early_reflection_zones=early_reflection_zones,
        detection_succeeded=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience-Singleton-Accessor
# ─────────────────────────────────────────────────────────────────────────────
def detect_breath_zones(audio: npt.NDArray[Any], sr: int) -> list[BreathZone]:
    """Direkt-Accessor für Atem-Schutzzonen (non-blocking)."""
    try:
        return _detect_breath_zones(_to_mono(audio, sr), sr)
    except Exception as exc:
        logger.debug("§2.46f erkennen_breath_zones fehlgeschlagen: %s", exc)
        return []


def detect_vibrato_zones(
    audio: npt.NDArray[Any],
    sr: int,
    panns_singing_confidence: float = 0.5,
) -> list[VibratoZone]:
    """Direkt-Accessor für Vibrato-Schutzzonen (non-blocking, nur bei Gesang)."""
    if panns_singing_confidence < 0.25:
        return []
    try:
        return _detect_vibrato_zones(_to_mono(audio, sr), sr)
    except Exception as exc:
        logger.debug("§2.46f erkennen_vibrato_zones fehlgeschlagen: %s", exc)
        return []
