"""Long-form Musical Coherence Guard (v10.0.0).

Stellt sicher, dass strukturell identische Song-Abschnitte (Vers-1 ↔ Vers-2,
Refrain-1 ↔ Refrain-2) nach der Restaurierung einen konsistenten Klang aufweisen.

Problem ohne diesen Guard:
    NR/DSP-Phasen arbeiten adaptiv. Wenn Refrain-1 und Refrain-2 unterschiedliche
    Signal-Rausch-Verhältnisse haben (z.B. wegen Trägerverschleiß), greifen die
    Phasen dort unterschiedlich stark ein — Ergebnis: Refrain-2 klingt anders als
    Refrain-1, obwohl beide strukturell identisch sind.

Algorithmus:
    1. Segmentierung: Song in ~15-Sekunden-Blöcke aufteilen
    2. Fingerprinting: MFCC (13 Koeffizienten, Mittelwert) pro Block
    3. Gruppenbildung: Cosinus-Ähnlichkeit ≥ 0.92 → gleiche strukturelle Sektion
    4. Konsistenzprüfung: Mittlere Oktav-Spektralfarbe pro Gruppe vergleichen
    5. Korrektur: Sanfte Oktav-EQ-Angleichung (max ±2 dB per Band, non-blocking)
    6. Blend: Korrigierte Segmente werden 40/60 in Restored-Audio zurückgeblendet

Kanonische Nutzung (UV3 post-pipeline, vor HPG):
    from backend.core.musical_coherence_guard import check_musical_coherence
    restored_audio, mcg_report = check_musical_coherence(
        original_audio, restored_audio, sr
    )
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Mindest-Cosinus-Ähnlichkeit für "gleiche strukturelle Sektion"
_FINGERPRINT_SIMILARITY_THRESHOLD: float = 0.92
# Maximale spektrale Korrektur pro Oktavband (dB)
_MAX_CORRECTION_DB: float = 2.0
# Blend-Gewicht für Korrektur (0.4 = 40 % korrigiert, 60 % original)
_CORRECTION_BLEND: float = 0.40
# Segment-Dauer in Sekunden
_SEGMENT_DURATION_S: float = 15.0
# Mindest-Segmentanzahl für sinnvolle Gruppen-Erkennung
_MIN_SEGMENTS: int = 2
# Mindest-Gruppengrüße (mind. 2 Segmente = 1 Wiederholung)
_MIN_GROUP_SIZE: int = 2
# Anzahl der Mel-Koeffizienten für Fingerprinting
_N_MFCC: int = 13
# Anzahl der Oktavbänder für Konsistenzprüfung
_N_OCTAVE_BANDS: int = 8


@dataclass
class MusicalCoherenceReport:
    """Ergebnis des Musical Coherence Guards."""

    groups_found: int = 0
    corrections_applied: int = 0
    max_spectral_deviation_db: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = False


def _mfcc_fingerprint(audio_1ch: np.ndarray, sr: int, n_mfcc: int = _N_MFCC) -> np.ndarray:
    """Einfacher MFCC-Fingerabdruck (ohne librosa — nur numpy/scipy).

    Verwendet eine vereinfachte Mel-Filterbank-Annäherung für Geschwindigkeit.
    Für Fingerprinting (nicht Analyse) ist die Genauigkeit ausreichend.
    """
    n = len(audio_1ch)
    if n < 512:
        return np.zeros(n_mfcc, dtype=np.float32)  # type: ignore[no-any-return]

    # Kurze STFT (fensterbasiert, ~50 ms Frames)
    frame_len = min(2048, max(512, n // 64))
    hop_len = frame_len // 2
    n_frames = max(1, (n - frame_len) // hop_len)

    # Hanning-Fenster
    window = np.hanning(frame_len).astype(np.float32)
    power_spectrum = np.zeros(frame_len // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        seg = audio_1ch[start : start + frame_len]
        if len(seg) < frame_len:
            seg = np.pad(seg, (0, frame_len - len(seg)))
        spec = np.abs(np.fft.rfft(seg * window)) ** 2
        power_spectrum += spec
    power_spectrum /= max(1, n_frames)

    # Log-Mel-Annäherung: 13 logarithmisch verteilte Bänder
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr)
    f_min, f_max = 80.0, min(8000.0, sr / 2.0)
    mel_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_mfcc + 1)
    mel_features = np.zeros(n_mfcc, dtype=np.float32)
    for k in range(n_mfcc):
        lo, hi = mel_edges[k], mel_edges[k + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            mel_features[k] = float(np.log1p(np.mean(power_spectrum[mask])))

    # Normieren auf Einheitsnorm für Cosinus-Ähnlichkeit
    norm = np.linalg.norm(mel_features) + 1e-10
    return (mel_features / norm).astype(np.float32)  # type: ignore[no-any-return]


def _octave_band_spectrum(audio_1ch: np.ndarray, sr: int, n_bands: int = _N_OCTAVE_BANDS) -> np.ndarray:
    """Mittleres Leistungsspektrum in logarithmisch verteilten Oktavbändern (dBFS).

    Args:
        audio_1ch: 1-Kanal-Audio.
        sr: Sample-Rate.
        n_bands: Anzahl der Oktavbänder.

    Returns:
        Mittlere Leistung pro Band in dBFS (float32, Länge = n_bands).
    """
    n = len(audio_1ch)
    if n < 256:
        return np.zeros(n_bands, dtype=np.float32)  # type: ignore[no-any-return]
    fft = np.abs(np.fft.rfft(audio_1ch.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    f_min, f_max = 80.0, min(20000.0, sr / 2.0)
    band_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bands + 1)
    band_power = np.zeros(n_bands, dtype=np.float32)
    for k in range(n_bands):
        mask = (freqs >= band_edges[k]) & (freqs < band_edges[k + 1])
        if np.any(mask):
            band_power[k] = float(np.mean(fft[mask]) + 1e-20)
    # In dBFS umrechnen
    band_db: np.ndarray = 10.0 * np.log10(np.maximum(band_power, 1e-20)).astype(np.float32)
    return band_db


def _apply_octave_correction(
    audio_1ch: np.ndarray,
    sr: int,
    correction_db: np.ndarray,
) -> np.ndarray:
    """Wendet eine sanfte Oktav-EQ-Korrektur auf audio_1ch an.

    Die Korrektur wird im Frequenzbereich angewendet (lineares Gain per Band).
    Übergänge zwischen Bändern werden interpoliert.

    Args:
        audio_1ch: 1-Kanal-Audio (float32).
        sr: Sample-Rate.
        correction_db: Korrektur pro Band in dB (float32, Länge = _N_OCTAVE_BANDS).

    Returns:
        Korrigiertes Audio (float32).
    """
    n = len(audio_1ch)
    if n < 256:
        return audio_1ch

    # Klemme Korrektur auf ±_MAX_CORRECTION_DB
    correction_db_clamped = np.clip(correction_db, -_MAX_CORRECTION_DB, _MAX_CORRECTION_DB)

    # Gain-Kurve im Frequenzbereich aufbauen (interpoliert)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    f_min, f_max = 80.0, min(20000.0, sr / 2.0)
    band_edges = np.logspace(np.log10(f_min), np.log10(f_max), len(correction_db_clamped) + 1)
    band_centers = np.sqrt(band_edges[:-1] * band_edges[1:])

    # Interpolation der Korrektur-dB auf Frequenzachse
    gain_db_curve = np.zeros(len(freqs), dtype=np.float32)
    log_freqs = np.log10(np.maximum(freqs, 1.0))
    log_centers = np.log10(band_centers)
    gain_db_curve = np.interp(log_freqs, log_centers, correction_db_clamped).astype(np.float32)

    # Sub- und Superhochfrequenzen: keine Korrektur (Edges fade to 0)
    gain_db_curve[freqs < f_min] = 0.0
    gain_db_curve[freqs > f_max] = 0.0

    # Anwenden im Frequenzbereich
    spectrum = np.fft.rfft(audio_1ch.astype(np.float64))
    gain_linear = 10.0 ** (gain_db_curve / 20.0)
    spectrum *= gain_linear.astype(np.complex128)
    corrected = np.fft.irfft(spectrum, n=n).real.astype(np.float32)
    return corrected  # type: ignore[no-any-return]


def check_musical_coherence(
    restored_audio: np.ndarray,
    sr: int,
    *,
    panns_singing: float = 0.0,
) -> tuple[np.ndarray, MusicalCoherenceReport]:
    """Prüft und korrigiert Long-form Kohärenz über strukturell identische Sektionen.

    Nicht-blockierend: Bei Fehler wird ``restored_audio`` unverändert zurückgegeben.

    Args:
        restored_audio: Restauriertes Audio. Shape [N] oder [2, N].
        sr: Sample-Rate (muss 48000 sein).
        panns_singing: PANNs-Singing-Score (0–1). Wenn ≥ 0.35, ist der Guard
            konservativer (kleinere Korrektur, um Stimmfarbe nicht zu verfremden).

    Returns:
        Tuple (korrigiertes_audio, MusicalCoherenceReport).
    """
    report = MusicalCoherenceReport()
    try:
        assert sr == 48000
        restored_audio = np.nan_to_num(restored_audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        is_stereo = restored_audio.ndim == 2

        # Mono-Mix für Fingerprinting (nur Kanal 0 / Downmix)
        if is_stereo:
            fingerprint_audio = restored_audio[0].astype(np.float32)
        else:
            fingerprint_audio = restored_audio.astype(np.float32)

        n_samples = len(fingerprint_audio)
        seg_len = int(_SEGMENT_DURATION_S * sr)
        if seg_len < 1:
            return restored_audio, report
        n_segments = n_samples // seg_len
        if n_segments < _MIN_SEGMENTS:
            return restored_audio, report

        # Fingerprinting aller Segmente
        fingerprints = []
        for i in range(n_segments):
            seg = fingerprint_audio[i * seg_len : (i + 1) * seg_len]
            fp = _mfcc_fingerprint(seg, sr)
            fingerprints.append(fp)
        fps = np.array(fingerprints, dtype=np.float32)  # [n_segments, n_mfcc]

        # Cosinus-Ähnlichkeits-Matrix
        similarity = fps @ fps.T  # Einheitsnorm → dot product = cosinus sim

        # Gruppen bilden (einfaches greedy clustering)
        used = np.zeros(n_segments, dtype=bool)
        groups: list[list[int]] = []
        for i in range(n_segments):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, n_segments):
                if not used[j] and float(similarity[i, j]) >= _FINGERPRINT_SIMILARITY_THRESHOLD:
                    group.append(j)
                    used[j] = True
            if len(group) >= _MIN_GROUP_SIZE:
                groups.append(group)

        if not groups:
            return restored_audio, report

        report.is_active = True
        report.groups_found = len(groups)

        # Konservativerer Blend bei Gesang
        blend_weight = _CORRECTION_BLEND * (0.6 if panns_singing >= 0.35 else 1.0)

        result = restored_audio.copy()
        max_dev_db = 0.0

        for group in groups:
            # Oktav-Spektren aller Gruppe-Segmente ermitteln
            spectra = []
            for seg_idx in group:
                start = seg_idx * seg_len
                seg_ch = fingerprint_audio[start : start + seg_len]
                spectra.append(_octave_band_spectrum(seg_ch, sr))
            spectra_arr = np.array(spectra, dtype=np.float32)  # [n_seg, n_bands]

            # Ziel-Spektrum = Mittelwert aller Segment-Spektren der Gruppe
            mean_spectrum = np.mean(spectra_arr, axis=0)

            for k, seg_idx in enumerate(group):
                deviation_db = mean_spectrum - spectra_arr[k]
                max_dev = float(np.max(np.abs(deviation_db)))
                if max_dev > max_dev_db:
                    max_dev_db = max_dev

                # Nur korrigieren wenn Abweichung > 0.5 dB (spürbar)
                if max_dev < 0.5:
                    continue

                correction_db = deviation_db  # Richtung: segment → Gruppen-Mittelwert
                start = seg_idx * seg_len
                end = start + seg_len

                if is_stereo:
                    for ch in range(result.shape[0]):
                        seg_ch = result[ch, start:end]
                        corrected = _apply_octave_correction(seg_ch, sr, correction_db)
                        result[ch, start:end] = ((1.0 - blend_weight) * seg_ch + blend_weight * corrected).astype(
                            np.float32
                        )
                else:
                    seg_ch = result[start:end]
                    corrected = _apply_octave_correction(seg_ch, sr, correction_db)
                    result[start:end] = ((1.0 - blend_weight) * seg_ch + blend_weight * corrected).astype(np.float32)

                report.corrections_applied += 1

        report.max_spectral_deviation_db = round(max_dev_db, 2)
        report.metadata = {
            "groups_found": report.groups_found,
            "corrections_applied": report.corrections_applied,
            "max_spectral_deviation_db": report.max_spectral_deviation_db,
            "n_segments": n_segments,
            "similarity_threshold": _FINGERPRINT_SIMILARITY_THRESHOLD,
            "blend_weight": round(blend_weight, 3),
        }

        logger.debug(
            "MusicalCoherenceGuard: %d Gruppen, %d Korrekturen, max_dev=%.2f dB",
            report.groups_found,
            report.corrections_applied,
            report.max_spectral_deviation_db,
        )

        result = np.clip(result, -1.0, 1.0).astype(np.float32)
        return result, report

    except Exception as exc:
        logger.debug("MusicalCoherenceGuard (nicht blockierend): %s", exc)
        return restored_audio, report


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: MusicalCoherenceGuardSingleton | None = None
_lock = threading.Lock()


class MusicalCoherenceGuardSingleton:
    """Thread-sicherer Singleton-Wrapper."""

    def analyze_and_correct(
        self,
        restored_audio: np.ndarray,
        sr: int,
        panns_singing: float = 0.0,
    ) -> tuple[np.ndarray, MusicalCoherenceReport]:
        """Delegiert an :func:`check_musical_coherence`."""
        return check_musical_coherence(restored_audio, sr, panns_singing=panns_singing)


def get_musical_coherence_guard() -> MusicalCoherenceGuardSingleton:
    """Gibt den globalen MusicalCoherenceGuard-Singleton zurück (thread-safe)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MusicalCoherenceGuardSingleton()
    return _instance
