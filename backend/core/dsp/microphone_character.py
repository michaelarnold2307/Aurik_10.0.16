"""
Microphone Character Detector — Lücke 6 (v10.0.0.x)
==================================================

Erkennt und schützt mikrofonspezifische spektrale Charakteristika
in historischen Vokalaufnahmen. Ergänzt RecordingProductionKB um eine
neue Dimension: den Mikrofon-Typ als Authentizitätsmarker.

MIKROFON-PROFILE (15 historische Typen, Mai 2026):

    Kondensatormikrofone (Großmembran):
      - Neumann U47 (1947–1966): Proximity-Effekt +6 dB/Okt <300 Hz,
        Präsenzanhebung +4 dB bei 8–10 kHz, Roll-off >14 kHz
      - Neumann U67 (1960er): ähnlich U47, weicherer Präsenzbereich
      - AKG C12 (1953+): breitere Präsenz 5–12 kHz, heller Charakter
      - Telefunken ELA M 250 (1957+): voller, wärmerer Unterton

    Bändchenmikrofone:
      - RCA 44-BX (1930er–1950er): sanfter Roll-off >8 kHz,
        voller "Radio-Klang", Bidirektional
      - RCA 77-DX (1954+): ähnlich, etwas heller
      - Coles 4038 (1954+): UK-Rundfunk-Klang, 15 kHz Roll-off

    Kristall-/Keramik-Mikrofone:
      - "Crystal" (1930er–1940er): starke Resonanz 2–5 kHz,
        starker Roll-off <200 Hz und >10 kHz

    Dynamische Mikrofone:
      - Shure SM7B (1973+): Mittenbetonung 2–5 kHz, Präsenzanhebung
      - EV RE20 (1968+): Flat, kein Proximity-Effekt

    Kohlenmikrofone (Telefon-Ära):
      - Carbon (vor 1940): BW 300–3400 Hz, starkes Eigen-Rauschen

ALGORITHMUS:
    1. Spektrale Hüllkurve des Signals (gemittelt über stimmhafte Frames)
    2. Vergleich mit Mikrofon-Fingerprints über normiertes Spektrum
    3. Proximity-Effekt-Detektion: log-lineare Energiezunahme < 300 Hz
    4. Präsenz-Peak-Detektion: Lokales Maximum im 4–12 kHz Band
    5. High-Frequency-Roll-off-Schätzung: −3 dB Grenzfrequenz
    6. Pattern-Matching gegen Fingerprint-Bibliothek → Nearest-Neighbour

Ausgabe: MicrophoneSignature + MicrophoneProtectionEQ für phase_04.

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0.x — Lücke 6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mikrofon-Fingerprint-Bibliothek
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrophoneFingerprint:
    """Spektraler Fingerprint eines historischen Mikrofons."""

    name: str
    era_range: tuple[int, int]  # Produktionsjahre (start, end)
    proximity_slope_db_oct: float  # Tiefton-Anhebung durch Proximity (dB/Okt)
    presence_peak_hz: float  # Zentrum Präsenz-Anhebung (Hz), 0 = keine
    presence_peak_db: float  # Stärke der Präsenz-Anhebung (dB)
    rolloff_3db_hz: float  # Obere −3 dB Grenzfrequenz (Hz)
    bass_rolloff_hz: float  # Untere −3 dB Grenzfrequenz (Hz)
    character: str  # "warm", "bright", "neutral", "limited_bw"
    protection_priority: str  # "strict", "standard", "relaxed"


_MIC_LIBRARY: list[MicrophoneFingerprint] = [
    # Großmembran-Kondensatoren
    MicrophoneFingerprint(
        "Neumann U47",
        (1947, 1966),
        proximity_slope_db_oct=4.5,
        presence_peak_hz=9000.0,
        presence_peak_db=3.5,
        rolloff_3db_hz=14000.0,
        bass_rolloff_hz=40.0,
        character="warm",
        protection_priority="strict",
    ),
    MicrophoneFingerprint(
        "Neumann U67",
        (1960, 1975),
        proximity_slope_db_oct=3.5,
        presence_peak_hz=10000.0,
        presence_peak_db=2.5,
        rolloff_3db_hz=16000.0,
        bass_rolloff_hz=35.0,
        character="warm",
        protection_priority="strict",
    ),
    MicrophoneFingerprint(
        "AKG C12",
        (1953, 1970),
        proximity_slope_db_oct=3.0,
        presence_peak_hz=8000.0,
        presence_peak_db=4.5,
        rolloff_3db_hz=18000.0,
        bass_rolloff_hz=30.0,
        character="bright",
        protection_priority="strict",
    ),
    MicrophoneFingerprint(
        "Telefunken ELA M 250",
        (1957, 1972),
        proximity_slope_db_oct=5.0,
        presence_peak_hz=7000.0,
        presence_peak_db=3.0,
        rolloff_3db_hz=13000.0,
        bass_rolloff_hz=50.0,
        character="warm",
        protection_priority="strict",
    ),
    # Bändchenmikrofone
    MicrophoneFingerprint(
        "RCA 44-BX",
        (1931, 1960),
        proximity_slope_db_oct=6.0,
        presence_peak_hz=0.0,
        presence_peak_db=0.0,
        rolloff_3db_hz=8000.0,
        bass_rolloff_hz=60.0,
        character="warm",
        protection_priority="strict",
    ),
    MicrophoneFingerprint(
        "RCA 77-DX",
        (1954, 1975),
        proximity_slope_db_oct=5.0,
        presence_peak_hz=6000.0,
        presence_peak_db=1.5,
        rolloff_3db_hz=10000.0,
        bass_rolloff_hz=50.0,
        character="warm",
        protection_priority="standard",
    ),
    MicrophoneFingerprint(
        "Coles 4038",
        (1954, 1990),
        proximity_slope_db_oct=4.0,
        presence_peak_hz=0.0,
        presence_peak_db=0.0,
        rolloff_3db_hz=15000.0,
        bass_rolloff_hz=40.0,
        character="neutral",
        protection_priority="standard",
    ),
    # Kristall-/Keramik-Mikrofone
    MicrophoneFingerprint(
        "Crystal Mic",
        (1930, 1950),
        proximity_slope_db_oct=0.0,
        presence_peak_hz=3500.0,
        presence_peak_db=6.0,
        rolloff_3db_hz=8000.0,
        bass_rolloff_hz=300.0,
        character="limited_bw",
        protection_priority="relaxed",
    ),
    # Dynamische Mikrofone
    MicrophoneFingerprint(
        "Shure SM7B",
        (1973, 2030),
        proximity_slope_db_oct=2.0,
        presence_peak_hz=5000.0,
        presence_peak_db=3.0,
        rolloff_3db_hz=18000.0,
        bass_rolloff_hz=60.0,
        character="neutral",
        protection_priority="standard",
    ),
    MicrophoneFingerprint(
        "EV RE20",
        (1968, 2030),
        proximity_slope_db_oct=0.5,
        presence_peak_hz=3000.0,
        presence_peak_db=1.0,
        rolloff_3db_hz=20000.0,
        bass_rolloff_hz=45.0,
        character="neutral",
        protection_priority="relaxed",
    ),
    # Kohlenmikrofone (stark limitiert)
    MicrophoneFingerprint(
        "Carbon Mic",
        (1920, 1945),
        proximity_slope_db_oct=0.0,
        presence_peak_hz=1500.0,
        presence_peak_db=8.0,
        rolloff_3db_hz=3400.0,
        bass_rolloff_hz=300.0,
        character="limited_bw",
        protection_priority="relaxed",
    ),
    # Generisch: unbekanntes Vintage-Kondensatormikrofon
    MicrophoneFingerprint(
        "Vintage Condenser (generic)",
        (1940, 1970),
        proximity_slope_db_oct=3.0,
        presence_peak_hz=8000.0,
        presence_peak_db=2.0,
        rolloff_3db_hz=12000.0,
        bass_rolloff_hz=50.0,
        character="warm",
        protection_priority="standard",
    ),
]


# ---------------------------------------------------------------------------
# Gemessene Signatur
# ---------------------------------------------------------------------------


@dataclass
class MicrophoneSignature:
    """Aus dem Audio extrahierte Mikrofon-Charakteristik."""

    detected_mic: str = "unknown"
    protection_priority: str = "standard"  # "strict" / "standard" / "relaxed"
    proximity_slope_db_oct: float = 0.0
    presence_peak_hz: float = 0.0
    presence_peak_db: float = 0.0
    rolloff_3db_hz: float = 20000.0
    bass_rolloff_hz: float = 40.0
    character: str = "neutral"
    match_confidence: float = 0.0  # 0–1, Sicherheit des Matchings

    # Schutz-EQ-Parameter für phase_04
    protect_proximity_below_hz: float = 0.0  # Bass-Proximity-Bereich schützen
    protect_presence_band: tuple[float, float] = (0.0, 0.0)  # Präsenz-Band schützen
    protect_rolloff_above_hz: float = 20000.0  # Ab hier nichts hinzufügen

    def should_protect_bass(self) -> bool:
        return self.proximity_slope_db_oct > 2.0 and self.protect_proximity_below_hz > 0

    def has_detectable_presence(self) -> bool:
        return self.presence_peak_hz > 0 and self.presence_peak_db > 1.5


# ---------------------------------------------------------------------------
# DSP-Hilfsfunktionen
# ---------------------------------------------------------------------------


def _smooth_spectrum(spectrum_db: np.ndarray, octave_smooth: float = 1 / 3) -> np.ndarray:
    """1/3-Oktav-Glättung des Spektrums."""
    N = len(spectrum_db)
    smoothed = spectrum_db.copy()
    for i in range(N):
        # Fenster in Bins ≈ ±octave_smooth/2 Oktaven
        half_win = max(1, int(N * octave_smooth / 4))
        i0 = max(0, i - half_win)
        i1 = min(N, i + half_win + 1)
        smoothed[i] = float(np.mean(spectrum_db[i0:i1]))
    return smoothed


def _estimate_proximity_slope(spectrum_db: np.ndarray, freqs: np.ndarray) -> float:
    """
    Schätzt den Proximity-Effekt-Anstieg in dB/Oktave im Bass (<400 Hz).
    Positiver Wert = Bassanhebung = Proximity-Effekt vorhanden.
    """
    bass_mask = (freqs >= 60.0) & (freqs <= 400.0)
    if not np.any(bass_mask):
        return 0.0
    x = np.log2(freqs[bass_mask] + 1e-6)
    y = spectrum_db[bass_mask]
    if len(x) < 2:
        return 0.0
    slope = float(np.polyfit(x, y, 1)[0])
    return -slope  # Positiver Slope = Zunahme zu tiefen Frequenzen


def _estimate_presence_peak(spectrum_db: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    """
    Findet Präsenz-Peak im Band 3–14 kHz.
    Gibt (peak_hz, peak_db_über_Baseline) zurück.
    """
    pres_mask = (freqs >= 3000.0) & (freqs <= 14000.0)
    if not np.any(pres_mask):
        return 0.0, 0.0
    pres_spectrum = spectrum_db[pres_mask]
    pres_freqs = freqs[pres_mask]
    # Baseline: Medianwert des Präsenz-Bandes
    baseline = float(np.median(pres_spectrum))
    peak_idx = int(np.argmax(pres_spectrum))
    peak_db = pres_spectrum[peak_idx] - baseline
    peak_hz = float(pres_freqs[peak_idx])
    return peak_hz, float(peak_db)


def _estimate_rolloff_hz(spectrum_db: np.ndarray, freqs: np.ndarray, direction: str = "high") -> float:
    """
    Schätzt −3 dB Grenzfrequenz (oben oder unten).
    direction: "high" = obere Grenzfrequenz, "low" = untere Grenzfrequenz.
    """
    if len(spectrum_db) < 4:
        return 20000.0 if direction == "high" else 40.0

    if direction == "high":
        # Suche von oben die Frequenz, wo Pegel auf Spitzenpegel -3 dB fällt
        peak_db = float(np.max(spectrum_db))
        threshold = peak_db - 3.0
        for i in range(len(spectrum_db) - 1, -1, -1):
            if spectrum_db[i] >= threshold:
                return float(freqs[i])
        return float(freqs[-1])
    else:
        # Suche von unten
        peak_db = float(np.max(spectrum_db))
        threshold = peak_db - 3.0
        for i in range(len(spectrum_db)):
            if spectrum_db[i] >= threshold:
                return float(freqs[i])
        return float(freqs[0])


# ---------------------------------------------------------------------------
# Matching-Algorithmus
# ---------------------------------------------------------------------------


def _score_fingerprint(
    fp: MicrophoneFingerprint,
    measured_proximity: float,
    measured_presence_hz: float,
    measured_presence_db: float,
    measured_rolloff_hz: float,
    measured_bass_hz: float,
    era_decade: int | None,
) -> float:
    """Näherungs-Score eines Fingerprints zu den Messwerten (0–1)."""

    score = 0.0
    n_features = 0

    # Ära-Match (Bonus, nicht zwingend)
    if era_decade is not None:
        era_match = fp.era_range[0] <= era_decade <= fp.era_range[1] + 10
        score += 0.20 if era_match else 0.0
    n_features += 1

    # Proximity-Slope
    prox_diff = abs(fp.proximity_slope_db_oct - measured_proximity)
    score += max(0.0, 0.20 * (1.0 - prox_diff / 6.0))
    n_features += 1

    # Presence Peak
    if fp.presence_peak_hz > 0 and measured_presence_hz > 0:
        pres_hz_diff = abs(fp.presence_peak_hz - measured_presence_hz) / fp.presence_peak_hz
        pres_db_diff = abs(fp.presence_peak_db - measured_presence_db)
        score += max(0.0, 0.25 * (1.0 - pres_hz_diff) * (1.0 - pres_db_diff / 6.0))
    elif fp.presence_peak_hz == 0 and measured_presence_db < 1.5:
        score += 0.20  # Kein Präsenz-Peak → passt zu Bändchen
    n_features += 1

    # High-Frequency-Rolloff
    rolloff_diff = abs(fp.rolloff_3db_hz - measured_rolloff_hz) / (fp.rolloff_3db_hz + 1e-6)
    score += max(0.0, 0.20 * (1.0 - rolloff_diff))
    n_features += 1

    # Bass-Rolloff
    bass_diff = abs(fp.bass_rolloff_hz - measured_bass_hz) / (fp.bass_rolloff_hz + 100.0)
    score += max(0.0, 0.15 * (1.0 - bass_diff))
    n_features += 1

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def detect_microphone_character(
    audio: np.ndarray,
    sr: int = 48_000,
    era_decade: int | None = None,
) -> MicrophoneSignature:
    """
    Erkennt Mikrofon-Charakter aus dem Audio-Spektrum.

    Non-blocking: Exception → MicrophoneSignature() mit defaults.

    Args:
        audio:      Mono oder Stereo float32/64
        sr:         Abtastrate (48 000 Hz)
        era_decade: Ära für Plausibilitätsprüfung (z.B. 1955)

    Returns:
        MicrophoneSignature mit erkanntem Mikrofon und Schutz-EQ-Parametern.
    """
    _default = MicrophoneSignature()
    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Mono + Clip
        if audio.ndim == 2:
            mono = audio.mean(axis=0).astype(np.float64)
        else:
            mono = audio.astype(np.float64)

        if len(mono) < sr // 4:  # < 250 ms
            return _default

        # Langzeit-Spektrum (gemittelt, repräsentativ für den Aufnahmecharakter)
        # Nur stimmhafte Frames (Energie > -60 dBFS) mitteln
        n_fft = 4096
        hop = n_fft // 4
        n_frames = (len(mono) - n_fft) // hop
        if n_frames < 1:
            return _default

        spectra: list[np.ndarray] = []
        for i in range(n_frames):
            frame = mono[i * hop : i * hop + n_fft]
            frame_rms = float(np.sqrt(np.mean(frame**2) + 1e-30))
            if frame_rms < 1e-4:  # Stille überspringen
                continue
            window = np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
            spectra.append(spectrum)

        if len(spectra) < 3:
            return _default

        # Geometrischer Mittelwert (robust gegen Outlier)
        avg_spectrum = np.exp(np.mean(np.log(np.array(spectra) + 1e-30), axis=0))
        avg_spectrum_db = 10.0 * np.log10(avg_spectrum + 1e-30)

        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        # 1/3-Oktav-Glättung
        avg_spectrum_db_smooth = _smooth_spectrum(avg_spectrum_db)

        # Merkmals-Extraktion
        meas_prox = _estimate_proximity_slope(avg_spectrum_db_smooth, freqs)
        meas_pres_hz, meas_pres_db = _estimate_presence_peak(avg_spectrum_db_smooth, freqs)
        meas_rolloff = _estimate_rolloff_hz(avg_spectrum_db_smooth, freqs, "high")
        meas_bass = _estimate_rolloff_hz(avg_spectrum_db_smooth, freqs, "low")

        # Fingerprint-Matching
        best_fp: MicrophoneFingerprint | None = None
        best_score = 0.0
        for fp in _MIC_LIBRARY:
            s = _score_fingerprint(fp, meas_prox, meas_pres_hz, meas_pres_db, meas_rolloff, meas_bass, era_decade)
            if s > best_score:
                best_score = s
                best_fp = fp

        if best_fp is None or best_score < 0.25:
            logger.debug("mic_character: no confident match (best_Wert=%.2f) → unknown", best_score)
            return _default

        # Schutz-EQ-Parameter ableiten
        prot_pres_low = max(0.0, meas_pres_hz * 0.7) if meas_pres_hz > 0 else 0.0
        prot_pres_high = meas_pres_hz * 1.4 if meas_pres_hz > 0 else 0.0

        sig_out = MicrophoneSignature(
            detected_mic=best_fp.name,
            protection_priority=best_fp.protection_priority,
            proximity_slope_db_oct=meas_prox,
            presence_peak_hz=meas_pres_hz,
            presence_peak_db=meas_pres_db,
            rolloff_3db_hz=meas_rolloff,
            bass_rolloff_hz=meas_bass,
            character=best_fp.character,
            match_confidence=best_score,
            protect_proximity_below_hz=min(300.0, meas_bass * 5.0) if meas_prox > 2.0 else 0.0,
            protect_presence_band=(prot_pres_low, prot_pres_high),
            protect_rolloff_above_hz=meas_rolloff * 0.95,
        )

        logger.debug(
            "mic_character: erkannt=%s confidence=%.2f proximity=%.1f dB/oct presence=%.0f Hz rolloff=%.0f Hz",
            sig_out.detected_mic,
            best_score,
            sig_out.proximity_slope_db_oct,
            sig_out.presence_peak_hz,
            sig_out.rolloff_3db_hz,
        )
        return sig_out

    except Exception as exc:
        logger.debug("erkennen_microphone_character: nicht blockierend Ersatzpfad — %s", exc)
        return _default
