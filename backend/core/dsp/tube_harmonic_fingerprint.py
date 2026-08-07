"""
Tube/Tape Harmonic Saturation Fingerprint — §Lücke-B (v10.0.0.x)
================================================================
Erkennt und schützt die authentische H2/H4-Röhrensignatur in historischen
Aufnahmen.  Verhindert, dass NR-Phasen diese musikologisch bedeutsame
Sättigung als Rauschen abtragen.

Physikalische Basis:
    Röhrenverstärker (Trioden): Klirrspektrum dominiert von H2 (∼-20 dBc)
    und H4 (∼-35 dBc).  H3/H5 sind ≥ 10 dB schwächer als H2.
    → "Warm"-Klang = gerade Harmonische dominieren.

    Bandmaschinen (Magnetband): Klirrspektrum H2+H3 ähnlich stark,
    H4 präsent, H5 schwach.
    → "Tape"-Klang = gerade + leichte ungerade Mischung.

    Clipping/Verzerrung: H3 und H5 dominant, H2 sekundär.
    → Klirr-Profil: ungerade Harmonische dominant.

Nur-Lese-Modul — erzeugt kein Audio, nur ein Profil.

§0p Vocal-Supremacy: Profil wird für Vokal-Material priorisiert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate as _scipy_correlate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typ-Definitionen
# ---------------------------------------------------------------------------

_SIGNATURE_TYPES = frozenset({"tube", "tape", "tape_tube", "clip", "digital_noise", "neutral"})


@dataclass
class TubeHarmonicProfile:
    """Ergebnis der Röhren/Band-Harmonik-Fingerabdruck-Analyse.

    Alle Amplituden relativ zu H1 (Grundton = 1.0).
    """

    # Harmonische Amplituden (relativ zu H1)
    h2_ratio: float = 0.0  # 2. Harmonische (gerade)
    h3_ratio: float = 0.0  # 3. Harmonische (ungerade)
    h4_ratio: float = 0.0  # 4. Harmonische (gerade)
    h5_ratio: float = 0.0  # 5. Harmonische (ungerade)

    # Klassifikation
    signature_type: str = "neutral"  # "tube" | "tape" | "tape_tube" | "clip" | "neutral"
    is_tube_character: bool = False  # H2+H4 dominieren gegenüber H3+H5
    is_tape_character: bool = False  # H2+H3 gleichmäßig, H4 präsent
    is_clip_distortion: bool = False  # H3+H5 dominant → Clipping, NICHT schützen

    # Schutzparameter für NR-Integration
    protect_harmonic_bins: bool = False  # True = G_floor für harm. Bins erhöhen
    g_floor_boost_harmonic: float = 0.0  # Zusätzlicher G_floor-Boost (0.0–0.35)
    confidence: float = 0.0  # Erkennungs-Konfidenz (0.0–1.0)

    # Für Protokoll
    voiced_frames_analyzed: int = 0
    f0_estimate_hz: float = 0.0

    def to_dict(self) -> dict:
        """Serialisiert the harmonic profile as a dictionary for metadata and NR integration."""
        return {
            "h2_ratio": round(self.h2_ratio, 4),
            "h3_ratio": round(self.h3_ratio, 4),
            "h4_ratio": round(self.h4_ratio, 4),
            "h5_ratio": round(self.h5_ratio, 4),
            "signature_type": self.signature_type,
            "is_tube_character": self.is_tube_character,
            "is_tape_character": self.is_tape_character,
            "is_clip_distortion": self.is_clip_distortion,
            "protect_harmonic_bins": self.protect_harmonic_bins,
            "g_floor_boost_harmonic": round(self.g_floor_boost_harmonic, 4),
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Interne Helpers
# ---------------------------------------------------------------------------


def _estimate_f0_median(audio_mono: np.ndarray, sr: int) -> float:
    """Grobe F0-Schätzung via Autocorrelation (AMDF-Approximation).

    Gibt 0.0 zurück wenn Schätzung fehlschlägt.  Nicht-blockierend.
    """
    try:
        hop = 512
        frame_len = 2048
        f0_candidates = []
        for i in range(0, len(audio_mono) - frame_len, hop * 4):
            frame = audio_mono[i : i + frame_len]
            rms = float(np.sqrt(np.mean(frame**2)))
            if rms < 1e-5:
                continue
            # Autocorrelation
            corr = _scipy_correlate(frame, frame, mode="full", method="fft")
            corr = corr[len(corr) // 2 :]
            corr /= max(corr[0], 1e-10)
            # Suche Peak im F0-Bereich (50–1000 Hz)
            min_lag = max(1, int(sr / 1000.0))
            max_lag = int(sr / 50.0)
            if max_lag >= len(corr):
                continue
            peak_lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
            if corr[peak_lag] > 0.35:
                f0_candidates.append(float(sr) / float(peak_lag))
        if not f0_candidates:
            return 0.0
        return float(np.median(f0_candidates))
    except Exception as e:
        logger.warning("tube_harmonic_fingerprint.py::_estimate_f0_median Ersatzpfad: %s", e)
        return 0.0


def _measure_harmonic_ratios(audio_mono: np.ndarray, sr: int, f0_hz: float) -> tuple[float, float, float, float, int]:
    """Misst H2–H5 relativ zu H1 über voiced Frames.

    Returns: (h2_ratio, h3_ratio, h4_ratio, h5_ratio, n_voiced_frames)
    Alle Werte ∈ [0, 1], relativ zu H1.
    """
    n_fft = 4096
    hop = 1024
    # BW für jede Harmonische: ±1 Semitone = ±5.9 %
    _bw_frac = 0.06

    harm_energy = np.zeros(5, dtype=np.float64)  # H1..H5
    n_voiced = 0

    for i in range(0, len(audio_mono) - n_fft, hop):
        frame = audio_mono[i : i + n_fft]
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < 1e-5:
            continue
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Energie pro Harmonische
        for k in range(1, 6):  # H1..H5
            center = f0_hz * k
            if center >= sr / 2.0:
                continue
            bw = center * _bw_frac
            mask = (freqs >= center - bw) & (freqs <= center + bw)
            if not np.any(mask):
                continue
            harm_energy[k - 1] += float(np.sum(spectrum[mask] ** 2))

        n_voiced += 1

    if n_voiced == 0 or harm_energy[0] < 1e-10:
        return 0.0, 0.0, 0.0, 0.0, 0

    h1 = harm_energy[0]
    h2 = harm_energy[1] / h1
    h3 = harm_energy[2] / h1
    h4 = harm_energy[3] / h1
    h5 = harm_energy[4] / h1

    return (
        float(np.clip(h2, 0.0, 1.0)),
        float(np.clip(h3, 0.0, 1.0)),
        float(np.clip(h4, 0.0, 1.0)),
        float(np.clip(h5, 0.0, 1.0)),
        n_voiced,
    )


def _classify_signature(h2: float, h3: float, h4: float, h5: float) -> tuple[str, bool, bool, bool, float]:
    """Klassifiziert das Harmonik-Profil.

    Returns: (signature_type, is_tube, is_tape, is_clip, confidence)
    """
    even = h2 + h4
    odd = h3 + h5

    # Clipping: ungerade dominieren deutlich
    if odd > even * 1.5 and h3 > 0.02:
        conf = float(np.clip((odd - even) / max(odd + even, 1e-6), 0.0, 1.0))
        return "clip", False, False, True, conf * 0.85

    # Neutrales Rauschen: alle Harmonischen sehr schwach
    if even + odd < 0.004:
        return "neutral", False, False, False, 0.5

    # Tube: H2 klar dominant, H4 präsent, H3/H5 schwach
    tube_score = 0.0
    if h2 > 0.01:
        tube_score += 0.5
    if h2 > h3 * 2.0:
        tube_score += 0.25
    if h4 > h5 * 1.5:
        tube_score += 0.15
    if h3 < 0.01:
        tube_score += 0.10

    # Tape: H2 und H3 ähnlich, H4 präsent
    tape_score = 0.0
    if h2 > 0.005 and h3 > 0.003:
        tape_score += 0.4
    if 0.3 <= (h3 / max(h2, 1e-8)) <= 3.0:  # H2 ≈ H3
        tape_score += 0.3
    if h4 > 0.001:
        tape_score += 0.2

    if tube_score > tape_score and tube_score > 0.5:
        conf = float(np.clip(tube_score, 0.0, 1.0))
        return "tube", True, False, False, conf
    if tape_score > tube_score and tape_score > 0.5:
        conf = float(np.clip(tape_score, 0.0, 1.0))
        sig = "tape_tube" if tube_score > 0.35 else "tape"
        return sig, sig == "tape_tube", True, False, conf
    if tube_score > 0.3 or tape_score > 0.3:
        conf = max(tube_score, tape_score)
        return "tape_tube", tube_score > 0.3, tape_score > 0.3, False, conf * 0.7

    return "neutral", False, False, False, 0.3


# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------


def detect_tube_harmonic_fingerprint(
    audio: np.ndarray,
    sr: int,
    f0_estimate_hz: float = 0.0,
    *,
    material_type: str = "unknown",
) -> TubeHarmonicProfile:
    """Erkennt die Röhren/Band-Harmonik-Signatur im Audio.

    Args:
        audio:           Input Audio (mono oder stereo channels-first).
        sr:              Sample-Rate (Hz).
        f0_estimate_hz:  Bekannte F0 in Hz.  0 = automatisch schätzen.
        material_type:   Materialtyp für era-Adaption ("shellac", "vinyl", ...).

    Returns:
        TubeHarmonicProfile — niemals None (Non-blocking).
    """
    _default = TubeHarmonicProfile()
    try:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Mono
        mono = audio.mean(axis=0) if audio.ndim == 2 else audio
        if len(mono) < sr * 0.5:  # < 0.5 s → nicht analysierbar
            return _default

        # F0-Schätzung wenn nicht übergeben
        f0_hz = float(f0_estimate_hz) if float(f0_estimate_hz) > 40.0 else _estimate_f0_median(mono, sr)
        if f0_hz < 40.0 or f0_hz > 1200.0:
            # Kein plausibles F0 → Profil unbekannt
            return _default

        h2, h3, h4, h5, n_voiced = _measure_harmonic_ratios(mono, sr, f0_hz)
        if n_voiced < 3:
            return _default

        sig_type, is_tube, is_tape, is_clip, confidence = _classify_signature(h2, h3, h4, h5)

        # G_floor-Boost: Röhren/Band-Charakter schützen
        protect = (is_tube or is_tape) and not is_clip and confidence >= 0.45
        g_floor_boost = 0.0
        if protect:
            # Tube: stärker schützen (H2 ist wertvoller)
            g_floor_boost = 0.25 if is_tube else 0.15
            # Shellac/Vinyl: maximaler Schutz
            if material_type in ("shellac", "wax_cylinder"):
                g_floor_boost = min(g_floor_boost + 0.10, 0.35)
            elif material_type in ("vinyl", "reel_tape", "tape"):
                g_floor_boost = min(g_floor_boost + 0.05, 0.30)

        profile = TubeHarmonicProfile(
            h2_ratio=h2,
            h3_ratio=h3,
            h4_ratio=h4,
            h5_ratio=h5,
            signature_type=sig_type,
            is_tube_character=is_tube,
            is_tape_character=is_tape,
            is_clip_distortion=is_clip,
            protect_harmonic_bins=protect,
            g_floor_boost_harmonic=float(g_floor_boost),
            confidence=float(confidence),
            voiced_frames_analyzed=n_voiced,
            f0_estimate_hz=float(f0_hz),
        )
        logger.info(
            "§Lücke-B TubeHarmonicFingerprint: sig=%s tube=%s tape=%s clip=%s "
            "h2=%.3f h3=%.3f h4=%.3f h5=%.3f conf=%.2f g_floor_boost=%.2f",
            sig_type,
            is_tube,
            is_tape,
            is_clip,
            h2,
            h3,
            h4,
            h5,
            confidence,
            g_floor_boost,
        )
        return profile

    except Exception as exc:
        logger.debug("TubeHarmonicFingerprint nicht blockierend Fehlschlag: %s", exc)
        return _default
