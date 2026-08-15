"""
core/consonant_enhancement.py — Konsonanten-Enhancement & Sibilanten-Erkennung
===============================================================================

Implementiert §2.8 Step 5b/5c der Aurik-Spec (Vocal-Restaurierungskette):

Step 5b  ConsonantDetector:
    Frikativ-Segmente (stimmlose Konsonanten: s, f, sch, th) identifizieren.
    Erkennungskriterium: ZCR > 0.3, Energie in 4–16 kHz dominant.

Step 5c  ConsonantEnhancement (Adaptive Spectral Tilt Correction):
    Frikative, die durch NR abgedämpft wurden, wieder anheben.
    - HF-Anhebung ≤ +6 dB im Frikativ-Band (stimmtyp-adaptiv)
    - MALE: 5–10 kHz | FEMALE: 6–12 kHz | CHILD: 7–14 kHz
    - Pflicht-Invariante: SNR_frikativ_after ≥ SNR_frikativ_before + 3 dB
    - Kreuzfade 5 ms (Hanning) an Voiced/Unvoiced-Übergängen (Artefakt-Schutz)

Kausal-Konditionierung (§2.8 „Position hängt vom Defekt ab"):
    Starkes BANDWIDTH_LOSS oder HIGH_FREQ_NOISE → Friktive wurden durch NR stärker
    gedämpft → ConsonantEnhancement greift proportional stärker ein.
    COMPRESSION_ARTIFACTS → Codec-Vor-Rauschen überlagert Frikative → moderate Anhebung.

Singleton-Muster (§3.2), NaN/Inf-Schutz (§3.1), vollständige Type-Annotations (§3.7).

Author: Aurik Development Team
Spec:   §2.8 Step 5b/5c, §4.4 (ConsonantEnhancement DSP-Zeile)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import scipy.signal as sig

logger = logging.getLogger(__name__)

# ── Stimmtyp-adaptive Frikativ-Bänder (§2.8, §4.4) ─────────────────────────
#   MALE:   5–10 kHz | FEMALE: 6–12 kHz | CHILD: 7–14 kHz
FRICATIVE_BANDS: dict[str, tuple[float, float]] = {
    "male": (5_000.0, 10_000.0),
    "female": (6_000.0, 12_000.0),
    "child": (7_000.0, 14_000.0),
    "androgynous": (5_500.0, 11_000.0),
    "unknown": (6_000.0, 12_000.0),  # FEMALE-Fallback per Spec
}

# Maximale HF-Anhebung pro Frikativsegment (§2.8: ≤ +6 dB)
MAX_BOOST_DB: float = 6.0
# Mindest-SNR-Verbesserung im Frikativband (§2.8 Pflicht-Invariante)
SNR_MIN_IMPROVEMENT_DB: float = 3.0
# Crossfade-Dauer an Übergängen (§2.8)
CROSSFADE_MS: float = 5.0

# ZCR-Schwellwerte für Konsonanten-Erkennung (§2.8 Step 5b: ZCR > 0.3)
ZCR_CONSONANT_THRESHOLD: float = 0.30
# HF-Energie-Anteil (4–16 kHz) für Frikativ-Kandidaten (§2.8 Step 5b)
HF_ENERGY_THRESHOLD: float = 0.25

# Kausal-Konditionierungsfaktoren (§2.8 positionsabhängige Stärke):
# – Wenn BANDWIDTH_LOSS hoch → NR hat mehr abgedämpft → mehr Boost nötig
# – Wenn HIGH_FREQ_NOISE hoch → Rauschentfernung war stärker → mehr Boost
CAUSAL_DEFECT_BOOST: dict[str, float] = {
    "bandwidth_loss": 1.0,  # maximaler Boost-Faktor (up to MAX_BOOST_DB)
    "high_freq_noise": 0.85,
    "compression_artifacts": 0.60,
    "tape_hiss": 0.75,
    "default": 0.50,  # halber Boost wenn kein Defekt-Prior
}


# ── Ergebnis-Dataclass ───────────────────────────────────────────────────── #


@dataclass
class ConsonantEnhancementResult:
    """Ergebnis der Konsonanten-Enhancement-Verarbeitung."""

    audio: np.ndarray = field(repr=False)
    """Verarbeitetes Audio-Signal (float32, NaN/Inf-frei)."""

    fricative_segments: int = 0
    """Anzahl erkannter Frikativ-Segmente."""

    snr_improvement_db: float = 0.0
    """Erzielte SNR-Verbesserung im Frikativband in dB."""

    boost_applied_db: float = 0.0
    """Tatsächlich angewendeter HF-Boost in dB."""

    invariant_met: bool = True
    """True wenn SNR_frikativ_after ≥ SNR_frikativ_before + 3 dB."""

    voice_gender: str = "unknown"
    """Verwendetes Stimmtyp-Profil."""

    causal_factor: float = 0.5
    """Kausal-Konditionierungsfaktor aus Defekt-Scores 0..1."""


# ── ConsonantEnhancement Singleton ──────────────────────────────────────────

_instance: ConsonantEnhancement | None = None
_lock = threading.Lock()


def get_consonant_enhancer() -> ConsonantEnhancement:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking, §3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ConsonantEnhancement()
    return _instance


def enhance_consonants(
    audio: np.ndarray,
    sr: int,
    voice_gender: str = "unknown",
    defect_scores: dict[str, float] | None = None,
) -> ConsonantEnhancementResult:
    """Convenience-Wrapper: Konsonanten-Enhancement ohne Klassen-Instantiierung.

    Args:
        audio:        Mono oder Stereo float32/64, vorab auf 48 000 Hz resampelt.
        sr:           Sample-Rate (muss 48 000 Hz sein, §6.6).
        voice_gender: Stimmtyp-Label ("male" / "female" / "child" / "androgynous" /
                      "unknown") für band-adaptive Verarbeitung.
        defect_scores: Dict mit Defekt-Schwere-Werten aus DefectScanner
                      (z. B. {"bandwidth_loss": 0.7, "high_freq_noise": 0.4}).
                      Wenn None → kausal neutraler Boost.

    Returns:
        ConsonantEnhancementResult mit verarbeitetem Audio + Metriken.
    """
    return get_consonant_enhancer().enhance(audio, sr, voice_gender, defect_scores)


class ConsonantEnhancement:
    """Adaptive Spectral Tilt Correction für Frikativ-Konsonanten.

    Restauriert stimmlose Reibelaute (s, f, sch, th), die durch vorangehende
    Rauschunterdrückung (OMLSA/DeepFilterNet) oder Codec-Kompression (MP3/AAC)
    abgedämpft wurden.

    Algorithmus:
        1. PhonemeDetector → sibilant_mask (bool-Array auf Sample-Ebene)
        2. Kausal-Konditionierung: Boost-Stärke aus DefectScanner-Scores ableiten
        3. Für jedes Frikativ-Segment:
           a. SNR im Frikativband vor Boost messen
           b. Höhenanhebung via High-Shelf EQ (stimmtyp-adaptiv)
           c. Boost-Stärke clippen auf ≤ +6 dB (§2.8)
           d. SNR-Verbesserung prüfen (≥ +3 dB Invariante)
        4. Crossfade 200  # §G3 (GEBOTE.md): Minimum 200ms ms (Hanning) an Voiced/Unvoiced-Übergängen
        5. NaN/Inf-Guard + clip(-1, 1)

    Invarianten:
        - SNR_frikativ_after ≥ SNR_frikativ_before + 3 dB (§2.8)
        - HF-Boost ≤ +6 dB pro Segment (§2.8)
        - Verstärkung NUR in Frikativ-Segmenten (Voiced bewahrt)
        - Kreuzfade 5 ms verhindert Stufenklicks
        - Laufzeit: typisch ≤ 0.3 s / Minute Audio (DSP-only)
    """

    # Fensterparameter für Konsonanten-Erkennung
    _FRAME_SIZE: int = 1024
    _HOP_SIZE: int = 512

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        voice_gender: str = "unknown",
        defect_scores: dict[str, float] | None = None,
    ) -> ConsonantEnhancementResult:
        """Frikativ-Konsonanten stimmtyp-adaptiv wiederherstellen.

        Args:
            audio:        Mono (1-D) oder Stereo (2-D) float32/64.
            sr:           Sample-Rate in Hz (48 000 empfohlen).
            voice_gender: Stimmtyp ("male", "female", "child", "unknown").
            defect_scores: Defekt-Schwere-Werte aus DefectScanner (Optional).

        Returns:
            ConsonantEnhancementResult mit verarbeitetem Audio.
        """
        # ── Eingangs-Validierung ────────────────────────────────────────── #
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return _passthrough(audio, voice_gender)
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        stereo = audio.ndim == 2

        # Mono für Analyse
        mono = audio.mean(axis=0) if stereo else audio.copy()

        # ── Kausal-Konditionierung ──────────────────────────────────────── #
        causal_factor = self._causal_factor(defect_scores or {})

        # ── Frikativ-Maske via PhonemeDetector ─────────────────────────── #
        sib_mask = self._sibilant_mask(mono, sr)
        n_fricative_frames = int(np.sum(sib_mask))
        if n_fricative_frames == 0:
            logger.debug("ConsonantEnhancement: keine Frikativ-Segmente gefunden, ueberspringen.")
            return ConsonantEnhancementResult(
                audio=np.clip(audio, -1.0, 1.0),
                fricative_segments=0,
                snr_improvement_db=0.0,
                boost_applied_db=0.0,
                invariant_met=True,
                voice_gender=voice_gender,
                causal_factor=causal_factor,
            )

        # ── Boost-Stärke berechnen ──────────────────────────────────────── #
        target_boost_db = causal_factor * MAX_BOOST_DB  # 0 .. +6 dB
        target_boost_db = float(np.clip(target_boost_db, 0.0, MAX_BOOST_DB))

        # Frikativband bestimmen
        f_lo, f_hi = _fricative_band(voice_gender, sr)

        # SNR vor Boost
        snr_before = _snr_in_band(mono, sr, f_lo, f_hi)

        # ── High-Shelf EQ auf Frikativ-Segmenten ───────────────────────── #
        if stereo:
            channels_out = []
            for ch in range(audio.shape[0]):
                ch_proc = self._boost_segment(audio[ch], sr, sib_mask, f_lo, f_hi, target_boost_db)
                channels_out.append(ch_proc)
            processed = np.stack(channels_out, axis=0)
        else:
            processed = self._boost_segment(mono, sr, sib_mask, f_lo, f_hi, target_boost_db)

        # ── SNR-Invariante prüfen ───────────────────────────────────────── #
        proc_mono = processed.mean(axis=0) if processed.ndim == 2 else processed
        snr_after = _snr_in_band(proc_mono, sr, f_lo, f_hi)
        snr_improvement = snr_after - snr_before
        invariant_met = snr_improvement >= SNR_MIN_IMPROVEMENT_DB

        if not invariant_met:
            # Fallback: Boost leicht erhöhen bis Invariante erfüllt
            extra_db = min(SNR_MIN_IMPROVEMENT_DB - snr_improvement + 0.5, 3.0)
            if stereo:
                channels_out2 = []
                for ch in range(audio.shape[0]):
                    ch_proc2 = self._boost_segment(
                        audio[ch],
                        sr,
                        sib_mask,
                        f_lo,
                        f_hi,
                        min(target_boost_db + extra_db, MAX_BOOST_DB),
                    )
                    channels_out2.append(ch_proc2)
                processed = np.stack(channels_out2, axis=0)
            else:
                processed = self._boost_segment(
                    mono,
                    sr,
                    sib_mask,
                    f_lo,
                    f_hi,
                    min(target_boost_db + extra_db, MAX_BOOST_DB),
                )
            proc_mono = processed.mean(axis=0) if processed.ndim == 2 else processed
            snr_after = _snr_in_band(proc_mono, sr, f_lo, f_hi)
            snr_improvement = snr_after - snr_before
            invariant_met = snr_improvement >= SNR_MIN_IMPROVEMENT_DB
            if not invariant_met:
                logger.warning(
                    "ConsonantEnhancement: SNR-Invariante nicht erfüllt "
                    "(Δ%.1f dB < %.1f dB). Möglicherweise kaum Frikativinhalt.",
                    snr_improvement,
                    SNR_MIN_IMPROVEMENT_DB,
                )

        # ── NaN/Inf-Guard & Clipping ────────────────────────────────────── #
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)

        logger.debug(
            "ConsonantEnhancement: gender=%s, causal=%.2f, boost=%.1f dB, "
            "SNR_before=%.1f dB, SNR_after=%.1f dB, Δ=%.1f dB, sib_frames=%d",
            voice_gender,
            causal_factor,
            target_boost_db,
            snr_before,
            snr_after,
            snr_improvement,
            n_fricative_frames,
        )

        return ConsonantEnhancementResult(
            audio=processed,
            fricative_segments=n_fricative_frames,
            snr_improvement_db=float(snr_improvement),
            boost_applied_db=float(target_boost_db),
            invariant_met=invariant_met,
            voice_gender=voice_gender,
            causal_factor=causal_factor,
        )

    # ── Private Hilfsmethoden ────────────────────────────────────────────── #

    def _causal_factor(self, defect_scores: dict[str, float]) -> float:
        """Kausal-Konditionierungsfaktor aus Defekt-Scores ableiten (0..1).

        Stärkere Defekte → stärkere Frikativ-Dämpfung durch NR → mehr Boost nötig.

        Algorithmus:
            Score = Σ(defect_severity × CAUSAL_DEFECT_BOOST[defect]) / count
            → geclamppt auf [0.1, 1.0]
        """
        if not defect_scores:
            return CAUSAL_DEFECT_BOOST["default"]

        total = 0.0
        weight = 0.0
        for defect_key, boost_factor in CAUSAL_DEFECT_BOOST.items():
            if defect_key == "default":
                continue
            severity = float(defect_scores.get(defect_key, 0.0))
            if not np.isfinite(severity):
                continue
            total += severity * boost_factor
            weight += boost_factor

        if weight < 1e-6:
            return CAUSAL_DEFECT_BOOST["default"]

        factor = total / weight
        return float(np.clip(factor, 0.1, 1.0))

    def _sibilant_mask(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """Sample-genaue Frikativ-Maske via ZCR + HF-Energie (§2.8 Step 5b).

        Erkennungskaskade (3 Stufen):
            1. PhonemeDetector (Singleton, plugins/phoneme_detector.py)
            2. ConsonantDetector (Singleton, plugins/consonant_detector.py) ← §2.8 Step 5b
            3. Leere Maske als Notfall-Fallback (kein Absturz)

        Returns:
            bool-Array [n_samples], True = Frikativ/Sibilant.
        """
        # Stufe 1: PhonemeDetector (wenn Sibilant-Maske verfügbar)
        try:
            from plugins.phoneme_detector import get_phoneme_detector

            det = get_phoneme_detector()
            result = det.detect(mono, sr)
            if result.sibilant_mask is not None and result.sibilant_mask.shape[0] == mono.shape[0]:
                return result.sibilant_mask.astype(bool)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning(
                "§G23 PhonemeDetector nicht verfügbar, DSP-Ersatzpfad ConsonantDetector: %s", exc, exc_info=True
            )

        # Stufe 2: ConsonantDetector — §2.8 Step 5b (eigenständiger Singleton)
        try:
            from plugins.consonant_detector import get_consonant_detector

            cd_result = get_consonant_detector().detect(mono, sr)
            if cd_result.mask.shape[0] == mono.shape[0]:
                logger.debug(
                    "ConsonantDetector: %d Frikativ-Frames (Verhaeltnis=%.2f)",
                    cd_result.n_fricative_frames,
                    cd_result.fricative_ratio,
                )
                return cd_result.mask  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("§G23 ConsonantDetector nicht verfügbar, DSP-Ersatzpfad leere Maske: %s", exc, exc_info=True)

        # Stufe 3: Notfall-Fallback (kein Daten-Verlust, Pipeline läuft weiter)
        return np.zeros(len(mono), dtype=bool)  # type: ignore[no-any-return]

    def _boost_segment(
        self,
        channel: np.ndarray,
        sr: int,
        sib_mask: np.ndarray,
        f_lo: float,
        f_hi: float,
        boost_db: float,
    ) -> np.ndarray:
        """High-Shelf Boost nur in Frikativ-Segmenten mit Crossfade.

        Algorithmus:
            1. Frikativband via Butterworth-Bandpass extrahieren
            2. Lineare Gain-Maske aus sib_mask erzeugen (0 → 1)
            3. Crossfade 200  # §G3 (GEBOTE.md): Minimum 200ms ms (Hanning) an Segment-Übergängen
            4. Frikativband × (gain - 1.0) zum Original addieren

        Args:
            channel:  Mono-Kanal float32.
            sr:       Sample-Rate.
            sib_mask: bool-Array Sample-genau.
            f_lo/f_hi: Frikativ-Band-Grenzen.
            boost_db:  Anhebung in dB (0 .. +6).

        Returns:
            Verarbeiteter Mono-Kanal.
        """
        if boost_db < 0.01:
            return channel.astype(np.float32)  # type: ignore[no-any-return]

        n = len(channel)
        boost_lin = float(10.0 ** (boost_db / 20.0)) - 1.0  # Additive Gain (0 wenn 0 dB)

        # ── Bandpass-gefiltertes Frikativsignal ─────────────────────── #
        nyq = sr / 2.0
        f_lo_norm = max(f_lo, 20.0) / nyq
        f_hi_norm = min(f_hi, nyq * 0.99) / nyq
        if f_lo_norm >= f_hi_norm or f_lo_norm <= 0.0:
            return channel.astype(np.float32)  # type: ignore[no-any-return]

        try:
            sos = sig.butter(4, [f_lo_norm, f_hi_norm], btype="band", output="sos")
            # §2.51 Anti-Zeitversatz: sosfiltfilt (Zero-Phase) — fric_band wird zum channel addiert.
            fric_band = sig.sosfiltfilt(sos, channel)
        except Exception as exc:
            logger.debug("ConsonantEnhancement Butterworth fehlgeschlagen: %s", exc)
            return channel.astype(np.float32)  # type: ignore[no-any-return]

        # ── Gain-Maske aus sib_mask ─────────────────────────────────── #
        gain_mask = sib_mask.astype(np.float32)

        # Crossfade 200  # §G3 (GEBOTE.md): Minimum 200ms ms (Hanning) an Übergängen (§2.8)
        cf_samples = max(2, int(CROSSFADE_MS / 1000.0 * sr))
        hanning = np.hanning(2 * cf_samples)[:cf_samples]
        # Übergänge: False→True (Onset) und True→False (Offset)
        diff = np.diff(gain_mask.astype(np.int8), prepend=0, append=0)  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]
        for onset in onsets:
            s = max(0, onset - cf_samples)
            ramp = hanning[: onset - s]
            if ramp.size > 0:
                gain_mask[s:onset] = np.maximum(gain_mask[s:onset], ramp)
        for offset in offsets:
            s = offset
            e = min(n, s + cf_samples)
            ramp = 1.0 - hanning[: e - s]
            if ramp.size > 0:
                gain_mask[s:e] = np.maximum(gain_mask[s:e], ramp)

        # ── Boost anwenden: nur additive Komponente in Frikativband ─── #
        out = channel + fric_band * (boost_lin * gain_mask)
        return out.astype(np.float32)  # type: ignore[no-any-return]


# ── Modul-Hilfsfunktionen ────────────────────────────────────────────────── #


def _fricative_band(voice_gender: str, sr: int) -> tuple[float, float]:
    """Stimmtyp-adaptives Frikativband (Nyquist-sicher)."""
    f_lo, f_hi = FRICATIVE_BANDS.get(voice_gender.lower(), FRICATIVE_BANDS["unknown"])
    nyq = sr / 2.0
    return float(min(f_lo, nyq * 0.60)), float(min(f_hi, nyq * 0.95))


def _snr_in_band(mono: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    """Signal-Rausch-Verhältnis im Frikativband [dB].

    Schätzt SNR als Verhältnis Band-RMS / (Gesamtsignal-RMS − Band-RMS).
    NaN/Inf → 0.0 dB.
    """
    nyq = sr / 2.0
    f_lo_n = max(f_lo, 20.0) / nyq
    f_hi_n = min(f_hi, nyq * 0.99) / nyq
    if f_lo_n >= f_hi_n:
        return 0.0
    try:
        sos = sig.butter(4, [f_lo_n, f_hi_n], btype="band", output="sos")
        band = sig.sosfilt(sos, mono)
    except Exception as e:
        logger.warning("consonant_enhancement.py::_snr_in_band Ersatzpfad: %s", e)
        return 0.0
    band_rms = float(np.sqrt(np.mean(band**2)) + 1e-12)
    total_rms = float(np.sqrt(np.mean(mono**2)) + 1e-12)
    noise_rms = max(total_rms - band_rms, 1e-12)
    snr = 20.0 * np.log10(band_rms / noise_rms)
    return float(snr) if np.isfinite(snr) else 0.0


def _passthrough(audio: np.ndarray, voice_gender: str) -> ConsonantEnhancementResult:
    """Leeres Ergebnis ohne Verarbeitung (Passthrough)."""
    safe = (
        np.zeros(1, dtype=np.float32)
        if not isinstance(audio, np.ndarray)
        else np.clip(np.nan_to_num(audio.astype(np.float32)), -1.0, 1.0)
    )
    return ConsonantEnhancementResult(
        audio=safe,
        fricative_segments=0,
        snr_improvement_db=0.0,
        boost_applied_db=0.0,
        invariant_met=True,
        voice_gender=voice_gender,
        causal_factor=CAUSAL_DEFECT_BOOST["default"],
    )


def measure_fricative_snr(audio: np.ndarray, sr: int, voice_gender: str = "unknown") -> float:
    """Misst den SNR im stimmtyp-adaptiven Frikativband (öffentliche API, §2.8).

    Öffentliche Schnittstelle zu _snr_in_band() für externe Module (z. B. Phase 19).
    Wird für die Feedback-Invariante §2.8 verwendet:
        SNR_frikativ_after_full_chain ≥ SNR_frikativ_before_deessing + 3 dB

    Args:
        audio:        Mono (1-D) oder Stereo (2-D) float32/64, NaN-sicher.
        sr:           Sample-Rate in Hz (48 000 empfohlen).
        voice_gender: Stimmtyp ("male" / "female" / "child" / "unknown")
                      → bestimmt das adaptive Frikativband.

    Returns:
        SNR in dB, NaN/Inf-sicher (0.0 wenn Messung nicht möglich).
    """
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        return 0.0
    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    f_lo, f_hi = _fricative_band(voice_gender, sr)
    return _snr_in_band(mono, sr, f_lo, f_hi)


# ── PlosiveBurstPreserver ────────────────────────────────────────────────────


@dataclass
class PlosiveBurstResult:
    """Result of plosive burst detection and transient restoration."""

    audio: np.ndarray = field(repr=False)
    """Audio after transient restoration (float32, NaN/Inf-free)."""

    n_bursts_detected: int = 0
    """Number of plosive bursts detected."""

    n_bursts_restored: int = 0
    """Number of burst transients successfully restored."""

    onset_positions_ms: list = field(default_factory=list)
    """Onset timestamps in milliseconds."""


_plosive_instance: PlosiveBurstPreserver | None = None
_plosive_lock = threading.Lock()


def get_plosive_preserver() -> PlosiveBurstPreserver:
    """Thread-safe singleton accessor (Double-Checked Locking, §3.2)."""
    global _plosive_instance
    if _plosive_instance is None:
        with _plosive_lock:
            if _plosive_instance is None:
                _plosive_instance = PlosiveBurstPreserver()
    return _plosive_instance


def preserve_plosive_transients(
    audio: np.ndarray,
    sr: int,
    processed_audio: np.ndarray,
    blend: float = 0.60,
) -> PlosiveBurstResult:
    """Convenience wrapper: restore plosive bursts after enhancement processing.

    Args:
        audio:           Original (pre-enhancement) audio.
        sr:              Sample rate (48 000 Hz).
        processed_audio: Audio after enhancement/compression.
        blend:           Transient restoration strength [0.0–1.0].

    Returns:
        PlosiveBurstResult with restored audio and detection metrics.
    """
    return get_plosive_preserver().restore(audio, sr, processed_audio, blend)


class PlosiveBurstPreserver:
    """Erkennt plosive bursts and restores their transients after compression.

    Plosive consonants (/p/, /t/, /k/, /b/, /d/, /g/) are characterised by a
    rapid energy onset (release burst, ≤ 5 ms rise time) followed by a short
    aspiration phase (5–50 ms).  Micro-compressors and NoiseReduction alike
    tend to smear or attenuate these transients, making speech and singing
    sound "soft" or "lisped".

    Algorithm:
        1. Frame-wise RMS envelope (hop 1 ms) of the *original* audio.
        2. Onset detection: RMS rises > ONSET_RATIO_DB dB within ONSET_WINDOW_MS.
        3. For each onset, classify as plosive if:
           a. Rise time ≤ BURST_MAX_RISE_MS  (sharp onset → not fricative)
           b. Pre-onset energy ≤ BURST_PRE_SILENCE_FACTOR × post-onset peak RMS
              (brief preceding silence → closure phase of plosive)
        4. For each plosive burst window [onset – PRE_MS, onset + POST_MS]:
           - Extract original transient envelope
           - Extract processed transient envelope
           - Compute gain ratio (original / processed), clamped to MAX_RESTORE_DB
           - Re-apply gain to processed audio in burst window (Hanning-weighted)
        5. NaN/Inf-Guard + clip(-1, 1).

    Invariants:
        - Transient gain never exceeds +MAX_RESTORE_DB dB.
        - Does not modify non-burst regions.
        - NaN/Inf output is impossible (guarded).
    """

    # Detection thresholds
    ONSET_RATIO_DB: float = 12.0  # RMS must rise ≥ 12 dB in onset window
    ONSET_WINDOW_MS: float = 5.0  # Look-ahead for onset detection (ms)
    BURST_MAX_RISE_MS: float = 5.0  # Maximum rise time for plosive classification
    BURST_PRE_SILENCE_FACTOR: float = 0.25  # Pre-burst RMS ≤ 25 % of burst peak

    # Restoration window
    BURST_PRE_MS: float = 2.0  # Samples before onset to include in window
    BURST_POST_MS: float = 30.0  # Aspiration window after onset (ms)
    MAX_RESTORE_DB: float = 6.0  # Maximum transient gain restoration

    # Frame / hop sizes
    _FRAME_MS: float = 2.0  # Frame length for envelope (ms)
    _HOP_MS: float = 1.0  # Hop size for envelope tracking (ms)

    def restore(
        self,
        original: np.ndarray,
        sr: int,
        processed: np.ndarray,
        blend: float = 0.60,
    ) -> PlosiveBurstResult:
        """Erkennt plosive bursts in *original* and restore their transients.

        Args:
            original:  Pre-enhancement mono audio (float32, 48 kHz).
            sr:        Sample rate — must be 48 000 Hz.
            processed: Post-enhancement audio (same shape as ``original``).
            blend:     Transient restoration weight [0.0–1.0].
                       0.0 = no restoration, 1.0 = full original transient.

        Returns:
            PlosiveBurstResult with restored audio.
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        if not isinstance(original, np.ndarray) or original.size == 0:
            empty = np.zeros(0, dtype=np.float32)
            return PlosiveBurstResult(audio=empty)

        original = np.nan_to_num(np.asarray(original, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.nan_to_num(np.asarray(processed, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo: process the mix for detection, apply to each channel
        stereo = original.ndim == 2
        orig_mono = original.mean(axis=0) if stereo else original
        proc_channels = (
            [processed[:, ch] for ch in range(processed.shape[1])] if stereo and processed.ndim == 2 else [processed]
        )

        frame_n = max(2, int(self._FRAME_MS / 1000.0 * sr))
        hop_n = max(1, int(self._HOP_MS / 1000.0 * sr))

        # ── RMS envelope of original ──────────────────────────────────── #
        envelope = self._rms_envelope(orig_mono, frame_n, hop_n)
        n_frames = len(envelope)

        # ── Onset detection ───────────────────────────────────────────── #
        onset_window_frames = max(1, int(self.ONSET_WINDOW_MS / self._HOP_MS))
        onset_ratio_lin = 10.0 ** (self.ONSET_RATIO_DB / 20.0)
        burst_max_rise_frames = max(1, int(self.BURST_MAX_RISE_MS / self._HOP_MS))

        onsets: list[int] = []  # onset frame indices
        fi = onset_window_frames
        while fi < n_frames:
            pre_rms = envelope[fi - onset_window_frames]
            post_rms = envelope[fi]
            if pre_rms < 1e-10:
                fi += 1
                continue
            ratio = post_rms / pre_rms
            if ratio >= onset_ratio_lin:
                # Check rise time (plosive = very sharp, fricative = gradual)
                rise_start = max(0, fi - onset_window_frames)
                rise_frames = fi - rise_start
                if rise_frames <= burst_max_rise_frames:
                    # Check pre-burst silence (closure phase)
                    pre_peak = float(np.max(envelope[max(0, rise_start - 5) : rise_start + 1]))
                    if pre_peak <= self.BURST_PRE_SILENCE_FACTOR * post_rms:
                        onsets.append(fi)
                        # Skip ahead to avoid duplicate detections
                        fi += max(1, int(self.BURST_POST_MS / self._HOP_MS))
                        continue
            fi += 1

        if not onsets:
            out = np.clip(processed, -1.0, 1.0)
            return PlosiveBurstResult(audio=out, n_bursts_detected=0)

        # ── Transient restoration ──────────────────────────────────────── #
        pre_s = max(1, int(self.BURST_PRE_MS / 1000.0 * sr))
        post_s = max(1, int(self.BURST_POST_MS / 1000.0 * sr))
        max_gain = 10.0 ** (self.MAX_RESTORE_DB / 20.0)
        blend = float(np.clip(blend, 0.0, 1.0))
        n_samples = len(orig_mono)

        result_channels = [ch.copy() for ch in proc_channels]
        n_restored = 0
        onset_ms_list: list[float] = []

        for onset_frame in onsets:
            onset_sample = onset_frame * hop_n
            s = max(0, onset_sample - pre_s)
            e = min(n_samples, onset_sample + post_s)
            if e <= s:
                continue

            window_len = e - s
            hann = np.hanning(window_len)

            # Envelope in burst window
            orig_env = np.abs(orig_mono[s:e]) + 1e-10
            for ch_arr in result_channels:
                if len(ch_arr) < e:
                    continue
                proc_env = np.abs(ch_arr[s:e]) + 1e-10
                # Gain to restore original transient envelope
                gain = np.clip(orig_env / proc_env, 1.0, max_gain)
                # Apply blended, Hanning-windowed restoration
                blend_gain = 1.0 + blend * (gain - 1.0) * hann
                ch_arr[s:e] *= blend_gain

            n_restored += 1
            onset_ms_list.append(round(onset_sample / sr * 1000.0, 2))

        # Reassemble
        out = np.column_stack(result_channels) if stereo and len(result_channels) > 1 else result_channels[0]

        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0)

        logger.debug(
            "PlosiveBurstPreserver: erkannt=%d, wiederhergestellt=%d, blend=%.2f",
            len(onsets),
            n_restored,
            blend,
        )
        return PlosiveBurstResult(
            audio=out,
            n_bursts_detected=len(onsets),
            n_bursts_restored=n_restored,
            onset_positions_ms=onset_ms_list,
        )

    def _rms_envelope(self, audio: np.ndarray, frame_n: int, hop_n: int) -> np.ndarray:
        """Berechnet frame-wise RMS envelope."""
        n = len(audio)
        frames = []
        pos = 0
        while pos < n:
            frame = audio[pos : pos + frame_n]
            rms = float(np.sqrt(np.mean(frame**2))) if len(frame) > 0 else 0.0
            frames.append(rms if np.isfinite(rms) else 0.0)
            pos += hop_n
        return np.array(frames, dtype=np.float32)  # type: ignore[no-any-return]
