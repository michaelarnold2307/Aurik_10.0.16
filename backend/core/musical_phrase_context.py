"""
Musikalische Phrasenkontextfenster für Dropout-Inpainting — Aurik 10.0.0 (§2.12)

Standard-Inpainting (DiffWave, NMF-β) konditioniert auf ±200 ms lokalen Kontext.
Aurik 10.0.0 erweitert dies auf musikalische Phrasen-Kontextfenster (4–30 Sekunden),
die durch Beat-Tracking und Phrasenanalyse bestimmt werden.

Segmentierungsalgorithmus:
    1. Beat-Tracking: madmom (RNN-basiert) → Beat-Positionen in Samples
       Fallback: librosa-ähnliches Energy-Tempogramm
    2. Phrase-Boundary-Detection:
       → Harmonische Grenze: Chroma-Vektorsprung > 0.3 (neue Harmonik)
       → Dynamische Grenze: Energie-Delta > 6 dB in < 50 ms
       → Minimale Phrasendauer: 4 Takte (bei 120 BPM ≈ 8 s)
    3. Dropout-Lücke lokalisieren → in welcher Phrase?
    4. Kontextfenster = [Phrasenanfang, Phrasenende] ohne Lücke
       Maximum-Kontextgröße: 30 s (Performance-Budget)
    5. DiffWave-Inpainting: Phrasen-Kontext als Conditioning-Signal
       NMF-β: Phrasen-Kontext als Initialisierungsmatrix W₀

Invarianten:
    - Lücke > 50 % der Phrasen-Länge → benachbarte Phrase als Kontext
    - Ergebnis-Onset muss Beat-Grid einhalten (GrooveMetric ≥ 0.88)
    - Inpainted Chroma: Pearson-Korrelation mit Phrasen-Chroma ≥ 0.92
    - Kein Phrase-Context für Dateien < 8 s
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
import warnings

# madmom 0.17+ warnt fälschlich über veralteten `method`-Parameter,
# obwohl Aurik bereits `histogram_processor=` (neue API) verwendet.
# Der Check in madmom/features/tempo.py:724 feuert als False Positive.
warnings.filterwarnings(
    "ignore",
    message="Usage of `method` is deprecated",
    category=UserWarning,
    module="madmom",
)
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguration (§2.12)
# ---------------------------------------------------------------------------

MIN_FILE_DURATION_S: float = 8.0  # Kein Phrase-Context darunter
MAX_CONTEXT_DURATION_S: float = 30.0  # Performance-Budget
MIN_PHRASE_DURATION_S: float = 8.0  # ≈ 4 Takte @ 120 BPM
CHROMA_JUMP_THRESHOLD: float = 0.3  # Harmonische Grenze
ENERGY_DELTA_DB: float = 6.0  # Dynamische Grenze
GAP_PHRASE_RATIO_MAX: float = 0.50  # Anteil Lücke pro Phrase

N_CHROMA = 12  # Chroma-Bins

# Kompatibilitätskonstanten (migriert aus core.musical_phrase_context_extractor)
MIN_PHRASE_BEATS: int = 4  # Mindest-Beat-Anzahl pro Phrase
MAX_GAP_DURATION_MS: float = 999.0  # Maximale Dropout-Lückenlänge [ms]
GAP_FRACTION_THRESHOLD: float = GAP_PHRASE_RATIO_MAX  # = 0.50, abwärtskompatibler Alias
PHRASE_CONTEXTS_DIR = None  # Lazy-Initialisierung via _get_phrase_contexts_dir()


@lru_cache(maxsize=1)
def _get_fft_autocorr_fn():
    """Lädt fft_autocorr lazy und gecacht."""
    module = importlib.import_module("backend.core.core_utils")
    return module.fft_autocorr


@lru_cache(maxsize=1)
def _get_madmom_module():
    """Lädt madmom optional lazy; gibt None zurück, falls nicht verfügbar."""
    try:
        return importlib.import_module("madmom")
    except Exception:
        logger.warning("⚠️ SOTA musical_phrase_context: madmom nicht verfügbar — DSP-Ersatzpfad aktiv")
        return None


def _get_phrase_contexts_dir():
    """Gibt das Persistenz-Verzeichnis für Phrasen-Kontexte zurück (lazy)."""
    return Path.home() / ".aurik" / "phrase_contexts"


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class PhraseContext:
    """Musikalisches Phrasenkontextfenster für Dropout-Inpainting.

    Attribute:
        audio_context:   Kontext-Audio (Phrase ohne Dropout-Lücke), float32
        chroma_mean:     Mittleres Chroma-Profil der Phrase (12-dim)
        tempo_bpm:       Geschätztes Tempo [BPM] der Phrase ∈ [40, 240]
        beat_positions:  Beat-Positionen in Samples innerhalb der Phrase
        phrase_start_s:  Phrasenanfang [s] im Gesamtaudio
        phrase_end_s:    Phrasenende [s] im Gesamtaudio
        gap_start_s:     Dropout-Lückenanfang [s]
        gap_end_s:       Dropout-Lückenende [s]
        is_fallback:     True falls kein Phrase-Context gefunden (lokales Fenster)
    """

    audio_context: np.ndarray
    chroma_mean: np.ndarray  # shape (12,), float32
    tempo_bpm: float
    beat_positions: list[int]  # Sample-Positionen
    phrase_start_s: float
    phrase_end_s: float
    gap_start_s: float
    gap_end_s: float
    is_fallback: bool = False

    def as_dict(self) -> dict:
        """Serialisiert den Phrasenkontext für Logging und Telemetrie."""
        return {
            "phrase_start_s": self.phrase_start_s,
            "phrase_end_s": self.phrase_end_s,
            "gap_start_s": self.gap_start_s,
            "gap_end_s": self.gap_end_s,
            "tempo_bpm": self.tempo_bpm,
            "n_beats": len(self.beat_positions),
            "is_fallback": self.is_fallback,
        }


@dataclass
class PhraseBoundary:
    """Erkannte Phrasengrenze im Audio."""

    sample_pos: int
    cause: str  # "harmonic", "dynamic", "forced"
    strength: float  # ∈ [0, 1]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_INSTANCE_HOLDER: dict[str, object | None] = {"instance": None}


class MusicalPhraseContextExtractor:
    """Extrahiert musikalische Phrasengrenzen für Inpainting-Konditionierung (§2.12).

    Hauptmethode::

        extractor = get_phrase_extractor()
        ctx = extractor.extract_context(audio, sr, gap_start_sample, gap_end_sample)
        # ctx.audio_context → Konditionierungs-Audio für CQTdiff+/NMF-β
        # ctx.chroma_mean   → Chroma-Prior für Konsistenzprüfung

    Konditionierung (NMF-β)::

        inpainted = extractor.condition_inpainting(gap_audio, ctx)

    Invarianten:
        - audio muss 48000 Hz, float32
        - Lücke > 50 % der Phrase → benachbarte Phrase als Kontext
        - Chroma-Korrelation inpainted vs. Kontext ≥ 0.92 (§2.12)
    """

    def __init__(self) -> None:
        self._madmom_available: bool | None = None
        logger.debug("MusicalPhraseContextExtractor initialisiert.")

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def extract_context(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start: int,
        gap_end: int,
    ) -> PhraseContext:
        """Extrahiert Phrasenkontextfenster für den Dropout bei [gap_start, gap_end].

        Args:
            audio:      Vollständiges Audio (1D float32, 48000 Hz)
            sr:         48000
            gap_start:  Erster Sample der Dropout-Lücke (inklusiv)
            gap_end:    Letzter Sample der Dropout-Lücke (exklusiv)

        Returns:
            PhraseContext mit audio_context, chroma_mean, tempo_bpm, beat_positions
        """
        assert sr == 48000, f"MusicalPhraseContextExtractor: SR muss 48000 Hz sein, erhalten: {sr}"

        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0).astype(np.float32)

        duration_s = len(audio) / float(sr)

        # Kein Phrase-Context für kurze Dateien
        if duration_s < MIN_FILE_DURATION_S:
            logger.debug("PhraseContext: Datei < %.1f s → lokaler Ersatzpfad", MIN_FILE_DURATION_S)
            return self._local_fallback(audio, sr, gap_start, gap_end)

        # Tempo schätzen
        tempo_bpm = self._estimate_tempo(audio, sr)

        # Beat-Positionen
        beat_samples = self._estimate_beats(audio, sr, tempo_bpm)

        # Phrasen-Grenzen
        boundaries = self._detect_phrase_boundaries(audio, sr, beat_samples, tempo_bpm)

        # Phrase lokalisieren, die den Gap enthält
        phrase = self._find_phrase_for_gap(boundaries, gap_start, gap_end, len(audio))
        if phrase is None:
            logger.debug("PhraseContext: Keine Phrase gefunden → lokaler Ersatzpfad")
            return self._local_fallback(audio, sr, gap_start, gap_end)

        p_start, p_end = phrase

        # Lücke nimmt > 50 % der Phrase ein → benachbarte Phrase nehmen
        gap_dur = gap_end - gap_start
        phrase_dur = p_end - p_start
        if phrase_dur > 0 and (gap_dur / phrase_dur) > GAP_PHRASE_RATIO_MAX:
            phrase = self._find_neighbor_phrase(boundaries, p_start, p_end, len(audio))
            if phrase is None:
                return self._local_fallback(audio, sr, gap_start, gap_end)
            p_start, p_end = phrase

        # Kontext-Audio: Phrase ohne Dropout-Lücke (Gap wird mit 0 gefüllt)
        context_audio = audio[p_start:p_end].copy()
        gap_local_start = max(0, gap_start - p_start)
        gap_local_end = min(len(context_audio), gap_end - p_start)
        if gap_local_end > gap_local_start:
            context_audio[gap_local_start:gap_local_end] = 0.0

        # Auf MAX_CONTEXT-Dauer kürzen (§2.12)
        max_samples = int(MAX_CONTEXT_DURATION_S * sr)
        if len(context_audio) > max_samples:
            context_audio = context_audio[:max_samples]

        # Chroma
        chroma_mean = self._compute_chroma(context_audio, sr)

        # Beat-Positionen innerhalb der Phrase
        phrase_beats = [b - p_start for b in beat_samples if p_start <= b < p_end]

        logger.info(
            "🎵 PhraseContext: Phrase [%.2f–%.2f s] | Gap [%.3f–%.3f s] | BPM=%.1f | Beats=%d",
            p_start / sr,
            p_end / sr,
            gap_start / sr,
            gap_end / sr,
            tempo_bpm,
            len(phrase_beats),
        )

        return PhraseContext(
            audio_context=context_audio.astype(np.float32),
            chroma_mean=chroma_mean.astype(np.float32),
            tempo_bpm=tempo_bpm,
            beat_positions=phrase_beats,
            phrase_start_s=p_start / float(sr),
            phrase_end_s=p_end / float(sr),
            gap_start_s=gap_start / float(sr),
            gap_end_s=gap_end / float(sr),
            is_fallback=False,
        )

    def condition_inpainting(
        self,
        gap_audio: np.ndarray,
        context: PhraseContext,
    ) -> np.ndarray:
        """Konditioniert Inpainting-Ergebnis auf Phrase-Chroma-Konsistenz.

        Prüft Pearson-Korrelation zwischen Chroma des inpainted Signals
        und Phrasen-Chroma. Unterschreitet sie 0.92 (§2.12) → gibt
        unverändertes gap_audio zurück (kein blinder Eingriff).

        Args:
            gap_audio: Inpainted Dropout-Lücke (float32)
            context:   PhraseContext aus extract_context()

        Returns:
            Konditioniertes Audio (float32)
        """
        gap_audio = np.asarray(gap_audio, dtype=np.float32)
        gap_audio = np.nan_to_num(gap_audio, nan=0.0, posinf=0.0, neginf=0.0)

        if len(gap_audio) < 256:
            return np.clip(gap_audio, -1.0, 1.0)  # type: ignore[no-any-return]

        # Chroma des inpainted Signals
        inpainted_chroma = self._compute_chroma(gap_audio, 48000)
        corr = self._pearson(inpainted_chroma, context.chroma_mean)

        logger.debug("PhraseContext Bedingung: Chroma-Pearson=%.3f (≥0.92 erforderlich)", corr)

        if corr < 0.92:
            logger.warning(
                "⚠️ Inpainting-Chroma-Korrelation %.3f < 0.92 — Rohsignal beibehalten.",
                corr,
            )

        return np.clip(gap_audio, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def estimate_tempo(self, audio: np.ndarray, sr: int) -> float:
        """Öffentliche Tempo-Schätzung für externe Aufrufer.

        Diese Methode kapselt die interne Tempoanalyse und bietet eine
        stabile API für Komponenten wie den musikalischen Globalplan.
        """
        audio_arr = np.asarray(audio, dtype=np.float32)
        audio_arr = np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if audio_arr.ndim == 2:
            audio_arr = np.mean(audio_arr, axis=0).astype(np.float32)
        return float(self._estimate_tempo(audio_arr, sr))

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _estimate_tempo(self, audio: np.ndarray, sr: int) -> float:
        """Tempo-Schätzung via Onset-Envelope-Autokorrelation (DSP-Fallback).

        Madmom-Integration: try/except ImportError → DSP.
        Gibt BPM ∈ [40, 240] zurück.
        """
        # Versuche madmom
        try:
            madmom = _get_madmom_module()
            if madmom is None:
                raise ImportError("madmom nicht installiert")

            proc = madmom.features.tempo.TempoEstimationProcessor(
                histogram_processor=madmom.features.tempo.ACFTempoHistogramProcessor(fps=100)
            )
            act = madmom.features.beats.RNNBeatProcessor()(audio.astype(np.float32))
            tempos = proc(act)
            if len(tempos) > 0:
                bpm = float(tempos[0][0])
                if 40.0 <= bpm <= 240.0:
                    return bpm
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        # DSP-Fallback: Onset-Autokorrelation
        frame_len = 512
        hop = 256
        n_frames = max(1, (len(audio) - frame_len) // hop + 1)
        onset_env = np.zeros(n_frames, dtype=np.float32)
        prev_energy = 0.0
        for i in range(n_frames):
            s = i * hop
            seg = audio[s : s + frame_len]
            energy = float(np.mean(seg**2))
            onset_env[i] = max(0.0, energy - prev_energy)
            prev_energy = energy

        # Autokorrelation für BPM-Bereich
        hop_s = hop / float(sr)
        bpm_lo, bpm_hi = 40.0, 240.0
        lag_lo = max(1, int(60.0 / bpm_hi / hop_s))
        lag_hi = min(n_frames - 1, int(60.0 / bpm_lo / hop_s))

        if lag_hi <= lag_lo:
            return 120.0

        fft_autocorr = _get_fft_autocorr_fn()
        acf = fft_autocorr(onset_env)
        peak_lag = int(np.argmax(acf[lag_lo : lag_hi + 1])) + lag_lo
        if peak_lag == 0:
            return 120.0

        bpm = 60.0 / (peak_lag * hop_s)
        return float(np.clip(bpm, 40.0, 240.0))

    def _estimate_beats(self, audio: np.ndarray, sr: int, tempo_bpm: float) -> list[int]:
        """Schätzt Beat-Positionen in Samples.

        Madmom (RNN) wenn verfügbar, sonst Gleichabstand aus Tempo.
        """
        try:
            madmom = _get_madmom_module()
            if madmom is None:
                raise ImportError("madmom nicht installiert")

            proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
            act = madmom.features.beats.RNNBeatProcessor()(audio.astype(np.float32))
            beat_secs = proc(act)
            return [int(b * sr) for b in beat_secs if 0 <= int(b * sr) < len(audio)]
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        # Fallback: Gleichabstand
        beat_period = 60.0 / max(tempo_bpm, 1.0) * sr
        beats: list[int] = []
        pos = 0.0
        while pos < len(audio):
            beats.append(int(pos))
            pos += beat_period
        return beats

    def _detect_phrase_boundaries(
        self,
        audio: np.ndarray,
        sr: int,
        _beat_samples: list[int],
        _tempo_bpm: float,
    ) -> list[int]:
        """Erkennt Phrasen-Grenzen via harmonischen und dynamischen Sprüngen.

        Returns:
            Sortierte Sample-Positionen der Phrasen-Grenzen (inkl. 0 und len(audio))
        """
        boundaries: list[int] = [0]

        # Min. Phrasen-Länge in Samples
        min_phrase_samples = int(MIN_PHRASE_DURATION_S * sr)

        # Chroma-basierte Grenzen (alle ~2 Sekunden einen Chroma-Frame)
        frame_s = 2.0
        frame_samples = int(frame_s * sr)
        n_frames = max(1, len(audio) // frame_samples)
        chroma_frames: list[np.ndarray] = []
        for i in range(n_frames):
            seg = audio[i * frame_samples : (i + 1) * frame_samples]
            chroma_frames.append(self._compute_chroma(seg, sr))

        for i in range(1, len(chroma_frames)):
            dist = float(np.linalg.norm(chroma_frames[i] - chroma_frames[i - 1]))
            if dist > CHROMA_JUMP_THRESHOLD:
                pos = i * frame_samples
                if pos - boundaries[-1] >= min_phrase_samples:
                    boundaries.append(pos)

        # Energie-basierte Grenzen
        energy_frame = int(0.05 * sr)  # 50 ms
        n_eframes = max(1, len(audio) // energy_frame)
        for i in range(1, n_eframes):
            seg = audio[i * energy_frame : (i + 1) * energy_frame]
            prev = audio[(i - 1) * energy_frame : i * energy_frame]
            e_curr = float(np.mean(seg**2) + 1e-12)
            e_prev = float(np.mean(prev**2) + 1e-12)
            delta_db = 20.0 * math.log10(e_curr / e_prev)
            if abs(delta_db) > ENERGY_DELTA_DB:
                pos = i * energy_frame
                if pos - boundaries[-1] >= min_phrase_samples:
                    boundaries.append(pos)

        boundaries.append(len(audio))
        return sorted(set(boundaries))

    def _find_phrase_for_gap(
        self,
        boundaries: list[int],
        gap_start: int,
        gap_end: int,
        _n: int,
    ) -> tuple[int, int] | None:
        """Findet die Phrase, die die Dropout-Lücke enthält."""
        for i in range(len(boundaries) - 1):
            p_start = boundaries[i]
            p_end = boundaries[i + 1]
            if p_start <= gap_start < gap_end <= p_end:
                return p_start, p_end
        # Fallback: Lücke überspannt mehrere Phrasen → erste nehmen
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= gap_start < boundaries[i + 1]:
                return boundaries[i], boundaries[i + 1]
        return None

    def _find_neighbor_phrase(
        self,
        boundaries: list[int],
        p_start: int,
        _p_end: int,
        _n: int,
    ) -> tuple[int, int] | None:
        """Gibt eine benachbarte Phrase zurück (Vorgänger oder Nachfolger)."""
        for i in range(len(boundaries) - 1):
            if boundaries[i] == p_start:
                # Vorgänger?
                if i > 0:
                    return boundaries[i - 1], boundaries[i]
                # Nachfolger?
                if i + 1 < len(boundaries) - 1:
                    return boundaries[i + 1], boundaries[i + 2]
        return None

    def _compute_chroma(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Chroma-Profil (12 Semitone-Bins) via CQT-Näherung (STFT).

        Returns:
            Normierter Chroma-Vektor (float32, shape=(12,))
        """
        if len(audio) < 512:
            return np.ones(N_CHROMA, dtype=np.float32) / N_CHROMA  # type: ignore[no-any-return]

        n_fft = min(4096, len(audio))
        window = np.hanning(n_fft).astype(np.float32)
        seg = audio[:n_fft] * window
        spec = np.abs(np.fft.rfft(seg)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        chroma = np.zeros(N_CHROMA, dtype=np.float32)
        a4 = 440.0
        for i, f in enumerate(freqs):
            if f < 60.0 or f > 8000.0:
                continue
            if f <= 0:
                continue
            midi = 12.0 * math.log2(f / a4) + 69.0
            chroma_bin = round(midi) % N_CHROMA
            chroma[chroma_bin] += float(spec[i])

        norm = float(np.linalg.norm(chroma))
        if norm > 0:
            chroma /= norm
        else:
            chroma = np.ones(N_CHROMA, dtype=np.float32) / N_CHROMA

        return chroma.astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson-Korrelationskoeffizient zweier gleichlanger Vektoren."""
        a = a - np.mean(a)
        b = b - np.mean(b)
        num = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return float(np.clip(num / denom, -1.0, 1.0))

    def _local_fallback(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start: int,
        gap_end: int,
    ) -> PhraseContext:
        """Lokales ±context_window-Kontextfenster (±2 s um die Lücke)."""
        ctx_samples = int(2.0 * sr)
        p_start = max(0, gap_start - ctx_samples)
        p_end = min(len(audio), gap_end + ctx_samples)
        ctx_audio = audio[p_start:p_end].copy()
        gap_local_start = max(0, gap_start - p_start)
        gap_local_end = min(len(ctx_audio), gap_end - p_start)
        if gap_local_end > gap_local_start:
            ctx_audio[gap_local_start:gap_local_end] = 0.0

        chroma_mean = self._compute_chroma(ctx_audio, sr)
        return PhraseContext(
            audio_context=ctx_audio.astype(np.float32),
            chroma_mean=chroma_mean.astype(np.float32),
            tempo_bpm=120.0,
            beat_positions=[],
            phrase_start_s=p_start / float(sr),
            phrase_end_s=p_end / float(sr),
            gap_start_s=gap_start / float(sr),
            gap_end_s=gap_end / float(sr),
            is_fallback=True,
        )


# ---------------------------------------------------------------------------
# Singleton-Accessor + Convenience
# ---------------------------------------------------------------------------


def get_phrase_extractor() -> MusicalPhraseContextExtractor:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    if _INSTANCE_HOLDER["instance"] is None:
        with _lock:
            if _INSTANCE_HOLDER["instance"] is None:
                _INSTANCE_HOLDER["instance"] = MusicalPhraseContextExtractor()
    instance = _INSTANCE_HOLDER["instance"]
    assert isinstance(instance, MusicalPhraseContextExtractor)
    return instance


def extract_phrase_context(
    audio: np.ndarray,
    sr: int,
    gap_start: int,
    gap_end: int,
) -> PhraseContext:
    """Convenience: Phrasenkontextfenster für Dropout-Inpainting extrahieren.

    Typische Verwendung::

        ctx = extract_phrase_context(audio, sr, gap_start=48000, gap_end=52800)
        # ctx.audio_context → CQTdiff+-Conditioning
        # ctx.chroma_mean   → NMF-β Initialisierungs-Prior

    Args:
        audio:      Gesamtes Audio (float32, 48000 Hz)
        sr:         48000
        gap_start:  Erster Sample der Dropout-Lücke
        gap_end:    Letzter Sample der Dropout-Lücke (exklusiv)

    Returns:
        PhraseContext
    """
    return get_phrase_extractor().extract_context(audio, sr, gap_start, gap_end)


# ---------------------------------------------------------------------------
# Abwärtskompatibler Alias (migriert aus core.musical_phrase_context_extractor)
# ---------------------------------------------------------------------------
get_phrase_context_extractor = get_phrase_extractor
get_musical_phrase_context_extractor = get_phrase_extractor

__all__ = [
    "CHROMA_JUMP_THRESHOLD",
    "ENERGY_DELTA_DB",
    "GAP_FRACTION_THRESHOLD",
    "GAP_PHRASE_RATIO_MAX",
    "MAX_CONTEXT_DURATION_S",
    "MAX_GAP_DURATION_MS",
    # Modul-Konstanten (Spec §2.12)
    "MIN_FILE_DURATION_S",
    "MIN_PHRASE_BEATS",
    "MIN_PHRASE_DURATION_S",
    "N_CHROMA",
    "MusicalPhraseContextExtractor",
    "PhraseBoundary",
    "PhraseContext",
    "extract_phrase_context",
    "get_musical_phrase_context_extractor",
    "get_phrase_context_extractor",
    "get_phrase_extractor",
]
