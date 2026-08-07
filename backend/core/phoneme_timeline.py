"""
backend/core/phoneme_timeline.py
Aurik 10.0.0 — §2.36a PhonemeTimeline

Pipeline-Datenstruktur für phoneme-aware Restaurierung.
Bietet zeitlich aufgelöste Phonemklassen-Information an alle konsumierenden Phasen.

Sprachdetektion: DSP-basiert (LPC-Formanten-Analyse), SR-agnostisch.
IPA-Symbole: nur wenn Decoder läuft (has_ipa=False für aktuellen Stand ohne Decoder-ONNX).

Singleton-Zugriff: get_phoneme_timeline_builder()

Referenzen:
    Peterson & Barney 1952; Hillenbrand 1995; Sendlmeier & Seebode 2006.

Spec §2.36a, §3.2 (Singleton DCL), §3.1 (NaN-Guard), §13.3 (offline)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ─── IPA classification sets ─────────────────────────────────────────────────

SIBILANT_IPA: frozenset[str] = frozenset({"s", "z", "ʃ", "ʒ", "ts", "dz", "tʃ", "dʒ"})
FRICATIVE_IPA: frozenset[str] = frozenset({"f", "v", "θ", "ð", "x", "ɣ", "h", "ç"})
PLOSIVE_IPA: frozenset[str] = frozenset({"p", "b", "t", "d", "k", "g", "ʔ"})
VOWEL_IPA: frozenset[str] = frozenset(
    {
        "a",
        "e",
        "i",
        "o",
        "u",
        "ɐ",
        "ɑ",
        "æ",
        "ɒ",
        "ə",
        "ɛ",
        "ɪ",
        "ɨ",
        "ɔ",
        "ø",
        "œ",
        "ʊ",
        "ʌ",
        "iː",
        "uː",
        "eː",
        "oː",
        "aː",
        "ü",
        "ö",
        "ä",
        "y",
    }
)


# ─── Formant reference tables (F1 Hz, F2 Hz) per language / vowel ────────────
# Sources: Peterson & Barney 1952, Hillenbrand 1995, Sendlmeier & Seebode 2006
_FORMANT_TABLE: dict[str, dict[str, tuple[float, float]]] = {
    "de": {
        "a": (800.0, 1200.0),
        "e": (390.0, 2300.0),
        "i": (270.0, 2750.0),
        "o": (440.0, 900.0),
        "u": (300.0, 800.0),
        "ä": (590.0, 2000.0),
        "ö": (450.0, 1550.0),
        "ü": (300.0, 1950.0),
        "ə": (500.0, 1500.0),
    },
    "en": {
        "ɑ": (750.0, 1150.0),
        "æ": (660.0, 1720.0),
        "ɪ": (430.0, 2480.0),
        "iː": (280.0, 2620.0),
        "ɒ": (570.0, 830.0),
        "uː": (300.0, 870.0),
        "ʊ": (440.0, 1020.0),
        "ɛ": (580.0, 1800.0),
        "ʌ": (700.0, 1250.0),
        "ə": (500.0, 1500.0),
    },
    "fr": {
        "a": (780.0, 1300.0),
        "e": (420.0, 2200.0),
        "i": (290.0, 2700.0),
        "o": (420.0, 960.0),
        "u": (290.0, 840.0),
        "ø": (460.0, 1700.0),
        "œ": (540.0, 1600.0),
        "y": (280.0, 1960.0),
        "ə": (490.0, 1600.0),
    },
    "it": {
        "a": (760.0, 1240.0),
        "e": (410.0, 2150.0),
        "i": (280.0, 2680.0),
        "o": (430.0, 920.0),
        "u": (310.0, 820.0),
        "ə": (490.0, 1550.0),
    },
    "es": {
        "a": (770.0, 1220.0),
        "e": (400.0, 2150.0),
        "i": (285.0, 2720.0),
        "o": (425.0, 910.0),
        "u": (305.0, 810.0),
    },
}


# ─── Sibilant bands per language (Hz) ────────────────────────────────────────
_SIBILANT_BAND_MAP: dict[str, tuple[float, float]] = {
    "de": (5500.0, 8500.0),
    "en": (5000.0, 8000.0),
    "fr": (4800.0, 7500.0),
    "it": (5000.0, 8000.0),
    "es": (4500.0, 7000.0),
    "unknown": (4000.0, 8000.0),
}

# Mapping from LyricsTranscriptionResult.phoneme_type to PhonemeTimelineSegment.phoneme_class
_PTYPE_TO_CLASS: dict[str, str] = {
    "fricative_stressed": "fricative_stressed",
    "fricative_unstressed": "sibilant",  # treat unstressed fricatives as sibilant
    "fricative": "sibilant",  # legacy alias used by LyricsTranscriber (plugin)
    "plosive": "plosive",
    "vowel_stressed": "vowel_stressed",
    "vowel_unstressed": "vowel_unstressed",
    "vowel": "vowel_unstressed",  # legacy alias
    "silence": "silence",
    "mixed": "silence",
}


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class PhonemeTimelineSegment:
    """single phoneme segment with timing, class, and confidence.

    Attributes:
        start_s:       Start time in seconds.
        end_s:         End time in seconds.
        phoneme_class: One of "fricative_stressed" | "plosive" | "vowel_stressed" |
                       "vowel_unstressed" | "silence" | "sibilant".
        phoneme_ipa:   IPA symbol when available (e.g. "s", "ʃ", "a"); empty string
                       when has_ipa=False (no decoder ONNX loaded).
        confidence:    Detection confidence 0.0–1.0.
        is_stressed:   True when the segment carries lexical stress.
    """

    start_s: float
    end_s: float
    phoneme_class: str
    phoneme_ipa: str
    confidence: float
    is_stressed: bool


@dataclass
class PhonemeTimeline:
    """Language-aware phoneme timeline for a full audio file.

    Built from LyricsTranscriptionResult; consumed by phase_19, phase_24,
    phase_43, phase_56 and MDEM for phoneme-targeted processing.

    Singleton factory: get_phoneme_timeline_builder().
    """

    language: str  # ISO 639-1: "de"|"en"|"fr"|"it"|"es"|"unknown"
    language_confidence: float  # 0.0–1.0
    segments: list[PhonemeTimelineSegment]
    duration_s: float
    has_ipa: bool  # True only when IPA symbols are populated

    # ── Query methods ─────────────────────────────────────────────────────

    def segments_in_range(self, start_s: float, end_s: float) -> list[PhonemeTimelineSegment]:
        """Gibt all segments that overlap with [start_s, end_s] zurück.

        Args:
            start_s: Window start in seconds.
            end_s:   Window end in seconds.

        Returns:
            Overlapping segments sorted by start_s; empty list on invalid range.
        """
        if not (np.isfinite(start_s) and np.isfinite(end_s)) or end_s <= start_s:
            return []
        return [s for s in self.segments if s.end_s > start_s and s.start_s < end_s]

    def sibilant_segments(self) -> list[PhonemeTimelineSegment]:
        """Gibt segments representing sibilant or fricative sounds zurück.

        Used by de-esser phases (19 + 43) for targeted processing.

        Returns:
            Segments where phoneme_class ∈ {"sibilant", "fricative_stressed",
            "fricative_unstressed"}.
        """
        sibilant_classes = {"sibilant", "fricative_stressed", "fricative_unstressed"}
        return [s for s in self.segments if s.phoneme_class in sibilant_classes]

    def stressed_vowel_segments(self) -> list[PhonemeTimelineSegment]:
        """Gibt vowel segments with lexical stress zurück.

        Used by MDEM and phase_56 for formant-guided processing.

        Returns:
            Segments where phoneme_class == "vowel_stressed".
        """
        return [s for s in self.segments if s.phoneme_class == "vowel_stressed"]

    def formant_target_for_range(self, start_s: float, end_s: float) -> tuple[float, float] | None:
        """Gibt (F1_hz, F2_hz) for the dominant vowel phoneme in [start_s, end_s] zurück.

        Looks up the language-specific formant reference table for the most
        confident vowel segment in the range.

        If has_ipa is False or no vowel segments exist, returns None.

        Args:
            start_s: Window start in seconds.
            end_s:   Window end in seconds.

        Returns:
            (F1_hz, F2_hz) tuple or None.
        """
        if not (np.isfinite(start_s) and np.isfinite(end_s)):
            return None
        vowel_classes = {"vowel_stressed", "vowel_unstressed"}
        vowel_segs = [s for s in self.segments_in_range(start_s, end_s) if s.phoneme_class in vowel_classes]
        if not vowel_segs:
            return None

        best = max(vowel_segs, key=lambda s: s.confidence)

        lang = self.language if self.language in _FORMANT_TABLE else "de"
        table = _FORMANT_TABLE[lang]

        if best.phoneme_ipa and best.phoneme_ipa in table:
            f1, f2 = table[best.phoneme_ipa]
            return (
                float(np.nan_to_num(f1, nan=500.0)),
                float(np.nan_to_num(f2, nan=1500.0)),
            )
        # Fallback: schwa as generic vowel centroid
        f1, f2 = table.get("ə", (500.0, 1500.0))
        return (
            float(np.nan_to_num(f1, nan=500.0)),
            float(np.nan_to_num(f2, nan=1500.0)),
        )

    def sibilant_band_hz(self) -> tuple[float, float]:
        """Gibt (f_low_hz, f_high_hz) of expected sibilant energy for detected language zurück.

        Returns:
            Frequency band tuple from _SIBILANT_BAND_MAP, defaulting to (4000, 8000).
        """
        return _SIBILANT_BAND_MAP.get(self.language, _SIBILANT_BAND_MAP["unknown"])

    # ── Factory methods ───────────────────────────────────────────────────

    @classmethod
    def build_empty(cls, duration_s: float = 0.0) -> PhonemeTimeline:
        """Erstellt an empty PhonemeTimeline with safe defaults.

        Args:
            duration_s: Audio duration in seconds (NaN-safe: clamps to 0.0).

        Returns:
            Empty PhonemeTimeline with language="unknown", no segments.
        """
        safe_dur = float(np.nan_to_num(float(duration_s), nan=0.0, posinf=0.0, neginf=0.0))
        return cls(
            language="unknown",
            language_confidence=0.0,
            segments=[],
            duration_s=max(0.0, safe_dur),
            has_ipa=False,
        )

    @classmethod
    def build_from_transcription(
        cls,
        result: object,  # LyricsTranscriptionResult
        language: str = "unknown",
    ) -> PhonemeTimeline:
        """Erstellt PhonemeTimeline from a LyricsTranscriptionResult.

        Maps WordTimestamp.phoneme_type → PhonemeTimelineSegment.phoneme_class:
            fricative_stressed  → "fricative_stressed"
            fricative_unstressed → "sibilant"
            fricative           → "sibilant"  (legacy alias)
            plosive             → "plosive"
            vowel_stressed      → "vowel_stressed"
            vowel_unstressed    → "vowel_unstressed"
            vowel               → "vowel_unstressed"  (legacy alias)
            silence / mixed     → "silence"

        has_ipa is always False (no Decoder-ONNX available; IPA empty string).

        Args:
            result:   LyricsTranscriptionResult with .words and .duration_s.
            language: ISO 639-1 code from result.language.

        Returns:
            PhonemeTimeline with one segment per WordTimestamp.
        """
        safe_lang = str(language or "unknown").lower().strip()
        if safe_lang not in _FORMANT_TABLE and safe_lang != "unknown":
            safe_lang = "unknown"

        duration_s = float(np.nan_to_num(float(getattr(result, "duration_s", 0.0)), nan=0.0))
        words = getattr(result, "words", None) or []
        segments: list[PhonemeTimelineSegment] = []

        for word in words:
            ptype = str(getattr(word, "phoneme_type", "silence") or "silence")
            pclass = _PTYPE_TO_CLASS.get(ptype, "silence")

            start_s = float(np.nan_to_num(float(getattr(word, "start_s", 0.0)), nan=0.0))
            end_s = float(np.nan_to_num(float(getattr(word, "end_s", 0.0)), nan=0.0))
            if end_s <= start_s:
                continue

            conf = float(
                np.clip(
                    np.nan_to_num(float(getattr(word, "confidence", 0.5)), nan=0.5),
                    0.0,
                    1.0,
                )
            )
            is_stressed = bool(getattr(word, "is_stressed", False))

            # Promote unstressed vowel to stressed if flag is set
            if pclass == "vowel_unstressed" and is_stressed:
                pclass = "vowel_stressed"

            segments.append(
                PhonemeTimelineSegment(
                    start_s=start_s,
                    end_s=end_s,
                    phoneme_class=pclass,
                    phoneme_ipa="",  # IPA not available without Decoder-ONNX
                    confidence=conf,
                    is_stressed=is_stressed,
                )
            )

        segments.sort(key=lambda s: s.start_s)

        lang_conf = float(
            np.clip(
                np.nan_to_num(float(getattr(result, "overall_confidence", 0.0)), nan=0.0),
                0.0,
                1.0,
            )
        )

        return cls(
            language=safe_lang,
            language_confidence=lang_conf,
            segments=segments,
            duration_s=duration_s,
            has_ipa=False,
        )


# ─── Language detection ──────────────────────────────────────────────────────


def _detect_language(mono: np.ndarray, sr: int = 16_000) -> tuple[str, float]:
    """Erkennt spoken language from audio via LPC formant analysis.

    Algorithm (DSP-only, SR-agnostic, no ML model):
        1. Limit to first 30 s of audio; no assert sr==48000 (analysis-agnostic).
        2. Compute 25 ms frames with 10 ms hop.
        3. Keep only voiced frames (ZCR < 0.15).
        4. For each voiced frame: LPC ord=12 → complex roots → F1/F2
           (filter roots inside the unit circle with positive imaginary part).
        5. Collect (F1, F2) pairs across all voiced frames.
        6. Compute mean (F1_mean, F2_mean) as observed vowel centroid.
        7. Compute scaled Euclidean distance to each language's formant centroid.
        8. Language with minimum distance wins.
        9. Confidence: 1.0 − (best_dist / 2.0), clipped to [0, 1].
           If confidence < 0.35 → "unknown".
       10. Fallback on any exception → ("unknown", 0.0).

    Args:
        mono: 1-D float32/64 audio (any sample rate, analysis-agnostic).
        sr:   Sample rate of mono in Hz (default 16 000 Hz).

    Returns:
        (language_code, confidence) where language_code ∈
        {"de", "en", "fr", "it", "es", "unknown"} and confidence ∈ [0.0, 1.0].
    """
    try:
        mono_f = np.nan_to_num(np.asarray(mono, dtype=np.float32))
        if mono_f.ndim != 1 or len(mono_f) < 64:
            return ("unknown", 0.0)

        max_samples = int(30 * sr)
        audio = mono_f[:max_samples].astype(np.float64)

        frame_size = max(64, int(0.025 * sr))  # 25 ms
        hop = max(32, int(0.010 * sr))  # 10 ms
        lpc_order = max(16, min(40, int(sr / 1200)))  # §4.5: >= 16, 30-40 @ 48 kHz

        formant_pairs: list[tuple[float, float]] = []

        for i in range(0, max(1, len(audio) - frame_size), hop):
            frame = audio[i : i + frame_size]
            if len(frame) < lpc_order + 2:
                continue

            # Voiced gate: low zero-crossing rate
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) / 2.0))
            if zcr >= 0.15:
                continue

            # LPC via scipy (preferred) with numpy fallback
            roots: np.ndarray | None = None
            try:
                from scipy.signal import lpc as _scipy_lpc  # pylint: disable=no-name-in-module

                A = _scipy_lpc(frame - np.mean(frame), lpc_order)
                if not np.isfinite(A).all():
                    continue
                roots = np.roots(A)
            except Exception:
                try:
                    from backend.core.core_utils import fft_autocorr

                    r = fft_autocorr(frame, max_lag=lpc_order)
                    R = np.array([[r[abs(ii - jj)] for jj in range(lpc_order)] for ii in range(lpc_order)])
                    r_vec = -r[1 : lpc_order + 1]
                    if np.linalg.matrix_rank(R) < lpc_order:
                        continue
                    a_lpc = np.linalg.solve(R, r_vec)
                    coeff = np.concatenate([[1.0], a_lpc])
                    roots = np.roots(coeff)
                except Exception:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                    continue

            if roots is None:
                continue

            # Select roots inside unit circle with positive imaginary part
            voiced_roots = roots[(np.abs(roots) < 1.0) & (np.imag(roots) > 0.01)]
            freqs = np.angle(voiced_roots) * float(sr) / (2.0 * np.pi)
            freqs = np.sort(freqs[freqs > 50.0])

            if len(freqs) < 2:
                continue

            f1 = float(freqs[0])
            f2 = float(freqs[1])
            if not (50.0 < f1 < 1200.0 and 500.0 < f2 < 3500.0):
                continue

            formant_pairs.append((f1, f2))

        if len(formant_pairs) < 3:
            return ("unknown", 0.0)

        pairs_arr = np.array(formant_pairs, dtype=np.float64)
        mean_f1 = float(np.mean(pairs_arr[:, 0]))
        mean_f2 = float(np.mean(pairs_arr[:, 1]))

        # Euclidean distance (normalised: F1/400 Hz, F2/800 Hz → similar scales)
        distances: dict[str, float] = {}
        for lang, vowels in _FORMANT_TABLE.items():
            if not vowels:
                continue
            vf1 = np.array([v[0] for v in vowels.values()], dtype=np.float64)
            vf2 = np.array([v[1] for v in vowels.values()], dtype=np.float64)
            cent_f1 = float(np.mean(vf1))
            cent_f2 = float(np.mean(vf2))
            d = float(np.sqrt(((mean_f1 - cent_f1) / 400.0) ** 2 + ((mean_f2 - cent_f2) / 800.0) ** 2))
            distances[lang] = d

        if not distances:
            return ("unknown", 0.0)

        best_lang = min(distances, key=distances.__getitem__)
        best_dist = distances[best_lang]
        confidence = float(max(0.0, 1.0 - best_dist / 2.0))

        if confidence < 0.35:
            return ("unknown", 0.0)

        return (best_lang, float(np.clip(confidence, 0.0, 1.0)))

    except Exception as exc:
        logger.debug("_erkennen_language fehlgeschlagen: %s", exc)
        return ("unknown", 0.0)


# ─── Singleton ───────────────────────────────────────────────────────────────


class PhonemeTimelineBuilder:
    """Convenience wrapper around PhonemeTimeline factory methods.

    Singleton access via get_phoneme_timeline_builder() (§3.2 DCL pattern).
    """

    def detect_language(self, mono: np.ndarray, sr: int = 16_000) -> tuple[str, float]:
        """Erkennt language via LPC formant analysis (SR-agnostic).

        Delegates to module-level _detect_language().

        Args:
            mono: 1-D float32 audio at any sample rate.
            sr:   Sample rate in Hz.

        Returns:
            (language_code, confidence) tuple.
        """
        return _detect_language(mono, sr)

    def build_empty(self, duration_s: float = 0.0) -> PhonemeTimeline:
        """Erstellt empty PhonemeTimeline."""
        return PhonemeTimeline.build_empty(duration_s)

    def build_from_transcription(self, result: object, language: str = "unknown") -> PhonemeTimeline:
        """Erstellt PhonemeTimeline from LyricsTranscriptionResult."""
        return PhonemeTimeline.build_from_transcription(result, language)


_builder_instance: PhonemeTimelineBuilder | None = None
_builder_lock = threading.Lock()


def get_phoneme_timeline_builder() -> PhonemeTimelineBuilder:
    """Gibt thread-safe Singleton PhonemeTimelineBuilder (§3.2 DCL) zurück.

    Returns:
        Module-level PhonemeTimelineBuilder instance.
    """
    global _builder_instance
    if _builder_instance is None:
        with _builder_lock:
            if _builder_instance is None:
                _builder_instance = PhonemeTimelineBuilder()
    return _builder_instance
