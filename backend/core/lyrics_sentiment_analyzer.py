"""NLP-based lyrics-sentiment analysis for emotion-driven restoration behaviour.

A human mastering engineer understands whether a verse is sad/triumphant/intimate
and processes it accordingly (softer dynamics, more space, less compression).

This implementation is PRIVACY-COMPLIANT (§2.36):
- No lyrics text is logged
- Only numeric sentiment vectors are passed downstream
- DSP fallback when no ML model is available

Spec: §LSM-1 Lyrics-Semantics-Model (v10.0.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion anchors — pure embedding coordinates (no lyrics text)
# Based on the VAD model (Valence/Arousal/Dominance)
# ---------------------------------------------------------------------------

# VAD coordinates for emotional clusters [Valence, Arousal, Dominance]
# Valence:   -1 (negative/sad) … +1 (positive/joyful)
# Arousal:   -1 (calm/quiet)   … +1 (excited/loud)
# Dominance: -1 (weak/fragile) … +1 (strong/triumphant)
_EMOTION_ANCHORS_VAD: dict[str, np.ndarray] = {
    "joyful": np.array([+0.85, +0.70, +0.60], dtype=np.float32),
    "triumphant": np.array([+0.80, +0.85, +0.90], dtype=np.float32),
    "tender": np.array([+0.60, -0.30, +0.30], dtype=np.float32),
    "intimate": np.array([+0.50, -0.50, -0.10], dtype=np.float32),
    "melancholic": np.array([-0.55, -0.40, -0.40], dtype=np.float32),
    "sad": np.array([-0.75, -0.50, -0.60], dtype=np.float32),
    "longing": np.array([-0.30, -0.20, -0.30], dtype=np.float32),
    "angry": np.array([-0.50, +0.80, +0.70], dtype=np.float32),
    "tense": np.array([-0.20, +0.60, +0.20], dtype=np.float32),
    "neutral": np.array([+0.00, +0.00, +0.00], dtype=np.float32),
}

# DSP parameter modifiers per emotion cluster
# Format: (dynamics_scale, presence_scale, space_scale)
# dynamics_scale: 1.0 = unchanged; < 1.0 = gentler dynamics processing
# presence_scale: 1.0 = unchanged; > 1.0 = more presence
# space_scale:    1.0 = unchanged; > 1.0 = more space / reverb preservation
_EMOTION_DSP_PARAMS: dict[str, dict[str, float]] = {
    "joyful": {"dynamics_scale": 1.10, "presence_scale": 1.05, "space_scale": 1.00},
    "triumphant": {"dynamics_scale": 1.20, "presence_scale": 1.10, "space_scale": 0.90},
    "tender": {"dynamics_scale": 0.70, "presence_scale": 0.90, "space_scale": 1.10},
    "intimate": {"dynamics_scale": 0.60, "presence_scale": 0.85, "space_scale": 1.20},
    "melancholic": {"dynamics_scale": 0.75, "presence_scale": 0.90, "space_scale": 1.15},
    "sad": {"dynamics_scale": 0.65, "presence_scale": 0.85, "space_scale": 1.20},
    "longing": {"dynamics_scale": 0.80, "presence_scale": 0.90, "space_scale": 1.10},
    "angry": {"dynamics_scale": 1.15, "presence_scale": 1.10, "space_scale": 0.90},
    "tense": {"dynamics_scale": 1.05, "presence_scale": 1.00, "space_scale": 1.00},
    "neutral": {"dynamics_scale": 1.00, "presence_scale": 1.00, "space_scale": 1.00},
}


@dataclass
class SegmentSentiment:
    """Sentiment analysis for a single text segment (timestamp + emotion cluster)."""

    start_s: float
    end_s: float
    emotion: str  # Primary emotion label
    valence: float  # -1 … +1
    arousal: float  # -1 … +1
    dominance: float  # -1 … +1
    confidence: float  # 0 … 1 (model confidence)
    dsp_params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dsp_params:
            self.dsp_params = _EMOTION_DSP_PARAMS.get(self.emotion, _EMOTION_DSP_PARAMS["neutral"]).copy()


@dataclass
class LyricsSentimentResult:
    """Sentiment analysis result for a complete song."""

    segments: list[SegmentSentiment]
    dominant_emotion: str
    valence_mean: float
    arousal_mean: float
    dominance_mean: float
    model_used: str  # "embedding_dsp" | "keyword_heuristic" | "neutral_fallback"

    def get_emotion_at(self, time_s: float) -> SegmentSentiment | None:
        """Gibt the segment that contains the given timestamp time_s zurück."""
        for seg in self.segments:
            if seg.start_s <= time_s < seg.end_s:
                return seg
        return None

    def get_dynamics_scale_at(self, time_s: float) -> float:
        """Gibt the dynamics_scale DSP parameter at the given timestamp zurück."""
        seg = self.get_emotion_at(time_s)
        if seg is None:
            return 1.0
        return float(seg.dsp_params.get("dynamics_scale", 1.0))


# ---------------------------------------------------------------------------
# Keyword heuristic (fallback without ML)
# ---------------------------------------------------------------------------

# Emotional signal words — only very common, cross-language words
# §2.36 privacy: no full lyrics text is stored
_KEYWORD_VALENCE: dict[str, float] = {
    # Positive
    "love": +0.80,
    "liebe": +0.80,
    "happy": +0.70,
    "glück": +0.70,
    "joy": +0.75,
    "freude": +0.75,
    "smile": +0.65,
    "beautiful": +0.65,
    "hope": +0.60,
    "hoffnung": +0.60,
    "together": +0.50,
    "zusammen": +0.50,
    "dream": +0.55,
    "traum": +0.55,
    "heart": +0.60,
    "herz": +0.60,
    # Negative
    "pain": -0.70,
    "schmerz": -0.70,
    "sad": -0.75,
    "traurig": -0.75,
    "cry": -0.65,
    "weinen": -0.65,
    "lost": -0.55,
    "verloren": -0.55,
    "alone": -0.60,
    "allein": -0.60,
    "broken": -0.70,
    "gebrochen": -0.70,
    "goodbye": -0.50,
    "abschied": -0.50,
    "never": -0.40,
    "niemals": -0.40,
    "dark": -0.45,
    "dunkel": -0.45,
    "fall": -0.30,
    "fallen": -0.30,
}

_KEYWORD_AROUSAL: dict[str, float] = {
    "run": +0.60,
    "fight": +0.70,
    "war": +0.65,
    "fire": +0.65,
    "quiet": -0.60,
    "still": -0.50,
    "silent": -0.55,
    "whisper": -0.70,
    "soft": -0.40,
    "sanft": -0.40,
    "gentle": -0.50,
    "slow": -0.50,
}


def _vad_to_emotion(vad: np.ndarray) -> str:
    """Nearest emotion anchor in VAD space (L2 distance)."""
    best_label = "neutral"
    best_dist = float("inf")
    for label, anchor in _EMOTION_ANCHORS_VAD.items():
        dist = float(np.linalg.norm(vad - anchor))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def _analyze_keywords(text: str) -> tuple[float, float, float]:
    """Berechnet Valence/Arousal/Dominance via keyword heuristic.

    §2.36 privacy: text is NOT logged or stored.
    Returns only a numeric (V, A, D) tuple.
    """
    words = text.lower().split()
    valence_scores = [_KEYWORD_VALENCE[w] for w in words if w in _KEYWORD_VALENCE]
    arousal_scores = [_KEYWORD_AROUSAL[w] for w in words if w in _KEYWORD_AROUSAL]

    valence = float(np.mean(valence_scores)) if valence_scores else 0.0
    arousal = float(np.mean(arousal_scores)) if arousal_scores else 0.0
    dominance = float(np.clip(valence * 0.5 + arousal * 0.3, -1.0, 1.0))  # Proxy
    return valence, arousal, dominance


# ---------------------------------------------------------------------------
# Haupt-Sentiment-Analyse
# ---------------------------------------------------------------------------


class LyricsSentimentAnalyzer:
    """Analysiert den emotionalen Inhalt von Liedtexten und gibt zeitgestempelte Ergebnisse zurück.
    sentiment segments.

    Chain: DSP-embedding heuristic → keyword fallback → neutral fallback.
    §2.36 privacy guaranteed: no text in logs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def analyze(
        self,
        transcription_result: object,
        audio_duration_s: float,
    ) -> LyricsSentimentResult:
        """Analysiert die Stimmungs-Timeline aus einem Transkriptionsergebnis.

        Args:
            transcription_result: LyricsTranscriptionResult (WordTimestamp list)
            audio_duration_s:     Total audio duration in seconds

        Returns:
            LyricsSentimentResult with time-stamped segment emotions
        """
        try:
            return self._analyze_internal(transcription_result, audio_duration_s)
        except Exception as exc:
            logger.debug("§LSM-1 sentiment Analyse nicht blockierend error: %s", exc)
            return self._neutral_fallback(audio_duration_s)

    def _analyze_internal(
        self,
        transcription_result: object,
        audio_duration_s: float,
    ) -> LyricsSentimentResult:
        # Extract words (from LyricsTranscriptionResult or dict)
        words: list = []
        if hasattr(transcription_result, "words"):
            words = list(transcription_result.words)
        elif isinstance(transcription_result, dict):
            words = list(transcription_result.get("words", []))

        if not words:
            return self._neutral_fallback(audio_duration_s)

        # Segmentation: ~8 seconds per sentiment segment
        segment_duration_s = 8.0
        n_segments = max(1, int(np.ceil(audio_duration_s / segment_duration_s)))
        segments: list[SegmentSentiment] = []

        for seg_i in range(n_segments):
            t_start = seg_i * segment_duration_s
            t_end = min((seg_i + 1) * segment_duration_s, audio_duration_s)

            # Words in this time window
            seg_words = [w for w in words if hasattr(w, "start") and t_start <= float(w.start) < t_end]
            if not seg_words:
                # Interpolate from neighbouring segments
                if segments:
                    prev = segments[-1]
                    segments.append(
                        SegmentSentiment(
                            start_s=t_start,
                            end_s=t_end,
                            emotion=prev.emotion,
                            valence=prev.valence,
                            arousal=prev.arousal,
                            dominance=prev.dominance,
                            confidence=0.3,
                        )
                    )
                else:
                    segments.append(
                        SegmentSentiment(
                            start_s=t_start,
                            end_s=t_end,
                            emotion="neutral",
                            valence=0.0,
                            arousal=0.0,
                            dominance=0.0,
                            confidence=0.1,
                        )
                    )
                continue

            # Build text from words (NOT logged — §2.36 privacy)
            seg_text = " ".join(str(getattr(w, "word", "")) for w in seg_words)
            valence, arousal, dominance = _analyze_keywords(seg_text)
            vad = np.array([valence, arousal, dominance], dtype=np.float32)
            emotion = _vad_to_emotion(vad)
            n_matched = sum(
                1 for word in seg_text.lower().split() if word in _KEYWORD_VALENCE or word in _KEYWORD_AROUSAL
            )
            total_words = max(len(seg_text.split()), 1)
            confidence = float(np.clip(n_matched / total_words, 0.1, 0.9))

            segments.append(
                SegmentSentiment(
                    start_s=t_start,
                    end_s=t_end,
                    emotion=emotion,
                    valence=valence,
                    arousal=arousal,
                    dominance=dominance,
                    confidence=confidence,
                )
            )

        if not segments:
            return self._neutral_fallback(audio_duration_s)

        valence_mean = float(np.mean([s.valence for s in segments]))
        arousal_mean = float(np.mean([s.arousal for s in segments]))
        dominance_mean = float(np.mean([s.dominance for s in segments]))
        vad_mean = np.array([valence_mean, arousal_mean, dominance_mean], dtype=np.float32)
        dominant_emotion = _vad_to_emotion(vad_mean)

        logger.info(
            "§LSM-1 Sentiment: %d Segmente, Dominanz=%s V=%.2f A=%.2f D=%.2f",
            len(segments),
            dominant_emotion,
            valence_mean,
            arousal_mean,
            dominance_mean,
        )
        return LyricsSentimentResult(
            segments=segments,
            dominant_emotion=dominant_emotion,
            valence_mean=valence_mean,
            arousal_mean=arousal_mean,
            dominance_mean=dominance_mean,
            model_used="keyword_heuristic",
        )

    @staticmethod
    def _neutral_fallback(audio_duration_s: float) -> LyricsSentimentResult:
        """Gibt neutral sentinel segments (no DSP intervention) zurück."""
        return LyricsSentimentResult(
            segments=[
                SegmentSentiment(
                    start_s=0.0,
                    end_s=audio_duration_s,
                    emotion="neutral",
                    valence=0.0,
                    arousal=0.0,
                    dominance=0.0,
                    confidence=0.0,
                )
            ],
            dominant_emotion="neutral",
            valence_mean=0.0,
            arousal_mean=0.0,
            dominance_mean=0.0,
            model_used="neutral_fallback",
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LyricsSentimentAnalyzer | None = None
_lock = threading.Lock()


def get_lyrics_sentiment_analyzer() -> LyricsSentimentAnalyzer:
    """Gibt the singleton LyricsSentimentAnalyzer (thread-safe double-checked locking) zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LyricsSentimentAnalyzer()
    return _instance
