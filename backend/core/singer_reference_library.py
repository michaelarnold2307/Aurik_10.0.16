"""External singer reference library for Aurik 10.0.0.

Motivation: A human mastering engineer knows the typical sound of many
well-known artists. When restoring Ella Fitzgerald, they compare the
result mentally against the original Ella sound — not against the degraded
version. This difference is crucial.

VQI problem without this library: `singer_identity_cosine` is computed between
pre-restoration (degraded) and post-restoration audio — which favours results
closer to the degraded input rather than the true artist sound.

With this library: when an artist is identified with confidence > 0.70,
VQI can compute `singer_identity_cosine` against the true artist reference
instead of the degraded input.

Implementation: DSP-only, no ML model required.
- MFCC 20 coefficients (mean + std per segment) → 40-dim fingerprint
- Spectral centroid mean
- Total fingerprint: 41-dim float32 vector
- Library: embedded approximations for voice-class prototypes
  (statistical estimates as proxy; real fingerprints from licensed audio
  would yield better results but are excluded for copyright reasons).

Spec: §SRL-1 Singer-Reference-Library (v10.0.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fingerprint format
# ---------------------------------------------------------------------------
# dim 0–19: MFCC means (normalised)
# dim 20–39: MFCC standard deviations (normalised)
# dim 40:   Spectral centroid mean (normalised, 0=0 Hz, 1=Nyquist)
# All values in [-1, +1] or [0, 1]
_FINGERPRINT_DIM = 41


@dataclass
class SingerMatchResult:
    """Result of a singer-matching operation."""

    artist_id: str  # Matched voice class (empty if no reliable match)
    confidence: float  # 0.0–1.0
    fingerprint_distance: float  # L2 distance to nearest reference fingerprint
    reference_fingerprint: np.ndarray | None  # Reference fingerprint of the matched class

    def is_reliable(self) -> bool:
        """Gibt True when the match is usable for VQI singer identity checks zurück."""
        return (
            self.confidence >= 0.70
            and bool(self.artist_id)
            and np.isfinite(self.fingerprint_distance)
            and self.reference_fingerprint is not None
        )


# ---------------------------------------------------------------------------
# Embedded reference fingerprints
# ---------------------------------------------------------------------------
# These vectors are NOT extracted fingerprints from real audio — they are
# approximate expected values based on published MFCC statistics from MIR
# research literature.
# Privacy: no artist audio is embedded.
#
# Format: artist_id → np.array(shape=(41,), dtype=float32)
#
# The MFCC values are intentionally only coarsely approximated — they allow
# discrimination between very different voice types:
# - Deep male voices (bass-baritone)
# - Bright female voices (soprano)
# - Belcanto tenors
# - Jazz scat singers
# etc.
#
# For production use, real fingerprints should be computed from royalty-free
# audio and embedded in song_fingerprints.npz.


def _make_fp(
    mfcc_mean: list[float],
    mfcc_std: list[float],
    centroid_norm: float,
) -> np.ndarray:
    """Erstellt a normalised fingerprint vector from MFCC mean/std and spectral centroid."""
    assert len(mfcc_mean) == 20
    assert len(mfcc_std) == 20
    fp = np.zeros(_FINGERPRINT_DIM, dtype=np.float32)
    fp[:20] = np.clip(mfcc_mean, -1.0, 1.0)
    fp[20:40] = np.clip(mfcc_std, 0.0, 1.0)
    fp[40] = float(np.clip(centroid_norm, 0.0, 1.0))
    return fp  # type: ignore[no-any-return]


# Voice-class prototypes (no real artist fingerprints)
# Based on the MFCC characteristics of typical voice types
_STIMMKLASSE_PROTOTYPEN: dict[str, np.ndarray] = {
    # Bright female soprano (1940–1960s: high overtone energy)
    "voice_light_soprano": _make_fp(
        mfcc_mean=[
            -0.3,
            0.7,
            -0.1,
            0.5,
            -0.2,
            0.3,
            -0.1,
            0.2,
            0.0,
            0.1,
            0.0,
            -0.1,
            0.1,
            -0.1,
            0.1,
            0.0,
            0.0,
            0.1,
            0.0,
            0.0,
        ],
        mfcc_std=[
            0.3,
            0.25,
            0.2,
            0.2,
            0.15,
            0.15,
            0.12,
            0.12,
            0.1,
            0.1,
            0.1,
            0.1,
            0.08,
            0.08,
            0.08,
            0.07,
            0.07,
            0.06,
            0.06,
            0.05,
        ],
        centroid_norm=0.45,
    ),
    # Jazz alto (warm midrange voice, 1940–1970s)
    "voice_jazz_alto": _make_fp(
        mfcc_mean=[-0.1, 0.4, 0.1, 0.2, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mfcc_std=[
            0.25,
            0.2,
            0.18,
            0.16,
            0.14,
            0.13,
            0.12,
            0.11,
            0.1,
            0.1,
            0.09,
            0.09,
            0.08,
            0.08,
            0.07,
            0.07,
            0.06,
            0.06,
            0.05,
            0.05,
        ],
        centroid_norm=0.32,
    ),
    # Deep baritone (classical/opera)
    "voice_deep_baritone": _make_fp(
        mfcc_mean=[
            0.2,
            -0.3,
            0.3,
            -0.2,
            0.2,
            -0.1,
            0.1,
            -0.1,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        mfcc_std=[
            0.2,
            0.18,
            0.16,
            0.15,
            0.14,
            0.13,
            0.12,
            0.11,
            0.1,
            0.1,
            0.09,
            0.08,
            0.08,
            0.07,
            0.07,
            0.06,
            0.06,
            0.05,
            0.05,
            0.04,
        ],
        centroid_norm=0.20,
    ),
    # Lyric tenor (belcanto, high energy in 2–4 kHz range)
    "voice_tenor_lyric": _make_fp(
        mfcc_mean=[
            -0.1,
            0.5,
            0.1,
            0.3,
            -0.1,
            0.2,
            0.0,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        mfcc_std=[
            0.22,
            0.19,
            0.17,
            0.15,
            0.14,
            0.13,
            0.11,
            0.10,
            0.09,
            0.09,
            0.08,
            0.08,
            0.07,
            0.07,
            0.06,
            0.06,
            0.05,
            0.05,
            0.04,
            0.04,
        ],
        centroid_norm=0.38,
    ),
    # Pop male voice (modern sound, 1980–2010)
    "voice_pop_male": _make_fp(
        mfcc_mean=[0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mfcc_std=[
            0.2,
            0.17,
            0.15,
            0.14,
            0.13,
            0.12,
            0.11,
            0.10,
            0.09,
            0.09,
            0.08,
            0.08,
            0.07,
            0.07,
            0.06,
            0.06,
            0.05,
            0.05,
            0.04,
            0.04,
        ],
        centroid_norm=0.28,
    ),
    # Pop female voice (modern sound, 1980–2010)
    "voice_pop_female": _make_fp(
        mfcc_mean=[-0.1, 0.4, 0.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mfcc_std=[
            0.22,
            0.19,
            0.17,
            0.15,
            0.13,
            0.12,
            0.11,
            0.10,
            0.09,
            0.08,
            0.08,
            0.07,
            0.07,
            0.06,
            0.06,
            0.05,
            0.05,
            0.04,
            0.04,
            0.04,
        ],
        centroid_norm=0.37,
    ),
    # Schlager/chanson (DE/FR, 1960–1980: warm midrange dynamics)
    "voice_schlager_chanson": _make_fp(
        mfcc_mean=[0.0, 0.3, 0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mfcc_std=[
            0.23,
            0.20,
            0.18,
            0.16,
            0.14,
            0.13,
            0.11,
            0.10,
            0.09,
            0.09,
            0.08,
            0.07,
            0.07,
            0.06,
            0.06,
            0.05,
            0.05,
            0.04,
            0.04,
            0.04,
        ],
        centroid_norm=0.30,
    ),
}


# ---------------------------------------------------------------------------
# Fingerprint computation from audio
# ---------------------------------------------------------------------------


def compute_vocal_fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet a 41-dim voice-class fingerprint from an audio signal.

    Features used:
    - MFCC 20 (mean + std, time-averaged)
    - Spectral centroid (normalised to Nyquist)

    Args:
        audio: numpy array (channels-first or mono)
        sr:    sample rate in Hz

    Returns:
        np.array(shape=(41,), dtype=float32)
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = _to_mono(audio_f32)
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        # Cap at 30 s (fingerprint does not need more)
        max_samples = min(len(audio_f32), sr * 30)
        audio_f32 = audio_f32[:max_samples]
        if audio_f32.size < max(32, sr // 10):
            return np.zeros(_FINGERPRINT_DIM, dtype=np.float32)  # type: ignore[no-any-return]
        rms = float(np.sqrt(np.mean(audio_f32**2)))
        if not np.isfinite(rms) or rms < 1e-6:
            return np.zeros(_FINGERPRINT_DIM, dtype=np.float32)  # type: ignore[no-any-return]

        # MFCC 20 Koeffizienten
        mfcc = librosa.feature.mfcc(y=audio_f32, sr=sr, n_mfcc=20, hop_length=512)
        mfcc_mean = mfcc.mean(axis=1)  # (20,)
        mfcc_std = mfcc.std(axis=1)  # (20,)

        # Normalise: mfcc_mean to [-1, 1], std to [0, 1]
        _mfcc_range = np.abs(mfcc_mean).max() + 1e-9
        mfcc_mean_norm = np.clip(mfcc_mean / _mfcc_range, -1.0, 1.0).astype(np.float32)
        _std_range = mfcc_std.max() + 1e-9
        mfcc_std_norm = np.clip(mfcc_std / _std_range, 0.0, 1.0).astype(np.float32)

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio_f32, sr=sr, hop_length=512)
        centroid_mean = float(centroid.mean())
        centroid_norm = float(np.clip(centroid_mean / (sr / 2.0), 0.0, 1.0))

        fp = np.zeros(_FINGERPRINT_DIM, dtype=np.float32)
        fp[:20] = mfcc_mean_norm
        fp[20:40] = mfcc_std_norm
        fp[40] = centroid_norm
        return fp  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug("berechnen_vocal_fingerprint fehlgeschlagen: %s", exc)
        return np.zeros(_FINGERPRINT_DIM, dtype=np.float32)  # type: ignore[no-any-return]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Gibt mono audio from mono, channel-first, or channel-last input zurück."""
    if audio.ndim != 2:
        return audio.reshape(-1)
    rows, cols = audio.shape
    if rows <= 8 and cols > rows:
        return audio.mean(axis=0)  # type: ignore[no-any-return]
    if cols <= 8 and rows > cols:
        return audio.mean(axis=1)  # type: ignore[no-any-return]
    return audio.mean(axis=0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_singer(
    audio: np.ndarray,
    sr: int,
    min_confidence: float = 0.55,
) -> SingerMatchResult:
    """Compare audio fingerprint against known voice-class prototypes.

    Matching method: cosine similarity (more robust than L2 across
    different recording loudness levels).

    Args:
        audio: input audio (channels-first or mono)
        sr:    sample rate
        min_confidence: minimum confidence for a valid match

    Returns:
        SingerMatchResult (artist_id="" if no reliable match found)
    """
    query_fp = compute_vocal_fingerprint(audio, sr)
    query_norm_raw = float(np.linalg.norm(query_fp))
    if not np.isfinite(query_norm_raw) or query_norm_raw < 1e-8:
        return SingerMatchResult(
            artist_id="",
            confidence=0.0,
            fingerprint_distance=float("inf"),
            reference_fingerprint=None,
        )
    query_norm = query_norm_raw + 1e-12

    best_id = ""
    best_sim = -1.0
    best_fp: np.ndarray | None = None
    best_dist = float("inf")

    for artist_id, ref_fp in _STIMMKLASSE_PROTOTYPEN.items():
        ref_norm = np.linalg.norm(ref_fp) + 1e-12
        cosine_sim = float(np.dot(query_fp, ref_fp) / (query_norm * ref_norm))
        l2_dist = float(np.linalg.norm(query_fp - ref_fp))
        if cosine_sim > best_sim:
            best_sim = cosine_sim
            best_id = artist_id
            best_fp = ref_fp.copy()
            best_dist = l2_dist

    # Confidence: cosine-sim [-1, 1] → [0, 1]
    confidence = float(np.clip((best_sim + 1.0) / 2.0, 0.0, 1.0))

    if confidence < min_confidence:
        return SingerMatchResult(
            artist_id="",
            confidence=confidence,
            fingerprint_distance=best_dist,
            reference_fingerprint=None,
        )

    logger.info(
        "§SRL-1 Singer-Match: class=%s confidence=%.2f dist=%.3f",
        best_id,
        confidence,
        best_dist,
    )
    return SingerMatchResult(
        artist_id=best_id,
        confidence=confidence,
        fingerprint_distance=best_dist,
        reference_fingerprint=best_fp,
    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: SingerReferenceLibrary | None = None
_lock = threading.Lock()


class SingerReferenceLibrary:
    """Thread-safe singleton wrapper for singer-class matching."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def match(
        self,
        audio: np.ndarray,
        sr: int,
        min_confidence: float = 0.55,
    ) -> SingerMatchResult:
        """Match audio against the voice-class library (non-blocking)."""
        try:
            return match_singer(audio, sr, min_confidence=min_confidence)
        except Exception as exc:
            logger.debug("§SRL-1 Singer-Match nicht blockierend: %s", exc)
            return SingerMatchResult(
                artist_id="",
                confidence=0.0,
                fingerprint_distance=float("inf"),
                reference_fingerprint=None,
            )


def get_singer_reference_library() -> SingerReferenceLibrary:
    """Gibt the singleton SingerReferenceLibrary (thread-safe double-checked locking) zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SingerReferenceLibrary()
    return _instance
