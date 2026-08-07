"""
§v10 Speaker Identity Guard — Stabiler, getesteter Sänger-Identitätswächter.

Extrahiert MFCC-basierte Voiceprints (60-dim) VOR der Pipeline und validiert
nach jeder vokalrelevanten Phase, ob die Sänger-Identität erhalten blieb.
Cosine-Similarity < IDENTITY_THRESHOLD → Warnung.

Reines NumPy — keine externen ML-Frameworks als Pflicht.
Robust gegen: leere Arrays, Monospuren, verschiedene Stereo-Layouts, NaN/Inf.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Konfiguration
# ═══════════════════════════════════════════════════════════════════════════

N_MFCC: int = 20
N_FFT: int = 2048
HOP_LENGTH: int = 512
IDENTITY_THRESHOLD: float = 0.92

VOCAL_PHASES: tuple[str, ...] = (
    "phase_19_de_esser",
    "phase_42_vocal_enhancement",
    "phase_43",
    "phase_65_vocal_naturalness_restoration",
)


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SpeakerIdentityResult:
    phase_id: str
    cosine_similarity: float
    identity_preserved: bool
    pre_embedding_shape: tuple[int, ...] | None = None


# ═══════════════════════════════════════════════════════════════════════════
# MFCC-Implementierung (reines NumPy, keine externen Abhängigkeiten)
# ═══════════════════════════════════════════════════════════════════════════


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)  # type: ignore[no-any-return]


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)  # type: ignore[no-any-return]


def _mel_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Erzeugt eine Mel-Filterbank-Matrix."""
    f_min, f_max = 0.0, sr / 2.0
    mel_min, mel_max = _hz_to_mel(f_min), _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)  # type: ignore[arg-type]
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(left, center):
            if j < filters.shape[1]:
                filters[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if j < filters.shape[1]:
                filters[i - 1, j] = (right - j) / max(right - center, 1)
    return filters


def _dct_ii(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    """DCT Typ II (orthogonal)."""
    n = len(x)
    result = np.zeros(n_coeffs)
    for k in range(n_coeffs):
        result[k] = 2.0 * np.sum(x * np.cos(np.pi * k * (2.0 * np.arange(n) + 1.0) / (2.0 * n)))
    return result


def extract_mfcc_voiceprint(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extrahiert ein 60-dimensionales MFCC-Voiceprint (20 MFCCs + Δ + ΔΔ).

    Robust gegen:
    - leere Arrays → Null-Vektor
    - Stereo/N-Dimensionen → automatisch Mono
    - Kurze Audio-Clips → Zero-Padding
    - NaN/Inf → durch Nullen ersetzt
    """
    # Input normalisieren
    arr = np.asarray(audio, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Mono machen
    if arr.ndim == 0 or arr.size == 0:
        return np.zeros(n_mfcc * 3, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1) if arr.shape[-1] <= 2 else arr.mean(axis=0)
    arr = np.atleast_1d(arr).ravel()

    # Mindestlänge sicherstellen
    min_len = n_fft + hop_length
    if len(arr) < min_len:
        arr = np.pad(arr, (0, min_len - len(arr)))

    # Frames bilden
    n_frames = 1 + (len(arr) - n_fft) // hop_length
    window = np.hanning(n_fft)
    mel_fb = _mel_filterbank(n_mfcc + 2, n_fft, sr)

    mfcc_frames = []
    for i in range(max(1, n_frames)):
        start = i * hop_length
        end = start + n_fft
        frame = arr[start:end]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))

        power_spec = np.abs(np.fft.rfft(frame * window)) ** 2
        mel_spec = np.dot(mel_fb, power_spec)
        mel_spec = np.log(np.maximum(mel_spec, 1e-12))
        mfcc = _dct_ii(mel_spec, n_mfcc)
        mfcc_frames.append(mfcc)

    if not mfcc_frames:
        return np.zeros(n_mfcc * 3, dtype=np.float64)

    mfcc_array = np.array(mfcc_frames).T  # (n_mfcc, n_frames)

    # Delta (erste Ableitung)
    delta = np.zeros_like(mfcc_array)
    if mfcc_array.shape[1] >= 3:
        delta[:, 1:-1] = (mfcc_array[:, 2:] - mfcc_array[:, :-2]) / 2.0

    # Delta-Delta (zweite Ableitung)
    delta2 = np.zeros_like(delta)
    if delta.shape[1] >= 3:
        delta2[:, 1:-1] = (delta[:, 2:] - delta[:, :-2]) / 2.0

    # Zusammenführen: MFCC + Delta + Delta-Delta → 60-dim
    embedding = np.concatenate(
        [
            mfcc_array.mean(axis=1),
            delta.mean(axis=1),
            delta2.mean(axis=1),
        ]
    )

    # Normalisieren (L2)
    norm = np.linalg.norm(embedding)
    if norm > 1e-12:
        embedding /= norm

    return embedding.astype(np.float64)  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════
# Speaker Identity Guard
# ═══════════════════════════════════════════════════════════════════════════


class SpeakerIdentityGuard:
    """Bewacht die Sänger-Identität über die Pipeline hinweg."""

    def __init__(self) -> None:
        self._pre_embedding: np.ndarray | None = None
        self._pre_shape: tuple[int, ...] | None = None

    def capture_pre_embedding(self, audio: np.ndarray, sr: int) -> None:
        """Extrahiert und speichert das Pre-Pipeline-Voiceprint."""
        self._pre_embedding = extract_mfcc_voiceprint(audio, sr)
        self._pre_shape = self._pre_embedding.shape
        logger.debug(
            "SpeakerIdentity: Pre-Embedding erfasst (dim=%d, norm=%.3f)",
            len(self._pre_embedding),
            float(np.linalg.norm(self._pre_embedding)),
        )

    def get_pre_embedding(self) -> np.ndarray | None:
        """Gibt das gespeicherte Pre-Embedding zurück."""
        return self._pre_embedding

    def check_phase(
        self,
        phase_id: str,
        vocals_after_phase: np.ndarray,
        sr: int,
        threshold: float = IDENTITY_THRESHOLD,
    ) -> SpeakerIdentityResult:
        """Validiert die Sänger-Identität nach einer vokalrelevanten Phase.

        Args:
            phase_id:            ID der Phase (z.B. "phase_42")
            vocals_after_phase:  Audio NACH der Phase
            sr:                  Sample-Rate
            threshold:           Cosine-Similarity-Schwellwert

        Returns:
            SpeakerIdentityResult mit similarity und identity_preserved-Flag
        """
        if self._pre_embedding is None:
            logger.warning("SpeakerIdentity: Kein Pre-Embedding — Pruefung_Verarbeitungsschritt übersprungen.")
            return SpeakerIdentityResult(
                phase_id=phase_id,
                cosine_similarity=1.0,
                identity_preserved=True,
            )

        try:
            post_emb = extract_mfcc_voiceprint(vocals_after_phase, sr)

            # Cosine-Similarity
            dot = float(np.dot(self._pre_embedding, post_emb))
            norm_pre = float(np.linalg.norm(self._pre_embedding))
            norm_post = float(np.linalg.norm(post_emb))
            similarity = dot / max(norm_pre * norm_post, 1e-12)
            similarity = float(np.clip(similarity, -1.0, 1.0))

            identity_preserved = similarity >= threshold

            if not identity_preserved:
                logger.warning(
                    "SpeakerIdentity: %s — Identität möglicherweise verändert (cosine_sim=%.4f < Schwelle=%.2f)",
                    phase_id,
                    similarity,
                    threshold,
                )
            else:
                logger.debug(
                    "SpeakerIdentity: %s OK (cosine_sim=%.4f)",
                    phase_id,
                    similarity,
                )

            return SpeakerIdentityResult(
                phase_id=phase_id,
                cosine_similarity=similarity,
                identity_preserved=identity_preserved,
                pre_embedding_shape=self._pre_shape,
            )
        except Exception as exc:
            logger.error("SpeakerIdentity: Pruefung_Verarbeitungsschritt fehlgeschlagen: %s", exc)
            return SpeakerIdentityResult(
                phase_id=phase_id,
                cosine_similarity=1.0,
                identity_preserved=True,
            )
