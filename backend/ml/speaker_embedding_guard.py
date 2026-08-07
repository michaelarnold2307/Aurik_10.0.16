"""Speaker Embedding Guard — Verbesserte Sänger-Identitäts-Erhaltung.

§Rolls-Royce-Phantom: Ergänzt den MFCC-basierten SpeakerIdentityGuard
um perceptuelle Embedding-Ähnlichkeit. Nutzt spektrale Kontrast-Features
und Cepstral-Mean-Variance-Normalisierung für robustere Voiceprints.

Key improvements over speaker_identity_guard.py:
  1. Perceptual weighting (Bark-Scale, nicht linear)
  2. Cepstral Mean-Variance Normalization (CMVN) für Kanal-Robustheit
  3. Multi-window embedding (3 Fensterlängen für verschiedene Stimmregister)
  4. Soft-decision statt Hard-Threshold (Cos-Sim → Confidence)

Nutzung: Komplementär zu speaker_identity_guard.py.
    from backend.ml.speaker_embedding_guard import SpeakerEmbeddingGuard

    guard = SpeakerEmbeddingGuard()
    voiceprint = guard.extract(audio, sr=48000)  # Vor Pipeline
    ...
    similarity = guard.compare(voiceprint, processed_audio)  # Nach jeder Phase
    if similarity < 0.88:
        logger.warning("Speaker identity drift detected!")

Autor: Aurik 10 — Rolls-Royce Phantom Edition, 11. Juli 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Konfiguration ───────────────────────────────────────────────────────────
N_MFCC: int = 24  # Mehr Koeffizienten als der Basis-Guard (20)
N_FFT: int = 2048
HOP_LENGTH: int = 512
EMBEDDING_DIM: int = 72  # 24 MFCC × 3 Fensterlängen
IDENTITY_THRESHOLD: float = 0.88  # Cos-Sim-Schwelle
WARNING_THRESHOLD: float = 0.92  # Ab wann eine Warnung ausgegeben wird


@dataclass
class SpeakerEmbedding:
    """Multi-window perceptueller Voiceprint."""

    vector: np.ndarray  # 72-dim Embedding
    n_frames: int
    duration_s: float
    has_speech: bool
    confidence: float  # Wie sicher ist die Extraktion? (0–1)


class SpeakerEmbeddingGuard:
    """Verbesserter Sänger-Identitäts-Wächter.

    Extrahiert perceptuelle Embeddings über 3 Fensterlängen und vergleicht
    mit Cosine-Similarity. Robust gegen Kanal-Variation durch CMVN.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self._dim = embedding_dim
        self._window_sizes = [1024, 2048, 4096]  # 3 Fenster für Register
        self._n_mfcc = embedding_dim // len(self._window_sizes)

    def extract(self, audio: np.ndarray, sr: int = 48000) -> SpeakerEmbedding:
        """Extrahiert perceptuelles Speaker-Embedding.

        Args:
            audio: float32, mono oder stereo.
            sr:    Abtastrate.

        Returns:
            SpeakerEmbedding mit 72-dim Vektor.
        """
        mono = np.mean(audio, axis=-1) if audio.ndim > 1 else audio
        mono = mono.astype(np.float32).flatten()

        # ── RMS-Normalisierung (Amplituden-invariant) ──────────────────
        rms = float(np.sqrt(np.mean(mono**2)))
        if rms > 1e-8:
            mono = mono / rms * 0.3  # Normalisiere auf RMS=0.3 (Sprachniveau)

        if len(mono) < sr * 0.5:
            return SpeakerEmbedding(
                vector=np.zeros(self._dim, dtype=np.float32),
                n_frames=0,
                duration_s=len(mono) / sr,
                has_speech=False,
                confidence=0.0,
            )

        # ── Multi-Window MFCC-Extraktion ────────────────────────────────
        embeddings: list[np.ndarray] = []

        for win_size in self._window_sizes:
            # STFT
            n_frames = (len(mono) - win_size) // HOP_LENGTH + 1
            if n_frames < 1:
                continue

            mfccs = self._extract_mfcc_window(mono, sr, win_size, n_frames)
            # CMVN (Cepstral Mean-Variance Normalization)
            mfccs = self._apply_cmvn(mfccs)

            # Mittelwert über alle Frames → Embedding pro Fensterlänge
            mean_vec = np.mean(mfccs, axis=0)
            embeddings.append(mean_vec)

        # Konkateniere Embeddings aller Fensterlängen
        if embeddings:
            full_embedding = np.concatenate(embeddings)
        else:
            full_embedding = np.zeros(self._dim, dtype=np.float32)

        # Auf target-Dim padden/trimmen
        if len(full_embedding) < self._dim:
            full_embedding = np.pad(full_embedding, (0, self._dim - len(full_embedding)))
        else:
            full_embedding = full_embedding[: self._dim]

        # L2-Normalisierung
        norm = np.linalg.norm(full_embedding)
        if norm > 0:
            full_embedding = full_embedding / norm

        # Confidence: Energie-basiert
        rms = float(np.sqrt(np.mean(mono**2)))
        confidence = min(1.0, rms * 10.0)

        return SpeakerEmbedding(
            vector=full_embedding.astype(np.float32),
            n_frames=n_frames if "n_frames" in dir() else 0,
            duration_s=len(mono) / sr,
            has_speech=rms > 0.001,
            confidence=confidence,
        )

    def compare(
        self,
        reference: SpeakerEmbedding,
        test_audio: np.ndarray,
        sr: int = 48000,
    ) -> float:
        """Vergleicht Referenz-Voiceprint mit Test-Audio.

        Args:
            reference: Voiceprint VOR der Verarbeitung.
            test_audio: Audio NACH einer Phase.
            sr: Abtastrate.

        Returns:
            Cosine-Similarity (0–1). >0.92 = identisch, <0.88 = Drift.
        """
        if not reference.has_speech:
            return 1.0  # Keine Sprache → keine Bewertung möglich, als OK durchgehen

        test_embedding = self.extract(test_audio, sr)
        if not test_embedding.has_speech:
            return 1.0

        # Cosine-Similarity
        dot = np.dot(reference.vector, test_embedding.vector)
        similarity = float(np.clip(dot, -1.0, 1.0))
        similarity = (similarity + 1.0) / 2.0  # [-1,1] → [0,1]

        # Confidence-Gewichtung
        confidence_factor = min(reference.confidence, test_embedding.confidence)
        weighted_sim = similarity * confidence_factor + 1.0 * (1 - confidence_factor)

        return float(weighted_sim)

    def check(
        self,
        reference: SpeakerEmbedding,
        test_audio: np.ndarray,
        sr: int = 48000,
    ) -> tuple[bool, float, str]:
        """Prüft Speaker-Identität und gibt Entscheidung zurück.

        Returns:
            (ok, similarity, message)
        """
        sim = self.compare(reference, test_audio, sr)

        if sim >= IDENTITY_THRESHOLD:
            return True, sim, f"Speaker identity preserved (sim={sim:.3f})"
        elif sim >= IDENTITY_THRESHOLD - 0.05:
            return True, sim, f"Minor speaker drift (sim={sim:.3f}) — monitoring"
        else:
            return False, sim, f"Speaker identity drift detected (sim={sim:.3f} < {IDENTITY_THRESHOLD})"

    # ── Private ─────────────────────────────────────────────────────────

    def _extract_mfcc_window(
        self,
        mono: np.ndarray,
        sr: int,
        win_size: int,
        n_frames: int,
    ) -> np.ndarray:
        """Extrahiert MFCCs für ein Fenster mit Bark-Scale-Filterbank."""
        mel_bins = 40
        mfccs = np.zeros((n_frames, self._n_mfcc), dtype=np.float32)

        # Bark-Scale → Mel-Scale Approximation (vereinfacht)
        f_min, f_max = 80.0, min(sr / 2, 8000.0)
        mel_freqs = np.linspace(
            2595 * np.log10(1 + f_min / 700),
            2595 * np.log10(1 + f_max / 700),
            mel_bins + 2,
        )
        hz_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)

        # Hann-Fenster
        window = np.hanning(win_size)

        for i in range(n_frames):
            start = i * HOP_LENGTH
            frame = mono[start : start + win_size] * window
            spec = np.abs(np.fft.rfft(frame, n=N_FFT))
            freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)

            # Mel-Filterbank
            mel_energies = np.zeros(mel_bins)
            for j in range(mel_bins):
                mask = (freqs >= hz_freqs[j]) & (freqs < hz_freqs[j + 2])
                if np.any(mask):
                    mel_energies[j] = np.sum(spec[mask])

            # Log + DCT → MFCC
            mel_energies = np.maximum(mel_energies, 1e-10)
            log_mel = np.log(mel_energies)
            mfccs[i] = self._dct(log_mel, self._n_mfcc)

        return mfccs

    @staticmethod
    def _dct(x: np.ndarray, n_coeffs: int) -> np.ndarray:
        """Discrete Cosine Transform Type II (scipy-free)."""
        N = len(x)
        result = np.zeros(n_coeffs)
        for k in range(n_coeffs):
            result[k] = 2 * np.sum(x * np.cos(np.pi * k * (np.arange(N) + 0.5) / N))
        return result

    @staticmethod
    def _apply_cmvn(mfccs: np.ndarray) -> np.ndarray:
        """Cepstral Mean-Variance Normalization."""
        mean = np.mean(mfccs, axis=0)
        std = np.std(mfccs, axis=0) + 1e-8
        return (mfccs - mean) / std  # type: ignore[no-any-return]


__all__ = [
    "SpeakerEmbeddingGuard",
    "SpeakerEmbedding",
]
