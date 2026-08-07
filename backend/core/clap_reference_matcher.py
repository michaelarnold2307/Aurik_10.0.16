"""
core/clap_reference_matcher.py — CLAP-basiertes Referenz-Matching
==================================================================

CLAP (Contrastive Language-Audio Pretraining) ermöglicht semantisches
Audio-Matching: "Restauriere so, dass das Ergebnis klingt wie dieses
Originaldokument von 1960."

ARCHITEKTUR:
  1. **DSP-Embedding** (Primärpfad, kein ML nötig):
     Vektor dim=32 aus [Spectral Centroid, MFCC-Mittel×13, Harmonizität,
     Dynamikbreite, Rauschpegel, Spectral Rolloff, ZCR, Spectral Contrast×6]

  2. **CLAP-Plugin-Pfad** (wenn `plugins/clap_plugin.py` vorhanden):
     Nutzt LAION-CLAP 512-dim-Embeddings für exaktes semantisches Matching.

MATCHING-VERFAHREN:
  - Kosinus-Ähnlichkeit zwischen Referenz-Embedding und Ziel-Embedding
  - Wenn similarity < threshold: Parameter-Anpassung empfehlen
  - **Spektral-Transfer**: MUSIC-MOS-Energie-Profil von Referenz auf Ergebnis
    übertragen (Spektralformung)

VERWENDUNG im Restaurierungsworkflow:
  matcher = CLAPReferenceMatcher()
  matcher.load_reference(reference_audio, sample_rate)
  result = matcher.match_and_adapt(processed_audio, sample_rate)
  # result.adapted_audio: Spektral an Referenz angeglichenes Audio
  # result.similarity: Kosinus-Ähnlichkeit [0, 1]
  # result.recommendations: Empfohlene Parameter-Anpassungen

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ─── Embedding-Dimension ─────────────────────────────────────────────────
_DSP_EMBED_DIM = 32
_N_MFCC = 13
_N_CONTRAST = 6


# ─── DSP-Embedding ───────────────────────────────────────────────────────


def _safe_mono(audio: np.ndarray) -> np.ndarray:
    return audio.flatten() if audio.ndim > 1 else audio


def _mfcc_approx(audio: np.ndarray, sample_rate: int, n_mfcc: int = _N_MFCC) -> np.ndarray:
    """DCT-basierte MFCC-Näherung ohne librosa."""
    frame_size = 1024
    n_frames = max(1, min(200, len(audio) // frame_size))
    mfcc_sum = np.zeros(n_mfcc)

    for i in range(n_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        if len(frame) < frame_size:
            break
        window = frame * np.hanning(len(frame))
        mag = np.abs(np.fft.rfft(window))
        log_mag = np.log(mag + 1e-10)
        # Vereinfachte DCT-II
        for k in range(n_mfcc):
            mfcc_sum[k] += np.sum(log_mag * np.cos(np.pi * k * np.arange(len(log_mag)) / len(log_mag)))

    return mfcc_sum / max(n_frames, 1)  # type: ignore[no-any-return]


def _spectral_contrast(audio: np.ndarray, n_bands: int = _N_CONTRAST) -> np.ndarray:
    """Spektral-Kontrast (Peak/Valley Energie je Subbands)."""
    mag = np.abs(np.fft.rfft(audio[:4096] if len(audio) >= 4096 else audio))
    band_size = len(mag) // n_bands
    contrast = np.zeros(n_bands)
    for b in range(n_bands):
        band_mag = mag[b * band_size : (b + 1) * band_size]
        if len(band_mag) > 0:
            contrast[b] = np.log(np.percentile(band_mag, 90) / (np.percentile(band_mag, 10) + 1e-10) + 1)
    return contrast  # type: ignore[no-any-return]


def compute_dsp_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Berechnet DSP-Feature-Vektor dim={_DSP_EMBED_DIM} für Audio.
    Normiert auf Einheitslänge (für Kosinus-Ähnlichkeit).
    """
    mono = _safe_mono(audio)
    if len(mono) < 512:
        return np.zeros(_DSP_EMBED_DIM)  # type: ignore[no-any-return]

    features = []

    # Spectral Centroid (1)
    mag = np.abs(np.fft.rfft(mono[:4096] if len(mono) >= 4096 else mono))
    freqs = np.fft.rfftfreq(min(4096, len(mono)), 1 / sample_rate)
    centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-10)
    features.append(centroid / (sample_rate / 2))  # Normiert [0, 1]

    # MFCC × 13 (13)
    mfcc = _mfcc_approx(mono, sample_rate)
    mfcc_norm = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-10)
    features.extend(mfcc_norm.tolist())

    # Harmonizität (1) — vereinfacht
    from backend.core.music_quality_scorer import _frame_audio, _harmonicity, _resample_to_16k

    resampled = _resample_to_16k(mono, sample_rate)
    frames = _frame_audio(resampled)
    harm = _harmonicity(frames)
    features.append(harm)

    # Dynamikbreite in dB, normiert (1)
    rms_frames = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    dyn_range = float(np.percentile(20 * np.log10(rms_frames), 90) - np.percentile(20 * np.log10(rms_frames), 10))
    features.append(min(1.0, dyn_range / 60))

    # Rauschpegel normiert (1)
    noise_floor = float(np.percentile(20 * np.log10(rms_frames), 5))
    features.append(max(0.0, min(1.0, (noise_floor + 80) / 80)))

    # Spectral Rolloff (1)
    cumsum_mag = np.cumsum(mag)
    threshold = 0.85 * cumsum_mag[-1]
    rolloff_idx = np.searchsorted(cumsum_mag, threshold)
    rolloff = freqs[rolloff_idx] if rolloff_idx < len(freqs) else sample_rate / 2
    features.append(rolloff / (sample_rate / 2))

    # ZCR (1)
    zcr = np.sum(np.diff(np.sign(mono)) != 0) / len(mono)
    features.append(float(zcr))

    # Spectral Contrast × 6 (6)
    contrast = _spectral_contrast(mono)
    contrast_norm = contrast / (np.max(contrast) + 1e-10)
    features.extend(contrast_norm.tolist())

    # Auf genau _DSP_EMBED_DIM kürzen/padden
    feat_arr = np.array(features[:_DSP_EMBED_DIM], dtype=np.float32)
    if len(feat_arr) < _DSP_EMBED_DIM:
        feat_arr = np.pad(feat_arr, (0, _DSP_EMBED_DIM - len(feat_arr)))

    # L2-Normierung
    norm = np.linalg.norm(feat_arr) + 1e-10
    return feat_arr / norm  # type: ignore[no-any-return]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Kosinus-Ähnlichkeit zwischen zwei Embeddings."""
    a_norm = np.linalg.norm(a) + 1e-10
    b_norm = np.linalg.norm(b) + 1e-10
    return float(np.dot(a, b) / (a_norm * b_norm))


# ─── Spektral-Transfer ───────────────────────────────────────────────────


def _spectral_envelope(audio: np.ndarray, n_bands: int = 32) -> np.ndarray:
    """Berechnet mittlere Energie je Frequenzband als Envelope-Vektor."""
    mag = np.abs(np.fft.rfft(audio[:8192] if len(audio) >= 8192 else audio))
    band_size = max(1, len(mag) // n_bands)
    envelope = np.array([np.mean(mag[b * band_size : (b + 1) * band_size] ** 2) for b in range(n_bands)])
    return envelope + 1e-10  # type: ignore[no-any-return]


def spectral_transfer(
    audio: np.ndarray,
    sample_rate: int,
    reference_envelope: np.ndarray,
    transfer_strength: float = 0.4,
    n_bands: int = 32,
) -> np.ndarray:
    """
    Überträgt die spektrale Energieverteilung der Referenz auf das Ziel-Audio.

    Args:
        audio: Ziel-Audio (wird angepasst)
        sample_rate: Samplerate
        reference_envelope: Energie-Envelope der Referenz (Ausgabe von _spectral_envelope)
        transfer_strength: Stärke der Anpassung [0, 1] (0=kein Transfer, 1=vollständig)
        n_bands: Anzahl Frequenzbänder

    Returns:
        Spektral angepasstes Audio
    """
    mono = _safe_mono(audio)
    if len(mono) < 512:
        return audio.copy()

    target_envelope = _spectral_envelope(mono, n_bands)

    # Equalization-Kurve
    # Guard: reference_envelope kann negative CLAP-Embedding-Werte enthalten
    # => sqrt(negativ/positiv) = sqrt(NaN) => RuntimeWarning: invalid value in sqrt
    # Fix: beide Seiten auf >= 0 clampen vor sqrt
    eq_curve = np.sqrt(np.maximum(reference_envelope, 0.0) / np.maximum(target_envelope, 1e-12))
    eq_curve = np.clip(eq_curve, 0.1, 10.0)  # Maximale ±20 dB Anpassung

    # FFT-basierte EQ-Anwendung (Filterbank)
    n_fft = min(8192, len(mono))
    mag = np.fft.rfft(mono[:n_fft])
    freqs_idx = np.linspace(0, len(mag) - 1, n_bands + 1).astype(int)

    for b in range(n_bands):
        start_idx = freqs_idx[b]
        end_idx = freqs_idx[b + 1]
        gain = 1.0 + transfer_strength * (eq_curve[b] - 1.0)
        mag[start_idx:end_idx] *= gain

    corrected = np.fft.irfft(mag, n=n_fft)

    # Zurück auf Originallänge anpassen
    result = audio.copy()
    if audio.ndim == 1:
        min_len = min(len(corrected), len(result))
        result[:min_len] = corrected[:min_len]
    else:
        for ch in range(audio.shape[0]):
            ch_audio = _safe_mono(audio[ch])
            n_fft_ch = min(8192, len(ch_audio))
            ch_mag = np.fft.rfft(ch_audio[:n_fft_ch])
            for b in range(n_bands):
                start_idx = freqs_idx[b]
                end_idx = freqs_idx[b + 1]
                gain = 1.0 + transfer_strength * (eq_curve[b] - 1.0)
                ch_mag[start_idx:end_idx] *= gain
            ch_corrected = np.fft.irfft(ch_mag, n=n_fft_ch)
            min_len = min(len(ch_corrected), audio.shape[1])
            result[ch, :min_len] = ch_corrected[:min_len]

    return result


# ─── CLAPReferenceMatcher ────────────────────────────────────────────────


@dataclass
class MatchResult:
    """Ergebnis eines Referenz-Matching-Vorgangs."""

    adapted_audio: np.ndarray
    similarity: float
    recommendations: list[str] = field(default_factory=list)
    reference_embedding: np.ndarray | None = None
    target_embedding: np.ndarray | None = None


class CLAPReferenceMatcher:
    """
    Semantisches Audio-Matching via CLAP-Embeddings (DSP-Fallback).

    Workflow:
        1. ``load_reference(audio, sr)`` — Referenz-Embedding berechnen
        2. ``match_and_adapt(audio, sr)`` — Ziel an Referenz angleichen
    """

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        transfer_strength: float = 0.40,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.transfer_strength = transfer_strength
        self._ref_embedding: np.ndarray | None = None
        self._ref_envelope: np.ndarray | None = None
        self._ref_sample_rate: int = 44100

    def _clap_plugin_embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """Versucht CLAP-Plugin für 512-dim-Embedding. None wenn nicht verfügbar."""
        try:
            import importlib

            clap = importlib.import_module("clap_plugin")
            if hasattr(clap, "embed"):
                return clap.embed(audio, sample_rate)  # type: ignore[no-any-return]
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        return None

    def load_reference(self, audio: np.ndarray, sample_rate: int) -> None:
        """Lädt und berechnet Referenz-Embedding + Spektral-Envelope."""
        plugin_emb = self._clap_plugin_embed(audio, sample_rate)
        if plugin_emb is not None:
            self._ref_embedding = plugin_emb
        else:
            self._ref_embedding = compute_dsp_embedding(audio, sample_rate)

        mono = _safe_mono(audio)
        self._ref_envelope = _spectral_envelope(mono)
        self._ref_sample_rate = sample_rate
        logger.info("CLAP-Referenz geladen, Embedding-Dim=%d", len(self._ref_embedding))

    def match_and_adapt(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> MatchResult:
        """
        Passt audio spektral an die geladene Referenz an.

        Returns:
            MatchResult mit adapted_audio und Empfehlungen
        """
        if self._ref_embedding is None:
            raise RuntimeError("load_reference() muss zuerst aufgerufen werden.")

        plugin_emb = self._clap_plugin_embed(audio, sample_rate)
        target_emb = plugin_emb if plugin_emb is not None else compute_dsp_embedding(audio, sample_rate)

        similarity = cosine_similarity(self._ref_embedding, target_emb)
        recommendations: list[str] = []

        adapted = audio.copy()

        if similarity < self.similarity_threshold and self._ref_envelope is not None:
            adapted = spectral_transfer(
                audio,
                sample_rate,
                self._ref_envelope,
                transfer_strength=self.transfer_strength,
            )
            recommendations.append(
                f"Spektral-Transfer angewendet (similarity={similarity:.3f} < {self.similarity_threshold})"
            )

        if similarity < 0.60:
            recommendations.append(
                "Sehr geringe Ähnlichkeit zur Referenz — Rauschreduzierung oder EQ-Korrektur empfohlen"
            )
        elif similarity < 0.75:
            recommendations.append("Mäßige Ähnlichkeit — leichtes Spektral-Shaping empfohlen")

        return MatchResult(
            adapted_audio=adapted,
            similarity=round(similarity, 4),
            recommendations=recommendations,
            reference_embedding=self._ref_embedding,
            target_embedding=target_emb,
        )
