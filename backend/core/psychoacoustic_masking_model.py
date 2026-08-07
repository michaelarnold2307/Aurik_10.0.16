"""
backend/core/psychoacoustic_masking_model.py — Psychoakustisches Masking-Modell (Aurik 10.0.0 §4.5)
===========================================================================
ISO 11172-3 Simultane + Temporale Maskierung als OMLSA-Gain-Modifier.
Stille-Segmente (<= SILENCE_DB) erhalten Gain ≤ SILENCE_GAIN_MAX = 0.30.
SR-Invariante: assert sr == 48000 in compute_threshold().
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# threadpoolctl: available in all Aurik venv builds; used to limit BLAS to
# 1 thread inside compute_threshold to prevent OpenBLAS worker-thread
# segfaults under concurrent numpy load (documented 2026-04-05 crash).
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits

    _THREADPOOL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _THREADPOOL_AVAILABLE = False
    _threadpool_limits = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Öffentliche Konstanten (§4.5, §8.2)
# ---------------------------------------------------------------------------
SILENCE_DB: float = -40.0  # Schwelle Stille (≤ -30 dBFS, §8.2)
GAIN_FLOOR: float = 0.10  # Mindest-Gain (G_floor, §4.5)
N_BARK: int = 24  # Anzahl Bark-Bänder
SILENCE_GAIN_MAX: float = 0.30  # Maximaler Gain in Stille-Frames
POST_MASK_MS: float = 100.0  # Temporale Post-Masking-Dauer in ms

# ISO 11172-3 Masking-Slopes: alpha_b ≈ 14.5 + b [dB/Bark] — monoton steigend
_MASKING_SLOPE_DB: list[float] = [14.5 + float(b) for b in range(N_BARK)]


# ---------------------------------------------------------------------------
# MaskingResult
# ---------------------------------------------------------------------------
@dataclass
class MaskingResult:
    """Rückgabe-Container von PsychoacousticMaskingModel.compute_threshold()."""

    gain_modifier: np.ndarray  # [n_frames × 24] float32 ∈ [GAIN_FLOOR, 1.0]
    masking_threshold: np.ndarray  # [n_frames × 24] float32 ≥ 0
    silence_frames: np.ndarray  # [n_frames] bool — True = Stille (<= SILENCE_DB)
    post_mask_frames: np.ndarray  # [n_frames] bool — True = temporale Maskierung aktiv
    n_frames: int
    n_bark_bands: int = N_BARK

    # Rückwärtskomp. Alias
    @property
    def gain_mask(self) -> np.ndarray:
        return self.gain_modifier

    @property
    def threshold_bark(self) -> np.ndarray:
        return self.masking_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_bark_bands": self.n_bark_bands,
            "silence_fraction": float(np.mean(self.silence_frames)),
            "post_mask_fraction": float(np.mean(self.post_mask_frames)),
            "gain_modifier_mean": float(np.mean(self.gain_modifier)),
        }


# ---------------------------------------------------------------------------
# PsychoacousticMaskingModel
# ---------------------------------------------------------------------------
class PsychoacousticMaskingModel:
    """Psychoakustisches Masking-Modell (ISO 11172-3, §4.5).

    Gleichzeitige Maskierung: pro Bark-Band b:
        MT_b = Signal_b · 10^(-alpha_b / 10)
    Stille-Segmente: Gain ≤ SILENCE_GAIN_MAX = 0.30
    Perkussive Transienten: Post-Masking 50–200 ms
    """

    def __init__(self, g_floor: float = GAIN_FLOOR) -> None:
        self.g_floor = float(np.clip(g_floor, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Interne Hilfsmethode
    # ------------------------------------------------------------------
    def _build_bark_bins(self, sr: int) -> np.ndarray:
        """Gibt N_BARK+1 Kanten der Bark-Bänder als float32-Array zurück.

        24 Bänder = 25 Kantenpositionen in Hz.
        Basiert auf Zwicker & Fastl (1990): BARK_EDGES_HZ.
        """
        bark_edges_hz = np.array(
            [
                20,
                100,
                200,
                300,
                400,
                510,
                630,
                770,
                920,
                1080,
                1270,
                1480,
                1720,
                2000,
                2320,
                2700,
                3150,
                3700,
                4400,
                5300,
                6400,
                7700,
                9500,
                12000,
                15500,
            ],
            dtype=np.float32,
        )
        return bark_edges_hz  # type: ignore[no-any-return]  # 25 Elemente = N_BARK + 1

    # ------------------------------------------------------------------
    # ATH (Absolute Threshold of Hearing) — ISO 226:2023 0-Phon-Kontur
    # ------------------------------------------------------------------
    @staticmethod
    def _ath_threshold_db(freq_hz: float) -> float:
        """Absolute Hörschwelle (ATH) nach ISO 226:2023 0-Phon-Kontur.

        Approximiert die absolute Hörschwelle in dB SPL für eine gegebene
        Frequenz mittels der ISO 226:2023 Annex A Näherungsformel.

        Formel (0 Phon):
            ath_spl_db = 3.64 * (f_khz ** -0.8)
                       - 6.5 * exp(-0.6 * (f_khz - 3.3) ** 2)
                       + 1e-3 * (f_khz ** 4)

        wobei f_khz = freq_hz / 1000.0.

        Ergebnis in dB SPL, dann auf dBFS umgerechnet:
            ath_dbfs = ath_spl_db - 100.0  (100 dB SPL ≡ 0 dBFS)

        Args:
            freq_hz: Frequenz in Hz (z.B. Bandmittenfrequenz).

        Returns:
            ATH-Wert in dBFS (negativ, z.B. −80 dBFS bei 1 kHz).
        """
        f_khz = freq_hz / 1000.0
        ath_spl_db = 3.64 * (f_khz ** (-0.8)) - 6.5 * math.exp(-0.6 * (f_khz - 3.3) ** 2) + 1e-3 * (f_khz**4)
        ath_dbfs = ath_spl_db - 100.0
        return ath_dbfs  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Binaural Masking Level Difference (BMLD) — IACC-basiert
    # ------------------------------------------------------------------
    @staticmethod
    def compute_interaural_cross_correlation(audio_left: np.ndarray, audio_right: np.ndarray) -> float:
        """Berechnet die Interaurale Kreuzkorrelation (IACC).

        Die IACC misst die Ähnlichkeit zwischen linkem und rechtem Kanal
        und dient als Grundlage für die Binaural Masking Level Difference (BMLD).

        IACC = |R_lr(0)| / sqrt(R_ll(0) · R_rr(0))

        wobei R_lr(τ) die Kreuzkorrelation zwischen links und rechts ist.

        Args:
            audio_left: 1-D numpy array, linker Kanal.
            audio_right: 1-D numpy array, rechter Kanal (muss gleiche Länge haben).

        Returns:
            IACC-Wert im Bereich [0.0, 1.0].
            1.0 = perfekt korreliert (mono), 0.0 = unkorreliert.
        """
        if audio_left.ndim > 1:
            audio_left = np.mean(audio_left, axis=-1)
        if audio_right.ndim > 1:
            audio_right = np.mean(audio_right, axis=-1)

        # Truncate to common length
        min_len = min(len(audio_left), len(audio_right))
        if min_len == 0:
            return 1.0
        left = np.asarray(audio_left[:min_len], dtype=np.float64)
        right = np.asarray(audio_right[:min_len], dtype=np.float64)

        # Normalized cross-correlation at zero lag
        r_ll = float(np.dot(left, left))
        r_rr = float(np.dot(right, right))
        r_lr = float(np.dot(left, right))

        denom = math.sqrt(max(r_ll * r_rr, 1e-30))
        iacc = r_lr / denom
        return max(0.0, min(1.0, abs(iacc)))

    @staticmethod
    def _binaural_masking_level_difference(iacc: float) -> float:
        """Binaurale Mithörschwellen-Differenz (BMLD) aus IACC.

        BMLD quantifiziert den psychoakustischen Vorteil binauralen Hörens:
        räumlich verteilte Signale sind leichter zu detektieren als monaurale.

        Mapping:
            IACC = 1.0 (perfekt korreliert, mono)   → BMLD =  0 dB
            IACC = 0.0 (unkorreliert)                → BMLD =  6 dB
            IACC = −1.0 (anti-phasig)                → BMLD = 12 dB

        Der BMLD-Wert wird von der Maskierungsschwelle subtrahiert,
        d.h. eine positive BMLD senkt die effektive Schwelle.

        Args:
            iacc: Interaurale Kreuzkorrelation in [−1.0, 1.0].

        Returns:
            BMLD-Wert in dB [0, 12] — von der Maskierungsschwelle zu subtrahieren.
        """
        iacc_val = min(1.0, max(-1.0, float(iacc)))
        # Lineare Abbildung: IACC ∈ [−1, 1] → BMLD ∈ [12, 0]
        #   IACC =  1.0 (perfekt korreliert) → BMLD = 0 dB
        #   IACC =  0.0 (unkorreliert)       → BMLD = 6 dB
        #   IACC = −1.0 (anti-phasig)        → BMLD = 12 dB
        bmld = 6.0 * (1.0 - iacc_val)
        return float(bmld)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_bark_band_mask(fft_freqs: np.ndarray, bark_edges: np.ndarray) -> np.ndarray:
        """Pre-compute boolean band assignment matrix [N_BARK × n_bins].

        Returns float32 mean-weights: for each band, 1/count where bin belongs,
        so that matrix-multiply with spec_sq gives mean energy per band.
        """
        n_bins = fft_freqs.shape[0]
        mask = np.zeros((N_BARK, n_bins), dtype=np.float32)
        for b in range(N_BARK):
            sel = (fft_freqs >= bark_edges[b]) & (fft_freqs < bark_edges[b + 1])
            cnt = int(sel.sum())
            if cnt > 0:
                mask[b, sel] = 1.0 / cnt
        return mask  # type: ignore[no-any-return]

    # Lock ensures only one thread runs compute_threshold at a time.
    # Prevents concurrent OpenBLAS worker threads (spawned by @, np.mean,
    # np.fft.rfft) from racing against each other → eliminates SIGSEGV
    # documented 2026-04-05 (crash at line 193 via _wrapreduction thread).
    _compute_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    def compute_threshold(self, audio: np.ndarray, sr: int) -> MaskingResult:
        """Berechnet Masking-Schwelle und Gain-Modifier (vektorisiert).

        Args:
            audio: float32/64 mono oder stereo (channel-first (2,N) oder channel-last (N,2))
            sr:    Sample-Rate — MUSS 48000 sein (§6.6 SR-Invariante)

        Returns:
            MaskingResult mit gain_modifier, masking_threshold,
            silence_frames, post_mask_frames
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"

        # Make a contiguous float32 copy immediately so all subsequent numpy
        # operations work on owned, aligned memory — never on a non-contiguous
        # view whose underlying buffer could be concurrently modified.
        arr = np.ascontiguousarray(
            np.nan_to_num(np.asarray(audio, dtype=np.float32)),
            dtype=np.float32,
        )
        # Stereo → Mono: axis-korrekt für beide Konventionen (N,2) und (2,N)
        if arr.ndim == 2:
            if arr.shape[0] <= arr.shape[1]:
                # channel-first (2, N) → mean over axis=0 → (N,)
                mono = np.ascontiguousarray(arr.mean(axis=0), dtype=np.float32)
            else:
                # channel-last (N, 2) → mean over axis=1 → (N,)
                mono = np.ascontiguousarray(arr.mean(axis=1), dtype=np.float32)
        else:
            mono = arr  # already contiguous copy from above

        if mono.size == 0:
            empty = np.zeros((1, N_BARK), dtype=np.float32)
            return MaskingResult(
                gain_modifier=np.full((1, N_BARK), self.g_floor, dtype=np.float32),
                masking_threshold=empty,
                silence_frames=np.array([True], dtype=bool),
                post_mask_frames=np.array([False], dtype=bool),
                n_frames=1,
            )

        frame = max(256, sr // 100)  # 480 samples @ 48 kHz
        hop = frame // 2

        # Cap at 10 s: greatly reduces BLAS allocation footprint, preventing
        # OpenBLAS worker-thread malloc failures under concurrent numpy load.
        # 10 s is still sufficient for stable masking-scalar estimation.
        _max_samples = sr * 10  # 480 000 samples @ 48 kHz
        if mono.size > _max_samples:
            _start = (mono.size - _max_samples) // 2
            mono = np.ascontiguousarray(mono[_start : _start + _max_samples], dtype=np.float32)

        n_frames = max(1, (mono.size - frame) // hop + 1)

        # ── Frame extraction via advanced indexing ───────────────────────────
        # SAFETY: fancy indexing always produces a fresh copy, never a strided
        # view — prevents SIGSEGV when numpy C malloc returns NULL.
        mono_safe = np.ascontiguousarray(np.clip(np.nan_to_num(mono), -1.0, 1.0), dtype=np.float32)
        _starts = np.arange(n_frames, dtype=np.int32) * hop
        _fi = _starts[:, np.newaxis] + np.arange(frame, dtype=np.int32)
        _fi = np.clip(_fi, 0, mono_safe.size - 1)
        segments = mono_safe[_fi]  # (n_frames, frame) — always a safe copy

        # ── All BLAS-intensive ops run single-threaded + behind instance lock ─
        # Prevents OpenBLAS worker threads from racing → eliminates SIGSEGV
        # (crash at _wrapreduction documented 2026-04-05).
        with self._compute_lock:
            _ctx = _threadpool_limits(limits=1, user_api="blas") if _THREADPOOL_AVAILABLE else None
            try:
                if _ctx is not None:
                    _ctx.__enter__()

                # RMS per frame via einsum (no BLAS multi-thread spawn)
                seg_f64 = segments.astype(np.float64)
                rms_sq = np.einsum("ij,ij->i", seg_f64, seg_f64) / frame + 1e-12
                del seg_f64  # free before FFT allocation
                rms_db = 10.0 * np.log10(rms_sq)

                sil_frms = rms_db <= SILENCE_DB

                # Temporal post-masking (sequential — state dependency)
                post_mask_frames_count = max(1, int(POST_MASK_MS * 1e-3 * sr / hop))
                post_frms = np.zeros(n_frames, dtype=bool)
                prev_loud = -999
                for i in range(n_frames):
                    if not sil_frms[i]:
                        prev_loud = i
                    elif prev_loud >= 0 and (i - prev_loud) <= post_mask_frames_count:
                        post_frms[i] = True

                # Batch FFT → power spectrum (n_frames, n_bins)
                spec_sq = np.abs(np.fft.rfft(segments, n=frame, axis=1)).astype(np.float32) ** 2

                # Pre-computed bark-band mean-weight matrix [N_BARK × n_bins]
                bark_edges = self._build_bark_bins(sr)
                fft_freqs = np.fft.rfftfreq(frame, d=1.0 / sr).astype(np.float32)
                band_mask = self._build_bark_band_mask(fft_freqs, bark_edges)

                # Band energies via einsum (avoids multi-threaded @ operator)
                # (n_frames, n_bins) · (N_BARK, n_bins)ᵀ → (n_frames, N_BARK)
                band_energy = np.einsum("ij,kj->ik", spec_sq, band_mask, optimize=False)

                # Masking threshold: ISO 11172-3 — slope attenuation per band
                slope_atten = np.array([10.0 ** (-s / 10.0) for s in _MASKING_SLOPE_DB], dtype=np.float32)
                mask_thr = np.maximum(0.0, band_energy * slope_atten[np.newaxis, :])

                # ATH (Absolute Threshold of Hearing) nach ISO 226:2023
                # Für jedes Bark-Band die Mittenfrequenz → ATH in dBFS → linear
                _bark_centers = 0.5 * (bark_edges[:-1] + bark_edges[1:])
                _ath_dbfs = np.array(
                    [self._ath_threshold_db(float(fc)) for fc in _bark_centers],
                    dtype=np.float32,
                )
                # ATH als lineare Energie-Schwelle pro Band (gleiche Einheit wie mask_thr)
                _ath_linear = 10.0 ** (_ath_dbfs / 10.0)
                mask_thr = np.maximum(mask_thr, _ath_linear[np.newaxis, :])

                # Gain modifier: relative band energy → gain
                total_e = np.sum(band_energy, axis=1, keepdims=True) + 1e-12
                rel = band_energy / total_e * N_BARK

                gain_mod = np.clip(0.3 + 0.7 * rel, self.g_floor, 1.0).astype(np.float32)

                if sil_frms.any():
                    g_sil = np.clip(SILENCE_GAIN_MAX * rel[sil_frms], self.g_floor, SILENCE_GAIN_MAX)
                    gain_mod[sil_frms] = g_sil.astype(np.float32)

                pm_only = post_frms & ~sil_frms
                if pm_only.any():
                    g_pm = np.clip(0.5 + 0.5 * rel[pm_only], self.g_floor, 1.0)
                    gain_mod[pm_only] = g_pm.astype(np.float32)

                gain_mod = np.nan_to_num(gain_mod)
                mask_thr = np.nan_to_num(mask_thr)
                np.clip(gain_mod, self.g_floor, 1.0, out=gain_mod)
                np.clip(mask_thr, 0.0, None, out=mask_thr)

            finally:
                if _ctx is not None:
                    try:
                        _ctx.__exit__(None, None, None)
                    except Exception as _ctx_exc:
                        logger.debug("Context manager exit fehlgeschlagen in masking model: %s", _ctx_exc)

        return MaskingResult(
            gain_modifier=gain_mod,
            masking_threshold=mask_thr,
            silence_frames=sil_frms,
            post_mask_frames=post_frms,
            n_frames=n_frames,
        )

    def apply_adaptive_gain(
        self,
        gain_mask: np.ndarray,
        masking_result: MaskingResult | np.ndarray,
    ) -> np.ndarray:
        """Skaliert gain_mask mit dem Masking-Modifier.

        Args:
            gain_mask:       [n_frames × 24] Eingangs-Gain
            masking_result:  MaskingResult oder np.ndarray (Rückwärtskomp.)

        Returns:
            Gain-Maske geclippt auf [G_floor, 1.0], NaN-frei
        """
        g = np.nan_to_num(np.asarray(gain_mask, dtype=np.float32))
        if g.ndim != 2:
            return np.clip(g, self.g_floor, 1.0)  # type: ignore[no-any-return]

        # MaskingResult oder raw ndarray als Masking-Modifier
        if isinstance(masking_result, MaskingResult):
            t = masking_result.gain_modifier
        else:
            t = np.nan_to_num(np.asarray(masking_result, dtype=np.float32))

        rows = min(g.shape[0], t.shape[0]) if t.ndim >= 1 else g.shape[0]
        cols = min(g.shape[1], t.shape[1]) if t.ndim == 2 else g.shape[1]
        out = g.copy()
        if t.ndim == 2 and rows > 0 and cols > 0:
            scale = np.clip(t[:rows, :cols], 0.3, 1.0)
            out[:rows, :cols] *= scale
        return np.clip(np.nan_to_num(out), self.g_floor, 1.0)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton (§3.2 Double-Checked Locking)
# ---------------------------------------------------------------------------
_instance: PsychoacousticMaskingModel | None = None
_lock = threading.Lock()


def get_masking_model() -> PsychoacousticMaskingModel:
    """Thread-sicherer Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PsychoacousticMaskingModel()
    return _instance


# Alias für Rückwärtskompatibilität
def get_psychoacoustic_masking_model() -> PsychoacousticMaskingModel:
    return get_masking_model()


# ---------------------------------------------------------------------------
# Convenience-Funktion
# ---------------------------------------------------------------------------
def compute_masking_threshold(audio: np.ndarray, sr: int) -> MaskingResult:
    """Convenience-Wrapper: Masking-Schwelle berechnen."""
    return get_masking_model().compute_threshold(audio, sr)


__all__ = [
    "GAIN_FLOOR",
    "N_BARK",
    "SILENCE_DB",
    "SILENCE_GAIN_MAX",
    "_MASKING_SLOPE_DB",
    "MaskingResult",
    "PsychoacousticMaskingModel",
    "compute_masking_threshold",
    "get_masking_model",
    "get_psychoacoustic_masking_model",
]
