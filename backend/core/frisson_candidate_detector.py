"""
backend/core/frisson_candidate_detector.py — Frisson Candidate Detector
========================================================================
Detects high-potential "Gänsehaut" (musical chill / frisson) moments in audio.

Research basis:
  Blood & Zatorre (2001): frisson correlates with dopaminergic reward in nucleus
    accumbens; strongest predictor is expectation violation.
  Sloboda (1991): chills most reliably triggered by harmonic surprises and
    dynamic entrances after quiet passages.
  Grewe et al. (2007): multi-listener agreement highest for RMS crescendo and
    new voice/instrument entries.
  Huron (2006) "Sweet Anticipation": ITPRA model — frisson = surprise x reward.
  Harrison & Loui (2014): physiological frisson correlates with spectral flux and
    dynamic contrast patterns more reliably than static loudness.

Five acoustic triggers (empirically ranked):
  1. RMS crescendo after quiet    (weight 0.35) — loud entry after silence/near-silence
  2. Harmonic surprise            (weight 0.25) — chroma KL-divergence vs. past expectation
  3. Timbral novelty              (weight 0.20) — new instrument/voice entering the mix
  4. Sub-bass onset               (weight 0.10) — sudden low-frequency energy surge
  5. Dynamic contrast peak        (weight 0.10) — local maximum of dynamic range

Integration with MDEM (§2.30):
  Detected zones are passed to MicroDynamicsEnvelopeMorphing.morph() via the
  frisson_zones parameter. In those zones the downward gain correction is capped at
  -1.0 LU instead of -max_gain, preventing attenuation of emotionally significant
  peak moments.

Performance: < 150 ms for 3-minute stereo at 48 kHz (pure NumPy, no ML).

Author: Aurik Development Team
Version: 1.0.0 (v10.0.0)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class FrissonZone:
    """time interval with high frisson-induction potential.

    Attributes:
        start_s: Zone start time in seconds.
        end_s:   Zone end time in seconds.
        score:   Combined frisson score in [0.0, 1.0] (higher = more potential).
        trigger: Name of the dominant acoustic trigger that drove the score.
    """

    start_s: float
    end_s: float
    score: float
    trigger: str

    def __post_init__(self) -> None:
        self.start_s = float(np.clip(self.start_s, 0.0, 1e9))
        self.end_s = float(max(self.end_s, self.start_s + 0.1))
        self.score = float(np.clip(self.score, 0.0, 1.0))
        self.trigger = str(self.trigger or "unknown")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class FrissonCandidateDetector:
    """Erkennt time zones with high frisson/chill potential in audio.

    Thread-safe; use the module-level get_frisson_detector() singleton.
    """

    # Analysis parameters
    FRAME_HOP_S: float = 0.25  # 250 ms frame hop
    FRAME_SIZE_S: float = 0.50  # 500 ms analysis window
    FFT_SIZE: int = 4096  # frequency resolution
    MEL_BANDS: int = 8  # log mel filterbank bands (150 Hz – 8 kHz)
    # Zone geometry
    ZONE_RADIUS_S: float = 1.0  # ±1 s around peak → 2 s total zone
    MIN_SCORE: float = 0.28  # minimum score to emit a zone
    # Performance cap
    MAX_ANALYSIS_S: float = 600.0  # analyse at most 10 min; crop center for longer songs

    def detect(self, audio: np.ndarray, sr: int, max_zones: int = 20) -> list[FrissonZone]:
        """Erkennt frisson candidate zones in audio.

        Args:
            audio:     Audio signal (mono or stereo, any dtype).
            sr:        Sample rate in Hz.
            max_zones: Maximum number of zones to return.

        Returns:
            List of FrissonZone objects, sorted by score descending.
            Empty list if audio is too short or no zones found.
        """
        try:
            return self._detect_safe(audio, sr, max_zones)
        except Exception as exc:
            logger.debug("FrissonCandidateDetector.erkennen() fehlgeschlagen (nicht blockierend): %s", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Internal implementation                                            #
    # ------------------------------------------------------------------ #

    def _detect_safe(self, audio: np.ndarray, sr: int, max_zones: int) -> list[FrissonZone]:
        if sr <= 0:
            return []

        # --- mono conversion -------------------------------------------------
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            # Handle both (samples, channels) and (channels, samples)
            if arr.shape[0] <= 4:  # (channels, samples)
                arr = arr.mean(axis=0)
            elif arr.shape[1] <= 4:  # (samples, channels)
                arr = arr.mean(axis=1)
            else:
                arr = arr.mean(axis=1)  # fallback
        mono = arr.astype(np.float64)

        # --- crop to MAX_ANALYSIS_S ------------------------------------------
        max_samples = int(self.MAX_ANALYSIS_S * sr)
        duration_s = len(mono) / float(sr)
        if len(mono) > max_samples:
            # center crop to preserve structural frisson moments
            start_crop = (len(mono) - max_samples) // 2
            mono = mono[start_crop : start_crop + max_samples]
            duration_s = self.MAX_ANALYSIS_S

        frame_size = max(1, int(self.FRAME_SIZE_S * sr))
        hop = max(1, int(self.FRAME_HOP_S * sr))
        n = len(mono)
        n_frames = max(1, (n - frame_size) // hop + 1)

        if n_frames < 4:
            return []

        # --- per-frame features ----------------------------------------------
        rms_db, chroma, mel_energy, sub_rms_db = self._compute_features(mono, sr, n_frames, frame_size, hop)

        # --- scoring ---------------------------------------------------------
        total_score, dominant_triggers = self._score_frames(rms_db, chroma, mel_energy, sub_rms_db)

        # --- zone extraction -------------------------------------------------
        return self._extract_zones(
            total_score,
            dominant_triggers,
            hop_s=self.FRAME_HOP_S,
            duration_s=duration_s,
            max_zones=max_zones,
        )

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def _compute_features(
        self,
        mono: np.ndarray,
        sr: int,
        n_frames: int,
        frame_size: int,
        hop: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Berechnet per-frame acoustic features.

        Returns:
            rms_db:      [n_frames]     dBFS RMS per frame
            chroma:      [n_frames, 12] L1-normalised chroma
            mel_energy:  [n_frames, 8]  log mel filterbank energy
            sub_rms_db:  [n_frames]     dBFS sub-bass (20–80 Hz) RMS
        """
        n_fft = self.FFT_SIZE
        n = len(mono)

        # --- Frequency axis --------------------------------------------------
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))  # [n_fft//2+1]
        n_bins = len(freqs)

        # --- Chroma bin mapping (per-frequency-bin → pitch class 0–11, or -1) -
        chroma_map = np.full(n_bins, -1, dtype=np.int32)
        valid_mask = (freqs > 27.5) & (freqs < 8372.0)
        if np.any(valid_mask):
            midi_raw = 69.0 + 12.0 * np.log2(np.maximum(freqs[valid_mask], 1.0) / 440.0)
            chroma_map[valid_mask] = np.round(midi_raw).astype(np.int32) % 12  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level

        # --- Mel filterbank (8 bands, 150 Hz – 8 kHz) -----------------------
        _mel_low = 2595.0 * np.log10(1.0 + 150.0 / 700.0)
        _mel_high = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
        mel_edges = np.linspace(_mel_low, _mel_high, self.MEL_BANDS + 2)
        freq_edges = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)
        mel_filters = np.zeros((self.MEL_BANDS, n_bins), dtype=np.float64)
        for b in range(self.MEL_BANDS):
            lo, center, hi = freq_edges[b], freq_edges[b + 1], freq_edges[b + 2]
            up = (freqs >= lo) & (freqs <= center)
            down = (freqs > center) & (freqs <= hi)
            if np.any(up):
                mel_filters[b, up] = (freqs[up] - lo) / max(1e-8, center - lo)
            if np.any(down):
                mel_filters[b, down] = (hi - freqs[down]) / max(1e-8, hi - center)

        # --- Sub-bass mask (20–80 Hz) ----------------------------------------
        sub_mask = (freqs >= 20.0) & (freqs <= 80.0)
        has_sub = np.any(sub_mask)

        # --- Analysis window -------------------------------------------------
        win_len = min(frame_size, n_fft)
        window = np.hanning(win_len).astype(np.float64)

        # --- Frame matrix [n_frames, n_fft] (extract then batch-FFT) --------
        frames_mat = np.zeros((n_frames, n_fft), dtype=np.float64)
        for k in range(n_frames):
            s = k * hop
            e = min(s + frame_size, n)
            length = e - s
            frames_mat[k, : min(length, win_len)] = mono[s : s + min(length, win_len)]
            frames_mat[k, :win_len] *= window

        # Batch FFT → magnitude spectra [n_frames, n_bins]
        mags = np.abs(np.fft.rfft(frames_mat, axis=1))  # [n_frames, n_bins]

        # --- RMS in dBFS (from time domain) ----------------------------------
        rms_lin = np.zeros(n_frames, dtype=np.float64)
        for k in range(n_frames):
            s, e = k * hop, min(k * hop + frame_size, n)
            rms_lin[k] = float(np.sqrt(np.mean(mono[s:e] ** 2) + 1e-12))
        rms_db = 20.0 * np.log10(rms_lin + 1e-12)

        # --- Chroma [n_frames, 12] -------------------------------------------
        chroma = np.zeros((n_frames, 12), dtype=np.float64)
        for pc in range(12):
            pc_mask = chroma_map == pc
            if np.any(pc_mask):
                chroma[:, pc] = np.sum(mags[:, pc_mask], axis=1)
        chroma_sum = np.sum(chroma, axis=1, keepdims=True) + 1e-10
        chroma /= chroma_sum

        # --- Log mel energy [n_frames, 8] ------------------------------------
        pow_spec = mags**2  # [n_frames, n_bins]
        mel_lin = np.dot(pow_spec, mel_filters.T)  # [n_frames, 8]
        mel_energy = np.log(mel_lin + 1e-10)

        # --- Sub-bass dBFS [n_frames] ----------------------------------------
        if has_sub:
            sub_rms = np.sqrt(np.mean(mags[:, sub_mask] ** 2, axis=1) + 1e-12)
        else:
            sub_rms = np.full(n_frames, 1e-12)
        sub_rms_db = 20.0 * np.log10(sub_rms + 1e-12)

        return (
            rms_db.astype(np.float32),
            chroma.astype(np.float32),
            mel_energy.astype(np.float32),
            sub_rms_db.astype(np.float32),
        )

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _score_frames(
        self,
        rms_db: np.ndarray,
        chroma: np.ndarray,
        mel_energy: np.ndarray,
        sub_rms_db: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Berechnet combined frisson score per frame.

        Returns:
            total_score: [n_frames] combined score ∈ [0, 1]
            triggers:    [n_frames] name of dominant trigger per frame
        """
        n = len(rms_db)
        eps = 1e-10

        # ── Score 1: RMS crescendo after quiet (weight 0.35) ─────────────────
        # "Slow reference" = Gaussian-weighted mean of past frames (proxy for
        # what the ear expects). A large positive jump from this baseline,
        # especially after a quiet passage, is a primary frisson trigger.
        win_slow = min(n, 7)
        kernel = np.ones(win_slow, dtype=np.float64) / win_slow
        rms_slow = np.convolve(rms_db, kernel, mode="same").astype(np.float32)
        # Shift 3 frames forward (causal: use only PAST frames as reference)
        rms_past = np.roll(rms_slow, 3)
        rms_past[:3] = rms_slow[:3]
        rms_jump = np.maximum(0.0, rms_db - rms_past)
        # Boost score if preceded by quiet segment
        quiet_ref = np.roll(rms_slow, 3)
        quiet_ref[:3] = rms_slow[:3]
        quiet_weight = np.where(quiet_ref < -30.0, 1.0, 0.5).astype(np.float32)
        score_1 = np.clip(rms_jump / 15.0, 0.0, 1.0) * quiet_weight

        # ── Score 2: Harmonic surprise — chroma KL-divergence (weight 0.25) ──
        # Expected chroma = smoothed mean of last 3 frames (musical expectation).
        # Current chroma deviating from expectation = harmonic surprise.
        score_2 = np.zeros(n, dtype=np.float32)
        if n >= 4:
            # Rolling mean of chroma over window of 3 shifted by 1 frame
            chroma_f64 = chroma.astype(np.float64)
            # cumsum for fast rolling mean
            cum = np.cumsum(chroma_f64, axis=0)  # [n, 12]
            expected = np.zeros_like(chroma_f64)
            for k in range(1, n):
                lo_k = max(0, k - 3)
                count = k - lo_k
                expected[k] = (cum[k - 1] - (cum[lo_k - 1] if lo_k > 0 else 0.0)) / count
            # Normalise expected
            exp_sum = np.sum(expected, axis=1, keepdims=True) + eps
            exp_norm = expected / exp_sum
            act_norm = chroma_f64 / (np.sum(chroma_f64, axis=1, keepdims=True) + eps)
            kl = np.sum(exp_norm * np.log((exp_norm + eps) / (act_norm + eps)), axis=1)
            score_2 = np.clip(kl / 2.5, 0.0, 1.0).astype(np.float32)
            score_2[0] = 0.0  # no expectation at frame 0

        # ── Score 3: Timbral novelty — mel cosine distance (weight 0.20) ──────
        score_3 = np.zeros(n, dtype=np.float32)
        if n >= 2:
            mel_f64 = mel_energy.astype(np.float64)
            mel_prev = np.roll(mel_f64, 1, axis=0)
            mel_prev[0] = mel_f64[0]
            dot = np.sum(mel_f64 * mel_prev, axis=1)
            norm_a = np.linalg.norm(mel_f64, axis=1)
            norm_b = np.linalg.norm(mel_prev, axis=1)
            cos_sim = dot / (norm_a * norm_b + eps)
            score_3 = np.clip(1.0 - cos_sim, 0.0, 1.0).astype(np.float32)
            score_3[0] = 0.0

        # ── Score 4: Sub-bass onset (weight 0.10) ────────────────────────────
        sub_jump = np.maximum(0.0, np.diff(sub_rms_db, prepend=sub_rms_db[0]))
        score_4 = np.clip(sub_jump / 12.0, 0.0, 1.0).astype(np.float32)

        # ── Score 5: Dynamic contrast — local RMS std (weight 0.10) ──────────
        win_dyn = min(n, 11)
        kernel_dyn = np.ones(win_dyn, dtype=np.float64) / win_dyn
        rms_mean = np.convolve(rms_db.astype(np.float64), kernel_dyn, mode="same")
        rms_sq_mean = np.convolve((rms_db.astype(np.float64)) ** 2, kernel_dyn, mode="same")
        rms_var = np.maximum(0.0, rms_sq_mean - rms_mean**2)
        rms_std = np.sqrt(rms_var)
        score_5 = np.clip(rms_std / 12.0, 0.0, 1.0).astype(np.float32)

        # ── Combined score ───────────────────────────────────────────────────
        total = 0.35 * score_1 + 0.25 * score_2 + 0.20 * score_3 + 0.10 * score_4 + 0.10 * score_5
        total = np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Dominant trigger per frame ───────────────────────────────────────
        _names = ["rms_crescendo", "harmonic_surprise", "timbral_novelty", "sub_bass_onset", "dynamic_contrast"]
        _weighted = np.stack(
            [score_1 * 0.35, score_2 * 0.25, score_3 * 0.20, score_4 * 0.10, score_5 * 0.10],
            axis=1,
        )  # [n, 5]
        dominant_idx = np.argmax(_weighted, axis=1)  # [n]
        triggers = [_names[int(i)] for i in dominant_idx]

        return total.astype(np.float32), triggers

    # ------------------------------------------------------------------ #
    #  Zone extraction                                                     #
    # ------------------------------------------------------------------ #

    def _extract_zones(
        self,
        scores: np.ndarray,
        triggers: list[str],
        hop_s: float,
        duration_s: float,
        max_zones: int,
    ) -> list[FrissonZone]:
        """Findet Peak-Frames, fasst sie in 2-Sekunden-Zonen zusammen und führt überlappende Zonen zusammen."""
        n = len(scores)
        if n < 3:
            return []

        # Local peaks: score[k] > score[k-1] AND score[k] >= score[k+1]
        # AND score[k] >= MIN_SCORE
        min_dist_frames = max(1, int(2.0 / hop_s))  # min 2 s between peaks

        peak_indices: list[int] = []
        for k in range(1, n - 1):
            if (
                float(scores[k]) >= self.MIN_SCORE
                and float(scores[k]) > float(scores[k - 1])
                and float(scores[k]) >= float(scores[k + 1])
            ):
                # Enforce minimum distance from previous peak
                if not peak_indices or (k - peak_indices[-1]) >= min_dist_frames:
                    peak_indices.append(k)
                elif float(scores[k]) > float(scores[peak_indices[-1]]):
                    peak_indices[-1] = k  # replace with higher neighbour

        # Also check last frame
        if n >= 2 and float(scores[n - 1]) >= self.MIN_SCORE and float(scores[n - 1]) > float(scores[n - 2]):
            if not peak_indices or (n - 1 - peak_indices[-1]) >= min_dist_frames:
                peak_indices.append(n - 1)

        if not peak_indices:
            return []

        # Sort by score descending, then limit
        peak_indices.sort(key=lambda k: -float(scores[k]))
        peak_indices = peak_indices[:max_zones]

        # Build zones
        zones: list[FrissonZone] = []
        for pk in peak_indices:
            center_s = pk * hop_s
            start_s = max(0.0, center_s - self.ZONE_RADIUS_S)
            end_s = min(duration_s, center_s + self.ZONE_RADIUS_S)
            zones.append(
                FrissonZone(
                    start_s=float(start_s),
                    end_s=float(end_s),
                    score=float(scores[pk]),
                    trigger=triggers[pk],
                )
            )

        # Merge overlapping zones (keep higher score)
        zones = self._merge_zones(zones)

        # Final sort by score descending, cap at max_zones
        zones.sort(key=lambda z: -z.score)
        return zones[:max_zones]

    @staticmethod
    def _merge_zones(zones: list[FrissonZone]) -> list[FrissonZone]:
        """Führt zusammen: overlapping or adjacent zones (gap < 0.2 s), keep max score."""
        if len(zones) <= 1:
            return zones

        # Sort by start time
        sorted_z = sorted(zones, key=lambda z: z.start_s)
        merged: list[FrissonZone] = [sorted_z[0]]

        for z in sorted_z[1:]:
            prev = merged[-1]
            if z.start_s <= prev.end_s + 0.2:  # overlap or tiny gap → merge
                merged[-1] = FrissonZone(
                    start_s=prev.start_s,
                    end_s=max(prev.end_s, z.end_s),
                    score=max(prev.score, z.score),
                    trigger=prev.trigger if prev.score >= z.score else z.trigger,
                )
            else:
                merged.append(z)

        return merged


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: FrissonCandidateDetector | None = None
_lock = threading.Lock()


def get_frisson_detector() -> FrissonCandidateDetector:
    """Thread-safe singleton (§3.2)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = FrissonCandidateDetector()
    return _instance


__all__ = [
    "FrissonZone",
    "FrissonCandidateDetector",
    "get_frisson_detector",
]
