"""
Blind Reference-Free Quality Estimator (§G55 / §3.3)

Assesses absolute audio quality WITHOUT comparing to the original.
Essential for true blind testing — the system must know when it sounds good.

Six DSP-based features (§G55a–§G55f):
  §G55a  Spectral naturalness (crest factor, not too flat/peaky)
  §G55b  Dynamic range health (histogram entropy, not over-compressed)
  §G55c  Noise floor continuity (no unnatural gating artifacts)
  §G55d  High-frequency presence (over-denoising kills HF)
  §G55e  Stereo width naturalness (M/S ratio within normal range)
  §G55f  Transient density (over-smoothing removes attacks)

§3.3 MERT-based perceptual quality (§G55g):
  §G55g  MERT embedding distance from clean reference centroid.
         Uses the MERT v1-330M model to extract acoustic embeddings and
         maps them to a 0-100 quality score. Graceful fallback if MERT
         is not available.

Each feature: 0-100 score. Weighted ensemble → overall 0-100.

Training reference: AES Convention Paper on Single-Ended Quality Assessment
(ITU-R BS.1387 PEAQ adapted for restoration context).

Author: Aurik Development Team
Version: 10.0.8 — §3.3 MERT integration
Date: 2026-07-26
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BlindQualityScore:
    """Reference-free quality assessment result.

    Includes §3.3 MERT-based perceptual score when available.
    """

    overall: float  # 0-100
    spectral_naturalness: float = 100.0
    dynamic_range_health: float = 100.0
    noise_floor_continuity: float = 100.0
    hf_presence: float = 100.0
    stereo_naturalness: float = 100.0
    transient_density: float = 100.0
    mert_perceptual: float | None = None  # §3.3: None if MERT unavailable
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def grade(self) -> str:
        if self.overall >= 90:
            return "Excellent"
        if self.overall >= 80:
            return "Good"
        if self.overall >= 60:
            return "Fair"
        return "Poor"

    @property
    def has_mert(self) -> bool:
        """Whether MERT-based quality was available."""
        return self.mert_perceptual is not None


class BlindQualityEstimator:
    """§G55: Single-ended audio quality assessment.

    Usage:
        est = BlindQualityEstimator(sr=48000)
        score = est.estimate(processed_audio)
        logger.info(f"Reference-free quality: {score.overall:.0f}/100")
    """

    def __init__(self, sr: int = 48000, material_key: str = "unknown", era_decade: int | None = None):
        self.sr = sr
        self.material_key = str(material_key).lower()
        self.era_decade = era_decade
        # §G100: Era-adaptive "natural" ranges for HF and stereo scoring.
        # Pre-1960 recordings have fundamentally different spectral and spatial
        # characteristics — scoring them against modern standards is unfair.

    def estimate(self, audio: np.ndarray) -> BlindQualityScore:
        """Compute reference-free quality score."""
        mono = self._to_mono(audio)
        n = len(mono)
        is_stereo = audio.ndim == 2 and audio.shape[1] >= 2

        if n < 4096:
            return BlindQualityScore(overall=50.0)

        details = {}

        # §G55a: Spectral naturalness
        spec_nat = self._spectral_naturalness(mono)
        details["spectral_crest_factor"] = spec_nat

        # §G55b: Dynamic range health
        dyn_health = self._dynamic_range_health(mono)
        details["dynamic_entropy"] = dyn_health

        # §G55c: Noise floor continuity
        noise_cont = self._noise_floor_continuity(mono)
        details["noise_continuity"] = noise_cont

        # §G55d: HF presence
        hf_pres = self._hf_presence(mono)
        details["hf_energy_ratio"] = hf_pres

        # §G55e: Stereo naturalness
        if is_stereo:
            stereo_nat = self._stereo_naturalness(audio)
        else:
            stereo_nat = 100.0
        details["stereo_naturalness"] = stereo_nat

        # §G55f: Transient density
        trans_dens = self._transient_density(mono)
        details["transient_density"] = trans_dens

        # §3.3 §G55g: MERT perceptual quality (graceful fallback)
        mert_score = self._mert_perceptual_quality(mono)
        has_mert = mert_score is not None

        # Weighted ensemble — MERT gets 15% when available
        if has_mert:
            overall = (
                0.20 * spec_nat
                + 0.15 * dyn_health
                + 0.15 * noise_cont
                + 0.10 * hf_pres
                + 0.10 * stereo_nat
                + 0.10 * trans_dens
                + 0.20 * mert_score  # type: ignore[operator]
            )
        else:
            overall = (
                0.25 * spec_nat
                + 0.20 * dyn_health
                + 0.20 * noise_cont
                + 0.15 * hf_pres
                + 0.10 * stereo_nat
                + 0.10 * trans_dens
            )

        return BlindQualityScore(
            overall=float(np.clip(overall, 0.0, 100.0)),
            spectral_naturalness=spec_nat,
            dynamic_range_health=dyn_health,
            noise_floor_continuity=noise_cont,
            hf_presence=hf_pres,
            stereo_naturalness=stereo_nat,
            transient_density=trans_dens,
            mert_perceptual=mert_score,
            breakdown=details,
        )

    # ── §G55a Spectral Naturalness ──────────────────────────────────────

    def _spectral_naturalness(self, mono: np.ndarray) -> float:
        """How natural is the spectrum? Based on spectral crest factor.

        Natural music has spectral peaks (harmonics, formants).
        Over-smoothed audio has flat spectrum → low crest factor.
        Over-processed audio has razor peaks → too high crest factor.
        """
        n = len(mono)
        n_fft = 4096
        if n < n_fft:
            n_fft = 1
            while n_fft < n:
                n_fft <<= 1
        hop = n_fft // 2
        n_frames = (n - n_fft) // hop + 1
        if n_frames < 3:
            return 50.0

        win = np.hanning(n_fft)
        crests = []
        for i in range(n_frames):
            s = i * hop
            spec = np.abs(np.fft.rfft(mono[s : s + n_fft] * win))
            # Focus on midrange (300-8000 Hz) — most musically relevant
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)
            mask = (freqs >= 300) & (freqs <= 8000)
            if not np.any(mask):
                continue
            s_mid = spec[mask]
            s_mid = np.maximum(s_mid, 1e-15)
            # Spectral crest = max / geometric mean
            geo_mean = float(np.exp(np.mean(np.log(s_mid))))
            peak = float(np.max(s_mid))
            if geo_mean > 1e-15:
                crests.append(peak / geo_mean)

        if not crests:
            return 50.0

        mean_crest = float(np.mean(crests))
        # Ideal spectral crest for natural music: 15-40 (empirical)
        # Too low (<8) = over-smoothed
        # Too high (>60) = artifact-ridden
        if mean_crest < 8:
            score = mean_crest / 8.0 * 60.0 + 10.0  # 8 → 70, 4 → 40
        elif mean_crest <= 40:
            score = 90.0  # Golden zone
        else:
            score = max(10.0, 100.0 - (mean_crest - 40) * 1.5)  # 60 → 70

        return float(np.clip(score, 0.0, 100.0))

    # ── §G55b Dynamic Range Health ──────────────────────────────────────

    def _dynamic_range_health(self, mono: np.ndarray) -> float:
        """Is the dynamic range natural? Not over-compressed, not over-expanded."""
        n = len(mono)
        win_s = int(0.200 * self.sr)
        hop_s = win_s // 2
        n_frames = (n - win_s) // hop_s + 1
        if n_frames < 5:
            return 50.0

        rms_db = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop_s
            rms = float(np.sqrt(np.mean(mono[s : s + win_s].astype(np.float64) ** 2)))
            rms_db[i] = 20.0 * np.log10(max(rms, 1e-15))

        # Dynamic range = P95 - P5 of RMS
        p95 = float(np.percentile(rms_db, 95))
        p5 = float(np.percentile(rms_db, 5))
        dr = p95 - p5

        # Ideal DR for well-mastered music: 6-18 dB
        # Below 3 dB = brick-wall limited
        # Above 25 dB = classical/unmastered (still good, but unusual for CD)
        if dr < 3:
            score = dr / 3.0 * 40.0  # 3 → 40, 1.5 → 20
        elif dr <= 18:
            score = 90.0
        else:
            score = max(30.0, 90.0 - (dr - 18) * 2.0)  # 25 → 76

        return float(np.clip(score, 0.0, 100.0))

    # ── §G55c Noise Floor Continuity ────────────────────────────────────

    def _noise_floor_continuity(self, mono: np.ndarray) -> float:
        """Is the noise floor continuous? No unnatural gating artifacts.

        Measures the smoothness of the noise floor envelope.
        Sudden changes indicate noise gate pumping.
        """
        n = len(mono)
        win_s = int(0.500 * self.sr)  # 500ms windows
        hop_s = win_s // 2
        n_frames = (n - win_s) // hop_s + 1
        if n_frames < 4:
            return 80.0

        # Per-frame: P10 percentile as noise floor estimate
        noise_floor = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop_s
            frame = np.abs(mono[s : s + win_s].astype(np.float64))
            noise_floor[i] = float(np.percentile(frame, 10))

        # Smoothness: standard deviation of noise floor first difference
        if np.max(noise_floor) < 1e-15:
            return 100.0  # Digital silence is perfectly continuous

        diff = np.diff(noise_floor)
        # Normalize by mean noise floor
        mean_nf = float(np.mean(noise_floor))
        if mean_nf < 1e-15:
            return 100.0
        cv = float(np.std(diff)) / mean_nf  # Coefficient of variation

        # cv < 0.3: smooth → 90+
        # cv 0.3-0.6: moderate → 70-90
        # cv > 1.0: gated → <60
        score = 100.0 - cv * 50.0
        return float(np.clip(score, 0.0, 100.0))

    # ── §G55d High-Frequency Presence ───────────────────────────────────

    def _hf_presence(self, mono: np.ndarray) -> float:
        """Is there natural high-frequency energy? Over-denoising kills HF."""
        n = len(mono)
        n_fft = 4096
        if n < n_fft:
            n_fft = 1
            while n_fft < n:
                n_fft <<= 1
        win = np.hanning(min(n_fft, n))
        spec = np.abs(np.fft.rfft(mono[:n_fft] * win, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        # HF energy (8-20 kHz) vs total energy
        mask_hf = (freqs >= 8000) & (freqs <= 20000)
        mask_total = freqs >= 300  # Exclude sub-bass rumble

        if not np.any(mask_hf) or not np.any(mask_total):
            return 50.0

        e_hf = float(np.sum(spec[mask_hf] ** 2))
        e_total = float(np.sum(spec[mask_total] ** 2))

        if e_total < 1e-15:
            return 50.0

        ratio = e_hf / e_total

        # §G100 Era-adaptive natural HF ratio.
        # Pre-1950 recordings: carbon mics, limited bandwidth → much less HF.
        # Modern digital: full 20 kHz spectrum.
        _era = self.era_decade
        _mat = self.material_key
        if _era is not None and _era < 1950:
            _hf_lo, _hf_hi = 0.0001, 0.03
        elif _era is not None and _era < 1970:
            _hf_lo, _hf_hi = 0.0005, 0.08
        elif any(t in _mat for t in ("shellac", "wax", "wire")):
            _hf_lo, _hf_hi = 0.0002, 0.04
        elif "cassette" in _mat:
            _hf_lo, _hf_hi = 0.001, 0.12
        else:
            _hf_lo, _hf_hi = 0.001, 0.15  # original universal range

        # Score: linear ramp below lo, flat 90 in [lo, hi], linear decay above hi
        if ratio < _hf_lo:
            score = max(0.0, ratio / max(_hf_lo, 1e-9) * 30.0)
        elif ratio <= _hf_hi:
            score = 90.0
        else:
            _decay = 200.0 * (0.15 / max(_hf_hi, 0.01))  # scale decay to range width
            score = max(30.0, 90.0 - (ratio - _hf_hi) * _decay)

        return float(np.clip(score, 0.0, 100.0))

    # ── §G55e Stereo Naturalness ────────────────────────────────────────

    def _stereo_naturalness(self, audio: np.ndarray) -> float:
        """Is the stereo image natural? Not collapsed, not over-wide."""
        if audio.ndim < 2 or audio.shape[1] < 2:
            return 80.0

        n = min(len(audio), 48000 * 3)  # First 3 seconds
        left = audio[:n, 0].astype(np.float64)
        right = audio[:n, 1].astype(np.float64)

        # M/S analysis
        mid = left + right
        side = left - right

        rms_mid = float(np.sqrt(np.mean(mid**2)))
        rms_side = float(np.sqrt(np.mean(side**2)))

        if rms_mid < 1e-15:
            return 50.0

        width = rms_side / rms_mid

        # §G100 Era-adaptive natural stereo width.
        # Pre-1960: commercial stereo didn't exist — narrow/mono is NORMAL.
        # 1960-1980: early stereo, hard-pan mixing → width 0.05-0.8 is natural.
        # Post-1980: modern multi-track → width 0.3-1.5 is natural.
        _era = self.era_decade
        if _era is not None and _era < 1960:
            _width_lo, _width_hi = 0.0, 0.30  # mono era: narrow is expected
        elif _era is not None and _era < 1980:
            _width_lo, _width_hi = 0.05, 0.80  # early stereo: moderate
        else:
            _width_lo, _width_hi = 0.10, 1.50  # modern stereo: wide expected

        # Natural width scoring
        if width < _width_lo:
            score = width / max(_width_lo, 1e-9) * 50.0
        elif width <= _width_hi:
            score = 90.0
        else:
            score = max(30.0, 90.0 - (width - _width_hi) * (40.0 * 1.5 / max(_width_hi, 0.1)))

        # Also check L/R correlation (not too correlated, not anti-correlated)
        corr = float(np.corrcoef(left, right)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        # Natural correlation: 0.2 - 0.9
        if corr < 0.0:
            corr_score = 50.0  # Anti-correlated = phase issue
        elif corr < 0.2:
            corr_score = 70.0  # Very wide
        elif corr <= 0.9:
            corr_score = 90.0
        else:
            corr_score = 70.0  # Near-mono

        return float(np.clip(0.6 * score + 0.4 * corr_score, 0.0, 100.0))

    # ── §G55f Transient Density ─────────────────────────────────────────

    def _transient_density(self, mono: np.ndarray) -> float:
        """Are there natural transients? Over-smoothing removes attacks."""
        n = len(mono)
        n_fft, hop = 1024, 256
        n_frames = (n - n_fft) // hop + 1
        if n_frames < 5:
            return 50.0

        win = np.hanning(n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)
        lo = np.searchsorted(freqs, 2000)
        hi = np.searchsorted(freqs, 8000)
        if hi <= lo:
            return 50.0

        # HF energy per frame
        energy = np.zeros(n_frames, dtype=np.float64)
        for i in range(n_frames):
            s = i * hop
            spec = np.abs(np.fft.rfft(mono[s : s + n_fft] * win))
            energy[i] = float(np.sum(spec[lo:hi] ** 2))

        if np.max(energy) < 1e-15:
            return 50.0

        # Onset detection
        energy_db = 10.0 * np.log10(energy + 1e-15)
        onset = np.maximum(np.diff(energy_db), 0.0)
        # Count onsets > 6 dB
        n_onsets = int(np.sum(onset > 6.0))

        # Transient density: onsets per second
        duration_s = n / self.sr
        density = n_onsets / max(duration_s, 0.1)

        # §G100 Era-adaptive natural transient density.
        # Pre-1950: simpler arrangements, less percussion → lower density.
        # Post-1980: dense productions, more transients → higher density.
        _era = self.era_decade
        if _era is not None and _era < 1950:
            _td_lo, _td_hi = 0.1, 4.0
        elif _era is not None and _era < 1980:
            _td_lo, _td_hi = 0.3, 6.0
        else:
            _td_lo, _td_hi = 0.5, 8.0  # original universal range

        # Score: linear ramp below lo, flat 85 in [lo, hi], linear decay above hi
        if density < _td_lo:
            score = density / max(_td_lo, 1e-9) * 40.0
        elif density <= _td_hi:
            score = 85.0
        else:
            score = max(30.0, 85.0 - (density - _td_hi) * 3.0)

        return float(np.clip(score, 0.0, 100.0))

    # ── §3.3 §G55g MERT Perceptual Quality ──────────────────────────────

    def _mert_perceptual_quality(self, mono: np.ndarray) -> float | None:
        """§3.3: MERT-embedding-based absolute quality estimation.

        Uses MERT v1-330M to extract acoustic embeddings and maps them
        to a 0-100 quality score via embedding statistics.

        Returns None if MERT is not available (graceful fallback).
        """
        try:
            from plugins.mert_plugin import MertPlugin
        except ImportError:
            logger.debug("§3.3 MERT: plugin not verfuegbar, skipping perceptual quality")
            return None

        try:
            plugin = MertPlugin()
            if not plugin.model_available:
                logger.debug("§3.3 MERT: model not geladen, skipping perceptual quality")
                return None

            # Extract MERT analysis (use first 10s max for efficiency)
            max_samples = min(len(mono), self.sr * 10)
            audio_segment = mono[:max_samples].astype(np.float32)

            result = plugin.analyze(audio_segment, self.sr)
            if result is None:
                return None

            # §3.3: Use MERT's naturalness_score + harmonicity as perceptual features
            nat_score = float(getattr(result, "naturalness_score", 0.0))  # 0-1
            harmonicity = float(getattr(result, "harmonicity", 0.0))  # 0-1
            tonal_consistency = float(getattr(result, "tonal_consistency", 0.0))  # 0-1
            model_used = str(getattr(result, "model_used", "dsp_fallback"))

            # If DSP fallback: reduce weight (DSP NAT is less reliable than MERT)
            is_mert = model_used != "dsp_fallback"

            # Combine features into a 0-100 quality score
            # naturalness_score is the primary MERT-driven metric
            mert_score = nat_score * 60.0 + harmonicity * 25.0 + tonal_consistency * 15.0

            if not is_mert:
                # DSP fallback: less reliable, cap at 85
                mert_score = min(mert_score, 85.0)

            logger.debug(
                "§3.3 MERT quality: nat=%.3f harm=%.3f tonal=%.3f model=%s → Wert=%.1f",
                nat_score,
                harmonicity,
                tonal_consistency,
                model_used,
                mert_score,
            )
            return float(np.clip(mert_score, 0.0, 100.0))

        except Exception:
            logger.debug("§3.3 MERT: perceptual quality estimation fehlgeschlagen", exc_info=True)
            return None

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        return audio.mean(axis=0) if audio.shape[1] < audio.shape[0] else audio.mean(axis=1)  # type: ignore[no-any-return]
