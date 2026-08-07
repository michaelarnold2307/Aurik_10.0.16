"""

Professional formant tracking, correction, and enhancement:
- LPC-based Formant Tracking
- Formant Drift Detection & Correction
- Singer's Formant Enhancement (2.5-3.5 kHz "ring")
- Voice Identity Protection

Author: AURIK Development Team
Version: 1.0.0
Date: 9. Februar 2026
"""

import logging

logger = logging.getLogger(__name__)

import warnings

import numpy as np
from scipy import signal
from scipy.signal import lfilter

try:
    import pyworld as _pw  # type: ignore[import-untyped]

    _HAS_PYWORLD: bool = True
except ImportError:
    _pw = None  # type: ignore[assignment]
    _HAS_PYWORLD: bool = False  # type: ignore[no-redef]

warnings.filterwarnings("ignore", category=RuntimeWarning)


class FormantTracker:
    """
    LPC-based formant tracking for voice analysis.
    """

    def __init__(self, n_formants: int = 5, frame_length_ms: float = 25.0, hop_length_ms: float = 10.0):
        """
        Parameters
        ----------
        n_formants : int
            Number of formants to track (typically 4-5)
        frame_length_ms : float
            Frame length in milliseconds
        hop_length_ms : float
            Hop length in milliseconds
        """
        self.n_formants = n_formants
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms

    def track(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Verfolgt formants over time using LPC analysis.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz

        Returns
        -------
        formant_freqs : np.ndarray
            Formant frequencies over time, shape (n_frames, n_formants)
        formant_bandwidths : np.ndarray
            Formant bandwidths over time, shape (n_frames, n_formants)
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        frame_length = int(self.frame_length_ms * sr / 1000)
        hop_length = int(self.hop_length_ms * sr / 1000)

        # Pre-emphasis (boost high frequencies for better formant detection)
        audio_emphasized = self._pre_emphasis(audio)

        # Extract frames
        frames = self._extract_frames(audio_emphasized, frame_length, hop_length)

        # Track formants per frame
        formant_freqs = []
        formant_bandwidths = []

        for frame in frames:
            freqs, bws = self._analyze_frame(frame, sr)
            formant_freqs.append(freqs)
            formant_bandwidths.append(bws)

        formant_freqs = np.array(formant_freqs)  # type: ignore[assignment]
        formant_bandwidths = np.array(formant_bandwidths)  # type: ignore[assignment]

        # Smooth trajectories (removes outliers)
        formant_freqs = self._smooth_trajectories(formant_freqs)  # type: ignore[arg-type, assignment]

        # NaN/Inf-Guard
        formant_freqs = np.nan_to_num(formant_freqs, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[assignment]
        formant_bandwidths = np.nan_to_num(formant_bandwidths, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[assignment]

        return formant_freqs, formant_bandwidths  # type: ignore[return-value]

    def _pre_emphasis(self, audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """
        Wendet an: pre-emphasis filter to boost high frequencies.
        """
        return lfilter([1, -coeff], [1], audio)  # type: ignore[no-any-return]

    def _extract_frames(self, audio: np.ndarray, frame_length: int, hop_length: int) -> list[np.ndarray]:
        """
        Extrahiert overlapping frames from audio.
        """
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            # Apply Hamming window
            frame = frame * np.hamming(len(frame))
            frames.append(frame)

        return frames

    def _analyze_frame(self, frame: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Analysiert einen einzelnen Frame auf Formanten mittels LPC.

        LPC (Linear Predictive Coding) models the vocal tract as an all-pole filter.
        The poles correspond to formant frequencies.
        """
        # LPC order: textbook sr_kHz*2+4 ≈ 100 at 48 kHz is overkill for F1–F5.
        # Spec §4.1: 30–40 at 48 kHz sufficient for F1–F5; 16–20 at 16 kHz.
        lpc_order = min(int(sr / 1000) + 4, 40)  # 52@48kHz → clamped to 40

        # Compute LPC coefficients
        lpc_coeffs = self._compute_lpc(frame, lpc_order)

        # Guard: degenerate LPC coefficients → LAPACK DLASCL failure
        if not np.all(np.isfinite(lpc_coeffs)):
            return np.zeros(self.n_formants), np.full(self.n_formants, 500.0)

        # Find poles (roots of LPC polynomial)
        try:
            roots = np.roots(lpc_coeffs)
        except (np.linalg.LinAlgError, ValueError):
            return np.zeros(self.n_formants), np.full(self.n_formants, 500.0)

        # Convert to frequencies and bandwidths
        freqs, bws = self._roots_to_formants(roots, sr)

        # Select top n_formants (by bandwidth - narrower is more prominent)
        if len(freqs) > self.n_formants:
            indices = np.argsort(bws)[: self.n_formants]
            freqs = freqs[indices]
            bws = bws[indices]
            # Sort by frequency
            sort_idx = np.argsort(freqs)
            freqs = freqs[sort_idx]
            bws = bws[sort_idx]

        # Pad if necessary
        if len(freqs) < self.n_formants:
            freqs = np.pad(freqs, (0, self.n_formants - len(freqs)), constant_values=0)
            bws = np.pad(bws, (0, self.n_formants - len(bws)), constant_values=0)

        return freqs, bws

    def _compute_lpc(self, frame: np.ndarray, order: int) -> np.ndarray:
        """
        Berechnet LPC coefficients using autocorrelation method.
        """
        # Autocorrelation
        r = signal.correlate(frame, frame, mode="full", method="fft")
        r = r[len(r) // 2 :]
        r = r[: order + 1]

        # Guard: degenerate autocorrelation (NaN/Inf/near-zero) triggers LAPACK DLASCL failure
        if not np.isfinite(r).all() or r[0] < 1e-12:
            return np.array([1.0] + [0.0] * order)  # neutral LPC polynomial (passthrough)

        # Levinson-Durbin recursion
        lpc_coeffs = self._levinson_durbin(r, order)

        return lpc_coeffs

    def _levinson_durbin(self, r: np.ndarray, order: int) -> np.ndarray:
        """
        Levinson-Durbin recursion for LPC coefficients.
        """
        a = np.zeros(order + 1)
        a[0] = 1.0

        e = r[0]

        for i in range(1, order + 1):
            lambda_i = -np.sum(a[:i] * r[i:0:-1]) / e if e != 0 else 0.0  # §3.1

            a_new = np.zeros(i + 1)
            a_new[:i] = a[:i] + lambda_i * a[i - 1 :: -1]
            a_new[i] = lambda_i

            a = a_new
            e *= 1 - lambda_i**2

        return a

    def _roots_to_formants(self, roots: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Konvertiert LPC polynomial roots to formant frequencies and bandwidths.
        """
        # Keep only roots inside unit circle
        roots = roots[np.abs(roots) < 1.0]

        # Keep only complex conjugate pairs (formants)
        roots = roots[np.imag(roots) >= 0]

        if len(roots) == 0:
            return np.array([]), np.array([])

        # Convert to frequency (Hz)
        angles = np.angle(roots)
        freqs = angles * sr / (2 * np.pi)

        # Convert to bandwidth (Hz)
        radii = np.abs(roots)
        radii_safe = np.maximum(radii, 1e-10)  # §3.1: log(0) bei Stille verhindern
        bws = -sr / (2 * np.pi) * np.log(radii_safe)

        # Filter: keep only positive frequencies in vocal range (50-5000 Hz)
        valid = (freqs > 50) & (freqs < 5000) & (bws > 0) & (bws < 1000)
        freqs = freqs[valid]
        bws = bws[valid]

        return freqs, bws

    def _smooth_trajectories(self, formant_freqs: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Glättet formant trajectories using median filter.
        """
        if len(formant_freqs) < window_size:
            return formant_freqs

        smoothed = formant_freqs.copy()

        for i in range(self.n_formants):
            trajectory = formant_freqs[:, i]
            # Median filter to remove outliers
            smoothed[:, i] = signal.medfilt(trajectory, kernel_size=window_size)

        return smoothed


class FormantCorrector:
    """
    Corrects formant drift and preserves voice identity.
    """

    def __init__(self, max_drift_hz: float = 50.0, correction_strength: float = 0.7):
        """
        Parameters
        ----------
        max_drift_hz : float
            Maximum allowed formant drift in Hz
        correction_strength : float
            Correction strength (0.0-1.0)
        """
        self.max_drift_hz = max_drift_hz
        self.correction_strength = np.clip(correction_strength, 0.0, 1.0)

    def detect_drift(
        self, formant_freqs: np.ndarray, reference_formants: np.ndarray | None = None
    ) -> tuple[bool, dict]:
        """
        Erkennt formant drift.

        Parameters
        ----------
        formant_freqs : np.ndarray
            Formant frequencies over time, shape (n_frames, n_formants)
        reference_formants : np.ndarray, optional
            Reference formant frequencies (median or provided)

        Returns
        -------
        has_drift : bool
            Whether drift was detected
        drift_info : Dict
            Drift statistics per formant
        """
        if reference_formants is None:
            # Use median as reference
            reference_formants = np.asarray(np.median(formant_freqs, axis=0))

        # Compute drift per formant
        drift_info = {}
        has_drift = False

        for i in range(formant_freqs.shape[1]):
            trajectory = formant_freqs[:, i]
            ref_freq = reference_formants[i]

            if ref_freq == 0:  # Empty formant
                continue

            # Drift = difference from reference
            drift = trajectory - ref_freq
            max_drift = np.max(np.abs(drift))
            mean_drift = np.mean(np.abs(drift))

            drift_info[f"F{i + 1}"] = {
                "reference_hz": ref_freq,
                "max_drift_hz": max_drift,
                "mean_drift_hz": mean_drift,
                "needs_correction": max_drift > self.max_drift_hz,
            }

            if max_drift > self.max_drift_hz:
                has_drift = True

        return has_drift, drift_info

    def correct(
        self, audio: np.ndarray, sr: int, formant_freqs: np.ndarray, target_formants: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Correct formant drift using formant shifting.

        This is a simplified implementation. Full implementation would use
        sophisticated formant shifting (e.g., WORLD vocoder, PSOLA).

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz
        formant_freqs : np.ndarray
            Detected formant frequencies
        target_formants : np.ndarray, optional
            Target formant frequencies

        Returns
        -------
        audio_corrected : np.ndarray
            Corrected audio
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if target_formants is None:
            # Use median as target
            target_formants = np.asarray(np.median(formant_freqs, axis=0))

        # Median detected formants for this correction pass
        n_correct = min(3, len(target_formants))
        current_formants = np.array(
            [float(np.median(formant_freqs[:, i])) if formant_freqs.shape[1] > i else 0.0 for i in range(n_correct)]
        )

        # Primary: WORLD spectral-envelope warp (Morise et al. 2016) — phase-transparent.
        # Fallback: biquad EQ approximation when pyworld is unavailable.
        audio_full = audio.copy()
        if _HAS_PYWORLD and _pw is not None:
            try:
                audio_full = self._correct_with_world(audio, sr, current_formants, target_formants[:n_correct])
            except Exception:
                for i in range(n_correct):
                    if target_formants[i] == 0 or current_formants[i] == 0:
                        continue
                    shift_hz = target_formants[i] - current_formants[i]
                    if np.abs(shift_hz) > self.max_drift_hz:
                        audio_full = self._apply_formant_shift_eq(audio_full, sr, current_formants[i], shift_hz)
        else:
            for i in range(n_correct):
                if target_formants[i] == 0 or current_formants[i] == 0:
                    continue
                shift_hz = target_formants[i] - current_formants[i]
                if np.abs(shift_hz) > self.max_drift_hz:
                    audio_full = self._apply_formant_shift_eq(audio_full, sr, current_formants[i], shift_hz)

        # Blend with original
        audio_corrected = self.correction_strength * audio_full + (1 - self.correction_strength) * audio

        # NaN/Inf-Guard and clipping
        audio_corrected = np.nan_to_num(audio_corrected, nan=0.0, posinf=0.0, neginf=0.0)
        audio_corrected = np.clip(audio_corrected, -1.0, 1.0)

        return audio_corrected  # type: ignore[no-any-return]

    def _apply_formant_shift_eq(self, audio: np.ndarray, sr: int, center_freq: float, shift_hz: float) -> np.ndarray:
        """
        Wendet an: EQ-based formant shift (approximation).
        """
        # Target frequency
        target_freq = center_freq + shift_hz

        # Bandwidth (10% of center frequency)
        bandwidth = center_freq * 0.1

        # Attenuation at original position
        audio = self._apply_notch(audio, sr, center_freq, bandwidth, attenuation_db=3)

        # Boost at target position
        audio = self._apply_peak(audio, sr, target_freq, bandwidth, gain_db=3)

        return audio

    def _apply_notch(
        self, audio: np.ndarray, sr: int, freq: float, bandwidth: float, attenuation_db: float
    ) -> np.ndarray:
        """
        Wendet an: notch filter (attenuation).
        """
        Q = freq / bandwidth

        # Biquad notch filter
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = 1
        b1 = -2 * np.cos(w0)
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        # Apply with attenuation strength
        gain = 10 ** (-attenuation_db / 20)
        audio_filtered = signal.lfilter(b, a, audio)
        audio = audio * gain + audio_filtered * (1 - gain)

        return audio

    def _apply_peak(self, audio: np.ndarray, sr: int, freq: float, bandwidth: float, gain_db: float) -> np.ndarray:
        """
        Wendet an: peak filter (boost).
        """
        Q = freq / bandwidth
        A = 10 ** (gain_db / 40)

        # Biquad peak filter
        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        audio = np.asarray(signal.lfilter(b, a, audio), dtype=np.float64)

        return audio

    def _correct_with_world(
        self,
        audio: np.ndarray,
        sr: int,
        current_formants: np.ndarray,
        target_formants: np.ndarray,
    ) -> np.ndarray:
        """WORLD spectral-envelope warp for phase-transparent formant correction.

        Decomposes audio into F0, spectral envelope SP, and aperiodicity AP via
        WORLD (Morise et al. 2016).  For each of the first three formants a
        Gaussian-weighted frequency remap moves the detected SP peak toward the
        target frequency.  Audio is resynthesised with the original F0 and AP,
        preserving timing and voice naturalness without EQ phase distortion.

        Scientific basis:
            Morise et al. (2016) — WORLD: A Vocoder-Based High-Quality Speech
            Synthesis System for Real-Time Applications. IEICE Trans. A.
            Toda & Shikano (2005) — Voice conversion based on maximum likelihood
            estimation of spectral parameter trajectory. Interspeech.

        Args:
            audio:            Mono float32/64 audio, 48 000 Hz.
            sr:               Sample rate — must be 48 000 Hz.
            current_formants: Detected median formant frequencies (Hz), shape (≤3,).
            target_formants:  Target formant frequencies (Hz), same shape.

        Returns:
            Corrected mono audio (float32, same length as input).
        """
        audio_f64 = np.asarray(audio.flatten(), dtype=np.float64)
        sr_f64 = float(sr)

        f0, timeaxis = _pw.harvest(audio_f64, sr_f64)
        f0_sm = _pw.stonemask(audio_f64, f0, timeaxis, sr_f64)
        sp = _pw.cheaptrick(audio_f64, f0_sm, timeaxis, sr_f64)  # (n_frames, bins)
        ap = _pw.d4c(audio_f64, f0_sm, timeaxis, sr_f64)

        freq_axis = np.linspace(0.0, sr_f64 / 2.0, sp.shape[1])
        sp_warped = np.empty_like(sp)
        for fi in range(sp.shape[0]):
            sp_warped[fi] = self._warp_sp_frame(sp[fi], freq_axis, current_formants, target_formants)

        out = _pw.synthesize(f0_sm, sp_warped, ap, sr_f64)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # Match original length (WORLD synthesis may pad by a few samples)
        n = len(audio_f64)
        if len(out) >= n:
            out = out[:n]
        else:
            out = np.pad(out, (0, n - len(out)))
        return np.asarray(out, dtype=np.float32)

    def _warp_sp_frame(
        self,
        sp_frame: np.ndarray,
        freq_axis: np.ndarray,
        src_formants: np.ndarray,
        tgt_formants: np.ndarray,
    ) -> np.ndarray:
        """Gaussian-weighted frequency remap of a single spectral-envelope frame.

        For each formant pair (src_f, tgt_f) a Gaussian centred at *tgt_f* with
        bandwidth max(200, 0.15 × tgt_f) Hz pulls energy from *src_f* toward the
        target position via interpolation of the SP vector.  The warp magnitude
        is scaled by ``self.correction_strength`` so identity is preserved when
        strength = 0.  This achieves formant relocation without IIR phase
        distortion (Toda & Shikano 2005).

        Args:
            sp_frame:     1-D spectral envelope frame (linear power, not dB).
            freq_axis:    Corresponding frequency axis in Hz, same length.
            src_formants: Source formant frequencies (Hz).
            tgt_formants: Target formant frequencies (Hz).

        Returns:
            Warped spectral envelope frame, same shape as *sp_frame*.
        """
        warp = freq_axis.copy()
        for src_f, tgt_f in zip(src_formants, tgt_formants):
            if src_f <= 0.0 or tgt_f <= 0.0:
                continue
            bw = max(200.0, tgt_f * 0.15)
            weights = np.exp(-0.5 * ((freq_axis - tgt_f) / bw) ** 2)
            # Inverse warp: when reading at tgt_f, pull energy from src_f
            delta = (src_f - tgt_f) * self.correction_strength
            warp += weights * delta
        warp = np.clip(warp, 0.0, freq_axis[-1])
        return np.interp(warp, freq_axis, sp_frame)


class SingersFormantEnhancer:
    """
    Enhances the "singer's formant" (2.5-3.5 kHz resonance ring).

    The singer's formant is a clustering of F3, F4, F5 around 2.5-3.5 kHz,
    creating a "ring" that helps trained singers project over orchestras.
    """

    def __init__(self, target_freq_hz: float = 3000.0, bandwidth_hz: float = 250.0, gain_db: float = 3.0):
        """
        Parameters
        ----------
        target_freq_hz : float
            Target singer's formant frequency (2500-3500 Hz)
        bandwidth_hz : float
            Bandwidth of enhancement
        gain_db : float
            Gain in dB
        """
        self.target_freq_hz = target_freq_hz
        self.bandwidth_hz = bandwidth_hz
        self.gain_db = gain_db

    def detect_singers_formant(self, formant_freqs: np.ndarray) -> tuple[bool, float]:
        """
        Erkennt if singer's formant is present.

        Parameters
        ----------
        formant_freqs : np.ndarray
            Formant frequencies, shape (n_frames, n_formants)

        Returns
        -------
        has_singers_formant : bool
            Whether singer's formant is detected
        strength : float
            Strength of singer's formant (0.0-1.0)
        """
        # Look for clustering of F3, F4, F5 in 2.5-3.5 kHz range
        target_range = (2500, 3500)

        if formant_freqs.shape[1] < 3:
            return False, 0.0

        # Check F3, F4, F5 (indices 2, 3, 4)
        f3 = formant_freqs[:, 2] if formant_freqs.shape[1] > 2 else np.zeros(len(formant_freqs))
        f4 = formant_freqs[:, 3] if formant_freqs.shape[1] > 3 else np.zeros(len(formant_freqs))
        f5 = formant_freqs[:, 4] if formant_freqs.shape[1] > 4 else np.zeros(len(formant_freqs))

        # Count frames where formants cluster in target range
        in_range_3 = (f3 >= target_range[0]) & (f3 <= target_range[1])
        in_range_4 = (f4 >= target_range[0]) & (f4 <= target_range[1])
        in_range_5 = (f5 >= target_range[0]) & (f5 <= target_range[1])

        # At least 2 formants in range = singer's formant
        clustering = (in_range_3.astype(int) + in_range_4.astype(int) + in_range_5.astype(int)) >= 2

        strength = np.mean(clustering)
        has_singers_formant = strength > 0.3

        return has_singers_formant, strength

    def enhance(self, audio: np.ndarray, sr: int, formant_freqs: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """
        Enhance singer's formant.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz
        formant_freqs : np.ndarray, optional
            Formant frequencies for detection

        Returns
        -------
        audio_enhanced : np.ndarray
            Enhanced audio
        metrics : Dict
            Enhancement metrics
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples)
                left = self.enhance(audio[0], sr, formant_freqs)[0]
                right = self.enhance(audio[1], sr, formant_freqs)[0]
                return np.vstack([left, right]), {"stereo": True}
            else:
                # Format: (samples, channels) - transpose for processing
                audio_T = audio.T
                left = self.enhance(audio_T[0], sr, formant_freqs)[0]
                right = self.enhance(audio_T[1], sr, formant_freqs)[0]
                # Return in original format
                return np.column_stack([left, right]), {"stereo": True}

        # Detect existing singer's formant
        has_formant = False
        strength = 0.0

        if formant_freqs is not None:
            has_formant, strength = self.detect_singers_formant(formant_freqs)

        # Compute dynamic cluster center from actual F4/F5 median in 2800-3200 Hz
        # (Sundberg 1987/2015: Singer's formant = tight F3-F5 cluster at 2.8-3.2 kHz
        # in classically trained voices.  Pop/speech vocals lack this cluster.)
        cluster_center = self.target_freq_hz
        if has_formant and formant_freqs is not None and formant_freqs.shape[1] >= 4:
            f4 = formant_freqs[:, 3] if formant_freqs.shape[1] > 3 else np.zeros(len(formant_freqs))
            f5 = formant_freqs[:, 4] if formant_freqs.shape[1] > 4 else np.zeros(len(formant_freqs))
            cluster_vals = np.concatenate(
                [
                    f4[(f4 >= 2500.0) & (f4 <= 3500.0)],
                    f5[(f5 >= 2500.0) & (f5 <= 3500.0)],
                ]
            )
            if len(cluster_vals) > 0:
                cluster_center = float(np.median(cluster_vals))

        if not has_formant:
            # No Singer's formant detected — passthrough.  Imposing a 3 kHz boost
            # on pop or folk vocals creates nasal/honky coloration (Sundberg 1987 §4.2).
            audio_enhanced = audio.copy()
            adaptive_gain = 0.0
        else:
            # Scale gain by clustering strength; boost at the actual cluster center
            adaptive_gain = self.gain_db * (1.0 - strength * 0.5)
            audio_enhanced = self._apply_peak_eq(audio, sr, cluster_center, self.bandwidth_hz, adaptive_gain)

        # NaN/Inf-Guard and clipping
        audio_enhanced = np.nan_to_num(audio_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        audio_enhanced = np.clip(audio_enhanced, -1.0, 1.0)

        metrics = {
            "has_singers_formant": has_formant,
            "strength_before": strength,
            "gain_applied_db": adaptive_gain,
            "target_freq_hz": self.target_freq_hz,
            "cluster_center_hz": cluster_center,
        }

        return audio_enhanced, metrics

    def _apply_peak_eq(self, audio: np.ndarray, sr: int, freq: float, bandwidth: float, gain_db: float) -> np.ndarray:
        """
        Wendet an: peak EQ filter.
        """
        Q = freq / bandwidth
        A = 10 ** (gain_db / 40)

        w0 = 2 * np.pi * freq / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])

        audio = np.asarray(signal.lfilter(b, a, audio), dtype=np.float64)

        return audio


class FormantSystem:
    """
    Einheitliche API für Formant-Tracking, -Korrektur und -Verbesserung.
    """

    def __init__(self, n_formants: int = 5, correction_strength: float = 0.7, enhance_singers_formant: bool = True):
        """
        Parameters
        ----------
        n_formants : int
            Number of formants to track
        correction_strength : float
            Formant correction strength (0.0-1.0)
        enhance_singers_formant : bool
            Whether to enhance singer's formant
        """
        self.tracker = FormantTracker(n_formants=n_formants)
        self.corrector = FormantCorrector(correction_strength=correction_strength)
        self.singers_formant_enhancer = SingersFormantEnhancer()
        self.enhance_singers_formant = enhance_singers_formant

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full formant processing pipeline.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_processed : np.ndarray
            Processed audio
        report : Dict
            Processing report
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Handle stereo
        if audio.ndim == 2:
            # Auto-detect format: (channels, samples) vs (samples, channels)
            # Heuristic: If first dimension is small and < second dimension, likely channels
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples) - average over channels (axis 0)
                audio_mono = np.mean(audio, axis=0)
            else:
                # Format: (samples, channels) - average over channels (axis 1)
                audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        # Track formants
        formant_freqs, _formant_bws = self.tracker.track(audio_mono, sr)

        # Detect drift
        has_drift, drift_info = self.corrector.detect_drift(formant_freqs)

        # Correct drift if needed
        audio_corrected = self.corrector.correct(audio, sr, formant_freqs) if has_drift else audio

        # Enhance singer's formant
        if self.enhance_singers_formant:
            audio_enhanced, singers_metrics = self.singers_formant_enhancer.enhance(audio_corrected, sr, formant_freqs)
        else:
            audio_enhanced = audio_corrected
            singers_metrics = {}

        report = {
            "formant_tracking": {
                "n_formants": formant_freqs.shape[1],
                "n_frames": len(formant_freqs),
                "mean_formants_hz": np.mean(formant_freqs[formant_freqs > 0], axis=0).tolist(),
            },
            "drift_detection": {"has_drift": has_drift, "drift_info": drift_info},
            "singers_formant": singers_metrics,
        }

        # Final guards
        audio_enhanced = np.nan_to_num(audio_enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        audio_enhanced = np.clip(audio_enhanced, -1.0, 1.0)

        return audio_enhanced, report

    def phoneme_guided_enhance(
        self,
        audio: np.ndarray,
        sr: int,
        phoneme_segments: list | None = None,
        gender: str = "male",
        correction_strength: float = 0.25,
    ) -> tuple[np.ndarray, dict]:
        """Phoneme-guided formant enhancement using per-vowel canonical targets.

        For each detected vowel-phoneme segment the method applies a gentle EQ
        boost toward the canonical (F1, F2, F3) frequencies from Peterson &
        Barney (1952) and Hillenbrand et al. (1995).  This complements
        ``process()`` (which corrects temporal drift) by addressing *phonemic
        accuracy*: a damaged /i/ whose F1 shifted upward is steered back toward
        the canonical /i/ target.

        Args:
            audio:             Mono float32 signal (48 kHz).  Stereo is handled
                               channel-wise.
            sr:                Sample rate — must be 48000 Hz.
            phoneme_segments:  Optional list of objects with ``.phoneme``
                               (IPA str) and ``.start_time`` / ``.end_time``
                               (float, seconds).  If *None*, the detected LPC
                               formants are used to classify the vowel class
                               automatically (DSP fallback).
            gender:            ``'male'`` | ``'female'`` | ``'child'`` |
                               ``'unknown'``
            correction_strength: Blend ratio toward canonical targets — keep
                               ≤ 0.35 to preserve voice identity [0.0–0.35].

        Returns:
            ``(enhanced_audio, report_dict)``

        References:
            Peterson & Barney (1952): "Control Methods Used in a Study of Vowels"
            Hillenbrand et al. (1995): "Acoustic Characteristics of American
            English Vowels"
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # ── Stereo: recurse per channel ──────────────────────────────── #
        if audio.ndim == 2:
            if audio.shape[0] <= 32:  # (channels, samples) format
                channels = [
                    self.phoneme_guided_enhance(audio[i], sr, phoneme_segments, gender, correction_strength)[0]
                    for i in range(audio.shape[0])
                ]
                result = np.stack(channels)
            else:  # (samples, channels) format
                channels = [
                    self.phoneme_guided_enhance(audio[:, i], sr, phoneme_segments, gender, correction_strength)[0]
                    for i in range(audio.shape[1])
                ]
                result = np.column_stack(channels)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            result = np.clip(result, -1.0, 1.0)
            return result, {"stereo": True}

        # ── LPC formant tracking ──────────────────────────────────────── #
        try:
            formant_freqs, _ = self.tracker.track(audio, sr)  # (n_frames, n_formants)
        except Exception as _e:
            logger.debug("phoneme_guided_verbessern: FormantTracker fehlgeschlagen: %s", _e)
            return audio.copy(), {"vowel_segments_processed": 0, "error": str(_e)}

        n_frames = len(formant_freqs)
        if n_frames == 0:
            return audio.copy(), {"vowel_segments_processed": 0}

        hop = max(1, int(self.tracker.hop_length_ms * sr / 1000))
        frame_len = max(1, int(self.tracker.frame_length_ms * sr / 1000))
        strength = float(np.clip(correction_strength, 0.0, 0.35))
        n_samples = len(audio)
        enhanced = audio.copy()
        vowel_count = 0

        # ── Build frame→IPA map from optional segments ────────────────── #
        frame_ipa: list[str | None] = [None] * n_frames
        if phoneme_segments is not None:
            for seg in phoneme_segments:
                ipa = getattr(seg, "phoneme", None)
                if ipa is None:
                    continue
                if VowelPhonemeFormantTargets.get_targets(ipa, gender) is None:
                    continue  # not a vowel symbol
                start_f = max(0, int(getattr(seg, "start_time", 0.0) * sr / hop))
                end_f = min(n_frames, int(getattr(seg, "end_time", 0.0) * sr / hop) + 1)
                for fi in range(start_f, end_f):
                    frame_ipa[fi] = ipa

        # ── Per-frame phonem-guided EQ ────────────────────────────────── #
        for fi in range(n_frames):
            f1_det = float(formant_freqs[fi, 0]) if formant_freqs.shape[1] > 0 else 0.0
            f2_det = float(formant_freqs[fi, 1]) if formant_freqs.shape[1] > 1 else 0.0

            ipa = frame_ipa[fi]
            if ipa is None:
                # DSP fallback: classify vowel from F1/F2 position
                ipa = VowelPhonemeFormantTargets.classify_from_formants(f1_det, f2_det, gender)
            if ipa is None:
                continue

            targets = VowelPhonemeFormantTargets.get_targets(ipa, gender)
            if targets is None:
                continue
            t_f1, t_f2, _ = targets

            start_s = fi * hop
            end_s = min(n_samples, start_s + frame_len)
            if end_s <= start_s:
                continue

            frame_audio = enhanced[start_s:end_s].copy()
            frame_mod = frame_audio.copy()

            for f_det, f_tgt in ((f1_det, t_f1), (f2_det, t_f2)):
                if f_det <= 0 or f_tgt <= 0:
                    continue
                deviation = abs(f_tgt - f_det)
                if deviation < 80:
                    continue  # within tolerance
                bw = max(150.0, f_tgt * 0.15)
                boost_db = min(2.0, deviation / 200.0 * 2.0)
                frame_mod = _apply_peak_eq_frame(frame_mod, sr, f_tgt, bw, boost_db)

            enhanced[start_s:end_s] = (1.0 - strength) * frame_audio + strength * frame_mod
            vowel_count += 1

        enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
        enhanced = np.clip(enhanced, -1.0, 1.0)

        logger.debug(
            "phoneme_guided_verbessern: gender=%s, vowel_frames=%d/%d, strength=%.2f",
            gender,
            vowel_count,
            n_frames,
            strength,
        )
        return enhanced, {
            "vowel_segments_processed": vowel_count,
            "total_frames": n_frames,
            "gender": gender,
            "correction_strength": strength,
        }


# ── Module-level helper (used by phoneme_guided_enhance) ────────────────────


def _apply_peak_eq_frame(audio: np.ndarray, sr: int, freq: float, bandwidth: float, gain_db: float) -> np.ndarray:
    """Wendet einen Biquad-Peak-EQ auf einen kurzen Audio-Frame an (DSP-Hilfsfunktion).

    Args:
        audio:     1-D float32 audio frame.
        sr:        Sample rate in Hz.
        freq:      Center frequency in Hz.
        bandwidth: Bandwidth in Hz (Q = freq/bandwidth).
        gain_db:   Boost in dB (positive = boost).

    Returns:
        EQ-filtered frame (same shape as input).
    """
    if gain_db < 0.01 or freq <= 0 or bandwidth <= 0 or freq >= sr / 2.0:
        return audio
    Q = max(0.1, freq / bandwidth)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b_coef = np.array([b0, b1, b2]) / a0
    a_coef = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b_coef, a_coef, audio)  # type: ignore[no-any-return]


# ── VowelPhonemeFormantTargets ───────────────────────────────────────────────


class VowelPhonemeFormantTargets:
    """Canonical vowel formant targets per gender class.

    Maps IPA vowel symbols to reference (F1, F2, F3) frequencies in Hz.
    Used by :meth:`FormantSystem.phoneme_guided_enhance` to steer the
    formant EQ toward phonemically correct targets.

    Sources:
        Peterson & Barney (1952): "Control Methods Used in a Study of Vowels"
        Hillenbrand et al. (1995): "Acoustic Characteristics of American
        English Vowels"

    Layout of ``_IPA_TARGETS``:
        ipa_sym → (F1_male, F2_male, F3_male,
                   F1_female, F2_female, F3_female)  — all in Hz.
    """

    # fmt: off
    _IPA_TARGETS: dict[str, tuple[float, float, float, float, float, float]] = {
        # Close front /i/ — "heed"  (Peterson & Barney 1952, Table II)
        "i":   (270, 2290, 3010,  310, 2790, 3310),
        "iː":  (270, 2290, 3010,  310, 2790, 3310),
        # Near-close near-front /ɪ/ — "hid"
        "ɪ":   (390, 1990, 2550,  430, 2480, 3070),
        # Close-mid front /e/ — German "See"
        "e":   (400, 2000, 2550,  470, 2500, 3050),
        "eː":  (400, 2000, 2550,  470, 2500, 3050),
        # Open-mid front /ɛ/ — "head"
        "ɛ":   (530, 1840, 2480,  610, 2330, 2990),
        "ɛː":  (530, 1840, 2480,  610, 2330, 2990),
        # Near-open front /æ/ — "had"
        "æ":   (660, 1720, 2410,  860, 2050, 2850),
        # Open back /ɑ/ — "father"  (Hillenbrand 1995)
        "ɑ":   (730, 1090, 2440,  850, 1220, 2810),
        "ɑː":  (730, 1090, 2440,  850, 1220, 2810),
        # Open central /a/ — German "Mann"
        "a":   (700, 1220, 2600,  800, 1450, 2800),
        "aː":  (700, 1220, 2600,  800, 1450, 2800),
        # Open-mid back (rounded) /ɒ/ — "hod"
        "ɒ":   (570,  840, 2410,  590,  920, 2710),
        # Close-mid back rounded /o/ — German "Boot"
        "o":   (450,  720, 2600,  470,  820, 2700),
        "oː":  (450,  720, 2600,  470,  820, 2700),
        # Open-mid back /ʌ/ — "hud"
        "ʌ":   (640, 1190, 2390,  760, 1400, 2780),
        # Mid central schwa /ə/
        "ə":   (490, 1350, 1690,  500, 1640, 2480),
        # Open-mid central /ɜ/ — "heard"
        "ɜ":   (490, 1350, 1690,  500, 1640, 2480),
        "ɜː":  (490, 1350, 1690,  500, 1640, 2480),
        # Near-close near-back rounded /ʊ/ — "hood"
        "ʊ":   (440, 1020, 2240,  470, 1160, 2680),
        # Close back rounded /u/ — "who'd"
        "u":   (300,  870, 2240,  370,  950, 2670),
        "uː":  (300,  870, 2240,  370,  950, 2670),
        # Close-mid front rounded /ø/ — German "Höhle"
        "ø":   (370, 1480, 1990,  400, 1650, 2400),
        "øː":  (370, 1480, 1990,  400, 1650, 2400),
        # Open-mid front rounded /œ/ — German "Hölle"
        "œ":   (450, 1180, 1900,  530, 1320, 2300),
        # Close front rounded /y/ — German "über"
        "y":   (235, 1870, 2150,  260, 2080, 2760),
        "yː":  (235, 1870, 2150,  260, 2080, 2760),
    }
    # fmt: on

    @classmethod
    def get_targets(cls, ipa_symbol: str, gender: str = "male") -> tuple[float, float, float] | None:
        """Gibt canonical (F1, F2, F3) targets for an IPA vowel symbol zurück.

        Args:
            ipa_symbol: IPA character (e.g. ``'i'``, ``'a'``, ``'ɛ'``).
            gender:     ``'male'`` | ``'female'`` | ``'child'`` | ``'unknown'``.

        Returns:
            ``(F1_hz, F2_hz, F3_hz)`` or *None* if symbol is not a vowel.
        """
        row = cls._IPA_TARGETS.get(ipa_symbol)
        if row is None:
            return None
        if gender == "female":
            return (row[3], row[4], row[5])
        if gender == "child":
            # Child formant ranges are approx. 15/10/5 % above female
            return (row[3] * 1.15, row[4] * 1.10, row[5] * 1.05)
        return (row[0], row[1], row[2])

    @classmethod
    def classify_from_formants(cls, f1_hz: float, f2_hz: float, gender: str = "male") -> str | None:
        """Identify nearest vowel class from detected LPC F1/F2 values.

        Uses Euclidean distance in a normalized F1/F2 space
        (F1 normalised over 200–900 Hz, F2 over 600–3000 Hz) so that
        both axes contribute equally regardless of absolute frequency range.

        Args:
            f1_hz: Detected first formant in Hz.
            f2_hz: Detected second formant in Hz.
            gender: Speaker gender for target selection.

        Returns:
            IPA symbol of nearest canonical vowel, or *None* if formants
            are invalid (≤ 0).
        """
        if f1_hz <= 0.0 or f2_hz <= 0.0:
            return None

        best_key: str | None = None
        best_dist = float("inf")

        n_f1 = (f1_hz - 200.0) / 700.0
        n_f2 = (f2_hz - 600.0) / 2400.0

        for ipa_sym, row in cls._IPA_TARGETS.items():
            t_f1, t_f2 = (row[3], row[4]) if gender == "female" else (row[0], row[1])
            n_t_f1 = (t_f1 - 200.0) / 700.0
            n_t_f2 = (t_f2 - 600.0) / 2400.0
            dist = (n_f1 - n_t_f1) ** 2 + (n_f2 - n_t_f2) ** 2
            if dist < best_dist:
                best_dist = dist
                best_key = ipa_sym

        return best_key


# ── InstrumentFormantTargets ─────────────────────────────────────────────────


class InstrumentFormantTargets:
    """Canonical resonance-peak (formant) targets for acoustic instruments.

    Maps instrument type strings to reference (F1, F2, F3) characteristic
    resonance frequencies in Hz with associated Q-values.
    Used by :meth:`FormantSystem.instrument_guided_enhance` to steer
    the EQ toward the instrument's timbral identity.

    Sources:
        STRINGS (violin/cello):
            McIntyre & Woodhouse (1978): "Acoustics of bowed instruments"
            Jansson (2002): "Acoustics for Violin and Guitar Makers"
        GUITAR (acoustic):
            Christensen (1982): "Structural-acoustical analysis of guitar"
            Reboursière et al. (2012): "Guitar body resonances"
        BRASS (trumpet/trombone):
            Benade (1976): "Fundamentals of Musical Acoustics"
            Campbell (1999): "Brass instrument acoustics"
        KEYS / PIANO:
            Young (1952): "Inharmonicity of plain wire piano strings"
            Weinreich (1977): "Coupled piano strings"
        BASS (electric/double bass):
            Rossing et al. (2002): "Science of String Instruments"
        DRUMS / PERCUSSION:
            Rossing (1992): "Science of Percussion Instruments"
            Fletcher & Rossing (1998): "Physics of Musical Instruments"

    Layout of ``_INSTRUMENT_TARGETS``:
        instrument_str → (F1_hz, F2_hz, F3_hz, Q1, Q2, Q3)

    Instrument strings correspond to ``InstrumentType`` enum values from
    ``backend.semantic.semantic_audio_analyzer`` (used as plain strings here
    to avoid cross-package import).
    """

    # fmt: off
    _INSTRUMENT_TARGETS: dict[str, tuple[float, float, float, float, float, float]] = {
        # ── STRINGS (violin / viola / cello / contrabass) ────────────────────
        # McIntyre & Woodhouse 1978: A0 Helmholtz ~275 Hz, B1- wood ~450 Hz,
        # bridge hill ~2500 Hz (Cremer 1984)
        "strings": (275.0, 450.0, 2500.0,  8.0, 6.0, 4.0),

        # ── GUITAR (acoustic steel-string / classical) ────────────────────────
        # Helmholtz resonance ~110 Hz, body resonance ~200 Hz,
        # top-plate resonance ~420 Hz  (Christensen 1982)
        "guitar":  (110.0, 200.0, 420.0,   5.0, 4.0, 3.5),

        # ── BRASS (trumpet / trombone / French horn) ─────────────────────────
        # Benade 1976: characteristic formant peak 600–800 Hz,
        # second resonance 1800–2000 Hz, brilliance ~3000 Hz
        "brass":   (700.0, 1900.0, 3000.0, 6.0, 4.0, 3.0),

        # ── KEYS / PIANO ──────────────────────────────────────────────────────
        # Grand piano: body resonance ~80 Hz, mid-register presence ~500 Hz,
        # upper partial brightness ~2500 Hz  (Young 1952)
        "keys":    (90.0,  500.0, 2500.0,  4.0, 3.0, 2.5),

        # ── BASS (electric bass / double bass) ───────────────────────────────
        # Fundamental region ~80 Hz, first strong harmonic ~160 Hz,
        # attack definition ~450 Hz  (Rossing et al. 2002)
        "bass":    (80.0,  160.0, 450.0,   5.0, 4.0, 3.0),

        # ── DRUMS / KICK & SNARE ─────────────────────────────────────────────
        # Kick: sub-punch ~65 Hz, click ~400 Hz; Snare: body ~200 Hz
        # (Fletcher & Rossing 1998)
        "drums":   (65.0,  200.0, 400.0,   3.5, 3.0, 3.0),

        # ── PERCUSSION (mallet: xylophone / marimba / vibraphone) ────────────
        # Bar resonance ~250 Hz, tube resonance ~500 Hz, shimmer ~3000 Hz
        "percussion": (250.0, 500.0, 3000.0, 5.0, 4.0, 3.5),

        # ── SYNTH / ELECTRONIC ───────────────────────────────────────────────
        # No acoustic body; target spectral centroid regions of typical leads:
        # Sub-bass ~80 Hz, mid presence ~1000 Hz, high harmonic ~4000 Hz
        "synth":   (80.0, 1000.0, 4000.0,  3.0, 2.5, 2.0),

        # ── WOODWINDS (flute / clarinet / oboe / saxophone) ──────────────────
        # Clarinet: first resonance ~300 Hz, second ~1000 Hz, brightness ~2200 Hz
        # (Benade 1976 — applied generically to woodwinds)
        "woodwinds": (300.0, 1000.0, 2200.0, 6.0, 4.0, 3.5),
    }
    # fmt: on

    @classmethod
    def get_targets(cls, instrument: str) -> tuple[float, float, float, float, float, float] | None:
        """Gibt canonical (F1, F2, F3, Q1, Q2, Q3) for an instrument string zurück.

        Args:
            instrument: Instrument type string, e.g. ``'guitar'``, ``'strings'``,
                ``'brass'``.  Case-insensitive.

        Returns:
            ``(F1_hz, F2_hz, F3_hz, Q1, Q2, Q3)`` or *None* if unknown.
        """
        return cls._INSTRUMENT_TARGETS.get(instrument.lower())

    @classmethod
    def all_instruments(cls) -> list[str]:
        """Gibt sorted list of supported instrument type strings zurück."""
        return sorted(cls._INSTRUMENT_TARGETS.keys())


# ── FormantSystem.instrument_guided_enhance (monkey-patched in) ───────────────


def _instrument_guided_enhance(
    self: "FormantSystem",
    audio: np.ndarray,
    sr: int,
    instrument: str = "guitar",
    correction_strength: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """Instrument-specific formant enhancement toward physical resonance targets.

    Applies a gentle, frame-wise peak-EQ boost toward the canonical (F1, F2, F3)
    resonance peaks of the given instrument type, using targets from
    :class:`InstrumentFormantTargets` (McIntyre & Woodhouse 1978, Benade 1976,
    Christensen 1982 et al.).

    This complements :meth:`process` (which corrects temporal drift) by
    steering the timbral identity of a degraded recording back toward the
    characteristic body resonances of the instrument.

    Algorithm per frame:
        1. Compute deviation between tracked LPC-F1 and instrument target F1.
        2. If |deviation| > 60 Hz: apply a proportional peak-EQ boost
           (max 2 dB, scaled by deviation / 300 Hz) at the target frequency.
        3. Repeat for F2 and F3.
        4. Blend corrected frame at *correction_strength* ≤ 0.30 (identity-safe).

    Args:
        audio:               Mono or stereo input at 48 000 Hz.
        sr:                  Sample rate — must be 48 000 Hz.
        instrument:          Instrument type string (see :class:`InstrumentFormantTargets`).
        correction_strength: Blend factor 0.0–1.0; clamped to 0.30.

    Returns:
        Tuple of ``(enhanced_audio, report_dict)`` where *report_dict* keys:
        ``instrument``, ``frames_processed``, ``total_frames``,
        ``correction_strength``, ``f_targets_hz``.
    """
    assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    correction_strength = float(np.clip(correction_strength, 0.0, 0.30))

    # Look up targets — graceful no-op when instrument is unsupported
    row = InstrumentFormantTargets.get_targets(instrument)
    if row is None:
        logger.debug("instrument_guided_verbessern: unknown instrument '%s' — passthrough", instrument)
        audio = np.clip(audio, -1.0, 1.0)
        return audio, {
            "instrument": instrument,
            "frames_processed": 0,
            "total_frames": 0,
            "correction_strength": correction_strength,
            "f_targets_hz": None,
        }

    f1_tgt, f2_tgt, f3_tgt, q1, q2, q3 = row

    # Work on mono for formant tracking, preserve original shape for mixing
    is_stereo = audio.ndim == 2
    if is_stereo:
        audio_mono = np.mean(audio, axis=0) if audio.shape[0] < audio.shape[1] else np.mean(audio, axis=1)
    else:
        audio_mono = audio.copy()

    # Frame parameters (25 ms / 10 ms hop)
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)

    # Track formants (LPC-based)
    try:
        formant_freqs, _ = self.tracker.track(audio_mono, sr)
    except Exception as _e:
        logger.debug("instrument_guided_verbessern: FormantTracker fehlgeschlagen: %s", _e)
        audio = np.clip(audio, -1.0, 1.0)
        return audio, {
            "instrument": instrument,
            "frames_processed": 0,
            "total_frames": 0,
            "correction_strength": correction_strength,
            "f_targets_hz": (f1_tgt, f2_tgt, f3_tgt),
        }

    total_frames = len(formant_freqs)
    n_samples = len(audio_mono)
    enhanced = audio_mono.copy()
    frames_done = 0

    # Per-frame EQ toward instrument resonance targets
    for fi in range(total_frames):
        t0 = fi * hop_len
        t1 = min(t0 + frame_len, n_samples)
        if t1 <= t0:
            break

        frame = enhanced[t0:t1].copy()
        frame_mod = frame.copy()

        tracked_f1 = float(formant_freqs[fi, 0]) if formant_freqs.shape[1] > 0 else 0.0

        # F1 boost when deviation > 60 Hz
        if tracked_f1 > 0.0 and abs(tracked_f1 - f1_tgt) > 60.0:
            boost_db = min(2.0, abs(tracked_f1 - f1_tgt) / 300.0 * 2.0) * correction_strength / 0.30
            frame_mod = _apply_peak_eq_frame(frame_mod, sr, f1_tgt, sr / (2.0 * q1 * f1_tgt + 1e-6), boost_db)

        # F2 boost (always applied at low strength — body colour)
        bw2 = f2_tgt / q2
        boost_f2 = 1.5 * correction_strength / 0.30
        frame_mod = _apply_peak_eq_frame(frame_mod, sr, f2_tgt, bw2, boost_f2)

        # F3 boost (brilliance / air region)
        bw3 = f3_tgt / q3
        boost_f3 = 1.0 * correction_strength / 0.30
        frame_mod = _apply_peak_eq_frame(frame_mod, sr, f3_tgt, bw3, boost_f3)

        # Blend
        enhanced[t0:t1] = (1.0 - correction_strength) * frame + correction_strength * frame_mod[: t1 - t0]
        frames_done += 1

    # Apply to stereo by re-mixing
    if is_stereo:
        if audio.shape[0] < audio.shape[1]:
            orig_mono = np.mean(audio, axis=0)
            ratio = np.where(np.abs(orig_mono) > 1e-10, enhanced / (orig_mono + 1e-12), 1.0)
            ratio = np.clip(ratio, 0.5, 2.0)
            out = audio * ratio[np.newaxis, :]
        else:
            orig_mono = np.mean(audio, axis=1)
            ratio = np.where(np.abs(orig_mono) > 1e-10, enhanced / (orig_mono + 1e-12), 1.0)
            ratio = np.clip(ratio, 0.5, 2.0)
            out = audio * ratio[:, np.newaxis]
    else:
        out = enhanced

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -1.0, 1.0)

    logger.debug(
        "instrument_guided_verbessern: instrument=%s, frames=%d/%d, strength=%.2f",
        instrument,
        frames_done,
        total_frames,
        correction_strength,
    )

    return out, {
        "instrument": instrument,
        "frames_processed": frames_done,
        "total_frames": total_frames,
        "correction_strength": correction_strength,
        "f_targets_hz": (f1_tgt, f2_tgt, f3_tgt),
    }


# Attach as method on FormantSystem (avoids re-opening the class block)
FormantSystem.instrument_guided_enhance = _instrument_guided_enhance  # type: ignore[attr-defined]


# CLI interface
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Formant System - Professional formant processing")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", help="Output audio file")
    parser.add_argument("--correction-strength", type=float, default=0.7, help="Correction strength (0.0-1.0)")
    parser.add_argument("--no-singers-formant", action="store_true", help="Disable singer's formant enhancement")

    args = parser.parse_args()

    # Load audio
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Process
    formant_sys = FormantSystem(
        correction_strength=args.correction_strength, enhance_singers_formant=not args.no_singers_formant
    )

    audio_processed, report = formant_sys.process(audio, sr)

    # Print report
    logger.info("")
    logger.info("=" * 70)
    logger.info("FORMANT SYSTEM REPORT")
    logger.info("=" * 70)
    logger.info("")
    logger.info("[Formant Tracking]")
    logger.info("  Formants tracked: %d", report["formant_tracking"]["n_formants"])
    logger.info("  Frames analyzed:  %d", report["formant_tracking"]["n_frames"])

    logger.info("")
    logger.info("[Drift Detection]")
    logger.info("  Has drift: %s", report["drift_detection"]["has_drift"])
    if report["drift_detection"]["has_drift"]:
        for formant, info in report["drift_detection"]["drift_info"].items():
            if info["needs_correction"]:
                logger.info("  %s: %.1f Hz drift (corrected)", formant, info["max_drift_hz"])

    if "has_singers_formant" in report["singers_formant"]:
        logger.info("")
        logger.info("[Singer's Formant]")
        logger.info("  erkannt: %s", report["singers_formant"]["has_singers_formant"])
        logger.info("  Enhancement: %.1f dB", report["singers_formant"]["gain_applied_db"])

    logger.info("=" * 70)

    # Save
    if args.output:
        sf.write(args.output, audio_processed, sr)
        logger.info("")
        logger.info("✅ gespeichert to: %s", args.output)
