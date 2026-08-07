"""
PROFESSIONAL AUDIO METERING
============================

ITU-R BS.1770-4 compliant LUFS metering, True Peak detection,
Phase Correlation, and Spectrum Analysis for professional audio workflows.
"""

import logging
from dataclasses import dataclass

import numpy as np
import scipy.signal

_logger = logging.getLogger(__name__)


@dataclass
class MeteringResult:
    """Complete metering analysis result"""

    # Loudness (LUFS)
    integrated_lufs: float  # Overall loudness
    loudness_range: float  # LRA (dynamic range)
    momentary_lufs: np.ndarray  # 400ms windows
    short_term_lufs: np.ndarray  # 3s windows

    # Peak levels
    true_peak_db: float  # True peak (oversampled)
    sample_peak_db: float  # Sample peak
    true_peak_exceeded: bool  # > -1 dBTP alert

    # Phase correlation (stereo only)
    phase_correlation: float | None  # -1 to +1
    phase_coherence: float | None  # 0 to 1

    # Spectral analysis
    spectrum_freqs: np.ndarray  # Frequency bins
    spectrum_db: np.ndarray  # Power spectrum in dB
    spectral_centroid: float  # Brightness (Hz)
    spectral_rolloff: float  # High-freq cutoff (Hz)


class LUFSMeter:
    """
    ITU-R BS.1770-4 compliant loudness metering

    References:
    - ITU-R BS.1770-4: Algorithms to measure audio programme loudness
    - EBU R128: Loudness normalisation and permitted maximum level
    """

    def __init__(self, sr: int = 48000):
        self.sr = sr

        # K-weighting filter coefficients (ITU-R BS.1770-4)
        self.prefilter = self._design_prefilter()
        self.rlb_filter = self._design_rlb_filter()

    def _design_prefilter(self) -> tuple[np.ndarray, np.ndarray]:
        """Design high-shelf pre-filter (stage 1 K-weighting)"""
        # ITU-R BS.1770-4 pre-filter at 48 kHz
        # High-shelf filter: +4 dB at high frequencies

        if self.sr == 48000:
            # Pre-computed coefficients for 48 kHz
            b = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
            a = np.array([1.0, -1.69065929318241, 0.73248077421585])
        else:
            # Design for arbitrary sample rate
            f0 = 1681.974  # Hz
            Q = 0.7071067811865476
            K = np.tan(np.pi * f0 / self.sr)
            Vh = 10 ** (4.0 / 20)  # +4 dB gain

            Vb = Vh**0.5
            a0 = 1 + K / Q + K**2
            b = np.array([(Vh + Vb * K / Q + K**2) / a0, 2 * (K**2 - Vh) / a0, (Vh - Vb * K / Q + K**2) / a0])
            a = np.array([1.0, 2 * (K**2 - 1) / a0, (1 - K / Q + K**2) / a0])

        return b, a

    def _design_rlb_filter(self) -> tuple[np.ndarray, np.ndarray]:
        """Design RLB (Revised Low-frequency B) high-pass filter (stage 2)"""
        # ITU-R BS.1770-4 RLB filter at 48 kHz

        if self.sr == 48000:
            # Pre-computed coefficients for 48 kHz
            b = np.array([1.0, -2.0, 1.0])
            a = np.array([1.0, -1.99004745483398, 0.99007225036621])
        else:
            # Design for arbitrary sample rate
            f0 = 38.13547  # Hz
            Q = 0.5003270373238773
            K = np.tan(np.pi * f0 / self.sr)

            a0 = 1 + K / Q + K**2
            b = np.array([1.0, -2.0, 1.0])
            a = np.array([1.0, 2 * (K**2 - 1) / a0, (1 - K / Q + K**2) / a0])

        return b, a

    def measure(self, audio: np.ndarray, sr: int, gating: bool = True) -> dict[str, float]:
        """
        Misst integrated loudness (LUFS).

        Args:
            audio: Mono or stereo audio (channels, samples) or (samples,)
            sr: Sample rate
            gating: Apply gating per ITU-R BS.1770-4

        Returns:
            Dictionary with LUFS measurements
        """
        # Ensure 2D array (channels, samples)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Guard: NaN/Inf audio → LAPACK DLASCL failure in cascaded filters
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # Apply K-weighting filters (sosfilt supports multi-channel along axis)
        stage1 = scipy.signal.lfilter(self.prefilter[0], self.prefilter[1], audio, axis=1)
        filtered = scipy.signal.lfilter(self.rlb_filter[0], self.rlb_filter[1], stage1, axis=1)

        # Mean square per channel
        ms_per_channel = np.mean(filtered**2, axis=1)

        # Channel weights (ITU-R BS.1770-4)
        if audio.shape[0] == 1:  # Mono
            channel_weights = np.array([1.0])
        elif audio.shape[0] == 2:  # Stereo
            channel_weights = np.array([1.0, 1.0])  # L, R
        else:  # Multi-channel (5.1, etc.)
            # L, R, C, LFE, Ls, Rs
            channel_weights = np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
            channel_weights = channel_weights[: audio.shape[0]]

        # Weighted sum
        mean_square = np.sum(ms_per_channel * channel_weights) / np.sum(channel_weights)

        # Convert to LUFS
        lufs = -0.691 + 10 * np.log10(mean_square) if mean_square > 0 else -np.inf

        # Compute gated loudness if requested
        if gating:
            lufs = self._apply_gating(filtered, channel_weights)

        return {"integrated_lufs": float(lufs), "sample_rate": sr, "channels": audio.shape[0]}

    def _apply_gating(self, filtered: np.ndarray, channel_weights: np.ndarray) -> float:
        """Wendet absolutes und relatives Gating gemäß ITU-R BS.1770-4 an."""
        # Block size: 400ms (overlap 75%)
        block_samples = int(0.4 * self.sr)
        hop_samples = int(0.1 * self.sr)  # 100ms hop

        n_blocks = (filtered.shape[1] - block_samples) // hop_samples + 1

        if n_blocks <= 0:
            return -np.inf

        # Vectorized block extraction via index array
        starts = np.arange(n_blocks) * hop_samples
        indices = starts[:, np.newaxis] + np.arange(block_samples)  # (n_blocks, block_samples)

        # Extract all blocks at once: (n_channels, n_blocks, block_samples)
        blocks = filtered[:, indices]

        # Mean square per channel per block: (n_channels, n_blocks)
        ms_per_channel = np.mean(blocks**2, axis=2)

        # Weighted sum per block: (n_blocks,)
        weight_sum = np.sum(channel_weights) + 1e-30
        mean_square = np.dot(channel_weights, ms_per_channel) / weight_sum

        # Convert to LUFS, filter out silent blocks
        valid = mean_square > 0
        if not np.any(valid):
            return -np.inf

        block_loudness = -0.691 + 10 * np.log10(mean_square[valid])

        # Absolute gate: -70 LUFS
        gated_blocks = block_loudness[block_loudness >= -70.0]

        if len(gated_blocks) == 0:
            return -np.inf

        # Relative gate: -10 LU relative to absolute gated loudness
        absolute_gated = 10 * np.log10(np.mean(10 ** (gated_blocks / 10)))
        relative_gate = absolute_gated - 10.0

        final_blocks = gated_blocks[gated_blocks >= relative_gate]

        if len(final_blocks) == 0:
            return -np.inf

        # Final integrated loudness
        integrated = 10 * np.log10(np.mean(10 ** (final_blocks / 10)))

        return float(integrated)

    def measure_blocks(
        self,
        audio: np.ndarray,
        sr: int,
        block_sec: float = 0.4,
        hop_sec: float = 0.1,
    ) -> np.ndarray:
        """Berechnet LUFS-Werte für überlappende Blöcke (EBU R128).

        Wird für Momentary (400 ms, 100 ms Hop) und Short-term (3 s, 1 s Hop)
        verwendet. Werte unterhalb von -70 LUFS werden als -np.inf zurückgegeben
        (absolutes Gate nach EBU R128).

        Args:
            audio:      Mono- oder Stereo-Audio (channels, samples) oder (samples,)
            sr:         Sample-Rate in Hz.
            block_sec:  Fenstergröße in Sekunden (0.4 for momentary, 3.0 for short-term).
            hop_sec:    Hop-Größe in Sekunden.

        Returns:
            np.ndarray: LUFS-Wert je Block, shape=(n_blocks,).
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # K-Gewichtung anwenden (lfilter unterstützt axis=1 für multi-channel)
        audio_f64 = audio.astype(np.float64)
        s1 = scipy.signal.lfilter(self.prefilter[0], self.prefilter[1], audio_f64, axis=1)
        filtered = scipy.signal.lfilter(self.rlb_filter[0], self.rlb_filter[1], s1, axis=1)

        # Kanalgewichte
        n_ch = audio.shape[0]
        if n_ch == 1:
            ch_weights = np.array([1.0])
        elif n_ch == 2:
            ch_weights = np.array([1.0, 1.0])
        else:
            _w = np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
            ch_weights = _w[:n_ch]

        block_samps = max(int(block_sec * sr), 1)
        hop_samps = max(int(hop_sec * sr), 1)
        n_total = filtered.shape[1]
        n_blocks = max((n_total - block_samps) // hop_samps + 1, 0)

        if n_blocks <= 0:
            return np.array([], dtype=np.float64)

        # Vectorized block extraction
        starts = np.arange(n_blocks) * hop_samps
        indices = starts[:, np.newaxis] + np.arange(block_samps)  # (n_blocks, block_samps)
        # Use np.take for type-safe advanced indexing along sample axis.
        blocks = np.take(filtered, indices, axis=1)  # (n_ch, n_blocks, block_samps)
        ms = np.mean(blocks**2, axis=2)  # (n_ch, n_blocks)
        weight_sum = np.sum(ch_weights) + 1e-30
        weighted = np.dot(ch_weights, ms) / weight_sum  # (n_blocks,)

        lufs_blocks = np.full(n_blocks, -np.inf, dtype=np.float64)
        valid = weighted > 0
        lufs_blocks[valid] = -0.691 + 10.0 * np.log10(weighted[valid])

        return lufs_blocks


class TruePeakDetector:
    """
    ITU-R BS.1770-4 True Peak detection

    True peak is measured by oversampling the signal 4x
    to detect inter-sample peaks.
    """

    def __init__(self, sr: int = 48000, oversample_factor: int = 4):
        self.sr = sr
        self.oversample_factor = oversample_factor

    def measure(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Misst true peak level.

        Args:
            audio: Mono or stereo audio
            sr: Sample rate

        Returns:
            Dictionary with peak measurements
        """
        # Ensure 2D array
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        true_peaks = []
        sample_peaks = []

        for ch in range(audio.shape[0]):
            # Sample peak
            sample_peak = np.max(np.abs(audio[ch]))
            sample_peaks.append(sample_peak)

            # True peak (oversample 4x)
            upsampled = scipy.signal.resample_poly(audio[ch], self.oversample_factor, 1)
            true_peak = np.max(np.abs(upsampled))
            true_peaks.append(true_peak)

        true_peak_max = np.max(true_peaks)
        sample_peak_max = np.max(sample_peaks)

        # Convert to dB
        true_peak_db = 20 * np.log10(true_peak_max + 1e-10)
        sample_peak_db = 20 * np.log10(sample_peak_max + 1e-10)

        # EBU R128 limit: -1 dBTP
        exceeded = true_peak_db > -1.0

        return {
            "true_peak_db": float(true_peak_db),
            "sample_peak_db": float(sample_peak_db),
            "true_peak_exceeded": bool(exceeded),
            "headroom_db": float(-1.0 - true_peak_db),
        }


class PhaseCorrelationMeter:
    """
    Stereo phase correlation measurement

    Phase correlation ranges from -1 (out of phase) to +1 (in phase)
    - +1.0: Perfect mono (L = R)
    - 0.0: Uncorrelated (wide stereo)
    - -1.0: Perfect anti-phase (L = -R, problematic for mono compatibility)
    """

    def measure(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Misst phase correlation.

        Args:
            audio: Stereo audio (2, samples)
            sr: Sample rate

        Returns:
            Dictionary with phase measurements
        """
        if audio.ndim == 1 or audio.shape[0] != 2:
            # Mono or not stereo
            return {"phase_correlation": None, "phase_coherence": None, "stereo_width": None}  # type: ignore[dict-item]

        left = audio[0]
        right = audio[1]

        # Guarded Pearson correlation — avoids NaN on silent/constant channels (§VERBOTEN: np.corrcoef)
        _lc = left.astype(float) - float(np.mean(left))
        _rc = right.astype(float) - float(np.mean(right))
        _nl = float(np.linalg.norm(_lc))
        _nr = float(np.linalg.norm(_rc))
        if _nl < 1e-12 or _nr < 1e-12:
            correlation = 1.0
        else:
            correlation = float(np.clip(np.dot(_lc, _rc) / (_nl * _nr), -1.0, 1.0))

        # Phase coherence (magnitude of correlation, 0 to 1)
        coherence = np.abs(correlation)

        # Stereo width (0 = mono, 1 = wide)
        width = 1.0 - coherence

        return {
            "phase_correlation": float(correlation),
            "phase_coherence": float(coherence),
            "stereo_width": float(width),
        }


class SpectrumAnalyzer:
    """
    Frequency spectrum analysis with perceptual weighting
    """

    def __init__(self, sr: int = 48000, n_fft: int = 8192):
        self.sr = sr
        self.n_fft = n_fft

    def analyze(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray | float]:
        """
        Berechnet frequency spectrum.

        Args:
            audio: Mono or stereo audio
            sr: Sample rate

        Returns:
            Dictionary with spectral data
        """
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Compute power spectrum
        freqs = np.fft.rfftfreq(self.n_fft, 1 / sr)
        fft = np.fft.rfft(audio, n=self.n_fft)
        power = np.abs(fft) ** 2

        # Convert to dB
        power_db = 10 * np.log10(power + 1e-10)

        # Spectral centroid (brightness)
        centroid = np.sum(freqs * power) / (np.sum(power) + 1e-10)

        # Spectral rolloff (90% of energy)
        cumsum = np.cumsum(power)
        rolloff_idx = np.where(cumsum >= 0.9 * cumsum[-1])[0][0]
        rolloff = freqs[rolloff_idx]

        return {
            "freqs": freqs,
            "power_db": power_db,
            "spectral_centroid": float(centroid),
            "spectral_rolloff": float(rolloff),
        }


class MeterV9:
    """
    Complete professional metering suite

    Combines LUFS, True Peak, Phase Correlation, and Spectrum Analysis
    """

    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.lufs_meter = LUFSMeter(sr)
        self.peak_detector = TruePeakDetector(sr)
        self.phase_meter = PhaseCorrelationMeter()
        self.spectrum_analyzer = SpectrumAnalyzer(sr)

    def analyze(self, audio: np.ndarray, sr: int, verbose: bool = True) -> MeteringResult:
        """
        Complete metering analysis

        Args:
            audio: Audio array (channels, samples) or (samples,)
            sr: Sample rate
            verbose: Print results

        Returns:
            MeteringResult with all measurements
        """
        # LUFS measurement
        lufs_result = self.lufs_meter.measure(audio, sr)

        # True Peak measurement
        peak_result = self.peak_detector.measure(audio, sr)

        # Phase correlation (stereo only)
        phase_result = self.phase_meter.measure(audio, sr)

        # Spectrum analysis
        spectrum_result = self.spectrum_analyzer.analyze(audio, sr)

        # ─ Momentary LUFS (400 ms, 100 ms Hop — EBU R128 Annex 1) ─────────────
        momentary_raw = self.lufs_meter.measure_blocks(audio, sr, block_sec=0.4, hop_sec=0.1)
        # Publiziere nur finite Werte (stille Blöcke bleiben -inf)
        momentary_lufs = np.where(np.isfinite(momentary_raw), momentary_raw, -np.inf)

        # ─ Short-term LUFS (3 s, 1 s Hop — EBU R128) ────────────────────────
        short_term_raw = self.lufs_meter.measure_blocks(audio, sr, block_sec=3.0, hop_sec=1.0)
        short_term_lufs = np.where(np.isfinite(short_term_raw), short_term_raw, -np.inf)

        # ─ Loudness Range (LRA) — EBU R128 Tech Doc 3342 v3 ───────────────
        #   Algorithmus (normativ):                                          │
        #     1. Absolutes Gate: Blöcke mit L_ST ≥ -70 LUFS behalten          │
        #     2. Relatives Gate: L_ST ≥ (power_avg - 20 LU) behalten          │
        #     3. LRA = Perzentil_95 − Perzentil_10 der gegateten Blöcke      │
        loudness_range = 0.0
        _st_finite = short_term_raw[np.isfinite(short_term_raw)]
        if len(_st_finite) >= 2:
            # Absolutes Gate
            _abs_gated = _st_finite[_st_finite >= -70.0]
            if len(_abs_gated) >= 2:
                # Relatives Gate (−20 LU relativ zum Mittelwert absolut-gegateter Blöcke)
                _pwr_avg = 10.0 * np.log10(np.mean(10.0 ** (_abs_gated / 10.0)) + 1e-30)
                _rel_gate = _pwr_avg - 20.0
                _rel_gated = _abs_gated[_abs_gated >= _rel_gate]
                if len(_rel_gated) >= 2:
                    loudness_range = float(np.percentile(_rel_gated, 95) - np.percentile(_rel_gated, 10))

        # Create result object
        result = MeteringResult(
            integrated_lufs=lufs_result["integrated_lufs"],
            loudness_range=loudness_range,
            momentary_lufs=momentary_lufs,
            short_term_lufs=short_term_lufs,
            true_peak_db=peak_result["true_peak_db"],
            sample_peak_db=peak_result["sample_peak_db"],
            true_peak_exceeded=peak_result["true_peak_exceeded"],  # type: ignore[arg-type]
            phase_correlation=phase_result["phase_correlation"],
            phase_coherence=phase_result["phase_coherence"],
            spectrum_freqs=spectrum_result["freqs"],
            spectrum_db=spectrum_result["power_db"],
            spectral_centroid=spectrum_result["spectral_centroid"],  # type: ignore[arg-type]
            spectral_rolloff=spectrum_result["spectral_rolloff"],  # type: ignore[arg-type]
        )

        if verbose:
            self._print_report(result)

        return result

    def _print_report(self, result: MeteringResult) -> None:
        """Protokolliert metering analysis report via standard logging."""
        _logger.info("\n" + "=" * 60)
        _logger.info("PROFESSIONAL AUDIO METERING REPORT")
        _logger.info("=" * 60)

        _logger.info("\n📊 LOUDNESS (ITU-R BS.1770-4)")
        _logger.info("  Integrated:  %7.1f LUFS", result.integrated_lufs)

        _logger.info("\n🔊 PEAK LEVELS")
        _logger.info("  True Peak:   %7.2f dBTP", result.true_peak_db)
        _logger.info("  Sample Peak: %7.2f dBFS", result.sample_peak_db)
        if result.true_peak_exceeded:
            _logger.warning("  ⚠️  True Peak überschreitet -1 dBTP (EBU R128 Limit)")
        else:
            headroom = -1.0 - result.true_peak_db
            _logger.info("  ✅ Headroom: %7.2f dB", headroom)

        if result.phase_correlation is not None:
            _logger.info("\n🎧 STEREO PHASE")
            _logger.info("  Correlation: %7.3f", result.phase_correlation)
            _logger.info("  Coherence:   %7.3f", result.phase_coherence)
            if result.phase_correlation < -0.5:
                _logger.warning("  ⚠️  Starker Gegenphasenanteil — Mono-Inkompatibilität möglich")
            elif result.phase_correlation > 0.9:
                _logger.info("  ℹ️  Nahezu Mono (schmales Stereobild)")

        # Momentary / Short-term Zusammenfassung
        if result.momentary_lufs.size > 0:
            _m_finite = result.momentary_lufs[np.isfinite(result.momentary_lufs)]
            if _m_finite.size > 0:
                _logger.info("\n⚡ MOMENTARY LUFS (400 ms, EBU R128)")
                _logger.info("  Max:         %7.1f LUFS", float(np.max(_m_finite)))
                _logger.info("  Min:         %7.1f LUFS", float(np.min(_m_finite)))

        if result.short_term_lufs.size > 0:
            _s_finite = result.short_term_lufs[np.isfinite(result.short_term_lufs)]
            if _s_finite.size > 0:
                _logger.info("\n📈 SHORT-TERM LUFS (3 s, EBU R128)")
                _logger.info("  Max:         %7.1f LUFS", float(np.max(_s_finite)))
                _logger.info("  Min:         %7.1f LUFS", float(np.min(_s_finite)))

        if result.loudness_range > 0.0:
            _logger.info("  LRA:         %7.1f LU", result.loudness_range)

        _logger.info("\n🎵 SPECTRAL ANALYSIS")
        _logger.info("  Centroid:    %7.1f Hz (Helligkeit)", result.spectral_centroid)
        _logger.info("  Rolloff:     %7.1f Hz (90 %% Energie)", result.spectral_rolloff)
        _logger.info("\n" + "=" * 60 + "\n")


# Convenience function
def meter_audio(audio: np.ndarray, sr: int = 48000, verbose: bool = True) -> MeteringResult:
    """
    Quick professional metering analysis

    Args:
        audio: Audio array
        sr: Sample rate
        verbose: Print report

    Returns:
        MeteringResult
    """
    meter = MeterV9(sr)
    return meter.analyze(audio, sr, verbose=verbose)
