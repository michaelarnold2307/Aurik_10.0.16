"""
vocal_dynamics_intelligence.py - Intelligent Vocal Dynamics (Phase 2.2)

Surgical dynamics processing for vocals:
- Micro-Compression (syllable-level dynamics)
- Consonant Punch (transient preservation)
- Breath-Aware Gating (musical ducking)

Author: AURIK Development Team
Version: 1.0.0
Date: 9. Februar 2026
"""

import logging
import warnings

import numpy as np
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


def _chunked_hilbert_envelope(audio: np.ndarray, sr: int) -> np.ndarray:
    """Berechnet |hilbert(audio)| in 30-s chunks to prevent OOM on long audio."""
    max_chunk = 30 * sr  # 30 s — ~46 MB complex128 per chunk
    n = len(audio)
    if n <= max_chunk:
        return np.abs(np.asarray(hilbert(audio), dtype=np.complex128))

    overlap = int(0.01 * sr)  # 10 ms overlap
    envelope = np.empty(n, dtype=np.float64)
    pos = 0
    while pos < n:
        end = min(pos + max_chunk, n)
        chunk_env = np.abs(np.asarray(hilbert(audio[pos:end]), dtype=np.complex128))
        if pos == 0:
            envelope[pos:end] = chunk_env
        else:
            fade = np.linspace(0.0, 1.0, overlap)
            envelope[pos : pos + overlap] = envelope[pos : pos + overlap] * (1.0 - fade) + chunk_env[:overlap] * fade
            envelope[pos + overlap : end] = chunk_env[overlap:]
        pos = end - overlap if end < n else n

    return envelope


warnings.filterwarnings("ignore", category=RuntimeWarning)


class MicroCompressor:
    """
    Syllable-level micro-compression for consistent vocal energy.
    """

    def __init__(self, threshold: float = -20.0, ratio: float = 2.0, attack_ms: float = 5.0, release_ms: float = 50.0):
        """
        Parameters
        ----------
        threshold : float
            Compression threshold (dB)
        ratio : float
            Compression ratio
        attack_ms : float
            Attack time in milliseconds
        release_ms : float
            Release time in milliseconds
        """
        self.threshold = threshold
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def compress(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Wendet an: micro-compression.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_compressed : np.ndarray
            Compressed audio
        metrics : Dict
            Compression metrics
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        # Compute envelope
        envelope = self._compute_envelope(audio, sr)

        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Compute gain reduction
        gain_reduction_db = np.zeros_like(envelope_db)
        above_threshold = envelope_db > self.threshold
        gain_reduction_db[above_threshold] = (envelope_db[above_threshold] - self.threshold) * (1 - 1 / self.ratio)

        # Apply ballistics (attack/release)
        gain_reduction_db = self._apply_ballistics(gain_reduction_db, sr, self.attack_ms, self.release_ms)

        # Convert to linear gain
        gain = 10 ** (-gain_reduction_db / 20)

        # Apply gain
        audio_compressed = audio * gain

        # NaN/Inf-Guard + Clipping
        audio_compressed = np.nan_to_num(audio_compressed, nan=0.0, posinf=0.0, neginf=0.0)
        audio_compressed = np.clip(audio_compressed, -1.0, 1.0)

        # Metrics
        max_gr = np.max(gain_reduction_db)
        avg_gr = np.mean(gain_reduction_db[gain_reduction_db > 0]) if np.any(gain_reduction_db > 0) else 0

        metrics = {
            "max_gain_reduction_db": max_gr,
            "avg_gain_reduction_db": avg_gr,
            "ratio": self.ratio,
            "threshold_db": self.threshold,
        }

        return audio_compressed, metrics

    def _compute_envelope(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Berechnet amplitude envelope using chunked Hilbert transform (OOM-safe).
        """
        envelope = _chunked_hilbert_envelope(audio, sr)

        # Smooth (10ms window)
        window_samples = int(0.01 * sr)
        if window_samples > 0:
            kernel = np.ones(window_samples) / window_samples
            envelope = np.convolve(envelope, kernel, mode="same")

        return envelope

    def _apply_ballistics(self, gain_reduction: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
        """
        Wendet Attack/Release-Ballistiken an.
        """
        attack_coeff = np.exp(-1000 / (attack_ms * sr))
        release_coeff = np.exp(-1000 / (release_ms * sr))

        gain_smooth = np.empty_like(gain_reduction)
        prev = 0.0

        for i in range(len(gain_reduction)):
            target = gain_reduction[i]
            if target > prev:
                prev = attack_coeff * prev + (1 - attack_coeff) * target
            else:
                prev = release_coeff * prev + (1 - release_coeff) * target
            gain_smooth[i] = prev

        return gain_smooth


class ConsonantPunchEnhancer:
    """
    Preserves/enhances consonant transients for clarity.
    """

    def __init__(self, enhance_db: float = 3.0):
        """
        Parameters
        ----------
        enhance_db : float
            Transient enhancement in dB
        """
        self.enhance_db = enhance_db

    def enhance(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Enhance consonant punch.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_enhanced : np.ndarray
            Enhanced audio
        metrics : Dict
            Enhancement metrics
        """
        # Detect transients
        transient_mask = self._detect_transients(audio, sr)

        # Create transient enhancement gain
        gain = np.ones_like(audio)
        gain[transient_mask] = 10 ** (self.enhance_db / 20)

        # Smooth gain transitions
        window_samples = int(0.005 * sr)  # 5ms
        if window_samples > 0:
            kernel = np.hanning(window_samples)
            kernel /= kernel.sum()
            gain = np.convolve(gain, kernel, mode="same")

        # Apply
        audio_enhanced = audio * gain

        metrics = {"transients_detected": np.sum(transient_mask), "enhancement_db": self.enhance_db}

        return audio_enhanced, metrics

    def _detect_transients(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Erkennt transients using envelope derivative.
        """
        # Compute envelope (chunked Hilbert — OOM-safe)
        envelope = _chunked_hilbert_envelope(audio, sr)

        # Compute derivative
        derivative = np.diff(envelope, prepend=0)

        # Threshold (adaptive)
        threshold = np.percentile(np.abs(derivative), 95)

        # Detect transients
        transient_mask = np.abs(derivative) > threshold

        return transient_mask  # type: ignore[no-any-return]


class BreathAwareGate:
    """
    Musical gating that respects breath events.
    """

    def __init__(self, threshold_db: float = -50.0, reduction_db: float = 12.0, hold_ms: float = 100.0):
        """
        Parameters
        ----------
        threshold_db : float
            Gate threshold (dB)
        reduction_db : float
            Amount of reduction when gate closes (dB)
        hold_ms : float
            Hold time in milliseconds
        """
        self.threshold_db = threshold_db
        self.reduction_db = reduction_db
        self.hold_ms = hold_ms

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Wendet atemgesteuertes Gating an.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono)
        sr : int
            Sample rate in Hz

        Returns
        -------
        audio_gated : np.ndarray
            Gated audio
        metrics : Dict
            Gating metrics
        """
        # Compute envelope (chunked Hilbert — OOM-safe)
        envelope = _chunked_hilbert_envelope(audio, sr)

        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)

        # Gate decision
        gate_open = envelope_db > self.threshold_db

        # Apply hold
        gate_open = self._apply_hold(gate_open, sr, self.hold_ms)

        # Compute gain
        reduction_linear = 10 ** (-self.reduction_db / 20)
        gain = np.where(gate_open, 1.0, reduction_linear)

        # Smooth transitions
        window_samples = int(0.01 * sr)  # 10ms
        if window_samples > 0:
            kernel = np.hanning(window_samples)
            kernel /= kernel.sum()
            gain = np.convolve(gain, kernel, mode="same")

        # Apply
        audio_gated = audio * gain

        metrics = {
            "gate_open_percent": np.mean(gate_open) * 100,
            "threshold_db": self.threshold_db,
            "reduction_db": self.reduction_db,
        }

        return audio_gated, metrics

    def _apply_hold(self, gate_open: np.ndarray, sr: int, hold_ms: float) -> np.ndarray:
        """
        Wendet an: hold time to gate (vectorised).
        """
        hold_samples = int(hold_ms * sr / 1000)
        if hold_samples <= 0:
            return gate_open.copy()

        gate_bool = gate_open.astype(bool)
        if not gate_bool.any():
            return gate_bool

        gate_with_hold = gate_bool.copy()
        # Only process True→False transitions (very few, typically 50-200
        # for a 225 s file) instead of iterating over all True indices.
        padded = np.concatenate([gate_bool, np.array([False], dtype=bool)])
        closes = np.where(padded[:-1] & ~padded[1:])[0]
        for close_idx in closes:
            end = min(close_idx + 1 + hold_samples, len(gate_with_hold))
            gate_with_hold[close_idx + 1 : end] = True

        return gate_with_hold


class VocalDynamicsIntelligence:
    """
    Einheitliche API für Vokal-Dynamik-Intelligenz.
    """

    def __init__(self, compression_ratio: float = 2.0, enhancement_db: float = 3.0, gate_enabled: bool = True):
        """
        Parameters
        ----------
        compression_ratio : float
            Micro-compression ratio
        enhancement_db : float
            Consonant transient enhancement (dB)
        gate_enabled : bool
            Enable breath-aware gating
        """
        self.micro_compressor = MicroCompressor(ratio=compression_ratio)
        self.consonant_enhancer = ConsonantPunchEnhancer(enhance_db=enhancement_db)
        self.breath_gate = BreathAwareGate()
        self.gate_enabled = gate_enabled

    def process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """
        Full vocal dynamics processing.

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
        # Handle stereo
        if audio.ndim == 2:
            logger.debug("VocalDynamicsIntelligence: Eingabe shape %s", audio.shape)
            # Auto-detect format: (channels, samples) vs (samples, channels)
            # Heuristic: If first dimension is small and < second dimension, likely channels
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 32:
                # Format: (channels, samples) - process each channel
                logger.debug("VocalDynamicsIntelligence: format (channels, samples)")
                left, left_report = self.process(audio[0], sr)
                right, right_report = self.process(audio[1], sr)
                result = np.vstack([left, right])
                logger.debug("VocalDynamicsIntelligence: Ausgabe shape %s", result.shape)
                return result, {**left_report, "stereo": True}
            else:
                # Format: (samples, channels) - transpose to (channels, samples) for processing
                logger.debug("VocalDynamicsIntelligence: format (samples, channels)")
                audio_transposed = audio.T
                logger.debug("VocalDynamicsIntelligence: transposed shape %s", audio_transposed.shape)
                left, left_report = self.process(audio_transposed[0], sr)
                logger.debug("VocalDynamicsIntelligence: left shape %s", left.shape)
                right, _right_report = self.process(audio_transposed[1], sr)
                logger.debug("VocalDynamicsIntelligence: right shape %s", right.shape)
                # Return in original format: (samples, channels)
                result = np.column_stack([left, right])
                logger.debug("VocalDynamicsIntelligence: Ausgabe shape %s", result.shape)
                return result, {**left_report, "stereo": True}

        # Stage 1: Micro-compression
        audio_comp, comp_metrics = self.micro_compressor.compress(audio, sr)

        # Stage 2: Consonant enhancement
        audio_cons, cons_metrics = self.consonant_enhancer.enhance(audio_comp, sr)

        # Stage 3: Breath-aware gating (optional)
        if self.gate_enabled:
            audio_final, gate_metrics = self.breath_gate.process(audio_cons, sr)
        else:
            audio_final = audio_cons
            gate_metrics = {"enabled": False}

        report = {
            "micro_compression": comp_metrics,
            "consonant_enhancement": cons_metrics,
            "breath_aware_gating": gate_metrics,
        }

        return audio_final, report


# CLI interface
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser(description="Vocal Dynamics Intelligence - Surgical dynamics processing")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output", help="Output audio file")
    parser.add_argument("--compression-ratio", type=float, default=2.0, help="Compression ratio")
    parser.add_argument("--consonant-enhance", type=float, default=3.0, help="Consonant enhancement (dB)")
    parser.add_argument("--no-gate", action="store_true", help="Disable breath-aware gating")

    args = parser.parse_args()

    # Load audio
    from backend.file_import import load_audio_file

    _res = load_audio_file(args.input)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Process
    dynamics = VocalDynamicsIntelligence(
        compression_ratio=args.compression_ratio, enhancement_db=args.consonant_enhance, gate_enabled=not args.no_gate
    )

    audio_processed, report = dynamics.process(audio, sr)

    # Print report
    logger.info(str("\n" + "=" * 70))
    logger.info("VOCAL DYNAMICS INTELLIGENCE REPORT")
    logger.info(str("=" * 70))

    logger.info("\n[Micro-Compression]")
    comp = report["micro_compression"]
    logger.info("  Max gain reduction: %.1f dB", comp["max_gain_reduction_db"])
    logger.info("  Avg gain reduction: %.1f dB", comp["avg_gain_reduction_db"])
    logger.info("  Verhaeltnis:              %.1f:1", comp["ratio"])

    logger.info("\n[Consonant Enhancement]")
    cons = report["consonant_enhancement"]
    logger.info("  Transients erkannt: %s", cons["transients_detected"])
    logger.info("  Enhancement:         %.1f dB", cons["enhancement_db"])

    if report["breath_aware_gating"].get("enabled", True):
        logger.info("\n[Breath-Aware Gating]")
        gate = report["breath_aware_gating"]
        logger.info("  Gate open:    %.1f%%", gate["gate_open_percent"])
        logger.info("  Schwelle:    %.1f dB", gate["threshold_db"])
        logger.info("  Reduction:    %.1f dB", gate["reduction_db"])

    logger.info(str("=" * 70))

    # Save
    if args.output:
        sf.write(args.output, audio_processed, sr)
        logger.info("\n✅ gespeichert to: %s", args.output)
