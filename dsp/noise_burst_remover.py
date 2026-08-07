"""
AURIK 8.0: Noise Burst Removal
Entfernt transiente Störungen (plötzliche laute Impulse, Spikes, Pops)

Integration: Phase 2.2 (nach Click/Crackle, vor Dropout Repair)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DSPContract:
    """Contract definition für DSP-Module"""

    id: str = "noise_burst_remover"
    category: str = "restoration"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


noise_burst_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "NOISE_BURST detected in forensic_analysis", "reason": "Only process when needed"}],
    params={
        "defaults": {
            "threshold_db": -20.0,  # Local peak threshold over RMS
            "min_burst_ms": 5.0,  # Minimum burst duration
            "max_burst_ms": 100.0,  # Maximum burst duration
            "lookahead_ms": 10.0,  # Context window for analysis
        },
        "safe_ranges": {
            "threshold_db": {"min": -40.0, "max": -10.0},
            "min_burst_ms": {"min": 1.0, "max": 20.0},
            "max_burst_ms": {"min": 30.0, "max": 500.0},
        },
        "trial_profile": {
            "wet": 1.0,
            "segment_sec": 1.0,
            "warmup_ms": 0,
        },
    },
    budgets={
        "artifact_budget": 0.04,  # Max 4% artifacts introduced
        "identity_budget": 0.90,  # Min 90% signal preservation
        "spectral_change_budget": 0.05,
        "temporal_change_budget": 0.05,
        "compute_cost": 0.15,
    },
    side_effects=[
        {"risk": "Transient smearing", "expected_when": "threshold_db too low", "severity": 0.3},
        {"risk": "False positives", "expected_when": "natural drum hits detected", "severity": 0.2},
    ],
    reports={
        "self_metrics": ["bursts_detected", "bursts_removed", "avg_reduction_db"],
        "confidence": 0.80,
    },
    rollback={
        "strategy": "snapshot_restore",
        "supports_partial": True,
    },
)


class NoiseBurstRemover:
    """
    Entfernt transiente Störungen (Noise Bursts, Pops, Spikes)

    Features:
    - Adaptive threshold basierend auf lokalem RMS-Level
    - Unterscheidet zwischen musikalischen Transienten und Störungen
    - Spectral smoothing für natürliche Reparatur
    - Instrumenten-bewusste Verarbeitung (preserve_transients Mode)

    Integration Point: Phase 2.2 in unified_restorer_v2.py
    """

    def __init__(
        self,
        sr: int,
        threshold_db: float = -20.0,
        min_burst_ms: float = 5.0,
        max_burst_ms: float = 100.0,
        preserve_transients: bool = False,
    ):
        """
        Args:
            sr: Sample rate
            threshold_db: Threshold über lokalem RMS (in dB) für Burst-Detektion
            min_burst_ms: Minimale Burst-Dauer
            max_burst_ms: Maximale Burst-Dauer
            preserve_transients: Falls True, schonender Modus für Drum-Heavy Material
        """
        self.sr = sr
        self.threshold_db = threshold_db
        self.min_burst_samples = int((min_burst_ms / 1000.0) * sr)
        self.max_burst_samples = int((max_burst_ms / 1000.0) * sr)
        self.preserve_transients = preserve_transients
        self.contract = noise_burst_contract

        # Statistics for reporting
        self.bursts_detected = 0
        self.bursts_removed = 0
        self.avg_reduction_db = 0.0

    def process(self, audio: np.ndarray, severity: str = "MEDIUM") -> np.ndarray:
        """
        Entfernt Noise Bursts aus Audio-Signal

        Args:
            audio: Audio signal (mono oder stereo)
            severity: Severity level from forensic analysis ('LOW', 'MEDIUM', 'HIGH')

        Returns:
            Processed audio
        """
        # Adjust threshold based on severity
        if severity == "HIGH":
            effective_threshold = self.threshold_db - 5.0  # More aggressive
        elif severity == "LOW":
            effective_threshold = self.threshold_db + 5.0  # More conservative
        else:
            effective_threshold = self.threshold_db

        # Process channels independently
        if audio.ndim == 2:
            processed = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                processed[:, ch] = self._process_mono(audio[:, ch], effective_threshold)
        else:
            processed = self._process_mono(audio, effective_threshold)

        return processed

    def _process_mono(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Verarbeitet single channel."""
        # Step 1: Detect noise bursts
        burst_regions = self._detect_bursts(audio, threshold_db)

        if len(burst_regions) == 0:
            return audio

        self.bursts_detected = len(burst_regions)
        logging.info(f"[NoiseBurstRemover] Detected {len(burst_regions)} bursts")

        # Step 2: Remove each burst
        audio_cleaned = audio.copy()
        total_reduction = 0.0

        for start, end in burst_regions:
            try:
                audio_cleaned, reduction = self._remove_burst(audio_cleaned, start, end)
                total_reduction += reduction
                self.bursts_removed += 1
            except Exception as e:
                logging.warning(f"[NoiseBurstRemover] Failed to remove burst [{start}, {end}]: {e}")

        self.avg_reduction_db = total_reduction / max(self.bursts_removed, 1)

        return audio_cleaned

    def _detect_bursts(self, audio: np.ndarray, threshold_db: float) -> list[tuple[int, int]]:
        """
        Erkennt transient noise bursts using local RMS analysis.

        Returns:
            List of (start, end) tuples marking burst regions
        """
        # Compute local RMS envelope
        window_size = int(0.010 * self.sr)  # 10ms window
        hop_size = window_size // 2

        # Pad audio for windowing
        padded = np.pad(audio, (window_size // 2, window_size // 2), mode="reflect")

        # Local RMS computation
        rms_envelope = []
        for i in range(0, len(audio), hop_size):
            window = padded[i : i + window_size]
            rms = np.sqrt(np.mean(window**2))
            rms_envelope.append(rms)

        rms_envelope = np.array(rms_envelope)  # type: ignore[assignment]

        # Compute threshold: median RMS + threshold_db
        median_rms = np.median(rms_envelope)
        threshold_linear = median_rms * (10 ** (threshold_db / 20.0))

        # Detect peaks above threshold
        burst_candidate_indices = np.where(rms_envelope > threshold_linear)[0]

        if len(burst_candidate_indices) == 0:
            return []

        # Group consecutive indices into regions
        burst_regions = []
        current_start = burst_candidate_indices[0]
        current_end = burst_candidate_indices[0]

        for idx in burst_candidate_indices[1:]:
            if idx == current_end + 1:
                current_end = idx
            else:
                # Convert from envelope indices back to sample indices
                sample_start = current_start * hop_size
                sample_end = (current_end + 1) * hop_size
                burst_length = sample_end - sample_start

                # Filter by duration
                if self.min_burst_samples <= burst_length <= self.max_burst_samples:
                    burst_regions.append((sample_start, sample_end))

                current_start = idx
                current_end = idx

        # Add final region
        sample_start = current_start * hop_size
        sample_end = (current_end + 1) * hop_size
        burst_length = sample_end - sample_start
        if self.min_burst_samples <= burst_length <= self.max_burst_samples:
            burst_regions.append((sample_start, min(sample_end, len(audio))))

        # Filter false positives if preserve_transients is enabled
        if self.preserve_transients:
            burst_regions = self._filter_musical_transients(audio, burst_regions)

        return burst_regions

    def _filter_musical_transients(
        self, audio: np.ndarray, burst_regions: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        Filtert out likely musical transients (drum hits, piano attacks).
        using spectral characteristics
        """
        filtered_regions = []

        for start, end in burst_regions:
            # Analyze spectral content
            burst_segment = audio[start:end]

            # Musical transients have:
            # 1. Richer harmonic structure (nicht nur hohe Frequenzen)
            # 2. Längere Decay-Zeit
            # 3. Nicht-clipping Charakteristik

            # Check for clipping (strong indicator of noise, not music)
            max_abs = np.max(np.abs(burst_segment))
            if max_abs > 0.95:  # Near clipping
                filtered_regions.append((start, end))
                continue

            # Check spectral balance
            # Noise bursts haben oft unnatürliche Spektren (z.B. nur Highs)
            if len(burst_segment) >= 512:
                try:
                    spectrum = np.abs(np.fft.rfft(burst_segment))
                    freqs = np.fft.rfftfreq(len(burst_segment), 1 / self.sr)

                    # Compute energy in low (<2kHz) vs high (>5kHz) bands
                    low_mask = freqs < 2000
                    high_mask = freqs > 5000

                    low_energy = np.sum(spectrum[low_mask])
                    high_energy = np.sum(spectrum[high_mask])

                    # Noise bursts sind oft high-frequency dominant
                    if high_energy > 3 * low_energy:
                        filtered_regions.append((start, end))
                except Exception:
                    # Analysis failure → keep candidate
                    filtered_regions.append((start, end))
            else:
                # Too short to analyze reliably → keep
                filtered_regions.append((start, end))

        logging.info(f"[NoiseBurstRemover] Filtered {len(burst_regions) - len(filtered_regions)} musical transients")

        return filtered_regions

    def _remove_burst(self, audio: np.ndarray, start: int, end: int) -> tuple[np.ndarray, float]:
        """
        Entfernt a single burst using spectral smoothing.

        Returns:
            (cleaned_audio, reduction_db)
        """
        burst_length = end - start

        # Strategy 1: For very short bursts (<100 samples), use cubic spline
        if burst_length < 100:
            if start > 0 and end < len(audio):
                try:
                    from scipy.interpolate import CubicSpline

                    # Use surrounding context for smooth interpolation
                    ctx = min(20, start, len(audio) - end)
                    x_ctx = list(range(start - ctx, start)) + list(range(end, end + ctx))
                    y_ctx = list(audio[start - ctx : start]) + list(audio[end : end + ctx])
                    if len(x_ctx) >= 4:
                        cs = CubicSpline(x_ctx, y_ctx)
                        audio[start:end] = cs(np.arange(start, end))
                    else:
                        audio[start:end] = np.interp(
                            np.arange(start, end), [start - 1, end], [audio[start - 1], audio[end]]
                        )
                except Exception:
                    audio[start:end] = np.interp(
                        np.arange(start, end), [start - 1, end], [audio[start - 1], audio[end]]
                    )
            reduction_db = 20.0  # Estimate
            return audio, reduction_db

        # Strategy 2: For longer bursts, use spectral attenuation
        # Extract context (before and after burst)
        context_samples = min(burst_length * 2, 2048)
        pre_start = max(0, start - context_samples)
        post_end = min(len(audio), end + context_samples)

        pre_context = audio[pre_start:start]
        post_context = audio[end:post_end]
        burst_segment = audio[start:end]

        if len(pre_context) < 10 or len(post_context) < 10:
            # Not enough context → fallback to interpolation
            return self._remove_burst(audio, start, min(start + 99, end))

        # Compute "expected" amplitude based on context
        pre_rms = np.sqrt(np.mean(pre_context**2))
        post_rms = np.sqrt(np.mean(post_context**2))
        expected_rms = (pre_rms + post_rms) / 2

        burst_rms = np.sqrt(np.mean(burst_segment**2))

        # Attenuation factor
        if burst_rms > expected_rms:
            attenuation = expected_rms / (burst_rms + 1e-10)
            reduction_db = 20 * np.log10((burst_rms + 1e-10) / (expected_rms + 1e-10))
        else:
            attenuation = 1.0
            reduction_db = 0.0

        # Apply gentle attenuation with crossfade
        fade_len = min(50, burst_length // 4)
        fade_in = np.linspace(1.0, attenuation, fade_len)
        fade_out = np.linspace(attenuation, 1.0, fade_len)

        burst_attenuated = burst_segment * attenuation

        if fade_len > 0:
            burst_attenuated[:fade_len] *= fade_in
            burst_attenuated[-fade_len:] *= fade_out

        audio[start:end] = burst_attenuated

        return audio, reduction_db

    def get_metrics(self) -> dict[str, Any]:
        """Gibt processing metrics zurück."""
        return {
            "bursts_detected": self.bursts_detected,
            "bursts_removed": self.bursts_removed,
            "avg_reduction_db": self.avg_reduction_db,
        }
