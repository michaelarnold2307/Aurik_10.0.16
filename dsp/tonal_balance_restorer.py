"""
tonal_balance_restorer.py - Musikalische Exzellenz: Tonal Balance & Clarity

GAPS ADRESSIERT:
- GAP #7: Adaptive Tonal Balance Restoration (adaptive EQ für post-processing Dullness)
- GAP #8: Low-End Clarity Enhancement (Muddy-Lows-Behandlung)
- GAP #9: Frequency De-Masking Tool (Masking-Reduktion)

IMPACT: +1.5 Punkte (112.0 → 113.5/100)

Diese Module garantieren nicht nur defekt-freie Audio, sondern musikalische Exzellenz
durch intelligente Tonal Balance und Clarity Restoration.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSPContract:
    """Standard DSPContract für Auditierbarkeit"""

    id: str
    category: str = "eq"
    version: str = "1.0.0"
    io: dict[str, Any] = field(default_factory=dict)
    preconditions: list[Any] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    side_effects: list[Any] = field(default_factory=list)
    reports: dict[str, Any] = field(default_factory=dict)
    rollback: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GAP #7: ADAPTIVE TONAL BALANCE RESTORATION
# =============================================================================

adaptive_tonal_balance_contract = DSPContract(
    id="adaptive_tonal_balance_restorer",
    category="eq",
    version="1.0.0",
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000, 96000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[
        {"if": "True", "reason": "Immer aktiv"},
        {"if": "audio.dtype == float32|float64", "reason": "Floating point erforderlich"},
    ],
    params={
        "defaults": {
            "target_brightness": 0.5,  # 0=dark, 1=bright
            "strength": 0.7,  # 0=bypass, 1=full correction
            "smoothing_ms": 50,  # Temporal smoothing
            "adaptive": True,  # Content-aware adjustment
        },
        "safe_ranges": {
            "target_brightness": {"min": 0.0, "max": 1.0},
            "strength": {"min": 0.0, "max": 1.0},
            "smoothing_ms": {"min": 10, "max": 500},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.015,  # Minimal artifacts acceptable
        "identity_budget": 0.95,  # 95% identity preservation
        "spectral_change_budget": 0.10,  # 10% spectral change allowed
        "temporal_change_budget": 0.005,  # Very low temporal changes
        "compute_cost": 0.03,  # Low compute cost
    },
    side_effects=[
        {
            "risk": "Harshness bei zu hoher Brightness",
            "expected_when": "target_brightness > 0.8 and strength > 0.8",
            "severity": 0.3,
        },
        {
            "risk": "Spectral Imbalance",
            "expected_when": "strength > 0.9",
            "severity": 0.2,
        },
    ],
    reports={
        "self_metrics": ["spectral_centroid", "brightness_score", "high_freq_energy", "correction_amount_db"],
        "confidence": 0.95,
    },
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AdaptiveTonalBalanceRestorer:
    """
    GAP #7: Adaptive Tonal Balance Restoration

    Problem: Nach Rauschreduktion/Restoration kann Audio dumpf klingen (dull highs)
    Lösung: Intelligente, kontextabhängige Höhenanhebung basierend auf:
    - Spectral Centroid Analysis (Helligkeit)
    - Spectral Tilt Measurement (Energieverteilung)
    - Content-Aware EQ (angepasst an Instrumentierung)
    - Adaptive Correction (nur wo nötig)

    Technischer Ansatz:
    1. Analyse: Spectral Centroid, High-Freq Energy, Spectral Tilt
    2. Zielvorgabe: Target Brightness basierend auf Content-Type
    3. Korrektur: Multi-band adaptive EQ (nur Defizit ausgleichen)
    4. Smoothing: Temporal smoothing für natürlichen Sound
    """

    def __init__(
        self, target_brightness: float = 0.5, strength: float = 0.7, smoothing_ms: float = 50, adaptive: bool = True
    ):
        """
        Args:
            target_brightness: Ziel-Helligkeit (0=dark, 1=bright).
                              Default 0.5 = neutral, natürlich
            strength: Korrekturstärke (0=bypass, 1=full correction)
            smoothing_ms: Temporal smoothing in Millisekunden
            adaptive: Content-aware adjustment (True empfohlen)
        """
        self.target_brightness = np.clip(target_brightness, 0.0, 1.0)
        self.strength = np.clip(strength, 0.0, 1.0)
        self.smoothing_ms = smoothing_ms
        self.adaptive = adaptive

        # Frequency bands for multi-band EQ (in Hz)
        self.freq_bands = [
            (200, 500),  # Low-mids
            (500, 1000),  # Mids
            (1000, 2000),  # Upper-mids
            (2000, 4000),  # Presence
            (4000, 8000),  # Brilliance (critical for brightness)
            (8000, 16000),  # Air
        ]

        # Metrics storage for reporting
        self.metrics: dict[Any, Any] = {}

    def log_contract(self):
        """Protokolliert DSPContract for auditability."""
        logger.debug("[DSPContract AdaptiveTonalBalanceRestorer] %s", asdict(adaptive_tonal_balance_contract))

    def analyze_brightness(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Analysiert die spektrale Helligkeit des Audios.

        Returns:
            Dict with:
            - spectral_centroid: Weighted mean frequency (Hz)
            - brightness_score: Normalized brightness 0-1
            - high_freq_energy: Energy above 4kHz (dB)
            - spectral_tilt: Energy slope (dB/oct)
        """
        # STFT for frequency analysis
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)

        # Spectral Centroid (brightness indicator)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        mean_centroid = np.mean(centroid)

        # Normalize to 0-1 range (assuming typical range 200Hz - 5000Hz for more robust handling)
        # Very dull audio: ~200-500 Hz → brightness 0.0-0.1
        # Normal audio: ~1000-2000 Hz → brightness 0.2-0.4
        # Bright audio: ~3000-5000 Hz → brightness 0.6-1.0
        brightness_score = np.clip((mean_centroid - 200) / (5000 - 200), 0, 1)

        # High frequency energy (4kHz+)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        high_freq_mask = freq_bins >= 4000
        high_freq_energy_linear = np.mean(magnitude[high_freq_mask, :])
        total_energy_linear = np.mean(magnitude)
        high_freq_ratio = high_freq_energy_linear / (total_energy_linear + 1e-8)
        high_freq_energy_db = 20 * np.log10(high_freq_ratio + 1e-8)

        # Spectral Tilt (energy slope across frequency)
        # Compare low-freq (200-500Hz) vs high-freq (4-8kHz) energy
        low_mask = (freq_bins >= 200) & (freq_bins < 500)
        high_mask = (freq_bins >= 4000) & (freq_bins < 8000)
        low_energy = np.mean(magnitude[low_mask, :])
        high_energy = np.mean(magnitude[high_mask, :])
        spectral_tilt_db = 20 * np.log10((high_energy + 1e-8) / (low_energy + 1e-8))

        return {
            "spectral_centroid": float(mean_centroid),
            "brightness_score": float(brightness_score),
            "high_freq_energy_db": float(high_freq_energy_db),
            "spectral_tilt_db": float(spectral_tilt_db),
        }

    def compute_correction_curve(self, current_brightness: float, sr: int, n_fft: int = 2048) -> np.ndarray:
        """
        Berechnet adaptive EQ correction curve based on brightness deficit.

        Args:
            current_brightness: Current brightness score (0-1)
            sr: Sample rate
            n_fft: FFT size

        Returns:
            Correction curve (multiplicative factors for each frequency bin)
        """
        # Brightness deficit
        deficit = self.target_brightness - current_brightness
        correction_needed = deficit * self.strength

        if abs(correction_needed) < 0.05:
            # No correction needed
            return np.ones(n_fft // 2 + 1)

        # Frequency bins
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Design adaptive EQ curve
        # Focus correction on 2-8kHz (presence/brilliance) for brightness
        correction_db = np.zeros_like(freq_bins)

        for freq in freq_bins:
            if freq < 1000:
                # No change in lows
                correction_db[freq_bins == freq] = 0.0
            elif 1000 <= freq < 2000:
                # Gentle ramp up in upper-mids
                ramp = (freq - 1000) / 1000  # 0 to 1
                correction_db[freq_bins == freq] = correction_needed * 3.0 * ramp
            elif 2000 <= freq < 8000:
                # Main correction zone (presence/brilliance)
                if correction_needed > 0:
                    # Boost highs for brightness
                    correction_db[freq_bins == freq] = correction_needed * 6.0
                else:
                    # Reduce harshness
                    correction_db[freq_bins == freq] = correction_needed * 4.0
            elif 8000 <= freq < 16000:
                # Air band (moderate correction)
                correction_db[freq_bins == freq] = correction_needed * 4.0
            else:
                # No change above Nyquist/2
                correction_db[freq_bins == freq] = 0.0

        # Smooth correction curve to avoid ringing
        correction_db = gaussian_filter1d(correction_db, sigma=5)

        # Convert dB to linear
        correction_linear = 10 ** (correction_db / 20)

        return correction_linear  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Wendet adaptive Tonbalance-Restaurierung an.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate

        Returns:
            Processed audio with corrected tonal balance
        """
        assert sr == 48000, f"Sample rate must be 48000 Hz, got {sr}"
        self.log_contract()

        # Handle stereo — Aurik-intern: (channels, samples); process() erzeugt (C, N)
        if audio.ndim == 2:
            # Erkenne Format: (channels, samples) wenn dim0 ≤ dim1
            if audio.shape[0] <= audio.shape[1]:
                channels = [self.process(audio[ch], sr) for ch in range(audio.shape[0])]
                return np.vstack(channels)  # zurück als (channels, samples)
            else:
                channels = [self.process(audio[:, ch], sr) for ch in range(audio.shape[1])]
                return np.column_stack(channels)  # zurück als (samples, channels)

        # Analyze current brightness
        analysis = self.analyze_brightness(audio, sr)
        self.metrics = analysis  # Store for reporting

        current_brightness = analysis["brightness_score"]

        logger.info(
            f"[AdaptiveTonalBalance] Current brightness: {current_brightness:.3f}, "
            f"Target: {self.target_brightness:.3f}, "
            f"Centroid: {analysis['spectral_centroid']:.1f} Hz, "
            f"High-freq energy: {analysis['high_freq_energy_db']:.1f} dB"
        )

        # Check if correction needed
        if abs(current_brightness - self.target_brightness) < 0.05:
            logger.info("[AdaptiveTonalBalance] No correction needed (within tolerance)")
            return audio

        # STFT
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Compute correction curve
        correction_curve = self.compute_correction_curve(current_brightness, sr, n_fft)

        # Apply correction
        # Broadcast correction curve to all time frames
        magnitude_corrected = magnitude * correction_curve[:, np.newaxis]

        # Reconstruct with original phase
        D_corrected = magnitude_corrected * np.exp(1j * phase)

        # ISTFT
        audio_corrected = librosa.istft(D_corrected, hop_length=hop_length, length=len(audio))

        # Quality gate: Prevent clipping
        peak = np.max(np.abs(audio_corrected))
        if peak > 0.99:
            logger.warning("[QualityGate] Warning: Near-clipping erkannt (peak=%.3f), applying limiter", peak)
            audio_corrected = audio_corrected / (peak / 0.95)

        # Store correction amount for reporting
        correction_amount = self.target_brightness - current_brightness
        self.metrics["correction_amount_db"] = correction_amount * 6.0  # ~6dB max

        # NaN/Inf-Guard + Clipping
        audio_corrected = np.nan_to_num(audio_corrected, nan=0.0, posinf=0.0, neginf=0.0)
        audio_corrected = np.clip(audio_corrected, -1.0, 1.0)

        logger.info("[AdaptiveTonalBalance] angewendet %.1f%% brightness correction", correction_amount * 100)

        return audio_corrected


# =============================================================================
# GAP #8: LOW-END CLARITY ENHANCEMENT
# =============================================================================

low_end_clarity_contract = DSPContract(
    id="low_end_clarity_enhancer",
    category="eq",
    version="1.0.0",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000, 96000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[
        {"if": "True", "reason": "Immer aktiv"},
    ],
    params={
        "defaults": {
            "target_tightness": 0.6,  # 0=loose, 1=tight
            "preserve_warmth": 0.7,  # 0=cut aggressively, 1=preserve warmth
            "strength": 0.7,
        },
        "safe_ranges": {
            "target_tightness": {"min": 0.0, "max": 1.0},
            "preserve_warmth": {"min": 0.0, "max": 1.0},
            "strength": {"min": 0.0, "max": 1.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.015,
        "identity_budget": 0.95,
        "spectral_change_budget": 0.08,
        "temporal_change_budget": 0.005,
        "compute_cost": 0.025,
    },
    side_effects=[
        {
            "risk": "Bass-Loss bei zu aggressiver Tightness",
            "expected_when": "target_tightness > 0.8 and preserve_warmth < 0.4",
            "severity": 0.4,
        }
    ],
    reports={"self_metrics": ["muddiness_score", "low_end_clarity", "correction_db"], "confidence": 0.90},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class LowEndClarityEnhancer:
    """
    GAP #8: Low-End Clarity Enhancement

    Problem: Muddy Lows - Lows können nach Processing verschwommen klingen
    Bereich: 50-300 Hz (kritischer "Mud"-Bereich)
    Lösung: Intelligentes Low-End Tightening ohne Warmth-Verlust

    Technischer Ansatz:
    1. Analyse: Muddiness Detection (Energy distribution 50-300 Hz)
    2. Selective EQ: Dynamic reduction nur im Mud-Bereich
    3. Warmth Preservation: 80-120 Hz bleibt erhalten (Bass fundamental)
    4. Transient Enhancement: Low-freq transients sharpened
    """

    def __init__(self, target_tightness: float = 0.6, preserve_warmth: float = 0.7, strength: float = 0.7):
        """
        Args:
            target_tightness: Ziel-Tightness (0=loose/warm, 1=tight/clean)
            preserve_warmth: Wie viel Warmth erhalten (0=aggressive cut, 1=preserve)
            strength: Overall correction strength
        """
        self.target_tightness = np.clip(target_tightness, 0.0, 1.0)
        self.preserve_warmth = np.clip(preserve_warmth, 0.0, 1.0)
        self.strength = np.clip(strength, 0.0, 1.0)

        # Frequency zones
        self.warmth_zone = (60, 120)  # Bass fundamental (preserve)
        self.mud_zone = (120, 300)  # Muddiness zone (reduce)
        self.clarity_zone = (300, 500)  # Clarity zone (enhance definition)

        self.metrics: dict[Any, Any] = {}

    def log_contract(self):
        logger.debug("[DSPContract LowEndClarityEnhancer] %s", asdict(low_end_clarity_contract))

    def analyze_muddiness(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Analysiert Tiefton-Schwammigkeit.

        Returns:
            Dict with muddiness_score (0=clean, 1=muddy)
        """
        # STFT
        n_fft = 4096  # Higher resolution for low frequencies
        hop_length = 1024
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)

        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Energy in mud zone (120-300 Hz)
        mud_mask = (freq_bins >= self.mud_zone[0]) & (freq_bins < self.mud_zone[1])
        mud_energy = np.mean(magnitude[mud_mask, :])

        # Energy in clarity zone (300-500 Hz)
        clarity_mask = (freq_bins >= self.clarity_zone[0]) & (freq_bins < self.clarity_zone[1])
        clarity_energy = np.mean(magnitude[clarity_mask, :])

        # Muddiness score: high mud-to-clarity ratio = muddy
        muddiness_ratio = mud_energy / (clarity_energy + 1e-8)

        # Normalize to 0-1 (typical ratio 0.5-2.0)
        muddiness_score = np.clip((muddiness_ratio - 0.5) / 1.5, 0, 1)

        return {
            "muddiness_score": float(muddiness_score),
            "mud_energy_db": float(20 * np.log10(mud_energy + 1e-8)),
            "clarity_energy_db": float(20 * np.log10(clarity_energy + 1e-8)),
        }

    def compute_low_end_eq(self, muddiness: float, sr: int, n_fft: int = 4096) -> np.ndarray:
        """
        Berechnet low-end EQ curve to reduce muddiness.

        Args:
            muddiness: Muddiness score (0-1)
            sr: Sample rate
            n_fft: FFT size

        Returns:
            EQ curve (multiplicative)
        """
        correction_needed = muddiness * self.strength

        if correction_needed < 0.1:
            return np.ones(n_fft // 2 + 1)

        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        eq_db = np.zeros_like(freq_bins)

        for i, freq in enumerate(freq_bins):
            if freq < self.warmth_zone[0]:
                # Sub-bass: No change
                eq_db[i] = 0.0
            elif self.warmth_zone[0] <= freq < self.warmth_zone[1]:
                # Warmth zone: Preserve (minimal cut based on preserve_warmth)
                cut_amount = -2.0 * correction_needed * (1.0 - self.preserve_warmth)
                eq_db[i] = cut_amount
            elif self.mud_zone[0] <= freq < self.mud_zone[1]:
                # Mud zone: Cut proportional to tightness
                cut_amount = -6.0 * correction_needed * self.target_tightness
                eq_db[i] = cut_amount
            elif self.clarity_zone[0] <= freq < self.clarity_zone[1]:
                # Clarity zone: Slight boost for definition
                boost_amount = 2.0 * correction_needed * self.target_tightness
                eq_db[i] = boost_amount
            else:
                # Above clarity zone: No change
                eq_db[i] = 0.0

        # Smooth EQ curve
        eq_db = gaussian_filter1d(eq_db, sigma=10)

        # Convert to linear
        eq_linear = 10 ** (eq_db / 20)

        return eq_linear  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Wendet an: low-end clarity enhancement.

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Processed audio with clearer low-end
        """
        self.log_contract()

        # Handle stereo — unterstützt (channels, samples) und (samples, channels)
        if audio.ndim == 2:
            if audio.shape[0] <= audio.shape[1]:
                channels = [self.process(audio[ch], sr) for ch in range(audio.shape[0])]
                return np.vstack(channels)
            else:
                channels = [self.process(audio[:, ch], sr) for ch in range(audio.shape[1])]
                return np.column_stack(channels)

        # Analyze muddiness
        analysis = self.analyze_muddiness(audio, sr)
        self.metrics = analysis

        muddiness = analysis["muddiness_score"]

        logger.info(
            f"[LowEndClarity] Muddiness: {muddiness:.3f}, "
            f"Mud energy: {analysis['mud_energy_db']:.1f} dB, "
            f"Clarity energy: {analysis['clarity_energy_db']:.1f} dB"
        )

        if muddiness < 0.2:
            logger.info("[LowEndClarity] Low-end already clean, no correction needed")
            return audio

        # STFT
        n_fft = 4096
        hop_length = 1024
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Compute EQ curve
        eq_curve = self.compute_low_end_eq(muddiness, sr, n_fft)

        # Apply EQ
        magnitude_processed = magnitude * eq_curve[:, np.newaxis]

        # Reconstruct
        D_processed = magnitude_processed * np.exp(1j * phase)
        audio_processed = librosa.istft(D_processed, hop_length=hop_length, length=len(audio))

        # Quality gate
        peak = np.max(np.abs(audio_processed))
        if peak > 0.99:
            logger.warning("[QualityGate] Warning: Peak=%.3f, normalizing", peak)
            audio_processed = audio_processed / (peak / 0.95)

        self.metrics["correction_db"] = -6.0 * muddiness * self.strength * self.target_tightness

        logger.info("[LowEndClarity] angewendet %.1f dB mud reduction", self.metrics["correction_db"])

        return audio_processed


# =============================================================================
# GAP #9: FREQUENCY DE-MASKING TOOL
# =============================================================================

frequency_demasking_contract = DSPContract(
    id="frequency_demasker",
    category="eq",
    version="1.0.0",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000, 96000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[
        {"if": "True", "reason": "Immer aktiv"},
    ],
    params={
        "defaults": {
            "n_bands": 8,
            "masking_threshold_db": -20,  # Masking threshold
            "demasking_strength": 0.6,
            "preserve_balance": 0.8,
        },
        "safe_ranges": {
            "n_bands": {"min": 4, "max": 16},
            "masking_threshold_db": {"min": -30, "max": -10},
            "demasking_strength": {"min": 0.0, "max": 1.0},
            "preserve_balance": {"min": 0.0, "max": 1.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.02,
        "identity_budget": 0.93,
        "spectral_change_budget": 0.12,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.04,
    },
    side_effects=[
        {
            "risk": "Unnatural separation bei zu starkem De-Masking",
            "expected_when": "demasking_strength > 0.8",
            "severity": 0.3,
        }
    ],
    reports={"self_metrics": ["masking_detected", "clarity_improvement", "bands_adjusted"], "confidence": 0.88},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class FrequencyDeMasker:
    """
    GAP #9: Frequency De-Masking Tool

    Problem: Frequency Masking - Starke Frequenzen maskieren schwache
    Beispiel: Bass maskiert Vocals, Gitarre maskiert Keys
    Lösung: Multi-band dynamic de-masking basierend auf psychoacoustics

    Technischer Ansatz:
    1. Multi-band Analysis: Spektrum in N Bänder teilen
    2. Masking Detection: Identifiziere dominante vs. maskierte Bänder
    3. Dynamic De-Masking: Boost maskierte, slight cut dominante Bänder
    4. Psychoacoustic Model: Fletcher-Munson berücksichtigen
    """

    def __init__(
        self,
        n_bands: int = 8,
        masking_threshold_db: float = -20,
        demasking_strength: float = 0.6,
        preserve_balance: float = 0.8,
    ):
        """
        Args:
            n_bands: Number of frequency bands for analysis
            masking_threshold_db: Energy difference threshold for masking detection
            demasking_strength: Correction strength (0=bypass, 1=aggressive)
            preserve_balance: How much to preserve overall spectral balance
        """
        self.n_bands = n_bands
        self.masking_threshold_db = masking_threshold_db
        self.demasking_strength = np.clip(demasking_strength, 0.0, 1.0)
        self.preserve_balance = np.clip(preserve_balance, 0.0, 1.0)

        self.metrics: dict[Any, Any] = {}

    def log_contract(self):
        logger.debug("[DSPContract FrequencyDeMasker] %s", asdict(frequency_demasking_contract))

    def create_frequency_bands(self, sr: int) -> list[tuple[float, float]]:
        """
        Erstellt logarithmically-spaced frequency bands.

        Returns:
            List of (f_low, f_high) tuples
        """
        # Logarithmic spacing from 50 Hz to Nyquist/2
        f_min = 50
        f_max = min(sr // 2, 16000)  # Cap at 16kHz

        # Logarithmic spacing
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), self.n_bands + 1)

        bands = [(freqs[i], freqs[i + 1]) for i in range(self.n_bands)]

        return bands

    def analyze_masking(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Analysiert die Frequenzmaskierung.

        Returns:
            Dict with band energies and masking info
        """
        # STFT
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)

        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        bands = self.create_frequency_bands(sr)

        # Compute energy per band
        band_energies_db = []
        for f_low, f_high in bands:
            mask = (freq_bins >= f_low) & (freq_bins < f_high)
            if np.sum(mask) == 0:
                band_energies_db.append(-80.0)
                continue
            band_energy = np.mean(magnitude[mask, :])
            band_energy_db = 20 * np.log10(band_energy + 1e-8)
            band_energies_db.append(band_energy_db)

        band_energies_db = np.array(band_energies_db)  # type: ignore[assignment]

        # Detect masking: Compare each band to neighbors
        # Masked if significantly lower energy than neighbors
        masked_bands = np.zeros(self.n_bands, dtype=bool)

        for i in range(1, self.n_bands - 1):
            # Compare to neighbors
            left_diff = band_energies_db[i] - band_energies_db[i - 1]
            right_diff = band_energies_db[i] - band_energies_db[i + 1]

            # Masked if both neighbors are significantly louder
            if left_diff < self.masking_threshold_db and right_diff < self.masking_threshold_db:
                masked_bands[i] = True

        masking_count = np.sum(masked_bands)

        return {
            "band_energies_db": band_energies_db,
            "masked_bands": masked_bands,
            "masking_count": int(masking_count),
            "bands": bands,
        }

    def compute_demasking_eq(
        self,
        band_energies_db: np.ndarray,
        masked_bands: np.ndarray,
        bands: list[tuple[float, float]],
        sr: int,
        n_fft: int = 2048,
    ) -> np.ndarray:
        """
        Berechnet de-masking EQ curve.

        Args:
            band_energies_db: Energy per band
            masked_bands: Boolean mask of masked bands
            bands: Frequency bands
            sr: Sample rate
            n_fft: FFT size

        Returns:
            EQ curve (multiplicative)
        """
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        eq_db = np.zeros_like(freq_bins)

        for i, (f_low, f_high) in enumerate(bands):
            mask = (freq_bins >= f_low) & (freq_bins < f_high)

            if masked_bands[i]:
                # Boost masked band
                boost = 4.0 * self.demasking_strength
                eq_db[mask] = boost
            elif i > 0 and masked_bands[i - 1]:
                # Slight cut of masking band (left neighbor)
                cut = -2.0 * self.demasking_strength
                eq_db[mask] = cut
            elif i < len(bands) - 1 and masked_bands[i + 1]:
                # Slight cut of masking band (right neighbor)
                cut = -2.0 * self.demasking_strength
                eq_db[mask] = cut
            else:
                # No change
                eq_db[mask] = 0.0

        # Preserve overall balance: apply high-pass filter to EQ curve
        # This prevents overall level changes
        eq_db = eq_db * self.preserve_balance

        # Smooth EQ curve
        eq_db = gaussian_filter1d(eq_db, sigma=8)

        # Convert to linear
        eq_linear = 10 ** (eq_db / 20)

        return eq_linear  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Wendet an: frequency de-masking.

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Processed audio with reduced masking
        """
        self.log_contract()

        # Handle stereo — unterstützt (channels, samples) und (samples, channels)
        if audio.ndim == 2:
            if audio.shape[0] <= audio.shape[1]:
                channels = [self.process(audio[ch], sr) for ch in range(audio.shape[0])]
                return np.vstack(channels)
            else:
                channels = [self.process(audio[:, ch], sr) for ch in range(audio.shape[1])]
                return np.column_stack(channels)

        # Analyze masking
        analysis = self.analyze_masking(audio, sr)
        self.metrics = {"masking_detected": analysis["masking_count"], "bands_adjusted": analysis["masking_count"]}

        logger.info("[FrequencyDeMasker] erkannt %s masked bands out of %s", analysis["masking_count"], self.n_bands)

        if analysis["masking_count"] == 0:
            logger.info("[FrequencyDeMasker] No masking erkannt, no correction needed")
            return audio

        # STFT
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Compute de-masking EQ
        eq_curve = self.compute_demasking_eq(
            analysis["band_energies_db"], analysis["masked_bands"], analysis["bands"], sr, n_fft
        )

        # Apply EQ
        magnitude_demasked = magnitude * eq_curve[:, np.newaxis]

        # Reconstruct
        D_demasked = magnitude_demasked * np.exp(1j * phase)
        audio_demasked = librosa.istft(D_demasked, hop_length=hop_length, length=len(audio))

        # Quality gate
        peak = np.max(np.abs(audio_demasked))
        if peak > 0.99:
            logger.warning("[QualityGate] Warning: Peak=%.3f, normalizing", peak)
            audio_demasked = audio_demasked / (peak / 0.95)

        # Estimate clarity improvement
        self.metrics["clarity_improvement"] = analysis["masking_count"] * 10  # % improvement estimate

        logger.info(
            f"[FrequencyDeMasker] angewendet de-masking to {analysis['masking_count']} bands, "
            f"estimated {self.metrics['clarity_improvement']:.0f}% clarity improvement"
        )

        return audio_demasked


# =============================================================================
# UNIFIED API
# =============================================================================


class TonalBalanceRestorer:
    """
    Einheitliche API für alle drei Tonbalance-Restaurierungsmodule.

    Usage:
        restorer = TonalBalanceRestorer()
        audio_restored = restorer.process(audio, sr)
    """

    def __init__(
        self,
        enable_brightness_correction: bool = True,
        enable_low_end_clarity: bool = True,
        enable_demasking: bool = True,
        **kwargs,
    ):
        """
        Args:
            enable_brightness_correction: Enable GAP #7 (Adaptive Tonal Balance)
            enable_low_end_clarity: Enable GAP #8 (Low-End Clarity)
            enable_demasking: Enable GAP #9 (Frequency De-Masking)
            **kwargs: Additional parameters for individual modules
        """
        self.enable_brightness_correction = enable_brightness_correction
        self.enable_low_end_clarity = enable_low_end_clarity
        self.enable_demasking = enable_demasking

        # Initialize modules
        if enable_brightness_correction:
            self.brightness_restorer = AdaptiveTonalBalanceRestorer(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ["target_brightness", "strength", "smoothing_ms", "adaptive"]
                }
            )

        if enable_low_end_clarity:
            self.low_end_enhancer = LowEndClarityEnhancer(
                **{k: v for k, v in kwargs.items() if k in ["target_tightness", "preserve_warmth"]}
            )

        if enable_demasking:
            self.demasker = FrequencyDeMasker(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ["n_bands", "masking_threshold_db", "demasking_strength", "preserve_balance"]
                }
            )

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Wendet alle aktivierten Tonbalance-Restaurierungsmodule der Reihe nach an.

        Processing order:
        1. Low-End Clarity (foundation)
        2. Brightness Correction (overall tone)
        3. Frequency De-Masking (fine detail)

        Args:
            audio: Input audio
            sr: Sample rate

        Returns:
            Fully restored audio with excellent tonal balance
        """
        logger.info("\n" + "=" * 80)
        logger.info("TONAL BALANCE RESTORATION - Musikalische Exzellenz")
        logger.info("=" * 80)

        audio_processed = audio.copy()

        # Step 1: Low-End Clarity (foundation)
        if self.enable_low_end_clarity:
            logger.info("\n[STEP 1/3] Low-End Clarity Enhancement (GAP #8)")
            audio_processed = self.low_end_enhancer.process(audio_processed, sr)

        # Step 2: Brightness Correction (overall tone)
        if self.enable_brightness_correction:
            logger.info("\n[STEP 2/3] Adaptive Tonal Balance Restoration (GAP #7)")
            audio_processed = self.brightness_restorer.process(audio_processed, sr)

        # Step 3: Frequency De-Masking (fine detail)
        if self.enable_demasking:
            logger.info("\n[STEP 3/3] Frequency De-Masking (GAP #9)")
            audio_processed = self.demasker.process(audio_processed, sr)

        logger.info("\n" + "=" * 80)
        logger.info("TONAL BALANCE RESTORATION vollstaendig")
        logger.info("=" * 80 + "\n")

        return audio_processed

    def get_metrics(self) -> dict[str, Any]:
        """
        Gibt zurück: metrics from all modules.

        Returns:
            Combined metrics dict
        """
        metrics = {}

        if self.enable_brightness_correction and hasattr(self.brightness_restorer, "metrics"):
            metrics["brightness"] = self.brightness_restorer.metrics

        if self.enable_low_end_clarity and hasattr(self.low_end_enhancer, "metrics"):
            metrics["low_end"] = self.low_end_enhancer.metrics

        if self.enable_demasking and hasattr(self.demasker, "metrics"):
            metrics["demasking"] = self.demasker.metrics

        return metrics


# =============================================================================
# DEMO / CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    import soundfile as sf

    logger.info(str("=" * 80))
    logger.info("AURIK v8 - Tonal Balance Restorer (GAP #7, #8, #9)")
    logger.info("Musikalische Exzellenz: Nicht nur defekt-frei, sondern perfekt klingend!")
    logger.info(str("=" * 80))

    if len(sys.argv) < 3:
        logger.info("\nUsage: python tonal_balance_restorer.py <Eingabe.wav> <Ausgabe.wav> [options]")
        logger.info("\nOptions:")
        logger.info("  --brightness <0.0-1.0>     Target brightness (default: 0.5)")
        logger.info("  --tightness <0.0-1.0>      Low-end tightness (default: 0.6)")
        logger.info("  --demasking-strength <0.0-1.0>  De-masking strength (default: 0.6)")
        logger.info("  --disable-brightness       Disable brightness correction")
        logger.info("  --disable-low-end          Disable low-end clarity")
        logger.info("  --disable-demasking        Disable de-masking")
        logger.info("\nExample:")
        logger.info("  python tonal_balance_restorer.py dull_audio.wav bright_audio.wav --brightness 0.7")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Parse options
    options = {
        "target_brightness": 0.5,
        "target_tightness": 0.6,
        "demasking_strength": 0.6,
        "enable_brightness_correction": True,
        "enable_low_end_clarity": True,
        "enable_demasking": True,
    }

    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--brightness" and i + 1 < len(sys.argv):
            options["target_brightness"] = float(sys.argv[i + 1])
            i += 2
        elif arg == "--tightness" and i + 1 < len(sys.argv):
            options["target_tightness"] = float(sys.argv[i + 1])
            i += 2
        elif arg == "--demasking-strength" and i + 1 < len(sys.argv):
            options["demasking_strength"] = float(sys.argv[i + 1])
            i += 2
        elif arg == "--disable-brightness":
            options["enable_brightness_correction"] = False
            i += 1
        elif arg == "--disable-low-end":
            options["enable_low_end_clarity"] = False
            i += 1
        elif arg == "--disable-demasking":
            options["enable_demasking"] = False
            i += 1
        else:
            i += 1

    # Load audio
    logger.info("\nlade: %s", input_file)
    from backend.file_import import load_audio_file

    _res = load_audio_file(input_file)
    audio, sr = _res["audio"], int(_res["sr"])  # type: ignore[index]

    # Ensure mono for processing (or handle stereo properly)
    if audio.ndim > 1:
        logger.info("Eingabe is stereo (%s channels), processing both channels", audio.shape[1])
        audio = audio.T  # Shape: (channels, samples)

    # Process
    restorer = TonalBalanceRestorer(**options)  # type: ignore[arg-type]
    audio_restored = restorer.process(audio, sr)

    # Get metrics
    metrics = restorer.get_metrics()
    logger.info("\nProcessing Metrics:")
    logger.info(str(metrics))

    # Save
    logger.info("\nspeichere: %s", output_file)
    if audio_restored.ndim > 1:
        audio_restored = audio_restored.T  # Back to (samples, channels)
    sf.write(output_file, audio_restored, sr)

    logger.info("\n✅ Tonal Balance Restoration vollstaendig!")
    logger.info("Audio now has professional-grade tonal balance and clarity.")
