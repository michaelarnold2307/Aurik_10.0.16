"""
forensics/defect_quantifier.py
Defect Quantification Engine
=============================

Quantifiziert Audio-Defekte mit präzisen Messwerten:
- Clicks: Count, Average Severity, Peak Amplitude
- Hum: Frequency, Amplitude (dB), Harmonics Strength
- Distortion: THD%, Clipping%, IMD
- Dropout: Count, Duration, Depth (dB)
- Noise Burst: Count, Peak Level, Spectral Shape

Im Gegensatz zum MLDefectDetector (Binary: Detected/Not Detected)
liefert dieser Quantizer präzise Messwerte für Parameter-Inferenz.

Author: AURIK Team
Date: 11. Februar 2026
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def _as_real_array(audio: np.ndarray) -> np.ndarray:
    """Gibt a contiguous real-valued ndarray for signal processing calls zurück."""
    return np.asarray(audio, dtype=np.float64).reshape(-1)  # type: ignore[no-any-return]


def _analytic_envelope(audio: np.ndarray) -> np.ndarray:
    """Gibt absolute Hilbert envelope as ndarray with explicit typing zurück."""
    analytic = np.asarray(signal.hilbert(_as_real_array(audio)), dtype=np.complex128)
    return np.abs(analytic)  # type: ignore[no-any-return]


def _sosfilt_array(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Gibt only filtered audio ndarray from scipy.signal.sosfilt zurück."""
    filtered = signal.sosfilt(sos, _as_real_array(audio))
    if isinstance(filtered, tuple):
        return np.asarray(filtered[0], dtype=np.float64)  # type: ignore[no-any-return]
    return np.asarray(filtered, dtype=np.float64)  # type: ignore[no-any-return]


def _rfft_magnitude(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Gibt typed rFFT magnitudes and frequencies zurück."""
    audio_array = _as_real_array(audio)
    fft_vals = np.asarray(np.fft.rfft(audio_array), dtype=np.complex128)
    freqs = np.asarray(np.fft.rfftfreq(len(audio_array), 1.0 / float(sample_rate)), dtype=np.float64)
    return np.abs(fft_vals), freqs


@dataclass
class ClickMetrics:
    """Quantified click/pop metrics."""

    count: int = 0  # Number of clicks detected
    density_per_sec: float = 0.0  # Clicks per second
    avg_amplitude_db: float = -100.0  # Average click amplitude
    max_amplitude_db: float = -100.0  # Strongest click
    avg_duration_ms: float = 0.0  # Average click duration
    severity: str = "NONE"  # NONE/LOW/MEDIUM/HIGH/EXTREME

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "density_per_sec": self.density_per_sec,
            "avg_amplitude_db": self.avg_amplitude_db,
            "max_amplitude_db": self.max_amplitude_db,
            "avg_duration_ms": self.avg_duration_ms,
            "severity": self.severity,
        }


@dataclass
class HumMetrics:
    """Quantified hum metrics."""

    present: bool = False
    fundamental_freq_hz: float = 0.0  # 50 or 60 Hz
    fundamental_level_db: float = -100.0  # Level relative to signal
    harmonics_detected: list[int] = field(default_factory=list)  # [2, 3, 4, ...]
    harmonics_level_db: float = -100.0  # Average harmonics level
    total_hum_level_db: float = -100.0  # Combined level
    modulation_percent: float = 0.0  # Amplitude modulation
    severity: str = "NONE"  # NONE/LOW/MEDIUM/HIGH/EXTREME

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "fundamental_freq_hz": self.fundamental_freq_hz,
            "fundamental_level_db": self.fundamental_level_db,
            "harmonics_detected": self.harmonics_detected,
            "harmonics_level_db": self.harmonics_level_db,
            "total_hum_level_db": self.total_hum_level_db,
            "modulation_percent": self.modulation_percent,
            "severity": self.severity,
        }


@dataclass
class DistortionMetrics:
    """Quantified distortion metrics."""

    thd_percent: float = 0.0  # Total Harmonic Distortion
    thd_plus_noise_percent: float = 0.0  # THD+N
    clipping_percent: float = 0.0  # % of samples clipped
    peak_clipping_level: float = 0.0  # Highest clip level
    imd_percent: float = 0.0  # Intermodulation Distortion
    harmonic_spread: float = 0.0  # Distribution of harmonics
    severity: str = "NONE"  # NONE/LOW/MEDIUM/HIGH/EXTREME

    def to_dict(self) -> dict:
        return {
            "thd_percent": self.thd_percent,
            "thd_plus_noise_percent": self.thd_plus_noise_percent,
            "clipping_percent": self.clipping_percent,
            "peak_clipping_level": self.peak_clipping_level,
            "imd_percent": self.imd_percent,
            "harmonic_spread": self.harmonic_spread,
            "severity": self.severity,
        }


@dataclass
class DropoutMetrics:
    """Quantified dropout metrics."""

    count: int = 0  # Number of dropouts
    total_duration_ms: float = 0.0  # Total silence/dropout time
    avg_duration_ms: float = 0.0  # Average dropout duration
    max_depth_db: float = 0.0  # Deepest amplitude drop
    discontinuities: int = 0  # Sudden amplitude jumps
    severity: str = "NONE"  # NONE/LOW/MEDIUM/HIGH/EXTREME

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "max_depth_db": self.max_depth_db,
            "discontinuities": self.discontinuities,
            "severity": self.severity,
        }


@dataclass
class NoiseBurstMetrics:
    """Quantified noise burst metrics."""

    count: int = 0  # Number of bursts
    avg_level_db: float = -100.0  # Average burst level
    max_level_db: float = -100.0  # Loudest burst
    avg_duration_ms: float = 0.0  # Average burst duration
    spectral_content: str = "UNKNOWN"  # LOW_FREQ/MID_FREQ/HIGH_FREQ/BROADBAND
    severity: str = "NONE"  # NONE/LOW/MEDIUM/HIGH/EXTREME

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "avg_level_db": self.avg_level_db,
            "max_level_db": self.max_level_db,
            "avg_duration_ms": self.avg_duration_ms,
            "spectral_content": self.spectral_content,
            "severity": self.severity,
        }


@dataclass
class DefectQuantification:
    """Complete defect quantification."""

    clicks: ClickMetrics = field(default_factory=ClickMetrics)
    hum: HumMetrics = field(default_factory=HumMetrics)
    distortion: DistortionMetrics = field(default_factory=DistortionMetrics)
    dropout: DropoutMetrics = field(default_factory=DropoutMetrics)
    noise_burst: NoiseBurstMetrics = field(default_factory=NoiseBurstMetrics)

    # Overall metrics
    overall_quality: float = 1.0  # 0-1 (1 = perfect)
    restoration_required: bool = False
    priority_defects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clicks": self.clicks.to_dict(),
            "hum": self.hum.to_dict(),
            "distortion": self.distortion.to_dict(),
            "dropout": self.dropout.to_dict(),
            "noise_burst": self.noise_burst.to_dict(),
            "overall_quality": self.overall_quality,
            "restoration_required": self.restoration_required,
            "priority_defects": self.priority_defects,
        }


class DefectQuantifier:
    """
    Defect Quantifier - Präzise Messung von Audio-Defekten.

    Quantifiziert 5 Defekt-Typen mit detaillierten Messwerten:
    1. Clicks/Pops: Transiente Störungen
    2. Hum: 50/60Hz Brummen
    3. Distortion: Clipping, THD, IMD
    4. Dropout: Silence, Amplitude Drops
    5. Noise Bursts: Transiente Rauschen

    Verwendet im Adaptive Chain Builder für Parameter-Inferenz.
    """

    VERSION = "1.0.0"

    def __init__(self, sample_rate: int = 48000):
        """
        Initialisiert defect quantifier.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def quantify(self, audio: np.ndarray) -> DefectQuantification:
        """
        Quantify all defects in audio signal.

        Args:
            audio: Audio signal (mono or stereo)

        Returns:
            DefectQuantification with all metrics
        """
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Quantify each defect type
        clicks = self._quantify_clicks(audio)
        hum = self._quantify_hum(audio)
        distortion = self._quantify_distortion(audio)
        dropout = self._quantify_dropout(audio)
        noise_burst = self._quantify_noise_burst(audio)

        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(clicks, hum, distortion, dropout, noise_burst)

        # Determine restoration priority
        priority_defects = self._determine_priority_defects(clicks, hum, distortion, dropout, noise_burst)

        restoration_required = len(priority_defects) > 0

        return DefectQuantification(
            clicks=clicks,
            hum=hum,
            distortion=distortion,
            dropout=dropout,
            noise_burst=noise_burst,
            overall_quality=overall_quality,
            restoration_required=restoration_required,
            priority_defects=priority_defects,
        )

    def _quantify_clicks(self, audio: np.ndarray) -> ClickMetrics:
        """Quantify clicks and pops."""
        # Detect transients using envelope following
        _analytic_envelope(audio)

        # High-pass filter for click detection (> 5kHz)
        sos = signal.butter(4, 5000, "hp", fs=self.sample_rate, output="sos")
        audio_hp = _sosfilt_array(sos, audio)

        # Detect sharp peaks
        threshold = np.std(audio_hp) * 5  # 5 sigma threshold
        peaks, properties = signal.find_peaks(
            np.abs(audio_hp),
            height=threshold,
            distance=int(0.01 * self.sample_rate),  # Min 10ms between clicks
        )

        count = len(peaks)
        density_per_sec = count / (len(audio) / self.sample_rate)

        if count > 0:
            amplitudes_db = 20 * np.log10(properties["peak_heights"] / np.max(np.abs(audio)) + 1e-10)
            avg_amplitude_db = float(np.mean(amplitudes_db))
            max_amplitude_db = float(np.max(amplitudes_db))

            # Estimate duration (time above 50% peak)
            durations = []
            for peak in peaks:
                start = max(0, peak - 100)
                end = min(len(audio_hp), peak + 100)
                segment = audio_hp[start:end]
                above_threshold = np.where(np.abs(segment) > properties["peak_heights"][peaks == peak][0] * 0.5)[0]
                if len(above_threshold) > 0:
                    durations.append((above_threshold[-1] - above_threshold[0]) / self.sample_rate * 1000)

            avg_duration_ms = float(np.mean(durations)) if durations else 0.0
        else:
            avg_amplitude_db = -100.0
            max_amplitude_db = -100.0
            avg_duration_ms = 0.0

        # Severity classification
        if count == 0:
            severity = "NONE"
        elif density_per_sec < 0.5:
            severity = "LOW"
        elif density_per_sec < 2.0:
            severity = "MEDIUM"
        elif density_per_sec < 5.0:
            severity = "HIGH"
        else:
            severity = "EXTREME"

        return ClickMetrics(
            count=count,
            density_per_sec=float(density_per_sec),
            avg_amplitude_db=avg_amplitude_db,
            max_amplitude_db=max_amplitude_db,
            avg_duration_ms=avg_duration_ms,
            severity=severity,
        )

    def _quantify_hum(self, audio: np.ndarray) -> HumMetrics:
        """Quantify hum (50/60Hz)."""
        # FFT analysis
        magnitudes, freqs = _rfft_magnitude(audio, self.sample_rate)
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)

        # Check 50Hz and 60Hz
        def get_level_at_freq(target_freq: float, bandwidth: float = 2.0) -> float:
            idx = np.where((freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth))[0]
            if len(idx) > 0:
                return float(np.max(magnitudes_db[idx]))
            return -100.0

        level_50hz = get_level_at_freq(50.0)
        level_60hz = get_level_at_freq(60.0)

        # Determine fundamental
        if level_50hz > level_60hz and level_50hz > -60:
            fundamental_freq = 50.0
            fundamental_level = level_50hz
        elif level_60hz > -60:
            fundamental_freq = 60.0
            fundamental_level = level_60hz
        else:
            # No hum detected
            return HumMetrics(present=False)

        # Check harmonics (2x, 3x, 4x, 5x)
        harmonics_detected = []
        harmonics_levels = []
        for harmonic in [2, 3, 4, 5]:
            harmonic_freq = fundamental_freq * harmonic
            level = get_level_at_freq(harmonic_freq)
            if level > -70:
                harmonics_detected.append(harmonic)
                harmonics_levels.append(level)

        harmonics_level_db = float(np.mean(harmonics_levels)) if harmonics_levels else -100.0

        # Total hum level (fundamental + harmonics)
        total_hum_level_db = 10 * np.log10(
            10 ** (fundamental_level / 10) + sum(10 ** (h / 10) for h in harmonics_levels)
        )

        # Modulation detection (amplitude variation)
        # Low-pass filter to get envelope
        sos = signal.butter(2, 20, "lp", fs=self.sample_rate, output="sos")
        envelope = _analytic_envelope(_sosfilt_array(sos, audio))
        modulation_percent = float((np.std(envelope) / np.mean(envelope)) * 100)

        # Severity
        if total_hum_level_db < -60:
            severity = "LOW"
        elif total_hum_level_db < -45:
            severity = "MEDIUM"
        elif total_hum_level_db < -30:
            severity = "HIGH"
        else:
            severity = "EXTREME"

        return HumMetrics(
            present=True,
            fundamental_freq_hz=fundamental_freq,
            fundamental_level_db=float(fundamental_level),
            harmonics_detected=harmonics_detected,
            harmonics_level_db=harmonics_level_db,
            total_hum_level_db=float(total_hum_level_db),
            modulation_percent=modulation_percent,
            severity=severity,
        )

    def _quantify_distortion(self, audio: np.ndarray) -> DistortionMetrics:
        """Quantify distortion (clipping, THD, IMD)."""
        # Clipping detection
        threshold = 0.99  # 99% of full scale
        clipped = np.abs(audio) > threshold
        clipping_percent = float(np.sum(clipped) / len(audio) * 100)
        peak_clipping_level = float(np.max(np.abs(audio[clipped]))) if np.any(clipped) else 0.0

        # THD estimation using FFT
        magnitudes, freqs = _rfft_magnitude(audio, self.sample_rate)

        # Find fundamental frequency (strongest component in 50-1000Hz)
        low_idx = np.searchsorted(freqs, 50)
        high_idx = np.searchsorted(freqs, 1000)
        if high_idx > low_idx:
            fundamental_idx = low_idx + np.argmax(magnitudes[low_idx:high_idx])
            fundamental_power = magnitudes[fundamental_idx] ** 2

            # Sum harmonic powers (2x, 3x, 4x, 5x fundamental)
            harmonic_powers = []
            for h in [2, 3, 4, 5]:
                harmonic_freq = freqs[fundamental_idx] * h
                harmonic_idx = np.searchsorted(freqs, harmonic_freq)
                if harmonic_idx < len(magnitudes):
                    harmonic_powers.append(magnitudes[harmonic_idx] ** 2)

            if fundamental_power > 0 and harmonic_powers:
                thd_percent = float(np.sqrt(sum(harmonic_powers)) / np.sqrt(fundamental_power) * 100)
                harmonic_spread = float(np.std(harmonic_powers) / np.mean(harmonic_powers)) if harmonic_powers else 0.0
            else:
                thd_percent = 0.0
                harmonic_spread = 0.0
        else:
            thd_percent = 0.0
            harmonic_spread = 0.0

        # THD+N (simplified: use RMS of full spectrum minus fundamental)
        signal_power: float = float(np.sum(magnitudes**2))
        thd_plus_noise_percent = (
            float(np.sqrt((signal_power - fundamental_power) / signal_power) * 100) if signal_power > 0 else 0.0
        )

        # IMD (simplified estimate from spectral irregularity)
        imd_percent = thd_percent * 0.5  # Rough estimate

        # Severity
        if clipping_percent == 0 and thd_percent < 0.1:
            severity = "NONE"
        elif clipping_percent < 0.01 and thd_percent < 1.0:
            severity = "LOW"
        elif clipping_percent < 0.1 and thd_percent < 3.0:
            severity = "MEDIUM"
        elif clipping_percent < 1.0 and thd_percent < 10.0:
            severity = "HIGH"
        else:
            severity = "EXTREME"

        return DistortionMetrics(
            thd_percent=thd_percent,
            thd_plus_noise_percent=thd_plus_noise_percent,
            clipping_percent=clipping_percent,
            peak_clipping_level=peak_clipping_level,
            imd_percent=imd_percent,
            harmonic_spread=harmonic_spread,
            severity=severity,
        )

    def _quantify_dropout(self, audio: np.ndarray) -> DropoutMetrics:
        """Quantify dropouts (silence, amplitude drops)."""
        # Envelope detection
        envelope = _analytic_envelope(audio)

        # Threshold for dropout (< -40dB relative to RMS)
        rms = np.sqrt(np.mean(audio**2))
        dropout_threshold = rms * 0.01  # -40dB

        # Find dropout regions
        is_dropout = envelope < dropout_threshold

        # Find contiguous dropout regions
        dropout_regions = []
        in_dropout = False
        start_idx = 0

        for i, drop in enumerate(is_dropout):
            if drop and not in_dropout:
                start_idx = i
                in_dropout = True
            elif not drop and in_dropout:
                dropout_regions.append((start_idx, i))
                in_dropout = False

        if in_dropout:
            dropout_regions.append((start_idx, len(is_dropout)))

        count = len(dropout_regions)

        if count > 0:
            durations_ms = [(end - start) / self.sample_rate * 1000 for start, end in dropout_regions]
            total_duration_ms = float(sum(durations_ms))
            avg_duration_ms = float(np.mean(durations_ms))

            # Measure dropout depth
            depths_db = []
            for start, end in dropout_regions:
                if end > start:
                    dropout_level = np.mean(envelope[start:end])
                    depth_db = 20 * np.log10(dropout_level / rms + 1e-10)
                    depths_db.append(depth_db)

            max_depth_db = float(min(depths_db)) if depths_db else 0.0
        else:
            total_duration_ms = 0.0
            avg_duration_ms = 0.0
            max_depth_db = 0.0

        # Detect discontinuities (sudden amplitude jumps)
        diff = np.abs(np.diff(envelope))
        discontinuities = int(np.sum(diff > (np.std(diff) * 10)))

        # Severity
        if count == 0 and discontinuities == 0:
            severity = "NONE"
        elif count < 3 and total_duration_ms < 100:
            severity = "LOW"
        elif count < 10 and total_duration_ms < 500:
            severity = "MEDIUM"
        elif count < 30 and total_duration_ms < 2000:
            severity = "HIGH"
        else:
            severity = "EXTREME"

        return DropoutMetrics(
            count=count,
            total_duration_ms=total_duration_ms,
            avg_duration_ms=avg_duration_ms,
            max_depth_db=max_depth_db,
            discontinuities=discontinuities,
            severity=severity,
        )

    def _quantify_noise_burst(self, audio: np.ndarray) -> NoiseBurstMetrics:
        """Quantify noise bursts."""
        # High-pass filter for burst detection
        sos = signal.butter(4, 2000, "hp", fs=self.sample_rate, output="sos")
        audio_hp = _sosfilt_array(sos, audio)

        # Envelope
        envelope = _analytic_envelope(audio_hp)

        # Detect bursts (sudden energy increases)
        threshold = np.mean(envelope) + 3 * np.std(envelope)
        peaks, properties = signal.find_peaks(
            envelope,
            height=threshold,
            distance=int(0.05 * self.sample_rate),  # Min 50ms between bursts
        )

        count = len(peaks)

        if count > 0:
            levels_db = 20 * np.log10(properties["peak_heights"] / np.max(np.abs(audio)) + 1e-10)
            avg_level_db = float(np.mean(levels_db))
            max_level_db = float(np.max(levels_db))

            # Estimate duration
            durations = []
            for peak in peaks:
                start = max(0, peak - 500)
                end = min(len(envelope), peak + 500)
                segment = envelope[start:end]
                above_threshold = np.where(segment > threshold)[0]
                if len(above_threshold) > 0:
                    durations.append((above_threshold[-1] - above_threshold[0]) / self.sample_rate * 1000)

            avg_duration_ms = float(np.mean(durations)) if durations else 0.0

            # Spectral content analysis
            # Average FFT of burst regions
            magnitudes, freqs = _rfft_magnitude(audio_hp, self.sample_rate)

            low_energy: float = float(np.sum(magnitudes[freqs < 2000]))
            mid_energy: float = float(np.sum(magnitudes[(freqs >= 2000) & (freqs < 8000)]))
            high_energy: float = float(np.sum(magnitudes[freqs >= 8000]))
            total = low_energy + mid_energy + high_energy

            if total > 0:
                if low_energy / total > 0.5:
                    spectral_content = "LOW_FREQ"
                elif high_energy / total > 0.5:
                    spectral_content = "HIGH_FREQ"
                elif mid_energy / total > 0.5:
                    spectral_content = "MID_FREQ"
                else:
                    spectral_content = "BROADBAND"
            else:
                spectral_content = "UNKNOWN"
        else:
            avg_level_db = -100.0
            max_level_db = -100.0
            avg_duration_ms = 0.0
            spectral_content = "UNKNOWN"

        # Severity
        if count == 0:
            severity = "NONE"
        elif count < 5:
            severity = "LOW"
        elif count < 15:
            severity = "MEDIUM"
        elif count < 30:
            severity = "HIGH"
        else:
            severity = "EXTREME"

        return NoiseBurstMetrics(
            count=count,
            avg_level_db=avg_level_db,
            max_level_db=max_level_db,
            avg_duration_ms=avg_duration_ms,
            spectral_content=spectral_content,
            severity=severity,
        )

    def _calculate_overall_quality(
        self,
        clicks: ClickMetrics,
        hum: HumMetrics,
        distortion: DistortionMetrics,
        dropout: DropoutMetrics,
        noise_burst: NoiseBurstMetrics,
    ) -> float:
        """
        Calculate overall quality score (0-1).

        1.0 = Perfect, no defects
        0.0 = Severe defects, heavy restoration needed
        """
        # Severity scores (0-1, 1 = no defect)
        severity_map = {"NONE": 1.0, "LOW": 0.85, "MEDIUM": 0.65, "HIGH": 0.35, "EXTREME": 0.1}

        scores = [
            severity_map.get(clicks.severity, 0.5),
            severity_map.get(hum.severity, 0.5),
            severity_map.get(distortion.severity, 0.5),
            severity_map.get(dropout.severity, 0.5),
            severity_map.get(noise_burst.severity, 0.5),
        ]

        # Weighted average (distortion and dropout are more critical)
        weights = [1.0, 1.0, 1.5, 1.5, 1.0]
        overall = float(np.average(scores, weights=weights))

        return overall

    def _determine_priority_defects(
        self,
        clicks: ClickMetrics,
        hum: HumMetrics,
        distortion: DistortionMetrics,
        dropout: DropoutMetrics,
        noise_burst: NoiseBurstMetrics,
    ) -> list[str]:
        """Bestimmt which defects need priority restoration."""
        priority = []

        if clicks.severity in ["HIGH", "EXTREME"]:
            priority.append("clicks")
        if hum.severity in ["HIGH", "EXTREME"]:
            priority.append("hum")
        if distortion.severity in ["MEDIUM", "HIGH", "EXTREME"]:
            priority.append("distortion")  # Distortion has lower threshold
        if dropout.severity in ["HIGH", "EXTREME"]:
            priority.append("dropout")
        if noise_burst.severity in ["HIGH", "EXTREME"]:
            priority.append("noise_burst")

        return priority


if __name__ == "__main__":
    # Demo usage
    quantifier = DefectQuantifier(sample_rate=48000)

    # Generate test signal with defects
    duration = 2.0
    sr = 48000
    t = np.linspace(0, duration, int(sr * duration))

    # Clean sine wave + some distortion
    audio = np.sin(2 * np.pi * 440 * t)
    audio += 0.1 * np.sin(2 * np.pi * 50 * t)  # Add 50Hz hum
    audio = np.clip(audio * 1.2, -1, 1)  # Add clipping

    # Add some clicks
    for i in range(10):
        pos = np.random.randint(0, len(audio) - 100)
        audio[pos : pos + 10] += np.random.randn(10) * 0.5

    # Quantify
    result = quantifier.quantify(audio)

    logger.debug("Defect Quantification Results:")
    logger.debug("  Clicks: %s erkannt (%s)", result.clicks.count, result.clicks.severity)
    logger.debug(
        f"  Hum: {result.hum.fundamental_freq_hz}Hz at {result.hum.fundamental_level_db:.1f}dB ({result.hum.severity})"
    )
    logger.debug(
        f"  Distortion: THD={result.distortion.thd_percent:.2f}%, Clipping={result.distortion.clipping_percent:.2f}% ({result.distortion.severity})"
    )
    logger.debug("  Dropout: %s erkannt (%s)", result.dropout.count, result.dropout.severity)
    logger.debug("  Noise Bursts: %s erkannt (%s)", result.noise_burst.count, result.noise_burst.severity)
    logger.debug("\nOverall Quality: %.2f", result.overall_quality)
    logger.debug("Priority Defects: %s", ", ".join(result.priority_defects) if result.priority_defects else "None")
