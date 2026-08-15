"""
Comprehensive Audio Quality Metrics for Aurik 10.0.0
==================================================

Vollständiges System für psychoakustische, musikalische und emotionale Metriken.

Phase: Entwicklung psychoakustischer, musikalischer und emotionaler Metriken
Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
Version: 10.0.0

Metrik-Kategorien:
1. PSYCHOAKUSTISCHE METRIKEN: SNR, THD, Dynamikbereich, LUFS, Maskierung, etc.
2. MUSIKALISCHE METRIKEN: Tonalität, Harmonie, Rhythmus, Artikulation, Timbre
3. EMOTIONALE METRIKEN: Valenz, Arousal, Spannung, Energie, etc.

Keine Dummys/Mocks - nur reale, wissenschaftlich fundierte Implementierungen.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import fft, signal
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type-stable wrappers for scipy functions whose stubs use Dispatchable types
# that Pylance cannot resolve via overload matching (scipy stub limitation).
# ---------------------------------------------------------------------------


def _rfft(x: np.ndarray, n: int | None = None) -> np.ndarray:
    """scipy.fft.rfft with explicit ndarray return type for Pylance."""
    return np.asarray(fft.rfft(x, n=n))  # type: ignore[no-any-return,arg-type]


def _irfft(x: np.ndarray, n: int | None = None) -> np.ndarray:
    """scipy.fft.irfft with explicit ndarray return type for Pylance."""
    return np.asarray(fft.irfft(x, n))  # type: ignore[no-any-return,arg-type]


def _hilbert(x: np.ndarray) -> np.ndarray:
    """scipy.signal.hilbert with explicit ndarray return type for Pylance."""
    return np.asarray(signal.hilbert(x))  # type: ignore[no-any-return,arg-type]


# Import existing metrics modules
try:
    from backend.core.enhanced_metrics import EnhancedMetrics

    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    ENHANCED_METRICS_AVAILABLE = False
    logger.warning("EnhancedMetrics not verfuegbar")

try:
    from dsp.professional_meters import LUFSMeter

    PROFESSIONAL_METERS_AVAILABLE = True
except ImportError:
    PROFESSIONAL_METERS_AVAILABLE = False
    logger.warning("Professional meters not verfuegbar")


# ============================================================
# DATA STRUCTURES
# ============================================================


class PsychoAcousticMetrics:
    """Psychoakustische Metriken (objektiv messbar, perzeptuell relevant).

    §5/5: Dient sowohl als Datencontainer als auch als Calculator.
    Methoden wie calculate_roughness() werden an den internen
    ComprehensiveMetricsCalculator delegiert.
    """

    # Signal Quality
    snr_db: float  # Signal-to-Noise Ratio
    thd_percent: float  # Total Harmonic Distortion
    sinad_db: float  # Signal-to-Noise-and-Distortion

    # Loudness & Dynamics
    integrated_lufs: float  # ITU-R BS.1770-4 compliant
    loudness_range_lu: float  # Dynamic range (LU)
    true_peak_dbtp: float  # True Peak level
    crest_factor_db: float  # Peak-to-RMS ratio

    # Frequency Response
    frequency_response_flatness: float  # 0-1 (1=flat)
    spectral_centroid_hz: float  # "Brightness"
    spectral_rolloff_hz: float  # High-freq cutoff
    spectral_flux: float  # Spectral change rate

    # Masking & Perception
    perceptual_sharpness: float  # Sharpness sensation (acum)
    perceptual_roughness: float  # Roughness sensation
    tonality: float  # Tonal vs. noise content (0-1)

    # Artifacts
    pre_echo_score: float  # Pre-echo artifacts (0-1, 1=clean)
    click_detection: int  # Number of detected clicks
    clipping_percent: float  # % of clipped samples

    def __init__(self, sample_rate: int = 48000, **kwargs: Any) -> None:
        self.sample_rate = sample_rate
        # Alle übergebenen Felder als Attribute setzen (Rückwärtskompatibilität)
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Lazy-init: Calculator wird erst bei Bedarf erstellt
        self._calculator: Any = None

    def _get_calculator(self) -> Any:
        if self._calculator is None:
            self._calculator = ComprehensiveMetricsCalculator(sample_rate=self.sample_rate)
        return self._calculator

    def calculate_roughness(self, audio: np.ndarray) -> float:
        return float(self._get_calculator()._compute_roughness(audio))

    def calculate_sharpness(self, audio: np.ndarray) -> float:
        return float(self._get_calculator()._compute_sharpness(audio))

    def calculate_spectral_flatness(self, audio: np.ndarray) -> float:
        flatness, _centroid, _rolloff, _flux = self._get_calculator()._compute_spectral_features(audio)
        return float(flatness)

    def calculate_harmonic_coherence(self, audio: np.ndarray) -> float:
        return float(self._get_calculator()._compute_tonality(audio))

    def calculate_naturalness_score(self, audio: np.ndarray, reference: np.ndarray | None = None) -> dict[str, float]:
        result = self._get_calculator().compute_all(audio, reference)
        _d = result.to_dict()
        # Nur endliche Zahlen zurückgeben — to_dict() kann verschachtelte
        # Objekt-Referenzen enthalten, die math.isfinite() crashen lassen.
        return {str(k): float(v) for k, v in _d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def calculate_temporal_smoothness(self, audio: np.ndarray) -> float:
        """Zeitliche Glätte: 1 − normalisierte Streuung der Frame-RMS (0–1)."""
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        n = len(audio)
        frame = max(64, self.sample_rate // 50)
        n_frames = n // frame
        if n_frames < 2:
            return 1.0
        rms = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1) + 1e-12)
        spread = float(np.std(rms) / (np.mean(rms) + 1e-12))
        return float(np.clip(1.0 - min(spread, 1.0), 0.0, 1.0))

    def calculate_noise_floor_consistency(self, audio: np.ndarray) -> float:
        """Konsistenz des Rauschbodens über Segmente (0–1, 1 = konstant)."""
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        n = len(audio)
        frame = max(64, self.sample_rate // 50)
        n_frames = n // frame
        if n_frames < 2:
            return 1.0
        floors = np.percentile(np.abs(audio[: n_frames * frame].reshape(n_frames, frame)), 5, axis=1)
        spread = float(np.std(floors) / (np.mean(floors) + 1e-12))
        return float(np.clip(1.0 - min(spread, 1.0), 0.0, 1.0))

    def calculate_roughness_zwicker_detailed(self, audio: np.ndarray) -> float:
        """Detaillierte Zwicker-Rauhigkeit — delegiert an den Roughness-Pfad."""
        return self.calculate_roughness(audio)


@dataclass
class MusicalMetrics:
    """Musikalische Metriken (musikalisch relevante Eigenschaften)."""

    # Harmonic Content
    harmonic_clarity: float  # Harmonic vs. inharmonic (0-1)
    harmonic_to_noise_ratio_db: float  # HNR
    fundamental_stability: float  # Pitch stability (0-1)

    # Tonal Properties
    key_confidence: float  # Confidence in detected key (0-1)
    detected_key: str  # e.g., "C major"
    consonance: float  # Harmonic consonance (0-1)

    # Rhythm & Timing
    tempo_bpm: float  # Detected tempo
    tempo_stability: float  # Tempo consistency (0-1)
    rhythmic_regularity: float  # Beat regularity (0-1)

    # Articulation & Dynamics
    attack_sharpness: float  # Transient sharpness (0-1)
    decay_smoothness: float  # Decay envelope smoothness (0-1)
    dynamic_contrast: float  # Micro-dynamics (0-1)

    # Timbre & Texture
    spectral_complexity: float  # Harmonic richness (0-1)
    spectral_balance: float  # Bass/mid/treble balance (0-1)
    warmth: float  # Low-mid richness (0-1)
    brightness: float  # High-freq presence (0-1)
    fullness: float  # Mid-range presence (0-1)


@dataclass
class EmotionalMetrics:
    """Emotionale Metriken (affektive Eigenschaften nach Russell's Circumplex Model)."""

    # Core Dimensions (Russell's Circumplex Model)
    valence: float  # Pleasantness (-1=negative, +1=positive)
    arousal: float  # Energy/Activation (-1=calm, +1=energetic)

    # Energy & Intensity
    energy: float  # Overall energy (0-1)
    intensity: float  # Emotional intensity (0-1)
    tension: float  # Harmonic/rhythmic tension (0-1)

    # Emotional Categories (Geneva Emotional Music Scale)
    power: float  # Feeling of power/confidence (0-1)
    joyful_activation: float  # Happy, cheerful (0-1)
    nostalgia: float  # Nostalgic, sentimental (0-1)
    sadness: float  # Sad, melancholic (0-1)
    peacefulness: float  # Calm, peaceful (0-1)
    transcendence: float  # Spiritual, transcendent (0-1)

    # Perceived Affect
    perceived_happiness: float  # 0-1
    perceived_sadness: float  # 0-1
    perceived_anger: float  # 0-1
    perceived_fear: float  # 0-1
    perceived_surprise: float  # 0-1


@dataclass
class ComprehensiveMetricsResult:
    """Vollständiges Metrik-Ergebnis."""

    psychoacoustic: PsychoAcousticMetrics
    musical: MusicalMetrics
    emotional: EmotionalMetrics

    # Overall Quality Scores
    overall_technical_quality: float  # 0-1
    overall_musical_quality: float  # 0-1
    overall_emotional_impact: float  # 0-1

    # Composite Score
    aurik_quality_score: float  # 0-100 (Weltklasse: >90)

    def to_dict(self) -> dict:
        """Konvertiert to dictionary."""
        return asdict(self)

    def passes_aurik_standards(self) -> bool:
        """Prüft if metrics meet Aurik Weltklasse standards."""
        return self.aurik_quality_score >= 90.0


# ============================================================
# COMPREHENSIVE METRICS CALCULATOR
# ============================================================


class ComprehensiveMetricsCalculator:
    """
    Berechnet alle psychoakustischen, musikalischen und emotionalen Metriken.

    Verwendung:
        calculator = ComprehensiveMetricsCalculator(sample_rate=48000)
        result = calculator.compute_all(audio)
    """

    def __init__(self, sample_rate: int = 48000):
        self.sr = sample_rate

        # Initialize external metrics modules
        if ENHANCED_METRICS_AVAILABLE:
            self.enhanced_metrics = EnhancedMetrics()
        if PROFESSIONAL_METERS_AVAILABLE:
            self.lufs_meter = LUFSMeter(sr=sample_rate)

    # ========================================
    # MAIN COMPUTATION
    # ========================================

    def compute_all(self, audio: np.ndarray, reference: np.ndarray | None = None) -> ComprehensiveMetricsResult:
        """
        Berechnet alle Metriken für gegebenes Audio.

        Args:
            audio: Audio signal (mono or stereo)
            reference: Optional reference signal for comparison metrics

        Returns:
            ComprehensiveMetricsResult with all metrics
        """
        # Normalize stereo layout to (samples, channels) and derive mono robustly.
        # Some callers provide channel-first audio with shape (channels, samples).
        audio_for_metrics = np.asarray(audio)
        if audio_for_metrics.ndim == 2:
            if audio_for_metrics.shape[0] <= 8 and audio_for_metrics.shape[1] > audio_for_metrics.shape[0]:
                audio_for_metrics = audio_for_metrics.T
            audio_mono = np.mean(audio_for_metrics, axis=1)
        else:
            audio_mono = audio_for_metrics

        # Compute each category
        psychoacoustic = self._compute_psychoacoustic(audio_for_metrics, audio_mono, reference)
        musical = self._compute_musical(audio_mono)
        emotional = self._compute_emotional(audio_mono)

        # Compute overall scores
        tech_quality = self._compute_technical_quality(psychoacoustic)
        music_quality = self._compute_musical_quality(musical)
        emotional_impact = self._compute_emotional_impact(emotional)

        # Aurik Quality Score (weighted combination)
        aurik_score = (
            tech_quality * 40.0  # 40% technical
            + music_quality * 40.0  # 40% musical
            + emotional_impact * 20.0  # 20% emotional
        )

        return ComprehensiveMetricsResult(
            psychoacoustic=psychoacoustic,
            musical=musical,
            emotional=emotional,
            overall_technical_quality=tech_quality,
            overall_musical_quality=music_quality,
            overall_emotional_impact=emotional_impact,
            aurik_quality_score=aurik_score,
        )

    # ========================================
    # PSYCHOACOUSTIC METRICS
    # ========================================

    def _compute_psychoacoustic(
        self, audio: np.ndarray, audio_mono: np.ndarray, reference: np.ndarray | None
    ) -> PsychoAcousticMetrics:
        """Berechnet psychoakustische Metriken."""

        # Signal Quality
        snr = self._compute_snr(audio_mono)
        thd = self._compute_thd(audio_mono)
        sinad = self._compute_sinad(audio_mono)

        # Loudness & Dynamics
        lufs, lra, true_peak = self._compute_loudness(audio)
        crest_factor = self._compute_crest_factor(audio_mono)

        # Frequency Response
        flatness, centroid, rolloff, flux = self._compute_spectral_features(audio_mono)

        # Masking & Perception
        sharpness = self._compute_sharpness(audio_mono)
        roughness = self._compute_roughness(audio_mono)
        tonality = self._compute_tonality(audio_mono)

        # Artifacts
        pre_echo = self._detect_pre_echo(audio_mono)
        clicks = self._detect_clicks(audio_mono)
        clipping = self._detect_clipping(audio_mono)

        return PsychoAcousticMetrics(
            snr_db=snr,
            thd_percent=thd,
            sinad_db=sinad,
            integrated_lufs=lufs,
            loudness_range_lu=lra,
            true_peak_dbtp=true_peak,
            crest_factor_db=crest_factor,
            frequency_response_flatness=flatness,
            spectral_centroid_hz=centroid,
            spectral_rolloff_hz=rolloff,
            spectral_flux=flux,
            perceptual_sharpness=sharpness,
            perceptual_roughness=roughness,
            tonality=tonality,
            pre_echo_score=pre_echo,
            click_detection=clicks,
            clipping_percent=clipping,
        )

    def _compute_snr(self, audio: np.ndarray) -> float:
        """Signal-to-Noise Ratio in dB (spektrale Methode).

        Algorithmus:
            1. Hanning-gewichtetes FFT des gesamten Signals
            2. Signal-Anteil = Energie in Top-5%-Bins (dominante Frequenzen)
            3. Rausch-Anteil  = Energie der verbleibenden 95%-Bins
            4. SNR = 10·log10(E_signal / (E_noise + ε))

        Diese Methode ist robust für reine Sinustöne (alle Energie in wenigen Bins)
        und breitrauschen (Energie gleichmäßig verteilt → niedriger SNR).
        """
        if len(audio) < 64:
            return 0.0

        # Hanning-Fenster reduziert Spectral-Leakage
        window = np.hanning(len(audio))
        spectrum = np.abs(_rfft(audio * window)) ** 2

        n_bins = len(spectrum)
        if n_bins < 4:
            return 0.0

        sorted_spectrum = np.sort(spectrum)
        split_idx = max(1, int(0.95 * n_bins))  # 95% Rauschen, 5% Signal

        noise_power = np.mean(sorted_spectrum[:split_idx]) + 1e-30
        signal_power = np.mean(sorted_spectrum[split_idx:]) + 1e-30

        if noise_power < 1e-20:
            return 80.0  # Vernachlässigbares Rauschen → cap bei 80 dB

        snr = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr, 0, 100))

    def _compute_thd(self, audio: np.ndarray) -> float:
        """Total Harmonic Distortion in %."""
        # Use FFT to find fundamental and harmonics
        spectrum = np.abs(_rfft(audio))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Find fundamental (assume speech/music range 80-1000 Hz)
        valid_range = (freqs >= 80) & (freqs <= 1000)
        if not np.any(valid_range):
            return 0.0

        fund_idx = np.argmax(spectrum[valid_range]) + np.where(valid_range)[0][0]
        fund_freq = freqs[fund_idx]
        fund_magnitude = spectrum[fund_idx]

        # Sum harmonics (2x, 3x, 4x, 5x fundamental)
        harmonic_power = 0.0
        for h in range(2, 6):
            harm_freq = fund_freq * h
            harm_idx = np.argmin(np.abs(freqs - harm_freq))
            if harm_idx < len(spectrum):
                harmonic_power += spectrum[harm_idx] ** 2

        thd = 100 * np.sqrt(harmonic_power) / (fund_magnitude + 1e-10)
        return float(np.clip(thd, 0, 100))

    def _compute_sinad(self, audio: np.ndarray) -> float:
        """Signal-to-Noise-And-Distortion in dB."""
        # SINAD combines SNR and THD
        snr = self._compute_snr(audio)
        thd = self._compute_thd(audio)

        # Convert THD to dB
        thd_db = 20 * np.log10(thd / 100 + 1e-10)

        # SINAD = 10 * log10(signal / (noise + distortion))
        sinad = snr + thd_db
        return float(np.clip(sinad, 0, 100))

    def _compute_loudness(self, audio: np.ndarray) -> tuple[float, float, float]:
        """Berechnet LUFS, LRA, True Peak."""
        if PROFESSIONAL_METERS_AVAILABLE and audio.ndim == 2:
            try:
                # LUFSMeter expects (channels, samples); many callers provide (samples, channels).
                meter_audio = audio.T if audio.shape[0] > audio.shape[1] else audio
                result = self.lufs_meter.measure(meter_audio, self.sr)
                integrated_lufs = float(result.get("integrated_lufs", -23.0))
                # LUFSMeter.measure returns integrated LUFS only; estimate LRA/TP in fallback path.
                # Keep estimate behavior while still using ITU-integrated LUFS when available.
                rms = np.sqrt(np.mean(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0) ** 2))
                peak = 20 * np.log10(np.max(np.abs(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0))) + 1e-10)
                frame_size = int(0.1 * self.sr)
                flat = audio.flatten()
                frames = np.array_split(flat, max(1, len(flat) // max(frame_size, 1)))
                frame_rms = [np.sqrt(np.mean(f**2)) for f in frames if len(f) == frame_size]
                if frame_rms:
                    lra = np.percentile(frame_rms, 95) - np.percentile(frame_rms, 10)
                    lra_lu = 20 * np.log10(lra + 1e-10)
                else:
                    lra_lu = 0.0
                return (integrated_lufs, float(lra_lu), float(peak))
            except Exception as e:
                logger.warning("LUFS meter fehlgeschlagen: %s", e)

        # Fallback: Simple RMS-based estimation
        rms = np.sqrt(np.mean(audio**2))
        lufs_approx = -23.0 + 20 * np.log10(rms + 1e-10)
        peak = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)

        # Estimate LRA from percentile differences
        frame_size = int(0.1 * self.sr)
        n_frames = max(1, len(audio.flatten()) // max(frame_size, 1))
        frames = np.array_split(audio.flatten(), n_frames)
        frame_rms = [np.sqrt(np.mean(f**2)) for f in frames if len(f) == frame_size]
        if frame_rms:
            lra = np.percentile(frame_rms, 95) - np.percentile(frame_rms, 10)
            lra_lu = 20 * np.log10(lra + 1e-10)
        else:
            lra_lu = 0.0

        return float(lufs_approx), float(lra_lu), float(peak)

    def _compute_crest_factor(self, audio: np.ndarray) -> float:
        """Crest Factor = Peak / RMS (in dB)."""
        peak: float = float(np.max(np.abs(audio)))
        rms = np.sqrt(np.mean(audio**2))
        crest_db = 20 * np.log10((peak + 1e-10) / (rms + 1e-10))
        return float(crest_db)

    def _safe_stft(
        self, audio: np.ndarray, nperseg: int = 2048, noverlap: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Berechnet STFT with adaptive parameters for short clips.

        This avoids SciPy warnings/exceptions when input is shorter than fixed STFT windows.
        """
        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            return np.array([0.0]), np.array([0.0]), np.zeros((1, 1), dtype=np.complex64)

        seg = max(1, min(int(nperseg), x.size))
        ov = min(seg // 2, seg - 1) if noverlap is None else min(int(noverlap), seg - 1)

        return signal.stft(x, self.sr, nperseg=seg, noverlap=ov, boundary="even")  # type: ignore[no-any-return]

    def _compute_spectral_features(self, audio: np.ndarray) -> tuple[float, float, float, float]:
        """Spektrale Merkmale: Flatness, Centroid, Rolloff, Flux (vektorisiert)."""
        f, _t, Zxx = self._safe_stft(audio, nperseg=2048)
        magnitude = np.abs(Zxx)  # shape: (freq_bins, time_frames)

        # Spektrale Flatness — vektorisiert (kein Python-Loop)
        log_mean = np.mean(np.log(magnitude + 1e-10), axis=0)  # (time_frames,)
        arith_mean = np.mean(magnitude, axis=0)
        flatness_per_frame = np.exp(log_mean) / (arith_mean + 1e-10)
        flatness = float(np.mean(flatness_per_frame))

        # Spektraler Schwerpunkt — vektorisiert
        f_col = f[:, np.newaxis]  # (freq_bins, 1) für Broadcasting
        energy_per_frame = np.sum(magnitude, axis=0)
        valid = energy_per_frame > 1e-10
        centroid_per_frame = np.where(valid, np.sum(f_col * magnitude, axis=0) / (energy_per_frame + 1e-10), 0.0)
        centroid = float(np.mean(centroid_per_frame[valid])) if valid.any() else 0.0

        # Spektraler Rolloff (95%) — vektorisiert
        cumsum = np.cumsum(magnitude, axis=0)  # (freq_bins, time_frames)
        threshold = 0.95 * cumsum[-1, :]  # (time_frames,)
        # Für jeden Frame: Index des ersten Bins wo cumsum ≥ threshold
        above = cumsum >= threshold[np.newaxis, :]  # (freq_bins, time_frames)
        rolloff_idx = np.argmax(above, axis=0)  # (time_frames,)
        rolloff = float(np.mean(f[rolloff_idx]))

        # Spektraler Flux — vektorisiert (diff entlang Zeit-Achse)
        diff = np.diff(magnitude, axis=1)  # (freq_bins, time_frames-1)
        flux_per_frame = np.sqrt(np.sum(diff**2, axis=0))
        flux = float(np.mean(flux_per_frame)) if flux_per_frame.size > 0 else 0.0

        return flatness, centroid, rolloff, flux

    def _compute_sharpness(self, audio: np.ndarray) -> float:
        """Perceptual sharpness (simplified Zwicker model)."""
        # High-frequency energy weighted by critical bands
        spectrum = np.abs(_rfft(audio))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Weight higher frequencies more
        weights = np.log10(freqs + 100) / np.log10(10000)
        weighted_energy: float = float(np.sum(spectrum * weights))
        total_energy: float = float(np.sum(spectrum))

        sharpness = weighted_energy / (total_energy + 1e-10)
        return float(np.clip(sharpness, 0, 1))

    def _compute_roughness(self, audio: np.ndarray) -> float:
        """Perceptual roughness (amplitude modulation 15-300 Hz)."""
        # Detect amplitude modulation in roughness range
        envelope = np.abs(_hilbert(audio))
        envelope_spectrum = np.abs(_rfft(envelope))
        freqs = fft.rfftfreq(len(envelope), 1 / self.sr)

        # Roughness band: 15-300 Hz modulation
        rough_mask = (freqs >= 15) & (freqs <= 300)
        roughness_energy: float = float(np.sum(envelope_spectrum[rough_mask]))
        total_energy: float = float(np.sum(envelope_spectrum))

        roughness = roughness_energy / (total_energy + 1e-10)
        return float(np.clip(roughness, 0, 1))

    def _compute_tonality(self, audio: np.ndarray) -> float:
        """Tonalität (tonal vs. Rauschen) via FFT-basierter Autokorrelation.

        O(N log N) — ersetzt O(N²) np.correlate(mode='full').
        """
        n = len(audio)
        if n < 64:
            return 0.0
        X = _rfft(audio, n=2 * n)
        autocorr = _irfft(np.abs(X) ** 2)[:n]
        ac0 = autocorr[0] + 1e-30
        autocorr = autocorr / ac0

        peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)

        tonality = float(autocorr[peaks[0] + 1]) if len(peaks) > 0 else 0.0

        return float(np.clip(tonality, 0, 1))

    def _detect_pre_echo(self, audio: np.ndarray) -> float:
        """Erkennt pre-echo artifacts (score: 1=clean, 0=artifacts)."""
        # Detect transients
        envelope = np.abs(_hilbert(audio))
        transients, _ = signal.find_peaks(envelope, height=np.percentile(envelope, 90))

        if len(transients) == 0:
            return 1.0

        # Check for energy spikes before transients
        pre_echo_count = 0
        window = int(0.005 * self.sr)  # 5ms before

        for t in transients:
            if t >= window:
                pre_window = envelope[t - window : t]
                if np.max(pre_window) > 0.3 * envelope[t]:
                    pre_echo_count += 1

        score = 1.0 - (pre_echo_count / (len(transients) + 1))
        return float(np.clip(score, 0, 1))

    def _detect_clicks(self, audio: np.ndarray) -> int:
        """Zählt detected clicks/pops."""
        # High-pass filter to isolate clicks
        sos = signal.butter(4, 2000, "high", fs=self.sr, output="sos")
        filtered: np.ndarray = np.asarray(signal.sosfilt(sos, audio))  # type: ignore[arg-type]

        # Detect spikes
        threshold = float(5 * np.std(filtered))
        clicks, _ = signal.find_peaks(np.abs(filtered), height=threshold, distance=int(0.001 * self.sr))  # type: ignore[call-overload]

        return len(clicks)

    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Percentage of clipped samples."""
        clipped: int = int(np.sum(np.abs(audio) > 0.99))
        percent = 100 * clipped / len(audio)
        return float(percent)

    # ========================================
    # MUSICAL METRICS
    # ========================================

    def _compute_musical(self, audio: np.ndarray) -> MusicalMetrics:
        """Berechnet musikalische Metriken."""

        # Harmonic Content
        clarity = self._compute_harmonic_clarity(audio)
        hnr = self._compute_hnr(audio)
        fund_stability = self._compute_fundamental_stability(audio)

        # Tonal Properties
        key, key_conf = self._detect_key(audio)
        consonance = self._compute_consonance(audio)

        # Rhythm & Timing
        tempo, tempo_stab = self._detect_tempo(audio)
        rhythm_reg = self._compute_rhythmic_regularity(audio)

        # Articulation & Dynamics
        attack = self._compute_attack_sharpness(audio)
        decay = self._compute_decay_smoothness(audio)
        dyn_contrast = self._compute_dynamic_contrast(audio)

        # Timbre & Texture
        complexity = self._compute_spectral_complexity(audio)
        balance = self._compute_spectral_balance(audio)
        warmth, brightness, fullness = self._compute_timbral_qualities(audio)

        return MusicalMetrics(
            harmonic_clarity=clarity,
            harmonic_to_noise_ratio_db=hnr,
            fundamental_stability=fund_stability,
            key_confidence=key_conf,
            detected_key=key,
            consonance=consonance,
            tempo_bpm=tempo,
            tempo_stability=tempo_stab,
            rhythmic_regularity=rhythm_reg,
            attack_sharpness=attack,
            decay_smoothness=decay,
            dynamic_contrast=dyn_contrast,
            spectral_complexity=complexity,
            spectral_balance=balance,
            warmth=warmth,
            brightness=brightness,
            fullness=fullness,
        )

    def _compute_harmonic_clarity(self, audio: np.ndarray) -> float:
        """Harmonischer Anteil vs. inharmonisches Rauschen (0–1).

        Algorithmus:
            1. Leistungsspektrum via rfft (quadriert für höhere Diskriminationskraft)
            2. Stärkstes Peak identifizieren
            3. Obertöne (Harmonische 1–6) suchen: Energiefenster ±3 Bins
            4. Harmonische Klarheit = harmonische Energie / Gesamtenergie × Skalierung
        """
        n = len(audio)
        if n < 64:
            return 0.0

        spectrum = np.abs(_rfft(audio)) ** 2  # Leistungsspektrum
        total_energy = float(np.sum(spectrum)) + 1e-30

        # Stärkstes Peak (Fundamental-Kandidat)
        peak_bin = int(np.argmax(spectrum))

        # Oberton-Kette prüfen: 6 Harmonische
        harmonic_energy = 0.0
        for h in range(1, 7):
            target_bin = peak_bin * h
            if target_bin >= len(spectrum):
                break
            lo = max(0, target_bin - 3)
            hi = min(len(spectrum), target_bin + 4)
            harmonic_energy += float(np.sum(spectrum[lo:hi]))

        clarity = harmonic_energy / total_energy
        return float(np.clip(clarity * 8.0, 0.0, 1.0))

    def _compute_hnr(self, audio: np.ndarray) -> float:
        """Harmonic-to-Noise Ratio (dB) via FFT-basierter Autokorrelation.

        Algorithmus: R(τ) = IFFT(|FFT(x)|²) — O(N log N) statt O(N²).
        """
        n = len(audio)
        if n < 64:
            return 0.0
        # Zero-padded FFT für lineare Autokorrelation
        X = _rfft(audio, n=2 * n)
        autocorr = _irfft(np.abs(X) ** 2)[:n]
        ac0 = autocorr[0] + 1e-30
        autocorr = autocorr / ac0  # Normalisierung auf [0, 1]

        search_end = min(int(0.02 * self.sr), n - 1)
        if search_end < 2:
            return 0.0
        periodic_energy = float(np.max(autocorr[1:search_end]))
        noise_energy = 1.0 - periodic_energy

        hnr = 10.0 * np.log10((periodic_energy + 1e-10) / (noise_energy + 1e-10))
        return float(np.clip(hnr, -10, 40))

    def _compute_fundamental_stability(self, audio: np.ndarray) -> float:
        """Pitch-Stabilität über Zeit via FFT-basierter Frame-Autokorrelation.

        Algorithmus: FFT-ACF pro Frame — O(N log N) statt O(N²).
        Hop = frame_size (kein Overlap) für optimierte Performance bei langen Signalen.
        """
        frame_size = int(0.05 * self.sr)  # 50 ms Frames
        hop = frame_size  # Kein Overlap → halb so viele Frames, 2× schneller

        if len(audio) < frame_size * 2:
            return 0.5

        pitches = []
        search_end = min(int(self.sr / 50), frame_size - 1)  # Bis 50 Hz Untergrenze

        for start in range(0, len(audio) - frame_size, hop):
            frame = audio[start : start + frame_size]
            n = len(frame)
            # FFT-basierte lineare Autokorrelation
            X = _rfft(frame, n=2 * n)
            autocorr = _irfft(np.abs(X) ** 2)[:n]
            ac0 = autocorr[0] + 1e-30
            autocorr = autocorr / ac0

            # Fundamental-Peak suchen (Bins 20 … search_end)
            search_region = autocorr[20:search_end]
            if len(search_region) < 2:
                continue
            peaks, _ = signal.find_peaks(search_region, height=0.4)
            if len(peaks) > 0:
                period = peaks[0] + 20
                freq = self.sr / (period + 1e-10)
                if 50.0 < freq < 2000.0:
                    pitches.append(freq)

        if len(pitches) < 2:
            return 0.5

        stability = 1.0 - (np.std(pitches) / (np.mean(pitches) + 1e-10))
        return float(np.clip(stability, 0, 1))

    def _detect_key(self, audio: np.ndarray) -> tuple[str, float]:
        """Erkennt musical key (simplified Krumhansl-Schmuckler)."""
        # Chromagram
        chroma = self._compute_chromagram(audio)

        # Key profiles (major/minor templates)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)

        # Test all 24 keys
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        best_corr = -1
        best_key = "C major"

        for shift in range(12):
            # Major
            shifted_major = np.roll(major_profile, shift)
            corr_major = pearsonr(chroma, shifted_major)[0] if not np.all(chroma == chroma[0]) else 0
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = f"{keys[shift]} major"

            # Minor
            shifted_minor = np.roll(minor_profile, shift)
            corr_minor = pearsonr(chroma, shifted_minor)[0] if not np.all(chroma == chroma[0]) else 0
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = f"{keys[shift]} minor"

        confidence = float(np.clip((best_corr + 1) / 2, 0, 1))  # Map [-1,1] to [0,1]
        return best_key, confidence

    def _compute_chromagram(self, audio: np.ndarray) -> np.ndarray:
        """Berechnet 12-bin chromagram (pitch class profile)."""
        spectrum = np.abs(_rfft(audio))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        chroma = np.zeros(12)

        for i, freq in enumerate(freqs):
            if 50 < freq < 4000:  # Musical range
                # Convert frequency to pitch class (0-11)
                pitch_class = int(np.round(12 * np.log2(freq / 440.0))) % 12
                chroma[pitch_class] += spectrum[i]

        chroma = chroma / (np.sum(chroma) + 1e-10)
        return chroma  # type: ignore[no-any-return]

    def _compute_consonance(self, audio: np.ndarray) -> float:
        """Harmonic consonance (Helmholtz consonance)."""
        spectrum = np.abs(_rfft(audio))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Find peaks (partials)
        peaks, _ = signal.find_peaks(spectrum, height=np.percentile(spectrum, 90))

        if len(peaks) < 2:
            return 0.5

        # Compute frequency ratios
        peak_freqs = freqs[peaks]
        consonance_score = 0
        comparisons = 0

        for i in range(min(10, len(peak_freqs))):
            for j in range(i + 1, min(10, len(peak_freqs))):
                ratio = peak_freqs[j] / (peak_freqs[i] + 1e-10)
                # Simple integer ratio consonance (octave, fifth, fourth, etc.)
                simple_ratios = [2.0, 1.5, 4 / 3, 5 / 4, 6 / 5]
                min_dist = min([abs(ratio - r) for r in simple_ratios])
                consonance_score += np.exp(-10 * min_dist)
                comparisons += 1

        if comparisons == 0:
            return 0.5

        consonance = consonance_score / comparisons
        return float(np.clip(consonance, 0, 1))

    def _detect_tempo(self, audio: np.ndarray) -> tuple[float, float]:
        """Erkennt tempo (BPM) and stability."""
        if len(audio) < 16:
            return 120.0, 0.5

        nperseg = min(2048, len(audio))
        if nperseg < 2:
            return 120.0, 0.5
        noverlap = min(1536, int(0.75 * nperseg), nperseg - 1)

        # Onset detection via spectral flux
        _f, _t, Zxx = self._safe_stft(audio, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)

        # Spectral flux
        flux = np.zeros(magnitude.shape[1])
        for i in range(1, magnitude.shape[1]):
            diff = magnitude[:, i] - magnitude[:, i - 1]
            flux[i] = np.sum(np.maximum(diff, 0))

        # Autocorrelation of onset strength — FFT-based O(N log N)
        from backend.core.core_utils import fft_autocorr

        autocorr = fft_autocorr(flux)
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Find tempo in reasonable range (60-180 BPM)
        hop_time = max(1 / self.sr, (nperseg - noverlap) / self.sr)
        min_lag = int(60 / 180 / hop_time)  # 180 BPM
        max_lag = int(60 / 60 / hop_time)  # 60 BPM

        if max_lag >= len(autocorr):
            return 120.0, 0.5

        tempo_range = autocorr[min_lag:max_lag]
        if len(tempo_range) == 0:
            return 120.0, 0.5

        peak_lag = np.argmax(tempo_range) + min_lag
        tempo = 60.0 / (peak_lag * hop_time)

        # Stability = height of autocorrelation peak
        stability = float(autocorr[peak_lag])

        return float(tempo), float(np.clip(stability, 0, 1))

    def _compute_rhythmic_regularity(self, audio: np.ndarray) -> float:
        """Beat regularity (periodicity of onset envelope)."""
        if len(audio) < 32:
            return 0.3

        # Onset envelope
        envelope = np.abs(_hilbert(audio))

        # Downsample to ~100 Hz for beat tracking
        target_rate = 100
        decimation = max(1, self.sr // target_rate)
        # signal.decimate with zero_phase=True can fail on very short vectors due to padlen.
        # Use simple stride-based downsampling fallback when the clip is too short.
        if len(envelope) <= 27:
            envelope_ds = envelope[::decimation] if decimation > 1 else envelope
        else:
            try:
                envelope_ds = signal.decimate(envelope, decimation, zero_phase=True)
            except ValueError:
                envelope_ds = envelope[::decimation] if decimation > 1 else envelope

        if len(envelope_ds) < 16:
            return 0.3

        # Autocorrelation — FFT-based O(N log N)
        from backend.core.core_utils import fft_autocorr

        autocorr = fft_autocorr(envelope_ds)
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Regularity = strength of first peak
        peaks, _ = signal.find_peaks(autocorr[10:100], height=0.2)

        regularity = float(autocorr[peaks[0] + 10]) if len(peaks) > 0 else 0.3

        return float(np.clip(regularity, 0, 1))

    def _compute_attack_sharpness(self, audio: np.ndarray) -> float:
        """Transient attack sharpness."""
        # Detect transients
        envelope = np.abs(_hilbert(audio))
        transients, _ = signal.find_peaks(envelope, height=np.percentile(envelope, 90), distance=int(0.1 * self.sr))

        if len(transients) == 0:
            return 0.5

        # Measure slope at each transient
        attack_window = int(0.01 * self.sr)  # 10ms
        sharpness_values = []

        for t in transients:
            if t >= attack_window and t + attack_window < len(envelope):
                pre = envelope[t - attack_window : t]
                post = envelope[t : t + attack_window]
                slope = (np.mean(post) - np.mean(pre)) / (attack_window / self.sr)
                sharpness_values.append(slope)

        if not sharpness_values:
            return 0.5

        mean_sharpness = np.mean(sharpness_values)
        # Normalize to 0-1 (assuming typical range 0-10)
        sharpness = np.clip(mean_sharpness / 10, 0, 1)

        return float(sharpness)

    def _compute_decay_smoothness(self, audio: np.ndarray) -> float:
        """Decay envelope smoothness."""
        # Envelope
        envelope = np.abs(_hilbert(audio))

        # Find peaks (note onsets)
        peaks, _ = signal.find_peaks(envelope, height=np.percentile(envelope, 80), distance=int(0.2 * self.sr))

        if len(peaks) == 0:
            return 0.5

        # Measure decay smoothness after each peak
        decay_window = int(0.1 * self.sr)  # 100ms
        smoothness_values = []

        for peak in peaks:
            if peak + decay_window < len(envelope):
                decay_segment = envelope[peak : peak + decay_window]

                # Fit exponential decay
                x = np.arange(len(decay_segment))
                y = np.log(decay_segment + 1e-10)

                # Smoothness = how well it fits exponential
                if len(x) > 1:
                    fit = np.polyfit(x, y, 1)
                    predicted = np.poly1d(fit)(x)
                    r2 = 1 - (np.sum((y - predicted) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-10))
                    smoothness_values.append(r2)

        if not smoothness_values:
            return 0.5

        smoothness = np.clip(np.mean(smoothness_values), 0, 1)
        return float(smoothness)

    def _compute_dynamic_contrast(self, audio: np.ndarray) -> float:
        """Micro-dynamics (small-scale amplitude variations)."""
        # Compute RMS in small windows
        window = int(0.01 * self.sr)  # 10ms
        hop = window // 2

        rms_values = []
        for i in range(0, len(audio) - window, hop):
            frame = audio[i : i + window]
            rms = np.sqrt(np.mean(frame**2))
            rms_values.append(rms)

        if not rms_values:
            return 0.5

        # Contrast = standard deviation of RMS
        contrast = np.std(rms_values) / (np.mean(rms_values) + 1e-10)
        return float(np.clip(contrast, 0, 1))

    def _compute_spectral_complexity(self, audio: np.ndarray) -> float:
        """Harmonic richness (number of significant partials)."""
        spectrum = np.abs(_rfft(audio))

        # Find peaks
        threshold = np.percentile(spectrum, 85)
        peaks, _ = signal.find_peaks(spectrum, height=threshold)

        # Normalize by typical range
        complexity = len(peaks) / 50  # Assume 50 partials is very complex
        return float(np.clip(complexity, 0, 1))

    def _compute_spectral_balance(self, audio: np.ndarray) -> float:
        """Bass/mid/treble balance."""
        spectrum = np.abs(_rfft(audio))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Define bands
        bass = freqs < 250
        mid = (freqs >= 250) & (freqs < 4000)
        treble = freqs >= 4000

        bass_energy: float = float(np.sum(spectrum[bass]))
        mid_energy: float = float(np.sum(spectrum[mid]))
        treble_energy: float = float(np.sum(spectrum[treble]))

        total = bass_energy + mid_energy + treble_energy + 1e-10

        # Ideal balance: 30% bass, 50% mid, 20% treble
        ideal = np.array([0.3, 0.5, 0.2])
        actual = np.array([bass_energy, mid_energy, treble_energy]) / total

        # Balance = 1 - normalized distance from ideal
        distance = np.sqrt(np.sum((ideal - actual) ** 2))
        balance = 1.0 - distance

        return float(np.clip(balance, 0, 1))

    def _compute_timbral_qualities(self, audio: np.ndarray) -> tuple[float, float, float]:
        """Berechnet warmth, brightness, fullness."""
        spectrum = np.abs(_rfft(audio))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Warmth: low-mid energy (150-500 Hz)
        warmth_mask = (freqs >= 150) & (freqs <= 500)
        warmth = np.sum(spectrum[warmth_mask]) / (np.sum(spectrum) + 1e-10)
        warmth = float(np.clip(warmth / 0.3, 0, 1))  # Normalize

        # Brightness: high-freq energy (4-10 kHz)
        bright_mask = (freqs >= 4000) & (freqs <= 10000)
        brightness = np.sum(spectrum[bright_mask]) / (np.sum(spectrum) + 1e-10)
        brightness = float(np.clip(brightness / 0.2, 0, 1))

        # Fullness: mid-range energy (500-2000 Hz)
        full_mask = (freqs >= 500) & (freqs <= 2000)
        fullness = np.sum(spectrum[full_mask]) / (np.sum(spectrum) + 1e-10)
        fullness = float(np.clip(fullness / 0.4, 0, 1))

        return warmth, brightness, fullness

    # ========================================
    # EMOTIONAL METRICS
    # ========================================

    def _compute_emotional(self, audio: np.ndarray) -> EmotionalMetrics:
        """Berechnet emotionale Metriken."""

        # Core Dimensions
        valence, arousal = self._compute_valence_arousal(audio)

        # Energy & Intensity
        energy = self._compute_energy(audio)
        intensity = self._compute_intensity(audio)
        tension = self._compute_tension(audio)

        # Emotional Categories (Geneva scale)
        power = self._compute_power(audio)
        joyful = self._compute_joyful_activation(audio)
        nostalgia = self._compute_nostalgia(audio)
        sadness = self._compute_sadness(audio)
        peace = self._compute_peacefulness(audio)
        transcend = self._compute_transcendence(audio)

        # Perceived Affect
        happiness = self._compute_perceived_happiness(audio)
        sad = self._compute_perceived_sadness(audio)
        anger = self._compute_perceived_anger(audio)
        fear = self._compute_perceived_fear(audio)
        surprise = self._compute_perceived_surprise(audio)

        return EmotionalMetrics(
            valence=valence,
            arousal=arousal,
            energy=energy,
            intensity=intensity,
            tension=tension,
            power=power,
            joyful_activation=joyful,
            nostalgia=nostalgia,
            sadness=sadness,
            peacefulness=peace,
            transcendence=transcend,
            perceived_happiness=happiness,
            perceived_sadness=sad,
            perceived_anger=anger,
            perceived_fear=fear,
            perceived_surprise=surprise,
        )

    def _compute_valence_arousal(self, audio: np.ndarray) -> tuple[float, float]:
        """
        Berechnet valence and arousal (Russell's Circumplex Model).

        Valence: Pleasantness (consonance, harmonic content)
        Arousal: Energy/Activation (tempo, loudness, spectral flux)
        """
        # Valence: based on consonance and harmonic clarity
        spectrum = np.abs(_rfft(audio))
        fft.rfftfreq(len(audio), 1 / self.sr)

        # Harmonic content suggests pleasantness
        peaks, _ = signal.find_peaks(spectrum, height=np.percentile(spectrum, 80))
        harmonic_ratio = len(peaks) / (len(spectrum) / 100)  # More peaks = more harmonic

        # Low dissonance = positive valence
        valence = np.clip(harmonic_ratio / 5, 0, 1)  # Normalize
        valence = 2 * valence - 1  # Map to [-1, +1]

        # Arousal: based on tempo, loudness, and spectral flux
        rms = np.sqrt(np.mean(audio**2))
        loudness_factor = np.clip(rms / 0.3, 0, 1)

        # Spectral flux
        _f, _t, Zxx = self._safe_stft(audio, nperseg=2048)
        magnitude = np.abs(Zxx)
        flux_values = []
        for i in range(1, magnitude.shape[1]):
            flux = np.sqrt(np.sum((magnitude[:, i] - magnitude[:, i - 1]) ** 2))
            flux_values.append(flux)
        flux_mean = np.mean(flux_values) if flux_values else 0
        flux_factor = np.clip(flux_mean / 100, 0, 1)

        arousal = (loudness_factor + flux_factor) / 2
        arousal = 2 * arousal - 1  # Map to [-1, +1]

        return float(valence), float(arousal)

    def _compute_energy(self, audio: np.ndarray) -> float:
        """Overall energy level."""
        rms = np.sqrt(np.mean(audio**2))
        energy = np.clip(rms / 0.5, 0, 1)
        return float(energy)

    def _compute_intensity(self, audio: np.ndarray) -> float:
        """Emotional intensity (dynamic range + spectral spread)."""
        # Dynamic range
        peak: float = float(np.max(np.abs(audio)))
        rms = np.sqrt(np.mean(audio**2))
        dyn_range = peak / (rms + 1e-10)

        # Spectral spread
        spectrum = np.abs(_rfft(audio))
        spread = np.std(spectrum) / (np.mean(spectrum) + 1e-10)

        intensity = np.clip((dyn_range + spread) / 20, 0, 1)
        return float(intensity)

    def _compute_tension(self, audio: np.ndarray) -> float:
        """Harmonic/rhythmic tension."""
        # Dissonance
        consonance = self._compute_consonance(audio)
        dissonance = 1 - consonance

        # Rhythmic irregularity
        regularity = self._compute_rhythmic_regularity(audio)
        irregularity = 1 - regularity

        tension = (dissonance + irregularity) / 2
        return float(np.clip(tension, 0, 1))

    def _compute_power(self, audio: np.ndarray) -> float:
        """Feeling of power/confidence (low-freq presence + loudness)."""
        spectrum = np.abs(_rfft(audio))
        freqs = fft.rfftfreq(len(audio), 1 / self.sr)

        # Low-freq energy (50-200 Hz)
        bass_mask = (freqs >= 50) & (freqs <= 200)
        bass_energy = np.sum(spectrum[bass_mask]) / (np.sum(spectrum) + 1e-10)

        # Loudness
        rms = np.sqrt(np.mean(audio**2))

        power = (bass_energy * 3 + rms) / 4  # Weight bass more
        return float(np.clip(power, 0, 1))

    def _compute_joyful_activation(self, audio: np.ndarray) -> float:
        """Happy, cheerful (high arousal + positive valence)."""
        valence, arousal = self._compute_valence_arousal(audio)

        # Tempo (fast = joyful)
        tempo, _ = self._detect_tempo(audio)
        tempo_factor = np.clip((tempo - 100) / 80, 0, 1)  # 100-180 BPM

        joyful = (valence + 1) / 2 * (arousal + 1) / 2 * tempo_factor
        return float(np.clip(joyful, 0, 1))

    def _compute_nostalgia(self, audio: np.ndarray) -> float:
        """Nostalgic, sentimental (moderate arousal, warm timbre)."""
        # Warmth
        warmth, _, _ = self._compute_timbral_qualities(audio)

        # Moderate tempo
        tempo, _ = self._detect_tempo(audio)
        tempo_factor = 1.0 - abs(tempo - 90) / 90  # Peak at 90 BPM
        tempo_factor = np.clip(tempo_factor, 0, 1)

        nostalgia = (warmth + tempo_factor) / 2
        return float(np.clip(nostalgia, 0, 1))

    def _compute_sadness(self, audio: np.ndarray) -> float:
        """Sad, melancholic (low arousal + negative valence)."""
        valence, arousal = self._compute_valence_arousal(audio)

        # Slow tempo
        tempo, _ = self._detect_tempo(audio)
        slow_factor = np.clip(1.0 - (tempo - 60) / 60, 0, 1)  # Slower = sadder

        sadness = (1 - (valence + 1) / 2) * (1 - (arousal + 1) / 2) * slow_factor
        return float(np.clip(sadness, 0, 1))

    def _compute_peacefulness(self, audio: np.ndarray) -> float:
        """Calm, peaceful (low arousal + moderate valence)."""
        valence, arousal = self._compute_valence_arousal(audio)

        # Low energy
        energy = self._compute_energy(audio)

        peacefulness = (valence + 1) / 2 * (1 - (arousal + 1) / 2) * (1 - energy)
        return float(np.clip(peacefulness, 0, 1))

    def _compute_transcendence(self, audio: np.ndarray) -> float:
        """Spiritual, transcendent (spectral complexity + reverb)."""
        # Spectral complexity
        complexity = self._compute_spectral_complexity(audio)

        # High-frequency content (ethereal)
        _, brightness, _ = self._compute_timbral_qualities(audio)

        transcendence = (complexity + brightness) / 2
        return float(np.clip(transcendence, 0, 1))

    def _compute_perceived_happiness(self, audio: np.ndarray) -> float:
        """Perceived happiness (high valence + high arousal)."""
        valence, arousal = self._compute_valence_arousal(audio)
        happiness = ((valence + 1) / 2) * ((arousal + 1) / 2)
        return float(np.clip(happiness, 0, 1))

    def _compute_perceived_sadness(self, audio: np.ndarray) -> float:
        """Perceived sadness (low valence + low arousal)."""
        valence, arousal = self._compute_valence_arousal(audio)
        sadness = (1 - (valence + 1) / 2) * (1 - (arousal + 1) / 2)
        return float(np.clip(sadness, 0, 1))

    def _compute_perceived_anger(self, audio: np.ndarray) -> float:
        """Perceived anger (low valence + high arousal)."""
        valence, arousal = self._compute_valence_arousal(audio)
        anger = (1 - (valence + 1) / 2) * ((arousal + 1) / 2)
        return float(np.clip(anger, 0, 1))

    def _compute_perceived_fear(self, audio: np.ndarray) -> float:
        """Perceived fear (low valence + high tension)."""
        valence, _ = self._compute_valence_arousal(audio)
        tension = self._compute_tension(audio)
        fear = (1 - (valence + 1) / 2) * tension
        return float(np.clip(fear, 0, 1))

    def _compute_perceived_surprise(self, audio: np.ndarray) -> float:
        """Perceived surprise (high spectral flux + transients)."""
        # Spectral flux
        _f, _t, Zxx = self._safe_stft(audio, nperseg=2048)
        magnitude = np.abs(Zxx)
        flux_values = []
        for i in range(1, magnitude.shape[1]):
            flux = np.sqrt(np.sum((magnitude[:, i] - magnitude[:, i - 1]) ** 2))
            flux_values.append(flux)
        flux_mean = np.mean(flux_values) if flux_values else 0

        # Transient density
        envelope = np.abs(_hilbert(audio))
        transients, _ = signal.find_peaks(envelope, height=np.percentile(envelope, 85))
        transient_density = len(transients) / (len(audio) / self.sr)  # per second

        surprise = (np.clip(flux_mean / 100, 0, 1) + np.clip(transient_density / 10, 0, 1)) / 2
        return float(np.clip(surprise, 0, 1))

    # ========================================
    # QUALITY SCORING
    # ========================================

    def _compute_technical_quality(self, psycho: PsychoAcousticMetrics) -> float:
        """Berechnet overall technical quality score (0-1)."""
        scores = []

        # Signal quality
        scores.append(np.clip(psycho.snr_db / 40, 0, 1))
        scores.append(np.clip(1 - psycho.thd_percent / 5, 0, 1))

        # Loudness compliance (target: -16 LUFS for music)
        lufs_offset = abs(psycho.integrated_lufs + 16)
        scores.append(np.clip(1 - lufs_offset / 10, 0, 1))

        # Dynamics
        scores.append(np.clip(psycho.loudness_range_lu / 15, 0, 1))

        # Artifacts
        scores.append(psycho.pre_echo_score)
        scores.append(np.clip(1 - psycho.clipping_percent / 1, 0, 1))

        return float(np.mean(scores))

    def _compute_musical_quality(self, musical: MusicalMetrics) -> float:
        """Berechnet overall musical quality score (0-1)."""
        scores = []

        # Harmonic content
        scores.append(musical.harmonic_clarity)
        scores.append(np.clip(musical.harmonic_to_noise_ratio_db / 20, 0, 1))

        # Tonal properties
        scores.append(musical.consonance)

        # Articulation
        scores.append(musical.attack_sharpness)
        scores.append(musical.decay_smoothness)

        # Timbre
        scores.append(musical.spectral_balance)
        scores.append((musical.warmth + musical.brightness + musical.fullness) / 3)

        return float(np.mean(scores))

    def _compute_emotional_impact(self, emotional: EmotionalMetrics) -> float:
        """Berechnet overall emotional impact score (0-1)."""
        scores = []

        # Energy and intensity
        scores.append(emotional.energy)
        scores.append(emotional.intensity)

        # Emotional diversity (presence of multiple emotions)
        emotion_values = [
            emotional.joyful_activation,
            emotional.nostalgia,
            emotional.sadness,
            emotional.peacefulness,
            emotional.power,
            emotional.transcendence,
        ]
        diversity = np.std(emotion_values)
        scores.append(np.clip(diversity, 0, 1))

        # Overall emotional clarity (strong emotions)
        max_emotion = max(emotion_values)
        scores.append(max_emotion)

        return float(np.mean(scores))


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def generate_metrics_report(result: ComprehensiveMetricsResult) -> str:
    """Generiert human-readable metrics report."""
    report = []
    report.append("=" * 70)
    report.append("AURIK 9.0 COMPREHENSIVE AUDIO QUALITY METRICS")
    report.append("=" * 70)

    # Psychoacoustic
    report.append("\n📊 PSYCHOACOUSTIC METRICS")
    report.append("-" * 70)
    p = result.psychoacoustic
    report.append(f"SNR:                {p.snr_db:.1f} dB")
    report.append(f"THD:                {p.thd_percent:.2f} %")
    report.append(f"Integrated LUFS:    {p.integrated_lufs:.1f} LUFS")
    report.append(f"Loudness Range:     {p.loudness_range_lu:.1f} LU")
    report.append(f"True Peak:          {p.true_peak_dbtp:.1f} dBTP")
    report.append(f"Crest Factor:       {p.crest_factor_db:.1f} dB")
    report.append(f"Tonality:           {p.tonality:.2f}")
    report.append(f"Pre-Echo Score:     {p.pre_echo_score:.2f}")
    report.append(f"Clicks Detected:    {p.click_detection}")
    report.append(f"Clipping:           {p.clipping_percent:.2f} %")

    # Musical
    report.append("\n🎵 MUSICAL METRICS")
    report.append("-" * 70)
    m = result.musical
    report.append(f"Key:                {m.detected_key} (conf: {m.key_confidence:.2f})")
    report.append(f"Tempo:              {m.tempo_bpm:.1f} BPM (stability: {m.tempo_stability:.2f})")
    report.append(f"Harmonic Clarity:   {m.harmonic_clarity:.2f}")
    report.append(f"HNR:                {m.harmonic_to_noise_ratio_db:.1f} dB")
    report.append(f"Consonance:         {m.consonance:.2f}")
    report.append(f"Spectral Balance:   {m.spectral_balance:.2f}")
    report.append(f"Warmth:             {m.warmth:.2f}")
    report.append(f"Brightness:         {m.brightness:.2f}")
    report.append(f"Fullness:           {m.fullness:.2f}")

    # Emotional
    report.append("\n💫 EMOTIONAL METRICS")
    report.append("-" * 70)
    e = result.emotional
    report.append(f"Valence:            {e.valence:+.2f} (negative ← → positive)")
    report.append(f"Arousal:            {e.arousal:+.2f} (calm ← → energetic)")
    report.append(f"Energy:             {e.energy:.2f}")
    report.append(f"Intensity:          {e.intensity:.2f}")
    report.append(f"Tension:            {e.tension:.2f}")
    report.append(f"Joyful Activation:  {e.joyful_activation:.2f}")
    report.append(f"Nostalgia:          {e.nostalgia:.2f}")
    report.append(f"Sadness:            {e.sadness:.2f}")
    report.append(f"Peacefulness:       {e.peacefulness:.2f}")
    report.append(f"Power:              {e.power:.2f}")

    # Overall
    report.append("\n⭐ OVERALL QUALITY SCORES")
    report.append("-" * 70)
    report.append(f"Technical Quality:  {result.overall_technical_quality * 100:.1f} / 100")
    report.append(f"Musical Quality:    {result.overall_musical_quality * 100:.1f} / 100")
    report.append(f"Emotional Impact:   {result.overall_emotional_impact * 100:.1f} / 100")
    report.append(f"\n🏆 AURIK QUALITY SCORE: {result.aurik_quality_score:.1f} / 100")

    if result.passes_aurik_standards():
        report.append("\n✅ WELTKLASSE - Meets Aurik 10.0.0 Standards!")
    else:
        report.append("\n⚠️  Below Weltklasse standards")

    report.append("=" * 70)

    return "\n".join(report)


# ============================================================
# MAIN ENTRY POINT FOR TESTING
# ============================================================

if __name__ == "__main__":
    """Test comprehensive metrics with synthetic audio."""
    logger.debug("Testing Aurik 10.0.0 Comprehensive Metrics System...")
    logger.debug("=" * 70)

    # Generate test audio
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Complex musical signal: fundamental + harmonics + modulation
    freq_fund = 440.0  # A4
    audio = 0.3 * np.sin(2 * np.pi * freq_fund * t)  # Fundamental
    audio += 0.15 * np.sin(2 * np.pi * 2 * freq_fund * t)  # 2nd harmonic
    audio += 0.1 * np.sin(2 * np.pi * 3 * freq_fund * t)  # 3rd harmonic
    audio += 0.05 * np.sin(2 * np.pi * 5 * freq_fund * t)  # 5th harmonic

    # Add rhythmic envelope
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz rhythm
    audio = audio * envelope

    # Add subtle noise
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise

    # Compute metrics
    logger.debug("berechne comprehensive metrics...")
    calculator = ComprehensiveMetricsCalculator(sample_rate=sr)
    result = calculator.compute_all(audio)

    # Print report
    report = generate_metrics_report(result)
    logger.debug(report)

    # Export to dict
    logger.debug("\n" + "=" * 70)
    logger.debug("Exporting to dictionary...")
    metrics_dict = result.to_dict()
    logger.debug("Total metrics computed: %s values", len(str(metrics_dict).split(",")))
    logger.debug("✅ Comprehensive Metrics System Test vollstaendig!")
