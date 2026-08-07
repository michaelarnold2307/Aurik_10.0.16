"""
forensics/feature_extractor.py
Unified Feature Extraction für Signal Forensics
================================================

Extrahiert 30+ Audio-Features für ML-basierte Detection:
- Spectral: Centroid, Rolloff, Flux, Flatness, Contrast
- Temporal: ZCR, RMS, Attack, Decay, Transients
- Harmonic: MFCC, Chroma, Tonnetz
- Defect: Clicks, Hum, Wow/Flutter, Noise Floor

USAGE:
    from backend.core.forensics.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    features = extractor.extract_all(audio, sr)
    # Returns: Dict mit 30+ Features als numpy arrays
"""

import logging
from dataclasses import dataclass, field

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Container für extrahierte Audio-Features."""

    # Spectral Features
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_rolloff_std: float = 0.0
    spectral_flux_mean: float = 0.0
    spectral_flatness_mean: float = 0.0
    spectral_contrast_mean: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # Temporal Features
    zero_crossing_rate_mean: float = 0.0
    rms_energy_mean: float = 0.0
    rms_energy_std: float = 0.0
    transient_density: float = 0.0  # Transients per second
    attack_time_mean: float = 0.0

    # Harmonic Features
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(13))
    mfcc_std: np.ndarray = field(default_factory=lambda: np.zeros(13))
    chroma_mean: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # Bandwidth & Dynamic Range
    bandwidth_3db_low: float = 0.0
    bandwidth_3db_high: float = 0.0
    dynamic_range_db: float = 0.0
    crest_factor: float = 0.0

    # Noise & Artifacts
    noise_floor_db: float = 0.0
    hum_presence: float = 0.0  # 50/60 Hz energy
    clicks_per_second: float = 0.0
    wow_strength: float = 0.0
    flutter_strength: float = 0.0

    # Statistical
    audio_kurtosis: float = 0.0
    audio_skewness: float = 0.0
    peak_to_rms_db: float = 0.0

    # Stereo (if applicable)
    stereo_width: float = 0.0
    phase_correlation: float = 0.0
    channel_imbalance_db: float = 0.0

    def to_array(self) -> np.ndarray:
        """Konvertiert Features zu Numpy Array für ML."""
        features = []

        # Scalar features
        features.extend(
            [
                self.spectral_centroid_mean,
                self.spectral_centroid_std,
                self.spectral_rolloff_mean,
                self.spectral_rolloff_std,
                self.spectral_flux_mean,
                self.spectral_flatness_mean,
                self.zero_crossing_rate_mean,
                self.rms_energy_mean,
                self.rms_energy_std,
                self.transient_density,
                self.attack_time_mean,
                self.bandwidth_3db_low,
                self.bandwidth_3db_high,
                self.dynamic_range_db,
                self.crest_factor,
                self.noise_floor_db,
                self.hum_presence,
                self.clicks_per_second,
                self.wow_strength,
                self.flutter_strength,
                self.audio_kurtosis,
                self.audio_skewness,
                self.peak_to_rms_db,
                self.stereo_width,
                self.phase_correlation,
                self.channel_imbalance_db,
            ]
        )

        # Vector features (flatten)
        features.extend(self.spectral_contrast_mean.tolist())
        features.extend(self.mfcc_mean.tolist())
        features.extend(self.mfcc_std.tolist())
        features.extend(self.chroma_mean.tolist())

        return np.array(features)  # type: ignore[no-any-return]


class FeatureExtractor:
    """
    Extrahiert Audio-Features für Signal Forensics.
    Wiederverwendbar für Medium, Era und Defect Detection.
    """

    def __init__(self, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512) -> None:
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_all(self, audio: np.ndarray, sr: int, verbose: bool = False) -> AudioFeatures:
        """
        Extrahiert alle Features aus Audio-Signal.

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate
            verbose: Print extraction progress

        Returns:
            AudioFeatures mit allen extrahierten Features
        """
        features = AudioFeatures()

        # Konvertiere zu Mono für die meisten Features
        if audio.ndim == 2:
            audio_mono = librosa.to_mono(audio.T)
            is_stereo = True
        else:
            audio_mono = audio
            is_stereo = False

        if verbose:
            logger.debug("🔬 extrahiere features...")

        # 1. Spectral Features
        features = self._extract_spectral_features(audio_mono, sr, features)

        # 2. Temporal Features
        features = self._extract_temporal_features(audio_mono, sr, features)

        # 3. Harmonic Features
        features = self._extract_harmonic_features(audio_mono, sr, features)

        # 4. Bandwidth & Dynamic Range
        features = self._extract_bandwidth_features(audio_mono, sr, features)

        # 5. Noise & Artifacts
        features = self._extract_artifact_features(audio_mono, sr, features)

        # 6. Statistical Features
        features = self._extract_statistical_features(audio_mono, features)

        # 7. Stereo Features (if applicable)
        if is_stereo:
            features = self._extract_stereo_features(audio, sr, features)

        if verbose:
            logger.debug("✅ Extracted %s features", len(features.to_array()))

        return features

    def _extract_spectral_features(self, audio: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extrahiert spektrale Features."""
        # Clamp n_fft to the next-lower power of 2 that fits in the signal —
        # prevents librosa from zero-padding and issuing UserWarning.
        safe_n_fft = min(self.n_fft, max(32, 1 << int(np.floor(np.log2(max(1, len(audio)))))))

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=safe_n_fft, hop_length=self.hop_length)[0]
        features.spectral_centroid_mean = np.mean(centroid)  # type: ignore[assignment]
        features.spectral_centroid_std = np.std(centroid)  # type: ignore[assignment]

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=safe_n_fft, hop_length=self.hop_length)[0]
        features.spectral_rolloff_mean = np.mean(rolloff)  # type: ignore[assignment]
        features.spectral_rolloff_std = np.std(rolloff)  # type: ignore[assignment]

        # Spectral Flux (energy difference between frames)
        S = np.abs(librosa.stft(audio, n_fft=safe_n_fft, hop_length=self.hop_length))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        features.spectral_flux_mean = np.mean(flux)  # type: ignore[assignment]

        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=safe_n_fft, hop_length=self.hop_length)[0]
        features.spectral_flatness_mean = np.mean(flatness)  # type: ignore[assignment]

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=safe_n_fft, hop_length=self.hop_length)
        features.spectral_contrast_mean = np.mean(contrast, axis=1)

        return features

    def _extract_temporal_features(self, audio: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extrahiert temporale Features."""
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        features.zero_crossing_rate_mean = np.mean(zcr)  # type: ignore[assignment]

        # RMS Energy
        rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        features.rms_energy_mean = np.mean(rms)  # type: ignore[assignment]
        features.rms_energy_std = np.std(rms)  # type: ignore[assignment]

        # Transient Density (onset detection)
        onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)  # type: ignore[attr-defined]
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr)  # type: ignore[attr-defined]
        duration_sec = len(audio) / sr
        features.transient_density = len(onset_frames) / duration_sec if duration_sec > 0 else 0.0

        # Attack Time (average rise time of onsets)
        if len(onset_frames) > 0:
            attack_times = []
            for onset in onset_frames[:10]:  # First 10 onsets
                onset_sample = librosa.frames_to_samples(onset, hop_length=self.hop_length)
                if onset_sample + 100 < len(audio):
                    envelope = np.abs(audio[onset_sample : onset_sample + 100])
                    peak_idx = np.argmax(envelope)
                    attack_times.append(peak_idx / sr * 1000)  # ms
            features.attack_time_mean = np.mean(attack_times) if attack_times else 0.0  # type: ignore[assignment]

        return features

    def _extract_harmonic_features(self, audio: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extrahiert harmonische Features."""
        # Clamp n_fft so librosa never zero-pads short segments.
        safe_n_fft = min(self.n_fft, max(32, 1 << int(np.floor(np.log2(max(1, len(audio)))))))

        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=safe_n_fft, hop_length=self.hop_length)
        features.mfcc_mean = np.mean(mfcc, axis=1)
        features.mfcc_std = np.std(mfcc, axis=1)

        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=safe_n_fft, hop_length=self.hop_length)
        features.chroma_mean = np.mean(chroma, axis=1)

        return features

    def _extract_bandwidth_features(self, audio: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extrahiert Bandwidth und Dynamic Range Features."""
        # Power Spectral Density
        freqs, psd = scipy_signal.welch(audio, sr, nperseg=min(self.n_fft, len(audio) // 4))

        # 3dB Bandwidth
        max_psd: float = float(np.max(psd))
        threshold_3db = max_psd / 2.0  # -3 dB = half power

        above_threshold = psd > threshold_3db
        if np.any(above_threshold):
            indices = np.where(above_threshold)[0]
            features.bandwidth_3db_low = freqs[indices[0]]
            features.bandwidth_3db_high = freqs[indices[-1]]

        # Dynamic Range
        peak: float = float(np.max(np.abs(audio)))
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            features.dynamic_range_db = 20 * np.log10(peak / rms)
            features.crest_factor = peak / rms

        # Peak-to-RMS ratio
        features.peak_to_rms_db = features.dynamic_range_db

        return features

    def _extract_artifact_features(self, audio: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extrahiert Artifact Detection Features."""
        # Noise Floor (10th percentile of PSD)
        freqs, psd = scipy_signal.welch(audio, sr, nperseg=min(self.n_fft, len(audio) // 4))
        noise_floor_power = np.percentile(psd, 10)
        if noise_floor_power > 0:
            features.noise_floor_db = 10 * np.log10(noise_floor_power)
        else:
            features.noise_floor_db = -100.0

        # Hum Detection (50/60 Hz energy)
        hum_freqs = [50, 60, 100, 120]  # Hz
        hum_energy = 0.0
        for hum_freq in hum_freqs:
            hum_idx = np.argmin(np.abs(freqs - hum_freq))
            hum_energy += psd[hum_idx]
        total_energy: float = float(np.sum(psd))
        features.hum_presence = hum_energy / (total_energy + 1e-10)

        # Click Detection (impulse noise)
        diff = np.abs(np.diff(audio))
        click_threshold = np.percentile(diff, 99.9)
        clicks = np.where(diff > click_threshold)[0]
        duration_sec = len(audio) / sr
        features.clicks_per_second = len(clicks) / duration_sec if duration_sec > 0 else 0.0

        # Wow & Flutter (pitch instability)
        # Simplified detection via Hilbert transform
        try:
            analytic_signal = np.asarray(
                scipy_signal.hilbert(audio[: min(sr * 10, len(audio))]),
                dtype=np.complex128,
            )  # First 10s
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * sr

            # Wow: 0.5-6 Hz modulation
            sos_wow = scipy_signal.butter(2, [0.5, 6], btype="band", fs=sr, output="sos")
            wow_component = scipy_signal.sosfilt(sos_wow, instantaneous_freq)
            features.wow_strength = np.std(wow_component) / (np.mean(np.abs(instantaneous_freq)) + 1e-10)  # type: ignore[assignment]

            # Flutter: 6-100 Hz modulation
            sos_flutter = scipy_signal.butter(2, [6, 100], btype="band", fs=sr, output="sos")
            flutter_component = scipy_signal.sosfilt(sos_flutter, instantaneous_freq)
            features.flutter_strength = np.std(flutter_component) / (np.mean(np.abs(instantaneous_freq)) + 1e-10)  # type: ignore[assignment]
        except Exception:
            features.wow_strength = 0.0
            features.flutter_strength = 0.0

        return features

    def _extract_statistical_features(self, audio: np.ndarray, features: AudioFeatures) -> AudioFeatures:
        """Extrahiert statistische Features."""
        # Kurtosis (tailedness of distribution)
        features.audio_kurtosis = kurtosis(audio)

        # Skewness (asymmetry of distribution)
        features.audio_skewness = skew(audio)

        return features

    def _extract_stereo_features(self, audio: np.ndarray, sr: int, features: AudioFeatures) -> AudioFeatures:
        """Extrahiert Stereo-spezifische Features."""
        if audio.ndim != 2 or audio.shape[1] != 2:
            return features

        left = audio[:, 0]
        right = audio[:, 1]

        # Stereo Width (M/S ratio)
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_energy: float = float(np.sum(mid**2))
        side_energy: float = float(np.sum(side**2))
        features.stereo_width = side_energy / (mid_energy + 1e-10)

        # Phase Correlation — guarded dot-product (§VERBOTEN: np.corrcoef on non-guarded signals)
        if len(left) == len(right):
            _lc = left.astype(float) - float(np.mean(left))
            _rc = right.astype(float) - float(np.mean(right))
            _nl = float(np.linalg.norm(_lc))
            _nr = float(np.linalg.norm(_rc))
            if _nl < 1e-12 or _nr < 1e-12:
                features.phase_correlation = 0.0
            else:
                features.phase_correlation = float(np.clip(np.dot(_lc, _rc) / (_nl * _nr), -1.0, 1.0))

        # Channel Imbalance
        left_rms = np.sqrt(np.mean(left**2))
        right_rms = np.sqrt(np.mean(right**2))
        if right_rms > 0:
            features.channel_imbalance_db = 20 * np.log10(left_rms / right_rms)

        return features

    def extract_batch(self, audio_list: list[np.ndarray], sr: int, verbose: bool = False) -> list[AudioFeatures]:
        """
        Extrahiert Features für eine Liste von Audio-Signalen.

        Args:
            audio_list: Liste von Audio-Signalen
            sr: Sample rate
            verbose: Print progress

        Returns:
            Liste von AudioFeatures
        """
        features_list = []

        for i, audio in enumerate(audio_list):
            if verbose and i % 10 == 0:
                logger.debug("  Processing %s/%s...", i + 1, len(audio_list))

            features = self.extract_all(audio, sr, verbose=False)
            features_list.append(features)

        return features_list

    def features_to_matrix(self, features_list: list[AudioFeatures]) -> np.ndarray:
        """
        Konvertiert Liste von AudioFeatures zu Feature-Matrix für ML.

        Args:
            features_list: Liste von AudioFeatures

        Returns:
            Feature-Matrix (n_samples, n_features)
        """
        feature_arrays = [f.to_array() for f in features_list]
        return np.vstack(feature_arrays)  # type: ignore[no-any-return]
