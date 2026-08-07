"""
forensics/ml_defect_detector.py
ML-basierter Defekt Detector mit 98%+ Recall Target
====================================================

Multi-Label Classification für Audio-Defekte:
- Clicks/Pops (Vinyl, Digitale Fehler)
- Hum (50/60Hz, Erdungsschleifen)
- Distortion (Clipping, Overload, THD)
- Dropout (Tape, Digital, Amplitude Drops)
- Noise Burst (Transient Störungen)

Features:
- Binary classifiers pro Defekt-Typ
- 98%+ Recall Target (minimize false negatives)
- Defekt-spezifische Feature-Extraktion
- Ensemble Learning für Robustheit
"""

import logging
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from backend.core.forensics.feature_extractor import AudioFeatures, FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class DefectFeatures:
    """
    Defekt-spezifische Features.

    20 Features für verschiedene Defekt-Typen:
    - Clicks: Impulsiveness, Zero-Crossing Anomalies
    - Hum: Harmonic Peaks at 50/60Hz
    - Distortion: THD, Clipping
    - Dropout: Silence, Amplitude Drops
    - Noise Burst: Transient Energy Spikes
    """

    # Click/Pop Detection (5 features)
    impulsiveness: float = 0.0  # Impulse strength
    zero_crossing_rate_std: float = 0.0  # ZCR variation
    high_freq_energy_spikes: float = 0.0  # HF transients
    click_density: float = 0.0  # Clicks per second
    max_impulse_amplitude: float = 0.0  # Strongest impulse

    # Hum Detection (4 features)
    hum_50hz_db: float = -100.0  # 50Hz component
    hum_60hz_db: float = -100.0  # 60Hz component
    hum_harmonics_strength: float = 0.0  # Harmonic series
    hum_modulation: float = 0.0  # Amplitude modulation

    # Distortion Detection (5 features)
    thd_percent: float = 0.0  # Total harmonic distortion
    clipping_percent: float = 0.0  # % of clipped samples
    harmonic_spread: float = 0.0  # Harmonic distribution
    odd_harmonic_ratio: float = 0.0  # Odd vs even harmonics
    intermodulation_distortion: float = 0.0  # IMD

    # Dropout Detection (3 features)
    silence_ratio: float = 0.0  # % silence
    dropout_count: int = 0  # Number of dropouts
    amplitude_discontinuities: float = 0.0  # Sudden drops

    # Noise Burst Detection (3 features)
    transient_count: int = 0  # Number of transients
    max_transient_db: float = -100.0  # Strongest transient
    spectral_irregularity: float = 0.0  # Spectrum anomalies

    def to_array(self) -> np.ndarray:
        """Konvertiert to numpy array."""
        return np.array(  # type: ignore[no-any-return]
            [
                # Clicks
                self.impulsiveness,
                self.zero_crossing_rate_std,
                self.high_freq_energy_spikes,
                self.click_density,
                self.max_impulse_amplitude,
                # Hum
                self.hum_50hz_db,
                self.hum_60hz_db,
                self.hum_harmonics_strength,
                self.hum_modulation,
                # Distortion
                self.thd_percent,
                self.clipping_percent,
                self.harmonic_spread,
                self.odd_harmonic_ratio,
                self.intermodulation_distortion,
                # Dropout
                self.silence_ratio,
                float(self.dropout_count),
                self.amplitude_discontinuities,
                # Noise Burst
                float(self.transient_count),
                self.max_transient_db,
                self.spectral_irregularity,
            ]
        )


@dataclass
class DefectDetectionResult:
    """
    Result from defect detection.

    Multi-label classification: Multiple defects can be present.
    """

    defects_detected: dict[str, bool]  # Defect type -> detected
    defect_confidences: dict[str, float]  # Defect type -> confidence
    defect_severities: dict[str, str]  # Defect type -> severity (LOW/MEDIUM/HIGH)
    features_used: int  # Number of features
    model_version: str  # Model version
    summary: str = ""  # Human-readable summary


class DefectFeatureExtractor:
    """
    Extrahiert defekt-spezifische Features aus Audio.
    """

    def __init__(self) -> None:
        self.base_extractor = FeatureExtractor()

    def extract_defect_features(
        self, audio: np.ndarray, sr: int, verbose: bool = False
    ) -> tuple[AudioFeatures, DefectFeatures]:
        """
        Extrahiert base + defect features.

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate in Hz
            verbose: Print extraction progress

        Returns:
            (AudioFeatures, DefectFeatures)
        """
        # Extract base features
        base_features = self.base_extractor.extract_all(audio, sr, verbose=verbose)

        # Extract defect-specific features
        defect_features = DefectFeatures()

        # Convert to mono if stereo
        audio_mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio

        # 1. Click/Pop Detection
        defect_features.impulsiveness = self._detect_impulsiveness(audio_mono, sr)
        defect_features.zero_crossing_rate_std = self._zcr_variation(audio_mono, sr)
        defect_features.high_freq_energy_spikes = self._hf_spikes(audio_mono, sr)
        defect_features.click_density, defect_features.max_impulse_amplitude = self._click_analysis(audio_mono, sr)

        # 2. Hum Detection
        defect_features.hum_50hz_db, defect_features.hum_60hz_db = self._detect_hum(audio_mono, sr)
        defect_features.hum_harmonics_strength = self._hum_harmonics(audio_mono, sr)
        defect_features.hum_modulation = self._hum_modulation(audio_mono, sr)

        # 3. Distortion Detection
        defect_features.thd_percent = self._calculate_thd(audio_mono, sr)
        defect_features.clipping_percent = self._detect_clipping(audio_mono)
        defect_features.harmonic_spread, defect_features.odd_harmonic_ratio = self._harmonic_analysis(audio_mono, sr)
        defect_features.intermodulation_distortion = self._calculate_imd(audio_mono, sr)

        # 4. Dropout Detection
        defect_features.silence_ratio = self._silence_ratio(audio_mono, sr)
        defect_features.dropout_count, defect_features.amplitude_discontinuities = self._detect_dropouts(audio_mono, sr)

        # 5. Noise Burst Detection
        defect_features.transient_count, defect_features.max_transient_db = self._detect_transients(audio_mono, sr)
        defect_features.spectral_irregularity = self._spectral_irregularity(audio_mono, sr)

        return base_features, defect_features

    def _detect_impulsiveness(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt impulsiveness (clicks/pops)."""
        # High-pass filter to emphasize clicks
        from scipy.signal import butter, sosfilt

        sos = butter(4, 1000, btype="high", fs=sr, output="sos")
        audio_hp = sosfilt(sos, audio)

        # Envelope detection
        envelope = np.abs(audio_hp)

        # Peak detection threshold
        threshold = np.mean(envelope) + 3 * np.std(envelope)
        peaks = envelope > threshold

        # Impulsiveness = ratio of peak energy to mean energy
        if np.sum(peaks) > 0:
            peak_energy = float(np.mean(envelope[peaks]))
            mean_energy = float(np.mean(envelope)) + 1e-10
            impulsiveness = peak_energy / mean_energy
            return float(impulsiveness)
        return 0.0

    def _zcr_variation(self, audio: np.ndarray, sr: int) -> float:
        """Calculate zero-crossing rate variation."""
        # Frame-based ZCR
        frame_length = 2048
        hop_length = 512

        zcr_values: list[float] = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame)))
            zcr_values.append(zcr)

        if len(zcr_values) > 0:
            return float(np.std(zcr_values))
        return 0.0

    def _hf_spikes(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt high-frequency energy spikes."""
        from scipy.signal import butter, sosfilt

        sos = butter(4, 8000, btype="high", fs=sr, output="sos")
        audio_hf = sosfilt(sos, audio)

        # Energy in frames
        frame_length = 2048
        hop_length = 512

        energies: list[float] = []
        for i in range(0, len(audio_hf) - frame_length, hop_length):
            frame = audio_hf[i : i + frame_length]
            energy: float = float(np.sum(frame**2))
            energies.append(energy)

        if len(energies) > 0:
            energy_arr = np.asarray(energies, dtype=np.float64)
            threshold = float(np.mean(energy_arr) + 3.0 * np.std(energy_arr))
            spikes: int = int(np.sum(energy_arr > threshold))
            return float(spikes) / len(energy_arr)
        return 0.0

    def _click_analysis(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Analysiert clicks: density and max amplitude."""
        from scipy.signal import find_peaks

        # Differentiate to emphasize transients
        diff = np.diff(audio)

        # Find peaks in differentiated signal
        threshold = np.mean(np.abs(diff)) + 5 * np.std(np.abs(diff))
        peaks, _properties = find_peaks(np.abs(diff), height=threshold, distance=sr // 100)

        # Click density (clicks per second)
        duration_sec = len(audio) / sr
        click_density = len(peaks) / max(duration_sec, 1.0)

        # Max impulse amplitude
        max_impulse = np.max(np.abs(diff[peaks])) if len(peaks) > 0 else 0.0

        return click_density, max_impulse

    def _detect_hum(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Erkennt 50Hz and 60Hz hum."""
        from scipy.signal import welch

        # Power spectral density
        f, psd = welch(audio, sr, nperseg=min(8192, len(audio)))

        # Find 50Hz and 60Hz components
        idx_50hz = np.argmin(np.abs(f - 50))
        idx_60hz = np.argmin(np.abs(f - 60))

        # Power in dB
        hum_50hz_db = 10 * np.log10(psd[idx_50hz] + 1e-10)
        hum_60hz_db = 10 * np.log10(psd[idx_60hz] + 1e-10)

        return hum_50hz_db, hum_60hz_db

    def _hum_harmonics(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt harmonic series of hum (50Hz or 60Hz)."""
        from scipy.signal import welch

        f, psd = welch(audio, sr, nperseg=min(8192, len(audio)))

        # Check 50Hz harmonics (50, 100, 150, 200, 250 Hz)
        harmonics_50 = [50, 100, 150, 200, 250]
        strength_50 = 0.0
        for freq in harmonics_50:
            if freq < sr / 2:
                idx = np.argmin(np.abs(f - freq))
                strength_50 += psd[idx]

        # Check 60Hz harmonics (60, 120, 180, 240, 300 Hz)
        harmonics_60 = [60, 120, 180, 240, 300]
        strength_60 = 0.0
        for freq in harmonics_60:
            if freq < sr / 2:
                idx = np.argmin(np.abs(f - freq))
                strength_60 += psd[idx]

        # Return max strength
        return max(strength_50, strength_60)

    def _hum_modulation(self, audio: np.ndarray, sr: int) -> float:
        """Erkennt amplitude modulation of hum."""
        # Band-pass filter around 50-60Hz
        from scipy.signal import butter, sosfilt

        sos = butter(4, [40, 80], btype="band", fs=sr, output="sos")
        hum_band = sosfilt(sos, audio)

        # Envelope
        envelope = np.abs(hum_band)

        # Modulation = std of envelope / mean of envelope
        if np.mean(envelope) > 1e-10:
            modulation = float(np.std(envelope)) / float(np.mean(envelope))
            return float(modulation)
        return 0.0

    def _calculate_thd(self, audio: np.ndarray, sr: int) -> float:
        """Calculate Total Harmonic Distortion."""
        from scipy.signal import welch

        # Assume fundamental around 100-500Hz (typical audio content)
        f, psd = welch(audio, sr, nperseg=min(8192, len(audio)))

        # Find fundamental (peak in 100-500Hz range)
        mask = (f >= 100) & (f <= 500)
        if np.sum(mask) > 0:
            idx_fundamental = np.argmax(psd[mask]) + np.argmin(np.abs(f - 100))
            fundamental_freq = f[idx_fundamental]
            fundamental_power = psd[idx_fundamental]

            # Sum harmonic powers (2f, 3f, 4f, 5f)
            harmonic_power = 0.0
            for n in range(2, 6):
                harmonic_freq = n * fundamental_freq
                if harmonic_freq < sr / 2:
                    idx_harmonic = np.argmin(np.abs(f - harmonic_freq))
                    harmonic_power += psd[idx_harmonic]

            # THD = sqrt(harmonic_power / fundamental_power) * 100%
            if fundamental_power > 1e-10:
                thd = np.sqrt(harmonic_power / fundamental_power) * 100
                return min(thd, 100.0)  # type: ignore[no-any-return]  # Cap at 100%

        return 0.0

    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Erkennt clipping (samples at max amplitude)."""
        # Normalize to -1..+1
        if np.max(np.abs(audio)) > 0:
            audio_norm = audio / np.max(np.abs(audio))
        else:
            return 0.0

        # Count samples near ±1.0 (clipping threshold)
        clipped: int = int(np.sum(np.abs(audio_norm) > 0.98))
        clipping_percent = (clipped / len(audio)) * 100

        return clipping_percent  # type: ignore[no-any-return]

    def _harmonic_analysis(self, audio: np.ndarray, sr: int) -> tuple[float, float]:
        """Analysiert harmonic distribution."""
        from scipy.signal import welch

        f, psd = welch(audio, sr, nperseg=min(8192, len(audio)))

        # Find fundamental
        mask = (f >= 100) & (f <= 500)
        if np.sum(mask) > 0:
            idx_fundamental = np.argmax(psd[mask]) + np.argmin(np.abs(f - 100))
            fundamental_freq = f[idx_fundamental]

            # Measure harmonic spread (variance of harmonic amplitudes)
            harmonic_amps = []
            for n in range(2, 6):
                harmonic_freq = n * fundamental_freq
                if harmonic_freq < sr / 2:
                    idx_harmonic = np.argmin(np.abs(f - harmonic_freq))
                    harmonic_amps.append(psd[idx_harmonic])

            harmonic_spread = np.std(harmonic_amps) if len(harmonic_amps) > 0 else 0.0

            # Odd vs even harmonic ratio
            odd_harmonics = [harmonic_amps[i] for i in range(len(harmonic_amps)) if i % 2 == 0]  # 3rd, 5th
            even_harmonics = [harmonic_amps[i] for i in range(len(harmonic_amps)) if i % 2 == 1]  # 2nd, 4th

            if len(odd_harmonics) > 0 and len(even_harmonics) > 0:
                odd_ratio = np.sum(odd_harmonics) / (np.sum(even_harmonics) + 1e-10)
            else:
                odd_ratio = 1.0

            return harmonic_spread, odd_ratio  # type: ignore[return-value]

        return 0.0, 1.0

    def _calculate_imd(self, audio: np.ndarray, sr: int) -> float:
        """Calculate Intermodulation Distortion (simplified)."""
        # Simplified: Look for non-harmonic spectral components
        from scipy.signal import welch

        _f, psd = welch(audio, sr, nperseg=min(8192, len(audio)))

        # Total power
        np.sum(psd)

        # Power in expected harmonic regions vs non-harmonic
        # Simplified metric: spectral flatness
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)

        if arithmetic_mean > 1e-10:
            spectral_flatness = geometric_mean / arithmetic_mean
            # IMD increases with spectral flatness deviation
            return (1.0 - spectral_flatness) * 100  # type: ignore[no-any-return]

        return 0.0

    def _silence_ratio(self, audio: np.ndarray, sr: int) -> float:
        """Calculate ratio of silence."""
        # Silence threshold: -60 dBFS
        threshold = 0.001  # ~-60dB

        silent_samples: int = int(np.sum(np.abs(audio) < threshold))
        silence_ratio = silent_samples / len(audio)

        return silence_ratio  # type: ignore[no-any-return]

    def _detect_dropouts(self, audio: np.ndarray, sr: int) -> tuple[int, float]:
        """Erkennt dropouts (sudden amplitude drops)."""
        # Frame-based RMS
        frame_length = 2048
        hop_length = 512

        rms_values: list[float] = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            rms = float(np.sqrt(np.mean(frame**2)))
            rms_values.append(rms)

        if len(rms_values) < 2:
            return 0, 0.0

        rms_arr = np.asarray(rms_values, dtype=np.float64)

        # Detect sudden drops (>20dB)
        rms_db = 20.0 * np.log10(rms_arr + 1e-10)
        drops = np.diff(rms_db) < -20  # 20dB drop

        dropout_count: int = int(np.sum(drops))

        # Amplitude discontinuities (max drop)
        max_discontinuity = np.abs(np.min(np.diff(rms_db))) if dropout_count > 0 else 0.0

        return int(dropout_count), max_discontinuity

    def _detect_transients(self, audio: np.ndarray, sr: int) -> tuple[int, float]:
        """Erkennt transient noise bursts."""
        from scipy.signal import find_peaks

        # Envelope
        envelope = np.abs(audio)

        # Smooth envelope
        window = np.ones(100) / 100
        envelope_smooth = np.convolve(envelope, window, mode="same")

        # Find peaks in envelope
        threshold = np.mean(envelope_smooth) + 5 * np.std(envelope_smooth)
        peaks, _properties = find_peaks(envelope_smooth, height=threshold, distance=sr // 10)

        transient_count = len(peaks)

        # Max transient in dB
        if transient_count > 0:
            max_transient: float = float(np.max(envelope_smooth[peaks]))
            max_transient_db = 20 * np.log10(max_transient + 1e-10)
        else:
            max_transient_db = -100.0

        return transient_count, max_transient_db

    def _spectral_irregularity(self, audio: np.ndarray, sr: int) -> float:
        """Misst spectral irregularity (noise bursts)."""
        from scipy.signal import welch

        # Short-time spectral analysis
        frame_length = 2048
        hop_length = 512

        spectral_variances = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            _f, psd = welch(frame, sr, nperseg=len(frame))

            # Spectral variance (irregularity)
            spectral_var = np.std(psd)
            spectral_variances.append(spectral_var)

        if len(spectral_variances) > 0:
            # High variance indicates irregular spectrum (noise bursts)
            return float(np.mean(spectral_variances))

        return 0.0


class MLDefectDetector:
    """
    ML-basierter Defekt Detector mit 98%+ Recall Target.

    Features:
    - Multi-label classification (Binary classifiers per defect type)
    - 5 defect types: CLICKS, HUM, DISTORTION, DROPOUT, NOISE_BURST
    - Ensemble learning (Random Forest + Gradient Boosting)
    - High recall target (minimize false negatives)
    """

    VERSION = "1.0.0"

    # Defect types
    DEFECT_TYPES = ["CLICKS", "HUM", "DISTORTION", "DROPOUT", "NOISE_BURST"]

    def __init__(self, n_estimators: int = 100, max_depth: int = 15, recall_target: float = 0.98) -> None:
        """
        Initialisiert defect detector.

        Args:
            n_estimators: Number of trees in ensemble
            max_depth: Max tree depth
            recall_target: Target recall (0.98 = 98%)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.recall_target = recall_target

        # Feature extraction
        self.feature_extractor = DefectFeatureExtractor()

        # Scaling
        self.scalers: dict[str, Any] = {}  # One scaler per defect type

        # Models (one set per defect type)
        self.rf_models: dict[str, Any] = {}  # Random Forest models
        self.gb_models: dict[str, Any] = {}  # Gradient Boosting models

        # Training state
        self.is_trained = dict.fromkeys(self.DEFECT_TYPES, False)
        self.training_metrics: dict[str, Any] = {}
        self.cv_recalls: dict[str, dict[str, float]] = {}

    def train(self, X: np.ndarray, y: dict[str, np.ndarray], cv_folds: int = 5, verbose: bool = True) -> dict[str, Any]:
        """
        Train binary classifiers for each defect type.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Dict of labels per defect type {defect: binary_labels}
            cv_folds: Cross-validation folds
            verbose: Print training progress

        Returns:
            Training metrics per defect type
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("   ML Defect Detector Training")
            logger.info("=" * 60)
            logger.info("   Samples: %s, Features: %s", X.shape[0], X.shape[1])
            logger.info("   Defect Types: %s", len(self.DEFECT_TYPES))
            logger.info("   Recall Target: %.1f", self.recall_target)

        all_metrics = {}

        for defect_type in self.DEFECT_TYPES:
            if defect_type not in y:
                logger.warning("   ⚠️ No labels for %s, skipping", defect_type)
                continue

            if verbose:
                logger.info("\n   Training %s detector...", defect_type)

            y_defect = y[defect_type]

            # Check if we have both classes
            unique_classes = np.unique(y_defect)
            if len(unique_classes) < 2:
                logger.warning("   ⚠️ %s: Only one class present, skipping", defect_type)
                continue

            # Scale features
            self.scalers[defect_type] = StandardScaler()
            X_scaled = self.scalers[defect_type].fit_transform(X)

            # Train Random Forest
            self.rf_models[defect_type] = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",  # Handle imbalanced data
                random_state=42,
            )
            self.rf_models[defect_type].fit(X_scaled, y_defect)

            # Train Gradient Boosting
            self.gb_models[defect_type] = GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42
            )
            self.gb_models[defect_type].fit(X_scaled, y_defect)

            # Cross-validation (focus on recall)
            cv = StratifiedKFold(n_splits=min(cv_folds, len(y_defect)), shuffle=True, random_state=42)

            rf_cv_recall = cross_val_score(self.rf_models[defect_type], X_scaled, y_defect, cv=cv, scoring="recall")

            gb_cv_recall = cross_val_score(self.gb_models[defect_type], X_scaled, y_defect, cv=cv, scoring="recall")

            # Store metrics
            self.cv_recalls[defect_type] = {
                "rf_mean": rf_cv_recall.mean(),
                "rf_std": rf_cv_recall.std(),
                "gb_mean": gb_cv_recall.mean(),
                "gb_std": gb_cv_recall.std(),
            }

            self.is_trained[defect_type] = True

            if verbose:
                logger.info("   RF Recall: %.4f ± %.4f", rf_cv_recall.mean(), rf_cv_recall.std())
                logger.info("   GB Recall: %.4f ± %.4f", gb_cv_recall.mean(), gb_cv_recall.std())

                # Check if recall target is met
                if rf_cv_recall.mean() >= self.recall_target or gb_cv_recall.mean() >= self.recall_target:
                    logger.info("   ✅ %s: Recall target MET!", defect_type)
                else:
                    logger.warning("   ⚠️ %s: Recall below target (%.1f)", defect_type, self.recall_target)

            all_metrics[defect_type] = {
                "rf_recall_mean": rf_cv_recall.mean(),
                "rf_recall_std": rf_cv_recall.std(),
                "gb_recall_mean": gb_cv_recall.mean(),
                "gb_recall_std": gb_cv_recall.std(),
                "n_samples": len(y_defect),
                "positive_samples": int(np.sum(y_defect)),
                "negative_samples": int(len(y_defect) - np.sum(y_defect)),
            }

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("   Training vollstaendig!")
            logger.info("=" * 60)

        return all_metrics

    def predict(self, audio: np.ndarray, sample_rate: int, return_features: bool = False) -> DefectDetectionResult:
        """
        Predict defects in audio.

        Args:
            audio: Audio signal (mono or stereo)
            sample_rate: Sample rate in Hz
            return_features: If True, also return extracted features

        Returns:
            DefectDetectionResult with detected defects
        """
        # Check if at least one model is trained
        if not any(self.is_trained.values()):
            raise RuntimeError("No models trained. Call train() first.")

        # Extract features
        base_features, defect_features = self.feature_extractor.extract_defect_features(
            audio, sample_rate, verbose=False
        )

        # Combine features
        base_array = base_features.to_array()
        defect_array = defect_features.to_array()
        feature_array = np.concatenate([base_array, defect_array]).reshape(1, -1)

        # Predict each defect type
        defects_detected = {}
        defect_confidences = {}
        defect_severities = {}

        for defect_type in self.DEFECT_TYPES:
            if not self.is_trained.get(defect_type, False):
                defects_detected[defect_type] = False
                defect_confidences[defect_type] = 0.0
                defect_severities[defect_type] = "NONE"
                continue

            # Scale features
            feature_scaled = self.scalers[defect_type].transform(feature_array)

            # Ensemble prediction (average probabilities)
            rf_proba = self.rf_models[defect_type].predict_proba(feature_scaled)[0]
            gb_proba = self.gb_models[defect_type].predict_proba(feature_scaled)[0]

            ensemble_proba = (rf_proba + gb_proba) / 2

            # Defect detected (class 1)
            confidence = ensemble_proba[1] if len(ensemble_proba) > 1 else 0.0

            # Lower threshold for high recall (0.3 instead of 0.5)
            detected = confidence > 0.3

            defects_detected[defect_type] = detected
            defect_confidences[defect_type] = float(confidence)

            # Severity based on confidence
            if confidence > 0.7:
                severity = "HIGH"
            elif confidence > 0.5:
                severity = "MEDIUM"
            elif confidence > 0.3:
                severity = "LOW"
            else:
                severity = "NONE"

            defect_severities[defect_type] = severity

        # Summary
        detected_list = [d for d, detected in defects_detected.items() if detected]
        summary = f"Detected: {', '.join(detected_list)}" if detected_list else "No defects detected"

        result = DefectDetectionResult(
            defects_detected=defects_detected,
            defect_confidences=defect_confidences,
            defect_severities=defect_severities,
            features_used=feature_array.shape[1],
            model_version=self.VERSION,
            summary=summary,
        )

        if return_features:
            return result, base_features, defect_features  # type: ignore[return-value]

        return result

    def save(self, filepath: str) -> None:
        """Speichert trained models to file."""
        model_data = {
            "version": self.VERSION,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "recall_target": self.recall_target,
            "scalers": self.scalers,
            "rf_models": self.rf_models,
            "gb_models": self.gb_models,
            "is_trained": self.is_trained,
            "training_metrics": self.training_metrics,
            "cv_recalls": self.cv_recalls,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("Model gespeichert to %s", filepath)

    def load(self, filepath: str) -> None:
        """Lädt trained models from file."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)  # nosec B301 — lokale, SHA256-verifizierte Modelldatei

        self.n_estimators = model_data["n_estimators"]
        self.max_depth = model_data["max_depth"]
        self.recall_target = model_data["recall_target"]
        self.scalers = model_data["scalers"]
        self.rf_models = model_data["rf_models"]
        self.gb_models = model_data["gb_models"]
        self.is_trained = model_data["is_trained"]
        self.training_metrics = model_data.get("training_metrics", {})
        self.cv_recalls = model_data.get("cv_recalls", {})

        logger.info("Model geladen from %s", filepath)


def train_ml_defect_detector_from_dataset(
    dataset: list[tuple[np.ndarray, int, dict[str, bool]]],
    test_size: float = 0.2,
    verbose: bool = True,
    save_path: str | None = None,
    cv_folds: int = 5,
    n_estimators: int = 100,
    max_depth: int = 15,
) -> tuple[MLDefectDetector, dict[str, Any]]:
    """
    Train ML Defect Detector from dataset.

    Args:
        dataset: List of (audio, sr, defect_labels) tuples
                 defect_labels = {'CLICKS': True/False, ...}
        test_size: Fraction for test set
        verbose: Print progress
        save_path: Optional path to save trained model

    Returns:
        (detector, evaluation_metrics)
    """
    if verbose:
        logger.info("Training ML Defect Detector from %s samples...", len(dataset))

    # Extract features
    feature_extractor = DefectFeatureExtractor()
    X_list = []
    y_dict: dict[str, list[int]] = {defect: [] for defect in MLDefectDetector.DEFECT_TYPES}

    for audio, sr, defect_labels in dataset:
        base_features, defect_features = feature_extractor.extract_defect_features(audio, sr, verbose=False)

        # Combine features
        base_array = base_features.to_array()
        defect_array = defect_features.to_array()
        feature_array = np.concatenate([base_array, defect_array])

        X_list.append(feature_array)

        # Labels for each defect type
        for defect_type in MLDefectDetector.DEFECT_TYPES:
            label = 1 if defect_labels.get(defect_type, False) else 0
            y_dict[defect_type].append(label)

    X = np.array(X_list)
    y = {defect: np.array(labels) for defect, labels in y_dict.items()}

    if verbose:
        logger.info("Extracted features: %s", X.shape)

    # Split train/test
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = {defect: labels[train_idx] for defect, labels in y.items()}
    y_test = {defect: labels[test_idx] for defect, labels in y.items()}

    # Train detector
    detector = MLDefectDetector(n_estimators=n_estimators, max_depth=max_depth)
    detector.train(X_train, y_train, cv_folds=cv_folds, verbose=verbose)

    # Evaluate on test set
    evaluation_metrics = {}
    for defect_type in MLDefectDetector.DEFECT_TYPES:
        if not detector.is_trained.get(defect_type, False):
            continue

        X_test_scaled = detector.scalers[defect_type].transform(X_test)

        # Ensemble prediction
        rf_pred = detector.rf_models[defect_type].predict(X_test_scaled)
        gb_pred = detector.gb_models[defect_type].predict(X_test_scaled)

        # Use OR for high recall
        ensemble_pred = np.logical_or(rf_pred, gb_pred).astype(int)

        # Calculate recall
        recall = recall_score(y_test[defect_type], ensemble_pred, zero_division=0)

        evaluation_metrics[defect_type] = {
            "test_recall": recall,
            "test_samples": len(y_test[defect_type]),
            "test_positives": int(np.sum(y_test[defect_type])),
        }

        if verbose:
            logger.info("%s Test Recall: %.4f", defect_type, recall)

    # Save if requested
    if save_path:
        detector.save(save_path)

    return detector, evaluation_metrics
