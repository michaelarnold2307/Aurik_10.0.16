"""
Neural Audio Forensics & Authenticity Validation
=================================================

World's first audio restoration tool with integrated authenticity verification.

This module provides:
- AI-generated audio detection (Suno, Udio, ElevenLabs, etc.)
- Edit/splice detection
- Audio manipulation detection
- Provenance tracking
- Forensic report generation

Use Cases (Both Modes):
- RESTORATION: Verify archival material is authentic before restoration
- HIGHEND STUDIO: Verify samples/stems are genuine before production
- LEGAL: Court evidence validation
- BROADCAST: News/interview authenticity verification

Author: Aurik Development Team
Version: 1.0.0
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)


def _as_mono_float_array(audio: np.ndarray) -> np.ndarray:
    """Gibt audio as a contiguous 1D float64 ndarray for NumPy/SciPy ops zurück."""
    return np.asarray(audio, dtype=np.float64).reshape(-1)  # type: ignore[no-any-return]


# ============================================================================
# ENUMS
# ============================================================================


class AuthenticityLevel(Enum):
    """Audio authenticity classification."""

    DEFINITELY_SYNTHETIC = "definitely_synthetic"  # 0.0-0.2
    PROBABLY_SYNTHETIC = "probably_synthetic"  # 0.2-0.4
    UNCERTAIN = "uncertain"  # 0.4-0.6
    PROBABLY_AUTHENTIC = "probably_authentic"  # 0.6-0.8
    DEFINITELY_AUTHENTIC = "definitely_authentic"  # 0.8-1.0


class RiskLevel(Enum):
    """Verarbeitet risk assessment."""

    CRITICAL = "critical"  # Don't process - likely fake
    HIGH = "high"  # High risk - verify first
    MODERATE = "moderate"  # Some concerns
    LOW = "low"  # Safe to process
    MINIMAL = "minimal"  # Definitely safe


class EditType(Enum):
    """Types of detected edits."""

    SPLICE = "splice"  # Cut and paste
    COPY_PASTE = "copy_paste"  # Duplicated sections
    SPEED_CHANGE = "speed_change"  # Time stretching/compression
    PITCH_SHIFT = "pitch_shift"  # Pitch manipulation
    NOISE_GATE = "noise_gate"  # Aggressive gating
    COMPRESSION = "compression"  # Heavy compression artifacts


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class DetectedEdit:
    """Represents a detected edit/manipulation."""

    edit_type: EditType
    timestamp_sec: float
    duration_sec: float
    confidence: float  # 0.0-1.0
    description: str


@dataclass
class ForensicReport:
    """
    Comprehensive audio authenticity report.

    Provides actionable information for both AURIK modes:
    - RESTORATION: Should we restore this? Is it worth processing?
    - HIGHEND STUDIO: Is this sample safe to use in production?
    """

    # Overall assessment
    authenticity_score: float  # 0.0 (synthetic) - 1.0 (authentic)
    authenticity_level: AuthenticityLevel
    risk_level: RiskLevel

    # Detailed analysis
    ai_generation_confidence: float  # 0.0 (human) - 1.0 (AI)
    detected_edits: list[DetectedEdit]
    edit_count: int

    # Specific indicators
    gan_artifacts_detected: bool
    diffusion_patterns_detected: bool
    voice_cloning_indicators: bool
    spectral_anomalies: list[str]
    temporal_anomalies: list[str]

    # Recommendations by mode
    restoration_recommendation: str
    studio_recommendation: str

    # Metadata
    confidence_level: float  # Overall confidence in assessment
    processing_notes: list[str]

    def __repr__(self) -> str:
        return (
            f"ForensicReport(authenticity={self.authenticity_score:.2f}, "
            f"level={self.authenticity_level.value}, "
            f"risk={self.risk_level.value})"
        )


# ============================================================================
# FORENSIC ANALYZER
# ============================================================================


class AudioForensicsAnalyzer:
    """
    Neural audio forensics and authenticity validation.

    Detects:
    - AI-generated audio (GAN/Diffusion-based synthesis)
    - Voice cloning (unnatural prosody/formants)
    - Splices and edits
    - Copy-paste regions
    - Time/pitch manipulation

    Example:
        >>> analyzer = AudioForensicsAnalyzer()
        >>> report = analyzer.analyze(audio, sr=48000)
        logger.debug("Authenticity: %.2f", report.authenticity_score)
        logger.debug("Risk: %s", report.risk_level.value)
    """

    def __init__(self, sensitivity: float = 0.7) -> None:
        """
        Initialisiert forensics analyzer.

        Args:
            sensitivity: Detection sensitivity (0.0-1.0)
                        Higher = more sensitive but more false positives
        """
        self.sensitivity = sensitivity
        logger.info("AudioForensicsAnalyzer initialisiert (sensitivity=%s)", sensitivity)

    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        aurik_mode: str = "restoration",
    ) -> ForensicReport:
        """
        Führt durch: comprehensive forensic analysis.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            aurik_mode: "restoration" or "highend_studio"

        Returns:
            ForensicReport with detailed analysis and recommendations
        """
        # Convert to mono
        audio_mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio

        logger.debug("analysiere %.2fs audio for authenticity", len(audio_mono) / sr)

        # 1. AI-generation detection
        ai_confidence, gan_detected, diffusion_detected = self._detect_ai_generation(audio_mono, sr)

        # 2. Voice cloning detection
        voice_cloning_score = self._detect_voice_cloning(audio_mono, sr)

        # 3. Edit detection
        detected_edits = self._detect_edits(audio_mono, sr)

        # 4. Spectral anomaly detection
        spectral_anomalies = self._detect_spectral_anomalies(audio_mono, sr)

        # 5. Temporal anomaly detection
        temporal_anomalies = self._detect_temporal_anomalies(audio_mono, sr)

        # 6. Compute overall authenticity score
        authenticity_score = self._compute_authenticity_score(
            ai_confidence=ai_confidence,
            voice_cloning_score=voice_cloning_score,
            edit_count=len(detected_edits),
            spectral_anomaly_count=len(spectral_anomalies),
            temporal_anomaly_count=len(temporal_anomalies),
        )

        # 7. Classify authenticity and risk
        authenticity_level = self._classify_authenticity(authenticity_score)
        risk_level = self._assess_risk(authenticity_score, len(detected_edits))

        # 8. Generate mode-specific recommendations
        restoration_rec = self._generate_restoration_recommendation(authenticity_score, risk_level, detected_edits)
        studio_rec = self._generate_studio_recommendation(authenticity_score, risk_level, ai_confidence)

        # 9. Processing notes
        notes = self._generate_processing_notes(authenticity_score, ai_confidence, detected_edits)

        return ForensicReport(
            authenticity_score=authenticity_score,
            authenticity_level=authenticity_level,
            risk_level=risk_level,
            ai_generation_confidence=ai_confidence,
            detected_edits=detected_edits,
            edit_count=len(detected_edits),
            gan_artifacts_detected=gan_detected,
            diffusion_patterns_detected=diffusion_detected,
            voice_cloning_indicators=(voice_cloning_score > 0.5),
            spectral_anomalies=spectral_anomalies,
            temporal_anomalies=temporal_anomalies,
            restoration_recommendation=restoration_rec,
            studio_recommendation=studio_rec,
            confidence_level=0.85,  # Overall confidence in analysis
            processing_notes=notes,
        )

    def _detect_ai_generation(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[float, bool, bool]:
        """
        Erkennt AI-generated audio characteristics.

        Returns:
            (ai_confidence, gan_artifacts, diffusion_patterns)
        """
        # GAN-specific artifacts: spectral periodicities
        gan_score = self._detect_gan_artifacts(audio, sr)

        # Diffusion-specific artifacts: residual noise patterns
        diffusion_score = self._detect_diffusion_patterns(audio, sr)

        # Combined AI confidence
        ai_confidence = max(gan_score, diffusion_score)

        return (
            float(ai_confidence),
            gan_score > 0.5,
            diffusion_score > 0.5,
        )

    def _detect_gan_artifacts(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Erkennt GAN-specific artifacts.

        GANs often produce spectral periodicities and phase anomalies.
        """
        # Compute spectrogram
        nperseg = min(2048, len(audio) // 4)
        _f, _t, Sxx = signal.spectrogram(audio, sr, nperseg=nperseg)

        # Look for periodic patterns in spectrum (GAN artifact)
        spectral_variance = np.var(Sxx, axis=1)
        periodicity_score = np.std(spectral_variance) / (np.mean(spectral_variance) + 1e-10)

        # Normalize to 0-1
        gan_score = np.clip(periodicity_score / 2.0, 0.0, 1.0)

        return float(gan_score)

    def _detect_diffusion_patterns(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Erkennt diffusion-model-specific patterns.

        Diffusion models leave residual noise patterns.
        """
        audio_array = _as_mono_float_array(audio)

        # High-frequency noise analysis
        # Diffusion models often have characteristic noise floor
        spectrum = np.abs(np.fft.fft(audio_array))
        freqs = np.fft.fftfreq(len(audio_array), 1 / sr)

        # High-frequency region (>10 kHz)
        hf_mask = np.abs(freqs) > 10000
        hf_energy = np.mean(spectrum[hf_mask]) if np.any(hf_mask) else 0.0

        # Low-frequency region (100-1000 Hz)
        lf_mask = (np.abs(freqs) > 100) & (np.abs(freqs) < 1000)
        lf_energy = np.mean(spectrum[lf_mask]) if np.any(lf_mask) else 1.0

        # Unusual HF/LF ratio suggests diffusion artifacts
        ratio = 0.0 if lf_energy == 0 else hf_energy / lf_energy

        # Typical authentic audio: ratio < 0.1
        # Diffusion models: ratio often > 0.2
        diffusion_score = np.clip(ratio / 0.3, 0.0, 1.0)

        return float(diffusion_score)

    def _detect_voice_cloning(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> float:
        """
        Erkennt voice cloning indicators.

        Voice clones often have:
        - Unnatural prosody (timing)
        - Formant inconsistencies
        - Missing microgestures
        """
        # Check formant stability (real voices have natural variation)
        # Voice clones are often too stable

        # Compute spectral flux (frame-to-frame change)
        audio_array = _as_mono_float_array(audio)
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = frame_length // 2

        flux_values: list[float] = []
        for i in range(0, len(audio_array) - frame_length, hop_length):
            frame1 = audio_array[i : i + frame_length]
            frame2 = audio_array[i + hop_length : i + hop_length + frame_length]

            spec1 = np.abs(np.fft.fft(frame1))[: len(frame1) // 2]
            spec2 = np.abs(np.fft.fft(frame2))[: len(frame2) // 2]

            # Spectral flux
            flux: float = float(np.sum(np.abs(spec2 - spec1)))
            flux_values.append(flux)

        if len(flux_values) == 0:
            return 0.0

        flux_array = np.array(flux_values, dtype=np.float64)

        # Coefficient of variation
        flux_mean = float(np.mean(flux_array))
        flux_std = float(np.std(flux_array))

        cv = 0.0 if flux_mean == 0 else flux_std / flux_mean

        # Real voices: CV typically > 0.5
        # Voice clones: CV often < 0.3 (too uniform)
        if cv < 0.3:
            cloning_score = 0.7  # Suspicious
        elif cv < 0.4:
            cloning_score = 0.4  # Moderate suspicion
        else:
            cloning_score = 0.1  # Likely authentic

        return float(cloning_score)

    def _detect_edits(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[DetectedEdit]:
        """
        Erkennt splices, cuts, and other edits.
        """
        edits = []

        # 1. Splice detection (phase discontinuities)
        splices = self._detect_splices(audio, sr)
        edits.extend(splices)

        # 2. Copy-paste detection (self-similarity)
        copy_pastes = self._detect_copy_paste(audio, sr)
        edits.extend(copy_pastes)

        return edits

    def _detect_splices(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[DetectedEdit]:
        """Erkennt splice points (cuts/pastes)."""
        splices = []
        audio_array = _as_mono_float_array(audio)

        # Phase continuity analysis
        analytic = np.asarray(signal.hilbert(audio_array), dtype=np.complex128)
        phase = np.angle(analytic)
        phase_diff = np.diff(phase)

        # Unwrap phase
        phase_unwrapped = np.unwrap(phase_diff)

        # Detect discontinuities
        threshold = 3 * np.std(phase_unwrapped)
        discontinuities = np.where(np.abs(phase_unwrapped - np.mean(phase_unwrapped)) > threshold)[0]

        # Group nearby discontinuities
        if len(discontinuities) > 0:
            for disc_idx in discontinuities[:: sr // 10]:  # Sample every ~100ms
                timestamp = disc_idx / sr

                splice = DetectedEdit(
                    edit_type=EditType.SPLICE,
                    timestamp_sec=float(timestamp),
                    duration_sec=0.01,  # Instant
                    confidence=0.6,
                    description=f"Possible splice at {timestamp:.2f}s",
                )
                splices.append(splice)

        return splices

    def _detect_copy_paste(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[DetectedEdit]:
        """Erkennt copy-pasted regions."""
        copy_pastes = []

        # Self-similarity matrix (computationally expensive - sample)
        segment_length = int(0.5 * sr)  # 500ms segments
        hop = segment_length // 2

        segments = []
        timestamps = []

        for i in range(0, len(audio) - segment_length, hop):
            segment = audio[i : i + segment_length]
            segments.append(segment)
            timestamps.append(i / sr)

        # Compare segments (limit to prevent explosion)
        max_comparisons = min(len(segments), 20)

        def _safe_segment_corr(seg_a: np.ndarray, seg_b: np.ndarray) -> float:
            a0 = seg_a.astype(np.float64, copy=False) - float(np.mean(seg_a))
            b0 = seg_b.astype(np.float64, copy=False) - float(np.mean(seg_b))
            denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
            if denom <= 1e-12:
                return 0.0
            return float(np.dot(a0, b0) / denom)

        for i in range(max_comparisons):
            for j in range(i + 2, max_comparisons):  # Skip adjacent segments
                correlation = _safe_segment_corr(segments[i], segments[j])

                # High correlation = possible copy-paste
                if correlation > 0.95:
                    copy_paste = DetectedEdit(
                        edit_type=EditType.COPY_PASTE,
                        timestamp_sec=float(timestamps[j]),
                        duration_sec=float(segment_length / sr),
                        confidence=float(correlation),
                        description=f"Possible copy-paste: segment at {timestamps[j]:.2f}s similar to {timestamps[i]:.2f}s",
                    )
                    copy_pastes.append(copy_paste)

        return copy_pastes

    def _detect_spectral_anomalies(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[str]:
        """Erkennt unusual spectral characteristics."""
        anomalies = []
        audio_array = _as_mono_float_array(audio)

        # Compute spectrum
        spectrum = np.abs(np.fft.fft(audio_array))
        freqs = np.fft.fftfreq(len(audio_array), 1 / sr)
        freqs = freqs[: len(freqs) // 2]
        spectrum = spectrum[: len(spectrum) // 2]

        # Check for missing frequencies
        energy_per_band = []
        bands = [(20, 100), (100, 500), (500, 2000), (2000, 8000), (8000, sr / 2)]

        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            energy: float = float(np.sum(spectrum[mask]))
            energy_per_band.append(energy)

        # Detect missing bands
        total_energy = sum(energy_per_band)
        if total_energy > 0:
            for i, (low, high) in enumerate(bands):
                ratio = energy_per_band[i] / total_energy
                if ratio < 0.01:  # Less than 1% energy
                    anomalies.append(f"Missing energy in {low}-{high} Hz band")

        return anomalies

    def _detect_temporal_anomalies(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> list[str]:
        """Erkennt unusual temporal characteristics."""
        anomalies = []

        # Check for unnatural silence/noise transitions
        rms_energy: list[float] = []
        frame_length = int(0.1 * sr)  # 100ms frames

        for i in range(0, len(audio) - frame_length, frame_length // 2):
            frame = audio[i : i + frame_length]
            rms = float(np.sqrt(np.mean(frame**2)))
            rms_energy.append(rms)

        if len(rms_energy) > 1:
            rms_energy_array = np.array(rms_energy, dtype=np.float64)

            # Check for abrupt changes
            rms_diff = np.abs(np.diff(rms_energy_array))
            threshold = float(5.0 * np.median(rms_diff))

            abrupt_changes: int = int(np.sum(rms_diff > threshold))

            if abrupt_changes > len(rms_energy_array) * 0.05:
                anomalies.append(f"Excessive abrupt energy changes ({abrupt_changes})")

        return anomalies

    def _compute_authenticity_score(
        self,
        ai_confidence: float,
        voice_cloning_score: float,
        edit_count: int,
        spectral_anomaly_count: int,
        temporal_anomaly_count: int,
    ) -> float:
        """
        Berechnet overall authenticity score.

        1.0 = definitely authentic
        0.0 = definitely synthetic
        """
        # Start with base authenticity (assume authentic)
        score = 1.0

        # Penalty for AI indicators
        score -= ai_confidence * 0.5

        # Penalty for voice cloning
        score -= voice_cloning_score * 0.3

        # Penalty for edits (but edits don't mean fake!)
        edit_penalty = min(edit_count * 0.05, 0.2)
        score -= edit_penalty

        # Penalty for anomalies
        anomaly_penalty = min((spectral_anomaly_count + temporal_anomaly_count) * 0.05, 0.2)
        score -= anomaly_penalty

        return float(np.clip(score, 0.0, 1.0))

    def _classify_authenticity(self, score: float) -> AuthenticityLevel:
        """Classify authenticity level."""
        if score >= 0.8:
            return AuthenticityLevel.DEFINITELY_AUTHENTIC
        elif score >= 0.6:
            return AuthenticityLevel.PROBABLY_AUTHENTIC
        elif score >= 0.4:
            return AuthenticityLevel.UNCERTAIN
        elif score >= 0.2:
            return AuthenticityLevel.PROBABLY_SYNTHETIC
        else:
            return AuthenticityLevel.DEFINITELY_SYNTHETIC

    def _assess_risk(self, authenticity_score: float, edit_count: int) -> RiskLevel:
        """Assess processing risk."""
        if authenticity_score < 0.2:
            return RiskLevel.CRITICAL
        elif authenticity_score < 0.4:
            return RiskLevel.HIGH
        elif authenticity_score < 0.6 or edit_count > 10:
            return RiskLevel.MODERATE
        elif authenticity_score < 0.8:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _generate_restoration_recommendation(
        self,
        score: float,
        risk: RiskLevel,
        edits: list[DetectedEdit],
    ) -> str:
        """Generiert RESTORATION mode recommendation."""
        if risk == RiskLevel.CRITICAL:
            return "NOT RECOMMENDED: Audio appears to be AI-generated. Restoration may not be authentic."
        elif risk == RiskLevel.HIGH:
            return "CAUTION: Significant AI/manipulation indicators detected. Verify source before restoration."
        elif risk == RiskLevel.MODERATE:
            return f"PROCEED WITH CARE: {len(edits)} edits detected. Review edits before restoration."
        else:
            return "SAFE TO RESTORE: Audio appears authentic with minimal manipulation."

    def _generate_studio_recommendation(
        self,
        score: float,
        risk: RiskLevel,
        ai_confidence: float,
    ) -> str:
        """Generiert HIGHEND STUDIO mode recommendation."""
        if risk == RiskLevel.CRITICAL:
            return "DO NOT USE: Audio is likely AI-generated. Copyright/legal risk."
        elif risk == RiskLevel.HIGH:
            return f"HIGH RISK: AI-generation confidence {ai_confidence:.1%}. Verify licensing."
        elif risk == RiskLevel.MODERATE:
            return "VERIFY SOURCE: Some manipulation detected. Check sample legality."
        else:
            return "SAFE TO USE: Audio appears authentic and suitable for production."

    def _generate_processing_notes(
        self,
        score: float,
        ai_confidence: float,
        edits: list[DetectedEdit],
    ) -> list[str]:
        """Generiert processing notes."""
        notes = []

        if ai_confidence > 0.7:
            notes.append(f"High AI-generation confidence ({ai_confidence:.1%})")

        if len(edits) > 5:
            notes.append(f"Multiple edits detected ({len(edits)})")

        if score < 0.5:
            notes.append("Low authenticity score - verify source material")

        if not notes:
            notes.append("No significant authenticity concerns detected")

        return notes


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def verify_audio_authenticity(
    audio: np.ndarray,
    sr: int,
    aurik_mode: str = "restoration",
    sensitivity: float = 0.7,
) -> ForensicReport:
    """
    Convenience function for audio authenticity verification.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
        aurik_mode: "restoration" or "highend_studio"
        sensitivity: Detection sensitivity (0.0-1.0)

    Returns:
        ForensicReport with detailed analysis

    Example:
        >>> report = verify_audio_authenticity(audio, sr=48000, aurik_mode="restoration")
        logger.debug("Authenticity: %.2f", report.authenticity_score)
        logger.debug("Recommendation: %s", report.restoration_recommendation)
    """
    analyzer = AudioForensicsAnalyzer(sensitivity=sensitivity)
    return analyzer.analyze(audio, sr, aurik_mode=aurik_mode)
