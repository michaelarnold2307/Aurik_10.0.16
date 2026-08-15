"""
AURIK Aesthetic Judgment Model (AJM)

Implements the Composite Aesthetic Score (CAS) calculation per AURIK Architecture Specification.
Translates subjective musical goals into measurable proxy metrics and optimizes aesthetic quality.

Key Features:
- 7 aesthetic dimensions (Brilliance, Transparency, Naturalness, Authenticity, Emotionality, Warmth, Spatiality)
- Genre-adaptive weighting (Classical, Jazz, Rock/Metal, Electronic, Vocal/Pop, Vintage/Analog)
- Constraint system enforcement (Spec 3.2.3)
- CAS formula implementation: CAS = Σ (wᵢ × normalized_proxyᵢ) × (1 - penalty_artifacts) × authenticity_factor

Spec Reference:
- Section 1.2: Musikalische Zielgrößen
- Section 3.2.1: Composite Aesthetic Score
- Section 3.2.2: Genre-adaptive Gewichtung
- Section 3.2.3: Constraint System
"""

import logging
from typing import Any

import numpy as np

from .data_models import (
    AestheticScores,
    AnalysisProfile,
    ConstraintCheckResult,
    Genre,
    GenreWeights,
    QualityReport,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Genre Weight Registry (Spec 3.2.2)
# ============================================================================


class GenreWeightRegistry:
    """
    Genre-adaptive weighting for aesthetic dimensions.

    Spec Reference: Section 3.2.2 - Genre-adaptive Gewichtung

    Each genre prioritizes different aesthetic qualities:
    - Classical: Emphasizes transparency, naturalness, spatiality
    - Jazz: Emphasizes warmth, transparency, emotionality
    - Rock/Metal: Emphasizes emotionality, warmth, brilliance
    - Electronic: Emphasizes brilliance, spatiality, transparency
    - Vocal/Pop: Emphasizes emotionality, brilliance, transparency
    - Vintage/Analog: Emphasizes authenticity, warmth, naturalness
    """

    # Exact weights from Spec Table 3.2.2
    GENRE_WEIGHTS = {
        Genre.CLASSICAL: GenreWeights(
            brilliance=0.10,
            transparency=0.20,
            naturalness=0.20,
            authenticity=0.15,
            emotionality=0.15,
            warmth=0.05,
            spatiality=0.15,
        ),
        Genre.JAZZ: GenreWeights(
            brilliance=0.10,
            transparency=0.15,
            naturalness=0.15,
            authenticity=0.10,
            emotionality=0.15,
            warmth=0.20,
            spatiality=0.15,
        ),
        Genre.ROCK_METAL: GenreWeights(
            brilliance=0.15,
            transparency=0.10,
            naturalness=0.10,
            authenticity=0.15,
            emotionality=0.20,
            warmth=0.20,
            spatiality=0.10,
        ),
        Genre.ELECTRONIC: GenreWeights(
            brilliance=0.20,
            transparency=0.15,
            naturalness=0.05,
            authenticity=0.10,
            emotionality=0.15,
            warmth=0.15,
            spatiality=0.20,
        ),
        Genre.VOCAL_POP: GenreWeights(
            brilliance=0.15,
            transparency=0.15,
            naturalness=0.15,
            authenticity=0.10,
            emotionality=0.20,
            warmth=0.15,
            spatiality=0.10,
        ),
        Genre.VINTAGE_ANALOG: GenreWeights(
            brilliance=0.05,
            transparency=0.10,
            naturalness=0.15,
            authenticity=0.25,
            emotionality=0.10,
            warmth=0.25,
            spatiality=0.10,
        ),
        Genre.UNKNOWN: GenreWeights(
            # Balanced default weights
            brilliance=0.14,
            transparency=0.14,
            naturalness=0.14,
            authenticity=0.15,
            emotionality=0.15,
            warmth=0.14,
            spatiality=0.14,
        ),
    }

    @classmethod
    def get_weights(cls, genre: Genre, confidence: float = 1.0) -> GenreWeights:
        """
        Gibt zurück: genre-adaptive weights with confidence-based blending.

        Args:
            genre: Detected genre
            confidence: Genre detection confidence (0-1)

        Returns:
            GenreWeights for the specified genre

        Note:
            If confidence < 0.8, blends genre weights with default weights
        """
        genre_weights = cls.GENRE_WEIGHTS[genre]

        if confidence < 0.8:
            # Blend with default weights for low confidence
            default_weights = cls.GENRE_WEIGHTS[Genre.UNKNOWN]
            blended = GenreWeights(
                brilliance=confidence * genre_weights.brilliance + (1 - confidence) * default_weights.brilliance,
                transparency=confidence * genre_weights.transparency + (1 - confidence) * default_weights.transparency,
                naturalness=confidence * genre_weights.naturalness + (1 - confidence) * default_weights.naturalness,
                authenticity=confidence * genre_weights.authenticity + (1 - confidence) * default_weights.authenticity,
                emotionality=confidence * genre_weights.emotionality + (1 - confidence) * default_weights.emotionality,
                warmth=confidence * genre_weights.warmth + (1 - confidence) * default_weights.warmth,
                spatiality=confidence * genre_weights.spatiality + (1 - confidence) * default_weights.spatiality,
            )
            return blended

        return genre_weights

    @classmethod
    def validate_all_weights(cls) -> bool:
        """Validiert that all genre weight tables sum to 1.0."""
        for genre, weights in cls.GENRE_WEIGHTS.items():
            if not weights.validate_sum():
                logger.warning("Genre %s weights do not sum to 1.0", genre)
                return False
        return True


# ============================================================================
# Aesthetic Proxy Calculator
# ============================================================================


class AestheticProxyCalculator:
    """
    Calculates proxy metrics for all 7 aesthetic dimensions.

    Spec Reference: Section 1.2 - Musikalische Zielgrößen

    Each dimension has multiple technical proxies that serve as measurable
    indicators (though not absolute truth) of subjective aesthetic quality.
    """

    @staticmethod
    def calculate_brilliance_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Brilliance score (Spec 1.2: Brillanz).

        Proxies:
        - High-Frequency Energy Ratio (12-20kHz)
        - Spectral Centroid (normalized)
        - Air-Band Presence (15-20kHz)
        - Harmonic Brightness Index

        Args:
            audio: Audio signal (mono or stereo)
            sr: Sample rate
            profile: Analysis profile with pre-computed features

        Returns:
            Tuple of (brilliance_score, proxy_details)
        """
        details = {}

        # 1. HF Energy Ratio (12-20kHz)
        nyquist = sr / 2
        if nyquist >= 20000:
            fft = np.fft.rfft(audio.flatten())
            freqs = np.fft.rfftfreq(len(audio.flatten()), 1 / sr)

            hf_mask = (freqs >= 12000) & (freqs <= 20000)
            hf_energy: float = float(np.sum(np.abs(fft[hf_mask]) ** 2))
            total_energy: float = float(np.sum(np.abs(fft) ** 2))

            hf_ratio = hf_energy / (total_energy + 1e-10)
            details["hf_energy_ratio"] = float(hf_ratio)
        else:
            hf_ratio = 0.0
            details["hf_energy_ratio"] = 0.0

        # 2. Spectral Centroid (normalized to 0-1, typical range 1000-8000 Hz)
        spectral_centroid = profile.spectral.spectral_centroid
        centroid_norm = np.clip((spectral_centroid - 1000) / 7000, 0.0, 1.0)
        details["spectral_centroid_normalized"] = float(centroid_norm)

        # 3. Air-Band Presence (15-20kHz)
        if nyquist >= 20000:
            air_mask = (freqs >= 15000) & (freqs <= 20000)
            air_energy: float = float(np.sum(np.abs(fft[air_mask]) ** 2))
            air_presence = air_energy / (total_energy + 1e-10)
            details["air_band_presence"] = float(air_presence)
        else:
            air_presence = 0.0
            details["air_band_presence"] = 0.0

        # 4. Harmonic Brightness (approximated from spectral rolloff)
        rolloff = profile.spectral.spectral_rolloff
        brightness_index = np.clip((rolloff - 4000) / 12000, 0.0, 1.0)
        details["harmonic_brightness_index"] = float(brightness_index)

        # Weighted combination
        brilliance = 0.3 * hf_ratio + 0.2 * centroid_norm + 0.3 * air_presence + 0.2 * brightness_index
        brilliance = np.clip(brilliance, 0.0, 1.0)

        return float(brilliance), details

    @staticmethod
    def calculate_transparency_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Transparency score (Spec 1.2: Transparenz).

        Proxies:
        - Spectral Flatness
        - Inter-Source Masking Index (estimated)
        - Transient Sharpness
        - Frequency-Band Separation Score

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (transparency_score, proxy_details)
        """
        details: dict[str, Any] = {}

        # 1. Spectral Flatness — aus Audio berechnen (SpectralAnalysis hat kein spectral_flatness-Feld)
        try:
            import librosa as _lr_sf

            _audio_mono_sf = audio.flatten().astype(np.float32) if audio.ndim > 1 else audio.astype(np.float32)
            _sf_vals = _lr_sf.feature.spectral_flatness(y=_audio_mono_sf)
            spectral_flatness = float(np.mean(_sf_vals))
        except Exception:
            spectral_flatness = 0.5
        details["spectral_flatness"] = float(spectral_flatness)

        # 2. Inter-Source Masking Index via Spectral Contrast (Moore & Glasberg 1983)
        # Higher contrast across frequency bands → less inter-source masking → higher transparency
        try:
            import librosa as _lr

            audio_mono = audio.flatten()
            S = np.abs(_lr.stft(audio_mono, n_fft=2048, hop_length=512))
            contrast = _lr.feature.spectral_contrast(S=S, sr=sr, n_bands=6, fmin=200.0)
            # Mean contrast across bands and frames; typical range 10-40 dB
            mean_contrast = float(np.mean(contrast))
            masking_estimate = float(np.clip((mean_contrast - 10.0) / 30.0, 0.0, 1.0))
        except Exception:
            masking_estimate = 0.7
        details["inter_source_masking_index"] = masking_estimate

        # 3. Transient Sharpness (from onset count and strength)
        onset_count = len(profile.feature_vectors.onset_times)
        duration = profile.format_info.sample_rate / sr if audio.size > 0 else 1.0
        onset_density = onset_count / duration if duration > 0 else 0.0
        transient_sharpness = np.clip(onset_density / 10.0, 0.0, 1.0)  # Normalize by typical density
        details["transient_sharpness"] = float(transient_sharpness)

        # 4. Frequency-Band Separation (estimated from crest factor and dynamics)
        crest_factor_db = profile.dynamics.crest_factor_db
        separation_score = np.clip(crest_factor_db / 20.0, 0.0, 1.0)
        details["frequency_band_separation"] = float(separation_score)

        # Weighted combination
        transparency = (
            0.25 * spectral_flatness + 0.35 * masking_estimate + 0.20 * transient_sharpness + 0.20 * separation_score
        )
        transparency = np.clip(transparency, 0.0, 1.0)

        return float(transparency), details

    @staticmethod
    def calculate_naturalness_score(
        audio: np.ndarray,
        sr: int,
        profile: AnalysisProfile,
        original_audio: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        """
        Calculate Naturalness score (Spec 1.2: Natürlichkeit).

        Proxies:
        - Foundation Model Deviation Score (MERT-Cosine-Similarity, Fallback: CLAP)
        - Artifact Likelihood Estimator
        - Harmonic Distortion Profile

        Args:
            audio: Audio signal (restauriertes Audio)
            sr: Sample rate
            profile: Analysis profile
            original_audio: Original-Audio für MERT-Similarity (optional)

        Returns:
            Tuple of (naturalness_score, proxy_details)
        """
        details: dict[str, Any] = {}

        # 1. Artifact Likelihood (inverted - lower=more natural)
        artifact_severity_sum = sum(d.severity for d in profile.detected_defects)
        artifact_likelihood = np.clip(artifact_severity_sum / 5.0, 0.0, 1.0)
        naturalness_from_artifacts = 1.0 - artifact_likelihood
        details["artifact_likelihood"] = float(artifact_likelihood)

        # 2. Foundation Model Deviation (CLAP-based; fallback to quality proxy)
        foundation_deviation = 1.0 - profile.overall_quality_score
        _foundation_source = "overall_quality_proxy"
        try:
            _audio_np = np.asarray(audio, dtype=np.float32)
            _audio_mono = _audio_np
            if _audio_np.ndim == 2:
                if _audio_np.shape[1] <= 2 and _audio_np.shape[0] > _audio_np.shape[1]:
                    _audio_mono = _audio_np.mean(axis=1)
                elif _audio_np.shape[0] <= 2 and _audio_np.shape[1] > _audio_np.shape[0]:
                    _audio_mono = _audio_np.mean(axis=0)
                else:
                    _audio_mono = _audio_np.mean(axis=-1)

            # Runtime budget: CLAP auf repräsentativem 30s Zentrumsslice.
            _max_len = int(sr * 30)
            if _audio_mono.size > _max_len:
                _s = (_audio_mono.size - _max_len) // 2
                _audio_mono = _audio_mono[_s : _s + _max_len]

            if int(sr) == 48000 and _audio_mono.size >= 2048:
                from plugins.laion_clap_plugin import get_laion_clap, get_loaded_laion_clap

                _clap = get_loaded_laion_clap()
                if _clap is None:
                    _clap = get_laion_clap()
                _tag_res = _clap.tag(_audio_mono, sr)

                _genre_map = {
                    Genre.CLASSICAL: "classical",
                    Genre.JAZZ: "jazz",
                    Genre.ROCK_METAL: "rock",
                    Genre.ELECTRONIC: "electronic",
                    Genre.VOCAL_POP: "pop",
                    Genre.VINTAGE_ANALOG: "blues",
                    Genre.UNKNOWN: "pop",
                }
                _genre_key = _genre_map.get(profile.musical_context.genre, "pop")
                _genre_match = float(np.clip(_tag_res.genre_tags.get(_genre_key, 0.0), 0.0, 1.0))
                _dominant_genre = (
                    float(np.clip(max(_tag_res.genre_tags.values()), 0.0, 1.0)) if _tag_res.genre_tags else 0.0
                )
                _model_conf = float(np.clip(_tag_res.confidence, 0.0, 1.0))
                _semantic_alignment = float(np.clip(0.65 * _genre_match + 0.35 * _dominant_genre, 0.0, 1.0))

                # Niedrige Abweichung = hohe Foundation-Nähe.
                _foundation_similarity = float(np.clip(0.55 * _model_conf + 0.45 * _semantic_alignment, 0.0, 1.0))
                foundation_deviation = 1.0 - _foundation_similarity
                _foundation_source = str(_tag_res.model_used)
                details["foundation_model_confidence"] = _model_conf
                details["foundation_genre_match"] = _genre_match
                details["foundation_semantic_alignment"] = _semantic_alignment
        except Exception as exc:
            logger.warning("§G23 Naturalness CLAP foundation deviation DSP-Ersatzpfad: %s", exc, exc_info=True)

        # Foundation Model Deviation Score: MERT-Cosine-Similarity als Proxy
        # Berechne Ähnlichkeit zwischen Original und Restauriert im MERT-Embedding-Raum.
        # Verbessert foundation_deviation wenn Original verfügbar (Blending 50/50 mit CLAP).
        if original_audio is not None:
            try:
                from backend.core.musical_goals.musical_goals_metrics import (
                    _compute_mert_similarity,  # pylint: disable=import-outside-toplevel
                )

                _fmds = float(_compute_mert_similarity(original_audio, audio, sr))
                foundation_model_deviation_score = max(0.0, 1.0 - _fmds)  # Abweichung, nicht Ähnlichkeit
                # Blend: 50 % CLAP-Abweichung + 50 % MERT-Abweichung
                foundation_deviation = 0.5 * foundation_deviation + 0.5 * foundation_model_deviation_score
                details["foundation_mert_deviation"] = float(foundation_model_deviation_score)
                details["foundation_model_source"] = (
                    str(details.get("foundation_model_source", _foundation_source)) + "+mert"
                )
            except Exception:
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
                foundation_model_deviation_score = 0.0  # Graceful fallback
        naturalness_from_foundation = 1.0 - foundation_deviation
        details["foundation_model_deviation"] = float(foundation_deviation)
        if "foundation_model_source" not in details:
            details["foundation_model_source"] = _foundation_source

        # 3. Harmonic Distortion Profile (from harmonicity if available)
        if profile.feature_vectors.harmonicity is not None:
            harmonicity = profile.feature_vectors.harmonicity
            # High harmonicity suggests low distortion
            naturalness_from_harmonics = np.clip(harmonicity / 30.0, 0.0, 1.0)
        else:
            naturalness_from_harmonics = 0.7  # Default
        details["harmonic_distortion_profile"] = float(1.0 - naturalness_from_harmonics)

        # Weighted combination
        naturalness = (
            0.4 * naturalness_from_artifacts + 0.3 * naturalness_from_foundation + 0.3 * naturalness_from_harmonics
        )
        naturalness = np.clip(naturalness, 0.0, 1.0)

        return float(naturalness), details

    @staticmethod
    def calculate_authenticity_score(
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        sr: int,
        profile: AnalysisProfile,
    ) -> tuple[float, dict]:
        """Calculate Authenticity score (Spec 1.2: Authentizität).

        Perceptual timbral-authenticity via three complementary proxy metrics:
        1. MFCC correlation   — timbral fingerprint similarity (DCT of log-spectrogram),
                                robust to loudness differences unlike MSE/RMS-ratio.
        2. Spectral centroid  — brightness preservation; large centroid drift indicates
                                over-denoising (HF loss) or additive HF hallucination.
        3. Chroma correlation — harmonic/key authenticity; captures whether the pitch-class
                                distribution of the restored signal matches the original.

        Weights: 0.45 MFCC + 0.25 centroid + 0.30 chroma  (same as GoosebumpsQualityChecker).

        Args:
            original_audio: Original (reference) audio, mono or stereo.
            processed_audio: Restored audio, mono or stereo.
            sr: Sample rate in Hz.
            profile: Analysis profile (unused; kept for API compatibility).

        Returns:
            Tuple of (authenticity_score ∈ [0, 1], proxy_details dict).
        """
        # ── Flatten stereo → mono ──────────────────────────────────────────────
        orig_m = original_audio.flatten() if original_audio.ndim == 1 else np.mean(original_audio, axis=1)
        proc_m = processed_audio.flatten() if processed_audio.ndim == 1 else np.mean(processed_audio, axis=1)

        min_len = min(len(orig_m), len(proc_m))
        orig = orig_m[:min_len]
        proc = proc_m[:min_len]

        n_fft = 2048
        hop = 512
        n_mfcc = 13

        # ── 1. MFCC correlation (timbral fingerprint) ─────────────────────────
        def _mfcc_frames(audio: np.ndarray) -> np.ndarray:
            """DCT-II of log-magnitude spectrum per frame (no external deps)."""
            n_frames = max(1, (len(audio) - n_fft) // hop)
            window = np.hanning(n_fft)
            coeffs = np.zeros((n_frames, n_mfcc))
            half = n_fft // 2 + 1
            cos_tbl = np.array(
                [np.cos(np.pi * k * (2 * np.arange(half) + 1) / (2 * half)) for k in range(n_mfcc)]
            )  # shape (n_mfcc, half)
            for i in range(n_frames):
                frame = audio[i * hop : i * hop + n_fft]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                log_spec = np.log(np.maximum(np.abs(np.fft.rfft(frame * window)), 1e-10))
                coeffs[i] = cos_tbl @ log_spec
            return coeffs  # type: ignore[no-any-return]

        mfcc_o = _mfcc_frames(orig).flatten()
        mfcc_p = _mfcc_frames(proc).flatten()
        min_f = min(len(mfcc_o), len(mfcc_p))
        mfcc_o = mfcc_o[:min_f]
        mfcc_p = mfcc_p[:min_f]
        std_o, std_p = np.std(mfcc_o), np.std(mfcc_p)
        if std_o > 1e-8 and std_p > 1e-8:
            _cc = np.dot(mfcc_o - mfcc_o.mean(), mfcc_p - mfcc_p.mean()) / (
                np.linalg.norm(mfcc_o - mfcc_o.mean()) * np.linalg.norm(mfcc_p - mfcc_p.mean()) + 1e-10
            )
            mfcc_corr = float(np.clip(_cc, 0.0, 1.0))
        elif std_o < 1e-8 and std_p < 1e-8:
            mfcc_corr = 1.0  # both constant (silence)
        else:
            mfcc_corr = 0.3  # one constant, one not → timbral mismatch

        # ── 2. Spectral centroid stability ────────────────────────────────────
        def _centroid(audio: np.ndarray) -> float:
            spec = np.abs(np.fft.rfft(audio[:n_fft]))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            total = np.sum(spec) + 1e-10
            return float(np.sum(freqs * spec) / total)

        centroid_o = _centroid(orig)
        centroid_p = _centroid(proc)
        centroid_dev = abs(centroid_p - centroid_o) / (sr / 2.0)
        centroid_score = float(max(0.0, 1.0 - centroid_dev * 10.0))  # 10 % Nyquist drift → 0

        # ── 3. Chroma correlation (harmonic/key authenticity, first 30 s) ─────
        def _chroma(audio: np.ndarray) -> np.ndarray:
            n = min(len(audio), int(30.0 * sr))
            spec = np.abs(np.fft.rfft(audio[:n]))
            freqs = np.fft.rfftfreq(n, 1.0 / sr)
            chroma = np.zeros(12)
            for i, f in enumerate(freqs):
                if f < 20 or f > 8000:
                    continue
                chroma[int(round(69 + 12 * np.log2(f / 440.0 + 1e-12))) % 12] += spec[i] ** 2
            norm = np.linalg.norm(chroma)
            return chroma / norm if norm > 0 else chroma  # type: ignore[no-any-return]

        chroma_corr = float(np.dot(_chroma(orig), _chroma(proc)))

        # ── Weighted combination ───────────────────────────────────────────────
        authenticity = float(np.clip(0.45 * mfcc_corr + 0.25 * centroid_score + 0.30 * chroma_corr, 0.0, 1.0))

        details = {
            "mfcc_corr": round(mfcc_corr, 4),
            "centroid_score": round(centroid_score, 4),
            "centroid_orig_hz": round(centroid_o, 1),
            "centroid_proc_hz": round(centroid_p, 1),
            "chroma_corr": round(chroma_corr, 4),
        }
        return authenticity, details

    @staticmethod
    def calculate_emotionality_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Emotionality score (Spec 1.2: Emotionalität).

        Proxies:
        - Micro-Dynamics Preservation
        - Vibrato/Tremolo Integrity
        - Dynamic Range Variance
        - Expressive Feature Retention

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (emotionality_score, proxy_details)
        """
        details = {}

        # 1. Micro-Dynamics (from dynamic range)
        dr_db = profile.dynamics.dynamic_range_db
        # Typical excellent DR: 12-20 dB
        microdynamics = np.clip((dr_db - 5) / 15, 0.0, 1.0)
        details["microdynamics_preservation"] = float(microdynamics)

        # 2. Vibrato/Tremolo Integrity (from pitch variance if available)
        if profile.feature_vectors.pitch_contour:
            pitch_variance = np.std(profile.feature_vectors.pitch_contour)
            vibrato_integrity = np.clip(pitch_variance / 50.0, 0.0, 1.0)
        else:
            vibrato_integrity = 0.7  # Default
        details["vibrato_tremolo_integrity"] = float(vibrato_integrity)

        # 3. Dynamic Range Variance (from loudness range)
        lra_lu = profile.dynamics.loudness_range_lu
        # Typical good LRA: 6-15 LU
        dr_variance = np.clip((lra_lu - 3) / 12, 0.0, 1.0)
        details["dynamic_range_variance"] = float(dr_variance)

        # 4. Expressive Feature Retention (from harmonicity and transients)
        if profile.feature_vectors.harmonicity:
            expressive_retention = np.clip(profile.feature_vectors.harmonicity / 25.0, 0.0, 1.0)
        else:
            expressive_retention = 0.7
        details["expressive_feature_retention"] = float(expressive_retention)

        # Weighted combination
        emotionality = (
            0.30 * microdynamics + 0.20 * vibrato_integrity + 0.25 * dr_variance + 0.25 * expressive_retention
        )
        emotionality = np.clip(emotionality, 0.0, 1.0)

        return float(emotionality), details

    @staticmethod
    def calculate_warmth_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Warmth score (Spec 1.2: Wärme).

        Proxies:
        - Low-Mid Energy Balance (200-800Hz)
        - Even Harmonic Content
        - Crest Factor
        - Tube Saturation Profile Similarity

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (warmth_score, proxy_details)
        """
        details = {}

        # 1. Low-Mid Energy Balance (200-800Hz)
        fft = np.fft.rfft(audio.flatten())
        freqs = np.fft.rfftfreq(len(audio.flatten()), 1 / sr)

        lowmid_mask = (freqs >= 200) & (freqs <= 800)
        lowmid_energy: float = float(np.sum(np.abs(fft[lowmid_mask]) ** 2))
        total_energy: float = float(np.sum(np.abs(fft) ** 2))

        lowmid_balance = lowmid_energy / (total_energy + 1e-10)
        # Normalize to typical range
        lowmid_balance_norm = np.clip(lowmid_balance * 10, 0.0, 1.0)
        details["lowmid_energy_balance"] = float(lowmid_balance_norm)

        # 2. Even Harmonic Content (approximation)
        # Would require pitch detection and harmonic analysis
        # Placeholder: use inverse of crest factor as proxy
        crest_factor_db = profile.dynamics.crest_factor_db
        even_harmonic_estimate = np.clip(1.0 - (crest_factor_db - 10) / 15, 0.0, 1.0)
        details["even_harmonic_content"] = float(even_harmonic_estimate)

        # 3. Crest Factor (inverted for warmth - lower crest = warmer)
        warmth_from_crest = np.clip(1.0 - (crest_factor_db - 8) / 12, 0.0, 1.0)
        details["crest_factor_warmth"] = float(warmth_from_crest)

        # 4. Tube Saturation Profile — Even/Odd Harmonic Ratio (Rossing 2007)
        # Tube amps produce predominantly even harmonics (H2, H4, H6).
        # A high even/odd ratio indicates warm tube-like saturation.
        try:
            audio_flat = audio.flatten()
            fft_full = np.fft.rfft(audio_flat)
            mag = np.abs(fft_full)
            freqs_full = np.fft.rfftfreq(len(audio_flat), 1.0 / sr)
            # Find dominant fundamental in 50-500 Hz range
            f_mask = (freqs_full >= 50) & (freqs_full <= 500)
            if np.any(f_mask):
                f0_idx = int(np.where(f_mask)[0][0] + np.argmax(mag[f_mask]))
                f0 = freqs_full[f0_idx]
                if f0 > 0:
                    even_pwr = 0.0
                    odd_pwr = 0.0
                    for h in range(2, 9):
                        h_idx = int(np.argmin(np.abs(freqs_full - f0 * h)))
                        if h_idx < len(mag):
                            pwr = float(mag[h_idx] ** 2)
                            if h % 2 == 0:
                                even_pwr += pwr
                            else:
                                odd_pwr += pwr
                    ratio = even_pwr / (even_pwr + odd_pwr + 1e-12)
                    tube_saturation_similarity = float(np.clip(ratio, 0.0, 1.0))
                else:
                    tube_saturation_similarity = 0.5
            else:
                tube_saturation_similarity = 0.5
        except Exception:
            tube_saturation_similarity = 0.5
        details["tube_saturation_profile"] = tube_saturation_similarity

        # Weighted combination
        warmth = (
            0.30 * lowmid_balance_norm
            + 0.25 * even_harmonic_estimate
            + 0.25 * warmth_from_crest
            + 0.20 * tube_saturation_similarity
        )
        warmth = np.clip(warmth, 0.0, 1.0)

        return float(warmth), details

    @staticmethod
    def calculate_spatiality_score(audio: np.ndarray, sr: int, profile: AnalysisProfile) -> tuple[float, dict]:
        """
        Calculate Spatiality score (Spec 1.2: Räumlichkeit).

        Proxies:
        - IACC (Interaural Cross-Correlation)
        - Stereo Width
        - Early/Late Energy Ratio (approximated)
        - Envelopment Index
        - Depth Cue Preservation

        Args:
            audio: Audio signal
            sr: Sample rate
            profile: Analysis profile

        Returns:
            Tuple of (spatiality_score, proxy_details)
        """
        details = {}

        # 1. IACC (from profile)
        iacc = profile.stereo.iacc
        # IACC ranges from -1 to 1; lower values = wider stereo image
        # Convert to 0-1 scale where higher = better spatiality
        iacc_score = (1.0 - iacc) / 2.0  # Maps [-1, 1] to [1, 0]
        details["iacc"] = float(iacc_score)

        # 2. Stereo Width (from profile)
        stereo_width = profile.stereo.stereo_width
        # Normalize: 0=mono, 1=normal, >1=wide. Clip to [0, 1.5] and normalize
        width_norm = np.clip(stereo_width / 1.5, 0.0, 1.0)
        details["stereo_width"] = float(width_norm)

        # 3. Early/Late Energy Ratio (approximation using onset density)
        # Higher onset density suggests more early energy (transients)
        onset_count = len(profile.feature_vectors.onset_times)
        duration_sec = len(audio) / sr if sr > 0 else 1.0
        early_late_ratio = np.clip(onset_count / (duration_sec * 5), 0.0, 1.0)
        details["early_late_energy_ratio"] = float(early_late_ratio)

        # 4. Envelopment Index (approximation from stereo width and phase coherence)
        phase_coherence = profile.stereo.phase_coherence
        envelopment = (stereo_width + (1.0 - phase_coherence)) / 2.0
        envelopment = np.clip(envelopment, 0.0, 1.0)
        details["envelopment_index"] = float(envelopment)

        # 5. Depth Cue Preservation (approximation from spectral rolloff)
        # Higher rolloff suggests more depth/air
        rolloff = profile.spectral.spectral_rolloff
        depth_cue = np.clip((rolloff - 5000) / 10000, 0.0, 1.0)
        details["depth_cue_preservation"] = float(depth_cue)

        # Weighted combination
        spatiality = (
            0.25 * iacc_score + 0.25 * width_norm + 0.15 * early_late_ratio + 0.20 * envelopment + 0.15 * depth_cue
        )
        spatiality = np.clip(spatiality, 0.0, 1.0)

        return float(spatiality), details


# ============================================================================
# Composite Aesthetic Score Calculator (Spec 3.2.1)
# ============================================================================


class CompositeAestheticScoreCalculator:
    """
    Calculates the Composite Aesthetic Score (CAS) using the formula from Spec 3.2.1.

    Formula:
        CAS = Σ (wᵢ × normalized_proxyᵢ) × (1 - penalty_artifacts) × authenticity_factor

    Where:
        - wᵢ are genre-adaptive weights
        - normalized_proxyᵢ are the 7 aesthetic dimension scores (0-1)
        - penalty_artifacts is artifact penalty (0-1)
        - authenticity_factor is authenticity preservation (0-1)
    """

    def __init__(self):
        self.proxy_calculator = AestheticProxyCalculator()

    def calculate_cas(
        self,
        audio: np.ndarray,
        sr: int,
        profile: AnalysisProfile,
        original_audio: np.ndarray | None = None,
        genre: Genre = Genre.UNKNOWN,
        genre_confidence: float = 1.0,
    ) -> tuple[float, AestheticScores]:
        """
        Calculate Composite Aesthetic Score.

        Args:
            audio: Processed audio signal
            sr: Sample rate
            profile: Analysis profile
            original_audio: Original audio (for authenticity calculation)
            genre: Detected genre
            genre_confidence: Genre detection confidence

        Returns:
            Tuple of (cas_score, aesthetic_scores)
        """
        # Get genre-adaptive weights
        weights = GenreWeightRegistry.get_weights(genre, genre_confidence)

        # Calculate all 7 proxy scores
        brilliance, brilliance_details = self.proxy_calculator.calculate_brilliance_score(audio, sr, profile)
        transparency, transparency_details = self.proxy_calculator.calculate_transparency_score(audio, sr, profile)
        naturalness, naturalness_details = self.proxy_calculator.calculate_naturalness_score(
            audio, sr, profile, original_audio=original_audio
        )

        if original_audio is not None:
            authenticity, authenticity_details = self.proxy_calculator.calculate_authenticity_score(
                original_audio, audio, sr, profile
            )
        else:
            # No original for comparison - assume high authenticity
            authenticity = 0.9
            authenticity_details = {"note": "no_original_for_comparison"}

        emotionality, emotionality_details = self.proxy_calculator.calculate_emotionality_score(audio, sr, profile)
        warmth, warmth_details = self.proxy_calculator.calculate_warmth_score(audio, sr, profile)
        spatiality, spatiality_details = self.proxy_calculator.calculate_spatiality_score(audio, sr, profile)

        # Create AestheticScores object
        scores = AestheticScores(
            brilliance=brilliance,
            transparency=transparency,
            naturalness=naturalness,
            authenticity=authenticity,
            emotionality=emotionality,
            warmth=warmth,
            spatiality=spatiality,
            proxy_details={
                "brilliance": brilliance_details,
                "transparency": transparency_details,
                "naturalness": naturalness_details,
                "authenticity": authenticity_details,
                "emotionality": emotionality_details,
                "warmth": warmth_details,
                "spatiality": spatiality_details,
            },
        )

        # Calculate weighted sum: Σ (wᵢ × normalized_proxyᵢ)
        weighted_sum = (
            weights.brilliance * brilliance
            + weights.transparency * transparency
            + weights.naturalness * naturalness
            + weights.authenticity * authenticity
            + weights.emotionality * emotionality
            + weights.warmth * warmth
            + weights.spatiality * spatiality
        )

        # Calculate artifact penalty: (1 - penalty_artifacts)
        artifact_severity_sum = sum(d.severity * d.confidence for d in profile.detected_defects)
        penalty_artifacts = np.clip(artifact_severity_sum / 3.0, 0.0, 1.0)  # Normalize by typical max
        artifact_factor = 1.0 - penalty_artifacts

        # Apply formula: CAS = weighted_sum × (1 - penalty) × authenticity
        cas = weighted_sum * artifact_factor * authenticity
        cas = np.clip(cas, 0.0, 1.0)

        logger.info(
            f"CAS Calculation: weighted_sum={weighted_sum:.3f}, artifact_factor={artifact_factor:.3f}, authenticity={authenticity:.3f}, CAS={cas:.3f}"
        )

        return float(cas), scores

    def check_constraints(
        self,
        cas_before: float,
        cas_after: float,
        scores_before: AestheticScores,
        scores_after: AestheticScores,
        profile_before: AnalysisProfile,
        profile_after: AnalysisProfile,
    ) -> list[ConstraintCheckResult]:
        """
        Prüft constraint system per Spec 3.2.3.

        Constraints:
        1. Authenticity Floor: Perceptual Similarity > 0.85
        2. Artifact Ceiling: New artifacts < existing
        3. Dynamic Preservation: Mikrodynamik > 90%
        4. Spectral Integrity: No bands > 3dB changed without defect indication

        Args:
            cas_before: CAS before processing
            cas_after: CAS after processing
            scores_before: Aesthetic scores before
            scores_after: Aesthetic scores after
            profile_before: Analysis profile before
            profile_after: Analysis profile after

        Returns:
            List of ConstraintCheckResult objects
        """
        results = []

        # Constraint 1: Authenticity Floor
        authenticity = scores_after.authenticity
        passed_auth = authenticity > 0.85
        results.append(
            ConstraintCheckResult(
                constraint_name="Authenticity Floor",
                passed=passed_auth,
                measured_value=authenticity,
                threshold_value=0.85,
                severity="error" if not passed_auth else "info",
                message=f"Perceptual similarity to original: {authenticity:.3f} ({'PASS' if passed_auth else 'FAIL'}: must be > 0.85)",
            )
        )

        # Constraint 2: Artifact Ceiling
        artifacts_before = sum(d.severity for d in profile_before.detected_defects)
        artifacts_after = sum(d.severity for d in profile_after.detected_defects)
        passed_artifact = artifacts_after < artifacts_before or abs(artifacts_after - artifacts_before) < 0.1
        results.append(
            ConstraintCheckResult(
                constraint_name="Artifact Ceiling",
                passed=passed_artifact,
                measured_value=artifacts_after,
                threshold_value=artifacts_before,
                severity="warning" if not passed_artifact else "info",
                message=f"New artifact level: {artifacts_after:.3f} vs original: {artifacts_before:.3f} ({'PASS' if passed_artifact else 'FAIL'}: must not increase)",
            )
        )

        # Constraint 3: Dynamic Preservation
        dr_before = profile_before.dynamics.dynamic_range_db
        dr_after = profile_after.dynamics.dynamic_range_db
        dr_preservation = dr_after / (dr_before + 1e-6)
        passed_dynamics = dr_preservation > 0.9
        results.append(
            ConstraintCheckResult(
                constraint_name="Dynamic Preservation",
                passed=passed_dynamics,
                measured_value=dr_preservation,
                threshold_value=0.9,
                severity="warning" if not passed_dynamics else "info",
                message=f"Dynamic range preservation: {dr_preservation:.1%} ({'PASS' if passed_dynamics else 'FAIL'}: must be > 90%)",
            )
        )

        # Constraint 4: Spectral Integrity
        centroid_before = profile_before.spectral.spectral_centroid
        centroid_after = profile_after.spectral.spectral_centroid
        centroid_change_db = 20 * np.log10((centroid_after + 1) / (centroid_before + 1))
        passed_spectral = abs(centroid_change_db) < 3.0
        results.append(
            ConstraintCheckResult(
                constraint_name="Spectral Integrity",
                passed=passed_spectral,
                measured_value=abs(centroid_change_db),
                threshold_value=3.0,
                severity="warning" if not passed_spectral else "info",
                message=f"Spectral centroid change: {centroid_change_db:+.1f} dB ({'PASS' if passed_spectral else 'FAIL'}: must be < 3 dB)",
            )
        )

        return results


# ============================================================================
# Aesthetic Judgment Model (Main Facade)
# ============================================================================


class AestheticJudgmentModel:
    """
    Haupt-facade for the Aesthetic Judgment Model.

    Orchestrates CAS calculation, genre-adaptive weighting, and constraint checking.
    """

    def __init__(self):
        self.cas_calculator = CompositeAestheticScoreCalculator()
        GenreWeightRegistry.validate_all_weights()  # Validate on init

    def evaluate(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        profile_before: AnalysisProfile,
        profile_after: AnalysisProfile,
        genre: Genre | None = None,
        genre_confidence: float = 1.0,
    ) -> QualityReport:
        """
        Complete aesthetic evaluation: CAS calculation + constraint checking.

        Args:
            audio_before: Original audio signal
            audio_after: Processed audio signal
            sr: Sample rate
            profile_before: Analysis profile of original
            profile_after: Analysis profile of processed
            genre: Detected genre (if None, uses profile)
            genre_confidence: Genre detection confidence

        Returns:
            QualityReport with CAS scores and constraint results
        """
        # Determine genre
        if genre is None:
            genre = profile_after.musical_context.genre
            genre_confidence = profile_after.musical_context.genre_confidence

        # Calculate CAS before and after
        cas_before, scores_before = self.cas_calculator.calculate_cas(
            audio_before, sr, profile_before, None, genre, genre_confidence
        )

        cas_after, scores_after = self.cas_calculator.calculate_cas(
            audio_after, sr, profile_after, audio_before, genre, genre_confidence
        )

        cas_improvement = cas_after - cas_before

        # Check constraints
        constraint_results = self.cas_calculator.check_constraints(
            cas_before,
            cas_after,
            scores_before,
            scores_after,
            profile_before,
            profile_after,
        )

        constraints_satisfied = all(c.passed for c in constraint_results)

        # Collect warnings and errors
        warnings = [c.message for c in constraint_results if c.severity == "warning"]
        errors = [c.message for c in constraint_results if c.severity == "error"]

        # Create QualityReport
        report = QualityReport(
            cas_before=cas_before,
            cas_after=cas_after,
            cas_improvement=cas_improvement,
            aesthetic_scores_before=scores_before,
            aesthetic_scores_after=scores_after,
            dnsmos_score=None,
            cdpam_score=None,
            constraints_satisfied=constraints_satisfied,
            constraint_checks=constraint_results,
            warnings=warnings,
            errors=errors,
        )

        logger.info("Quality evaluation vollstaendig: CAS %.3f → %.3f (Δ%+.3f)", cas_before, cas_after, cas_improvement)

        return report


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "AestheticJudgmentModel",
    "AestheticProxyCalculator",
    "CompositeAestheticScoreCalculator",
    "GenreWeightRegistry",
]
