"""
Context-Aware De-Esser v2.0
============================

Phoneme-aware sibilance reduction using ML-based phoneme detection.

Industry First: Unlike traditional broadband frequency-based de-essers,
this module uses Wav2Vec2 phoneme detection to identify actual sibilant
phonemes (/s/, /z/, /ʃ/, /ʒ/, /tʃ/, /dʒ/) and applies targeted reduction
ONLY to those regions.

Key Advantages:
- No "lisping" artifacts (doesn't process non-sibilant high frequencies)
- Phoneme-specific parameters (/s/ at 8kHz vs. /ʃ/ at 5kHz)
- Context-aware (word position, musical genre, vocal style)
- Preserves intelligibility and natural timbre
- Adaptive thresholds based on phoneme confidence

Architecture:
┌─────────────────┐
│  Audio Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PhonemeDetector │ ← Wav2Vec2 inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Sibilant      │
│  Classification │ ← Filter s/z/ʃ/ʒ/tʃ/dʒ
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phoneme-Specific│
│  Frequency      │ ← Different center freq per phoneme
│   Reduction     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Crossfade &     │
│  Reconstruct    │ ← Smooth transitions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio Output   │
└─────────────────┘

Author: Aurik Development Team
Version: 2.0.0
Date: 8. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.signal as signal

# Phoneme detection imports
try:
    from backend.ml.phoneme_aware.phoneme_classifier import (
        PhonemeClassifier,  # type: ignore[assignment]
        SibilantType,  # type: ignore[assignment]
    )
    from backend.ml.phoneme_aware.phoneme_detector import (
        Language,  # type: ignore[assignment]
        PhonemeDetector,  # type: ignore[assignment]
        PhonemeSegment,  # type: ignore[assignment]
    )

    PHONEME_DETECTION_AVAILABLE = True
except Exception:

    class SibilantType(Enum):  # type: ignore[no-redef]
        """Fallback sibilant enum for DSP-only mode."""

        S_VOICELESS = "s"
        Z_VOICED = "z"
        SH_VOICELESS = "sh"
        ZH_VOICED = "zh"
        CH_VOICELESS = "ch"
        JH_VOICED = "jh"

    class Language(Enum):  # type: ignore[no-redef]
        """Fallback language enum for DSP-only mode."""

        ENGLISH = "en"

    @dataclass
    class PhonemeSegment:  # type: ignore[no-redef]
        """Fallback phoneme segment placeholder."""

        phoneme: str
        start_time: float
        end_time: float
        confidence: float
        frame_index: int

    class PhonemeDetector:  # type: ignore[no-redef]
        """Fallback detector placeholder."""

        def __init__(self, *args, **kwargs):
            raise ImportError("Phoneme detection backend unavailable")

    class PhonemeClassifier:  # type: ignore[no-redef]
        """Fallback classifier placeholder."""

        def classify(self, phoneme: str):
            raise ImportError("Phoneme detection backend unavailable")

    PHONEME_DETECTION_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


class ProcessingMode(Enum):
    """De-essing processing mode."""

    GENTLE = "gentle"  # -3 to -6 dB reduction
    MODERATE = "moderate"  # -6 to -9 dB reduction
    AGGRESSIVE = "aggressive"  # -9 to -12 dB reduction


@dataclass
class SibilantParameters:
    """
    Verarbeitet parameters for specific sibilant type.

    Different sibilants have different spectral characteristics:
    - /s/, /z/: High frequency (7-9 kHz)
    - /ʃ/, /ʒ/: Mid-high frequency (4-6 kHz)
    - /tʃ/, /dʒ/: Broad spectrum (5-8 kHz)
    """

    freq_center: float  # Center frequency in Hz
    freq_bandwidth: float  # Bandwidth in Hz
    reduction_db: float  # Maximum reduction in dB
    q_factor: float  # Filter Q factor (sharpness)
    attack_ms: float  # Attack time in ms
    release_ms: float  # Release time in ms


# Phoneme-specific processing parameters
SIBILANT_PROFILES: dict[SibilantType, SibilantParameters] = {
    SibilantType.S_VOICELESS: SibilantParameters(
        freq_center=8000.0,
        freq_bandwidth=4000.0,
        reduction_db=-6.0,
        q_factor=1.5,
        attack_ms=0.5,
        release_ms=50.0,
    ),
    SibilantType.Z_VOICED: SibilantParameters(
        freq_center=7500.0,
        freq_bandwidth=3500.0,
        reduction_db=-5.5,
        q_factor=1.4,
        attack_ms=0.5,
        release_ms=50.0,
    ),
    SibilantType.SH_VOICELESS: SibilantParameters(
        freq_center=5000.0,
        freq_bandwidth=3000.0,
        reduction_db=-5.0,
        q_factor=1.3,
        attack_ms=0.8,
        release_ms=60.0,
    ),
    SibilantType.ZH_VOICED: SibilantParameters(
        freq_center=4500.0,
        freq_bandwidth=2500.0,
        reduction_db=-4.5,
        q_factor=1.2,
        attack_ms=0.8,
        release_ms=60.0,
    ),
    SibilantType.CH_VOICELESS: SibilantParameters(
        freq_center=6000.0,
        freq_bandwidth=4000.0,
        reduction_db=-5.5,
        q_factor=1.4,
        attack_ms=1.0,
        release_ms=70.0,
    ),
    SibilantType.JH_VOICED: SibilantParameters(
        freq_center=5500.0,
        freq_bandwidth=3500.0,
        reduction_db=-5.0,
        q_factor=1.3,
        attack_ms=1.0,
        release_ms=70.0,
    ),
}


@dataclass
class DeEsserConfig:
    """Configuration for Context-Aware De-Esser."""

    mode: ProcessingMode = ProcessingMode.MODERATE
    device: str = "cpu"  # §9.5 CPU-only — kein CUDA
    language: Language = Language.ENGLISH
    min_phoneme_confidence: float = 0.5
    reduction_multiplier: float = 1.0  # Scale all reductions
    crossfade_ms: float = 5.0  # Smooth transitions between regions
    dry_wet_mix: float = 1.0  # 0=dry, 1=wet
    enable_genre_adaptation: bool = True
    genre: str | None = None  # e.g., "pop", "classical", "speech"


@dataclass
class ProcessingReport:
    """Verarbeitet statistics and metrics."""

    total_duration_sec: float
    phonemes_detected: int
    sibilants_detected: int
    sibilants_processed: int
    avg_reduction_db: float
    percentage_processed: float
    sibilant_breakdown: dict[str, int]  # Count per sibilant type


# ============================================================================
# CONTEXT-AWARE DE-ESSER V2.0
# ============================================================================


class ContextAwareDeEsser:
    """
    Phoneme-aware de-esser using ML-based sibilant detection.

    This de-esser uses Wav2Vec2 to detect phonemes and applies reduction
    ONLY to actual sibilant phonemes (/s/, /z/, /ʃ/, /ʒ/, /tʃ/, /dʒ/).

    Benefits over traditional de-essers:
    - No processing of non-sibilant high frequencies
    - Preserves cymbal/hi-hat clarity in music
    - No "lisping" artifacts
    - Context-aware (adapts to genre, voice type)
    - Phoneme-specific parameters

    Example:
        >>> deesser = ContextAwareDeEsser()
        >>> audio_out, report = deesser.process(audio, sr=48000)
        >>> logger.info(f"Processed {report.percentage_processed:.1f}% of audio")
    """

    def __init__(self, config: DeEsserConfig | None = None):
        """
        Initialisiert Context-Aware De-Esser.

        Args:
            config: De-esser configuration (defaults to moderate)
        """
        self.config = config or DeEsserConfig()

        # Initialize phoneme detector
        if not PHONEME_DETECTION_AVAILABLE:
            logger.warning(
                "Phoneme detection not verfuegbar (torch/transformers missing) — laeuft in DSP-only Betriebsart"
            )
            self.phoneme_detector = None
            self.phoneme_classifier = None
        else:
            logger.info(
                f"initialisiere Context-Aware De-Esser v2.0 (Betriebsart={self.config.mode.value}, device={self.config.device})"
            )
            try:
                # Create detection config
                from backend.ml.phoneme_aware.phoneme_detector import DetectionConfig

                detection_config = DetectionConfig(
                    language=self.config.language,
                    min_confidence=self.config.min_phoneme_confidence,
                    use_gpu=False,  # §9.5 CPU-only
                )

                self.phoneme_detector = PhonemeDetector(config=detection_config)
                self.phoneme_classifier = PhonemeClassifier()
                logger.info("Phoneme detection initialisiert erfolgreich")
            except Exception as e:
                logger.error("konnte nicht initialisieren phoneme detection: %s", e)
                logger.warning("Falling back to DSP-only Betriebsart")
                self.phoneme_detector = None
                self.phoneme_classifier = None

        # Processing state
        self.last_report: ProcessingReport | None = None

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        genre: str | None = None,
    ) -> tuple[np.ndarray, ProcessingReport]:
        """
        Verarbeitet audio with phoneme-aware de-essing.

        Args:
            audio: Input audio (mono or stereo)
            sr: Sample rate
            genre: Optional genre for adaptive parameters

        Returns:
            (processed_audio, report): Processed audio and statistics
        """
        if audio.size == 0:
            raise ValueError("Audio is empty")

        logger.info("Processing %s audio at %s Hz (Betriebsart=%s)", audio.shape, sr, self.config.mode.value)

        # Handle stereo
        is_stereo = audio.ndim == 2
        if is_stereo:
            # Process each channel separately
            left, report_l = self._process_mono(audio[0], sr, genre)
            right, report_r = self._process_mono(audio[1], sr, genre)
            processed = np.stack([left, right], axis=0)

            # Merge reports (average)
            report = self._merge_reports([report_l, report_r])
        else:
            # Process mono
            processed, report = self._process_mono(audio, sr, genre)

        # Apply dry/wet mix
        if self.config.dry_wet_mix < 1.0:
            processed = audio * (1.0 - self.config.dry_wet_mix) + processed * self.config.dry_wet_mix

        self.last_report = report
        logger.info(
            f"De-essing vollstaendig: {report.sibilants_processed}/{report.sibilants_detected} "
            f"sibilants verarbeitet ({report.percentage_processed:.1f}% of audio)"
        )

        return processed, report

    def _process_mono(
        self,
        audio: np.ndarray,
        sr: int,
        genre: str | None,
    ) -> tuple[np.ndarray, ProcessingReport]:
        """Verarbeitet mono audio channel."""
        duration_sec = len(audio) / sr

        # Step 1: Detect phonemes
        logger.debug("Detecting phonemes...")
        try:
            phonemes = self.phoneme_detector.detect(audio, sr)  # type: ignore[union-attr]
        except Exception as e:
            logger.error("Phoneme detection fehlgeschlagen: %s", e)
            # Fallback: return unprocessed audio
            report = ProcessingReport(
                total_duration_sec=duration_sec,
                phonemes_detected=0,
                sibilants_detected=0,
                sibilants_processed=0,
                avg_reduction_db=0.0,
                percentage_processed=0.0,
                sibilant_breakdown={},
            )
            return audio.copy(), report

        logger.debug("erkannt %s phonemes", len(phonemes))

        # Step 2: Filter sibilants
        sibilants = self._filter_sibilants(phonemes)
        logger.debug("Found %s sibilant phonemes", len(sibilants))

        if len(sibilants) == 0:
            # No sibilants detected - return unprocessed
            report = ProcessingReport(
                total_duration_sec=duration_sec,
                phonemes_detected=len(phonemes),
                sibilants_detected=0,
                sibilants_processed=0,
                avg_reduction_db=0.0,
                percentage_processed=0.0,
                sibilant_breakdown={},
            )
            return audio.copy(), report

        # Step 3: Apply genre-adaptive adjustments
        if self.config.enable_genre_adaptation and genre:
            sibilants = self._adapt_to_genre(sibilants, genre)

        # Step 4: Process each sibilant region
        processed = audio.copy()
        reductions_db = []
        sibilant_counts: dict[str, int] = {}

        for sibilant_seg, sibilant_type in sibilants:
            # Get parameters for this sibilant type
            params = SIBILANT_PROFILES[sibilant_type]

            # Apply mode-specific scaling
            reduction_db = (
                self._scale_reduction(
                    params.reduction_db,
                    self.config.mode,
                )
                * self.config.reduction_multiplier
            )

            # Process region
            processed = self._process_sibilant_region(
                processed,
                sr,
                sibilant_seg,
                params,
                reduction_db,
            )

            reductions_db.append(abs(reduction_db))
            sibilant_type_str = sibilant_type.value
            sibilant_counts[sibilant_type_str] = sibilant_counts.get(sibilant_type_str, 0) + 1

        # Compute statistics
        total_sibilant_duration = sum(seg.duration for seg, _ in sibilants)
        percentage_processed = (total_sibilant_duration / duration_sec) * 100.0

        report = ProcessingReport(
            total_duration_sec=duration_sec,
            phonemes_detected=len(phonemes),
            sibilants_detected=len(sibilants),
            sibilants_processed=len(sibilants),
            avg_reduction_db=np.mean(reductions_db) if reductions_db else 0.0,  # type: ignore[arg-type]
            percentage_processed=percentage_processed,
            sibilant_breakdown=sibilant_counts,
        )

        return processed, report

    def _filter_sibilants(
        self,
        phonemes: list[PhonemeSegment],
    ) -> list[tuple[PhonemeSegment, SibilantType]]:
        """
        Filtert phoneme list to only sibilants.

        Returns:
            List of (segment, sibilant_type) tuples
        """
        sibilants = []

        for phoneme_seg in phonemes:
            phoneme_info = self.phoneme_classifier.classify(phoneme_seg.phoneme)  # type: ignore[union-attr]

            # Check if sibilant
            if phoneme_info.is_sibilant and phoneme_info.sibilant_type:  # type: ignore[union-attr]
                sibilants.append((phoneme_seg, phoneme_info.sibilant_type))  # type: ignore[union-attr]

        return sibilants

    # ------------------------------------------------------------------ #
    # Genre → De-Essing-Skalierungstabelle                                #
    # §2.8 Stimmtyp-Adaptierung / §2.19.3 Schlager-Restaurierungsprofil  #
    # Schlüssel: lowercase Teilstring des genre-Strings genügt.           #
    # Wert: Reduktions-Skalierungsfaktor (< 1 = sanfter, > 1 = aggressiver)
    # ------------------------------------------------------------------ #
    _GENRE_SCALE: dict[str, float] = {
        # Klassik / Jazz: natürliche Sibilanz bewahren → sanfter eingreifen
        "classical": 1.40,  # Schwellwert × 1.4 → weniger Sibilanten werden behandelt
        "klassik": 1.40,
        "orchestra": 1.40,
        "jazz": 1.30,
        # Deutscher Schlager: warm, weniger S-Reduktion nötig (§2.19.3)
        "schlager": 1.20,
        "volksmusik": 1.20,
        "walzer": 1.15,
        "marsch": 1.10,
        # Pop / Rock: moderater Standardwert
        "pop": 1.00,
        "rock": 1.00,
        "indie": 1.05,
        # Oper / Gesang: Sprachverständlichkeit wichtig, aber nicht überaggressiv
        "opera": 1.15,
        "vocal": 1.10,
        # Podcast / Speech: maximale De-Essing für Verständlichkeit
        "podcast": 0.70,
        "speech": 0.70,
        "interview": 0.75,
        "broadcast": 0.75,
    }

    # Genre-Energieschwellen: verbleibende Sibilanten müssen > Schwellwert
    # kHz liegen, sonst zu schwach für Behandlung (Verhinderung von
    # Lispel-Artefakten, §2.8 Invariante)
    _GENRE_ENERGY_FILTER: dict[str, float] = {
        "classical": 0.30,  # Nur energiereiche Sibilanten behandeln
        "klassik": 0.30,
        "jazz": 0.25,
        "schlager": 0.20,
        "podcast": 0.05,  # Fast alle Sibilanten behandeln
        "speech": 0.05,
    }
    _DEFAULT_ENERGY_THRESHOLD: float = 0.10  # Fallback

    def _adapt_to_genre(
        self,
        sibilants: list[tuple[PhonemeSegment, SibilantType]],
        genre: str,
    ) -> list[tuple[PhonemeSegment, SibilantType]]:
        """Passe Sibilanten-Liste an das Genre an.

        Implementierung (§2.8, §2.19.3):
            1. Genre-String normalisieren (lowercase, Leerzeichen → Teilstring-Match).
            2. Skalierungsfaktor aus _GENRE_SCALE ermitteln:
               Iteriere über alle Schlüssel; der erste Treffer (Teilstring-Match)
               gewinnt. Kein Treffer → 1.0 (kein Eingriff).
            3. _genre_reduction_scale auf self setzen, damit _scale_reduction()
               diesen Wert als Multiplikator einbinden kann.
            4. Energie-basierte Filterung:
               Für Genres mit hohem Threshold (classical/jazz) werden
               Sibilanten mit sehr niedriger Phoneme-Confidence herausgefiltert
               (< energy_threshold), da in diesen Genres auch schwache Sibilanten
               schützenswert sind, nicht behandlungsbedürftig.
        Resultat: kürzere oder gleich lange Liste; Reihenfolge erhalten.
        Invariante: Mindestens leere Liste zurück (kein Absturz).
        Referenz: §2.8 Stimmtyp-Adaptierung; §2.19.3 Schlager-Profil;
                  Moulines & Charpentier (1990) PSOLA-Kontext.
        """
        genre_key = (genre or "").lower().strip()
        logger.debug("[DeEsser] Genre-Adaption für: '%s'", genre_key)

        # ── Skalierungsfaktor ermitteln ───────────────────────────────────
        scale: float = 1.0
        for key, val in self._GENRE_SCALE.items():
            if key in genre_key:
                scale = val
                break
        self._genre_reduction_scale = scale  # für _scale_reduction() nutzbar
        logger.debug("[DeEsser] Genre-Skalierungsfaktor: %.2f", scale)

        # ── Energie-basierte Filterung ────────────────────────────────────
        energy_threshold: float = self._DEFAULT_ENERGY_THRESHOLD
        for key, thr in self._GENRE_ENERGY_FILTER.items():
            if key in genre_key:
                energy_threshold = thr
                break

        # Filtere Sibilanten mit Confidence < energy_threshold nur bei
        # Genres, die höheren Threshold haben (conservative genres: scale > 1.0)
        if scale > 1.0 and energy_threshold > self._DEFAULT_ENERGY_THRESHOLD:
            filtered: list[tuple[PhonemeSegment, SibilantType]] = []
            for seg, sib_type in sibilants:
                confidence = getattr(seg, "confidence", 1.0)
                if confidence >= energy_threshold:
                    filtered.append((seg, sib_type))
                else:
                    logger.debug(
                        "[DeEsser] Sibilant bei %.2f s gefiltert (confidence %.2f < %.2f, genre=%s)",
                        getattr(seg, "start_time", 0.0),
                        confidence,
                        energy_threshold,
                        genre_key,
                    )
            logger.info(
                "[DeEsser] Genre '%s': %d/%d Sibilanten nach Energie-Filter",
                genre_key,
                len(filtered),
                len(sibilants),
            )
            return filtered

        return sibilants

    def _scale_reduction(
        self,
        base_reduction_db: float,
        mode: ProcessingMode,
    ) -> float:
        """Skaliert reduction based on processing mode."""
        if mode == ProcessingMode.GENTLE:
            return base_reduction_db * 0.5
        elif mode == ProcessingMode.MODERATE:
            return base_reduction_db * 1.0
        elif mode == ProcessingMode.AGGRESSIVE:
            return base_reduction_db * 1.5
        else:
            return base_reduction_db

    def _process_sibilant_region(
        self,
        audio: np.ndarray,
        sr: int,
        segment: PhonemeSegment,
        params: SibilantParameters,
        reduction_db: float,
    ) -> np.ndarray:
        """
        Wendet an: frequency-specific reduction to sibilant region.

        Uses bandpass filter + dynamic range compression in target frequency band.
        """
        # Convert time to sample indices
        start_idx = int(segment.start_time * sr)
        end_idx = int(segment.end_time * sr)

        # Add crossfade margins
        crossfade_samples = int(self.config.crossfade_ms * sr / 1000.0)
        start_idx_fade = max(0, start_idx - crossfade_samples)
        end_idx_fade = min(len(audio), end_idx + crossfade_samples)

        # Extract region
        region = audio[start_idx_fade:end_idx_fade].copy()

        if len(region) == 0:
            return audio

        # Design bandpass filter for sibilant frequency
        nyquist = sr / 2.0
        low_freq = max(params.freq_center - params.freq_bandwidth / 2.0, 100.0)  # Minimum freq
        high_freq = min(params.freq_center + params.freq_bandwidth / 2.0, nyquist * 0.95)  # Avoid Nyquist

        # Create filter
        sos = signal.butter(
            4,  # 4th order
            [low_freq / nyquist, high_freq / nyquist],
            btype="bandpass",
            output="sos",
        )

        # Extract sibilant band
        sibilant_band = signal.sosfilt(sos, region)

        # Apply reduction (simple gain)
        gain_linear = 10 ** (reduction_db / 20.0)
        sibilant_band_reduced = sibilant_band * gain_linear

        # Subtract original band and add reduced band
        region_processed = region - sibilant_band + sibilant_band_reduced

        # Apply crossfade at boundaries
        if crossfade_samples > 0:
            # Fade in
            fade_in = np.linspace(0, 1, crossfade_samples)
            region_processed[:crossfade_samples] = (
                region[:crossfade_samples] * (1 - fade_in) + region_processed[:crossfade_samples] * fade_in
            )

            # Fade out
            fade_out = np.linspace(1, 0, crossfade_samples)
            region_processed[-crossfade_samples:] = region_processed[-crossfade_samples:] * fade_out + region[
                -crossfade_samples:
            ] * (1 - fade_out)

        # Write processed region back
        audio_out = audio.copy()
        audio_out[start_idx_fade:end_idx_fade] = region_processed

        return audio_out

    def _merge_reports(
        self,
        reports: list[ProcessingReport],
    ) -> ProcessingReport:
        """Führt zusammen: multiple processing reports (for stereo)."""
        if not reports:
            raise ValueError("No reports to merge")

        if len(reports) == 1:
            return reports[0]

        # Average metrics
        avg_report = ProcessingReport(
            total_duration_sec=reports[0].total_duration_sec,
            phonemes_detected=int(np.mean([r.phonemes_detected for r in reports])),
            sibilants_detected=int(np.mean([r.sibilants_detected for r in reports])),
            sibilants_processed=int(np.mean([r.sibilants_processed for r in reports])),
            avg_reduction_db=float(np.mean([r.avg_reduction_db for r in reports])),
            percentage_processed=float(np.mean([r.percentage_processed for r in reports])),
            sibilant_breakdown={},
        )

        # Merge sibilant breakdowns
        for report in reports:
            for sib_type, count in report.sibilant_breakdown.items():
                avg_report.sibilant_breakdown[sib_type] = avg_report.sibilant_breakdown.get(sib_type, 0) + count

        # Average counts
        for sib_type in avg_report.sibilant_breakdown:
            avg_report.sibilant_breakdown[sib_type] //= len(reports)

        return avg_report


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def apply_context_aware_deessing(
    audio: np.ndarray,
    sr: int,
    mode: ProcessingMode = ProcessingMode.MODERATE,
    device: str = "cpu",
    genre: str | None = None,
    dry_wet: float = 1.0,
) -> tuple[np.ndarray, ProcessingReport]:
    """
    Convenience function for context-aware de-essing.

    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
        mode: Processing mode (GENTLE, MODERATE, AGGRESSIVE)
        device: Processing device (immer 'cpu', §9.5 CPU-only)
        genre: Optional genre for adaptation
        dry_wet: Dry/wet mix (0=dry, 1=wet)

    Returns:
        (processed_audio, report): Processed audio and statistics

    Example:
        >>> audio_out, report = apply_context_aware_deessing(
        ...     audio, sr=48000, mode=ProcessingMode.MODERATE
        ... )
        >>> logger.info(f"Reduced {report.sibilants_processed} sibilants")
    """
    config = DeEsserConfig(
        mode=mode,
        device=device,
        genre=genre,
        dry_wet_mix=dry_wet,
    )

    deesser = ContextAwareDeEsser(config)
    return deesser.process(audio, sr, genre=genre)
