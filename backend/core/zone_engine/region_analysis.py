"""
Region Analysis System für AURIK v8.0 (NORMATIVE COMPLIANCE)

**WICHTIG: Dieses Modul führt KEINE eigenständige Verarbeitung durch!**

Gemäß AURIK v8 Architekturregeln (docs/aurik_v_8_projektstruktur_ki_programmierregeln.md):
- Jede Verarbeitung läuft über: Epistemic Gate → Zonen → Conduct → Regulator
- Dieses Modul liefert NUR Analysedaten und Empfehlungen
- Verarbeitung erfolgt durch adaptive_pipeline.py unter Einhaltung aller Regulator-Vorgaben

Komponenten (ANALYSE-ONLY):
1. RegionDetector - Klassifiziert Audio-Segmente (Silence, Music, Speech, Noise)
2. RegionAnalyzer - Analysiert Region-Eigenschaften (Spectral, Dynamic, Quality)
3. RegionMetadataProvider - Liefert region-spezifische Empfehlungen an Pipeline

Vorteile:
- Informiert Pipeline über heterogene Audio-Bereiche
- Ermöglicht region-aware Parameter-Empfehlungen
- Unterstützt präzisere Zonen-Klassifizierung
- Compliance mit normativer Architektur

Autor: AURIK Team
Version: 8.0 (Normative Compliance Edition)
Datum: 7. Februar 2026
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np

# Handle both module import and direct execution
try:
    from backend.logging_config import get_logger

    logger = get_logger(__name__)
except ImportError:
    # Direct execution fallback
    logger = logging.getLogger(__name__)


# ==============================================================================
# Data Models
# ==============================================================================


class RegionType(Enum):
    """Audio-Region Typen."""

    SILENCE = "silence"
    SPEECH = "speech"
    MUSIC = "music"
    NOISE = "noise"
    MIXED = "mixed"  # Speech + Music or unclear


@dataclass
class AudioRegion:
    """Eine Audio-Region mit Metadaten."""

    region_type: RegionType
    start_sample: int
    end_sample: int
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_samples(self) -> int:
        """Duration in samples."""
        return self.end_sample - self.start_sample

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary."""
        return {
            "type": self.region_type.value,
            "start": self.start_sample,
            "end": self.end_sample,
            "duration": self.duration_samples,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class RegionRecommendation:
    """Empfohlene Parameter für eine Region (NICHT BINDEND!)."""

    region_type: RegionType
    recommended_enhancement_strength: float  # 0-1 (Empfehlung)
    recommended_noise_reduction_strength: float  # 0-1 (Empfehlung)
    suggested_deessing: bool  # Empfehlung
    suggested_dynamic_eq: bool  # Empfehlung
    preserve_transients_priority: bool  # Empfehlung
    quality_sensitivity: float  # Empfohlener Threshold
    reasoning: str  # Begründung für Empfehlung


# ==============================================================================
# Region Detector
# ==============================================================================


class RegionDetector:
    """
    Detektiert und klassifiziert Audio-Regionen.

    Methoden:
    - Energy-based silence detection
    - Spectral analysis for music vs speech
    - Harmonic-percussive separation
    - Zero-crossing rate analysis
    """

    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        min_region_duration_ms: float = 100.0,
        hop_length_ms: float = 10.0,
    ):
        """
        Args:
            silence_threshold_db: Threshold for silence detection (dB)
            min_region_duration_ms: Minimum duration for a region (ms)
            hop_length_ms: Hop length for analysis (ms)
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_region_duration_ms = min_region_duration_ms
        self.hop_length_ms = hop_length_ms

    def detect_regions(self, audio: np.ndarray, sr: int) -> list[AudioRegion]:
        """
        Detektiert alle Regionen im Audio.

        Args:
            audio: Audio signal (mono)
            sr: Sample rate

        Returns:
            List[AudioRegion]: Detected regions
        """
        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Compute frame-level features
        hop_length = int(sr * self.hop_length_ms / 1000.0)
        frame_length = hop_length * 2

        # 1. Energy (for silence detection)
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        energy_db = librosa.power_to_db(energy**2, ref=np.max)

        # 2. Spectral features (for music vs speech)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]

        # 3. Zero-crossing rate (for noise vs tonal)
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # 4. Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)  # type: ignore[attr-defined]
        harmonic_ratio = self._compute_harmonic_ratio_per_frame(harmonic, percussive, hop_length)

        # Classify each frame
        frame_types = []
        for i in range(len(energy_db)):
            frame_type, confidence = self._classify_frame(
                energy_db[i],
                spectral_centroid[i] if i < len(spectral_centroid) else 0,
                spectral_rolloff[i] if i < len(spectral_rolloff) else 0,
                zcr[i] if i < len(zcr) else 0,
                harmonic_ratio[i] if i < len(harmonic_ratio) else 0.5,  # type: ignore[arg-type]
            )
            frame_types.append((frame_type, confidence))

        # Merge consecutive frames into regions
        regions = self._merge_frames_to_regions(frame_types, hop_length, sr)

        # Filter by minimum duration
        min_samples = int(sr * self.min_region_duration_ms / 1000.0)
        regions = [r for r in regions if r.duration_samples >= min_samples]

        logger.info("erkannt %s regions in audio", len(regions))
        return regions

    def _classify_frame(
        self,
        energy_db: float,
        spectral_centroid: float,
        spectral_rolloff: float,
        zcr: float,
        harmonic_ratio: float,
    ) -> tuple[RegionType, float]:
        """
        Klassifiziert einen einzelnen Frame.

        Returns:
            Tuple[RegionType, confidence]
        """
        # 1. Check for silence
        if energy_db < self.silence_threshold_db:
            return RegionType.SILENCE, 0.95

        # 2. High ZCR + low harmonic ratio = Noise
        if zcr > 0.15 and harmonic_ratio < 0.3:
            return RegionType.NOISE, 0.8

        # 3. High harmonic ratio = Music or Speech
        if harmonic_ratio > 0.6:
            # Music: higher spectral centroid + rolloff
            if spectral_centroid > 3000 and spectral_rolloff > 8000:
                return RegionType.MUSIC, 0.85
            # Speech: lower centroid, focused spectrum
            elif spectral_centroid < 2000:
                return RegionType.SPEECH, 0.8
            else:
                return RegionType.MIXED, 0.6

        # 4. Default: Mixed
        return RegionType.MIXED, 0.5

    def _compute_harmonic_ratio_per_frame(
        self, harmonic: np.ndarray, percussive: np.ndarray, hop_length: int
    ) -> np.ndarray:
        """Berechnet Harmonic-Ratio pro Frame."""
        # RMS per frame
        harmonic_rms = librosa.feature.rms(y=harmonic, hop_length=hop_length)[0]
        percussive_rms = librosa.feature.rms(y=percussive, hop_length=hop_length)[0]

        total_energy = harmonic_rms**2 + percussive_rms**2
        harmonic_ratio = np.divide(
            harmonic_rms**2,
            total_energy,
            out=np.full_like(harmonic_rms, 0.5),
            where=total_energy > 1e-10,
        )

        return harmonic_ratio  # type: ignore[no-any-return]

    def _merge_frames_to_regions(
        self, frame_types: list[tuple[RegionType, float]], hop_length: int, sr: int
    ) -> list[AudioRegion]:
        """Fasst aufeinanderfolgende Frames gleichen Typs zu Regionen zusammen."""
        if not frame_types:
            return []

        regions = []
        current_type = frame_types[0][0]
        current_start = 0
        confidences = [frame_types[0][1]]

        for i in range(1, len(frame_types)):
            frame_type, confidence = frame_types[i]

            if frame_type == current_type:
                # Same type, continue
                confidences.append(confidence)
            else:
                # Type changed, create region
                start_sample = current_start * hop_length
                end_sample = i * hop_length
                avg_confidence = float(np.mean(confidences))

                regions.append(
                    AudioRegion(
                        region_type=current_type,
                        start_sample=start_sample,
                        end_sample=end_sample,
                        confidence=avg_confidence,
                    )
                )

                # Start new region
                current_type = frame_type
                current_start = i
                confidences = [confidence]

        # Add last region
        start_sample = current_start * hop_length
        end_sample = len(frame_types) * hop_length
        avg_confidence = float(np.mean(confidences))

        regions.append(
            AudioRegion(
                region_type=current_type,
                start_sample=start_sample,
                end_sample=end_sample,
                confidence=avg_confidence,
            )
        )

        return regions


# ==============================================================================
# Region Analyzer
# ==============================================================================


class RegionAnalyzer:
    """Analysiert Eigenschaften von Audio-Regionen."""

    def analyze_region(self, audio: np.ndarray, region: AudioRegion, sr: int) -> dict[str, Any]:
        """
        Analysiert eine Region detailliert.

        Args:
            audio: Full audio signal
            region: AudioRegion to analyze
            sr: Sample rate

        Returns:
            Dict mit Analysis-Resultaten
        """
        # Extract region audio
        region_audio = audio[region.start_sample : region.end_sample]

        if len(region_audio) == 0:
            return {"error": "Empty region"}

        analysis = {
            "region_type": region.region_type.value,
            "duration_s": len(region_audio) / sr,
        }

        # Spectral analysis
        try:
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=region_audio, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=region_audio, sr=sr)))
            spectral_contrast = librosa.feature.spectral_contrast(y=region_audio, sr=sr)

            analysis.update(
                {
                    "spectral_centroid": spectral_centroid,
                    "spectral_bandwidth": spectral_bandwidth,
                    "spectral_contrast_mean": float(np.mean(spectral_contrast)),
                }
            )
        except Exception as e:
            logger.warning("Spectral Analyse fehlgeschlagen: %s", e)

        # Dynamic analysis
        rms = np.sqrt(np.mean(region_audio**2))
        peak = float(np.max(np.abs(region_audio)))
        crest_factor = peak / (rms + 1e-10)

        analysis.update(
            {
                "rms": float(rms),
                "peak": peak,
                "crest_factor": float(crest_factor),
            }
        )

        # Region-specific features
        if region.region_type == RegionType.SPEECH:
            # F0 for speech
            try:
                f0, voiced_flag, _ = librosa.pyin(region_audio, sr=sr, fmin=80, fmax=400)
                voiced_f0 = f0[voiced_flag]
                if len(voiced_f0) > 0:
                    analysis["f0_mean"] = float(np.nanmean(voiced_f0))
                    analysis["f0_std"] = float(np.nanstd(voiced_f0))
                    analysis["voiced_ratio"] = float(np.sum(voiced_flag) / len(voiced_flag))
            except Exception as e:
                logger.warning("F0 Analyse fehlgeschlagen: %s", e)

        elif region.region_type == RegionType.MUSIC:
            # Tempo for music
            try:
                tempo, _ = librosa.beat.beat_track(y=region_audio, sr=sr)  # type: ignore[attr-defined]
                analysis["tempo_bpm"] = float(np.asarray(tempo).flat[0])
            except Exception as e:
                logger.warning("Tempo Analyse fehlgeschlagen: %s", e)

        return analysis


# ==============================================================================
# Region Metadata Provider (Normative Compliance)
# ==============================================================================


class RegionMetadataProvider:
    """
    Liefert region-spezifische Metadaten und Empfehlungen.

    **WICHTIG:** Trifft KEINE Entscheidungen, liefert nur Daten!
    Eigentliche Parameter werden vom Regulator gesetzt.
    """

    def __init__(self):
        # Empfehlungs-Templates (nicht bindend!)
        self.recommendation_templates = {
            RegionType.SILENCE: RegionRecommendation(
                region_type=RegionType.SILENCE,
                recommended_enhancement_strength=0.0,
                recommended_noise_reduction_strength=0.0,
                suggested_deessing=False,
                suggested_dynamic_eq=False,
                preserve_transients_priority=True,
                quality_sensitivity=0.95,
                reasoning="Silence region: Minimal intervention to preserve authenticity",
            ),
            RegionType.SPEECH: RegionRecommendation(
                region_type=RegionType.SPEECH,
                recommended_enhancement_strength=0.6,
                recommended_noise_reduction_strength=0.7,
                suggested_deessing=True,
                suggested_dynamic_eq=True,
                preserve_transients_priority=True,
                quality_sensitivity=0.85,
                reasoning="Speech region: Balance clarity enhancement with natural character",
            ),
            RegionType.MUSIC: RegionRecommendation(
                region_type=RegionType.MUSIC,
                recommended_enhancement_strength=0.4,
                recommended_noise_reduction_strength=0.5,
                suggested_deessing=False,
                suggested_dynamic_eq=True,
                preserve_transients_priority=True,
                quality_sensitivity=0.90,
                reasoning="Music region: Preserve dynamics and tonal character",
            ),
            RegionType.NOISE: RegionRecommendation(
                region_type=RegionType.NOISE,
                recommended_enhancement_strength=0.8,
                recommended_noise_reduction_strength=0.9,
                suggested_deessing=False,
                suggested_dynamic_eq=False,
                preserve_transients_priority=False,
                quality_sensitivity=0.80,
                reasoning="Noise region: Aggressive reduction acceptable",
            ),
            RegionType.MIXED: RegionRecommendation(
                region_type=RegionType.MIXED,
                recommended_enhancement_strength=0.5,
                recommended_noise_reduction_strength=0.6,
                suggested_deessing=False,
                suggested_dynamic_eq=True,
                preserve_transients_priority=True,
                quality_sensitivity=0.85,
                reasoning="Mixed region: Balanced approach for heterogeneous content",
            ),
        }

    def get_region_recommendation(self, region: AudioRegion, analysis: dict[str, Any]) -> RegionRecommendation:
        """
        Liefert Empfehlung für Region (NICHT BINDEND!).

        **WICHTIG:** Diese Empfehlung ist NUR eine Datenbasis.
        Finale Entscheidung trifft der Regulator!

        Args:
            region: AudioRegion
            analysis: Analysis results from RegionAnalyzer

        Returns:
            RegionRecommendation (Empfehlung, nicht Entscheidung)
        """
        # Start with template recommendation
        recommendation = self.recommendation_templates[region.region_type]

        # Adjust recommendation based on analysis (still only suggestion)
        if region.region_type == RegionType.SPEECH:
            rms = analysis.get("rms", 0.1)
            if rms < 0.05:  # Very quiet speech
                recommendation = RegionRecommendation(
                    region_type=RegionType.SPEECH,
                    recommended_enhancement_strength=0.7,
                    recommended_noise_reduction_strength=0.8,
                    suggested_deessing=True,
                    suggested_dynamic_eq=True,
                    preserve_transients_priority=True,
                    quality_sensitivity=0.80,
                    reasoning=f"Low RMS speech ({rms:.3f}): Higher enhancement suggested",
                )

        elif region.region_type == RegionType.MUSIC:
            crest_factor = analysis.get("crest_factor", 3.0)
            if crest_factor > 5.0:  # High dynamics
                recommendation = RegionRecommendation(
                    region_type=RegionType.MUSIC,
                    recommended_enhancement_strength=0.3,
                    recommended_noise_reduction_strength=0.4,
                    suggested_deessing=False,
                    suggested_dynamic_eq=False,
                    preserve_transients_priority=True,
                    quality_sensitivity=0.92,
                    reasoning=f"High dynamics music (crest={crest_factor:.1f}): Preserve original character",
                )

        return recommendation


# ==============================================================================
# Main Region Analysis System (Normative Compliance)
# ==============================================================================


class RegionAnalysisSystem:
    """
    Region-Analyse-System (NORMATIVE COMPLIANCE).

    **WICHTIG: Dieses System führt KEINE Verarbeitung durch!**

    Gemäß AURIK v8 Architekturregeln:
    - Detektiert und analysiert nur
    - Liefert Metadaten und Empfehlungen
    - Verarbeitung erfolgt durch adaptive_pipeline.py
    - Respektiert Epistemic Gate → Zonen → Conduct → Regulator

    Verwendung: Informiert Pipeline über Audio-Struktur für präzisere Entscheidungen.
    """

    def __init__(self):
        self.detector = RegionDetector()
        self.analyzer = RegionAnalyzer()
        self.metadata_provider = RegionMetadataProvider()
        # Use module-level logger (already configured with fallback)
        self.logger = logger

        self.logger.info("RegionAnalysisSystem initialized (analysis-only, non-invasive)")

    def analyze_audio_regions(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        Analysiert Audio-Regionen (KEINE VERARBEITUNG!).

        **WICHTIG:** Diese Methode verändert das Audio NICHT.
        Sie liefert nur Metadaten für die Pipeline.

        Args:
            audio: Input audio (mono)
            sr: Sample rate

        Returns:
            Dict mit Region-Analysen und Empfehlungen
        """
        self.logger.info("🔍 Starting region analysis (non-invasive)...")

        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # 1. Detect regions
        regions = self.detector.detect_regions(audio, sr)
        logger.info("  ├─ erkannt %s regions", len(regions))

        # 2. Analyze each region
        region_data = []
        for i, region in enumerate(regions):
            analysis = self.analyzer.analyze_region(audio, region, sr)
            recommendation = self.metadata_provider.get_region_recommendation(region, analysis)

            region_data.append(
                {
                    "region_id": i,
                    "type": region.region_type.value,
                    "start_sample": region.start_sample,
                    "end_sample": region.end_sample,
                    "duration_s": region.duration_samples / sr,
                    "confidence": region.confidence,
                    "analysis": analysis,
                    "recommendation": {
                        "enhancement_strength": recommendation.recommended_enhancement_strength,
                        "noise_reduction_strength": recommendation.recommended_noise_reduction_strength,
                        "deessing": recommendation.suggested_deessing,
                        "dynamic_eq": recommendation.suggested_dynamic_eq,
                        "preserve_transients": recommendation.preserve_transients_priority,
                        "quality_sensitivity": recommendation.quality_sensitivity,
                        "reasoning": recommendation.reasoning,
                    },
                }
            )

        logger.info("  ├─ Analyzed %s regions", len(region_data))

        # 3. Generate summary report
        report = {
            "total_regions": len(regions),
            "region_breakdown": {
                region_type.value: sum(1 for r in regions if r.region_type == region_type) for region_type in RegionType
            },
            "regions": region_data,
            "metadata": {"analysis_only": True, "no_audio_modification": True, "normative_compliance": True},
        }

        logger.info("  └─ Region Analyse vollstaendig: %s regions identified", len(regions))

        return report

    def export_region_analysis(self, report: dict[str, Any], output_path: str) -> None:
        """
        Exportiert Region-Analyse für Integration mit Pipeline.

        Args:
            report: Report from analyze_audio_regions()
            output_path: Path to export JSON
        """
        import json

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Region Analyse exported to %s", output_path)


# ==============================================================================
# Example Usage & Test
# ==============================================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    logging.info("\n" + "=" * 80)
    logging.info("REGION ANALYSIS TEST (Non-Invasive, Normative Compliance)")
    logging.info("=" * 80 + "\n")

    # Generate test signal with different regions
    sr = 44100
    duration = 5.0

    # Silence (0-0.5s)
    silence = np.zeros(int(sr * 0.5))

    # Speech-like (0.5-2s): 200 Hz fundamental + harmonics
    t_speech = np.linspace(0, 1.5, int(sr * 1.5))
    speech = (
        0.3 * np.sin(2 * np.pi * 200 * t_speech)
        + 0.1 * np.sin(2 * np.pi * 400 * t_speech)
        + 0.05 * np.sin(2 * np.pi * 600 * t_speech)
    )

    # Music-like (2-4s): Chord progression
    t_music = np.linspace(0, 2.0, int(sr * 2.0))
    music = (
        0.2 * np.sin(2 * np.pi * 440 * t_music)  # A
        + 0.2 * np.sin(2 * np.pi * 554.37 * t_music)  # C#
        + 0.2 * np.sin(2 * np.pi * 659.25 * t_music)  # E
    )

    # Noise (4-5s)
    noise = 0.1 * np.random.randn(int(sr * 1.0))

    # Concatenate
    audio = np.concatenate([silence, speech, music, noise])
    audio_original = audio.copy()

    # Test system (ANALYSIS ONLY)
    system = RegionAnalysisSystem()
    report = system.analyze_audio_regions(audio, sr)

    logging.info("\n📊 REGION BREAKDOWN:")
    logging.info("-" * 80)
    for region_type, count in report["region_breakdown"].items():
        logging.info(f"  {region_type:12s}: {count} region(s)")

    logging.info("\n✅ NORMATIVE COMPLIANCE:")
    logging.info("-" * 80)
    logging.info(f"  Analysis only: {report['metadata']['analysis_only']}")
    logging.info(f"  No audio modification: {report['metadata']['no_audio_modification']}")
    logging.info(f"  Audio unchanged: {np.array_equal(audio, audio_original)}")

    logging.info("\n📋 REGION RECOMMENDATIONS (for Pipeline):")
    logging.info("-" * 80)
    for region in report["regions"][:5]:  # Show first 5
        logging.info(f"Region {region['region_id']}: {region['type']:12s} ({region['duration_s']:.2f}s)")
        logging.info(f"  Recommendation: {region['recommendation']['reasoning']}")
        logging.info(
            f"  Enhancement: {region['recommendation']['enhancement_strength']:.1f}, "
            f"Noise Reduction: {region['recommendation']['noise_reduction_strength']:.1f}"
        )

    logging.info("\n" + "=" * 80)
    logging.info("✅ Test completed successfully (Normative Compliance Verified)!")
    logging.info("=" * 80 + "\n")
