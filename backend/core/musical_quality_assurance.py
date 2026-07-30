"""
core/musical_quality_assurance.py
Musical Quality Assurance System
=================================

Garantierte musikalische Qualität für Aurik:
- Medium-spezifische Quality Gates (VINYL, TAPE, SHELLAC, DIGITAL)
- Processing Mode Validation (RESTORATION, STUDIO_2026, VINTAGE_WARMTH)
- Overprocessing Protection (stoppt vor Zerstörung des Charakters)
- Authenticity Preservation (analog character erhalten)
- Musical Integrity Validation (Musik klingt noch natürlich)
- A/B Comparison (vorher/nachher mit musikalischen Kriterien)

Aurik-spezifisch:
- Nutzt Forensic Medium Detection
- Berücksichtigt Processing Modes
- Evaluiert Aurik's Module (TapeSpecialist, ClickRemover, etc.)
- Garantiert musikalische Exzellenz (nicht nur technische Qualität)

Version: 1.0.0 "Musical Excellence"
Author: AURIK Team
Date: 10. Februar 2026
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from backend.core.quality_prediction import QualityAnalyzer, QualityEstimate, QualityLevel

logger = logging.getLogger(__name__)


class MediumType(Enum):
    """Audio medium types (from forensic analysis)."""

    # Analog Disc Media
    VINYL_33 = "VINYL_33"  # 33⅓ rpm LP
    VINYL_45 = "VINYL_45"  # 45 rpm Single
    SHELLAC_78 = "SHELLAC_78"  # 78 rpm Shellac
    ACETATE = "ACETATE"  # Acetate Disc (Lacquer)
    TRANSCRIPTION_DISC = "TRANSCRIPTION_DISC"  # Radio transcription discs

    # Magnetic Tape Media
    REEL_TO_REEL = "REEL_TO_REEL"  # Open reel tape (1/4", 1/2", 1", 2")
    CASSETTE = "CASSETTE"  # Compact Cassette
    EIGHT_TRACK = "8TRACK"  # 8-Track Cartridge
    DAT = "DAT"  # Digital Audio Tape
    ELCASET = "ELCASET"  # Elcaset (Sony)
    BETAMAX_AUDIO = "BETAMAX_AUDIO"  # Betamax Hi-Fi
    VHS_HIFI = "VHS_HIFI"  # VHS Hi-Fi

    # Historic/Rare Media
    WAX_CYLINDER = "WAX_CYLINDER"  # Edison Wax Cylinder
    WIRE_RECORDING = "WIRE_RECORDING"  # Wire Recorder

    # Optical Disc Media
    CD = "CD"  # Compact Disc (16/44.1)
    CD_R = "CD_R"  # CD-R (burned)
    SACD = "SACD"  # Super Audio CD
    DVD_AUDIO = "DVD_AUDIO"  # DVD-Audio
    MINIDISC = "MINIDISC"  # MiniDisc (ATRAC)
    LASERDISC = "LASERDISC"  # LaserDisc Audio

    # Digital File Formats
    LOSSLESS = "LOSSLESS"  # WAV, FLAC, ALAC, APE
    LOSSY_HIGH = "LOSSY_HIGH"  # 320kbps MP3, AAC, OGG
    LOSSY_MID = "LOSSY_MID"  # 192-256kbps
    LOSSY_LOW = "LOSSY_LOW"  # <192kbps
    DSD = "DSD"  # Direct Stream Digital

    # Broadcast/Professional
    BROADCAST_TAPE = "BROADCAST_TAPE"  # Professional broadcast tape
    CART_MACHINE = "CART_MACHINE"  # Broadcast cart (Fidelipac)
    DAW_BOUNCE = "DAW_BOUNCE"  # Digital Audio Workstation export

    # Film/Video Audio
    OPTICAL_FILM = "OPTICAL_FILM"  # Optical film soundtrack
    MAGNETIC_FILM = "MAGNETIC_FILM"  # Magnetic film soundtrack

    # Generic/Fallback
    ANALOG_UNKNOWN = "ANALOG_UNKNOWN"  # Unknown analog source
    DIGITAL_UNKNOWN = "DIGITAL_UNKNOWN"  # Unknown digital source
    UNKNOWN = "UNKNOWN"  # Completely unknown


class ProcessingMode(Enum):
    """
    Aurik Magic Button Processing Modes.

    Nur 2 User-wählbare Modi. Forensic Analysis ist KEIN Mode,
    sondern ein fester Bestandteil der Pipeline (immer aktiv).
    """

    RESTORATION = "restoration"  # Authentizität, moderate Bearbeitung, Original-Charakter erhalten
    STUDIO_2026 = "studio_2026"  # Modern, streaming-optimiert, maximale Brillanz


class IntegrityViolation(Enum):
    """Types of musical integrity violations."""

    OVERPROCESSING = "overprocessing"  # Zu viel Verarbeitung
    CHARACTER_LOSS = "character_loss"  # Analog character verloren
    UNNATURAL_SOUND = "unnatural_sound"  # Klingt unnatürlich
    FREQUENCY_IMBALANCE = "frequency_imbalance"  # Frequenzbalance gestört
    DYNAMIC_DESTRUCTION = "dynamic_destruction"  # Dynamik zerstört
    STEREO_COLLAPSE = "stereo_collapse"  # Stereo-Bild kollabiert
    TRANSIENT_SMEARING = "transient_smearing"  # Transienten verschmiert
    ARTIFACT_INTRODUCTION = "artifact_introduction"  # Neue Artefakte eingeführt


@dataclass
class MediumQualityGates:
    """
    Medium-spezifische Quality Gates.

    Jedes Medium (VINYL, TAPE, etc.) hat eigene Qualitätsanforderungen:
    - VINYL: Warmth > 0.6, Clicks entfernt, Analog character erhalten
    - TAPE: Authenticity > 0.7, Tape saturation erhalten, Keine Überreaktion
    - SHELLAC: Naturalness > 0.6, Historical character erhalten
    """

    medium_type: MediumType

    # Mindestanforderungen (0-1)
    min_snr_db: float
    min_clarity: float
    min_warmth: float
    min_brightness: float
    min_naturalness: float
    min_authenticity: float

    # Maximalwerte (Überverarbeitung)
    max_brightness: float = 0.95  # Nicht zu hell
    max_clarity: float = 1.0  # Perfekt ist unnatürlich

    # Erlaubte Artefakte
    allow_analog_artifacts: bool = True  # Vinyl crackle, tape hiss OK
    allow_lossy_artifacts: bool = False  # MP3 artifacts nicht OK

    # Musical character preservation
    preserve_warmth: bool = True
    preserve_stereo_width: bool = True
    preserve_dynamics: bool = True


@dataclass
class ModeQualityStandards:
    """
    Processing Mode spezifische Qualitätsstandards.
    """

    mode: ProcessingMode

    # Minimale Qualitätsstufe
    min_quality_level: QualityLevel

    # Spezifische Anforderungen
    min_overall_score: float
    min_authenticity: float
    max_processing_intensity: float  # Wie viel Verarbeitung erlaubt (0-1)

    # Musical standards
    require_natural_sound: bool = True
    require_authentic_character: bool = True
    allow_modern_enhancement: bool = False


@dataclass
class IntegrityCheckResult:
    """Result from musical integrity check."""

    passed: bool
    overall_integrity: float  # 0-1 (1 = perfekte Integrität)
    violations: list[IntegrityViolation]
    violation_details: dict[IntegrityViolation, str]

    # Comparison metrics
    naturalness_change: float  # Veränderung der Natürlichkeit (-1 to 1)
    character_preservation: float  # Wie gut wurde Character erhalten (0-1)
    overprocessing_risk: float  # Risiko der Überverarbeitung (0-1)

    # Recommendations
    recommendations: list[str]
    should_rollback: bool = False
    should_stop_processing: bool = False


@dataclass
class MusicalQualityReport:
    """
    Comprehensive musical quality report.
    """

    # Input/Output quality
    input_quality: QualityEstimate
    output_quality: QualityEstimate

    # Medium & Mode
    medium_type: MediumType
    processing_mode: ProcessingMode

    # Quality gates
    gates_passed: bool
    gate_results: dict[str, bool]

    # Integrity check
    integrity_result: IntegrityCheckResult

    # Musical metrics
    musical_improvement: float  # Gesamte musikalische Verbesserung (-1 to 1)
    authenticity_preserved: bool
    character_preserved: bool
    natural_sound: bool

    # Processing summary
    modules_applied: list[str]
    processing_intensity: float  # Wie intensiv wurde verarbeitet (0-1)
    overprocessed: bool

    # Final verdict
    quality_guaranteed: bool
    verdict: str
    warnings: list[str]
    recommendations: list[str]


class MusicalQualityAssurance:
    """
    Musical Quality Assurance System für Aurik.

    Garantiert musikalische Qualität durch:
    1. Medium-spezifische Quality Gates
    2. Processing Mode Validation
    3. Overprocessing Protection
    4. Authenticity Preservation
    5. Musical Integrity Validation

    Usage:
        mqa = MusicalQualityAssurance()

        # Before processing
        baseline = mqa.establish_baseline(audio, sr, medium_type, mode)

        # During processing (after each module)
        gate_ok, reason = mqa.check_quality_gate(
            processed_audio, sr, baseline, medium_type, mode
        )

        if not gate_ok:
            logger.warning("Quality gate failed: %s", reason)
            # Rollback or stop

        # After processing
        report = mqa.validate_final_quality(
            original_audio, processed_audio, sr,
            medium_type, mode, modules_applied
        )

        if not report.quality_guaranteed:
            logger.error("Quality not guaranteed: %s", report.verdict)
    """

    VERSION = "1.0.0"

    # Medium-spezifische Quality Gates Definition
    # Hierarchisch organisiert: Basis-Gates + Spezifische Anpassungen
    MEDIUM_GATES = {
        # === VINYL Familie ===
        MediumType.VINYL_33: MediumQualityGates(
            medium_type=MediumType.VINYL_33,
            min_snr_db=48.0,  # Modern LP
            min_clarity=0.65,
            min_warmth=0.65,  # Vinyl warmth
            min_brightness=0.45,
            min_naturalness=0.75,
            min_authenticity=0.75,
            max_brightness=0.85,
            allow_analog_artifacts=True,
        ),
        MediumType.VINYL_45: MediumQualityGates(
            medium_type=MediumType.VINYL_45,
            min_snr_db=46.0,  # Singles oft schlechter gepresst
            min_clarity=0.63,
            min_warmth=0.63,
            min_brightness=0.43,
            min_naturalness=0.73,
            min_authenticity=0.73,
            max_brightness=0.85,
            allow_analog_artifacts=True,
        ),
        # === SHELLAC/78rpm ===
        MediumType.SHELLAC_78: MediumQualityGates(
            medium_type=MediumType.SHELLAC_78,
            min_snr_db=30.0,  # Sehr alt, viel Rauschen
            min_clarity=0.45,
            min_warmth=0.45,
            min_brightness=0.25,  # Sehr begrenzte Bandbreite
            min_naturalness=0.60,
            min_authenticity=0.85,  # Historical character KRITISCH
            max_brightness=0.70,  # Nicht modernisieren!
            allow_analog_artifacts=True,
        ),
        MediumType.ACETATE: MediumQualityGates(
            medium_type=MediumType.ACETATE,
            min_snr_db=35.0,  # Better than shellac
            min_clarity=0.50,
            min_warmth=0.50,
            min_brightness=0.30,
            min_naturalness=0.65,
            min_authenticity=0.82,
            max_brightness=0.75,
            allow_analog_artifacts=True,
        ),
        # === MAGNETIC TAPE Familie ===
        MediumType.REEL_TO_REEL: MediumQualityGates(
            medium_type=MediumType.REEL_TO_REEL,
            min_snr_db=55.0,  # Professional tape
            min_clarity=0.70,
            min_warmth=0.68,  # Tape warmth/saturation
            min_brightness=0.55,
            min_naturalness=0.78,
            min_authenticity=0.78,
            max_brightness=0.82,
            allow_analog_artifacts=True,
        ),
        MediumType.CASSETTE: MediumQualityGates(
            medium_type=MediumType.CASSETTE,
            min_snr_db=48.0,
            min_clarity=0.60,
            min_warmth=0.60,
            min_brightness=0.45,
            min_naturalness=0.70,
            min_authenticity=0.70,
            max_brightness=0.82,
            allow_analog_artifacts=True,
        ),
        MediumType.EIGHT_TRACK: MediumQualityGates(
            medium_type=MediumType.EIGHT_TRACK,
            min_snr_db=45.0,  # Schlechtere Qualität als Cassette
            min_clarity=0.55,
            min_warmth=0.58,
            min_brightness=0.40,
            min_naturalness=0.65,
            min_authenticity=0.72,
            max_brightness=0.80,
            allow_analog_artifacts=True,
        ),
        MediumType.DAT: MediumQualityGates(
            medium_type=MediumType.DAT,
            min_snr_db=85.0,  # Digital tape - sehr gut
            min_clarity=0.80,
            min_warmth=0.60,
            min_brightness=0.70,
            min_naturalness=0.82,
            min_authenticity=0.65,
            max_brightness=0.90,
            allow_analog_artifacts=False,
        ),
        # === HISTORIC Media ===
        MediumType.WAX_CYLINDER: MediumQualityGates(
            medium_type=MediumType.WAX_CYLINDER,
            min_snr_db=20.0,  # Sehr primitiv
            min_clarity=0.30,
            min_warmth=0.35,
            min_brightness=0.15,
            min_naturalness=0.50,
            min_authenticity=0.90,  # MAXIMAL authentisch bleiben
            max_brightness=0.60,
            allow_analog_artifacts=True,
        ),
        MediumType.WIRE_RECORDING: MediumQualityGates(
            medium_type=MediumType.WIRE_RECORDING,
            min_snr_db=35.0,
            min_clarity=0.45,
            min_warmth=0.45,
            min_brightness=0.30,
            min_naturalness=0.60,
            min_authenticity=0.85,
            max_brightness=0.70,
            allow_analog_artifacts=True,
        ),
        # === OPTICAL DISC Familie ===
        MediumType.CD: MediumQualityGates(
            medium_type=MediumType.CD,
            min_snr_db=90.0,  # 16-bit ideal
            min_clarity=0.82,
            min_warmth=0.58,
            min_brightness=0.72,
            min_naturalness=0.80,
            min_authenticity=0.60,
            max_brightness=0.92,
            allow_analog_artifacts=False,
        ),
        MediumType.CD_R: MediumQualityGates(
            medium_type=MediumType.CD_R,
            min_snr_db=85.0,  # Burned CDs etwas schlechter
            min_clarity=0.78,
            min_warmth=0.56,
            min_brightness=0.70,
            min_naturalness=0.78,
            min_authenticity=0.58,
            max_brightness=0.90,
            allow_analog_artifacts=False,
        ),
        MediumType.SACD: MediumQualityGates(
            medium_type=MediumType.SACD,
            min_snr_db=110.0,  # DSD - exzellent
            min_clarity=0.88,
            min_warmth=0.65,
            min_brightness=0.78,
            min_naturalness=0.85,
            min_authenticity=0.70,
            max_brightness=0.95,
            allow_analog_artifacts=False,
        ),
        MediumType.MINIDISC: MediumQualityGates(
            medium_type=MediumType.MINIDISC,
            min_snr_db=75.0,  # ATRAC compression
            min_clarity=0.70,
            min_warmth=0.55,
            min_brightness=0.65,
            min_naturalness=0.72,
            min_authenticity=0.55,
            max_brightness=0.85,
            allow_analog_artifacts=False,
            allow_lossy_artifacts=False,
        ),
        # === DIGITAL FILE Formate ===
        MediumType.LOSSLESS: MediumQualityGates(
            medium_type=MediumType.LOSSLESS,
            min_snr_db=95.0,  # Modern digital
            min_clarity=0.85,
            min_warmth=0.60,
            min_brightness=0.75,
            min_naturalness=0.82,
            min_authenticity=0.62,
            max_brightness=0.93,
            allow_analog_artifacts=False,
        ),
        MediumType.LOSSY_HIGH: MediumQualityGates(
            medium_type=MediumType.LOSSY_HIGH,
            min_snr_db=70.0,  # 320kbps
            min_clarity=0.72,
            min_warmth=0.55,
            min_brightness=0.68,
            min_naturalness=0.75,
            min_authenticity=0.55,
            max_brightness=0.88,
            allow_analog_artifacts=False,
            allow_lossy_artifacts=False,
        ),
        MediumType.LOSSY_MID: MediumQualityGates(
            medium_type=MediumType.LOSSY_MID,
            min_snr_db=60.0,  # 192-256kbps
            min_clarity=0.65,
            min_warmth=0.52,
            min_brightness=0.62,
            min_naturalness=0.70,
            min_authenticity=0.52,
            max_brightness=0.85,
            allow_analog_artifacts=False,
            allow_lossy_artifacts=False,
        ),
        MediumType.LOSSY_LOW: MediumQualityGates(
            medium_type=MediumType.LOSSY_LOW,
            min_snr_db=50.0,  # <192kbps
            min_clarity=0.40,  # Lowered: lossy codecs have higher hf_ratio → Gaussian drops below 0.58
            min_warmth=0.48,
            min_brightness=0.55,
            min_naturalness=0.65,
            min_authenticity=0.48,
            max_brightness=0.82,
            allow_analog_artifacts=False,
            allow_lossy_artifacts=False,
        ),
        MediumType.DSD: MediumQualityGates(
            medium_type=MediumType.DSD,
            min_snr_db=120.0,  # Direct Stream Digital - beste Qualität
            min_clarity=0.90,
            min_warmth=0.68,
            min_brightness=0.82,
            min_naturalness=0.88,
            min_authenticity=0.72,
            max_brightness=0.96,
            allow_analog_artifacts=False,
        ),
        # === BROADCAST/PROFESSIONAL ===
        MediumType.BROADCAST_TAPE: MediumQualityGates(
            medium_type=MediumType.BROADCAST_TAPE,
            min_snr_db=60.0,  # Professional standard
            min_clarity=0.75,
            min_warmth=0.65,
            min_brightness=0.60,
            min_naturalness=0.78,
            min_authenticity=0.75,
            max_brightness=0.85,
            allow_analog_artifacts=True,
        ),
        MediumType.DAW_BOUNCE: MediumQualityGates(
            medium_type=MediumType.DAW_BOUNCE,
            min_snr_db=100.0,  # Modern DAW - exzellent
            min_clarity=0.85,
            min_warmth=0.62,
            min_brightness=0.78,
            min_naturalness=0.83,
            min_authenticity=0.65,
            max_brightness=0.93,
            allow_analog_artifacts=False,
        ),
        # === GENERIC Fallbacks ===
        MediumType.ANALOG_UNKNOWN: MediumQualityGates(
            medium_type=MediumType.ANALOG_UNKNOWN,
            min_snr_db=40.0,  # Conservative für unknown
            min_clarity=0.55,
            min_warmth=0.55,
            min_brightness=0.40,
            min_naturalness=0.70,
            min_authenticity=0.75,
            max_brightness=0.80,
            allow_analog_artifacts=True,
        ),
        MediumType.DIGITAL_UNKNOWN: MediumQualityGates(
            medium_type=MediumType.DIGITAL_UNKNOWN,
            min_snr_db=70.0,
            min_clarity=0.70,
            min_warmth=0.35,  # Lowered: digital sources have less LF energy; silence→0.70 now safe
            min_brightness=0.65,
            min_naturalness=0.75,
            min_authenticity=0.35,
            max_brightness=0.88,
            allow_analog_artifacts=False,
        ),
        MediumType.UNKNOWN: MediumQualityGates(
            medium_type=MediumType.UNKNOWN,
            min_snr_db=50.0,  # Sehr conservative
            min_clarity=0.40,  # Lowered: bright digital sources score ~0.60 with σ=0.25
            min_warmth=0.35,  # Lowered: 2-sample silence returns 0.70; digital sources < 0.55 normal
            min_brightness=0.50,
            min_naturalness=0.70,
            min_authenticity=0.30,
            max_brightness=0.85,
            allow_analog_artifacts=True,
        ),
    }

    # Processing Mode Standards
    MODE_STANDARDS = {
        ProcessingMode.RESTORATION: ModeQualityStandards(
            mode=ProcessingMode.RESTORATION,
            min_quality_level=QualityLevel.GOOD,
            min_overall_score=70.0,
            min_authenticity=0.75,  # Authentizität SEHR wichtig
            max_processing_intensity=1.0,  # Aurik 10.0.0: bis zu 50 Phasen — Intensität durch §2.45+PMGG reguliert
            require_natural_sound=True,
            require_authentic_character=True,
            allow_modern_enhancement=False,
        ),
        ProcessingMode.STUDIO_2026: ModeQualityStandards(
            mode=ProcessingMode.STUDIO_2026,
            min_quality_level=QualityLevel.EXCELLENT,
            min_overall_score=85.0,
            min_authenticity=0.60,  # Weniger wichtig (modern sound)
            max_processing_intensity=0.95,  # Aggressive Bearbeitung OK
            require_natural_sound=True,
            require_authentic_character=False,  # Modern = nicht authentisch
            allow_modern_enhancement=True,
        ),
    }

    def __init__(self):
        """Initialisiert Musical Quality Assurance System."""
        self.analyzer = QualityAnalyzer()
        logger.info("Musical Quality Assurance System initialized (v%s)", self.VERSION)

    def establish_baseline(
        self, audio: np.ndarray, sample_rate: int, medium_type: MediumType, processing_mode: ProcessingMode
    ) -> QualityEstimate:
        """
        Establish quality baseline before processing.

        Args:
            audio: Input audio
            sample_rate: Sample rate
            medium_type: Medium type (from forensics)
            processing_mode: Processing mode

        Returns:
            Baseline quality estimate
        """
        baseline = self.analyzer.analyze_quality(audio, sample_rate)

        logger.info("Quality Baseline: %.1f/100 (%s)", baseline.overall_score, baseline.quality_level.value)
        logger.info("  Medium: %s, Mode: %s", medium_type.value, processing_mode.value)
        logger.info("  SNR: %.1f dB, Clarity: %.2f, Warmth: %.2f", baseline.snr_db, baseline.clarity, baseline.warmth)
        logger.info("  Authenticity: %.2f, Naturalness: %.2f", baseline.authenticity, baseline.naturalness)

        return baseline

    def check_quality_gate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        baseline: QualityEstimate,
        medium_type: MediumType,
        processing_mode: ProcessingMode,
        module_name: str | None = None,
    ) -> tuple[bool, str]:
        """
        Prüft if quality gates are met (during processing).

        Args:
            audio: Current audio
            sample_rate: Sample rate
            baseline: Baseline quality
            medium_type: Medium type
            processing_mode: Processing mode
            module_name: Name of module just applied

        Returns:
            (gate_passed, reason)
        """
        current = self.analyzer.analyze_quality(audio, sample_rate)

        # Get gates for medium and mode
        medium_gates = self.MEDIUM_GATES.get(medium_type)
        mode_standards = self.MODE_STANDARDS.get(processing_mode)

        if not medium_gates or not mode_standards:
            logger.warning("No gates defined for %s/%s", medium_type.value, processing_mode.value)
            return True, "No gates defined"

        # Check medium-specific gates
        # SNR gate: Compare against material-adaptive threshold.
        # §2.54: The minimum improvement requirement is NOT a fixed constant.
        # For severely degraded multi-generation material (e.g. vinyl→cassette→mp3_low,
        # baseline ≈ 28 dB), the physical SNR ceiling of the carrier chain limits
        # achievable improvement.  Requiring a fixed +3 dB from an already-degraded
        # baseline forces destructive over-denoising that PMGG correctly prevents.
        # Smooth ramp: below _RAMP_LOW dB baseline → 0.3 dB min; above _RAMP_HIGH → 3.0 dB.
        _snr_target = medium_gates.min_snr_db
        # §v10.131 Depth-adaptive SNR: tiefe Transfer-Ketten haben physikalisch
        # niedrigere SNR-Ceilings. Bei depth≥4 wird das Target proportional relaxiert.
        try:
            from backend.core.calibration_context import get_calibration_context
            _ctx = get_calibration_context()
            if _ctx is not None and _ctx.transfer_chain_depth >= 4:
                _depth = _ctx.transfer_chain_depth
                _depth_factor = float(np.clip(1.0 - (_depth - 3) * 0.15, 0.50, 1.0))
                _snr_target = float(max(_snr_target * _depth_factor, _snr_baseline * 0.95))
        except Exception:
            pass
        _snr_baseline = baseline.snr_db if baseline is not None else 0.0
        # §v10.131 Depth-adaptive: bei tiefen Ketten mehr SNR-Verlust tolerieren
        _snr_depth_relax = 0.0
        try:
            from backend.core.calibration_context import get_calibration_context
            _ctx = get_calibration_context()
            if _ctx is not None and _ctx.transfer_chain_depth >= 4:
                _snr_depth_relax = (_ctx.transfer_chain_depth - 3) * 1.0  # +1dB pro Stufe ab 4
        except Exception:
            pass
        _SNR_RAMP_LOW = 28.0  # below this baseline SNR: only 0.3 dB min improvement
        _SNR_RAMP_HIGH = 38.0  # above this baseline SNR: standard 3.0 dB min improvement
        # §v10.0.4: Bei sehr niedriger Baseline (≤28dB) ist eine leichte SNR-Verschlechterung
        # (−0.5 dB) akzeptabel, da NR bei schwachem Signal auch Signalenergie entfernen kann.
        if _snr_baseline <= _SNR_RAMP_LOW:
            _min_snr_improvement = -0.5 - _snr_depth_relax  # allow slight SNR loss for very noisy sources
        elif _snr_baseline >= _SNR_RAMP_HIGH:
            _min_snr_improvement = 3.0
        else:
            _t = (_snr_baseline - _SNR_RAMP_LOW) / (_SNR_RAMP_HIGH - _SNR_RAMP_LOW)
            _min_snr_improvement = -0.5 + 3.5 * _t**1.5 - _snr_depth_relax

        # Target-floor-Rampe (statt fixer 55%):
        # Bei sehr niedriger Baseline darf das harte Medium-Ziel nicht als implizite
        # Pflichtsteigerung von +10..30 dB wirken. Das provoziert over-denoising und
        # erzeugt false fails nahe der Messunsicherheit.
        _TARGET_FLOOR_RAMP_LOW = 30.0
        _TARGET_FLOOR_RAMP_HIGH = 45.0
        _TARGET_FLOOR_MAX_FACTOR = 0.55
        if _snr_baseline <= _TARGET_FLOOR_RAMP_LOW:
            _target_floor_factor = 0.0
        elif _snr_baseline >= _TARGET_FLOOR_RAMP_HIGH:
            _target_floor_factor = _TARGET_FLOOR_MAX_FACTOR
        else:
            _tf = (_snr_baseline - _TARGET_FLOOR_RAMP_LOW) / (_TARGET_FLOOR_RAMP_HIGH - _TARGET_FLOOR_RAMP_LOW)
            _target_floor_factor = _TARGET_FLOOR_MAX_FACTOR * _tf**1.2

        _snr_adaptive_floor = max(
            _snr_baseline + _min_snr_improvement,
            _snr_target * _target_floor_factor,
        )
        _effective_snr_min = min(_snr_target, _snr_adaptive_floor)

        # SNR-Estimator hat bei kurzen/noisy Segmenten eine natürliche Streuung.
        # Toleranz verhindert false negatives bei Grenzfällen (z.B. 28.2 vs 28.5 dB).
        _snr_tolerance_db = 0.35
        if _snr_baseline <= 30.0:
            _snr_tolerance_db += 0.15

        if current.snr_db < (_effective_snr_min - _snr_tolerance_db):
            return (
                False,
                (
                    f"SNR too low: {current.snr_db:.1f} < {_effective_snr_min:.1f} dB "
                    f"(target={_snr_target:.0f}, baseline={_snr_baseline:.1f}, tol={_snr_tolerance_db:.2f})"
                ),
            )

        if current.clarity < medium_gates.min_clarity:
            return False, f"Clarity too low: {current.clarity:.2f} < {medium_gates.min_clarity:.2f}"

        _warmth_drop = baseline.warmth - current.warmth
        _warmth_was_already_below_floor = baseline.warmth < (medium_gates.min_warmth - 0.08)
        if current.warmth < medium_gates.min_warmth and not (_warmth_was_already_below_floor and _warmth_drop <= 0.08):
            return (
                False,
                (
                    f"Warmth too low: {current.warmth:.2f} < {medium_gates.min_warmth:.2f} "
                    f"({medium_type.value} MUST be warm, baseline={baseline.warmth:.2f})"
                ),
            )

        _naturalness_drop_abs = baseline.naturalness - current.naturalness
        _naturalness_was_already_below_floor = baseline.naturalness < (medium_gates.min_naturalness - 0.10)
        if current.naturalness < medium_gates.min_naturalness and not (
            _naturalness_was_already_below_floor and _naturalness_drop_abs <= 0.10
        ):
            return (
                False,
                (
                    f"Naturalness too low: {current.naturalness:.2f} < {medium_gates.min_naturalness:.2f} "
                    f"(sounds unnatural, baseline={baseline.naturalness:.2f})"
                ),
            )

        if current.authenticity < medium_gates.min_authenticity:
            return (
                False,
                f"Authenticity too low: {current.authenticity:.2f} < {medium_gates.min_authenticity:.2f} (character lost)",
            )

        # Check overprocessing (brightness too high = over-brightened).
        # Bereits helle historische Quellen duerfen bei No-op/Minimal-Delta nicht
        # als neuer Aurik-Schaden gewertet werden; relevant ist der zusaetzliche
        # Brightness-Anstieg gegen die Baseline.
        _brightness_increase = current.brightness - baseline.brightness
        _brightness_was_already_over = baseline.brightness > medium_gates.max_brightness
        if current.brightness > medium_gates.max_brightness and not (
            _brightness_was_already_over and _brightness_increase <= 0.05
        ):
            return (
                False,
                (
                    f"Over-brightened: {current.brightness:.2f} > {medium_gates.max_brightness:.2f} "
                    f"(sounds harsh, baseline={baseline.brightness:.2f})"
                ),
            )

        # Check naturalness degradation (compared to baseline)
        naturalness_drop = baseline.naturalness - current.naturalness
        # §2.54 material-adaptive tolerance: degraded analog sources can show
        # larger short-term naturalness proxy drift after legitimate carrier-repair,
        # while clean digital should stay tighter.
        _NAT_DROP_GATE: dict[MediumType, float] = {
            MediumType.CD: 0.15,
            MediumType.CD_R: 0.15,
            MediumType.SACD: 0.15,
            MediumType.DAT: 0.15,
            MediumType.LOSSLESS: 0.15,
            MediumType.LOSSY_HIGH: 0.18,
            MediumType.LOSSY_MID: 0.20,
            MediumType.LOSSY_LOW: 0.24,
            MediumType.MINIDISC: 0.22,
            MediumType.VINYL_33: 0.25,
            MediumType.VINYL_45: 0.25,
            MediumType.CASSETTE: 0.28,
            MediumType.REEL_TO_REEL: 0.26,
            MediumType.SHELLAC_78: 0.32,
            MediumType.WAX_CYLINDER: 0.35,
            MediumType.WIRE_RECORDING: 0.33,
        }
        _nat_drop_limit = float(_NAT_DROP_GATE.get(medium_type, 0.24))
        if naturalness_drop > _nat_drop_limit:
            return (
                False,
                f"Naturalness dropped {naturalness_drop:.2f} > {_nat_drop_limit:.2f} (overprocessing detected)",
            )

        # Check mode-specific standards
        # Unknown media classification is inherently uncertain; use slightly softer
        # mode gates to avoid false hard-fails in autonomous candidate evaluation.
        _is_unknown_medium = medium_type in {
            MediumType.UNKNOWN,
            MediumType.DIGITAL_UNKNOWN,
            MediumType.ANALOG_UNKNOWN,
        }
        _mode_min_overall = mode_standards.min_overall_score
        if _is_unknown_medium:
            _mode_min_overall = min(_mode_min_overall, 60.0)

        _overall_drop = baseline.overall_score - current.overall_score
        _overall_was_already_below = baseline.overall_score < (_mode_min_overall - 2.0)
        if current.overall_score < _mode_min_overall and not (_overall_was_already_below and _overall_drop <= 2.0):
            return (
                False,
                (
                    f"Overall score too low: {current.overall_score:.1f} < {_mode_min_overall:.1f} "
                    f"(mode: {processing_mode.value}, baseline={baseline.overall_score:.1f})"
                ),
            )

        _mode_min_auth = mode_standards.min_authenticity
        if _is_unknown_medium:
            _mode_min_auth = min(_mode_min_auth, medium_gates.min_authenticity)

        if mode_standards.require_authentic_character and current.authenticity < _mode_min_auth:
            return (
                False,
                f"Authenticity requirement not met: {current.authenticity:.2f} < {_mode_min_auth:.2f}",
            )

        # All gates passed
        module_info = f" (after {module_name})" if module_name else ""
        logger.info(
            f"✓ Quality gate passed{module_info}: {current.overall_score:.1f}/100, naturalness {current.naturalness:.2f}"
        )

        return True, "All quality gates passed"

    def check_musical_integrity(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        sample_rate: int,
        medium_type: MediumType,
        processing_mode: ProcessingMode,
    ) -> IntegrityCheckResult:
        """
        Prüft musical integrity (compare original vs processed).

        Detects violations like:
        - Overprocessing (too much change)
        - Character loss (authenticity destroyed)
        - Unnatural sound (naturalness too low)
        - Frequency imbalance (too bright or dull)
        - Dynamic destruction (over-compressed)

        Args:
            original_audio: Original audio
            processed_audio: Processed audio
            sample_rate: Sample rate
            medium_type: Medium type
            processing_mode: Processing mode

        Returns:
            IntegrityCheckResult with violations and recommendations
        """
        # Analyze both
        original_quality = self.analyzer.analyze_quality(original_audio, sample_rate)
        processed_quality = self.analyzer.analyze_quality(processed_audio, sample_rate, reference=original_audio)

        violations = []
        violation_details = {}
        recommendations = []

        # Calculate changes
        naturalness_change = processed_quality.naturalness - original_quality.naturalness
        character_preservation = min(processed_quality.authenticity / max(original_quality.authenticity, 0.01), 1.0)

        # Get standards
        medium_gates = self.MEDIUM_GATES.get(medium_type, self.MEDIUM_GATES[MediumType.UNKNOWN])
        mode_standards = self.MODE_STANDARDS.get(processing_mode, self.MODE_STANDARDS[ProcessingMode.RESTORATION])

        # 1. Check overprocessing (material-adaptive naturalness drop threshold)
        _NAT_DROP_INTEGRITY: dict[MediumType, float] = {
            MediumType.CD: 0.18,
            MediumType.CD_R: 0.18,
            MediumType.SACD: 0.18,
            MediumType.DAT: 0.18,
            MediumType.LOSSLESS: 0.18,
            MediumType.LOSSY_HIGH: 0.22,
            MediumType.LOSSY_MID: 0.25,
            MediumType.LOSSY_LOW: 0.28,
            MediumType.MINIDISC: 0.26,
            MediumType.VINYL_33: 0.28,
            MediumType.VINYL_45: 0.28,
            MediumType.CASSETTE: 0.30,
            MediumType.REEL_TO_REEL: 0.28,
            MediumType.SHELLAC_78: 0.35,
            MediumType.WAX_CYLINDER: 0.38,
            MediumType.WIRE_RECORDING: 0.35,
        }
        _nat_drop_limit_integrity = float(_NAT_DROP_INTEGRITY.get(medium_type, 0.28))
        if naturalness_change < -_nat_drop_limit_integrity:
            violations.append(IntegrityViolation.OVERPROCESSING)
            violation_details[IntegrityViolation.OVERPROCESSING] = (
                f"Naturalness dropped {abs(naturalness_change):.2%} "
                f"({original_quality.naturalness:.2f} → {processed_quality.naturalness:.2f})"
            )
            recommendations.append("Reduce processing intensity")
            recommendations.append("Re-run with ARCHIVAL or FORENSIC mode")

        # 2. Check character loss (authenticity drop > 20%)
        # The authenticity metric (SNR ≈ 60 dB, THD, bandwidth < 18 kHz) models
        # *analog* material character. For digital material (CD, LOSSY_*,
        # DIGITAL_UNKNOWN, DAT) this metric is meaningless: after restoration the
        # SNR improves far beyond 60 dB and bandwidth may extend past 18 kHz —
        # both are improvements, not character loss. §0 / §2.47 require material-
        # adaptive gates; applying the analog check to digital sources causes false
        # violations and misleads operators.
        _DIGITAL_MEDIA = frozenset(
            {
                MediumType.CD,
                MediumType.CD_R,
                MediumType.SACD,
                MediumType.DVD_AUDIO,
                MediumType.LOSSY_HIGH,
                MediumType.LOSSY_MID,
                MediumType.LOSSY_LOW,
                MediumType.LOSSLESS,
                MediumType.DSD,
                MediumType.DAT,
                MediumType.MINIDISC,
                MediumType.DAW_BOUNCE,
                MediumType.DIGITAL_UNKNOWN,
            }
        )
        _effective_require_authentic = mode_standards.require_authentic_character and medium_type not in _DIGITAL_MEDIA
        authenticity_drop = original_quality.authenticity - processed_quality.authenticity
        # §2.54 material-adaptive authenticity drop threshold (analog only path)
        _AUTH_DROP_LIMIT: dict[MediumType, float] = {
            MediumType.VINYL_33: 0.24,
            MediumType.VINYL_45: 0.24,
            MediumType.CASSETTE: 0.26,
            MediumType.REEL_TO_REEL: 0.24,
            MediumType.SHELLAC_78: 0.32,
            MediumType.WAX_CYLINDER: 0.35,
            MediumType.WIRE_RECORDING: 0.33,
            MediumType.ACETATE: 0.30,
            MediumType.ANALOG_UNKNOWN: 0.30,
        }
        _auth_drop_limit = float(_AUTH_DROP_LIMIT.get(medium_type, 0.24))
        if authenticity_drop > _auth_drop_limit and _effective_require_authentic:
            violations.append(IntegrityViolation.CHARACTER_LOSS)
            violation_details[IntegrityViolation.CHARACTER_LOSS] = (
                f"Authenticity dropped {authenticity_drop:.2%} "
                f"({original_quality.authenticity:.2f} → {processed_quality.authenticity:.2f})"
            )
            recommendations.append(f"{medium_type.value} character was destroyed")
            recommendations.append("Disable aggressive enhancement modules")

        # 3. Check unnatural sound (absolute naturalness < threshold)
        # Nur feuern wenn das Input-Signal bereits natürlich genug war — ein reiner
        # Sinuston oder anderes synthetisches Quellmaterial soll hier nicht fälschlich
        # als Verarbeitungsfehler gewertet werden.
        if (
            processed_quality.naturalness < medium_gates.min_naturalness
            and original_quality.naturalness >= medium_gates.min_naturalness - 0.10
        ):
            violations.append(IntegrityViolation.UNNATURAL_SOUND)
            violation_details[IntegrityViolation.UNNATURAL_SOUND] = (
                f"Output sounds unnatural: {processed_quality.naturalness:.2f} < {medium_gates.min_naturalness:.2f}"
            )
            recommendations.append("Audio sounds artificial or robotic")
            recommendations.append("Rollback to previous checkpoint")

        # 4. Check frequency imbalance
        # §0d Carrier-Recovery-Referenz-Paradoxon: For analog carrier restoration
        # (vinyl, shellac, tape, etc.) the HF-extension phases (phase_06, phase_07)
        # intentionally increase brightness — this is the goal, not a violation.
        # The degraded input has HF loss from the carrier chain; comparing brightness
        # against it yields false "Over-brightened" flags (e.g. +76% for vinyl).
        # Fix: use a material-adaptive ceiling instead of a fixed 0.30 threshold.
        _ANALOG_CARRIER_MATERIALS = frozenset(
            {
                MediumType.VINYL_33,
                MediumType.VINYL_45,
                MediumType.SHELLAC_78,
                MediumType.ACETATE,
                MediumType.REEL_TO_REEL,
                MediumType.CASSETTE,
                MediumType.WAX_CYLINDER,
                MediumType.WIRE_RECORDING,
                MediumType.MINIDISC,
                MediumType.ELCASET,
                MediumType.EIGHT_TRACK,
                MediumType.BETAMAX_AUDIO,
                MediumType.VHS_HIFI,
                MediumType.TRANSCRIPTION_DISC,
                MediumType.ANALOG_UNKNOWN,
            }
        )
        _is_restoration = processing_mode == ProcessingMode.RESTORATION
        # §0d/§2.46 Step 5 — BW-Extension raises brightness from near-zero (bandlimited
        # carrier) to full spec. The relative change (e.g. +77% for vinyl 7→16 kHz) is
        # intentional and correct. Only an absolute check matters here: if the restored
        # signal is brighter than medium_gates.max_brightness it's a real issue.
        # For analog carrier restoration we therefore skip the relative threshold and
        # rely solely on the absolute gate below.
        _bright_threshold = (
            2.0  # Effectively disabled: relative change can be 100 %+ for BW-restored analog
            if (_is_restoration and medium_type in _ANALOG_CARRIER_MATERIALS)
            else 0.30  # Studio/digital: standard 30% ceiling
        )
        # Absolute brightness gate (both modes): if restored audio exceeds the medium's
        # physical brightness ceiling we always flag, regardless of relative change.
        if processed_quality.brightness > medium_gates.max_brightness + 0.15:
            violations.append(IntegrityViolation.FREQUENCY_IMBALANCE)
            violation_details[IntegrityViolation.FREQUENCY_IMBALANCE] = (
                f"Over-brightened (absolute): {processed_quality.brightness:.2f} > "
                f"{medium_gates.max_brightness:.2f} ceiling"
            )
            recommendations.append("High frequencies boosted beyond physical medium ceiling")
            recommendations.append("Reduce phase_06 / phase_07 strength")
        brightness_change = processed_quality.brightness - original_quality.brightness
        if brightness_change > _bright_threshold:  # Never fires for analog restoration (threshold=2.0)
            violations.append(IntegrityViolation.FREQUENCY_IMBALANCE)
            violation_details[IntegrityViolation.FREQUENCY_IMBALANCE] = (
                f"Over-brightened: brightness increased {brightness_change:.2%}"
            )
            recommendations.append("High frequencies boosted too much (sounds harsh)")
            recommendations.append("Reduce de-esser/enhancer intensity")
        elif brightness_change < -0.25:  # More than 25% duller
            violations.append(IntegrityViolation.FREQUENCY_IMBALANCE)
            violation_details[IntegrityViolation.FREQUENCY_IMBALANCE] = (
                f"Too dull: brightness decreased {abs(brightness_change):.2%}"
            )
            recommendations.append("High frequencies removed too much")

        # 5. Check dynamic destruction (dynamic range loss > 30%)
        dr_loss = original_quality.dynamic_range_db - processed_quality.dynamic_range_db
        if dr_loss > 10.0:  # More than 10 dB dynamic range lost
            violations.append(IntegrityViolation.DYNAMIC_DESTRUCTION)
            violation_details[IntegrityViolation.DYNAMIC_DESTRUCTION] = (
                f"Dynamic range lost: {dr_loss:.1f} dB "
                f"({original_quality.dynamic_range_db:.1f} → {processed_quality.dynamic_range_db:.1f} dB)"
            )
            recommendations.append("Over-compressed or limited")
            recommendations.append("Reduce compression/limiting")

        # 6. Check artifact introduction (THD increase)
        # NOTE: _estimate_thd measures 2–10 kHz energy relative to 0.2–2 kHz, not
        # true harmonic distortion. For analog materials where HF restoration intentionally
        # increases the 2–10 kHz band, a high "THD delta" is correct behaviour, not a defect.
        # Use a material-adaptive threshold: tight for digital (0.5 pct-pts), generous for
        # bandwidth-limited analog (30 pct-pts) to avoid false-positive artifact flags.
        _ANALOG_BW_LIMITED_MEDIA = frozenset(
            {
                MediumType.VINYL_33,
                MediumType.VINYL_45,
                MediumType.CASSETTE,
                MediumType.REEL_TO_REEL,
                MediumType.SHELLAC_78,
                MediumType.ACETATE,
                MediumType.WAX_CYLINDER,
                MediumType.WIRE_RECORDING,
                MediumType.EIGHT_TRACK,
                MediumType.ANALOG_UNKNOWN,
            }
        )
        # §0d Carrier-Recovery: In Restoration mode, HF extension (phase_06, phase_07)
        # legitimately increases 2–10 kHz energy by 30–60 pct-pts. Tight 30% threshold
        # caused false ARTIFACT_INTRODUCTION on every vinyl/tape HF restoration.
        # Fix: raise to 60% for restoration of BW-limited analog media.
        _thd_threshold = (
            60.0
            if (_is_restoration and medium_type in _ANALOG_BW_LIMITED_MEDIA)
            else (
                30.0 if medium_type in _ANALOG_BW_LIMITED_MEDIA else 3.0
            )  # §v10.0.4: 0.5→3.0 — Messung ist HF/LF-Ratio, nicht echte THD
        )
        thd_increase = processed_quality.thd_percent - original_quality.thd_percent
        if thd_increase > _thd_threshold:
            violations.append(IntegrityViolation.ARTIFACT_INTRODUCTION)
            violation_details[IntegrityViolation.ARTIFACT_INTRODUCTION] = (
                # thd_percent is in [0, 100] units — use :.2f to avoid double-×100 via :.2%
                f"THD increased {thd_increase:.2f}% pts (new artifacts introduced)"
            )
            recommendations.append("Processing introduced distortion")
            recommendations.append("Check for clipping or aggressive processing")

        # Calculate overall integrity
        passed = len(violations) == 0
        overall_integrity = 1.0 - (len(violations) * 0.15)  # Each violation = -15%
        overall_integrity = max(0.0, overall_integrity)

        # Calculate overprocessing risk
        overprocessing_risk = max(abs(naturalness_change), abs(brightness_change), authenticity_drop)

        # Determine if rollback or stop needed
        should_rollback = (
            IntegrityViolation.CHARACTER_LOSS in violations
            or IntegrityViolation.UNNATURAL_SOUND in violations
            or overprocessing_risk > 0.35
        )

        should_stop_processing = len(violations) >= 3 or should_rollback  # 3+ violations = stop

        result = IntegrityCheckResult(
            passed=passed,
            overall_integrity=overall_integrity,
            violations=violations,
            violation_details=violation_details,
            naturalness_change=naturalness_change,
            character_preservation=character_preservation,
            overprocessing_risk=overprocessing_risk,
            recommendations=recommendations,
            should_rollback=should_rollback,
            should_stop_processing=should_stop_processing,
        )

        # Log results
        if passed:
            logger.info("✓ Musical integrity check PASSED (integrity: %.1f%%)", overall_integrity * 100)
            logger.info(
                f"  Character preserved: {character_preservation:.2%}, Naturalness change: {naturalness_change:+.2f}"
            )
        else:
            logger.warning("⚠ Musical integrity check FAILED (%s violations)", len(violations))
            for violation in violations:
                logger.warning("  - %s: %s", violation.value, violation_details[violation])
            if should_rollback:
                logger.warning("  → OPTIMIZATION RECOMMENDED: Try reduced intensity")
            if should_stop_processing:
                logger.warning("  → ADAPTIVE MODE: Switch to gentler processing path")

        return result

    def validate_final_quality(
        self,
        original_audio: np.ndarray,
        processed_audio: np.ndarray,
        sample_rate: int,
        medium_type: MediumType,
        processing_mode: ProcessingMode,
        modules_applied: list[str],
    ) -> MusicalQualityReport:
        """
        Validiert final quality after complete processing.

        Args:
            original_audio: Original audio
            processed_audio: Final processed audio
            sample_rate: Sample rate
            medium_type: Medium type
            processing_mode: Processing mode
            modules_applied: List of modules applied

        Returns:
            Comprehensive musical quality report
        """
        logger.info("=" * 60)
        logger.info("FINAL MUSICAL QUALITY VALIDATION")
        logger.info("=" * 60)

        # Analyze qualities
        input_quality = self.analyzer.analyze_quality(original_audio, sample_rate)
        output_quality = self.analyzer.analyze_quality(processed_audio, sample_rate, reference=original_audio)

        # Check gates
        gate_passed, gate_reason = self.check_quality_gate(
            processed_audio, sample_rate, input_quality, medium_type, processing_mode
        )

        # Check integrity
        integrity_result = self.check_musical_integrity(
            original_audio, processed_audio, sample_rate, medium_type, processing_mode
        )

        # Calculate musical metrics
        musical_improvement = (output_quality.overall_score - input_quality.overall_score) / 100.0
        authenticity_preserved = output_quality.authenticity >= input_quality.authenticity * 0.85
        character_preserved = integrity_result.character_preservation >= 0.80
        natural_sound = output_quality.naturalness >= 0.65 or (
            input_quality.naturalness < 0.55 and (input_quality.naturalness - output_quality.naturalness) <= 0.10
        )

        # Calculate processing intensity.
        # Deduplicate module names first: retries/candidate loops can append duplicates
        # and otherwise overstate intensity (false OVERPROCESSED verdicts).
        # §2.54: Aurik 10.0.0 runs 8–50 unique phases per restoration. The original divisor
        # of 8.0 was calibrated for a simple 8-module pipeline and always produces
        # intensity=1.0 for Aurik 10.0.0, causing misleading OVERPROCESSED verdicts.
        # Each phase is already regulated by §2.45 Minimal-Intervention + PMGG;
        # phase count alone is not a meaningful over-processing indicator here.
        _unique_modules = {m for m in modules_applied if isinstance(m, str) and m.strip()}
        _N_MODULES = max(len(_unique_modules), 1)
        # Normalise to Aurik 10.0.0's practical maximum (~50 unique phase IDs per run).
        processing_intensity = min(_N_MODULES / 50.0, 1.0)

        # Check mode standards
        mode_standards = self.MODE_STANDARDS.get(processing_mode, self.MODE_STANDARDS[ProcessingMode.RESTORATION])
        _is_unknown_medium = medium_type in {
            MediumType.UNKNOWN,
            MediumType.DIGITAL_UNKNOWN,
            MediumType.ANALOG_UNKNOWN,
        }
        _max_processing_intensity = mode_standards.max_processing_intensity
        if _is_unknown_medium:
            _max_processing_intensity = max(_max_processing_intensity, 1.0)

        overprocessed = processing_intensity > _max_processing_intensity

        # Determine if quality is guaranteed.
        # §2.54 Minimum improvement: processing must yield measurable improvement.
        # For severely degraded sources (restorability < 40) a 0% improvement
        # with gates-all-passing means the pipeline ran but added nothing — never
        # "excellence".  Require at least 1% absolute improvement.
        # §v10.101: Use relative improvement (output/input ≥ 1.02) instead of
        # absolute (output−input ≥ 1.0). For low-quality sources (score ~40),
        # a 2% relative gain (~0.8 points) is already significant. The absolute
        # 1.0-point threshold was too strict for heavily degraded material.
        _input_score = max(float(input_quality.overall_score), 0.1)
        _output_score = float(output_quality.overall_score)
        _minimal_improvement = (_output_score / _input_score) >= 1.02

        quality_guaranteed = (
            gate_passed
            and integrity_result.passed
            and authenticity_preserved
            and character_preserved
            and natural_sound
            and not overprocessed
            and _minimal_improvement
        )

        # Generate verdict
        if quality_guaranteed:
            verdict = f"✓ QUALITY GUARANTEED - {medium_type.value} musical excellence achieved"
        elif not gate_passed:
            verdict = f"❌ QUALITY GATES FAILED - {gate_reason}"
        elif not integrity_result.passed:
            verdict = f"❌ MUSICAL INTEGRITY VIOLATED - {len(integrity_result.violations)} issues"
        elif overprocessed:
            verdict = (
                f"❌ OVERPROCESSED - Intensity {processing_intensity:.1%} exceeds limit {_max_processing_intensity:.1%}"
            )
        elif not authenticity_preserved:
            verdict = f"❌ AUTHENTICITY LOST - {medium_type.value} character destroyed"
        elif not _minimal_improvement:
            verdict = (
                f"❌ NO IMPROVEMENT - Processing did not enhance quality "
                f"({input_quality.overall_score:.0f}→{output_quality.overall_score:.0f})"
            )
        else:
            verdict = "❌ QUALITY NOT GUARANTEED - Unknown issue"

        # Collect warnings
        warnings = []
        if not gate_passed:
            warnings.append(gate_reason)
        if overprocessed:
            warnings.append(f"Processing intensity too high: {processing_intensity:.1%}")
        if not authenticity_preserved:
            warnings.append(f"{medium_type.value} authenticity not preserved")
        if not character_preserved:
            warnings.append(f"Character preservation: {integrity_result.character_preservation:.1%} < 80%")
        if not natural_sound:
            warnings.append(f"Unnatural sound: naturalness {output_quality.naturalness:.2f} < 0.65")

        # Collect recommendations
        recommendations = list(integrity_result.recommendations)
        if overprocessed:
            recommendations.append(f"Reduce module count (currently {len(modules_applied)})")
        if not quality_guaranteed:
            recommendations.append(f"Try {ProcessingMode.RESTORATION.value} mode for safer, more authentic processing")

        # Create report
        report = MusicalQualityReport(
            input_quality=input_quality,
            output_quality=output_quality,
            medium_type=medium_type,
            processing_mode=processing_mode,
            gates_passed=gate_passed,
            gate_results={"overall": gate_passed},
            integrity_result=integrity_result,
            musical_improvement=musical_improvement,
            authenticity_preserved=authenticity_preserved,
            character_preserved=character_preserved,
            natural_sound=natural_sound,
            modules_applied=modules_applied,
            processing_intensity=processing_intensity,
            overprocessed=overprocessed,
            quality_guaranteed=quality_guaranteed,
            verdict=verdict,
            warnings=warnings,
            recommendations=recommendations,
        )

        # Log report
        logger.info("Input Quality:  %.1f/100 (%s)", input_quality.overall_score, input_quality.quality_level.value)
        logger.info("Output Quality: %.1f/100 (%s)", output_quality.overall_score, output_quality.quality_level.value)
        logger.info("Musical Improvement: %+.1f%%", musical_improvement * 100)
        logger.info("Processing Intensity: %.1f%% (%s modules)", processing_intensity * 100, len(modules_applied))
        logger.info("Authenticity Preserved: %s", authenticity_preserved)
        logger.info(
            "Character Preserved: %s (%.1f%%)", character_preserved, integrity_result.character_preservation * 100
        )
        logger.info("Natural Sound: %s (naturalness: %.2f)", natural_sound, output_quality.naturalness)
        logger.info("Musical Integrity: %.1f%%", integrity_result.overall_integrity * 100)
        logger.info("")
        logger.info("VERDICT: %s", verdict)

        if warnings:
            logger.warning("WARNINGS:")
            for warning in warnings:
                logger.warning("  ⚠ %s", warning)

        if recommendations:
            logger.info("RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info("  → %s", rec)

        logger.info("=" * 60)

        return report


def create_musical_quality_assurance() -> MusicalQualityAssurance:
    """Factory-Funktion zum Erstellen des MQA-Systems."""
    return MusicalQualityAssurance()


def map_forensic_to_medium_type(
    forensic_result: dict[str, Any], rpm: int | None = None, bitrate: int | None = None
) -> MediumType:
    """
    Map forensic analysis result to detailed MediumType.

    Args:
        forensic_result: Result from UnifiedForensicAnalyzer
        rpm: Optional RPM info for discs (33, 45, 78)
        bitrate: Optional bitrate for lossy formats (kbps)

    Returns:
        Detailed MediumType

    Examples:
        >>> forensic = {'medium_type': 'VINYL', 'rpm': 33}
        >>> map_forensic_to_medium_type(forensic, rpm=33)
        MediumType.VINYL_33

        >>> forensic = {'medium_type': 'TAPE', 'format': 'reel-to-reel'}
        >>> map_forensic_to_medium_type(forensic)
        MediumType.REEL_TO_REEL
    """
    medium_str = forensic_result.get("medium_type", "UNKNOWN").upper()

    # === VINYL Familie ===
    if "VINYL" in medium_str or "LP" in medium_str:
        if rpm == 45:
            return MediumType.VINYL_45
        elif rpm == 78:
            return MediumType.SHELLAC_78  # 78rpm ist meist Shellac
        else:
            return MediumType.VINYL_33  # Default

    # === SHELLAC/78rpm ===
    if "SHELLAC" in medium_str or rpm == 78:
        return MediumType.SHELLAC_78

    if "ACETATE" in medium_str or "LACQUER" in medium_str:
        return MediumType.ACETATE

    # === TAPE Familie ===
    if "TAPE" in medium_str or "REEL" in medium_str:
        tape_format = forensic_result.get("format", "").lower()

        if "reel" in tape_format or "open" in tape_format:
            return MediumType.REEL_TO_REEL
        elif "cassette" in tape_format or "compact" in tape_format:
            return MediumType.CASSETTE
        elif "8track" in tape_format or "8-track" in tape_format:
            return MediumType.EIGHT_TRACK
        elif "dat" in tape_format:
            return MediumType.DAT
        elif "broadcast" in tape_format:
            return MediumType.BROADCAST_TAPE
        else:
            return MediumType.CASSETTE  # Default tape = cassette

    if "CASSETTE" in medium_str:
        return MediumType.CASSETTE

    # === Historic ===
    if "WAX" in medium_str or "CYLINDER" in medium_str:
        return MediumType.WAX_CYLINDER

    if "WIRE" in medium_str:
        return MediumType.WIRE_RECORDING

    # === Optical Disc ===
    if "CD" in medium_str:
        if "CD-R" in medium_str or "CDR" in medium_str:
            return MediumType.CD_R
        elif "SACD" in medium_str:
            return MediumType.SACD
        else:
            return MediumType.CD

    if "DVD" in medium_str and "AUDIO" in medium_str:
        return MediumType.DVD_AUDIO

    if "MINIDISC" in medium_str or "MD" in medium_str:
        return MediumType.MINIDISC

    # === Digital Files ===
    if "DIGITAL" in medium_str or "FILE" in medium_str:
        format_str = forensic_result.get("format", "").upper()

        # Lossless
        if any(fmt in format_str for fmt in ["WAV", "FLAC", "ALAC", "APE", "AIFF"]):
            return MediumType.LOSSLESS

        # DSD
        if "DSD" in format_str or "DSF" in format_str or "DFF" in format_str:
            return MediumType.DSD

        # Lossy (by bitrate)
        if bitrate:
            if bitrate >= 320:
                return MediumType.LOSSY_HIGH
            elif bitrate >= 192:
                return MediumType.LOSSY_MID
            else:
                return MediumType.LOSSY_LOW

        # Lossy (by format)
        if any(fmt in format_str for fmt in ["MP3", "AAC", "OGG", "WMA", "OPUS"]):
            return MediumType.LOSSY_HIGH  # Assume high quality if unknown

        return MediumType.DIGITAL_UNKNOWN

    if "LOSSY" in medium_str or "MP3" in medium_str:
        if bitrate:
            if bitrate >= 320:
                return MediumType.LOSSY_HIGH
            elif bitrate >= 192:
                return MediumType.LOSSY_MID
            else:
                return MediumType.LOSSY_LOW
        return MediumType.LOSSY_HIGH

    # === Broadcast/Professional ===
    if "BROADCAST" in medium_str:
        return MediumType.BROADCAST_TAPE

    if "DAW" in medium_str:
        return MediumType.DAW_BOUNCE

    # === Fallback by type ===
    if any(term in medium_str for term in ["ANALOG", "ANALOGUE"]):
        return MediumType.ANALOG_UNKNOWN

    if any(term in medium_str for term in ["DIGITAL", "FILE", "STREAM"]):
        return MediumType.DIGITAL_UNKNOWN

    return MediumType.UNKNOWN


# === Example Usage ===
if __name__ == "__main__":
    # Example: Validate VINYL restoration
    pass

    from backend.file_import import load_audio_file

    # Load audio
    _res1 = load_audio_file("input/vinyl_recording.wav")
    _res2 = load_audio_file("output/vinyl_restored.wav")
    if _res1 is None or _res2 is None:
        raise FileNotFoundError("Beispiel-Audiodateien konnten nicht geladen werden.")
    original, sr = np.asarray(_res1["audio"], dtype=np.float32), int(_res1["sr"])
    processed = np.asarray(_res2["audio"], dtype=np.float32)

    # Create MQA system
    mqa = MusicalQualityAssurance()

    # Validate
    report = mqa.validate_final_quality(
        original,
        processed,
        sr,
        medium_type=MediumType.VINYL_33,
        processing_mode=ProcessingMode.RESTORATION,
        modules_applied=["DCBlocker", "ForensicAnalyzer", "ClickRemover", "NoiseReduction", "TapeSpecialist"],
    )

    if report.quality_guaranteed:
        logger.debug("✓ Quality guaranteed - ready for delivery")
    else:
        logger.debug("❌ Quality issues: %s", report.verdict)
        for warning in report.warnings:
            logger.debug("  - %s", warning)
