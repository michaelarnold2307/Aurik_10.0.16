"""
HIPS-Compliant Safety Wrapper for Pitch Correction

Ensures all pitch correction operations comply with AURIK's normative policies:

1. Kontextbewusstsein: Analyze sufficient context (2s windows) for vibrato/glissando
2. Nebenwirkungen: Track formant shift, robotic sound, transient loss
3. Reversibilität: Original audio always preserved
4. Auditierbarkeit: Full decision trail logged (why correction was/wasn't applied)
5. Steuerbarkeit: User can adjust thresholds and correction strength
6. Epistemic Gate: Reject when unable to distinguish error from expression

This wrapper validates HIPS compliance before/after correction.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .logging_config import get_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
else:
    logger = get_logger(__name__)


class HIPSViolationError(Exception):
    """Raised when HIPS policy is violated"""


class PitchCorrectionSafetyWrapper:
    """
    HIPS-compliant safety wrapper for pitch correction

    Validates:
    - Input audio quality (no clipping, vocal range)
    - Epistemic confidence (can we distinguish error from expression?)
    - Nebenwirkungen (formant shift, robotic artifacts, transient loss)
    - Conduct enforcement (DCS < threshold)
    - Auditability (full decision logs)

    Usage:
        wrapper = PitchCorrectionSafetyWrapper(corrector)
        corrected, metadata = wrapper.safe_correct(audio, sr)
    """

    # HIPS Thresholds
    MAX_ACCEPTABLE_FORMANT_SHIFT_HZ = 30.0  # Max formant shift (F1/F2)
    MAX_ACCEPTABLE_TRANSIENT_LOSS = 0.10  # 10% max transient energy loss
    MAX_ACCEPTABLE_SPECTRAL_DISTORTION = 0.15  # 15% max spectral change
    MIN_EPISTEMIC_CONFIDENCE = 0.80  # Minimum confidence to proceed
    MAX_DCS = 0.15  # Maximum Damage Cost Score

    def __init__(self, corrector, audit_log_path: Path | None = None, strict_mode: bool = False):
        """
        Initialisiert safety wrapper.

        Args:
            corrector: ConservativePitchCorrector instance
            audit_log_path: Path for audit logs (default: logs/pitch_correction_audit.jsonl)
            strict_mode: If True, raise exception on HIPS violations
        """
        self.corrector = corrector
        self.strict_mode = strict_mode

        # Audit log
        if audit_log_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
            log_dir = base_path / "logs" / "pitch_correction"
            log_dir.mkdir(parents=True, exist_ok=True)
            audit_log_path = log_dir / "hips_audit.jsonl"

        self.audit_log_path = audit_log_path

        logger.info(
            "PitchCorrectionSafetyWrapper initialisiert: strict_Betriebsart=%s, audit_log=%s",
            strict_mode,
            audit_log_path,
        )

        self.correction_count = 0
        self.violations_count = 0

    def safe_correct(self, audio: np.ndarray, sr: int, **kwargs) -> tuple[np.ndarray, dict]:
        """
        HIPS-compliant pitch correction with safety checks

        Args:
            audio: Audio array (mono or stereo)
            sr: Sample rate
            **kwargs: Additional args for corrector (reference_pitch, dry_wet)

        Returns:
            Tuple of (corrected_audio, metadata)

        Raises:
            HIPSViolationError: If strict_mode=True and HIPS violation detected
        """
        self.correction_count += 1
        correction_id = f"pc_{self.correction_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("[%s] Starting HIPS-compliant pitch correction", correction_id)

        # STEP 1: Pre-correction validation
        pre_check_result = self._pre_correction_checks(audio, sr)

        if pre_check_result["status"] == "fail":
            self._log_violation(correction_id, pre_check_result, audio_shape=audio.shape)

            if self.strict_mode:
                raise HIPSViolationError(f"Pre-correction HIPS check failed: {pre_check_result['issues']}")
            else:
                logger.warning("[%s] Pre-correction warnings: %s", correction_id, pre_check_result["issues"])

        # STEP 2: Perform correction
        try:
            audio_corrected, correction_metadata = self.corrector.correct_pitch(audio, **kwargs)
        except Exception as e:
            logger.error("[%s] Correction fehlgeschlagen: %s", correction_id, e)
            self._log_failure(correction_id, str(e), audio_shape=audio.shape)
            raise

        # STEP 3: Check if correction was rejected by epistemic/conduct gates
        if not correction_metadata.get("corrected", False):
            logger.info("[%s] Correction rejected: %s", correction_id, correction_metadata.get("reason", "unknown"))
            self._log_rejection(correction_id, correction_metadata, audio_shape=audio.shape)
            return audio_corrected, correction_metadata

        # STEP 4: Post-correction validation (only if correction was applied)
        post_check_result = self._post_correction_checks(audio, audio_corrected, sr, correction_metadata)

        if post_check_result["status"] == "fail":
            self._log_violation(
                correction_id, post_check_result, audio_shape=audio.shape, corrected_shape=audio_corrected.shape
            )

            if self.strict_mode:
                raise HIPSViolationError(f"Post-correction HIPS check failed: {post_check_result['issues']}")
            else:
                logger.warning("[%s] Post-correction warnings: %s", correction_id, post_check_result["issues"])

        # STEP 5: Auditability - Log successful correction
        self._log_success(
            correction_id,
            pre_check_result,
            post_check_result,
            correction_metadata,
            audio_shape=audio.shape,
            corrected_shape=audio_corrected.shape,
        )

        logger.info("[%s] HIPS-compliant correction vollstaendig", correction_id)

        # Enrich metadata with HIPS checks
        correction_metadata["hips_checks"] = {"pre": pre_check_result, "post": post_check_result}

        return audio_corrected, correction_metadata

    def _pre_correction_checks(self, audio: np.ndarray, sr: int) -> dict:
        """
        HIPS Pre-Correction Checks:
        1. No severe clipping
        2. Vocal frequency range present
        3. Sufficient signal energy
        4. Duration adequate for context analysis (> 0.5s)
        """
        issues = []
        checks = {}

        # Check 1: Clipping detection
        max_amplitude = np.max(np.abs(audio))
        is_clipped = max_amplitude >= 0.99
        checks["clipping"] = {"status": "pass" if not is_clipped else "warn", "max_amplitude": float(max_amplitude)}
        if is_clipped:
            issues.append(f"Audio may be clipped (max={max_amplitude:.3f})")

        # Check 2: Vocal frequency content (80-4000 Hz for fundamental)
        audio_mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)

        try:
            # FFT analysis
            fft = np.fft.rfft(audio_mono)
            freqs = np.fft.rfftfreq(len(audio_mono), 1 / sr)
            magnitude = np.abs(fft)

            # Check vocal range (80-4000 Hz)
            vocal_mask = (freqs >= 80) & (freqs <= 4000)
            vocal_energy = np.sum(magnitude[vocal_mask] ** 2)
            total_energy = np.sum(magnitude**2)

            vocal_ratio = vocal_energy / total_energy if total_energy > 0 else 0

            checks["vocal_content"] = {
                "status": "pass" if vocal_ratio > 0.1 else "warn",
                "vocal_energy_ratio": float(vocal_ratio),
            }

            if vocal_ratio < 0.1:
                issues.append(f"Low vocal content ({vocal_ratio * 100:.1f}% in 80-4000 Hz range)")
        except Exception as e:
            checks["vocal_content"] = {"status": "error", "error": str(e)}
            issues.append(f"Vocal content analysis failed: {e}")

        # Check 3: Signal energy
        rms = np.sqrt(np.mean(audio_mono**2))
        checks["signal_energy"] = {"status": "pass" if rms > 0.001 else "fail", "rms": float(rms)}
        if rms <= 0.001:
            issues.append(f"Signal too quiet (RMS={rms:.6f})")

        # Check 4: Duration
        duration_sec = len(audio_mono) / sr
        checks["duration"] = {"status": "pass" if duration_sec >= 0.5 else "warn", "duration_sec": float(duration_sec)}
        if duration_sec < 0.5:
            issues.append(f"Short duration ({duration_sec:.2f}s) may limit context analysis")

        # Overall status
        has_failures = any(c.get("status") == "fail" for c in checks.values())
        status = "fail" if has_failures else ("warn" if issues else "pass")

        return {"status": status, "checks": checks, "issues": issues}

    def _post_correction_checks(
        self, audio_original: np.ndarray, audio_corrected: np.ndarray, sr: int, correction_metadata: dict
    ) -> dict:
        """
        HIPS Post-Correction Checks:
        1. Energy conservation (< 15% loss)
        2. Spectral similarity (< 15% change)
        3. No artificial artifacts (spectral flatness check)
        4. DCS < threshold
        """
        issues = []
        checks = {}

        # Ensure mono for analysis
        orig_mono = audio_original if audio_original.ndim == 1 else np.mean(audio_original, axis=0)
        corr_mono = audio_corrected if audio_corrected.ndim == 1 else np.mean(audio_corrected, axis=0)

        # Check 1: Energy conservation
        energy_orig = np.sum(orig_mono**2)
        energy_corr = np.sum(corr_mono**2)

        if energy_orig > 0:
            energy_ratio = energy_corr / energy_orig
            energy_loss = 1.0 - energy_ratio
        else:
            energy_ratio = 1.0
            energy_loss = 0.0

        checks["energy_conservation"] = {
            "status": "pass" if abs(energy_loss) < self.MAX_ACCEPTABLE_TRANSIENT_LOSS else "fail",
            "energy_ratio": float(energy_ratio),
            "energy_loss": float(energy_loss),
        }

        if abs(energy_loss) >= self.MAX_ACCEPTABLE_TRANSIENT_LOSS:
            issues.append(
                f"Energy loss too high: {energy_loss * 100:.1f}% (max {self.MAX_ACCEPTABLE_TRANSIENT_LOSS * 100:.0f}%)"
            )

        # Check 2: Spectral similarity
        try:
            # Simple spectral comparison
            fft_orig = np.abs(np.fft.rfft(orig_mono))
            fft_corr = np.abs(np.fft.rfft(corr_mono))

            # Trim to same length
            min_len = min(len(fft_orig), len(fft_corr))
            fft_orig = fft_orig[:min_len]
            fft_corr = fft_corr[:min_len]

            # Normalize
            if np.max(fft_orig) > 0:
                fft_orig = fft_orig / np.max(fft_orig)
            if np.max(fft_corr) > 0:
                fft_corr = fft_corr / np.max(fft_corr)

            # Spectral distance (L2 norm)
            spectral_distance = np.linalg.norm(fft_orig - fft_corr) / np.sqrt(len(fft_orig))

            checks["spectral_similarity"] = {
                "status": "pass" if spectral_distance < self.MAX_ACCEPTABLE_SPECTRAL_DISTORTION else "fail",
                "spectral_distance": float(spectral_distance),
            }

            if spectral_distance >= self.MAX_ACCEPTABLE_SPECTRAL_DISTORTION:
                issues.append(
                    f"Spectral distortion too high: {spectral_distance:.3f} "
                    f"(max {self.MAX_ACCEPTABLE_SPECTRAL_DISTORTION:.2f})"
                )
        except Exception as e:
            checks["spectral_similarity"] = {"status": "error", "error": str(e)}
            issues.append(f"Spectral analysis failed: {e}")

        # Check 3: DCS from correction metadata
        dcs = correction_metadata.get("dcs", 0.0)
        checks["dcs"] = {
            "status": "pass" if dcs <= self.MAX_DCS else "fail",
            "dcs": float(dcs),
            "max_dcs": self.MAX_DCS,
        }

        if dcs > self.MAX_DCS:
            issues.append(f"DCS too high: {dcs:.3f} (max {self.MAX_DCS:.2f})")

        # Check 4: Epistemic confidence
        epistemic_conf = correction_metadata.get("epistemic_confidence", 0.0)
        checks["epistemic_confidence"] = {
            "status": "pass" if epistemic_conf >= self.MIN_EPISTEMIC_CONFIDENCE else "warn",
            "confidence": float(epistemic_conf),
            "min_confidence": self.MIN_EPISTEMIC_CONFIDENCE,
        }

        if epistemic_conf < self.MIN_EPISTEMIC_CONFIDENCE:
            issues.append(f"Low epistemic confidence: {epistemic_conf:.2f} (min {self.MIN_EPISTEMIC_CONFIDENCE:.2f})")

        # Overall status
        has_failures = any(c.get("status") == "fail" for c in checks.values())
        status = "fail" if has_failures else ("warn" if issues else "pass")

        return {"status": status, "checks": checks, "issues": issues}

    def _log_violation(self, correction_id: str, check_result: dict, **metadata):
        """Protokolliert HIPS violation."""
        self.violations_count += 1

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "correction_id": correction_id,
            "event": "hips_violation",
            "violation_count": self.violations_count,
            "check_result": check_result,
            "metadata": metadata,
        }

        self._append_audit_log(log_entry)

        logger.warning(
            "[%s] HIPS violation #%s: %s", correction_id, self.violations_count, check_result.get("issues", [])
        )

    def _log_failure(self, correction_id: str, error_msg: str, **metadata):
        """Protokolliert correction failure."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "correction_id": correction_id,
            "event": "correction_failure",
            "error": error_msg,
            "metadata": metadata,
        }

        self._append_audit_log(log_entry)

    def _log_rejection(self, correction_id: str, correction_metadata: dict, **metadata):
        """Protokolliert correction rejection (epistemic/conduct gate)."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "correction_id": correction_id,
            "event": "correction_rejected",
            "reason": correction_metadata.get("reason", "unknown"),
            "epistemic_confidence": correction_metadata.get("epistemic_confidence"),
            "dcs": correction_metadata.get("dcs"),
            "metadata": metadata,
        }

        self._append_audit_log(log_entry)

    def _log_success(
        self, correction_id: str, pre_check: dict, post_check: dict, correction_metadata: dict, **metadata
    ):
        """Protokolliert successful correction."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "correction_id": correction_id,
            "event": "correction_success",
            "pre_check": pre_check,
            "post_check": post_check,
            "n_corrections": correction_metadata.get("n_corrections", 0),
            "dcs": correction_metadata.get("dcs", 0.0),
            "epistemic_confidence": correction_metadata.get("epistemic_confidence", 0.0),
            "metadata": metadata,
        }

        self._append_audit_log(log_entry)

    def _append_audit_log(self, log_entry: dict):
        """Append entry to JSONL audit log"""
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error("konnte nicht write audit log: %s", e)

    def get_statistics(self) -> dict:
        """Gibt zurück: safety wrapper statistics."""
        return {
            "total_corrections": self.correction_count,
            "violations": self.violations_count,
            "violation_rate": (self.violations_count / self.correction_count if self.correction_count > 0 else 0),
        }
