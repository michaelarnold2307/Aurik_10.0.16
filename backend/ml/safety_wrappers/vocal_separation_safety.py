"""
HIPS-Compliant Safety Wrapper for Vocal Separation

Ensures all vocal separation operations comply with AURIK's normative policies:

1. Kontextbewusstsein: Model must capture sufficient musical context
2. Nebenwirkungen: All artifacts must be tracked and reported
3. Reversibilität: Original can be reconstructed from stems
4. Auditierbarkeit: Full decision trail logged
5. Steuerbarkeit: User can adjust aggressiveness/quality
6. Bedeutungsagnostik: No aesthetic judgments, signal-level only

This wrapper validates HIPS compliance before/after separation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HIPSViolationError(Exception):
    """Raised when HIPS policy is violated"""


class VocalSeparationSafetyWrapper:
    """
    HIPS-compliant safety wrapper for vocal separation

    Validates:
    - Input audio quality (no clipping, adequate SNR)
    - Separation nebenwirkungen (phase, stereo, artifacts)
    - Reversibility (stems recombine to original)
    - Auditability (full logs)

    Usage:
        wrapper = VocalSeparationSafetyWrapper(separator)
        stems = wrapper.safe_separate(audio, sr)
    """

    # HIPS Thresholds
    MAX_ACCEPTABLE_ENERGY_LOSS = 0.15  # 15% max energy loss
    MAX_ACCEPTABLE_PHASE_LOSS = 0.25  # 25% max phase coherence loss
    MAX_ACCEPTABLE_STEREO_WIDTH_CHANGE = 0.30  # 30% max stereo width change
    MIN_ACCEPTABLE_SNR_DB = -40.0  # Minimum SNR for quality separation

    def __init__(self, separator, audit_log_path: Path | None = None, strict_mode: bool = False):
        """
        Initialisiert safety wrapper.

        Args:
            separator: VocalSeparator instance (MDX, Demucs, or Hybrid)
            audit_log_path: Path for audit logs (default: logs/vocal_separation_audit.jsonl)
            strict_mode: If True, raise exception on HIPS violations
        """
        self.separator = separator
        self.strict_mode = strict_mode

        # Audit log
        if audit_log_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
            log_dir = base_path / "logs" / "vocal_separation"
            log_dir.mkdir(parents=True, exist_ok=True)
            audit_log_path = log_dir / "hips_audit.jsonl"

        self.audit_log_path = audit_log_path

        logger.info(
            "VocalSeparationSafetyWrapper initialisiert: strict_Betriebsart=%s, audit_log=%s",
            strict_mode,
            audit_log_path,
        )

        self.separation_count = 0
        self.violations_count = 0

    def safe_separate(self, audio: np.ndarray, sr: int, **kwargs) -> dict[str, np.ndarray]:
        """
        HIPS-compliant vocal separation with safety checks

        Args:
            audio: Audio array
            sr: Sample rate
            **kwargs: Additional args for separator

        Returns:
            Separated stems dictionary

        Raises:
            HIPSViolationError: If strict_mode=True and HIPS violation detected
        """
        self.separation_count += 1
        separation_id = f"sep_{self.separation_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("[%s] Starting HIPS-compliant separation", separation_id)

        # STEP 1: Pre-separation validation
        pre_check_result = self._pre_separation_checks(audio, sr)

        if pre_check_result["status"] == "fail":
            self._log_violation(separation_id, pre_check_result, audio_shape=audio.shape)

            if self.strict_mode:
                raise HIPSViolationError(f"Pre-separation HIPS check failed: {pre_check_result['issues']}")
            else:
                logger.warning("[%s] Pre-separation warnings: %s", separation_id, pre_check_result["issues"])

        # STEP 2: Perform separation
        try:
            stems = self.separator.separate(audio, sr=sr, **kwargs)
        except Exception as e:
            logger.error("[%s] Separation fehlgeschlagen: %s", separation_id, e)
            self._log_failure(separation_id, str(e), audio_shape=audio.shape)
            raise

        # STEP 3: Post-separation validation
        post_check_result = self._post_separation_checks(audio, stems, sr)

        if post_check_result["status"] == "fail":
            self._log_violation(
                separation_id,
                post_check_result,
                audio_shape=audio.shape,
                stems_info={k: v.shape for k, v in stems.items()},
            )

            if self.strict_mode:
                raise HIPSViolationError(f"Post-separation HIPS check failed: {post_check_result['issues']}")
            else:
                logger.warning("[%s] Post-separation warnings: %s", separation_id, post_check_result["issues"])

        # STEP 4: Auditability - Log successful separation
        self._log_success(
            separation_id,
            pre_check_result,
            post_check_result,
            audio_shape=audio.shape,
            stems_info={k: v.shape for k, v in stems.items()},
        )

        logger.info("[%s] HIPS-compliant separation vollstaendig", separation_id)

        return stems  # type: ignore[no-any-return]

    def _pre_separation_checks(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """
        HIPS Pre-Separation Checks:
        1. No clipping
        2. Adequate SNR
        3. Valid stereo field
        4. Sufficient duration
        """
        issues = []
        checks = {}

        # Check 1: Clipping detection
        max_amplitude = np.max(np.abs(audio))
        is_clipped = max_amplitude >= 0.99
        checks["clipping"] = {"status": "pass" if not is_clipped else "fail", "max_amplitude": float(max_amplitude)}
        if is_clipped:
            issues.append(f"Input audio is clipped (max={max_amplitude:.3f})")

        # Check 2: SNR estimation
        # Estimate noise floor from quietest segments
        audio_flat = audio.flatten()
        noise_percentile = 10  # Bottom 10% assumed as noise
        noise_threshold = np.percentile(np.abs(audio_flat), noise_percentile)
        signal_power = np.mean(audio_flat**2)
        noise_power = noise_threshold**2

        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 100.0  # Practically noiseless
        # NaN/Inf-Guard
        snr_db = 0.0 if not np.isfinite(snr_db) else snr_db

        checks["snr"] = {"status": "pass" if snr_db > self.MIN_ACCEPTABLE_SNR_DB else "fail", "snr_db": float(snr_db)}
        if snr_db < self.MIN_ACCEPTABLE_SNR_DB:
            issues.append(f"Low SNR: {snr_db:.1f} dB")

        # Check 3: Stereo field validity
        if audio.shape[0] >= 2:
            _std0 = float(np.std(audio[0]))
            _std1 = float(np.std(audio[1]))
            if _std0 > 1e-8 and _std1 > 1e-8:
                _a0 = audio[0] - audio[0].mean()
                _a1 = audio[1] - audio[1].mean()
                _n0 = float(np.linalg.norm(_a0))
                _n1 = float(np.linalg.norm(_a1))
                correlation = float(np.dot(_a0, _a1) / (_n0 * _n1 + 1e-10))
            else:
                correlation = 1.0 if (_std0 < 1e-8 and _std1 < 1e-8) else 0.0
            correlation = 0.0 if not np.isfinite(correlation) else correlation
            checks["stereo"] = {"status": "pass", "lr_correlation": float(correlation)}
            if abs(correlation) > 0.999:
                issues.append("Audio is essentially mono (L/R correlation > 0.999)")
        else:
            checks["stereo"] = {"status": "pass", "note": "mono_input"}

        # Check 4: Duration
        duration_sec = audio.shape[1] / sr
        checks["duration"] = {"status": "pass" if duration_sec >= 0.1 else "fail", "duration_sec": float(duration_sec)}
        if duration_sec < 0.1:
            issues.append(f"Audio too short: {duration_sec:.2f}s")

        # Overall status
        status = "pass" if len(issues) == 0 else "fail"

        return {"stage": "pre_separation", "status": status, "checks": checks, "issues": issues}

    def _post_separation_checks(self, original: np.ndarray, stems: dict[str, np.ndarray], sr: int) -> dict:
        """
        HIPS Post-Separation Checks:
        1. Reversibility (stems recombine to original)
        2. Energy conservation
        3. Phase coherence preservation
        4. Stereo width preservation
        """
        issues = []
        checks = {}

        # Prepare stems for checks
        vocals = stems.get("vocals")
        instrumental = stems.get("instrumental")

        if vocals is None or instrumental is None:
            return {
                "stage": "post_separation",
                "status": "fail",
                "checks": {},
                "issues": ["Missing required stems (vocals or instrumental)"],
            }

        # Ensure same length
        min_len = min(original.shape[1], vocals.shape[1], instrumental.shape[1])
        original = original[:, :min_len]
        vocals = vocals[:, :min_len]
        instrumental = instrumental[:, :min_len]

        # Check 1: Reversibility (recombination error)
        recombined = vocals + instrumental
        reconstruction_error = np.mean((original - recombined) ** 2)
        reconstruction_error_db = 10 * np.log10(reconstruction_error + 1e-10)

        checks["reversibility"] = {
            "status": "pass" if reconstruction_error < 0.01 else "warn",
            "reconstruction_error_db": float(reconstruction_error_db),
        }
        if reconstruction_error > 0.01:
            issues.append(f"Reconstruction error: {reconstruction_error_db:.1f} dB (stems don't perfectly recombine)")

        # Check 2: Energy conservation
        energy_original = np.sum(original**2)
        energy_recombined = np.sum(recombined**2)
        energy_loss = abs(1.0 - energy_recombined / (energy_original + 1e-10))
        # NaN/Inf-Guard
        energy_loss = 0.0 if not np.isfinite(energy_loss) else energy_loss

        checks["energy_conservation"] = {
            "status": "pass" if energy_loss < self.MAX_ACCEPTABLE_ENERGY_LOSS else "fail",
            "energy_loss_ratio": float(energy_loss),
        }
        if energy_loss > self.MAX_ACCEPTABLE_ENERGY_LOSS:
            issues.append(f"Energy loss: {energy_loss * 100:.1f}%")

        # Check 3: Phase coherence
        def phase_coherence(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape[0] < 2 or b.shape[0] < 2:
                return 1.0
            xcorr = np.correlate(a[0], b[0], mode="valid")
            norm = np.linalg.norm(a[0]) * np.linalg.norm(b[0]) + 1e-10
            return float(np.max(np.abs(xcorr)) / norm)

        phase_original = phase_coherence(original, original)
        phase_recombined = phase_coherence(original, recombined)
        phase_loss = abs(phase_original - phase_recombined)
        # NaN/Inf-Guard
        phase_loss = 0.0 if not np.isfinite(phase_loss) else phase_loss

        checks["phase_coherence"] = {
            "status": "pass" if phase_loss < self.MAX_ACCEPTABLE_PHASE_LOSS else "fail",
            "phase_loss": float(phase_loss),
        }
        if phase_loss > self.MAX_ACCEPTABLE_PHASE_LOSS:
            issues.append(f"Phase coherence loss: {phase_loss * 100:.1f}%")

        # Check 4: Stereo width preservation
        def stereo_width(audio: np.ndarray) -> float:
            if audio.shape[0] < 2:
                return 0.0
            _s0 = float(np.std(audio[0]))
            _s1 = float(np.std(audio[1]))
            if _s0 < 1e-8 and _s1 < 1e-8:
                corr = 1.0  # Both constant — trivially correlated
            elif _s0 < 1e-8 or _s1 < 1e-8:
                corr = 0.0
            else:
                _a0 = audio[0] - audio[0].mean()
                _a1 = audio[1] - audio[1].mean()
                _n0 = float(np.linalg.norm(_a0))
                _n1 = float(np.linalg.norm(_a1))
                corr = float(np.dot(_a0, _a1) / (_n0 * _n1 + 1e-10))
                if not np.isfinite(corr):
                    corr = 0.0
            return float(1.0 - abs(corr))

        width_original = stereo_width(original)
        width_vocals = stereo_width(vocals)
        width_change = abs(width_original - width_vocals)

        checks["stereo_width"] = {
            "status": "pass" if width_change < self.MAX_ACCEPTABLE_STEREO_WIDTH_CHANGE else "fail",
            "original_width": float(width_original),
            "vocals_width": float(width_vocals),
            "width_change": float(width_change),
        }
        if width_change > self.MAX_ACCEPTABLE_STEREO_WIDTH_CHANGE:
            issues.append(f"Stereo width change: {width_change * 100:.1f}%")

        # Overall status
        status = "pass" if len(issues) == 0 else "fail"

        return {"stage": "post_separation", "status": status, "checks": checks, "issues": issues}

    def _log_success(self, separation_id: str, pre_check: dict, post_check: dict, audio_shape: tuple, stems_info: dict):
        """Protokolliert successful HIPS-compliant separation."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "separation_id": separation_id,
            "status": "success",
            "audio_shape": audio_shape,
            "stems_info": stems_info,
            "pre_checks": pre_check,
            "post_checks": post_check,
        }

        self._write_audit_log(log_entry)

    def _log_violation(
        self, separation_id: str, check_result: dict, audio_shape: tuple, stems_info: dict | None = None
    ):
        """Protokolliert HIPS violation."""
        self.violations_count += 1

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "separation_id": separation_id,
            "status": "hips_violation",
            "audio_shape": audio_shape,
            "stems_info": stems_info,
            "check_result": check_result,
        }

        self._write_audit_log(log_entry)

    def _log_failure(self, separation_id: str, error: str, audio_shape: tuple) -> None:
        """Protokolliert separation failure."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "separation_id": separation_id,
            "status": "failure",
            "audio_shape": audio_shape,
            "error": error,
        }

        self._write_audit_log(log_entry)

    def _write_audit_log(self, entry: dict) -> None:
        """Schreibt audit log entry (JSONL format)."""
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("konnte nicht write audit log: %s", e)

    def get_compliance_report(self) -> dict[str, Any]:
        """
        Gibt zurück: HIPS compliance report.

        Returns summary of all separations and violations.
        """
        if self.separation_count == 0:
            return {"total_separations": 0, "violations": 0, "compliance_rate": 1.0}

        compliance_rate = 1.0 - (self.violations_count / self.separation_count)

        return {
            "total_separations": self.separation_count,
            "violations": self.violations_count,
            "compliance_rate": compliance_rate,
            "audit_log_path": str(self.audit_log_path),
            "strict_mode": self.strict_mode,
        }


if __name__ == "__main__":
    # Test safety wrapper
    from backend.ml.inference_only.vocal_separation.hybrid_separation import HybridVocalSeparator

    separator = HybridVocalSeparator()
    wrapper = VocalSeparationSafetyWrapper(separator, strict_mode=False)

    # Generate test signal
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    mixed = np.sin(2 * np.pi * 440 * t) + np.random.randn(len(t)) * 0.1
    audio = np.stack([mixed, mixed])

    # Separate with safety checks
    stems = wrapper.safe_separate(audio, sr=sr)

    logger.info("✓ Safety wrapper test passed")
    logger.info("  Compliance report: %s", wrapper.get_compliance_report())
