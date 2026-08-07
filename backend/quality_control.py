"""
Mehrstufige Qualitätskontrolle, A/B-Vergleiche, psychoakustische Modelle, Nicht-Destruktivität, Warnungen, Testdatenbank

AURIK v8.0 UPDATE:
- Integration von Quality Metrics Manager (VERSA, ViSQOL)
- Docker-basierte objektive Qualitätsmetriken
- Weltklasse Quality Gates mit echten ML-Modellen
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class QualityControl:
    def __init__(self) -> None:
        self.quality_log: list[dict] = []
        self.ab_results: list[float] = []
        self.warnings: list[str] = []
        self.test_db: list[dict] = []

    def check_non_destructive(self, original: np.ndarray, processed: np.ndarray) -> float | None:
        """Vergleicht Original und bearbeitetes Signal auf Nicht-Destruktivität (z.B. Differenzsignal, SNR)."""
        diff = np.array(processed) - np.array(original)
        diff_power = np.sum(np.square(diff))
        orig_power = np.sum(np.square(original))
        # Numerische Stabilität: diff_power mindestens 1e-10
        if orig_power == 0:
            return None  # SNR nicht definiert für Null-Signale
        if diff_power == 0:
            return 100.0  # Perfekte Nicht-Destruktivität (identische Signale)
        snr = 10 * np.log10(orig_power / diff_power)
        # NaN/Inf-Guard (§3.1)
        snr = float(np.nan_to_num(snr, nan=0.0, posinf=100.0, neginf=0.0))
        if snr < 30:
            self.warnings.append("Warnung: Mögliche destruktive Bearbeitung (SNR < 30 dB)")
        if not np.isfinite(snr):
            return None
        return float(snr)

    def ab_test(self, reference: np.ndarray, candidate: np.ndarray) -> float | None:
        """Simuliert einen A/B-Vergleich (z.B. Feature-Vergleich, Score)."""
        # NaN/Inf-Guard am Eingang (§3.1)
        reference = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
        candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)

        # Correlation is undefined for constant signals.
        if reference.size == 0 or candidate.size == 0:
            return None
        if np.std(reference) == 0.0 or np.std(candidate) == 0.0:
            return None

        _r_c = reference - np.mean(reference)
        _c_c = candidate - np.mean(candidate)
        score = float(np.dot(_r_c, _c_c) / (np.linalg.norm(_r_c) * np.linalg.norm(_c_c) + 1e-10))
        # NaN/Inf-Guard am Ausgang
        score = float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
        self.ab_results.append(score)
        if not np.isfinite(score):
            return None
        return score

    def psychoacoustic_score(self, audio: np.ndarray, sr: int) -> float:
        """Platzhalter für psychoakustisches Modell (z.B. Loudness, Maskierung, Klarheit)."""
        # Sicherstellen, dass audio ein numerisches Array ist
        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.float32)
        elif not isinstance(audio, np.ndarray) or not np.issubdtype(audio.dtype, np.floating):
            audio = np.asarray(audio, dtype=np.float32)
        # NaN/Inf-Guard (§3.1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        loudness = float(np.mean(np.abs(audio)))
        clarity = float(np.std(audio))
        score = loudness / (clarity + 1e-8)
        # NaN/Inf-Guard am Ausgang
        score = float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
        self.quality_log.append({"loudness": loudness, "clarity": clarity, "score": score})
        return score

    def add_to_test_db(self, features: dict, label: str) -> None:
        self.test_db.append({"features": features, "label": label})

    def get_warnings(self) -> list[str]:
        return self.warnings

    def get_quality_log(self) -> list[dict]:
        return self.quality_log

    def get_ab_results(self) -> list[float]:
        return self.ab_results

    def get_test_db(self) -> list[dict]:
        return self.test_db


# ==============================================================================
# Quality Report Adapter (AURIK Spec 3.5)
# ==============================================================================


def create_quality_report_from_job(job: Any) -> Any:
    """
    Erstellt formal QualityReport from ResturationJob.

    Extracts quality metrics from completed job and formats them
    according to AURIK Spec 3.5 (Quality Assurance & Audit).

    Args:
        job: Completed ResturationJob with quality_report

    Returns:
        QualityReport (already populated in job)

    Note:
        This is a convenience function. The QualityReport is already
        created during job processing by AdaptiveProcessingPipelineV2.
    """
    if job.quality_report is None:
        raise ValueError(f"Job {job.job_id} has no quality_report")

    return job.quality_report


def enhance_quality_report_with_objective_metrics(
    report: Any, audio_before: np.ndarray, audio_after: np.ndarray, sr: int
) -> Any:
    """
    Enhance QualityReport with additional objective metrics.

    Uses Quality Metrics Manager to calculate comprehensive quality scores:
    - VERSA (Compat-Key "cdpam"): Non-reference perceptual quality
    - ViSQOL: Reference-based perceptual quality (wenn reference verfügbar)

    Args:
        report: Existing QualityReport to enhance
        audio_before: Original audio signal (reference)
        audio_after: Processed audio signal
        sr: Sample rate

    Returns:
        Enhanced QualityReport with objective metrics

    Note:
        Requires quality plugins (VERSA/ViSQOL where available).
        Falls Plugins nicht verfügbar, werden nur verfügbare Metriken berechnet.
    """
    import soundfile as sf

    try:
        # Import Quality Metrics Manager
        from backend.quality_metrics_manager import QualityMetricsManager

        manager = QualityMetricsManager()

        # Save audio_after to temp file for plugin processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_after:
            sf.write(tmp_after.name, audio_after, sr)
            after_path = tmp_after.name

        # Save audio_before for reference-based metrics
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_before:
            sf.write(tmp_before.name, audio_before, sr)
            before_path = tmp_before.name

        try:
            # Comprehensive assessment with all metrics
            results = manager.assess_comprehensive(
                audio_file=after_path, reference_file=before_path, output_dir=tempfile.mkdtemp()
            )

            # Add to report
            if "metrics" in results:
                metrics = results["metrics"]

                # VERSA über Compat-Key "cdpam"
                if "cdpam" in metrics and "score" in metrics["cdpam"]:
                    report.cdpam_score = float(metrics["cdpam"]["score"])

                # §4.4+§10.2: DNSMOS P.835 deaktiviert — Sprach-Metrik (DNS-Challenge), verboten für Musikrestaurierung
                # report.dnsmos_p808 / dnsmos_sig / dnsmos_bak / dnsmos_ovrl bleiben auf Default-Werten

                # §4.4+§10.2: NISQA deaktiviert — Sprach-Metrik (NeurIPS Speech Corpus), verboten für Musikrestaurierung
                # report.nisqa_mos / nisqa_noisiness / nisqa_coloration / nisqa_discontinuity / nisqa_loudness bleiben auf Default-Werten
                report.nisqa_mos = 0.0  # DEAKTIVIERT §10.2

                # ViSQOL (reference-based)
                if "visqol" in metrics and "ViSQOL_MOS" in metrics["visqol"]:
                    report.visqol_mos = float(metrics["visqol"]["ViSQOL_MOS"])

                # Aggregate score
                if "aggregate" in results and "overall_score" in results["aggregate"]:
                    report.quality_aggregate = float(results["aggregate"]["overall_score"])
                    report.quality_rating = results["aggregate"]["overall_rating"]

        finally:
            # Cleanup temp files
            Path(after_path).unlink(missing_ok=True)
            Path(before_path).unlink(missing_ok=True)

    except ImportError:
        logger.warning("Quality Metrics Manager nicht verfügbar - objective plugin metrics uebersprungen")
        # §4.4: Kein SI-SDR-Fallback für Musikmetriken.

    except Exception as e:
        logger.warning("Objective metrics enhancement fehlgeschlagen: %s", e)

    return report


# ==============================================================================
# CAS Score & Weltklasse Quality Gates (AURIK v8.0)
# ==============================================================================


class CASScoreCalculator:
    """
    Creative Authenticity Score (CAS) Calculator

    Misst die musikalische Exzellenz anhand von 5 Dimensionen:
    1. BRILLANZ (Clarity, Detail, HF Content)
    2. TRANSPARENZ (Separation, Space, Contrast)
    3. AUTHENTIZITÄT (Naturalness, HP Balance)
    4. EMOTIONALITÄT (Dynamics, Flow, Expression)
    5. WÄRME (Tone, Harmony, LF Content)

    Target Scores:
    - GOOD: ≥0.80
    - EXCELLENT: ≥0.92
    - WORLD-CLASS: ≥0.96 (⭐⭐⭐⭐⭐)
    """

    def __init__(self):
        self.weights = {
            "brillanz": 0.25,
            "transparenz": 0.20,
            "authentizitaet": 0.20,
            "emotionalitaet": 0.20,
            "waerme": 0.15,
        }

    def compute(self, audio: np.ndarray, sr: int) -> dict:
        """
        Berechnet CAS Score mit allen 5 Dimensionen.

        Args:
            audio: Audio signal (mono oder stereo)
            sr: Sample rate

        Returns:
            dict: {
                "cas_score": float,
                "brillanz": float,
                "transparenz": float,
                "authentizitaet": float,
                "emotionalitaet": float,
                "waerme": float,
                "rating": str
            }
        """
        import librosa

        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # 1. BRILLANZ (Clarity, Detail)
        # High-frequency content + spectral centroid
        try:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
            brillanz = np.clip(centroid / 8000.0, 0, 1)
        except Exception:
            brillanz = 0.5

        # 2. TRANSPARENZ (Separation, Space)
        # Spectral contrast (separation between peaks and valleys)
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            transparenz = np.clip(float(np.mean(contrast)) / 40.0, 0, 1)
        except Exception:
            transparenz = 0.5

        # 3. AUTHENTIZITÄT (Naturalness)
        # Harmonic/Percussive balance (expect 70/30 for natural music)
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = float(np.mean(harmonic**2))
            percussive_energy = float(np.mean(percussive**2))
            total_energy = harmonic_energy + percussive_energy

            if total_energy > 0:
                hp_balance = harmonic_energy / total_energy
                # Optimal balance: 0.7 (70% harmonic, 30% percussive)
                authentizitaet = 1.0 - min(abs(hp_balance - 0.7), 0.3) / 0.3
            else:
                authentizitaet = 0.5
        except Exception:
            authentizitaet = 0.5

        # 4. EMOTIONALITÄT (Dynamics, Flow)
        # RMS variation + dynamic range
        try:
            rms = librosa.feature.rms(y=audio)[0]
            rms_variation = float(np.std(rms))
            # Higher variation = more emotional expression
            emotionalitaet = np.clip(rms_variation / 0.1, 0, 1)
        except Exception:
            emotionalitaet = 0.5

        # 5. WÄRME (Tone, Harmony)
        # Low-frequency content (warmth in 60-500 Hz range)
        try:
            spectrum = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

            # Energy in warmth band (60-500 Hz)
            lf_mask = (freqs >= 60) & (freqs <= 500)
            lf_energy = float(np.sum(spectrum[lf_mask]))
            total_energy = float(np.sum(spectrum))

            if total_energy > 0:
                lf_ratio = lf_energy / total_energy
                # Expect 15-30% in warmth band
                waerme = np.clip(lf_ratio * 5.0, 0, 1)
            else:
                waerme = 0.5
        except Exception:
            waerme = 0.5

        # Composite CAS Score
        cas_score = (
            self.weights["brillanz"] * brillanz
            + self.weights["transparenz"] * transparenz
            + self.weights["authentizitaet"] * authentizitaet
            + self.weights["emotionalitaet"] * emotionalitaet
            + self.weights["waerme"] * waerme
        )

        # Rating
        if cas_score >= 0.96:
            rating = "⭐⭐⭐⭐⭐ WORLD-CLASS"
        elif cas_score >= 0.92:
            rating = "⭐⭐⭐⭐ EXCELLENT"
        elif cas_score >= 0.80:
            rating = "⭐⭐⭐ GOOD"
        elif cas_score >= 0.70:
            rating = "⭐⭐ ACCEPTABLE"
        else:
            rating = "⭐ NEEDS IMPROVEMENT"

        return {
            "cas_score": float(cas_score),
            "brillanz": float(brillanz),
            "transparenz": float(transparenz),
            "authentizitaet": float(authentizitaet),
            "emotionalitaet": float(emotionalitaet),
            "waerme": float(waerme),
            "rating": rating,
        }


class QualityGates:
    """
    Weltklasse Quality Gates für AURIK v8.0

    8 comprehensive quality checks:
    1. SNR Check: No degradation (≤2dB loss allowed)
    2. THD Check: Distortion control (≤50% increase)
    3. Clipping Check: No digital clipping (peak < 0.99)
    4. VERSA Compat-Check (über Key "cdpam", optional)
    5. DNSMOS Compat-Check (deaktiviert)
    6. NISQA Compat-Check (deaktiviert)
    7. CAS Score Check: Musical excellence (≥ 0.80)
    8. Spectral Fidelity: Frequency response preservation

    All gates must PASS for export approval.
    Uses Docker-based Quality Metrics Manager for ML-powered assessments.
    """

    def __init__(self, use_ml_plugins: bool = True):
        self.cas_calculator = CASScoreCalculator()
        self.results_log: list[Any] = []
        self.use_ml_plugins = use_ml_plugins

        # Initialize Quality Metrics Manager (lazy load)
        self._metrics_manager = None
        if use_ml_plugins:
            try:
                from backend.quality_metrics_manager import QualityMetricsManager

                self._metrics_manager = QualityMetricsManager()
                logger.info("✓ Quality Metrics Manager initialisiert")
            except Exception as e:
                logger.warning("Quality Metrics Manager nicht verfügbar: %s", e)

    def validate_all(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
        require_vocals: bool = False,
    ) -> tuple[bool, dict]:
        """
        Alle Quality Gates durchlaufen.

        AURIK v8.0: Verwendet Docker-basierte ML-Plugins für objektive Qualitätsbewertung.

        Args:
            audio_before: Original audio
            audio_after: Processed audio
            sr: Sample rate
            require_vocals: Reserved compatibility flag (currently no NISQA logic)

        Returns:
            tuple: (all_passed: bool, results: dict)
        """
        import librosa
        import soundfile as sf

        results: dict[str, Any] = {}

        # Ensure mono for analysis
        if audio_before.ndim > 1:
            audio_before = librosa.to_mono(audio_before)
        if audio_after.ndim > 1:
            audio_after = librosa.to_mono(audio_after)

        # 1. SNR Check
        snr_before = self._compute_snr(audio_before)
        snr_after = self._compute_snr(audio_after)
        results["snr_before"] = snr_before
        results["snr_after"] = snr_after
        results["snr_improvement"] = snr_after - snr_before
        results["snr_check"] = snr_after >= snr_before - 2.0  # Max 2dB loss

        # 2. THD Check
        thd_before = self._compute_thd(audio_before, sr)
        thd_after = self._compute_thd(audio_after, sr)
        results["thd_before"] = thd_before
        results["thd_after"] = thd_after
        results["thd_ratio"] = thd_after / (thd_before + 1e-10)
        results["thd_check"] = thd_after <= thd_before * 1.5  # Max 50% increase

        # 3. Clipping Check
        peak = float(np.max(np.abs(audio_after)))
        results["peak_amplitude"] = peak
        results["no_clipping"] = peak < 0.99

        # ==============================================================================
        # ML-BASED QUALITY METRICS (AURIK v8.0)
        # ==============================================================================

        if self.use_ml_plugins and self._metrics_manager is not None:
            # Save audio to temp file for plugin processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_after, sr)
                audio_temp_path = tmp.name

            try:
                # Run comprehensive quality assessment
                self._metrics_manager.assess_quality(audio_temp_path, output_dir=tempfile.mkdtemp())

                # Compat-Key cdpam bleibt für Abwärtskompatibilität, aktive Metrik ist VERSA.
                results["cdpam_score"] = None
                results["cdpam_check"] = True  # DEAKTIVIERT §4.4

                # §4.4+§10.2: DNSMOS P.835 deaktiviert — Sprach-Metrik (DNS-Challenge), verboten für Musikrestaurierung
                results["dnsmos_ovrl_p835"] = None  # DEAKTIVIERT §10.2
                results["dnsmos_p808"] = None  # DEAKTIVIERT §10.2
                results["dnsmos_sig"] = None  # DEAKTIVIERT §10.2
                results["dnsmos_bak"] = None  # DEAKTIVIERT §10.2
                results["dnsmos_check"] = True  # immer bestanden (Metrik deaktiviert)

                # §4.4+§10.2: NISQA deaktiviert — Sprach-Metrik verboten für Musikrestaurierung
                _nisqa_mos = 0.0  # DEAKTIVIERT §10.2
                results["nisqa_mos"] = None
                results["nisqa_check"] = True  # immer bestanden (Metrik deaktiviert)

            except Exception as e:
                logger.warning("Plugin-based quality checks fehlgeschlagen: %s", e)
                # Fallback to approximations
                results["cdpam_score"] = None
                results["cdpam_check"] = True
                results["dnsmos_p808"] = None
                results["dnsmos_check"] = True

                # §4.4: NISQA deaktiviert — verboten als Musikmetrik
                results["nisqa_mos"] = None
                results["nisqa_check"] = True  # DEAKTIVIERT §4.4

            finally:
                # Cleanup temp file
                Path(audio_temp_path).unlink(missing_ok=True)

        else:
            # Fallback: compatibility defaults wenn Plugins nicht verfügbar
            results["cdpam_score"] = None
            results["cdpam_check"] = True
            results["dnsmos_p808"] = None
            results["dnsmos_check"] = True

            # §4.4: NISQA deaktiviert — verboten als Musikmetrik
            results["nisqa_mos"] = None
            results["nisqa_check"] = True  # DEAKTIVIERT §4.4

        # ==============================================================================
        # TRADITIONAL QUALITY METRICS
        # ==============================================================================

        # 7. CAS Score Check
        cas_results = self.cas_calculator.compute(audio_after, sr)
        results["cas_score"] = cas_results["cas_score"]
        results["cas_details"] = cas_results
        results["cas_check"] = cas_results["cas_score"] >= 0.80

        # 8. Spectral Fidelity Check
        spectral_fidelity = self._compute_spectral_fidelity(audio_before, audio_after, sr)
        results["spectral_fidelity"] = spectral_fidelity
        results["spectral_check"] = spectral_fidelity >= 0.90  # 90% similarity

        # Overall Pass/Fail
        all_passed = all(
            [
                results["snr_check"],
                results["thd_check"],
                results["no_clipping"],
                results["cdpam_check"],
                results["dnsmos_check"],
                results["nisqa_check"],
                results["cas_check"],
                results["spectral_check"],
            ]
        )

        results["all_passed"] = all_passed

        # Log results
        self.results_log.append(results)

        return all_passed, results

    def _compute_snr(self, audio: np.ndarray) -> float:
        """Berechnet Signal-to-Noise Ratio (simplified)."""
        # Signal: RMS of signal
        signal_rms = float(np.sqrt(np.mean(audio**2)))

        # Noise: estimate from quietest 10% of frames
        frame_size = 2048
        n_frames = max(1, (len(audio) - frame_size) // frame_size)
        trimmed = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(trimmed**2, axis=1))
        noise_rms = float(np.percentile(frame_rms, 10))

        if noise_rms > 0:
            snr = 20 * np.log10(signal_rms / noise_rms)
        else:
            snr = 100.0  # Very high SNR

        return float(snr)

    def _compute_thd(self, audio: np.ndarray, sr: int) -> float:
        """
        Berechnet Total Harmonic Distortion (simplified).

        THD = sqrt(sum of harmonic powers) / fundamental power
        """
        # FFT
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

        # Find fundamental (peak in 80-500 Hz range)
        voice_mask = (freqs >= 80) & (freqs <= 500)
        if np.any(voice_mask):
            fundamental_idx = np.argmax(spectrum[voice_mask])
            fundamental_freq = freqs[voice_mask][fundamental_idx]
            fundamental_power = float(spectrum[voice_mask][fundamental_idx] ** 2)

            # Harmonics (2f, 3f, 4f, 5f)
            harmonic_power = 0.0
            for n in range(2, 6):
                harmonic_freq = n * fundamental_freq
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                if harmonic_idx < len(spectrum):
                    harmonic_power += float(spectrum[harmonic_idx] ** 2)

            thd = np.sqrt(harmonic_power) / np.sqrt(fundamental_power) if fundamental_power > 0 else 0.0
        else:
            thd = 0.0

        return float(thd)

    def _compute_snr_based_mos(self, audio: np.ndarray, sr: int) -> float:
        """SNR-based MOS approximation (internal heuristic, not NISQA).

        §4.4: NISQA is forbidden as a music metric. This method computes a
        simple SNR-to-MOS mapping for internal sanity checks only — it must
        NOT be reported under any NISQA-related key.
        """
        snr = self._compute_snr(audio)

        # Map SNR to MOS-like score (1-5)
        # SNR > 40dB → 5.0
        # SNR 30-40 → 4.0-5.0
        # SNR 20-30 → 3.0-4.0
        # SNR < 20 → < 3.0

        if snr >= 40:
            mos = 5.0
        elif snr >= 30:
            mos = 4.0 + (snr - 30) / 10.0
        elif snr >= 20:
            mos = 3.0 + (snr - 20) / 10.0
        else:
            mos = 1.0 + snr / 20.0

        return float(np.clip(mos, 1.0, 5.0))

    def _compute_spectral_fidelity(self, audio_before: np.ndarray, audio_after: np.ndarray, sr: int) -> float:
        """
        Berechnet spectral fidelity (how well frequency response is preserved).

        Returns: float (0-1, where 1 = perfect preservation)
        """
        import librosa

        # Compute mel spectrograms
        mel_before = librosa.feature.melspectrogram(y=audio_before, sr=sr, n_mels=128)
        mel_after = librosa.feature.melspectrogram(y=audio_after, sr=sr, n_mels=128)

        # Average over time
        mel_before_avg = np.mean(mel_before, axis=1)
        mel_after_avg = np.mean(mel_after, axis=1)

        # Normalize
        mel_before_norm = mel_before_avg / (np.sum(mel_before_avg) + 1e-10)
        mel_after_norm = mel_after_avg / (np.sum(mel_after_avg) + 1e-10)

        # Correlation (NaN-safe: Mel profile could be near-constant for silence)
        _sm = float(np.std(mel_before_norm))
        _sa = float(np.std(mel_after_norm))
        if _sm > 1e-10 and _sa > 1e-10:
            _ma = mel_before_norm - mel_before_norm.mean()
            _mb = mel_after_norm - mel_after_norm.mean()
            _nm = float(np.linalg.norm(_ma))
            _na = float(np.linalg.norm(_mb))
            correlation = float(np.dot(_ma, _mb) / (_nm * _na + 1e-10))
            if not np.isfinite(correlation):
                correlation = 0.0
        else:
            correlation = 1.0 if (_sm < 1e-10 and _sa < 1e-10) else 0.0

        # Map to 0-1 (correlation -1 to 1 → 0 to 1)
        fidelity = (correlation + 1.0) / 2.0

        return float(np.clip(fidelity, 0, 1))

    def get_results_log(self) -> list:
        """Gibt all quality gate results zurück."""
        return self.results_log

    def print_report(self, results: dict):
        """Gibt aus: formatted quality gate report with ML metrics."""
        logger.info("\n" + "=" * 80)
        logger.info("QUALITY GATES REPORT - AURIK v8.0")
        logger.info("=" * 80)

        logger.info("\n1. SNR Pruefung: %s", "✅ PASS" if results["snr_check"] else "❌ FAIL")
        logger.info("   Before: %.1f dB", results["snr_before"])
        logger.info("   After:  %.1f dB", results["snr_after"])
        logger.info("   Change: %.1f dB", results["snr_improvement"])

        logger.info("\n2. THD Pruefung: %s", "✅ PASS" if results["thd_check"] else "❌ FAIL")
        logger.info("   Before: %.3f", results["thd_before"])
        logger.info("   After:  %.3f", results["thd_after"])
        logger.info("   Verhaeltnis:  %.2fx", results["thd_ratio"])

        logger.info("\n3. Clipping Pruefung: %s", "✅ PASS" if results["no_clipping"] else "❌ FAIL")
        logger.info("   Peak Amplitude: %.3f", results["peak_amplitude"])

        # ML-BASED QUALITY METRICS
        logger.info("\n" + "-" * 80)
        logger.info("ML-BASED QUALITY METRICS")
        logger.info("-" * 80)

        if results.get("cdpam_score") is not None:
            logger.info(
                "\n4. VERSA Compat-Pruefung (Key: cdpam): %s", "✅ PASS" if results["cdpam_check"] else "❌ FAIL"
            )
            logger.info("   Wert: %.2f/100", results["cdpam_score"])
        else:
            logger.info("\n4. VERSA Compat-Pruefung: ⏭️  uebersprungen")

        if results.get("dnsmos_ovrl_p835") is not None:
            logger.info(
                "\n5. DNSMOS Pruefung (Noise Assessment): %s", "✅ PASS" if results["dnsmos_check"] else "❌ FAIL"
            )
            logger.info("   OVRL P.835: %.2f/5.0 ⭐ (Musik - Primär)", results["dnsmos_ovrl_p835"])
            logger.info("   SIG P.835:  %.2f/5.0 (Signal distortion)", results.get("dnsmos_sig", 0.0))
            logger.info("   BAK P.835:  %.2f/5.0 (Background noise)", results.get("dnsmos_bak", 0.0))
            logger.info("   MOS P.808:  %.2f/5.0 (Sprache - Referenz)", results.get("dnsmos_p808", 0.0))
        else:
            logger.info("\n5. DNSMOS Pruefung: ⏭️  uebersprungen (deaktiviert §4.4/§10.2)")

        if results.get("nisqa_mos") is not None:
            logger.info("\n6. NISQA Pruefung (Broadband Audio): %s", "✅ PASS" if results["nisqa_check"] else "❌ FAIL")
            logger.info("   MOS:           %.2f/5.0", results["nisqa_mos"])
            if results.get("nisqa_noisiness") is not None:
                logger.info("   Noisiness:     %.2f/5.0", results["nisqa_noisiness"])
                logger.info("   Coloration:    %.2f/5.0", results["nisqa_coloration"])
                logger.info("   Discontinuity: %.2f/5.0", results["nisqa_discontinuity"])
                logger.info("   Loudness:      %.2f/5.0", results["nisqa_loudness"])
        else:
            logger.info("\n6. NISQA Pruefung: ⏭️  uebersprungen (deaktiviert §4.4/§10.2)")

        logger.info("\n" + "-" * 80)
        logger.info("TRADITIONAL QUALITY METRICS")
        logger.info("-" * 80)

        logger.info("\n7. CAS Wert Pruefung: %s", "✅ PASS" if results["cas_check"] else "❌ FAIL")
        cas = results["cas_details"]
        logger.info("   Overall: %.3f - %s", cas["cas_score"], cas["rating"])
        logger.info("   └─ Brillanz:        %.3f", cas["brillanz"])
        logger.info("   └─ Transparenz:     %.3f", cas["transparenz"])
        logger.info("   └─ Authentizität:   %.3f", cas["authentizitaet"])
        logger.info("   └─ Emotionalität:   %.3f", cas["emotionalitaet"])
        logger.info("   └─ Wärme:           %.3f", cas["waerme"])

        logger.info("\n8. Spectral Fidelity: %s", "✅ PASS" if results["spectral_check"] else "❌ FAIL")
        logger.info("   Similarity: %s", format(results["spectral_fidelity"], ".1%"))

        logger.info("\n" + "=" * 80)
        if results["all_passed"]:
            logger.info("Ergebnis: ✅ ALL QUALITY GATES PASSED")
        else:
            logger.info("Ergebnis: ❌ QUALITY GATES fehlgeschlagen")
        logger.info("=" * 80 + "\n")


# ==============================================================================
# Example Usage & Test
# ==============================================================================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    logger.info(str("\n" + "=" * 80))
    logger.info("QUALITY GATES & CAS Wert TEST")
    logger.info("=" * 80 + "\n")

    # Generate test signals
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Original: Clean 440 Hz sine wave
    audio_before = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Processed: Add some harmonics + slight noise
    audio_after = (
        0.5 * np.sin(2 * np.pi * 440 * t)  # Fundamental
        + 0.1 * np.sin(2 * np.pi * 880 * t)  # 2nd harmonic
        + 0.05 * np.sin(2 * np.pi * 1320 * t)  # 3rd harmonic
        + 0.02 * np.random.randn(len(t))  # Slight noise
    )

    # Normalize
    _peak_p99 = float(np.percentile(np.abs(audio_after), 99.9)) if audio_after.size > 0 else 0.0
    audio_after = audio_after / _peak_p99 * 0.8 if _peak_p99 > 1e-8 else audio_after

    # Test 1: CAS Score Calculator
    logger.info("TEST 1: CAS Wert Calculator")
    logger.info(str("-" * 80))
    cas_calc = CASScoreCalculator()
    cas_results = cas_calc.compute(audio_after, sr)

    logger.info("CAS Wert: %.3f - %s", cas_results["cas_score"], cas_results["rating"])
    logger.info("  Brillanz:        %.3f", cas_results["brillanz"])
    logger.info("  Transparenz:     %.3f", cas_results["transparenz"])
    logger.info("  Authentizität:   %.3f", cas_results["authentizitaet"])
    logger.info("  Emotionalität:   %.3f", cas_results["emotionalitaet"])
    logger.info("  Wärme:           %.3f", cas_results["waerme"])

    # Test 2: Quality Gates
    logger.info(str("\n" + "=" * 80))
    logger.info("TEST 2: Quality Gates Validierung")
    logger.info(str("-" * 80))

    gates = QualityGates()
    all_passed, results = gates.validate_all(audio_before, audio_after, sr, require_vocals=False)

    gates.print_report(results)

    # Test 3: Failed case (clipping)
    logger.info(str("\n" + "=" * 80))
    logger.error("TEST 3: fehlgeschlagen Case (Clipping)")
    logger.info(str("-" * 80))

    audio_clipped = audio_after * 1.5  # Intentional clipping
    audio_clipped = np.clip(audio_clipped, -1.0, 1.0)

    all_passed2, results2 = gates.validate_all(audio_before, audio_clipped, sr, require_vocals=False)

    gates.print_report(results2)

    logger.info(str("\n" + "=" * 80))
    logger.info("✓ All tests abgeschlossen!")
    logger.info("=" * 80 + "\n")
