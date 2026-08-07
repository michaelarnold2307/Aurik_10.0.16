"""
PermanentAudioMonitor – Continuous Quality Tracking für AURIK v8.0

Erfasst PRE/POST-Processing Metriken für JEDES Audio-File:
- Spectral Snapshot (24 Bark Bands)
- RMS, Crest Factor, Dynamic Range
- F0 (Fundamental Frequency) Statistiken
- HNR (Harmonic-to-Noise Ratio)

Ermöglicht:
- Per-Module Tracking (Input/Output Metrics, Confidence, Processing Time)
- Quality Gate Validation
- Audit Export (JSON/YAML/CSV)
- Continuous Learning (nach 1000+ Files)

Autor: AURIK Team
Version: 8.0
Datum: 7. Februar 2026
"""

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np
import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Models
# ==============================================================================


@dataclass
class AudioMetrics:
    """Strukturierte Audio-Metriken für Monitoring."""

    # Spectral Features
    bark_spectrum: list[float] = field(default_factory=list)  # 24 Bark bands
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0

    # Dynamic Features
    rms: float = 0.0
    crest_factor: float = 0.0
    dynamic_range_db: float = 0.0
    peak_amplitude: float = 0.0

    # Pitch Features
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    voiced_ratio: float = 0.0  # Anteil voiced frames

    # Harmonic Features
    hnr_mean: float = 0.0  # Harmonic-to-Noise Ratio
    hnr_std: float = 0.0

    # Signal Quality
    snr_estimate: float = 0.0
    clipping_ratio: float = 0.0  # Anteil geclippter Samples

    # Timestamp
    timestamp: str | None = None


@dataclass
class ModuleLog:
    """Protokolliert eines einzelnen Processing-Moduls."""

    module_name: str
    metrics_pre: AudioMetrics | None = None
    metrics_post: AudioMetrics | None = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    quality_gate_passed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Kompletter Audit-Report für ein Audio-File."""

    file_path: str
    baseline_metrics: AudioMetrics | None = None
    final_metrics: AudioMetrics | None = None
    module_logs: list[ModuleLog] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    overall_quality_passed: bool = True
    cas_improvement: float = 0.0  # Creative Authenticity Score delta
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Helper Functions: Bark Spectrum
# ==============================================================================


def hz_to_bark(f: float) -> float:
    """
    Konvertiert Frequenz (Hz) zu Bark-Skala.

    Zwicker & Terhardt (1980): Bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f / 7500)^2)
    """
    return float(13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2))


def bark_to_hz(bark: float) -> float:
    """
    Konvertiert Bark-Skala zu Frequenz (Hz).

    Approximation für Inversion.
    """
    # Simplified inverse (Traunmüller 1990)
    return 1960.0 * (bark + 0.53) / (26.28 - bark)


def compute_bark_spectrum(audio: np.ndarray, sr: int, n_bands: int = 24) -> list[float]:
    """
    Berechnet Bark-Spektrum mit n_bands (Standard: 24).

    Args:
        audio: Audio-Signal (mono)
        sr: Sample Rate
        n_bands: Anzahl Bark-Bänder (Standard 24)

    Returns:
        List[float]: Energie pro Bark-Band (dB)
    """
    # STFT berechnen
    n_fft = min(2048, max(32, 1 << int(np.floor(np.log2(max(1, len(audio)))))))
    hop_length = 512
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    power = stft**2

    # Frequenz-Bins zu Bark konvertieren
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bark_freqs = np.array([hz_to_bark(f) for f in freqs])

    # Bark-Bänder definieren (0-24 Bark ≈ 0-15kHz)
    bark_edges = np.linspace(0, 24, n_bands + 1)
    bark_spectrum = []

    for i in range(n_bands):
        # Finde FFT-Bins in diesem Bark-Band
        mask = (bark_freqs >= bark_edges[i]) & (bark_freqs < bark_edges[i + 1])
        if np.any(mask):
            band_energy = np.mean(power[mask, :])
            band_db = 10 * np.log10(band_energy + 1e-10)
            bark_spectrum.append(float(band_db))
        else:
            bark_spectrum.append(-80.0)  # Silence

    return bark_spectrum


# ==============================================================================
# Helper Functions: F0 & HNR
# ==============================================================================


def compute_f0_stats(audio: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> dict[str, float]:
    """
    Berechnet F0-Statistiken mit librosa.pyin.

    Args:
        audio: Audio-Signal (mono)
        sr: Sample Rate
        fmin: Minimale F0 (Hz) - 50 Hz für tiefe Männerstimmen
        fmax: Maximale F0 (Hz) - 500 Hz für hohe Frauenstimmen

    Returns:
        Dict mit f0_mean, f0_std, f0_range, voiced_ratio
    """
    try:
        # PYIN: Probabilistic YIN for pitch tracking
        f0, voiced_flag, _voiced_probs = librosa.pyin(audio, sr=sr, fmin=fmin, fmax=fmax, frame_length=2048)

        # Nur voiced frames berücksichtigen
        voiced_f0 = f0[voiced_flag]

        if len(voiced_f0) > 0:
            return {
                "f0_mean": float(np.nanmean(voiced_f0)),
                "f0_std": float(np.nanstd(voiced_f0)),
                "f0_range": float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0)),
                "voiced_ratio": float(np.sum(voiced_flag) / len(voiced_flag)),
            }
        else:
            return {"f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0, "voiced_ratio": 0.0}

    except Exception as e:
        logger.warning("F0 computation fehlgeschlagen: %s", e)
        return {"f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0, "voiced_ratio": 0.0}


def compute_hnr(audio: np.ndarray, _sr: int, _f0: np.ndarray | None = None) -> dict[str, float]:
    """
    Berechnet Harmonic-to-Noise Ratio (HNR).

    Simplified implementation: Vergleicht harmonische vs. residuale Energie.

    Args:
        audio: Audio-Signal (mono)
        sr: Sample Rate
        f0: Optional F0-Track (falls bereits berechnet)

    Returns:
        Dict mit hnr_mean, hnr_std (in dB)
    """
    try:
        # STFT-basierte HNR-Schätzung
        y_harmonic, y_percussive = librosa.effects.hpss(audio)

        # Energie-Verhältnis (harmonisch / perkussiv als Proxy für Noise)
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)

        if percussive_energy > 0:
            hnr_db = 10 * np.log10(harmonic_energy / percussive_energy + 1e-10)
        else:
            hnr_db = 40.0  # Very high HNR (pure harmonic)

        return {"hnr_mean": float(hnr_db), "hnr_std": 0.0}  # Simplified

    except Exception as e:
        logger.warning("HNR computation fehlgeschlagen: %s", e)
        return {"hnr_mean": 0.0, "hnr_std": 0.0}


# ==============================================================================
# Main Monitor Class
# ==============================================================================


class PermanentAudioMonitor:
    """
    Permanent Audio Monitoring System.

    Tracking ALLER Processing-Schritte für:
    - Quality Assurance
    - Continuous Learning
    - Audit Compliance
    - Performance Optimization

    Usage:
        monitor = PermanentAudioMonitor()
        monitor.capture_baseline(audio, sr, file_path="input.wav")

        # In jedem Processing-Modul:
        monitor.start_module("denoiser")
        processed = module.process(audio)
        monitor.end_module(audio, processed, sr, confidence=0.95)

        # Am Ende:
        monitor.export_audit_report(output_dir="./audits")
    """

    def __init__(self):
        self.baseline_metrics: AudioMetrics | None = None
        self.final_metrics: AudioMetrics | None = None
        self.module_logs: list[ModuleLog] = []
        self.current_module: str | None = None
        self.current_module_start_time: float | None = None
        self.file_path: str | None = None
        self.metadata: dict[str, Any] = {}

    def _compute_metrics(self, audio: np.ndarray, sr: int) -> AudioMetrics:
        """Berechnet alle Audio-Metriken."""
        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        # Spectral Features
        bark_spectrum = compute_bark_spectrum(audio, sr, n_bands=24)
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))

        # Dynamic Features
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        crest_factor = peak / (rms + 1e-10)
        dynamic_range_db = 20 * np.log10(peak / (rms + 1e-10))

        # Clipping Detection
        clipping_ratio = float(np.sum(np.abs(audio) > 0.99) / len(audio))

        # Pitch Features
        f0_stats = compute_f0_stats(audio, sr)

        # Harmonic Features
        hnr_stats = compute_hnr(audio, sr)

        # SNR Estimate (simplified: RMS vs. noise floor)
        noise_floor = np.percentile(np.abs(audio), 10)
        snr_estimate = 20 * np.log10(rms / (noise_floor + 1e-10))

        return AudioMetrics(
            bark_spectrum=bark_spectrum,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            rms=rms,
            crest_factor=crest_factor,
            dynamic_range_db=dynamic_range_db,
            peak_amplitude=peak,
            f0_mean=f0_stats["f0_mean"],
            f0_std=f0_stats["f0_std"],
            f0_range=f0_stats["f0_range"],
            voiced_ratio=f0_stats["voiced_ratio"],
            hnr_mean=hnr_stats["hnr_mean"],
            hnr_std=hnr_stats["hnr_std"],
            snr_estimate=snr_estimate,
            clipping_ratio=clipping_ratio,
            timestamp=datetime.now().isoformat(),
        )

    def capture_baseline(
        self, audio: np.ndarray, sr: int, file_path: str | None = None, metadata: dict[str, Any] | None = None
    ):
        """
        Erfasst Baseline-Metriken (vor jeglichem Processing).

        Args:
            audio: Input Audio (mono oder stereo)
            sr: Sample Rate
            file_path: Pfad zur Input-Datei
            metadata: Zusätzliche Metadaten
        """
        logger.info("📊 Capturing baseline metrics for: %s", file_path or "unknown")
        self.file_path = file_path
        self.metadata = metadata or {}
        self.baseline_metrics = self._compute_metrics(audio, sr)
        logger.info(
            "   ✅ Baseline captured: RMS=%.4f, F0=%.1f Hz, HNR=%.1f dB",
            self.baseline_metrics.rms,
            self.baseline_metrics.f0_mean,
            self.baseline_metrics.hnr_mean,
        )

    def start_module(self, module_name: str):
        """Startet Tracking eines Processing-Moduls."""
        self.current_module = module_name
        self.current_module_start_time = time.time()
        logger.debug("🔧 Starting module: %s", module_name)

    def end_module(
        self,
        audio_in: np.ndarray,
        audio_out: np.ndarray,
        sr: int,
        confidence: float = 1.0,
        quality_gate_passed: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Beendet Tracking eines Processing-Moduls und loggt Metriken.

        Args:
            audio_in: Input Audio des Moduls
            audio_out: Output Audio des Moduls
            sr: Sample Rate
            confidence: Confidence Score (0-1)
            quality_gate_passed: Quality Gate Status
            metadata: Zusätzliche Metadaten
        """
        if self.current_module is None:
            logger.warning("end_module() called without start_module()")
            return

        processing_time_ms = (time.time() - (self.current_module_start_time or 0.0)) * 1000

        module_log = ModuleLog(
            module_name=self.current_module,
            metrics_pre=self._compute_metrics(audio_in, sr),
            metrics_post=self._compute_metrics(audio_out, sr),
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            quality_gate_passed=quality_gate_passed,
            metadata=metadata or {},
        )

        self.module_logs.append(module_log)
        logger.info(
            "   ✅ Module %s abgeschlossen: %.1fms, confidence=%.2f, gate=%s",
            self.current_module,
            processing_time_ms,
            confidence,
            "PASS" if quality_gate_passed else "FAIL",
        )

        self.current_module = None
        self.current_module_start_time = None

    def capture_final(self, audio: np.ndarray, sr: int):
        """Erfasst finale Metriken (nach allem Processing)."""
        logger.info("📊 Capturing final metrics...")
        self.final_metrics = self._compute_metrics(audio, sr)

    def compute_cas_improvement(self) -> float:
        """
        Berechnet CAS (Creative Authenticity Score) Verbesserung.

        Simplified: Vergleicht HNR, SNR, Dynamic Range.

        Returns:
            float: CAS improvement (positive = better, negative = worse)
        """
        if self.baseline_metrics is None or self.final_metrics is None:
            return 0.0

        # Simplified CAS based on key metrics
        baseline_score = (
            self.baseline_metrics.hnr_mean * 0.3
            + self.baseline_metrics.snr_estimate * 0.3
            + self.baseline_metrics.dynamic_range_db * 0.2
            - self.baseline_metrics.clipping_ratio * 100 * 0.2
        )

        final_score = (
            self.final_metrics.hnr_mean * 0.3
            + self.final_metrics.snr_estimate * 0.3
            + self.final_metrics.dynamic_range_db * 0.2
            - self.final_metrics.clipping_ratio * 100 * 0.2
        )

        return final_score - baseline_score

    def export_audit_report(self, output_dir: str = "./audits", formats: list[str] | None = None):
        """
        Exportiert Audit-Report in verschiedenen Formaten.

        Args:
            output_dir: Output-Verzeichnis
            formats: Liste von Formaten ["json", "yaml", "csv"] (default: alle)
        """
        if formats is None:
            formats = ["json", "yaml", "csv"]

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Compute CAS improvement
        cas_improvement = self.compute_cas_improvement()

        # Build report
        report = AuditReport(
            file_path=self.file_path or "unknown",
            baseline_metrics=self.baseline_metrics,
            final_metrics=self.final_metrics,
            module_logs=self.module_logs,
            total_processing_time_ms=sum(log.processing_time_ms for log in self.module_logs),
            overall_quality_passed=all(log.quality_gate_passed for log in self.module_logs),
            cas_improvement=cas_improvement,
            metadata=self.metadata,
        )

        # Generate filename base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base = f"audit_{timestamp}"

        # Export JSON (machine-readable)
        if "json" in formats:
            json_path = output_path / f"{file_base}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2)
            logger.info("✅ Exported JSON audit: %s", json_path)

        # Export YAML (human-readable)
        if "yaml" in formats:
            yaml_path = output_path / f"{file_base}.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(asdict(report), f, sort_keys=False, default_flow_style=False)
            logger.info("✅ Exported YAML audit: %s", yaml_path)

        # Export CSV (Excel-compatible)
        if "csv" in formats:
            csv_path = output_path / f"{file_base}.csv"
            self._export_csv(report, csv_path)
            logger.info("✅ Exported CSV audit: %s", csv_path)

        logger.info("📊 Audit Report Summary:")
        logger.info("   ├─ File: %s", report.file_path)
        logger.info("   ├─ Modules: %s", len(report.module_logs))
        logger.info("   ├─ Total Time: %.1f ms", report.total_processing_time_ms)
        logger.info("   ├─ Quality Gates: %s", "ALL PASS ✅" if report.overall_quality_passed else "FAILED ❌")
        logger.info("   └─ CAS Improvement: %+.2f", report.cas_improvement)

    def _export_csv(self, report: AuditReport, csv_path: Path):
        """Exportiert Report als CSV (simplified - nur Module Summary)."""
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Module",
                    "Processing Time (ms)",
                    "Confidence",
                    "Quality Gate",
                    "RMS Delta",
                    "F0 Delta (Hz)",
                    "HNR Delta (dB)",
                ]
            )

            # Module rows
            for log in report.module_logs:
                if log.metrics_pre and log.metrics_post:
                    rms_delta = log.metrics_post.rms - log.metrics_pre.rms
                    f0_delta = log.metrics_post.f0_mean - log.metrics_pre.f0_mean
                    hnr_delta = log.metrics_post.hnr_mean - log.metrics_pre.hnr_mean

                    writer.writerow(
                        [
                            log.module_name,
                            f"{log.processing_time_ms:.1f}",
                            f"{log.confidence:.2f}",
                            "PASS" if log.quality_gate_passed else "FAIL",
                            f"{rms_delta:+.4f}",
                            f"{f0_delta:+.1f}",
                            f"{hnr_delta:+.1f}",
                        ]
                    )

            # Summary row
            writer.writerow([])
            writer.writerow(["TOTAL", f"{report.total_processing_time_ms:.1f}", "", "", "", "", ""])
            writer.writerow(["CAS Improvement", f"{report.cas_improvement:+.2f}", "", "", "", "", ""])


# ==============================================================================
# Example Usage & Test
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate test signal
    _sr = 44100
    _duration = 2.0
    _t = np.linspace(0, _duration, int(_sr * _duration))
    _audio = 0.5 * np.sin(2 * np.pi * 440 * _t)  # 440 Hz sine wave

    # Add some noise
    _audio += 0.05 * np.random.randn(len(_audio))

    logging.info("=" * 80)
    logging.info("PERMANENTAUDIOMONITOR TEST")
    logging.info("=" * 80)

    # Create monitor
    monitor = PermanentAudioMonitor()

    # Capture baseline
    monitor.capture_baseline(_audio, _sr, file_path="test_input.wav", metadata={"source": "test"})

    # Simulate Module 1: Denoiser
    monitor.start_module("adaptive_denoiser")
    time.sleep(0.05)  # Simulate processing
    audio_denoised = _audio * 0.95  # Simulate slight change
    monitor.end_module(_audio, audio_denoised, _sr, confidence=0.92, quality_gate_passed=True)

    # Simulate Module 2: Voice Enhancer
    monitor.start_module("voice_enhancer")
    time.sleep(0.03)
    audio_enhanced = audio_denoised * 1.1  # Simulate enhancement
    monitor.end_module(audio_denoised, audio_enhanced, _sr, confidence=0.88, quality_gate_passed=True)

    # Capture final
    monitor.capture_final(audio_enhanced, _sr)

    # Export report
    monitor.export_audit_report(output_dir="/tmp/aurik_audits", formats=["json", "yaml", "csv"])  # nosec B108

    logging.info("=" * 80)
    logging.info("✅ Test completed successfully!")
    logging.info("=" * 80)
