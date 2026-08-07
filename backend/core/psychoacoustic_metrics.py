"""
Psychoacoustic Metrics Module - Aurik 10.0.0
==========================================

Objective audio quality metrics for validation and optimization.

Metrics:
- PESQ: Perceptual Evaluation of Speech Quality
- SI-SDR: Scale-Invariant Signal-to-Distortion Ratio
- Spectral Distortion: Log-Spectral Distance
- Roughness (Zwicker): Psychoacoustic roughness
- Sharpness (Aures): High-frequency emphasis
- Naturalness Score: Custom composite metric

Scientific Foundation:
- ITU-T P.862: PESQ standard
- Roux et al. (2019): SI-SDR for source separation
- Zwicker & Fastl (1999): Psychoacoustics of Hearing
- Aures (1985): Berechnungsverfahren für den sensorischen Wohlklang

Author: Aurik 10.0.0 Development Team
Date: 15. Februar 2026
"""

import logging

import numpy as np

from backend.core.comprehensive_metrics import PsychoAcousticMetrics  # canonical (§dedup)

logger = logging.getLogger(__name__)


def measure_quality_improvement(
    original: np.ndarray, processed: np.ndarray, sample_rate: int = 44100
) -> dict[str, float]:
    """
    Misst quality improvement from processing.

    Args:
        original: Original audio
        processed: Processed audio
        sample_rate: Sample rate

    Returns:
        Dictionary with improvement metrics
    """
    metrics = PsychoAcousticMetrics(sample_rate)

    original_quality = metrics.calculate_naturalness_score(original)
    processed_quality = metrics.calculate_naturalness_score(processed, reference=original)

    improvement = {
        "original_naturalness": original_quality["naturalness_overall"],
        "processed_naturalness": processed_quality["naturalness_overall"],
        "improvement": processed_quality["naturalness_overall"] - original_quality["naturalness_overall"],
        "improvement_percent": (
            (processed_quality["naturalness_overall"] - original_quality["naturalness_overall"])
            / (original_quality["naturalness_overall"] + 1e-10)
            * 100
        ),
    }

    # Add detailed metrics
    for key in ["spectral_flatness", "temporal_smoothness", "harmonic_coherence", "noise_floor_consistency"]:
        improvement[f"original_{key}"] = original_quality[key]
        improvement[f"processed_{key}"] = processed_quality[key]

    # sisdr_db entfernt — verboten §4.4+§10.2 (SI-SDR Sprach-Metrik)
    if "spectral_distortion_db" in processed_quality:
        improvement["spectral_distortion_db"] = processed_quality["spectral_distortion_db"]

    return improvement


if __name__ == "__main__":
    # Test with synthetic audio
    logger.debug("\n" + "=" * 70)
    logger.debug("Psychoacoustic Metrics Test")
    logger.debug("=" * 70)

    # Generate test signals
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sr))

    # Clean signal
    clean = np.sin(2 * np.pi * 440 * t) * 0.3

    # Degraded signal (with artifacts)
    degraded = clean.copy()
    degraded += np.random.randn(len(clean)) * 0.05  # Add noise

    # Add clicks
    for _ in range(10):
        pos = np.random.randint(0, len(clean))
        degraded[pos] += 0.5

    # Calculate metrics
    metrics = PsychoAcousticMetrics(sr)

    logger.debug("\nClean Signal:")
    clean_scores = metrics.calculate_naturalness_score(clean)
    for key, val in clean_scores.items():
        logger.debug("  %s: %.3f", key, val)

    logger.debug("\nDegraded Signal:")
    degraded_scores = metrics.calculate_naturalness_score(degraded, reference=clean)
    for key, val in degraded_scores.items():
        logger.debug("  %s: %.3f", key, val)

    logger.debug("\nImprovement Analyse:")
    improvement = measure_quality_improvement(degraded, clean, sr)
    for key, val in improvement.items():
        logger.debug("  %s: %.3f", key, val)

    logger.debug("\n" + "=" * 70)
    logger.debug("✅ Psychoacoustic Metrics Module operational")
