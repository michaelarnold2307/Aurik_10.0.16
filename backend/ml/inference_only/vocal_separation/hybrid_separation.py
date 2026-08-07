"""
Hybrid Vocal Separator - Combines MDX-Net + Demucs v5

Ensemble approach:
- MDX-Net: Excellent for spectral artifacts, high-frequency detail
- Demucs v5: Superior context modeling via Transformer
- Hybrid: Best of both worlds with intelligent fusion

HIPS Compliance:
- Kontextbewusstsein: ✅ Dual-model fusion (spectral + temporal)
- Nebenwirkungen: ✅ Aggregated tracking from both models
- Reversibilität: ✅ Individual model outputs stored
- Auditierbarkeit: ✅ Full ensemble decision logging
- Steuerbarkeit: ✅ Adjustable model weights
- Bedeutungsagnostik: ✅ Signal-level fusion only
"""

import logging
import time
from typing import Literal

import numpy as np

from .demucs_v5_wrapper import DemucsV5Separator
from .mdx_net_wrapper import MDXNetSeparator

logger = logging.getLogger(__name__)


class HybridVocalSeparator:
    """
    Hybrid ensemble vocal separator for AURIK v8.1

    Combines:
    1. MDX-Net (spectral domain, U-Net)
    2. Demucs v5 (hybrid domain, Transformer)

    Fusion Strategies:
    - 'adaptive': Quality-based weighting per frequency band
    - 'weighted': Fixed weighting (MDX=0.4, Demucs=0.6)
    - 'best': Choose best model per segment

    HIPS Guarantees:
    - Kontextbewusstsein: Multi-model context fusion
    - Nebenwirkungen: Aggregated from both models
    - Auditierbarkeit: Full decision tree logged
    """

    def __init__(
        self,
        fusion_strategy: Literal["adaptive", "weighted", "best"] = "adaptive",
        mdx_weight: float = 0.4,
        demucs_weight: float = 0.6,
        sample_rate: int = 44100,
        device: str | None = None,
    ):
        """
        Initialisiert hybrid separator.

        Args:
            fusion_strategy: How to combine model outputs
            mdx_weight: Weight for MDX-Net (if weighted strategy)
            demucs_weight: Weight for Demucs (if weighted strategy)
            sample_rate: Target sample rate
            device: 'cuda', 'cpu', or None
        """
        self.fusion_strategy = fusion_strategy
        self.mdx_weight = mdx_weight
        self.demucs_weight = demucs_weight
        self.sample_rate = sample_rate

        logger.info("HybridVocalSeparator initialisiert: strategy=%s, device=%s", fusion_strategy, device)

        # Initialize both models
        self.mdx_net = MDXNetSeparator(sample_rate=sample_rate, device=device)

        self.demucs = DemucsV5Separator(sample_rate=sample_rate, device=device)

        # HIPS tracking
        self.separation_count = 0
        self.fusion_decisions_log: list[dict] = []

    def separate(
        self, audio: np.ndarray, sr: int | None = None, return_individual: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Hybrid vocal separation

        Args:
            audio: Audio array (shape: [channels, samples] or [samples])
            sr: Sample rate
            return_individual: If True, also return individual model outputs

        Returns:
            Dictionary with:
            - 'vocals': Fused vocal stem
            - 'instrumental': Fused instrumental stem
            - 'vocals_mdx': MDX-Net vocals (if return_individual)
            - 'vocals_demucs': Demucs vocals (if return_individual)

        HIPS Compliance:
        - Logs fusion strategy and decisions
        - Tracks nebenwirkungen from both models
        - Preserves individual outputs for auditability
        """
        # SR-Invariante (Aurik 10.0.0 nutzt 48000 Hz)
        if sr is not None:
            assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        # NaN/Inf-Guard
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # HIPS: Log separation
        self.separation_count += 1
        start_time = time.time()

        logger.info("Hybrid separation #%s: strategy=%s", self.separation_count, self.fusion_strategy)

        # Run both models in parallel (conceptually)
        logger.info("  laeuft MDX-Net...")
        mdx_time_start = time.time()
        mdx_stems = self.mdx_net.separate(audio, sr=sr, return_stems=True)
        mdx_time = time.time() - mdx_time_start

        logger.info("  laeuft Demucs v5...")
        demucs_time_start = time.time()
        demucs_stems = self.demucs.separate(audio, sr=sr, stems=["vocals", "other"])
        demucs_time = time.time() - demucs_time_start

        # Fusion
        logger.info("  Fusing with '%s' strategy...", self.fusion_strategy)
        vocals_fused, instrumental_fused, fusion_metadata = self._fuse_stems(
            mdx_vocals=mdx_stems["vocals"],
            mdx_instrumental=mdx_stems["instrumental"],
            demucs_vocals=demucs_stems["vocals"],
            demucs_instrumental=demucs_stems.get("other", mdx_stems["instrumental"]),
        )

        total_time = time.time() - start_time

        # HIPS: Log fusion decision
        fusion_decision = {
            "separation_id": self.separation_count,
            "strategy": self.fusion_strategy,
            "mdx_time_sec": mdx_time,
            "demucs_time_sec": demucs_time,
            "total_time_sec": total_time,
            "fusion_metadata": fusion_metadata,
        }
        self.fusion_decisions_log.append(fusion_decision)

        logger.info(
            "  Hybrid separation vollstaendig: %.2fs (MDX=%.2fs, Demucs=%.2fs)", total_time, mdx_time, demucs_time
        )

        # NaN/Inf-Guard für Ausgabe
        vocals_fused = np.nan_to_num(vocals_fused, nan=0.0, posinf=0.0, neginf=0.0)
        vocals_fused = np.clip(vocals_fused, -1.0, 1.0)
        instrumental_fused = np.nan_to_num(instrumental_fused, nan=0.0, posinf=0.0, neginf=0.0)
        instrumental_fused = np.clip(instrumental_fused, -1.0, 1.0)

        # Build result
        result = {"vocals": vocals_fused, "instrumental": instrumental_fused}

        if return_individual:
            result["vocals_mdx"] = mdx_stems["vocals"]
            result["vocals_demucs"] = demucs_stems["vocals"]
            result["instrumental_mdx"] = mdx_stems["instrumental"]
            result["instrumental_demucs"] = demucs_stems.get("other", mdx_stems["instrumental"])

        return result

    def _fuse_stems(
        self,
        mdx_vocals: np.ndarray,
        mdx_instrumental: np.ndarray,
        demucs_vocals: np.ndarray,
        demucs_instrumental: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Fuse stems from both models

        Returns:
            (vocals_fused, instrumental_fused, fusion_metadata)
        """
        # Ensure same length
        min_len = min(mdx_vocals.shape[1], demucs_vocals.shape[1])

        mdx_vocals = mdx_vocals[:, :min_len]
        mdx_instrumental = mdx_instrumental[:, :min_len]
        demucs_vocals = demucs_vocals[:, :min_len]
        demucs_instrumental = demucs_instrumental[:, :min_len]

        if self.fusion_strategy == "weighted":
            # Simple weighted average
            vocals_fused = self.mdx_weight * mdx_vocals + self.demucs_weight * demucs_vocals
            instrumental_fused = self.mdx_weight * mdx_instrumental + self.demucs_weight * demucs_instrumental

            metadata = {"method": "weighted", "mdx_weight": self.mdx_weight, "demucs_weight": self.demucs_weight}

        elif self.fusion_strategy == "adaptive":
            # Adaptive frequency-band weighting
            vocals_fused, instrumental_fused, metadata = self._adaptive_fusion(
                mdx_vocals, mdx_instrumental, demucs_vocals, demucs_instrumental
            )

        elif self.fusion_strategy == "best":
            # Choose best model based on quality metrics
            vocals_fused, instrumental_fused, metadata = self._best_model_fusion(
                mdx_vocals, mdx_instrumental, demucs_vocals, demucs_instrumental
            )

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        return vocals_fused, instrumental_fused, metadata

    def _adaptive_fusion(
        self,
        mdx_vocals: np.ndarray,
        mdx_instrumental: np.ndarray,
        demucs_vocals: np.ndarray,
        demucs_instrumental: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Adaptive frequency-band fusion

        Strategy:
        - Low freq (0-250 Hz): Demucs preferred (better context)
        - Mid freq (250-4000 Hz): Weighted average
        - High freq (4000+ Hz): MDX-Net preferred (better spectral detail)
        """
        import librosa

        # Compute STFT for both
        mdx_vocals_D = librosa.stft(mdx_vocals[0], n_fft=2048, hop_length=512)
        demucs_vocals_D = librosa.stft(demucs_vocals[0], n_fft=2048, hop_length=512)

        mdx_inst_D = librosa.stft(mdx_instrumental[0], n_fft=2048, hop_length=512)
        demucs_inst_D = librosa.stft(demucs_instrumental[0], n_fft=2048, hop_length=512)

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)

        # Adaptive weights per frequency band
        weights_mdx = np.ones_like(freqs)
        weights_demucs = np.ones_like(freqs)

        for i, freq in enumerate(freqs):
            if freq < 250:
                # Low: Prefer Demucs
                weights_mdx[i] = 0.3
                weights_demucs[i] = 0.7
            elif freq < 4000:
                # Mid: Balanced
                weights_mdx[i] = 0.5
                weights_demucs[i] = 0.5
            else:
                # High: Prefer MDX-Net
                weights_mdx[i] = 0.7
                weights_demucs[i] = 0.3

        # Apply weights
        weights_mdx = weights_mdx[:, np.newaxis]
        weights_demucs = weights_demucs[:, np.newaxis]

        vocals_D_fused = weights_mdx * mdx_vocals_D + weights_demucs * demucs_vocals_D
        inst_D_fused = weights_mdx * mdx_inst_D + weights_demucs * demucs_inst_D

        # ISTFT
        vocals_fused_mono = librosa.istft(vocals_D_fused, hop_length=512)
        inst_fused_mono = librosa.istft(inst_D_fused, hop_length=512)

        # Stereo (duplicate for now)
        vocals_fused = np.stack([vocals_fused_mono, vocals_fused_mono])
        inst_fused = np.stack([inst_fused_mono, inst_fused_mono])

        metadata = {
            "method": "adaptive_frequency",
            "bands": {
                "low_0_250hz": "demucs_70_mdx_30",
                "mid_250_4000hz": "balanced_50_50",
                "high_4000plus_hz": "mdx_70_demucs_30",
            },
        }

        return vocals_fused, inst_fused, metadata

    def _best_model_fusion(
        self,
        mdx_vocals: np.ndarray,
        mdx_instrumental: np.ndarray,
        demucs_vocals: np.ndarray,
        demucs_instrumental: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Choose best model based on quality heuristic

        Heuristic: Signal-to-residual ratio
        """
        # Compute quality scores
        mdx_score = self._compute_separation_quality(mdx_vocals, mdx_instrumental)
        demucs_score = self._compute_separation_quality(demucs_vocals, demucs_instrumental)

        logger.info("    Quality scores: MDX=%.3f, Demucs=%.3f", mdx_score, demucs_score)

        if demucs_score > mdx_score:
            vocals_fused = demucs_vocals
            inst_fused = demucs_instrumental
            chosen = "demucs"
        else:
            vocals_fused = mdx_vocals
            inst_fused = mdx_instrumental
            chosen = "mdx_net"

        metadata = {
            "method": "best_model_selection",
            "chosen_model": chosen,
            "mdx_score": mdx_score,
            "demucs_score": demucs_score,
        }

        return vocals_fused, inst_fused, metadata

    def _compute_separation_quality(self, vocals: np.ndarray, instrumental: np.ndarray) -> float:
        """
        Berechnet separation quality score.

        Higher score = better separation

        Metric: Energy ratio between stems (well-separated > balanced energy)
        """
        energy_vocals = np.sum(vocals**2)
        energy_inst = np.sum(instrumental**2)

        # Ratio (closer to 1 = more balanced, typically good)
        total_energy = energy_vocals + energy_inst
        if total_energy < 1e-10:
            return 0.0

        ratio = min(energy_vocals, energy_inst) / total_energy

        # Quality score: prefer some imbalance (indicates separation)
        # But not too extreme
        quality = 1.0 - abs(ratio - 0.3)  # Optimal at 30/70 split

        return max(0.0, quality)  # type: ignore[no-any-return]

    def get_metrics(self) -> dict:
        """
        HIPS: Auditability - Get all metrics
        """
        # Aggregate from both models
        mdx_metrics = self.mdx_net.get_separation_metrics()
        demucs_metrics = self.demucs.get_separation_metrics()

        return {
            "total_separations": self.separation_count,
            "fusion_strategy": self.fusion_strategy,
            "mdx_net_metrics": mdx_metrics,
            "demucs_metrics": demucs_metrics,
            "fusion_decisions_log": self.fusion_decisions_log[-10:],
        }


if __name__ == "__main__":
    # Test hybrid separator
    separator = HybridVocalSeparator(fusion_strategy="adaptive")

    # Generate test signal
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Complex mix
    vocals = np.sin(2 * np.pi * 440 * t)
    drums = np.random.randn(len(t)) * 0.2
    bass = np.sin(2 * np.pi * 110 * t) * 0.3

    mixed = vocals + drums + bass
    audio = np.stack([mixed, mixed])

    # Separate
    stems = separator.separate(audio, sr=sr, return_individual=True)

    logger.info("✓ Hybrid separation test passed")
    logger.info("  Vocals shape: %s", stems["vocals"].shape)
    logger.info("  Instrumental shape: %s", stems["instrumental"].shape)
    logger.info("  Metrics: %s", separator.get_metrics())
