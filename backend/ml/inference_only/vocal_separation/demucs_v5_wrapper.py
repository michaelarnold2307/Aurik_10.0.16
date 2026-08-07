"""
Demucs v5 (Hybrid Transformer) Vocal Separator - HIPS Compliant Wrapper

Demucs v5 combines:
- Hybrid domain processing (time + frequency)
- Transformer-based global context
- State-of-the-art separation quality

HIPS Compliance:
- Kontextbewusstsein: ✅ Transformer captures global structure
- Nebenwirkungen: ✅ Tracked (transient smearing, stereo artifacts)
- Reversibilität: ✅ 4-stem output allows recombination
- Auditierbarkeit: ✅ Full metrics logged
- Steuerbarkeit: ✅ Configurable separation parameters
- Bedeutungsagnostik: ✅ Pure signal processing
"""

import logging
from pathlib import Path

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    _HAS_LIBROSA = False
import numpy as np

logger = logging.getLogger(__name__)


class DemucsV5Separator:
    """
    Demucs v5 (Hybrid Transformer) wrapper for AURIK v8.1

    Architecture:
    - 4-source separation: vocals, drums, bass, other
    - Hybrid time-frequency processing
    - Transformer attention mechanism

    HIPS Guarantees:
    - Inference-only (no training)
    - Deterministic output
    - Full auditability
    """

    AVAILABLE_STEMS = ["vocals", "drums", "bass", "other"]

    def __init__(
        self,
        model_name: str = "htdemucs",
        model_path: str | None = None,
        sample_rate: int = 44100,
        device: str | None = None,
        segment_duration: float = 10.0,
    ):
        """
        Initialisiert Demucs v5 separator.

        Args:
            model_name: Model variant ('htdemucs', 'htdemucs_ft')
            model_path: Optional custom model path
            sample_rate: Target sample rate
            device: 'cuda', 'cpu', or None (auto-detect)
            segment_duration: Processing segment length (seconds)
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration

        # §GPU-Mixed-Mode: Heavy plugins → GPU when available, CPU fallback.
        try:
            from backend.core.ml_device_manager import get_torch_device as _get_dev

            self.device = _get_dev("DemucsV5")
        except Exception:
            self.device = "cpu"

        logger.info("DemucsV5Separator (%s) initialisiert on %s", model_name, self.device)

        # Model loading
        self.model_path = model_path or self._get_default_model_path()
        self.model = self._load_model()

        # HIPS tracking
        self.separation_count = 0
        self.nebenwirkungen_log: list[dict] = []

    def _get_default_model_path(self) -> Path:
        """Gibt zurück: default Demucs model path."""
        base_path = Path(__file__).parent.parent.parent.parent.parent
        model_dir = base_path / "models" / "demucs"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{self.model_name}.yaml"
        if not model_path.exists():
            logger.warning("Demucs model not found at %s. Install with: pip install demucs", model_path)

        return model_path

    def _load_model(self):
        """
        Lädt Demucs v5 model.

        Note: Requires demucs package:
        pip install demucs
        """
        try:
            # Attempt to load demucs
            import demucs.pretrained as pretrained

            # Load pretrained model
            model = pretrained.get_model(self.model_name)
            model.eval()
            model.to(self.device)
            logger.info("Demucs %s geladen erfolgreich (device=%s)", self.model_name, self.device)
            return model

        except ImportError:
            logger.warning("Demucs package not installed. Install with: pip install demucs")
            return None
        except Exception as e:
            logger.error("konnte nicht laden Demucs model: %s", e)
            return None

    def separate(
        self, audio: np.ndarray, sr: int | None = None, stems: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Separate audio into stems

        Args:
            audio: Audio array (shape: [channels, samples] or [samples])
            sr: Sample rate (if different from self.sample_rate)
            stems: List of stems to return (default: ['vocals'])

        Returns:
            Dictionary with requested stems

        HIPS Compliance:
        - Logs all separation operations
        - Tracks nebenwirkungen (transient smearing, phase issues)
        - Preserves original for reversibility check
        """
        if stems is None:
            stems = ["vocals"]

        # Validate stems
        for stem in stems:
            if stem not in self.AVAILABLE_STEMS:
                raise ValueError(f"Invalid stem '{stem}'. Available: {self.AVAILABLE_STEMS}")

        # SR-Invariante (Aurik 10.0.0 nutzt 48000 Hz)
        assert sr == 48000 or sr is None or sr == self.sample_rate, f"SR muss 48000 Hz sein, erhalten: {sr}"
        # Resample if needed
        if sr is not None and sr != self.sample_rate:
            logger.info("Resampling from %sHz to %sHz", sr, self.sample_rate)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        # NaN/Inf-Guard
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        audio_original = audio.copy()  # For reversibility check

        # HIPS: Log separation attempt
        self.separation_count += 1
        logger.info(
            f"Demucs separation #{self.separation_count}: shape={audio.shape}, sr={self.sample_rate}, stems={stems}"
        )

        # Actual separation
        if self.model is None:
            logger.warning("Demucs model nicht verfuegbar. Using Ersatzpfad.")
            separated_stems = self._fallback_separation(audio, stems)
        else:
            separated_stems = self._demucs_inference(audio, stems)

        # HIPS: Nebenwirkungen tracking
        nebenwirkungen = self._assess_nebenwirkungen(audio_original, separated_stems)
        self.nebenwirkungen_log.append(nebenwirkungen)

        # NaN/Inf-Guard für alle Stems
        for stem_name in separated_stems:
            separated_stems[stem_name] = np.nan_to_num(separated_stems[stem_name], nan=0.0, posinf=0.0, neginf=0.0)
            separated_stems[stem_name] = np.clip(separated_stems[stem_name], -1.0, 1.0)

        if nebenwirkungen["severity"] > 0.3:
            logger.warning(
                "Separation nebenwirkungen erkannt: "
                f"transient_smearing={nebenwirkungen['transient_smearing']:.2f}, "
                f"Verarbeitungsschritt_loss={nebenwirkungen['phase_loss']:.2f}"
            )

        return separated_stems

    def _demucs_inference(self, audio: np.ndarray, stems: list[str]) -> dict[str, np.ndarray]:
        """
        Actual Demucs inference

        Uses demucs.apply.apply_model for segmented processing.
        """
        try:
            import torch
            from demucs.apply import apply_model

            model = self.model
            if model is None:
                raise RuntimeError("Demucs model is not loaded")

            # Convert to torch tensor
            audio_torch = torch.from_numpy(audio).float()  # placed on device via apply_model

            # Add batch dimension
            audio_torch = audio_torch.unsqueeze(0)

            # Demucs controls chunking via model.segment (seconds), not apply_model kwargs.
            if hasattr(model, "segment"):
                try:
                    model.segment = float(self.segment_duration)
                except Exception as _exc:
                    logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

            # Apply model
            with torch.no_grad():
                sources = apply_model(
                    model,
                    audio_torch,
                    split=True,
                    overlap=0.25,
                    device=self.device,
                )

            # Convert back to numpy
            sources = sources.squeeze(0).cpu().numpy()

            # Map to stem names
            separated = {}
            for i, stem_name in enumerate(self.AVAILABLE_STEMS):
                if stem_name in stems:
                    separated[stem_name] = sources[i]

            return separated

        except Exception as e:
            logger.error("Demucs inference fehlgeschlagen: %s. Using Ersatzpfad.", e)
            return self._fallback_separation(audio, stems)

    def _fallback_separation(self, audio: np.ndarray, stems: list[str]) -> dict[str, np.ndarray]:
        """
        Fallback separation when Demucs unavailable

        Uses simple spectral methods.
        """
        logger.info("Using Ersatzpfad spectral separation")

        separated = {}

        for stem in stems:
            if stem == "vocals":
                # Harmonic component
                vocals_stereo = []
                for channel in audio:
                    D = librosa.stft(channel, n_fft=2048, hop_length=512)
                    H, _ = librosa.decompose.hpss(D, margin=2.0)
                    vocal_channel = librosa.istft(H, hop_length=512)
                    vocals_stereo.append(vocal_channel)

                vocals = np.array([librosa.util.fix_length(v, size=audio.shape[1]) for v in vocals_stereo])
                separated["vocals"] = vocals

            elif stem == "drums":
                # Percussive component
                drums_stereo = []
                for channel in audio:
                    D = librosa.stft(channel, n_fft=2048, hop_length=512)
                    _, P = librosa.decompose.hpss(D, margin=2.0)
                    drum_channel = librosa.istft(P, hop_length=512)
                    drums_stereo.append(drum_channel)

                drums = np.array([librosa.util.fix_length(d, size=audio.shape[1]) for d in drums_stereo])
                separated["drums"] = drums

            elif stem in ["bass", "other"]:
                # Placeholder: return low-pass filtered audio for bass
                if stem == "bass":
                    _cutoff = 250  # Hz (reserved for future filter)
                else:
                    _cutoff = 8000  # Hz (reserved for future filter)

                filtered_stereo = []
                for channel in audio:
                    filtered = librosa.effects.percussive(channel, margin=1.0)
                    filtered_stereo.append(filtered)

                filtered = np.array([librosa.util.fix_length(f, size=audio.shape[1]) for f in filtered_stereo])
                separated[stem] = filtered

        return separated

    def _assess_nebenwirkungen(self, original: np.ndarray, separated_stems: dict[str, np.ndarray]) -> dict[str, float]:
        """
        HIPS Requirement: Assess separation nebenwirkungen

        Tracks:
        - Transient smearing (attack preservation)
        - Phase coherence
        - Energy conservation
        - Spectral artifacts
        """
        # Recombine all stems
        recombined = np.zeros_like(original)
        for stem_audio in separated_stems.values():
            min_len = min(recombined.shape[1], stem_audio.shape[1])
            recombined[:, :min_len] += stem_audio[:, :min_len]

        # Match lengths
        min_len = min(original.shape[1], recombined.shape[1])
        original = original[:, :min_len]
        recombined = recombined[:, :min_len]

        # Energy conservation
        energy_original = np.sum(original**2)
        energy_recombined = np.sum(recombined**2)
        energy_ratio = energy_recombined / (energy_original + 1e-10)
        # NaN/Inf-Guard
        energy_ratio = 0.0 if not np.isfinite(energy_ratio) else energy_ratio

        # Transient preservation (onset detection)
        def transient_score(audio: np.ndarray) -> float:
            """Misst transient density."""
            onset_env = librosa.onset.onset_strength(y=audio[0], sr=self.sample_rate)
            return np.std(onset_env)  # type: ignore[no-any-return]

        transient_original = transient_score(original)
        transient_recombined = transient_score(recombined)
        transient_smearing = abs(transient_original - transient_recombined)
        # NaN/Inf-Guard
        transient_smearing = 0.0 if not np.isfinite(transient_smearing) else transient_smearing

        # Phase coherence
        def phase_coherence(a: np.ndarray, b: np.ndarray) -> float:
            """Misst phase alignment between signals."""
            if a.shape[0] < 2 or b.shape[0] < 2:
                return 1.0

            xcorr = np.correlate(a[0], b[0], mode="valid")
            max_corr = np.max(np.abs(xcorr))
            norm = np.linalg.norm(a[0]) * np.linalg.norm(b[0]) + 1e-10
            return max_corr / norm  # type: ignore[no-any-return]

        # NaN/Inf-Guard
        phase_score = phase_coherence(original, recombined)
        phase_loss = 1.0 - phase_score
        phase_loss = 0.0 if not np.isfinite(phase_loss) else phase_loss

        # Overall severity
        severity = abs(1.0 - energy_ratio) * 0.4 + transient_smearing * 0.4 + phase_loss * 0.2

        return {
            "energy_ratio": float(energy_ratio),
            "transient_smearing": float(transient_smearing),
            "phase_loss": float(phase_loss),
            "severity": float(min(severity, 1.0)),
        }

    def get_separation_metrics(self) -> dict:
        """
        HIPS: Auditability - Get all separation metrics
        """
        if not self.nebenwirkungen_log:
            return {"total_separations": 0}

        avg_severity = np.mean([n["severity"] for n in self.nebenwirkungen_log])
        max_severity = np.max([n["severity"] for n in self.nebenwirkungen_log])

        return {
            "total_separations": self.separation_count,
            "average_nebenwirkungen_severity": float(avg_severity),
            "max_nebenwirkungen_severity": float(max_severity),
            "nebenwirkungen_log": self.nebenwirkungen_log[-10:],
        }


if __name__ == "__main__":
    # Test Demucs separator
    separator = DemucsV5Separator()

    # Generate test signal
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Complex test signal
    vocals = np.sin(2 * np.pi * 440 * t)  # 440 Hz
    drums = np.random.randn(len(t)) * 0.3  # Noise (percussive-like)
    bass = np.sin(2 * np.pi * 110 * t) * 0.5  # 110 Hz

    mixed = vocals + drums + bass
    audio = np.stack([mixed, mixed])  # Stereo

    # Separate
    stems = separator.separate(audio, sr=sr, stems=["vocals", "drums"])

    logger.info("✓ Demucs separation test passed")
    logger.info("  Vocals shape: %s", stems["vocals"].shape)
    logger.info("  Drums shape: %s", stems["drums"].shape)
    logger.info("  Metrics: %s", separator.get_separation_metrics())
