"""
forensics/dataset_generator.py
Golden Sample Generator für Signal Forensics Training
======================================================

Generiert Trainings-Datasets für Medium/Era/Defect Detection:
- Nutzt vorhandene golden_samples/ Struktur
- Extrahiert Features aus echten Samples
- Generiert synthetische Samples mit DSP Emulation
- Augmentation für robuste ML-Modelle

USAGE:
    from backend.core.forensics.dataset_generator import DatasetGenerator

    gen = DatasetGenerator()
    dataset = gen.generate_medium_dataset(n_samples=1000)
    # Returns: {'X': features, 'y': labels, 'metadata': info}
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from backend.core.forensics.signatures import ERA_SIGNATURES, EraType, MediaType

logger = logging.getLogger(__name__)


def _sosfilt_array(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Gibt only the filtered audio array from scipy.signal.sosfilt zurück."""
    filtered = scipy_signal.sosfilt(sos, audio)
    if isinstance(filtered, tuple):
        return np.asarray(filtered[0])  # type: ignore[no-any-return]
    return np.asarray(filtered)  # type: ignore[no-any-return]


@dataclass
class SyntheticSample:
    """Metadata für synthetisches Sample."""

    audio: np.ndarray
    sample_rate: int
    medium_type: MediaType
    era_type: EraType
    defects: list[str]
    generation: str  # "real" or "synthetic"
    source_file: str | None = None


class DatasetGenerator:
    """
    Generiert Training-Datasets für Signal Forensics.
    Kombiniert echte Golden Samples mit synthetischen Samples.
    """

    def __init__(
        self, golden_samples_dir: str = "golden_samples", output_dir: str = "forensics/datasets", target_sr: int = 48000
    ) -> None:
        self.golden_samples_dir = Path(golden_samples_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr

    def generate_medium_dataset(
        self, n_synthetic_per_medium: int = 100, real_samples_only: bool = False
    ) -> dict[str, Any]:
        """
        Generiert Dataset für Medium Detection Training.

        Args:
            n_synthetic_per_medium: Anzahl synthetischer Samples pro Medium
            real_samples_only: Nur echte Samples verwenden (kein Synthetic)

        Returns:
            {
                'samples': List[SyntheticSample],
                'n_real': int,
                'n_synthetic': int,
                'medium_distribution': Dict[str, int]
            }
        """
        samples = []

        # 1. Lade echte Golden Samples (wenn vorhanden)
        real_samples = self._load_real_samples()
        samples.extend(real_samples)

        if not real_samples_only:
            # 2. Generiere synthetische Samples für jedes Medium
            media_types = [
                MediaType.VINYL_LP_STEREO,  # VINYL category
                MediaType.TAPE_15IPS,  # TAPE category
                MediaType.CASSETTE_TYPE_I,  # CASSETTE category
                MediaType.CD_STANDARD,  # CD category
                MediaType.HIRES_PCM,  # DIGITAL category
                MediaType.MP3_320,  # LOSSY category
            ]

            for medium in media_types:
                for i in range(n_synthetic_per_medium):
                    synthetic_sample = self._generate_synthetic_medium(medium)
                    samples.append(synthetic_sample)

        # 3. Statistiken
        medium_distribution: dict[str, int] = {}
        for sample in samples:
            medium_name = sample.medium_type.name
            medium_distribution[medium_name] = medium_distribution.get(medium_name, 0) + 1

        return {
            "samples": samples,
            "n_real": len(real_samples),
            "n_synthetic": len(samples) - len(real_samples),
            "medium_distribution": medium_distribution,
        }

    def generate_era_dataset(self, n_synthetic_per_era: int = 50) -> dict[str, Any]:
        """
        Generiert Dataset für Era Detection Training.

        Args:
            n_synthetic_per_era: Anzahl synthetischer Samples pro Era

        Returns:
            {
                'samples': List[SyntheticSample],
                'era_distribution': Dict[str, int]
            }
        """
        samples = []

        # Generiere für jede Era
        for era_type in EraType:
            if era_type == EraType.UNKNOWN:
                continue

            for i in range(n_synthetic_per_era):
                synthetic_sample = self._generate_synthetic_era(era_type)
                samples.append(synthetic_sample)

        # Statistiken
        era_distribution: dict[str, int] = {}
        for sample in samples:
            era_name = sample.era_type.value
            era_distribution[era_name] = era_distribution.get(era_name, 0) + 1

        return {"samples": samples, "era_distribution": era_distribution}

    def _load_real_samples(self) -> list[SyntheticSample]:
        """
        Lädt echte Golden Samples aus golden_samples/ Verzeichnis.
        """
        samples: list[SyntheticSample] = []

        if not self.golden_samples_dir.exists():
            logger.debug("⚠️  Golden samples directory not found: %s", self.golden_samples_dir)
            return samples

        # Suche nach Audio-Dateien
        audio_files = list(self.golden_samples_dir.rglob("*.wav"))
        audio_files.extend(list(self.golden_samples_dir.rglob("*.flac")))

        for audio_file in audio_files[:100]:  # Limit auf 100 für Performance
            try:
                from backend.file_import import load_audio_file

                _af_res = load_audio_file(str(audio_file), do_carrier_analysis=False)
                if _af_res is None or _af_res.get("error") or _af_res["audio"] is None:
                    continue
                audio, sr = _af_res["audio"], int(_af_res["sr"])

                # Resample wenn nötig
                if sr != self.target_sr:
                    audio = self._resample(audio, sr, self.target_sr)

                # Versuche Medium aus Pfad zu inferieren
                medium_type = self._infer_medium_from_path(audio_file)
                era_type = EraType.UNKNOWN  # Würde aus metadata.json kommen

                samples.append(
                    SyntheticSample(
                        audio=audio,
                        sample_rate=self.target_sr,
                        medium_type=medium_type,
                        era_type=era_type,
                        defects=[],
                        generation="real",
                        source_file=str(audio_file),
                    )
                )
            except Exception as e:
                logger.debug("⚠️  konnte nicht laden %s: %s", audio_file, e)

        return samples

    def _generate_synthetic_medium(self, medium_type: MediaType) -> SyntheticSample:
        """
        Generiert synthetisches Sample für ein spezifisches Medium.
        Nutzt DSP-Emulation für realistische Artefakte.
        """
        # Basis-Audio: Sauberes Signal (Sine + Harmonics oder White Noise)
        duration_sec = random.uniform(5.0, 10.0)
        n_samples = int(duration_sec * self.target_sr)

        # Generiere Test-Signal (Music-like)
        t = np.linspace(0, duration_sec, n_samples)
        fundamental = random.uniform(100, 400)  # Hz

        audio = np.zeros(n_samples)
        for harmonic in range(1, 8):
            amplitude = 1.0 / harmonic  # Harmonische Reihe
            audio += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)

        # Normalisiere
        audio = audio / np.max(np.abs(audio)) * 0.5

        # Medium-spezifische Degradierung (unterstützt alle Typen)
        if medium_type in [
            MediaType.VINYL_LP_STEREO,
            MediaType.VINYL_LP_MONO,
            MediaType.VINYL_45_STEREO,
            MediaType.VINYL,
        ]:
            audio = self._apply_vinyl_artifacts(audio, self.target_sr)
        elif medium_type in [MediaType.TAPE_15IPS, MediaType.TAPE_7_5IPS, MediaType.TAPE]:
            audio = self._apply_tape_artifacts(audio, self.target_sr)
        elif medium_type in [MediaType.CASSETTE_TYPE_I, MediaType.CASSETTE_TYPE_II, MediaType.CASSETTE]:
            audio = self._apply_cassette_artifacts(audio, self.target_sr)
        elif medium_type in [MediaType.CD_STANDARD, MediaType.CD_HDCD, MediaType.CD]:
            audio = self._apply_cd_artifacts(audio, self.target_sr)
        elif medium_type in [MediaType.MP3_320, MediaType.MP3_192, MediaType.AAC_256]:
            audio = self._apply_lossy_artifacts(audio, self.target_sr)
        elif medium_type in [MediaType.HIRES_PCM, MediaType.DIGITAL_NATIVE]:
            audio = self._apply_digital_artifacts(audio, self.target_sr)

        # Inferiere Era aus Medium
        era_type = self._infer_era_from_medium(medium_type)

        return SyntheticSample(
            audio=audio,
            sample_rate=self.target_sr,
            medium_type=medium_type,
            era_type=era_type,
            defects=[],
            generation="synthetic",
        )

    def _generate_synthetic_era(self, era_type: EraType) -> SyntheticSample:
        """
        Generiert synthetisches Sample für eine spezifische Era.
        """
        era_sig = ERA_SIGNATURES.get(era_type)
        if not era_sig:
            raise ValueError(f"Unknown era type: {era_type}")

        # Basis-Audio
        duration_sec = random.uniform(5.0, 10.0)
        n_samples = int(duration_sec * self.target_sr)
        t = np.linspace(0, duration_sec, n_samples)
        fundamental = 220.0  # A3

        audio = np.sin(2 * np.pi * fundamental * t)

        # Era-spezifische Charakteristiken
        audio = self._apply_era_characteristics(audio, self.target_sr, era_sig)

        # Wähle typisches Medium für diese Era
        typical_medium = random.choice(era_sig.typical_media)

        return SyntheticSample(
            audio=audio,
            sample_rate=self.target_sr,
            medium_type=typical_medium,
            era_type=era_type,
            defects=[],
            generation="synthetic",
        )

    def _apply_vinyl_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fügt Vinyl-typische Artefakte hinzu."""
        # 1. Rumble (Low-frequency noise <30 Hz)
        rumble = np.random.randn(len(audio)) * 0.005
        sos_low = scipy_signal.butter(4, 30, btype="low", fs=sr, output="sos")
        rumble = scipy_signal.sosfilt(sos_low, rumble)
        audio = audio + rumble

        # 2. Clicks (Random impulses)
        n_clicks = random.randint(5, 20)
        for _ in range(n_clicks):
            click_pos = random.randint(0, len(audio) - 1)
            audio[click_pos] += random.uniform(0.1, 0.3) * random.choice([-1, 1])

        # 3. Surface noise (High-frequency hiss)
        hiss = np.random.randn(len(audio)) * 0.01
        sos_high = scipy_signal.butter(4, [2000, 10000], btype="band", fs=sr, output="sos")
        hiss = scipy_signal.sosfilt(sos_high, hiss)
        audio = audio + hiss

        return audio

    def _apply_cassette_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fügt Cassette-typische Artefakte hinzu."""
        # 1. Tape Hiss
        hiss = np.random.randn(len(audio)) * 0.02
        audio = audio + hiss

        # 2. High-frequency rolloff
        sos_lpf = scipy_signal.butter(4, 12000, btype="low", fs=sr, output="sos")
        audio = _sosfilt_array(sos_lpf, audio)

        # 3. Wow & Flutter (pitch modulation)
        t = np.arange(len(audio)) / sr
        wow_freq = 0.5  # Hz
        flutter_freq = 5.0  # Hz
        modulation = 1.0 + 0.001 * np.sin(2 * np.pi * wow_freq * t) + 0.0005 * np.sin(2 * np.pi * flutter_freq * t)

        # Simple pitch modulation (approximation)
        audio = audio * modulation

        return audio

    def _apply_tape_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Fügt Reel-to-Reel Tape-typische Artefakte hinzu."""
        # 1. Print-through (echo from adjacent tape layer)
        print_through = np.roll(audio, int(sr * 0.05))  # 50ms delay
        audio = audio + print_through * 0.03

        # 2. Tape hiss (less than cassette)
        hiss = np.random.randn(len(audio)) * 0.008
        audio = audio + hiss

        # 3. Flutter (less than cassette, better quality)
        t = np.arange(len(audio)) / sr
        flutter_freq = 2.0  # Hz
        modulation = 1.0 + 0.0003 * np.sin(2 * np.pi * flutter_freq * t)
        audio = audio * modulation

        return audio

    def _apply_lossy_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simuliert Lossy Compression Artifacts (MP3, AAC)."""
        # 1. High-frequency cutoff (bitrate-dependent)
        cutoff_freq = random.choice([14000, 15000, 16000, 18000])
        sos_lpf = scipy_signal.butter(8, cutoff_freq, btype="low", fs=sr, output="sos")
        audio = _sosfilt_array(sos_lpf, audio)

        # 2. Pre-echo artifacts
        audio = self._add_pre_echo(audio)

        # 3. Quantization noise (subtle)
        quant_noise = np.random.randn(len(audio)) * 0.001
        audio = audio + quant_noise

        return audio

    def _apply_cd_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """CD hat kaum Artefakte - sehr sauber."""
        # Minimales Quantization Noise (16-bit)
        quantization_noise = np.random.randn(len(audio)) * 1e-5
        audio = audio + quantization_noise
        return audio

    def _apply_digital_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simuliert MP3/Lossy Compression Artifacts.
        Vereinfachte Simulation (echte MP3 Encoding ist komplex).
        """
        # High-frequency cutoff (z.B. 16 kHz für 128 kbps MP3)
        cutoff_freq = random.choice([15000, 16000, 18000, 20000])
        sos_lpf = scipy_signal.butter(8, cutoff_freq, btype="low", fs=sr, output="sos")
        audio = _sosfilt_array(sos_lpf, audio)

        # Pre-echo artifacts (vereinfacht)
        audio = self._add_pre_echo(audio)

        return audio

    def _apply_era_characteristics(self, audio: np.ndarray, sr: int, era_sig) -> np.ndarray:
        """Wendet Era-spezifische Charakteristiken an."""
        # 1. Bandwidth limiting
        low_cutoff = era_sig.freq_bandwidth_hz[0]
        high_cutoff = era_sig.freq_bandwidth_hz[1]

        if low_cutoff > 20:
            sos_hp = scipy_signal.butter(2, low_cutoff, btype="high", fs=sr, output="sos")
            audio = _sosfilt_array(sos_hp, audio)

        if high_cutoff < sr / 2:
            sos_lp = scipy_signal.butter(4, high_cutoff, btype="low", fs=sr, output="sos")
            audio = _sosfilt_array(sos_lp, audio)

        # 2. Noise floor
        noise_floor_db = random.uniform(*era_sig.noise_floor_db)
        noise_amplitude = 10 ** (noise_floor_db / 20.0)
        noise = np.random.randn(len(audio)) * noise_amplitude
        audio = audio + noise

        # 3. Dynamic range compression (für Loudness War Eras)
        if era_sig.brick_wall_limiting:
            audio = np.clip(audio, -0.95, 0.95)

        return audio

    def _add_pre_echo(self, audio: np.ndarray) -> np.ndarray:
        """Fügt MP3-typische Pre-Echo artifacts hinzu."""
        # Vereinfachte Simulation
        delayed = np.roll(audio, 100)
        audio = audio + delayed * 0.05
        return audio

    def _resample(self, audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        """Resampled Audio."""
        from scipy.signal import resample_poly

        if audio.ndim == 1:
            return resample_poly(audio, sr_out, sr_in)  # type: ignore[no-any-return]
        else:
            # Stereo
            resampled = np.zeros((int(len(audio) * sr_out / sr_in), 2))
            resampled[:, 0] = resample_poly(audio[:, 0], sr_out, sr_in)
            resampled[:, 1] = resample_poly(audio[:, 1], sr_out, sr_in)
            return resampled  # type: ignore[no-any-return]

    def _infer_medium_from_path(self, path: Path) -> MediaType:
        """Inferiert Medium Type aus Datei-Pfad."""
        path_str = str(path).lower()

        if "vinyl" in path_str or "lp" in path_str:
            return MediaType.VINYL
        elif "cassette" in path_str or "tape" in path_str:
            return MediaType.CASSETTE
        elif "cd" in path_str:
            return MediaType.CD
        elif "digital" in path_str or "wav" in path_str:
            return MediaType.DIGITAL_NATIVE
        else:
            return MediaType.DIGITAL_NATIVE  # Default

    def _infer_era_from_medium(self, medium_type: MediaType) -> EraType:
        """Inferiert typische Era aus Medium."""
        # Vinyl
        if medium_type in [
            MediaType.VINYL_LP_STEREO,
            MediaType.VINYL_LP_MONO,
            MediaType.VINYL_45_STEREO,
            MediaType.VINYL,
        ] or medium_type in [MediaType.TAPE_15IPS, MediaType.TAPE_7_5IPS, MediaType.TAPE]:
            return random.choice([EraType.ERA_1960s, EraType.ERA_1970s, EraType.ERA_1980s])
        # Cassette
        elif medium_type in [MediaType.CASSETTE_TYPE_I, MediaType.CASSETTE_TYPE_II, MediaType.CASSETTE]:
            return random.choice([EraType.ERA_1970s, EraType.ERA_1980s, EraType.ERA_1990s])
        # CD
        elif medium_type in [MediaType.CD_STANDARD, MediaType.CD_HDCD, MediaType.CD]:
            return random.choice([EraType.ERA_1980s, EraType.ERA_1990s, EraType.ERA_2000s])
        # Lossy
        elif medium_type in [MediaType.MP3_320, MediaType.MP3_192, MediaType.AAC_256]:
            return random.choice([EraType.ERA_1990s, EraType.ERA_2000s, EraType.ERA_2010s])
        # Digital
        elif medium_type in [MediaType.HIRES_PCM, MediaType.DIGITAL_NATIVE]:
            return random.choice([EraType.ERA_2000s, EraType.ERA_2010s, EraType.ERA_2020s])
        else:
            return EraType.UNKNOWN

    def save_dataset(self, dataset: dict[str, Any], name: str) -> Path:
        """
        Speichert Dataset als .npz und metadata.json.

        Args:
            dataset: Dataset von generate_*_dataset()
            name: Dataset Name (z.B. "medium_training_v1")

        Returns:
            Path zum gespeicherten Dataset
        """
        output_path = self.output_dir / f"{name}.npz"
        metadata_path = self.output_dir / f"{name}_metadata.json"

        # Extrahiere Audio und Labels
        samples = dataset["samples"]
        audio_list = [s.audio for s in samples]
        labels = [s.medium_type.name for s in samples]

        # Speichere als NPZ
        np.savez_compressed(output_path, audio=audio_list, labels=labels, sample_rate=self.target_sr)

        # Speichere Metadata
        metadata = {k: v for k, v in dataset.items() if k != "samples"}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug("✅ Dataset gespeichert: %s", output_path)
        logger.debug("✅ Metadata gespeichert: %s", metadata_path)

        return output_path
