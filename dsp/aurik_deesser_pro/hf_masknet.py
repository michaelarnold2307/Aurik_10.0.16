# === Validierungs- und Auto-Abbruch-Logik (SOTA, produktiv) ===
import logging
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

AudioArray: TypeAlias = npt.NDArray[np.float32]


def hf_retention(x: AudioArray, y: AudioArray, sr: int) -> float:
    return float(band_energy(y, sr, 6000, 12000) / (band_energy(x, sr, 6000, 12000) + 1e-9))


def residual_sibilance(x: AudioArray, y: AudioArray, sr: int) -> float:
    return float(band_energy(y, sr, 7000, 11000) / (band_energy(x, sr, 7000, 11000) + 1e-9))


def formant_safety(x: AudioArray, y: AudioArray, sr: int) -> float:
    lf_x = band_energy(x, sr, 300, 3000)
    lf_y = band_energy(y, sr, 300, 3000)
    return float(lf_y / (lf_x + 1e-9))


def burst_sharpness(x: AudioArray, y: AudioArray, sr: int) -> float:
    return float(np.std(y) / (np.std(x) + 1e-9))


def validate(before: AudioArray, after: AudioArray, sr: int) -> dict[str, float]:
    return {
        "hf": hf_retention(before, after, sr),
        "res": residual_sibilance(before, after, sr),
        "formant": formant_safety(before, after, sr),
        "sharp": burst_sharpness(before, after, sr),
    }


def accept(metrics: dict[str, float]) -> bool:
    if metrics["hf"] < 0.75:
        return False
    if metrics["formant"] < 0.98:
        return False
    if metrics["res"] > 0.95:
        return False
    return not metrics["sharp"] < 0.8


def auto_abort(before: AudioArray, after: AudioArray, sr: int) -> dict[str, float]:
    metrics = validate(before, after, sr)
    if metrics["hf"] < 0.75:
        raise RuntimeError("ABORT: HF loss")
    if metrics["formant"] < 0.98:
        raise RuntimeError("ABORT: Formant damage")
    return metrics


def auto_rollback(before: AudioArray, after: AudioArray, sr: int) -> str | None:
    metrics = validate(before, after, sr)
    if metrics["res"] > 0.95:
        return "ROLLBACK_PASS: ineffective"
    if metrics["sharp"] < 0.80:
        return "ROLLBACK_PASS: over-smoothing"
    return None


def process_and_validate(
    before: AudioArray, after: AudioArray, sr: int, ml_used: bool = False
) -> tuple[bool, dict[str, float] | None]:
    try:
        metrics = auto_abort(before, after, sr)
    except RuntimeError as e:
        logger.info(str(e))
        return False, None
    rollback_reason = auto_rollback(before, after, sr)
    if rollback_reason:
        logger.info(rollback_reason)
        return False, metrics
    if ml_used and not accept(metrics):
        logger.info("ML-Kill-Switch: disable ML for this track")
        return False, metrics
    return True, metrics


# === ML-Entscheidungslogik für produktive Musik-Vocal-Pipeline ===


def band_energy(audio: AudioArray, sr: int, f_low: float, f_high: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    mask = (freqs > f_low) & (freqs < f_high)
    return float(np.mean(spec[mask.astype(bool)]))


def residual_sibilance_old(before: AudioArray, after: AudioArray, sr: int) -> float:
    hf_b = band_energy(before, sr, 6000, 12000)
    hf_a = band_energy(after, sr, 6000, 12000)
    return float((hf_a / (hf_b + 1e-9)) > 0.85)


def dsp_at_limit(before: AudioArray, after: AudioArray, sr: int) -> float:
    hf_loss = band_energy(after, sr, 6000, 12000) / band_energy(before, sr, 6000, 12000)
    return float(hf_loss < 0.75)


def low_formant_overlap(stft: AudioArray, band_bins: Any, t: int) -> float:
    lf = np.mean(np.abs(stft[: band_bins.start, t]))
    hf = np.mean(np.abs(stft[band_bins, t]))
    return float(hf / (lf + 1e-9) > 2.5)


def should_use_ml(
    before: AudioArray,
    after: AudioArray,
    stft: AudioArray,
    band_bins: Any,
    t: int,
    profile: Any,
    sr: int,
) -> bool:
    if not profile.allow_ml:
        return False
    if not residual_sibilance(before, after, sr):
        return False
    if not dsp_at_limit(before, after, sr):
        return False
    return bool(low_formant_overlap(stft, band_bins, t))


from collections.abc import Sequence

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

# === HF-Trainingsdaten-Pipeline (STFT → HF-Patches) ===
SR = 48000
N_FFT = 4096
HOP = 512
WIN = "hann"
HF_LOW = 6000
HF_HIGH = 12000
HF_BINS = 64
CTX_FRAMES = 3


def stft_hf(audio: AudioArray, sr: int = SR) -> AudioArray:
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, window=WIN)
    mag = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    hf_mask = (freqs >= HF_LOW) & (freqs <= HF_HIGH)
    hf = mag[hf_mask, :]
    return np.asarray(hf)


def compress_bins(hf: AudioArray, target_bins: int = HF_BINS) -> AudioArray:
    idx = list(np.linspace(0, hf.shape[0], target_bins + 1, dtype=int))
    out = librosa.util.sync(hf, idx, aggregate=np.mean).astype(np.float32)
    return out


def extract_patches(hf: AudioArray, ctx: int = CTX_FRAMES) -> AudioArray:
    patches: list[AudioArray] = []
    for t in range(ctx, hf.shape[1] - ctx):
        patch: AudioArray = hf[:, t - ctx : t + ctx + 1]
        if is_valid_patch(patch):
            patches.append(patch)
    if patches:
        return np.stack(patches).astype(np.float32)
    else:
        return np.empty((0, HF_BINS, 2 * CTX_FRAMES + 1), dtype=np.float32)


def add_sibilance(hf_patch: AudioArray) -> AudioArray:
    hf = hf_patch.copy()
    hf *= np.random.uniform(1.2, 1.5)
    noise = np.random.randn(*hf.shape) * np.random.uniform(0.05, 0.15)
    hf += noise
    hf = np.tanh(hf * np.random.uniform(1.2, 1.4))
    return np.asarray(hf, dtype=np.float32)


def make_target(hf_clean: AudioArray) -> AudioArray:
    return np.clip(hf_clean, 0, np.percentile(hf_clean, 98))


def is_valid_patch(hf: AudioArray) -> bool:
    return bool(np.mean(hf) > 0.01 and np.std(hf) > 0.005)


def build_dataset(audio: AudioArray) -> tuple[np.ndarray, np.ndarray]:
    hf = stft_hf(audio)
    hf = compress_bins(hf)
    clean_patches = extract_patches(hf)
    noisy_patches = np.stack([add_sibilance(p) for p in clean_patches])
    targets = np.stack([make_target(p) for p in clean_patches])
    return noisy_patches, targets


class HFTextureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, audio_files: Sequence[str]) -> None:
        from backend.file_import import load_audio_file

        inputs_list: list[npt.NDArray[np.float32]] = []
        targets_list: list[npt.NDArray[np.float32]] = []
        for path in audio_files:
            _res = load_audio_file(path, target_sr=SR, mono=True)
            audio = np.asarray(_res["audio"], dtype=np.float32)  # type: ignore[index]
            x, y = build_dataset(audio)
            if len(x) > 0:
                inputs_list.append(x)
                targets_list.append(y)
        self.inputs: torch.Tensor = torch.tensor(np.concatenate(inputs_list)).unsqueeze(1)
        self.targets: torch.Tensor = torch.tensor(np.concatenate(targets_list)).unsqueeze(1)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _add_sibilance_torch(hf: torch.Tensor) -> torch.Tensor:
    noise: torch.Tensor = torch.randn_like(hf) * 0.15
    return hf * 1.3 + noise


class HFMaskNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, F, T]
        g: torch.Tensor = self.net(x)
        return 0.75 + 0.25 * g  # HARD RANGE LIMIT


def hf_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1: torch.Tensor = torch.mean(torch.abs(pred - target))
    smooth: torch.Tensor = torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
    penalty: torch.Tensor = torch.mean(torch.relu(pred - 1.0))
    return l1 + 0.1 * smooth + 0.2 * penalty


# Trainings-Setup (Beispiel)
def train_hf_masknet(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    epochs: int = 30,
    lr: float = 1e-4,
    device: str = "cpu",  # §9.5 CPU-only — kein CUDA
) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for hf_in, hf_target in loader:
            hf_in = hf_in.to(device)
            hf_target = hf_target.to(device)
            pred = model(hf_in)
            loss = hf_loss(pred, hf_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


# Export (Beispiel)
def export_scripted(model: nn.Module, path: str = "hf_masknet.pt") -> None:
    scripted = torch.jit.script(model)
    scripted.save(path, _extra_files={})
