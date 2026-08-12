#!/usr/bin/env python3
"""
§v10.20: Whisper-gesteuertes Musik-Denoising (Ansatz 1 aus "unkonventionelle Lösungswege").

Architektur:
  Whisper-tiny (39M, frozen) → Feature-Extraktor (hört "durchs Rauschen")
  Lightweight Decoder (2M) → rekonstruiert sauberes Spektrogramm
  RXApproximator → perceptual loss (spectral subtraction + dynamic EQ)

Training: MUSDB18-HQ, 48kHz, komplexe STFT.
"""

from __future__ import annotations

import argparse, random, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

SR = 48_000
N_FFT = 960
HOP = 480
CHUNK_SEC = 4.0
CHUNK_SAMPLES = int(CHUNK_SEC * SR)
SNR_RANGE = (5.0, 20.0)

WHISPER_SR = 16_000  # Whisper native sample rate

CHECKPOINT_DIR = _PROJECT / "models" / "miipher_dit"
BEST_PT = CHECKPOINT_DIR / "whisper_denoiser_best.pt"
LATEST_PT = CHECKPOINT_DIR / "whisper_denoiser_latest.pt"

# ═════════════════════════════════════════════════════════════════════════════
# Whisper Feature Extractor (frozen)
# ═════════════════════════════════════════════════════════════════════════════

class WhisperFeatureExtractor:
    """Extract frozen Whisper encoder features. Input: 16kHz mono audio."""

    def __init__(self, device="cpu", model_name="openai/whisper-tiny"):
        from transformers import WhisperModel
        self.device = device
        print(f"  Loading Whisper {model_name}...", flush=True)
        self.model = WhisperModel.from_pretrained(model_name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_dim = self.model.config.d_model  # 384 for tiny

    @torch.no_grad()
    def __call__(self, audio_16k):
        """audio_16k: [B, T_16k] mono at 16kHz → features [B, T_w, hidden_dim]"""
        # Whisper expects 16kHz mono
        outputs = self.model.encoder(audio_16k.unsqueeze(1))  # [B, 1, T] needed?
        # Actually WhisperModel.encoder expects input_features [B, 80, 3000]
        # Let's use the proper interface
        return outputs.last_hidden_state  # [B, T_w, hidden_dim]


# ═════════════════════════════════════════════════════════════════════════════
# Lightweight Decoder
# ═════════════════════════════════════════════════════════════════════════════

class LightweightDecoder(nn.Module):
    """Project Whisper features + expand to conditioned UNet input."""

    def __init__(self, whisper_dim=384, cond_dim=256):
        super().__init__()
        self.proj = nn.Linear(whisper_dim, cond_dim)
        # Upsample Whisper time-grid (~50 Hz) to STFT time-grid (~100 Hz)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, w_features):
        """w_features: [B, T_w, D] → [B, T_s, cond_dim]"""
        x = self.proj(w_features)  # [B, T_w, cond_dim]
        x = x.transpose(1, 2)  # [B, cond_dim, T_w]
        x = self.upsample(x)  # [B, cond_dim, T_w*2]
        return x.transpose(1, 2)  # [B, T_upsampled, cond_dim]


# ═════════════════════════════════════════════════════════════════════════════
# Conditioned UNet (2D on spectrogram)
# ═════════════════════════════════════════════════════════════════════════════
class ConditionedUNet(nn.Module):
    """Small UNet that denoises STFT conditioned on Whisper features. Uses interpolate for robust shapes."""

    def __init__(self, cond_dim=256):
        super().__init__()
        c = cond_dim
        self.enc1 = nn.Sequential(nn.Conv2d(2, 48, 3, padding=1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(48, 96, 3, padding=1, stride=2), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(96, 192, 3, padding=1, stride=2), nn.GELU())
        self.enc4 = nn.Sequential(nn.Conv2d(192, 256, 3, padding=1, stride=2), nn.GELU())

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + c, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GELU(),
        )

        self.dec4_conv = nn.Sequential(nn.Conv2d(256, 192, 3, padding=1), nn.GELU())
        self.dec3_conv = nn.Sequential(nn.Conv2d(192 + 192, 96, 3, padding=1), nn.GELU())
        self.dec2_conv = nn.Sequential(nn.Conv2d(96 + 96, 48, 3, padding=1), nn.GELU())
        self.dec1_conv = nn.Sequential(nn.Conv2d(48 + 48, 2, 3, padding=1))

    def forward(self, spec, condition):
        """spec: [B, 2, F, T]; condition: [B, T_c, cond_dim]"""
        B, _, n_freq, T = spec.shape

        e1 = self.enc1(spec)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        cond = condition.transpose(1, 2).unsqueeze(2)
        cond = F.interpolate(cond, size=(e4.shape[2], e4.shape[3]), mode="bilinear", align_corners=False)

        b = self.bottleneck(torch.cat([e4, cond], dim=1))

        d4 = self.dec4_conv(b)
        d4_up = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3_conv(torch.cat([d4_up, e3], dim=1))
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2_conv(torch.cat([d3_up, e2], dim=1))
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1_conv(torch.cat([d2_up, e1], dim=1))

        return d1


class RXApproximator(nn.Module):
    """Differentiable approximation of RX-style processing for perceptual loss."""

    def __init__(self, n_bands=4):
        super().__init__()
        self.n_bands = n_bands
        self.gains = nn.Parameter(torch.ones(n_bands))
        self.centroids = nn.Parameter(torch.linspace(0.1, 0.9, n_bands))

    def forward(self, log_mag_spec):
        """log_mag_spec: [B, F, T] in log scale. Returns processed version."""
        B, F_bins, T = log_mag_spec.shape
        device = log_mag_spec.device

        # Frequency positions normalized to [0, 1]
        freq_norm = torch.linspace(0, 1, F_bins, device=device)

        # Compute band gains
        band_gains = self.gains.softmax(dim=0)  # [n_bands]
        # Distance of each freq bin to each centroid
        dist = torch.abs(freq_norm.unsqueeze(0) - self.centroids.unsqueeze(1))  # [n_bands, F_bins]
        # Weighted gain map per frequency bin
        gain_map = (band_gains.unsqueeze(1) * torch.exp(-dist * 10)).sum(dim=0)  # [F_bins]

        # Noise floor estimation (smoothed version)
        noise_floor = F.avg_pool1d(
            log_mag_spec.permute(0, 2, 1).reshape(B * T, 1, F_bins),
            kernel_size=11, stride=1, padding=5
        ).reshape(B, T, F_bins).permute(0, 2, 1)

        alpha = 0.8
        cleaned = log_mag_spec - alpha * noise_floor
        output = cleaned * gain_map.unsqueeze(1)

        return output


# ═════════════════════════════════════════════════════════════════════════════
# Full Model
# ═════════════════════════════════════════════════════════════════════════════

class WhisperDenoiser(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.whisper = WhisperFeatureExtractor(device)
        self.decoder = LightweightDecoder(whisper_dim=384, cond_dim=256)
        self.unet = ConditionedUNet(cond_dim=256)

        # STFT window
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.to(device)

    def extract_whisper_features(self, audio_48k):
        """audio_48k: [B, T_48k] → Whisper features [B, T_w, 384]"""
        # Resample to 16kHz
        B, T = audio_48k.shape
        # Simple approach: use torchaudio if available, else manual
        try:
            import torchaudio
            audio_16k = torchaudio.functional.resample(audio_48k, SR, WHISPER_SR)
        except Exception:
            # Fallback: librosa on CPU
            audio_np = audio_48k.cpu().numpy()
            audio_16k_np = np.stack([librosa.resample(a, orig_sr=SR, target_sr=WHISPER_SR) for a in audio_np])
            audio_16k = torch.from_numpy(audio_16k_np).float().to(self.device)

        # Manual Whisper mel extraction (avoids broken torchvision import)
        audio_16k_padded = audio_16k
        # Whisper needs at least 0.5s for its conv layers
        if audio_16k_padded.shape[1] < WHISPER_SR // 2:
            audio_16k_padded = F.pad(audio_16k_padded, (0, WHISPER_SR // 2 - audio_16k_padded.shape[1]))

        # Compute 80-bin log-Mel spectrogram (same as WhisperProcessor)
        with torch.no_grad():
            if not hasattr(self, '_whisper_mel_fb'):
                # Build mel filterbank once
                n_fft_w = 400  # 25ms at 16kHz
                hop_w = 160    # 10ms at 16kHz
                self._whisper_mel_fb = torchaudio.functional.melscale_fbanks(
                    n_freqs=n_fft_w // 2 + 1, f_min=0, f_max=WHISPER_SR // 2,
                    n_mels=80, sample_rate=WHISPER_SR
                ).to(self.device)
                self._whisper_n_fft = n_fft_w
                self._whisper_hop = hop_w

            spec_w = torch.stft(audio_16k_padded, n_fft=self._whisper_n_fft,
                                hop_length=self._whisper_hop, return_complex=True,
                                window=torch.hann_window(self._whisper_n_fft, device=self.device))
            mag_w = spec_w.abs()  # [B, 201, T_w]
            mel_w = torch.matmul(self._whisper_mel_fb.T.unsqueeze(0), mag_w)  # [1,80,201]@[B,201,T]→[B,80,T]
            log_mel = torch.log1p(mel_w)  # Log scale

        # Normalize to Whisper's expected range (approx [-1, 1], zero-mean)
        log_mel = log_mel - log_mel.mean(dim=(1, 2), keepdim=True)
        log_mel = log_mel / (log_mel.std(dim=(1, 2), keepdim=True) + 1e-5)

        # Pad to 3000 frames (Whisper expects exactly 30s at 10ms hop)
        if log_mel.shape[2] < 3000:
            log_mel = F.pad(log_mel, (0, 3000 - log_mel.shape[2]))

        # Whisper encoder
        with torch.no_grad():
            outputs = self.whisper.model.encoder(input_features=log_mel)
        return outputs.last_hidden_state  # [B, T_w, 384]

    def forward(self, noisy_audio, return_specs=False):
        """noisy_audio: [B, T_48k] → clean_audio: [B, T_48k]"""
        B = noisy_audio.shape[0]

        # 1. STFT
        spec = torch.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP,
                          window=self.window.to(noisy_audio.device), return_complex=True)
        T_s = spec.shape[2]
        F_bins = spec.shape[1]  # 481

        # 2. Whisper features
        w_feat = self.extract_whisper_features(noisy_audio)  # [B, T_w, 384]

        # 3. Decoder: project + upsample to match STFT time dim
        cond = self.decoder(w_feat)  # [B, T_cond, 128]
        # Match time dimension
        if cond.shape[1] < T_s:
            cond = F.pad(cond, (0, 0, 0, T_s - cond.shape[1]))
        else:
            cond = cond[:, :T_s, :]

        # 4. UNet: spec as [B, 2, F, T] (real+imag)
        spec_real_imag = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        enhanced = self.unet(spec_real_imag, cond)  # [B, 2, F, T]
        enhanced_spec = torch.complex(enhanced[:, 0], enhanced[:, 1])  # [B, F, T]

        # 5. iSTFT
        clean_audio = torch.istft(enhanced_spec, n_fft=N_FFT, hop_length=HOP,
                                  window=self.window.to(noisy_audio.device), length=noisy_audio.shape[1])

        if return_specs:
            return clean_audio, enhanced_spec, spec
        return clean_audio

    def compute_loss(self, noisy_audio, clean_audio):
        """AurikLoss: 0.7 x MSE(spec) + 0.3 x BarkLoss (psychoacoustic)."""
        clean_out, pred_spec, noisy_spec = self.forward(noisy_audio, return_specs=True)
        _, _, clean_spec_gt = self.forward(clean_audio, return_specs=True)

        # Multi-band spectral MSE — bands weighted by bin count (no artificial scaling)
        F_bins = pred_spec.shape[1]
        f_low = F_bins // 6       # 0-80 bins (bass)
        f_mid = F_bins // 3       # 80-160 bins (vocals)
        # Each band's MSE is naturally weighted by its bin count
        loss_low = F.mse_loss(pred_spec.real[:, :f_low], clean_spec_gt.real[:, :f_low]) +                    F.mse_loss(pred_spec.imag[:, :f_low], clean_spec_gt.imag[:, :f_low])
        loss_mid = F.mse_loss(pred_spec.real[:, f_low:f_mid], clean_spec_gt.real[:, f_low:f_mid]) +                    F.mse_loss(pred_spec.imag[:, f_low:f_mid], clean_spec_gt.imag[:, f_low:f_mid])
        loss_high = F.mse_loss(pred_spec.real[:, f_mid:], clean_spec_gt.real[:, f_mid:]) +                     F.mse_loss(pred_spec.imag[:, f_mid:], clean_spec_gt.imag[:, f_mid:])
        # Reconstruct full MSE from per-band components (bin-count weighted)
        n_low, n_mid, n_high = f_low, f_mid - f_low, F_bins - f_mid
        loss_mse = (loss_low * n_low + loss_mid * n_mid + loss_high * n_high) / F_bins

        # BarkLoss on log magnitude
        pred_mag = torch.log1p(pred_spec.abs())
        clean_mag = torch.log1p(clean_spec_gt.abs())
        loss_bark = F.l1_loss(
            self._bark_projection(pred_mag),
            self._bark_projection(clean_mag)
        )

        loss = 0.7 * loss_mse + 0.3 * loss_bark
        return loss, loss_mse.item(), loss_bark.item()

    def _bark_projection(self, mag):
        """Psychoacoustic Bark-band projection for perceptual loss."""
        B, F, T = mag.shape
        n_bins = F
        proj = torch.zeros_like(mag)
        bark_freqs = [80, 150, 250, 400, 550, 700, 900, 1100,
                      1350, 1650, 2000, 2400, 2850, 3400, 4000, 4650,
                      5400, 6200, 7050, 8000, 9000, 10000]
        for bf in bark_freqs:
            idx = int((bf / 24000) * n_bins)  # SR/2 = 24000
            if 0 <= idx < n_bins:
                w = min(3, n_bins)
                s, e = max(0, idx - w // 2), min(n_bins, idx + w // 2)
                proj[:, s:e, :] += mag[:, s:e, :] / w
        return proj


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class MusicDenoiseDataset(Dataset):
    def __init__(self, audio_files, noise_files=None):
        self.files = audio_files
        self.noise_files = noise_files or []

    def __len__(self):
        return len(self.files) * 30

    def _load(self, path):
        with sf.SoundFile(str(path)) as snd:
            sr = snd.samplerate
            chunk_native = min(int(4.5 * sr), snd.frames)
            max_start = max(0, snd.frames - chunk_native)
            start_frame = random.randint(0, max_start)
            snd.seek(start_frame)
            y = snd.read(chunk_native, dtype='float32')
            if y.ndim > 1:
                y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        y = y.astype(np.float32)
        if len(y) < CHUNK_SAMPLES:
            y = np.pad(y, (0, CHUNK_SAMPLES - len(y)), mode="reflect")
        else:
            y = y[:CHUNK_SAMPLES]
        return y, start_frame / sr if sr > 0 else 0.0

    def _noise(self, length):
        if self.noise_files and random.random() < 0.5:
            try:
                n, _ = self._load(random.choice(self.noise_files))
                return n / (np.abs(n).max() + np.float32(1e-8))
            except Exception:
                pass
        n = np.random.randn(length).astype(np.float32)
        c = random.choice(["white", "pink", "brown"])
        if c == "pink": n = np.cumsum(n)
        elif c == "brown": n = np.cumsum(np.cumsum(n))
        return n / (np.abs(n).max() + np.float32(1e-8))

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        clean, _ = self._load(self.files[file_idx])
        peak = np.abs(clean).max() + np.float32(1e-8)
        clean = clean / peak

        gain = np.float32(10 ** (random.uniform(-3.0, 3.0) / 20.0))
        clean = clean * gain

        noise = self._noise(CHUNK_SAMPLES)
        snr_db = random.uniform(*SNR_RANGE)
        cr = np.sqrt(np.mean(clean**2) + np.float32(1e-8))
        nr = np.sqrt(np.mean(noise**2) + np.float32(1e-8))
        noise = noise * (cr / np.float32(10**(snr_db/20))) / (nr + np.float32(1e-8))
        degraded = clean + noise

        dp = np.abs(degraded).max() + np.float32(1e-8)
        return {
            "clean": torch.from_numpy((clean / dp).astype(np.float32)),
            "degraded": torch.from_numpy((degraded / dp).astype(np.float32)),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train(epochs=50, batch_size=32, lr=1e-4, steps_per_epoch=200, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    musdb = _PROJECT / "data" / "musdb18hq" / "train"
    corpus = _PROJECT / "corpus"
    vocals = sorted(f for f in musdb.rglob("vocals.wav") if f.is_file())
    instruments = sorted(f for f in musdb.rglob("*.wav") if f.is_file() and "vocals" not in f.stem)
    noise_files = sorted(corpus.rglob("*.wav")) if corpus.is_dir() else []
    noise_files = [f for f in noise_files if "clean" not in f.stem.lower()]
    corpus_clean = sorted(corpus.rglob("*clean*.wav")) if corpus.is_dir() else []
    all_files = vocals + instruments[:len(vocals)] + corpus_clean

    if not all_files:
        print("ERROR: No files found"); return

    n_train = int(0.8 * len(all_files))
    rng = random.Random(42)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    train_files, val_files = shuffled[:n_train], shuffled[n_train:]
    train_ds = MusicDenoiseDataset(train_files, noise_files)
    val_ds = MusicDenoiseDataset(val_files, noise_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, drop_last=True, prefetch_factor=2)

    # Model
    print(f"WhisperDenoiser | Files: {len(all_files)} | Noise: {len(noise_files)}")
    model = WhisperDenoiser(device=device).to(device)
    n_p = sum(p.numel() for p in model.unet.parameters()) / 1e6
    n_p += sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"  Trainable params: {n_p:.2f}M (Whisper 39M frozen)")

    optimizer = torch.optim.AdamW(
        list(model.unet.parameters()) + list(model.decoder.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.unet.load_state_dict(ckpt["unet_state_dict"])
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
        best_val = ckpt.get("val_loss", float("inf"))
        print(f"  Loaded from {resume} (val_loss={best_val:.4f})")

    print(f"Epochs: {epochs} | Batch: {batch_size} | Steps/ep: {steps_per_epoch} | LR: {lr}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        print(f"Epoch {epoch+1}/{epochs} — LR {scheduler.get_last_lr()[0]:.1e} — starting", flush=True)

        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            clean = batch["clean"].to(device)
            noisy = batch["degraded"].to(device)

            optimizer.zero_grad()
            loss, loss_spec, loss_rx = model.compute_loss(noisy, clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            train_loss += loss.item()

            if (step + 1) % 10 == 0:
                e = time.time() - t0
                eta = e / (step + 1) * (steps_per_epoch - step - 1) if step > 0 else 0
                print(f"  Ep {epoch+1:3d}/{epochs} | St {step+1:3d}/{steps_per_epoch} | "
                      f"L {train_loss/(step+1):.4f} (spec {loss_spec:.4f} rx {loss_rx:.4f}) | "
                      f"{e:.0f}s/{eta:.0f}s", flush=True)

        scheduler.step()
        avg_train = train_loss / min(steps_per_epoch, len(train_loader))

        # Validation
        model.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for vb in val_loader:
                if vn >= 20: break
                cv = vb["clean"].to(device)
                nv = vb["degraded"].to(device)
                _, pred_spec, _ = model(nv, return_specs=True)
                _, _, clean_spec = model(cv, return_specs=True)
                val_loss += (F.mse_loss(pred_spec.real, clean_spec.real) +
                             F.mse_loss(pred_spec.imag, clean_spec.imag)).item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        print(f"Ep {epoch+1:3d}/{epochs} | Tr {avg_train:.4f} | Val {avg_val:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.1e} | {time.time()-t0:.0f}s", flush=True)

        torch.save({
            "unet_state_dict": model.unet.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            
            "epoch": epoch + 1,
            "val_loss": avg_val,
            "train_loss": avg_train,
        }, LATEST_PT)
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "unet_state_dict": model.unet.state_dict(),
                "decoder_state_dict": model.decoder.state_dict(),
                
                "epoch": epoch + 1,
                "val_loss": avg_val,
            }, BEST_PT)
            print(f"  >> Best: {best_val:.4f}")

    print(f"\nDone. Best val: {best_val:.4f} | {BEST_PT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Whisper-guided music denoising (§v10.20)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.steps_per_epoch, args.resume)
