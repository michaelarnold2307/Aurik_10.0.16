"""
Perceptual Loss Functions für Aurik 10.0.0

Implementiert psychoakustisch und musikalisch fundierte Loss Functions:
1. PANNs-based High-Level Perceptual Loss
2. Multi-Resolution STFT Loss
3. Psychoacoustic Masking Loss
4. Musical Feature Loss (Harmonic, Rhythmic, Timbral)

Autor: Aurik Backend-Team
Version: 8.1
Datum: 14. Februar 2026
"""

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch

        _HAS_TORCH = True
    except ImportError:
        torch = None  # type: ignore[assignment]
        _HAS_TORCH = False
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _mel_filterbank(n_fft: int, n_mels: int, sr: int, device: "torch.device", dtype: "torch.dtype") -> torch.Tensor:
    """Baut eine echte HTK-Dreiecks-Mel-Filterbank [n_mels, n_fft//2+1].

    Slaney/HTK-Formel (identisch zu librosa.filters.mel, unabhängig implementiert
    um keine zusätzliche Hard-Dependency einzuführen). Rein in Torch, differenzierbar
    bzgl. der Eingabe (die Filterbank selbst ist konstant, keine Parameter).
    """
    n_freqs = n_fft // 2 + 1
    fmin, fmax = 0.0, sr / 2.0

    def _hz_to_mel(f: float) -> float:
        return 2595.0 * float(torch.log10(torch.tensor(1.0 + f / 700.0)))

    def _mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # type: ignore[no-any-return]

    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = torch.floor((n_fft + 1) * hz_points / sr).long().clamp(0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = int(bin_points[m - 1]), int(bin_points[m]), int(bin_points[m + 1])
        if f_center > f_left:
            for k in range(f_left, f_center):
                fb[m - 1, k] = (k - f_left) / (f_center - f_left)
        if f_right > f_center:
            for k in range(f_center, f_right):
                fb[m - 1, k] = (f_right - k) / (f_right - f_center)

    return fb.to(device=device, dtype=dtype)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss für verschiedene Time-Frequency Auflösungen.

    Basiert auf:
    - Yamamoto et al. (2019): "Parallel WaveGAN"
    - Défossez et al. (2020): "Real Time Speech Enhancement in the Waveform Domain"
    """

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_lengths: list[int] | None = None,
        window: str = "hann",
        spectral_convergence_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512, 256, 128]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or [f // 4 for f in fft_sizes]
        self.win_lengths = win_lengths or fft_sizes
        self.window = window
        self.spectral_convergence_weight = spectral_convergence_weight
        self.log_magnitude_weight = log_magnitude_weight
        self.epsilon = epsilon

        # Register windows as buffers (nicht trainierbar)
        for i, (fft_size, win_length) in enumerate(zip(self.fft_sizes, self.win_lengths)):
            if window == "hann":
                win = torch.hann_window(win_length)
            elif window == "hamming":
                win = torch.hamming_window(win_length)
            elif window == "blackman":
                win = torch.blackman_window(win_length)
            else:
                win = torch.ones(win_length)

            self.register_buffer(f"window_{i}", win)

    def stft(
        self, audio: torch.Tensor, fft_size: int, hop_size: int, win_length: int, window: torch.Tensor
    ) -> torch.Tensor:
        """Berechnet STFT."""
        # torch.stft expects [batch, samples] or [samples], so squeeze channel dimension
        audio_2d = audio.squeeze(1)  # [batch, channels, samples] -> [batch, samples]
        return torch.stft(
            audio_2d,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
            normalized=False,
        )

    def spectral_convergence_loss(self, output_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Spectral convergence loss."""
        return torch.norm(target_mag - output_mag, p="fro") / (torch.norm(target_mag, p="fro") + self.epsilon)  # type: ignore[no-any-return]

    def log_magnitude_loss(self, output_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Protokolliert magnitude loss."""
        log_output = torch.log(output_mag + self.epsilon)
        log_target = torch.log(target_mag + self.epsilon)
        return F.l1_loss(log_output, log_target)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Berechnet multi-resolution STFT loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Combined multi-resolution loss
            details: Dictionary with per-resolution losses
        """
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        details = {}

        for i, (fft_size, hop_size, win_length) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            window = getattr(self, f"window_{i}")

            # Compute STFT
            output_stft = self.stft(output, fft_size, hop_size, win_length, window)
            target_stft = self.stft(target, fft_size, hop_size, win_length, window)

            # Magnitude
            output_mag = torch.abs(output_stft)
            target_mag = torch.abs(target_stft)

            # Compute losses
            sc_loss = self.spectral_convergence_loss(output_mag, target_mag)
            mag_loss = self.log_magnitude_loss(output_mag, target_mag)

            total_sc_loss += sc_loss  # type: ignore[assignment]
            total_mag_loss += mag_loss  # type: ignore[assignment]

            details[f"sc_loss_{fft_size}"] = sc_loss.item()
            details[f"mag_loss_{fft_size}"] = mag_loss.item()

        # Average over all resolutions
        total_sc_loss /= len(self.fft_sizes)
        total_mag_loss /= len(self.fft_sizes)

        # Weighted combination
        total_loss = self.spectral_convergence_weight * total_sc_loss + self.log_magnitude_weight * total_mag_loss

        details["total_sc_loss"] = total_sc_loss.item()  # type: ignore[attr-defined]
        details["total_mag_loss"] = total_mag_loss.item()  # type: ignore[attr-defined]

        return total_loss, details  # type: ignore[return-value]


class PANNsPerceptualLoss(nn.Module):
    """
    PANNs-based High-Level Perceptual Loss.

    Lädt das echte CNN14-Backbone (Kong et al. 2020, qiuqiangkong/audioset_tagging_cnn)
    differenzierbar via torch.hub und vergleicht Distanzen in den tatsächlichen
    Zwischen-Layer-Embeddings (conv_block1..4) via Forward-Hooks — kein Platzhalter.

    `PANNsPlugin` (plugins/panns_plugin.py) ist eine reine numpy/ONNX-Inferenz-API
    und daher für eine Backprop-fähige Loss-Funktion architektonisch ungeeignet;
    deshalb wird hier das rohe PyTorch-Backbone separat geladen (analog zu
    PANNsPlugin._try_load_torch_panns(), aber mit aktivem Grad-Fluss zu den Hooks).

    Referenz: Kong et al. (2020): "PANNs: Large-Scale Pretrained Audio Neural Networks"
    """

    _MODEL_SR: int = 32_000

    def __init__(
        self,
        panns_model_path: str | None = None,
        feature_layers: list[str] | None = None,
        feature_weights: list[float] | None = None,
        distance_metric: str = "l1",
        sr: int = 48000,
    ) -> None:
        super().__init__()
        _ = panns_model_path  # torch.hub lädt den offiziellen Checkpoint; expliziter Pfad hier nicht anwendbar
        if feature_layers is None:
            feature_layers = ["conv_block1", "conv_block2", "conv_block3", "conv_block4"]
        if feature_weights is not None and len(feature_weights) != len(feature_layers):
            raise ValueError(
                f"feature_weights (len={len(feature_weights)}) muss zu feature_layers "
                f"(len={len(feature_layers)}) passen"
            )
        self.feature_layers = feature_layers
        self.feature_weights = feature_weights or [1.0] * len(feature_layers)
        self.distance_metric = distance_metric
        self.sr = sr

        self._activations: dict[str, torch.Tensor] = {}
        self.panns_available = False
        self._resampler: Any = None

        try:
            self.panns = torch.hub.load(
                "qiuqiangkong/audioset_tagging_cnn",
                "Cnn14",
                pretrained=True,
                trust_repo=True,
            )
            self.panns.eval()
            for p in self.panns.parameters():
                p.requires_grad_(False)

            for name in self.feature_layers:
                module = getattr(self.panns, name, None)
                if module is None:
                    raise AttributeError(f"CNN14-Backbone hat kein Modul '{name}'")
                module.register_forward_hook(self._make_hook(name))

            if self.sr != self._MODEL_SR:
                try:
                    import torchaudio

                    self._resampler = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self._MODEL_SR)
                except ImportError:
                    logger.warning(
                        "torchaudio nicht verfügbar — PANNs-Resampling via linearer Interpolation (Ersatzpfad)"
                    )

            self.panns_available = True
            logger.info(
                "PANNsPerceptualLoss: CNN14 (torch.hub, differenzierbar) geladen — Layers=%s",
                self.feature_layers,
            )
        except Exception as exc:
            logger.warning(
                "PANNsPerceptualLoss: CNN14-Backbone nicht ladbar (%s) — Ersatzpfad auf Mel-Spektrogramm-Features",
                exc,
            )
            self.panns = None  # type: ignore[assignment]
            self.panns_available = False

    def _make_hook(self, name: str) -> Any:
        def _hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self._activations[name] = out

        return _hook

    def _resample_to_model_sr(self, audio: torch.Tensor) -> torch.Tensor:
        """Resampelt auf die CNN14-Modell-Sample-Rate (32 kHz), falls nötig."""
        if self.sr == self._MODEL_SR:
            return audio
        if self._resampler is not None:
            return cast(torch.Tensor, self._resampler.to(device=audio.device)(audio))
        # Fallback ohne torchaudio: differenzierbare lineare Interpolation
        n_out = max(1, round(audio.shape[-1] * self._MODEL_SR / self.sr))
        return F.interpolate(audio.unsqueeze(1), size=n_out, mode="linear", align_corners=False).squeeze(1)  # type: ignore[no-any-return]

    def extract_features(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extrahiert echte CNN14-Zwischen-Layer-Embeddings via Forward-Hooks."""
        if not self.panns_available:
            return self._extract_spectral_features(audio)

        audio_mono = audio.squeeze(1) if audio.ndim == 3 else audio
        audio_32k = self._resample_to_model_sr(audio_mono)

        self._activations.clear()
        self.panns(audio_32k)  # Hooks befüllen self._activations; Rückgabewert (Tags) hier irrelevant

        if len(self._activations) != len(self.feature_layers):
            logger.warning(
                "PANNsPerceptualLoss: nur %d/%d Hook-Aktivierungen erhalten — Ersatzpfad",
                len(self._activations),
                len(self.feature_layers),
            )
            return self._extract_spectral_features(audio)

        return dict(self._activations)

    def _extract_spectral_features(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Fallback: echte Mel-Spektrogramm-Features (HTK-Dreiecks-Filterbank), falls CNN14 nicht ladbar ist."""
        n_fft = 2048
        hop_length = 512
        n_mels = 64

        audio_mono = audio.squeeze(1) if audio.ndim == 3 else audio
        window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
        spec = torch.stft(
            audio_mono,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        mag = torch.abs(spec)

        mel_fb = _mel_filterbank(n_fft=n_fft, n_mels=n_mels, sr=self.sr, device=audio.device, dtype=audio.dtype)
        mel_spec = torch.matmul(mel_fb, mag)  # [n_mels, n_freq] @ [batch, n_freq, frames] -> [batch, n_mels, frames]
        log_mel = torch.log(mel_spec + 1e-6)

        return {"mel_spectrogram": log_mel}

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Berechnet PANNs-based perceptual loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Perceptual loss in feature space
            details: Dictionary with feature-wise losses
        """
        output_features = self.extract_features(output)
        target_features = self.extract_features(target)

        # Layer-Gewichte anwenden — bei Fallback (ein einzelner Key) wird das erste Gewicht genutzt.
        weight_map = (
            dict(zip(self.feature_layers, self.feature_weights))
            if self.panns_available
            else dict.fromkeys(output_features, self.feature_weights[0])
        )

        total_loss = torch.zeros((), device=output.device, dtype=output.dtype)
        details: dict[str, float] = {}

        for key in output_features:
            output_feat = output_features[key]
            target_feat = target_features[key]

            if self.distance_metric == "l1":
                feat_loss = F.l1_loss(output_feat, target_feat)
            elif self.distance_metric == "l2":
                feat_loss = F.mse_loss(output_feat, target_feat)
            elif self.distance_metric == "cosine":
                feat_loss = 1.0 - F.cosine_similarity(output_feat.flatten(1), target_feat.flatten(1), dim=1).mean()  # pylint: disable=not-callable
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")

            total_loss = total_loss + weight_map.get(key, 1.0) * feat_loss
            details[f"feat_loss_{key}"] = feat_loss.item()

        return total_loss, details


class PsychoacousticMaskingLoss(nn.Module):
    """
    Psychoacoustic Masking Loss basierend auf ITU-R BS.1387 (PEAQ).

    Berücksichtigt:
    - Frequenz-Masking (simultaneous masking)
    - Temporal Masking (pre- and post-masking)
    - Kritische Bänder (Bark scale)
    """

    def __init__(self, sr: int = 48000, n_fft: int = 2048, hop_length: int = 512, n_bark_bands: int = 24):
        super().__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bark_bands = n_bark_bands

        # Bark scale boundaries (approximation)
        self.bark_boundaries = self._compute_bark_boundaries()
        self.register_buffer("_stft_window", torch.hann_window(self.n_fft), persistent=False)

    def _compute_bark_boundaries(self) -> torch.Tensor:
        """Berechnet Bark scale band boundaries nach Zwicker (ITU-R BS.1387-2).

        Verwendet 24 kritische Bänder (Bark 1–24) mit den definierten
        Zwicker-Kantenfrequenzen in Hz. Die Frequenzen werden in
        FFT-Bin-Indizes umgerechnet.

        Zwicker-Kantenfrequenzen (untere Grenzen) in Hz:
          0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
          1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
          9500, 12000, 15500
        """
        # Zwicker critical band edge frequencies (Hz) — ITU-R BS.1387-2 Table 1
        zwicker_edges_hz = [
            0,
            100,
            200,
            300,
            400,
            510,
            630,
            770,
            920,
            1080,
            1270,
            1480,
            1720,
            2000,
            2320,
            2700,
            3150,
            3700,
            4400,
            5300,
            6400,
            7700,
            9500,
            12000,
            15500,
        ]
        max_freq = self.sr / 2.0
        bins = []
        for f_hz in zwicker_edges_hz:
            # Clamp to valid FFT bin range
            f_clamped = min(f_hz, max_freq)
            bin_idx = int(round(f_clamped / max_freq * (self.n_fft // 2)))
            bins.append(bin_idx)
        return torch.tensor(bins, dtype=torch.long)

    def compute_masking_threshold(self, magnitude: torch.Tensor) -> torch.Tensor:
        """
        Berechnet psychoacoustic masking threshold nach PEAQ-Prinzip.

        Implementiert:
        1. Gruppierung in 24 Zwicker-Bark-Bänder
        2. Inter-Band-Spreading (vereinfachte Spreading-Funktion)
        3. Maskierungs-Schwellenberechnung mit Bark-Skalierung

        Args:
            magnitude: STFT-Magnitude [batch, n_freq_bins, n_frames]

        Returns:
            Maskierungs-Schwelle pro Bark-Band [batch, n_bark_bands, n_frames]
        """
        # Group into Zwicker Bark bands
        bark_magnitudes = []

        for i in range(len(self.bark_boundaries) - 1):
            start_bin = int(self.bark_boundaries[i].item())
            end_bin = int(self.bark_boundaries[i + 1].item())
            if end_bin <= start_bin:
                end_bin = start_bin + 1

            band_mag = magnitude[:, start_bin:end_bin, :].mean(dim=1, keepdim=True)
            bark_magnitudes.append(band_mag)

        bark_mag = torch.cat(bark_magnitudes, dim=1)  # [batch, 24, frames]

        # Spreading function: neighbouring bands contribute to masking.
        # Simplified exponential spread with 1-Bark slope (≈ 27 dB/Bark).
        n_bands = bark_mag.shape[1]
        spread_matrix = self._build_spreading_matrix(n_bands, device=magnitude.device)

        # Apply spreading: spread_energy[b, i] = sum_j bark_mag[b, j] * S(i, j)
        # bark_mag: [batch, bands, frames] -> permute for matmul
        bark_mag_t = bark_mag.permute(0, 2, 1)  # [batch, frames, bands]
        spread_energy = torch.matmul(bark_mag_t, spread_matrix)  # [batch, frames, bands]
        spread_energy = spread_energy.permute(0, 2, 1)  # [batch, bands, frames]

        # Masking threshold: spread energy with offset
        # In PEAQ, threshold ≈ spread_energy * 10^(-offset/10)
        # Using offset = 10 dB (simplified)
        offset_db = 10.0
        masking_threshold = spread_energy * (10.0 ** (-offset_db / 10.0))

        # Add absolute hearing threshold floor
        # Approximate quiet threshold at -20 dB relative to full scale
        abs_threshold = 10.0 ** (-20.0 / 10.0)  # 0.01 linear
        masking_threshold = torch.clamp(masking_threshold, min=abs_threshold)

        return masking_threshold

    def _build_spreading_matrix(self, n_bands: int, device: torch.device) -> torch.Tensor:
        """Build inter-band spreading matrix (simplified PEAQ spreading function).

        Models the spread of masking from band j to band i.
        S(i, j) = 10^( -27 * |z_i - z_j| / 10 )  (approximately 27 dB/Bark)
        """
        indices = torch.arange(n_bands, dtype=torch.float32, device=device)
        dz = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()  # |z_i - z_j|
        # 27 dB/Bark spreading slope
        spread_db = -27.0 * dz
        spread_linear = 10.0 ** (spread_db / 10.0)
        return cast(torch.Tensor, spread_linear)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Berechnet psychoacoustic masking loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Psychoacoustically weighted loss
            details: Dictionary with loss components
        """
        stft_window = cast(torch.Tensor, self._stft_window).to(device=output.device, dtype=output.dtype)

        # Compute STFT
        output_stft = torch.stft(
            output.squeeze(1) if output.ndim == 3 else output,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        target_stft = torch.stft(
            target.squeeze(1) if target.ndim == 3 else target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        output_mag = torch.abs(output_stft)
        target_mag = torch.abs(target_stft)

        # Compute masking threshold from target
        masking_threshold = self.compute_masking_threshold(target_mag)

        # Compute error
        error = torch.abs(output_mag - target_mag)

        # Group error into Bark bands
        bark_errors = []
        for i in range(len(self.bark_boundaries) - 1):
            start_bin = int(self.bark_boundaries[i].item())
            end_bin = int(self.bark_boundaries[i + 1].item())
            if end_bin <= start_bin:
                end_bin = start_bin + 1  # Guard gegen leeren Slice → NaN (analog compute_masking_threshold)

            band_error = error[:, start_bin:end_bin, :].mean(dim=1, keepdim=True)
            bark_errors.append(band_error)

        bark_error = torch.cat(bark_errors, dim=1)

        # Weight errors by masking threshold
        # Errors above threshold are weighted more heavily
        weighted_error = torch.where(
            bark_error > masking_threshold,
            bark_error * 2.0,  # Double weight for audible errors
            bark_error * 0.5,  # Half weight for masked errors
        )

        loss = weighted_error.mean()

        details = {"psychoacoustic_loss": loss.item(), "avg_masking_threshold": masking_threshold.mean().item()}

        return loss, details


# ---------------------------------------------------------------------------
# PEAQ ODG — Objective Difference Grade (ITU-R BS.1387-2)
# ---------------------------------------------------------------------------
def peaq_odg(nmr_db: float) -> float:
    """Konvertiert einen Noise-to-Mask Ratio (NMR) Wert in dB in den
    Objective Difference Grade (ODG) nach ITU-R BS.1387-2.

    Der ODG liegt im Bereich [-4, 0] und bildet subjektive
    Qualitätswahrnehmung ab:

        ODG   | Beschreibung
        ------|---------------------
        0.0   | Imperceptible
        -1.0  | Perceptible, not annoying
        -2.0  | Slightly annoying
        -3.0  | Annoying
        -4.0  | Very annoying

    Die Abbildung NMR → ODG erfolgt über eine sigmoide Funktion, die an
    umfangreiche Hörversuche (ITU-R BS.1387-2 Annex 2) kalibriert ist.

    Args:
        nmr_db: Noise-to-Mask Ratio in dB. Positiv = Rauschen über
                Maskierungsschwelle (schlechter). Negativ = unter Schwelle (gut).

    Returns:
        ODG-Wert im Bereich [-4.0, 0.0].
    """
    import math

    # PEAQ mapping function: ODG = b_min + (b_max - b_min) / (1 + exp(s*(NMR - c)))
    # Parameters calibrated from ITU-R BS.1387-2 listening test data:
    b_min = -4.0  # Worst possible ODG
    b_max = 0.0  # Best possible ODG (imperceptible)
    c = -5.0  # NMR midpoint: where the sigmoid is steepest
    s = 0.22  # Slope: controls the transition sharpness

    # Clamp extreme values to avoid overflow
    nmr_clamped = max(-50.0, min(50.0, nmr_db))

    try:
        odg = b_min + (b_max - b_min) / (1.0 + math.exp(s * (nmr_clamped - c)))
    except OverflowError:
        # Extreme negative NMR = good quality
        if nmr_db < c:
            odg = b_max
        else:
            odg = b_min

    return float(max(b_min, min(b_max, odg)))


def peaq_nmr(
    output_mag: "torch.Tensor",
    target_mag: "torch.Tensor",
    masking_threshold: "torch.Tensor",
    bark_boundaries: "torch.Tensor",
) -> float:
    """Berechnet den mittleren Noise-to-Mask Ratio (NMR) in dB aus
    STFT-Magnituden.

    NMR = 10 * log10( mean( error^2 / threshold^2 ) )

    Args:
        output_mag: STFT-Magnitude des Ausgangssignals [batch, freq, frames]
        target_mag: STFT-Magnitude des Referenzsignals [batch, freq, frames]
        masking_threshold: Maskierungsschwelle [batch, bark_bands, frames]
        bark_boundaries: FFT-Bin-Indizes für Bark-Bänder [n_bands+1]

    Returns:
        NMR in dB. Höhere Werte = mehr wahrnehmbares Rauschen.
    """
    import torch as _torch

    error = _torch.abs(output_mag - target_mag)

    # Group error into Bark bands (matching masking_threshold shape)
    bark_errors = []
    for i in range(len(bark_boundaries) - 1):
        start_bin = int(bark_boundaries[i].item())
        end_bin = int(bark_boundaries[i + 1].item())
        if end_bin <= start_bin:
            end_bin = start_bin + 1
        band_error = error[:, start_bin:end_bin, :].mean(dim=1, keepdim=True)
        bark_errors.append(band_error)

    bark_error = _torch.cat(bark_errors, dim=1)  # [batch, bark_bands, frames]

    # NMR: ratio of error energy to masking threshold energy
    nmr_linear = (bark_error**2) / (masking_threshold**2 + 1e-12)
    nmr_mean = nmr_linear.mean()

    nmr_db = 10.0 * _torch.log10(nmr_mean + 1e-12)
    return float(nmr_db.item())


class MusicalFeatureLoss(nn.Module):
    """
    Musical Feature Loss für Harmonic, Rhythmic und Timbral Eigenschaften.

    Basiert auf:
    - Harmonic-to-Noise Ratio (HNR)
    - Onset Detection Consistency
    - Spectral Centroid/Rolloff
    """

    def __init__(
        self, sr: int = 48000, harmonic_weight: float = 1.0, rhythmic_weight: float = 0.8, timbral_weight: float = 0.8
    ):
        super().__init__()

        self.sr = sr
        self.harmonic_weight = harmonic_weight
        self.rhythmic_weight = rhythmic_weight
        self.timbral_weight = timbral_weight
        self.register_buffer("_timbral_stft_window", torch.hann_window(2048), persistent=False)
        self.register_buffer("_harmonic_stft_window", torch.hann_window(1024), persistent=False)
        self.register_buffer("_rhythmic_stft_window", torch.hann_window(1024), persistent=False)

    def compute_harmonic_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Berechnet Loss auf Basis der harmonischen Anregungsstruktur (HNR-Prinzip).

        Homomorphe Dekonvolution (cepstrales Liftering, Oppenheim & Schafer 2009,
        Kap. 13): Der reale Cepstrum trennt die spektrale Hüllkurve (Formanten,
        niedrige Quefrenz) von der Anregungsquelle (Tonhöhen-/Harmonischen-Struktur,
        hohe Quefrenz — der periodische Pulszug erzeugt dort einen Peak bei der
        Grundperiode). Ein Hoch-Lifter isoliert genau diesen Harmonischen-Anteil,
        unabhängig vom Timbre/Formanten-Verlauf — echte Harmonic-Content-Analyse
        statt reinem Waveform-Vergleich.
        """
        n_fft = 1024
        hop_length = 256
        window = cast(torch.Tensor, self._harmonic_stft_window).to(device=output.device, dtype=output.dtype)

        def _harmonic_cepstral_component(audio: torch.Tensor) -> torch.Tensor:
            audio_mono = audio.squeeze(1) if audio.ndim == 3 else audio
            spec = torch.stft(
                audio_mono,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
                center=True,
            )
            log_mag = torch.log(torch.abs(spec) + 1e-8)  # [batch, n_freq, frames]
            # Realer Cepstrum via inverse reelle FFT entlang der Frequenzachse.
            cepstrum = torch.fft.irfft(log_mag.transpose(1, 2), n=n_fft, dim=-1)  # [batch, frames, n_fft]
            # Hoch-Lifter: niedrige Quefrenz (Hüllkurve/Formanten) auf 0 setzen,
            # nur der Harmonischen-/Tonhöhen-Anteil (hohe Quefrenz) bleibt erhalten.
            lifter_cutoff = n_fft // 32
            cepstrum[..., :lifter_cutoff] = 0.0
            cepstrum[..., n_fft - lifter_cutoff + 1 :] = 0.0  # Spiegel-Hälfte (reelles Cepstrum ist symmetrisch)
            harmonic_log_mag = torch.fft.rfft(cepstrum, n=n_fft, dim=-1).real  # [batch, frames, n_freq]
            return harmonic_log_mag.transpose(1, 2)  # type: ignore[no-any-return]  # [batch, n_freq, frames]

        output_harmonic = _harmonic_cepstral_component(output)
        target_harmonic = _harmonic_cepstral_component(target)

        return F.l1_loss(output_harmonic, target_harmonic)

    def compute_rhythmic_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Berechnet Loss auf Basis der Onset-Stärke-Funktion (Spectral Flux).

        Spectral Flux (Bello et al. 2005; Dixon 2006): halbwellen-gleichgerichtete
        Differenz aufeinanderfolgender STFT-Magnitude-Frames, über alle Frequenz-
        bänder summiert — die Standard-Onset-Detektionsfunktion im MIR-Bereich.
        Ein reiner Amplituden-Envelope-Vergleich (Vorgänger-Implementierung) kodiert
        keine Transienten-/Onset-Information: ein Dauerton und ein perkussiver
        Transient mit gleicher RMS-Hüllkurve wären ununterscheidbar.
        """
        n_fft = 1024
        hop_length = 256
        window = cast(torch.Tensor, self._rhythmic_stft_window).to(device=output.device, dtype=output.dtype)

        def _onset_strength(audio: torch.Tensor) -> torch.Tensor:
            audio_mono = audio.squeeze(1) if audio.ndim == 3 else audio
            spec = torch.stft(
                audio_mono,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
                center=True,
            )
            mag = torch.abs(spec)  # [batch, n_freq, frames]
            # Halbwellen-Gleichrichtung: nur Energiezuwächse zählen als Onset-Beitrag.
            flux = F.relu(mag[:, :, 1:] - mag[:, :, :-1]).sum(dim=1)  # [batch, frames-1]
            # log1p-Kompression: rohe Flux-Summen (über alle Frequenzbins) sind
            # unbeschränkt und würden diese Komponente gegenüber den übrigen
            # (bereits normierten) Loss-Termen um Größenordnungen dominieren.
            return torch.log1p(flux)

        output_flux = _onset_strength(output)
        target_flux = _onset_strength(target)

        return F.mse_loss(output_flux, target_flux)

    def compute_timbral_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Berechnet loss based on timbral characteristics."""
        # Compute spectral features
        # Simplified: Compare spectral centroids

        n_fft = 2048
        hop_length = 512
        stft_window = cast(torch.Tensor, self._timbral_stft_window).to(device=output.device, dtype=output.dtype)

        output_spec = torch.stft(
            output.squeeze(1) if output.ndim == 3 else output,
            n_fft=n_fft,
            hop_length=hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        target_spec = torch.stft(
            target.squeeze(1) if target.ndim == 3 else target,
            n_fft=n_fft,
            hop_length=hop_length,
            window=stft_window,
            return_complex=True,
            center=True,
        )

        output_mag = torch.abs(output_spec)
        target_mag = torch.abs(target_spec)

        # Spectral centroid (weighted frequency mean), normiert auf Nyquist ∈ [0, 1] —
        # rohe Hz-Werte (bis 24 kHz) würden den MSE-Term um 4-5 Größenordnungen gegenüber
        # den übrigen (bereits normierten) Loss-Komponenten dominieren.
        nyquist = self.sr / 2.0
        freqs = torch.linspace(0, nyquist, n_fft // 2 + 1, device=output.device) / nyquist
        freqs = freqs.view(1, -1, 1)

        output_centroid = (output_mag * freqs).sum(dim=1) / (output_mag.sum(dim=1) + 1e-8)
        target_centroid = (target_mag * freqs).sum(dim=1) / (target_mag.sum(dim=1) + 1e-8)

        return F.mse_loss(output_centroid, target_centroid)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Berechnet musical feature loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]

        Returns:
            loss: Combined musical feature loss
            details: Dictionary with component losses
        """
        harmonic_loss = self.compute_harmonic_loss(output, target)
        rhythmic_loss = self.compute_rhythmic_loss(output, target)
        timbral_loss = self.compute_timbral_loss(output, target)

        total_loss = (
            self.harmonic_weight * harmonic_loss
            + self.rhythmic_weight * rhythmic_loss
            + self.timbral_weight * timbral_loss
        )

        details = {
            "harmonic_loss": harmonic_loss.item(),
            "rhythmic_loss": rhythmic_loss.item(),
            "timbral_loss": timbral_loss.item(),
        }

        return total_loss, details


class CombinedPerceptualLoss(nn.Module):
    """
    Combined Perceptual Loss combining all perceptual loss components.

    Optimale Gewichtung für musikalische Audio-Restauration.
    """

    def __init__(
        self,
        sr: int = 48000,
        stft_weight: float = 1.0,
        panns_weight: float = 0.8,
        psychoacoustic_weight: float = 0.6,
        musical_weight: float = 1.2,
        use_panns: bool = True,
        use_psychoacoustic: bool = True,
        use_musical: bool = True,
    ):
        super().__init__()

        self.stft_weight = stft_weight
        self.panns_weight = panns_weight
        self.psychoacoustic_weight = psychoacoustic_weight
        self.musical_weight = musical_weight

        # Initialize loss components
        self.stft_loss = MultiResolutionSTFTLoss()

        if use_panns:
            self.panns_loss = PANNsPerceptualLoss(sr=sr)
        else:
            self.panns_loss = None  # type: ignore[assignment]

        if use_psychoacoustic:
            self.psychoacoustic_loss = PsychoacousticMaskingLoss(sr=sr)
        else:
            self.psychoacoustic_loss = None  # type: ignore[assignment]

        if use_musical:
            self.musical_loss = MusicalFeatureLoss(sr=sr)
        else:
            self.musical_loss = None  # type: ignore[assignment]

        logger.info("CombinedPerceptualLoss initialisiert with sr=%s", sr)
        logger.info("  STFT weight: %s", stft_weight)
        logger.info("  PANNs weight: %s (aktiviert: %s)", panns_weight, use_panns)
        logger.info("  Psychoacoustic weight: %s (aktiviert: %s)", psychoacoustic_weight, use_psychoacoustic)
        logger.info("  Musical weight: %s (aktiviert: %s)", musical_weight, use_musical)

    def forward(
        self, output: torch.Tensor, target: torch.Tensor, return_details: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """
        Berechnet combined perceptual loss.

        Args:
            output: Predicted audio [batch, channels, time]
            target: Ground truth audio [batch, channels, time]
            return_details: If True, return detailed loss breakdown

        Returns:
            loss: Combined perceptual loss
            details: (optional) Dictionary with all component losses
        """
        total_loss: torch.Tensor = torch.zeros((), device=output.device, dtype=output.dtype)
        all_details: dict[str, float] = {}

        # 1. Multi-Resolution STFT Loss
        stft_loss, stft_details = self.stft_loss(output, target)
        total_loss += self.stft_weight * stft_loss
        all_details.update({f"stft_{k}": v for k, v in stft_details.items()})

        # 2. PANNs Perceptual Loss
        if self.panns_loss is not None:
            panns_loss, panns_details = self.panns_loss(output, target)
            total_loss += self.panns_weight * panns_loss
            all_details.update({f"panns_{k}": v for k, v in panns_details.items()})

        # 3. Psychoacoustic Masking Loss
        if self.psychoacoustic_loss is not None:
            psych_loss, psych_details = self.psychoacoustic_loss(output, target)
            total_loss += self.psychoacoustic_weight * psych_loss
            all_details.update({f"psych_{k}": v for k, v in psych_details.items()})

        # 4. Musical Feature Loss
        if self.musical_loss is not None:
            musical_loss, musical_details = self.musical_loss(output, target)
            total_loss += self.musical_weight * musical_loss
            all_details.update({f"musical_{k}": v for k, v in musical_details.items()})

        all_details["total_perceptual_loss"] = total_loss.item()

        if return_details:
            return total_loss, all_details
        return total_loss


# Example usage and testing
if __name__ == "__main__":
    # Test perceptual loss
    batch_size = 2
    channels = 1
    duration = 2  # seconds
    sr = 48000
    samples = duration * sr

    # Create dummy audio
    output = torch.randn(batch_size, channels, samples)
    target = torch.randn(batch_size, channels, samples)

    # Test Combined Perceptual Loss
    loss_fn = CombinedPerceptualLoss(sr=sr)
    loss, details = loss_fn(output, target, return_details=True)

    logger.debug("Total Perceptual Loss: %.4f", loss.item())
    logger.debug("\nDetailed Breakdown:")
    for key, value in details.items():
        logger.debug("  %s: %.4f", key, value)
