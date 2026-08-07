"""
CQTdiff Score-Network → TorchScript Exporter für Aurik 10.0.0
=========================================================
Exportiert das Score-Netzwerk (UNet-CQT) des CQTdiff-Modells als TorchScript (.pt),
sodass es von cqtdiff_plus_plugin.py mit torch.jit.load geladen werden kann.

Checkpoint: models/cqtdiff/src/models/cqt_weights.pt  (119 MB, EMA step 319999)
Ausgabe:    models/cqtdiff/score_network.pt             (~62 MB, TorchScript)

Das exportierte Modell erwartet:
    Input  "x_noisy"   shape [1, 65536]   float32  (konditioniertes Audio @ 22050 Hz)
    Input  "sigma"     shape [1, 1]       float32  (Rauschpegel σ für EDM-Preconditioning)
    Output              shape [1, 65536]  float32  (Schätzung des clean signal D(x_noisy, σ))

EDM-Preconditioning (Karras et al. 2022, Gleichungen 7):
    c_skip = σ_data² / (σ² + σ_data²)
    c_out  = σ · σ_data / √(σ² + σ_data²)
    c_in   = 1 / √(σ² + σ_data²)
    c_noise= ln(σ) / 4
    D(x,σ) = c_skip·x + c_out · UNet(c_in·x, c_noise)

EMA-Weight-Mapping:
    Das Checkpoint-Format speichert 171 EMA-Tensoren (= Parameter-Count) als Liste.
    10 Resample-Kernel sind registered buffers und werden aus dem Original-Checkpoint übernommen.
Aurik-Spec: §4.4 — CQTdiff (IEEE TASLP 2022) als Primär-Inpainting ≥ 50 ms.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Workspace-Root und Pfade
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).parent.parent
CQTDIFF_DIR = WORKSPACE / "models" / "cqtdiff"
CHECKPOINT = CQTDIFF_DIR / "src" / "models" / "cqt_weights.pt"
OUTPUT_PT = CQTDIFF_DIR / "score_network.pt"

# CQTdiff-Quellcode zum Suchpfad hinzufügen
sys.path.insert(0, str(CQTDIFF_DIR))

# plotly ist in cqtdiff nur für Logging-Visualisierung — hier nicht benötigt
for _mod in ["plotly", "plotly.express", "plotly.graph_objects", "plotly.subplots"]:
    _m = types.ModuleType(_mod)
    _m.__spec__ = None  # type: ignore[assignment]
    sys.modules[_mod] = _m


def _make_args() -> types.SimpleNamespace:
    """Erstellt Konfigurations-Namespace, der das hydra-args-Objekt simuliert."""

    def make_ns(d: dict) -> types.SimpleNamespace:
        ns = types.SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, make_ns(v) if isinstance(v, dict) else v)
        return ns

    return make_ns(
        {
            "sample_rate": 22050,
            "audio_len": 65536,
            "cqt": {"binsoct": 64, "numocts": 7, "use_norm": False},
            "unet_STFT": {"depth": 5},
        }
    )


def _load_ema_weights(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Lädt EMA-Gewichte aus dem Checkpoint.

    Der Checkpoint speichert 171 EMA-Tensoren (in model.parameters()-Reihenfolge)
    sowie 10 Resample-Puffer (registered buffers). Die Puffer werden aus dem
    Original-Checkpoint übernommen.

    Args:
        model:           Frisch initialisiertes Unet_CQT-Modell
        checkpoint_path: Pfad zur .pt-Checkpoint-Datei

    Returns:
        Modell mit geladenen EMA-Gewichten
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ema_weights = ckpt["ema_weights"]  # list[Tensor], len=171
    orig_model_sd = ckpt["model"]  # OrderedDict, len=181

    # Baue State-Dict: Parameter aus EMA, Puffer aus Original
    param_names = [name for name, _ in model.named_parameters()]
    buffer_names = {name for name, _ in model.named_buffers()}

    assert len(param_names) == len(ema_weights), (
        f"EMA-Zähler ({len(ema_weights)}) passt nicht zur Parameteranzahl ({len(param_names)})"
    )

    state_dict: dict = {}
    for name, ema_tensor in zip(param_names, ema_weights):
        state_dict[name] = ema_tensor

    for name in buffer_names:
        state_dict[name] = orig_model_sd[name]

    model.load_state_dict(state_dict, strict=True)
    return model


def main() -> None:
    # ------------------------------------------------------------------
    # Checkpoint-Validierung
    # ------------------------------------------------------------------
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint nicht gefunden: {CHECKPOINT}\nErwartet in: models/cqtdiff/src/models/cqt_weights.pt"
        )

    try:
        import torch
    except ImportError as e:
        raise ImportError(f"torch nicht verfügbar: {e}") from e

    # ------------------------------------------------------------------
    # Modell laden
    # ------------------------------------------------------------------
    print(f"Lade Checkpoint: {CHECKPOINT} (EMA step 319999)")
    args = _make_args()

    from src.models.unet_cqt import Unet_CQT  # type: ignore[import]

    model = Unet_CQT(args, "cpu")
    model.eval()
    model = _load_ema_weights(model, str(CHECKPOINT))
    model.eval()
    print(f"  Modell: {sum(p.numel() for p in model.parameters()):,} Parameter")

    # ------------------------------------------------------------------
    # EDM-Preconditioning-Wrapper
    # ------------------------------------------------------------------
    SIGMA_DATA = 0.057  # Maestro-Trainingsparameter

    class ScoreNetWrapper(torch.nn.Module):
        """Wraps UNet-CQT with EDM preconditioning (Karras et al. 2022, Eq. 7).

        Forward: D(x_noisy, σ) = c_skip·x_noisy + c_out · UNet(c_in·x_noisy, c_noise)
        """

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.sigma_data = SIGMA_DATA

        def forward(self, x_noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            sigma = sigma.view(-1, 1)  # [B, 1]
            sd2 = self.sigma_data**2
            c_skip = sd2 / (sigma**2 + sd2)
            c_out = sigma * self.sigma_data / (sigma**2 + sd2).sqrt()
            c_in = 1.0 / (sigma**2 + sd2).sqrt()
            c_noise = sigma.log() / 4.0  # [B, 1]

            x_in = c_in * x_noisy
            raw_out = self.inner(x_in, c_noise)  # [B, 65536]
            return c_skip * x_noisy + c_out * raw_out  # type: ignore[no-any-return]

    wrapper = ScoreNetWrapper(model)
    wrapper.eval()

    # ------------------------------------------------------------------
    # Forward-Probe
    # ------------------------------------------------------------------
    audio_len = args.audio_len  # 65536
    dummy_x = torch.zeros(1, audio_len)
    dummy_sigma = torch.ones(1, 1) * 1.0

    with torch.no_grad():
        out = wrapper(dummy_x, dummy_sigma)
    assert out.shape == (1, audio_len), f"Forward-Shape fehlerhaft: {out.shape}"
    print(f"  Forward-Test OK — Ausgabe: {list(out.shape)}, Bereich: [{out.min():.4f}, {out.max():.4f}]")

    # ------------------------------------------------------------------
    # TorchScript-Export via torch.jit.trace
    # ------------------------------------------------------------------
    print(f"\nExportiere nach: {OUTPUT_PT}")
    OUTPUT_PT.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_x, dummy_sigma), strict=False)

    # Zweite Probe mit anderen Werten
    dummy_x2 = torch.randn(1, audio_len) * 0.1
    dummy_sigma2 = torch.ones(1, 1) * 0.3
    with torch.no_grad():
        out_trace = traced(dummy_x2, dummy_sigma2)
    assert out_trace.shape == (1, audio_len), f"Trace-Shape fehlerhaft: {out_trace.shape}"

    traced.save(str(OUTPUT_PT))
    size_mb = OUTPUT_PT.stat().st_size / 1e6
    print(f"✓ TorchScript gespeichert — Größe: {size_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Lade-Validierung
    # ------------------------------------------------------------------
    loaded = torch.jit.load(str(OUTPUT_PT), map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        out_val = loaded(dummy_x2, dummy_sigma2)

    import numpy as np

    assert out_val.shape == (1, audio_len), f"Validierungs-Shape fehlerhaft: {out_val.shape}"
    assert np.isfinite(out_val.numpy()).all(), "NaN/Inf in TorchScript-Ausgabe!"

    diff = (out_val - out_trace).abs().max().item()
    assert diff < 1e-4, f"Ausgabedifferenz zu groß: {diff}"

    print(f"✓ Lade-Validierung OK — Max-Diff: {diff:.2e}")
    print(f"\n✓ Exportiert: {OUTPUT_PT}")
    print("  Nächster Schritt: Aurik starten — CQTdiff wird automatisch geladen.")


if __name__ == "__main__":
    main()
