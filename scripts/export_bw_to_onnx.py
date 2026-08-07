#!/usr/bin/env python3
"""
Export trainiertes Bandwidth-Reconstructor-Modell nach ONNX.

Voraussetzung: train_bw_reconstructor.py wurde erfolgreich ausgeführt und
die Checkpoint-Datei liegt unter models/bw_reconstructor/best_model.pt.

Usage:
    python scripts/export_bw_to_onnx.py [--checkpoint PATH] [--output PATH]
"""

import argparse
import sys
from pathlib import Path

import torch

# Projekt-Root zum Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class BWReconstructor(torch.nn.Module):
    """Muss exakt mit der Architektur aus train_bw_reconstructor.py übereinstimmen."""

    def __init__(self, in_channels=1, base_channels=24):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = self._conv_block(in_channels, c)
        self.enc2 = self._conv_block(c, c * 2)
        self.enc3 = self._conv_block(c * 2, c * 4)
        self.enc4 = self._conv_block(c * 4, c * 8)

        self.pool = torch.nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(c * 8, c * 8)

        # Decoder
        self.up4 = torch.nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec4 = self._conv_block(c * 8, c * 4)
        self.up3 = torch.nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec3 = self._conv_block(c * 4, c * 2)
        self.up2 = torch.nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec2 = self._conv_block(c * 2, c)
        self.up1 = torch.nn.ConvTranspose2d(c, c, 2, 2)
        self.dec1 = self._conv_block(c * 2, c)

        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(c, c, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c, in_channels, 1),
        )

    def _conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


def export_to_onnx(checkpoint_path: str, output_path: str, dynamic_batch: bool = True):
    """Lädt Checkpoint und exportiert nach ONNX."""

    device = torch.device("cpu")
    model = BWReconstructor(in_channels=1, base_channels=24)

    print(f"Lade Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    # Entferne 'module.'-Präfix falls nötig (DataParallel)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    # Dummy-Input: (batch=1, channels=1, freq_bins=256, time_frames=256)
    dummy_input = torch.randn(1, 1, 256, 256, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch", 2: "freq", 3: "time"},
            "output": {0: "batch", 2: "freq", 3: "time"},
        }

    print(f"Exportiere nach ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    print(f"✅ ONNX-Modell exportiert: {output_path}")

    # Validierung
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX-Validierung bestanden")

    # Größe
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"📦 Modellgröße: {size_mb:.1f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export BW Reconstructor nach ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/bw_reconstructor/best_model.pt",
        help="Pfad zum PyTorch-Checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/bw_reconstructor/bw_reconstructor.onnx",
        help="Pfad für das ONNX-Modell",
    )
    parser.add_argument(
        "--static-shape",
        action="store_true",
        help="Deaktiviere dynamische Batch/Shape-Achsen",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"❌ Checkpoint nicht gefunden: {checkpoint}")
        print("   Bitte zuerst train_bw_reconstructor.py ausführen.")
        sys.exit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        str(checkpoint),
        str(output),
        dynamic_batch=not args.static_shape,
    )


if __name__ == "__main__":
    main()
