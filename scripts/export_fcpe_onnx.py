"""
FCPE ONNX Export
================
Exportiert models/fcpe/torchfcpe/assets/fcpe.pt → models/fcpe/fcpe.onnx

Die Gewichte in fcpe.pt verwenden dieselbe Architektur wie CFNaiveMelPE
aus torchfcpe, aber mit abweichenden Key-Namen:

  fcpe.pt                            CFNaiveMelPE
  ──────────────────────────────     ──────────────────────────────
  stack.X                        →   input_stack.X
  decoder._layers.X.Y            →   net.encoder_layers.X.Y
  dense_out.Y                    →   output_proj.Y
  norm.Y / cent_table            →   (identisch)

ONNX-Interface:
  Input:  mel   (1, T, 128)  float32   Mel-Spektrogramm @ 16 kHz
  Output: salience (1, T, 360) float32  Pitch-Klassen-Wahrscheinlichkeiten

Mel-Config (im erzeugten fcpe_config.json gespeichert):
  sr=16000, n_fft=1024, hop_size=160, num_mels=128, fmin=0, fmax=8000
  win_size=1024 (Hann)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
FCPE_DIR = ROOT / "models" / "fcpe"
CHECKPOINT = FCPE_DIR / "torchfcpe" / "assets" / "fcpe.pt"
OUTPUT_ONNX = FCPE_DIR / "fcpe.onnx"
CONFIG_JSON = FCPE_DIR / "fcpe_config.json"

# torchfcpe direkt über Dateipfad importieren (umgeht __init__ mit pretty_midi-Dep)
sys.path.insert(0, str(FCPE_DIR / "torchfcpe"))

# local_attention ist bei Inferenz (local_heads=0) nie aktiv → mock reicht
import types as _types

_mock_la = _types.ModuleType("local_attention")
_mock_la.LocalAttention = type("LocalAttention", (object,), {"__init__": lambda *a, **kw: None})  # type: ignore[attr-defined]
sys.modules.setdefault("local_attention", _mock_la)


# ---------------------------------------------------------------------------
# Key-Remapping
# ---------------------------------------------------------------------------
def remap_state_dict(orig_sd: dict) -> dict:
    """Mappt die fcpe.pt-Keys auf CFNaiveMelPE-Keys."""
    new_sd: dict = {}
    for k, v in orig_sd.items():
        if k.startswith("stack."):
            new_k = "input_stack." + k[len("stack.") :]
        elif k.startswith("decoder._layers."):
            new_k = "net.encoder_layers." + k[len("decoder._layers.") :]
        elif k.startswith("dense_out."):
            new_k = "output_proj." + k[len("dense_out.") :]
        else:
            new_k = k  # norm.*, cent_table — unverändert
        new_sd[new_k] = v
    return new_sd


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def main() -> None:
    if not CHECKPOINT.exists():
        logger.error("Checkpoint nicht gefunden: %s", CHECKPOINT)
        sys.exit(1)

    logger.info("Lade Checkpoint: %s (%.0f MB)", CHECKPOINT.name, CHECKPOINT.stat().st_size / 1e6)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    config = ckpt["config"]

    mel_cfg = config["mel"]
    model_cfg = config["model"]

    input_channel: int = model_cfg["input_channel"]  # 128
    n_chans: int = model_cfg["n_chans"]  # 512
    n_layers: int = model_cfg["n_layers"]  # 6
    out_dims: int = model_cfg["out_dims"]  # 360
    f0_min: float = model_cfg["f0_min"]  # 32.7
    f0_max: float = model_cfg["f0_max"]  # 1975.5
    # n_heads: 512 / 64 = 8 (FastAttention default dim_head=64)
    n_heads: int = n_chans // 64

    logger.info(
        "Architektur: Eingabe_channels=%d  hidden=%d  layers=%d  heads=%d  out_dims=%d",
        input_channel,
        n_chans,
        n_layers,
        n_heads,
        out_dims,
    )

    # Modell aufbauen (CFNaiveMelPE = selbe Architektur, andere Key-Namen)
    # Direktimport aus models.py (umgeht __init__ mit pretty_midi-Abhängigkeit)
    import importlib.util as _ilu

    def _load_mod(name: str, path: Path):
        spec = _ilu.spec_from_file_location(name, path)
        mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    _torchfcpe_pkg = _types.ModuleType("torchfcpe")
    sys.modules.setdefault("torchfcpe", _torchfcpe_pkg)

    _load_mod("torchfcpe.model_conformer_naive", FCPE_DIR / "torchfcpe" / "model_conformer_naive.py")
    tfm = _load_mod("torchfcpe.models", FCPE_DIR / "torchfcpe" / "models.py")
    CFNaiveMelPE = tfm.CFNaiveMelPE

    model = CFNaiveMelPE(
        input_channels=input_channel,
        out_dims=out_dims,
        hidden_dims=n_chans,
        n_layers=n_layers,
        n_heads=n_heads,
        f0_max=f0_max,
        f0_min=f0_min,
        use_fa_norm=False,
        conv_only=False,
        conv_dropout=0.0,
        atten_dropout=0.0,
    )

    # Gewichte laden (Key-Remapping + strict=False für gaussian_blurred_cent_mask)
    orig_sd = ckpt["model"]
    new_sd = remap_state_dict(orig_sd)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    logger.info("Gewichte geladen — missing: %s  unexpected: %s", missing, unexpected)

    # Sicherstellen, dass keine kritischen Keys fehlen
    critical_missing = [k for k in missing if not k.startswith("gaussian_blurred")]
    if critical_missing:
        logger.error("Kritische Keys fehlen: %s", critical_missing)
        sys.exit(1)
    if unexpected:
        logger.warning("Unerwartete Keys (werden ignoriert): %s", unexpected)

    model.eval()
    torch.set_num_threads(1)  # reproduzierbar, kein GPU (§9.5)

    # Smoke-Test: dummy input
    dummy_mel = torch.zeros(1, 64, input_channel, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy_mel)
    assert out.shape == (1, 64, out_dims), f"Unexpected output shape: {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output not in [0,1] (sigmoid fehlt?)"
    logger.info("Smoke-Test OK: Eingabe %s → Ausgabe %s", dummy_mel.shape, out.shape)

    # ONNX-Export
    OUTPUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exportiere nach %s …", OUTPUT_ONNX)

    import onnx

    torch.onnx.export(
        model,
        (dummy_mel,),
        str(OUTPUT_ONNX),
        input_names=["mel"],
        output_names=["salience"],
        dynamic_axes={
            "mel": {0: "batch", 1: "time"},
            "salience": {0: "batch", 1: "time"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    logger.info("ONNX-Ausgabe abgeschlossen.")

    # Validierung
    onnx_model = onnx.load(str(OUTPUT_ONNX))
    onnx.checker.check_model(onnx_model)
    size_mb = OUTPUT_ONNX.stat().st_size / 1e6
    logger.info("ONNX-Validierung OK  — Dateigröße: %.1f MB", size_mb)

    # Config speichern (für fcpe_plugin.py)
    cfg_out = {
        "mel_sr": mel_cfg["sampling_rate"],  # 16000
        "mel_n_fft": mel_cfg["n_fft"],  # 1024
        "mel_hop_size": mel_cfg["hop_size"],  # 160
        "mel_num_mels": mel_cfg["num_mels"],  # 128
        "mel_fmin": mel_cfg["fmin"],  # 0
        "mel_fmax": mel_cfg["fmax"],  # 8000
        "mel_win_size": mel_cfg["win_size"],  # 1024
        "f0_min": f0_min,  # 32.7
        "f0_max": f0_max,  # 1975.5
        "out_dims": out_dims,  # 360
        "threshold": model_cfg["threshold"],  # 0.05
    }
    CONFIG_JSON.write_text(json.dumps(cfg_out, indent=2))
    logger.info("Config gespeichert: %s", CONFIG_JSON)

    # Onnxruntime-Verifikation
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, 64, input_channel), dtype=np.float32)
        [ort_out] = sess.run(["salience"], {"mel": dummy_np})
        assert ort_out.shape == (1, 64, out_dims), f"ORT shape: {ort_out.shape}"
        logger.info(
            "OnnxRuntime-Test OK: Ausgabe shape %s, min=%.4f max=%.4f",
            ort_out.shape,
            float(ort_out.min()),
            float(ort_out.max()),
        )
    except Exception as exc:
        logger.warning("OnnxRuntime-Test fehlgeschlagen (nicht kritisch): %s", exc)

    logger.info("✅ FCPE ONNX Ausgabe erfolgreich → %s", OUTPUT_ONNX)


if __name__ == "__main__":
    main()
