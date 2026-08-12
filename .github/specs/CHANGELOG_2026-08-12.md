# Changelog — 2026-08-11/12

## Neue Dateien

| Datei | Zweck |
|-------|-------|
| `backend/core/migraphx_adapter.py` | Python-Wrapper für MIGraphX C-Bridge: `MIGraphXSession`, `is_migraphx_available()`, `create_session_with_fallback()` |
| `backend/core/lib/libmigraphx_bridge.so` | C++ Shared Library: `mgx_load_onnx()`, `mgx_run()`, `mgx_destroy()` via MIGraphX C API + HIP |
| `/tmp/migraphx_bridge_v3.cpp` | Bridge-Quellcode (reproduzierbarer Build) |
| `.github/specs/v10.40_migraphx_gpu_integration.md` | GPU-Architektur-Doku |
| `.github/specs/v10.41_model_compatibility_matrix.md` | 36-Modell-Kompatibilitätsmatrix |

## Geänderte Dateien

| Datei | Änderung | Grund |
|-------|----------|------|
| `backend/core/ml_device_manager.py` | `_detect_migraphx_bridge()` in `_detect_backend()` | MIGraphX-GPU-Erkennung |
| `backend/core/onnx/runtime.py` | MIGraphX-Interception in `__init__` | GPU-Fallback in ONNX-Session |
| `backend/core/ml/session_manager.py` | MIGraphX-Pfad in `_load_session()` | GPU-Fallback in Session-Manager |
| `plugins/miipher_dit_plugin.py` | Flow-Matching-Korrektur: `x + (1-t)·v̂` | **Bug-Fix**: Modell gab Velocity statt Audio aus |
| `plugins/cqtdiff_plugin.py` | TorchScript-Fallback für fehlendes `score_network.onnx` | **Bug-Fix**: Plugin war ohne ONNX-Modell funktionsunfähig |
| `scripts/train_miipher_dit.py` | `default batch_size: 8 → 2` | OOM-Vermeidung auf RX 7900 XTX |

## Gefundene & behobene Bugs

| # | Bug | Datei | Fix |
|---|-----|-------|-----|
| B1 | MIIPHER DiT gab Velocity-Feld statt Audio aus | `miipher_dit_plugin.py:365` | Flow-Matching-Formel: `ŷ = x + (1-t)·v̂` |
| B2 | CQTDiff ONNX-Modell fehlte → DSP-Fallback | `cqtdiff_plugin.py:160` | TorchScript-Direktladung als Fallback |
| B3 | DFN-Training Gradient-Explosion bei batch_size-Wechsel | `train_miipher_dit.py:350` | Kein Resume mit geändertem batch_size |
| B4 | KIM-Modelle an falschen Pfaden | - | `kim_inst.onnx` (aus Backup), `kim_vocal_2.onnx` (aus mdx23c) |
| B5 | PANNs ONNX external-data Symlink | - | ONNX lädt trotz Warnung korrekt |

## Installationen

| Paket | Version | Zweck |
|-------|---------|-------|
| MIGraphX | 2.10.0 (ROCm 6.2) | AMD GPU-Compiler für ONNX→HIP |
| PyTorch ROCm | 2.5.1+rocm6.2 | GPU-Training & -Inferenz |
| onnx2torch | 1.5.15 | ONNX→PyTorch-Konvertierung |
| loguru | 0.7.3 | DFN-Abhängigkeit |


## Modell-Status (alle 53 ONNX geladen)

| Status | Anzahl | Modelle |
|--------|--------|---------|
| ✅ Einsatzbereit | 52 | Alle ONNX + TorchScript |
| 🔴 Fehlt | 0 | — (cqtdiff via TorchScript-Fallback) |
| 🚧 Training | 1 | MIIPHER DiT (batch_size=2, neu starten) |
