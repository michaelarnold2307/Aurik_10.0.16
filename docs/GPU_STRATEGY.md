# Aurik 10 — GPU-Strategie

> Spec 15 §15.5 | Best-Effort Multi-Backend GPU Support

## Architektur

```
detect_gpu_capabilities.py → gpu_capabilities.json → backend_router.py → ONNX-Provider
```

Aurik wählt automatisch das beste verfügbare ML-Backend. Kein manuelles Konfigurieren nötig. Alle Backends sind "best effort" — fällt eines aus, greift das nächste.

## Unterstützte Backends

| Backend | Provider | Plattform | ONNX-Execution-Provider |
|---------|----------|-----------|------------------------|
| **ROCm** | AMD GPU | Linux | `ROCMExecutionProvider` |
| **CUDA** | NVIDIA GPU | Linux, Windows | `CUDAExecutionProvider` |
| **MPS** | Apple Silicon | macOS 12+ | `CoreMLExecutionProvider` |
| **DirectML** | Any GPU | Windows | `DmlExecutionProvider` |
| **CPU** | Fallback | Alle | `CPUExecutionProvider` |

## Prioritätsreihenfolge

1. ROCm (AMD) — Entwicklungsplattform, primär getestet
2. CUDA (NVIDIA) — BREITESTE GPU-Abdeckung im Profi-Audio-Markt
3. MPS (Apple Silicon) — macOS Standard
4. DirectML (Windows) — Universeller Windows-GPU-Zugriff
5. CPU — Immer verfügbar, keine GPU nötig

## Konfiguration

Alle Einstellungen in `~/.aurik/gpu_capabilities.json`:

```json
{
  "detected_backend": "cuda",
  "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "torch_cuda_available": true,
  "ort_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "fallback_chain": ["cuda", "cpu"]
}
```

## Startup-Sequenz

1. `detect_gpu_capabilities.py` wird beim ersten Start ausgeführt
2. Ergebnis wird in `~/.aurik/gpu_capabilities.json` gecacht
3. `backend_router.py` liest die gecachte Konfiguration
4. ONNX-Sessions werden mit dem erkannten Provider gestartet
5. Bei Fehler: nächster Provider in der Fallback-Kette

## Modell-Kompatibilität

Alle Aurik-Modelle sind ONNX — plattformunabhängig. Kein Modell muss für eine bestimmte GPU konvertiert werden. Der ONNX-Provider übernimmt die hardware-spezifische Optimierung.

## Performance-Erwartungen

| Backend | Relative Geschwindigkeit | RAM-Bedarf |
|---------|:----------------------:|:----------:|
| CUDA | 1.0× (Referenz) | GPU-RAM |
| ROCm | 0.9× | GPU-RAM |
| MPS | 0.7× | Unified Memory |
| DirectML | 0.6× | GPU-RAM |
| CPU | 0.3× | System-RAM |

## Fehlerbehandlung

1. GPU nicht verfügbar → CPU-Fallback (immer)
2. ONNX-Provider nicht installiert → nächster Provider
3. Out-of-Memory → Session-Recycling + CPU-Fallback für diese Session
4. Treiber-Version inkompatibel → Log-Warnung + CPU

## Installation

```bash
# ROCm (AMD)
pip install onnxruntime-rocm

# CUDA (NVIDIA)  
pip install onnxruntime-gpu

# MPS (Apple Silicon)
pip install onnxruntime-silicon  # oder coremltools für Konvertierung

# DirectML (Windows)
pip install onnxruntime-directml

# CPU (immer)
pip install onnxruntime
```
