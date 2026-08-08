# Aurik 10 — Installation Guide

> Stand: 10.0.20 | Alle Betriebssysteme

---

## Linux (Ubuntu 22.04+)

### Schritt 1: System vorbereiten
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git libportaudio2
```

### Schritt 2: GPU-Treiber (optional)
```bash
# AMD ROCm
sudo apt install rocm-libs

# NVIDIA CUDA
sudo apt install nvidia-cuda-toolkit
```

### Schritt 3: Aurik installieren
```bash
git clone https://github.com/aurik-audio/Aurik_Standalone.git
cd Aurik_Standalone
bash scripts/install_linux.sh
```

### Schritt 4: Starten
```bash
source .venv_aurik/bin/activate
python3 Aurik10/main.py
```

---

## macOS (12 Monterey+)

### Schritt 1: Homebrew installieren
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Schritt 2: Aurik installieren
```bash
git clone https://github.com/aurik-audio/Aurik_Standalone.git
cd Aurik_Standalone
bash scripts/install_macos.sh
```

### Schritt 3: Starten
```bash
source .venv_aurik/bin/activate
python3 Aurik10/main.py
```

> **Apple Silicon (M1/M2/M3):** Das Skript erkennt den Chip automatisch und installiert `onnxruntime-silicon` für GPU-Beschleunigung.

---

## Windows (10/11)

### Schritt 1: Python installieren
1. https://python.org → Download Python 3.10+
2. **Wichtig:** „Add Python to PATH" ankreuzen
3. Installation abschließen

### Schritt 2: Git installieren
1. https://git-scm.com → Download → Installation mit Defaults

### Schritt 3: Aurik installieren
```powershell
git clone https://github.com/aurik-audio/Aurik_Standalone.git
cd Aurik_Standalone
powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1
```

### Schritt 4: Starten
```powershell
.\.venv_aurik\Scripts\Activate.ps1
python Aurik10/main.py
```

> **NVIDIA GPU:** Treiber von nvidia.com installieren. Das Skript erkennt CUDA automatisch.

---

## Nach der Installation

### Erster Start
Beim ersten Start lädt Aurik automatisch SOTA-Modelle herunter (Hintergrund, nicht blockierend).

### Modelle prüfen
```bash
python3 -c "
from backend.core.model_downloader import get_model_downloader
dl = get_model_downloader()
print(dl.get_download_progress())
"
```

### GPU verifizieren
```bash
python3 scripts/detect_gpu_capabilities.py --json
```

---

## Deinstallation

```bash
# Linux/macOS
rm -rf Aurik_Standalone
rm -rf ~/.aurik

# Windows
# Ordner Aurik_Standalone löschen
# %USERPROFILE%\.aurik löschen
```

---

## Fehlerbehebung

| Problem | Lösung |
|---------|--------|
| `portaudio not found` | Linux: `sudo apt install libportaudio2` |
| `No module named 'PyQt5'` | `pip install PyQt5` |
| `soundfile: OSError` | Linux: `sudo apt install libsndfile1` |
| GPU nicht erkannt | CPU-Fallback automatisch aktiv |
| Model-Download hängt | `export OFFLINE_MODE=true` für Offline-Betrieb |
