#!/bin/bash
# Aurik 10 — Linux Installation Script
set -euo pipefail

echo "=== Aurik 10 — Linux Installation ==="
echo ""

# Prerequisites check
MISSING=""
for cmd in python3 pip3 git; do
    if ! command -v $cmd &>/dev/null; then
        MISSING="$MISSING $cmd"
    fi
done
if [ -n "$MISSING" ]; then
    echo "Fehlende Pakete: $MISSING"
    echo "Installiere mit: sudo apt install python3 python3-pip python3-venv git"
    exit 1
fi

# Python version check
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PYVER gefunden"
if [ "$(python3 -c "print(int('$PYVER' >= '3.10'))")" != "1" ]; then
    echo "Python >= 3.10 benötigt. Aktuell: $PYVER"
    exit 1
fi

# Audio backend
if ! python3 -c "import sounddevice" 2>/dev/null; then
    echo "Audio-Backend: PortAudio + sounddevice"
    sudo apt-get install -y libportaudio2 portaudio19-dev 2>/dev/null || true
fi

# GPU detection
echo "GPU-Erkennung..."
if command -v rocminfo &>/dev/null; then
    echo "  ROCm (AMD GPU) gefunden"
elif command -v nvidia-smi &>/dev/null; then
    echo "  CUDA (NVIDIA GPU) gefunden"
else
    echo "  CPU-Modus (keine GPU)"
fi

# Virtual environment
if [ ! -d ".venv_aurik" ]; then
    echo "Erstelle Virtual Environment..."
    python3 -m venv .venv_aurik
fi
source .venv_aurik/bin/activate

# Install dependencies
echo "Installiere Abhängigkeiten..."
pip install --upgrade pip
pip install -r requirements/requirements_aurik.txt

# Verify installation
echo ""
echo "=== Installation prüfen ==="
python3 -c "
import numpy; print(f'  numpy {numpy.__version__}')
import scipy; print(f'  scipy {scipy.__version__}')
print('  ✅ Core dependencies OK')
" || echo "  ⚠️ Core dependencies check failed"

python3 -c "import soundfile" 2>/dev/null && echo "  ✅ Audio I/O OK" || echo "  ⚠️ Audio I/O not available"

echo ""
echo "=== Installation abgeschlossen ==="
echo "Start: source .venv_aurik/bin/activate && python3 Aurik10/main.py"
