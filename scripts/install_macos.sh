#!/bin/bash
# Aurik 10 — macOS Installation Script
set -euo pipefail

echo "=== Aurik 10 — macOS Installation ==="
echo ""

# Homebrew check
if ! command -v brew &>/dev/null; then
    echo "Homebrew nicht gefunden. Installiere:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi

# Python
if ! command -v python3 &>/dev/null; then
    echo "Installiere Python 3.10+..."
    brew install python@3.12
fi

PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python $PYVER"

# PortAudio
if ! brew list portaudio &>/dev/null; then
    echo "Installiere PortAudio..."
    brew install portaudio
fi

# GPU detection (Apple Silicon)
if sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -q "Apple"; then
    echo "Apple Silicon (MPS GPU) erkannt"
    PIP_EXTRA="onnxruntime-silicon"
else
    echo "Intel Mac — CPU-Modus"
    PIP_EXTRA=""
fi

# Virtual environment
if [ ! -d ".venv_aurik" ]; then
    echo "Erstelle Virtual Environment..."
    python3 -m venv .venv_aurik
fi
source .venv_aurik/bin/activate

# Install
echo "Installiere Abhängigkeiten..."
pip install --upgrade pip
pip install -r requirements/requirements_aurik.txt

# MPS backend for Apple Silicon
if [ -n "$PIP_EXTRA" ]; then
    pip install "$PIP_EXTRA" 2>/dev/null || echo "  ⚠️ MPS backend optional"
fi

echo ""
echo "=== Installation abgeschlossen ==="
echo "Start: source .venv_aurik/bin/activate && python3 Aurik10/main.py"
