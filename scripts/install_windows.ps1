# Aurik 10 — Windows Installation Script
# Run: powershell -ExecutionPolicy Bypass -File scripts/install_windows.ps1

Write-Host "=== Aurik 10 — Windows Installation ===" -ForegroundColor Cyan
Write-Host ""

# Python check
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python nicht gefunden. Installiere von https://python.org (3.10+)" -ForegroundColor Red
    Write-Host "Wähle: Add Python to PATH"
    exit 1
}

$pyver = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python $pyver"

if ([version]$pyver -lt [version]"3.10") {
    Write-Host "Python >= 3.10 benötigt" -ForegroundColor Red
    exit 1
}

# Virtual environment
if (-not (Test-Path ".venv_aurik")) {
    Write-Host "Erstelle Virtual Environment..."
    python -m venv .venv_aurik
}
.\.venv_aurik\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installiere Abhängigkeiten..."
python -m pip install --upgrade pip
pip install -r requirements/requirements_aurik.txt

# GPU detection
Write-Host "GPU-Erkennung..."
$cuda = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($cuda) {
    Write-Host "  CUDA (NVIDIA) gefunden"
    pip install onnxruntime-gpu 2>$null
}

# PortAudio (bundled with sounddevice on Windows via pip)
pip install sounddevice 2>$null

Write-Host ""
Write-Host "=== Installation abgeschlossen ===" -ForegroundColor Green
Write-Host "Start: .\.venv_aurik\Scripts\Activate.ps1 && python Aurik10/main.py"
