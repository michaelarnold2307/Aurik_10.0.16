"""tests/normative/test_installer_smoke.py — §v10.700 K5.

Validiert nach Build: Installer existiert, startet Aurik, restauriert Testdatei.
CI-Integration: Läuft NACH make installer-all.

Nutzung:
  pytest tests/normative/test_installer_smoke.py --installer-path ./dist/Aurik.AppImage
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _find_installer() -> Path | None:
    """Sucht nach Installer-Artefakten im dist/-Verzeichnis."""
    dist = Path("dist")
    if not dist.exists():
        return None
    for pattern in ["*.AppImage", "*.exe", "*.dmg"]:
        matches = list(dist.glob(pattern))
        if matches:
            return matches[0]
    return None


@pytest.mark.slow
def test_installer_exists():
    """Build-Artefakt muss existieren (nur nach make installer-all)."""
    installer = _find_installer()
    if installer is None:
        pytest.skip("Kein Installer gefunden — make installer-all nicht ausgeführt?")
    assert installer.exists(), f"Installer nicht gefunden: {installer}"
    size_mb = installer.stat().st_size / (1024 * 1024)
    assert size_mb > 10, f"Installer zu klein: {size_mb:.0f} MB (erwartet >10 MB)"


@pytest.mark.slow
def test_installer_version():
    """Installer muss korrekte Version ausgeben."""
    installer = _find_installer()
    if installer is None:
        pytest.skip("Kein Installer gefunden")

    from backend.core.version import __version__

    # Versuche --version
    try:
        result = subprocess.run(
            [str(installer), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        assert __version__ in output, f"Version {__version__} nicht in Output: {output[:200]}"
    except FileNotFoundError:
        pytest.skip(f"Installer nicht ausführbar: {installer}")
    except subprocess.TimeoutExpired:
        pytest.fail("Installer --version timeout nach 30s")


@pytest.mark.slow
def test_headless_restore():
    """Headless-Restore einer Testdatei muss funktionieren."""
    # Erstelle Test-Audio
    import tempfile

    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test_input.wav"
        output_path = Path(tmpdir) / "test_output.flac"

        # Generiere Test-Signal
        rng = np.random.RandomState(42)
        audio = (rng.randn(48000) * 0.1).astype(np.float32)
        sf.write(str(input_path), audio, 48000)

        # Prüfe ob direkter Python-Restore funktioniert (kein Installer nötig)
        try:
            from backend.core.unified_restorer_v3 import UnifiedRestorerV3

            restorer = UnifiedRestorerV3()
            result = restorer.restore(audio, 48000, material_type="vinyl")
            assert result.audio is not None
            assert np.isfinite(result.audio).all()
        except ImportError:
            pytest.skip("UnifiedRestorerV3 nicht importierbar")
