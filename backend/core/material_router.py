"""
MaterialRouter: Erkennt Materialtyp (Vinyl, Digital, Tape, Broadcast, Shellac, CD)
und liefert passende Chain-Templates.

Erkennung via spektrale Features (kein ML benötigt):
  - Vinyl:    Charakteristisches Rumpeln (< 30 Hz), Knistern (hochfrequente Impulse),
               starkes Hiss (2–8 kHz)
  - Tape:     WOW/Flutter (Pitch-Modulation bei 0.5–4 Hz), Bandrauschen
  - Shellac:  Sehr hohes Rauschen, starke Hochpasscharakteristik, 78rpm
  - Broadcast: Komprimiert, Bandpass-Charakter (100 Hz – 10 kHz)
  - CD/Digital: Geringes Rauschen, clipping-Artefakte erkennbar
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_instance: MaterialRouter | None = None
_lock = threading.Lock()


def get_material_router() -> MaterialRouter:
    """Gibt zurück: or create MaterialRouter singleton.

    Returns:
        MaterialRouter singleton instance
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MaterialRouter()
    return _instance


def _spectral_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Berechnet schnelle spektrale Features aus einem kurzen Ausschnitt."""
    n = min(len(audio), sr * 5)  # Maximal 5 Sekunden
    x = audio[:n].astype(np.float64)
    if len(x) == 0:
        return {"rumble": 0.0, "hiss": 0.0, "noise_floor": 0.0, "clipping_ratio": 0.0, "centroid": 1000.0}

    fft_len = min(n, 8192)
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / sr)
    power = np.abs(np.fft.rfft(x[:fft_len])) ** 2
    total_power = np.sum(power) + 1e-30

    # Rumpeln (20–80 Hz)
    rumble_mask = (freqs >= 20) & (freqs <= 80)
    rumble = float(np.sum(power[rumble_mask]) / total_power) if rumble_mask.any() else 0.0

    # Hiss: Rauschen bei 4–12 kHz
    hiss_mask = (freqs >= 4000) & (freqs <= 12000)
    hiss = float(np.sum(power[hiss_mask]) / total_power) if hiss_mask.any() else 0.0

    # Noise-Floor: unterste 5 % der Energiedichte
    sorted_power = np.sort(power)
    noise_floor = float(np.mean(sorted_power[: max(1, len(sorted_power) // 20)]) / (np.mean(power) + 1e-30))

    # Clipping-Ratio: Anteil Samples >= 0.98 Peak
    peak: float = float(np.max(np.abs(x)))
    thresh = peak * 0.98
    clipping_ratio = float(np.mean(np.abs(x) >= thresh)) if peak > 1e-6 else 0.0

    # Spektrales Zentroid (Hz)
    centroid = float(np.sum(freqs * power) / np.sum(power)) if total_power > 1e-30 else 1000.0

    return {
        "rumble": rumble,
        "hiss": hiss,
        "noise_floor": noise_floor,
        "clipping_ratio": clipping_ratio,
        "centroid": centroid,
    }


class MaterialRouter:
    """
    Erkennt Materialtyp aus Metadaten und/oder Audiomaterial.
    Gibt passende Chain-Templates zurück.
    """

    def __init__(self) -> None:
        """Initialisiert MaterialRouter."""
        logger.info("MaterialRouter initialisiert")

    def detect_material(
        self,
        audio_metadata: dict[str, Any],
        audio: np.ndarray | None = None,
        sr: int = 44100,
    ) -> str:
        """
        Erkennt den Materialtyp.

        Reihenfolge:
        1. Explizites 'material'-Feld in audio_metadata (höchste Priorität).
        2. 'format'-Feld (z.B. 'vinyl', 'cassette', 'cd').
        3. Spektrale Feature-Analyse falls `audio` übergeben.
        4. Fallback: 'vinyl'.

        Returns
        -------
        Einer von: 'vinyl', 'digital', 'tape', 'shellac', 'broadcast', 'cd'
        """
        # 1. Expliziter Materialtyp in Metadaten
        material = str(audio_metadata.get("material", "")).lower().strip()
        if material in ("vinyl", "digital", "tape", "shellac", "broadcast", "cd", "78rpm"):
            return "shellac" if material == "78rpm" else material

        # 2. Format-Feld
        fmt = str(audio_metadata.get("format", "")).lower().strip()
        for kw, mat in [
            ("vinyl", "vinyl"),
            ("lp", "vinyl"),
            ("ep", "vinyl"),
            ("tape", "tape"),
            ("cassette", "tape"),
            ("reel", "tape"),
            ("shellac", "shellac"),
            ("78", "shellac"),
            ("digital", "digital"),
            ("wav", "digital"),
            ("flac", "digital"),
            ("cd", "cd"),
            ("aiff", "cd"),
            ("broadcast", "broadcast"),
            ("radio", "broadcast"),
            ("am", "broadcast"),
        ]:
            if kw in fmt:
                return mat

        # 3. Spektrale Feature-Analyse
        if audio is not None and len(audio) > 0:
            features = _spectral_features(np.asarray(audio, dtype=np.float64), sr)
            rumble = features["rumble"]
            hiss = features["hiss"]
            noise_floor = features["noise_floor"]
            clipping_ratio = features["clipping_ratio"]
            centroid = features["centroid"]

            # Shellac: extrem hohes Rauschen + hohes Hiss
            if hiss > 0.15 and noise_floor > 0.1:
                return "shellac"
            # Vinyl: Rumpeln + moderates Hiss
            if rumble > 0.02 and hiss > 0.05:
                return "vinyl"
            # Tape: geringes Rumpeln, moderates Rauschen, niedriger Centroid
            if noise_floor > 0.01 and centroid < 3000 and rumble < 0.02:
                return "tape"
            # CD/Digital mit Clipping
            if clipping_ratio > 0.005:
                return "digital"
            # Broadcast: bandgefiltert (centroid zwischen 500–5000 Hz, wenig bass)
            if 500 < centroid < 5000 and rumble < 0.01 and hiss < 0.05:
                return "broadcast"
            # Sauber/digital
            if noise_floor < 0.005:
                return "digital"

        # 4. Fallback
        return audio_metadata.get("material", "vinyl")  # type: ignore[no-any-return]

    def get_chain_template(self, material: str) -> list[str]:
        if material == "vinyl":
            from chain_templates.vinyl_chain import VINYL_CHAIN

            return VINYL_CHAIN  # type: ignore[no-any-return]
        elif material == "digital":
            from chain_templates.digital_chain import DIGITAL_CHAIN

            return DIGITAL_CHAIN  # type: ignore[no-any-return]
        elif material == "tape":
            from chain_templates.tape_chain import TAPE_CHAIN

            return TAPE_CHAIN  # type: ignore[no-any-return]
        elif material == "broadcast":
            from chain_templates.broadcast_chain import BROADCAST_CHAIN

            return BROADCAST_CHAIN  # type: ignore[no-any-return]
        else:
            from chain_templates.vinyl_chain import VINYL_CHAIN

            return VINYL_CHAIN  # type: ignore[no-any-return]
