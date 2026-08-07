"""§NTI (V19) Noise-Textur-Invariante.

Prüft nach NR-Phasen ob das Residualrauschen (entfernter Inhalt = pre − post)
zum erwarteten Defektprofil der Materialklasse passt.

Kanonische Nutzung (UV3 post-phase hook):
    from backend.core.dsp.noise_texture_guard import compute_noise_texture_distance
    dist = compute_noise_texture_distance(residual, material)
    if dist > 0.25:  # → nr_strength × 0.5 (WARNING)
        ...

Konzeptueller Unterschied zu v10.0.0:
    Die frühere _MATERIAL_SLOPE_RANGES modellierte den Slope des in-situ Materiallauschens
    (dunkel, negativ für Shellac). Das Residual (entfernter Inhalt) hat jedoch ein
    ANDERES Profil als das in-situ Rauschen:
    - Shellac-Oberflächen-Kratzen/Knistern = HF-dominant → Residual-Slope positiv (+2..+6)
    - Tape-Hiss = breitbandig mit leichter HF-Betonung → Residual-Slope flach (−3..+1)
    - Whitening = NR entfernt Musik statt Rauschen → Residual-Slope musikähnlich (−12..−5)

    Die neue _MATERIAL_RESIDUAL_SLOPE_RANGES definiert den ERWARTETEN Slope des Residuals
    nach einer KORREKTEN NR. Außerhalb dieser Range = Whitening-Warnung.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Erwarteter Spektral-Slope (dB/Oktave) des RESIDUALS (pre − post) nach korrekter NR.
# Physikalisches Modell: Das Residual = entferntes Rauschprofil.
#   Shellac/Lacquer: Oberflächen-Kratzen/-Knistern = HF-betont → Slope nahe 0 bis stark positiv
#   Tape/Cassette: Bandrauschen breitbandig mit leichter HF-Betonung → flach bis leicht positiv
#   Vinyl: Oberflächen-Rauschen gemischt → leicht negativ bis leicht positiv
#   Digital: Quantisierungsrauschen ≈ flach
#   Whitening-Indikator: Residual musikähnlich (stark negativ, < −5 dB/oct) = NR hat Musik entfernt
# Werte: (min_slope, max_slope)
_MATERIAL_RESIDUAL_SLOPE_RANGES: dict[str, tuple[float, float]] = {
    "shellac": (-2.0, 8.0),  # Kratzen/Knistern HF-betont; Whitening = musikähnlicher LF-Slope
    "wax_cylinder": (-1.5, 9.0),  # Wachszylinder: noch stärker HF-betont
    "lacquer_disc": (-2.0, 7.0),  # Lackscheibe: ähnlich Shellac
    "wire_recording": (-1.0, 8.0),  # Draht-Aufnahme: HF-Rauschen dominant
    "reel_tape": (-4.0, 2.0),  # Bandrauschen: breitbandig, leichter HF-Anteil
    "tape": (-4.0, 2.0),  # Tape (allgemein)
    "vinyl": (-4.5, 1.5),  # Vinyl-Oberfläche: moderate HF-Betonung
    "cassette": (-4.0, 2.0),  # Kassetten-Rauschen: ähnlich Bandmaterial
    "minidisc": (-3.0, 2.5),  # MiniDisc: leichtes HF-Residual
    "cd_digital": (-2.5, 2.5),  # Quantisierungsrauschen: nahezu flach
    "dat": (-2.5, 2.5),  # DAT: ebenfalls flach
    "mp3_low": (-3.5, 2.0),  # MP3-Artefakte: leicht LF-betont
    "mp3_high": (-3.0, 2.0),  # MP3 high: gemischt
    "unknown": (-5.0, 4.0),  # Fallback: breite Toleranz
}

# Rückwärtskompatibilität — Legacy-Alias (nicht mehr primär genutzt)
_MATERIAL_SLOPE_RANGES = _MATERIAL_RESIDUAL_SLOPE_RANGES

# Maximaler Steigungsbereich für Normierungszwecke
_MAX_DEVIATION = 5.0  # dB/oct — 1 dB/oct Abweichung → dist 0.20 (sensitiver für Whitening)


def _estimate_spectral_slope(audio_mono: np.ndarray, sr: int) -> float:
    """Schätzt die Spektralsteigung (dB/oct) via log-log Regression.

    Args:
        audio_mono: Mono-Audio-Signal (float32).
        sr: Sample-Rate.

    Returns:
        Steigung in dB/Oktave (negativ = Roll-Off nach oben).
    """
    n_fft = min(8192, len(audio_mono))
    if n_fft < 256:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio_mono[:n_fft].astype(np.float32), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    # Analyse im Band 100 Hz – 8 kHz
    mask = (freqs >= 100.0) & (freqs <= 8000.0) & (spectrum > 1e-14)
    if mask.sum() < 8:
        return 0.0
    log_f = np.log2(freqs[mask])
    log_p = 10.0 * np.log10(spectrum[mask] + 1e-14)
    try:
        slope = float(np.polyfit(log_f, log_p, 1)[0])
    except Exception:
        slope = 0.0
    return float(np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0))


def compute_noise_texture_distance(
    residual: np.ndarray,
    material: str,
    sr: int = 48000,
) -> float:
    """Berechnet die Distanz zwischen dem Residual (entferntem Rauschen) und dem
    erwarteten Defektprofil der Materialklasse.

    Das Residual = pre_audio − post_audio = der durch NR entfernte Inhalt.
    Sein Spektral-Slope wird gegen _MATERIAL_RESIDUAL_SLOPE_RANGES geprüft.
    Ein Residual-Slope außerhalb der erwarteten Range deutet auf Whitening hin
    (NR hat musikähnlichen Inhalt entfernt statt Rauschen).

    Args:
        residual: Differenz pre_audio − post_audio (entfernter Inhalt). Shape [N] oder [2, N].
        material: Materialklasse (z.B. ``"vinyl"``, ``"shellac"``).
        sr: Sample-Rate. Standardmäßig 48000 Hz (wird nicht assertions-geprüft,
            da auch in Analyse-Kontexten aufrufbar).

    Returns:
        Normierte Distanz [0.0 … 1.0]. 0 = materialkonformes Residual, > 0.25 = Whitening-Warnung.
    """
    try:
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        if residual.ndim == 2:
            residual_mono = residual.mean(axis=0).astype(np.float32)
        else:
            residual_mono = residual.astype(np.float32)

        if len(residual_mono) < 256 or float(np.abs(residual_mono).max()) < 1e-9:
            return 0.0

        slope = _estimate_spectral_slope(residual_mono, sr)

        mat_key = str(material).lower().strip()
        lo, hi = _MATERIAL_RESIDUAL_SLOPE_RANGES.get(mat_key, _MATERIAL_RESIDUAL_SLOPE_RANGES["unknown"])

        if lo <= slope <= hi:
            return 0.0  # materialkonformes Residual

        # Abstand zur nächsten Grenze normieren auf [0, 1]
        dist = max(0.0, lo - slope) if slope < lo else max(0.0, slope - hi)
        normalized = float(np.clip(dist / _MAX_DEVIATION, 0.0, 1.0))
        return float(np.nan_to_num(normalized, nan=0.0))

    except Exception as exc:
        logger.debug("berechnen_noise_texture_distance nicht blockierend: %s", exc)
        return 0.0
