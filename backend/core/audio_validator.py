"""Audio-Validator — MAX_AUDIO_BYTES_RAM Konfigurationskonstante.
Spec 08 paragraph 8.1.4. Begrenzt die maximale Audio-Groesse im RAM.

Autor: Aurik 10
"""

from __future__ import annotations

# Maximale Audio-Datenmenge im RAM (4 GB = 4 * 1024**3 Bytes)
# Entspricht ~3.5 Stunden Stereo 48kHz float32
MAX_AUDIO_BYTES_RAM: int = 4 * 1024**3

# Maximale Sample-Rate (Hz)
MAX_SAMPLE_RATE: int = 384000

# Maximale Kanalzahl
MAX_CHANNELS: int = 8


def validate_audio_size(num_samples: int, num_channels: int = 2, bytes_per_sample: int = 4) -> bool:
    """Prueft ob die Audio-Daten ins RAM-Budget passen.

    Args:
        num_samples: Anzahl Samples pro Kanal
        num_channels: Anzahl Kanaele (default 2 = Stereo)
        bytes_per_sample: Bytes pro Sample (default 4 = float32)

    Returns:
        True wenn innerhalb des Budgets
    """
    total_bytes = num_samples * num_channels * bytes_per_sample
    return total_bytes <= MAX_AUDIO_BYTES_RAM


def get_max_duration_s(sample_rate: int, num_channels: int = 2) -> float:
    """Maximale Audio-Dauer in Sekunden bei gegebener Sample-Rate."""
    bytes_per_second = sample_rate * num_channels * 4  # float32
    return MAX_AUDIO_BYTES_RAM / bytes_per_second
