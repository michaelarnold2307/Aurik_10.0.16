"""Vocoder Chain — BigVGAN-v2 + HiFi-GAN + PGHI-ISTFT Fallback.

Spec 02 §1.5 Schritt 13. Aktiviert wenn PQS-MOS < 4.3 nach Phase-Pipeline.

Kette: vocos_48khz → BigVGAN-v2 → HiFi-GAN → PGHI-ISTFT (Fallback)
VERBOTEN: vocos_mel_spec_24khz.onnx als primäres Modell (§4.4 SOTA-Matrix)
"""
from __future__ import annotations
import logging
import numpy as np

logger = logging.getLogger(__name__)


def activate_vocoder_chain(audio: np.ndarray, sample_rate: int = 48000,
                           pqs_mos: float = 4.5) -> np.ndarray | None:
    """Aktiviert die Vocoder-Kette wenn PQS-MOS unter Schwellwert.

    Args:
        audio: Restauriertes Audio
        sample_rate: Sample-Rate
        pqs_mos: Aktueller PQS-MOS-Wert

    Returns:
        Vocoder-verarbeitetes Audio, oder None wenn nicht aktiviert.
    """
    if pqs_mos >= 4.3:
        return None  # Keine Vocoder-Kette nötig

    logger.info("Vocoder-Kette aktiviert (PQS-MOS %.1f < 4.3)", pqs_mos)
    arr = np.asarray(audio, dtype=np.float32)

    # Stufe 1: BigVGAN-v2 (primär)
    try:
        from plugins.bigvgan_v2_plugin import BigVGANv2Plugin
        bigvgan = BigVGANv2Plugin()
        result = bigvgan.synthesize(arr, sample_rate)
        if result is not None and isinstance(result, np.ndarray) and result.size > 0:
            logger.info("Vocoder-Kette: BigVGAN-v2 erfolgreich")
            return result.astype(np.float32)
    except Exception as e:
        logger.warning("BigVGAN-v2 fehlgeschlagen: %s — Fallback zu HiFi-GAN", e)

    # Stufe 2: HiFi-GAN (sekundär)
    try:
        from plugins.hifigan_plugin import HiFiGANPlugin
        hifi = HiFiGANPlugin()
        result = hifi.synthesize(arr, sample_rate)
        if result is not None and isinstance(result, np.ndarray) and result.size > 0:
            logger.info("Vocoder-Kette: HiFi-GAN Fallback erfolgreich")
            return result.astype(np.float32)
    except Exception as e:
        logger.warning("HiFi-GAN fehlgeschlagen: %s — Fallback zu PGHI-ISTFT", e)

    # Stufe 3: PGHI-ISTFT (deterministischer Letzter-Ausweg)
    try:
        from dsp.pghi import pghi_istft
        result = pghi_istft(arr, sample_rate)
        logger.info("Vocoder-Kette: PGHI-ISTFT Fallback erfolgreich")
        return result.astype(np.float32)
    except Exception as e:
        logger.error("Vocoder-Kette: ALLE Stufen fehlgeschlagen — Original zurueck: %s", e)
        return audio


def is_vocoder_available() -> bool:
    """Prueft ob mindestens ein Vocoder-Backend verfuegbar ist."""
    try:
        from plugins.bigvgan_v2_plugin import BigVGANv2Plugin
        return True
    except ImportError:
        pass
    try:
        from plugins.hifigan_plugin import HiFiGANPlugin
        return True
    except ImportError:
        pass
    return False
