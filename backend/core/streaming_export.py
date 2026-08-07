"""backend/core/streaming_export.py — §v10.700 Streaming-Export.

Ermöglicht Preview-Audio während der Pipeline noch läuft.
Sobald Phase 1-8 (Defect-Scan + Basis-DSP) durchgelaufen sind (~30s),
wird eine Preview der ersten 30 Sekunden als FLAC generiert.

Architektur:
  1. Pipeline läuft normal
  2. Nach Phase 08: `StreamingExport.generate_preview()` wird aufgerufen
  3. Erste 30s des aktuellen Audio-Zustands werden als FLAC kodiert
  4. Base64-kodiert → via Bridge/API an GUI/CLI auslieferbar

Nutzung:
    exporter = StreamingPreview()

    # Während der Pipeline (nach Phase 08):
    preview_b64 = exporter.generate_preview(audio, sample_rate)

    # Nach der Pipeline:
    full_preview = exporter.generate_preview(audio, sample_rate, duration_s=30.0)
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StreamingPreview:
    """Generiert Audio-Preview während der Pipeline läuft."""

    def __init__(self, preview_duration_s: float = 30.0, target_sr: int = 48000):
        self.preview_duration_s = preview_duration_s
        self.target_sr = target_sr
        self._last_preview_time: float = 0.0
        self._preview_count: int = 0

    def generate_preview(
        self,
        audio: np.ndarray,
        sample_rate: int,
        duration_s: float | None = None,
        format: str = "flac",
    ) -> dict[str, Any]:
        """Generiert eine Preview der ersten N Sekunden.

        Args:
            audio: Audio-Array (beliebige Shape, float32)
            sample_rate: Aktuelle Samplerate
            duration_s: Preview-Dauer (default: self.preview_duration_s)
            format: "flac" oder "wav"

        Returns:
            {"audio_b64": "...", "sample_rate": 48000, "duration_s": 30.0, "format": "flac"}
            oder {} bei Fehler
        """
        dur = duration_s or self.preview_duration_s
        try:
            import soundfile as sf

            # Auf Mono reduzieren für kleine Preview
            if audio.ndim > 1:
                audio_mono = np.mean(audio, axis=-1)
            else:
                audio_mono = audio

            audio_f32 = np.asarray(audio_mono, dtype=np.float32).ravel()

            # Erste N Sekunden
            n_samples = min(int(dur * sample_rate), len(audio_f32))
            preview = audio_f32[:n_samples]

            # Auf Ziel-SR resamplen falls nötig
            if sample_rate != self.target_sr:
                try:
                    import scipy.signal

                    new_len = int(len(preview) * self.target_sr / sample_rate)
                    preview = scipy.signal.resample(preview, new_len)
                except ImportError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

            # Als FLAC/WAV in Memory
            buf = io.BytesIO()
            sf.write(buf, preview, self.target_sr, format=format.upper())
            buf.seek(0)

            audio_b64 = base64.b64encode(buf.read()).decode("ascii")

            self._preview_count += 1
            self._last_preview_time = time.monotonic()

            return {
                "audio_b64": audio_b64,
                "sample_rate": self.target_sr,
                "duration_s": float(len(preview) / self.target_sr),
                "format": format,
                "preview_number": self._preview_count,
            }
        except Exception as e:
            logger.debug("StreamingPreview: Fehler bei Preview-Generierung: %s", e)
            return {}

    def should_generate_new_preview(self, min_interval_s: float = 5.0) -> bool:
        """Prüft ob genug Zeit seit letzter Preview vergangen ist."""
        return (time.monotonic() - self._last_preview_time) >= min_interval_s

    @property
    def preview_count(self) -> int:
        return self._preview_count


# ── Singleton für Pipeline-Integration ──────────────────────────

_streaming_preview: StreamingPreview | None = None


def get_streaming_preview() -> StreamingPreview:
    """Gibt die globale StreamingPreview-Instanz zurück."""
    global _streaming_preview
    if _streaming_preview is None:
        _streaming_preview = StreamingPreview()
    return _streaming_preview
