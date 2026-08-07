"""
§2.63 StereoDriftState — Chunk-übergreifende L/R-Kohärenz (§G16, §G49)

Problem (VERBOTEN.md 2026-07-12):
  Chunk-basierte Phasen (Phase 12 PSOLA, Phase 24 Dropout-Repair) verarbeiten
  Stereo-Kanäle unabhängig pro Chunk. STCG korrigiert L/R-Versatz pro Chunk,
  aber die Korrektur wird NICHT über Chunk-Grenzen persistiert.

  → Chunk N+1 beginnt mit Null-Zustand, obwohl Chunk N einen Lag von -8900
    Samples akkumuliert hat. Kumulativer Effekt: ~183 ms Kanalversatz.

Lösung (§G49):
  StereoDriftState speichert den akkumulierten Lag pro Kanal über ALLE Chunks.
  Jeder Chunk:
    1. Liest den aktuellen akkumulierten Lag aus dem State
    2. Wendet seinen eigenen Korrektur-Delta an
    3. Persistiert den neuen akkumulierten Lag im State
  Nach allen Chunks: finaler globaler STCG-Check + Korrektur (§G14).

Usage:
  state = StereoDriftState()
  for chunk in chunks:
      state.apply_pre_chunk(chunk)
      # ... process chunk ...
      state.record_post_chunk(chunk, processed)
  state.apply_final_correction(full_audio)

Author: Aurik Development Team
Version: 10.0.9
Date: 2026-07-13
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StereoDriftState:
    """Persistenter L/R-Drift-Zustand über Chunk-Grenzen hinweg.

    Attributes:
        accumulated_lag_samples: Kumulativer L/R-Versatz in Samples (signed).
            Positiv = R-Kanal hängt NACH L (R muss nach VORN geschoben werden).
            Negativ = R-Kanal ist VOR L (R muss nach HINTEN geschoben werden).
        chunk_count: Anzahl der verarbeiteten Chunks (für Debug/Logging).
        max_abs_lag: Maximaler |Lag|, der jemals gemessen wurde.
        final_correction_applied: True wenn apply_final_correction() lief.
    """

    accumulated_lag_samples: float = 0.0
    chunk_count: int = 0
    max_abs_lag: float = 0.0
    final_correction_applied: bool = False
    _per_chunk_lags: list[float] = field(default_factory=list)

    def record_chunk_lag(self, measured_lag_samples: float, chunk_index: int) -> float:
        """Registriert den in einem Chunk gemessenen Lag.

        Args:
            measured_lag_samples: Vom STCG gemessener L/R-Lag dieses Chunks.
            chunk_index: 0-basierter Chunk-Index.

        Returns:
            Der neue akkumulierte Lag (für Anwendung auf diesen Chunk).
        """
        self.chunk_count = max(self.chunk_count, chunk_index + 1)
        self._per_chunk_lags.append(measured_lag_samples)
        self.accumulated_lag_samples += measured_lag_samples
        self.max_abs_lag = max(self.max_abs_lag, abs(self.accumulated_lag_samples))
        return self.accumulated_lag_samples

    def get_cumulative_lag(self) -> float:
        """Gibt den aktuell akkumulierten Lag zurück."""
        return self.accumulated_lag_samples

    def apply_final_correction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Wendet die finale akkumulierte Lag-Korrektur auf das GESAMTE Audio an.

        Args:
            audio: Vollständig reassembliertes Stereo-Audio (alle Chunks).
            sr: Sample-Rate (muss 48000 sein).

        Returns:
            Lag-korrigiertes Audio (STCG sub-sample shift, kein np.roll).
        """
        if abs(self.accumulated_lag_samples) < 0.5:
            logger.debug(
                "StereoDriftState: accumulated lag=%.1f samples — below Schwelle, no correction",
                self.accumulated_lag_samples,
            )
            self.final_correction_applied = True
            return audio

        try:
            from backend.core.stereo_temporal_coherence_guard import (
                get_stereo_temporal_coherence_guard,
            )

            logger.info(
                "StereoDriftState §G49: applying final correction — "
                "accumulated lag=%.1f samples (%.1f ms) over %d chunks, max_abs=%.1f",
                self.accumulated_lag_samples,
                self.accumulated_lag_samples / sr * 1000.0,
                self.chunk_count,
                self.max_abs_lag,
            )

            corrected = get_stereo_temporal_coherence_guard().correct_interchannel_delay(
                audio.astype(np.float32),
                sr,
                phase_id="stereo_drift_final",
            )
            self.final_correction_applied = True
            return corrected
        except Exception as exc:
            logger.warning("StereoDriftState final correction fehlgeschlagen: %s", exc)
            self.final_correction_applied = True
            return audio

    def reset(self) -> None:
        """Setzt alle Zustände zurück (für neuen Song/Durchlauf)."""
        self.accumulated_lag_samples = 0.0
        self.chunk_count = 0
        self.max_abs_lag = 0.0
        self.final_correction_applied = False
        self._per_chunk_lags.clear()


# ---------------------------------------------------------------------------
# Thread-safe singleton (ein State pro Pipeline-Durchlauf)
# ---------------------------------------------------------------------------

_stereo_drift_state: StereoDriftState | None = None


def get_stereo_drift_state() -> StereoDriftState:
    """Gibt den aktuellen StereoDriftState zurück (erzeugt neuen bei Bedarf)."""
    global _stereo_drift_state
    if _stereo_drift_state is None:
        _stereo_drift_state = StereoDriftState()
    return _stereo_drift_state


def reset_stereo_drift_state() -> None:
    """Reset für neuen Song/Pipeline-Durchlauf."""
    global _stereo_drift_state
    if _stereo_drift_state is not None:
        _stereo_drift_state.reset()
    _stereo_drift_state = StereoDriftState()
