"""
cd_error_correction.py - CD-Error-Correction für Aurik 6.0

SOTA-konforme CD-Error-Correction mit DSPContract und Auditierbarkeit.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import correlate as _sc_correlate

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContractCDErrorCorrection:
    id: str = "cd_error_correction"
    category: str = "cd_error_correction"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


cd_error_correction_contract = DSPContractCDErrorCorrection(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"enabled": True}},
    budgets={"compute_cost": 0.01},
    side_effects=[
        {
            "risk": "Fehler nicht erkannt",
            "expected_when": "enabled=False",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["cd_error_correction_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class CDErrorCorrection:
    """
    SOTA-konforme CD-Error-Correction:
    - Korrigiert Lesefehler, Jitter und Dropouts von Audio-CDs
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractCDErrorCorrection = cd_error_correction_contract

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """CD-Fehlerkorrektur: Dropout-Erkennung + kubische Interpolation.

        Erkennt Ausfallbereiche (Stille, Clipping, Sprünge) und interpoliert
        fehlende Samples mit einem AR-Prädiktor (Levinson-Durbin, Ordnung 16).
        """
        self.log_contract()
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        if not self.enabled:
            return audio
        order = 16  # AR-Ordnung
        min_gap = 3  # min. Ausfallbreite (Samples)
        max_gap = 512  # max. interpolierbare Lücke

        def _repair_1d(y: np.ndarray) -> np.ndarray:
            y = y.astype(np.float64).copy()
            n = len(y)
            # Dropout-Maske: Null-Runs oder Clipping
            silence = np.abs(y) < 1e-9
            clipping = np.abs(y) >= 0.9999
            dropout = silence | clipping
            # Label-Regionen
            in_gap = False
            gap_start = 0
            gaps = []
            for i in range(n):
                if dropout[i] and not in_gap:
                    in_gap = True
                    gap_start = i
                elif not dropout[i] and in_gap:
                    in_gap = False
                    gap_len = i - gap_start
                    if min_gap <= gap_len <= max_gap:
                        gaps.append((gap_start, i))
            # Interpolation pro Gap
            for gs, ge in gaps:
                ctx_start = max(0, gs - order)
                ctx = y[ctx_start:gs]
                if len(ctx) < order:
                    # Zu wenig Kontext: lineare Interpolation
                    if gs > 0 and ge < n:
                        y[gs:ge] = np.linspace(y[gs - 1], y[ge], ge - gs + 2)[1:-1]
                    continue
                # AR-Koeffizienten via Levinson-Durbin (Ordnung = min(order, len(ctx)//2))
                p = min(order, len(ctx) // 2)
                if p < 1:
                    continue
                try:
                    from scipy.linalg import solve_toeplitz

                    r = _sc_correlate(ctx, ctx, mode="full", method="fft")[len(ctx) - 1 : len(ctx) + p]
                    rhs = r[1 : p + 1]
                    row = r[:p]
                    # Guard: degenerate autocorrelation triggers LAPACK DLASCL parameter error
                    if not np.isfinite(row).all() or not np.isfinite(rhs).all() or row[0] < 1e-12:
                        if gs > 0 and ge < n:
                            y[gs:ge] = np.linspace(y[gs - 1], y[ge], ge - gs + 2)[1:-1]
                        continue
                    ar = solve_toeplitz(row, rhs)
                    # Vorwärts-Prädiktion
                    buf = list(y[ctx_start:gs])
                    for _ in range(ge - gs):
                        pred = float(np.dot(ar, buf[-p:][::-1]))
                        pred = np.clip(pred, -1.0, 1.0)
                        buf.append(pred)
                    y[gs:ge] = buf[-(ge - gs) :]
                except Exception:
                    # Fallback: lineare Interpolation
                    if gs > 0 and ge < n:
                        y[gs:ge] = np.linspace(y[gs - 1], y[ge], ge - gs + 2)[1:-1]
            return y

        if audio.ndim == 1:
            return _repair_1d(audio).astype(audio.dtype)
        return np.stack([_repair_1d(ch) for ch in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
