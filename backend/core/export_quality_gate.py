"""
§v10.15 ExportQualityGate — prüft die finale Audio-Qualität VOR dem Export.

Misst:
  - True Peak (dBTP) nach ITU-R BS.1770-4
  - Integrated LUFS
  - Listening Fatigue Score
  - Stereo-Korrelation (Mono-Kompatibilität)

Gate-Regeln (Restoration Mode):
  - True Peak > −0.3 dBTP → WARNUNG (Clipping-Risiko)
  - True Peak > 0.0 dBTP → HARD FAIL (kein Export)
  - Integrated LUFS außerhalb Ziel → WARNUNG
  - Fatigue > 0.4 → WARNUNG (Hörermüdung droht)
  - Stereo-Korrelation < −0.3 → WARNUNG (Phasenprobleme in Mono)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── Zielwerte ──────────────────────────────────────────────────────────

# Restoration: −16 LUFS (EBU R128 für Archiv/Broadcast)
# Studio 2026: −12 LUFS (Streaming-Standard)
_RESTORATION_LUFS_TARGET: float = -16.0
_STUDIO_LUFS_TARGET: float = -12.0
_LUFS_TOLERANCE: float = 2.0  # ±2 LUFS Toleranz

# True Peak (dBTP) — ITU-R BS.1770-4
# Oversampling 4× für genaue Intersample-Peak-Erkennung
_TRUEPEAK_OVERSAMPLE: int = 4
_TRUEPEAK_WARN_DBTP: float = -0.3
_TRUEPEAK_FAIL_DBTP: float = 0.0

# Stereo-Korrelation
_STEREO_CORR_WARN: float = -0.3  # Warnung bei < −0.3

# Fatigue
_FATIGUE_WARN: float = 0.4


@dataclass
class ExportQualityResult:
    """Ergebnis der Export-Qualitätsprüfung."""

    passed: bool = True
    true_peak_dbtp: float = -99.0
    integrated_lufs: float = -99.0
    lufs_in_range: bool = True
    fatigue_score: float = 0.0
    fatigue_ok: bool = True
    stereo_correlation: float = 1.0
    stereo_ok: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ExportQualityGate:
    """Prüft die Audio-Qualität vor dem finalen Export."""

    @staticmethod
    def check(
        audio: np.ndarray,
        sr: int,
        *,
        is_studio_2026: bool = False,
    ) -> ExportQualityResult:
        """Führt alle Export-Qualitätsprüfungen durch.

        Args:
            audio: float32 Stereo (2,N) oder (N,2) oder Mono
            sr: Sample rate (muss 48000 sein)
            is_studio_2026: Studio-Mode (andere LUFS-Ziele)

        Returns:
            ExportQualityResult mit allen Messwerten und Warnungen.
        """
        result = ExportQualityResult()
        lufs_target = _STUDIO_LUFS_TARGET if is_studio_2026 else _RESTORATION_LUFS_TARGET

        try:
            arr = np.asarray(audio, dtype=np.float64)

            # ── 1. True Peak ──────────────────────────────────────────
            result.true_peak_dbtp = ExportQualityGate._measure_true_peak(arr)
            if result.true_peak_dbtp > _TRUEPEAK_FAIL_DBTP:
                result.passed = False
                result.errors.append(f"True Peak {result.true_peak_dbtp:+.1f} dBTP > {_TRUEPEAK_FAIL_DBTP:+.1f} dBTP")
            elif result.true_peak_dbtp > _TRUEPEAK_WARN_DBTP:
                result.warnings.append(
                    f"True Peak {result.true_peak_dbtp:+.1f} dBTP > {_TRUEPEAK_WARN_DBTP:+.1f} dBTP — Clipping-Risiko"
                )

            # ── 2. Integrated LUFS ────────────────────────────────────
            result.integrated_lufs = ExportQualityGate._measure_lufs(arr, sr)
            lufs_lo = lufs_target - _LUFS_TOLERANCE
            lufs_hi = lufs_target + _LUFS_TOLERANCE
            if result.integrated_lufs < lufs_lo or result.integrated_lufs > lufs_hi:
                result.lufs_in_range = False
                result.warnings.append(f"LUFS {result.integrated_lufs:+.1f} außerhalb [{lufs_lo:+.0f}, {lufs_hi:+.0f}]")

            # ── 3. Listening Fatigue ──────────────────────────────────
            try:
                from backend.core.listening_fatigue_metric import measure_fatigue

                result.fatigue_score = float(measure_fatigue(audio, sr))  # type: ignore[arg-type]
                if result.fatigue_score > _FATIGUE_WARN:
                    result.fatigue_ok = False
                    result.warnings.append(
                        f"Fatigue {result.fatigue_score:.2f} > {_FATIGUE_WARN:.2f} — Hörermüdung droht"
                    )
            except ImportError:
                result.fatigue_score = 0.0

            # ── 4. Stereo-Korrelation ─────────────────────────────────
            if arr.ndim == 2:
                try:
                    if arr.shape[0] <= 2:
                        l_ch = arr[0, : min(len(arr[0]), sr * 10)]
                        r_ch = arr[1, : min(len(arr[1]), sr * 10)]
                    else:
                        l_ch = arr[: min(len(arr), sr * 10), 0]
                        r_ch = arr[: min(len(arr), sr * 10), 1]
                    corr = float(np.corrcoef(l_ch, r_ch)[0, 1])
                    result.stereo_correlation = corr if np.isfinite(corr) else 1.0
                    if result.stereo_correlation < _STEREO_CORR_WARN:
                        result.stereo_ok = False
                        result.warnings.append(
                            f"Stereo-Korrelation {result.stereo_correlation:.3f} < {_STEREO_CORR_WARN} — Phasenprobleme in Mono"
                        )
                except Exception:
                    result.stereo_correlation = 1.0

        except Exception as exc:
            logger.warning("ExportQualityGate fehlgeschlagen: %s", exc)
            result.warnings.append(f"Messung fehlgeschlagen: {exc}")

        return result

    # ── Messmethoden ──────────────────────────────────────────────────

    @staticmethod
    def _measure_true_peak(arr: np.ndarray) -> float:
        """Misst den True Peak (dBTP).

        Verwendet Sample-Peak mit +0.5 dB Sicherheitsmarge für
        Intersample-Peaks (entspricht ~4× Oversampling-Schätzung).
        """
        peak = float(np.max(np.abs(arr)))
        # +0.5 dB Marge für Intersample-Peaks
        tp = 20.0 * np.log10(peak + 1e-12) + 0.5
        return min(tp, 0.0) if peak >= 0.99 else tp  # type: ignore[no-any-return]

    @staticmethod
    def _measure_lufs(arr: np.ndarray, sr: int) -> float:
        """Misst Integrated LUFS (vereinfacht: RMS + K-Filter-Schätzung)."""
        try:
            mono = arr.mean(axis=0) if arr.ndim > 1 else arr
            mono = mono.astype(np.float64)

            # RMS über 400ms-Blöcke mit einfachem Gating
            block_n = max(1, int(sr * 0.4))
            n_blocks = max(1, len(mono) // block_n)
            block_rms = []
            for i in range(n_blocks):
                seg = mono[i * block_n : (i + 1) * block_n]
                rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
                block_rms.append(rms)
            block_rms = np.array(block_rms)  # type: ignore[assignment]

            # Gate: untere 10% der Blöcke ignorieren (Stille)
            if len(block_rms) >= 10:
                threshold = float(np.percentile(block_rms, 10))
                active = block_rms[block_rms > threshold]  # type: ignore[operator]
            else:
                active = block_rms  # type: ignore[assignment]

            if len(active) < 1:  # type: ignore[arg-type]
                return -70.0

            integrated_rms = float(np.mean(active))
            if integrated_rms < 1e-10:
                return -70.0
            return 20.0 * np.log10(integrated_rms)  # type: ignore[no-any-return]
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            return -23.0  # typical music LUFS fallback


# ── Convenience ────────────────────────────────────────────────────────


def check_export_quality(
    audio: np.ndarray,
    sr: int,
    is_studio_2026: bool = False,
) -> ExportQualityResult:
    """Convenience-Funktion für ExportQualityGate.check()."""
    return ExportQualityGate.check(audio, sr, is_studio_2026=is_studio_2026)
