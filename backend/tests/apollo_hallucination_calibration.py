"""§v10.303.22 Hallucination-Guard-Kalibrierung für Phase 0.

Kalibriert die spectral_novelty-Schwellen für Apollo, DeepFilterNet v3 und
Resemble Enhance anhand eines Batch von echten Importsongs.

Usage:
  python backend/tests/apollo_hallucination_calibration.py --dir /path/to/songs/

Output:
  reports/phase0_calibration_YYYY-MM-DD.json mit empfohlenen Schwellwerten.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase0_calibration")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Spectral Novelty (kopiert aus apollo_phase0_integration) ────────────


def spectral_novelty(original: np.ndarray, processed: np.ndarray, sr: int = 48000) -> float:
    _a = original if original.ndim == 1 else np.mean(original, axis=0)
    _b = processed if processed.ndim == 1 else np.mean(processed, axis=0)
    n_fft, hop = 2048, 512
    _min_len = min(len(_a), len(_b))
    if _min_len < n_fft:
        return 0.0
    _a, _b = _a[:_min_len], _b[:_min_len]
    spec_a = np.abs(np.stack([np.fft.rfft(_a[i : i + n_fft]) for i in range(0, _min_len - n_fft, hop)]))
    spec_b = np.abs(np.stack([np.fft.rfft(_b[i : i + n_fft]) for i in range(0, _min_len - n_fft, hop)]))
    _n = min(spec_a.shape[0], spec_b.shape[0])
    diff = np.abs(spec_b[:_n] - spec_a[:_n])
    return float(np.clip(np.mean(diff) / (np.mean(spec_a[:_n]) + 1e-10), 0.0, 1.0))


# ── Kalibrierung ────────────────────────────────────────────────────────


def calibrate_thresholds(song_dir: str, min_songs: int = 10) -> dict:
    """Kalibriert Hallucination-Guard-Schwellen.

    Lädt alle WAV/FLAC/MP3 im Verzeichnis, führt jede Phase-0-Stufe einzeln aus
    und misst die spectral_novelty. Empfiehlt Schwellen basierend auf Perzentilen.

    Returns:
        dict mit empfohlenen Schwellen pro Stufe.
    """
    from plugins.apollo_phase0_integration import (
        ApolloPhase0Guard,
        DeepFilterNetGuard,
        ResembleEnhanceGuard,
    )

    _audio_extensions = {".wav", ".flac", ".mp3", ".aac", ".m4a", ".ogg"}

    _files = []
    for _root, _dirs, _filenames in os.walk(song_dir):
        for _fn in _filenames:
            if Path(_fn).suffix.lower() in _audio_extensions:
                _files.append(os.path.join(_root, _fn))

    if len(_files) < min_songs:
        logger.warning("Nur %d Songs gefunden (min %d). Kalibrierung ggf. ungenau.", len(_files), min_songs)

    _novelties: dict[str, list[float]] = {
        "apollo": [],
        "deepfilternet": [],
        "resemble_enhance": [],
    }

    for _i, _file in enumerate(_files[: max(min_songs, 50)]):
        logger.info("[%d/%d] %s", _i + 1, min(len(_files), 50), os.path.basename(_file))
        try:
            import soundfile as sf

            _audio, _sr = sf.read(_file, dtype="float32")
            if _sr != 48000:
                continue
            _audio = np.asarray(_audio, dtype=np.float32)
            _audio = _audio[: min(len(_audio), 48000 * 30)]  # Max 30s
        except Exception as exc:
            logger.warning("  ueberspringen: %s", exc)
            continue

        # ── Apollo ──
        try:
            _apollo = ApolloPhase0Guard()
            _apollo._hallucination_threshold = 999.0  # Disable guard
            _out, _applied = _apollo.process(_audio, 48000, "mp3_high")  # type: ignore[misc]
            # _apollo.process returns ApolloResult now, but we need raw
            # Actually ApolloPhase0Guard.process returns (audio, bool)
            # Let me use the raw model
            if hasattr(_out, "audio"):  # type: ignore[has-type]
                _out_audio = _out.audio  # type: ignore[has-type]
            else:
                _out_audio = _out  # type: ignore[has-type]
            _nov = spectral_novelty(_audio[: len(_out_audio)], _out_audio[: len(_audio)], 48000)
            _novelties["apollo"].append(_nov)
            logger.info("  apollo: novelty=%.4f", _nov)
        except Exception as exc:
            logger.debug("  apollo fehlgeschlagen: %s", exc)

        # ── DeepFilterNet ──
        try:
            _dfn = DeepFilterNetGuard()
            _dfn._threshold = 999.0
            _dfn_out, _dfn_applied = _dfn.process(_audio, 48000)
            _nov_df = spectral_novelty(_audio[: len(_dfn_out)], _dfn_out, 48000)
            _novelties["deepfilternet"].append(_nov_df)
            logger.info("  deepfilternet: novelty=%.4f", _nov_df)
        except Exception as exc:
            logger.debug("  deepfilternet fehlgeschlagen: %s", exc)

        # ── Resemble Enhance ──
        try:
            _re = ResembleEnhanceGuard()
            _re._threshold = 999.0
            _re_out, _re_applied = _re.process(_audio, 48000)
            _nov_re = spectral_novelty(_audio[: len(_re_out)], _re_out, 48000)
            _novelties["resemble_enhance"].append(_nov_re)
            logger.info("  resemble_verbessern: novelty=%.4f", _nov_re)
        except Exception as exc:
            logger.debug("  resemble_verbessern fehlgeschlagen: %s", exc)

    # ── Empfehlungen ──
    _recommendations = {}
    _STATS = {}
    for _stufe, _values in _novelties.items():
        if len(_values) < 3:
            _recommendations[_stufe] = {"status": "insufficient_data", "threshold": None}
            continue
        _arr = np.array(_values)
        _p50 = float(np.percentile(_arr, 50))
        _p75 = float(np.percentile(_arr, 75))
        _p90 = float(np.percentile(_arr, 90))
        _p95 = float(np.percentile(_arr, 95))

        # Empfehlung: P95 + 10% Margin (konservativ)
        _rec = float(np.clip(_p95 * 1.10, 0.05, 0.50))
        _recommendations[_stufe] = {
            "p50": round(_p50, 4),  # type: ignore[dict-item]
            "p75": round(_p75, 4),  # type: ignore[dict-item]
            "p90": round(_p90, 4),  # type: ignore[dict-item]
            "p95": round(_p95, 4),  # type: ignore[dict-item]
            "recommended_threshold": round(_rec, 4),  # type: ignore[dict-item]
            "n_samples": len(_values),  # type: ignore[dict-item]
        }
        _STATS[_stufe] = _arr

    _result = {
        "calibration_date": datetime.now().isoformat(),
        "n_songs_analyzed": len(_files[: max(min_songs, 50)]),
        "recommendations": _recommendations,
        "current_thresholds": {
            "apollo": 0.15,
            "deepfilternet": 0.25,
            "resemble_enhance": 0.20,
        },
    }

    # ── Output ──
    _report_dir = _PROJECT_ROOT / "reports"
    _report_dir.mkdir(exist_ok=True)
    _report_path = _report_dir / f"phase0_calibration_{datetime.now():%Y-%m-%d}.json"
    with open(_report_path, "w") as _f:
        json.dump(_result, _f, indent=2, ensure_ascii=False)
    logger.info("Kalibrierung gespeichert: %s", _report_path)

    return _result


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase-0 Hallucination-Guard-Kalibrierung")
    parser.add_argument("--dir", required=True, help="Verzeichnis mit Songs (WAV/FLAC/MP3)")
    parser.add_argument("--min-songs", type=int, default=10, help="Minimale Anzahl Songs")
    args = parser.parse_args()

    result = calibrate_thresholds(args.dir, args.min_songs)
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))

    # Empfehlungen ausgeben
    print("\n=== EMPFEHLUNGEN ===")
    for _stufe, _rec in result["recommendations"].items():
        _current = result["current_thresholds"].get(_stufe, "?")
        _new = _rec.get("recommended_threshold", "?")
        print(f"  {_stufe}: {_current} → {_new} (n={_rec.get('n_samples', 0)})")


if __name__ == "__main__":
    main()
