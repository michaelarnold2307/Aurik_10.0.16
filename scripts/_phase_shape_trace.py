#!/usr/bin/env python3
"""
§ Phase-Shape-Tracer — findet die Phase die als erste truncatet oder Stereo zerstört.
Ausgabe: tabellarisch, bricht bei erstem Problem ab.
Usage: .venv_aurik/bin/python scripts/_phase_shape_trace.py
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Minimales Logging (nur WARNING+ von Drittanbieter-Libs) ──────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-7s %(name)s: %(message)s",
    stream=sys.stdout,
)
for lib in ("matplotlib", "PIL", "urllib3", "numba", "filelock", "acoustics", "torch"):
    logging.getLogger(lib).setLevel(logging.CRITICAL)

import numpy as np

INPUT_FILE = str(
    PROJECT_ROOT / "test_audio" / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
)
if not os.path.exists(INPUT_FILE):
    print(f"[FEHLER] Input nicht gefunden: {INPUT_FILE}")
    sys.exit(1)

SR = 48_000
_trace_rows: list[Any] = []
_input_n_samples = None  # wird nach erstem load gesetzt
_first_anomaly = None


def _nonzero_len(audio: np.ndarray, sr: int) -> float:
    mono = audio[:, 0] if audio.ndim == 2 else audio
    nz = np.where(np.abs(mono) > 1e-7)[0]
    return float(nz[-1]) / sr if len(nz) else 0.0


def _is_stereo(audio: np.ndarray) -> bool:
    return audio.ndim == 2 and audio.shape[1] == 2


def _active_frac(audio: np.ndarray) -> float:
    mono = audio[:, 0] if audio.ndim == 2 else audio
    return float((np.abs(mono) > 1e-3).mean())


# ── Monkeypatch ────────────────────────────────────────────────────────────
from backend.core import unified_restorer_v3 as _uv3_mod

_UVcls = _uv3_mod.UnifiedRestorerV3
_original_profiled = _UVcls._profiled_phase_call


def _patched_profiled_phase_call(self, phase, audio: np.ndarray, **kwargs):
    global _first_anomaly

    phase_id = "?"
    try:
        phase_id = phase.get_metadata().phase_id
    except Exception:
        logger.warning(
            "_Verarbeitungsschritt_shape_trace.py::_patched_profiled_Verarbeitungsschritt_call Ersatzpfad",
            exc_info=True,
        )

    n_before = audio.shape[0]
    stereo_before = _is_stereo(audio)
    nz_before = _nonzero_len(audio, SR)
    act_before = _active_frac(audio)

    # Aufruf der echten Implementierung
    result = _original_profiled(self, phase, audio, **kwargs)

    # Extrahiere Audio aus dem Ergebnis
    out_audio = result.audio if hasattr(result, "audio") else result
    if not isinstance(out_audio, np.ndarray):
        out_audio = audio  # Fallback

    n_after = out_audio.shape[0]
    stereo_after = _is_stereo(out_audio)
    nz_after = _nonzero_len(out_audio, SR)
    act_after = _active_frac(out_audio)

    anomaly = ""
    if n_after < n_before - 100:
        anomaly += f" TRUNCATED({n_before}→{n_after}={n_after / SR:.1f}s)"
    if not stereo_after and stereo_before:
        anomaly += " STEREO→MONO"
    if nz_after < nz_before - 2.0:
        anomaly += f" NONZERO-DROP({nz_before:.1f}s→{nz_after:.1f}s)"
    if act_after < act_before - 0.10:
        anomaly += f" ACTIVE-DROP({act_before:.2%}→{act_after:.2%})"

    row = {
        "phase": phase_id,
        "shape_before": str(audio.shape),
        "shape_after": str(out_audio.shape),
        "nz_before": f"{nz_before:.2f}s",
        "nz_after": f"{nz_after:.2f}s",
        "active_before": f"{act_before:.1%}",
        "active_after": f"{act_after:.1%}",
        "anomaly": anomaly.strip(),
    }
    _trace_rows.append(row)

    flag = "‼️ " if anomaly else "   "
    print(
        f"{flag}{phase_id:45s} "
        f"shape:{audio.shape!s:12s}→{out_audio.shape!s:12s} "
        f"nz:{nz_before:7.2f}s→{nz_after:7.2f}s "
        f"active:{act_before:.0%}→{act_after:.0%}" + (f"  *** {anomaly} ***" if anomaly else ""),
        flush=True,
    )

    if anomaly and _first_anomaly is None:
        _first_anomaly = row
        print("\n" + "=" * 80)
        print(f"ERSTE ANOMALIE GEFUNDEN: {phase_id}")
        print(f"  Shape vorher : {audio.shape}")
        print(f"  Shape nachher: {out_audio.shape}")
        print(f"  Nonzero vorher / nachher: {nz_before:.2f}s / {nz_after:.2f}s")
        print(f"  Anomalie: {anomaly}")
        print("=" * 80 + "\n")

    return result


_UVcls._profiled_phase_call = _patched_profiled_phase_call  # type: ignore[method-assign]

# ── Load & Run ──────────────────────────────────────────────────────────────
print("=" * 80)
print("PHASE SHAPE TRACER")
print(f"  Input: {INPUT_FILE}")
print(f"  SR: {SR} Hz")
print("=" * 80)
print(
    f"{'Phase':45s} {'shape_before':12s} {'shape_after':12s} "
    f"{'nz_before':10s} {'nz_after':10s} {'act_b':7s} {'act_a':7s}"
)
print("-" * 120)

from backend.file_import import load_audio_file

result_dict = load_audio_file(INPUT_FILE)
if result_dict is None or result_dict.get("error") or result_dict.get("audio") is None:
    print(f"[FEHLER] load_audio_file: {result_dict}")
    sys.exit(1)
audio_raw = result_dict["audio"]
sr_raw = result_dict["sr"]
print(f"\nLoaded: shape={audio_raw.shape} sr={sr_raw}")

# Resample to 48kHz
import librosa

if sr_raw != SR:
    if audio_raw.ndim == 2:
        audio_48k = np.stack(
            [librosa.resample(audio_raw[:, ch], orig_sr=sr_raw, target_sr=SR) for ch in range(audio_raw.shape[1])],
            axis=1,
        ).astype(np.float32)
    else:
        audio_48k = librosa.resample(audio_raw, orig_sr=sr_raw, target_sr=SR).astype(np.float32)
else:
    audio_48k = audio_raw.astype(np.float32)

print(f"At 48kHz:  shape={audio_48k.shape}  duration={audio_48k.shape[0] / SR:.2f}s\n")
_input_n_samples = audio_48k.shape[0]

from denker.aurik_denker import get_aurik_denker

logger = logging.getLogger(__name__)

t0 = time.perf_counter()


def _progress(pct, msg, elapsed=0.0):
    pass  # suppress progress noise


denker = get_aurik_denker()
result = denker.denke(
    audio_48k,
    sr=SR,
    mode="Restoration",
    no_rt_limit=True,
    input_path=os.path.abspath(INPUT_FILE),
    progress_callback=_progress,
)

elapsed = time.perf_counter() - t0
print(f"\n{'=' * 80}")
print(f"FERTIG in {elapsed:.1f}s")

# Zusammenfassung
anomalies = [r for r in _trace_rows if r["anomaly"]]
print(f"\nPhasen gesamt: {len(_trace_rows)}")
print(f"Anomalien: {len(anomalies)}")
if anomalies:
    print("\nANOMALIE-ÜBERSICHT:")
    for r in anomalies:
        print(f"  {r['phase']:45s} {r['anomaly']}")
else:
    print("\nKeine Truncation-/Stereo-Anomalien gefunden!")
    # Zeige Ergebnis-Audio-Shape
    out_audio = result.audio if hasattr(result, "audio") else None
    if isinstance(out_audio, np.ndarray):
        nz_out = _nonzero_len(out_audio, SR)
        print(f"\nFinal audio: shape={out_audio.shape}  nz={nz_out:.2f}s  n={out_audio.shape[0] / SR:.2f}s")
