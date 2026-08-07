"""backend/core/recovery_checkpoint.py — OOM-Recovery-Checkpoint-System.

Persists pipeline state to disk on OOM so the next Aurik startup can
resume restoration from the last successful phase without quality loss.

Checkpoint lifecycle:
  1. OOM in _execute_pipeline → save_checkpoint() writes atomic JSON + WAV
  2. Next Aurik startup → find_pending_checkpoints() discovers unfinished jobs
  3. User confirms → resume_from_checkpoint() re-enters UV3 at the failed phase
  4. Successful completion → delete_checkpoint() removes temporary files

Files on disk per checkpoint:
  sessions/<stem>_oom_checkpoint.json   — serialised analysis caches + phase list
  sessions/<stem>_oom_audio.wav         — current_audio at point of failure (48 kHz)

Thread-safe: all public functions use atomic writes (tmp + os.replace).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SESSIONS_DIR: Path = Path(__file__).resolve().parents[2] / "sessions"
_MAX_CHECKPOINT_AGE_S: float = 7 * 24 * 3600.0  # 7 days


# ---------------------------------------------------------------------------
# Runtime version helper
# ---------------------------------------------------------------------------


def _get_aurik_version() -> str:
    """Liest Aurik version dynamically at checkpoint creation time.

    Priority:
    1. importlib.metadata (works when installed as editable package via pip install -e .)
    2. Aurik10.__version__ (works for desktop runtime without package install)
    3. regex parse of pyproject.toml  (always present in repository root)
    4. "unknown" as last-resort fallback
    """
    try:
        from importlib.metadata import PackageNotFoundError  # Python 3.8+  # pylint: disable=import-outside-toplevel
        from importlib.metadata import version as _pkg_version  # pylint: disable=import-outside-toplevel

        return _pkg_version("aurik9")
    except PackageNotFoundError:
        # Typischer Desktop-Fall: Aurik läuft aus dem Repo ohne pip-Installation.
        pass
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
    try:
        _pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        content = _pyproject.read_text(encoding="utf-8")
        m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        if m:
            return m.group(1)
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
    return "unknown"


# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------


@dataclass
class RecoveryCheckpoint:
    """Serialisable snapshot of pipeline state at OOM failure point."""

    # File identity
    input_path: str
    output_path: str

    # Pipeline position
    phases_executed: list[str]
    phases_remaining: list[str]
    mode: str  # "restoration" | "studio_2026"

    # Cached analysis results (avoid re-scan on resume)
    material_type: str  # MaterialType.value
    era_decade: int | None
    defect_scores: dict[str, float]  # {defect_type.value: severity}
    defect_scores_full: dict[str, dict[str, Any]]  # Full DefectScore dicts
    restorability_score: float | None
    spectral_fingerprint: dict[str, float]

    # Quality tracking
    quality_estimate_at_failure: float
    musical_goals_at_failure: dict[str, float]

    # Audio reference
    audio_wav_path: str  # Path to the temporary WAV with current_audio
    sample_rate: int
    original_input_path: str  # To re-read original for pre-repair reference

    # Metadata
    timestamp: float = field(default_factory=time.time)
    aurik_version: str = field(default_factory=_get_aurik_version)
    failure_phase: str = ""  # Phase that caused OOM
    failure_reason: str = "MemoryError"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _checkpoint_json_path(input_path: str) -> Path:
    """Derive checkpoint JSON path from input file path."""
    stem = Path(input_path).stem
    # Sanitise stem for filesystem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)[:120]
    return _SESSIONS_DIR / f"{safe_stem}_oom_checkpoint.json"


def _checkpoint_audio_path(input_path: str) -> Path:
    """Derive checkpoint WAV path from input file path."""
    stem = Path(input_path).stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)[:120]
    return _SESSIONS_DIR / f"{safe_stem}_oom_audio.wav"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    *,
    input_path: str,
    output_path: str,
    current_audio: np.ndarray,
    sample_rate: int,
    phases_executed: list[str],
    phases_remaining: list[str],
    failure_phase: str,
    mode: str,
    defect_result: Any,
    era_decade: int | None = None,
    restorability_score: float | None = None,
    quality_estimate: float = 0.0,
    musical_goals: dict[str, float] | None = None,
) -> str | None:
    """Speichert OOM recovery checkpoint to disk.

    Returns the JSON checkpoint path on success, None on failure.
    Designed to work under extreme memory pressure — minimal allocations.
    """
    try:
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        json_path = _checkpoint_json_path(input_path)
        audio_path = _checkpoint_audio_path(input_path)

        # 1. Write audio to WAV (atomic: .tmp → rename)
        try:
            import soundfile as sf  # pylint: disable=import-outside-toplevel
        except ImportError:
            logger.error("Wiederherstellung: soundfile not verfuegbar — cannot speichern checkpoint")
            return None

        audio_tmp = str(audio_path) + ".tmp"
        # Ensure mono/stereo shape is correct for soundfile
        audio_to_write = current_audio
        if audio_to_write.ndim == 1:
            pass  # mono, fine
        elif audio_to_write.ndim == 2 and audio_to_write.shape[0] == 2:
            audio_to_write = audio_to_write.T  # (2, N) → (N, 2) for soundfile
        sf.write(audio_tmp, audio_to_write, sample_rate, subtype="FLOAT", format="WAV")
        os.replace(audio_tmp, str(audio_path))
        logger.info("Wiederherstellung: Audio-Checkpoint gespeichert: %s", audio_path)

        # 2. Serialise defect scores
        defect_scores_simple: dict[str, float] = {}
        defect_scores_full: dict[str, dict[str, Any]] = {}
        if defect_result is not None and hasattr(defect_result, "scores"):
            for dt, ds in defect_result.scores.items():
                key = dt.value if hasattr(dt, "value") else str(dt)
                if hasattr(ds, "severity"):
                    defect_scores_simple[key] = float(ds.severity)
                    defect_scores_full[key] = {
                        "severity": float(ds.severity),
                        "confidence": float(getattr(ds, "confidence", 0.0)),
                        "locations": [[float(s), float(e)] for s, e in getattr(ds, "locations", [])],
                    }
                else:
                    defect_scores_simple[key] = float(ds)

        material_type_str = ""
        if defect_result is not None and hasattr(defect_result, "material_type"):
            mt = defect_result.material_type
            material_type_str = mt.value if hasattr(mt, "value") else str(mt)

        spectral_fp: dict[str, float] = {}
        if defect_result is not None and hasattr(defect_result, "spectral_fingerprint"):
            spectral_fp = {k: float(v) for k, v in (defect_result.spectral_fingerprint or {}).items()}

        # 3. Build checkpoint
        checkpoint = RecoveryCheckpoint(
            input_path=input_path,
            output_path=output_path,
            phases_executed=list(phases_executed),
            phases_remaining=list(phases_remaining),
            mode=mode,
            material_type=material_type_str,
            era_decade=era_decade,
            defect_scores=defect_scores_simple,
            defect_scores_full=defect_scores_full,
            restorability_score=restorability_score,
            spectral_fingerprint=spectral_fp,
            quality_estimate_at_failure=float(quality_estimate),
            musical_goals_at_failure=dict(musical_goals or {}),
            audio_wav_path=str(audio_path),
            sample_rate=sample_rate,
            original_input_path=input_path,
            failure_phase=failure_phase,
        )

        # 4. Write JSON (atomic)
        json_tmp = str(json_path) + ".tmp"
        with open(json_tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False)
        os.replace(json_tmp, str(json_path))
        logger.info(
            "Wiederherstellung: Checkpoint gespeichert — %d Phasen abgeschlossen, %d verbleibend. %s",
            len(phases_executed),
            len(phases_remaining),
            json_path,
        )
        return str(json_path)

    except Exception as exc:
        logger.error("Wiederherstellung: Checkpoint-Speicherung fehlgeschlagen: %s", exc)
        return None


def find_pending_checkpoints() -> list[RecoveryCheckpoint]:
    """Findet alle gültigen OOM-Checkpoints im Sessions-Verzeichnis.

    Filters out expired (> 7 days) and orphaned (missing audio WAV) checkpoints.
    """
    results: list[RecoveryCheckpoint] = []
    if not _SESSIONS_DIR.exists():
        return results

    now = time.time()
    for json_file in _SESSIONS_DIR.glob("*_oom_checkpoint.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Age check
            ts = data.get("timestamp", 0.0)
            if now - ts > _MAX_CHECKPOINT_AGE_S:
                logger.debug("Wiederherstellung: Checkpoint zu alt (%.0f Tage): %s", (now - ts) / 86400, json_file)
                _cleanup_checkpoint_files(json_file)
                continue

            # Audio file must exist
            audio_path = data.get("audio_wav_path", "")
            if not os.path.isfile(audio_path):
                logger.debug("Wiederherstellung: Audio-Datei fehlt: %s", audio_path)
                _cleanup_checkpoint_files(json_file)
                continue

            checkpoint = RecoveryCheckpoint(
                **{k: v for k, v in data.items() if k in RecoveryCheckpoint.__dataclass_fields__}
            )
            results.append(checkpoint)

        except Exception as exc:
            logger.debug("Wiederherstellung: Ungültiger Checkpoint %s: %s", json_file, exc)
            _cleanup_checkpoint_files(json_file)

    return results


def load_checkpoint_audio(checkpoint: RecoveryCheckpoint) -> np.ndarray | None:
    """Lädt resume source audio with §2.39 priority rules.

    Priority:
      1) Original input audio (full-quality resume path)
      2) Checkpoint WAV only as emergency fallback if original is unavailable

    Returns ``None`` only if both paths fail.
    """
    from backend.file_import import load_audio_file  # pylint: disable=import-outside-toplevel

    orig_exc: Exception | None = None

    def _resample_to_checkpoint_sr(audio: np.ndarray, sr_in: int) -> np.ndarray:
        """Resample loaded recovery audio to the checkpoint sample rate.

        Recovery resume in UV3 uses ``checkpoint.sample_rate`` as authoritative SR.
        If the loaded source has a different native SR (e.g. 44.1 kHz original vs
        48 kHz checkpoint), resampling is mandatory; otherwise the waveform is
        interpreted at the wrong SR and pitch/tempo shift occurs.
        """
        if sr_in == checkpoint.sample_rate:
            return np.asarray(audio, dtype=np.float32)  # type: ignore[no-any-return]

        import librosa  # pylint: disable=import-outside-toplevel

        logger.warning(
            "Wiederherstellung: Resampling resume source from %d Hz to checkpoint SR %d Hz.",
            sr_in,
            checkpoint.sample_rate,
        )

        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 1:
            return np.asarray(  # type: ignore[no-any-return]
                librosa.resample(x, orig_sr=sr_in, target_sr=checkpoint.sample_rate),
                dtype=np.float32,
            )

        # load_audio_file returns (samples, channels), but keep a robust fallback
        # for channel-first layouts coming from future/import variants.
        if x.shape[1] <= 8 and x.shape[0] > x.shape[1]:
            ch = [
                np.asarray(librosa.resample(x[:, i], orig_sr=sr_in, target_sr=checkpoint.sample_rate), dtype=np.float32)
                for i in range(x.shape[1])
            ]
            return np.stack(ch, axis=1).astype(np.float32, copy=False)  # type: ignore[no-any-return]

        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            ch = [
                np.asarray(librosa.resample(x[i], orig_sr=sr_in, target_sr=checkpoint.sample_rate), dtype=np.float32)
                for i in range(x.shape[0])
            ]
            return np.stack(ch, axis=0).astype(np.float32, copy=False)  # type: ignore[no-any-return]

        ch = [
            np.asarray(librosa.resample(x[:, i], orig_sr=sr_in, target_sr=checkpoint.sample_rate), dtype=np.float32)
            for i in range(x.shape[1])
        ]
        return np.stack(ch, axis=1).astype(np.float32, copy=False)  # type: ignore[no-any-return]

    # Primary source per §2.39: original input audio
    try:
        _res = load_audio_file(checkpoint.original_input_path, do_carrier_analysis=False)
        if _res is not None and not _res.get("error"):
            audio = np.asarray(_res["audio"], dtype=np.float32)
            sr = int(_res["sr"])
            if sr != checkpoint.sample_rate:
                logger.warning(
                    "Wiederherstellung: SR mismatch in Originalsignal — checkpoint %d Hz, Originalsignal %d Hz",
                    checkpoint.sample_rate,
                    sr,
                )
                audio = _resample_to_checkpoint_sr(audio, sr)
            logger.info(
                "Wiederherstellung: Originalsignal-Datei als Primärquelle geladen (Checkpoint nur Notfall-Ersatzpfad gemäß §2.39)."
            )
            return audio  # type: ignore[no-any-return]
        orig_exc = RuntimeError(str((_res or {}).get("error", "load_audio_file returned invalid result")))
    except Exception as _exc:
        orig_exc = _exc

    logger.warning(
        "Wiederherstellung: Originalsignal-Datei konnte nicht geladen werden (%s) — Notfall-Ersatzpfad auf Checkpoint-Audio: %s",
        type(orig_exc).__name__ if orig_exc is not None else "UnknownError",
        checkpoint.audio_wav_path,
    )

    # Emergency fallback: checkpoint audio WAV
    try:
        _res_cp = load_audio_file(checkpoint.audio_wav_path, do_carrier_analysis=False)
        if _res_cp is None or _res_cp.get("error"):
            raise RuntimeError(str((_res_cp or {}).get("error", "load_audio_file returned invalid result")))
        audio = np.asarray(_res_cp["audio"], dtype=np.float32)
        sr = int(_res_cp["sr"])
        if sr != checkpoint.sample_rate:
            logger.warning(
                "Wiederherstellung: SR mismatch — checkpoint %d Hz, WAV %d Hz",
                checkpoint.sample_rate,
                sr,
            )
            audio = _resample_to_checkpoint_sr(audio, sr)
        logger.info(
            "§2.39 OOM-Checkpoint-Ausnahme aktiv: Checkpoint-Audio wird verwendet, "
            "weil das Originalsignal nicht verfügbar/lesbar ist."
        )
        return audio  # type: ignore[no-any-return]
    except Exception as cp_exc:
        logger.error(
            "Wiederherstellung: Weder Originalsignal noch Checkpoint-Audio konnte geladen werden: Originalsignal: %s, Checkpoint: %s",
            orig_exc,
            cp_exc,
        )
        return None


def delete_checkpoint(input_path: str) -> None:
    """Delete checkpoint files after successful completion or user dismissal."""
    json_path = _checkpoint_json_path(input_path)
    _cleanup_checkpoint_files(json_path)


def cleanup_expired_checkpoints() -> int:
    """Entfernt all expired checkpoints. Called at startup."""
    if not _SESSIONS_DIR.exists():
        return 0
    removed = 0
    now = time.time()
    for json_file in _SESSIONS_DIR.glob("*_oom_checkpoint.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if now - data.get("timestamp", 0.0) > _MAX_CHECKPOINT_AGE_S:
                _cleanup_checkpoint_files(json_file)
                removed += 1
        except Exception:
            _cleanup_checkpoint_files(json_file)
            removed += 1
    if removed:
        logger.info("Wiederherstellung: %d abgelaufene Checkpoints bereinigt.", removed)
    return removed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cleanup_checkpoint_files(json_path: Path) -> None:
    """Entfernt checkpoint JSON and associated audio WAV."""
    try:
        # Derive audio path from JSON path
        audio_path = str(json_path).replace("_oom_checkpoint.json", "_oom_audio.wav")
        for path in [str(json_path), audio_path, audio_path + ".tmp", str(json_path) + ".tmp"]:
            try:
                os.remove(path)
            except OSError as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
