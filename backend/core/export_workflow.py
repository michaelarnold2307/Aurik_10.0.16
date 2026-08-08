"""
Aurik v8 – Professional Export Workflow

GAPs ADRESSIERT:
- GAP #41: Enhanced Format Support (AIFF, CAF, Ogg Vorbis, Opus)
- GAP #42: Basic Metadata Preservation
- GAP #44: Multi-Version Export (multiple modes/qualities)

Exportiert restauriertes Audio in verschiedenen Formaten mit Metadata-Erhaltung
und Multi-Version Support für professionelle Workflows.

Version: 2.0.0
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.core.pipeline_health_state import (
    PipelineHealthState,
    pipeline_health_from_fail_reasons,
    primary_fail_reason_from_fail_reasons,
)

logger = logging.getLogger(__name__)


def _evaluate_export_quality_gate(
    quality_gate: dict | None,
) -> tuple[bool | None, str | None, str | None, list[dict[str, str]]]:
    """Bewertet quality gate payload.

    Expected schema:
        {
            "passed": bool,
            "fail_reason": str | None,
            "fail_reasons": list[dict] | None,
            "required_gates": list[str] | None,
        }
    """
    if quality_gate is None:
        return None, None, None, []

    _fqf = (
        quality_gate.get("fallback_quality_floor")
        if isinstance(quality_gate.get("fallback_quality_floor"), dict)
        else {}
    )
    _fqf_triggered = bool(_fqf.get("triggered", False))  # type: ignore[union-attr]
    _fqf_status = str(_fqf.get("status", "")).strip().lower()  # type: ignore[union-attr]

    # Deterministic coupling: fallback_quality_floor is authoritative for degraded/recovered
    # export states when triggered.
    if _fqf_triggered and _fqf_status in {"recovered", "degraded", "failed", "fail"}:
        _fqf_reason = str(_fqf.get("reason", "fallback_quality_floor_triggered") or "fallback_quality_floor_triggered")  # type: ignore[union-attr]
        _fqf_failed = _fqf_status in {"degraded", "failed", "fail"}
        _severity = "failed" if _fqf_failed else "degraded"
        _error_code = "FALLBACK_QUALITY_FLOOR_FAILED" if _fqf_failed else "FALLBACK_QUALITY_FLOOR_RECOVERED"
        return (
            False,
            _fqf_reason,
            (_fqf_status if _fqf_status in {"recovered", "degraded"} else "degraded"),
            [
                {
                    "component": "FallbackQualityFloor",
                    "error_code": _error_code,
                    "severity": _severity,
                    "exc_type": "",
                    "exc_msg": _fqf_reason,
                }
            ],
        )

    passed = bool(quality_gate.get("passed", False))
    if passed:
        return True, None, PipelineHealthState.OK.value, []

    raw_fail_reasons = quality_gate.get("fail_reasons") or []
    normalized_fail_reasons: list[dict[str, str]] = []
    if isinstance(raw_fail_reasons, list):
        for entry in raw_fail_reasons:
            if not isinstance(entry, dict):
                continue
            normalized_fail_reasons.append(
                {
                    "component": str(entry.get("component", "quality_gate") or "quality_gate"),
                    "error_code": str(entry.get("error_code", "QUALITY_GATE_FAILED") or "QUALITY_GATE_FAILED"),
                    "severity": str(entry.get("severity", "blocked") or "blocked"),
                    "exc_type": str(entry.get("exc_type", "") or ""),
                    "exc_msg": str(entry.get("exc_msg", entry.get("message", "")) or ""),
                }
            )

    fail_reason = str(quality_gate.get("fail_reason") or "").strip()
    if not fail_reason:
        fail_reason = primary_fail_reason_from_fail_reasons(
            normalized_fail_reasons,
            default="Unbekannter Quality-Gate-Fehler",
        )
    if not normalized_fail_reasons:
        normalized_fail_reasons = [
            {
                "component": "quality_gate",
                "error_code": "QUALITY_GATE_FAILED",
                "severity": "blocked",
                "exc_type": "",
                "exc_msg": fail_reason,
            }
        ]
    degradation_status = pipeline_health_from_fail_reasons(normalized_fail_reasons).value
    required = quality_gate.get("required_gates") or []
    required_str = ", ".join(str(x) for x in required) if required else "nicht angegeben"
    logger.warning(
        "Quality gate fehlgeschlagen. Wiederherstellung metadata required before Ausgabe. reason=%s required_gates=%s degradation=%s",
        fail_reason,
        required_str,
        degradation_status,
    )
    return False, fail_reason, degradation_status, normalized_fail_reasons


@dataclass
class ExportMetadata:
    """Metadata for exported audio files"""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    date: str | None = None
    genre: str | None = None
    comment: str | None = None
    # Processing metadata
    aurik_version: str = "v8.0"
    processing_date: str | None = None
    restoration_applied: bool = True
    sample_rate: int | None = None
    bit_depth: int | None = None
    channels: int | None = None
    export_strategy: str = "success"
    quality_gate_passed: bool | None = None
    quality_gate_fail_reason: str | None = None
    quality_gate_degradation_status: str | None = None
    quality_gate_fail_reasons: list[dict[str, str]] | None = None
    # §2.46b / §2.49 Fidelity-Guard telemetry — populated from RestorationResult.metadata
    fidelity_guards: dict | None = None


SUPPORTED_FORMATS = {
    # Lossless formats (via soundfile)
    "wav": {"subtype": "PCM_24", "extension": ".wav", "description": "WAV (PCM 24-bit)"},
    "flac": {"subtype": "PCM_24", "extension": ".flac", "description": "FLAC (Lossless)"},
    "aiff": {"subtype": "PCM_24", "extension": ".aif", "description": "AIFF (Apple)"},
    "caf": {"subtype": "PCM_24", "extension": ".caf", "description": "CAF (Core Audio)"},
    # Compressed formats (via soundfile - Ogg Vorbis)
    "ogg": {"subtype": "VORBIS", "extension": ".ogg", "description": "Ogg Vorbis"},
    # Note: Opus requires libsndfile 1.0.29+ with Opus support
    # 'opus': {'subtype': 'OPUS', 'extension': '.opus', 'description': 'Opus'},
}


def _resolve_export_strategy(quality_gate: dict | None, gate_passed: bool | None) -> str:
    """Löst auf: export strategy with mandatory recovery contract.

    Rules:
    - passed=True  -> "success"
    - passed=False -> requires recovery_attempted=True, else RuntimeError
                     best_possible_reached=True -> "recovered"
                     otherwise                  -> "degraded"
    - quality_gate=None -> "success" (legacy/no gate payload)
    """
    if quality_gate is not None:
        _fqf = (
            quality_gate.get("fallback_quality_floor")
            if isinstance(quality_gate.get("fallback_quality_floor"), dict)
            else {}
        )
        if bool(_fqf.get("triggered", False)):  # type: ignore[union-attr]
            _fqf_status = str(_fqf.get("status", "")).strip().lower()  # type: ignore[union-attr]
            if _fqf_status == "recovered":
                return "recovered"
            if _fqf_status in {"degraded", "failed", "fail"}:
                return "degraded"

    if gate_passed is True or quality_gate is None:
        return "success"

    if gate_passed is False:
        recovery_attempted = bool(quality_gate.get("recovery_attempted", False))
        if not recovery_attempted:
            raise RuntimeError(
                "Export blocked: quality_gate failed and no recovery_attempted flag was provided. "
                "Run recovery cascade first and pass recovery metadata."
            )
        best_possible_reached = bool(quality_gate.get("best_possible_reached", False))
        return "recovered" if best_possible_reached else "degraded"

    return "success"


def export_audio(
    audio: np.ndarray,
    sr: int,
    filename: str,
    format: str = "wav",
    metadata: ExportMetadata | None = None,
    quality_gate: dict | None = None,
    output_dir: str = "export",
) -> str:
    """
    Export audio with format and metadata support.

    Args:
        audio: Audio data (mono or stereo)
        sr: Sample rate
        filename: Base filename (without extension)
        format: Export format ('wav', 'flac', 'aiff', 'caf', 'ogg')
        metadata: Optional metadata to embed
        quality_gate: Optional quality gate payload; non-blocking and documented in metadata
        output_dir: Output directory

    Returns:
        Path to exported file

    Raises:
        ValueError: If format not supported
        RuntimeError: If export fails
    """
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Format '{format}' not supported. Supported: {', '.join(SUPPORTED_FORMATS.keys())}")

    gate_passed, gate_fail_reason, gate_degradation_status, gate_fail_reasons = _evaluate_export_quality_gate(
        quality_gate
    )
    export_strategy = _resolve_export_strategy(quality_gate, gate_passed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build full path with correct extension
    fmt_info = SUPPORTED_FORMATS[format]
    base_name = Path(filename).stem  # Remove any existing extension
    export_path = os.path.join(output_dir, base_name + fmt_info["extension"])

    # Prepare metadata
    if metadata is None:
        metadata = ExportMetadata()

    # Update processing metadata
    metadata.processing_date = datetime.now().isoformat()
    metadata.sample_rate = sr
    metadata.bit_depth = 24  # Default to 24-bit for quality
    metadata.channels = audio.shape[1] if audio.ndim == 2 else 1
    metadata.quality_gate_passed = gate_passed
    metadata.quality_gate_fail_reason = gate_fail_reason
    metadata.quality_gate_degradation_status = gate_degradation_status
    metadata.quality_gate_fail_reasons = gate_fail_reasons
    metadata.export_strategy = export_strategy

    try:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        # Write audio file
        tmp_export_path = export_path + ".tmp"
        try:
            sf.write(tmp_export_path, audio, sr, format=format.upper(), subtype=fmt_info["subtype"])
            os.replace(tmp_export_path, export_path)
        finally:
            if os.path.exists(tmp_export_path):
                os.remove(tmp_export_path)

        # Write sidecar metadata (JSON) for formats that don't support embedded tags
        if metadata:
            _write_metadata_sidecar(export_path, metadata)

        logger.debug("✓ Exported: %s (%s)", export_path, fmt_info["description"])
        return export_path

    except Exception as e:
        raise RuntimeError(f"Export failed for '{format}': {e}") from e


def export_multi_version(
    audio: np.ndarray,
    sr: int,
    base_filename: str,
    formats: list[str] | None = None,
    metadata: ExportMetadata | None = None,
    quality_gate: dict | None = None,
    output_dir: str = "export",
) -> dict[str, str]:
    """
    GAP #44: Export multiple versions in different formats.

    Useful for:
    - Archive: FLAC (lossless)
    - Distribution: WAV (universal)
    - Streaming: Ogg Vorbis (compressed)

    Args:
        audio: Audio data
        sr: Sample rate
        base_filename: Base filename (without extension)
        formats: List of formats to export (default: ['wav', 'flac'])
        metadata: Optional metadata
        quality_gate: Optional quality gate payload; blocks export if failed
        output_dir: Output directory

    Returns:
        Dict mapping format -> file path
    """
    if formats is None:
        formats = ["wav", "flac"]  # Default: lossless archive + universal

    results = {}
    for fmt in formats:
        try:
            path = export_audio(
                audio,
                sr,
                base_filename,
                format=fmt,
                metadata=metadata,
                quality_gate=quality_gate,
                output_dir=output_dir,
            )
            results[fmt] = path
        except Exception as e:
            logger.debug("⚠️ Ausgabe fehlgeschlagen for format '%s': %s", fmt, e)
            results[fmt] = None  # type: ignore[assignment]

    logger.debug(
        "✓ Multi-version Ausgabe vollstaendig: %s / %s formats", len([p for p in results.values() if p]), len(formats)
    )
    return results


def _write_metadata_sidecar(audio_path: str, metadata: ExportMetadata):
    """
    Schreibt metadata as JSON sidecar file (GAP #42: Basic Metadata Preservation).

    For formats that don't support embedded tags (or as backup).
    Creates a .json file alongside the audio file.
    """
    sidecar_path = Path(audio_path).with_suffix(".json")

    metadata_dict = asdict(metadata)
    # Remove None values
    metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

    return sidecar_path


def export_audit_log(audit_log: list, filename: str):
    """Export audit log (legacy function, kept for compatibility)"""
    log_path = os.path.join("logs", filename)
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "a") as f:
        for entry in audit_log:
            f.write(str(entry) + "\n")
    logger.debug("✓ Audit-Log gespeichert: %s", log_path)
    return log_path


# Example usage:
# metadata = ExportMetadata(title="My Song", artist="Artist Name", album="Album")
# export_audio(restored_audio, 48000, "result", format='flac', metadata=metadata)
# export_multi_version(restored_audio, 48000, "result", formats=['wav', 'flac', 'ogg'])
# export_stems(restored_audio, 48000, "result", backend='auto')  # GAP #43


# ==============================================================================
# GAP #43: STEM EXPORT
# ==============================================================================


def export_stems(
    audio: np.ndarray,
    sr: int,
    base_filename: str,
    format: str = "wav",
    backend: str = "auto",
    metadata: ExportMetadata | None = None,
    output_dir: str = "export/stems",
) -> dict[str, str]:
    """
    GAP #43: Export audio as separate stems (vocals, drums, bass, other).

    Separates the mixed audio into individual stems using source separation,
    then exports each stem as a separate file.

    Args:
        audio: Input audio (mixed/full track)
        sr: Sample rate
        base_filename: Base filename for stems (will add _vocals, _drums, etc.)
        format: Export format ('wav', 'flac', 'aiff', etc.)
        backend: Separation backend ('auto', 'spectral', 'banquet')
        metadata: Optional metadata (copied to all stems)
        output_dir: Output directory for stems

    Returns:
        Dict mapping stem name -> file path

    Example:
        >>> stems_paths = export_stems(audio, 48000, "my_song")
        >>> # Creates: my_song_vocals.wav, my_song_drums.wav, my_song_bass.wav, my_song_other.wav
    """
    try:
        from dsp.stem_separator import StemSeparator
    except ImportError as e:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
        raise RuntimeError("Stem separator not available. Make sure dsp/ module is in your Python path.") from e

    logger.debug("Separating stems (backend: %s)...", backend)

    # Separate into stems
    separator = StemSeparator(backend=backend)
    stems = separator.separate(audio, sr)

    backend_info = separator.get_backend_info()
    logger.debug("✓ Separated using %s (%s quality)", backend_info["backend"], backend_info["quality"])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export each stem
    results = {}
    for stem_name, stem_audio in stems.items():
        stem_filename = f"{base_filename}_{stem_name}"

        # Copy metadata and add stem-specific info
        if metadata:
            stem_metadata = ExportMetadata(
                title=metadata.title,
                artist=metadata.artist,
                album=metadata.album,
                date=metadata.date,
                genre=metadata.genre,
                comment=f"{metadata.comment or ''} [Stem: {stem_name}]".strip(),
                aurik_version=metadata.aurik_version,
                processing_date=metadata.processing_date,
                restoration_applied=metadata.restoration_applied,
            )
        else:
            stem_metadata = ExportMetadata(comment=f"Stem: {stem_name}")

        try:
            path = export_audio(
                stem_audio, sr, stem_filename, format=format, metadata=stem_metadata, output_dir=output_dir
            )
            results[stem_name] = path
            logger.debug("  ✓ %s → %s", stem_name, os.path.basename(path))
        except Exception as e:
            logger.debug("  ⚠️  %s Ausgabe fehlgeschlagen: %s", stem_name, e)
            results[stem_name] = None  # type: ignore[assignment]

    # Print summary
    successful = len([p for p in results.values() if p])
    logger.debug("✓ Stem Ausgabe vollstaendig: %s/%s stems exported", successful, len(stems))

    # Print metrics
    metrics = separator.get_metrics()
    if metrics:
        logger.debug("  Backend: %s", metrics.get("backend", "unknown"))
        logger.debug("  Quality: %s", metrics.get("quality", "unknown"))

    return results
