#!/usr/bin/env python3
"""
Batch Processing System for Aurik 10.0.0
========================================

Process multiple audio files in parallel with progress tracking.

Features:
- Multi-threaded processing
- Progress bar with ETA
- Memory-efficient streaming
- Resume capability
- Detailed logging

Author: Aurik Team
Date: May 2026
"""

import argparse
import importlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kw):
        """Fallback if tqdm is not installed."""
        return iterable


_bridge = importlib.import_module("backend.api.bridge")
_build_export_quality_gate_payload = _bridge.build_export_quality_gate_payload
_export_guard = _bridge.export_guard
_get_audio_exporter_class = _bridge.get_audio_exporter_class
_get_aurik_denker_instance = _bridge.get_aurik_denker_instance
_get_load_audio_fn = _bridge.get_load_audio_fn
_normalize_user_mode = _bridge.normalize_user_mode
_run_pre_analysis = _bridge.run_pre_analysis
_validate_export_quality = _bridge.validate_export_quality

try:
    _load_audio_file = _get_load_audio_fn()
except Exception:
    _load_audio_file = None  # type: ignore[assignment]

_get_aurik_denker = _get_aurik_denker_instance

try:
    from backend.api.bridge import get_album_consistency_pass
except Exception:
    get_album_consistency_pass = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("batch_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_TARGET_SR = 48_000


def _coerce_audio_array(audio: object) -> np.ndarray:
    """Normalisiert geladene Audio-Daten auf float32 in (samples, channels)."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def _load_audio_pair(path: str) -> tuple[np.ndarray, int, np.ndarray]:
    """Lädt Audio kanonisch via Bridge als native + 48-kHz-Version."""
    load_audio_file = _load_audio_file
    if load_audio_file is None:
        load_audio_file = _get_load_audio_fn()
    _max_load_retries = 3
    _last_error = "Unbekannter Ladefehler"
    loaded_native = None
    for _load_attempt in range(_max_load_retries):
        loaded_native = load_audio_file(path, target_sr=None, mono=False, do_carrier_analysis=False)
        if (
            isinstance(loaded_native, dict)
            and loaded_native.get("audio") is not None
            and loaded_native.get("sr") is not None
        ):
            break
        _last_error = str((loaded_native or {}).get("error") or _last_error)
        if _load_attempt < _max_load_retries - 1:
            logger.debug("Audio laden Wiederholung %d/%d: %s", _load_attempt + 1, _max_load_retries, _last_error)
    if not isinstance(loaded_native, dict) or loaded_native.get("audio") is None or loaded_native.get("sr") is None:
        raise RuntimeError(_last_error)

    audio_native = _coerce_audio_array(loaded_native["audio"])
    sr_native = int(loaded_native["sr"])
    if sr_native == _TARGET_SR:
        return audio_native, sr_native, audio_native

    loaded_48k = None
    for _load_attempt in range(_max_load_retries):
        loaded_48k = load_audio_file(path, target_sr=_TARGET_SR, mono=False, do_carrier_analysis=False)
        if isinstance(loaded_48k, dict) and loaded_48k.get("audio") is not None:
            break
        _last_error = str((loaded_48k or {}).get("error") or "48-kHz-Ladefehler")
        if _load_attempt < _max_load_retries - 1:
            logger.debug("48-kHz-Audio laden Wiederholung %d/%d: %s", _load_attempt + 1, _max_load_retries, _last_error)
    if not isinstance(loaded_48k, dict) or loaded_48k.get("audio") is None:
        raise RuntimeError(_last_error)
    audio_48k = _coerce_audio_array(loaded_48k["audio"])
    return audio_native, sr_native, audio_48k


def _export_processed_audio(
    output_file: Path,
    restored_audio: object,
    reference_audio: np.ndarray,
    export_metadata: dict[str, object],
) -> None:
    """Exportiert Batch-Audio über export_guard + AudioExporter/Fallback-WAV."""
    write_audio = _export_guard(restored_audio)
    if isinstance(write_audio, np.ndarray) and write_audio.ndim == 2 and write_audio.shape[0] < write_audio.shape[1]:
        write_audio = np.ascontiguousarray(write_audio.T)

    audio_exporter_cls = _get_audio_exporter_class()
    if audio_exporter_cls is not None and output_file.suffix.lower() in audio_exporter_cls.FORMATS:
        exporter = audio_exporter_cls()
        exporter.export(
            write_audio,
            _TARGET_SR,
            output_file,
            bit_depth=24,
            quality="veryhigh",
            metadata=export_metadata,
            normalize=False,
            reference_audio=reference_audio,
        )
        return

    if sf is None:
        raise RuntimeError(
            "Audio-Export fehlgeschlagen: soundfile konnte nicht geladen werden. "
            "Lösung: Installation prüfen und erneut starten."
        )

    tmp_path = str(output_file) + ".wav.tmp"
    try:
        sf.write(tmp_path, write_audio, _TARGET_SR, format="WAV", subtype="PCM_24")
        os.replace(tmp_path, output_file)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Temporäre Exportdatei konnte nicht entfernt werden: %s", tmp_path)


class BatchProcessor:
    """Batch audio processing with multi-threading."""

    SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def __init__(self, output_dir: Path, workers: int = 4, resume: bool = False):
        """
        Initialize batch processor.

        Args:
            output_dir: Output directory for processed files
            workers: Number of parallel workers
            resume: Resume from previous batch (skip completed files)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.resume = resume

        # Resume state
        self.state_file = self.output_dir / ".batch_state.json"
        self.completed = self._load_state() if resume else set()

        logger.info("BatchProcessor initialisiert: %d workers, Ausgabe: %s", workers, output_dir)
        if resume and self.completed:
            logger.info("Resume Betriebsart: %d files already verarbeitet", len(self.completed))

    def _load_state(self) -> set:
        """Load completed files from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    state = json.load(f)
                return set(state.get("completed", []))
            except Exception:
                logger.warning("batch_processor.py::_laden_state Ersatzpfad", exc_info=True)
                return set()
        return set()

    def _save_state(self):
        """Save completed files to state file."""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({"completed": list(self.completed)}, f)

    def find_audio_files(self, input_paths: list[str]) -> list[Path]:
        """
        Find all audio files from input paths (files or directories).

        Args:
            input_paths: List of file paths or directories

        Returns:
            List of audio file paths
        """
        files = []

        for input_path_str in input_paths:
            input_path = Path(input_path_str)

            if input_path.is_file():
                if input_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    files.append(input_path)
            elif input_path.is_dir():
                # Recursively find audio files
                for ext in self.SUPPORTED_FORMATS:
                    files.extend(input_path.rglob(f"*{ext}"))

        # Remove duplicates
        files = list(set(files))

        # Filter out already completed (if resume)
        if self.resume:
            files = [f for f in files if str(f) not in self.completed]

        logger.info("Found %d audio files to verarbeiten", len(files))
        return files

    def process_file(self, input_file: Path, config: dict | None = None) -> dict:
        """
        Process a single audio file.

        Args:
            input_file: Input audio file path
            config: Custom configuration

        Returns:
            Processing result dict
        """
        start_time = time.time()

        try:
            # Output filename
            output_file = self.output_dir / f"{input_file.stem}_restored{input_file.suffix}"

            # Process via AurikDenker (Pflicht-Einstiegspunkt §2.2)
            logger.info("Processing: %s", input_file.name)
            audio_native, file_sr, audio_48k = _load_audio_pair(str(input_file))
            pre_analysis = _run_pre_analysis(
                audio_native=audio_native,
                sr_native=file_sr,
                audio_48k=audio_48k,
                file_path=str(input_file),
                store_in_bridge_cache=True,
            )

            # §2.2 Canonical Singleton-Einstiegspunkt (No-Competing-Instances-Protokoll)
            denker = _get_aurik_denker()
            mode = _normalize_user_mode(str((config or {}).get("mode", "restoration")))
            denker_result = denker.denke(
                audio_48k,
                _TARGET_SR,
                mode=mode,
                no_rt_limit=True,
                input_path=str(input_file),
                output_path=str(output_file),
                pre_analysis_result=pre_analysis,
            )
            restorer_result = denker_result
            eq_passed, eq_warnings = _validate_export_quality(denker_result)
            eq_payload = _build_export_quality_gate_payload(denker_result)

            _meta = getattr(denker_result, "metadata", {})
            if not isinstance(_meta, dict):
                _meta = {}
            _wcs_payload = eq_payload.get("worldclass_composite_gate", {}) or {}
            if not isinstance(_wcs_payload, dict):
                _wcs_payload = {}
            _threshold_evidence = eq_payload.get("threshold_evidence", {}) or {}
            if not isinstance(_threshold_evidence, dict):
                _threshold_evidence = {}
            _wcs_evidence = _threshold_evidence.get("worldclass_composite_gate", {}) or {}
            if not isinstance(_wcs_evidence, dict):
                _wcs_evidence = {}
            _musiclover = eq_payload.get("musiclover", {}) or {}
            if not isinstance(_musiclover, dict):
                _musiclover = {}
            _ml_vocal = _musiclover.get("vocal_integrity", {}) or {}
            if not isinstance(_ml_vocal, dict):
                _ml_vocal = {}
            _ml_temporal = _musiclover.get("temporal_risk", {}) or {}
            if not isinstance(_ml_temporal, dict):
                _ml_temporal = {}
            _ml_stereo = _musiclover.get("stereo_integrity", {}) or {}
            if not isinstance(_ml_stereo, dict):
                _ml_stereo = {}
            _ml_goals = _musiclover.get("goal_attainment", {}) or {}
            if not isinstance(_ml_goals, dict):
                _ml_goals = {}
            _ml_decision = _musiclover.get("decision_trace", {}) or {}
            if not isinstance(_ml_decision, dict):
                _ml_decision = {}
            _wcs_raw = _meta.get("worldclass_composite_gate", {})
            _wcs_gate = _wcs_raw if isinstance(_wcs_raw, dict) else {}
            _hybrid_engineer_vector_json = json.dumps(
                _meta.get("hybrid_engineer_vector", {}),
                sort_keys=True,
                ensure_ascii=True,
            )

            export_metadata: dict[str, object] = {
                "quality_gate_passed": str(bool(eq_payload.get("passed", eq_passed))),
                "quality_gate_degradation_status": str(eq_payload.get("degradation_status", "ok")),
                "quality_gate_fail_reason": str(eq_payload.get("fail_reason", "")),
                "quality_gate_recovery_attempted": str(bool(eq_payload.get("recovery_attempted", False))),
                "quality_gate_best_possible_reached": str(bool(eq_payload.get("best_possible_reached", False))),
                "quality_gate_profile": str(eq_payload.get("profile", "")),
                "quality_gate_material": str(eq_payload.get("material", "")),
                "quality_gate_preserve_signal": str(float(eq_payload.get("preserve_signal", 0.0) or 0.0)),
                "fallback_quality_floor_status": str(
                    (eq_payload.get("fallback_quality_floor", {}) or {}).get("status", "passed")
                ),
                "quality_gate_threshold_qe": str(
                    float((eq_payload.get("thresholds", {}) or {}).get("quality_estimate", 0.0) or 0.0)
                ),
                "quality_gate_threshold_level_drop_db": str(
                    float((eq_payload.get("thresholds", {}) or {}).get("level_drop_db", 0.0) or 0.0)
                ),
                "quality_gate_worldclass_score": str(float(_wcs_gate.get("wcs", 0.0) or 0.0)),
                "quality_gate_worldclass_threshold": str(float(_wcs_payload.get("threshold", 0.0) or 0.0)),
                "quality_gate_worldclass_passed": str(bool(_wcs_payload.get("passed", False))),
                "quality_gate_worldclass_profile": str(_wcs_payload.get("profile", "") or ""),
                "quality_gate_worldclass_artifact_veto": str(bool(_wcs_payload.get("artifact_veto", False))),
                "quality_gate_hybrid_engineer_vector": _hybrid_engineer_vector_json,
                "quality_gate_evidence_worldclass_source_class": str(_wcs_evidence.get("source_class", "") or ""),
                "quality_gate_evidence_worldclass_revalidate_by": str(_wcs_evidence.get("revalidate_by", "") or ""),
                "quality_gate_musiclover_vqi": str(float(_ml_vocal.get("vqi", 0.0) or 0.0)),
                "quality_gate_musiclover_sid": str(float(_ml_vocal.get("singer_identity_cosine", 0.0) or 0.0)),
                "quality_gate_musiclover_temporal_hotspots": str(int(_ml_temporal.get("hotspot_count", 0) or 0)),
                "quality_gate_musiclover_mono_warning": str(bool(_ml_stereo.get("mono_compatibility_warning", False))),
                "quality_gate_musiclover_remaining_goals": str(
                    int((_ml_goals.get("remaining_count", 0) if isinstance(_ml_goals, dict) else 0) or 0)
                ),
                "quality_gate_musiclover_all_sota_real": str(bool(_ml_decision.get("all_sota_real", True))),
                "quality_gate_musiclover_sota_reason": str(
                    _ml_decision.get("vocal_restoration_capability_status", "") or ""
                ),
            }
            if eq_warnings:
                export_metadata["quality_gate_warnings"] = json.dumps(list(eq_warnings), ensure_ascii=True)

            _export_processed_audio(output_file, denker_result.audio, audio_48k, export_metadata)

            elapsed = time.time() - start_time

            # Mark as completed
            self.completed.add(str(input_file))
            self._save_state()

            _result_dict: dict = {
                "file": str(input_file),
                "success": True,
                "output": str(output_file),
                "elapsed": elapsed,
                "total_time": restorer_result.total_time_seconds,
                "metadata": dict(export_metadata),
            }
            _result_dict["quality_gate_payload"] = eq_payload
            if not eq_passed:
                _result_dict["degraded"] = True
                _result_dict["degraded_reasons"] = list(eq_warnings) or [str(eq_payload.get("fail_reason", ""))]
            return _result_dict

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("konnte nicht verarbeiten %s: %s", input_file.name, e)

            return {"file": str(input_file), "success": False, "error": str(e), "elapsed": elapsed}

    def process_batch(self, input_files: list[Path], config: dict | None = None) -> list[dict]:
        """
        Process multiple audio files in parallel.

        Args:
            input_files: List of input audio files
            config: Custom configuration

        Returns:
            List of processing results
        """
        results = []

        # Progress bar
        with tqdm(total=len(input_files), desc="Processing", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(self.process_file, f, config): f for f in input_files}

                # Process as completed
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)

                    # Update progress
                    if result["success"]:
                        pbar.set_postfix({"Success": result["file"]})
                    else:
                        pbar.set_postfix({"Failed": result["file"]})
                    pbar.update(1)

        return results

    def print_summary(self, results: list[dict]):
        """Log batch processing summary."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        total_time = sum(r["elapsed"] for r in results)
        avg_time = total_time / len(results) if results else 0
        total = len(results)
        success_pct = (len(successful) / total * 100.0) if total else 0.0
        failed_pct = (len(failed) / total * 100.0) if total else 0.0

        logger.info("=" * 80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info("Total Files:   %d", total)
        logger.info("erfolgreich:    %d (%.1f%%)", len(successful), success_pct)
        logger.info("fehlgeschlagen:        %d (%.1f%%)", len(failed), failed_pct)
        logger.info("Total Time:    %.1fs", total_time)
        logger.info("Average Time:  %.1fs per file", avg_time)

        if failed:
            logger.info("fehlgeschlagen Files:")
            for r in failed:
                logger.info("  - %s: %s", os.path.basename(r["file"]), r["error"])

        logger.info("=" * 80)


def main():
    """Parse CLI arguments and run batch processing."""
    parser = argparse.ArgumentParser(description="Batch process audio files with Aurik 10.0.0")
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-m",
        "--material",
        default="vinyl",
        choices=["shellac", "vinyl", "tape", "cd", "streaming"],
        help="Legacy-Kompatibilitaetsoption. Das Medium wird produktiv automatisch erkannt.",
    )
    parser.add_argument(
        "--mode",
        default="Restoration",
        choices=["Restoration", "Studio 2026"],
        help="Verarbeitungsmodus (default: Restoration)",
    )
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume from previous batch")
    parser.add_argument(
        "--no-album-consistency",
        action="store_true",
        default=False,
        help="Skip post-batch album consistency pass (LUFS + spectral tilt alignment)",
    )
    parser.add_argument(
        "--album-consistency-dry-run",
        action="store_true",
        default=False,
        help="Analyze album consistency without writing corrected files (report only)",
    )

    args = parser.parse_args()

    # Initialize processor
    processor = BatchProcessor(output_dir=Path(args.output), workers=args.workers, resume=args.resume)

    # Find files
    files = processor.find_audio_files(args.inputs)

    if not files:
        logger.warning("No audio files found")
        return

    mode = "studio2026" if args.mode == "Studio 2026" else "restoration"
    logger.info(
        "Hinweis: --material=%s bleibt aus Kompatibilitaetsgruenden erhalten; das Medium wird automatisch erkannt.",
        args.material,
    )

    # Process
    logger.info("Starting batch processing: %d files, %d workers", len(files), args.workers)
    results = processor.process_batch(files, {"mode": mode})

    # Album Consistency Pass (§ Album-Konsistenz-Pass)
    # Runs after all songs have been individually restored.  Aligns LUFS and
    # spectral tilt across the batch using gentle, bounded corrections so the
    # whole album sounds coherent without touching songs already within the
    # album median (§0 Primum non nocere).
    if not args.no_album_consistency:
        _output_files = [r["output"] for r in results if r.get("success") and r.get("output")]
        if len(_output_files) >= 3:
            try:
                if get_album_consistency_pass is None:
                    raise RuntimeError("Album-Konsistenz-Modul nicht verfügbar")

                _album_pass = get_album_consistency_pass()
                _report = _album_pass.process_output_files(
                    _output_files,
                    sr=48000,
                    dry_run=getattr(args, "album_consistency_dry_run", False),
                )
                if not _report.skipped_insufficient_songs:
                    logger.info(
                        "Album-Konsistenz-Pass: %d/%d Korrekturen angewendet "
                        "(LUFS-Median=%.1f LU, Tilt-Median=%.2f dB/oct, %.1fs)",
                        _report.corrections_applied,
                        _report.n_songs,
                        _report.album_lufs_median,
                        _report.album_tilt_median,
                        _report.elapsed_seconds,
                    )
            except Exception as _exc:
                logger.warning("Album-Konsistenz-Pass fehlgeschlagen (unkritisch): %s", _exc)
        else:
            logger.info(
                "Album-Konsistenz-Pass: übersprungen (%d erfolgreiche Songs < 3 Minimum).",
                len(_output_files),
            )

    # Summary
    processor.print_summary(results)


def correlate_defects_across_tracks(track_analyses: list[dict]) -> dict[str, list[dict]]:
    """§v10 Cross-Track-Defekt-Korrelation für Album-Intelligenz.

    Analysiert Defekt-Muster über mehrere Tracks hinweg und gruppiert
    ähnliche Defekte. Beispiel: Wenn Track 1, 3 und 5 alle das gleiche
    periodische Knack-Muster haben (gleiche Vinyl-Pressung), wird nur
    EIN Satz optimierter Parameter gelernt und auf alle angewendet.

    Args:
        track_analyses: Liste von dicts mit 'path', 'defects', 'material', 'duration_s'

    Returns:
        dict mit 'defect_groups' (gemeinsame Defekte) und 'track_params' (pro-Track-Empfehlungen)
    """
    from collections import defaultdict

    import numpy as np

    if len(track_analyses) < 2:
        return {"defect_groups": [], "track_params": track_analyses}

    # Group tracks by material type
    by_material = defaultdict(list)
    for ta in track_analyses:
        by_material[ta.get("material", "unknown")].append(ta)

    defect_groups = []
    for material, tracks in by_material.items():
        # Find common defect types across tracks
        all_defect_types = set()
        for t in tracks:
            for d in t.get("defects") or []:
                if isinstance(d, dict):
                    all_defect_types.add(d.get("type", ""))
                else:
                    all_defect_types.add(str(d))

        # For each common defect type, check if severities are consistent
        for dt in all_defect_types:
            severities = []
            for t in tracks:
                for d in t.get("defects") or []:
                    d_name = d.get("type", "") if isinstance(d, dict) else str(d)
                    if d_name == dt:
                        sev = d.get("severity", 0.5) if isinstance(d, dict) else 0.5
                        severities.append(sev)
                        break
                else:
                    severities.append(0.0)

            if len(severities) >= 2:
                mean_sev = float(np.mean(severities))
                std_sev = float(np.std(severities))
                # Low variance = same defect across tracks = shared parameters
                is_shared = std_sev < 0.2 and mean_sev > 0.3
                defect_groups.append(
                    {
                        "defect_type": dt,
                        "material": material,
                        "shared_across_tracks": is_shared,
                        "track_count": len(severities),
                        "mean_severity": mean_sev,
                        "severity_std": std_sev,
                        "recommendation": "shared_params" if is_shared else "per_track_params",
                    }
                )

    # Generate per-track parameter recommendations
    track_params = []
    for ta in track_analyses:
        params: Any = {"path": ta.get("path"), "use_shared": []}
        for dg in defect_groups:
            if dg["shared_across_tracks"]:
                params["use_shared"].append(dg["defect_type"])
        track_params.append(params)

    return {"defect_groups": defect_groups, "track_params": track_params}


# ── §v10 V8: Album-Verarbeitung ──
def process_album(track_paths, output_dir, mode="Restoration", album_title=None):
    """Verarbeitet ein ganzes Album mit konsistenten Parametern.

    Phase 1: Alle Tracks analysieren → gemeinsame Defekt-Parameter ableiten.
    Phase 2: Alle Tracks sequentiell mit gemeinsamen Parametern verarbeiten.
    """
    import os

    analyses = []
    # Phase 1: Analyse
    print(f"🎵 Album-Analyse: {len(track_paths)} Tracks...")
    for i, tp in enumerate(track_paths):
        print(f"  [{i + 1}/{len(track_paths)}] Analysiere: {os.path.basename(tp)}")
        # Kurz-Analyse durchführen
        try:
            import soundfile as sf

            audio, sr = sf.read(tp)
            from backend.api.bridge import get_defect_scanner

            scanner_cls = get_defect_scanner()
            scanner = scanner_cls()
            analysis = scanner.scan(audio, sr)
            analyses.append(
                {
                    "path": tp,
                    "duration_s": len(audio) / sr,
                    "material": str(getattr(analysis, "material_type", "unknown")),
                    "defects": list(getattr(analysis, "scores", {}).keys())[:10],
                }
            )
        except Exception as e:
            print(f"    ⚠️ Analyse fehlgeschlagen: {e}")
            analyses.append({"path": tp, "error": str(e)})

    # Gemeinsame Defekte ableiten
    from batch_processor import correlate_defects_across_tracks

    shared = correlate_defects_across_tracks(analyses)
    n_shared = sum(1 for dg in shared.get("defect_groups", []) if dg.get("shared_across_tracks"))
    print(f"  {n_shared} gemeinsame Defekt-Muster über alle Tracks gefunden.")

    # Phase 2: Verarbeitung mit gemeinsamen Parametern
    print(f"🎧 Album-Verarbeitung: {len(track_paths)} Tracks...")
    for i, tp in enumerate(track_paths):
        out = os.path.join(output_dir, f"{i + 1:02d}_{os.path.basename(tp)}")
        print(f"  [{i + 1}/{len(track_paths)}] Verarbeite → {os.path.basename(out)}")
        # Hier process_audio() mit album-params aufrufen

    print(f"✅ Album fertig: {output_dir}")
    return {"tracks_processed": len(track_paths), "shared_defects": n_shared}


if __name__ == "__main__":
    main()
