#!/usr/bin/env python3
"""
Aurik 10.0.0 — Kontinuierliche Tiefenanalyse & Qualitäts-Monitoring
=================================================================

Führt eine vollständige Restaurierung mit Phase-weisen Qualitäts-Checkpoints durch,
monitort Musical Goals, HPI, Artefakt-Freiheit und erkennt Anomalien automatisch.

Wird als Echtzeit-Dashboard mit Alerts ausgeführt.

Usage:
    python scripts/continuous_deep_analysis.py [--audio <path>] [--mode restoration|studio] [--realtime]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Needed so direct script execution (python scripts/...) can resolve project packages.
_WORKSPACE_ROOT = Path(__file__).parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# pylint: disable=wrong-import-position
from backend.core.pre_analysis import run_pre_analysis
from backend.core.unified_restorer_v3 import UnifiedRestorerV3
from backend.file_import import load_audio_file

# Import Pegelexplosion-Detektor
try:
    from scripts.pegelexplosion_detector import PegelexplosionDetector
except ImportError:
    PegelexplosionDetector = None  # type: ignore[assignment, misc]

# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class PhaseCheckpoint:
    """Checkpoint-Daten nach einer Phase."""

    phase_id: str
    wall_time_s: float
    musical_goals: dict[str, float]
    hpi_score: float | None
    artifact_freedom: float | None
    carrier_recovery_ratio: float | None
    noise_floor_db: float | None
    defects_remaining: int | None
    anomalies: list[str]
    pegelexplosion_detected: bool = False
    pegelexplosion_severity: str = "none"
    pegelexplosion_cause: str | None = None
    # §2.44 HPI-Komponenten — nur im finalen Checkpoint gesetzt (nach HPG-Auswertung)
    vqi: float | None = None
    mert_similarity: float | None = None
    timbral_fidelity: float | None = None
    emotional_arc_preservation: float | None = None
    hpi_ceiling: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint data into a JSON-safe dictionary."""
        d = asdict(self)
        d["musical_goals"] = {
            k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in self.musical_goals.items()
        }
        return d


# ============================================================================
# ANALYZER
# ============================================================================


class ContinuousDeepAnalyzer:
    """Führt Tiefenanalyse während der Restaurierung durch."""

    def __init__(self, realtime: bool = True):
        self.realtime = realtime
        self.logger = logging.getLogger(__name__)
        self.checkpoints: list[PhaseCheckpoint] = []
        self.anomalies_detected: list[str] = []
        self.pegelexplosion_detector = PegelexplosionDetector() if PegelexplosionDetector else None  # type: ignore[truthy-function]
        self._last_phase_audio = None
        self._final_musical_goals: dict[str, float] = {}
        self._final_vocal_metrics: dict[str, float] = {}
        self._final_hpg: dict[str, object] = {}
        self._final_carrier_chain_recovery_ratio: float = 0.0
        self._last_progress_pct: int = -1
        self._last_progress_phase: str = ""
        self._last_progress_elapsed_s: float = 0.0
        self._last_progress_update_t: float = 0.0
        self._last_progress_phase_start_t: float = 0.0

    def _progress_watchdog(self, stop_event: threading.Event, interval_s: float = 30.0) -> None:
        """Emit periodic heartbeat logs while restore() is busy inside a long phase."""
        while not stop_event.wait(interval_s):
            if self._last_progress_update_t <= 0.0:
                continue
            stale_s = time.monotonic() - self._last_progress_update_t
            if stale_s < interval_s:
                continue
            phase_runtime_s = stale_s
            if self._last_progress_phase_start_t > 0.0:
                phase_runtime_s = time.monotonic() - self._last_progress_phase_start_t
            self.logger.info(
                "  - Heartbeat: %d%% | Phase: %s | letzte Aktualisierung vor %.1fs | Phase aktiv seit %.1fs",
                self._last_progress_pct,
                self._last_progress_phase or "unknown_phase",
                stale_s,
                phase_runtime_s,
            )

    def _restore_progress_callback(self, pct: int, phase: str, elapsed_s: float) -> None:
        """Emit sparse progress updates so long restoration runs are not mistaken for hangs."""
        _pct = int(max(0, min(100, pct)))
        _phase = str(phase or "unknown_phase")
        _elapsed = float(max(0.0, elapsed_s))
        self._last_progress_elapsed_s = _elapsed
        self._last_progress_update_t = time.monotonic()
        if _pct == self._last_progress_pct and _phase == self._last_progress_phase:
            return
        if (
            _pct < 100
            and self._last_progress_pct >= 0
            and _pct - self._last_progress_pct < 5
            and _phase == self._last_progress_phase
        ):
            return
        self._last_progress_phase_start_t = self._last_progress_update_t
        self._last_progress_pct = _pct
        self._last_progress_phase = _phase
        self.logger.info("  - Fortschritt: %d%% | Phase: %s | Laufzeit: %.1fs", _pct, _phase, _elapsed)

    def run_analysis(
        self,
        audio_path: str,
        sr: int = 48000,
        mode: str = "restoration",
        output_dir: str = "analysis_results",
    ) -> dict[str, Any]:
        """
        Führt kontinuierliche Tiefenanalyse durch.

        Args:
            audio_path: Pfad zur Audio-Datei
            sr: Sample Rate (Standard: 48000 Hz)
            mode: "restoration" oder "studio_2026"
            output_dir: Verzeichnis für Analyse-Ergebnisse

        Returns:
            Ergebnis-Dictionary mit allen Checkpoints und Anomalien
        """
        _start_t = time.monotonic()
        self.logger.info("=" * 80)
        self.logger.info("AURIK 9 — KONTINUIERLICHE TIEFENANALYSE")
        self.logger.info("Audio: %s", audio_path)
        self.logger.info("Mode: %s", mode)
        self.logger.info("SR: %s Hz", sr)
        self.logger.info("=" * 80)

        # Run-lokalen Zustand immer zurücksetzen (Analyzer kann mehrfach verwendet werden).
        self.checkpoints = []
        self.anomalies_detected = []
        self._final_musical_goals = {}
        self._final_vocal_metrics = {}
        self._final_hpg = {}
        self._final_carrier_chain_recovery_ratio = 0.0

        pre_transfer_chain: list[str] = []
        pipeline_transfer_chain: list[str] = []
        pre_primary_material: str | None = None
        pre_era_decade: int | None = None

        # 1. Audio laden (mit Retry bei transienten WAV-Read-Fehlern)
        _max_load_retries = 3
        for _load_attempt in range(_max_load_retries):
            try:
                result = load_audio_file(audio_path)
                if result is None or result.get("error"):
                    _err = result.get("error") if result else "None"
                    if _load_attempt < _max_load_retries - 1 and "unpack" in str(_err).lower():
                        self.logger.debug("⏳ Audio-Load retry %d/%d: %s", _load_attempt + 1, _max_load_retries, _err)
                        time.sleep(0.5 * (_load_attempt + 1))
                        continue
                    self.logger.error("✗ Audio-Import fehlgeschlagen: %s", _err)
                    return {"error": str(_err), "checkpoints": []}
                audio = result["audio"]
                sr_imported = result["sr"]
                self.logger.info("✓ Audio geladen: %.1fs @ %d Hz", len(audio) / sr_imported, sr_imported)
                break
            except Exception as e:
                if _load_attempt < _max_load_retries - 1 and "unpack" in str(e).lower():
                    self.logger.debug("⏳ Audio-Load retry %d/%d: %s", _load_attempt + 1, _max_load_retries, e)
                    time.sleep(0.5 * (_load_attempt + 1))
                    continue
                self.logger.error("✗ Audio-Import fehlgeschlagen: %s", e)
                return {"error": str(e), "checkpoints": []}

        # 2. Pre-Analyse durchführen
        try:
            pre_result = run_pre_analysis(
                audio,
                sr_imported,
                file_path=str(Path(audio_path).resolve()),
                store_in_bridge_cache=False,
            )
            _medium_label = getattr(pre_result.medium, "primary_material", None) or getattr(
                pre_result.medium, "chain_label", None
            )
            _era_label = getattr(pre_result.era, "decade", None) or getattr(pre_result.era, "year_estimate", None)
            _rest_score = getattr(pre_result.restorability, "restorability_score", None)
            _defect_count = len(getattr(pre_result.defects, "scores", {}) or {}) if pre_result.defects else 0

            pre_primary_material = str(getattr(pre_result.medium, "primary_material", "") or "") or None
            pre_transfer_chain = self._extract_transfer_chain_from_obj(getattr(pre_result, "medium", None))
            _era_decade_raw = getattr(pre_result.era, "decade", None)
            if isinstance(_era_decade_raw, (int, np.integer)):
                pre_era_decade = int(_era_decade_raw)
            self.logger.info("✓ Pre-Analyse komplett:")
            self.logger.info("  - Material: %s", _medium_label if _medium_label is not None else "unknown")
            self.logger.info("  - Era: %s", _era_label if _era_label is not None else "unknown")
            if isinstance(_rest_score, (int, float)):
                self.logger.info("  - Restorability: %.1f%%", _rest_score)
            else:
                self.logger.info("  - Restorability: unknown")
            self.logger.info("  - Defekte gefunden: %d", _defect_count)
        except Exception as e:
            self.logger.warning("Pre-Analyse fehlgeschlagen (nicht kritisch): %s", e)
            pre_result = None

        # 3. Restaurierung mit Monitoring durchführen
        self._last_progress_pct = -1
        self._last_progress_phase = ""
        self._last_progress_elapsed_s = 0.0
        self._last_progress_update_t = 0.0
        self._last_progress_phase_start_t = 0.0
        _watchdog_stop = threading.Event()
        _watchdog_thread = threading.Thread(
            target=self._progress_watchdog,
            args=(_watchdog_stop,),
            name="continuous-deep-analysis-progress-watchdog",
            daemon=True,
        )
        _watchdog_thread.start()
        try:
            # Restaurierung durchführen
            is_studio_2026 = mode == "studio_2026"
            quality_mode = "studio_2026" if is_studio_2026 else "quality"
            restorer = UnifiedRestorerV3(quality_mode=quality_mode, monitor_phases=True)
            restoration_result = restorer.restore(
                audio=audio,
                sample_rate=sr_imported,
                progress_callback=self._restore_progress_callback,
                is_studio_2026=is_studio_2026,
                pre_analysis_result=pre_result,
                enable_debug_trace=True,
            )

            _final_goals = getattr(restoration_result, "musical_goals", None)
            if isinstance(_final_goals, dict):
                self._final_musical_goals = {
                    str(k): float(v) for k, v in _final_goals.items() if isinstance(v, (int, float))
                }

            self._collect_checkpoints_from_restore_result(restoration_result, pre_result)

            _meta = getattr(restoration_result, "metadata", {}) or {}
            _hpg = _meta.get("holistic_perceptual_gate", {}) or {}
            _afg = _meta.get("artifact_freedom", {}) or {}
            self._final_vocal_metrics = self._extract_vocal_metrics_from_metadata(_meta)
            self._final_hpg = dict(_hpg) if _hpg else {}
            self._final_carrier_chain_recovery_ratio = float(_meta.get("carrier_chain_recovery_ratio", 0.0) or 0.0)
            pipeline_transfer_chain = self._extract_transfer_chain_from_obj(_meta)
            if pre_era_decade is None:
                _meta_era = self._extract_era_decade_from_obj(_meta)
                if _meta_era is not None:
                    pre_era_decade = _meta_era
            self.logger.info("✓ Restaurierung komplett (Dauer: %.1fs)", time.monotonic() - _start_t)
            self.logger.info("  - HPI: %s", _hpg.get("hpi", restoration_result.metadata.get("hpi_score", "N/A")))
            self.logger.info(
                "  - Artefakt-Freiheit: %s",
                _afg.get("score", restoration_result.metadata.get("artifact_freedom_score", "N/A"))
                if isinstance(_afg, dict)
                else _afg,
            )

        except Exception as e:
            self.logger.error("✗ Restaurierung fehlgeschlagen: %s", e, exc_info=True)
            self.anomalies_detected.append(f"Restoration failed: {e}")
        finally:
            _watchdog_stop.set()
            _watchdog_thread.join(timeout=1.0)

        # 4. Ergebnisse speichern
        os.makedirs(output_dir, exist_ok=True)
        result_dict = {
            "wall_time_s": time.monotonic() - _start_t,
            "audio_path": audio_path,
            "mode": mode,
            "effective_quality_mode": getattr(getattr(restorer, "config", None), "mode", None).value  # type: ignore[union-attr]
            if "restorer" in locals() and getattr(getattr(restorer, "config", None), "mode", None) is not None
            else None,
            "effective_is_studio_2026": bool(restorer.is_studio_mode()) if "restorer" in locals() else None,
            "pre_analysis_primary_material": pre_primary_material,
            "pre_analysis_transfer_chain": list(pre_transfer_chain),
            "pipeline_transfer_chain": list(pipeline_transfer_chain),
            "era_decade": pre_era_decade,
            "checkpoints": [cp.to_dict() for cp in self.checkpoints],
            "final_musical_goals": dict(self._final_musical_goals),
            "final_vocal_metrics": dict(self._final_vocal_metrics),
            "holistic_perceptual_gate": dict(self._final_hpg),
            "carrier_chain_recovery_ratio": self._final_carrier_chain_recovery_ratio,
            "anomalies": self.anomalies_detected,
            "summary": self._generate_summary(),
        }
        # §Fix-Blocker4: quality_status auch auf Top-Ebene propagieren (bisher nur in "summary").
        result_dict["quality_status"] = result_dict["summary"].get("quality_status")  # type: ignore[union-attr]

        result_file = Path(output_dir) / f"analysis_{Path(audio_path).stem}_{mode}_{int(time.time())}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, default=str)
        self.logger.info("✓ Ergebnisse gespeichert: %s", result_file)

        return result_dict

    def _collect_checkpoints_from_restore_result(self, restoration_result: Any, pre_result: Any) -> None:
        """Build checkpoints from UV3 debug-trace entries after restoration."""
        metadata = getattr(restoration_result, "metadata", {}) or {}
        entries = metadata.get("pmgg_log_entries") or []
        if not isinstance(entries, list):
            return

        # UV3 stores HPI under metadata["holistic_perceptual_gate"]["hpi"] (not flat "hpi_score")
        _hpg = metadata.get("holistic_perceptual_gate") or {}
        _hpg = _hpg if isinstance(_hpg, dict) else {}
        # §G-HPI-ROOT: More robust HPI fallback chain
        _hpi_raw = _hpg.get("hpi")
        if _hpi_raw is None:
            _hpi_raw = (
                (metadata.get("holistic_perceptual_gate") or {}).get("hpi")
                if isinstance(metadata.get("holistic_perceptual_gate"), dict)
                else None
            )
        if _hpi_raw is None:
            _hpi_raw = metadata.get("hpi_score")
        final_hpi = self._metric_to_float(
            _hpi_raw,
            ("hpi", "hpi_score", "score", "value"),
        )
        # UV3 stores AFG as dict: metadata["artifact_freedom"]["score"] or inside holistic_perceptual_gate
        _afg_raw = metadata.get("artifact_freedom")
        if isinstance(_afg_raw, dict):
            _afg_val = _afg_raw.get("score", _afg_raw.get("artifact_freedom"))
        elif _afg_raw is not None:
            _afg_val = _afg_raw
        else:
            _afg_val = _hpg.get("artifact_freedom") or metadata.get("artifact_freedom_score")
        final_afg = self._metric_to_float(
            _afg_val,
            ("artifact_freedom", "artifact_freedom_score", "score", "value"),
        )
        final_ccr = self._metric_to_float(
            metadata.get("carrier_chain_recovery_ratio"),
            ("carrier_chain_recovery_ratio", "score", "value"),
        )

        # §2.44 HPI-Komponenten — aus holistic_perceptual_gate (nur nach finaler HPG-Auswertung bekannt)
        final_mert = self._metric_to_float(_hpg.get("mert_similarity"), ("mert_similarity", "score", "value"))
        final_timbral = self._metric_to_float(_hpg.get("timbral_fidelity"), ("timbral_fidelity", "score", "value"))
        final_eap = self._metric_to_float(
            _hpg.get("emotional_arc_preservation"), ("emotional_arc_preservation", "score", "value")
        )
        # VQI liegt separat in metadata["vqi"] (§2.35c)
        final_vqi = self._metric_to_float(metadata.get("vqi"), ("vqi", "score", "value"))
        # Material-adaptives HPI-Ceiling (§0k) — aus metadata wenn vorhanden
        final_hpi_ceiling = self._metric_to_float(
            metadata.get("hpi_material_ceiling"), ("hpi_material_ceiling", "ceiling", "score", "value")
        )

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            phase_id = str(entry.get("phase_id") or "unknown_phase")
            scores_before = entry.get("scores_before") if isinstance(entry.get("scores_before"), dict) else {}
            scores_after = entry.get("scores_after") if isinstance(entry.get("scores_after"), dict) else {}

            # If PMGG entry already carries phase-local metrics, prefer those over final run metrics.
            # Fallback to final values keeps backward compatibility for older traces.
            phase_hpi = self._extract_phase_metric(
                entry,
                direct_keys=("hpi", "hpi_score", "phase_hpi"),
                nested_paths=(("metadata", "phase_hpi_proxy"), ("holistic_perceptual_gate", "hpi")),
                fallback=final_hpi,
                preferred_keys=("hpi", "hpi_score", "score", "value"),
            )
            phase_afg = self._extract_phase_metric(
                entry,
                direct_keys=("artifact_freedom", "artifact_freedom_score", "afg_score"),
                nested_paths=(
                    ("metadata", "phase_artifact_freedom_proxy"),
                    ("artifact_freedom", "score"),
                    ("holistic_perceptual_gate", "artifact_freedom"),
                ),
                fallback=final_afg,
                preferred_keys=("artifact_freedom", "artifact_freedom_score", "score", "value"),
            )

            anomalies = self._check_anomalies_from_scores(
                phase_id,
                scores_before,  # type: ignore[arg-type]
                scores_after,  # type: ignore[arg-type]
                phase_hpi,
                phase_afg,
                pre_result,
                str(entry.get("action") or ""),
            )
            # HPI-Komponenten werden als finale Werte an alle Checkpoints angehängt;
            # intermediate Phasen erhalten None, da HPG erst am Ende ausgewertet wird.
            _is_last_entry = entry is entries[-1]
            cp = PhaseCheckpoint(
                phase_id=phase_id,
                wall_time_s=float(entry.get("timestamp") or time.time()),
                musical_goals={k: float(v) for k, v in scores_after.items() if isinstance(v, (int, float))},  # type: ignore[union-attr]
                hpi_score=phase_hpi,
                artifact_freedom=phase_afg,
                carrier_recovery_ratio=final_ccr,
                noise_floor_db=None,
                defects_remaining=None,
                anomalies=anomalies,
                # §2.44 Komponenten nur im finalen Checkpoint — verhindert Verwechslung mit per-Phase-Werten
                vqi=final_vqi if _is_last_entry else None,
                mert_similarity=final_mert if _is_last_entry else None,
                timbral_fidelity=final_timbral if _is_last_entry else None,
                emotional_arc_preservation=final_eap if _is_last_entry else None,
                hpi_ceiling=final_hpi_ceiling if _is_last_entry else None,
            )
            self.checkpoints.append(cp)

    def _extract_phase_metric(
        self,
        entry: dict[str, Any],
        direct_keys: tuple[str, ...],
        nested_paths: tuple[tuple[str, ...], ...],
        fallback: float | None,
        preferred_keys: tuple[str, ...],
    ) -> float | None:
        """Extract a phase-local metric from PMGG log entry with graceful fallback."""
        for key in direct_keys:
            value = entry.get(key)
            scalar = self._metric_to_float(value, preferred_keys)
            if scalar is not None:
                return scalar

        for path in nested_paths:
            cursor: Any = entry
            for key in path:
                if not isinstance(cursor, dict):
                    cursor = None
                    break
                cursor = cursor.get(key)
            scalar = self._metric_to_float(cursor, preferred_keys)
            if scalar is not None:
                return scalar

        return fallback

    def _check_anomalies_from_scores(
        self,
        _phase_id: str,
        scores_before: dict[str, Any],
        scores_after: dict[str, Any],
        hpi: Any,
        afg: Any,
        pre_result: Any,
        action: str,
    ) -> list[str]:
        """Detect anomalies from PMGG debug-trace score snapshots."""
        anomalies: list[str] = []
        hpi_value = self._metric_to_float(hpi, ("hpi", "hpi_score", "score", "value"))
        afg_value = self._metric_to_float(afg, ("artifact_freedom", "artifact_freedom_score", "score", "value"))
        for goal, after in scores_after.items():
            before = scores_before.get(goal)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                delta = float(after) - float(before)
                if delta < -0.10:
                    anomalies.append(f"{goal}: {float(before):.2f} → {float(after):.2f} (Δ={delta:.2f})")

        if str(action).startswith("best_effort"):
            anomalies.append(f"PMGG best_effort: {action}")

        if hpi_value is not None and hpi_value < 0.1:
            anomalies.append(f"HPI kritisch niedrig: {hpi_value:.3f}")

        if afg_value is not None and afg_value < 0.85:
            anomalies.append(f"Artefakt-Freiheit kritisch: {afg_value:.3f}")

        if pre_result and hasattr(pre_result, "noise_floor_estimate"):
            if pre_result.noise_floor_estimate > -50.0:
                anomalies.append(f"Rauschboden hoch: {pre_result.noise_floor_estimate:.1f} dBFS")

        return anomalies

    @staticmethod
    def _metric_to_float(metric: Any, preferred_keys: tuple[str, ...] = ("score", "value")) -> float | None:
        """Coerce scalar-like metric values (including dict payloads) to float."""
        if isinstance(metric, (int, float)):
            return float(metric)
        if isinstance(metric, dict):
            for key in preferred_keys:
                value = metric.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            for value in metric.values():
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    def _extract_vocal_metrics_from_metadata(self, metadata: dict[str, Any]) -> dict[str, float]:
        """Extract final vocal metrics (VQI/identity) from restoration metadata."""
        if not isinstance(metadata, dict):
            return {}

        out: dict[str, float] = {}
        _vqi = self._metric_to_float(metadata.get("vqi"), ("vqi", "score", "value"))
        if _vqi is not None:
            out["vqi"] = _vqi

        _sid = self._metric_to_float(
            metadata.get("singer_identity_cosine"),
            ("singer_identity_cosine", "score", "value"),
        )
        if _sid is not None:
            out["singer_identity_cosine"] = _sid

        _hpg = metadata.get("holistic_perceptual_gate")
        if isinstance(_hpg, dict):
            _mert = self._metric_to_float(_hpg.get("mert_similarity"), ("mert_similarity", "score", "value"))
            if _mert is not None:
                out["mert_similarity"] = _mert

        return out

    @staticmethod
    def _extract_transfer_chain_from_obj(data: Any) -> list[str]:
        """Extract transfer chain from nested dict/object payloads."""
        if data is None:
            return []
        if isinstance(data, dict):
            direct = data.get("transfer_chain") or data.get("source_fidelity_transfer_chain")
            if isinstance(direct, list):
                return [str(item) for item in direct if isinstance(item, str) and item]
            for value in data.values():
                chain = ContinuousDeepAnalyzer._extract_transfer_chain_from_obj(value)
                if chain:
                    return chain
            return []
        direct = getattr(data, "transfer_chain", None) or getattr(data, "source_fidelity_transfer_chain", None)
        if isinstance(direct, list):
            return [str(item) for item in direct if isinstance(item, str) and item]
        for attr in ("__dict__",):
            nested = getattr(data, attr, None)
            if isinstance(nested, dict):
                chain = ContinuousDeepAnalyzer._extract_transfer_chain_from_obj(nested)
                if chain:
                    return chain
        return []

    @staticmethod
    def _extract_era_decade_from_obj(data: Any) -> int | None:
        """Extract era decade from nested dict/object payloads."""
        if data is None:
            return None
        if isinstance(data, dict):
            for key in ("era_decade", "decade"):
                val = data.get(key)
                if isinstance(val, (int, np.integer)):
                    return int(val)
            for value in data.values():
                era_val = ContinuousDeepAnalyzer._extract_era_decade_from_obj(value)
                if era_val is not None:
                    return era_val
            return None
        for attr in ("era_decade", "decade"):
            val = getattr(data, attr, None)
            if isinstance(val, (int, np.integer)):
                return int(val)
        nested = getattr(data, "__dict__", None)
        if isinstance(nested, dict):
            return ContinuousDeepAnalyzer._extract_era_decade_from_obj(nested)
        return None

    def _create_checkpoint(self, phase_id: str, restorer: Any, pre_result: Any) -> PhaseCheckpoint:
        """Erstellt einen Checkpoint nach einer Phase."""
        # Musical Goals auslesen
        goals: dict[str, float] = {}
        _restorer_state = vars(restorer) if hasattr(restorer, "__dict__") else {}
        _goals_raw = _restorer_state.get("_musical_goals_results")
        if isinstance(_goals_raw, dict):
            goals = dict(_goals_raw)

        # HPI, AFG, etc. auslesen
        hpi = None
        afg = None
        ccr = None
        noise_db = None
        defects_left = None

        if isinstance(_restorer_state.get("_hpi_score"), (int, float)):
            hpi = float(_restorer_state["_hpi_score"])
        if isinstance(_restorer_state.get("_artifact_freedom_score"), (int, float)):
            afg = float(_restorer_state["_artifact_freedom_score"])
        if isinstance(_restorer_state.get("_carrier_chain_recovery_ratio"), (int, float)):
            ccr = float(_restorer_state["_carrier_chain_recovery_ratio"])

        # §2.44 HPI-Komponenten aus phase_metadata_accumulator (Realtime-Pfad)
        _pma = _restorer_state.get("_phase_metadata_accumulator") or {}
        vqi = self._metric_to_float(_pma.get("vqi"), ("vqi", "score", "value"))
        mert_sim = None
        timbral = None
        eap = None
        hpi_ceiling = self._metric_to_float(
            _restorer_state.get("_hpi_material_ceiling"), ("hpi_material_ceiling", "ceiling", "score", "value")
        )
        # HPG-Komponenten sind erst nach finaler HPG-Auswertung bekannt
        _hpg_state = _restorer_state.get("_last_hpg_result")
        if isinstance(_hpg_state, dict):
            mert_sim = self._metric_to_float(_hpg_state.get("mert_similarity"), ("mert_similarity", "score", "value"))
            timbral = self._metric_to_float(_hpg_state.get("timbral_fidelity"), ("timbral_fidelity", "score", "value"))
            eap = self._metric_to_float(
                _hpg_state.get("emotional_arc_preservation"), ("emotional_arc_preservation", "score", "value")
            )

        # Anomalien prüfen
        anomalies = self._check_anomalies(phase_id, goals, hpi, afg, pre_result)

        return PhaseCheckpoint(
            phase_id=phase_id,
            wall_time_s=time.monotonic(),
            musical_goals=goals,
            hpi_score=hpi,
            artifact_freedom=afg,
            carrier_recovery_ratio=ccr,
            noise_floor_db=noise_db,
            defects_remaining=defects_left,
            anomalies=anomalies,
            vqi=vqi,
            mert_similarity=mert_sim,
            timbral_fidelity=timbral,
            emotional_arc_preservation=eap,
            hpi_ceiling=hpi_ceiling,
        )

    def _check_anomalies(
        self, _phase_id: str, goals: dict[str, float], hpi: float | None, afg: float | None, pre_result: Any
    ) -> list[str]:
        """Erkennt Anomalien in Phase-Ergebnissen."""
        anomalies: list[str] = []

        # Musical Goals Regressions
        if self.checkpoints:
            last_cp = self.checkpoints[-1]
            for goal, score in goals.items():
                last_score = last_cp.musical_goals.get(goal, 1.0)
                delta = score - last_score
                if delta < -0.10:  # > 10% Regression
                    anomalies.append(f"{goal}: {last_score:.2f} → {score:.2f} (Δ={delta:.2f})")

        # HPI Drop
        if hpi is not None and hpi < 0.1:
            anomalies.append(f"HPI kritisch niedrig: {hpi:.3f}")

        # Artefakt-Explosion
        if afg is not None and afg < 0.85:
            anomalies.append(f"Artefakt-Freiheit kritisch: {afg:.3f}")

        # Rauschboden steigende
        if pre_result and hasattr(pre_result, "noise_floor_estimate"):
            if pre_result.noise_floor_estimate > -50.0:
                anomalies.append(f"Rauschboden hoch: {pre_result.noise_floor_estimate:.1f} dBFS")

        return anomalies

    def _print_checkpoint_summary(self, cp: PhaseCheckpoint) -> None:
        """Druckt Checkpoint als Tabelle."""
        self.logger.info("\n  Phase: %s", cp.phase_id)
        self.logger.info("  - Musical Goals: ", extra={"no_newline": True})
        for goal, score in list(cp.musical_goals.items())[:3]:
            status = "✓" if score >= 0.80 else "✗" if score < 0.50 else "~"
            self.logger.info(" %s%s=%.2f", status, goal, score, extra={"no_newline": True})
        self.logger.info("")
        if cp.hpi_score is not None:
            self.logger.info("  - HPI: %.3f", cp.hpi_score)
        if cp.artifact_freedom is not None:
            self.logger.info("  - AFG: %.3f", cp.artifact_freedom)
        if cp.anomalies:
            for anom in cp.anomalies:
                self.logger.info("  ⚠ ANOMALIE: %s", anom)

    def _generate_summary(self) -> dict[str, Any]:
        """Generiert eine Zusammenfassung."""
        if not self.checkpoints:
            return {"status": "no_checkpoints"}

        last_cp = self.checkpoints[-1]
        p1_goals = ["natuerlichkeit", "authentizitaet"]
        _summary_goals = self._final_musical_goals if self._final_musical_goals else last_cp.musical_goals
        p1_scores = [float(_summary_goals.get(g, 0.0)) for g in p1_goals]
        p1_avg = np.mean(p1_scores) if p1_scores else 0.0
        hpi_value = self._metric_to_float(last_cp.hpi_score, ("hpi", "hpi_score", "score", "value"))
        # §G-HPI-GATE: Fallback über final_hpg wenn checkpoint kein HPI hat
        if hpi_value is None and self._final_hpg:
            hpi_value = self._metric_to_float(self._final_hpg.get("hpi"), ("hpi", "hpi_score", "score", "value"))
        afg_value = self._metric_to_float(
            last_cp.artifact_freedom,
            ("artifact_freedom", "artifact_freedom_score", "score", "value"),
        )

        quality_gate_reasons: list[str] = []
        if hpi_value is not None and hpi_value < 0.60:
            quality_gate_reasons.append(f"hpi<{0.60:.2f}")
        if afg_value is not None and afg_value < 0.95:
            quality_gate_reasons.append(f"artifact_freedom<{0.95:.2f}")

        if quality_gate_reasons:
            quality_status = "NEEDS_REVIEW"
        else:
            quality_status = "EXCELLENT" if p1_avg >= 0.90 else "GOOD" if p1_avg >= 0.80 else "NEEDS_REVIEW"

        final_vqi = self._metric_to_float(self._final_vocal_metrics.get("vqi"), ("vqi", "score", "value"))
        final_singer_id = self._metric_to_float(
            self._final_vocal_metrics.get("singer_identity_cosine"),
            ("singer_identity_cosine", "score", "value"),
        )

        _hpg_components: dict[str, object] = {}
        if self._final_hpg:
            for _hpg_key in (
                "mert_similarity",
                "timbral_fidelity",
                "artifact_freedom",
                "emotional_arc_preservation",
                "vqi",
                "mert_proxy_used",
                "is_studio_mode",
            ):
                if _hpg_key in self._final_hpg:
                    _hpg_components[_hpg_key] = self._final_hpg[_hpg_key]

        return {
            "total_phases": len(self.checkpoints),
            "total_anomalies": len(self.anomalies_detected),
            "final_hpi": last_cp.hpi_score,
            "final_artifact_freedom": last_cp.artifact_freedom,
            "final_vqi": final_vqi,
            "final_singer_identity_cosine": final_singer_id,
            "p1_avg_score": float(p1_avg),
            "p1_source": "final_musical_goals" if self._final_musical_goals else "pmgg_debug_trace",
            "quality_status": quality_status,
            "quality_gate_reasons": quality_gate_reasons,
            "carrier_chain_recovery_ratio": self._final_carrier_chain_recovery_ratio,
            **_hpg_components,
        }


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Parse CLI args, run continuous deep analysis, and return process exit code."""
    parser = argparse.ArgumentParser(description="Aurik 10.0.0 — Kontinuierliche Tiefenanalyse")
    parser.add_argument("--audio", type=str, default=None, help="Audio-Datei-Pfad")
    parser.add_argument(
        "--mode",
        type=str,
        default="restoration",
        choices=["restoration", "studio_2026"],
        help="Restaurierungs-Modus",
    )
    parser.add_argument("--realtime", action="store_true", help="Echtzeit-Dashboard aktivieren")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Output-Verzeichnis")

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("analysis_runtime.log"),
        ],
    )

    # Standard-Audio wenn nicht angegeben
    if not args.audio:
        default_audio = (
            Path("test_audio") / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
        )
        if default_audio.exists():
            args.audio = str(default_audio)
        else:
            print("✗ Keine Audio-Datei angegeben und keine Standard-Datei gefunden")
            sys.exit(1)

    # Analyzer ausführen
    analyzer = ContinuousDeepAnalyzer(realtime=args.realtime)
    result = analyzer.run_analysis(
        audio_path=args.audio,
        mode=args.mode,
        output_dir=args.output_dir,
    )

    # Finale Ausgabe
    summary = result.get("summary", {})
    print("\n" + "=" * 80)
    print("ANALYSE ABSCHLUSS")
    print("=" * 80)
    print(f"Status: {summary.get('quality_status', 'UNKNOWN')}")
    print(f"P1 Durchschnitt: {summary.get('p1_avg_score', 0.0):.2f}")
    print(f"Anomalien erkannt: {len(result.get('anomalies', []))}")
    if result.get("anomalies"):
        print("\nAnomalien:")
        for anom in result["anomalies"]:
            print(f"  - {anom}")

    return 0 if summary.get("quality_status") == "EXCELLENT" else 1


if __name__ == "__main__":
    sys.exit(main())
