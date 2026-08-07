"""
backend/core/metadata_aggregator.py — Vollständiger Metadaten-Aggregator (§v10.11)
===================================================================================

Sammelt und strukturiert ALLE Metadaten während der Pipeline-Ausführung.
7 neue Felder für maximale Transparenz und Selbstkalibrierung.

Felder:
    1. per_phase_hpe_delta     — HPE-Änderung pro Phase
    2. calibration_drift       — Preset-Abweichung + Outcome
    3. phase_group_quality     — Qualität pro Pipeline-Stufe
    4. model_loading_telemetry — Cold-Start-Analyse
    5. frequency_band_improvement — SNR pro Frequenzband
    6. processing_story        — Narrative Zusammenfassung
    7. comparative_benchmark   — Perzentil-Ranking
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_BENCHMARK_DIR = Path.home() / ".aurik" / "benchmarks"
_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Frequenzbänder für SNR-Analyse (ISO 266)
_FREQ_BANDS: dict[str, tuple[float, float]] = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mids": (250, 500),
    "mids": (500, 2000),
    "high_mids": (2000, 4000),
    "highs": (4000, 8000),
    "air": (8000, 16000),
}

# Pipeline-Stufen-Zuordnung
_PHASE_GROUP_MAP: dict[str, str] = {
    "01": "repair",
    "02": "repair",
    "03": "repair",
    "05": "repair",
    "06": "restoration",
    "07": "restoration",
    "09": "repair",
    "12": "transport",
    "14": "stereo",
    "16": "mastering",
    "17": "mastering",
    "19": "vocal",
    "20": "spatial",
    "23": "restoration",
    "24": "repair",
    "25": "transport",
    "26": "dynamics",
    "29": "repair",
    "31": "transport",
    "34": "stereo",
    "36": "dynamics",
    "37": "restoration",
    "38": "restoration",
    "39": "restoration",
    "40": "mastering",
    "42": "vocal",
    "43": "vocal",
    "47": "mastering",
    "49": "spatial",
    "50": "restoration",
    "53": "analysis",
    "54": "dynamics",
    "55": "restoration",
    "56": "restoration",
    "57": "repair",
    "58": "vocal",
    "59": "repair",
    "60": "repair",
    "61": "repair",
    "62": "stereo",
    "63": "repair",
    "64": "repair",
    "65": "vocal",
    "66": "mastering",
}


@dataclass
class PhaseHpeRecord:
    phase_id: str = ""
    hpe_before: float = 0.0
    hpe_after: float = 0.0
    delta: float = 0.0
    time_s: float = 0.0
    group: str = "general"


@dataclass
class CalibrationDrift:
    preset_name: str = ""
    strength_delta: float = 0.0
    hpe_outcome: float = 0.0
    user_rating: int = 0
    success: bool = False


@dataclass
class ModelTelemetry:
    model_name: str = ""
    load_ms: float = 0.0
    ram_gb: float = 0.0
    first_inference_ms: float = 0.0


@dataclass
class FreqBandImprovement:
    band: str = ""
    before_snr_db: float = 0.0
    after_snr_db: float = 0.0
    delta_db: float = 0.0


class MetadataAggregator:
    """Zentraler Metadaten-Sammler für die gesamte Pipeline."""

    def __init__(self, material: str = "unknown", era: int | None = None) -> None:
        self._lock = threading.Lock()
        self.material = material
        self.era = era

        # Feld 1: Per-Phase HPE
        self.phase_hpe_deltas: list[PhaseHpeRecord] = []

        # Feld 2: Calibration Drift
        self.calibration_drift: CalibrationDrift | None = None

        # Feld 3: Phase Group Quality
        self.group_quality: dict[str, dict[str, float]] = {}

        # Feld 4: Model Telemetry
        self.model_telemetry: list[ModelTelemetry] = []

        # Feld 5: Frequency Band Improvement
        self.freq_improvements: list[FreqBandImprovement] = []

        # Feld 6: Processing Story
        self.story_headline: str = ""
        self.story_highlights: list[str] = []

        # Feld 7: Benchmark
        self.benchmark_percentile: int = 50
        self.benchmark_total: int = 0

        # Intern
        self._start_time = time.perf_counter()
        self._total_defects: int = 0
        self._repaired_defects: int = 0
        self._phase_count: int = 0
        self._hpe_baseline: float = 0.0
        self._final_hpe: float = 0.0

    # ── Feld 1: Per-Phase HPE Delta ─────────────────────────────────

    def record_phase_hpe(
        self,
        phase_id: str,
        hpe_before: float,
        hpe_after: float,
        time_s: float = 0.0,
    ) -> None:
        """Zeichnet HPE-Delta für eine Phase auf."""
        _num = phase_id.split("_")[1] if "_" in phase_id else "99"
        _group = _PHASE_GROUP_MAP.get(_num, "general")
        _delta = hpe_after - hpe_before

        with self._lock:
            self.phase_hpe_deltas.append(
                PhaseHpeRecord(
                    phase_id=phase_id,
                    hpe_before=hpe_before,
                    hpe_after=hpe_after,
                    delta=_delta,
                    time_s=time_s,
                    group=_group,
                )
            )
            self._phase_count += 1

            # Group-Quality kumulieren (Feld 3)
            if _group not in self.group_quality:
                self.group_quality[_group] = {
                    "hpe_sum": 0.0,
                    "count": 0,
                    "time_s": 0.0,
                    "positive": 0,
                    "negative": 0,
                }
            _gq = self.group_quality[_group]
            _gq["hpe_sum"] += _delta
            _gq["count"] += 1
            _gq["time_s"] += time_s
            if _delta > 0:
                _gq["positive"] += 1
            elif _delta < 0:
                _gq["negative"] += 1

    # ── Feld 2: Calibration Drift ───────────────────────────────────

    def set_calibration_drift(
        self,
        preset_name: str,
        strength_delta: float,
        hpe_outcome: float,
        user_rating: int = 0,
    ) -> None:
        """Speichert die Abweichung vom Preset."""
        with self._lock:
            self.calibration_drift = CalibrationDrift(
                preset_name=preset_name,
                strength_delta=strength_delta,
                hpe_outcome=hpe_outcome,
                user_rating=user_rating,
                success=hpe_outcome >= self._hpe_baseline,
            )

    # ── Feld 3: Phase Group Quality ─────────────────────────────────

    def get_group_quality(self) -> dict[str, dict[str, float]]:
        """Gibt normalisierte Qualitätswerte pro Gruppe zurück."""
        with self._lock:
            result = {}
            for group, data in self.group_quality.items():
                _n = max(data["count"], 1)
                result[group] = {
                    "avg_hpe_delta": round(data["hpe_sum"] / _n, 4),
                    "phase_count": data["count"],
                    "total_time_s": round(data["time_s"], 1),
                    "positive_ratio": round(data["positive"] / _n, 3),
                    "quality_score": round(
                        np.clip(0.5 + data["hpe_sum"] / max(_n, 1) * 5.0, 0.0, 1.0),
                        3,
                    ),
                }
            return result

    # ── Feld 4: Model Loading Telemetry ─────────────────────────────

    def record_model_load(
        self,
        model_name: str,
        load_ms: float,
        ram_gb: float = 0.0,
        first_inference_ms: float = 0.0,
    ) -> None:
        """Zeichnet Modell-Ladezeiten auf."""
        with self._lock:
            self.model_telemetry.append(
                ModelTelemetry(
                    model_name=model_name,
                    load_ms=load_ms,
                    ram_gb=ram_gb,
                    first_inference_ms=first_inference_ms,
                )
            )

    # ── Feld 5: Frequency Band Improvement ──────────────────────────

    def compute_freq_improvements(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        sr: int,
    ) -> None:
        """Berechnet SNR-Verbesserung pro Frequenzband."""
        try:
            _mono_before = audio_before.mean(axis=0) if audio_before.ndim == 2 else audio_before
            _mono_after = audio_after.mean(axis=0) if audio_after.ndim == 2 else audio_after
            _min_len = min(len(_mono_before), len(_mono_after), sr * 30)
            _before = _mono_before[:_min_len].astype(np.float64)
            _after = _mono_after[:_min_len].astype(np.float64)

            _n_fft = 4096
            _spec_before = np.abs(np.fft.rfft(_before, n=_n_fft))
            _spec_after = np.abs(np.fft.rfft(_after, n=_n_fft))
            _freqs = np.fft.rfftfreq(_n_fft, 1.0 / sr)

            with self._lock:
                self.freq_improvements.clear()
                for band_name, (f_lo, f_hi) in _FREQ_BANDS.items():
                    _mask = (_freqs >= f_lo) & (_freqs <= f_hi)
                    if not np.any(_mask):
                        continue
                    _before_energy = float(np.mean(_spec_before[_mask]))
                    _after_energy = float(np.mean(_spec_after[_mask]))
                    _before_db = float(20.0 * np.log10(_before_energy + 1e-10))
                    _after_db = float(20.0 * np.log10(_after_energy + 1e-10))
                    _delta = _after_db - _before_db
                    self.freq_improvements.append(
                        FreqBandImprovement(
                            band=band_name,
                            before_snr_db=_before_db,
                            after_snr_db=_after_db,
                            delta_db=_delta,
                        )
                    )
        except Exception as exc:
            logger.debug("Freq improvement computation fehlgeschlagen: %s", exc)

    # ── Feld 6: Processing Story ────────────────────────────────────

    def generate_story(self, material_display: str = "", era_display: str = "") -> str:
        """Generiert narrative Zusammenfassung."""
        _elapsed = time.perf_counter() - self._start_time
        _mins = int(_elapsed // 60)
        _secs = int(_elapsed % 60)

        # Highlights sammeln
        highlights: list[str] = []

        # HPE-Trend
        _deltas = [r.delta for r in self.phase_hpe_deltas]
        _positive = sum(1 for d in _deltas if d > 0)
        if _deltas:
            _hpe_trend = "verbessert" if _positive > len(_deltas) / 2 else "stabil gehalten"
            highlights.append(f"HPE in {_positive}/{len(_deltas)} Phasen {_hpe_trend}")

        # Beste Phase
        if _deltas:
            _best = max(self.phase_hpe_deltas, key=lambda r: r.delta)
            if _best.delta > 0:
                highlights.append(f"Beste Phase: {_best.phase_id} (+{_best.delta:+.3f} HPE)")

        # Frequenz-Verbesserung
        _pos_bands = [f for f in self.freq_improvements if f.delta_db > 0.5]
        if _pos_bands:
            _top = max(_pos_bands, key=lambda f: f.delta_db)
            highlights.append(f"Größte Verbesserung: {_top.band} (+{_top.delta_db:+.1f} dB)")

        # Modell-Telemetrie
        if self.model_telemetry:
            _total_ram = sum(m.ram_gb for m in self.model_telemetry)
            highlights.append(f"{len(self.model_telemetry)} ML-Modelle ({_total_ram:.1f} GB RAM)")

        # Defekte
        if self._repaired_defects > 0:
            highlights.append(f"{self._repaired_defects} Defekte repariert")

        self.story_headline = (
            f"{material_display or self.material} restauriert — "
            f"{len(self.phase_hpe_deltas)} Phasen in {_mins}:{_secs:02d} min"
        )
        self.story_highlights = highlights

        return self.story_headline

    # ── Feld 7: Comparative Benchmark ───────────────────────────────

    def compute_benchmark(self) -> dict[str, Any]:
        """Vergleicht mit historischen Restaurationen."""
        _key = hashlib.md5(f"{self.material}:{self.era or 0}".encode(), usedforsecurity=False).hexdigest()[:8]

        _this_hpe = self._final_hpe or (self.phase_hpe_deltas[-1].hpe_after if self.phase_hpe_deltas else 0.7)

        # Historische Daten laden
        _history: list[float] = []
        try:
            _path = _BENCHMARK_DIR / f"{self.material}.json"
            if _path.exists():
                _data = json.loads(_path.read_text())
                _history = _data.get("hpe_scores", [])
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        # Perzentil berechnen
        if _history:
            _better = sum(1 for h in _history if h < _this_hpe)
            _percentile = int(_better / len(_history) * 100)
        else:
            _percentile = 50

        self.benchmark_percentile = _percentile
        self.benchmark_total = len(_history) + 1

        # Aktuellen Score speichern
        try:
            _path = _BENCHMARK_DIR / f"{self.material}.json"
            _data: dict = {}  # type: ignore[no-redef]
            if _path.exists():
                _data = json.loads(_path.read_text())
            _scores = _data.get("hpe_scores", [])
            _scores.append(round(_this_hpe, 4))
            if len(_scores) > 1000:
                _scores = _scores[-1000:]
            _data["hpe_scores"] = _scores
            _data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _path.write_text(json.dumps(_data, indent=2))
        except Exception as exc:
            logger.debug("Benchmark speichern fehlgeschlagen: %s", exc)

        return {
            "percentile": _percentile,
            "total_compared": len(_history) + 1,
            "avg_hpe_for_material": round(np.mean(_history), 3) if _history else 0.0,
            "this_hpe": round(_this_hpe, 4),
            "material": self.material,
        }

    # ── Finalize ────────────────────────────────────────────────────

    def set_baseline(self, hpe: float, total_defects: int = 0) -> None:
        self._hpe_baseline = hpe
        self._total_defects = total_defects

    def set_final(self, hpe: float, repaired: int = 0) -> None:
        self._final_hpe = hpe
        self._repaired_defects = repaired

    def to_dict(self) -> dict[str, Any]:
        """Exportiert alle gesammelten Metadaten."""
        return {
            "per_phase_hpe_delta": [
                {
                    "phase_id": r.phase_id,
                    "hpe_before": round(r.hpe_before, 4),
                    "hpe_after": round(r.hpe_after, 4),
                    "delta": round(r.delta, 4),
                    "time_s": round(r.time_s, 2),
                    "group": r.group,
                }
                for r in self.phase_hpe_deltas
            ],
            "calibration_drift": (
                {
                    "preset_name": self.calibration_drift.preset_name,
                    "strength_delta": round(self.calibration_drift.strength_delta, 4),
                    "hpe_outcome": round(self.calibration_drift.hpe_outcome, 4),
                    "success": self.calibration_drift.success,
                }
                if self.calibration_drift
                else None
            ),
            "phase_group_quality": self.get_group_quality(),
            "model_loading_telemetry": [
                {
                    "model": m.model_name,
                    "load_ms": round(m.load_ms, 0),
                    "ram_gb": round(m.ram_gb, 3),
                    "first_inference_ms": round(m.first_inference_ms, 0),
                }
                for m in self.model_telemetry
            ],
            "frequency_band_improvement": [
                {
                    "band": f.band,
                    "before_snr_db": round(f.before_snr_db, 1),
                    "after_snr_db": round(f.after_snr_db, 1),
                    "delta_db": round(f.delta_db, 1),
                }
                for f in self.freq_improvements
            ],
            "processing_story": {
                "headline": self.story_headline,
                "highlights": self.story_highlights,
                "phase_count": self._phase_count,
                "material": self.material,
            },
            "comparative_benchmark": self.compute_benchmark(),
        }
