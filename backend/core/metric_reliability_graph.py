"""MetricReliabilityGraph — Laufzeit-Kalibrierung der Proxy-Verlaesslichkeit.

Ziel:
- PMGG-/Phase-Delta-Telemetrie in goal-spezifische Zuverlaessigkeitswerte ueberfuehren
- Kontextsensitiv (Material, Chain-Komplexitaet, Modus, Era)
- Als advisory-only Layer fuer geschlossene Regelkreise (Rescheduler/PMGG)

Sicherheitsprinzip:
- Keine harten Gates werden ersetzt oder ueberstimmt.
- Werte sind rein zusaetzliche Gewichtung und stets auf [0.2, 0.98] begrenzt.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_RELIABILITY = 0.65
_MIN_RELIABILITY = 0.20
_MAX_RELIABILITY = 0.98
_MAX_CONTEXTS = 160
_ALPHA = 0.10  # EWMA Lernrate


@dataclass
class _GoalReliabilityState:
    value: float
    samples: int


class MetricReliabilityGraph:
    """Thread-sicherer Reliability-Graph fuer Goal-Proxys."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = Path(
            os.getenv("AURIK_METRIC_RELIABILITY_PATH", Path.home() / ".aurik" / "metric_reliability_graph.json")
        )
        self._store: dict[str, dict[str, _GoalReliabilityState]] = {}
        self._load()

    def update_from_phase_delta(
        self,
        *,
        phase_id: str,
        goal_deltas: dict[str, float] | None,
        phase_metadata: dict[str, Any] | None,
        material_type: str,
        transfer_chain: list[str] | None,
        is_studio_2026: bool,
        era_decade: int,
    ) -> None:
        """Aktualisiert Zuverlaessigkeit anhand realisierter Phase-Deltas."""
        _deltas = goal_deltas if isinstance(goal_deltas, dict) else {}
        if not _deltas:
            return
        _meta = phase_metadata if isinstance(phase_metadata, dict) else {}
        _context_key = self._make_context_key(
            material_type=material_type,
            transfer_chain=transfer_chain,
            is_studio_2026=is_studio_2026,
            era_decade=era_decade,
        )

        _team_net = float(_meta.get("pmgg_team_net_delta", 0.0) or 0.0)
        _recon_localized = bool(_meta.get("pmgg_reconstruction_localized", False))
        _recon_epistemic = float(_meta.get("pmgg_reconstruction_epistemic_confidence", 0.0) or 0.0)

        with self._lock:
            _ctx = self._store.setdefault(_context_key, {})
            for _goal, _delta_val in _deltas.items():
                try:
                    _delta = float(_delta_val)
                except (TypeError, ValueError):
                    continue

                _evidence = self._delta_to_evidence(_delta)
                if _team_net > 0.0:
                    _evidence = float(np.clip(_evidence + min(_team_net, 0.04) * 1.5, 0.0, 1.0))
                if _recon_localized and _goal in {"natuerlichkeit", "authentizitaet", "transparenz"}:
                    _evidence = float(np.clip(_evidence + 0.08 * max(_recon_epistemic, 0.5), 0.0, 1.0))

                _state = _ctx.get(_goal)
                if _state is None:
                    _state = _GoalReliabilityState(value=_DEFAULT_RELIABILITY, samples=0)
                    _ctx[_goal] = _state

                _state.value = float(
                    np.clip((1.0 - _ALPHA) * _state.value + _ALPHA * _evidence, _MIN_RELIABILITY, _MAX_RELIABILITY)
                )
                _state.samples += 1

            self._evict_if_needed_locked()
            self._save_locked_nonblocking()

        logger.debug(
            "MetricReliabilityGraph Aktualisierung: Verarbeitungsschritt=%s context=%s goals=%d",
            phase_id,
            _context_key,
            len(_deltas),
        )

    def get_goal_reliability(
        self,
        *,
        goal_scores: dict[str, float] | None,
        material_type: str,
        transfer_chain: list[str] | None,
        is_studio_2026: bool,
        era_decade: int,
    ) -> dict[str, float]:
        """Liefert goal-spezifische Reliability-Werte fuer den aktuellen Kontext."""
        _scores = goal_scores if isinstance(goal_scores, dict) else {}
        if not _scores:
            return {}

        _context_key = self._make_context_key(
            material_type=material_type,
            transfer_chain=transfer_chain,
            is_studio_2026=is_studio_2026,
            era_decade=era_decade,
        )
        with self._lock:
            _ctx = self._store.get(_context_key, {})
            _fallback = self._store.get(
                self._make_context_key(
                    material_type="unknown",
                    transfer_chain=[],
                    is_studio_2026=is_studio_2026,
                    era_decade=era_decade,
                ),
                {},
            )
            _result: dict[str, float] = {}
            for _goal, _score_val in _scores.items():
                _state = _ctx.get(_goal) or _fallback.get(_goal)
                _base = float(_state.value) if _state is not None else _DEFAULT_RELIABILITY
                # Score-Randbereiche sind volatiler -> kleine Drossel.
                try:
                    _score = float(np.clip(float(_score_val), 0.0, 1.0))
                except (TypeError, ValueError):
                    _score = 0.5
                _edge_penalty = 0.0
                if _score < 0.25:
                    _edge_penalty = 0.05
                elif _score > 0.92:
                    _edge_penalty = 0.03
                _result[_goal] = float(np.clip(_base - _edge_penalty, _MIN_RELIABILITY, _MAX_RELIABILITY))
            return _result

    def get_blend_weights(
        self,
        *,
        material_type: str,
        transfer_chain: list[str] | None,
        is_studio_2026: bool,
        era_decade: int,
    ) -> tuple[float, float]:
        """Liefert adaptive Blend-Gewichte (base, runtime) fuer Goal-Konfidenzen.

        Cross-run Gatekeeper:
        - wenig Evidenz -> mehr Gewicht auf base-confidence
        - viel konsistente Evidenz -> mehr Gewicht auf runtime-reliability
        """
        _context_key = self._make_context_key(
            material_type=material_type,
            transfer_chain=transfer_chain,
            is_studio_2026=is_studio_2026,
            era_decade=era_decade,
        )
        with self._lock:
            _ctx = self._store.get(_context_key, {})
            if not _ctx:
                return 0.75, 0.25

            _samples = [max(0, int(_state.samples)) for _state in _ctx.values()]
            _values = [float(_state.value) for _state in _ctx.values()]
            _sample_support = float(np.clip((sum(_samples) / max(len(_samples), 1)) / 24.0, 0.0, 1.0))
            _consistency = float(np.clip(1.0 - (float(np.std(_values)) / 0.22), 0.0, 1.0))
            _quality = float(np.clip((float(np.mean(_values)) - 0.45) / 0.45, 0.0, 1.0))

            # Runtime-Gewicht wächst nur bei tragfähiger und stabiler Evidenz.
            _runtime_weight = float(
                np.clip(
                    0.25 + 0.22 * _sample_support + 0.10 * _consistency + 0.03 * _quality,
                    0.25,
                    0.60,
                )
            )
            _base_weight = float(np.clip(1.0 - _runtime_weight, 0.40, 0.75))
            _norm = _base_weight + _runtime_weight
            if _norm <= 1e-9:
                return 0.75, 0.25
            return float(_base_weight / _norm), float(_runtime_weight / _norm)

    @staticmethod
    def _delta_to_evidence(delta: float) -> float:
        if delta >= 0.0:
            return float(np.clip(0.70 + min(delta, 0.12) * 2.2, 0.0, 1.0))
        return float(np.clip(0.45 + max(delta, -0.15) * 1.8, 0.0, 1.0))

    @staticmethod
    def _chain_complexity(transfer_chain: list[str] | None) -> float:
        _chain = [str(v).strip().lower() for v in (transfer_chain or []) if str(v).strip()]
        if not _chain:
            return 0.0
        _length_term = float(np.clip((len(_chain) - 1) / 5.0, 0.0, 1.0))
        _unique_term = float(np.clip((len(set(_chain)) - 1) / 4.0, 0.0, 1.0))
        _analog = {"vinyl", "shellac", "tape", "reel_tape", "cassette", "wire_recording", "wax_cylinder"}
        _lossy = {"mp3_low", "mp3_high", "aac", "streaming", "minidisc"}
        _mix = 1.0 if any(c in _analog for c in _chain) and any(c in _lossy for c in _chain) else 0.0
        return float(np.clip(0.42 * _length_term + 0.36 * _unique_term + 0.22 * _mix, 0.0, 1.0))

    def _make_context_key(
        self,
        *,
        material_type: str,
        transfer_chain: list[str] | None,
        is_studio_2026: bool,
        era_decade: int,
    ) -> str:
        _mode = "studio" if is_studio_2026 else "restoration"
        _mat = str(material_type or "unknown").strip().lower() or "unknown"
        _era_bin = int(np.clip((int(era_decade or 1970) // 10) * 10, 1900, 2030))
        _tcci_bin = int(np.clip(round(self._chain_complexity(transfer_chain) * 4.0), 0, 4))
        return f"{_mode}|{_mat}|era{_era_bin}|tcci{_tcci_bin}"

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            for _ctx, _goals in payload.items():
                if not isinstance(_goals, dict):
                    continue
                _ctx_map: dict[str, _GoalReliabilityState] = {}
                for _goal, _entry in _goals.items():
                    if not isinstance(_entry, dict):
                        continue
                    _val = float(
                        np.clip(float(_entry.get("value", _DEFAULT_RELIABILITY)), _MIN_RELIABILITY, _MAX_RELIABILITY)
                    )
                    _samples = int(max(0, int(_entry.get("samples", 0))))
                    _ctx_map[str(_goal)] = _GoalReliabilityState(value=_val, samples=_samples)
                if _ctx_map:
                    self._store[str(_ctx)] = _ctx_map
        except Exception as exc:
            logger.debug("MetricReliabilityGraph.laden nicht blockierend: %s", exc)

    def _save_locked_nonblocking(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                _ctx: {
                    _goal: {"value": round(_state.value, 6), "samples": _state.samples}
                    for _goal, _state in _goals.items()
                }
                for _ctx, _goals in self._store.items()
            }
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, dir=str(self._path.parent), suffix=".tmp"
            ) as tmp:
                json.dump(serializable, tmp, ensure_ascii=True, separators=(",", ":"))
                _tmp_name = tmp.name
            os.replace(_tmp_name, self._path)
        except Exception as exc:
            logger.debug("MetricReliabilityGraph.speichern nicht blockierend: %s", exc)

    def _evict_if_needed_locked(self) -> None:
        if len(self._store) <= _MAX_CONTEXTS:
            return
        # Einfacher LRU-Proxy: Kontexte mit geringster Samplesumme zuerst entfernen.
        _ctx_items = sorted(
            self._store.items(),
            key=lambda kv: sum(_s.samples for _s in kv[1].values()),
        )
        _to_remove = len(self._store) - _MAX_CONTEXTS
        for _idx in range(_to_remove):
            del self._store[_ctx_items[_idx][0]]


_instance: MetricReliabilityGraph | None = None
_instance_lock = threading.Lock()


def get_metric_reliability_graph() -> MetricReliabilityGraph:
    """Thread-safe Singleton-Accessor."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MetricReliabilityGraph()
    return _instance
