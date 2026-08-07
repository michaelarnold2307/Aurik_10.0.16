"""Consensus-Balance-Synthesizer (CBS) — §DGWCS, v10.0.0.

Dimension-Gap-Weighted Consensus Synthesis: Kombiniert bestehende Checkpoint-Referenzen
analytisch-optimal in einem einzigen, mathematisch begründeten globalen Alpha,
das die Imbalance über alle Perceptual-Dimensionen simultan minimiert.

Kein neues DSP. Keine neuen Datenquellen. Vollständig monoton:
psycho_score(output) >= psycho_score(input) oder kein Eingriff (Invariante).

Grundsatz: **Keine fiktiven Anchor-Werte.** Alle Dimensions-Werte der Referenz-Kandidaten
werden direkt aus dem Audio gemessen (3 schnelle DSP-Funktionen: Rauschtextur-Kohärenz,
Mikrodynamik-Korrelation, Spektralfarbe-Korrelation). Nur für `emotional_arc_preservation`
wird der aktuelle Pipeline-Wert als konservativer Fallback verwendet (MDEM zu aufwändig
für per-Kandidaten-Neuauswertung).

Physikalisches Prinzip:
    Gemessene Anchor-Dims → analytisch-optimales α in O(n_dims) ohne Brute-Force.
    Das kleinste α ≥ _ALPHA_MIN, das alle Per-Dim-Floors einhält und eine messbare
    Score-Verbesserung ≥ _MIN_IMPROVEMENT liefert.

Einschränkungen:
    - Nur Post-Pipeline (nach _execute_pipeline, vor RestorationResult-Export).
    - §0h: Strukturelle Stille-Zonen werden nie verändert.
    - §0a: Kein Phase-Aktivierungs-Bypass — reine Audio-Blend auf Array-Ebene.
    - artifact_freedom-Invariante: kein Eingriff wenn artifact_freedom < 0.95.
    - Alpha-Floor _ALPHA_MIN = 0.70: maximal 30 % Referenzanteil (Minimal-Intervention).
    - Zeit-Budget _MAX_CBS_SECONDS = 3.0: keine Messung über Limit hinaus.

Kanonische Nutzung (UV3, nach _evaluate_psychoacoustic_naturalness_gate):
    from backend.core.consensus_balance_synthesizer import (
        apply_dgwcs, compute_updated_vector_after_dgwcs
    )
    blended, result = apply_dgwcs(
        current_audio=restored_audio,
        original_audio=original_audio_for_goals,
        references={"hpi_best": ..., "carrier": ..., "original": ...},
        current_wcs_vector=dict(_wcs_vector_input),
        artifact_freedom=_artifact_freedom_for_hpi,
        panns_singing=..., sr=sample_rate, material=...,
        structural_silence_zones=...,
    )
    if result.applied:
        restored_audio = blended
        _wcs_vector_input = compute_updated_vector_after_dgwcs(_wcs_vector_input, result)

Exportierte API:
    apply_dgwcs(...)                        → (audio, CBSResult)
    compute_updated_vector_after_dgwcs(...) → dict
    measure_candidate_dims(...)             → dict | None  (für UV3-Psycho-Recovery)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# §DGWCS-Schwellwerte
_ALPHA_MIN: float = 0.70  # maximal 30 % Referenzanteil
_MIN_IMPROVEMENT: float = 0.005  # Mindest-Psycho-Score-Verbesserung für Eingriff
_MAX_CBS_SECONDS: float = 3.0  # Gesamt-Zeitbudget für alle Kandidatenmessungen

# Psycho-Naturalness-Gewichte (kanonisch aus UV3 §8.6g)
_PSYCHO_WEIGHTS: dict[str, float] = {
    "noise_texture_authenticity": 0.28,
    "micro_dynamic_correlation": 0.24,
    "emotional_arc_preservation": 0.24,
    "spectral_color_preservation": 0.24,
}


@dataclass
class CBSResult:
    """Ergebnis der Dimension-Gap-Weighted Consensus Synthesis (§DGWCS).

    Attributes:
        applied:                True wenn Blend angewendet wurde.
        reference_used:         Name der gewählten Referenz ("hpi_best"/"carrier"/"original").
        alpha:                  Blend-Gewicht der aktuellen Audio (1.0 = kein Eingriff).
        psycho_score_before:    Psycho-Score vor DGWCS.
        psycho_score_after:     Analytisch geschätzter Psycho-Score nach DGWCS.
        improvement:            psycho_score_after - psycho_score_before.
        gap_closure_per_dim:    Pro Dimension: geschlossener Gap (positiv = Verbesserung).
        silence_zones_protected: Anzahl Samples, die durch §0h-Stille-Maske geschützt wurden.
    """

    applied: bool
    reference_used: str = ""
    alpha: float = 1.0
    psycho_score_before: float = 0.0
    psycho_score_after: float = 0.0
    improvement: float = 0.0
    gap_closure_per_dim: dict[str, float] = field(default_factory=dict)
    silence_zones_protected: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialisiert für metadata-Telemetrie."""
        return {
            "applied": self.applied,
            "reference_used": self.reference_used,
            "alpha": round(float(self.alpha), 4),
            "psycho_score_before": round(float(self.psycho_score_before), 4),
            "psycho_score_after": round(float(self.psycho_score_after), 4),
            "improvement": round(float(self.improvement), 6),
            "gap_closure_per_dim": {k: round(float(v), 4) for k, v in self.gap_closure_per_dim.items()},
            "silence_zones_protected": int(self.silence_zones_protected),
        }


def measure_candidate_dims(
    original: np.ndarray,
    candidate: np.ndarray,
    sr: int,
    material: str,
    quality_mode: str,
    emotional_arc_fallback: float,
    *,
    _deadline: float | None = None,
) -> dict[str, float] | None:
    """Misst die 4 Psycho-Naturalness-Dimensionen eines Audio-Kandidaten gegen das Original.

    Nur schnelle DSP-Funktionen — kein ML. Budget: typisch < 0.5 s/Kandidat.
    `emotional_arc_preservation` wird nicht gemessen (MDEM zu aufwändig), sondern
    der aktuelle Pipeline-Wert als konservativer Fallback übernommen.
    Für 'original'-Kandidaten wird emotional_arc = 1.0 gesetzt (definitional).

    Args:
        original:              Degradierter Input (Referenz für Vergleich).
        candidate:             Zu messende Checkpoint-Variante.
        sr:                    Sample-Rate (erwartet 48000 Hz für DSP-Funktionen).
        material:              Materialklasse für Noise-Texture-Coherence.
        quality_mode:          'restoration' oder 'studio_2026'.
        emotional_arc_fallback: Aktueller Pipeline-Wert als Fallback [0,1].
        _deadline:             Optionales time.time()-Deadline (bricht Messung ab).

    Returns:
        Dict mit 4 gemessenen Dimensionen oder None bei Fehler / incompatiblen Shapes.
    """
    if not isinstance(original, np.ndarray) or not isinstance(candidate, np.ndarray):
        return None
    if original.shape != candidate.shape or original.size < 512:
        return None
    if sr != 48000:
        logger.debug("measure_candidate_dims: sr=%d != 48000, übersprungen.", sr)
        return None

    _t0 = time.monotonic()
    _failed: list[str] = []
    dims: dict[str, float] = {}

    # ── 1. Noise-Texture-Authentizität ────────────────────────────────────────
    # Residual = original - candidate → wie gut behält der Kandidat das Träger-
    # Rauschprofil des Originals? Coherence = 1 → materialkonforme Rauschstruktur.
    # Sonderfall: Null-Residual (candidate ≡ original) → perfekte Textur-Bewahrung = 1.0.
    # (compute_noise_texture_coherence liefert 0.0 für Null-Vektor, da Pearson(0,ref) = 0.)
    try:
        _residual_test = (original.mean(axis=0) if original.ndim == 2 else original) - (
            candidate.mean(axis=0) if candidate.ndim == 2 else candidate
        )
        if float(np.abs(_residual_test).max()) < 1e-7:
            # Kandidat identisch mit Original → alle Rausch-Textur bewahrt
            dims["noise_texture_authenticity"] = 1.0
        else:
            from backend.core.noise_texture_coherence import (
                get_noise_texture_coherence_guard as _get_ntcg,
            )

            _nt_result = _get_ntcg().check_end_of_pipeline(original, candidate, sr, material, quality_mode)
            dims["noise_texture_authenticity"] = float(np.clip(float(_nt_result.coherence), 0.0, 1.0))
    except Exception as _e:
        logger.debug("CBS: noise_texture Messung fehlgeschlagen: %s", _e)
        _failed.append("noise_texture_authenticity")

    # ── 2. Mikrodynamik-Korrelation ───────────────────────────────────────────
    # Pearson-Korrelation der Frame-Energien auf Voiced-Zonen.
    # Misst wie gut der Kandidat die Mikrodynamik der Original-Performance bewahrt.
    if _deadline is None or time.monotonic() < _deadline:
        try:
            from backend.core.dsp.mikrodynamik_guard import (
                frame_energy_correlation as _fec,
            )

            dims["micro_dynamic_correlation"] = float(np.clip(float(_fec(original, candidate, sr)), 0.0, 1.0))
        except Exception as _e:
            logger.debug("CBS: micro_dynamic Messung fehlgeschlagen: %s", _e)
            _failed.append("micro_dynamic_correlation")
    else:
        _failed.append("micro_dynamic_correlation")

    # ── 3. Spektralfarbe-Korrelation ──────────────────────────────────────────
    # 1/3-Oktav-Pearson-Korrelation (200–8000 Hz).
    # Misst wie gut der Kandidat die tonale Charakteristik des Originals bewahrt.
    if _deadline is None or time.monotonic() < _deadline:
        try:
            from backend.core.dsp.spectral_color_guard import (
                check_spectral_color_preservation as _scp,
            )

            _sc = _scp(original, candidate, sr)
            dims["spectral_color_preservation"] = float(np.clip(float(_sc.correlation), 0.0, 1.0))
        except Exception as _e:
            logger.debug("CBS: spectral_color Messung fehlgeschlagen: %s", _e)
            _failed.append("spectral_color_preservation")
    else:
        _failed.append("spectral_color_preservation")

    # ── 4. Emotionaler Bogen ──────────────────────────────────────────────────
    # MDEM-Neuauswertung pro Kandidat ist zu aufwändig.
    # Definitional: 'original' IS die Referenz → emotional_arc = 1.0.
    # Alle anderen: konservativer Fallback aus aktuellem Pipeline-Wert.
    dims["emotional_arc_preservation"] = float(np.clip(float(emotional_arc_fallback), 0.0, 1.0))

    # Wenn mehr als 1 Kerndimension nicht messbar → nicht zuverlässig → None.
    if len(_failed) >= 2:
        logger.debug(
            "CBS: %d Dimensionen nicht messbar (%s) → Kandidat wird übersprungen.",
            len(_failed),
            ", ".join(_failed),
        )
        return None

    # Fehlgeschlagene Einzelmessung: aktueller Wert (0.80) als konservativer Boden.
    for _k in _failed:
        dims[_k] = 0.80
        logger.debug("CBS: %s Messung fehlgeschlagen → konservativer Ersatzpfad 0.80.", _k)

    _elapsed = time.monotonic() - _t0
    logger.debug(
        "CBS: Kandidaten-Messung %.3fs — noise_tex=%.3f micro_dyn=%.3f spectral=%.3f arc=%.3f",
        _elapsed,
        dims.get("noise_texture_authenticity", 0.0),
        dims.get("micro_dynamic_correlation", 0.0),
        dims.get("spectral_color_preservation", 0.0),
        dims.get("emotional_arc_preservation", 0.0),
    )
    return {k: float(v) for k, v in dims.items()}


def _score_from_dims(dims: dict[str, float], weights: dict[str, float]) -> float:
    """Gewichteter Score aus Dimension-Werten."""
    return float(
        sum(float(weights.get(m, 0.0)) * float(np.clip(float(v), 0.0, 1.0)) for m, v in dims.items() if m in weights)
    )


def _compute_optimal_alpha(
    current_dims: dict[str, float],
    anchor_dims: dict[str, float],
    weights: dict[str, float],
    floor: float,
) -> float | None:
    """Berechnet das optimale Alpha analytisch aus GEMESSENEN Anchor-Dims.

    score(α) = α × score(current) + (1-α) × score(anchor) ist linear in α.
    Optimales α = kleinstes α ≥ _ALPHA_MIN das:
      a) alle Per-Dim-Floor-Constraints erfüllt (keine bereits-guten Dim verschlechtern)
      b) eine Verbesserung ≥ _MIN_IMPROVEMENT liefert

    Per-Dim-Floor-Constraint (schützend, nicht fördernd):
      Wenn A_m < floor UND C_m >= floor:
        α × C_m + (1-α) × A_m ≥ floor  →  α ≥ (floor - A_m) / (C_m - A_m)
      Wenn A_m < C_m (Referenz ist schlechter für diese Dim):
        Kein Eingriff in dieser Dim möglich ohne Verschlechterung →
        Falls diese Dim die Gesamtverbesserung nicht überwiegt, wird Alpha
        angehoben sodass die Verschlechterung minimal bleibt.

    Returns:
        Optimales Alpha ∈ [_ALPHA_MIN, 1.0) oder None bei keiner Verbesserung.
    """
    anchor_score = _score_from_dims(anchor_dims, weights)
    current_score = _score_from_dims(current_dims, weights)

    # Referenz muss einen ausreichenden Vorteil bieten
    if anchor_score <= current_score + _MIN_IMPROVEMENT * 0.5:
        return None

    alpha_min_eff = _ALPHA_MIN
    for m, w in weights.items():
        if float(w) <= 0.0:
            continue
        A_m = float(np.clip(float(anchor_dims.get(m, 1.0)), 0.0, 1.0))
        C_m = float(np.clip(float(current_dims.get(m, 1.0)), 0.0, 1.0))

        # Schutz-Constraint I: bereits-gute Dim darf nicht unter Floor fallen
        if A_m < floor and C_m >= floor:
            denom = C_m - A_m
            if denom > 1e-9:
                alpha_constraint = (floor - A_m) / denom
                alpha_min_eff = max(alpha_min_eff, float(alpha_constraint))

        # Schutz-Constraint II: Referenz schlechter in dieser Dim →
        # Alpha muss hoch genug sein, dass die absolute Verschlechterung
        # ≤ 5 % des Dim-Werts bleibt (kein hörbar relevanter Verlust).
        # α × C_m + (1-α) × A_m ≥ C_m × 0.95  →  α ≥ (0.95 × C_m - A_m) / (C_m - A_m)
        if A_m < C_m and C_m > 0.01:
            denom2 = C_m - A_m
            if denom2 > 1e-9:
                alpha_no_harm = (0.95 * C_m - A_m) / denom2
                if alpha_no_harm > 0.0:
                    alpha_min_eff = max(alpha_min_eff, float(alpha_no_harm))

    optimal_alpha = float(np.clip(alpha_min_eff, _ALPHA_MIN, 1.0 - 1e-6))
    if not math.isfinite(optimal_alpha):
        return None

    # Verbesserung bei gewähltem Alpha verifizieren
    blend_score = optimal_alpha * current_score + (1.0 - optimal_alpha) * anchor_score
    if blend_score <= current_score + _MIN_IMPROVEMENT:
        return None

    return optimal_alpha


def _build_silence_mask(
    n_samples: int,
    silence_zones: list[tuple[float, float]],
    sr: int,
    ndim: int,
) -> np.ndarray | None:
    """Erstellt §0h-Stille-Schutz-Maske (1.0 = blenden, 0.0 = Original behalten).

    Returns None wenn keine Stille-Zonen oder alle außerhalb des Signals.
    """
    if not silence_zones:
        return None

    mask = np.ones(n_samples, dtype=np.float32)
    for zone in silence_zones:
        try:
            start_s, end_s = float(zone[0]), float(zone[1])
            i_start = max(0, int(start_s * sr))
            i_end = min(n_samples, int(end_s * sr))
            if i_end > i_start:
                mask[i_start:i_end] = 0.0
        except Exception as _cbs_zone_exc:
            logger.debug(
                "consensus_balance_synthesizer: zone mask computation fehlgeschlagen (unkritisch): %s", _cbs_zone_exc
            )
            continue

    if float(mask.min()) > 0.999:
        return None  # Keine Zone im Signal-Bereich

    if ndim == 2:
        return mask[np.newaxis, :]  # type: ignore[no-any-return]  # (1, N) für (C, N)-Arrays
    return mask  # type: ignore[no-any-return]  # (N,) für Mono


def apply_dgwcs(
    current_audio: np.ndarray,
    original_audio: np.ndarray,
    references: dict[str, np.ndarray],
    current_wcs_vector: dict[str, float],
    artifact_freedom: float,
    panns_singing: float,
    sr: int,
    material: str = "digital",
    quality_mode: str = "restoration",
    structural_silence_zones: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, CBSResult]:
    """Dimension-Gap-Weighted Consensus Synthesis (§DGWCS).

    Findet analytisch-optimal den Checkpoint, der die Psycho-Naturalness-Imbalance
    am meisten reduziert, und blendet minimal-invasiv.

    Alle Kandidaten-Dimensionen werden GEMESSEN (3 DSP-Funktionen), nicht geschätzt.
    Nur `emotional_arc_preservation` nutzt den aktuellen Pipeline-Wert als Fallback.
    Für den 'original'-Kandidaten wird emotional_arc = 1.0 gesetzt (definitional).

    Monotonie-Invariante: psycho_score(output) >= psycho_score(input) immer.
    Kein Eingriff wenn: artifact_freedom < 0.95, keine Verbesserung ≥ _MIN_IMPROVEMENT,
    oder keine kompatible Referenz, oder Zeit-Budget überschritten.

    Args:
        current_audio:           Aktuelles restauriertes Audio (float32, (N,) oder (2, N)).
        original_audio:          Degradierter Input (Referenz für Dimensionsmessung).
        references:              Checkpoint-Varianten {"hpi_best"|"carrier"|"original": audio}.
        current_wcs_vector:      §8.6a HTEV-Vektor mit Psycho-Dimensionen.
        artifact_freedom:        Aktueller artifact_freedom-Score (§2.49 Veto-Faktor).
        panns_singing:           PANNs Gesangs-Score (0–1).
        sr:                      Sample-Rate (48000 Hz erwartet).
        material:                Materialklasse für Noise-Texture-Coherence.
        quality_mode:            'restoration' oder 'studio_2026'.
        structural_silence_zones: §2.68 SSIP Stille-Zonen (§0h-Schutz).

    Returns:
        (output_audio, CBSResult) — output_audio ist identisch mit current_audio
        wenn CBSResult.applied = False.
    """

    def _no_op() -> tuple[np.ndarray, CBSResult]:
        return current_audio, CBSResult(applied=False)

    # Sicherheitsprüfungen
    if not isinstance(current_audio, np.ndarray) or current_audio.size == 0:
        return _no_op()
    if not isinstance(original_audio, np.ndarray) or original_audio.size == 0:
        return _no_op()
    if not math.isfinite(float(artifact_freedom)) or float(artifact_freedom) < 0.95:
        return _no_op()  # §0h artifact_freedom-Invariante
    if not references:
        return _no_op()
    if sr != 48000:
        return _no_op()  # DSP-Funktionen erfordern 48 kHz

    # Psycho-Dimensionen aus aktuellem Vektor extrahieren
    current_dims: dict[str, float] = {
        m: float(np.clip(float(current_wcs_vector.get(m, 1.0)), 0.0, 1.0)) for m in _PSYCHO_WEIGHTS
    }
    _current_emotional_arc = float(current_dims.get("emotional_arc_preservation", 0.90))

    # Material-adaptiver Floor: Vokal strenger (§0p)
    _panns = float(np.clip(float(panns_singing), 0.0, 1.0))
    _floor = 0.80 if _panns >= 0.35 else 0.76

    current_psycho = _score_from_dims(current_dims, _PSYCHO_WEIGHTS)

    # Zeitbudget
    _t_start = time.monotonic()
    _deadline = _t_start + _MAX_CBS_SECONDS

    # ── Pro Kandidat: Dimensionen MESSEN, dann alpha analytisch berechnen ─────
    best_ref: str = ""
    best_alpha: float = 1.0
    best_improvement: float = 0.0
    best_anchor_dims: dict[str, float] = {}

    for ref_name, ref_audio in references.items():
        if time.monotonic() >= _deadline:
            logger.debug("CBS: Zeit-Grenze erschöpft, verbleibende Kandidaten übersprungen.")
            break
        if not isinstance(ref_audio, np.ndarray):
            continue
        if ref_audio.shape != current_audio.shape:
            continue

        # Emotionalen Bogen für diesen Kandidaten setzen
        _ea_fallback = 1.0 if ref_name == "original" else _current_emotional_arc

        # Dimensionen MESSEN (nicht schätzen)
        anchor_dims = measure_candidate_dims(
            original=original_audio,
            candidate=ref_audio,
            sr=sr,
            material=material,
            quality_mode=quality_mode,
            emotional_arc_fallback=_ea_fallback,
            _deadline=_deadline,
        )
        if anchor_dims is None:
            logger.debug("CBS: Kandidat '%s' nicht messbar → übersprungen.", ref_name)
            continue

        optimal_alpha = _compute_optimal_alpha(
            current_dims=current_dims,
            anchor_dims=anchor_dims,
            weights=_PSYCHO_WEIGHTS,
            floor=_floor,
        )
        if optimal_alpha is None:
            continue

        anchor_psycho = _score_from_dims(anchor_dims, _PSYCHO_WEIGHTS)
        blend_psycho = optimal_alpha * current_psycho + (1.0 - optimal_alpha) * anchor_psycho
        improvement = blend_psycho - current_psycho

        if improvement > best_improvement + 1e-7:
            best_improvement = improvement
            best_ref = ref_name
            best_alpha = optimal_alpha
            best_anchor_dims = anchor_dims

    if not best_ref or best_improvement < _MIN_IMPROVEMENT:
        return _no_op()

    ref_audio = references[best_ref]
    anchor_dims = best_anchor_dims

    # Audio-Blend (float32-Präzision)
    _cur = np.asarray(current_audio, dtype=np.float32)
    _ref = np.asarray(ref_audio, dtype=np.float32)
    blended = np.clip(
        best_alpha * _cur + (1.0 - best_alpha) * _ref,
        -1.0,
        1.0,
    ).astype(np.float32)

    # §0h [RELEASE_MUST] Stille-Zonen-Schutz
    n_samples = int(current_audio.shape[-1])
    ndim = int(current_audio.ndim)
    silence_mask = _build_silence_mask(
        n_samples=n_samples,
        silence_zones=list(structural_silence_zones or []),
        sr=int(sr),
        ndim=ndim,
    )
    n_protected = 0
    if silence_mask is not None:
        blended = np.where(silence_mask.astype(bool), blended, _cur)
        n_protected = int(np.sum(silence_mask == 0.0))

    blended = np.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)

    # Gap-Closure pro Dimension berechnen (Telemetrie)
    gap_closure: dict[str, float] = {}
    blend_score_after = 0.0
    for m, w in _PSYCHO_WEIGHTS.items():
        C_m = float(current_dims.get(m, 1.0))
        A_m = float(np.clip(float(anchor_dims.get(m, 1.0)), 0.0, 1.0))
        B_m = float(np.clip(best_alpha * C_m + (1.0 - best_alpha) * A_m, 0.0, 1.0))
        gap_closure[m] = round(B_m - C_m, 4)
        blend_score_after += float(w) * B_m

    result = CBSResult(
        applied=True,
        reference_used=best_ref,
        alpha=round(float(best_alpha), 4),
        psycho_score_before=round(float(current_psycho), 4),
        psycho_score_after=round(float(blend_score_after), 4),
        improvement=round(float(blend_score_after - current_psycho), 6),
        gap_closure_per_dim=gap_closure,
        silence_zones_protected=n_protected,
    )

    return blended, result


def compute_updated_vector_after_dgwcs(
    current_vector: dict[str, float],
    dgwcs_result: CBSResult,
) -> dict[str, float]:
    """Analytisch aktualisierter WCS-Vektor nach DGWCS-Blend.

    Konsistent mit der Blend-Formel: V_blend[m] = α × V_current[m] + (1-α) × anchor[m].
    Verwendet die gemessenen anchor_dims aus CBSResult.gap_closure_per_dim
    (rückgerechnet: A_m = (V_blend[m] - α × C_m) / (1 - α)).
    Nur Psycho-Naturalness-Dimensionen werden aktualisiert; alle anderen Keys bleiben.

    Für UV3-interne Gate-Neuauswertung nach DGWCS-Eingriff.
    """
    if not dgwcs_result.applied:
        return dict(current_vector)

    float(np.clip(float(dgwcs_result.alpha), 0.0, 1.0))
    updated = dict(current_vector)

    for m in _PSYCHO_WEIGHTS:
        C_m = float(np.clip(float(current_vector.get(m, 1.0)), 0.0, 1.0))
        # gap_closure = B_m - C_m = (1-α) × (A_m - C_m)
        # → B_m = C_m + gap_closure_per_dim[m]
        delta = float(dgwcs_result.gap_closure_per_dim.get(m, 0.0))
        updated[m] = float(np.clip(C_m + delta, 0.0, 1.0))

    return updated
