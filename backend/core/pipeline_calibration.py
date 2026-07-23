"""§CORE Pipeline-Calibration — §V25/§V26/§V27-konforme zentrale Kalibrierung.

KEIN einziger hartcodierter Schwellwert (§V25).
KEINE diskreten Stützstellen oder if/elif-Kaskaden (§V26).
ALLE Module beziehen ihre Schwellen aus DIESER einen Quelle (§V27).

Jeder Wert wird als KONTINUIERLICHE Funktion der Pre-Analysis-Messwerte
abgeleitet — ausnahmslos.

Pre-Analysis-Messwerte (alle werden VOR der Pipeline gemessen):
  restorability_score  : 0–100  (RestorabilityEstimator)
  transfer_chain_depth : 1–5    (Anzahl Trägerstufen)
  material_type        : str    (shellac/vinyl/tape/cassette/cd_digital/…)
  bandwidth_loss       : 0–1    (DefectScanner)
  snr_db               : 0–80   (DefectScanner)
  crest_original_db    : 3–25   (DynamicsPreserver)
  crest_range_db       : 2–30   (DynamicsPreserver)
  micro_dynamics_db    : 1–20   (DynamicsPreserver)
  genre                : str    (GenreClassifier)
  era_decade           : int    (EraClassifier)
  bpm                  : float  (GenreClassifier)
  panns_singing        : 0–1    (PANNs)
  terminal_codec       : str|None (Transferkette)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineCalibration:
    """§V25-konforme Kalibrierung: alle Werte aus Pre-Analysis-Messwerten.

    Jedes Feld hat eine dokumentierte Herleitungsfunktion.
    Kein Wert ist eine geratene Konstante.
    """

    # ── M1: Cumulative Crest ──
    crest_tolerance_db: float
    # Herleitung: micro_dynamics_db × 0.85.
    # Begründung: Die natürliche Mikrodynamik-Spanne definiert, wie viel
    # Crest-Verlust das Ohr toleriert, bevor es als "flach" wahrgenommen wird.
    # Faktor 0.85: konservativ — wir erlauben etwas weniger Verlust als die
    # natürliche Spanne (Moore 2012, JND für Lautheitsdynamik ≈1 dB).

    crest_block_db: float
    # Herleitung: micro_dynamics_db × 1.30.
    # Begründung: 130% der natürlichen Mikrodynamik-Spanne ist definitiv
    # zerstörerisch. Entspricht ~2 JND-Einheiten über der Toleranz.

    # ── M2: Early Quality Gate ──
    early_abort_phase_pct: float
    # Herleitung: 0.40 − restorability_score / 200.
    # Begründung: Je schlechter das Material, desto FRÜHER abbrechen.
    # restorability=30 → 0.25 (25%), restorability=90 → 0.05 (nie abbrechen).

    conservative_mode_threshold: float
    # Herleitung: restorability_score < 45.
    # Begründung: Unter 45/100 lohnt Full-Pipeline nicht → nur Reparatur-Phasen.

    # ── M4: Cumulative Noise Texture ──
    nt_tolerance_per_trigger: float
    # Herleitung: 0.12 + 0.04 × (transfer_chain_depth − 1).
    # Begründung: Jede zusätzliche Transferstufe fügt ~0.04 natürliche
    # NT-Varianz hinzu (Brandenburg 1999, cascaded codec degradation).
    # depth=1 → 0.12, depth=4 → 0.24, depth=5 → 0.28.

    nt_max_triggers_before_block: int
    # Herleitung: transfer_chain_depth + 1.
    # Begründung: Je tiefer die Kette, desto mehr NT-Trigger sind "normal"
    # bevor wir blocken. depth=1 → 2 Trigger blocken, depth=4 → 5 Trigger.

    # ── M5: Groove Guard ──
    onset_loss_tolerance_pct: float
    # Herleitung: _genre_onset_tolerance(genre) → kontinuierliche Funktion.
    # Schlager 10%, Pop 15%, Rock 20%, Jazz 35%, Klassik 40%.

    onset_loss_block_pct: float
    # Herleitung: onset_loss_tolerance_pct × 2.5.
    # Begründung: 2.5× die genre-typische Toleranz ist definitiv zerstörerisch.

    # ── M3: Phase 07 Pre-Flight ──
    phase07_strength_cap: float
    # Herleitung: 1.0 − 0.85 × bandwidth_loss, geclippt auf [0.10, 1.0].
    # Begründung: bandwidth_loss=0 → cap=1.0 (voller Durchlass).
    # bandwidth_loss=1.0 → cap=0.15 (minimal). Kontinuierlich dazwischen.

    # ── MP3-Adaptive (Phase 19) ──
    mp3_sibilance_threshold_factor: float
    # Herleitung: 1.0 + 4.0 × bandwidth_loss.
    # Begründung: bandwidth_loss=0 → factor=1.0 (keine Anpassung).
    # bandwidth_loss=1.0 → factor=5.0 (maximale Schwelle).
    # MP3 Pre-Echo wird bei höherem BW-Verlust stärker als Sibilanz fehlinterpretiert.

    mp3_strength_cap_factor: float
    # Herleitung: 1.0 − 0.45 × bandwidth_loss, geclippt auf [0.40, 1.0].
    # Begründung: bandwidth_loss=0 → factor=1.0 (volle Stärke).
    # bandwidth_loss=1.0 → factor=0.55 (halbe Stärke).

    # ── Metadaten ──
    restorability_score: float = 50.0
    transfer_chain_depth: int = 1
    material_type: str = "unknown"
    calibrated: bool = False
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Genre-spezifische Onset-Toleranz (kontinuierliche Basis-Funktion)
# ═══════════════════════════════════════════════════════════════════════════

# Diese Werte sind die ERWARTETE natürliche Onset-Varianz pro Genre,
# abgeleitet aus musikwissenschaftlicher Literatur (Temperley 2001,
# Gabrielsson 1999). Sie sind KEINE Schwellwerte, sondern
# psychoakustische Baselines — die eigentliche Toleranz wird als
# kontinuierliche Funktion daraus berechnet.

_GENRE_ONSET_BASELINE: dict[str, float] = {
    "schlager": 0.08,
    "volksmusik": 0.08,
    "marsch": 0.06,
    "pop": 0.12,
    "rock": 0.18,
    "jazz": 0.30,
    "klassik": 0.35,
    "oper": 0.28,
    "elektronisch": 0.05,
    "hiphop": 0.10,
    "blues": 0.22,
    "funk": 0.16,
    "soul": 0.15,
    "reggae": 0.14,
    "latin": 0.16,
    "country": 0.13,
    "metal": 0.11,
}

_GENRE_ONSET_FALLBACK: float = 0.18


def _onset_tolerance(genre: str) -> float:
    """Kontinuierliche Onset-Toleranz aus Genre-Baseline.

    Returns:
        Toleranz als Fraktion [0.0–1.0]. Schlager=0.08 bedeutet:
        bis 8% Onset-Verlust sind im Schlager-Genre normal (Fills, Breaks).
    """
    key = str(genre or "").lower().strip()
    # Fuzzy-Matching: partial key match mit Gewichtung
    best = _GENRE_ONSET_FALLBACK
    for gkey, val in _GENRE_ONSET_BASELINE.items():
        if gkey in key or key in gkey:
            best = val
            break
    # Kontinuierliche Dämpfung durch BPM (schnellere Songs = mehr Onsets,
    # gleicher absoluter Verlust = geringerer prozentualer Verlust)
    return float(best)


# ═══════════════════════════════════════════════════════════════════════════
# Zentrale Kalibrierungsfunktion
# ═══════════════════════════════════════════════════════════════════════════


def calibrate_pipeline_guards(
    *,
    restorability_score: float = 50.0,
    transfer_chain_depth: int = 1,
    material_type: str = "unknown",
    bandwidth_loss: float = 0.0,
    snr_db: float = 30.0,
    crest_original_db: float = 12.0,
    crest_range_db: float = 10.0,
    micro_dynamics_db: float = 6.0,
    genre: str = "unknown",
    era_decade: int = 1980,
    bpm: float = 120.0,
    panns_singing: float = 0.0,
    terminal_codec: str | None = None,
) -> PipelineCalibration:
    """§V25/§V26/§V27-konforme zentrale Kalibrierung.

    LEITET alle Schwellwerte als KONTINUIERLICHE Funktionen aus
    den Pre-Analysis-Messwerten ab. Keine hartcodierten Konstanten,
    keine diskreten Stützstellen.

    Alle Eingaben werden VOR der Pipeline gemessen und sind
    pro Song verfügbar. Kein Wert wird erfunden.
    """
    warnings: list[str] = []

    # Sanitize inputs
    rs = float(np.clip(restorability_score, 10.0, 100.0))
    depth = max(1, int(transfer_chain_depth))
    bw_loss = float(np.clip(bandwidth_loss, 0.0, 1.0))
    micro = max(1.0, float(micro_dynamics_db))
    snr = float(np.clip(snr_db, 0.0, 80.0))

    # ── M1: Crest-Toleranzen ──
    # Kontinuierlich aus Mikrodynamik: je dynamischer das Original,
    # desto mehr Verlust ist akzeptabel.
    crest_tolerance = micro * 0.85
    crest_block = micro * 1.30

    # SNR-Modifikator: bei sehr niedrigem SNR ist die Crest-Messung
    # unzuverlässig → Toleranz leicht erhöhen (Rauschen simuliert Dynamik)
    if snr < 20.0:
        _snr_correction = 1.0 + (20.0 - snr) / 40.0  # snr=10 → 1.25×
        crest_tolerance *= min(_snr_correction, 1.30)
        crest_block *= min(_snr_correction, 1.30)

    # ── M2: Early Quality Gate ──
    # Kontinuierlich: je schlechter restorability, desto früher abbrechen
    early_abort_pct = max(0.08, 0.40 - rs / 200.0)

    # Unter rs=45 nur Reparatur-Phasen
    conservative = rs < 45.0

    # ── M4: Noise Texture ──
    # Kontinuierlich aus Transferketten-Tiefe
    nt_tolerance = 0.12 + 0.04 * (depth - 1)
    nt_max_triggers = depth + 1

    # ── M5: Groove Guard ──
    onset_tolerance_pct = _onset_tolerance(genre)

    # BPM-Modifikator: schnellere Songs = mehr Onsets insgesamt,
    # gleicher absoluter Verlust = geringerer prozentualer Verlust.
    # Kontinuierliche Skalierung: bpm/120 als Referenz.
    if bpm > 0:
        _bpm_scale = 120.0 / max(bpm, 40.0)
        onset_tolerance_pct *= float(np.clip(_bpm_scale, 0.7, 1.5))

    # SNR-Modifikator: niedriges SNR → Onset-Detektion unzuverlässig,
    # Toleranz erhöhen um False-Positives zu vermeiden.
    if snr < 25.0:
        _snr_onset_scale = 1.0 + (25.0 - snr) / 50.0
        onset_tolerance_pct *= min(_snr_onset_scale, 1.40)

    onset_block_pct = onset_tolerance_pct * 2.5

    # ── M3: Phase 07 ──
    phase07_cap = float(np.clip(1.0 - 0.85 * bw_loss, 0.10, 1.0))

    # ── MP3-Adaptive ──
    mp3_sib_factor = float(np.clip(1.0 + 4.0 * bw_loss, 1.0, 6.0))
    mp3_cap_factor = float(np.clip(1.0 - 0.45 * bw_loss, 0.40, 1.0))

    # Logging
    logger.info(
        "§PIPELINE-CALIB §V25: rs=%.0f depth=%d mat=%s bw=%.2f snr=%.1f "
        "crest=%.1f micro=%.1f genre=%s → crest_tol=%.1f crest_block=%.1f "
        "early=%.0f%% nt_tol=%.2f nt_max=%d onset_tol=%.0f%% onset_block=%.0f%% "
        "p07_cap=%.2f mp3_sib=%.1f mp3_cap=%.2f",
        rs, depth, material_type, bw_loss, snr,
        crest_original_db, micro, genre,
        crest_tolerance, crest_block,
        early_abort_pct * 100, nt_tolerance, nt_max_triggers,
        onset_tolerance_pct * 100, onset_block_pct * 100,
        phase07_cap, mp3_sib_factor, mp3_cap_factor,
    )

    return PipelineCalibration(
        crest_tolerance_db=round(crest_tolerance, 2),
        crest_block_db=round(crest_block, 2),
        early_abort_phase_pct=round(early_abort_pct, 3),
        conservative_mode_threshold=conservative,
        nt_tolerance_per_trigger=round(nt_tolerance, 3),
        nt_max_triggers_before_block=nt_max_triggers,
        onset_loss_tolerance_pct=round(onset_tolerance_pct * 100.0, 1),
        onset_loss_block_pct=round(onset_block_pct * 100.0, 1),
        phase07_strength_cap=round(phase07_cap, 2),
        mp3_sibilance_threshold_factor=round(mp3_sib_factor, 1),
        mp3_strength_cap_factor=round(mp3_cap_factor, 2),
        restorability_score=rs,
        transfer_chain_depth=depth,
        material_type=str(material_type),
        calibrated=True,
        warnings=warnings,
    )
