"""backend/core/song_calibration_profile.py — §v10.700 I5.

SongCalibrationProfile: Pro-Song-Kalibrierungsprofil aus Pre-Analysis-Daten.

Speichert alle gemessenen und abgeleiteten Eigenschaften eines Songs
VOR dem Pipeline-Start. Wird von Pre-Analysis befüllt und von der
Pipeline als Entscheidungsgrundlage genutzt.

Wiederverwendbar im Batch-Modus: Gleiches Profil = gleiche Parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SongCalibrationProfile:
    """Pro-Song-Kalibrierungsdaten aus der Pre-Analysis.

    Alle Felder sind optional — die Pre-Analysis befüllt was sie messen kann.
    """

    # ── Basis-Metadaten ─────────────────────────────────────────
    song_name: str = ""
    file_path: str = ""
    duration_s: float = 0.0
    sample_rate: int = 48000
    channels: int = 2

    # ── Material & Medium ───────────────────────────────────────
    material_type: str = ""  # vinyl, tape, shellac, digital, ...
    medium_type: str = ""  # LP, 78rpm, cassette, CD, DAW, ...
    era_decade: int = 0  # 1920, 1950, 1980, ...
    transfer_chain_depth: int = 1  # Anzahl Tonträger-Generationen

    # ── Signal-Qualität ─────────────────────────────────────────
    input_snr_db: float = 0.0  # Geschätztes SNR in dB
    peak_db: float = 0.0  # True Peak in dBFS
    rms_db: float = -20.0  # RMS in dBFS
    bandwidth_hz: float = 20000.0  # Effektive Bandbreite (−3dB)
    stereo_correlation: float = 1.0  # L/R-Korrelation

    # ── Defekt-Profil ───────────────────────────────────────────
    max_defect_severity: float = 0.0  # Schwerwiegendster Defekt [0,1]
    defect_scores: dict[str, float] = field(default_factory=dict)
    defect_count: int = 0

    # ── Restaurierbarkeit ───────────────────────────────────────
    restorability_score: float = 50.0  # 0–100
    pipeline_confidence: float = 0.5  # 0–1
    estimated_phases: int = 30  # Geschätzte Anzahl nötiger Phasen

    # ── Genre & Inhalt ──────────────────────────────────────────
    genre_label: str = ""
    genre_confidence: float = 0.0
    has_vocals: bool = False
    language: str = ""

    # ── Kalibrierungs-Parameter ─────────────────────────────────
    global_scalar: float = 1.0  # Globaler Stärke-Multiplikator
    noise_floor_db: float = -80.0  # Geschätztes Noise Floor
    dynamic_range_db: float = 30.0  # Dynamik-Umfang

    # ── FastGoalProxy-Ergebnisse ─────────────────────────────────
    fast_goals: dict[str, float] = field(default_factory=dict)


def build_calibration_profile(
    material_type: str = "",
    restorability_score: float = 50.0,
    input_snr_db: float = 0.0,
    max_defect_severity: float = 0.0,
    pipeline_confidence: float = 0.5,
    era_decade: int = 0,
    defect_scores: dict[str, float] | None = None,
    genre_label: str = "",
    song_name: str = "",
    file_path: str = "",
    bandwidth_hz: float = 20000.0,
) -> SongCalibrationProfile:
    """Baut ein SongCalibrationProfile aus Pre-Analysis-Daten.

    Dies ist die kanonische Factory-Funktion — sie wird von
    UnifiedRestorerV3._build_song_calibration_profile() aufgerufen.
    """
    profile = SongCalibrationProfile(
        song_name=song_name,
        file_path=file_path,
        material_type=material_type,
        era_decade=era_decade,
        input_snr_db=input_snr_db,
        bandwidth_hz=bandwidth_hz,
        max_defect_severity=max_defect_severity,
        defect_scores=defect_scores or {},
        restorability_score=restorability_score,
        pipeline_confidence=pipeline_confidence,
        genre_label=genre_label,
        global_scalar=_compute_global_scalar(restorability_score, input_snr_db, max_defect_severity),
        noise_floor_db=_estimate_noise_floor(input_snr_db, material_type),
        dynamic_range_db=_estimate_dynamic_range(input_snr_db, max_defect_severity),
    )
    return profile


def _compute_global_scalar(
    restorability_score: float,
    input_snr_db: float,
    max_defect_severity: float,
) -> float:
    """Berechnet den globalen Stärke-Multiplikator.

    Gute Restaurierbarkeit → weniger Stärke nötig.
    Schlechtes SNR → mehr Stärke nötig.
    """
    base = 1.0
    # Restorability: 0 (schlecht) → +30%, 100 (gut) → −20%
    base += (100.0 - restorability_score) / 100.0 * 0.3 - 0.2
    # SNR: <10dB → +20%, >30dB → −10%
    if input_snr_db < 10:
        base += 0.2
    elif input_snr_db > 30:
        base -= 0.1
    # Defekt-Schwere: >0.7 → +25%
    if max_defect_severity > 0.7:
        base += 0.25
    return round(max(0.5, min(1.5, base)), 3)


def _estimate_noise_floor(input_snr_db: float, material_type: str) -> float:
    """Schätzt das Noise Floor in dBFS."""
    # Basis: −80 dBFS
    floor = -80.0
    # Material-spezifisch
    if material_type in ("shellac", "78rpm"):
        floor = -50.0  # Sehr lautes Grundrauschen
    elif material_type in ("vinyl",):
        floor = -65.0
    elif material_type in ("tape", "cassette"):
        floor = -60.0
    # SNR-Korrektur
    floor += max(0, 30 - input_snr_db) * 0.5
    return round(floor, 1)


def _estimate_dynamic_range(input_snr_db: float, max_defect_severity: float) -> float:
    """Schätzt den Dynamik-Umfang in dB."""
    dr = max(10.0, input_snr_db * 0.8)
    dr -= max_defect_severity * 10.0
    return round(dr, 1)
