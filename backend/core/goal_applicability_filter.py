"""
backend/core/goal_applicability_filter.py
Aurik 10.0.0 -- Spec §2.32: GoalApplicabilityFilter

Filtert physikalisch nicht messbare Musical Goals heraus.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GoalApplicabilityResult:
    """Spec §2.32"""

    applicable: frozenset[str]
    inapplicable: frozenset[str]
    reasons: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        """Serialisierungsformat f\u00fcr Logging und API."""
        return {
            "applicable": sorted(self.applicable),
            "inapplicable": sorted(self.inapplicable),
            "reasons": dict(self.reasons),
        }


_ALWAYS_APPLICABLE: frozenset[str] = frozenset(
    {
        "natuerlichkeit",
        "authentizitaet",
        "emotionalitaet",
        "transparenz",
        "timbre_authentizitaet",
        "artikulation",
    }
)

_ALL_GOALS: frozenset[str] = frozenset(
    {
        "brillanz",
        "waerme",
        "natuerlichkeit",
        "authentizitaet",
        "emotionalitaet",
        "transparenz",
        "bass_kraft",
        "groove",
        "spatial_depth",
        "timbre_authentizitaet",
        "tonal_center",
        "micro_dynamics",
        "separation_fidelity",
        "artikulation",
        "transient_energie",
        # §0p P0-Goals: nur applicable wenn panns_singing ≥ 0.35
        # (Guard in evaluate() via panns_singing-Parameter)
        "vocal_quality",
        "formant_fidelity",
    }
)

_MONO_MATERIALS = {
    "wax_cylinder",
    "wire_recording",
    "shellac",
    "lacquer_disc",
}

_PHYSICALLY_BASS_LIMITED_MATERIALS = frozenset(
    {
        "wax_cylinder",
        "wire_recording",
        "shellac",
        "lacquer_disc",
        "acoustic_78",
    }
)

# §6.2c BW-Ceiling je Material — Brillanz unerreichbar wenn Ceiling ≤ 8 kHz
_MATERIAL_BW_CEILING_HZ: dict[str, float] = {
    "wax_cylinder": 5000.0,
    "wire_recording": 6000.0,
    "shellac": 8000.0,
    "acoustic_78": 8000.0,
    "lacquer_disc": 12000.0,
    "cassette": 14000.0,  # central definition default (unknown type → conservative)
    "tape": 15000.0,
    "reel_tape": 18000.0,
}

# Für diese Materialien liegt der aufnehmbare Bass-Anteil strukturell unter 0.020
_LOW_BASS_MATERIALS = frozenset({"wax_cylinder", "wire_recording", "acoustic_78"})


class GoalApplicabilityFilter:
    """Spec §2.32: Filtert physikalisch nicht messbare Ziele.

    Deaktivierungs-Regeln:
        SpatialDepthMetric:  Mono-Aufnahme (decade <= 1950 oder mono-Material)
        BrillanzMetric:      Quell-BW < 8 kHz
        TonalCenterMetric:   wax_cylinder (Fix K v10.0.0: SNR-Bedingung entfernt —
                             K-S-Key-Detection ist SNR-invariant gemäß §9.7.11)
        GrooveMetric:        <10 s oder keine Percussion
        MicroDynamicsMetric: <20 s oder stark komprimiert
        SeparationFidelityMetric: Mono oder <2 Instrumente
    """

    ALWAYS_APPLICABLE: frozenset[str] = _ALWAYS_APPLICABLE

    def evaluate(
        self,
        audio: np.ndarray | None = None,
        sr: int = 48000,
        material: str = "unknown",
        era_decade: int | None = None,
        panns_tags: dict[str, float] | None = None,
        flashsr_available: bool = False,
        mode: str = "restoration",
        panns_singing: float = 0.0,
        transfer_chain: list[str] | None = None,
    ) -> GoalApplicabilityResult:
        """Spec §2.32: Wertet aus, welche Goals anwendbar sind.

        Args:
            audio: float32 ndarray
            sr: Sample-Rate
            material: MaterialType-Label
            era_decade: Jahrzehnt (aus EraClassifier)
            panns_tags: PANNs-Tag-Konfidenz Dict
            flashsr_available: ob FlashSR geladen ist

        Returns:
            GoalApplicabilityResult
        """
        inapplicable: dict[str, str] = {}
        _mat_key = str(getattr(material, "value", material) or "unknown").strip().lower()

        # Audio-Basis-Analysen
        duration_s = 0.0
        bw_hz = 20000.0
        is_mono_signal = False
        corr: float = 0.0  # L/R-Korrelation; 0.0 = kein Stereo-Audio verfügbar

        if audio is not None:
            arr = np.nan_to_num(np.asarray(audio, dtype=np.float32))
            if arr.ndim == 2:
                # Normierung auf (channels, samples) — toleriert (N, C) und (C, N)
                if arr.shape[0] > arr.shape[1]:  # (samples, channels) → transponieren
                    arr = arr.T
                duration_s = arr.shape[1] / sr
                # Stereo-Analyse
                if arr.shape[0] >= 2:
                    _std0 = float(np.std(arr[0]))
                    _std1 = float(np.std(arr[1]))
                    if _std0 > 1e-8 and _std1 > 1e-8:
                        _a0 = arr[0] - arr[0].mean()
                        _a1 = arr[1] - arr[1].mean()
                        _n0 = float(np.linalg.norm(_a0))
                        _n1 = float(np.linalg.norm(_a1))
                        corr = float(np.dot(_a0, _a1) / (_n0 * _n1 + 1e-10))
                    else:
                        corr = 1.0 if (_std0 < 1e-8 and _std1 < 1e-8) else 0.0
                    # Schwellwert 0.995 (vorher 0.97 — zu strict: typische
                    # Popmusik mit zentriertem Gesang hat corr ≈ 0.97–0.99
                    # und ist dennoch kein Mono-Signal).
                    is_mono_signal = bool(np.isnan(corr) or corr >= 0.995)
                else:
                    is_mono_signal = True
                mono = arr.mean(axis=0)
            else:
                duration_s = len(arr) / sr
                is_mono_signal = True
                mono = arr

            if len(mono) >= 512:
                spec = np.abs(np.fft.rfft(mono[: min(len(mono), 8192)])) ** 2 + 1e-15
                freqs = np.fft.rfftfreq(min(len(mono), 8192), d=1.0 / sr)

                # Effektive Bandbreite: höchste Frequenz mit Energie > Rauschboden + 20 dB.
                # Vorheriger Algorithmus (99th-Perzentil-Energie-Schwerpunkt) unterschätzte
                # die BW für typische Musik systematisch: Bassenergie dominiert →
                # 99 % der Energie unterhalb ~2–4 kHz → brillanz fälschlich deaktiviert.
                noise_floor_power = float(np.percentile(spec, 10)) + 1e-15
                noise_floor_db = 10.0 * math.log10(noise_floor_power)
                bw_hz = 0.0
                for _bi in range(len(freqs) - 1, 0, -1):
                    _e_db = 10.0 * math.log10(float(spec[_bi]) + 1e-15)
                    if _e_db > noise_floor_db + 18.0:  # 18 dB über Rauschboden
                        bw_hz = float(freqs[_bi])
                        break

        # REGEL: SpatialDepthMetric
        era_mono = era_decade is not None and era_decade <= 1950
        mat_mono = _mat_key in _MONO_MATERIALS
        # §GoalApplicability Mono-Fix: Materialien vor 1960 die in _MONO_MATERIALS sind,
        # werden auch dann als mono behandelt, wenn die Signal-Korrelation < 0.97 liegt
        # (z. B. Schellack mit Raumambience über Stereo-A/D-Wandler).
        mat_era_mono = mat_mono and (era_decade is not None and era_decade <= 1960)
        if era_mono or is_mono_signal or mat_mono or mat_era_mono:
            inapplicable["spatial_depth"] = "Mono-Aufnahme — Raumtiefe nicht messbar."
        elif "spatial_depth" not in inapplicable:
            # §S4 Near-Mono-Codec-Ausschluss: MP3/AAC/MiniDisc-Joint-Stereo zerstört Stereobreite
            # irreversibel bei near-mono Quellen (IACC ≥ 0.88 → score ≤ 0.12).
            # Restoration kann IACC nicht verbessern wenn das Codec-Bitstream kein Stereo
            # enthielt — spatial_depth wäre ein permanentes False-Positive ohne Lösbarkeit.
            _CODEC_JOINT_STEREO_MATS = frozenset({"mp3_low", "mp3_high", "aac", "streaming", "minidisc"})
            _is_near_mono_codec = _mat_key in _CODEC_JOINT_STEREO_MATS or any(
                str(s).strip().lower() in _CODEC_JOINT_STEREO_MATS for s in (transfer_chain or [])
            )
            if _is_near_mono_codec and not np.isnan(corr) and float(corr) >= 0.83:
                inapplicable["spatial_depth"] = (
                    f"Near-Mono-Codec ({_mat_key or 'codec'}, L/R-Korrelation={float(corr):.3f} ≥ 0.83) — "
                    "Joint-Stereo hat Raumtiefe irreversibel reduziert; Restoration kann IACC nicht erhöhen."
                )

        # REGEL: BrillanzMetric
        _mat_bw_ceiling = _MATERIAL_BW_CEILING_HZ.get(_mat_key, 22050.0)
        if _mat_bw_ceiling <= 8000.0:
            # §6.2c: Material-BW-Ceiling ≤ 8 kHz → Brillanz physikalisch unerreichbar nach BW-Cap
            inapplicable["brillanz"] = (
                f"Material '{_mat_key}' hat BW-Ceiling ≤ {_mat_bw_ceiling / 1000:.0f} kHz (§6.2c) — "
                "Brillanz-Threshold nach BW-Hard-Cap unerreichbar (Restoration: nie additiv über Ceiling)."
            )
        elif bw_hz < 8000.0 and not flashsr_available:
            inapplicable["brillanz"] = (
                "Hochfrequenz war nicht aufgezeichnet — Brillanz wird nach Bandbreiten-Erweiterung neu bewertet."
            )

        # REGEL: TonalCenterMetric
        # Fix K (v10.0.0): SNR-Bedingung entfernt — K-S-Key-Detection ist SNR-invariant
        # gemäß §9.7.11; Deaktivierung bei SNR < −5 dB war inkonsistent mit der
        # K-S-Invarianz-Aussage und hätte tonal_center auf stark degradiertem Material
        # blind abgeschaltet. Nur WAX_CYLINDER wird weiterhin deaktiviert (proprietäres
        # Tonleitersystem mit festen K, kein Western-Durtonart-Profil anwendbar).
        if _mat_key == "wax_cylinder":
            inapplicable["tonal_center"] = (
                "Wachswalze — K-S-Durtonart-Profil nicht anwendbar: proprietäres Tonleitersystem, kein Western-Key."
            )

        # REGEL: GrooveMetric
        no_percussion = (
            panns_tags is not None and panns_tags.get("Percussion", 0.0) < 0.15 and panns_tags.get("Drum", 0.0) < 0.15
        )
        if duration_s < 10.0 or no_percussion:
            reason = "Zu kurz fuer Beat-Tracking." if duration_s < 10.0 else "Kein messbares Rhythmusmuster erkannt."
            inapplicable["groove"] = reason

        # REGEL: MicroDynamicsMetric
        # §2.47: < 10 s → Groove/MicroDyn off (spec-konform; vorher fälschlich 20 s)
        if duration_s < 10.0:
            inapplicable["micro_dynamics"] = "Zu kurz fuer LUFS-Profil-Korrelation (<10 s)."

        # REGEL: SeparationFidelityMetric
        if is_mono_signal or mat_mono:
            inapplicable["separation_fidelity"] = "Mono-Quelle — kein mehrkanaliges Signal auflösbar."
        elif "separation_fidelity" not in inapplicable:
            # §S4 Near-Mono-Codec-Ausschluss: Joint-Stereo-Codecs (mp3_low, aac, streaming)
            # zerstören Stereo-Separation irreversibel — SeparationFidelity-Threshold
            # physikalisch unerreichbar in Restoration-Modus (keine additive Stereo-Synthese).
            # Parallel zur spatial_depth-Regel (gleiche CODEC_JOINT_STEREO_MATS-Logik).
            _CODEC_JOINT_STEREO_MATS = frozenset({"mp3_low", "mp3_high", "aac", "streaming", "minidisc"})
            _is_codec_joint_stereo = _mat_key in _CODEC_JOINT_STEREO_MATS or any(
                str(s).strip().lower() in _CODEC_JOINT_STEREO_MATS for s in (transfer_chain or [])
            )
            if _is_codec_joint_stereo and not np.isnan(corr) and float(corr) >= 0.83:
                inapplicable["separation_fidelity"] = (
                    f"Near-Mono-Codec ({_mat_key or 'codec'}, L/R-Korrelation={float(corr):.3f} ≥ 0.83) — "
                    "Joint-Stereo hat Stereo-Separation irreversibel reduziert; Restoration kann diese nicht wiederherstellen."
                )

        # REGEL: BassKraftMetric — physikalisch bassarmes Material in Restoration-Modus
        # §0c Universalitäts-Invariante: Regel basiert auf gemessener Spektraleigenschaft,
        # NICHT auf Genre-Labels oder Material-Namen. Trifft auf vocal-dominante Aufnahmen
        # (Schlager, Art Song, frühe Pop), Shellac, Wachswalzen und ähnliche Material zu.
        # §0 Primum non nocere: Restoration kann keinen Bass hinzufügen, der nicht
        # aufgezeichnet wurde — bass_kraft-Threshold 0.78 wäre physikalisch unerreichbar
        # und erzwingt Over-Processing-Versuche (Pipeline-Failure-Kaskade).
        # Studio 2026: Regel deaktiviert — Enhancement-Phasen (phase_06, phase_07) können
        # Bass aufbauen; bass_kraft bleibt als Ziel aktiv.
        # Threshold 0.015 (1.5 % der Spektralenergie in 20–250 Hz):
        #   < 0.015: vocal-dominante Aufnahmen, Shellac, WaxCyl, 1930s-1960s light music
        #   ≥ 0.015: Blues, Rock, Jazz mit Bass, Electronic, Hip-hop — bleibt anwendbar
        _mode_str = str(mode or "restoration").strip().lower()
        if _mode_str == "restoration" and audio is not None and duration_s >= 5.0:
            try:
                # 2-Sekunden-Segment aus Mitte für repräsentative Bass-Analyse.
                # Center-Segment vermeidet Intro-Stille und vermeidet Outro-Fade.
                _n_bass = min(len(mono), int(sr * 2))
                _bass_start = max(0, (len(mono) - _n_bass) // 2)
                _bass_seg = mono[_bass_start : _bass_start + _n_bass]
                if len(_bass_seg) >= 512:
                    _bass_spec = np.abs(np.fft.rfft(_bass_seg)) ** 2 + 1e-15
                    _bass_freqs = np.fft.rfftfreq(len(_bass_seg), d=1.0 / sr)
                    _bm = (_bass_freqs >= 20) & (_bass_freqs <= 250)
                    _input_bass_ratio = float(np.sum(_bass_spec[_bm])) / float(np.sum(_bass_spec))
                    _bass_floor = 0.010 if _mat_key in _LOW_BASS_MATERIALS else 0.015
                    if _input_bass_ratio < _bass_floor:
                        inapplicable["bass_kraft"] = (
                            f"Input-Bass-Anteil {_input_bass_ratio:.4f} < {_bass_floor:.3f} — "
                            "Restoration kann keinen aufnahmetypischen Bass erzeugen "
                            "(§0 Primum non nocere). "
                            "Studio-2026-Modus: Ziel bleibt aktiv (Enhancement-Phasen)."
                        )
                        logger.debug(
                            "§2.32 bass_kraft N/A: Eingabe_bass_Verhaeltnis=%.4f < %.3f (Restoration)",
                            _input_bass_ratio,
                            _bass_floor,
                        )
            except Exception as _bk_exc:
                logger.debug("§2.32 bass_kraft-Verhaeltnis-Pruefung fehlgeschlagen: %s", _bk_exc)

        # _ALWAYS_APPLICABLE NIE deaktivieren
        for g in _ALWAYS_APPLICABLE:
            inapplicable.pop(g, None)

        # §0p P0-Goals (vocal_quality, formant_fidelity):
        # Nur applicable wenn panns_singing >= 0.35 (Gesangsmaterial erkannt).
        # Unterhalb des Schwellwerts physikalisch nicht sinnvoll messbar
        # (VQI/Formant-Tracking auf Instrumental-Material liefert false-negatives).
        _p0_vocal_goals = frozenset({"vocal_quality", "formant_fidelity"})
        _panns_singing_f = float(panns_singing if panns_singing is not None else 0.0)
        if _panns_singing_f < 0.35:
            for _p0g in _p0_vocal_goals:
                inapplicable[_p0g] = (
                    f"panns_singing={_panns_singing_f:.3f} < 0.35 "
                    "— P0-Vokalziele nur bei erkanntem Gesangsmaterial anwendbar."
                )
        else:
            # Gesang erkannt: P0-Goals sicherstellen (nie durch andere Regeln deaktivierbar)
            for _p0g in _p0_vocal_goals:
                inapplicable.pop(_p0g, None)

        applicable = _ALL_GOALS - frozenset(inapplicable.keys())

        return GoalApplicabilityResult(
            applicable=frozenset(applicable),
            inapplicable=frozenset(inapplicable.keys()),
            reasons=inapplicable,
        )


# ---------------------------------------------------------------------------
# Singleton + Convenience
# ---------------------------------------------------------------------------

_instance: GoalApplicabilityFilter | None = None
_lock = threading.Lock()


def get_goal_filter() -> GoalApplicabilityFilter:
    """Thread-sicherer Singleton (§3.2)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = GoalApplicabilityFilter()
    return _instance


def evaluate_goal_applicability(
    audio: np.ndarray | None = None,
    sr: int = 48000,
    material: str = "unknown",
    era_decade: int | None = None,
    panns_tags: dict[str, float] | None = None,
    flashsr_available: bool = False,
    mode: str = "restoration",
    panns_singing: float = 0.0,
    transfer_chain: list[str] | None = None,
) -> GoalApplicabilityResult:
    """Convenience-Funktion."""
    return get_goal_filter().evaluate(
        audio=audio,
        sr=sr,
        material=material,
        era_decade=era_decade,
        panns_tags=panns_tags,
        flashsr_available=flashsr_available,
        mode=mode,
        panns_singing=panns_singing,
        transfer_chain=transfer_chain,
    )


# ── Öffentliche Aliase (Spec §2.32 — ohne führenden Unterstrich) ────────────
ALL_GOALS: frozenset[str] = _ALL_GOALS
ALWAYS_APPLICABLE: frozenset[str] = _ALWAYS_APPLICABLE

# Spec-konformer Funktionsname-Alias (§3.2)
get_goal_applicability_filter = get_goal_filter  # Alias für get_goal_filter() — Spec-konformer Name (§3.2)

__all__ = [
    "ALL_GOALS",
    "ALWAYS_APPLICABLE",
    "_ALL_GOALS",
    "_ALWAYS_APPLICABLE",
    "GoalApplicabilityFilter",
    "GoalApplicabilityResult",
    "evaluate_goal_applicability",
    "get_goal_applicability_filter",
    "get_goal_filter",
]
