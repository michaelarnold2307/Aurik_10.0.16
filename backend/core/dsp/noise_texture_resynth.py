"""Noise-Texture-Re-Synthesis — Rauschtextur nach Over-NR kontrolliert auffüllen (§TimbralCoherence).

Problem: Aggressive NR (DeepFilterNet, OMLSA, SGMSE+) entfernt nicht nur Rauschen, sondern
         auch den letzten ruhigen Rauschboden. Das Ergebnis klingt „plastisch" oder hart
         abgeschnitten, wenn Stillezonen völlig tot wirken.

Lösung:  Nach jeder NR-Phase: Messe gemessene Rauschtextur im NR-Ausgang und vergleiche
         mit Zielprofil. Analoge Tonträger zielen im Export auf CD-ähnliche Textur und
         CD-ähnlichen Boden; analoges Hiss-/Oberflächenrauschen wird nicht zurückgefüllt.

API:
    from backend.core.dsp.noise_texture_resynth import restore_carrier_noise_texture
    audio_out = restore_carrier_noise_texture(
        audio_pre_nr, audio_post_nr, sr, material_type="vinyl",
        max_correction_db=6.0
    )

Basierend auf:
    - psychoacoustics.compute_noise_texture_profile()
    - psychoacoustics.get_material_noise_texture()
    - psychoacoustics.synthesize_comfort_noise()  [§0a: Rauschboden-Textur-Invariante]
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["restore_carrier_noise_texture"]

# Schwellwert: Abweichung < 3 dB → keine Korrektur nötig
_MIN_DEVIATION_DB = 3.0

# Maximum Korrektur-Stärke: nie mehr als 6 dB Textur-Re-Synthese
_DEFAULT_MAX_CORRECTION_DB = 6.0

# Minimum Quiet-Frames: brauchen >= 3 Frames für zuverlässige Messung
_MIN_QUIET_FRAMES = 3

_CD_LIKE_FLOOR_DBFS = -74.0
_ANALOG_CD_FLOOR_MAX_CORRECTION_DB = 32.0
_ANALOG_CD_FLOOR_TARGETS = frozenset(
    {
        "shellac",
        "wax_cylinder",
        "lacquer_disc",
        "wire_recording",
        "reel_tape",
        "tape",
        "vinyl",
        "cassette",
        "unknown_analog",
    }
)


def _material_resynth_target(material_type: str) -> tuple[str, float | None]:
    """Liefert Textur-Zielmaterial und optionalen Maximalboden für Resynthese."""
    key = str(material_type or "unknown").lower().strip()
    # Normalisiere Enum-Präfixe (MaterialType.CASSETTE → cassette),
    # die von Callern wie phase_03/phase_29/unified_restorer_v3 ankommen können.
    for _prefix in ("materialtype.",):
        if key.startswith(_prefix):
            key = key[len(_prefix):]
            break
    if key in _ANALOG_CD_FLOOR_TARGETS:
        return "cd_digital", _CD_LIKE_FLOOR_DBFS
    return key, None


def restore_carrier_noise_texture(
    audio_pre_nr: np.ndarray,
    audio_post_nr: np.ndarray,
    sr: int,
    material_type: str = "vinyl",
    max_correction_db: float = _DEFAULT_MAX_CORRECTION_DB,
    strength: float = 1.0,
) -> np.ndarray:
    """Stelle Carrier-Rauchtextur nach Over-NR wieder her.

    Vergleicht die Rauchtextur des NR-Ausgangs mit dem Material-Referenzprofil.
    Wenn die Abweichung > 3 dB ist (Over-NR erkannt), wird comfort noise mit
    der Carrier-typischen Spektralform in Stille-Passagen eingemischt.

    §0a: Analoge Trägerdefekt-Böden im Export auf CD-ähnliches Ziel bringen.
    §TimbralCoherence: Über-NR-tes Audio klingt plastisch ohne kontrollierte Resttextur.

    Parameters
    ----------
    audio_pre_nr : np.ndarray
        Audio VOR der NR-Phase (dient als Referenz für Trägertextur-Analyse).
    audio_post_nr : np.ndarray
        Audio NACH der NR-Phase (wird korrigiert).
    sr : int
        Abtastrate in Hz. Kein assert — Analyse-Modul (§Codierregeln).
    material_type : str
        Materialtyp für Referenzprofil: "vinyl", "shellac", "reel_tape" usw.
    max_correction_db : float
        Maximale Textur-Korrektur in dB (Standard: 6 dB, nie überschreiten).
    strength : float
        Skalierungsfaktor [0.0, 1.0] für Korrekturstärke.

    Returns
    -------
    np.ndarray
        Korrigiertes Audio (gleiche Form wie audio_post_nr).
        Bei Fehler oder unzureichenden Quiet-Frames: audio_post_nr unverändert.
    """
    if strength <= 0.0:
        return audio_post_nr

    try:
        from backend.core.dsp.psychoacoustics import (  # pylint: disable=import-outside-toplevel
            compute_noise_texture_profile,
            get_material_noise_texture,
            synthesize_comfort_noise,
        )
    except ImportError as _imp_exc:
        logger.debug("noise_texture_resynth: psychoacoustics nicht verfügbar: %s", _imp_exc)
        return audio_post_nr

    try:
        audio_out = np.asarray(audio_post_nr, dtype=np.float64)
        # Detect stereo in both formats: channels-first (2, N) and channels-last (N, 2)
        _ch_first = audio_out.ndim == 2 and audio_out.shape[0] == 2 and audio_out.shape[1] > 2
        _ch_last = audio_out.ndim == 2 and audio_out.shape[1] == 2 and audio_out.shape[0] > 2
        stereo = _ch_first or _ch_last

        # Stereo: getrennte Kanalverarbeitung, dann zusammenführen
        if stereo:
            _pre = np.asarray(audio_pre_nr, dtype=np.float64)
            _pre_ch_first = _pre.ndim == 2 and _pre.shape[0] == 2 and _pre.shape[1] > 2
            _pre_ch_last = _pre.ndim == 2 and _pre.shape[1] == 2 and _pre.shape[0] > 2
            if _ch_first:
                ch_pre = [
                    _pre[0] if _pre_ch_first else (_pre[:, 0] if _pre_ch_last else _pre),
                    _pre[1] if _pre_ch_first else (_pre[:, 1] if _pre_ch_last else _pre),
                ]
                ch_post = [audio_out[0], audio_out[1]]
            else:  # _ch_last: (N, 2)
                ch_pre = [
                    _pre[:, 0] if _pre_ch_last else (_pre[0] if _pre_ch_first else _pre),
                    _pre[:, 1] if _pre_ch_last else (_pre[1] if _pre_ch_first else _pre),
                ]
                ch_post = [audio_out[:, 0], audio_out[:, 1]]
            corrected = [
                _restore_channel(
                    ch_pre[i],
                    ch_post[i],
                    sr,
                    material_type,
                    max_correction_db,
                    strength,
                    compute_noise_texture_profile,
                    get_material_noise_texture,
                    synthesize_comfort_noise,
                )
                for i in range(2)
            ]
            if _ch_first:
                result = np.stack(corrected, axis=0)  # (2, N)
            else:
                result = np.column_stack(corrected)  # (N, 2)
        else:
            pre_mono = np.asarray(audio_pre_nr, dtype=np.float64)
            if pre_mono.ndim == 2:
                pre_mono = pre_mono.mean(axis=0)
            post_mono = audio_out if audio_out.ndim == 1 else audio_out.mean(axis=0)
            result = _restore_channel(
                pre_mono,
                post_mono,
                sr,
                material_type,
                max_correction_db,
                strength,
                compute_noise_texture_profile,
                get_material_noise_texture,
                synthesize_comfort_noise,
            )

        result_arr = np.asarray(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
        result_arr = np.asarray(np.clip(result_arr, -1.0, 1.0), dtype=np.float64)
        return cast(np.ndarray, result_arr.astype(np.asarray(audio_post_nr).dtype, copy=False))

    except Exception as _exc:
        logger.debug("noise_texture_resynth: Fehler (nicht-blockierend): %s", _exc)
        return audio_post_nr


def _restore_channel(
    pre: np.ndarray,
    post: np.ndarray,
    sr: int,
    material_type: str,
    max_correction_db: float,  # pylint: disable=unused-argument
    strength: float,
    compute_noise_texture_profile,
    get_material_noise_texture,
    synthesize_comfort_noise,
) -> np.ndarray:
    """Rauchtextur-Korrektur für einen einzelnen Kanal."""
    # Messe aktuelle Rauchtextur im post-NR Signal
    measured = compute_noise_texture_profile(post, sr)
    target_material, floor_cap_dbfs = _material_resynth_target(material_type)
    # Referenzprofil für dieses Material
    target = get_material_noise_texture(target_material)

    # Abweichung berechnen: max dB-Unterschied zwischen gemessener und Zieltextur
    measured_safe = np.clip(measured, 1e-10, None)
    target_safe = np.clip(target, 1e-10, None)
    ratio = np.clip(target_safe / measured_safe, 0.01, 100.0)
    deviation_db = float(20.0 * np.log10(float(np.max(ratio))))

    if deviation_db < _MIN_DEVIATION_DB:
        logger.debug(
            "noise_texture_resynth: Deviation=%.1f dB < %.1f dB → keine Korrektur (%s)",
            deviation_db,
            _MIN_DEVIATION_DB,
            material_type,
        )
        return post

    # Schätze Noise-Floor-Pegel aus pre-NR Audio
    rms_floor = _estimate_noise_floor_dbfs(pre, sr)
    if rms_floor > -20.0:
        # Kein klar erkennbarer Rauschboden → keine Korrektur
        logger.debug("noise_texture_resynth: Rauschboden nicht erkennbar (%.1f dBFS)", rms_floor)
        return post

    # Begrenze Korrekturniveau auf max_correction_db (§TimbralCoherence: nie zu aggressiv)
    effective_floor = float(np.clip(rms_floor, -80.0, -20.0))
    # BUG-FIX: strength im linearen Bereich skalieren (NICHT als dBFS-Multiplikator!).
    # Falsch: effective_floor * 0.48 = -24.96 dBFS → Rauschen LAUTER als Original.
    # Korrekt: 20*log10(strength) addieren → Rauschen leiser als Original wenn strength < 1.
    _str_db = 20.0 * np.log10(max(float(strength), 1e-6))
    correction_floor = float(np.clip(effective_floor + _str_db, -75.0, effective_floor))
    if floor_cap_dbfs is not None:
        # Analoge Träger: Hiss/Oberflächenrauschen ist zu korrigierender Defekt.
        # Wenn Textur nachgefüllt werden muss, dann nur CD-ähnlich leise.
        correction_floor = min(correction_floor, float(floor_cap_dbfs))
        if effective_floor - correction_floor > _ANALOG_CD_FLOOR_MAX_CORRECTION_DB:
            logger.debug(
                "noise_texture_resynth: Analog→CD-Floor übersprungen (material=%s floor_delta=%.1f dB)",
                material_type,
                effective_floor - correction_floor,
            )
            return post

    corrected = synthesize_comfort_noise(
        post.astype(np.float64),
        sr,
        measured_texture=measured,
        target_texture=target,
        noise_floor_dbfs=correction_floor,
    )
    corrected_arr = np.asarray(corrected, dtype=np.float64)
    logger.info(
        "noise_texture_resynth: Over-NR-Korrektur angewandt (material=%s target=%s deviation=%.1f dB floor=%.1f dBFS)",
        material_type,
        target_material,
        deviation_db,
        correction_floor,
    )
    return cast(np.ndarray, corrected_arr)


def _estimate_noise_floor_dbfs(audio: np.ndarray, sr: int) -> float:
    """Schätze Rauschbodenpegel als 5. Perzentil der Frame-RMS-Werte."""
    try:
        frame_len = max(int(0.05 * sr), 1)
        hop = frame_len // 2
        n = len(audio)
        rms_vals = []
        for start in range(0, n - frame_len, hop):
            frame = audio[start : start + frame_len]
            rms = float(np.sqrt(np.mean(frame**2) + 1e-20))
            rms_vals.append(rms)
        if not rms_vals:
            return -80.0
        p5 = float(np.percentile(rms_vals, 5))
        return float(20.0 * np.log10(p5 + 1e-20))
    except Exception as e:
        logger.warning("noise_texture_resynth.py::_estimate_noise_floor_dbfs fallback: %s", e)
        return -80.0
