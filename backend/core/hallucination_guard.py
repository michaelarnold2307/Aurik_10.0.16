"""
§2.46e Hallucination-Guard — Schutz vor additiven Phasen, die Material halluzinieren
das im Eingangssignal physikalisch nicht vorhanden war.

Drei Kategorien halluzinierten Materials (alle verboten in Restoration):
1. Harmonik-Halluzination: Obertöne über physikalisches BW-Ceiling oder über Trägerprofil-Amplitude
2. Raum-Halluzination: Raumklang/Reverb/Stereobreite nicht im degradierten Signal
3. Textur-Halluzination: Spektrale Texturen von ML-Modellen ohne physikalisches Gegenstück

BUG-FIX v10.0.0: check_harmonic_ceiling_violation nutzt jetzt eine band-relative Delta-Metrik.
Die alte Metrik (energy_above_ceiling / total_energy) war für Air-Band-Halluzinationen blind:
Air-Band-Energie ist typisch < 0.1% der Gesamt-Energie => Threshold 3% war nie erreichbar.

Neue Metrik: (energy_above_after / energy_above_before) > 8.0
Ein 8-facher Anstieg (+9 dB) im Ceiling-Band = Violation.

BUG-FIX v10.0.0: spectral_novelty Rollback-Threshold 0.25 => 0.15 (Spec §2.46e).
"""

import logging

import numpy as np
from scipy import signal as _sp_signal

logger = logging.getLogger(__name__)


def compute_spectral_novelty(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int = 48000,
    n_fft: int = 2048,
) -> tuple[float, dict]:
    """
    §2.46e Hallucination-Guard: Misst Spektral-Neuheit (neue Bins, die nicht im Original waren).

    Args:
        audio_before: Original-Audio (mono)
        audio_after: Nach additive Phase (mono)
        sr: Sample rate
        n_fft: FFT-Größe

    Returns:
        (spectral_novelty: float [0, 1], metadata: dict)
    """
    try:
        # §2.61 Guard: nperseg darf nie größer sein als die Audio-Länge
        _nperseg = min(n_fft, audio_before.shape[-1], audio_after.shape[-1])
        if _nperseg < 4:
            return 0.0, {"error": "audio_too_short", "samples": _nperseg}
        _, _, Pxx_before = _sp_signal.spectrogram(audio_before, fs=sr, nperseg=_nperseg)
        _, _, Pxx_after = _sp_signal.spectrogram(audio_after, fs=sr, nperseg=_nperseg)

        # Frame-weise Energie
        E_before = np.mean(Pxx_before, axis=1)  # (n_freq,)
        E_after = np.mean(Pxx_after, axis=1)

        # Bins mit erhöhter Energie nach Phase (neue/halluzinierte Bins)
        E_delta = np.maximum(E_after - E_before, 0.0)
        E_total_after = np.sum(E_after) + 1e-12

        spectral_novelty = float(np.sum(E_delta) / E_total_after)

        return spectral_novelty, {
            "novelty": spectral_novelty,
            "n_fft": n_fft,
            "new_energy_db": float(10.0 * np.log10(np.sum(E_delta) + 1e-12)),
            "total_energy_db": float(10.0 * np.log10(E_total_after)),
        }
    except Exception as exc:
        logger.debug("Spectral novelty computation (nicht blockierend): %s", exc)
        return 0.0, {"error": str(exc)}


def check_harmonic_ceiling_violation(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    material_bw_ceiling_hz: float,
    sr: int = 48000,
    n_fft: int = 4096,
) -> tuple[bool, dict]:
    """
    §2.46e BUG-FIX v10.0.0: Band-relative Delta-Metrik statt ratio-zur-Gesamt-Energie.

    Die alte Metrik (energy_above_ceiling / total_energy) war fuer Air-Band-Halluzinationen
    blind: Air-Band-Energie ist < 0.1% der Gesamt-Energie => Threshold 3% nie erreichbar.

    Neue Metrik: (energy_above_after / energy_above_before) > 8.0
    Ein 8-facher Anstieg (+9 dB) im Ceiling-Band = Violation.

    Args:
        audio_before: Audio VOR der additiven Phase (Referenz)
        audio_after: Audio NACH der additiven Phase
        material_bw_ceiling_hz: Material-spezifisches BW-Ceiling (z.B. 16000 Hz fuer Vinyl)
        sr: Sample rate
        n_fft: FFT-Groesse

    Returns:
        (violation: bool, metadata: dict)
    """
    try:
        freq_resolution = sr / n_fft
        ceiling_bin = max(1, int(material_bw_ceiling_hz / freq_resolution))
        nyquist_bin = n_fft // 2 + 1
        if ceiling_bin >= nyquist_bin:
            return False, {"ceiling_hz": material_bw_ceiling_hz, "violation": False, "reason": "ceiling_at_nyquist"}

        n_before = audio_before.shape[-1] if audio_before.ndim == 2 else len(audio_before)
        n_after = audio_after.shape[-1] if audio_after.ndim == 2 else len(audio_after)
        n = min(n_before, n_after, n_fft * 8)  # cap at 8 frames for speed

        # Use mono (first channel if stereo)
        sig_before = audio_before[0, :n] if audio_before.ndim == 2 else audio_before[:n]
        sig_after = audio_after[0, :n] if audio_after.ndim == 2 else audio_after[:n]

        spec_before = np.abs(np.fft.rfft(sig_before, n=n_fft))
        spec_after = np.abs(np.fft.rfft(sig_after, n=n_fft))

        energy_above_before = float(np.sum(spec_before[ceiling_bin:] ** 2)) + 1e-12
        energy_above_after = float(np.sum(spec_after[ceiling_bin:] ** 2)) + 1e-12

        # Band-relative delta: how much did above-ceiling energy grow?
        ceiling_band_ratio = energy_above_after / energy_above_before
        ceiling_band_db = float(10.0 * np.log10(ceiling_band_ratio))

        # Violation: > 8x increase (approx. +9 dB) in ceiling band
        violation = ceiling_band_ratio > 8.0

        return violation, {
            "ceiling_hz": material_bw_ceiling_hz,
            "ceiling_bin": ceiling_bin,
            "ceiling_band_ratio": round(ceiling_band_ratio, 3),
            "ceiling_band_db": round(ceiling_band_db, 2),
            "energy_above_before_db": round(10.0 * np.log10(energy_above_before), 2),
            "energy_above_after_db": round(10.0 * np.log10(energy_above_after), 2),
            "violation": violation,
        }
    except Exception as exc:
        logger.debug("Harmonic ceiling violation Pruefung (nicht blockierend): %s", exc)
        return False, {"error": str(exc)}


def apply_hallucination_guard(
    audio_before: np.ndarray,
    audio_after: np.ndarray,
    sr: int = 48000,
    material_bw_ceiling_hz: float | None = None,
    mode: str = "restoration",  # "restoration" | "studio_2026"
) -> tuple[np.ndarray, dict]:
    """
    §2.46e Hallucination-Guard Master-Function: Prueft additive Phase auf Halluzinationen.

    Args:
        audio_before: Audio VOR der additiven Phase
        audio_after: Output der additiven Phase
        sr: Sample rate
        material_bw_ceiling_hz: Fuer BW-Check (z.B. 8000 Hz fuer Shellac)
        mode: "restoration" => Hard-Stop bei Violation | "studio_2026" => Penalty-Score

    Returns:
        (audio_result, metadata: dict mit {spectral_novelty, violation, decision})
    """
    try:
        metadata = {}

        novelty, novelty_meta = compute_spectral_novelty(audio_before, audio_after, sr=sr, n_fft=2048)
        metadata.update(novelty_meta)

        ceiling_violation = False
        if material_bw_ceiling_hz is not None and material_bw_ceiling_hz > 0:
            ceiling_violation, ceiling_meta = check_harmonic_ceiling_violation(
                audio_before,
                audio_after,
                material_bw_ceiling_hz,
                sr=sr,
                n_fft=4096,
            )
            metadata.update(ceiling_meta)

        if ceiling_violation and mode == "restoration":
            logger.warning(
                "§2.46e HARD-STOP: Harmonic ceiling violation in restoration Betriebsart "
                "(ceiling=%.0f Hz, Verhaeltnis=%.1fx, +%.1f dB)",
                material_bw_ceiling_hz,
                metadata.get("ceiling_band_ratio", 0.0),
                metadata.get("ceiling_band_db", 0.0),
            )
            metadata["hallucination_decision"] = "rollback"
            metadata["hallucination_severity"] = "ceiling_violation"
            return audio_before, metadata

        # Spec §2.46e: novelty rollback threshold in restoration mode.
        if novelty > 0.15 and mode == "restoration":
            logger.warning("§2.46e HARD-STOP: High spectral novelty (%.3f) in restoration Betriebsart", novelty)
            metadata["hallucination_decision"] = "rollback"
            metadata["hallucination_severity"] = "high_novelty"
            return audio_before, metadata

        penalty_score = 1.0
        if novelty > 0.12:
            penalty_score *= 0.8
            metadata["novelty_penalty_applied"] = True
        if ceiling_violation:
            penalty_score *= 0.6
            metadata["ceiling_penalty_applied"] = True

        metadata["hallucination_penalty_score"] = float(penalty_score)
        metadata["hallucination_decision"] = "pass_with_penalty" if penalty_score < 1.0 else "pass"
        return audio_after, metadata
    except Exception as exc:
        logger.debug("§2.46e Hallucination-Guard exception (nicht blockierend): %s", exc)
        return audio_after, {"error": str(exc), "hallucination_decision": "error_passthrough"}
