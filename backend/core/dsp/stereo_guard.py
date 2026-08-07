"""§MKI (V23) Mono-Kompatibilitätsprüfung — Stereo-Guard.

Prüft vor dem Export auf Phasenlöschung im 300 Hz–5 kHz Band bei
Vokal-Stereo-Material. Kein Veto — nur WARNING + Metadata-Flag.

Kanonische Nutzung (UV3 pre-export hook):
    from backend.core.dsp.stereo_guard import check_mono_compatibility, MonoCompatResult
    result = check_mono_compatibility(audio, sr)
    if result.phase_cancellation_db > 3.0:
        metadata["mono_compatibility_warning"] = True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class MonoCompatResult:
    """Ergebnis der Mono-Kompatibilitätsprüfung.

    Attributes:
        phase_cancellation_db: Energieverlust (dB) beim Summieren L+R im 300–5000 Hz Band.
            Positiv = Energieverlust (Phasenlöschung). Grenzwert: 3.0 dB → WARNING.
        ok: True wenn phase_cancellation_db <= 3.0.
        mono_rms: RMS des Mono-Summensignals (bandgefiltert).
        stereo_rms: RMS des Stereo-Originalsignals (bandgefiltert, L/R gemittelt).
    """

    phase_cancellation_db: float
    ok: bool
    mono_rms: float = 0.0
    stereo_rms: float = 0.0


def check_mono_compatibility(
    audio: np.ndarray,
    sr: int,
) -> MonoCompatResult:
    """Prüft Mono-Kompatibilität im 300 Hz–5 kHz Band.

    Args:
        audio: Stereo-Audio [2, N] oder [N, 2] oder Mono [N].
            Mono-Signale werden direkt als kompatibel zurückgegeben.
        sr: Sample-Rate (muss 48000 sein).

    Returns:
        MonoCompatResult mit phase_cancellation_db und ok-Flag.
    """
    assert sr == 48000
    _fallback = MonoCompatResult(phase_cancellation_db=0.0, ok=True)

    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Layout-Normierung: immer [2, N]
        if audio.ndim == 1:
            return _fallback  # Mono ist per Definition kompatibel
        if audio.ndim == 2:
            if audio.shape[0] == 2:
                ch_l, ch_r = audio[0], audio[1]
            elif audio.shape[1] == 2:
                ch_l, ch_r = audio[:, 0], audio[:, 1]
            else:
                return _fallback
        else:
            return _fallback

        # Bandpass 300 Hz – 5 kHz (Butterworth 4. Ordnung, zero-phase)
        nyq = sr / 2.0
        sos = butter(4, [300.0 / nyq, 5000.0 / nyq], btype="band", output="sos")

        l_bp = sosfiltfilt(sos, ch_l).astype(np.float32)
        r_bp = sosfiltfilt(sos, ch_r).astype(np.float32)

        # Mono-Summe (standard broadcasting)
        mono_bp = (l_bp + r_bp) * 0.5

        mono_rms = float(np.sqrt(np.mean(mono_bp**2) + 1e-12))
        stereo_rms = float(np.sqrt(np.mean((l_bp**2 + r_bp**2) * 0.5) + 1e-12))

        if stereo_rms < 1e-9:
            return _fallback

        # Phasenlöschung = Energieverlust beim Summieren
        cancellation_db = float(20.0 * np.log10((stereo_rms + 1e-12) / (mono_rms + 1e-12)))
        cancellation_db = float(np.nan_to_num(cancellation_db, nan=0.0, posinf=0.0, neginf=0.0))
        ok = cancellation_db <= 3.0

        if not ok:
            logger.info(
                "§V23 Mono-Kompatibilität: Phasenlöschung=%.2f dB > 3.0 dB (300–5000 Hz) → WARNING",
                cancellation_db,
            )

        return MonoCompatResult(
            phase_cancellation_db=round(cancellation_db, 2),
            ok=ok,
            mono_rms=round(mono_rms, 6),
            stereo_rms=round(stereo_rms, 6),
        )

    except Exception as exc:
        logger.debug("Pruefung_mono_compatibility nicht blockierend: %s", exc)
        return _fallback


# ─────────────────────────────────────────────────────────────────────────────
# §IACC Inter-Aural Cross-Correlation (ITU-R BS.1116 Annex B) — Gap 5
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IACCResult:
    """Inter-Aural Cross-Correlation nach ITU-R BS.1116 Annex B / Ando 1998.

    IACC = max_{τ ∈ [-1 ms, +1 ms]} R_LR(τ) / sqrt(R_LL(0) × R_RR(0))

    Attributes:
        iacc: IACC-Wert [0.0, 1.0]. 0.0 = maximale Stereobreite, 1.0 = Mono.
        spatial_depth_score: Raumtiefe-Score = 1.0 − iacc.
            Ersetzt M/S-Proxy für `spatial_depth`-Musical-Goal.
        tau_max_ms: Delay in ms bei dem R_LR maximal ist (Richtungshinweis).
        ok: True wenn iacc < 0.70 (ausreichend breit; Mono-Material → True trivial).
    """

    iacc: float = 0.0
    spatial_depth_score: float = 1.0
    tau_max_ms: float = 0.0
    ok: bool = True


def compute_iacc(
    audio: np.ndarray,
    sr: int,
    tau_max_ms: float = 1.0,
) -> IACCResult:
    """Berechnet IACC (Inter-Aural Cross-Correlation) nach ITU-R BS.1116 Annex B.

    IACC = max_{τ ∈ [-tau_max_ms, +tau_max_ms]} R_LR(τ) / sqrt(R_LL(0) × R_RR(0))

    Mono-Material: IACC = 1.0 (triviale Mono-Kompatibilität), spatial_depth_score = 0.0.

    Args:
        audio: Stereo [2, N] oder [N, 2] oder Mono [N].
        sr: Abtastrate in Hz (muss 48000 sein — Analyse-Schritt in UV3).
        tau_max_ms: Maximales IACC-Suchfenster in ms. ITU-R BS.1116: ±1 ms.

    Returns:
        IACCResult. Mono → iacc=1.0, spatial_depth_score=0.0, ok=True (Sonderfall).
    """
    assert sr == 48000
    _fallback_mono = IACCResult(iacc=1.0, spatial_depth_score=0.0, tau_max_ms=0.0, ok=True)
    _fallback_wide = IACCResult(iacc=0.0, spatial_depth_score=1.0, tau_max_ms=0.0, ok=True)

    try:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        # ── Layout-Normierung → immer [2, N] ─────────────────────────────────
        if audio.ndim == 1:
            return _fallback_mono
        if audio.ndim == 2:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                ch_l, ch_r = audio[0], audio[1]
            elif audio.shape[1] == 2 and audio.shape[0] > 2:
                ch_l, ch_r = audio[:, 0], audio[:, 1]
            else:
                return _fallback_mono
        else:
            return _fallback_mono

        n = len(ch_l)
        if n < 64:
            return _fallback_mono

        # ── R_LL(0) und R_RR(0) — Auto-Korrelationen bei τ=0 ─────────────────
        rll = float(np.dot(ch_l, ch_l))
        rrr = float(np.dot(ch_r, ch_r))

        if rll < 1e-20 or rrr < 1e-20:
            return _fallback_wide  # Kanal(e) stumm → keine sinnvolle IACC

        denom = float(np.sqrt(rll * rrr))

        # ── Kreuzkorrelation in ±tau_max_ms Fenster ───────────────────────────
        tau_max_samples = max(1, int(tau_max_ms * 1e-3 * sr))
        max_lag = min(tau_max_samples, n - 1)

        best_xcorr = 0.0
        best_tau_samples = 0

        # Direkte Berechnung für kleines Fenster (max_lag typisch 48 Samples @ 48 kHz)
        for tau in range(-max_lag, max_lag + 1):
            if tau == 0:
                xcorr = float(np.dot(ch_l, ch_r))
            elif tau > 0:
                # R_LR(τ): L verzögert um τ → L[τ:] × R[:-τ]
                xcorr = float(np.dot(ch_l[tau:], ch_r[: n - tau]))
            else:
                # tau < 0: R verzögert → L[:n+τ] × R[-τ:]
                t = -tau
                xcorr = float(np.dot(ch_l[: n - t], ch_r[t:]))

            if abs(xcorr) > abs(best_xcorr):
                best_xcorr = xcorr
                best_tau_samples = tau

        iacc = float(np.clip(abs(best_xcorr) / denom, 0.0, 1.0))
        iacc = float(np.nan_to_num(iacc, nan=0.0, posinf=1.0, neginf=0.0))
        tau_ms = float(best_tau_samples / sr * 1000.0)
        spatial_depth_score = float(np.clip(1.0 - iacc, 0.0, 1.0))
        ok = bool(iacc < 0.70)

        logger.debug(
            "IACC=%.3f spatial_depth=%.3f tau=%.3f ms ok=%s",
            iacc,
            spatial_depth_score,
            tau_ms,
            ok,
        )

        return IACCResult(
            iacc=round(iacc, 4),
            spatial_depth_score=round(spatial_depth_score, 4),
            tau_max_ms=round(tau_ms, 3),
            ok=ok,
        )

    except Exception as exc:
        logger.debug("berechnen_iacc nicht blockierend: %s", exc)
        return _fallback_wide
