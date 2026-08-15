"""Tonalitäts-Gate (§23-TONALITY, Vorschlag 01) — kanonische Sauberkeits-Prüfung für NR-Module.

Warum: Stationäre tonale Signale (Musik, Sinusanteile) werden von
Rauschschätzern systematisch als Rauschen klassifiziert (5 %-Perzentil-
Rauschboden, IMCRA). Mess-Evidenz (Session 2026-08):
  spectral flatness 0.00 = Sinus 440+880 Hz (tonal/sauber)
  spectral flatness 0.69 = Sinus + Rauschen σ=0.05
  spectral flatness 0.99 = reines Rauschen

Vertrag:
  NR-Module konsultieren `is_tonal_clean(audio, sr)` VOR der ML-NR-
  Entscheidung. Rückgabe True ⇒ ML-NR-Zweig überspringen; der DSP-Pfad
  bleibt aktiv. Definition: spectral flatness auf Welch-Spektrum
  (hann, nperseg=min(2048, n//2), f >= 100 Hz) < TONAL_CLEAN_FLATNESS.
  Die Konstante liegt (nach Übernahme) in CalibratedConstants; bis dahin
  getattr-Fallback auf den Referenzwert.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Kanonische Konstante: Vorschlag 01 verlangt CalibratedConstants als
# einzige Quelle. Solange die Übernahme (Maintainer-Sign-off) aussteht,
# greift der Referenzwert als Fallback — identisch zum Vorschlag.
_REFERENCE_FLATNESS = 0.05


def get_tonal_clean_flatness() -> float:
    """Liest TONAL_CLEAN_FLATNESS aus CalibratedConstants (Fallback 0.05)."""
    try:
        from backend.core import calibrated_constants as _cc

        return float(getattr(_cc, "TONAL_CLEAN_FLATNESS", _REFERENCE_FLATNESS))
    except Exception as _exc:  # nicht blockierend
        logger.debug("tonality_gate: CalibratedConstants nicht verfügbar (%s) — Referenzwert", _exc)
        return _REFERENCE_FLATNESS


def spectral_flatness(audio: np.ndarray, sample_rate: int) -> float:
    """Welch-basierte spectral flatness (0=totale Tonalität, 1=weißes Rauschen).

    nperseg=min(2048, n//2), hann, f >= 100 Hz; NaN/Inf-sicher.
    """
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = audio.size
    if n < 512:
        return 0.0  # zu kurz: konservativ als tonal behandeln (kein NR-Eingriff)
    try:
        from scipy import signal as _signal

        _nperseg = int(min(2048, max(128, n // 2)))
        _freqs, _psd = _signal.welch(audio, fs=int(sample_rate), nperseg=_nperseg, window="hann")
        _band = np.sqrt(np.maximum(_psd[_freqs >= 100.0], 1e-12))
        if _band.size < 4:
            return 0.0
        _geo = float(np.exp(np.mean(np.log(_band))))
        _ari = float(np.mean(_band))
        if _ari <= 1e-12:
            return 0.0
        return float(np.clip(_geo / _ari, 0.0, 1.0))
    except Exception as _exc:  # nicht blockierend — konservativ: NICHT tonal
        logger.debug("tonality_gate: flatness fehlgeschlagen (%s) — konservativ False", _exc)
        return 1.0


def is_tonal_clean(audio: np.ndarray, sample_rate: int) -> bool:
    """True, wenn das Signal als tonal/sauber gilt ⇒ ML-NR überspringen."""
    try:
        _flat = spectral_flatness(audio, sample_rate)
        _threshold = get_tonal_clean_flatness()
        _result = _flat < _threshold
        logger.debug(
            "tonality_gate: flatness=%.4f threshold=%.3f → tonal_clean=%s",
            _flat,
            _threshold,
            _result,
        )
        return _result
    except Exception as _exc:  # nicht blockierend — konservativ: NICHT tonal
        logger.debug("tonality_gate: is_tonal_clean fehlgeschlagen (%s)", _exc)
        return False
