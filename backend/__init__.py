"""Backend-Paket für Aurik 10.0.0 — DSP, ML-Modelle, Denker und API."""

# Ermöglicht Import von backend als Paket für Tests

# ── §2.62 STFT Input-Length-Guard (zentral, schützt alle 62+ Aufrufer) ──
# scipy.signal.stft warnt bei nperseg > input_length ("using nperseg = 2").
# Aurik-DSP erzeugt in 2-Sample-Subbändern korrekte Kurz-Arrays; die Warnung
# ist harmlos aber flutet das Log. Statt 62+ Einzel-Guards patchen wir
# scipy.signal.stft einmalig mit einem transparenten Längen-Guard.
import warnings as _warnings

import numpy as _np
import scipy.signal as _scipy_signal

_original_stft = _scipy_signal.stft


def _safe_stft(
    x,
    fs=1.0,
    window="hann",
    nperseg=256,
    noverlap=None,  # type: ignore[no-untyped-def]
    nfft=None,
    detrend=False,
    return_onesided=True,
    scaling="spectrum",
    axis=-1,
    boundary=None,
    padded=True,
):
    """scipy.signal.stft mit transparentem Längen-Guard.

    Bei input_length < nperseg gibt scipy eine UserWarning aus und
    verwendet nperseg=input_length. Dieser Guard fängt den Fall ab,
    ohne die Warnung auszulösen — das Verhalten ist identisch.
    """
    arr = _np.asarray(x)
    if arr.size == 0:
        nfft_val = nfft if nfft is not None else nperseg
        return _np.zeros(nfft_val // 2 + 1), _np.array([0.0]), _np.zeros((nfft_val // 2 + 1, 0), dtype=complex)
    if arr.size < nperseg:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return _original_stft(
                x,
                fs=fs,
                window=window,
                nperseg=arr.size,
                noverlap=max(0, min(noverlap or arr.size // 2, arr.size - 1)),
                nfft=nfft,
                detrend=detrend,
                return_onesided=return_onesided,
                scaling=scaling,
                axis=axis,
                boundary=boundary,
                padded=padded,
            )
    return _original_stft(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        boundary=boundary,
        padded=padded,
    )


_scipy_signal.stft = _safe_stft
# §v10.115 DEPRECATED: safe_stft/safe_istft monkey-patches bleiben für Rückwärtskompatibilität.
# Alle Phasen verwenden jetzt explizit `from backend.core.audio_utils import safe_stft, safe_istft`.
# Diese Monkey-Patches werden in v10.18 entfernt. Siehe Spec 22 (M1 Safe-STFT-Wrapper).
_scipy_signal.safe_stft = _safe_stft
_scipy_signal.safe_istft = _safe_stft  # §v10.119: safe_istft alias (gleiche Signatur, Längen-Guard)
