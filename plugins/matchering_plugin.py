"""matchering_plugin.py -- Reference-Mastering via Matchering 2.0 + DSP-Fallback.

Primary path (Matchering 2.0):
    Uses the matchering library (Kopylov 2019) which implements:
    - Separate Mid/Side processing
    - Multiband (Low/Mid/High) RMS matching
    - Wiener-filtered reference spectrum estimation
    - statsmodels-based spectral analysis
    API: matchering.process() -- works with WAV files (temp I/O, float32).

DSP fallback (if matchering not installed):
    Own implementation: Mid/Side-aware multiband STFT spectral matching.
    +-8 dB cap, 1/6-oct smoothing, OLA crossfade, NaN-guard.

Ref: Kopylov (2019) Matchering 2.0 -- https://github.com/sergree/matchering
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matchering as _mg

    _MATCHERING_AVAILABLE = True
except ImportError:
    _mg: Any = None  # type: ignore[no-redef]
    _MATCHERING_AVAILABLE = False

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst: MatcheringPlugin | None = None


class MatcheringPlugin:
    """Reference-Mastering via Matchering 2.0 library with DSP fallback.

    Primary: matchering.process() -- Multiband Mid/Side Wiener matching.
    Fallback: Custom STFT spectral matching with Mid/Side awareness.
    """

    # DSP-Fallback parameters
    N_FFT = 4096
    HOP = 1024
    SMOOTH_BANDS = 43  # ~1/6 oct at 48 kHz
    MAX_EQ_DB = 8.0  # Matchering uses up to 8 dB internally

    def process(
        self,
        target: np.ndarray,
        reference: np.ndarray,
        sr: int = 48000,
    ) -> np.ndarray:
        """Match spectral profile and loudness of target to reference.

        Uses Matchering 2.0 if available; otherwise falls back to DSP.
        Preserves stereo layout: returns array with same shape as target.

        Args:
            target: float32 ndarray, shape (samples,) or (2, samples)
            reference: float32 ndarray, shape (samples,) or (2, samples)
            sr: sample rate (must be 48000)

        Returns:
            Spectrally matched float32 ndarray clipped to [-1, 1].
        """
        assert sr == 48000, f"matchering_plugin: sr must be 48000, got {sr}"
        target = np.asarray(target, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)

        if _MATCHERING_AVAILABLE:
            return self._process_matchering(target, reference, sr)
        return self._process_dsp(target, reference, sr)

    # ------------------------------------------------------------------
    # Matchering 2.0 path
    # ------------------------------------------------------------------

    def _process_matchering(self, target: np.ndarray, reference: np.ndarray, sr: int) -> np.ndarray:
        """Use matchering.process() for Mid/Side multiband Wiener matching."""
        import soundfile as sf

        # matchering.process expects file paths; use temp WAV files.
        with tempfile.TemporaryDirectory(prefix="aurik_matchering_") as tmpdir:
            tmp = Path(tmpdir)
            tgt_path = tmp / "target.wav"
            ref_path = tmp / "reference.wav"
            out_path = tmp / "output.wav"

            tgt_interleaved = self._to_interleaved(target)
            ref_interleaved = self._to_interleaved(reference)

            sf.write(str(tgt_path), tgt_interleaved, sr, subtype="FLOAT")
            sf.write(str(ref_path), ref_interleaved, sr, subtype="FLOAT")

            try:
                from backend.file_import import load_audio_file

                _mg.process(
                    target=str(tgt_path),
                    reference=str(ref_path),
                    results=[_mg.pcm16(str(out_path))],
                )
                _res = load_audio_file(str(out_path), do_carrier_analysis=False)
                out_data = np.asarray(_res["audio"], dtype=np.float32)  # type: ignore[index]
                if out_data.ndim == 1:
                    out_data = out_data[:, np.newaxis]
                # load_audio_file returns (samples, channels); convert to (channels, samples)
                out_stereo = out_data.T
            except Exception as exc:
                logger.warning("Matchering 2.0 fehlgeschlagen, DSP-Ersatzpfad aktiv: %s", exc)
                return self._process_dsp(target, reference, sr)

        result = self._restore_shape(out_stereo, target)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # DSP fallback: Mid/Side-aware multiband spectral matching
    # ------------------------------------------------------------------

    def _process_dsp(self, target: np.ndarray, reference: np.ndarray, sr: int) -> np.ndarray:
        """Spectral matching fallback: Mid/Side STFT with per-band RMS normalisation.

        Algorithm:
            1. Decompose both signals into Mid and Side channels
            2. Per-channel: STFT -> mean magnitude spectrum
            3. H[f] = mean_ref[f] / mean_tgt[f], smoothed, capped +-MAX_EQ_DB
            4. Apply H[f] per frame -> ISTFT
            5. Global RMS match -> NaN-guard -> clip [-1, 1]
        """
        from scipy.ndimage import uniform_filter1d
        from scipy.signal import istft, stft
        from scipy.signal.windows import hann

        tgt_stereo = self._to_stereo(target)
        ref_stereo = self._to_stereo(reference)

        # Mid/Side decomposition
        tgt_mid = (tgt_stereo[0] + tgt_stereo[1]) * 0.5
        tgt_side = (tgt_stereo[0] - tgt_stereo[1]) * 0.5
        ref_mid = (ref_stereo[0] + ref_stereo[1]) * 0.5
        ref_side = (ref_stereo[0] - ref_stereo[1]) * 0.5

        win = hann(self.N_FFT)
        max_g = 10 ** (self.MAX_EQ_DB / 20.0)

        def _eq_channel(tgt_ch: np.ndarray, ref_ch: np.ndarray) -> np.ndarray:
            n = max(len(tgt_ch), len(ref_ch))
            tc = np.pad(tgt_ch, (0, max(0, n - len(tgt_ch))))
            rc = np.pad(ref_ch, (0, max(0, n - len(ref_ch))))
            _, _, Zt = stft(tc, fs=sr, window=win, nperseg=self.N_FFT, noverlap=self.N_FFT - self.HOP)
            _, _, Zr = stft(rc, fs=sr, window=win, nperseg=self.N_FFT, noverlap=self.N_FFT - self.HOP)
            mean_t = np.mean(np.abs(Zt), axis=1) + 1e-9
            mean_r = np.mean(np.abs(Zr), axis=1) + 1e-9
            ratio = np.clip(mean_r / mean_t, 1.0 / max_g, max_g)
            ratio = uniform_filter1d(ratio, size=self.SMOOTH_BANDS)
            rms_t = float(np.sqrt(np.mean(tc**2)) + 1e-9)
            rms_r = float(np.sqrt(np.mean(rc**2)) + 1e-9)
            g_rms = float(np.clip(rms_r / rms_t, 0.1, 10.0))
            Zout = Zt * ratio[:, np.newaxis] * g_rms
            _, out = istft(Zout, fs=sr, window=win, nperseg=self.N_FFT, noverlap=self.N_FFT - self.HOP)
            return np.nan_to_num(out[: len(tgt_ch)], nan=0.0).astype(np.float32)  # type: ignore[no-any-return]

        out_mid = _eq_channel(tgt_mid, ref_mid)
        out_side = _eq_channel(tgt_side, ref_side)

        # Back to L/R
        out_l = out_mid + out_side
        out_r = out_mid - out_side
        out_stereo = np.stack([out_l, out_r], axis=0)

        result = self._restore_shape(out_stereo, target)
        return np.clip(result, -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_stereo(self, audio: np.ndarray) -> np.ndarray:
        """Gibt (2, samples) float32, duplicating mono if needed zurück."""
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim == 1:
            return np.stack([a, a], axis=0)  # type: ignore[no-any-return]
        if a.shape[0] == 2:
            return a  # type: ignore[no-any-return]
        if a.ndim == 2 and a.shape[1] == 2:
            return a.T  # type: ignore[no-any-return]
        return np.stack([a[0], a[0]], axis=0)  # type: ignore[no-any-return]

    def _to_interleaved(self, audio: np.ndarray) -> np.ndarray:
        """Gibt (samples, 2) float32 interleaved for soundfile zurück."""
        stereo = self._to_stereo(audio)  # (2, samples)
        return stereo.T  # (samples, 2)

    def _restore_shape(self, out_stereo: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Restauriert original channel layout from (2, samples) result."""
        orig = np.asarray(original, dtype=np.float32)
        if orig.ndim == 1:
            n_orig = orig.shape[0]
        elif orig.shape[0] <= 8:
            n_orig = orig.shape[1]
        else:
            n_orig = orig.shape[0]
        # Trim/pad to original length
        n_out = out_stereo.shape[1]
        if n_out > n_orig:
            out_stereo = out_stereo[:, :n_orig]
        elif n_out < n_orig:
            out_stereo = np.pad(out_stereo, ((0, 0), (0, n_orig - n_out)))
        if orig.ndim == 1:
            return ((out_stereo[0] + out_stereo[1]) * 0.5).astype(np.float32)  # type: ignore[no-any-return]
        if orig.shape[0] == 2:  # (2, samples)
            return out_stereo.astype(np.float32)  # type: ignore[no-any-return]
        if orig.ndim == 2 and orig.shape[1] == 2:  # (samples, 2)
            return out_stereo.T.astype(np.float32)  # type: ignore[no-any-return]
        return ((out_stereo[0] + out_stereo[1]) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def get_matchering_plugin() -> MatcheringPlugin:
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = MatcheringPlugin()
    return _inst


def match_reference(target: np.ndarray, reference: np.ndarray, sr: int = 48000) -> np.ndarray:
    """Convenience wrapper. Returns spectrally matched audio (same shape as target)."""
    return get_matchering_plugin().process(target, reference, sr)


def is_matchering_available() -> bool:
    """Gibt True if the matchering 2.0 library is installed zurück."""
    return _MATCHERING_AVAILABLE
