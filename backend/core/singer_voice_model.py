"""
§SVM-1 SingerVoiceModel — singer-specific spectral voice model (v10.0.0).

Builds a perceptual voice model from the clean segments of a song and uses it
for spectral inpainting of heavily damaged vocal passages.

DSP-only: no ML model required. Runs fully offline, no network access.

Non-blocking: all public methods catch exceptions and return safe fallbacks.

Usage (after VocalFocusAnalyzer):
    from backend.core.singer_voice_model import get_singer_voice_model
    svm = get_singer_voice_model()
    model = svm.build_from_audio(audio, sr, panns_singing=0.7, vfa_result=vfa)
    if model is not None:
        repaired = svm.reconstruct_damaged_vocal(segment, sr, model, damage_mask)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate as _scipy_correlate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSP constants (§SVM-1)
# ---------------------------------------------------------------------------
_N_MELS: int = 80
_MEL_FMIN_HZ: float = 80.0
_MEL_FMAX_HZ: float = 8000.0
_HOP_LENGTH: int = 512
_N_FFT: int = 2048
_LPC_ORDER: int = 16
_LPC_ANALYSIS_SR: int = 16_000
_VIBRATO_MIN_HZ: float = 4.0
_VIBRATO_MAX_HZ: float = 7.0
_ENERGY_GATE_DBFS: float = -45.0
_MIN_CLEAN_DURATION_S: float = 1.0
_FULL_CONFIDENCE_S: float = 5.0
_FORMANT_BOOST_DB: float = 1.5  # max formant-guidance boost (§0h Primum non nocere)

# ---------------------------------------------------------------------------
# Singleton bookkeeping  (W0603 suppressed at call site)
# ---------------------------------------------------------------------------
_instance: SingerVoiceModel | None = None
_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SingerVoiceModelResult:
    """Output of SingerVoiceModel.build_from_audio() — §SVM-1."""

    spectral_envelope: np.ndarray  # mean mel-spectrum of clean segments (n_mels=80)
    formant_targets: dict[str, float]  # {"F1": Hz, "F2": Hz, "F3": Hz, "F4": Hz}
    vibrato_rate_hz: float  # dominant vibrato frequency (4–7 Hz), 0 if none
    vibrato_depth_cents: float  # vibrato depth in cents
    spectral_tilt_db_per_octave: float  # spectral slope (voice character)
    hnr_db: float  # harmonic-to-noise ratio of clean segments
    vocal_segments_seconds: float  # total clean duration used for model
    confidence: float  # 0.0–1.0; 5 s of clean audio = 1.0

    def to_dict(self) -> dict[str, object]:
        """Gibt JSON-serializable metadata for the UV3 restoration context zurück."""
        return {
            "spectral_envelope": self.spectral_envelope.astype(float).tolist(),
            "formant_targets": {key: float(value) for key, value in self.formant_targets.items()},
            "vibrato_rate_hz": float(self.vibrato_rate_hz),
            "vibrato_depth_cents": float(self.vibrato_depth_cents),
            "spectral_tilt_db_per_octave": float(self.spectral_tilt_db_per_octave),
            "hnr_db": float(self.hnr_db),
            "vocal_segments_seconds": float(self.vocal_segments_seconds),
            "confidence": float(self.confidence),
        }


# ---------------------------------------------------------------------------
# Module-level DSP helpers
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Konvertiert any channel layout to mono float32 (channels_last or leading)."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr  # type: ignore[no-any-return]
    # (channels, samples) when channels <= 2 and samples >> channels
    if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
        return np.mean(arr, axis=0)  # type: ignore[no-any-return]
    return np.mean(arr, axis=-1)  # type: ignore[no-any-return]


def _frame_rms_dbfs(frame: np.ndarray) -> float:
    """Gibt frame RMS in dBFS zurück."""
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    return float(20.0 * np.log10(rms + 1e-10))


def _burg_lpc(x: np.ndarray, order: int) -> np.ndarray:
    """Burg algorithm — LPC coefficients [1, a1, …, a_order], numerically stable."""
    n = len(x)
    f = x.copy().astype(np.float64)
    b = x.copy().astype(np.float64)
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    for m in range(1, order + 1):
        num = -2.0 * float(np.dot(f[m:], b[: n - m]))
        denom = float(np.dot(f[m:], f[m:]) + np.dot(b[: n - m], b[: n - m])) + 1e-10
        k = num / denom
        a_new = a.copy()
        for i in range(1, m + 1):
            a_new[i] = a[i] + k * a[m - i]
        a = a_new
        f_new, b_new = f[m:] + k * b[: n - m], b[: n - m] + k * f[m:]
        f, b = f_new, b_new
    return a  # type: ignore[no-any-return]


def _lpc_to_formants(a: np.ndarray, sr: int) -> dict[str, float]:
    """Extrahiert F1–F4 from LPC coefficients via pole analysis."""
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0.01]
    if len(roots) == 0:
        return {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0}
    angles = np.angle(roots)
    freqs = sorted(float(a * sr / (2.0 * np.pi)) for a in angles if a > 0.0)
    freqs = [f for f in freqs if 100.0 <= f <= 4500.0]
    return {k: freqs[i] if i < len(freqs) else 0.0 for i, k in enumerate(["F1", "F2", "F3", "F4"])}


def _compute_hnr_dsp(mono: np.ndarray, sr: int) -> float:
    """Schätzt HNR via normalised autocorrelation (Boersma 1993-inspired, DSP-only)."""
    frame = mono[: min(len(mono), int(0.04 * sr))].astype(np.float64)
    if len(frame) < 128:
        return 0.0
    frame -= np.mean(frame)
    acf = _scipy_correlate(frame, frame, mode="full", method="fft")[len(frame) - 1 :]
    if acf[0] < 1e-10:
        return 0.0
    acf_norm = acf / (acf[0] + 1e-10)
    t_min = max(1, int(sr / 800.0))
    t_max = min(len(acf_norm) - 1, int(sr / 50.0))
    if t_min >= t_max:
        return 0.0
    r = float(np.clip(np.max(acf_norm[t_min:t_max]), 0.0, 0.9999))
    if r < 0.01:
        return 0.0
    return float(np.clip(10.0 * np.log10(r / (1.0 - r) + 1e-10), -20.0, 40.0))


def _compute_spectral_tilt(mel_envelope: np.ndarray) -> float:
    """Spectral tilt (dB/octave) via linear regression on log-freq vs log-energy."""
    n = len(mel_envelope)
    if n < 4:
        return 0.0
    log_f = np.log2(np.arange(1, n + 1, dtype=np.float64))
    log_e = 10.0 * np.log10(np.maximum(mel_envelope.astype(np.float64), 1e-10))
    lf_c = log_f - np.mean(log_f)
    le_c = log_e - np.mean(log_e)
    denom = float(np.dot(lf_c, lf_c))
    if denom < 1e-10:
        return 0.0
    return float(np.nan_to_num(np.dot(lf_c, le_c) / denom, nan=0.0))


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Erstellt HTK mel filterbank (n_mels × n_freqs), DSP-only."""
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)  # type: ignore[no-any-return]

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # type: ignore[no-any-return]

    mel_pts = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, center, hi = bins[m - 1], bins[m], bins[m + 1]
        if center > lo:
            k = np.arange(lo, center)
            k = k[(k >= 0) & (k < n_freqs)]
            fb[m - 1, k] = (k - lo) / max(1, center - lo)
        if hi > center:
            k = np.arange(center, hi)
            k = k[(k >= 0) & (k < n_freqs)]
            fb[m - 1, k] = (hi - k) / max(1, hi - center)
    return fb  # type: ignore[no-any-return]


def _compute_mel_spectrum(mono: np.ndarray, sr: int) -> np.ndarray:
    """
    Berechnet mel spectrogram (n_mels=80, fmin=80 Hz, fmax=8 kHz).
    Tries librosa first; falls back to DSP STFT + filterbank.
    Returns shape (n_mels, n_frames), float32.
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        return librosa.feature.melspectrogram(  # type: ignore[no-any-return]
            y=mono,
            sr=sr,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            n_mels=_N_MELS,
            fmin=_MEL_FMIN_HZ,
            fmax=_MEL_FMAX_HZ,
        ).astype(np.float32)
    except Exception:  # pylint: disable=broad-except
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # DSP fallback
    from scipy.signal import get_window  # pylint: disable=import-outside-toplevel

    window = get_window("hann", _N_FFT).astype(np.float64)
    n_frames = max(1, (len(mono) - _N_FFT) // _HOP_LENGTH + 1)
    power = np.zeros((_N_FFT // 2 + 1, n_frames), dtype=np.float64)
    for i in range(n_frames):
        s = i * _HOP_LENGTH
        seg = mono[s : s + _N_FFT]
        if len(seg) < _N_FFT:
            seg = np.pad(seg, (0, _N_FFT - len(seg)))
        power[:, i] = np.abs(np.fft.rfft(seg.astype(np.float64) * window)) ** 2
    fb = _mel_filterbank(sr, _N_FFT, _N_MELS, _MEL_FMIN_HZ, _MEL_FMAX_HZ)
    return (fb @ power).astype(np.float32)  # type: ignore[no-any-return]


def _estimate_f0_dsp(mono: np.ndarray, sr: int) -> np.ndarray:
    """
    Schätzt F0 trajectory via pYIN (librosa) with autocorrelation DSP fallback.
    Returns array of F0 per hop frame (0 = unvoiced), float32.
    """
    try:
        import librosa  # pylint: disable=import-outside-toplevel

        f0, voiced, _ = librosa.pyin(
            mono,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
            hop_length=_HOP_LENGTH,
        )
        f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
        if voiced is not None:
            f0[~voiced] = 0.0
        return f0  # type: ignore[no-any-return]
    except Exception:  # pylint: disable=broad-except
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # DSP fallback: autocorrelation pitch per hop frame
    frame_len = min(len(mono), int(0.04 * sr))
    if frame_len < 64:
        return np.zeros(1, dtype=np.float32)  # type: ignore[no-any-return]
    n_frames = max(1, (len(mono) - frame_len) // _HOP_LENGTH + 1)
    f0_arr = np.zeros(n_frames, dtype=np.float32)
    t_min = max(1, int(sr / 800.0))
    t_max = int(sr / 50.0)
    for i in range(n_frames):
        frame = mono[i * _HOP_LENGTH : i * _HOP_LENGTH + frame_len].astype(np.float64)
        frame -= np.mean(frame)
        if np.max(np.abs(frame)) < 1e-6:
            continue
        acf = _scipy_correlate(frame, frame, mode="full", method="fft")[len(frame) - 1 :]
        if acf[0] < 1e-10 or t_max >= len(acf):
            continue
        acf_n = acf / acf[0]
        idx = t_min + int(np.argmax(acf_n[t_min:t_max]))
        if acf_n[idx] > 0.30:
            f0_arr[i] = float(sr) / float(idx)
    return f0_arr  # type: ignore[no-any-return]


def _estimate_vibrato(f0_arr: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Erkennt vibrato rate and depth from F0 trajectory.
    Returns (rate_hz, depth_cents); both 0.0 if no vibrato detected.
    """
    voiced = f0_arr[f0_arr > 50.0]
    if len(voiced) < 16:
        return 0.0, 0.0
    median_f0 = float(np.median(voiced))
    if median_f0 < 1.0:
        return 0.0, 0.0
    voiced_idx = np.where(f0_arr > 50.0)[0]
    cents = 1200.0 * np.log2(np.maximum(f0_arr[voiced_idx] / median_f0, 1e-6))
    frame_rate = float(sr) / float(_HOP_LENGTH)
    n = len(cents)
    fft_mag = np.abs(np.fft.rfft(cents - np.mean(cents)))
    freqs = np.fft.rfftfreq(n, d=1.0 / frame_rate)
    mask = (freqs >= _VIBRATO_MIN_HZ) & (freqs <= _VIBRATO_MAX_HZ)
    if not np.any(mask):
        return 0.0, 0.0
    sub_mag = fft_mag[mask]
    sub_f = freqs[mask]
    peak = int(np.argmax(sub_mag))
    vib_energy = float(sub_mag[peak] ** 2)
    total_energy = float(np.sum(fft_mag**2)) + 1e-10
    if vib_energy / total_energy < 0.05:
        return 0.0, 0.0
    depth = float(sub_mag[peak]) * 2.0 / n
    return float(sub_f[peak]), float(depth)


def _estimate_formants_lpc(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Schätzt F1–F4 from clean audio via Burg-LPC on 16 kHz downsampled signal."""
    try:
        from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

        gcd = _gcd_int(sr, _LPC_ANALYSIS_SR)
        audio_ds = resample_poly(audio.astype(np.float64), _LPC_ANALYSIS_SR // gcd, sr // gcd)
        n = min(len(audio_ds), _LPC_ANALYSIS_SR)
        frame = audio_ds[:n]
        frame = np.append(frame[0], np.diff(frame))  # pre-emphasis
        frame -= np.mean(frame)
        if len(frame) < _LPC_ORDER + 2:
            return {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0}
        a = _burg_lpc(frame, _LPC_ORDER)
        return _lpc_to_formants(a, _LPC_ANALYSIS_SR)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("_estimate_formants_lpc fehlgeschlagen: %s", exc)
        return {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0}


def _gcd_int(a: int, b: int) -> int:
    """Integer GCD (Euclidean algorithm)."""
    while b:
        a, b = b, a % b
    return a


def _apply_formant_guidance(
    mag: np.ndarray,
    formant_targets: dict[str, float],
    sr: int,
    damaged_frames: set[int],
) -> np.ndarray:
    """
    Gentle Gaussian boost at F1–F4 positions for damaged STFT frames (§0h: max 1.5 dB).
    """
    if not damaged_frames:
        return mag
    n_freqs = mag.shape[0]
    freq_axis = np.linspace(0.0, float(sr) / 2.0, n_freqs)
    boost_lin = float(10.0 ** (_FORMANT_BOOST_DB / 20.0))
    bw = 200.0  # Hz half-width
    formant_boost = np.ones(n_freqs, dtype=np.float64)
    for key in ("F1", "F2", "F3", "F4"):
        f_hz = formant_targets.get(key, 0.0)
        if not 100.0 <= f_hz <= 4500.0:
            continue
        formant_boost *= 1.0 + (boost_lin - 1.0) * np.exp(-0.5 * ((freq_axis - f_hz) / (bw / 2.0)) ** 2)
    mag_out = mag.copy()
    for idx in damaged_frames:
        if idx < mag_out.shape[1]:
            mag_out[:, idx] *= formant_boost
    return mag_out


def _apply_spectral_tilt(
    mag: np.ndarray,
    tilt_db_per_octave: float,
    sr: int,
) -> np.ndarray:
    """Wendet an: spectral tilt correction to magnitude spectrogram (capped ±6 dB/oct)."""
    tilt = float(np.clip(tilt_db_per_octave, -6.0, 6.0))
    if abs(tilt) < 0.01:
        return mag
    freq_axis = np.linspace(1.0, float(sr) / 2.0, mag.shape[0])
    gain_lin = 10.0 ** (tilt * np.log2(freq_axis / 1000.0) / 20.0)
    gain_lin = np.nan_to_num(gain_lin, nan=1.0).reshape(-1, 1)
    return (mag * gain_lin).astype(mag.dtype)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SingerVoiceModel:
    """
    Singer-specific spectral voice model for inpainting damaged vocal passages.

    DSP-only. Singleton via get_singer_voice_model(). Spec: §SVM-1 (v10.0.0).
    """

    # ------------------------------------------------------------------
    # Public API — build
    # ------------------------------------------------------------------

    def build_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        panns_singing: float = 0.0,
        vfa_result: dict | None = None,
    ) -> SingerVoiceModelResult | None:
        """
        Erstellt a singer voice model from clean segments of a full-mix audio.

        Args:
            audio:          Full mix (any channel layout), float32.
            sample_rate:    Must be 48000 Hz.
            panns_singing:  PANNs singing confidence from VocalFocusAnalyzer.
            vfa_result:     Optional VocalFocusAnalyzer result dict.

        Returns:
            SingerVoiceModelResult, or None if insufficient clean material.
        """
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
        try:
            return self._build_impl(audio, sample_rate, panns_singing, vfa_result)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("SingerVoiceModel.build_from_audio fehlgeschlagen: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public API — reconstruct
    # ------------------------------------------------------------------

    def reconstruct_damaged_vocal(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 48000,
        model: SingerVoiceModelResult | None = None,
        damage_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Rekonstruiert a damaged vocal segment guided by the singer voice model.

        Args:
            audio_segment:  Damaged short clip (< 2 s), any channel layout, float32.
            sample_rate:    Must be 48000 Hz.
            model:          SingerVoiceModelResult from build_from_audio().
            damage_mask:    Bool array (len == samples), True = damaged region.
                            None → treat all frames as damaged.

        Returns:
            Reconstructed audio, same shape as input, float32 in [-1, 1].
        """
        assert sample_rate == 48000, f"SR must be 48000 Hz, got: {sample_rate}"
        if model is None:
            logger.debug("SingerVoiceModel.reconstruct_damaged_vocal: no model, passthrough")
            return audio_segment
        try:
            return self._reconstruct_impl(audio_segment, sample_rate, model, damage_mask)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("SingerVoiceModel.reconstruct_damaged_vocal fehlgeschlagen: %s", exc)
            return audio_segment

    # ------------------------------------------------------------------
    # Internal — build
    # ------------------------------------------------------------------

    def _build_impl(
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing: float,
        _vfa_result: dict | None,
    ) -> SingerVoiceModelResult | None:
        # _vfa_result is reserved for VocalFocusAnalyzer clean_segments (§SVM-1 future)
        mono = _to_mono(audio)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

        if len(mono) < int(sr * 0.5):
            logger.debug("SingerVoiceModel: audio too short (< 0.5 s)")
            return None

        # Step 2 & 3: Energy-gate + transient filter on hop frames
        n_frames = len(mono) // _HOP_LENGTH
        clean_idx: list[int] = []
        for i in range(n_frames):
            frame = mono[i * _HOP_LENGTH : (i + 1) * _HOP_LENGTH]
            if _frame_rms_dbfs(frame) <= _ENERGY_GATE_DBFS:
                continue
            # Zero-crossing rate guard (transients have high ZCR)
            signs = np.sign(frame)
            signs[signs == 0] = 1
            zcr = float(np.mean(np.abs(np.diff(signs))) / 2.0)
            if zcr > 0.35:
                continue
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) + 1e-10
            if float(np.max(np.abs(frame))) / rms > 10.0:
                continue
            clean_idx.append(i)

        if len(clean_idx) < 4:
            logger.debug("SingerVoiceModel: <4 clean frames found (panns=%.2f)", panns_singing)
            return None

        vocal_seconds = float(len(clean_idx) * _HOP_LENGTH) / float(sr)
        if vocal_seconds < _MIN_CLEAN_DURATION_S:
            logger.debug("SingerVoiceModel: clean segments too short (%.2f s)", vocal_seconds)
            return None

        # Step 4 & 5: Mel spectrogram → mean spectral envelope
        mel_spec = _compute_mel_spectrum(mono, sr)
        valid_idx = [i for i in clean_idx if i < mel_spec.shape[1]]
        if not valid_idx:
            return None

        spectral_envelope = np.mean(mel_spec[:, valid_idx], axis=1)
        spectral_envelope = np.nan_to_num(spectral_envelope, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Step 6: Formants — VFA-Daten bevorzugen falls verfügbar
        _vfa_f1 = _vfa_result.get("formant_f1_mean", 0.0) if _vfa_result else 0.0
        _vfa_f2 = _vfa_result.get("formant_f2_mean", 0.0) if _vfa_result else 0.0
        if _vfa_f1 > 0 and _vfa_f2 > 0:
            # VFA hat Formanten aus dem Gesamtsignal — zuverlässiger als
            # LPC auf den kurzen Clean-Segmenten (die oft zu kurz/leise sind)
            formant_targets = {
                "F1": float(_vfa_f1),
                "F2": float(_vfa_f2),
                "F3": float(_vfa_result.get("formant_f3_mean", 0.0) if _vfa_result else 0.0),
                "F4": 0.0,
            }
        else:
            clean_audio = np.concatenate(
                [mono[i * _HOP_LENGTH : (i + 1) * _HOP_LENGTH] for i in valid_idx[: min(len(valid_idx), 200)]]
            )
            formant_targets = _estimate_formants_lpc(clean_audio, sr)

        # Step 7: Vibrato — VFA-Daten bevorzugen, sonst selbst schätzen
        # §FIX v10.0.0: VocalFocusAnalyzer hat bereits Vibrato aus dem
        # Gesamtsignal analysiert (robuster gegen Bandbreitenbegrenzung).
        # Nur wenn VFA keine Daten liefert, DSP-Eigenberechnung nutzen.
        vibrato_rate_hz = 0.0
        vibrato_depth_cents = 0.0
        _vfa_vibrato_zones = _vfa_result.get("vibrato_zones", []) if _vfa_result else []
        _vfa_vibrato_count = _vfa_result.get("vibrato_zones_count", len(_vfa_vibrato_zones)) if _vfa_result else 0
        if _vfa_vibrato_zones and _vfa_vibrato_count >= 3:
            # VFA hat Vibrato gefunden → typische Rate 4-7 Hz, Tiefe 50-200 cents
            vibrato_rate_hz = 5.0  # Mittelwert für ausgebildete Sänger
            vibrato_depth_cents = 150.0
            # Versuche genauere Werte aus VocalStyle zu extrahieren
            _vs = _vfa_result.get("vocal_style", {}) if _vfa_result else {}
            if isinstance(_vs, dict):
                vibrato_rate_hz = float(_vs.get("vibrato_rate_hz", 5.0))
                vibrato_depth_cents = float(_vs.get("vibrato_depth_cents", 150.0))
        else:
            f0_arr = _estimate_f0_dsp(mono, sr)
            vibrato_rate_hz, vibrato_depth_cents = _estimate_vibrato(f0_arr, sr)

        # Step 8: Spectral tilt
        spectral_tilt = _compute_spectral_tilt(spectral_envelope)

        # Step 9: HNR
        hnr_db = _compute_hnr_dsp(clean_audio, sr)

        # Step 10: Confidence
        confidence = float(np.clip(vocal_seconds / _FULL_CONFIDENCE_S, 0.0, 1.0))

        logger.info(
            "SingerVoiceModel built: duration=%.2fs conf=%.2f HNR=%.1fdB "
            "tilt=%.2fdB/oct vibrato=%.1fHz depth=%.0fcents F1=%.0fHz",
            vocal_seconds,
            confidence,
            hnr_db,
            spectral_tilt,
            vibrato_rate_hz,
            vibrato_depth_cents,
            formant_targets.get("F1", 0.0),
        )
        return SingerVoiceModelResult(
            spectral_envelope=spectral_envelope,
            formant_targets=formant_targets,
            vibrato_rate_hz=vibrato_rate_hz,
            vibrato_depth_cents=vibrato_depth_cents,
            spectral_tilt_db_per_octave=spectral_tilt,
            hnr_db=hnr_db,
            vocal_segments_seconds=vocal_seconds,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Internal — reconstruct
    # ------------------------------------------------------------------

    def _reconstruct_impl(
        self,
        audio_segment: np.ndarray,
        sr: int,
        model: SingerVoiceModelResult,
        damage_mask: np.ndarray | None,
    ) -> np.ndarray:
        original_shape = audio_segment.shape
        mono = _to_mono(audio_segment)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        n_original = len(mono)

        # Reflect-padding before STFT (§2.63 boundary guard)
        pad_len = _HOP_LENGTH * 4
        padded = np.pad(mono, pad_len, mode="reflect")

        # Step 1: STFT via scipy window + rfft
        from scipy.signal import get_window  # pylint: disable=import-outside-toplevel

        window = get_window("hann", _N_FFT).astype(np.float64)
        n_stft = max(1, (len(padded) - _N_FFT) // _HOP_LENGTH + 1)
        stft = np.zeros((_N_FFT // 2 + 1, n_stft), dtype=np.complex128)
        for i in range(n_stft):
            s = i * _HOP_LENGTH
            seg = padded[s : s + _N_FFT]
            if len(seg) < _N_FFT:
                seg = np.pad(seg, (0, _N_FFT - len(seg)))
            stft[:, i] = np.fft.rfft(seg.astype(np.float64) * window)

        mag = np.abs(stft)
        phase = np.angle(stft)

        # Step 2: Determine which STFT frames are damaged
        if damage_mask is not None and len(damage_mask) == n_original:
            damaged_frames: set[int] = set()
            for i in range(n_stft):
                s_orig = i * _HOP_LENGTH - pad_len
                e_orig = s_orig + _N_FFT
                s_clip = max(0, s_orig)
                e_clip = min(n_original, e_orig)
                if s_clip >= e_clip:
                    continue
                if float(np.mean(damage_mask[s_clip:e_clip])) > 0.5:
                    damaged_frames.add(i)
        else:
            damaged_frames = set(range(n_stft))

        if not damaged_frames:
            out = np.clip(np.nan_to_num(mono), -1.0, 1.0).astype(np.float32)
            return _restore_shape(out, original_shape)

        # Step 3: Map model envelope (mel→FFT) and energy-scale to segment
        fb = _mel_filterbank(sr, _N_FFT, _N_MELS, _MEL_FMIN_HZ, _MEL_FMAX_HZ)
        model_pow = fb.T @ model.spectral_envelope.astype(np.float64)
        model_mag = np.sqrt(np.maximum(model_pow, 0.0))
        model_mag = np.nan_to_num(model_mag, nan=0.0)
        scale = (float(np.mean(mag)) + 1e-10) / (float(np.mean(model_mag)) + 1e-10)
        model_mag_scaled = model_mag * scale

        # Step 4: Replace damaged frames (0.7 model + 0.3 original blend)
        mag_out = mag.copy()
        m_mean = float(np.mean(model_mag_scaled)) + 1e-10
        for fi in damaged_frames:
            if fi >= n_stft:
                continue
            f_scale = (float(np.mean(mag[:, fi])) + 1e-10) / m_mean
            interp = model_mag_scaled * f_scale
            mag_out[:, fi] = 0.7 * interp + 0.3 * mag[:, fi]

        # Step 4b: Formant-guided boost on damaged frames
        mag_out = _apply_formant_guidance(mag_out, model.formant_targets, sr, damaged_frames)

        # Step 5a: Spectral tilt
        mag_out = _apply_spectral_tilt(mag_out, model.spectral_tilt_db_per_octave, sr)

        # Step 5b: ISTFT — retain original phase (no Griffin-Lim drift)
        stft_out = mag_out * np.exp(1j * phase)
        reconstructed = np.zeros(len(padded), dtype=np.float64)
        win_sum = np.zeros(len(padded), dtype=np.float64)
        for i in range(n_stft):
            frame_t = np.fft.irfft(stft_out[:, i], n=_N_FFT).real[:_N_FFT] * window
            s = i * _HOP_LENGTH
            e = s + _N_FFT
            if e <= len(padded):
                reconstructed[s:e] += frame_t
                win_sum[s:e] += window**2
            else:
                reconstructed[s:] += frame_t[: len(padded) - s]
                win_sum[s:] += (window**2)[: len(padded) - s]

        reconstructed /= win_sum + 1e-8

        # Strip padding (§2.63 deterministic crop)
        reconstructed = reconstructed[pad_len : pad_len + n_original]

        # Fade boundaries to suppress edge clicks
        fade = min(256, n_original // 8)
        if fade > 0:
            reconstructed[:fade] *= np.linspace(0.0, 1.0, fade)
            reconstructed[-fade:] *= np.linspace(1.0, 0.0, fade)

        # Step 6: NaN guard + clip (§0h)
        reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        reconstructed = np.clip(reconstructed, -1.0, 1.0).astype(np.float32)
        return _restore_shape(reconstructed, original_shape)


# ---------------------------------------------------------------------------
# Shape utility
# ---------------------------------------------------------------------------


def _restore_shape(mono_out: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Broadcast mono result back to the original multi-channel shape if needed."""
    if len(original_shape) == 1:
        return mono_out
    # (channels, samples)
    if original_shape[0] <= 2 and original_shape[0] < original_shape[1]:
        return np.tile(mono_out[np.newaxis, :], (original_shape[0], 1)).astype(np.float32)  # type: ignore[no-any-return]
    # (samples, channels)
    return np.tile(mono_out[:, np.newaxis], (1, original_shape[-1])).astype(np.float32)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def get_singer_voice_model() -> SingerVoiceModel:
    """Gibt the thread-safe singleton SingerVoiceModel (§SVM-1 v10.0.0) zurück."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SingerVoiceModel()
                logger.debug("SingerVoiceModel singleton initialised (§SVM-1 v10.0.0)")
    return _instance
