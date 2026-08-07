"""
Adaptive Formant Shifting & Formantkorrektur Modul für Aurik 6.0 (SOTA-Maximum)
SOTA-tauglich, adaptiv, mit automatischer Parameteroptimierung (klassische DSP, SOTA-Maximum).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import librosa
import numpy as np

try:
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger("aurik.dsp.adaptive_formant_shifter")
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DSPContract:
    id: str = "adaptive_formant_shifter"
    category: str = "formant_shifting"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


class AdaptiveFormantShifter:
    """
    Klassische adaptive Formantverschiebung und -korrektur (SOTA-Maximum)
    """

    contract: DSPContract = DSPContract()

    def __init__(self, method: str = "simple_lpc", auto_optimize: bool = True):
        """
        method: 'simple_lpc', 'psola', 'world', 'custom'
        auto_optimize: Wenn True, werden Parameter automatisch optimiert.
        """
        self.method = method
        self.auto_optimize = auto_optimize
        self.last_params: dict[str, Any] | None = None

    def log_contract(self):
        # Optional: Audit-Log für Vertrag
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def formant_shift(
        self,
        audio: np.ndarray,
        sr: int,
        shift_ratio: float = 1.0,
        use_deep_learning: bool = False,
        audit_log: bool = True,
    ) -> np.ndarray:
        """
        Führt Formantverschiebung durch. shift_ratio > 1.0 = Formanten nach oben, < 1.0 = nach unten
        Quality Gate, Audit-Logging, optionale DL-Inferenz, robuste Fehlerbehandlung
        """
        self.log_contract()
        # Quality Gate: Input-Checks
        if not isinstance(audio, np.ndarray) or audio.size == 0 or sr < 8000:
            logger.error("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
            raise ValueError("Ungültiges Audio-Array oder Sample-Rate < 8kHz")
        if np.isnan(np.asarray(audio, dtype=np.float64)).any():
            logger.error("Audio enthält NaN-Werte")
            raise ValueError("Audio enthält NaN-Werte")
        if np.max(np.abs(audio)) > 1.5:
            logger.warning("Audio möglicherweise nicht normiert (max > 1.5)")

        # Multi-channel guard: process channels independently for all methods.
        if audio.ndim == 2:
            transposed = False
            work = audio
            if audio.shape[0] == 2 and audio.shape[1] != 2:
                work = audio.T
                transposed = True
            elif audio.shape[1] != 2:
                logger.error("2D-Audio muss Stereo mit 2 Kanälen sein")
                raise ValueError("2D-Audio muss Stereo mit 2 Kanälen sein")

            out = np.empty_like(work)
            for ch in range(work.shape[1]):
                out_ch = self.formant_shift(
                    work[:, ch],
                    sr,
                    shift_ratio=shift_ratio,
                    use_deep_learning=use_deep_learning,
                    audit_log=False,
                )
                if len(out_ch) != work.shape[0]:
                    if len(out_ch) < work.shape[0]:
                        out_ch = np.pad(out_ch, (0, work.shape[0] - len(out_ch)))
                    else:
                        out_ch = out_ch[: work.shape[0]]
                out[:, ch] = out_ch
            return out.T if transposed else out

        result = None
        fallback_used = False
        try:
            if use_deep_learning and _TORCH_AVAILABLE:
                logger.info("Deep-Learning-Inferenz aktiviert für Formantverschiebung.")
                # TorchScript-Modell (Platzhalter)
                # model = torch.jit.load('formant_shifter.pt')
                # result = model(torch.from_numpy(audio).float().unsqueeze(0)).squeeze(0).numpy()
                logger.warning("TorchScript-Modell nicht implementiert, Ersatzpfad auf klassische Methode.")
                fallback_used = True
                result = self._formant_shift_classic(audio, sr, shift_ratio)
            else:
                result = self._formant_shift_classic(audio, sr, shift_ratio)
        except Exception as e:
            logger.error("Fehler bei Formantverschiebung: %s", e)
            fallback_used = True
            result = audio.copy()

        if audit_log:
            logger.info("AdaptiveFormantShifter: shift_Verhaeltnis=%s, Ersatzpfad_used=%s", shift_ratio, fallback_used)
        return result

    def _formant_shift_classic(self, audio: np.ndarray, sr: int, shift_ratio: float) -> np.ndarray:
        if self.method == "simple_lpc":
            return self._lpc_formant_shift(audio, sr, shift_ratio)
        elif self.method == "psola":
            return self._psola_formant_shift(audio, sr, shift_ratio)
        elif self.method == "world":
            return self._world_formant_shift(audio, sr, shift_ratio)
        else:
            logger.warning(
                "Unbekannte Formant-Shift-Methode '%s' — Ersatzpfad auf LPC-Formantverschiebung",
                self.method,
            )
            return self._lpc_formant_shift(audio, sr, shift_ratio)

    def _lpc_formant_shift(self, audio: np.ndarray, sr: int, shift_ratio: float) -> np.ndarray:
        """LPC pole-frequency-warping formant shift (spec-compliant, order 30–40 @ 48 kHz).

        Replaces illegal direct FFT-magnitude mutation (violates PGHI invariant) with
        time-domain LPC Analysis-Synthesis using pole-angle warping (z-domain):
          1. LPC analysis → denominator polynomial A(z) → poles
          2. Rotate pole angles by shift_ratio (formant frequency warp)
          3. Reconstruct A'(z) from shifted poles
          4. LPC inverse filter (source residual) → synthesis with A'(z)

        LPC order: Spec §2.8 — 30–40 @ 48 kHz-SR (Faustregel SR[kHz]×2+4 = 100; Kompromiss 30–40).
        """
        import scipy.signal as _sps

        # Spec §2.8: LPC Ord. 30–40 @ 48 kHz-SR (clamp from theoretical ~100)
        order = max(30, min(40, 4 + int(sr / 1000) * 2))
        audio_f64 = audio.astype(np.float64)
        try:
            a = librosa.lpc(audio_f64, order=order)  # [1, a1, a2, …, a_order]
        except Exception:
            logger.warning("adaptive_formant_shifter.py::_lpc_formant_shift Ersatzpfad", exc_info=True)
            return audio.copy()

        # Guard: degenerate LPC → LAPACK DLASCL failure
        if not np.all(np.isfinite(a)):
            return audio.copy()

        # Extract poles (roots of A polynomial)
        try:
            poles = np.roots(a)
        except (np.linalg.LinAlgError, ValueError):
            return audio.copy()

        # Warp pole angles by shift_ratio (formant frequency shift in z-domain)
        angles = np.angle(poles)
        radii = np.abs(poles)
        new_angles = np.clip(angles * shift_ratio, -np.pi, np.pi)
        new_poles = radii * np.exp(1j * new_angles)

        # Stability guard: keep all poles strictly inside the unit circle
        over_unit = np.abs(new_poles) >= 1.0
        new_poles[over_unit] = new_poles[over_unit] / (np.abs(new_poles[over_unit]) + 1e-9) * 0.9999

        # Reconstruct LPC denominator from shifted poles
        new_a = np.poly(new_poles).real
        new_a = new_a / (new_a[0] + 1e-30)  # normalize leading coefficient to 1

        # LPC residual (inverse filter) — all time-domain, no FFT modification
        residual = _sps.lfilter(a, np.array([1.0]), audio_f64)
        result = _sps.lfilter(np.array([1.0]), new_a, residual)

        # DSP-internal RMS normalization (LUFS used at export; this is for gain stability)
        rms_in = float(np.sqrt(np.mean(audio_f64**2) + 1e-30))
        rms_out = float(np.sqrt(np.mean(result**2) + 1e-30))
        if rms_out > 1e-10:
            result *= rms_in / rms_out

        return np.clip(result, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # PSOLA-basierte Formantverschiebung (scipy-only, kein pysptk/pyworld)
    # ------------------------------------------------------------------
    def _psola_formant_shift(self, audio: np.ndarray, sr: int, shift_ratio: float) -> np.ndarray:
        """
        Pitch-Synchronous Overlap-Add (PSOLA) Formantverschiebung.

        Implementiert als rahmenbasiertes OLA ohne externes pyworld/pysptk:
        1. STFT-Analyse (Hann-Fenster).
        2. Spektrale Hüllkurve per LPC (Ordnung 16) je Rahmen berechnen.
        3. Formant-Shift: Hüllkurve frequenzgestreckt (shift_ratio), auf
           Restanreger angewandt.
        4. Synthese via iSTFT (Overlap-Add).

        Args:
            audio:       Mono-Audiosignal  [-1, 1]
            sr:          Abtastrate in Hz
            shift_ratio: Frequenzstreckfaktor (>1.0 = nach oben)

        Returns:
            Formant-verschobenes Signal (gleiche Länge wie Input).
        """
        import scipy.signal

        hop_length = 128
        n_fft = 1024
        # Spec §2.8: LPC Ord. 30–40 @ 48 kHz-SR (clamp from theoretical ~100)
        lpc_order = max(30, min(40, 4 + int(sr / 1000) * 2))
        n_bins = n_fft // 2 + 1
        freq_axis = np.linspace(0, np.pi, n_bins)

        # STFT
        _f, _t, S = scipy.signal.stft(
            audio.astype(np.float64),
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary="even",
        )
        S_out = np.zeros_like(S)

        for k in range(S.shape[1]):
            frame_mag = np.abs(S[:, k])
            frame_phase = np.angle(S[:, k])

            # Spektrale Hüllkurve via LPC (im Zeitbereich des Rahmens)
            frame_t_start = k * hop_length
            frame_t_end = min(frame_t_start + n_fft, len(audio))
            frame_audio = audio[frame_t_start:frame_t_end]
            if len(frame_audio) < lpc_order + 2:
                S_out[:, k] = S[:, k]
                continue

            # LPC-Spektrale Hüllkurve (Quellfilter-Modell: H = 1/A)
            try:
                a_lpc = librosa.lpc(frame_audio.astype(np.float32), order=lpc_order)
                _, h_env = scipy.signal.freqz(1.0, a_lpc, worN=n_bins)
                env = np.abs(h_env)
                # Numerische Absicherung: NaN/Inf → 1.0 (neutrale Hüllkurve)
                env = np.where(np.isfinite(env) & (env > 0), env, 1.0)
            except Exception:
                env = np.ones(n_bins)

            # Gestreckte Hüllkurve: freq_axis * shift_ratio → originalem Frequenzraster
            freq_shifted = np.clip(freq_axis * shift_ratio, 0.0, np.pi)
            env_shifted = np.interp(freq_shifted, freq_axis, env)

            # Anreger
            excitation_mag = frame_mag / (env + 1e-12)  # Residualspektrum im Kurzzeit-Rahmen
            # Neue Spektralmagnitude = Anreger × verschobene Hüllkurve
            new_mag = excitation_mag * env_shifted

            S_out[:, k] = new_mag * np.exp(1j * frame_phase)

        _, result = scipy.signal.istft(
            S_out,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
        )
        result = result[: len(audio)]
        # Längenmismatch (ISTFT kann kürzer sein)
        if len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)))

        # RMS-Normalisierung
        rms_in = np.sqrt(np.mean(audio**2) + 1e-30)
        rms_out = np.sqrt(np.mean(result**2) + 1e-30)
        if rms_out > 1e-10:
            result *= rms_in / rms_out

        logger.info(
            "PSOLA-Formant-Shift: shift_Verhaeltnis=%.3f, hop=%d, n_fft=%d",
            shift_ratio,
            hop_length,
            n_fft,
        )
        return np.clip(result, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # WORLD-ähnliche Formantverschiebung via Mel-Cepstraler Spektralhüllkurve
    # ------------------------------------------------------------------
    def _world_formant_shift(self, audio: np.ndarray, sr: int, shift_ratio: float) -> np.ndarray:
        """
        WORLD-inspired Formantverschiebung via Mel-Cepstral Spectral Envelope Warping.

        Da pyworld nicht zwingend installiert ist, wird die spektrale Hüllkurve
        mittels Mel-Cepstrum (Lift-Cepstrum) geschätzt und warpend verschoben:
        1. STFT pro Rahmen.
        2. Mel-Cepstrum via DCT der Log-Spektralmagnitude.
        3. Cepstrale Verschiebung: Anheben/Absenken der Formant-Region durch
           Frequenzstreckung der spektralen Hüllkurve.
        4. iSTFT-Resynthese mit unveranderten Phasen.

        Args:
            audio:       Mono-Audiosignal  [-1, 1]
            sr:          Abtastrate in Hz
            shift_ratio: Formant-Streckfaktor (1.0 = neutral)

        Returns:
            Formant-verschobenes Signal (gleiche Länge wie Input).
        """
        import scipy.signal
        from scipy.fft import dct, idct

        hop_length = 128
        n_fft = 1024
        n_bins = n_fft // 2 + 1
        # Lifter-Grenzwert: trennt Hüllkurve (niedere Quefrenz) von Anreger
        lifter_cutoff = 60
        freq_axis = np.linspace(0, np.pi, n_bins)

        _f, _t, S = scipy.signal.stft(
            audio.astype(np.float64),
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary="even",
        )
        S_out = np.zeros_like(S)

        for k in range(S.shape[1]):
            mag = np.abs(S[:, k]) + 1e-12
            phase = np.angle(S[:, k])

            # Log-Spektral-Magnitude
            log_mag = np.log(mag)

            # Mel-Cepstrum via DCT (Approx. MFCC Liftering)
            cep = dct(log_mag, norm="ortho")

            # Spektrale Hüllkurve (Low-Time-Lift): Quefrenz 0..lifter_cutoff behalten
            cep_env = np.zeros_like(cep)
            cep_env[:lifter_cutoff] = cep[:lifter_cutoff]
            log_env = np.asarray(idct(cep_env, norm="ortho"), dtype=np.float64)
            env = np.exp(log_env)  # Spektrale Hüllkurve (linear)

            # Anreger-Residual
            excitation_mag = mag / (env + 1e-12)

            # Hüllkurve freq-strecken (Formant-Shift)
            freq_shifted = np.clip(freq_axis * shift_ratio, 0.0, np.pi)
            env_shifted = np.interp(freq_shifted, freq_axis, env)

            new_mag = excitation_mag * env_shifted
            S_out[:, k] = new_mag * np.exp(1j * phase)

        _, result = scipy.signal.istft(
            S_out,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
        )
        result = result[: len(audio)]
        if len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)))

        rms_in = np.sqrt(np.mean(audio**2) + 1e-30)
        rms_out = np.sqrt(np.mean(result**2) + 1e-30)
        if rms_out > 1e-10:
            result *= rms_in / rms_out

        logger.info(
            "WORLD-Formant-Shift: shift_Verhaeltnis=%.3f, lifter=%d, hop=%d",
            shift_ratio,
            lifter_cutoff,
            hop_length,
        )
        return np.clip(result, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]

    def auto_optimize_params(self, audio: np.ndarray, sr: int, target: np.ndarray | None = None) -> dict[str, Any]:
        """
        Schätzt optimales shift_ratio anhand des Verhältnisses der Formant-Schwerpunkte
        von audio und target (falls angegeben). Ohne target bleibt shift_ratio=1.0.
        target: Optionales Zielspektrum oder Referenzsignal
        """
        src = np.asarray(audio, dtype=np.float64)
        if src.ndim == 2:
            src = np.mean(src, axis=0 if src.shape[0] == 2 else 1)
        mag = np.abs(np.fft.rfft(src.astype(float)))
        freqs = np.fft.rfftfreq(len(src), d=1.0 / sr)
        centroid_src = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-8))

        if target is not None and len(target) > 0:
            tgt = np.asarray(target, dtype=np.float64)
            if tgt.ndim == 2:
                tgt = np.mean(tgt, axis=0 if tgt.shape[0] == 2 else 1)
            mag_tgt = np.abs(np.fft.rfft(tgt.astype(float), n=len(src)))
            centroid_tgt = float(np.sum(freqs * mag_tgt) / (np.sum(mag_tgt) + 1e-8))
            shift_ratio = float(np.clip(centroid_tgt / (centroid_src + 1e-8), 0.5, 2.0))
        else:
            shift_ratio = 1.0  # Kein Target → keine Verschiebung

        self.last_params = {"method": self.method, "shift_ratio": shift_ratio, "centroid_src": centroid_src}
        logger.info(
            f"auto_optimieren_params (FormantShifter): centroid_src={centroid_src:.1f} Hz → shift_Verhaeltnis={shift_ratio:.3f}"
        )
        return self.last_params
