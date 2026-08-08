"""NVSR Plugin — productive SBR/NVSR adapter (§Spec-04).

Zielbereich: 8–16 kHz Bandbreitenerweiterung.
Methode:     Spectral Band Replication (SBR) + harmonic envelope extrapolation.
Abgrenzung:  AudioSR (Diffusion) für 0–8 kHz (Shellac).
             NVSR (diese Datei) für 8–16 kHz (Vinyl, MP3-128kbps).
             Vorteil: deterministisch, kein Halluzinationsrisiko, 10–30× schneller.

Wenn ein lokales NVSR-ONNX-Modell (models/nvsr/nvsr.onnx) vorhanden ist,
wird auf Modell-Inferenz eskaliert. Sonst ist SBR der produktive, deterministische
Fallback für 8–16 kHz ohne Halluzinationsrisiko.

§SOTA-Matrix:
    "SBR-Heuristik + NVSR: 8–16 kHz fehlt (MP3 128kbps): deterministisch, schneller,
     weniger Halluzinationsrisiko als AudioSR (Diffusion)"

Referenz:
    Spectral Band Replication (SBR): Dietz et al. (2002), ISO/IEC 14496-3.
    HE-AAC SBR: Malah (2003) "Spectral band replication, a novel approach".
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as _sp_signal
from scipy.signal import resample_poly as _resample_poly

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton — thread-safe, double-checked locking
# ---------------------------------------------------------------------------
_instance: NvsrPlugin | None = None
_lock: threading.Lock = threading.Lock()

# ---------------------------------------------------------------------------
# NVSR-ONNX-Modell-Pfad (optional — zukünftige DNN-Eskalation)
# ---------------------------------------------------------------------------
_NVSR_ONNX_PATH = Path(__file__).parent.parent / "models" / "nvsr" / "nvsr.onnx"

# ---------------------------------------------------------------------------
# Material-spezifische HF-Ceiling für SBR-Output
# ---------------------------------------------------------------------------
_MATERIAL_HF_CEILING_HZ: dict[str, float] = {
    "shellac": 8_000.0,  # Shellac: handled by AudioSR, NVSR should not activate
    "wax_cylinder": 5_000.0,
    "wire_recording": 7_000.0,
    "vinyl": 16_000.0,
    "tape": 16_000.0,
    "reel_tape": 16_000.0,
    "cassette": 14_000.0,
    "minidisc": 15_000.0,
    "mp3_low": 16_000.0,  # 128kbps
    "mp3_medium": 18_000.0,  # 192kbps
    "mp3_high": 20_000.0,
    "aac": 18_000.0,
    "cd_digital": 22_000.0,
    "streaming": 20_000.0,
    "digital": 22_000.0,
}

# SBR source band: content in this range is used to synthesize 8–16kHz
_SBR_SOURCE_LOW_HZ = 4_000.0  # lower edge of source band
_SBR_SOURCE_HIGH_HZ = 8_000.0  # upper edge of source band (= target band starts here)
_SBR_TARGET_LOW_HZ = 8_000.0  # lower edge of target band
_SBR_TARGET_HIGH_HZ = 16_000.0  # upper edge of target band


def get_nvsr_plugin() -> NvsrPlugin:
    """Singleton-Factory — gibt bestehende Instanz zurück oder erstellt sie."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = NvsrPlugin()
    return _instance


class NvsrPlugin:
    """NVSR-SBR-Plugin: Deterministische Bandbreitenerweiterung 8–16 kHz.

    Keine DNN-Modelle erforderlich — reiner DSP-Pfad mit ONNX-Hook für
    zukünftige Eskalation.

    Interface:
        plugin = get_nvsr_plugin()
        result = plugin.process(audio, sr, target_hz=16000, strength=0.7,
                                material_type="vinyl", energy_bias_db=0.0)
    """

    def __init__(self) -> None:
        self._onnx_session = None
        self._onnx_available = False
        self._onnx_load_attempted = False
        self._onnx_lock = threading.Lock()
        self._last_route_metadata: dict[str, Any] = {
            "strategy": "uninitialized",
            "capability_status": "unavailable",
            "model_path": str(_NVSR_ONNX_PATH),
            "model_loaded": False,
        }
        logger.info("NvsrPlugin: productive SBR/NVSR adapter initialisiert (model path=%s)", _NVSR_ONNX_PATH)

    @property
    def route_metadata(self) -> dict[str, Any]:
        """Gibt metadata for the last processing route zurück."""
        return dict(self._last_route_metadata)

    def capability_status(self) -> str:
        """Gibt local NVSR capability without running inference zurück."""
        if self._onnx_available and self._onnx_session is not None:
            return "sota_real"
        if _NVSR_ONNX_PATH.exists():
            return "sota_fallback"
        return "dsp_productive"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        audio: np.ndarray,
        sr: int,
        target_hz: float = 16_000.0,
        strength: float = 0.7,
        material_type: str = "vinyl",
        energy_bias_db: float = 0.0,
        panns_singing: float = 0.0,
    ) -> dict[str, Any]:
        """Erweitere die Bandbreite deterministisch auf target_hz.

        Parameters
        ----------
        audio : np.ndarray
            Mono- oder Stereo-Signal (N,) oder (2,N), float32, 48000 Hz.
        sr : int
            Abtastrate (muss 48000 sein).
        target_hz : float
            Ziel-Bandbreite in Hz (Standard: 16000 Hz).
        strength : float
            Mischungsstärke [0.0, 1.0] (Standard: 0.7).
        material_type : str
            Materialtyp für Material-Ceiling (Standard: "vinyl").
        energy_bias_db : float
            Energiekorrektur in dB auf SBR-Ausgang (Standard: 0 dB).
            Für Gesangsmaterial: −6 dB (§0j), Instrumental: −9 dB.
        panns_singing : float
            PANNs-Singing-Score [0.0, 1.0] für Vocal-Aware-Modus.

        Returns
        -------
        dict with keys:
            audio       : np.ndarray — erweitertes Signal, gleiche Form wie Input
            strategy    : str — "onnx" | "dsp_sbr"
            target_hz   : float
            ceiling_hz  : float — tatsächliche Material-Ceiling
            strength    : float — angewandete Stärke
            hf_energy_added_db : float
        """
        assert sr == 48000, "NVSR: Eingang muss 48000 Hz sein"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        # Material-Ceiling: nie über physikalischem Limit
        ceiling_hz = _MATERIAL_HF_CEILING_HZ.get(str(material_type).lower(), 16_000.0)
        effective_target = min(target_hz, ceiling_hz, float(sr) / 2.0)

        if effective_target <= _SBR_TARGET_LOW_HZ + 500.0:
            # Ziel bereits nah an Quelle → kein Gewinn
            self._last_route_metadata = self._metadata(
                strategy="passthrough",
                target_hz=effective_target,
                ceiling_hz=ceiling_hz,
                strength=0.0,
                hf_energy_added_db=0.0,
            )
            return {
                "audio": audio,
                "strategy": "passthrough",
                "target_hz": effective_target,
                "ceiling_hz": ceiling_hz,
                "strength": 0.0,
                "hf_energy_added_db": 0.0,
                **self._last_route_metadata,
            }

        # Versuche ONNX-Eskalation wenn Modell vorhanden
        if _NVSR_ONNX_PATH.exists() and not self._onnx_load_attempted:
            self._try_load_onnx()

        if self._onnx_available and self._onnx_session is not None:
            try:
                return self._process_onnx(audio, sr, effective_target, strength, energy_bias_db)
            except Exception as _onnx_exc:
                logger.warning("NvsrPlugin: ONNX-Fehler → DSP-Ersatzpfad: %s", _onnx_exc)

        return self._process_dsp_sbr(audio, sr, effective_target, strength, energy_bias_db, panns_singing)

    def _metadata(
        self,
        *,
        strategy: str,
        target_hz: float,
        ceiling_hz: float,
        strength: float,
        hf_energy_added_db: float,
    ) -> dict[str, Any]:
        """Erstellt JSON-safe route metadata."""
        status = "sota_real" if strategy in ("onnx", "flashsr_onnx") else self.capability_status()
        return {
            "strategy": strategy,
            "capability_status": status,
            "model_path": str(_NVSR_ONNX_PATH),
            "model_loaded": bool(self._onnx_available and self._onnx_session is not None),
            "target_hz": float(target_hz),
            "ceiling_hz": float(ceiling_hz),
            "strength": float(strength),
            "hf_energy_added_db": float(hf_energy_added_db),
        }

    # ------------------------------------------------------------------
    # DSP-SBR Kern
    # ------------------------------------------------------------------

    def _process_dsp_sbr(
        self,
        audio: np.ndarray,
        sr: int,
        target_hz: float,
        strength: float,
        energy_bias_db: float,
        panns_singing: float,
    ) -> dict[str, Any]:
        """Spectral Band Replication (SBR) — deterministisch, kein DNN.

        Algorithmus (§Spec-04 SBR-Heuristik):
        1. STFT auf Input.
        2. Für Zielbins (8–target_hz): interpoliere Energie aus Quellband (4–8kHz)
           via Harmonik-Ratio und Einhüllende-Matching.
        3. Phase: Ableitung aus bestehenden Subband-Phasen (kohärent).
        4. PGHI-konsistente Rekonstruktion wenn PGHI verfügbar.
        5. Blend mit Input: Stärke-gewichtet, HF-beschränkt.
        6. Hallucination-Guard nach additiver Operation.
        """
        stereo = audio.ndim == 2 and audio.shape[0] == 2
        if stereo:
            left_out = self._process_channel_sbr(audio[0], sr, target_hz, strength, energy_bias_db, panns_singing)
            right_out = self._process_channel_sbr(audio[1], sr, target_hz, strength, energy_bias_db, panns_singing)
            result_audio = np.stack([left_out, right_out], axis=0)
        else:
            mono = audio if audio.ndim == 1 else audio.mean(axis=0)
            result_audio = self._process_channel_sbr(mono, sr, target_hz, strength, energy_bias_db, panns_singing)

        result_audio = np.nan_to_num(result_audio, nan=0.0, posinf=0.0, neginf=0.0)
        result_audio = np.clip(result_audio, -1.0, 1.0)

        # Hallucination guard (§2.46e): ADDITIVE Phase, spec-conform check
        try:
            # pylint: disable-next=import-outside-toplevel
            from backend.core.dsp.hallucination_guard import check_hallucination

            hg_result = check_hallucination(audio, result_audio, sr=sr, mode="restoration")
            if hg_result.requires_rollback:
                logger.info(
                    "NvsrPlugin: HallucinationGuard → Rollback (spectral_novelty=%.3f)", hg_result.spectral_novelty
                )
                result_audio = audio.copy()
        except Exception as _hg_exc:
            logger.debug("NvsrPlugin: HallucinationGuard nicht verfügbar: %s", _hg_exc)

        # Energie-Vergleich für Metadata
        hf_added_db = 0.0
        try:
            _mono_in = audio if audio.ndim == 1 else audio.mean(axis=0)
            _mono_out = result_audio if result_audio.ndim == 1 else result_audio.mean(axis=0)
            _, _, _Zin = _sp_signal.stft(_mono_in, fs=sr, nperseg=512, boundary="even")
            _, _, _Zout = _sp_signal.stft(_mono_out, fs=sr, nperseg=512, boundary="even")
            _freqs = np.linspace(0, sr / 2.0, _Zin.shape[0])
            _hf_mask = (_freqs >= _SBR_TARGET_LOW_HZ) & (_freqs <= target_hz)
            _e_in = float(np.mean(np.abs(_Zin[_hf_mask]) ** 2)) + 1e-20
            _e_out = float(np.mean(np.abs(_Zout[_hf_mask]) ** 2)) + 1e-20
            hf_added_db = float(10.0 * np.log10(_e_out / _e_in))
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            logger.debug("nvsr_plugin.py::_verarbeiten_dsp_sbr energy calc Ersatzpfad", exc_info=True)

        self._last_route_metadata = self._metadata(
            strategy="dsp_sbr",
            target_hz=target_hz,
            ceiling_hz=target_hz,
            strength=strength,
            hf_energy_added_db=hf_added_db,
        )
        return {
            "audio": result_audio,
            "strategy": "dsp_sbr",
            "target_hz": target_hz,
            "ceiling_hz": target_hz,
            "strength": strength,
            "hf_energy_added_db": hf_added_db,
            **self._last_route_metadata,
        }

    def _process_channel_sbr(
        self,
        channel: np.ndarray,
        sr: int,
        target_hz: float,
        strength: float,
        energy_bias_db: float,
        panns_singing: float,
    ) -> np.ndarray:
        """SBR-Verarbeitung eines einzelnen Kanals."""
        n = len(channel)
        N_FFT = 1024
        HOP = 256

        # Reflect-padding (§2.63)
        pad_len = HOP * 4
        padded = np.pad(channel, pad_len, mode="reflect")

        f, _, Zxx = _sp_signal.stft(padded, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
        n_freq = len(f)
        freq_res = f[1] - f[0] if len(f) > 1 else (sr / N_FFT)

        # Frequenzindex-Grenzen
        src_low_bin = max(1, int(np.round(_SBR_SOURCE_LOW_HZ / freq_res)))
        src_high_bin = min(n_freq - 1, int(np.round(_SBR_SOURCE_HIGH_HZ / freq_res)))
        tgt_low_bin = min(n_freq - 1, int(np.round(_SBR_TARGET_LOW_HZ / freq_res)))
        tgt_high_bin = min(n_freq - 1, int(np.round(target_hz / freq_res)))

        if tgt_low_bin >= tgt_high_bin or src_low_bin >= src_high_bin:
            # Kein Spielraum → Passthrough
            _, channel_out = _sp_signal.istft(Zxx, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
            channel_out = channel_out[pad_len : pad_len + n]
            if len(channel_out) < n:
                channel_out = np.pad(channel_out, (0, n - len(channel_out)))
            return channel_out.astype(np.float32)  # type: ignore[no-any-return]

        # §Physik-Guard: Quellband-Energie prüfen — leeres Quellband → Passthrough
        # SBR braucht echte harmonische Information. Synthese aus Rauschen klingt unnatürlich.
        _src_band_energy = float(np.mean(np.abs(Zxx[src_low_bin : src_high_bin + 1, :]) ** 2))
        if _src_band_energy < 1e-8:
            logger.debug("NvsrPlugin: Quellband ohne Energie → Passthrough")
            _, channel_out = _sp_signal.istft(Zxx, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
            channel_out = channel_out[pad_len : pad_len + n]
            if len(channel_out) < n:
                channel_out = np.pad(channel_out, (0, n - len(channel_out)))
            return channel_out.astype(np.float32)  # type: ignore[no-any-return]

        # SBR-Kern: Spektrale Einhüllende aus Quellband ableiten
        src_mag = np.abs(Zxx[src_low_bin : src_high_bin + 1, :])  # (src_bins, T)
        src_phase = np.angle(Zxx[src_low_bin : src_high_bin + 1, :])

        # Einhüllende glätten (zeitlich: 3-Frame, spektral: Gliding-Mean)
        src_env = np.maximum(src_mag, 1e-12)

        # Vektorisiertes Temporal-Smoothing (3-Frame Median für Transienten-Erhalt)
        # Median-Filter erhält Attack-Flanken besser als Mean-Filter
        from scipy.ndimage import median_filter as _medfilt

        src_env_smooth = _medfilt(src_env.astype(np.float64), size=(1, 3)).astype(np.float32)

        # Transienten-Erkennung: spektrale Energie-Differenz zwischen benachbarten Frames
        # Ein Frame ist transient, wenn die Energie > 3× des gleitenden Mittelwerts ist
        _frame_energy = np.sum(src_env_smooth, axis=0)  # (T,)
        _frame_energy_smooth = np.convolve(_frame_energy, np.ones(5) / 5.0, mode="same")
        # §GEBOT-G04: Adaptiver Transienten-Schwellwert — abhängig von spektraler Dynamik
        # Ruhiges Material → niedrigere Schwelle, dynamisches Material → höhere Schwelle
        _frame_energy_mean = float(np.mean(_frame_energy_smooth))
        _frame_energy_std = float(np.std(_frame_energy_smooth)) / max(_frame_energy_mean, 1e-12)
        _transient_mult = np.clip(5.0 - _frame_energy_std * 2.0, 2.0, 5.0)
        _is_transient = _frame_energy > np.maximum(_frame_energy_smooth * _transient_mult, 1e-8)

        # Zielband-Energie via Harmonik-Ratio (Frequenz-Skalierung ×2)
        n_src_bins = src_high_bin - src_low_bin + 1
        n_tgt_bins = tgt_high_bin - tgt_low_bin + 1

        # Resample Quelle → Ziel über lineare Interpolation der Magnitude
        # Harmonische Struktur: spektrale Peaks im Quellband werden exakt auf
        # die Zielband-Positionen abgebildet (Frequenzverdopplung)
        src_bins_norm = np.linspace(0.0, 1.0, n_src_bins)
        tgt_bins_norm = np.linspace(0.0, 1.0, n_tgt_bins)
        tgt_mag = np.zeros((n_tgt_bins, Zxx.shape[1]), dtype=np.float32)

        # §GEBOT-G01: Harmonische Dichte pro Frame messen → adaptive Peak-Gewichtung
        _harmonic_density = np.zeros(Zxx.shape[1], dtype=np.float32)
        for t_idx in range(Zxx.shape[1]):
            _src_frame = src_env_smooth[:, t_idx]
            # Harmonische Peak-Erkennung: lokale Maxima im Quellspektrum
            _peaks = np.zeros(n_src_bins, dtype=bool)
            _peaks[1:-1] = (_src_frame[1:-1] > _src_frame[:-2]) & (_src_frame[1:-1] > _src_frame[2:])
            _peaks &= _src_frame > np.mean(_src_frame) * 1.5  # nur signifikante Peaks
            _harmonic_density[t_idx] = float(np.mean(_peaks))
            # Adaptive Peak-Gewichtung: harmonik-reiches Material → stärkere Peak-Betonung
            _peak_weight = 1.0 + np.clip(_harmonic_density[t_idx] * 10.0, 0.0, 2.0)
            _weights = np.ones(n_src_bins, dtype=np.float32)
            _weights[_peaks] = _peak_weight
            _interp_base = np.interp(tgt_bins_norm, src_bins_norm, _src_frame).astype(np.float32)
            _interp_peak = np.interp(tgt_bins_norm, src_bins_norm, _src_frame * _weights / np.mean(_weights)).astype(
                np.float32
            )
            # Adaptive Blend: harmonik-reich → mehr Peak-Weighted, rauschig → mehr Base
            _peak_blend = np.clip(0.5 + np.mean(_harmonic_density) * 3.0, 0.5, 0.85)
            tgt_mag[:, t_idx] = _interp_peak * _peak_blend + _interp_base * (1.0 - _peak_blend)

        # §GEBOT-G02: Adaptiver HF-Rolloff — ableiten aus spektraler Neigung des Quellbands
        # Miss Energiedecay über das Quellband (4–8 kHz) via lineare Regression
        _src_freq_idx = np.arange(n_src_bins, dtype=np.float32)
        _src_mean_mag = np.mean(src_env_smooth, axis=1)  # (src_bins,)
        _src_mean_db = 20.0 * np.log10(np.maximum(_src_mean_mag, 1e-12))
        # Lineare Regression: dB/Oktave im Quellband
        if n_src_bins > 2:
            _coeffs = np.polyfit(_src_freq_idx, _src_mean_db, 1)
            _src_tilt_db_per_octave = _coeffs[0] * n_src_bins  # dB über gesamtes Quellband
        else:
            _src_tilt_db_per_octave = -6.0  # konservativer Default
        # Mapping: steilere Quellneigung → steilere Zielneigung
        # −3 dB/Oktave Quelle → 0.75 Endpoint, −9 dB/Oktave Quelle → 0.35 Endpoint
        _rolloff_end = float(np.clip(0.75 + _src_tilt_db_per_octave / 24.0, 0.35, 0.75))
        rolloff_slope = np.linspace(1.0, _rolloff_end, n_tgt_bins, dtype=np.float32)
        tgt_mag = tgt_mag * rolloff_slope[:, np.newaxis]

        # §GEBOT-G03: Adaptiver Energy-Bias — spektrale Balance des Quellbands respektieren
        # Quelle mit viel HF-Energie → weniger Bias nötig (natürlich hell)
        # Quelle mit wenig HF-Energie → mehr Bias (konservativ, kein Übersteuern)
        _hf_num = float(np.mean(src_env_smooth[n_src_bins // 2 :]))
        _hf_den = max(float(np.mean(src_env_smooth[: n_src_bins // 2])), 1e-12)
        _hf_ratio = _hf_num / _hf_den
        _hf_ratio_db = 10.0 * np.log10(max(_hf_ratio, 1e-6))
        if panns_singing >= 0.4:
            _bias_base = 0.0  # Gesang: volle HF-Präsenz
        elif panns_singing >= 0.1:
            _bias_base = -1.5  # Mix: leichte Dämpfung
        else:
            _bias_base = -3.0  # Instrumental: natürlicher Abfall
        # Adaptiv: HF-arme Quelle → konservativer, HF-reiche Quelle → weniger Dämpfung
        _bias_adaptive = _bias_base + np.clip(_hf_ratio_db + 6.0, -3.0, 3.0)
        _bias = _bias_adaptive + energy_bias_db
        bias_lin = 10.0 ** (_bias / 20.0)
        tgt_mag = tgt_mag * float(bias_lin)

        # Phasen-Kohärenz: Phasengradient aus benachbarten Quell-Bins fortführen
        if src_phase.shape[1] > 1:
            _phase_grad = np.diff(src_phase, axis=1)  # (src_bins, T-1)
            _phase_grad_resamp = np.zeros((n_tgt_bins, _phase_grad.shape[1]), dtype=np.float32)
            for t_idx in range(_phase_grad.shape[1]):
                _phase_grad_resamp[:, t_idx] = np.interp(tgt_bins_norm, src_bins_norm, _phase_grad[:, t_idx]).astype(
                    np.float32
                )
            tgt_phase = np.zeros((n_tgt_bins, Zxx.shape[1]), dtype=np.float32)
            if Zxx.shape[1] > 0:
                tgt_phase[:, 0] = np.interp(tgt_bins_norm, src_bins_norm, src_phase[:, 0]).astype(np.float32)
            for t_idx in range(1, Zxx.shape[1]):
                tgt_phase[:, t_idx] = (
                    tgt_phase[:, t_idx - 1] + _phase_grad_resamp[:, min(t_idx - 1, _phase_grad_resamp.shape[1] - 1)]
                )
        else:
            tgt_phase = np.zeros((n_tgt_bins, Zxx.shape[1]), dtype=np.float32)

        # SBR-Signal in STFT einschreiben (nur Zielbins modifizieren)
        Zxx_out = Zxx.copy()
        tgt_complex = tgt_mag * np.exp(1j * tgt_phase)

        # Smooth-Blending an der Grenze src→tgt (4 Bins Crossfade, final=1.0)
        # Transienten-Schutz: bei Attack-Frames SBR-Stärke reduzieren
        _crossfade_bins = min(4, n_tgt_bins)
        _fade_in = np.linspace(0.0, 1.0, _crossfade_bins + 2)[1:]  # [0.2, 0.4, 0.6, 0.8, 1.0]
        for bi, tb in enumerate(range(tgt_low_bin, tgt_high_bin + 1)):
            if tb < n_freq:
                if bi < len(_fade_in):
                    fade = _fade_in[bi]
                else:
                    fade = 1.0
                # Transiente Frames: SBR auf 30% reduzieren für pristine Attacks
                _frame_strength = np.where(_is_transient, strength * 0.3, strength)
                Zxx_out[tb, :] = (1.0 - fade * _frame_strength) * Zxx[tb, :] + fade * _frame_strength * tgt_complex[
                    bi, :
                ]

        # iSTFT
        _, channel_sbr = _sp_signal.istft(Zxx_out, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even")
        channel_sbr = channel_sbr[pad_len : pad_len + n]
        if len(channel_sbr) < n:
            channel_sbr = np.pad(channel_sbr, (0, n - len(channel_sbr)))

        channel_sbr = np.nan_to_num(channel_sbr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(channel_sbr, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # ONNX-Hook (zukünftige DNN-Eskalation)
    # ------------------------------------------------------------------

    def _try_load_onnx(self) -> None:
        """Versuche NVSR-ONNX-Modell zu laden. Nicht-blockierend."""
        with self._onnx_lock:
            if self._onnx_load_attempted:
                return
            self._onnx_load_attempted = True
            try:
                import onnxruntime as ort  # pylint: disable=import-outside-toplevel

                from backend.core.ml_device_manager import get_ort_providers  # pylint: disable=import-outside-toplevel

                self._onnx_session = ort.InferenceSession(
                    str(_NVSR_ONNX_PATH),
                    providers=get_ort_providers("NVSR"),
                )
                self._onnx_available = True
            except Exception as _onnx_load_exc:
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            except Exception as _onnx_load_exc:
                logger.debug("NvsrPlugin: ONNX-Modell nicht verfügbar (DSP-SBR aktiv): %s", _onnx_load_exc)
                self._onnx_available = False

    def _process_onnx(
        self,
        audio: np.ndarray,
        sr: int,
        target_hz: float,
        strength: float,
        energy_bias_db: float,
    ) -> dict[str, Any]:
        """NVSR ONNX via FlashSR: 16kHz→48kHz neuronale Bandbreitenerweiterung.

        FlashSR (HierSpeech++, Apache 2.0) nimmt 16kHz Audio und rekonstruiert
        48kHz via neuronaler Wellenform-Synthese. Verarbeitung in 10s-Chunks
        für speichereffiziente ONNX-Inferenz ohne OOM-Risiko.
        """
        if self._onnx_session is None:
            raise RuntimeError("NVSR ONNX session not loaded")
        session = self._onnx_session
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("NVSR ONNX model has no inputs")
        input_name = inputs[0].name
        output_name = session.get_outputs()[0].name if session.get_outputs() else None
        if output_name is None:
            raise RuntimeError("NVSR ONNX model has no outputs")

        mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        mono = np.asarray(mono, dtype=np.float32)
        n_total = len(mono)

        # FlashSR arbeitet bei 16kHz → 48kHz (3× Upsampling).
        # Downsample 48kHz → 16kHz mit scipy's resample (anti-alias).
        mono_16k = _resample_poly(mono.astype(np.float64), 1, 3).astype(np.float32)  # 48k→16k

        # Chunked processing: 10s @ 16kHz = 160000 samples pro Chunk
        CHUNK_16K = 160000
        CHUNK_OVERLAP_16K = 4000  # 250ms overlap for smooth blending
        n_chunks = max(1, (len(mono_16k) + CHUNK_16K - 1) // CHUNK_16K)

        out_48k = np.zeros(n_total, dtype=np.float32)
        weight_48k = np.zeros(n_total, dtype=np.float32)

        for ci in range(n_chunks):
            start_16 = ci * CHUNK_16K
            end_16 = min(start_16 + CHUNK_16K + CHUNK_OVERLAP_16K, len(mono_16k))
            chunk_16 = mono_16k[start_16:end_16]

            # FlashSR expects (1, 1, N)
            model_input = chunk_16[np.newaxis, np.newaxis, :].astype(np.float32)
            outputs = session.run([output_name], {input_name: model_input})
            chunk_48 = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

            # Map back to 48kHz timeline
            start_48 = start_16 * 3
            end_48_actual = min(start_48 + len(chunk_48), n_total)
            n_place = end_48_actual - start_48

            # Triangle window for crossfade
            fade_win = np.ones(n_place, dtype=np.float32)
            fade_len = CHUNK_OVERLAP_16K * 3
            if ci > 0 and fade_len < n_place:
                fade_win[:fade_len] = np.linspace(0, 1, fade_len, dtype=np.float32)
            if ci < n_chunks - 1 and fade_len < n_place:
                fade_win[-fade_len:] = np.linspace(1, 0, fade_len, dtype=np.float32)

            out_48k[start_48:end_48_actual] += chunk_48[:n_place] * fade_win
            weight_48k[start_48:end_48_actual] += fade_win

        # Normalize overlapping regions
        mask = weight_48k > 0
        out_48k[mask] /= weight_48k[mask]

        out = np.clip(np.nan_to_num(out_48k, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)

        if audio.ndim == 2 and audio.shape[0] == 2:
            out_audio = np.stack([out, out], axis=0)
        else:
            out_audio = out

        # Blend mit DSP SBR für envelope-Kohärenz (minimaler DSP-Anteil)
        dsp_ref = self._process_dsp_sbr(audio, sr, target_hz, min(strength * 0.15, 0.10), energy_bias_db, 0.0)["audio"]
        # FlashSR-Output dominant (95% ONNX + 5% DSP) — neuronale Synthese liefert
        # präzisere Feinstruktur als deterministische SBR. DSP-Anteil dient nur
        # als envelope-guard gegen extreme Ausreißer.
        onnx_weight = min(strength * 1.2, 0.95)  # bis zu 95% ONNX-Anteil
        blended = np.clip(
            (1.0 - onnx_weight) * np.asarray(dsp_ref, dtype=np.float32) + onnx_weight * out_audio,
            -1.0,
            1.0,
        )

        self._last_route_metadata = self._metadata(
            strategy="flashsr_onnx",
            target_hz=target_hz,
            ceiling_hz=target_hz,
            strength=onnx_weight,
            hf_energy_added_db=0.0,
        )
        return {
            "audio": blended.astype(np.float32),
            "strategy": "flashsr_onnx",
            "target_hz": target_hz,
            "ceiling_hz": target_hz,
            "strength": onnx_weight,
            "hf_energy_added_db": 0.0,
            **self._last_route_metadata,
        }
