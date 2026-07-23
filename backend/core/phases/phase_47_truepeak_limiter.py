"""
Phase 47: TruePeak Limiter v2.0 — ITU-R BS.1770 / AES17 konform
================================================================

Vollständige DSP-Implementierung des True-Peak-Limiters ohne ML-Abhängigkeiten.
Ersetzt den kaputten aurik_ml.mastering-Stub.

ALGORITHMUS:
  1. 4× Oversampling via Polyphase-Filter (scipy.signal.resample_poly)
     → Ermittelt Zwischenwerte zwischen Samples; nur so messbar ob TP > Ceiling
  2. True-Peak-Messung auf überabgetasteten Signal (AES17-2020, §12.1.3)
  3. Lookahead-Limiter (12 ms Lookahead):
     - Smooth Gain Reduction: attack=0.1 ms, release=100 ms
     - Keine Saturation / Clipping — nur Pegelanpassung
  4. Downsample zurück auf Eingangs-SR
  5. Ceiling-Clip (Hard Limit) als Notfall-Fallback (−0.1 dBFS default)

WARUM 4×-OVERSAMPLING?
  Nyquist-Frequenz begrenzt die Rekonstruktion auf Sample-Werte; Sinussignale
  nahe Nyquist können Inter-Sample-Peaks haben, die 3 dB über dem Sample-Wert
  liegen. Standard-Peak-Messung übersieht diese. ITU-R BS.1770 schreibt daher
  4-faches Oversampling für True-Peak-Messung vor.

INDUSTRIAL STANDARD:
  - AES17-2020: True Peak Level Definition
  - ITU-R BS.1770-4: Loudness Measurement, Annex 2 (True Peak)
  - EBU R 128: −23 LUFS, Max TP = −1 dBTP

Author: Aurik Development Team
Version: 2.0.0 (keine ML-Import-Abhängigkeit)
"""

from __future__ import annotations

import os
import logging
import time

import numpy as np
import scipy.signal as sig

from backend.core.audio_utils import to_channels_last
from backend.core.phase_strength_contract import resolve_phase_strength_contract

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# Standard-Ceiling: −0.5 dBFS entspricht AES/EBU Produktions-Standard
_DEFAULT_CEILING_DBFS = -0.5
_DEFAULT_CEILING_LIN = 10 ** (_DEFAULT_CEILING_DBFS / 20.0)
_OVERSAMPLING = 4  # ITU-R BS.1770: 4× minimum
_LOOKAHEAD_MS = 12.0  # Lookahead-Fenster
_ATTACK_MS = 0.1  # Gain-Reduction Attack
_RELEASE_MS = 100.0  # Gain-Reduction Release


class TruePeakLimiterPhase(PhaseInterface):
    """
    TruePeak Limiter nach ITU-R BS.1770 / AES17.

    Letzte Instanz in der Signalkette (priority=9).
    Verhindert digitales Clipping und überschrittene True-Peak-Level.
    Kein ML-Import — reine DSP-Implementierung.
    """

    _PHASE_ID = "phase_47_truepeak_limiter"
    _NAME = "TruePeak Limiter (ITU-R BS.1770)"
    description = (
        "TruePeak Limiter nach ITU-R BS.1770: 4× Oversampling für exakte "
        "Zwischensample-Peak-Detektion, Lookahead Gain Reduction, "
        "keine Clipping-Ansteuerung. Last-in-chain (priority=9)."
    )

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self._PHASE_ID,
            name=self._NAME,
            category=PhaseCategory.DYNAMICS,
            priority=9,
            version="2.0.0",
            dependencies=[],
            estimated_time_factor=0.02,
            memory_requirement_mb=40,
            is_cpu_intensive=False,
            is_io_intensive=False,
            quality_impact=0.98,
            description=self.description,
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        material_type: str = "unknown",
        **kwargs,
    ) -> PhaseResult:
        """
        Limitiert True-Peak auf ceiling.

        Args:
            audio:       Mono oder Stereo (float32/64, ±1 normiert)
            sample_rate: Abtastrate in Hz
            material_type: Unbenutzt, nur fuer kanonische PhaseInterface-Signatur.
            **kwargs:    ceiling_dbfs (float, Default −0.5 dBFS)
        """
        sample_rate = kwargs.get("sample_rate", 48000)
        assert sample_rate == 48000, f"SR muss 48000 Hz sein, erhalten: {sample_rate}"
        self.validate_input(audio)
        audio, _p47_transposed = to_channels_last(audio)
        t0 = time.time()

        _strength_ctx = resolve_phase_strength_contract(kwargs)
        phase_locality_factor = float(_strength_ctx["phase_locality_factor"])
        effective_strength = float(_strength_ctx["effective_strength"])

        if effective_strength <= 1e-6:
            dry = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            dry = np.clip(dry, -1.0, 1.0)
            return PhaseResult(
                success=True,
                audio=dry.astype(audio.dtype),
                execution_time_seconds=time.time() - t0,
                metadata={
                    "algorithm": "skipped_zero_strength",
                    "phase_locality_factor": phase_locality_factor,
                    "effective_strength": 0.0,
                    "rms_drop_db": 0.0,
                    "loudness_makeup_db": 0.0,
                },
                metrics={"effective_strength": 0.0},
            )

        # ── §PRE-LIMITER-HIGHPASS: Infraschall-Energie vor Limiting entfernen ──
        # Sub-Bass < 20 Hz ist für das menschliche Ohr unhörbar (ISO 226),
        # verbraucht aber wertvollen Limiter-Headroom und triggert unnötige
        # Gain-Reduction auf hörbaren Frequenzen. Ein 20 Hz LR4 Highpass
        # (Linkwitz-Riley 4. Ordnung, −24 dB/Oktave) entfernt Infraschall,
        # ohne den hörbaren Bass (>30 Hz) anzutasten.
        try:
            from scipy.signal import butter, sosfiltfilt

            _hp_freq = 20.0
            _hp_sos = butter(4, _hp_freq / (sample_rate / 2), btype="high", output="sos")
            is_stereo = audio.ndim == 2
            if is_stereo:
                audio_hp = np.column_stack(
                    [
                        sosfiltfilt(_hp_sos, audio[:, 0]),
                        sosfiltfilt(_hp_sos, audio[:, 1]),
                    ]
                )
            else:
                audio_hp = sosfiltfilt(_hp_sos, audio)
            audio = audio_hp.astype(np.float32)
            logger.debug("Phase47 §Pre-Limiter-HP: 20 Hz LR4 applied")
        except Exception as _hp_exc:
            logger.debug("Phase47 §Pre-Limiter-HP non-blocking: %s", _hp_exc)

        ceiling_dbfs: float = float(kwargs.get("ceiling_dbfs", _DEFAULT_CEILING_DBFS))
        ceiling_lin: float = 10 ** (ceiling_dbfs / 20.0)

        is_stereo = audio.ndim == 2
        if is_stereo:
            # §2.51 Linked True Peak: derive gain reduction from the channel with the
            # higher instantaneous peak so both channels get the same gain curve.
            # This prevents independent L/R gain divergence → stereo phase cancellation.
            combined = np.where(np.abs(audio[:, 0]) >= np.abs(audio[:, 1]), audio[:, 0], audio[:, 1])
            n = len(combined)
            if n < 16:
                processed = np.clip(audio, -ceiling_lin, ceiling_lin)
            else:
                gain = self._compute_gain_curve(combined, sample_rate, ceiling_lin)
                left = audio[:n, 0] * gain
                right = audio[:n, 1] * gain
                processed = np.column_stack([left, right])
        else:
            processed = self._limit_channel(audio, sample_rate, ceiling_lin)

        if 0.0 < effective_strength < 1.0:
            processed = audio + effective_strength * (processed - audio)

        # Notfall-Hard-Clip (sollte nach Limiter nicht nötig sein)
        processed = np.clip(processed, -ceiling_lin, ceiling_lin)

        tp_before = self._true_peak_dbfs(audio, sample_rate)
        tp_after = self._true_peak_dbfs(processed, sample_rate)
        gain_reduction_db = tp_before - tp_after

        logger.info(
            "Phase 47 TruePeak: ceiling=%.1f dBFS, TP %+.2f → %+.2f dBFS, GR=%.2f dB, t=%.3fs",
            ceiling_dbfs,
            tp_before,
            tp_after,
            gain_reduction_db,
            time.time() - t0,
        )

        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        processed = np.clip(processed, -1.0, 1.0)
        return PhaseResult(
            success=True,
            audio=processed.astype(audio.dtype),
            execution_time_seconds=time.time() - t0,
            metadata={
                "ceiling_dbfs": ceiling_dbfs,
                "true_peak_before_dbfs": tp_before,
                "true_peak_after_dbfs": tp_after,
                "gain_reduction_db": gain_reduction_db,
                "oversampling": _OVERSAMPLING,
                "phase_locality_factor": phase_locality_factor,
                "effective_strength": effective_strength,
                "rms_drop_db": 0.0,
                "loudness_makeup_db": 0.0,
            },
            metrics={
                "true_peak_before_dbfs": tp_before,
                "true_peak_after_dbfs": tp_after,
                "gain_reduction_db": gain_reduction_db,
                "effective_strength": effective_strength,
            },
        )

    # ------------------------------------------------------------------
    # Kern-Algorithmus
    # ------------------------------------------------------------------

    def _compute_gain_curve(self, audio: np.ndarray, sample_rate: int, ceiling: float) -> np.ndarray:
        """Berechnet per-sample gain reduction curve for True-Peak limiting.

        Separated from _limit_channel so that stereo processing can derive
        a single linked gain curve from the combined peak of both channels
        (§2.51 Linked Stereo — avoids independent L/R gain divergence).

        Returns: gain_full array, shape (len(audio),), values in [0, 1].
        """
        n = len(audio)
        if n < 16:
            # Below ceiling → unity gain; above → hard clip handled by caller
            return np.ones(n)  # type: ignore[no-any-return]

        # 1. 4× Oversampling
        os_audio = sig.resample_poly(audio, _OVERSAMPLING, 1)

        # 2. True-Peak-Detection auf Oversampled Signal
        tp_env = np.abs(os_audio)

        # 3. Erforderliche Gain Reduction (frame-weise, 0.1ms Frames)
        frame_s = max(1, int(sample_rate * 0.001))  # 1ms Frames
        os_frame_s = frame_s * _OVERSAMPLING
        n_frames = len(os_audio) // os_frame_s + 1

        # True-Peak per Frame (Max im Oversampled)
        tp_per_frame = np.zeros(n_frames)
        for i in range(n_frames):
            s = i * os_frame_s
            e = min(s + os_frame_s, len(os_audio))
            tp_per_frame[i] = np.max(tp_env[s:e]) if e > s else 0.0

        # 4. Lookahead: für jeden Frame, schaue L ms voraus
        lookahead_f = max(1, int(_LOOKAHEAD_MS))
        peak_ahead = np.zeros(n_frames)
        for i in range(n_frames):
            lo = i
            hi = min(i + lookahead_f, n_frames)
            peak_ahead[i] = np.max(tp_per_frame[lo:hi])

        # 5. Gain Reduction: wie viel müssen wir runterregeln?
        gain_frames = np.minimum(1.0, ceiling / (peak_ahead + 1e-10))

        # 6. Smooth Gain: Angriff & Release
        sr_f = sample_rate / frame_s  # Frames pro Sekunde
        att_coeff = 1.0 - np.exp(-1.0 / (sr_f * _ATTACK_MS * 0.001 + 1e-10))
        rel_coeff = 1.0 - np.exp(-1.0 / (sr_f * _RELEASE_MS * 0.001 + 1e-10))

        gain_smooth = np.ones(n_frames)
        for i in range(1, n_frames):
            if gain_frames[i] < gain_smooth[i - 1]:
                gain_smooth[i] = gain_smooth[i - 1] + att_coeff * (gain_frames[i] - gain_smooth[i - 1])
            else:
                gain_smooth[i] = gain_smooth[i - 1] + rel_coeff * (gain_frames[i] - gain_smooth[i - 1])
        gain_smooth = np.clip(gain_smooth, 0.0, 1.0)

        # 7. Interpolate frame gains back to sample resolution
        gain_full = np.interp(np.arange(n), np.arange(n_frames) * frame_s, gain_smooth)
        return gain_full[:n]  # type: ignore[no-any-return]

    def _limit_channel(self, audio: np.ndarray, sample_rate: int, ceiling: float) -> np.ndarray:
        """Lookahead-Limiter für einen einzelnen Kanal."""
        n = len(audio)
        if n < 16:
            return np.clip(audio, -ceiling, ceiling)  # type: ignore[no-any-return]
        gain = self._compute_gain_curve(audio, sample_rate, ceiling)
        return audio * gain  # type: ignore[no-any-return]

    def _true_peak_dbfs(self, audio: np.ndarray, _sample_rate: int) -> float:
        """Misst True-Peak in dBFS durch 4× Oversampling."""
        mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
        if len(mono) < 4:
            return 0.0
        os_audio = sig.resample_poly(mono, _OVERSAMPLING, 1)
        peak = float(np.max(np.abs(os_audio)))
        return 20.0 * np.log10(peak + 1e-10)  # type: ignore[no-any-return]
