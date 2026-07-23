"""dac_plugin — Descript Audio Codec (DAC) 2023 ONNX, 44.1 kHz.

Provides neural discrete audio representation for inpainting conditioning.
Used by cqtdiff_plus_plugin and flow_matching_plugin for Dropout ≥ 50 ms.

Models (onnx-community/dac_44khz-ONNX):
    models/dac/encoder_model.onnx  (~87 MB)
        Input:  input_values  [B, C, T]       float32 (44.1 kHz PCM)
        Output: audio_codes   [B, 9, T//512]  int64   (9 codebooks, stride=512)

    models/dac/decoder_model.onnx  (~208 MB)
        Input:  audio_codes   [B, 9, T_codes] int64
        Output: audio_values  [B, 1, T_codes*512] float32 (44.1 kHz PCM)

Stride: 512 samples @ 44.1 kHz ≈ 11.6 ms per frame.
Codebook: 9 RVQ levels, vocab size 1024.

Resampling: Aurik pipeline uses 48 000 Hz; this plugin resamples
48 000 Hz → 44 100 Hz before encoding and 44 100 Hz → 48 000 Hz after
decoding (scipy.signal.resample_poly, Lanczos-equivalent).

Reference:
    Kumar et al. "High-Fidelity Audio Compression with Improved RVQGAN"
    (NeurIPS 2023). https://github.com/descriptinc/descript-audio-codec

Singleton pattern: use get_dac_plugin().
CPU-only: CPUExecutionProvider.
"""

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_ENCODER_PATH = _ROOT / "models" / "dac" / "encoder_model.onnx"
_DECODER_PATH = _ROOT / "models" / "dac" / "decoder_model.onnx"

# Native model sample rate
_MODEL_SR: int = 44_100
# Aurik pipeline sample rate
_AURIK_SR: int = 48_000
# RVQ codebook levels
_N_CODEBOOKS: int = 9
# Audio stride in model samples (2 * 4 * 8 * 8 = 512)
_STRIDE: int = 512
# Memory budget
_ENCODER_GB: float = 0.10
_DECODER_GB: float = 0.22

_lock = threading.Lock()
_INSTANCE_HOLDER: list[DacPlugin | None] = [None]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DacEncodeResult:
    """Result of DAC encoding: audio → discrete token codes.

    Attributes:
        codes:      Integer codebook indices, shape [B, 9, T_frames], int64.
        n_frames:   Number of code frames produced.
        model_used: "dac_onnx" | "unavailable"
    """

    codes: np.ndarray  # [B, 9, T_frames] int64
    n_frames: int
    model_used: str

    def __post_init__(self) -> None:
        assert self.codes.dtype == np.int64, "codes must be int64"


@dataclass
class DacDecodeResult:
    """Result of DAC decoding: discrete codes → audio.

    Attributes:
        audio:      Reconstructed audio float32 ∈ [-1, 1], shape [B, 1, T].
        sr:         48 000 (Aurik pipeline SR, after resampling).
        model_used: "dac_onnx" | "unavailable"
    """

    audio: np.ndarray  # [B, 1, T] float32
    sr: int
    model_used: str

    def __post_init__(self) -> None:
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = np.clip(self.audio, -1.0, 1.0)


@dataclass
class DacRoundTripResult:
    """Result of encode→decode round-trip (used for quality assessment).

    Attributes:
        audio_out:    Reconstructed audio float32 ∈ [-1, 1], shape [samples] or [2, samples].
        codes:        Intermediate code representation.
        sr:           48 000.
        model_used:   "dac_onnx" | "unavailable"
        snr_db:       Estimated signal-to-noise ratio of reconstruction vs. input.
    """

    audio_out: np.ndarray
    codes: np.ndarray
    sr: int
    model_used: str
    snr_db: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio using scipy.signal.resample_poly (polyphase, near-Lanczos quality)."""
    if from_sr == to_sr:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(from_sr, to_sr)
    up, down = to_sr // g, from_sr // g
    if audio.ndim == 1:
        return np.asarray(resample_poly(audio, up, down), dtype=np.float32)  # type: ignore[no-any-return]
    # [C, T] or [B, C, T]
    flat = np.reshape(audio, (-1, audio.shape[-1]))
    resampled_flat = [resample_poly(ch, up, down).astype(np.float32) for ch in flat]
    stacked = np.stack(resampled_flat, axis=0)
    new_len = stacked.shape[-1]
    reshaped = np.reshape(stacked, (*audio.shape[:-1], new_len))
    return np.asarray(reshaped, dtype=np.float32)  # type: ignore[no-any-return]


def _pad_to_stride(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Füllt auf: last axis to multiple of _STRIDE. Returns (padded, original_length)."""
    orig_len = audio.shape[-1]
    remainder = orig_len % _STRIDE
    if remainder:
        pad_len = _STRIDE - remainder
        pad_shape = list(audio.shape)
        pad_shape[-1] = pad_len
        audio = np.concatenate([audio, np.zeros(pad_shape, dtype=audio.dtype)], axis=-1)
    return audio, orig_len


def _make_session_options():
    """ONNX session options following Aurik §2.37 CPU-aware scheduling."""
    try:
        import os

        import onnxruntime as ort

        opts = ort.SessionOptions()
        n = os.cpu_count() or 4
        opts.intra_op_num_threads = n
        opts.inter_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts
    except Exception:
        logger.warning("dac_plugin.py::_make_session_options fallback", exc_info=True)
        return None


def _estimate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Schätzt SNR in dB between original and reconstructed signals."""
    # Align lengths
    n = min(original.shape[-1], reconstructed.shape[-1])
    orig = original.flat[:n]
    recon = reconstructed.flat[:n]
    sig_power = float(np.mean(orig**2))
    noise_power = float(np.mean((orig - recon) ** 2))
    if noise_power < 1e-12:
        return 60.0
    if sig_power < 1e-12:
        return 0.0
    return float(10.0 * math.log10(sig_power / noise_power))


# ---------------------------------------------------------------------------
# DacPlugin
# ---------------------------------------------------------------------------


class DacPlugin:
    """DAC 44.1 kHz ONNX encoder/decoder for neural discrete audio representation.

    Provides:
      - encode(audio, sr): audio → discrete codes [B, 9, T_frames]
      - decode(codes):     codes → audio [B, 1, T]  @ 48 kHz
      - round_trip(audio, sr): encode+decode for quality assessment / conditioning

    Inpainting use case:
      Known context windows are encoded into DAC codes, which condition the
      CQTdiff+ / flow_matching inpainting diffusion model at the gap boundaries.
    """

    def __init__(self) -> None:
        self._enc_session: Any = None
        self._dec_session: Any = None
        self._enc_loaded: bool = False
        self._dec_loaded: bool = False
        self._try_load()

    def _try_load(self) -> None:
        """Lädt encoder and decoder ONNX sessions."""
        if not _ENCODER_PATH.exists():
            logger.info(
                "DAC encoder ONNX not found (%s) — plugin unavailable. "
                "Download: https://huggingface.co/onnx-community/dac_44khz-ONNX",
                _ENCODER_PATH,
            )
            return

        try:
            import onnxruntime as ort

            # Memory budget check
            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("DacEncoder", size_gb=_ENCODER_GB):
                    logger.warning("DacPlugin: RAM-Budget erschöpft — Encoder nicht geladen.")
                    return
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

            opts = _make_session_options()
            if opts is not None:
                self._enc_session = ort.InferenceSession(
                    str(_ENCODER_PATH),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            else:
                self._enc_session = ort.InferenceSession(
                    str(_ENCODER_PATH),
                    providers=["CPUExecutionProvider"],
                )
            self._enc_loaded = True
            logger.info("✅ DAC encoder ONNX geladen (%s)", _ENCODER_PATH.name)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm("DacEncoder", size_gb=_ENCODER_GB, unload_fn=self._unload_encoder)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

            # Decoder is optional (larger, only needed for full round-trip)
            if _DECODER_PATH.exists():
                try:
                    from backend.core.ml_memory_budget import try_allocate as _try_alloc2

                    if not _try_alloc2("DacDecoder", size_gb=_DECODER_GB):
                        logger.info("DacPlugin: RAM-Budget erschöpft — Decoder nicht geladen (Encoder aktiv).")
                        return
                except Exception as _exc:
                    logger.debug("Plugin operation failed (non-critical): %s", _exc)

                if opts is not None:
                    self._dec_session = ort.InferenceSession(
                        str(_DECODER_PATH),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    self._dec_session = ort.InferenceSession(
                        str(_DECODER_PATH),
                        providers=["CPUExecutionProvider"],
                    )
                self._dec_loaded = True
                logger.info("✅ DAC decoder ONNX geladen (%s)", _DECODER_PATH.name)
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm("DacDecoder", size_gb=_DECODER_GB, unload_fn=self._unload_decoder)
                except Exception as _exc:
                    logger.debug("Plugin operation failed (non-critical): %s", _exc)

        except Exception as exc:
            logger.warning("DAC ONNX nicht ladbar: %s — Plugin deaktiviert.", exc)
            try:
                from backend.core.ml_memory_budget import release as _release

                if self._enc_loaded:
                    _release("DacEncoder")
                _release("DacDecoder")
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _unload_encoder(self) -> None:
        """Release encoder session (called by PLM eviction)."""
        self._enc_session = None
        self._enc_loaded = False
        logger.info("DAC encoder entladen (PLM-Eviction).")

    def _unload_decoder(self) -> None:
        """Release decoder session (called by PLM eviction)."""
        self._dec_session = None
        self._dec_loaded = False
        logger.info("DAC decoder entladen (PLM-Eviction).")

    @property
    def encoder_available(self) -> bool:
        """True if the encoder ONNX session is loaded and ready."""
        return self._enc_loaded and self._enc_session is not None

    @property
    def decoder_available(self) -> bool:
        """True if the decoder ONNX session is loaded and ready."""
        return self._dec_loaded and self._dec_session is not None

    def encode(self, audio: np.ndarray, sr: int) -> DacEncodeResult:
        """Kodiert audio to discrete DAC codes.

        Args:
            audio: float32 PCM, shape [T], [C, T], or [B, C, T].
                   Values expected ∈ [-1, 1].
            sr:    Sample rate of input (typically 48 000).

        Returns:
            DacEncodeResult with codes [B, 9, T_frames] int64.
            If encoder not available: returns zero-codes with model_used="unavailable".
        """
        assert isinstance(audio, np.ndarray), "audio must be np.ndarray"

        if not self.encoder_available:
            # Return minimal placeholder codes
            frames = max(1, audio.shape[-1] // _STRIDE)
            return DacEncodeResult(
                codes=np.zeros((1, _N_CODEBOOKS, frames), dtype=np.int64),
                n_frames=frames,
                model_used="unavailable",
            )

        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0)
        audio = np.clip(audio, -1.0, 1.0)

        # Normalize shape to [B, C, T]
        if audio.ndim == 1:
            audio = audio[np.newaxis, np.newaxis, :]  # [1, 1, T]
        elif audio.ndim == 2:
            audio = audio[np.newaxis, :]  # [1, C, T]
        # audio is now [B, C, T]

        # Resample to 44.1 kHz if needed
        if sr != _MODEL_SR:
            audio = _resample(audio, sr, _MODEL_SR)

        # Downmix to mono for encoder (model expects [B, 1, T])
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)

        # Pad to stride multiple
        audio, _ = _pad_to_stride(audio)

        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("DacEncoder", True)
        except Exception:
            logger.warning("dac_plugin.py::encode fallback", exc_info=True)
        try:
            outputs = self._enc_session.run(
                ["audio_codes"],
                {"input_values": audio},
            )
            codes = outputs[0]  # [B, 9, T_frames] int64
            return DacEncodeResult(
                codes=codes,
                n_frames=codes.shape[-1],
                model_used="dac_onnx",
            )
        except Exception as exc:
            logger.warning("DAC encode error: %s", exc)
            frames = max(1, audio.shape[-1] // _STRIDE)
            return DacEncodeResult(
                codes=np.zeros((1, _N_CODEBOOKS, frames), dtype=np.int64),
                n_frames=frames,
                model_used="unavailable",
            )
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("DacEncoder", False)
                except Exception:
                    logger.warning("dac_plugin.py::encode fallback", exc_info=True)

    def decode(self, codes: np.ndarray) -> DacDecodeResult:
        """Dekodiert discrete DAC codes back to audio.

        Args:
            codes: int64, shape [B, 9, T_frames] or [9, T_frames].

        Returns:
            DacDecodeResult with audio [B, 1, T] @ 48 000 Hz.
        """
        if not self.decoder_available:
            # Return silence placeholder
            t = codes.shape[-1] * _STRIDE if codes.ndim >= 2 else _STRIDE
            return DacDecodeResult(
                audio=np.zeros((1, 1, t), dtype=np.float32),
                sr=_AURIK_SR,
                model_used="unavailable",
            )

        assert isinstance(codes, np.ndarray)
        codes = codes.astype(np.int64)
        if codes.ndim == 2:
            codes = codes[np.newaxis, :]  # [1, 9, T]

        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("DacDecoder", True)
        except Exception:
            logger.warning("dac_plugin.py::decode fallback", exc_info=True)
        try:
            outputs = self._dec_session.run(
                ["audio_values"],
                {"audio_codes": codes},
            )
            audio_44k = outputs[0].astype(np.float32)  # [B, 1, T_44k]
            audio_44k = np.nan_to_num(audio_44k, nan=0.0, posinf=0.0, neginf=0.0)

            # Resample 44.1 kHz → 48 kHz
            audio_48k = _resample(audio_44k, _MODEL_SR, _AURIK_SR)

            return DacDecodeResult(
                audio=audio_48k,
                sr=_AURIK_SR,
                model_used="dac_onnx",
            )
        except Exception as exc:
            logger.warning("DAC decode error: %s", exc)
            t = codes.shape[-1] * _STRIDE
            return DacDecodeResult(
                audio=np.zeros((1, 1, t), dtype=np.float32),
                sr=_AURIK_SR,
                model_used="unavailable",
            )
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("DacDecoder", False)
                except Exception:
                    logger.warning("dac_plugin.py::decode fallback", exc_info=True)

    def round_trip(self, audio: np.ndarray, sr: int) -> DacRoundTripResult:
        """Kodiert then decode (round-trip). Used for conditioning context and quality checks.

        The reconstructed audio provides a perceptually quantized version of
        the input, useful as a conditioning signal for diffusion-based inpainting.

        Args:
            audio: float32 PCM, any shape with last axis = time.
            sr:    Sample rate of input (48 000 expected).

        Returns:
            DacRoundTripResult with reconstructed audio and intermediate codes.
        """
        enc = self.encode(audio, sr)
        dec = self.decode(enc.codes)

        # Bring output to same shape as input (squeeze batch+channel dims)
        audio_out = dec.audio
        if audio_out.ndim == 3 and audio_out.shape[0] == 1:
            audio_out = audio_out[0]  # [1, T]
        if audio_out.ndim == 2 and audio_out.shape[0] == 1:
            audio_out = audio_out[0]  # [T]

        # Trim/pad to original length
        orig_len = audio.shape[-1] if audio.ndim >= 1 else len(audio)
        if sr != _AURIK_SR:
            # Adjust expected output length for resampled input
            orig_len = int(orig_len * _AURIK_SR / sr)
        if audio_out.shape[-1] > orig_len:
            audio_out = audio_out[..., :orig_len]
        elif audio_out.shape[-1] < orig_len:
            pad = [(0, 0)] * (audio_out.ndim - 1) + [(0, orig_len - audio_out.shape[-1])]
            audio_out = np.pad(audio_out, pad)

        audio_out = np.nan_to_num(audio_out.astype(np.float32), nan=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)

        # Estimate SNR vs. original
        orig_flat = audio.astype(np.float32)
        if sr != _AURIK_SR:
            # Compare in 48 kHz domain
            orig_flat = _resample(orig_flat, sr, _AURIK_SR)
        snr = _estimate_snr(orig_flat, audio_out) if enc.model_used == "dac_onnx" else 0.0

        return DacRoundTripResult(
            audio_out=audio_out,
            codes=enc.codes,
            sr=_AURIK_SR,
            model_used=enc.model_used,
            snr_db=snr,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def get_dac_plugin() -> DacPlugin:
    """Gibt thread-safe singleton DacPlugin instance (double-checked locking) zurück."""
    if _INSTANCE_HOLDER[0] is None:
        with _lock:
            if _INSTANCE_HOLDER[0] is None:
                _INSTANCE_HOLDER[0] = DacPlugin()
    plugin = _INSTANCE_HOLDER[0]
    assert plugin is not None
    return plugin


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def dac_encode(audio: np.ndarray, sr: int = _AURIK_SR) -> DacEncodeResult:
    """Kodiert audio to discrete DAC codes. Convenience wrapper."""
    return get_dac_plugin().encode(audio, sr)


def dac_decode(codes: np.ndarray) -> DacDecodeResult:
    """Dekodiert DAC codes to audio. Convenience wrapper."""
    return get_dac_plugin().decode(codes)


def dac_round_trip(audio: np.ndarray, sr: int = _AURIK_SR) -> DacRoundTripResult:
    """Kodiert + decode round-trip for conditioning/quality check. Convenience wrapper."""
    return get_dac_plugin().round_trip(audio, sr)
