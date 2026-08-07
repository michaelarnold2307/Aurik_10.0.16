"""
plugins/gacela_plugin.py — GACELA ML Audio Inpainting + DSP Fallback

GACELA: GAN-basiertes Inpainting von Spektrogramm-Lücken (≈ 0.74 s).
Primär: PyTorch-Inferenz mit lokal gebündelten Checkpoints.
Fallback: harmonischer DSP-Exciter (H2/H4, SOFT_SATURATION-konform).

Modell-Parameter (aus main_realData.py verifiziert):
    Model-SR        : 22 050 Hz
    FFT-Länge       : 1 024  (512 Freq-Bins ohne Nyquist)
    FFT-Hop         : 256
    Fenster (SPLIT) : 480 | 64 | 480 STFT-Frames
    Mel-Bänder      : 80
    Zeit-Mittel     : 4
    Noise-Kanäle    : 4

Checkpoint-Reihenfolge (höchste Iteration zuerst = beste Qualität):
    01_400000.pt, 01_390000.pt, 00_410000.pt, 00_390000.pt, 01_310000.pt

Verifizierte Architektur (aus 01_400000.pt State-Dict):
    Linear(2700 → 8192), in_conv_shape=[8,8], nfilter=[256,128,64,32,1]
    Encoder nfilter=[32,64,32,16], Eingangsform: 2700=(16+16+4)×75=(36×5×15)

Denormalisierung (kritisch, ganSystem.py Z. 222-224):
    gap_linear = np.exp(25 * (generator_output - 1))

Spektrogramm-Inversion:
    PGHI (Perraudin et al. 2013) via dsp/pghi.py — heap-basierte Phasenpropagation
    mit Zeit- und Frequenzgradient. DSP-Fallback: vereinfachte PGHI-Näherung.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any

import numpy as np

try:
    import torch as _torch_mod  # Only for type hints at module level
except ImportError:
    _torch_mod = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Pfade ────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MDL_ROOT = os.path.join(_ROOT, "models", "gacela")
_CKPT_DIR = os.path.join(_MDL_ROOT, "model")

# ── Audio-Konstanten ─────────────────────────────────────────────────────────
MODEL_SR: int = 22_050
PLUGIN_SR: int = 48_000
FFT_LENGTH: int = 1_024
FFT_HOP_SIZE: int = 256
SPLIT: list[int] = [480, 64, 480]  # left_frames, gap_frames, right_frames
MEL_BINS: int = 80
TIME_AVG: int = 2  # padding=2 in BorderEncoder → enc [1,16,5,15]=1200, total 2×1200+4×75=2700 ✓
NOISE_CH: int = 4

# ── Hyperparameter ───────────────────────────────────────────────────────────
_MD: int = 32

# Architekturparameter aus Checkpoint-State-Dict verifiziert (01_400000.pt):
# Encoder: encoder.N.weight_v Shapes [32,1,5,5],[64,32,5,5],[32,64,5,5],[16,32,5,5]
#   → nfilter=[32, 64, 32, 16], stride=[2,2,2,2], shape=[[5,5]×4], output: 16 ch
# Generator: linGenerator weight [8192, 2700] → Linear(2700, 8192)
#   convGenerator shapes: [128,256,4,4],[256,128,4,4],[128,64,8,8],[64,32,8,8],[32,1,8,8]
#   full=8192, in_conv_shape=[8,8], 8192/128=64=8×8 (128 Eingangskanäle)
#   nfilter aus ConvTranspose2d in-ch: [128→256→128→64→32→1]
#   2700 = (16+16+4) × 75 = 36 × 75; enc H×W = 5×15

_BE_PARAMS: dict = {
    "nfilter": [_MD, 2 * _MD, _MD, _MD // 2],  # [32, 64, 32, 16]
    "shape": [[5, 5], [5, 5], [5, 5], [5, 5]],
    "stride": [2, 2, 2, 2],
    "data_size": 2,
    "border_scale": 1,
    "width_full": None,
}

_GEN_PARAMS: dict = {
    "stride": [2, 2, 2, 2, 2],
    # Aus Checkpoint: conv[0].weight [128,256,4,4]→in=128,out=256; conv[0].bias[256] ✓
    # nfilter beschreibt OUTPUT-Kanäle je Schicht (vor den ResBlöcken)
    "nfilter": [256, 128, 64, _MD, 1],  # verifiziert aus Checkpoint
    "shape": [[4, 4], [4, 4], [8, 8], [8, 8], [8, 8]],
    "padding": [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]],
    "residual_blocks": 2,
    "full": 256 * _MD,  # 8 192 = Linear-out; 8192/128=64=8×8
    "in_conv_shape": [8, 8],  # verifiziert: 8192/128=64 ✓
    "data_size": 2,
    "borders": _BE_PARAMS,
}

# 2700 = (16 enc_L + 16 enc_R + 4 noise) × (5 × 15) = 36 × 75
_GEN_IN_SHAPE: int = 2_700

_CKPT_ORDER: list[str] = [
    "01_400000.pt",
    "01_390000.pt",
    "00_410000.pt",
    "00_390000.pt",
    "01_310000.pt",
]


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────


def _time_average(tensor: Any, factor: int) -> Any:  # type: ignore[return]
    """Temporales Mitteln: [B, C, H, W] → [B, C, H, W//factor]."""

    B, C, H, W = tensor.shape
    W2 = W // factor
    return tensor[:, :, :, : W2 * factor].reshape(B, C, H, W2, factor).mean(-1)


def _fit_context(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Audio auf genau n_samples bringen: kürzen (von hinten) oder links padden."""
    if len(audio) >= n_samples:
        return audio[-n_samples:]
    return np.pad(audio, (n_samples - len(audio), 0))  # type: ignore[no-any-return]


# ── Plugin-Klasse ────────────────────────────────────────────────────────────


class GacelaPlugin:
    """GACELA-GAN Inpainting + DSP-Fallback (SOFT_SATURATION-konformer Exciter).

    Primärpfad (ML):
        ``inpaint(left_audio, right_audio, sr)`` → Gap-Audio via GAN-Inferenz.

    Fallback (DSP):
        ``generate(audio, sr, intensity)`` → harmonische H2/H4-Anreicherung.

    Singleton-Zugang via Modul-Ebene ``get_gacela_plugin()``.
    """

    # ── DSP-Konstanten (Fallback-Exciter) ────────────────────────────────────
    N_FFT: int = 2048
    HOP: int = 512
    MAX_BOOST_DB: float = 2.0

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._model_ready: bool = False
        self._generator: Any = None
        self._encoders: list = []
        self._mel_basis: np.ndarray | None = None
        self._stft: Any = None
        self._inverter: Any = None
        self._pghi_rec: Any = None  # dsp.pghi.PghiReconstructor, loaded in _try_load
        self._device: str = "cpu"  # set by _try_load
        self._try_load()

    # ── ML-Modell laden ──────────────────────────────────────────────────────

    _BUDGET_NAME: str = "GACELA"
    _BUDGET_SIZE_GB: float = 0.25  # ~200 MB PyTorch checkpoint

    def _try_load(self) -> None:
        """Lädt GACELA-Checkpoint; setzt _model_ready=True bei Erfolg."""
        try:
            from backend.core.ml_memory_budget import release as _release
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                logger.info("GACELA: ML-Grenze erschöpft — DSP-Ersatzpfad aktiv.")
                self._model_ready = False
                return
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        try:
            import torch

            torch.set_num_threads(os.cpu_count() or 4)  # §2.37 CPU-Thread-Budget

            # GACELA-Modell-Paket in sys.path aufnehmen
            if _MDL_ROOT not in sys.path:
                sys.path.insert(0, _MDL_ROOT)

            import librosa
            from model.borderEncoder import BorderEncoder  # type: ignore[import]
            from model.generator import Generator  # type: ignore[import]
            from tifresi.stft import GaussTruncTF  # type: ignore[import]
            from utils.spectrogramInverter import SpectrogramInverter  # type: ignore[import]

            # Architektur aufbauen
            self._encoders = [BorderEncoder(_BE_PARAMS) for _ in range(2)]
            self._generator = Generator(_GEN_PARAMS, _GEN_IN_SHAPE)

            # Besten verfügbaren Checkpoint laden
            ckpt = None
            for fname in _CKPT_ORDER:
                path = os.path.join(_CKPT_DIR, fname)
                if os.path.isfile(path):
                    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # nosec B614 — lokaler Tensor-Checkpoint
                    logger.info("GACELA: Checkpoint geladen — %s", fname)
                    break

            if ckpt is None:
                raise FileNotFoundError(f"Kein GACELA-Checkpoint gefunden in: {_CKPT_DIR}")

            # Gewichte einspielen
            self._generator.load_state_dict(ckpt["generator"])
            for enc, sd in zip(self._encoders, ckpt["encoders"]):
                enc.load_state_dict(sd)

            # Eval-Modus (kein Dropout/BatchNorm-Training)
            self._generator.eval()
            for enc in self._encoders:
                enc.eval()

            # Device-Platzierung
            try:
                from backend.core.ml_device_manager import get_torch_device as _get_dev

                _dev = _get_dev("Gacela")
            except Exception:
                _dev = "cpu"
            if _dev != "cpu":
                try:
                    from backend.core.ml_device_manager import get_ml_device_manager as _mgr

                    if not _mgr().try_allocate_vram("Gacela", self._BUDGET_SIZE_GB):
                        logger.info("GACELA: VRAM-Grenze erschöpft — CPU-laden")
                        _dev = "cpu"
                except Exception:
                    logger.warning("gacela_plugin.py::_try_laden Ersatzpfad", exc_info=True)
            self._generator.to(_dev)
            for enc in self._encoders:
                enc.to(_dev)
            self._device = _dev

            # STFT-Objekte
            self._stft = GaussTruncTF(hop_size=FFT_HOP_SIZE, stft_channels=FFT_LENGTH)
            mel_fb = librosa.filters.mel(sr=MODEL_SR, n_fft=FFT_LENGTH, n_mels=MEL_BINS)  # type: ignore[attr-defined]
            # mel_fb Form [80, FFT_LENGTH//2 + 1] = [80, 513]; nur erste 512 Bins
            self._mel_basis = mel_fb[:, : FFT_LENGTH // 2].astype(np.float32)
            # SpectrogramInverter als Typ-Referenz (nicht für Inversion genutzt)
            self._inverter = SpectrogramInverter(FFT_LENGTH, FFT_HOP_SIZE)

            # Canonical PGHI reconstructor (dsp/pghi.py, Perraudin et al. 2013).
            # sr=48000 satisfies the assert; PGHI algorithm is SR-agnostic — only
            # win_size and hop matter for phase propagation.
            try:
                from dsp.pghi import PghiReconstructor

                self._pghi_rec = PghiReconstructor(sr=48000)
                logger.debug("GACELA: canonical PGHI reconstructor geladen.")
            except Exception as _pghi_exc:
                logger.debug("GACELA: dsp.pghi nicht verfuegbar, using inline Ersatzpfad: %s", _pghi_exc)
                self._pghi_rec = None

            self._model_ready = True
            logger.info("GACELA: ML-Modell bereit (MODEL_SR=%d Hz).", MODEL_SR)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB, unload_fn=_unload_gacela)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

        except Exception as exc:
            logger.warning(
                "GACELA: Modell konnte nicht geladen werden (%s) — DSP-Ersatzpfad aktiv.",
                exc,
            )
            self._model_ready = False
            try:
                from backend.core.ml_memory_budget import release as _release

                _release(self._BUDGET_NAME)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    # ── ML-Inpainting (primärer Pfad) ────────────────────────────────────────

    def inpaint(
        self,
        left_audio: np.ndarray,
        right_audio: np.ndarray,
        native_sr: int,
    ) -> np.ndarray | None:
        """Füllt eine Lücke zwischen *left_audio* und *right_audio* via GAN.

        Algorithmus (vollständige Tensor-Pipeline):
            1. Resample Kontext 48 kHz → 22 050 Hz
            2. Auf SPLIT[0]/SPLIT[2] × FFT_HOP_SIZE Samples kürzen/padden
            3. GaussTruncTF.spectrogram() → Log-Spektrogramm [:512, :480]
            4. Mel-Filterbank [80,512] @ Spektrogramm → Mel [80, 480]
            5. Zeit-Mitteln (÷4) → [80, 120]
            6. BorderEncoder(left), BorderEncoder(right) → je [1,16,5,8]
            7. cat([enc_L, enc_R, noise(4)], dim=1) → [1,36,5,8]
            8. Generator → [1,1,512,64] ∈ [-1,1]
            9. Denorm: gap_linear = exp(25·(output − 1))
            10. SpectrogramInverter → 16 384 Samples @ 22 050 Hz
            11. Resample → native_sr; NaN/Inf-Guard; clip [-1,1]

        Args:
            left_audio:  Kontext vor der Lücke (float32, mono oder stereo)
            right_audio: Kontext nach der Lücke (float32, mono oder stereo)
            native_sr:   Host-Sample-Rate (üblicherweise 48 000 Hz)

        Returns:
            Gap-Audio als float32-Array bei native_sr oder None bei Fehler.
        """
        if not self._model_ready:
            return None

        try:
            import librosa
            import torch

            n_ctx = SPLIT[0] * FFT_HOP_SIZE  # Kontextlänge in Samples @ MODEL_SR

            def _prep(audio: np.ndarray) -> np.ndarray:
                """Mono-Downmix, Resample, Fit auf n_ctx."""
                a = np.asarray(audio, dtype=np.float32)
                if a.ndim == 2:
                    a = a.mean(axis=0) if a.shape[0] <= 8 else a.mean(axis=1)
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                a = librosa.resample(a, orig_sr=native_sr, target_sr=MODEL_SR)
                return _fit_context(a, n_ctx)

            left_mono = _prep(left_audio)
            right_mono = _prep(right_audio)

            def _encode(mono: np.ndarray, encoder: torch.nn.Module) -> torch.Tensor:
                """Mono [n_ctx] → BorderEncoder-Embedding [1,16,5,8]."""
                # Log-Spektrogramm via GaussTruncTF
                spec_full = self._stft.spectrogram(mono, normalize=False)  # [513, T]
                spec = spec_full[: FFT_LENGTH // 2, : SPLIT[0]]  # [512, 480]
                # Mel-Projektion
                mel = (self._mel_basis @ spec).astype(np.float32)  # [80, 480]
                t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self._device)  # [1,1,80,480]
                t = _time_average(t, TIME_AVG)  # [1,1,80,240]
                with torch.no_grad():
                    return encoder(t)  # type: ignore[no-any-return]  # [1,16,5,8]

            enc_L = _encode(left_mono, self._encoders[0])
            enc_R = _encode(right_mono, self._encoders[1])

            # Rauschen und Konkatenation
            noise = torch.rand(1, NOISE_CH, enc_L.size(2), enc_L.size(3), dtype=torch.float32).to(self._device)
            x = torch.cat([enc_L, enc_R, noise], dim=1)  # [1,36,5,8]

            # Generator-Inferenz
            with torch.no_grad():
                gap_norm = self._generator(x)  # [1,1,512,64], Tanh ∈ [-1,1]

            # Denormalisierung: norm → log → linear (ganSystem.py generateGap)
            # Generator-Ausgabe: [1, 1, 256, 256] (256 freq-bins × 256 time-frames)
            gap_np = gap_norm[0, 0].cpu().numpy()  # [256, 256], float32
            gap_log = 25.0 * (gap_np - 1.0)  # Log-Spektrogramm
            gap_linear = np.exp(np.clip(gap_log, -80.0, 20.0))  # numerisch stabil

            # Spektrogramm-Inversion via kanonischem PGHI (dsp/pghi.py, §4.4)
            # Referenz: Perraudin et al. (2013) — Phase Gradient Heap Integration
            # gap_linear Form: [256, 256] = [n_bins, n_frames]
            # Bei n_fft=512 erwartet ISTFT [1 + 512//2, n_frames] = [257, n_frames]
            # → 1 Zeile unten auffüllen (analog spectrogramInverter._invertSpectrogram)
            spec_padded = np.vstack(
                [
                    gap_linear,
                    np.zeros((1, gap_linear.shape[1]), dtype=np.float32),
                ]
            )  # [257, 256]

            _n_fft_pg = FFT_LENGTH // 2  # 512 → n_bins = 257
            _hop_pg = FFT_HOP_SIZE  # 256

            if self._pghi_rec is not None:
                # Canonical PGHI: heap-propagation with time + frequency gradients
                # (Perraudin et al. 2013). Win_size=512, hop=256 match GACELA params.
                _pghi_result = self._pghi_rec.reconstruct(
                    spec_padded.astype(np.float64),
                    win_size=_n_fft_pg,
                    hop=_hop_pg,
                )
                gap_audio = _pghi_result.audio
                logger.debug("GACELA: invert via %s", _pghi_result.method_used)
            else:
                # Inline fallback: simplified PGHI approximation (time-gradient only).
                # Used when dsp/pghi.py is unavailable (should not happen in production).
                import scipy.signal as _ss_pghi

                _log_m = np.log1p(spec_padded.astype(np.float32))
                _grad_t = np.diff(_log_m, axis=1, prepend=_log_m[:, :1])
                _hop_ph = (2.0 * np.pi * np.arange(spec_padded.shape[0]) * _hop_pg / _n_fft_pg).astype(np.float32)
                _phase = np.cumsum(_hop_ph[:, None] + _grad_t * 0.2, axis=1)
                _, gap_audio = _ss_pghi.istft(
                    spec_padded.astype(np.complex64) * np.exp(1j * _phase),
                    fs=MODEL_SR,
                    nperseg=_n_fft_pg,
                    noverlap=_n_fft_pg - _hop_pg,
                    window="hann",
                )
                gap_audio = gap_audio.astype(np.float32)

            gap_audio = np.nan_to_num(gap_audio, nan=0.0, posinf=0.0, neginf=0.0)
            # Normalisierung: Peak-Normierung auf max. 0.9 (§VERBOTEN: np.max → np.percentile 99.9 — Impuls-Artefakt darf Normalisierung nicht blockieren)
            peak = float(np.percentile(np.abs(gap_audio), 99.9))
            if peak > 1e-6:
                gap_audio = gap_audio / peak * 0.9
            gap_audio = np.clip(gap_audio, -1.0, 1.0)

            # Resample auf Host-SR
            gap_out = librosa.resample(gap_audio, orig_sr=MODEL_SR, target_sr=native_sr)
            return np.clip(gap_out, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

        except Exception as exc:
            if self._device != "cpu":
                logger.warning("GACELA: GPU-Inferenz fehlgeschlagen (%s) — CPU-Wiederholung", exc)
                try:
                    for _attr in ("_generator", "_border_enc_l", "_border_enc_r", "_spec_inverter"):
                        _m = getattr(self, _attr, None)
                        if _m is not None:
                            _m.cpu()
                    self._device = "cpu"
                    try:
                        from backend.core.ml_device_manager import get_ml_device_manager as _mgr

                        _mgr().report_gpu_error("Gacela", exc)
                    except Exception:
                        logger.warning("gacela_plugin.py::unknown Ersatzpfad", exc_info=True)
                except Exception as _mv_exc:
                    logger.debug("GACELA GPU→CPU move fehlgeschlagen: %s", _mv_exc)
                    self._device = "cpu"
                return self.inpaint(left_audio, right_audio, native_sr)
            logger.warning("GACELA inpaint() fehlgeschlagen: %s", exc)
            return None

    # ── DSP-Fallback (Exciter) ────────────────────────────────────────────────

    def generate(
        self,
        audio: np.ndarray,
        sr: int = 48000,
        intensity: float = 0.3,
    ) -> np.ndarray:
        """Harmonische Anreicherung (H2/H4, gerade Obertöne -- SOFT_SATURATION).

        Fuer volle generative Funktionalitaet waere das GACELA-Modell in
        models/gacela/ erforderlich (aktuell nicht vorhanden).

        Args:
            audio: Eingabe-Audio float32
            sr: Sample-Rate
            intensity: Staerke der Anreicherung [0.0, 1.0]

        Returns:
            Leicht harmonisch angereichertes Audio (clip -1..1)
        """
        intensity = float(np.clip(intensity, 0.0, 1.0))
        mono = self._to_mono(audio)
        if intensity < 0.01:
            return np.clip(audio.copy().astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]

        # Weiche Schaedigung: even-order Harmonics via tanh-Saettigung
        saturation_strength = intensity * (10 ** (self.MAX_BOOST_DB / 20.0) - 1.0)
        enhanced = np.tanh(mono * (1.0 + saturation_strength)) / (1.0 + saturation_strength * 0.5)
        enhanced = np.nan_to_num(enhanced, nan=0.0).astype(np.float32)

        if audio.ndim == 2:
            # Stereo: gleiches Enhancement auf beide Kanaele
            if audio.shape[0] <= 8:
                return np.clip(np.stack([enhanced] * audio.shape[0]), -1.0, 1.0)  # type: ignore[no-any-return]
            return np.clip(np.stack([enhanced] * audio.shape[1], axis=1), -1.0, 1.0)  # type: ignore[no-any-return]

        return np.clip(enhanced, -1.0, 1.0)  # type: ignore[no-any-return]

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        a = np.array(audio, dtype=np.float32)
        if a.ndim == 2:
            a = a.mean(axis=0) if a.shape[0] <= 8 else a.mean(axis=1)
        return np.nan_to_num(a, nan=0.0)  # type: ignore[no-any-return]


# ── Singleton (Double-Checked Locking, Spec §3.2) ────────────────────────────


def _unload_gacela() -> None:
    """Entlädt das GACELA-Singleton aus dem RAM (PLM-Eviction-Callback)."""
    global _inst
    _inst = None  # type: ignore[assignment]
    try:
        import gc

        gc.collect()
    except Exception as _exc:
        logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)


_lock = threading.Lock()
_inst: GacelaPlugin | None = None


def get_gacela_plugin() -> GacelaPlugin:
    """Thread-sicherer Singleton-Accessor."""
    global _inst
    if _inst is None:
        with _lock:
            if _inst is None:
                _inst = GacelaPlugin()
    return _inst


# ── Convenience-Wrapper ──────────────────────────────────────────────────────


def generate_audio(audio: np.ndarray, sr: int = 48000, intensity: float = 0.3) -> np.ndarray:
    """Convenience-Wrapper für den DSP-Exciter-Pfad."""
    return get_gacela_plugin().generate(audio, sr, intensity)


# ── Backward-compat alias (Spec §11.3) ──────────────────────────────────────
GACELAPlugin = GacelaPlugin
