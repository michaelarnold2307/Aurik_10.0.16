# §v10.19 DEPRECATED: ersetzt durch DFN+SGMSE++MP-SENet Musik-Kaskade
"""ResembleEnhancePlugin — Audio-Enhancement via lokales ONNX (kein Docker/HF).

Modell : models/resemble_enhance/model.onnx
ONNX   : mag+cos+sin[batch,841,T] → out_mag+out_cos+out_sin
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from backend.core.audio_utils import safe_to_mono

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst_holder: list[ResembleEnhancePlugin | None] = [None]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL = os.path.join(_ROOT, "models", "resemble_enhance", "model.onnx")
_SR = 44_100
_N = 1680
_HOP = 420
_BINS = 841


class ResembleEnhancePlugin:
    def __init__(self, model_path: str | None = None) -> None:
        self._session: Any = None
        self._try_load(model_path or _MODEL)

    def _try_load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning("Resemble-verbessern Modell fehlt: %s — DSP-Ersatzpfad.", path)
            return
        # ML-Budget-Guard: Resemble-Enhance model.onnx ~722 MB
        try:
            from backend.core.ml_memory_budget import try_allocate as _try_alloc

            if not _try_alloc("ResembleEnhance", size_gb=0.72):
                logger.warning("Resemble-verbessern: ML-Grenze erschöpft — DSP-Ersatzpfad.")
                return
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                _providers = _get_prov("ResembleEnhance")
            except Exception:
                _providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(path, sess_options=opts, providers=_providers)
            logger.info("Resemble-verbessern ONNX geladen: %s", path)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                def _unload_session() -> None:
                    self._session = None

                _reg_plm("ResembleEnhance", size_gb=0.72, unload_fn=_unload_session)
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        except Exception as exc:
            logger.warning("Resemble-verbessern Ladefehler: %s — DSP-Ersatzpfad.", exc)
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("ResembleEnhance")
            except Exception as _exc:
                logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)

    def enhance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        _was_channels_first = audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2
        mono = safe_to_mono(audio)
        n = len(mono)
        # RAM-Check: Resemble braucht ~130 MB pro 30s-Chunk + ONNX-Overhead.
        # Bei < 4 GB verfügbar → DSP-Fallback statt OOM-Risiko.
        try:
            import psutil

            _avail_gb = psutil.virtual_memory().available / (1024**3)
            if _avail_gb < 4.0:
                logger.warning(
                    "Resemble-verbessern: Nur %.1f GB RAM verfügbar (< 4.0 GB) — Wiener-DSP-Ersatzpfad.",
                    _avail_gb,
                )
                m44 = _resamp(mono, sr, _SR)
                out = _wiener(m44, _SR)
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                result = _resamp(out, _SR, sr)[:n]
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                if audio.ndim == 2:
                    result = np.stack([result, result], axis=0 if _was_channels_first else 1)
                return np.asarray(np.clip(result, -1.0, 1.0).astype(np.float32))  # type: ignore[no-any-return]
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        m44 = _resamp(mono, sr, _SR)
        out = self._onnx(m44) if self._session else _wiener(m44, _SR)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        result = _resamp(out, _SR, sr)[:n]
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim == 2:
            result = np.stack([result, result], axis=0 if _was_channels_first else 1)
        return np.asarray(np.clip(result, -1.0, 1.0).astype(np.float32))  # type: ignore[no-any-return]

    def process(self, input_path: str, output_path: str, denoise_level: float = 0.5, enhance_level: float = 1.0):
        """Datei-basierte Verarbeitung für hybrid_ml_denoiser Kompatibilität.

        Returns:
            (returncode, stdout, stderr) Tuple — 0 bei Erfolg
        """
        import soundfile as sf

        from backend.file_import import load_audio_file

        try:
            del denoise_level, enhance_level
            # Interner Plugin-Read: keine Carrier-Analyse triggern (vermeidet teure
            # CLAP/Medium-Classifier-Pfade und hält Hybrid-Denoise deterministisch).
            _res = load_audio_file(input_path, do_carrier_analysis=False)
            if not isinstance(_res, dict) or _res.get("audio") is None or _res.get("sr") is None:
                raise RuntimeError("load_audio_file lieferte kein valides Ergebnis")
            audio = np.asarray(_res["audio"], dtype=np.float32)
            sr = int(_res["sr"])
            audio_mono = audio.mean(axis=1) if audio.ndim == 2 else audio
            result = self.enhance(audio_mono.astype(np.float32), sr)
            # Ergebnis zurück in Stereo konvertieren wenn nötig
            result_out = np.stack([result, result], axis=1) if audio.ndim == 2 else result
            sf.write(output_path, result_out, sr)
            return (0, "OK", "")
        except Exception as e:
            return (1, "", str(e))

    def _onnx(self, mono: np.ndarray) -> np.ndarray:
        """STFT → ONNX → OLA resynthesis.  Chunked to cap peak RAM.

        Processes in ~30-second chunks (MAX_CHUNK_FRAMES) to prevent OOM on
        long files.  Each chunk is independent; overlap-add at chunk boundaries
        uses a Hanning crossfade of _N samples to avoid clicks.

        §2.38a ML-Headroom-Guard: Falls ONNX-Inferenz fehlschlägt, DSP-Fallback statt Exception.
        """
        try:
            win = np.hanning(_N).astype(np.float32)
            total_nf = max(1, (len(mono) + _HOP - 1) // _HOP)

            # ~30 s at 44.1 kHz  →  ~3150 STFT frames  →  ~130 MB peak per chunk
            _MAX_CHUNK_FRAMES = 3150
            if total_nf <= _MAX_CHUNK_FRAMES:
                return self._onnx_single(mono, win, total_nf)

            # ── Chunked processing ──────────────────────────────────────────
            overlap_samples = _N  # overlap between chunks for crossfade
            chunk_samples = _MAX_CHUNK_FRAMES * _HOP + _N
            step_samples = chunk_samples - overlap_samples
            out_full = np.zeros(len(mono), np.float32)
            pos = 0
            chunk_idx = 0
            while pos < len(mono):
                end = min(pos + chunk_samples, len(mono))
                chunk = mono[pos:end]
                nf_chunk = max(1, (len(chunk) + _HOP - 1) // _HOP)
                processed = self._onnx_single(chunk, win, nf_chunk)
                processed = processed[: len(chunk)]

                if chunk_idx == 0:
                    # First chunk: copy directly
                    out_full[pos : pos + len(processed)] = processed
                else:
                    # Crossfade in overlap region
                    ol = min(overlap_samples, len(processed), len(out_full) - pos)
                    if ol > 0:
                        fade_in = np.linspace(0.0, 1.0, ol, dtype=np.float32)
                        fade_out = 1.0 - fade_in
                        out_full[pos : pos + ol] = out_full[pos : pos + ol] * fade_out + processed[:ol] * fade_in
                    if ol < len(processed):
                        out_full[pos + ol : pos + len(processed)] = processed[ol:]

                pos += step_samples
                chunk_idx += 1
                # Explicit cleanup between chunks to prevent RAM drift
                del chunk, processed
            return out_full  # type: ignore[no-any-return]
        except Exception as _onnx_exc:
            logger.error("Resemble-verbessern ONNX-Pipeline fehlgeschlagen: %s — DSP-Wiener-Ersatzpfad", _onnx_exc)
            # Fallback: Wiener-Filterung statt ONNX
            return _wiener(mono, _SR)

    def _onnx_single(self, mono: np.ndarray, win: np.ndarray, nf: int) -> np.ndarray:
        """Verarbeitet a single audio chunk through ONNX (original algorithm)."""
        # Capture session reference locally to avoid race condition with PLM eviction
        session = self._session
        if session is None:
            return _wiener(mono, _SR)
        buf = np.zeros(nf * _HOP + _N, np.float32)
        buf[: len(mono)] = mono
        frames = [np.fft.rfft(buf[i * _HOP : i * _HOP + _N] * win)[:_BINS] for i in range(nf)]
        spec = np.array(frames, np.complex64).T  # [841,F]
        mag = np.abs(spec)
        cos = spec.real / (mag + 1e-8)
        sin_v = spec.imag / (mag + 1e-8)
        inp = lambda a: a[None].astype(np.float32)  # [1,841,F]
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("ResembleEnhance", True)
        except Exception:
            logger.warning("resemble_verbessern_plugin.py::_onnx_single Ersatzpfad", exc_info=True)
        try:
            outs = session.run(None, {"mag": inp(mag), "cos": inp(cos), "sin": inp(sin_v)})
            om, oc, os_ = outs[0][0], outs[1][0], outs[2][0]
            om = np.nan_to_num(om, nan=0.0, posinf=0.0, neginf=0.0)
            oc = np.nan_to_num(oc, nan=0.0, posinf=0.0, neginf=0.0)
            os_ = np.nan_to_num(os_, nan=0.0, posinf=0.0, neginf=0.0)
            out_spec = om * (oc + 1j * os_)
        except Exception as exc:
            logger.debug("Resemble-verbessern Ausfuehrung Fehler: %s", exc)
            out_spec = spec
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("ResembleEnhance", False)
                except Exception:
                    logger.warning("resemble_verbessern_plugin.py::_onnx_single Ersatzpfad", exc_info=True)
        n_out = nf * _HOP + _N
        res = np.zeros(n_out, np.float32)
        ws = np.zeros(n_out, np.float32)
        full = np.zeros((_N // 2 + 1, nf), np.complex64)
        full[:_BINS] = out_spec
        for i in range(nf):
            frame = np.fft.irfft(full[:, i], n=_N).real.astype(np.float32)
            res[i * _HOP : i * _HOP + _N] += frame * win
            ws[i * _HOP : i * _HOP + _N] += win**2
        return np.asarray((res / np.where(ws < 1e-8, 1.0, ws))[: len(mono)].astype(np.float32))  # type: ignore[no-any-return]


def _resamp(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return np.asarray(x)  # type: ignore[no-any-return]
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src, dst)
    return np.asarray(resample_poly(x, dst // g, src // g).astype(np.float32))  # type: ignore[no-any-return]


def _wiener(mono: np.ndarray, sr: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    # Guard short segments: scipy reduces effective nperseg to signal length,
    # so a fixed noverlap can become >= nperseg and raise ValueError.
    _sig_len = len(mono)
    _nperseg = int(min(_N, max(1, _sig_len)))
    _noverlap = int(min(_N - _HOP, max(0, _nperseg - 1)))

    _, _, Z = stft(mono, fs=sr, nperseg=_nperseg, noverlap=_noverlap, window="hann")
    mag = np.abs(Z)
    ne = np.maximum(uniform_filter(mag, (1, 9)), 1e-8)
    gain = np.maximum(mag**2 / (mag**2 + ne**2 + 1e-10), 0.15)
    _, o = istft(
        gain * mag * np.exp(1j * np.angle(Z)),
        fs=sr,
        nperseg=_nperseg,
        noverlap=_noverlap,
        window="hann",
    )
    return np.asarray(o[: len(mono)].astype(np.float32))  # type: ignore[no-any-return]


def get_resemble_enhance_plugin() -> ResembleEnhancePlugin:
    if _inst_holder[0] is None:
        with _lock:
            if _inst_holder[0] is None:
                _inst_holder[0] = ResembleEnhancePlugin()
    return _inst_holder[0]  # type: ignore[return-value]


def get_loaded_resemble_enhance_plugin() -> ResembleEnhancePlugin | None:
    """Gibt nur eine bereits geladene Resemble-Enhance-Instanz zurück, ohne Lazy-Load."""
    return _inst_holder[0]


def enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    return np.asarray(get_resemble_enhance_plugin().enhance(audio, sr))  # type: ignore[no-any-return]
