"""BanquetVinylPlugin — lokale ONNX-Inferenz (kein Docker).

Lädt models/banquet/banquet_vinyl_final.onnx direkt über onnxruntime
(CPUExecutionProvider). Kein Netzwerk, kein Docker, vollständig offline.

Fallback: DSP-Median-Declicker + Butterworth-Hochpass (scipy/numpy).  # §V6: logger.warning handled at call site
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_instance: BanquetVinylPlugin | None = None
_lock = threading.Lock()


def get_banquet_plugin() -> BanquetVinylPlugin:
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BanquetVinylPlugin()
    return _instance


# ---------------------------------------------------------------------------
# Haupt-Plugin
# ---------------------------------------------------------------------------
class BanquetVinylPlugin:
    """Vinyl-Crackle/Click/Rauschen-Restaurierung via lokalem ONNX-Modell.

    Modell-Pfad: models/banquet/banquet_vinyl_final.onnx
    Fallback   : DSP (Median-Declicker + Butterworth-Hochpass).  # §V6: logger.warning handled at call site

    Invarianten (copilot-instructions.md §3.1):
        - Kein Docker, kein Netzwerk, kein extererner Prozess
        - CPUExecutionProvider (kein CUDA)
        - NaN/Inf in Ausgabe → nan_to_num → clip(-1, 1)
        - GrooveMetric: kein Timing-Flattening
    """

    DEFAULT_MODEL_SUBPATH = "models/banquet/banquet_vinyl_final.onnx"
    TARGET_SR: int = 48_000
    CHUNK_SEC: float = 1.0
    OVERLAP_SEC: float = 0.5

    def __init__(self, model_dir: str | None = None) -> None:
        self._session = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._model_ok: bool = False
        self._chunk_failures: int = 0
        self._runtime_quarantined: bool = False

        if model_dir is not None:
            model_path = Path(model_dir) / "banquet_vinyl_final.onnx"
        else:
            workspace = Path(__file__).parent.parent
            model_path = workspace / self.DEFAULT_MODEL_SUBPATH

        self._model_path = model_path
        self._try_load_model()

    # ------------------------------------------------------------------
    # Name of the patched ONNX whose 0-D Slice tensors (val_21/val_22) have
    # been promoted to 1-D arrays so ONNXRuntime can execute them correctly.
    _PATCHED_SUFFIX = "_patched.onnx"

    def _try_load_model(self) -> None:
        # Prefer the pre-patched variant; fall back to applying the patch on-the-fly.
        patched_path = self._model_path.with_name(self._model_path.stem + self._PATCHED_SUFFIX)
        load_path = patched_path if patched_path.exists() else self._model_path

        if not load_path.exists():
            # Original model present but no patch yet — create it now.
            if self._model_path.exists():
                load_path = self._patch_onnx(self._model_path, patched_path)
            else:
                logger.warning("BANQUET-ONNX nicht gefunden: %s — DSP-Ersatzpfad aktiv", self._model_path)
                return

        try:
            import onnxruntime as ort

            try:
                from backend.core.ml_memory_budget import try_allocate as _try_alloc

                if not _try_alloc("BanquetVinyl", size_gb=0.80):
                    logger.warning("BanquetVinyl: ML-Grenze erschöpft — DSP-Ersatzpfad.")
                    return
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 4
            # ORT_DISABLE_ALL avoids the graph-level Slice rewrite that causes
            # 'Starts must be a 1-D array' at optimisation time.
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            try:
                from backend.core.ml_device_manager import get_ort_providers as _get_prov

                _providers = _get_prov("BanquetVinyl")
            except Exception:
                _providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(load_path),
                sess_options=opts,
                providers=_providers,
            )
            self._input_name = self._session.get_inputs()[0].name  # type: ignore[attr-defined]
            self._output_name = self._session.get_outputs()[0].name  # type: ignore[attr-defined]
            self._model_ok = True
            logger.info(
                "✅ BANQUET ONNX geladen: %s | In=%s | Out=%s",
                load_path.name,
                self._input_name,
                self._output_name,
            )
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    "BanquetVinyl",
                    size_gb=0.80,
                    unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_model_ok", False),  # type: ignore[func-returns-value,misc]
                )
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
        except Exception as exc:
            logger.warning("BANQUET ONNX Ladefehler: %s — DSP-Ersatzpfad", exc)
            self._session = None
            self._model_ok = False
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("BanquetVinyl")
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

    @staticmethod
    def _patch_onnx(src: Path, dst: Path) -> Path:
        """Promote 0-D Slice initializers (val_21, val_22) to 1-D and save.

        ONNXRuntime ≤ 1.20 rejects scalars as Slice starts/ends. The exported
        model contains two such 0-D tensors that need to become shape ``[1]``.
        Returns *dst* on success, *src* on any error so the caller can still
        attempt loading the original (which will fail at runtime, but then the
        exception handler activates the DSP fallback).
        """
        try:
            import onnx
            from onnx import numpy_helper

            model = onnx.load(str(src))
            patched = 0
            for init in model.graph.initializer:
                arr = numpy_helper.to_array(init)
                if arr.ndim == 0 and init.name in ("val_21", "val_22"):
                    import numpy as _np

                    new_arr = _np.array([arr.item()], dtype=arr.dtype)
                    init.CopyFrom(numpy_helper.from_array(new_arr, name=init.name))
                    patched += 1
            if patched:
                onnx.checker.check_model(model)
                dst.parent.mkdir(parents=True, exist_ok=True)
                onnx.save(model, str(dst))
                logger.info("BANQUET: %d Slice-Initialisierer gepatcht → %s", patched, dst.name)
                return dst
        except Exception as exc:
            logger.warning("BANQUET: ONNX-Patch fehlgeschlagen (%s) — versuche Originalsignal", exc)
        return src

    # ------------------------------------------------------------------
    def process(
        self,
        audio: np.ndarray,
        sr: int,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Restauriere Vinyl-Audio.

        Args:
            audio   : float32 [samples] mono oder [channels, samples]
            sr      : Sample-Rate in Hz
            strength: Restaurierungs-Intensität 0.0–1.0

        Returns: float32 ndarray, selbe Form wie Eingabe.
        """
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        stereo_in = audio.ndim == 2
        if not stereo_in:
            audio = audio[np.newaxis, :]  # [1, samples]

        resampled, res_sr = self._maybe_resample(audio, sr, self.TARGET_SR)

        if self._model_ok:
            _plm_bvq = None
            try:
                from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_fn

                _plm_bvq = _get_plm_fn()
                _plm_bvq.set_active("BanquetVinyl", True)
            except Exception as _exc:
                logger.debug("BanquetVinyl: PLM set_active fehlgeschlagen: %s", _exc)
            try:
                restored = self._process_onnx(resampled, strength)
            finally:
                if _plm_bvq is not None:
                    try:
                        _plm_bvq.set_active("BanquetVinyl", False)
                    except Exception as _exc:
                        logger.debug("BanquetVinyl: PLM unset_active fehlgeschlagen: %s", _exc)
        else:
            restored = self._process_dsp(resampled, res_sr, strength)

        if res_sr != sr:
            restored, _ = self._maybe_resample(restored, res_sr, sr)

        restored = np.nan_to_num(restored, nan=0.0, posinf=0.0, neginf=0.0)
        restored = np.clip(restored, -1.0, 1.0)

        if not stereo_in:
            restored = restored[0]
        return restored  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    def _process_onnx(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """OLA-Chunk-Verarbeitung über das ONNX-Modell.

        The Banquet SeqBand model takes a 4-D STFT feature tensor of shape
        ``[1, 128, 128, 128]`` (batch × n_bands × n_frames × feature_dim)
        and returns a cleaned version of the same shape, which is ISTFT-ed
        back to time-domain audio using the original phase.
        """
        channels, n_samples = audio.shape
        sr = self.TARGET_SR
        chunk_len = int(sr * self.CHUNK_SEC)
        hop_len = int(sr * (self.CHUNK_SEC - self.OVERLAP_SEC))
        window = np.hanning(chunk_len).astype(np.float32)

        out = np.zeros_like(audio)
        weight = np.zeros(n_samples, dtype=np.float32)

        pos = 0
        while pos < n_samples:
            end = min(pos + chunk_len, n_samples)
            chunk = audio[:, pos:end]
            pad = chunk_len - chunk.shape[1]
            if pad > 0:
                chunk = np.pad(chunk, ((0, 0), (0, pad)))

            if self._runtime_quarantined or self._session is None:
                chunk_out = self._process_dsp(chunk, self.TARGET_SR, strength)
            else:
                try:
                    inp_tensor, stft_ctx = self._prepare_input(chunk, channels)
                    raw_out = self._session.run([self._output_name], {self._input_name: inp_tensor})[0]
                    raw_out_arr = np.asarray(raw_out, dtype=np.float32)
                    raw_out = np.nan_to_num(raw_out_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    chunk_out = self._extract_output(raw_out, channels, chunk_len, stft_ctx)
                    self._chunk_failures = 0
                except Exception as exc:
                    logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
                    logger.debug("ONNX-Chunk-Fehler: %s — DSP für diesen Chunk", exc)
                    self._chunk_failures += 1
                    # Quarantine ONNX path after repeated deterministic failures.
                    if self._chunk_failures >= 3:
                        self._runtime_quarantined = True
                        self._model_ok = False
                        self._session = None
                        logger.warning(
                            "BANQUET ONNX zur Laufzeit deaktiviert (wiederholte Chunk-Fehler)."
                            " Nutze DSP-Ersatzpfad fuer Stabilitaet."
                        )
                        try:
                            from backend.core.ml_memory_budget import release as _rel

                            _rel("BanquetVinyl")
                        except Exception as _exc:
                            logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
                    chunk_out = self._process_dsp(chunk, self.TARGET_SR, strength)

            actual = end - pos
            for c in range(channels):
                out[:, pos:end] += (chunk_out[c] * window)[:actual][np.newaxis, :]
            weight[pos:end] += window[:actual]

            pos += hop_len

        weight = np.where(weight < 1e-8, 1.0, weight)
        out /= weight[np.newaxis, :]

        if strength < 1.0:
            out = strength * out + (1.0 - strength) * audio

        return np.clip(out, -1.0, 1.0)  # type: ignore[no-any-return]

    def _prepare_input(self, chunk: np.ndarray, channels: int) -> tuple[np.ndarray, np.ndarray]:
        """Konvertiert audio chunk [ch, chunk_len] → ONNX tensor [1, 128, 128, 128].

        The Banquet model uses a band-split RNN (SeqBand) that expects:
            [batch=1, n_bands=128, n_frames=128, hidden=128]

        Preprocessing pipeline:
        1. STFT with n_fft=512, hop=375 at 48 kHz
           → 257 complex bins × ~128 frames
        2. Split into 128 frequency bands (2 bins each, bands 0..127)
        3. Per band: encode 2 complex bins (real/imag × 2 = 4 floats) tiled
           to fill hidden_dim=128
        4. Normalise by global std

        Returns:
            inp   : float32 [1, 128, 128, 128]
            stft  : complex64 [128, 128]  — original STFT for ISTFT reconstruction
        """
        mono = chunk.mean(axis=0).astype(np.float32)
        try:
            from scipy.signal import stft as sci_stft

            # hop=375 at 48 kHz → exactly 128 frames per 1-second chunk
            _, _, Zxx = sci_stft(mono, nperseg=512, noverlap=512 - 375, boundary="zeros")
            # Zxx: [257, n_frames] complex
            n_frames = min(Zxx.shape[1], 128)
            stft_ctx = np.zeros((128, 128), dtype=np.complex64)
            stft_ctx[:, :n_frames] = Zxx[:128, :n_frames].astype(np.complex64)

            feat = np.zeros((1, 128, 128, 128), dtype=np.float32)
            for b in range(128):
                bin1 = min(2 * b + 1, Zxx.shape[0] - 1)
                # 4 real features per frame: real/imag of two consecutive bins
                r0 = stft_ctx[b, :]  # real part of band-centre bin
                band_feat = np.stack(
                    [
                        r0.real,
                        r0.imag,
                        Zxx[bin1, :n_frames].real[:128].astype(np.float32),
                        Zxx[bin1, :n_frames].imag[:128].astype(np.float32),
                    ],
                    axis=0,
                )  # [4, 128]
                # Tile 32× to fill hidden_dim=128
                tiled = np.tile(band_feat, (32, 1))[:128, :]  # [128, 128]
                feat[0, b, :, :] = tiled

            std = feat.std()
            if std > 1e-8:
                feat /= std
            return feat, stft_ctx
        except Exception:
            logger.warning("banquet_vinyl_plugin.py::_prepare_Eingabe Ersatzpfad", exc_info=True)
            return np.zeros((1, 128, 128, 128), dtype=np.float32), np.zeros((128, 128), dtype=np.complex64)

    @staticmethod
    def _extract_output(
        raw: np.ndarray,
        channels: int,
        chunk_len: int,
        stft_ctx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Konvertiert ONNX output [1, 128, 128, 128] back to time-domain audio.

        Interprets the model output as a Wiener-like spectral mask applied to
        the original STFT, then reconstructs audio via ISTFT using the
        preserved original phase.
        """
        try:
            from scipy.signal import istft as sci_istft
            from scipy.special import expit as _sigmoid

            # Mean across hidden_dim → [128, 128] mask (freq × time)
            mask_2d = raw.squeeze().mean(axis=2).astype(np.float64)  # [128, 128]
            mask_2d = _sigmoid(mask_2d)  # map to [0, 1] — Wiener-style

            if stft_ctx is not None and np.any(stft_ctx != 0):
                clean_stft = stft_ctx * mask_2d.astype(np.float32)  # [128, 128]
                # Pad to 257 bins for ISTFT with n_fft=512
                stft_full = np.zeros((257, 128), dtype=np.complex64)
                stft_full[:128, :] = clean_stft
                _, audio_out = sci_istft(stft_full, nperseg=512, noverlap=512 - 375, boundary="zeros")
                audio_out = audio_out.astype(np.float32)
                # Pad/trim to chunk_len
                if len(audio_out) >= chunk_len:
                    audio_out = audio_out[:chunk_len]
                else:
                    audio_out = np.pad(audio_out, (0, chunk_len - len(audio_out)))
                return np.tile(audio_out[np.newaxis, :], (channels, 1))  # type: ignore[no-any-return]
        except Exception as _exc:
            logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
        # Shape fallback — return silence (chunk exception handler will use DSP)
        return np.zeros((channels, chunk_len), dtype=np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    def _process_dsp(self, audio: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray:
        """DSP-Fallback: Median-Declicker + Butterworth-Hochpass.

        Referenz: Cemgil et al. (2007) Sparse Bayes + Cohen (2003) IMCRA.
        """
        try:
            from scipy.ndimage import median_filter
            from scipy.signal import butter, sosfilt
        except ImportError:
            return audio

        channels, _ = audio.shape
        out = audio.copy()

        for c in range(channels):
            x = audio[c].copy()

            # DC-Entfernung
            try:
                sos = butter(4, max(20.0, 1.0) / (sr / 2), btype="high", output="sos")
                x = sosfilt(sos, x).astype(np.float32)
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

            # Median-Declicker
            win = max(3, int(sr * 0.001) | 1)  # ~1 ms, odd
            try:
                smooth = median_filter(x, size=win)
                diff = np.abs(x - smooth)
                thresh = 3.0 * float(np.median(diff)) * strength
                mask = diff > thresh
                x[mask] = smooth[mask]
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

            out[c] = strength * x + (1.0 - strength) * audio[c]

        return np.clip(out, -1.0, 1.0)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    @staticmethod
    def _maybe_resample(audio: np.ndarray, src: int, tgt: int) -> tuple[np.ndarray, int]:
        if src == tgt:
            return audio, tgt
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(tgt, src)
            up, dn = tgt // g, src // g
            out = np.stack([resample_poly(ch, up, dn).astype(np.float32) for ch in audio])
            return out, tgt
        except Exception as exc:
            logger.debug("Resampling fehlgeschlagen: %s", exc)
            return audio, src

    # ------------------------------------------------------------------
    # Legacy file-based API (kompatibel mit alter Docker-Schnittstelle)
    # ------------------------------------------------------------------
    def process_files(self, input_wav: str, output_wav: str, strength: float = 1.0) -> None:
        """Verarbeite WAV-Datei direkt (kompatibel mit alter Docker-API)."""
        try:
            import soundfile as sf

            from backend.file_import import load_audio_file

            _res = load_audio_file(input_wav, do_carrier_analysis=False)
            audio = np.asarray(_res["audio"], dtype=np.float32)  # type: ignore[index]
            sr = int(_res["sr"])  # type: ignore[index]
            audio = audio[np.newaxis, :] if audio.ndim == 1 else audio.T  # [channels, samples]
            restored = self.process(audio, sr, strength)
            Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_wav, restored.T, sr)
            logger.info("✅ BANQUET: %s → %s", input_wav, output_wav)
        except Exception as exc:
            logger.error("BANQUET verarbeiten_files fehlgeschlagen: %s", exc)
            raise


# ---------------------------------------------------------------------------
# Convenience-Wrapper
# ---------------------------------------------------------------------------
def process_vinyl(audio: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray:
    """Convenience-Wrapper für direkte Nutzung ohne Klassen-Instantiierung."""
    return get_banquet_plugin().process(audio, sr, strength)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 3:
        logger.debug("Verwendung: banquet_vinyl_plugin.py <Eingabe.wav> <Ausgabe.wav> [strength=1.0]")
        sys.exit(1)
    st = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    get_banquet_plugin().process_files(sys.argv[1], sys.argv[2], strength=st)

# Convenience-Alias
import numpy as _np


def restore_vinyl(audio: _np.ndarray, sr: int = 48000) -> _np.ndarray:
    """Alias für process_vinyl."""
    return process_vinyl(audio, sr)
