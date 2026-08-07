"""
sota_denoiser.py - SOTA-konformer Denoiser für Aurik 6.0
Dieses Modul implementiert SOTA-orientiertes adaptives Denoising (DeepFilterNet2, DCCRN-ONNX, spektrale Maskierung als Fallback).
Es ist mit DSPContract, Auditierbarkeit und Rollback-Fähigkeit gemäß Dokumentation ausgestattet.
"""

import importlib
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import requests
import scipy.signal
import soundfile as sf

_logger = logging.getLogger(__name__)

df3: Any | None = None
try:
    df3 = importlib.import_module("deepfilter3ii")

    DEEPFILTER3II_AVAILABLE = True
except ImportError:
    DEEPFILTER3II_AVAILABLE = False

dfn: Any | None = None
try:
    dfn = importlib.import_module("deepfilternet2")

    DEEPFILTERNET_AVAILABLE = True
except ImportError:
    DEEPFILTERNET_AVAILABLE = False
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from dsp._memory_budget_guard import check_budget


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "sota_denoiser"
    category: str = "denoiser"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
sota_denoiser_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"strength": 0.8, "use_dccrn": True},
        "safe_ranges": {"strength": [0.0, 1.0], "use_dccrn": [True, False]},
    },
    budgets={"compute_cost": 0.05},
    side_effects=[{"risk": "Modellabhängigkeit", "expected_when": "Modell fehlt", "severity": 0.5}],
    reports={"self_metrics": ["denoising_quality"], "confidence": 1.0},
    rollback={"strategy": "bypass|spectral_masking", "supports_partial": True},
)

MODEL_PATH = "../../models/dccrn/dccrn.onnx"


class SotaDenoiser:
    """
    SOTA-orientierter Denoiser:
    - Deep-Learning (DeepFilterNet2, DCCRN-ONNX) oder spektrale Maskierung als Fallback
    - Auditierbar, rollback-fähig, SOTA-konform
    """

    contract: DSPContract = sota_denoiser_contract

    def __init__(self, strength: float = 0.8, use_dccrn: bool = True):
        self.strength = strength
        self.use_dccrn = use_dccrn
        self.dccrn_session = None
        # Versuche DCCRN-ONNX zu laden, falls gewünscht und verfügbar
        dccrn_path = os.path.join(os.path.dirname(__file__), "../models/dccrn/dccrn.onnx")
        dccrn_path = os.path.abspath(dccrn_path)
        if use_dccrn and ONNX_AVAILABLE and os.path.exists(dccrn_path):
            if not check_budget("sota_denoiser_dccrn", 0.15):
                _logger.warning("Memory budget exceeded for sota_denoiser DCCRN — using DSP fallback")
            else:
                self.dccrn_session = ort.InferenceSession(dccrn_path, providers=["CPUExecutionProvider"])

    def log_contract(self) -> None:
        """
        Gibt den DSPContract für Auditierbarkeit aus.
        """
        _logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        """
        Führt SOTA-Denoising durch (DeepFilterNet2, DCCRN-ONNX, spektrale Maskierung). Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Eingabe-Audiosignal (np.ndarray)
        :param sr: Sample-Rate
        :return: Denoised Signal (np.ndarray)
        """
        self.log_contract()
        # Quality-Gate: Input-Check
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            return np.zeros(0, dtype=np.float32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.ndim != 1:
            self._audit_log("error", "Input must be 1D array")
            return audio.astype(audio.dtype)
        if sr is None:
            self._audit_log("warn", "Sample-Rate ist None, Fallback Identität")
            return audio.copy().astype(audio.dtype)
        try:
            # DeepFilter3II
            if DEEPFILTER3II_AVAILABLE and df3 is not None:
                model = df3.DeepFilter3II()
                self._audit_log("success", "DeepFilter3II-Inferenz erfolgreich")
                return np.asarray(model.denoise(audio, sr))
            # REST/CLI-Fallback für DeepFilterNet3II
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                    sf.write(f_in.name, audio, sr)
                    input_path = f_in.name
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                    output_path = f_out.name
                # REST-API-Aufruf
                try:
                    with open(input_path, "rb") as f:
                        files = {"audio": f}
                        r = requests.post("http://localhost:8502/infer/", files=files, timeout=60)
                    if r.status_code == 200 and "result_wav" in r.json():
                        import binascii

                        wav_bytes = binascii.unhexlify(r.json()["result_wav"])
                        with open(output_path, "wb") as f:  # type: ignore[assignment]
                            f.write(wav_bytes)
                        from backend.file_import import load_audio_file

                        _res = load_audio_file(output_path, do_carrier_analysis=False)
                        result = np.asarray(_res["audio"], dtype=np.float32)  # type: ignore[index]
                        self._audit_log("success", "DeepFilterNet3II REST-API-Inferenz erfolgreich")
                        return np.asarray(result.astype(audio.dtype))
                except Exception as e:
                    self._audit_log("warn", f"DeepFilterNet3II REST-API nicht erreichbar: {e}")
                # CLI-Fallback
                try:
                    import subprocess

                    subprocess.run(
                        [
                            "docker",
                            "run",
                            "--rm",
                            "-v",
                            f"{input_path}:/workspace/input.wav",
                            "-v",
                            f"{output_path}:/workspace/output.wav",
                            "deepfilternet3ii-rest",
                            "python",
                            "/workspace/deepfilternet_v3_ii_infer.py",
                            "/workspace/input.wav",
                            "/workspace/output.wav",
                        ],
                        check=True,
                    )
                    from backend.file_import import load_audio_file

                    _res = load_audio_file(output_path, do_carrier_analysis=False)
                    result = np.asarray(_res["audio"], dtype=np.float32)  # type: ignore[index]
                    self._audit_log("success", "DeepFilterNet3II CLI-Inferenz erfolgreich")
                    return np.asarray(result.astype(audio.dtype))
                except Exception as e:
                    self._audit_log("warn", f"DeepFilterNet3II CLI nicht verfügbar: {e}")
            except Exception as e:
                self._audit_log("warn", f"DeepFilterNet3II Container-Integration fehlgeschlagen: {e}")
            # DCCRN-ONNX
            if self.dccrn_session is not None:
                x = audio.astype(np.float32)
                if x.ndim == 1:
                    x = x[None, None, :]
                elif x.ndim == 2:
                    x = x[None, :, :]
                elif x.ndim == 3:
                    x = x.squeeze(0)
                else:
                    self._audit_log("error", "Audio-Input hat ungültige Shape für DCCRN-ONNX")
                    raise ValueError("Audio-Input hat ungültige Shape für DCCRN-ONNX")
                ort_inputs = {self.dccrn_session.get_inputs()[0].name: x}
                out = self.dccrn_session.run(None, ort_inputs)[0]
                self._audit_log("success", "DCCRN-ONNX-Inferenz erfolgreich")
                return np.asarray(out.squeeze().astype(audio.dtype))
            # REST/CLI-Fallback für DCCRN
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
                    sf.write(f_in.name, audio, sr)
                    input_path = f_in.name
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                    output_path = f_out.name
                # REST-API-Aufruf
                try:
                    with open(input_path, "rb") as f:
                        files = {"audio": f}
                        r = requests.post("http://localhost:8501/infer/", files=files, timeout=60)
                    if r.status_code == 200 and "result_wav" in r.json():
                        import binascii

                        wav_bytes = binascii.unhexlify(r.json()["result_wav"])
                        with open(output_path, "wb") as f:  # type: ignore[assignment]
                            f.write(wav_bytes)
                        from backend.file_import import load_audio_file

                        _res = load_audio_file(output_path, do_carrier_analysis=False)
                        result = np.asarray(_res["audio"], dtype=np.float32)  # type: ignore[index]
                        self._audit_log("success", "DCCRN REST-API-Inferenz erfolgreich")
                        return np.asarray(result.astype(audio.dtype))
                except Exception as e:
                    self._audit_log("warn", f"DCCRN REST-API nicht erreichbar: {e}")
                # CLI-Fallback
                try:
                    import subprocess

                    subprocess.run(
                        [
                            "docker",
                            "run",
                            "--rm",
                            "-v",
                            f"{input_path}:/workspace/input.wav",
                            "-v",
                            f"{output_path}:/workspace/output.wav",
                            "dccrn-rest",
                            "python",
                            "/workspace/dccrn_infer.py",
                            "/workspace/input.wav",
                            "/workspace/output.wav",
                        ],
                        check=True,
                    )
                    from backend.file_import import load_audio_file

                    _res = load_audio_file(output_path, do_carrier_analysis=False)
                    result = np.asarray(_res["audio"], dtype=np.float32)  # type: ignore[index]
                    self._audit_log("success", "DCCRN CLI-Inferenz erfolgreich")
                    return np.asarray(result.astype(audio.dtype))
                except Exception as e:
                    self._audit_log("warn", f"DCCRN CLI nicht verfügbar: {e}")
            except Exception as e:
                self._audit_log("warn", f"DCCRN Container-Integration fehlgeschlagen: {e}")
            # DeepFilterNet2
            if DEEPFILTERNET_AVAILABLE and dfn is not None:
                model = dfn.DeepFilterNet2()
                self._audit_log("success", "DeepFilterNet2-Inferenz erfolgreich")
                return np.asarray(model.denoise(audio, sr))
            # Fallback: Spektrale Maskierung
            f, _t, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=1024, noverlap=512, boundary="even")
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            noise_mag = np.minimum.accumulate(mag, axis=1)
            mask = 1 - self.strength * (noise_mag / (mag + 1e-8))
            mask = np.clip(mask, 0, 1)
            mag_denoised = mag * mask
            Zxx_denoised = mag_denoised * np.exp(1j * phase)
            _, out = scipy.signal.istft(Zxx_denoised, fs=sr, nperseg=1024, noverlap=512, input_onesided=True)
            out = out[: len(audio)]
            if len(out) < len(audio):
                out = np.pad(out, (0, len(audio) - len(out)))
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            self._audit_log("success", "Spektrale Maskierung erfolgreich")
            return np.clip(out, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]
        except Exception as e:
            self._audit_log("error", str(e))
            return audio.copy().astype(audio.dtype)

    def _audit_log(self, level: str, message: str) -> None:
        _logger.debug("[AUR-AUDIT][%s][sota_denoiser] %s", level.upper(), message)
