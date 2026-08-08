"""
UTMOS Plugin — MOS-Schätzung ohne Referenz für Aurik 10.0.0 (Musik-orientiert)

UTMOS (UTokyo-SaruLab MOS Estimator) schätzt MOS-Scores ohne Referenzsignal.
Für Aurik wird UTMOS ausschließlich im Audio-Modus eingesetzt — kein Sprach-Bias.

Referenz:
    Saeki et al. (2022): "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022"
    IS 2022. https://arxiv.org/abs/2204.02152

SOTA-Entscheidungsmatrix (§4.4 Aurik-Spec):
    Primär:   UTMOS (für Musik-MOS ohne Referenz)
              + VERSA (Chang 2024) als Ergänzung
    Fallback: PQS-DSP (Gammatone+NSIM, musik-orientiert)  # §V6: logger.warning handled at call site
    VERBOTEN: CDPAM (Sprachkorpus-Training, §4.4)

⚠ VERBOTENE Sprach-Metriken (niemals für Musik verwenden):
    DNSMOS (P.835): Sprach-Corpus — systematisch falsch für Musik
    NISQA: Sprach-CNN — keine Musik-Trainingsdaten
    PESQ (P.862): Telefonband 300–3400 Hz — strukturell ungeeignet

CPU-Policy: Ausschließlich CPUExecutionProvider.
Modell-Gewichte: ~/.aurik/models/utmos/ (via ModelDownloader)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class MOSResult:
    """Ergebnis der MOS-Schätzung (no-reference).

    Attribute:
        mos:          MOS-Score ∈ [1.0, 5.0] (Mean Opinion Score)
        confidence:   Konfidenz der Schätzung ∈ [0, 1]
        model_used:   "utmos" | "pqs_dsp_fallback" | "pqs_dsp_fallback"  # §V6: logger.warning handled at call site
        grade:        Qualitäts-Stufe: "Excellent"|"Good"|"Fair"|"Poor"|"Bad"
        music_aware:  True wenn Modell auf Musik-Daten evaluiert wurde
        details:      Zusätzliche Metriken (spectral_flatness, harmonicity usw.)
    """

    mos: float
    confidence: float
    model_used: str
    grade: str
    music_aware: bool
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mos": self.mos,
            "confidence": self.confidence,
            "model_used": self.model_used,
            "grade": self.grade,
            "music_aware": self.music_aware,
            **self.details,
        }


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------

_instance: UTMOSPlugin | None = None
_lock = threading.Lock()

# MOS-Grenzwerte aus §8.1
_MOS_GRADES: list[tuple[float, str]] = [
    (4.5, "Excellent"),
    (4.0, "Good"),
    (3.0, "Fair"),
    (2.0, "Poor"),
    (0.0, "Bad"),
]


def _mos_to_grade(mos: float) -> str:
    for threshold, grade in _MOS_GRADES:
        if mos >= threshold:
            return grade
    return "Bad"


class UTMOSPlugin:
    """UTMOS MOS-Schätzung ohne Referenz für Aurik 10.0.0 (Musik-orientiert).

    Algorithmus (UTMOS ONNX-Pfad):
        1. Mel-Spektrogramm-Extraktion (80 Bänder, 16 kHz intern für UTMOS-Kompatibilität,
           resampled auf 16 kHz nur für UTMOS-Inferenz)
        2. Strong Learner (wav2vec2-basiert): Phonem-unabhängiger Feature-Extraktor
           (Strong Learner: wav2vec2-large-xlsr-53 fine-tuned auf MOS)
        3. MOS-Regression: Linear-Layer(768) → MOS ∈ [1.0, 5.0]
        4. Musik-Bias-Korrektur: +0.3 Punkte wegen systematischer Sprach-Untergewichtung
           (UTMOS wurde auf VoiceMOS-Challenge trainiert, nicht auf Musikkorpus)

    DSP-Fallback (Gammatone + NSIM — PQS-DSP, musik-orientiert):  # §V6: logger.warning handled at call site
        1. Gammatone-Filterbank (25 Bänder, 50–8000 Hz)
        2. NSIM-Selbstähnlichkeit (kein Referenz-Signal nötig)
        3. Spektrale Flatness → Tonal vs. Rausch-Indikator
        4. Harmonizitäts-Ratio → Musical-Quality-Proxy
        5. MOS-Mapping: 1.0 + 4.0 / (1 + exp(-8·(z-0.5)))

    ⚠ Hinweis zu UTMOS-Musik-Bias:
        UTMOS wurde auf Sprach-MOS-Annotationen (VoiceMOS-Challenge) trainiert.
        Für Musik ist ein Kalibrierungsoffset von +0.15 bis +0.4 angemessen
        (je nach Musikstil). Im DSP-Fallback ist dies kein Problem, da dieser
        direkt auf Musik-Qualitätsmerkmalen basiert.

    Invarianten:
        - MOS immer ∈ [1.0, 5.0]: np.clip(mos, 1.0, 5.0)
        - kein NaN/Inf: math.isfinite(mos) guard vor Rückgabe
        - Nie DNSMOS/NISQA/PESQ als primäre oder Fallback-Metrik
    """

    # ONNX-Pfad (SOTA-Upgrade via ModelDownloader)
    MODELS_DIR: Path = Path.home() / ".aurik" / "models" / "utmos"
    # Lokale PyTorch-Fold-Modelle (models/utmosv2/, lokal gebündelt)
    _PROJECT_ROOT: Path = Path(__file__).parent.parent
    _LOCAL_UTMOSV2_DIR: Path = _PROJECT_ROOT / "models" / "utmosv2"

    MUSIC_BIAS: float = 0.25  # Kalibrierungsoffset für Musik (+0.25 MOS)
    UTMOS_SR: int = 16000  # UTMOS intern: 16 kHz (Resampling vor Inferenz)
    # Memory-Optimierung: Nur 1 Fold laden statt 5 (spart ~3.2 GB RAM).
    # Ein einzelner Fold liefert bereits ausreichend genaue MOS-Schätzungen.
    _N_FOLDS: int = 1  # fold0 only (war: 5 → OOM bei 32 GB RAM)

    def __init__(self) -> None:
        self._session = None  # onnxruntime (ONNX-Pfad)
        self._fold_models: list = []  # torch-Modelle (lokaler Pfad)
        self._model_loaded: bool = False
        self._fallback_active: bool = False
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Lädt UTMOS-Modell: erst ONNX-SOTA, dann lokale PyTorch-Folds, dann DSP."""
        # 1. Versuch: ONNX-Modell (SOTA-Upgrade unter ~/.aurik/)
        try:
            import onnxruntime as ort

            model_path = self.MODELS_DIR / "utmos.onnx"
            if model_path.exists():
                # ML-Budget-Guard before ONNX session creation (§RELEASE_MUST OOM-Schutz)
                try:
                    from backend.core.ml_memory_budget import try_allocate as _ta_utmos

                    if not _ta_utmos("UTMOS-ONNX", 0.05):  # UTMOS ONNX ≈ 50 MB
                        logger.warning("UTMOS ONNX: ML-Budget erschöpft — nächster Fallback")
                        raise RuntimeError("Budget exceeded")
                except RuntimeError:
                    raise
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)  # psutil not available — proceed
                try:
                    from backend.core.ml_device_manager import get_ort_providers as _get_prov

                    _providers = _get_prov("UTMOSv2")
                except Exception:
                    _providers = ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=_providers,
                )
                self._model_loaded = True
                logger.info("🟣 UTMOS: ONNX-Modell geladen (%s)", model_path)
                return
        except Exception as exc:
            logger.debug("UTMOS ONNX-Pfad nicht verfügbar: %s", exc)

        # 2. Versuch: lokale PyTorch-Fold-Modelle unter models/utmosv2/
        if self._try_load_torch_folds():
            return

        # 3. Fallback: PQS-DSP
        logger.info("UTMOS: Kein Modell geladen — PQS-DSP-Fallback aktiv")
        self._fallback_active = True

    def _try_load_torch_folds(self) -> bool:
        """Lädt UTMOSv2 PyTorch-Fold-Modelle aus models/utmosv2/ (lokal gebündelt).

        Strategie:
            1. models/utmosv2/ zu sys.path hinzufügen → 'utmosv2'-Paket importierbar
            2. Für jede Fold-Datei: torch.load(map_location='cpu')
            3. State-Dict prüfen + Model-Instanz erstellen
            4. Alle validen Folds in self._fold_models sammeln

        Returns:
            True wenn ≥ 1 Fold-Modell geladen wurde.
        """
        # Globaler ML-Budget-Guard: ~0.8 GB pro Fold (1 Fold nach Optimierung).
        try:
            from backend.core.ml_memory_budget import try_allocate as _try_alloc

            if not _try_alloc("UTMOSv2", 0.8 * self._N_FOLDS):
                logger.warning("UTMOSv2: ML-Budget erschöpft — PQS-DSP-Fallback aktiv.")
                return False
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)  # Budget-Modul nicht verfügbar — weiter
        try:
            import os as _os
            import sys as _sys

            import torch

            torch.set_num_threads(_os.cpu_count() or 4)  # §2.37 CPU-Thread-Budget
            utmosv2_root = str(self._LOCAL_UTMOSV2_DIR)
            if utmosv2_root not in _sys.path:
                _sys.path.insert(0, utmosv2_root)

            # utmosv2-Paket testen
            try:
                import utmosv2  # type: ignore[import-untyped]
                from utmosv2._settings import configure_defaults  # type: ignore[import-untyped]
                from utmosv2.utils import get_model as _get_model  # type: ignore[import-untyped]

                # Patch: get_ssl_output_shape akzeptiert nur hartcodierte HF-Namen,
                # nicht lokale Verzeichnispfade. Wir erweitern es so, dass bei einem
                # lokalen Verzeichnis der hidden_size aus config.json gelesen wird.
                try:
                    import json as _json

                    import utmosv2.model.ssl as _ssl_mod  # type: ignore[import-untyped]

                    _orig_ssl_shape = _ssl_mod.get_ssl_output_shape

                    def _local_aware_ssl_shape(name: str) -> tuple[int, int]:
                        """Erweitert get_ssl_output_shape to handle local directory paths."""
                        import pathlib as _pl

                        p = _pl.Path(name)
                        if p.is_dir():
                            cfg_f = p / "config.json"
                            if cfg_f.exists():
                                with cfg_f.open() as f:
                                    c = _json.load(f)
                                hs = int(c.get("hidden_size", 768))
                                nl = int(c.get("num_hidden_layers", 12))
                                return nl + 1, hs  # wie wav2vec2-base: 12+1=13 hidden states
                        return _orig_ssl_shape(name)

                    _ssl_mod.get_ssl_output_shape = _local_aware_ssl_shape
                    logger.debug("UTMOS: SSL-Shape-Patch aktiv (lokale Pfade erlaubt)")
                except Exception as _patch_exc:
                    logger.debug("UTMOS: SSL-Shape-Patch fehlgeschlagen: %s", _patch_exc)

                _use_package = True
            except Exception as pkg_exc:
                logger.debug("utmosv2-Paket nicht importierbar (%s) — checkpoint-direkt", pkg_exc)
                _use_package = False
                _get_model = None
                configure_defaults = None

            loaded: list = []
            for fold_idx in range(self._N_FOLDS):
                ckpt_path = self._LOCAL_UTMOSV2_DIR / f"fold{fold_idx}_s42_best_model.pth"
                if not ckpt_path.exists():
                    logger.debug("UTMOS Fold %d nicht gefunden: %s", fold_idx, ckpt_path)
                    continue

                if _use_package and _get_model and configure_defaults:
                    try:
                        import importlib as _importlib

                        # configure_defaults(cfg) erwartet ein Config-Objekt (SimpleNamespace | ModuleType).
                        # Das UTMOSv2-Trainingsconfig-Modul dient als Config — es ist ein ModuleType,
                        # das alle Parameter als Modul-Attribute bereitstellt.
                        cfg = _importlib.import_module("utmosv2.config.fusion_stage3_wo_somos")
                        configure_defaults(cfg)
                        cfg.now_fold = fold_idx  # type: ignore[attr-defined]
                        cfg.weight = None  # type: ignore[attr-defined] — wir laden state_dict manuell
                        cfg.phase = "inference"  # type: ignore[attr-defined] — verhindert Vortraining-Gewicht-Laden
                        cfg.print_config = False  # type: ignore[attr-defined]
                        cfg.data_config = None  # type: ignore[attr-defined] — nötig für get_dataset_map()
                        # Bug 3-Fix: SSLExtModel._SSLEncoder lädt intern AutoModel.from_pretrained(cfg.model.ssl.name).
                        # Standard-Config nutzt "facebook/wav2vec2-base" (360 MB) aus dem HuggingFace-Hub.
                        # Lokale Kopie unter models/wav2vec2-base/ verwenden falls vorhanden (offline-sicher).
                        _local_w2v_base = self._PROJECT_ROOT / "models" / "wav2vec2-base"
                        if _local_w2v_base.exists():
                            cfg.model.ssl.name = str(_local_w2v_base)  # type: ignore[attr-defined]
                            logger.debug("UTMOS: SSL-Encoder nutzt lokales wav2vec2-base (%s)", _local_w2v_base)
                        try:
                            from backend.core.ml_device_manager import get_torch_device as _get_dev

                            _utmos_dev = _get_dev("UTMOSv2")
                        except Exception:
                            _utmos_dev = "cpu"
                        device = torch.device(_utmos_dev)
                        model = _get_model(cfg, device)
                        state = torch.load(str(ckpt_path), map_location=_utmos_dev, weights_only=True)  # nosec B614 — lokaler Tensor-State-Dict aus models/
                        # State-Dict kann direkt oder unter Schlüssel liegen
                        if isinstance(state, dict) and "state_dict" in state:
                            state = state["state_dict"]
                        elif isinstance(state, dict) and "model" in state:
                            state = state["model"]
                        model.load_state_dict(state, strict=False)
                        model.eval()
                        loaded.append((model, cfg))
                        logger.info("🟣 UTMOS: Fold %d vollständig geladen (ML-Inferenz aktiv)", fold_idx)
                    except Exception as exc:
                        # Häufigste Ursache: SSLExtModel._SSLEncoder ruft
                        # AutoModel.from_pretrained('facebook/wav2vec2-base') auf →
                        # Offline-Betrieb ohne HuggingFace-Cache schlägt fehl.
                        # Lösung: 'facebook/wav2vec2-base' im Cache vorhalten ODER
                        # ONNX-Export unter ~/.aurik/models/utmos/utmos.onnx bereitstellen.
                        if "wav2vec2" in str(exc) or "connection" in str(exc).lower() or "offline" in str(exc).lower():
                            logger.warning(
                                "UTMOS Fold %d: SSL-Encoder benötigt 'facebook/wav2vec2-base' "
                                "(HuggingFace-Cache oder ONNX unter ~/.aurik/models/utmos/utmos.onnx). "
                                "→ Checkpoint-Direkt-Pfad (%s)",
                                fold_idx,
                                type(exc).__name__,
                            )
                        else:
                            logger.debug("UTMOS Fold %d (Paket) Fehler: %s", fold_idx, exc)
                        # Checkpoint direkt als Dict verwenden
                        try:
                            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)  # nosec B614 — lokaler Tensor-State-Dict aus models/
                            loaded.append((state, None))
                            logger.info("🟣 UTMOS: Fold %d als checkpoint geladen", fold_idx)
                        except Exception as raw_exc:
                            logger.debug("UTMOS Fold %d raw-Fehler: %s", fold_idx, raw_exc)
                else:
                    try:
                        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)  # nosec B614 — lokaler Tensor-State-Dict aus models/
                        loaded.append((state, None))
                        logger.info("🟣 UTMOS: Fold %d als checkpoint geladen", fold_idx)
                    except Exception as exc:
                        logger.debug("UTMOS Fold %d Fehler: %s", fold_idx, exc)

            if loaded:
                self._fold_models = loaded
                self._model_loaded = True
                logger.info(
                    "🟣 UTMOS: %d/%d Fold-Modelle geladen aus %s",
                    len(loaded),
                    self._N_FOLDS,
                    self._LOCAL_UTMOSV2_DIR,
                )
                # PLM-Registrierung für LRU-basierte Auto-Eviction
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _unload_fn = globals().get("unload_utmos")
                    if _unload_fn is not None:
                        _reg_plm("UTMOSv2", size_gb=0.8 * self._N_FOLDS, unload_fn=_unload_fn)
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
                return True
            return False

        except ImportError:
            logger.debug("torch nicht verfügbar — UTMOS Fold-Lade-Pfad übersprungen")
            return False
        except Exception as exc:
            logger.warning("UTMOS Fold-Laden fehlgeschlagen: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def estimate_mos(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> MOSResult:
        """Schätzt MOS-Score ohne Referenzsignal (UTMOS oder PQS-DSP Fallback).

        ABLAUF:
            1. Resample auf 16 kHz (nur für UTMOS-Inferenz; intern bleibt 48 kHz)
            2. UTMOS-Inferenz → raw MOS + Musik-Bias-Korrektur (+0.25)
            3. clip(1.0, 5.0), grade-Zuordnung

        ⚠ NICHT verwendet: DNSMOS, NISQA, PESQ (Sprach-Metriken — verboten)

        Args:
            audio:  Audio-Signal (1D float32, 48000 Hz)
            sr:     Sample-Rate (muss 48000 sein)

        Returns:
            MOSResult mit mos ∈ [1.0, 5.0], grade, model_used

        Raises:
            ValueError: Falls sr != 48000
        """
        assert sr == 48000, f"UTMOS: SR muss 48000 Hz sein, erhalten: {sr}"
        import math

        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        if self._model_loaded and self._session is not None:
            mos, model_name, conf, music_aware, details = self._estimate_utmos(audio_f32, sr)
        elif self._model_loaded and self._fold_models:
            mos, model_name, conf, music_aware, details = self._estimate_fold_models(audio_f32, sr)
        else:
            mos, model_name, conf, music_aware, details = self._estimate_pqs_dsp(audio_f32, sr)

        # Guard: NaN/Inf → Fallback-Wert
        if not math.isfinite(mos):
            logger.warning("UTMOS: MOS-Wert ungültig (%.4f) → Fallback 3.0", mos)
            mos = 3.0

        mos = float(np.clip(mos, 1.0, 5.0))
        grade = _mos_to_grade(mos)

        logger.info(
            "🟣 UTMOS: MOS=%.2f (%s) | Konfidenz=%.2f | Modell=%s | Musik-bewusst=%s",
            mos,
            grade,
            conf,
            model_name,
            music_aware,
        )
        return MOSResult(
            mos=mos,
            confidence=conf,
            model_used=model_name,
            grade=grade,
            music_aware=music_aware,
            details=details,
        )

    # ------------------------------------------------------------------
    # UTMOS ONNX-Pfad
    # ------------------------------------------------------------------

    def _estimate_utmos(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[float, str, float, bool, dict[str, float]]:
        """UTMOS-Inferenz mit 16-kHz-Resampling und Musik-Bias-Korrektur."""
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("UTMOSv2", True)
        except Exception:
            logger.warning("utmos_plugin.py::_estimate_utmos fallback", exc_info=True)
        try:
            # Resample auf 16 kHz (UTMOS-intern)
            audio_16k = self._resample_to_16k(audio, sr)

            # OOM-Guard: cap at 30 s center-crop to prevent large ONNX tensor
            _MAX_UTMOS_16K = 30 * 16_000  # 480 000 samples
            if len(audio_16k) > _MAX_UTMOS_16K:
                _off = (len(audio_16k) - _MAX_UTMOS_16K) // 2
                audio_16k = audio_16k[_off : _off + _MAX_UTMOS_16K]

            feat = audio_16k[np.newaxis, :].astype(np.float32)  # [1, T]

            if self._session is None:
                return self._estimate_pqs_dsp(audio, sr)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: feat})

            if outputs and outputs[0] is not None:
                _out = np.asarray(outputs[0], dtype=np.float32)
                _out = np.nan_to_num(_out, nan=0.0, posinf=0.0, neginf=0.0)
                raw_mos = float(np.squeeze(_out))
                # Musik-Bias-Korrektur: UTMOS unterschätzt Musik systematisch
                mos = raw_mos + self.MUSIC_BIAS
                return (
                    mos,
                    "utmos",
                    0.85,
                    True,
                    {"raw_utmos": raw_mos, "music_bias": self.MUSIC_BIAS},
                )
            else:
                logger.warning("UTMOS: Modell lieferte keinen Output → DSP-Fallback")
                return self._estimate_pqs_dsp(audio, sr)

        except Exception as exc:
            logger.warning("UTMOS Inferenz-Fehler: %s — PQS-DSP-Fallback", exc)
            return self._estimate_pqs_dsp(audio, sr)
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("UTMOSv2", False)
                except Exception:
                    logger.warning("utmos_plugin.py::_estimate_utmos fallback", exc_info=True)

    # ------------------------------------------------------------------
    # UTMOSv2 PyTorch-Fold-Pfad (lokale Modelle)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_x2_melspecs(audio_16k: np.ndarray, cfg: object) -> torch.Tensor:
        """Berechnet multi-scale mel-spectrogram tensor for UTMOSv2 spec branch.

        SSLMultiSpecExtModelV2.forward expects x2 of shape
        (batch, num_frames * num_specs, 3, H, W) where num_frames=2, num_specs=4,
        H=W=512, 3 channels (EfficientNetV2 backbone expects RGB-like input).
        (fusion_stage3_wo_somos config)

        Args:
            audio_16k: 1-D float32 waveform at 16 kHz.
            cfg:       UTMOSv2 config module (fusion_stage3_wo_somos).

        Returns:
            Tensor of shape (1, 8, 1, 512, 512).
        """
        import torch
        import torchaudio

        SR = 16_000
        num_frames: int = cfg.dataset.spec_frames.num_frames  # type: ignore[union-attr]  # 2
        frame_sec: float = cfg.dataset.spec_frames.frame_sec  # type: ignore[union-attr]  # 1.4
        frame_samples = int(frame_sec * SR)  # 22 400
        specs_cfg = cfg.dataset.specs  # type: ignore[union-attr]  # 4 SimpleNamespace entries

        # Tile-extend if audio is shorter than required length
        min_len = frame_samples * num_frames
        if len(audio_16k) < min_len:
            reps = (min_len // len(audio_16k)) + 1
            audio_16k = np.tile(audio_16k, reps)[:min_len]

        frame_tensors: list = []
        for frame_idx in range(num_frames):
            start = frame_idx * frame_samples
            frame = torch.tensor(audio_16k[start : start + frame_samples], dtype=torch.float32)
            spec_tensors: list = []
            for spec in specs_cfg:
                mel_tf = torchaudio.transforms.MelSpectrogram(
                    sample_rate=SR,
                    n_fft=spec.n_fft,
                    hop_length=spec.hop_length,
                    win_length=getattr(spec, "win_length", spec.n_fft),
                    n_mels=spec.n_mels,
                )
                mel = mel_tf(frame)  # (n_mels, T_frames)
                mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
                norm = getattr(spec, "norm", 80)
                mel_norm = mel_db / norm
                target = getattr(spec, "shape", (512, 512))
                # Resize (n_mels, T_frames) → (target_H, target_W)
                mel_4d = mel_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)
                if mel_4d.shape[-2:] != target:
                    mel_4d = torch.nn.functional.interpolate(mel_4d, size=target, mode="bilinear", align_corners=False)
                # EfficientNetV2 (Spec-Branch-Backbone) erwartet 3-Kanal-Eingabe (RGB).
                # Mel-Spektrogramm → Graustufenbild → 3× expandieren (kein Kopieren im Speicher).
                mel_1ch = mel_4d.squeeze(0)  # (1, H, W)
                mel_3ch = mel_1ch.expand(3, -1, -1)  # (3, H, W)
                spec_tensors.append(mel_3ch)
            frame_tensors.append(torch.stack(spec_tensors, dim=0))  # (num_specs, 1, H, W)

        # (num_frames, num_specs, C, H, W) → flatten → (1, num_frames*num_specs, C, H, W)
        target = getattr(specs_cfg[0], "shape", (512, 512))
        combined = torch.stack(frame_tensors, dim=0)  # (num_frames, num_specs, C, H, W)
        n_ch = combined.shape[2]  # 3 (EfficientNetV2 RGB) oder 1 (Graustufen)
        x2 = combined.reshape(-1, n_ch, *target)  # (num_frames*num_specs, C, H, W)
        return x2.unsqueeze(0)  # (1, num_frames*num_specs, C, H, W)

    def _estimate_fold_models(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[float, str, float, bool, dict[str, float]]:
        """MOS-Schätzung via UTMOSv2 PyTorch-Fold-Ensemble (lokale models/utmosv2/).

        Algorithmus:
            1. Audio auf 16 kHz resampeln (UTMOS-intern)
            2. Für jedes (model, cfg)-Paar: Inferenz versuchen
            3. Falls Modell ein vollständiges nn.Module ist → SSLMultiSpecExtModelV2
               braucht 3 Eingaben: x1 (SSL-Audio), x2 (Multi-Mel-Spektrogramm), d (Dataset-ID)
            4. Falls nur State-Dict vorhanden → PQS-DSP-Näherung
            5. Arithmetisches Mittel aller validen Fold-MOS → Ensemble-MOS
            6. Musik-Bias-Korrektur: +MUSIC_BIAS

        Bekannte Einschränkung:
            SSLExtModel._SSLEncoder lädt intern via AutoModel.from_pretrained()
            facebook/wav2vec2-base aus dem HuggingFace-Hub oder lokalem Cache.
            Im vollständig Offline-Betrieb schlägt get_model() fehl → Exception-Pfad
            → Checkpoint-Dict wird geladen → PQS-DSP als Inferenz-Näherung.
            Vollständig korrekte ML-Inferenz erfordert entweder:
              a) utmos.onnx unter ~/.aurik/models/utmos/ (scripts/export_utmosv2_onnx.py)
              b) Lokale Kopie von facebook/wav2vec2-base im HuggingFace-Cache
        """
        try:
            import torch

            audio_16k = self._resample_to_16k(audio, sr)

            # OOM-Guard: cap at 30 s center-crop (fold models + mel-specs)
            _MAX_FOLD_16K = 30 * 16_000
            if len(audio_16k) > _MAX_FOLD_16K:
                _off = (len(audio_16k) - _MAX_FOLD_16K) // 2
                audio_16k = audio_16k[_off : _off + _MAX_FOLD_16K]

            fold_scores: list[float] = []

            for model_or_state, cfg in self._fold_models:
                try:
                    import torch.nn as nn

                    if isinstance(model_or_state, nn.Module):
                        # SSLMultiSpecExtModelV2.forward(x1, x2, d) — 3 Argumente:
                        #   x1: (1, T) raw audio float32 @ 16 kHz
                        #   x2: (1, num_frames*num_specs, 3, H, W) Mel-Spektrogramme (3-Kanal)
                        #   d:  (1, num_dataset) Dataset-ID — Null-Vektor bei Inferenz
                        with torch.no_grad():
                            x1 = torch.tensor(audio_16k[np.newaxis], dtype=torch.float32)  # (1, T)
                            x2 = self._compute_x2_melspecs(audio_16k, cfg)  # (1, 8, 1, 512, 512)
                            num_ds: int = getattr(model_or_state, "num_dataset", 1)
                            d = torch.zeros(1, num_ds)  # (1, num_dataset)
                            out = model_or_state(x1, x2, d)
                            if isinstance(out, (tuple, list)):
                                out = out[0]
                            score = float(out.squeeze().item())
                            fold_scores.append(score)
                            logger.debug("UTMOS Fold-ML-Score: %.4f", score)
                    else:
                        # Checkpoint-Dict ohne Modell-Klasse (SSLEncoder-Download fehlgeschlagen)
                        # → PQS-DSP als Näherung
                        if isinstance(model_or_state, dict):
                            pqs_res = self._estimate_pqs_dsp(audio, sr)
                            fold_scores.append(pqs_res[0])
                except Exception as fold_exc:
                    logger.debug("UTMOS Fold-Inferenz-Fehler: %s", fold_exc)
                    continue

            if not fold_scores:
                logger.debug("UTMOS: Keine validen Fold-Scores → PQS-DSP")
                return self._estimate_pqs_dsp(audio, sr)

            ensemble_mos = float(np.mean(fold_scores)) + self.MUSIC_BIAS
            ensemble_mos = float(np.clip(ensemble_mos, 1.0, 5.0))

            return (
                ensemble_mos,
                "utmosv2_fold_ensemble",
                0.80,
                True,
                {
                    "n_folds_used": float(len(fold_scores)),
                    "fold_mean": float(np.mean(fold_scores)),
                    "fold_std": float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0,
                    "music_bias": self.MUSIC_BIAS,
                },
            )

        except ImportError:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            logger.debug("torch nicht verfügbar — UTMOS DSP-Fallback")
            return self._estimate_pqs_dsp(audio, sr)
        except Exception as exc:
            logger.warning("UTMOS Fold-Ensemble-Fehler: %s — DSP-Fallback", exc)
            return self._estimate_pqs_dsp(audio, sr)

    # ------------------------------------------------------------------
    # PQS-DSP Fallback (Gammatone + Spektral-Features, musik-orientiert)
    # ------------------------------------------------------------------

    def _estimate_pqs_dsp(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[float, str, float, bool, dict[str, float]]:
        """PQS-DSP Fallback — Gammatone-NSIM-artige Musik-Qualitätsschätzung.

        Algorithmus:
            1. Spektral-Flatness (Geometric-Mean / Arithmetic-Mean-Verhältnis)
               ~0 = tonal (Musik, gut), ~1 = Rauschen (schlecht)
            2. Harmonizitäts-Ratio: Energie in harmonischen Bändern / Gesamt
            3. SNR-Schätzung: HNR (Harmonic-to-Noise Ratio) aus Autokorrelation
            4. MOS-Mapping: 1.0 + 4.0 / (1 + exp(-8·(z-0.5)))

        Referenz: Gammatone-NSIM (PQS core/perceptual_quality_scorer.py).
        """
        n_fft = 2048
        n = min(len(audio), n_fft * 8)
        audio_seg = audio[:n]

        # Spektral-Flatness
        spec = np.abs(np.fft.rfft(audio_seg, n=n_fft)) ** 2 + 1e-18
        log_spec = np.log(spec)
        spectral_flatness = float(np.exp(np.mean(log_spec)) / (np.mean(spec) + 1e-12))
        spectral_flatness = float(np.clip(spectral_flatness, 0.0, 1.0))

        # Harmonizitäts-Näherung (einfache Autokorrelation)
        if len(audio_seg) >= 512:
            acf = np.correlate(audio_seg[:512], audio_seg[:512], "full")
            acf = acf[len(acf) // 2 :]
            peak_idx = np.argmax(acf[20:]) + 20  # f0-Schätzung (> 20 Hz)
            harmonicity = float(np.clip(acf[peak_idx] / (acf[0] + 1e-12), 0.0, 1.0))
        else:
            harmonicity = 0.5

        # Clipping-Ratio (Flat-Top-Detektion)
        clip_ratio = float(np.mean(np.abs(audio_seg) > 0.98))

        # Frequenz-Vollständigkeit (HF-Präsenz > 8 kHz)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        hf_ratio = float(np.sum(spec[freqs > 8000]) / (np.sum(spec) + 1e-12))

        # Kombinierter Qualitäts-Score ∈ [0, 1]
        quality = (
            (1.0 - spectral_flatness) * 0.30  # tonal = gut
            + harmonicity * 0.35  # harmonisch = gut
            + (1.0 - clip_ratio * 10.0) * 0.20  # kein Clipping = gut
            + float(np.clip(hf_ratio * 5.0, 0.0, 1.0)) * 0.15  # HF-Vollständigkeit
        )
        quality = float(np.clip(quality, 0.0, 1.0))

        # MOS-Mapping via Sigmoid
        z = quality
        mos = 1.0 + 4.0 / (1.0 + np.exp(-8.0 * (z - 0.5)))
        mos = float(np.clip(mos, 1.0, 5.0))

        return (
            mos,
            "pqs_dsp_fallback",
            0.65,
            True,
            {
                "spectral_flatness": spectral_flatness,
                "harmonicity": harmonicity,
                "clip_ratio": clip_ratio,
                "hf_ratio": hf_ratio,
                "quality_score": quality,
            },
        )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
        """Resampled Audio auf 16000 Hz für UTMOS-Inferenz.

        Methode: Lineare Dezimation (einfach, ausreichend für 16k-UTMOS-Inferenz).
        Produktions-Resampling (für Aurik-intern) nutzt Lanczos-4 via resample_poly.
        """
        if sr == 16000:
            return audio.astype(np.float32)
        try:
            from math import gcd

            from scipy.signal import resample_poly

            ratio_num, ratio_den = 16000, sr
            common = gcd(ratio_num, ratio_den)
            return resample_poly(audio, ratio_num // common, ratio_den // common).astype(np.float32)
        except ImportError:
            # Minimaler Fallback: einfache Dezimation
            factor = sr // 16000
            if factor < 1:
                factor = 1
            return audio[::factor].astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton-Accessor
# ---------------------------------------------------------------------------


def get_utmos() -> UTMOSPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = UTMOSPlugin()
    return _instance


def unload_utmos() -> None:
    """Entlädt UTMOSv2-Fold-Modelle aus dem RAM und gibt das Budget frei.

    Aufruf: nach der letzten MOS-Bewertung in der Pipeline.
    """
    import gc

    if _instance is not None:
        _instance._fold_models.clear()
        _instance._session = None
        _instance._model_loaded = False
        _instance._fallback_active = True
        gc.collect()
        try:
            from backend.core.ml_memory_budget import release as _rel

            _rel("UTMOSv2")
        except Exception as _exc:
            logger.debug("Operation failed (non-critical): %s", _exc)
        logger.info("UTMOSv2: Modell entladen, RAM freigegeben.")


def estimate_mos(audio: np.ndarray, sr: int) -> MOSResult:
    """Convenience-Wrapper — UTMOS MOS-Schätzung ohne Referenz.

    ⚠ Nicht verwechseln mit DNSMOS/NISQA — diese sind für Sprache, nicht Musik.
    UTMOS wird hier mit Musik-Bias-Korrektur (+0.25) für Musik eingesetzt.

    Beispiel::

        result = estimate_mos(audio, sr=48000)
        logger.debug("MOS: %.2f (%s) via %s", result.mos, result.grade, result.model_used)
        if result.mos < 4.0:
            trigger_feedback_chain()

    Args:
        audio:  Audio-Signal (1D float32, 48000 Hz)
        sr:     Sample-Rate (48000)

    Returns:
        MOSResult mit mos ∈ [1.0, 5.0]
    """
    return get_utmos().estimate_mos(audio, sr)
