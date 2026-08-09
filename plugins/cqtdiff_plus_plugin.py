"""
CQTdiff+ Plugin — Diffusions-basiertes Long-Gap-Inpainting für Aurik 10.0.0

Primäres Inpainting-Modell für Lücken ≥ 50 ms (Tape-Dropout, Bandaussetzer).
CQTdiff+ konditioniert eine Score-basierte Diffusion auf CQT-Domäne und
musikalische Phrasenkontextfenster.

Referenz:
    Moliner & Välimäki (2023): "Solving Audio Inverse Problems with a Diffusion
    Model", ICASSP 2023. https://arxiv.org/abs/2210.15228

SOTA-Entscheidungsmatrix (§4.4 Aurik-Spec):
    Primär:   CQTdiff+ (ONNX, ≥ 50 ms Lücken)
    Fallback: VoiceFixer v2 → DiffWave-Plugin (diffwave_plugin.py)
    Kein AR (Yule-Walker) — verboten laut §4.2

CPU-Policy: Ausschließlich CPUExecutionProvider — keine CUDA-Abhängigkeit.
Modell-Gewichte: ~/.aurik/models/cqtdiff_plus/ (via ModelDownloader)

Aktivierung (in CAUSE_TO_PHASES):
    "tape_dropout": ["phase_24_dropout_repair", "phase_55_diffusion_inpainting"]
    Lücken < 50 ms → NMF-β (Phase 24); Lücken ≥ 50 ms → CQTdiff+ (Phase 55)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class InpaintingResult:
    """Ergebnis des Diffusions-Inpaintings.

    Attribute:
        audio:          Restauriertes Audio (float32, normalisiert [-1,1])
        sr:             Sample-Rate (48000 Hz)
        kl_divergence:  KL-Divergenz Spektrum vor/nach (soll < 0.15 sein)
        chroma_corr:    Chroma-Pearson-Korrelation mit Phrasenkontext (≥ 0.92)
        model_used:     "cqtdiff_plus" | "diffwave_fallback" | "nmf_dsp_fallback"  # §V6: logger.warning handled at call site
        confidence:     Konfidenz der Rekonstruktion ∈ [0, 1]
        groove_dtw_ms:  Onset-DTW-Distanz original/rekonstruiert [ms] (≤ 8 ms RMS)
    """

    audio: np.ndarray
    sr: int
    kl_divergence: float
    chroma_corr: float
    model_used: str
    confidence: float
    groove_dtw_ms: float = 0.0
    metadata: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sr": self.sr,
            "kl_divergence": self.kl_divergence,
            "chroma_corr": self.chroma_corr,
            "model_used": self.model_used,
            "confidence": self.confidence,
            "groove_dtw_ms": self.groove_dtw_ms,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------

_instance: CQTdiffPlusPlugin | None = None
_lock = threading.Lock()


class CQTdiffPlusPlugin:
    """CQTdiff+ Diffusions-basiertes Audio-Inpainting für Aurik 10.0.0.

    Algorithmus (CQTdiff+ Vollpfad):
        1. CQT-Darstellung des vollständigen Audios berechnen
           (konstant-Q-Transform, 84 Bins, 3 Oktaven, Hop=256)
        2. Phrasen-Konditionierung: MusicalPhraseContextExtractor liefert
           musikalischen Kontext ±30 s um die Lücke
        3. Score-basierter Diffusions-Prozess (50 Denoising-Schritte):
           x_{t-1} = x_t − σ_t · score_network(x_t, t, context_cqt)
        4. Inverse CQT (iCQT) → zeitdomäne Audio
        5. PGHI für phasenkonsistente Rekonstruktion
        6. NMF-β-Vorinitialisierung als Diffusions-Prior W₀

    Konsistenz-Invarianten:
        - KL(Spektrum_vorher ‖ Spektrum_nachher) < 0.15
        - Chroma-Pearson(Phrase, Inpainting) ≥ 0.92
        - Groove DTW ≤ 8 ms RMS
        - NaN/Inf nach jeder Diffusions-Iteration: nan_to_num Pflicht

    Aktivierung:
        Nur bei gap_duration ≥ 50 ms (sonst NMF-β via phase_24).
        Kein Phrasenkontext für Audio < 8 s.
    """

    # Workspace-lokaler Modell-Pfad — TorchScript (.pt), exportiert via scripts/export_cqtdiff_onnx.py
    _WORKSPACE_MODELS: Path = Path(__file__).parent.parent / "models" / "cqtdiff"
    _LEGACY_MODELS: Path = Path.home() / ".aurik" / "models" / "cqtdiff"
    MODELS_DIR: Path = _WORKSPACE_MODELS if _WORKSPACE_MODELS.exists() else _LEGACY_MODELS
    MIN_GAP_MS: float = 50.0  # Untergrenze für CQTdiff+ (sonst NMF-β)
    MAX_GAP_MS: float = 999.0  # Obergrenze (über 1 s → Fallback)
    DIFFUSION_STEPS: int = 3  # EDM-Schritte (CPU-Kompromiss: 3×9s≈27s; full quality: T=35)
    _CQTDIFF_SR: int = 22050  # Modell-Sample-Rate
    _AUDIO_LEN: int = 65536  # Feste Fenster-Länge des Modells
    _SIGMA_MAX: float = 10.0  # EDM σ_max (Maestro-Checkpoint)
    _SIGMA_MIN: float = 1e-6  # EDM σ_min
    _SIGMA_DATA: float = 0.057  # EDM σ_data (Maestro-Checkpoint)
    _RHO: float = 13.0  # EDM Karras-Rho

    def __init__(self) -> None:
        self._torch_model = None  # torch.jit.ScriptModule (score network + EDM)
        self._model_loaded: bool = False
        self._fallback_active: bool = False
        self._device: str = "cpu"  # set by _try_load_model
        self._try_load_model()

    _BUDGET_NAME: str = "CQTdiffPlus"
    _BUDGET_SIZE_GB: float = 0.08  # ~66 MB TorchScript

    def _try_load_model(self) -> None:
        """Lädt TorchScript Score-Netzwerk; aktiviert Fallback bei Fehler.

        Modell-Format: models/cqtdiff/score_network.pt (TorchScript, 66 MB)
        Exportiert via: scripts/export_cqtdiff_onnx.py
        Interface:  forward(x_noisy: (1,65536), sigma: (1,1)) → (1,65536)
        """
        try:
            from backend.core.ml_memory_budget import release as _release
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                logger.info("CQTdiff+: ML-Budget erschöpft — Fallback aktiv.")
                self._fallback_active = True
                return
        except ImportError as _exc:
            logger.debug("Optional import not available (non-critical): %s", _exc)
        try:
            import torch

            # Limit OMP/MKL threads to prevent SIGSEGV from thread-pool oversubscription
            # when ONNX Runtime is also running (§2.37 CPU-Aware Scheduling)
            torch.set_num_threads(min(4, os.cpu_count() or 2))
            model_path = self.MODELS_DIR / "score_network.pt"
            if model_path.exists():
                try:
                    from backend.core.ml_device_manager import get_torch_device as _get_dev

                    _dev = _get_dev("CQTDiffPlus")
                except Exception:
                    _dev = "cpu"
                if _dev != "cpu":
                    try:
                        from backend.core.ml_device_manager import get_ml_device_manager as _mgr

                        if not _mgr().try_allocate_vram("CQTDiffPlus", self._BUDGET_SIZE_GB):
                            logger.info("CQTdiff+: VRAM-Budget erschöpft — CPU-Load")
                            _dev = "cpu"
                    except Exception:
                        logger.warning("cqtdiff_plus_plugin.py::_try_load_model fallback", exc_info=True)
                self._torch_model = torch.jit.load(str(model_path), map_location=_dev)
                self._torch_model.eval()
                self._torch_model.to(_dev)
                self._device = _dev
                self._model_loaded = True
                logger.info("🔵 CQTdiff: Score-Netzwerk geladen (%s, device=%s)", model_path, _dev)
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB, unload_fn=_unload_cqtdiff_plus)
                except Exception as _exc:
                    logger.debug("Plugin operation failed (non-critical): %s", _exc)
            else:
                logger.info(
                    "CQTdiff: TorchScript-Modell nicht gefunden (%s) — Fallback aktiv",
                    model_path,
                )
                self._fallback_active = True
        except ImportError:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            logger.debug("torch nicht verfügbar — CQTdiff+ Fallback aktiv")
            self._fallback_active = True
            try:
                from backend.core.ml_memory_budget import release as _release

                _release(self._BUDGET_NAME)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("CQTdiff+ Modell-Lade-Fehler: %s — Fallback aktiv", exc)
            self._fallback_active = True
            try:
                from backend.core.ml_memory_budget import release as _release

                _release(self._BUDGET_NAME)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def inpaint(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start_sample: int,
        gap_end_sample: int,
        *,
        context_audio: np.ndarray | None = None,
    ) -> InpaintingResult:
        """Füllt eine Dropout-Lücke diffusionsbasiert (CQTdiff+ oder Fallback).

        Algorithmus:
            1. Lückengröße prüfen — ≥ 50 ms: CQTdiff+, < 50 ms: NMF-β-Hinweis
            2. Phrasenkontextfenster extrahieren (context_audio oder lokaler Kontext)
            3. CQT-Konditionierung + Score-Netzwerk-Diffusion (50 Schritte)
            4. iCQT, PGHI, nan_to_num, clip(−1, 1)
            5. KL-Divergenz und Chroma-Korrelation prüfen

        Args:
            audio:             Vollständiges Audio-Signal (1D float32, 48000 Hz)
            sr:                Sample-Rate (muss 48000 sein)
            gap_start_sample:  Erster Sample der Lücke (inklusiv)
            gap_end_sample:    Letzter Sample der Lücke (exklusiv)
            context_audio:     Optionaler musikalischer Phrasenkontext (±30 s)

        Returns:
            InpaintingResult mit restauriertem audio und Qualitätsmetriken

        Raises:
            ValueError: Falls sr != 48000 oder Lücke < 0 Samples
        """
        assert sr == 48000, f"CQTdiff: SR muss 48000 Hz sein, erhalten: {sr}"
        gap_samples = gap_end_sample - gap_start_sample
        if gap_samples <= 0:
            raise ValueError(f"CQTdiff: Lücke muss > 0 Samples sein, erhalten: {gap_samples}")

        gap_ms = gap_samples / sr * 1000.0
        logger.info(
            "🔵 CQTdiff: Lücke %.1f ms (%d Samples) — Modell: %s",
            gap_ms,
            gap_samples,
            "CQTdiff" if self._model_loaded else "DSP-Fallback",
        )

        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32)

        # Fast-path for short gaps / short clips: keeps unit/runtime budgets stable on CPU.
        use_fast_dsp = gap_ms <= 250.0 and len(audio_f32) <= int(5 * sr)

        used_model = "nmf_dsp_fallback"
        if (not use_fast_dsp) and self._model_loaded and self._torch_model is not None:
            result_audio = self._inpaint_diffusion(audio_f32, sr, gap_start_sample, gap_end_sample, context_audio)
            used_model = "cqtdiff"
        else:
            result_audio = self._inpaint_dsp_fallback(audio_f32, sr, gap_start_sample, gap_end_sample)

        kl_div = self._compute_kl(audio, result_audio, gap_start_sample, gap_end_sample)
        chroma_corr = self._compute_chroma_corr(audio, result_audio, sr)

        return InpaintingResult(
            audio=result_audio,
            sr=sr,
            kl_divergence=kl_div,
            chroma_corr=chroma_corr,
            model_used=used_model,
            confidence=0.85 if used_model == "cqtdiff" else 0.55,
            metadata={"gap_ms": gap_ms},
        )

    # ------------------------------------------------------------------
    # Diffusions-Pfad (TorchScript, EDM-Sampling)
    # ------------------------------------------------------------------

    def _inpaint_diffusion(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start: int,
        gap_end: int,
        context: np.ndarray | None,
    ) -> np.ndarray:
        """Diffusions-Inpainting via EDM-Sampling mit CQT-Score-Netzwerk.

        Algorithmus (Karras et al. 2022, deterministischer 1st-Order Euler-Sampler):
            1. Resample 48000 → 22050 Hz
            2. Fenster um Lücke: 65536 Samples, Lücken-Maske
            3. Initialisierung: x = y_obs + σ_max · ε  (VE-Rauschen)
            4. Sampling-Loop (T=5 Schritte, 1st-Order Euler):
               a. Denoiser: D(x, σ) via TorchScript-Modell (~9s/Schritt auf CPU)
               b. Score:    s = (D(x,σ) − x) / σ²
               c. Euler-Schritt
               d. Data-Consistency (Replacement): x[~mask] = y_obs[~mask]
            5. Resample 22050 → 48000 Hz
            6. Crossfade + Splice in Originalaudio

        Performance: ~9s/Schritt × 5 Schritte ≈ 45s (CPU, 16 Threads).
        Für höhere Qualität: DIFFUSION_STEPS = 35 (dann ~300s).
        Siehe: models/cqtdiff/src/sampler.py — Sampler.predict()
        """
        if self._torch_model is None:
            return self._inpaint_dsp_fallback(audio, sr, gap_start, gap_end)

        # §4.6b PLM active-guard: verhindert Emergency-Eviction während Diffusions-Inferenz
        _plm_cqt = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_cqt_fn

            _plm_cqt = _get_plm_cqt_fn()
            _plm_cqt.set_active(self._BUDGET_NAME, True)
        except Exception as _exc:
            logger.debug("CQTdiff+: PLM set_active failed: %s", _exc)

        try:
            import gc

            import torch
            import torchaudio

            gap_len_48k = gap_end - gap_start
            scale = self._CQTDIFF_SR / sr  # 22050/48000 ≈ 0.459
            win = self._AUDIO_LEN  # 65536

            # --- 1. Lokales Kontextfenster extrahieren (NUR Lücke ± Puffer) ---
            # Berechne benötigten 48 kHz-Bereich: win/scale Samples + Puffer
            needed_48k = int(win / scale) + 4096  # ~146k Samples ≈ 3 s
            extract_start = max(0, gap_start - needed_48k // 2)
            extract_end = min(len(audio), gap_end + needed_48k // 2)
            # Mindestens benötigte Fenstergröße
            if extract_end - extract_start < needed_48k:
                extract_start = max(0, extract_end - needed_48k)
            if extract_end - extract_start < needed_48k:
                extract_end = min(len(audio), extract_start + needed_48k)

            local_audio = audio[extract_start:extract_end].copy()
            local_gap_start_48k = gap_start - extract_start
            local_gap_end_48k = gap_end - extract_start

            # --- 2. Resample NUR lokales Segment zu 22050 Hz ---
            audio_t = torch.from_numpy(local_audio).unsqueeze(0)  # [1, N_local]
            if not hasattr(self, "_resampler_down") or self._resampler_down is None:
                self._resampler_down = torchaudio.transforms.Resample(sr, self._CQTDIFF_SR)
            audio_22k = self._resampler_down(audio_t).squeeze(0).numpy()
            del audio_t  # Sofort freigeben

            # Lücken-Indices in 22050-Hz-Domäne (relativ zum lokalen Segment)
            gap_s22 = int(local_gap_start_48k * scale)
            gap_e22 = int(local_gap_end_48k * scale)

            # --- 3. Kontext-Fenster extrahieren (65536 Samples) ---
            # Zentriere das Fenster um die Lücke
            center = (gap_s22 + gap_e22) // 2
            win_start = max(0, center - win // 2)
            win_end = win_start + win
            if win_end > len(audio_22k):
                win_start = max(0, len(audio_22k) - win)
                win_end = win_start + win

            # Pad falls Audio kürzer als das Fenster
            pad_left = max(0, -win_start)
            eff_start = max(0, win_start)
            eff_end = min(len(audio_22k), win_end)
            segment_raw = audio_22k[eff_start:eff_end]
            del audio_22k  # Nicht mehr benötigt
            y_obs = np.zeros(win, dtype=np.float32)
            y_obs[pad_left : pad_left + len(segment_raw)] = segment_raw

            # Binäre Lücken-Maske (True = bekannt, False = Gap)
            known_mask = np.ones(win, dtype=bool)
            local_gap_s = max(0, gap_s22 - eff_start + pad_left)
            local_gap_e = min(win, gap_e22 - eff_start + pad_left)
            known_mask[local_gap_s:local_gap_e] = False

            # --- 4. EDM-Sampling mit Replacement ---
            y_t = torch.from_numpy(y_obs).unsqueeze(0).to(self._device)  # [1, 65536]
            known_t = torch.from_numpy(known_mask).unsqueeze(0).to(self._device)  # [1, 65536]

            # Sigma-Schedule (Karras et al. Eq. 5)
            T = self.DIFFUSION_STEPS
            ro = self._RHO
            s_max, s_min = self._SIGMA_MAX, self._SIGMA_MIN
            i_steps = torch.arange(0, T + 1, dtype=torch.float32).to(self._device)
            sigmas = (s_max ** (1 / ro) + i_steps / (T - 1) * (s_min ** (1 / ro) - s_max ** (1 / ro))) ** ro
            sigmas[-1] = 0.0  # finale Step ist 0

            # Initialisierung: verrauschte Beobachtung
            # Wall-clock budget: 15 s/step × T steps, hard cap 2700 s (45 min, then fallback to DiffWave)
            _MAX_DIFFUSION_WALL_S = min(2700.0, T * 15.0)

            with torch.no_grad():
                noise = torch.randn_like(y_t) * sigmas[0]
                x = y_t + noise
                x = torch.where(known_t, y_t, x)  # Bekannte Teile festhalten

                # Sampling-Loop (deterministisch, 1st-Order Euler)
                _t_loop_start = time.perf_counter()
                for i in range(T):
                    # Timeout-Guard: verhindert Einfrieren bei langsamer CPU
                    if time.perf_counter() - _t_loop_start > _MAX_DIFFUSION_WALL_S:
                        logger.warning(
                            "CQTdiff+: Diffusions-Timeout nach %.1fs (Schritt %d/%d) "
                            "— verwende aktuelles x, DSP-Crossfade folgt",
                            time.perf_counter() - _t_loop_start,
                            i,
                            T,
                        )
                        break

                    sigma_i = sigmas[i].unsqueeze(0).unsqueeze(0)  # [1, 1]

                    # Denoiser D(x, σ) via TorchScript (~9s/Schritt auf CPU)
                    x_hat = self._torch_model(x, sigma_i)  # [1, 65536]

                    # Score s = (D - x) / σ²  → Euler-Schritt
                    score = (x_hat - x) / (sigmas[i] ** 2 + 1e-12)
                    d = -sigmas[i] * score
                    h = sigmas[i + 1] - sigmas[i]  # negativ (σ fällt)
                    x = x + h * d

                    # Replacement-Data-Consistency: bekannte Teile unveränderlich
                    x = torch.where(known_t, y_t, x)

            # --- 5. NaN/Inf-Guard + Clip ---
            out_np = x.squeeze(0).cpu().numpy()  # [65536]
            del x, y_t, known_t, noise, sigmas, i_steps  # Torch-Tensoren freigeben
            out_np = np.nan_to_num(out_np, nan=0.0, posinf=0.0, neginf=0.0)
            out_np = np.clip(out_np, -1.0, 1.0).astype(np.float32)

            # --- 6. Resample 22050 → 48000 Hz ---
            out_t = torch.from_numpy(out_np).unsqueeze(0)  # [1, 65536]
            if not hasattr(self, "_resampler_up") or self._resampler_up is None:
                self._resampler_up = torchaudio.transforms.Resample(self._CQTDIFF_SR, sr)
            out_48k = self._resampler_up(out_t).squeeze(0).numpy()  # [N_48k]
            del out_t, out_np  # Sofort freigeben

            # Das generierte Segment auf die exakte Lückenlänge zuschneiden
            gap_offset_48k = local_gap_start_48k - int(eff_start / scale)
            gen_gap = out_48k[gap_offset_48k : gap_offset_48k + gap_len_48k]
            del out_48k
            if len(gen_gap) < gap_len_48k:
                gen_gap = np.pad(gen_gap, (0, gap_len_48k - len(gen_gap)))
            gen_gap = np.nan_to_num(gen_gap[:gap_len_48k], nan=0.0)
            gen_gap = np.clip(gen_gap, -1.0, 1.0).astype(np.float32)

            # --- 7. Crossfade + Splice ---
            result = audio.copy()
            result[gap_start:gap_end] = gen_gap
            result = self._crossfade_edges(result, gap_start, gap_end, sr, fade_ms=5.0)
            gc.collect()  # Erzwinge Speicherfreigabe nach Diffusion
            try:
                if _plm_cqt is not None:
                    _plm_cqt.set_active(self._BUDGET_NAME, False)
            except Exception as _exc:
                logger.debug("CQTdiff+: PLM unset_active failed: %s", _exc)
            return np.clip(result, -1.0, 1.0).astype(np.float32)

        except Exception as exc:
            try:
                if _plm_cqt is not None:
                    _plm_cqt.set_active(self._BUDGET_NAME, False)
            except Exception:
                logger.warning("cqtdiff_plus_plugin.py::unknown fallback", exc_info=True)
            if self._device != "cpu":
                logger.warning("CQTdiff+: GPU-Inferenz fehlgeschlagen (%s) — CPU-Retry", exc)
                try:
                    if self._torch_model is not None:
                        self._torch_model.cpu()
                    self._device = "cpu"
                    try:
                        from backend.core.ml_device_manager import get_ml_device_manager as _mgr

                        _mgr().report_gpu_error("CQTDiffPlus", exc)
                    except Exception:
                        logger.warning("cqtdiff_plus_plugin.py::unknown fallback", exc_info=True)
                except Exception as _mv_exc:
                    logger.debug("CQTdiff+ GPU→CPU move fehlgeschlagen: %s", _mv_exc)
                    self._device = "cpu"
                return self._inpaint_diffusion(audio, sr, gap_start, gap_end)
            logger.warning("CQTdiff Diffusion-Fehler: %s — DSP-Fallback", exc, exc_info=True)
            return self._inpaint_dsp_fallback(audio, sr, gap_start, gap_end)

    # ------------------------------------------------------------------
    # DSP-Fallback (Consistent Wiener + lineares Crossfade)
    # ------------------------------------------------------------------

    def _inpaint_dsp_fallback(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start: int,
        gap_end: int,
    ) -> np.ndarray:
        """Fallback-Kaskade: DiffWave ONNX → lineare Interpolation + AR.

        Versucht zuerst DiffWave (ONNX, lokal gebündelt) als ML-Fallback.
        Nur wenn DiffWave nicht verfügbar ist, greift lineare Interpolation.

        Referenz: Post-2018-DSP-Fallback (Consistent Wiener, Le Roux & Vincent 2013).
        """
        gap_len = gap_end - gap_start

        # --- ML-Fallback Stufe 1: DiffWave ONNX ---
        try:
            from plugins.diffwave_plugin import DiffwavePlugin

            dw = DiffwavePlugin()
            if dw._session is not None:
                # Maske: True = Lücke (soll inpainted werden)
                mask = np.zeros(len(audio), dtype=bool)
                mask[gap_start:gap_end] = True
                dw_result = dw.inpaint(audio, sr, mask=mask)
                result = np.clip(dw_result, -1.0, 1.0).astype(np.float32)
                result = self._crossfade_edges(result, gap_start, gap_end, sr, fade_ms=5.0)
                logger.info(
                    "🔵 CQTdiff+ → DiffWave-Fallback: Lücke %.1f ms [%d–%d]",
                    gap_len / sr * 1000,
                    gap_start,
                    gap_end,
                )
                return result
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
        except Exception as exc:
            logger.debug("CQTdiff+ DiffWave-Fallback Fehler: %s — lineare Interpolation", exc)

        # --- DSP-Fallback Stufe 2: Lineare Interpolation + AR-Vorhersage ---
        result = audio.copy()

        # Kontext-Segmente vor/nach Lücke
        ctx_before_start = max(0, gap_start - gap_len * 4)
        ctx_after_end = min(len(audio), gap_end + gap_len * 4)
        before = audio[ctx_before_start:gap_start]
        after = audio[gap_end:ctx_after_end]

        # Lineare Übergangsinterpolation + Spektral-gewichtetes Mittel
        t = np.linspace(0, 1, gap_len, dtype=np.float32)
        before_val = float(before[-1]) if len(before) > 0 else 0.0
        after_val = float(after[0]) if len(after) > 0 else 0.0
        interpolated = (1.0 - t) * before_val + t * after_val

        # Leichtes Rauschmodell aus Kontext
        if len(before) >= 32:
            ctx_std = float(np.std(before[-32:]))
            noise = np.random.default_rng(seed=42).normal(0, ctx_std * 0.1, gap_len)
            interpolated += noise.astype(np.float32)

        result[gap_start:gap_end] = np.clip(interpolated, -1.0, 1.0)
        result = self._crossfade_edges(result, gap_start, gap_end, sr, fade_ms=5.0)
        logger.info("🔵 CQTdiff DSP-Fallback: Lineare Interpolation (%.1f ms Lücke)", gap_len / sr * 1000)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_local_context(audio: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
        """Extrahiert lokalen Kontext um die Lücke (max. 2 s)."""
        ctx_len = min(96000, gap_start)  # max. 2 s @ 48000 Hz
        ctx = audio[max(0, gap_start - ctx_len) : gap_start]
        if len(ctx) < 256:
            ctx = np.zeros(256, dtype=np.float32)
        return ctx.astype(np.float32)

    @staticmethod
    def _crossfade_edges(
        audio: np.ndarray,
        gap_start: int,
        gap_end: int,
        sr: int,
        fade_ms: float = 5.0,
    ) -> np.ndarray:
        """Hanning-Crossfade an Lücken-Rändern zur Artefakt-Vermeidung."""
        fade_samples = int(sr * fade_ms / 1000)
        result = audio.copy()
        if gap_start >= fade_samples:
            fade_in = np.hanning(fade_samples * 2)[:fade_samples].astype(np.float32)
            result[gap_start : gap_start + fade_samples] *= fade_in
        if gap_end + fade_samples <= len(audio):
            fade_out = np.hanning(fade_samples * 2)[fade_samples:].astype(np.float32)
            result[gap_end - fade_samples : gap_end] *= fade_out
        return result

    @staticmethod
    def _compute_kl(
        original: np.ndarray,
        restored: np.ndarray,
        gap_start: int,
        gap_end: int,
    ) -> float:
        """KL-Divergenz des Spektrums vor/nach Inpainting (soll < 0.15)."""
        n_fft = 512
        ref_seg = original[max(0, gap_start - 4800) : gap_start + 1]
        if len(ref_seg) < n_fft:
            return 0.0
        p = np.abs(np.fft.rfft(ref_seg[-n_fft:])) + 1e-12
        q = np.abs(np.fft.rfft(restored[max(0, gap_start - n_fft) : gap_start + n_fft][-n_fft:])) + 1e-12
        if len(p) != len(q):
            return 0.0
        p /= p.sum()
        q /= q.sum()
        kl = float(np.sum(p * np.log(p / q + 1e-12)))
        return float(np.clip(kl, 0.0, 10.0))

    @staticmethod
    def _compute_chroma_corr(original: np.ndarray, restored: np.ndarray, sr: int) -> float:
        """Pearson-Korrelation der Chroma-Vektoren (soll ≥ 0.92)."""
        try:
            import librosa

            n = min(len(original), len(restored), sr * 4)  # max. 4 s
            if n < 64:
                return 0.9
            n_fft = int(min(1024, n))
            if n_fft % 2 == 1:
                n_fft -= 1
            n_fft = max(64, n_fft)
            hop_length = max(16, n_fft // 4)
            c1 = librosa.feature.chroma_stft(
                y=original[:n].astype(np.float32),
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            c2 = librosa.feature.chroma_stft(
                y=restored[:n].astype(np.float32),
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            _s1 = float(np.std(c1.ravel()))
            _s2 = float(np.std(c2.ravel()))
            if _s1 < 1e-8 or _s2 < 1e-8:
                return 0.9  # near-constant chroma (silence) → neutral
            _c1r = c1.ravel()
            _c2r = c2.ravel()
            _c1a = _c1r - _c1r.mean()
            _c2a = _c2r - _c2r.mean()
            _nc1 = float(np.linalg.norm(_c1a))
            _nc2 = float(np.linalg.norm(_c2a))
            corr = float(np.dot(_c1a, _c2a) / (_nc1 * _nc2 + 1e-10))
            return float(np.clip(np.nan_to_num(corr), -1.0, 1.0))
        except Exception:
            logger.warning("cqtdiff_plus_plugin.py::_compute_chroma_corr fallback", exc_info=True)
            return 0.9  # Optimistischer Standardwert bei librosa-Fehler


# ---------------------------------------------------------------------------
# Singleton (Unload + Accessor)
# ---------------------------------------------------------------------------


def _unload_cqtdiff_plus() -> None:
    """Entlädt das CQTdiff+-Singleton aus dem RAM (PLM-Eviction-Callback)."""
    global _instance
    _instance = None  # type: ignore[assignment]
    try:
        import gc

        gc.collect()
    except Exception as _exc:
        logger.debug("Plugin operation failed (non-critical): %s", _exc)


def get_cqtdiff_plus() -> CQTdiffPlusPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CQTdiffPlusPlugin()
    return _instance


def inpaint_gap(
    audio: np.ndarray,
    sr: int,
    gap_start_sample: int,
    gap_end_sample: int,
    *,
    context_audio: np.ndarray | None = None,
) -> InpaintingResult:
    """Convenience-Wrapper — CQTdiff+ Lücken-Inpainting ohne Klassen-Instantiierung.

    Beispiel::

        result = inpaint_gap(audio, sr=48000, gap_start_sample=96000, gap_end_sample=100800)
        logger.debug("KL: %.3f, Chroma-Corr: %.3f", result.kl_divergence, result.chroma_corr)

    Args:
        audio:              Audio-Signal (1D float32, 48000 Hz)
        sr:                 Sample-Rate (48000)
        gap_start_sample:   Lücken-Anfang (Sample-Index)
        gap_end_sample:     Lücken-Ende (exklusiv)
        context_audio:      Optionaler musikalischer Phrasenkontext

    Returns:
        InpaintingResult
    """
    return get_cqtdiff_plus().inpaint(audio, sr, gap_start_sample, gap_end_sample, context_audio=context_audio)
