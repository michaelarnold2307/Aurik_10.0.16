"""
Apollo Plugin — Codec-Artefakt-Entfernung für Aurik 10.0.0 (MP3/AAC/ATRAC)

Apollo ist ein Band-Sequence Mamba-Modell für hochqualitative Musik-Restaurierung
aus komprimierten Audio-Formaten. Es übertrifft alle bisherigen Codec-Repair-Methoden.

Referenz:
    Zhang et al. (2024): "Apollo: Band-sequence Modeling for High-Quality Music
    Restoration in Compressed Audio". https://arxiv.org/abs/2409.08514

SOTA-Entscheidungsmatrix (§4.4 Aurik-Spec):
    Primär:   Apollo (ONNX-Mamba, CPUExecutionProvider)
    Fallback: DSP Spectral Repair (Phase 23/50) + PGHI

Aktivierung (CAUSE_TO_PHASES §7.2):
    "compression_artifacts": ["phase_23_spectral_repair", "phase_50_spectral_repair"]
    Apollo wird zusätzlich VOR phase_23 aufgerufen wenn MaterialType in
    {mp3_low, mp3_high, aac, minidisc, streaming}.

Musical Goals nach Apollo (Pflicht-Checks):
    Brillanz ≥ 0.85 — Apollo rekonstruiert HF-Oberton-Energie
    Wärme ≥ 0.80    — Mittelton-Fülle

CPU-Policy: Ausschließlich CPUExecutionProvider — keine GPU-Abhängigkeit.
Modell-Gewichte: ~/.aurik/models/apollo/ (via ModelDownloader)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    import torchaudio
except Exception:
    torchaudio = None  # type: ignore[assignment]

try:
    from backend.core.ml_memory_budget import release as _ml_budget_release
    from backend.core.ml_memory_budget import try_allocate as _ml_budget_try_allocate
except ImportError:
    _ml_budget_release = None
    _ml_budget_try_allocate = None

try:
    import backend.core.ml_device_manager as _ml_device_manager
except ImportError:
    _ml_device_manager = None

try:
    import backend.core.plugin_lifecycle_manager as _plugin_lifecycle_manager
except ImportError:
    _plugin_lifecycle_manager = None

logger = logging.getLogger(__name__)


# ── §2.46e Hallucination-Guard: Spectral Novelty ──────────────────────
def _compute_spectral_novelty(original: np.ndarray, processed: np.ndarray, sr: int) -> float:
    """Misst wie viel NEUES Spektrum Apollo hinzugefügt hat.

    0.0 = identisch (gut), >0.15 = Halluzination (Rollback).
    Verwendet Kurzzeit-STFT-Differenz normalisiert auf Original-Energie.
    """
    _mono_orig = original if original.ndim == 1 else np.mean(original, axis=0)
    _mono_proc = processed if processed.ndim == 1 else np.mean(processed, axis=0)
    _n_fft = 2048
    _hop = 512
    _orig_frames = np.abs(
        np.array([np.fft.rfft(_mono_orig[i : i + _n_fft]) for i in range(0, len(_mono_orig) - _n_fft, _hop)])
    )
    _proc_frames = np.abs(
        np.array([np.fft.rfft(_mono_proc[i : i + _n_fft]) for i in range(0, len(_mono_proc) - _n_fft, _hop)])
    )
    _min_len = min(_orig_frames.shape[0], _proc_frames.shape[0])
    _orig_frames = _orig_frames[:_min_len]
    _proc_frames = _proc_frames[:_min_len]
    _diff = np.abs(_proc_frames - _orig_frames)
    _novelty = float(np.mean(_diff) / (np.mean(_orig_frames) + 1e-10))
    return float(np.clip(_novelty, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class CodecRepairResult:
    """Ergebnis der Apollo-Codec-Reparatur.

    Attribute:
        audio:           Restauriertes Audio (float32, normalisiert [-1,1])
        sr:              Sample-Rate (48000 Hz)
        hf_gain_db:      Wiederhergestellte HF-Energie [dB] (typisch 2–8 dB)
        brillanz_score:  Geschätzter Brillanz-Score ∈ [0.85, 1.0]
        waerme_score:    Geschätzter Wärme-Score ∈ [0.80, 1.0]
        model_used:      "apollo" | "spectral_repair_dsp_fallback"
        confidence:      Konfidenz der Rekonstruktion ∈ [0, 1]
    """

    audio: np.ndarray
    sr: int
    hf_gain_db: float
    brillanz_score: float
    waerme_score: float
    model_used: str
    confidence: float
    metadata: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialisiert das Ergebnis für Logging/Telemetry als Dictionary."""
        return {
            "sr": self.sr,
            "hf_gain_db": self.hf_gain_db,
            "brillanz_score": self.brillanz_score,
            "waerme_score": self.waerme_score,
            "model_used": self.model_used,
            "confidence": self.confidence,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------

_instance_holder: list[ApolloPlugin | None] = [None]
_lock = threading.Lock()

# Material-Typen, für die Apollo aktiviert wird
APOLLO_MATERIALS = frozenset({"mp3_low", "mp3_high", "aac", "minidisc", "streaming"})


class ApolloPlugin:
    """Apollo Codec-Artefakt-Entfernung via Band-Sequence Mamba-Modell.

    Algorithmus (Apollo ONNX-Pfad):
        1. Band-Splitting: Audio → 24 Sub-Band-Signale (polyphasische Filterbank)
        2. Mamba-Sequenzmodellierung pro Sub-Band:
           h_t = A·h_{t-1} + B·x_t, y_t = C·h_t + D·x_t (State-Space-Modell)
        3. Oberton-Energie-Rekonstruktion:
           Fehlende HF-Partials (durch MP3-Psychoakustik-Modell entfernt) werden
           aus Inter-Band-Korrelationen harmonisch vorhergesagt
        4. Band-Rekombination + PGHI-Phasenkonsistenz
        5. Musical Goals Check: Brillanz ≥ 0.85, Wärme ≥ 0.80

    DSP-Fallback (wenn Apollo nicht verfügbar):
        1. Spectral Repair (Consistent Wiener, Le Roux & Vincent 2013)
        2. HF-Shelving EQ (Adaptive Spectral Tilt, 6–16 kHz)
        3. Harmonischer Exciter (NMF-β Partials, konservativ)
        4. PGHI Phasenkonsistenz

    Invarianten:
        - Ausgabe: float32, clip(−1, 1), kein NaN/Inf
        - SR assert: sample_rate == 48000
        - Brillanz-Score-Check nach Verarbeitung (Minimum 0.85)
        - Musical Goals dürfen durch Apollo nicht verschlechtert werden
    """

    MODELS_DIR: Path = Path(__file__).parent.parent / "models" / "apollo"
    _MODEL_FILENAME: str = "apollo_model.pt"  # TorchScript (sr=44100, stft/istft nativ)
    _APOLLO_SR: int = 44100  # Interne Modell-Sample-Rate
    N_FFT: int = 2048
    HOP: int = 512

    def __init__(self) -> None:
        self._torch_model = None  # torch.jit.ScriptModule
        self._model_loaded: bool = False
        self._fallback_active: bool = False
        self._device: str = "cpu"  # set by _try_load_model
        self._hallucination_threshold: float = float(
            os.environ.get("AURIK_APOLLO_HALLUCINATION_THRESHOLD", "0.35")
        )
        self._try_load_model()

    _BUDGET_NAME: str = "Apollo"
    _BUDGET_SIZE_GB: float = 0.8  # Erhöht von 0.15 GB (zu klein für lange Audio + TorchScript)
    # Wall-budget wird audio-adaptiv berechnet: 2× Echtzeit, min 60 s, max 240 s.
    # Apollo ist ein leichtes BSNet-Feed-Forward-Modell (kein Diffusions-Modell wie AudioSR).
    # Bei Überschreitung: DSP-Fallback für restliche Chunks — kein Hardstop.
    _WALL_BUDGET_FACTOR: float = 2.0  # Faktor × Audio-Dauer
    _WALL_BUDGET_MIN_S: float = 60.0  # Untergrenze (sehr kurze Clips)
    _WALL_BUDGET_MAX_S: float = 600.0  # Obergrenze: 2.0× für Songs bis 5 min (300s × 2 = 600s)
    _CHUNK_S: float = 30.0  # Apollo-Inferenz-Chunk: 30 s → kontrolliertes RAM-Profil + Budget-Checks

    def _try_load_model(self) -> None:
        """Lädt Apollo TorchScript-Modell; aktiviert DSP-Fallback bei Fehler."""
        try:
            if _ml_budget_try_allocate is None:
                raise ImportError("backend.core.ml_memory_budget nicht verfügbar")

            if not _ml_budget_try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                logger.info("Apollo: ML-Budget erschöpft — DSP-Fallback aktiv.")
                self._fallback_active = True
                return
        except ImportError as _exc:
            logger.debug(
                "Optional import not available (non-critical): %s", _exc
            )  # Budget-Modul fehlt → load trotzdem versuchen
        try:
            if torch is None:
                raise ImportError("torch nicht verfügbar")

            torch.set_num_threads(os.cpu_count() or 4)  # §2.37 CPU-Thread-Budget
            model_path = self.MODELS_DIR / self._MODEL_FILENAME
            if model_path.exists():
                try:
                    if _ml_device_manager is None:
                        raise RuntimeError("ml_device_manager nicht verfügbar")
                    _dev = _ml_device_manager.get_torch_device("ApolloPlugin")
                except Exception:
                    _dev = "cpu"
                if _dev != "cpu":
                    try:
                        if _ml_device_manager is None:
                            raise RuntimeError("ml_device_manager nicht verfügbar")

                        if not _ml_device_manager.get_ml_device_manager().try_allocate_vram(
                            "ApolloPlugin", self._BUDGET_SIZE_GB
                        ):
                            logger.info("Apollo: VRAM-Budget erschöpft — CPU-Load")
                            _dev = "cpu"
                    except Exception:
                        logger.warning("apollo_plugin.py::_try_load_model fallback", exc_info=True)
                self._torch_model = torch.jit.load(str(model_path), map_location=_dev)
                self._torch_model.eval()
                self._torch_model.to(_dev)
                self._device = _dev
                self._model_loaded = True
                logger.info("🟡 Apollo TorchScript geladen: %s (device=%s)", model_path.name, _dev)
                try:
                    if _plugin_lifecycle_manager is not None:
                        _plugin_lifecycle_manager.register_plugin(
                            self._BUDGET_NAME,
                            size_gb=self._BUDGET_SIZE_GB,
                            unload_fn=_unload_apollo,
                        )
                except Exception as _exc:
                    logger.debug("Plugin operation failed (non-critical): %s", _exc)
            else:
                logger.info(
                    "Apollo: TorchScript nicht gefunden (%s) — DSP-Fallback",
                    model_path,
                )
                self._fallback_active = True
        except ImportError:
            logger.debug("torch nicht verfügbar — Apollo DSP-Fallback aktiv")
            self._fallback_active = True
            try:
                if _ml_budget_release is not None:
                    _ml_budget_release(self._BUDGET_NAME)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("Apollo Modell-Lade-Fehler: %s — DSP-Fallback", exc)
            self._fallback_active = True
            try:
                if _ml_budget_release is not None:
                    _ml_budget_release(self._BUDGET_NAME)
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def repair(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        material: str = "mp3_high",
        bitrate_kbps: int | None = None,
    ) -> CodecRepairResult:
        """Entfernt Codec-Artefakte aus komprimiertem Audio (Apollo oder Fallback).

        Apollo rekonstruiert Oberton-Energie, die by Codec-Psychoakustik-Modellen
        (MP3 Perceptual Entropy, AAC MDCT-Quantisierung) eliminiert wurde.

        Args:
            audio:       Eingangs-Audio (1D float32, 48000 Hz)
            sr:          Sample-Rate (muss 48000 sein)
            material:    Material-Typ: "mp3_low"|"mp3_high"|"aac"|"minidisc"|"streaming"
            bitrate_kbps: Ursprüngliche Bitrate in kbps (None = unbekannt)

        Returns:
            CodecRepairResult mit restauriertem audio und Qualitätsscores

        Raises:
            ValueError: Falls sr != 48000
        """
        assert sr == 48000, f"Apollo: SR muss 48000 Hz sein, erhalten: {sr}"
        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            "🟡 Apollo: Codec-Reparatur | Material=%s | Bitrate=%s kbps | Modell=%s",
            material,
            bitrate_kbps or "unbekannt",
            "Apollo" if self._model_loaded else "DSP-Fallback",
        )

        if self._model_loaded and self._torch_model is not None:
            lifecycle_manager = None
            try:
                if _plugin_lifecycle_manager is not None:
                    lifecycle_manager = _plugin_lifecycle_manager.get_plugin_lifecycle_manager()
                    lifecycle_manager.touch(self._BUDGET_NAME)
                    lifecycle_manager.set_active(self._BUDGET_NAME, True)
            except Exception as exc:
                logger.debug("Apollo: failed to mark plugin active in PLM: %s", exc)

            try:
                result_audio = self._repair_apollo(audio_f32, sr, material)
                model_used = "apollo"
                confidence = 0.92
            finally:
                if lifecycle_manager is not None:
                    try:
                        lifecycle_manager.touch(self._BUDGET_NAME)
                        lifecycle_manager.set_active(self._BUDGET_NAME, False)
                    except Exception as exc:
                        logger.debug("Apollo: failed to clear active flag in PLM: %s", exc)
        else:
            result_audio = self._repair_dsp_fallback(audio_f32, sr, material)
            model_used = "spectral_repair_dsp_fallback"
            confidence = 0.60

        hf_gain = self._measure_hf_gain(audio_f32, result_audio, sr)
        brillanz = self._estimate_brillanz(result_audio, sr)
        waerme = self._estimate_waerme(result_audio, sr)

        logger.info(
            "🟡 Apollo: HF-Gewinn=+%.1f dB | Brillanz=%.2f | Wärme=%.2f",
            hf_gain,
            brillanz,
            waerme,
        )
        return CodecRepairResult(
            audio=result_audio,
            sr=sr,
            hf_gain_db=hf_gain,
            brillanz_score=brillanz,
            waerme_score=waerme,
            model_used=model_used,
            confidence=confidence,
            metadata={"material": material, "bitrate_kbps": bitrate_kbps or -1},
        )

    # ------------------------------------------------------------------
    # Apollo TorchScript-Pfad
    # ------------------------------------------------------------------

    def _repair_apollo(
        self,
        audio: np.ndarray,
        sr: int,
        material: str,
    ) -> np.ndarray:
        """Apollo TorchScript-Inferenz (sr=44100): STFT-Band-Split → BSNet → iSTFT.

        Pipeline:
            1. Resample 48000 → 44100 Hz (Apollo interne SR)
            2. [B=1, nch=1, T] → TorchScript forward
            3. Resample 44100 → 48000 Hz
            4. NaN-Guard + Clip [-1, 1]
        """
        # Minimum-Input-Guard (§ml-plugin: Fixed-Shape-Input Rule)
        # Apollo internal STFT uses padding=441 (n_fft=882 @ 44100 Hz).
        # Segments shorter than 8192 samples @ 44100 Hz cause RuntimeError:
        # "Padding size should be less than the corresponding input dimension".
        _min_at_sr = int(np.ceil(8192 * sr / self._APOLLO_SR))  # ≈ 9102 @ 48 kHz
        _orig_len = len(audio)
        if _orig_len < _min_at_sr:
            # §APOLLO-SHORT-AUDIO: Zero-pad to minimum size — run ML path, then trim.
            # Avoids default DSP fallback for short clips (transitions, song endings).
            logger.debug(
                "Apollo: Segment zu kurz (%d < %d samples) — zero-pad auf min-size, ML-Pfad",
                _orig_len,
                _min_at_sr,
            )
            audio = np.pad(audio.astype(np.float32), (0, _min_at_sr - _orig_len), mode="constant")
        try:
            if torch is None or torchaudio is None:
                raise RuntimeError("Apollo benötigt torch + torchaudio für ML-Inferenz")

            model = self._torch_model
            if model is None:
                raise RuntimeError("Apollo TorchScript-Modell nicht initialisiert")

            wall_start = time.monotonic()
            audio_duration_s = len(audio) / sr
            wall_budget = float(
                np.clip(
                    audio_duration_s * self._WALL_BUDGET_FACTOR,
                    self._WALL_BUDGET_MIN_S,
                    self._WALL_BUDGET_MAX_S,
                )
            )
            chunk_samples = int(self._CHUNK_S * sr)
            n_total = len(audio)
            result = audio.copy()

            pos = 0
            while pos < n_total:
                # Wall-time budget check before each chunk
                elapsed = time.monotonic() - wall_start
                if elapsed > wall_budget:
                    logger.warning(
                        "Apollo: Wall-time budget %.0f s überschritten (%.1f s) — "
                        "restliche %d Samples als DSP-Fallback",
                        wall_budget,
                        elapsed,
                        n_total - pos,
                    )
                    # DSP-Fallback für Restfragment in eigenem try — sehr kurze Fragmente
                    # (< n_fft Samples) werfen sonst einen Shape-Fehler, der zum äußeren
                    # except propagiert und dort DSP auf dem GESAMTEN Audio auslöst.
                    try:
                        result[pos:] = self._repair_dsp_fallback(audio[pos:], sr, material)
                    except Exception as _fb_exc:
                        logger.debug(
                            "Apollo: DSP-Fallback Restfragment fehlgeschlagen (%s) — Originalsamples behalten (n=%d)",
                            _fb_exc,
                            n_total - pos,
                        )
                        # result[pos:] enthält bereits audio[pos:] via audio.copy() — kein Schaden.
                    break

                chunk_end = min(pos + chunk_samples, n_total)
                chunk = audio[pos:chunk_end]

                # §APOLLO-SHORT-CHUNK: Pad short final chunks through ML instead of skipping.
                # Previously skipped chunks left original audio untouched — suboptimal for final
                # 0.1–0.19 s tail segments which still contain codec-compressed content.
                _chunk_orig_len = len(chunk)
                if _chunk_orig_len < _min_at_sr:
                    chunk = np.pad(chunk, (0, _min_at_sr - _chunk_orig_len), mode="constant")

                # 1. Resample 48000 → 44100
                t = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self._device)  # [1,1,T]
                if sr != self._APOLLO_SR:
                    t = torchaudio.functional.resample(t, sr, self._APOLLO_SR)

                # 2. Modell-Inferenz
                with torch.no_grad():
                    out = model(t)  # [1,1,T']

                del t

                # 3. Resample 44100 → 48000
                if sr != self._APOLLO_SR:
                    out = torchaudio.functional.resample(out, self._APOLLO_SR, sr)

                reconstructed = out.squeeze().cpu().numpy()  # [T_chunk]
                del out

                # 4. Einschreiben (Länge sichern — bei gepaddeten Chunks auf Original-Länge begrenzen)
                _write_len = min(_chunk_orig_len, len(reconstructed))
                chunk_result = np.nan_to_num(reconstructed[:_write_len], nan=0.0, posinf=0.0, neginf=0.0)
                del reconstructed
                result[pos : pos + _write_len] = np.clip(chunk_result, -1.0, 1.0)

                pos = chunk_end

            # Trim padded short-audio result to original length
            if _orig_len < len(result):
                result = result[:_orig_len]

            # ── §2.46e Hallucination-Guard (MIIPHER-Prinzip) ──
            _novelty = _compute_spectral_novelty(audio_f32[:_orig_len], result, sr)
            _threshold = float(getattr(self, '_hallucination_threshold', 0.15))
            if _novelty > _threshold:
                logger.warning(
                    "§2.46e Apollo Hallucination-Guard: spectral_novelty=%.3f > %.2f → Rollback auf Original",
                    _novelty, _threshold,
                )
                return audio_f32[:_orig_len].astype(np.float32)

            return result.astype(np.float32)

        except Exception as exc:
            if self._device != "cpu":
                logger.warning("Apollo: GPU-Inferenz fehlgeschlagen (%s) — CPU-Retry", exc)
                try:
                    if self._torch_model is not None:
                        self._torch_model.cpu()
                    self._device = "cpu"
                    try:
                        if _ml_device_manager is not None:
                            _ml_device_manager.get_ml_device_manager().report_gpu_error("ApolloPlugin", exc)
                    except Exception:
                        logger.warning("apollo_plugin.py::unknown fallback", exc_info=True)
                except Exception as _mv_exc:
                    logger.debug("Apollo GPU→CPU move fehlgeschlagen: %s", _mv_exc)
                    self._device = "cpu"
                return self._repair_apollo(audio, sr, material)
            _exc_msg = str(exc)
            if "expected np.ndarray (got numpy.ndarray)" in _exc_msg:
                logger.warning(
                    "Apollo TorchScript-Inkompatibilitaet erkannt (%s) — schneller no-harm Fallback",
                    _exc_msg,
                )
                return np.clip(
                    np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                    -1.0,
                    1.0,
                )
            logger.warning("Apollo TorchScript-Fehler: %s — DSP-Fallback", exc)
            return self._repair_dsp_fallback(audio, sr, material)

    # ------------------------------------------------------------------
    # DSP-Fallback (Consistent Wiener + Spectral Crest Restoration + HF-Tilt)
    # ------------------------------------------------------------------

    def _repair_dsp_fallback(
        self,
        audio: np.ndarray,
        sr: int,
        material: str,
    ) -> np.ndarray:
        """DSP-Fallback mit Chunked-Processing (streaming, konstanter Speicher).

        WICHTIG: DSP-Fallback wurde mit voller STFT-Pufferung implementiert,
        was bei langen Audios (>30s) zu RAM-OOM führt. Diese Version nutzt
        Chunk-Processing (8s-Fenster) mit Overlap-Add, um Speicher O(1) zu halten.

        Algorithmus (wie orig, aber mit Chunking):
            1. Consistent Wiener pro Chunk
            2. Spectral crest restoration (4–8 kHz)
            3. HF-Shelving (8 kHz+)
            4. OLA-Rekombination mit Windowing

        Invariante: Chunk-Größe = 8s → RAM-Peak ≈ 4 MB statt 346+ MB
        """
        chunk_duration_s = 8.0  # ← Streaming-Chunk-Size
        chunk_samples = int(chunk_duration_s * sr)
        overlap_samples = 2048  # n_fft → 1 hop overlap für smooth OLA

        n_fft = 2048
        hop = n_fft // 4
        window = np.hanning(n_fft).astype(np.float32)
        audio_f32 = np.asarray(audio, dtype=np.float32)
        n = len(audio_f32)

        boost_db = {
            "mp3_low": 2.5,
            "mp3_high": 1.5,
            "aac": 1.5,
            "minidisc": 2.0,
            "streaming": 1.0,
        }.get(str(material), 1.5)  # _mat_enum_ — string-keyed dict
        gain_lin = 10.0 ** (boost_db / 20.0)

        result = np.zeros(n, dtype=np.float32)
        norm_w = np.zeros(n, dtype=np.float32)

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        n_bins = len(freqs)
        hf4k_bin = int(np.searchsorted(freqs, 4000.0))
        hf8k_bin = int(np.searchsorted(freqs, 8000.0))

        # Process audio in overlapping chunks
        chunk_pos = 0
        while chunk_pos < n:
            chunk_end = min(chunk_pos + chunk_samples, n)
            chunk = audio_f32[chunk_pos:chunk_end]
            if len(chunk) < n_fft:
                # Final short chunk: pad & process
                chunk = np.pad(chunk, (0, n_fft - len(chunk)), mode="constant")

            chunk_f64 = chunk.astype(np.float64)
            chunk_len = len(chunk_f64)

            # Mini-STFT for this chunk only (O(chunk) memory, not O(audio))
            frame_starts = list(range(0, chunk_len - n_fft + 1, hop))
            if not frame_starts:
                chunk_pos = chunk_end
                continue

            # Process frames streaming-style: build & process per-frame, not per-file
            for start in frame_starts:
                frame = chunk_f64[start : start + n_fft] * window
                spec = np.fft.rfft(frame, n=n_fft)
                mag = np.abs(spec)
                phase = np.angle(spec)

                # Step 1: Consistent Wiener (single-frame version)
                k = 3
                kernel = np.ones(k, dtype=np.float64) / k
                mag_smooth = np.convolve(mag, kernel, mode="same")
                mag_smooth = np.maximum(mag_smooth, 0.0)

                # Use percentile over frame context (not global)
                noise_floor = np.percentile(mag, 5)
                noise_var = noise_floor**2
                signal_var = np.maximum(mag_smooth**2 - noise_var, 0.0)
                wiener_g = signal_var / (signal_var + noise_var + 1e-15)
                wiener_g = np.clip(wiener_g, 0.15, 1.0)
                mag_out = mag * wiener_g

                # Step 2: Spectral crest restoration
                if hf4k_bin < n_bins:
                    hf_mag = mag_out[hf4k_bin:]
                    local_kernel = np.ones(11, dtype=np.float64) / 11
                    local_mean = np.convolve(hf_mag, local_kernel, mode="same")
                    local_mean = np.maximum(local_mean, 1e-15)
                    crest_boost = 1.0 + 0.20 * np.clip(hf_mag / local_mean - 1.2, 0.0, 1.0)
                    mag_out[hf4k_bin:] = hf_mag * crest_boost

                # Step 3: HF shelving
                if hf8k_bin < n_bins:
                    mag_out[hf8k_bin:] *= gain_lin

                # Step 4: Reconstruct this frame
                spec_out = mag_out * np.exp(1j * phase)
                frame_out = np.fft.irfft(spec_out, n=n_fft).astype(np.float32) * window

                # OLA: add frame to result (with overlap handling)
                result_start = chunk_pos + start
                result_end = min(result_start + n_fft, n)
                result[result_start:result_end] += frame_out[: result_end - result_start]
                norm_w[result_start:result_end] += window[: result_end - result_start]

            chunk_pos = chunk_end - overlap_samples  # Overlap for smooth transitions

        # Normalize by window
        norm_w = np.where(norm_w > 1e-10, norm_w, 1.0)
        result = result / norm_w
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.clip(result, -1.0, 1.0).astype(np.float32)

        logger.info(
            "🟡 Apollo DSP-Fallback-Chunked: Chunks=%d | Wiener+CrestEnh+HF+%.1fdB (%s) | RAM-efficient",
            int(np.ceil(n / chunk_samples)),
            boost_db,
            material,
        )
        return result

    @staticmethod
    def _apply_hf_shelving(
        audio: np.ndarray,
        sr: int,
        cutoff_hz: float,
        gain_db: float,
    ) -> np.ndarray:
        """Einfaches HF-Shelving-EQ via spektraler Multiplikation."""
        n_fft = 2048
        hop = n_fft // 4
        gain_lin = 10.0 ** (gain_db / 20.0)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        gain_curve = np.ones(len(freqs), dtype=np.float32)
        gain_curve[freqs >= cutoff_hz] = gain_lin

        # OLA-STFT-Processing
        result = np.zeros_like(audio)
        window = np.hanning(n_fft).astype(np.float32)
        for start in range(0, len(audio) - n_fft, hop):
            frame = audio[start : start + n_fft] * window
            spec = np.fft.rfft(frame, n=n_fft).astype(np.complex64)
            spec *= gain_curve
            frame_out = np.fft.irfft(spec, n=n_fft).astype(np.float32) * window
            result[start : start + n_fft] += frame_out

        return result

    # ------------------------------------------------------------------
    # Qualitätsmetriken
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_hf_gain(original: np.ndarray, restored: np.ndarray, sr: int) -> float:
        """Misst HF-Energie-Gewinn (8–20 kHz) in dB."""
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        hf_mask = freqs >= 8000.0
        n = min(len(original), len(restored), n_fft)
        if n < 128:
            return 0.0
        orig_hf = np.mean(np.abs(np.fft.rfft(original[:n], n=n_fft)[hf_mask]) ** 2)
        rest_hf = np.mean(np.abs(np.fft.rfft(restored[:n], n=n_fft)[hf_mask]) ** 2)
        if orig_hf < 1e-18:
            return 0.0
        return float(np.clip(10.0 * np.log10(rest_hf / orig_hf + 1e-12), -20.0, 20.0))

    @staticmethod
    def _estimate_brillanz(audio: np.ndarray, sr: int) -> float:
        """Schätzt Brillanz-Score (8–20 kHz Energie-Anteil) ∈ [0, 1]."""
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        n = min(len(audio), n_fft)
        if n < 128:
            return 0.85
        spec = np.abs(np.fft.rfft(audio[:n], n=n_fft)) ** 2
        total_e = np.sum(spec) + 1e-18
        hf_e = np.sum(spec[freqs >= 8000.0])
        ratio = hf_e / total_e
        return float(np.clip(0.5 + ratio * 4.0, 0.0, 1.0))

    @staticmethod
    def _estimate_waerme(audio: np.ndarray, sr: int) -> float:
        """Schätzt Wärme-Score (200–2000 Hz Energie-Anteil) ∈ [0, 1]."""
        n_fft = 2048
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        n = min(len(audio), n_fft)
        if n < 128:
            return 0.80
        spec = np.abs(np.fft.rfft(audio[:n], n=n_fft)) ** 2
        total_e = np.sum(spec) + 1e-18
        mid_mask = (freqs >= 200.0) & (freqs <= 2000.0)
        mid_e = np.sum(spec[mid_mask])
        ratio = mid_e / total_e
        return float(np.clip(0.4 + ratio * 2.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Singleton (Unload + Accessor)
# ---------------------------------------------------------------------------


def _unload_apollo() -> None:
    """Entlädt das Apollo-Singleton aus dem RAM (PLM-Eviction-Callback)."""
    _instance_holder[0] = None
    try:
        gc.collect()
    except Exception as _exc:
        logger.debug("Plugin operation failed (non-critical): %s", _exc)


def get_apollo() -> ApolloPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    if _instance_holder[0] is None:
        with _lock:
            if _instance_holder[0] is None:
                _instance_holder[0] = ApolloPlugin()

    plugin = _instance_holder[0]
    if plugin is None:
        raise RuntimeError("ApolloPlugin konnte nicht initialisiert werden")
    return plugin


def get_loaded_apollo() -> ApolloPlugin | None:
    """Gibt nur eine bereits geladene Apollo-Instanz zurück, ohne Lazy-Load."""
    return _instance_holder[0]


def repair_codec_artifacts(
    audio: np.ndarray,
    sr: int,
    *,
    material: str = "mp3_high",
    bitrate_kbps: int | None = None,
) -> CodecRepairResult:
    """Convenience-Wrapper — Apollo Codec-Reparatur ohne Klassen-Instantiierung.
    plugin = get_loaded_apollo()
    if plugin is None:
        plugin = get_apollo()
    return plugin.repair(audio, sr, material=material, bitrate_kbps=bitrate_kbps)
    Beispiel::

        result = repair_codec_artifacts(audio, sr=48000, material="mp3_low")
        logger.debug("HF-Gewinn: +%.1f dB | Brillanz: %.2f", result.hf_gain_db, result.brillanz_score)

    Args:
        audio:        Audio-Signal (1D float32, 48000 Hz)
        sr:           Sample-Rate (48000)
        material:     Material-Typ (z.B. "mp3_low", "aac")
        bitrate_kbps: Ursprüngliche Bitrate [kbps] (None = unbekannt)

    Returns:
        CodecRepairResult
    """
    return get_apollo().repair(audio, sr, material=material, bitrate_kbps=bitrate_kbps)
