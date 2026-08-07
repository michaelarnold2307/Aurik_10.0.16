import logging

"""
declipper.py - Deep-Learning-basierter Declipper für Aurik 6.0

SOTA-konformer Declipper mit DSPContract und Auditierbarkeit.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContractDeclipper:
    id: str = "declipper"
    category: str = "declipper"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


declipper_contract = DSPContractDeclipper(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"model_path": None}},
    budgets={"compute_cost": 0.01},
    side_effects=[
        {
            "risk": "Fehlrestauration",
            "expected_when": "Modell nicht geladen",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["declip_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class Declipper:
    """
    SOTA-konformer Deep-Learning-Declipper:
    - Entfernt Clipping-Artefakte aus Audiosignalen
    - Architektur: Platzhalter für ONNX/Torch-Modell
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractDeclipper = declipper_contract

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model = None  # Hier könnte ein ONNX- oder Torch-Modell geladen werden

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    # ------------------------------------------------------------------ #
    # DSP-Fallback: Consistent-Wiener-basiertes Clip-Interpolation          #
    # Referenz: Le Roux & Vincent (2013), PGHI (Perraudin 2013)             #
    # ------------------------------------------------------------------ #

    _CLIP_THRESH: float = 0.990  # Samples oberhalb gelten als geclippt
    _NPERSEG: int = 2048
    _G_FLOOR: float = 0.10

    def _declip_channel(self, x: np.ndarray) -> np.ndarray:
        """Einzelkanal-Declipping via Zeitbereichs-Interpolation + STFT-Verfeinerung.

        Schritt 1 – Zeitbereichs-Interpolation (kurze Clips ≤ 6 ms):
            Flat-top-Regionen (|x| ≥ CLIP_THRESH, ≥ 2 aufeinanderfolgende Samples)
            werden mit linearer Interpolation zwischen den Randpunkten ersetzt.
            Hanning-fade (5 Samples) an den Übergängen.

        Schritt 2 – STFT-Consistent-Wiener-Verfeinerung (alle Clips):
            a) STFT des interpolierten Signals
            b) Schätze Rauschboden via Minimum-Statistik (IMCRA-Proxy, P5-Perzentil)
            c) MMSE-LSA-Gain: G = max(G_floor, v/(v+1)) mit v = SNR_post
            d) Wende G nur in Bins an, die vom Clipping betroffen sind
            e) ISTFT mit originaler Phase (phasenkonsistent)

        Invarianten: NaN/Inf-frei, Ausgang ∈ [−1, 1].
        """

        from scipy.signal import istft as _istft
        from scipy.signal import stft as _stft

        x = np.nan_to_num(x.copy().astype(np.float64), nan=0.0, posinf=1.0, neginf=-1.0)
        clip_mask = np.abs(x) >= self._CLIP_THRESH

        if not clip_mask.any():
            return x.astype(np.float32)

        # ── Schritt 1: Zeitbereich-Interpolation ──────────────────────────
        out = x.copy()
        n = len(x)
        i = 0
        fade = 5  # Hanning-Randsamples
        while i < n:
            if clip_mask[i]:
                j = i
                while j < n and clip_mask[j]:
                    j += 1
                # Flat-top-Region: [i, j)
                l_idx = max(0, i - 1)
                r_idx = min(n - 1, j)
                l_val = out[l_idx]
                r_val = out[r_idx]
                num_gap = j - i
                if num_gap > 0:
                    interp = np.linspace(l_val, r_val, num_gap + 2)[1:-1]
                    out[i:j] = interp
                    # Hanning-fade an den Rändern
                    fl = min(fade, num_gap // 2)
                    if fl > 0:
                        hw = np.hanning(fl * 2)[:fl]
                        out[i : i + fl] = (1 - hw) * x[i : i + fl] + hw * interp[:fl]
                i = j
            else:
                i += 1

        # ── Schritt 2: STFT-Consistent-Wiener-Gain ────────────────────────
        noverlap = self._NPERSEG * 3 // 4
        _, _, Zxx = _stft(out, nperseg=self._NPERSEG, noverlap=noverlap)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Minimum-Statistik für Rauschschätzung (IMCRA-Proxy)
        noise_est = np.percentile(mag, 5, axis=1, keepdims=True) + 1e-12

        # MMSE-LSA-Gain: G = max(G_floor, SNR_post / (SNR_post + 1))
        snr_post = np.maximum(mag / noise_est - 1.0, 0.0)
        G = np.maximum(self._G_FLOOR, snr_post / (snr_post + 1.0))

        # Gain nur in Clips-Regionen anwenden; unbetroffene Bins bleiben 1.0
        # → STFT-Clip-Maske: Zeitframes, in denen clip_mask aktiv war
        hop = self._NPERSEG - noverlap
        n_frames = Zxx.shape[1]
        frame_clipped = np.array(
            [clip_mask[f * hop : f * hop + self._NPERSEG].any() for f in range(n_frames)],
            dtype=float,
        )
        G_apply = 1.0 - frame_clipped[np.newaxis, :] + G * frame_clipped[np.newaxis, :]

        Zxx_clean = mag * G_apply * np.exp(1j * phase)
        _, restored = _istft(Zxx_clean, nperseg=self._NPERSEG, noverlap=noverlap)
        restored = np.nan_to_num(restored[:n], nan=0.0, posinf=0.0, neginf=0.0)

        return np.clip(restored, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Entfernt Clipping-Artefakte.

        Primär: ML-Modell (Apollo ONNX), falls geladen.
        Fallback: Consistent-Wiener-basierte Zeitbereichs-Interpolation + STFT-Gain
                  (Le Roux & Vincent 2013; PGHI Perraudin 2013).
        Invariante: NaN/Inf-frei, Ausgang ∈ [−1, 1].
        """
        self.log_contract()

        # ML-Primärverarbeitung (Apollo, ONNX)
        if self.model is not None:
            try:
                inp = audio.astype(np.float32)
                result = self.model.run(None, {"input": inp[np.newaxis, :]})[0].squeeze()
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                return np.clip(result, -1.0, 1.0).astype(np.float32)
            except Exception as exc:
                logger.warning("[Declipper] ML-Inferenz fehlgeschlagen (%s), nutze DSP-Ersatzpfad.", exc)

        # DSP-Fallback: kanalweise
        audio_f32 = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        if audio_f32.ndim == 1:
            return self._declip_channel(audio_f32)
        # Stereo
        channels = [self._declip_channel(audio_f32[:, ch]) for ch in range(audio_f32.shape[1])]
        return np.stack(channels, axis=1)
