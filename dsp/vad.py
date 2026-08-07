"""
ai_vad.py - Adaptive Voice Activity Detection (Deep-Learning) für Aurik 6.0

Dieses Modul stellt ein Deep-Learning-basiertes VAD-Modul bereit und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "ai_vad"
    category: str = "vad"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
ai_vad_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [16000, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"threshold": 0.5},
        "safe_ranges": {"threshold": {"min": 0.1, "max": 0.99}},
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Fehlerkennung",
            "expected_when": "threshold zu niedrig",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["vad_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AiVAD:
    """
    Adaptive Voice Activity Detection (Stub):
    - Erkennt Sprachsegmente im Audiosignal (Deep-Learning)
    """

    def __init__(self, model_path: str | None = None, threshold: float = 0.5):
        self.model_path = model_path
        self.model = None
        self.threshold = threshold

    def log_contract(self):
        import logging

        logging.info("[DSPContract] %s", asdict(ai_vad_contract))

    def detect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Frame-weise Energie + ZCR + Spektrale-Flatness-VAD (ITU-T G.729 Annex B Ansatz).

        Algorithmus:
            1. Mono-Konvertierung + NaN-Schutz
            2. 25 ms Frames, 10 ms Hop
            3. Pro Frame: RMS-Energie, ZCR, Spektrale Flatness (Geometrisches/Arithmetisches Mittel)
            4. Adaptive Schwellwerte aus ersten 50 Hintergrund-Frames (Stille-Prior)
            5. Entscheidung: Sprache wenn RMS > 4× Rauschboden ODER ZCR > 0.08
               UND Spektrale Flatness < Hintergrund-Median×1.2 (tonal = kein Rauschen)
            6. Morphologisches Glätten: Uniform-Filter (180 ms Kernel)
            7. Expansion auf Sample-Auflösung

        Returns:
            np.ndarray: Bool-Array shape=(n_samples,), True=Stimme, False=Stille
        """
        import logging as _log_mod

        from scipy.ndimage import uniform_filter1d

        _log = _log_mod.getLogger(__name__)
        self.log_contract()

        # ── Mono + NaN-Schutz ──────────────────────────────────────────
        mono = audio.mean(axis=0).astype(np.float64) if audio.ndim == 2 else audio.astype(np.float64)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(mono)
        if n == 0:
            return np.zeros(0, dtype=bool)

        # ── Frame-Parameter ────────────────────────────────────────────
        frame_len = max(int(0.025 * sr), 64)  # 25 ms
        hop_len = max(int(0.010 * sr), 16)  # 10 ms
        n_frames = max((n - frame_len) // hop_len + 1, 1)

        # ── Feature-Arrays ─────────────────────────────────────────────
        rms = np.empty(n_frames, dtype=np.float64)
        zcr = np.empty(n_frames, dtype=np.float64)
        sfm = np.empty(n_frames, dtype=np.float64)  # Spektrale Flatness

        window = np.hanning(frame_len)

        for i in range(n_frames):
            s = i * hop_len
            e = min(s + frame_len, n)
            fr = mono[s:e]
            if len(fr) < 4:
                rms[i] = 0.0
                zcr[i] = 0.0
                sfm[i] = 1.0
                continue

            # RMS-Energie
            rms[i] = float(np.sqrt(np.mean(fr**2) + 1e-12))

            # Zero-Crossing-Rate
            signs = np.diff(np.sign(fr))
            zcr[i] = float(np.sum(np.abs(signs)) / (2.0 * max(len(fr), 1)))

            # Spektrale Flatness: exp(mean(log(|STFT|²))) / mean(|STFT|²)
            _w = window[: len(fr)] if len(fr) < frame_len else window
            ps = np.abs(np.fft.rfft(fr * _w)) ** 2 + 1e-12
            log_mean = float(np.exp(np.mean(np.log(ps))))
            arith_mean = float(np.mean(ps))
            sfm[i] = log_mean / (arith_mean + 1e-12)

        # ── Adaptive Schwellwerte aus ersten 50 Stille-Frames ──────────
        n_calib = min(50, n_frames)
        rms_floor = float(np.median(rms[:n_calib])) if n_calib > 0 else 1e-6
        energy_thr = max(rms_floor * 4.0, 1e-5)  # 4× Rauschboden
        sfm_bg = float(np.median(sfm[:n_calib])) if n_calib > 0 else 0.8
        sfm_thr = min(sfm_bg * 1.2, 0.85)  # Tief = tonhaltiger Inhalt

        # ── Frame-Entscheidung ─────────────────────────────────────────
        # Sprache: (Energie HOCH ODER viele Nulldurchgänge [Frikative])
        #           UND spektraler Inhalt tonal (SFM < Schwelle)
        frame_voice = ((rms > energy_thr) | (zcr > 0.08)) & (sfm < sfm_thr)

        # ── Morphologisches Glätten (180 ms Kernel) ────────────────────
        smooth_kernel = max(int(0.180 / 0.010) // 2 * 2 + 1, 3)  # ungerade
        smoothed = uniform_filter1d(frame_voice.astype(np.float32), size=smooth_kernel)
        result = smoothed >= self.threshold

        # ── Expansion auf Sample-Auflösung ─────────────────────────────
        sample_mask = np.zeros(n, dtype=bool)
        for i in range(n_frames):
            s = i * hop_len
            e = min(s + hop_len, n)
            if result[i]:
                sample_mask[s:e] = True

        n_active = int(np.sum(result))
        _log.debug(
            "AiVAD: %d/%d Frames aktiv (%.1f%%), energy_thr=%.2e",
            n_active,
            n_frames,
            100.0 * n_active / max(n_frames, 1),
            energy_thr,
        )
        return sample_mask
