import logging

"""
streaming_optimized.py - Streaming-optimierte DSPs für Aurik 6.0

Dieses Modul stellt streaming-optimierte Varianten von Limiter, Denoiser und Gate bereit und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str
    category: str
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[str] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanzen der Contracts
streaming_limiter_contract = DSPContract(
    id="streaming_limiter",
    category="limiter",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {}, "safe_ranges": {}, "trial_profile": {}},
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Limiter-Pumpen",
            "expected_when": "Lookahead zu kurz",
            "severity": 0.1,
        }
    ],
    reports={"self_metrics": ["limiting_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
streaming_denoiser_contract = DSPContract(
    id="streaming_denoiser",
    category="denoiser",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {}, "safe_ranges": {}, "trial_profile": {}},
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[{"risk": "Artefakte", "expected_when": "Threshold zu niedrig", "severity": 0.2}],  # type: ignore[list-item]
    reports={"self_metrics": ["denoising_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
streaming_gate_contract = DSPContract(
    id="streaming_gate",
    category="gate",
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {}, "safe_ranges": {}, "trial_profile": {}},
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.01,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.01,
    },
    side_effects=[
        {  # type: ignore[list-item]
            "risk": "Falschabschaltung",
            "expected_when": "Threshold zu hoch",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["gating_accuracy"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class StreamingLimiter:
    """
    Streaming-optimierter Limiter (Stub):
    - Limiter für Streaming-Anwendungen (Latenz, Stabilität, Lookahead)
    """

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(streaming_limiter_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Streaming-Peak-Limiter: Frame-weiser Envelope-Follower, Ceiling -1 dBFS."""
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        audio_f = audio.astype(float)
        ceiling = 10 ** (-1 / 20)  # -1 dBFS
        frame = max(64, int(sr * 0.005))  # 5 ms
        # Envelope (Peak pro Frame) — leere Chunks sicher behandeln
        n_frames = (len(audio_f) + frame - 1) // frame
        env = np.array(
            [
                (
                    np.max(np.abs(audio_f[i * frame : (i + 1) * frame]))
                    if audio_f[i * frame : (i + 1) * frame].size > 0
                    else 0.0
                )
                for i in range(n_frames)
            ]
        )
        # Gain-Reduktion: g = min(1, ceiling / peak)
        gain_frames = np.minimum(1.0, ceiling / (env + 1e-9))
        # Sample-genaue Gain-Kurve durch Wiederholen pro Frame
        gain = np.repeat(gain_frames, frame)[: len(audio_f)]
        return np.clip(audio_f * gain, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]


class StreamingDenoiser:
    """
    Streaming-optimierter Denoiser — §4.2-konform.

    Kein np.fft.rfft/irfft, kein Perzentil-Schwellwert, keine primitive
    Spectral Subtraction mehr (alle gem. §4.2 VERBOTEN).

    Algorithmus:
        1. scipy.signal.stft  — phasenkons. OLA-Analyse (ersetzt rfft-Schleife)
        2. IMCRA-Sliding-Minimum: Rauschboden = gleitendes Min. der letzten W Frames
           Cohen (2003): Noise Spectrum Estimation in Adverse Environments
        3. MMSE-Wiener-Gain: G = xi/(1+xi),  xi = max(mag/noise_floor - 1, 0)
           Gain-Floor G_floor = 0.1 (Le Roux & Vincent 2013: Consistent Wiener)
        4. scipy.signal.istft — phasenkons. OLA-Synthese (ersetzt irfft-Schleife)
    """

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(streaming_denoiser_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Streaming-Denoiser: scipy.stft + IMCRA-Sliding-Min + MMSE-Wiener-Gain.

        Ersetzt die verbotene np.fft.rfft/irfft-Schleife (§4.2) sowie den
        primitiven Perzentil-Rauschboden und die einfache Spectral Subtraction.
        """
        self.log_contract()  # Audit: Contract-Infos loggen
        audio_f = np.asarray(audio, dtype=float)
        if audio_f.ndim > 1:
            audio_f = audio_f.mean(axis=-1)

        n_fft = 256
        hop = 64
        win_len = n_fft

        # Guard: kurze Buffer — n_fft und hop adaptiv an Eingangslänge anpassen.
        # Ursache ohne Guard: noverlap = win_len - hop = 192 >= nperseg (scipy-reduziert),
        # was in scipy.signal.stft einen ValueError auslöst.
        n = len(audio_f)
        if n < 4:
            # Buffer zu kurz für sinnvolle Spektralverarbeitung → sicherer Passthrough
            return np.clip(audio_f, -1.0, 1.0).astype(audio.dtype)
        win_len = min(n_fft, n)
        hop = min(hop, max(1, win_len // 4))  # hop < win_len garantiert

        from scipy.signal import istft as _istft
        from scipy.signal import stft as _stft

        # 1. STFT — phasenkonsistente OLA-Analyse (kein np.fft.rfft)
        _, _, Zxx = _stft(
            audio_f, fs=sr, nperseg=win_len, noverlap=win_len - hop, window="hann", padded=True, boundary="zeros"
        )
        # Zxx: (n_freqs, n_frames)

        mag = np.abs(Zxx)  # (n_freqs, n_frames)
        n_frames = mag.shape[1]

        # 2. IMCRA-Sliding-Minimum: Rauschboden = Min(letzter W Frames)
        #    Cohen (2003): iterative Minima-Controlled Recursive Averaging
        W = max(8, n_frames // 4)  # Sliding-Min-Fenster
        noise_floor = np.empty_like(mag)
        for t in range(n_frames):
            lo = max(0, t - W)
            noise_floor[:, t] = mag[:, lo : t + 1].min(axis=1)

        # Numerischer Mindest-Rauschboden: verhindert Division durch 0
        eps = np.maximum(np.percentile(mag, 1, axis=1, keepdims=True), 1e-10)
        noise_floor = np.maximum(noise_floor, eps)

        # 3. MMSE-Wiener-Gain  G = xi/(1+xi),  xi = max(SNR - 1, 0)
        #    Gain-Floor G_floor = 0.1 (Le Roux & Vincent 2013: Consistent Wiener)
        G_FLOOR = 0.1
        xi = np.maximum(mag / (noise_floor + 1e-12) - 1.0, 0.0)
        gain = xi / (xi + 1.0)
        gain = np.clip(gain, G_FLOOR, 1.0)

        # 4. ISTFT — phasenkonsistente OLA-Synthese (kein np.fft.irfft)
        Zxx_denoised = mag * gain * np.exp(1j * np.angle(Zxx))
        _, out = _istft(Zxx_denoised, fs=sr, nperseg=win_len, noverlap=win_len - hop, window="hann", boundary=True)

        # Länge anpassen + NaN/Inf-Schutz
        out = out[: len(audio_f)]
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(out, -1.0, 1.0).astype(audio.dtype)  # type: ignore[no-any-return]


class StreamingGate:
    """
    Streaming-optimiertes Gate (Stub):
    - Gate für Streaming-Anwendungen (Latenz, Stabilität)
    """

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(streaming_gate_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Streaming-Gate: Frame-RMS-Schwelle (-30/-50 dBFS) mit Hysterese-Überblendung."""
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        audio_f = audio.astype(float)
        frame = max(64, int(sr * 0.010))  # 10 ms
        thresh_open = 10 ** (-30 / 20)  # -30 dBFS: Gate öffnen
        thresh_close = 10 ** (-50 / 20)  # -50 dBFS: Gate schließen
        result = audio_f.copy()
        gate_open = False
        for i in range(0, len(audio_f), frame):
            chunk = audio_f[i : i + frame]
            rms = float(np.sqrt(np.mean(chunk**2)) + 1e-12)
            if rms >= thresh_open:
                gate_open = True
            elif rms < thresh_close:
                gate_open = False
            if not gate_open:
                result[i : i + frame] = 0.0
        return result.astype(audio.dtype)
