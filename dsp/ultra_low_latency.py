"""
ultra_low_latency.py - Ultra-Low-Latency-DSPs für Aurik 6.0

Dieses Modul stellt ultra-latenzoptimierte Varianten der wichtigsten DSPs bereit und ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
ull_limiter_contract = DSPContract(
    id="ultra_low_latency_limiter",
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
ull_denoiser_contract = DSPContract(
    id="ultra_low_latency_denoiser",
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
ull_gate_contract = DSPContract(
    id="ultra_low_latency_gate",
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


class UltraLowLatencyLimiter:
    """
    Ultra-Low-Latency-Limiter (Stub):
    - Limiter mit minimaler Latenz für Echtzeitanwendungen
    """

    def log_contract(self):
        import logging

        logging.info("[DSPContract] %s", asdict(ull_limiter_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """ULL-Limiter: Soft-Clipper mit tanh-Waveshaping (Ceiling = 0.9, keine Latenz)."""
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        ceiling = 0.9
        audio_f = audio.astype(float)
        # Tanh-Waveshaping: weich begrenzen, kein hartes Clipping
        result = ceiling * np.tanh(audio_f / (ceiling + 1e-9))
        return result.astype(audio.dtype)  # type: ignore[no-any-return]


class UltraLowLatencyDenoiser:
    """
    Ultra-Low-Latency-Denoiser (Stub):
    - Denoiser mit minimaler Latenz für Echtzeitanwendungen
    """

    def log_contract(self):
        logging.info("[DSPContract] %s", asdict(ull_denoiser_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """ULL-Denoiser: Spektrale Unterdrückung mit kurzem 128-Punkt FFT-Frame (4 ms @32kHz)."""
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        audio_f = audio.astype(float)
        n_fft = 128  # Minimale FFT-Größe für ULL
        result = np.zeros_like(audio_f)
        window = np.hanning(n_fft)
        # Überlappungsfreie Segmente (OLA 50%)
        for start in range(0, len(audio_f) - n_fft + 1, n_fft // 2):
            seg = audio_f[start : start + n_fft] * window
            spec = np.fft.rfft(seg)
            mag = np.abs(spec)
            phase = np.angle(spec)
            # Sehr sanfte spektrale Subtraktion (-6 dB Rauschboden)
            threshold = np.percentile(mag, 20)
            gain = np.maximum(1.0 - threshold / (mag + 1e-9), 0.4)
            end = min(len(result), start + n_fft)
            result[start:end] += np.fft.irfft(mag * gain * np.exp(1j * phase))[: end - start] * window[: end - start]
        return np.clip(result, -1.0, 1.0).astype(audio.dtype)


class UltraLowLatencyGate:
    """
    Ultra-Low-Latency-Gate (Stub):
    - Gate mit minimaler Latenz für Echtzeitanwendungen
    """

    def log_contract(self):
        logging.info("[DSPContract] %s", asdict(ull_gate_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """ULL-Gate: Sample-genauer Envelope-Follower + Schwellen-Gate (4 ms Attack/Release)."""
        self.log_contract()  # Audit: Contract-Infos loggen (optional)
        audio_f = audio.astype(float)
        threshold = 10 ** (-40 / 20)  # -40 dBFS
        attack = max(1, int(sr * 0.004))  # 4 ms
        release = max(1, int(sr * 0.020))  # 20 ms
        alpha_att = 1 - np.exp(-1.0 / attack)
        alpha_rel = 1 - np.exp(-1.0 / release)
        env = 0.0
        result = np.zeros_like(audio_f)
        for i, s in enumerate(audio_f):
            peak = abs(s)
            if peak > env:
                env += alpha_att * (peak - env)
            else:
                env += alpha_rel * (peak - env)
            result[i] = s if env >= threshold else 0.0
        return result.astype(audio.dtype)
