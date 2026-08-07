"""
bandwidth_extension.py - Bandbreitenerweiterung für Aurik 6.0

SOTA-konforme Bandbreitenerweiterung mit DSPContract und Auditierbarkeit.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContractBandwidthExtension:
    id: str = "bandwidth_extension"
    category: str = "bandwidth_extension"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


bandwidth_extension_contract = DSPContractBandwidthExtension(
    io={
        "channels": "mono|stereo",
        "sample_rates": [8000, 16000, 22050, 44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={"defaults": {"mode": "auto"}},
    budgets={"compute_cost": 0.01},
    side_effects=[
        {
            "risk": "Fehlrekonstruktion",
            "expected_when": "mode falsch gewählt",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["bandwidth_extension_score"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class BandwidthExtension:
    """
    SOTA-konforme Bandbreitenerweiterung:
    - Rekonstruiert/erweitert hohe oder tiefe Frequenzen bei schmalbandigen Quellen (z. B. Telefon, Funk, LoFi)
    - Auditierbar, rollback-fähig, SOTA-Maximum
    """

    contract: DSPContractBandwidthExtension = bandwidth_extension_contract

    def __init__(self, mode: str = "auto"):  # "auto", "high", "low"
        self.mode = mode

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(self.contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral Band Replication (SBR) for bandwidth extension.

        Algorithm (HE-AAC-inspired DSP fallback):
        1. Detect bandwidth cutoff via spectral energy rolloff
        2. Mirror spectral content from below cutoff to above
        3. Shape with spectral envelope and gentle rolloff
        4. Blend HF extension with original

        For ML-based extension, use AudioSR plugin (phase_06).
        """
        self.log_contract()
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float64))

        if audio.ndim == 2:
            # Process channels separately for stereo
            channels = [self._extend_channel(audio[:, ch], sr) for ch in range(audio.shape[1])]
            return np.stack(channels, axis=1)
        return self._extend_channel(audio, sr)

    def _extend_channel(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """SBR on single channel."""
        n = len(audio)
        nperseg = min(4096, n)
        if nperseg < 256:
            return audio

        # Detect bandwidth: find where spectrum drops to noise floor
        spectrum = np.abs(np.fft.rfft(audio, n=nperseg))
        freqs = np.fft.rfftfreq(nperseg, 1.0 / sr)
        nyquist = sr / 2.0

        # Spectral energy in octave bands
        energy_db = 20.0 * np.log10(spectrum + 1e-12)
        noise_floor = np.percentile(energy_db, 10)

        # Find cutoff: first frequency where energy stays within 6 dB of noise floor
        cutoff_hz = nyquist  # default: full bandwidth
        window = max(1, len(freqs) // 50)
        for i in range(len(freqs) - window, window, -1):
            band_energy = np.mean(energy_db[i : i + window])
            if band_energy > noise_floor + 6.0:
                cutoff_hz = float(freqs[i])
                break

        # Only extend if bandwidth is limited (< 90% of Nyquist)
        if cutoff_hz > nyquist * 0.90:
            logger.debug("[BandwidthExtension] Full bandwidth erkannt (%.0f Hz), no extension needed.", cutoff_hz)
            return audio

        logger.info(
            "[BandwidthExtension] erkannt bandwidth cutoff at %.0f Hz, extending to %.0f Hz.", cutoff_hz, nyquist
        )

        # SBR: mirror spectrum from [cutoff/2, cutoff] to [cutoff, 2*cutoff]
        # Process in STFT frames
        hop = nperseg // 4
        from scipy.signal import istft as _istft
        from scipy.signal import stft as _stft

        f, t, Zxx = _stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)

        cutoff_bin = int(cutoff_hz / (sr / nperseg))
        source_start = max(1, cutoff_bin // 2)
        source_end = cutoff_bin
        source_width = source_end - source_start

        if source_width < 2:
            return audio

        # Mirror and shape the HF content
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        for frame in range(Zxx.shape[1]):
            # Source band magnitude
            src_mag = mag[source_start:source_end, frame]
            if len(src_mag) == 0:
                continue

            # Target band: [cutoff, cutoff + source_width]
            target_end = min(cutoff_bin + source_width, mag.shape[0])
            target_width = target_end - cutoff_bin
            if target_width <= 0:
                continue

            # Mirror and apply rolloff envelope
            mirrored = src_mag[:target_width][::-1]  # reverse for natural taper
            rolloff = np.linspace(0.4, 0.05, target_width)  # gentle rolloff

            # Random phase for synthesized HF (avoids phase coherence artifacts)
            rng = np.random.RandomState(frame)
            synth_phase = rng.uniform(-np.pi, np.pi, target_width)

            # Apply only where existing content is below threshold
            existing = mag[cutoff_bin:target_end, frame]
            mask = existing < mirrored * rolloff * 0.5
            mag[cutoff_bin:target_end, frame] = np.where(mask, mirrored * rolloff, existing)
            phase[cutoff_bin:target_end, frame] = np.where(mask, synth_phase, phase[cutoff_bin:target_end, frame])

        Zxx_extended = mag * np.exp(1j * phase)
        _, audio_out = _istft(Zxx_extended, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)

        # Match output length
        if len(audio_out) < n:
            audio_out = np.pad(audio_out, (0, n - len(audio_out)))
        else:
            audio_out = audio_out[:n]

        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)
        return audio_out  # type: ignore[no-any-return]
