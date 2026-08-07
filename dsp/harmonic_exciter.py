from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.signal
from scipy.signal import butter, sosfilt


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "harmonic_exciter"
    category: str = "enhancer"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
harmonic_exciter_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {
            "band": (3000, 12000),
            "amount": 0.5,
            "saturation": 0.7,
            "formant_preserving": True,
        },
        "safe_ranges": {
            "amount": {"min": 0.0, "max": 1.0},
            "saturation": {"min": 0.0, "max": 1.0},
        },
        "trial_profile": {"wet": 1.0, "segment_sec": 1.0, "warmup_ms": 0},
    },
    budgets={
        "artifact_budget": 0.01,
        "identity_budget": 1.0,
        "spectral_change_budget": 0.05,
        "temporal_change_budget": 0.01,
        "compute_cost": 0.02,
    },
    side_effects=[{"risk": "Schärfe", "expected_when": "amount > 0.8", "severity": 0.2}],
    reports={"self_metrics": ["harmonic_enrichment"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


def _butter_sos(order: int, wn: float | list[float], btype: str) -> npt.NDArray[np.float64]:
    """Gibt validated Butterworth coefficients as SOS float64 matrix zurück."""
    sos = butter(order, wn, btype=btype, output="sos")
    return np.asarray(sos, dtype=np.float64)


class HarmonicExciter:
    """
    SOTA-konformer Harmonic Exciter:
    - Obertonanreicherung, Bandwahl, Sättigung, ML-ready
    """

    def __init__(
        self,
        band: tuple[float, float] = (3000, 12000),
        amount: float = 0.5,
        saturation: float = 0.7,
        formant_preserving: bool = True,
    ) -> None:
        """
        band: Frequenzbereich für Exciter (Hz)
        amount: Mischverhältnis (0...1)
        saturation: Sättigungsgrad (0...1)
        formant_preserving: Wenn True, wird ein Formant-Preserving-Ansatz genutzt (empfohlen für Vocals)
        """
        self.band = band
        self.amount = float(np.clip(amount, 0.0, 1.0))
        self.saturation = float(np.clip(saturation, 0.0, 1.0))
        self.formant_preserving = formant_preserving

    # Audit: Contract-Infos loggen (optional)
    def log_contract(self):
        import logging

        logging.info("[DSPContract] %s", asdict(harmonic_exciter_contract))

    def process(self, audio: npt.NDArray[np.float64], sr: int) -> npt.NDArray[np.float64]:
        """
        Verarbeitet das Eingangssignal mit Obertonanreicherung.
        audio: 1D numpy-Array (Mono)
        sr: Abtastrate (Hz)
        Rückgabe: angereichertes Signal (gleicher Typ wie audio)
        """
        # Audit: Contract-Infos loggen (optional)
        self.log_contract()
        if sr <= 0:
            raise ValueError("Ungültige Abtastrate für HarmonicExciter")
        orig_dtype = audio.dtype
        x = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # Band extrahieren (obere Grenzfrequenz muss < sr/2 sein)
        nyq = sr / 2
        low = self.band[0] / nyq
        high = min(self.band[1], nyq * 0.99) / nyq
        if not (0 < low < 1) or not (0 < high < 1) or not (low < high):
            raise ValueError(f"Ungültiges Band für HarmonicExciter: low={low}, high={high}, sr={sr}")
        sos_band = _butter_sos(4, [low, high], "band")
        band_sig = sosfilt(sos_band, x)

        # Soft saturation — scale factor 2 (not 4) to avoid near-clipping at
        # saturation=0.7: tanh(x * (1 + 0.7*2)) = tanh(x * 2.4) vs. tanh(x * 3.8)
        drive = 1.0 + self.saturation * 2.0
        excited = np.tanh(band_sig * drive)
        # Normalise excited to match RMS of band_sig (preserves level intent)
        rms_band = float(np.sqrt(np.mean(band_sig**2)) + 1e-10)
        rms_exc = float(np.sqrt(np.mean(excited**2)) + 1e-10)
        if rms_exc > 1e-10:
            excited = excited * (rms_band / rms_exc)
        # Formant-Preserving-Option (vereinfachtes Beispiel: Dry/Wet nur auf Obertöne, nicht auf Grundtonbereich)
        if self.formant_preserving:
            # Obertöne extrahieren (vereinfachtes Beispiel: Hochpass ab 2 kHz)
            sos_high = _butter_sos(2, 2000 / (sr / 2), "high")
            overtones = sosfilt(sos_high, excited - band_sig)
            out = x + self.amount * overtones
        else:
            # Standard: Mischung mit gesamtem Band
            out = x + self.amount * (excited - band_sig)
        # Pegel normalisieren
        maxval = np.max(np.abs(out))
        if maxval > 1.0:
            out = np.clip(out, -1.0, 1.0)
        return np.asarray(np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), dtype=orig_dtype)


class HarmonicExciterStudio:
    """
    SOTA Harmonic Exciter (Studio-Algorithmus):
    - Fügt Obertöne hinzu, um Brillanz und Präsenz zu steigern
    """

    def __init__(self, amount=0.3):
        self.amount = amount

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        x = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        # Soft Clipping: drive factor 2 matches class HarmonicExciter; avoids
        # near-clipping at amount=0.7: tanh(x*2.4) vs. old tanh(x*4.5)
        excited = np.tanh(x * (1.0 + self.amount * 2.0))
        # Highpass, um nur Höhen zu betonen
        sos = scipy.signal.iirfilter(2, 4000 / (sr / 2), btype="high", ftype="butter", output="sos")
        highs = scipy.signal.sosfilt(sos, excited - x)
        out = x + self.amount * highs
        peak = float(np.max(np.abs(out))) if out.size > 0 else 0.0
        if peak > 1.0:
            out = np.clip(out, -1.0, 1.0)
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)  # type: ignore[no-any-return]
