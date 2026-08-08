import logging

"""
auto_bypass_order.py - SOTA-konformes Auto-Bypass/Order Modul für Aurik 6.0

Dieses Modul ist jetzt mit DSPContract für Auditierbarkeit und SOTA-Konformität ausgestattet.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "auto_bypass_order"
    category: str = "bypass_order"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[Any] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    side_effects: list[Any] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
auto_bypass_order_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"model_path": None},
        "safe_ranges": {"model_path": "str|None"},
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
        {
            "risk": "Fehlentscheidung",
            "expected_when": "Modell nicht trainiert",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["bypass_decision"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)


class AutoBypassOrder:
    """
    Klassisches Auto-Bypass/Order (SOTA-Maximum):
    - Entscheidet regelbasiert, welche DSPs aktiviert und in welcher Reihenfolge sie geschaltet werden
    """

    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(auto_bypass_order_contract))

    def decide(self, audio: np.ndarray, sr: int, dsp_list: list[str], use_dl: bool = False) -> list[str]:
        """
        Entscheidet regelbasiert oder optional per DL-Inferenz die Reihenfolge der DSPs.
        Quality-Gate, Audit-Logging, robuste Fehlerbehandlung integriert.
        :param audio: Eingabe-Audiosignal (np.ndarray)
        :param sr: Sample-Rate
        :param dsp_list: Liste der verfügbaren DSPs
        :param use_dl: Optional Deep-Learning-Inferenz (Platzhalter)
        :return: Reihenfolge der DSPs (List[str])
        """
        self.log_contract()
        # Quality-Gate: Input-Check
        if not isinstance(audio, np.ndarray):
            self._audit_log("error", "Input is not a numpy array")
            raise ValueError("Input must be a numpy array")
        if audio.ndim != 1:
            self._audit_log("error", "Input must be 1D array")
            raise ValueError("Input must be 1D array")
        if not isinstance(dsp_list, list):
            self._audit_log("error", "dsp_list is not a list")
            raise ValueError("dsp_list must be a list")
        try:
            if use_dl:
                self._audit_log("info", "DL-Inferenz aktiviert (Platzhalter)")
                order = self._dl_decide(audio, sr, dsp_list)
            else:
                # ...bestehende Regel-Logik...
                order = []
                rms = np.sqrt(np.mean(audio**2))
                peak = np.max(np.abs(audio))
                zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))))
                if "Denoiser" in dsp_list and rms < 0.05:
                    order.append("Denoiser")
                if "Declipper" in dsp_list and peak > 0.98:
                    order.append("Declipper")
                if "Declicker" in dsp_list and zero_crossings > 0.1 * len(audio):
                    order.append("Declicker")
                if "Debuzzer" in dsp_list:
                    order.append("Debuzzer")
                if "EQ" in dsp_list:
                    order.append("EQ")
                if "Mastering" in dsp_list:
                    order.append("Mastering")
                for dsp in dsp_list:
                    if dsp not in order:
                        order.append(dsp)
            self._audit_log("success", "DSP-Order-Entscheidung erfolgreich")
            return order
        except Exception as e:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6
            self._audit_log("error", f"Fehler bei DSP-Order-Entscheidung: {e}")
            # Fallback: Rückgabe Originalreihenfolge
            return dsp_list

    def _audit_log(self, level: str, message: str) -> None:
        _fn = {"error": logger.error, "warn": logger.warning, "warning": logger.warning}.get(level.lower(), logger.info)
        _fn("[auto_bypass_order] %s", message)

    def _dl_decide(self, audio: np.ndarray, sr: int, dsp_list: list[str]) -> list[str]:
        """Spektral-basierte Heuristik für optimale DSP-Reihenfolge.

        Analyseleitfaden (Signal-Pathologie-Kaskade):
          1. Click/Crackle (Impulse) -> Declicker zuerst
          2. Clipping (peak >=0.99) -> Declipper
          3. Rauschen (niedriger SNR) -> Denoiser
          4. Brumm (50/60 Hz-Harmonische) -> Debuzzer
          5. Spektrale Korrektur -> EQ
          6. Dynamik -> Kompressor/Mastering
          7. Rest in Originalreihenfolge
        """
        order = []
        seen = set()

        rms = float(np.sqrt(np.mean(audio**2))) + 1e-12
        peak = float(np.max(np.abs(audio)))
        zcr = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * max(1, len(audio))))
        impulses = int(np.sum(np.abs(audio) > 0.7 * peak))

        # Spektrumanalyse
        n_fft = min(4096, len(audio))
        mag = np.abs(np.fft.rfft(audio[:n_fft] * np.hanning(n_fft), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        # Brummton-Energie: 50/60Hz Harmonische
        hum_mask = np.zeros(len(freqs), dtype=bool)
        for f0 in [50.0, 60.0]:
            for k in range(1, 6):
                fc = k * f0
                hum_mask |= np.abs(freqs - fc) < 1.5
        total_e = float(np.sum(mag**2)) + 1e-12
        hum_e = float(np.sum(mag[hum_mask] ** 2))
        hum_ratio = hum_e / total_e

        def _add(name: str):
            if name in dsp_list and name not in seen:
                order.append(name)
                seen.add(name)

        # Click: hohe Impulsrate oder hohe ZCR
        if impulses > int(len(audio) * 0.002) or zcr > 0.3:
            _add("Declicker")
        # Clipping
        if peak >= 0.99:
            _add("Declipper")
        # Rauschen: niedriger RMS relativ zu Peak
        if peak / rms > 15:
            _add("Denoiser")
        # Brumm
        if hum_ratio > 0.05:
            _add("Debuzzer")
        # Spektrale Korrektur
        _add("EQ")
        # Dynamik
        _add("Mastering")
        # Rest in Originalreihenfolge
        for dsp in dsp_list:
            if dsp not in seen:
                order.append(dsp)
                seen.add(dsp)
        return order
