import logging
from dataclasses import asdict, dataclass
from typing import Any


# DSPContract für Auditierbarkeit und SOTA-Konformität
@dataclass(frozen=True)
class DSPContract:
    id: str = "tape_equalizer"
    category: str = "restoration"
    version: str = "1.0.0"
    io: dict[str, Any] | None = None
    preconditions: list[dict[str, Any]] | None = None
    params: dict[str, Any] | None = None
    budgets: dict[str, float] | None = None
    side_effects: list[dict[str, Any]] | None = None
    reports: dict[str, Any] | None = None
    rollback: dict[str, Any] | None = None


# Instanz des Contracts (kann für Audit/Orchestrierung genutzt werden)
tape_equalizer_contract = DSPContract(
    io={
        "channels": "mono|stereo",
        "sample_rates": [44100, 48000],
        "latency_samples": 0,
        "supports_offline": True,
    },
    preconditions=[{"if": "True", "reason": "Immer aktiv"}],
    params={
        "defaults": {"standard": "IEC"},
        "safe_ranges": {"standard": ["IEC", "NAB"]},
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
            "risk": "Frequenzverfälschung",
            "expected_when": "standard falsch gewählt",
            "severity": 0.2,
        }
    ],
    reports={"self_metrics": ["eq_curve_applied"], "confidence": 1.0},
    rollback={"strategy": "wet_to_zero|snapshot_restore", "supports_partial": True},
)
"""
tape_equalizer.py - Kassettenspezifische Entzerrung für Aurik 6.0

Dieses Modul entzerrt oder simuliert IEC/NAB-Kennlinien für Kassette (Stub).
"""
import numpy as np

logger = logging.getLogger(__name__)


class TapeEqualizer:
    """
    Kassettenspezifische Entzerrung (Stub):
    - Wendet IEC/NAB-Entzerrungskurven auf Audiosignale an (z.B. für Digitalisierung)
    """

    def __init__(self, standard: str = "IEC"):  # "IEC" oder "NAB"
        self.standard = standard

    # Audit: Contract-Infos loggen (optional)
    def log_contract(self):
        logger.debug("[DSPContract] %s", asdict(tape_equalizer_contract))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """IEC/NAB Kassetten-Wiedergabe-Entzerrung.

        NAB (1965, 7.5 ips): \u03c41=3180\u00b5s (50Hz), \u03c42=100\u00b5s (1592Hz)
        IEC (Kompaktkassette Type I): 120\u00b5s, Type II/IV: 70\u00b5s
        CCIR (Rundfunk, 7.5 ips): 3180\u00b5s / 70\u00b5s
        Alle via bilinearer Transformation zu IIR-Shelving-Kette.
        """
        from scipy.signal import lfilter

        self.log_contract()
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return audio
        # Zeitkonstanten der Normen (in Sekunden)
        # (tau_bass, tau_treble) -> Shelving-Frequenzen: f = 1/(2*pi*tau)
        _tau = {
            "NAB": (3180e-6, 100e-6),  # 50Hz Bass, 1592Hz Treble
            "IEC": (3180e-6, 120e-6),  # 50Hz Bass, 1326Hz Treble (Type I)
            "CCIR": (3180e-6, 70e-6),  # 50Hz Bass, 2274Hz Treble (Rundfunk)
        }
        tau_b, tau_t = _tau.get(self.standard.upper(), _tau["IEC"])

        # Tiefpass-Shelving (Bass-Anhebung) via bilinearer Transformation
        # H_bass(s) = (1 + s*tau_b) -> entspricht: Verstärkung unterhalb 1/(2pi*tau_b)
        # Implementierung als 1. Ordnung: z-Koeffizienten aus H(s)=s*tau/(s*tau+1)
        # Hier: Hochpass-Shelving-Paar für Bandenzerrung
        # Bass-Boost: High-Pass shelving inverted
        def _first_order_shelf(tau, highpass=False):
            """Gibt (b, a) eines Hochpass- oder Tiefpass-1st-Order-Regal-IIR zurück."""
            k = 2.0 * sr * tau
            if highpass:
                # H(s) = s*tau/(1+s*tau): Hochpass-Charakter
                b = np.array([k / (k + 1.0), -k / (k + 1.0)])
                a = np.array([1.0, (1.0 - k) / (k + 1.0)])
            else:
                # H(s) = 1/(1+s*tau): Tiefpass-Charakter
                b = np.array([1.0 / (k + 1.0), 1.0 / (k + 1.0)])
                a = np.array([1.0, (1.0 - k) / (k + 1.0)])
            return b, a

        # Kassetten-Wiedergabe-Entzerrung:
        # Bass-Boost = inverse des Aufnahme-Tiefpassendes
        b_b, a_b = _first_order_shelf(tau_b, highpass=True)  # Bässe anheben
        b_t, a_t = _first_order_shelf(tau_t, highpass=False)  # Höhen dämpfen

        def _apply(ch):
            y = lfilter(b_b, a_b, ch.astype(np.float64))
            return lfilter(b_t, a_t, y)

        if audio.ndim == 1:
            return _apply(audio).astype(audio.dtype)  # type: ignore[no-any-return]
        return np.stack([_apply(ch) for ch in audio], axis=0).astype(audio.dtype)  # type: ignore[no-any-return]
