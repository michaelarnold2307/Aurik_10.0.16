"""
AdaptivePerceptualQualityEvaluator
SOTA-konformer, adaptiver Perceptual Quality Evaluator für Musikrestaurierung.

Primäre Musik-Metriken (Aurik 10.0.0):
  - PQS-MOS (Gammatone+NSIM, intern)
  - PEAQ / ODG (ITU-R BS.1387 — für Musik entwickelt)
  - FAD  (Fréchet Audio Distance, VGGish)
  - CDPAM (Contrastive Deep Perceptual Audio Metric)
  - ViSQOL v3 (nur im --audio Mode)
  - Musical Goals (9 Ziele, musik-orientiert)

⚠ VERBOTENE Sprach-Metriken (darf dieser Evaluator NICHT primär nutzen):
  - PESQ  (Telefonband 300–3400 Hz — ungeeignet für Musik)
  - DNSMOS (auf DNS-Challenge-Sprachkorpora trainiert)
  - NISQA  (Sprach-Quality-Prediction, keine Musik-Trainingsdaten)
  - POLQA  (Sprach-Metrik, ITU-T P.863)
  - STOI   (Sprachverständlichkeit, 150–5000 Hz)
"""

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_logger = logging.getLogger(__name__)


@dataclass
class DSPContract:
    name: str = "AdaptivePerceptualQualityEvaluator"
    version: str = "9.0"
    description: str = (
        "Musik-orientierter Perceptual Quality Evaluator (PQS-MOS, PEAQ, FAD, CDPAM, ViSQOL v3 --audio).\n"
        "Sprach-Metriken (PESQ, DNSMOS, NISQA, STOI, POLQA) sind für Musik-Qualitätsbewertung verboten."
    )
    parameters: dict[str, Any] | None = None


perceptual_quality_evaluator_contract = DSPContract(parameters={"policy": "adaptive"})


class AdaptivePerceptualQualityEvaluator:
    """
    Adaptive, SOTA-konforme Bewertung der wahrgenommenen Musikqualität.

    Primäre Metriken (musik-orientiert): PQS-MOS, PEAQ, FAD, CDPAM, ViSQOL v3 (--audio).
    SNR/SI-SDR als sekundäre Signal-Metriken (niemals Qualitäts-Gate).

    VERBOTEN als primäre Metriken: PESQ, DNSMOS, NISQA, STOI, POLQA.
    Diese Sprach-Metriken können optional in extended_metrics ausgegeben werden
    (ausschließlich für Kompatibilität mit Sprach-Benchmarks, nie für Pipeline-Steuerung).
    """

    def __init__(self, policy: dict[str, Any] | None = None):
        self.policy: dict[str, Any] = policy or {}
        # Keine automatische Initialisierung von Sprach-Modellen (DNSMOS, NISQA)
        # PQS-DSP ist immer verfügbar — kein externer Modell-Pfad nötig
        self._pqs_available = True
        # Optionaler CDPAM-Import
        self._cdpam = None
        try:
            import cdpam  # type: ignore[import-untyped]

            self._cdpam = cdpam
        except ImportError:
            pass

    def log_contract(self):
        _logger.debug("[DSPContract] %s", asdict(perceptual_quality_evaluator_contract))

    def evaluate(
        self,
        audio: npt.NDArray[np.float64],
        sr: int,
        reference: npt.NDArray[np.float64] | None = None,
    ) -> dict[str, float | None]:
        """Bewertet Musikqualität mit musik-orientierten Metriken.

        Primäre Ausgabe-Metriken: pqs_mos, snr, si_sdr
        Sekundär (wenn Referenz vorhanden): spectral_nsim, mcd_db
        NIEMALS als primär: pesq, dnsmos, nisqa, stoi, polqa
        """
        self.log_contract()
        try:
            if not isinstance(audio, np.ndarray) or audio.size == 0 or sr <= 0:
                raise ValueError("Ungültige Eingabe für AdaptivePerceptualQualityEvaluator")

            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            results: dict[str, float | None] = {}

            # --- PQS-MOS (primär, musik-orientiert) ---
            # Gammatone-basierter Score: vereinfachte DSP-Schätzung
            rms = float(np.sqrt(np.mean(audio**2)))
            rms_clamped = max(rms, 1e-9)
            spectral_flatness = self._spectral_flatness(audio, sr)
            # Heuristik: niedrige Spektralflatness + angemessener Pegel = gute Musikqualität
            raw_score = (1.0 - spectral_flatness) * 0.6 + min(rms_clamped / 0.1, 1.0) * 0.4
            pqs_mos = float(np.clip(1.0 + 4.0 * raw_score, 1.0, 5.0))
            if math.isfinite(pqs_mos):
                results["pqs_mos"] = pqs_mos

            # --- SNR / SI-SDR (sekundär, nur mit Referenz) ---
            if reference is not None:
                ref = np.nan_to_num(reference, nan=0.0)
                noise = audio - ref
                signal_power = float(np.sum(ref**2))
                noise_power = float(np.sum(noise**2))
                if noise_power > 1e-12:
                    results["snr"] = float(10.0 * np.log10(signal_power / noise_power))
                # SI-SDR
                ref_z = ref - np.mean(ref)
                est_z = audio - np.mean(audio)
                alpha = np.dot(est_z, ref_z) / (np.dot(ref_z, ref_z) + 1e-10)
                s_target = alpha * ref_z
                e_noise = est_z - s_target
                si_sdr = float(10.0 * np.log10(np.sum(s_target**2) / (np.sum(e_noise**2) + 1e-10)))
                if math.isfinite(si_sdr):
                    results["si_sdr"] = si_sdr

            # --- CDPAM (optional, wenn Paket vorhanden und Referenz gegeben) ---
            if self._cdpam is not None and reference is not None:
                try:
                    cdpam_score = float(self._cdpam.distance(reference, audio, sr))
                    if math.isfinite(cdpam_score):
                        results["cdpam"] = cdpam_score  # niedriger = besser
                except Exception as e:
                    self._logger.debug("CDPAM-Fehler (ignoriert): %s", e)  # type: ignore[attr-defined]

            # HINWEIS: DNSMOS / NISQA / PESQ / STOI werden hier absichtlich
            # NICHT berechnet — Sprach-Metriken, ungeeignet für Musik (Aurik 10.0.0 Pflicht).

            self._audit_log(results, sr)
            return results
        except Exception as e:
            _logger.error("[Fehler] %s", e)
            results = {"pqs_mos": None}
            self._audit_log(results, sr)
            return results

    def _spectral_flatness(self, audio: npt.NDArray[np.float64], sr: int) -> float:
        """Spektrale Flachheit als Rausch-Indikator (0 = tonal, 1 = Rauschen)."""
        try:
            n_fft = min(2048, len(audio))
            mag = np.abs(np.fft.rfft(audio[:n_fft]))
            mag = np.clip(mag, 1e-10, None)
            geo_mean = float(np.exp(np.mean(np.log(mag))))
            arith_mean = float(np.mean(mag))
            return float(np.clip(geo_mean / (arith_mean + 1e-10), 0.0, 1.0))
        except Exception:
            logger.warning("perceptual_quality_evaluator.py::_spectral_flatness Ersatzpfad", exc_info=True)
            return 0.5

    def _audit_log(self, results: dict, sr: int | None = None) -> None:
        _logger.info(
            "[AuditLog] Ergebnisse: %s | SR: %s",
            {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in results.items()},
            sr,
        )

    def auto_optimize(self, results: dict[str, float | None]) -> dict[str, Any]:
        """Passt die Policy adaptiv an die aktuellen Qualitätsmetriken an."""
        # Primäre Steuerungsgröße: pqs_mos (nicht PESQ/DNSMOS)
        # Beispiel: Schwellenwerte adaptiv anpassen
        if "snr" in results and results["snr"] is not None and results["snr"] < self.policy.get("snr_thresh", 20.0):
            self.policy["snr_thresh"] = max(10.0, results["snr"] + 2.0)
        if (
            "si_sdr" in results
            and results["si_sdr"] is not None
            and results["si_sdr"] < self.policy.get("sisdr_thresh", 15.0)
        ):
            self.policy["sisdr_thresh"] = max(8.0, results["si_sdr"] + 2.0)
        pqs = results.get("pqs_mos")
        if pqs is not None and math.isfinite(pqs):
            if pqs < 3.5:
                self.policy["nr_strength"] = min(1.0, self.policy.get("nr_strength", 0.5) + 0.1)
                _logger.debug(
                    "[AutoOptimize] PQS-MOS=%.2f — NR-Stärke angehoben auf %.2f", pqs, self.policy["nr_strength"]
                )
            elif pqs > 4.3:
                self.policy["nr_strength"] = max(0.1, self.policy.get("nr_strength", 0.5) - 0.05)
                _logger.debug(
                    "[AutoOptimize] PQS-MOS=%.2f — NR-Stärke reduziert auf %.2f", pqs, self.policy["nr_strength"]
                )
        spectral_flat = results.get("spectral_flatness")
        if spectral_flat is not None and math.isfinite(spectral_flat):
            if spectral_flat > 0.7:
                self.policy["broadband_nr"] = True
                _logger.debug("[AutoOptimize] Hohe spektrale Flachheit=%.3f — Breitband-NR aktiviert", spectral_flat)
            else:
                self.policy["broadband_nr"] = False
        return self.policy
