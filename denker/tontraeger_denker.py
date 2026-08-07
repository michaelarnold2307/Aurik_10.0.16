"""
TontraegerDenker — Domäne: Tonträger-/Material-Erkennung
=========================================================

Kapselt `forensics.medium_detector.MediumDetector` und liefert
einen strukturierten `TontraegerInfo`-Bericht über das erkannte
Quellmaterial (§6.1 / §2.14).

Singleton-Pattern nach §3.2 (Double-Checked Locking).
NaN/Inf-Schutz nach §3.1.
Type-Annotations nach §3.7.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class TontraegerInfo:
    """Strukturierter Bericht über das erkannte Quellmaterial."""

    material_type: str
    """Erkannter MaterialType-Name (z. B. 'vinyl', 'tape', 'unknown')."""

    confidence: float
    """Erkennungs-Konfidenz ∈ [0, 1]."""

    medium_details: dict[str, Any] = field(default_factory=dict)
    """Rohe Zusatz-Informationen aus MediumDetector (z. B. rolloff_hz, snr_db)."""

    chain_detected: bool = False
    """True, wenn eine Mehrstufen-Tonträgerkette erkannt wurde."""

    transfer_chain: list = field(default_factory=list)
    """Liste der erkannten Trägerstufen (z. B. ['tape', 'mp3_low'])."""

    reasoning: str = ""
    """Menschenlesbare Begründung der Erkennung (Deutsch)."""

    classification_result: Any = None
    """ClassificationResult-Objekt für Passthrough an UV3 (§6.7 v10.0.0)."""

    bayesian_scores: dict[str, float] = field(default_factory=dict)
    """Posterior-Wahrscheinlichkeiten aller 16 Materialtypen."""

    def as_dict(self) -> dict[str, Any]:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "material_type": self.material_type,
            "confidence": self.confidence,
            "chain_detected": self.chain_detected,
            "transfer_chain": self.transfer_chain,
            "reasoning": self.reasoning,
            **self.medium_details,
        }


@dataclass
class TontraegerErgebnis:
    """Ergebnis der Tonträger-Erkennung (kompaktes Reporting)."""

    material_type: str
    """Erkannter Träger-Typ (z. B. 'vinyl', 'tape', 'shellac')."""

    confidence: float
    """Erkennungs-Konfidenz ∈ [0, 1]."""

    detected_media: list
    """Liste von (typ: str, konfidenz: float)-Tupeln aller Kandidaten."""

    reasoning: str
    """Laienverständliche Begründung der Träger-Erkennung."""

    recommended_phases: list
    """Empfohlene Verarbeitungsphasen für diesen Träger."""


# ---------------------------------------------------------------------------
# TontraegerDenker
# ---------------------------------------------------------------------------


class TontraegerDenker:
    """Erkennt das Quellmaterial einer Aufnahme (Tonträger-Domäne).

    Wraps `forensics.medium_detector.MediumDetector` und ergänzt
    strukturiertes Reporting, NaN-Schutz und laienverständliche
    Begründungstexte.

    Verwendung::

        denker = get_tontraeger_denker()
        info = denker.erkenne(audio, sr=44100)
        logger.debug(info.material_type, info.confidence)
    """

    def __init__(self) -> None:
        self._detector: Any | None = None
        self._detector_lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy-Init des MediumDetector
    # ------------------------------------------------------------------

    def _get_detector(self) -> Any:
        """Gibt the MediumDetector instance, initializing it lazily zurück."""
        if not self._loaded:
            with self._detector_lock:
                if not self._loaded:
                    try:
                        from backend.core.forensics.medium_detector import MediumDetector

                        self._detector = MediumDetector()
                        logger.info("TontraegerDenker: MediumDetector geladen.")
                    except Exception as exc:
                        logger.warning(
                            "TontraegerDenker: MediumDetector nicht verfügbar (%s). Ersatzpfad auf 'unknown'.",
                            exc,
                        )
                        self._detector = None
                    self._loaded = True
        return self._detector

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def erkenne(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        validate_audio: bool = True,
        file_path: str = "",
    ) -> TontraegerInfo:
        """Erkennt das Quellmaterial und liefert eine strukturierten TontraegerInfo.

        Algorithmus:
            1. NaN/Inf-Schutz auf Eingabe
            2. MediumDetector.detect(audio, sr, file_ext=...) → Dict
            3. Strukturierung + Konfidenz-Normalisierung
            4. Laienverständliche Begründung erzeugen

        Args:
            audio: Eingabe-Audio, float32/64 mono oder stereo.
            sr:    Sample-Rate in Hz.
            validate_audio: NaN/Inf-Bereinigung durchführen (Standard: True).
            file_path: Optionaler Pfad zur Quelldatei.  Die Dateiendung wird als
                Prior für die Materialerkennung verwendet (§6.7b): digitale
                Formate (.mp3, .flac, …) schließen analoge Materialien aus.

        Returns:
            TontraegerInfo mit material_type, confidence und Details.

        Raises:
            ValueError: Falls sr < 1000 (plausibilitätsprüfung).
        """
        if sr < 1000:
            raise ValueError(f"Sample-Rate ungültig: {sr} Hz (mindestens 1000 Hz erwartet).")

        assert sr == 48000, f"TontraegerDenker.erkenne() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        if validate_audio:
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        detector = self._get_detector()

        if detector is None:
            return self._fallback_result()

        import os as _os

        _file_ext = _os.path.splitext(file_path)[1] if file_path else ""

        try:
            raw_result = detector.detect(audio, sr, file_ext=_file_ext)
            # Preserve classification_result + bayesian_scores for passthrough (§6.7 v10.0.0)
            _classification_result = getattr(raw_result, "classification_result", None)
            _bayesian_scores = getattr(raw_result, "bayesian_scores", {}) or {}
            # MediumDetectionResult (dataclass) → dict für _struktur()
            if isinstance(raw_result, dict):
                raw: dict[str, Any] = raw_result
            elif hasattr(raw_result, "as_dict"):
                raw = raw_result.as_dict()
                # Normalize field names: as_dict() uses "primary_material", _struktur() uses "material_type"
                if "material_type" not in raw and "primary_material" in raw:
                    raw["material_type"] = raw["primary_material"]
                # "detected_media" from transfer_chain
                if "detected_media" not in raw and "transfer_chain" in raw:
                    raw["detected_media"] = [(m, 1.0) for m in raw.get("transfer_chain", [])]
            else:
                raw = {
                    "material_type": str(getattr(raw_result, "primary_material", "unknown")),
                    "confidence": float(getattr(raw_result, "confidence", 0.5)),
                }
        except Exception as exc:
            logger.warning("TontraegerDenker: erkennen() fehlgeschlagen (%s). Ersatzpfad.", exc)
            return self._fallback_result()

        info = self._struktur(raw)
        info.classification_result = _classification_result
        info.bayesian_scores = _bayesian_scores
        return info

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _struktur(self, raw: dict[str, Any]) -> TontraegerInfo:
        """Strukturiert das Roh-Dict aus MediumDetector zu TontraegerInfo."""
        # Extrahiere bekannte Felder defensiv
        material = str(raw.get("material_type", raw.get("medium", "unknown"))).lower()
        confidence = float(raw.get("confidence", 0.5))

        # Konfidenz-Guard: muss finite sein
        if not math.isfinite(confidence):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        detected_media_raw: list[tuple] = raw.get("detected_media", [])
        chain = [m for m, _ in detected_media_raw] if detected_media_raw else []
        chain_detected = len(chain) > 1

        reasoning = self._begründung(material, confidence, chain_detected)

        # Rohe Details ohne bekannte Felder
        details = {k: v for k, v in raw.items() if k not in {"material_type", "medium", "confidence", "detected_media"}}

        logger.info(
            "TontraegerDenker: material=%s confidence=%.2f chain=%s",
            material,
            confidence,
            chain,
        )

        return TontraegerInfo(
            material_type=material,
            confidence=confidence,
            medium_details=details,
            chain_detected=chain_detected,
            transfer_chain=list(chain) if isinstance(chain, list) else [],
            reasoning=reasoning,
        )

    @staticmethod
    def _begründung(material: str, confidence: float, chain: bool) -> str:
        """Erzeugt eine laienverständliche Begründung (Deutsch)."""
        label_map = {
            "vinyl": "Schallplatte (Vinyl)",
            "tape": "Magnetband (Kassette/Tonband)",
            "reel_tape": "Profi-Spulenband",
            "shellac": "Schellack-Platte (78 rpm)",
            "wax_cylinder": "Phonograph-Wachswalze",
            "wire_recording": "Drahtbandaufnahme",
            "lacquer_disc": "Acetat-Lackfolie",
            "dat": "DAT (Digital Audio Tape)",
            "cd_digital": "CD / digitale Aufnahme",
            "mp3_low": "Stark komprimierte MP3-Datei",
            "mp3_high": "MP3-Datei",
            "aac": "AAC / M4A-Datei",
            "minidisc": "MiniDisc",
            "streaming": "Streaming-Kopie",
            "unknown": "Unbekanntes Material",
        }
        _mk = getattr(material, "value", material)  # §v10.113 — enum→string
        label = label_map.get(_mk, str(material))
        pct = int(confidence * 100)
        base = f"Erkanntes Material: {label} ({pct} % Konfidenz)."
        if chain:
            base += " Es wurde eine mehrstufige Übertragungskette entdeckt."
        return base

    @staticmethod
    def _fallback_result() -> TontraegerInfo:
        """Sicheres Fallback-Ergebnis wenn MediumDetector nicht verfügbar."""
        return TontraegerInfo(
            material_type="unknown",
            confidence=0.0,
            reasoning="Material-Erkennung nicht verfügbar — konservative Voreinstellung wird genutzt.",
        )


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: TontraegerDenker | None = None
_lock = threading.Lock()


def get_tontraeger_denker() -> TontraegerDenker:
    """Thread-sicherer Singleton-Accessor für TontraegerDenker."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = TontraegerDenker()
    return _instance
