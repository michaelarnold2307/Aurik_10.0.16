"""
TontraegerketteDenker — Domäne: Mehrstufige Tonträgerketten-Analyse
====================================================================

Analysiert mehrstufige Übertragungsketten in Audioaufnahmen gemäß §6.6
(Tonträgerketten-Erkennung, bindend ab v10.0.0).

In der Praxis landet eine Aufnahme selten auf nur einem Träger:
  Vinyl → Kassette → MP3 (klassische 3-Stufen-Kette)
  Spulenband → CD → Streaming (moderne Digitalisierung)

Aufgaben dieses Denkers:
  1. Erkennt alle beteiligten Träger-Medien via MediumDetector
  2. Ordnet die Kette *zeitlich* (Quelle → Zwischenstufen → Container)
  3. Beschreibt die Degradation jeder Übertragungsstufe
  4. Empfiehlt Restaurierungs-Phasen für jede Stufe (aus Phase-Map §7.2)
  5. Schätzt die Gesamt-Kettenkomplexität als Maß für den Restaurierungsaufwand

Singleton-Pattern nach §3.2 (Double-Checked Locking, thread-sicher).
NaN/Inf-Schutz nach §3.1.
Type-Annotations nach §3.7.
Docstrings mit mathematischen Formeln und deutschen Nutzer-Texten.
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
# Konstanten & Lookup-Tabellen
# ---------------------------------------------------------------------------

# Zeitliche Einordnung der Medien:
#   0 = physikalisch-analoges Original (Quelle)
#   1 = analoges Zwischenformat (Kassette, Spulenband)
#   2 = verlustfreies Digitalformat (CD, DAT)
#   3 = verlustbehaftetes Digitalformat (Container/End-Format)
# Höherer Wert = weiter hinten in der zeitlichen Kette.
_MEDIUM_ORDER: dict[str, int] = {
    # Physikalisch-analoge Quellmedien (Ära 0) — direkte Aufzeichnung
    "wax_cylinder": 0,
    "lacquer_disc": 0,
    "shellac": 0,
    "wire_recording": 0,
    "reel_tape": 0,  # Original-Aufnahmemedium (kommt VOR Vinyl-Pressung)
    "tape": 0,  # Original-Aufnahmemedium
    # Analoge Kopien/Vervielfältigungen (Ära 1) — vom Quellmedium abgeleitet
    "vinyl": 1,  # Schallplatten-Pressung aus Tape-Master
    "cassette": 1,  # Bandkopie
    # Verlustfreie Digitalformate (Ära 2)
    "dat": 2,
    "cd_digital": 2,
    "cd": 2,
    "digital": 2,
    "minidisc": 2,
    # Verlustbehaftete End-Container (Ära 3) — immer letztes Kettenglied
    "mp3_low": 3,
    "mp3_high": 3,
    "damaged_mp3": 3,
    "aac": 3,
    "streaming": 3,
}

# Restaurierungs-Phasen je Medium (§7.2 CAUSE_TO_PHASES-Mapping)
_PHASE_MAP: dict[str, list[str]] = {
    "vinyl": [
        "phase_09_crackle_removal",
        "phase_01_click_removal",
        "phase_05_rumble_filter",
    ],
    "shellac": [
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "phase_01_click_removal",
    ],
    "tape": [
        "phase_29_tape_hiss_reduction",
        "phase_24_dropout_repair",
        "phase_03_denoise",
    ],
    "reel_tape": [
        "phase_29_tape_hiss_reduction",
        "phase_03_denoise",
        "phase_24_dropout_repair",
    ],
    "cassette": [
        "phase_29_tape_hiss_reduction",
        "phase_12_wow_flutter_fix",
        "phase_03_denoise",
    ],
    "wax_cylinder": [
        "phase_03_denoise",
        "phase_06_frequency_restoration",
        "phase_01_click_removal",
    ],
    "lacquer_disc": [
        "phase_09_crackle_removal",
        "phase_01_click_removal",
        "phase_03_denoise",
    ],
    "wire_recording": [
        "phase_12_wow_flutter_fix",
        "phase_24_dropout_repair",
        "phase_03_denoise",
    ],
    "dat": [
        "phase_24_dropout_repair",
        "phase_23_spectral_repair",
    ],
    "cd_digital": [
        "phase_23_spectral_repair",
        "phase_06_frequency_restoration",
    ],
    "cd": [
        "phase_23_spectral_repair",
        "phase_06_frequency_restoration",
    ],
    "digital": [],
    "minidisc": [
        "phase_23_spectral_repair",
        "phase_06_frequency_restoration",
    ],
    "mp3_low": [
        "phase_23_spectral_repair",
        "phase_50_spectral_repair",
    ],
    "mp3_high": [
        "phase_23_spectral_repair",
    ],
    "damaged_mp3": [
        "phase_23_spectral_repair",
        "phase_50_spectral_repair",
    ],
    "aac": [
        "phase_23_spectral_repair",
        "phase_38_presence_boost",
    ],
    "streaming": [
        "phase_03_denoise",
        "phase_23_spectral_repair",
    ],
}

# Beschreibung der Degradation je Trägermedium (Deutsch)
_DEGRADATION: dict[str, str] = {
    "vinyl": "Klicks, Kratzer und Rillenrumpeln",
    "shellac": "Breites Grundrauschen, begrenzte Bandbreite (≤ 8 kHz)",
    "tape": "Bandrauschen, Dropout, Magnetisierungsinstabilität",
    "reel_tape": "Bandrauschen, Print-Through-Vorecho, Dropout",
    "cassette": "Bandrauschen, Wow & Flutter, HF-Dämpfung (≥ 8 kHz)",
    "wax_cylinder": "Extremes Grundrauschen, mechanische Verzerrung, HF ≤ 5 kHz",
    "lacquer_disc": "Rillenverschleiß, Substrat-Rauschen, Klicken",
    "wire_recording": "Jitter, Frequenzgang-Einbrüche, Magnetdraht-Modulation",
    "dat": "Jitter, Dropout, ATRAC-Artefakte",
    "cd_digital": "Quantisierungsrauschen, mögliches Clipping",
    "cd": "Quantisierungsrauschen, mögliches Clipping",
    "digital": "Keine wesentliche analoge Degradation",
    "minidisc": "ATRAC-Stufigkeit, HF-Verlust",
    "mp3_low": "Starke Codec-Artefakte, Frequenzbeschneidung (≤ 128 kbps)",
    "mp3_high": "Moderate Codec-Artefakte",
    "damaged_mp3": "Schwere Codec-Artefakte oder Dateikorruption",
    "aac": "AAC-Codec-Artefakte, Präsenzverlust",
    "streaming": "Variables Bitrate-Profil, Codec-Artefakte",
}

# Menschenlesbare Bezeichnungen (Deutsch)
_LABEL: dict[str, str] = {
    "vinyl": "Schallplatte (Vinyl)",
    "shellac": "Schellack (78 rpm)",
    "tape": "Magnetband",
    "reel_tape": "Profi-Spulenband",
    "cassette": "Kassette",
    "wax_cylinder": "Phonograph-Wachswalze",
    "lacquer_disc": "Acetat-Lackfolie",
    "wire_recording": "Drahtbandaufnahme",
    "dat": "DAT (Digital Audio Tape)",
    "cd_digital": "CD / Digitale Aufnahme",
    "cd": "CD",
    "digital": "Digitale Aufnahme",
    "minidisc": "MiniDisc",
    "mp3_low": "MP3 (stark komprimiert, ≤ 128 kbps)",
    "mp3_high": "MP3 (≥ 128 kbps)",
    "damaged_mp3": "Beschädigte MP3",
    "aac": "AAC / M4A",
    "streaming": "Streaming-Kopie",
}

# Komplexitätsfaktor je Medium (Beitrag zur Kettenkomplexi­tät)
# Höherer Wert = schwieriger zu restaurieren
_COMPLEXITY_WEIGHT: dict[str, float] = {
    "wax_cylinder": 0.95,
    "lacquer_disc": 0.80,
    "shellac": 0.85,
    "vinyl": 0.55,
    "wire_recording": 0.90,
    "reel_tape": 0.50,
    "tape": 0.55,
    "cassette": 0.60,
    "dat": 0.20,
    "cd_digital": 0.15,
    "cd": 0.15,
    "digital": 0.05,
    "minidisc": 0.40,
    "mp3_low": 0.70,
    "mp3_high": 0.40,
    "damaged_mp3": 0.80,
    "aac": 0.45,
    "streaming": 0.50,
}

# Additive Phasen im §2.46-Sinne (Stufe 5): ergänzen Energie/Spektrum statt zu subtrahieren.
# Alle anderen _PHASE_MAP-Einträge sind primär subtraktiv (Rauschen, Artefakte entfernen).
_ADDITIVE_PHASE_PREFIXES: frozenset[str] = frozenset(
    {
        "phase_06_",  # frequency_restoration — Bandbreiten-Erweiterung (Träger additiv)
        "phase_07_",  # harmonic_restoration — Harmonik-Rekonstruktion
        "phase_21_",  # harmonic_exciter — Oberton-Synthese
        "phase_38_",  # presence_boost — Präsenz-Anhebung (HF-Additiv)
        "phase_55_",  # diffusion_inpainting — spektrales Diffusions-Inpainting
    }
)


# ---------------------------------------------------------------------------
# Ergebnis-Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class KettenGlied:
    """Einzelne Übertragungsstufe innerhalb einer Tonträgerkette."""

    medium: str
    """Medien-Typ (z. B. 'vinyl', 'cassette', 'mp3_low')."""

    position: int
    """Zeitliche Position: 0 = Quelle, größer = später in der Kette."""

    score: float
    """Erkennungskonfidenz für dieses Medium, ∈ [0, 1]."""

    degradation_type: str
    """Art der durch dieses Medium eingebrachten Degradation (Deutsch)."""

    recommended_phases: list[str]
    """Aurik-Phasen, die für genau diese Stufe empfohlen werden."""

    label: str = ""
    """Menschenlesbare Bezeichnung (Deutsch)."""

    def as_dict(self) -> dict[str, Any]:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "medium": self.medium,
            "position": self.position,
            "score": float(self.score),
            "degradation_type": self.degradation_type,
            "recommended_phases": self.recommended_phases,
            "label": self.label,
        }


@dataclass
class KettenErgebnis:
    """Strukturierter Bericht einer Tonträgerketten-Analyse.

    Enthält die zeitlich geordnete Übertragungskette (Quelle→Container),
    die aggregierten Phasen-Empfehlungen und die berechnete Kettenkomplexität.

    Berechnung der Kettenkomplexität:
        complexity = clip(1 − ∏(1 − w_i), 0, 1)
        wobei w_i = _COMPLEXITY_WEIGHT[medium_i] für jedes Kettenglied.
    """

    chain: list[str]
    """Zeitlich geordnete Medien-Liste, Quelle zuerst
    (z. B. ['vinyl', 'cassette', 'mp3_low'])."""

    chain_string: str
    """Menschenlesbare Kettendarstellung (z. B. 'Vinyl → Kassette → MP3')."""

    is_multi_generation: bool
    """True wenn mehr als ein Trägermedium erkannt wurde."""

    generation_count: int
    """Anzahl der erkannten Übertragungsstufen."""

    primary_medium: str
    """Aktueller Container (letztes Glied der Kette, z. B. 'mp3_low')."""

    original_medium: str
    """Mutmaßlicher Ursprungsträger (erstes Glied, z. B. 'vinyl')."""

    glieder: list[KettenGlied]
    """Alle Kettenglieder mit Details und Phasen-Empfehlungen."""

    combined_phases: list[str]
    """Vereinigte, deduplizierte Phasen-Empfehlungen für die gesamte Kette."""

    chain_complexity: float
    """Kettenkomplexität ∈ [0, 1]; 1.0 = maximal schwierig zu restaurieren."""

    confidence: float
    """Gesamt-Konfidenz der Kettenerkennung ∈ [0, 1]."""

    spectral_evidence: dict[str, Any] = field(default_factory=dict)
    """Roh-Spektralmerkmale (wow_strength, flutter_strength, clicks_per_sec usw.)."""

    reasoning: str = ""
    """Laienverständliche Begründung auf Deutsch."""

    def as_dict(self) -> dict[str, Any]:
        """Serialisierungsformat für Logging und Persistenz."""
        return {
            "chain": self.chain,
            "chain_string": self.chain_string,
            "is_multi_generation": self.is_multi_generation,
            "generation_count": self.generation_count,
            "primary_medium": self.primary_medium,
            "original_medium": self.original_medium,
            "glieder": [g.as_dict() for g in self.glieder],
            "combined_phases": self.combined_phases,
            "chain_complexity": float(self.chain_complexity),
            "confidence": float(self.confidence),
            "spectral_evidence": {
                k: (float(v) if isinstance(v, float) else v) for k, v in self.spectral_evidence.items()
            },
            "reasoning": self.reasoning,
        }


@dataclass
class ChainPhasePlan:
    """§2.46-konformer Pflicht-Phasenplan aus der Trägerketten-Inversion.

    Enthält Phasen, die unabhängig vom DefectScanner-Score aktiv sein MÜSSEN —
    abgeleitet aus der erkannten Trägerkette (§6.2a Komplement).
    DefectScanner arbeitet statistisch; tiefe Einzeldefekte können unter Schwelle liegen.

    Reihenfolge-Invariante (§2.46):
        must_have_phases = (subtraktiv, Inversions-Reihenfolge) + (additiv)
        d.h.: letzter Träger (Container) → Zwischenstufen → Ursprungsträger → additiv
    """

    must_have_phases: list[str]
    """Pflicht-Phasen in §2.46-Inversionsreihenfolge.
    Subtraktive Phasen (Noise, Artefakte) kommen zuerst; additive (Bandbreite, Harmonik) am Ende."""

    additive_phases: list[str]
    """Additive Phasen aus dieser Kette (Teilmenge von must_have_phases)."""

    chain_string: str
    """Ketten-Darstellung für Logging (z. B. 'Vinyl → Kassette → MP3')."""

    stage_count: int
    """Anzahl erkannter Trägerstufen."""


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class TontraegerketteDenker:
    """Denker für mehrstufige Tonträgerketten.

    Analysiert, welche Übertragungsschritte eine Aufnahme durchlaufen hat
    (z. B. Vinyl → Kassette → MP3) und erstellt einen konsolidierten
    Restaurierungsplan für alle beteiligten Medien.

    Temporal-Ordnung:
        Die Kette wird in der Reihenfolge übernommen, in der sie ankommt — entweder
        aus `_erkennen()` (abgeleitet von `MediumDetectionResult.transfer_chain`, vom
        MediumDetector selbst mit vollem Signal-Kontext chronologisch geordnet) oder aus
        einem gecachten Ergebnis (§2.47a Direct Handover). `_MEDIUM_ORDER` dient nur noch
        als Referenztabelle (Degradationstyp/Phasen-Lookup), NICHT mehr als Re-Sort-Schlüssel
        — ein nachträglicher Re-Sort würde die bereits kontextreichere Ordnung verfälschen.

    Kettenkomplexität:
        complexity = clip(1 − ∏(1 − w_i), 0, 1)
        Ein einziges Schellack-Medium → complexity ≈ 0.85.
        Vinyl + Kassette + MP3 → complexity ≈ 1 − (1−0.55)(1−0.60)(1−0.70) ≈ 0.946.
    """

    def __init__(self) -> None:
        """Initialisiert den Denker (Lazy-Load der Forensics-Komponente)."""
        self._detector: object | None = None
        self._detector_lock = threading.Lock()
        logger.debug("TontraegerketteDenker initialisiert.")

    # ------------------------------------------------------------------
    # Singleton-Infrastruktur (wird von get_tontraegerkette_denker genutzt)
    # ------------------------------------------------------------------

    def analysiere(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        file_path: str = "",
        cached_medium_result: object | None = None,
    ) -> KettenErgebnis:
        """Analysiert die Tonträgerkette eines Audio-Signals.

        Algorithmus:
            1. NaN/Inf-Schutz (§3.1)
            2. MediumDetector.detect(audio, sr, file_ext=...) → Rohbefund
               (überspringen wenn cached_medium_result übergeben, §2.47a)
            3. detected_media (List[Tuple[str, float]]) extrahieren, bereits
               chronologisch geordnet (Quelle → Container)
            4. KettenGlieder mit Phasen-Empfehlungen aufbauen
            5. Kettenkomplexität berechnen: 1 − ∏(1 − w_i)
            6. Reasoning auf Deutsch formulieren
            7. KettenErgebnis zurückgeben

        Args:
            audio:                Float32-Array ∈ [-1, 1], mono oder stereo.
            sr:                   Abtastrate in Hz.
            file_path:            Optionaler Pfad zur Quelldatei.  Dateiendung wird als
                                  Prior für die Materialerkennung verwendet (§6.7b).
            cached_medium_result: Vorhandenes MediumDetectionResult aus Pre-Analysis
                                  (§2.47a Direct Handover). Falls übergeben, wird
                                  MediumDetector.detect() NICHT erneut aufgerufen.

        Returns:
            KettenErgebnis mit zeitlich geordneter Kette, Phasen und Komplexität.
        """
        assert sr == 48000, f"TontraegerketteDenker.analysiere() erwartet sr=48000 Hz, erhalten: {sr} Hz"
        # §3.1 NaN/Inf-Schutz
        audio = np.nan_to_num(
            audio.astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # §2.47a: Prefer cached result to avoid duplicate MediumDetector.detect() calls.
        # Guard: if cached chain is clearly under-informative (single-link + low confidence),
        # run one fresh detect() as recovery path so multi-generation chains are not lost.
        if cached_medium_result is not None:
            _cached_chain = getattr(cached_medium_result, "transfer_chain", None)
            _cached_conf = float(getattr(cached_medium_result, "confidence", 0.0) or 0.0)
            _cached_primary = (
                getattr(cached_medium_result, "primary_material", None)
                or getattr(cached_medium_result, "material_type", None)
                or "?"
            )
            _chain_len = len(_cached_chain) if isinstance(_cached_chain, (list, tuple)) else 0
            _weak_cached_chain = (_chain_len <= 1) and (_cached_conf < 0.55)

            if not _weak_cached_chain:
                logger.debug(
                    "TontraegerketteDenker.analysiere(): gecachtes MediumResult übernommen "
                    "(primary_material=%s, conf=%.2f, chain_len=%d) — erkennen() NICHT aufgerufen",
                    _cached_primary,
                    _cached_conf,
                    _chain_len,
                )
                raw = self._aufbereiten_from_cached(cached_medium_result)
                return raw

            logger.debug(
                "TontraegerketteDenker.analysiere(): schwaches Zwischenspeicher-Ergebnis erkannt "
                "(primary_material=%s, conf=%.2f, chain_len=%d) — erkennen() Wiederherstellung wird ausgeführt",
                _cached_primary,
                _cached_conf,
                _chain_len,
            )

        # Detektion durchführen (nur wenn kein cached result)
        import os as _os

        _file_ext = _os.path.splitext(file_path)[1] if file_path else ""
        raw = self._erkennen(audio, sr, file_ext=_file_ext)  # type: ignore[assignment]
        return self._aufbereiten(raw)  # type: ignore[arg-type]

    def leite_phasen_ab(self, ketten_ergebnis: KettenErgebnis) -> ChainPhasePlan:
        """Leitet §2.46-konformen Pflicht-Phasenplan aus der Trägerkette ab.

        §2.46 Carrier-Chain-Inversion: Container-Träger zuerst bearbeiten, dann
        Zwischenstufen, zuletzt Ursprungsträger — invers zur Aufnahme-Reihenfolge.
        Additive Phasen (Bandbreiten-Erweiterung, Harmonik) IMMER nach allen
        subtraktiven Phasen (§2.46 Stufe-4-vor-5-Invariante).

        Als Komplement zu §6.2a Material-Pflicht-Phasen: erzeugt kettenbasierte
        Pflicht-Phasen unabhängig vom DefectScanner-Score.

        Args:
            ketten_ergebnis: KettenErgebnis von TontraegerketteDenker.analysiere().

        Returns:
            ChainPhasePlan mit must_have_phases in korrekter Inversions-Reihenfolge.
        """
        # §2.46 Inversion: letztes Glied (Container) → erstes Glied (Ursprung)
        if ketten_ergebnis is None:
            return ChainPhasePlan(must_have_phases=[], additive_phases=[], chain_string="", stage_count=0)
        if not getattr(ketten_ergebnis, "glieder", None):
            return ChainPhasePlan(
                must_have_phases=[],
                additive_phases=[],
                chain_string=str(getattr(ketten_ergebnis, "chain_string", "")),
                stage_count=0,
            )
        glieder_inverted = list(reversed(ketten_ergebnis.glieder))

        seen: set[str] = set()
        subtractive: list[str] = []
        additive: list[str] = []

        for glied in glieder_inverted:
            for phase in glied.recommended_phases:
                if phase in seen:
                    continue
                seen.add(phase)
                # Klassifikation: additiv (Energie-Ergänzung) oder subtraktiv
                is_additive = any(phase.startswith(pfx) for pfx in _ADDITIVE_PHASE_PREFIXES)
                if is_additive:
                    additive.append(phase)
                else:
                    subtractive.append(phase)

        # §2.46: subtraktive Phasen vor additiven
        must_have = subtractive + additive

        logger.debug(
            "TontraegerketteDenker.leite_phasen_ab(): %s → %d Pflicht-Phasen (%d subtraktiv, %d additiv, %d Stufen)",
            ketten_ergebnis.chain_string,
            len(must_have),
            len(subtractive),
            len(additive),
            ketten_ergebnis.generation_count,
        )

        return ChainPhasePlan(
            must_have_phases=must_have,
            additive_phases=additive,
            chain_string=ketten_ergebnis.chain_string,
            stage_count=ketten_ergebnis.generation_count,
        )

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _get_detector(self) -> object:
        """Liefert den (lazy-initialisierten) MediumDetector (thread-sicher)."""
        if self._detector is None:
            with self._detector_lock:
                if self._detector is None:
                    from backend.core.forensics.medium_detector import MediumDetector  # lazy import

                    self._detector = MediumDetector()
                    logger.debug("MediumDetector lazy-initialisiert.")
        return self._detector

    def _erkennen(self, audio: np.ndarray, sr: int, *, file_ext: str = "") -> dict[str, Any]:
        """Ruft MediumDetector.detect() auf und normalisiert das Ergebnis auf dict.

        MediumDetector.detect() gibt ein MediumDetectionResult-Dataclass zurück.
        _aufbereiten() erwartet ein dict mit 'detected_media', 'is_multi_generation'
        und 'confidence'. Diese Methode übersetzt das Objekt auf das erwartete Format.
        """
        try:
            detector = self._get_detector()
            result = detector.detect(audio, sr, file_ext=file_ext)  # type: ignore[attr-defined]

            # MediumDetectionResult auf dict normalisieren
            if hasattr(result, "as_dict"):
                raw: dict[str, Any] = result.as_dict()
                # transfer_chain → detected_media (list[tuple[str, float]])
                chain: list[str] = raw.get("transfer_chain", [])
                conf: float = float(raw.get("confidence", 0.5))
                # Use per-link confidences when available (same length as chain).
                # Fallback to global confidence for every link if not present.
                per_link: list[float] = raw.get("medium_confidences", [])
                if len(per_link) == len(chain):
                    raw["detected_media"] = list(zip(chain, per_link))
                else:
                    raw["detected_media"] = [(m, conf) for m in chain]
                return raw
            # Fallback: dict wurde direkt zurückgegeben (Legacy)
            if isinstance(result, dict):
                return result
            logger.warning("MediumDetector.erkennen() gab unbekannten Typ zurück: %s", type(result))
            return {}
        except Exception as exc:
            logger.warning("MediumDetector fehlgeschlagen: %s", exc)
            return {}

    def _aufbereiten_from_cached(self, cached_medium_result: object) -> KettenErgebnis:
        """Baut KettenErgebnis aus einem gecachten MediumDetectionResult.

        §2.47a: Wird aufgerufen wenn cached_medium_result übergeben wurde,
        um detect() nicht erneut aufzurufen.
        """
        # Attribute mit Multi-Fallback (wie UV3 Zeile 1564)
        primary = str(
            getattr(cached_medium_result, "primary_material", None)
            or getattr(cached_medium_result, "material_type", None)
            or getattr(cached_medium_result, "material", None)
            or "unknown"
        )
        conf = float(getattr(cached_medium_result, "confidence", 0.5))

        # transfer_chain bevorzugen, fallback auf single primary
        chain: list[str] = []
        raw_chain = getattr(cached_medium_result, "transfer_chain", None)
        if raw_chain and isinstance(raw_chain, (list, tuple)) and len(raw_chain) >= 1:
            chain = [str(c) for c in raw_chain]
        else:
            chain = [primary]

        # Normiertes raw-dict für _aufbereiten() erzeugen
        per_link: list[float] = getattr(cached_medium_result, "medium_confidences", []) or []
        if len(per_link) == len(chain):
            detected_media = list(zip(chain, per_link))
        else:
            detected_media = [(m, conf) for m in chain]

        raw: dict[str, Any] = {
            "primary_material": primary,
            "confidence": conf,
            "transfer_chain": chain,
            "detected_media": detected_media,
            "is_multi_generation": len(chain) >= 2,
        }
        return self._aufbereiten(raw)

    def _aufbereiten(self, raw: dict[str, Any]) -> KettenErgebnis:
        """Wandelt das Rohresultat des MediumDetectors in ein KettenErgebnis um.

        Temporal ordering:
            `detected_media` kommt bereits chronologisch geordnet an (Quelle → Container):
            entweder aus `_erkennen()` (abgeleitet von `MediumDetectionResult.transfer_chain`,
            das der MediumDetector selbst mit vollem Signal-Kontext sortiert) oder aus einem
            gecachten Ergebnis (§2.47a Direct Handover — bereits von der Pre-Analysis
            geordnet). Ein zusätzlicher Re-Sort hier nach der simplen `_MEDIUM_ORDER`-Ära-
            Heuristik würde diese bereits korrekte, kontextreichere Ordnung überschreiben
            und in Sonderfällen (z. B. Vinyl-zu-Band-Überspielung) verfälschen — die
            gegebene Reihenfolge wird deshalb unverändert übernommen.

        Complexity formula:
            Gegeben Gewichte w_i aus _COMPLEXITY_WEIGHT:
            complexity = max(0, min(1, 1 − ∏(1 − w_i)))
        """
        # --- 1. detected_media extrahieren ---
        detected_media: list[tuple[str, float]] = raw.get("detected_media", [])
        is_multi = bool(raw.get("is_multi_generation", len(detected_media) >= 2))
        raw_confidence: float = float(raw.get("confidence", 0.5))
        confidence = float(np.clip(raw_confidence, 0.0, 1.0))
        if not math.isfinite(confidence):
            confidence = 0.5

        # Fallback: wenn detected_media leer, aber 'type' vorhanden
        if not detected_media and raw.get("type"):
            medium_type = str(raw["type"])
            detected_media = [(medium_type, confidence)]

        # --- 2. Gegebene chronologische Reihenfolge übernehmen (kein Re-Sort) ---
        chain_sorted = list(detected_media)

        # --- 3. Kettenglieder aufbauen ---
        glieder: list[KettenGlied] = []
        for pos, (medium, score) in enumerate(chain_sorted):
            safe_score = float(np.clip(score if math.isfinite(score) else 0.5, 0.0, 1.0))
            glied = KettenGlied(
                medium=medium,
                position=pos,
                score=safe_score,
                degradation_type=_DEGRADATION.get(medium, "Unbekannte Degradation"),
                recommended_phases=_PHASE_MAP.get(medium, []),
                label=_LABEL.get(medium, medium),
            )
            glieder.append(glied)
            logger.debug("Kettenglied %d: %s (Wert %.2f)", pos, medium, safe_score)

        # --- 4. Kettenliste & Strings ---
        chain: list[str] = [g.medium for g in glieder]

        labels = [_LABEL.get(m, m) for m in chain]
        chain_string = " → ".join(labels) if labels else "Unbekannt"

        # Kein Signal / keine Kette erkannt
        if not chain:
            chain = [str(raw.get("type", "unknown"))]
            chain_string = _LABEL.get(chain[0], chain[0])

        original_medium = chain[0] if chain else "unknown"
        primary_medium = chain[-1] if chain else "unknown"

        # --- 5. Gemeinsame Phasen (dedupliziert, Reihenfolge erhalten) ---
        seen_phases: set[str] = set()
        combined_phases: list[str] = []
        for glied in glieder:
            for phase in glied.recommended_phases:
                if phase not in seen_phases:
                    combined_phases.append(phase)
                    seen_phases.add(phase)

        # --- 6. Kettenkomplexität: 1 − ∏(1 − w_i) ---
        product = 1.0
        for m in chain:
            w = _COMPLEXITY_WEIGHT.get(m, 0.5)
            product *= max(0.0, 1.0 - w)
        chain_complexity = float(np.clip(1.0 - product, 0.0, 1.0))
        if not math.isfinite(chain_complexity):
            chain_complexity = 0.5

        # --- 7. Spektrale Evidenz ---
        spectral_evidence: dict[str, Any] = {}
        for key in ("wow_strength", "flutter_strength", "clicks_per_sec"):
            val = raw.get(key)
            if val is not None:
                fval = float(val) if math.isfinite(float(val)) else 0.0
                spectral_evidence[key] = fval

        # --- 8. Reasoning (Deutsch) ---
        reasoning = self._begründung(
            chain=chain,
            glieder=glieder,
            is_multi=is_multi,
            chain_complexity=chain_complexity,
            confidence=confidence,
            spectral_evidence=spectral_evidence,
        )

        return KettenErgebnis(
            chain=chain,
            chain_string=chain_string,
            is_multi_generation=is_multi,
            generation_count=len(chain),
            primary_medium=primary_medium,
            original_medium=original_medium,
            glieder=glieder,
            combined_phases=combined_phases,
            chain_complexity=chain_complexity,
            confidence=confidence,
            spectral_evidence=spectral_evidence,
            reasoning=reasoning,
        )

    def _begründung(
        self,
        chain: list[str],
        glieder: list[KettenGlied],
        is_multi: bool,
        chain_complexity: float,
        confidence: float,
        spectral_evidence: dict[str, Any],
    ) -> str:
        """Erstellt einen laienverständlichen deutschen Erklärungstext.

        Args:
            chain:             Zeitlich geordnete Medien-Liste.
            glieder:           Aufgebaute KettenGlied-Objekte.
            is_multi:          True wenn mehrere Medien erkannt.
            chain_complexity:  Berechnete Komplexität ∈ [0, 1].
            confidence:        Erkennungs-Konfidenz ∈ [0, 1].
            spectral_evidence: Roh-Spektralmerkmale.

        Returns:
            Formatierter Begründungstext (Deutsch).
        """
        parts: list[str] = []

        if len(chain) == 1:
            label = _LABEL.get(chain[0], chain[0])
            parts.append(
                f"Es wurde ein einzelner Tonträger erkannt: {label}. "
                f"Die Restaurierung wird auf die typischen Eigenschaften "
                f"dieses Formats abgestimmt."
            )
        else:
            labels = [_LABEL.get(m, m) for m in chain]
            chain_str = " → ".join(labels)
            parts.append(f"Es wurde eine {len(chain)}-stufige Übertragungskette erkannt: {chain_str}.")
            parts.append("Jede Übertragungsstufe hat typische Klangspuren hinterlassen, die separat behandelt werden.")

        # Degradationsbeschreibungen
        for glied in glieder:
            parts.append(f"• {glied.label}: {glied.degradation_type}.")

        # Komplexitätsbewertung
        if chain_complexity >= 0.85:
            parts.append(
                "Die Kettenkomplexi­tät ist sehr hoch — eine besonders sorgfältige, "
                "stufenweise Restaurierung ist erforderlich."
            )
        elif chain_complexity >= 0.60:
            parts.append("Die Kettenkomplexi­tät ist moderat — mehrere Restaurierungs-Phasen werden benötigt.")
        else:
            parts.append("Die Kettenkomplexi­tät ist gering — gezielte Korrekturen genügen.")

        # Spektrale Belege
        wow = spectral_evidence.get("wow_strength", 0.0)
        flutter = spectral_evidence.get("flutter_strength", 0.0)
        if wow > 0.01 or flutter > 0.005:
            parts.append(
                f"Messbarer Pitch-Jitter (Wow {wow:.3f}, Flutter {flutter:.3f}) "
                f"deutet auf eine Kassette oder ein Tonband hin."
            )

        # Phasen-Überblick
        if glieder:
            all_phases = [p for g in glieder for p in g.recommended_phases]
            parts.append(f"Insgesamt werden {len(set(all_phases))} Restaurierungs-Phasen empfohlen.")

        # Konfidenz-Hinweis
        conf_pct = int(confidence * 100)
        parts.append(f"Gesamt-Konfidenz der Kettenerkennung: {conf_pct} %.")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Singleton (§3.2 — Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: TontraegerketteDenker | None = None
_lock = threading.Lock()


def get_tontraegerkette_denker() -> TontraegerketteDenker:
    """Liefert den thread-sicheren Singleton des TontraegerketteDenkers.

    Implementiert Double-Checked Locking nach §3.2.

    Returns:
        Gemeinsame TontraegerketteDenker-Instanz.
    """
    global _instance
    if _instance is None:  # Schnellpfad ohne Lock
        with _lock:
            if _instance is None:  # Zweiter Check unter Lock (Race-Condition-sicher)
                _instance = TontraegerketteDenker()
                logger.debug("TontraegerketteDenker-Singleton erzeugt.")
    return _instance


def analysiere_kette(audio: np.ndarray, sr: int) -> KettenErgebnis:
    """Convenience-Wrapper: Analysiert die Tonträgerkette ohne Klassen-Zwang.

    Args:
        audio: Float32-Array ∈ [-1, 1], mono oder stereo.
        sr:    Abtastrate in Hz.

    Returns:
        KettenErgebnis mit zeitlich geordneter Kette und Phasen-Empfehlungen.
    """
    return get_tontraegerkette_denker().analysiere(audio, sr)
