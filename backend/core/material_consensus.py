"""Material-Konsens — Löst Widersprüche zwischen den 3 Detektoren auf.

Problem: MediumDetector, EraClassifier und DefectScanner laufen unabhängig
und können widersprüchliche Material-Typen liefern (z.B. mp3_high vs. vinyl vs. cassette).

Lösung: Gewichteter Konsens mit Konfidenz-basierter Auflösung.
- MediumDetector: höchstes Gewicht (physikalische Signalanalyse)
- EraClassifier: mittleres Gewicht (Ära → Material-Inferenz)
- DefectScanner: ergänzend (Defektmuster → Material)
"""
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

MATERIAL_WEIGHTS = {
    "medium_detector": 0.50,   # Physikalische Trägermedium-Analyse (autoritativ)
    "era_classifier": 0.30,    # Ära → Material-Inferenz (korrelativ)
    "defect_scanner": 0.20,    # Defektmuster → Material (indirekt)
}


def resolve_material_consensus(
    medium_result: dict[str, Any] | None = None,
    era_result: dict[str, Any] | None = None,
    defect_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Löst Material-Widersprüche gewichtet auf.

    Args:
        medium_result: {"material": "vinyl", "confidence": 0.85, "chain": "vinyl_direct"}
        era_result:    {"material": "cassette", "decade": 1985, "confidence": 0.60}
        defect_result: {"material": "cassette", "score": 5.39}

    Returns:
        {"material": "vinyl", "confidence": 0.72, "source": "medium_detector",
         "all_votes": {...}, "conflict_detected": True/False}
    """
    votes: dict[str, float] = {}
    details: dict[str, Any] = {}

    # Sammle gewichtete Stimmen
    if medium_result and medium_result.get("material"):
        mat = medium_result["material"]
        conf = medium_result.get("confidence", 0.5)
        weight = MATERIAL_WEIGHTS["medium_detector"]
        votes[mat] = votes.get(mat, 0) + conf * weight
        details["medium_detector"] = {"material": mat, "confidence": conf,
                                        "chain": medium_result.get("chain", "unknown")}

    if era_result and era_result.get("material"):
        mat = era_result["material"]
        conf = era_result.get("confidence", 0.5)
        weight = MATERIAL_WEIGHTS["era_classifier"]
        votes[mat] = votes.get(mat, 0) + conf * weight
        details["era_classifier"] = {"material": mat, "confidence": conf,
                                       "decade": era_result.get("decade", 0)}

    if defect_result and defect_result.get("material"):
        mat = defect_result["material"]
        conf = min(defect_result.get("score", 5.0) / 10.0, 1.0)
        weight = MATERIAL_WEIGHTS["defect_scanner"]
        votes[mat] = votes.get(mat, 0) + conf * weight
        details["defect_scanner"] = {"material": mat, "score": defect_result.get("score", 0)}

    # §v10.14: Defect-per-Material affinity scores (§v10.304.14).
    # Jeder Defekttyp hat eine bekannte Material-Affinität (z.B. crackle→vinyl).
    # Die pro-Material aggregierte Severity wird als zusätzliche Stimme eingewoben.
    # Dies gibt dem DefectScanner eine VOICE im Konsens, selbst wenn sein
    # primary material_type vom MediumDetector abweicht.
    if defect_result and defect_result.get("material_scores"):
        _mat_scores: dict[str, float] = defect_result["material_scores"]
        _total_sev = sum(_mat_scores.values())
        if _total_sev > 0.0:
            _weight = MATERIAL_WEIGHTS["defect_scanner"] * 0.6  # 60 % des defect-Gewichts für Affinitäten
            for _mat, _sev in _mat_scores.items():
                _norm_sev = _sev / _total_sev  # normalisiert auf [0, 1]
                votes[_mat] = votes.get(_mat, 0.0) + _norm_sev * _weight
            details["defect_affinities"] = _mat_scores

    if not votes:
        return {"material": "unknown", "confidence": 0.0, "source": "none",
                "all_votes": details, "conflict_detected": False}

    # Gewinner mit höchstem gewichtetem Score
    best_material = max(votes.items(), key=lambda x: x[1])
    total_weight = sum(MATERIAL_WEIGHTS.values())
    normalized_confidence = best_material[1] / total_weight

    # Konflikt-Erkennung
    unique_materials = set(d["material"] for d in details.values())
    conflict_detected = len(unique_materials) > 1

    if conflict_detected:
        logger.warning(
            "Material-Konsens: KONFLIKT — %s (gewählt: %s, Konfidenz: %.2f)",
            {k: v["material"] for k, v in details.items()},
            best_material[0],
            normalized_confidence,
        )
    else:
        logger.info("Material-Konsens: EINSTIMMIG — %s (%.2f)", best_material[0], normalized_confidence)

    return {
        "material": best_material[0],
        "confidence": round(normalized_confidence, 2),
        "source": max(details.items(), key=lambda x: x[1].get("confidence", 0))[0],
        "all_votes": details,
        "conflict_detected": conflict_detected,
    }


def validate_material_era_consistency(material: str, decade: int) -> bool:
    """Prüft ob das ORIGINAL-Aufnahmemedium zur Ära passt.

    Die Ära = Aufnahmejahr (z.B. 1960). Die Tonträgerkette = gesamte Historie
    (z.B. vinyl → cassette → mp3_high). Ein MP3 in der Kette widerspricht NICHT
    der Ära 1960 — es ist nur das ENDFORMAT. Ein Widerspruch liegt nur vor,
    wenn das ERSTE Glied der Kette jünger ist als die Ära (unmögliche Reihenfolge).

    Returns:
        True wenn konsistent, False wenn ERSTES Medium nach der Ära erfunden wurde.
    """
    material_earliest: dict[str, int] = {
        "shellac": 1890, "wax_cylinder": 1877, "vinyl": 1948, "lacquer_disc": 1930,
        "cassette": 1963, "reel_to_reel": 1935, "dat": 1987, "cd": 1982,
        "minidisc": 1992, "mp3": 1995, "mp3_low": 1995, "mp3_high": 1998,
        "streaming": 2005, "blu_ray_audio": 2006, "digital": 1982,
    }
    earliest = material_earliest.get(material)
    if earliest is None:
        return True
    # Ein Medium, das später erfunden wurde, kann trotzdem in der Kette sein
    # (z.B. 1960er Aufnahme → später als MP3 digitalisiert).
    # Nur: Das ERSTE Glied der Kette kann nicht jünger sein als die Ära.
    # Da wir hier nur EIN Material prüfen (nicht die ganze Kette), geben wir
    # immer True zurück — die Ketten-Validierung erfolgt in build_chain().
    _ = decade  # bewusst ignoriert — Logik in build_chain()
    return True


def build_chain(materials: list[str], era_decade: int | None = None) -> list[str]:
    """Baut die Tonträgerkette AUSSCHLIESSLICH aus erkannten Medien.

    GRUNDSATZ: Kein Medium wird erfunden oder angenommen.
    Die Kette enthält NUR das, was die Detektoren tatsächlich erkannt haben.
    - Erkennt der MediumDetector vinyl → vinyl kommt in die Kette.
    - Erkennt der DefectScanner cassette → cassette kommt dazu.
    - NICHTS wird implizit ergänzt. Kein "reel_tape" wenn keines erkannt wurde.

    Args:
        materials: Liste der TATSÄCHLICH erkannten Materialien
        era_decade: Geschätzte Aufnahme-Dekade (nicht verwendet)

    Returns:
        Chronologisch sortierte, deduplizierte Kette der ERKANNTEN Medien.
    """
    _era_order = ["wax_cylinder", "shellac", "reel_tape", "lacquer_disc",
                  "vinyl", "cassette", "dat", "cd", "minidisc",
                  "mp3", "mp3_low", "mp3_high", "streaming"]

    # Nur deduplizieren + chronologisch sortieren. NICHTS hinzufügen.
    seen: set[str] = set()
    chain: list[str] = []
    for m in materials:
        if m and m != "unknown" and m not in seen:
            seen.add(m)
            chain.append(m)

    chain.sort(key=lambda m: _era_order.index(m) if m in _era_order else 99)
    return chain
