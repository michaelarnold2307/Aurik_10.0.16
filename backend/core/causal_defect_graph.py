"""
CausalDefectGraph — Weltspitzen-Differenzierer #1
==================================================

Modelliert kausale Abhängigkeiten zwischen Tonträger-Defekten.
Root-Cause-Defekte werden vor ihren Symptomen repariert.

Kernthese: Viele scheinbare Defekte sind Symptome tiefer liegender Ursachen.
Aurik repariert in kausaler Reihenfolge — kein anderes Programm tut das.

Bekannte kausale Ketten:
    WOW             → BANDWIDTH_LOSS   (Kammfilter täuscht HF-Verlust vor)
    FLUTTER         → PITCH_DRIFT      (Motorinstabilität → Tonhöhenschwankung)
  DROPOUTS        → CLICKS           (Dropout-Flanken erzeugen Impuls-Artefakte)
  CRACKLE         → CLICKS           (Starke Crackling-Bursts erzeugen Click-Transienten)
  CRACKLE         → HIGH_FREQ_NOISE  (Oberflächen-Crackle erhöht breitbandigen HF-Rauschboden)
  PRINT_THROUGH   → REVERB_EXCESS    (Pre-Echo wirkt wie übermäßiger Raumhall)
  HUM             → LOW_FREQ_RUMBLE  (Brumm-Obertöne maskieren Rumpelrauschen)
  CLIPPING        → DIGITAL_ARTIFACTS (Clipping erzeugt Intermodulationsverzerrung)
  JITTER_ARTIFACTS → DIGITAL_ARTIFACTS
  QUANTIZATION_NOISE → HIGH_FREQ_NOISE
  PHASE_ISSUES    → STEREO_IMBALANCE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.core.defect_scanner import DefectScore, DefectType

logger = logging.getLogger(__name__)


@dataclass
class CausalNode:
    """Ein Knoten im kausalen Defektgraph mit Ursache-Wirkung-Metadaten."""

    defect_score: DefectScore
    caused_by: list[DefectType] = field(default_factory=list)
    """Erkannte Root-Causes dieses Defekts (Teilmenge der erkannten Defekte)."""
    causes: list[DefectType] = field(default_factory=list)
    """Effekte/Symptome, die dieser Defekt verursacht (Teilmenge erkannt)."""
    is_phantom: bool = False
    """True wenn dieser Defekt nur ein Symptom eines anderen erkannten Defekts ist."""
    causal_note: str = ""
    """Prosaerklärung der kausalen Rolle dieses Defekts."""


class CausalDefectGraph:
    """
    Kausaler Defektgraph für Aurik 10.0.0.

    Analysiert Ursache-Wirkung-Beziehungen zwischen erkannten Defekten und
    liefert eine topologisch geordnete Reparatursequenz: Root causes zuerst,
    Symptome danach — weil das Reparieren eines Symptoms ohne Root-Cause-Fix
    entweder ineffektiv ist oder die Restaurationsqualität aktiv verringert.

    Anwendungsbeispiel:
        graph = CausalDefectGraph()
        ordered = graph.resolve_causal_order(defect_result.get_top_defects(n=10))
        # ordered: Root causes zuerst, Symptome hinten
    """

    # Gerichteter azyklischer Graph: Ursache → [verursachte Symptome]
    CAUSAL_EDGES: dict[DefectType, list[DefectType]] = {
        DefectType.WOW: [
            DefectType.BANDWIDTH_LOSS,  # Kammfilter täuscht HF-Verlust vor
            DefectType.PITCH_DRIFT,  # Motorinstabilität → Tonhöhenschwankung
            DefectType.STEREO_IMBALANCE,  # Flutter kann Kanäle ungleichmäßig treffen
        ],
        DefectType.FLUTTER: [
            DefectType.PITCH_DRIFT,  # Mechanische Vibration → Tonhöhenschwankung (Hochfrequent)
        ],
        DefectType.DROPOUTS: [
            DefectType.CLICKS,  # Dropout-Flanken = Impuls-Artefakte
        ],
        DefectType.CRACKLE: [
            DefectType.CLICKS,  # Schwere Crackle-Bursts → Click-artige Transienten
            DefectType.HIGH_FREQ_NOISE,  # Vinyl/Shellac-Oberfläche → erhöhter HF-Rauschboden
        ],
        DefectType.PRINT_THROUGH: [
            DefectType.REVERB_EXCESS,  # Pre-Echo = scheinbarer übermäßiger Raumhall
        ],
        DefectType.HUM: [
            DefectType.LOW_FREQ_RUMBLE,  # Brumm-Obertöne überlagern Rumpelrauschen
        ],
        DefectType.LOW_FREQ_RUMBLE: [
            DefectType.WOW,  # Tonarm-Resonanz erzeugt Gleichlauf-Artefakte
            DefectType.FLUTTER,  # Tonarm-Resonanz erzeugt Flutter-Artefakte
        ],
        DefectType.CLIPPING: [
            DefectType.DIGITAL_ARTIFACTS,
            DefectType.DYNAMIC_COMPRESSION_EXCESS,
            DefectType.HIGH_FREQ_NOISE,  # Inter-Modulations-Verzerrung erzeugt HF-Rauschen
        ],
        DefectType.COMPRESSION_ARTIFACTS: [
            DefectType.DYNAMIC_COMPRESSION_EXCESS,
        ],
        DefectType.QUANTIZATION_NOISE: [
            DefectType.HIGH_FREQ_NOISE,  # Quantisierungsfehler → HF-Rauschboden
        ],
        DefectType.JITTER_ARTIFACTS: [
            DefectType.DIGITAL_ARTIFACTS,
            DefectType.HIGH_FREQ_NOISE,
        ],
        DefectType.BANDWIDTH_LOSS: [
            # HF-Restauration ohne korrekte Ursachenkenntnis → Rauschen
            DefectType.HIGH_FREQ_NOISE,
        ],
        DefectType.PHASE_ISSUES: [
            DefectType.STEREO_IMBALANCE,  # Phasenfehler → wahrgenommene Kanaltrennung
        ],
        DefectType.DC_OFFSET: [
            DefectType.CLIPPING,  # DC-Offset reduziert verfügbaren Headroom
            DefectType.LOW_FREQ_RUMBLE,
        ],
    }

    def __init__(self) -> None:
        # Invertierter Graph: Symptom → [Ursachen]
        self._symptom_to_causes: dict[DefectType, list[DefectType]] = {}
        self._incomplete_sort_warned = False
        for cause, effects in self.CAUSAL_EDGES.items():
            for effect in effects:
                self._symptom_to_causes.setdefault(effect, []).append(cause)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def resolve_causal_order(self, detected: list[DefectScore]) -> list[DefectScore]:
        """
        Topologische Sortierung nach kausaler Priorität (Kahn's Algorithmus).

        Root-Cause-Defekte kommen zuerst; Symptome werden erst repariert,
        nachdem ihre Ursachen behoben wurden.

        Args:
            detected: Liste der erkannten Defekte (aus DefectScanner).

        Returns:
            Kausal geordnete Liste — Root causes vorne, Symptome hinten.
        """
        if not detected:
            return []

        detected_types: set[DefectType] = {d.defect_type for d in detected}
        score_map: dict[DefectType, float] = {d.defect_type: d.severity for d in detected}

        # In-degree: Anzahl erkannter Ursachen pro Defekt
        in_degree: dict[DefectType, int] = {
            d.defect_type: sum(1 for c in self._symptom_to_causes.get(d.defect_type, []) if c in detected_types)
            for d in detected
        }

        # Startwarteschlange: Defekte ohne erkannte Ursachen (echte Root causes)
        queue: list[DefectType] = sorted(
            [dt for dt, deg in in_degree.items() if deg == 0],
            key=lambda dt: score_map.get(dt, 0),
            reverse=True,
        )

        ordered: list[DefectType] = []
        while queue:
            dt = queue.pop(0)
            ordered.append(dt)

            for effect in self.CAUSAL_EDGES.get(dt, []):
                if effect not in in_degree:
                    continue
                in_degree[effect] -= 1
                if in_degree[effect] == 0:
                    # Einfügen nach Schweregrad sortiert
                    idx = next(
                        (i for i, q in enumerate(queue) if score_map.get(q, 0) < score_map.get(effect, 0)),
                        len(queue),
                    )
                    queue.insert(idx, effect)

        # Defekte ohne Graph-Kante → nach Schweregrad ans Ende
        remaining = sorted(
            [dt for dt in detected_types if dt not in ordered],
            key=lambda dt: score_map.get(dt, 0),
            reverse=True,
        )
        ordered.extend(remaining)

        # Build lookup — keep highest-severity entry when duplicates exist.
        # DefectScanner can theoretically emit the same DefectType more than once
        # (e.g. two scoring passes).  A plain dict comprehension would silently
        # drop all but the last entry, making len(result) < len(detected) and
        # triggering the fallback on every call.  We deduplicate *before* the
        # length check so the kausal order is actually used.
        score_lookup: dict = {}
        for d in detected:
            if d.defect_type not in score_lookup or d.severity > score_lookup[d.defect_type].severity:
                score_lookup[d.defect_type] = d
        # Work with deduplicated list from here on
        detected_dedup = list(score_lookup.values())

        result = [score_lookup[dt] for dt in ordered if dt in score_lookup]

        if len(result) != len(detected_dedup):
            if not self._incomplete_sort_warned:
                logger.warning("Kausale Sortierung unvollständig — Ersatzpfad auf Schweregrad.")
                self._incomplete_sort_warned = True
            else:
                logger.debug("Kausale Sortierung weiterhin unvollständig — Ersatzpfad auf Schweregrad.")
            return sorted(detected_dedup, key=lambda d: d.severity, reverse=True)

        logger.debug(
            "Kausale Reparaturreihenfolge: %s",
            " → ".join(d.defect_type.value for d in result),
        )
        return result

    def build(self, detected: list[DefectScore]) -> list[CausalNode]:
        """Baut annotierte CausalNode-Liste für alle erkannten Defekte."""
        detected_types: set[DefectType] = {d.defect_type for d in detected}
        nodes: list[CausalNode] = []

        for score in detected:
            dt = score.defect_type
            causes_here = [c for c in self._symptom_to_causes.get(dt, []) if c in detected_types]
            effects_here = [e for e in self.CAUSAL_EDGES.get(dt, []) if e in detected_types]
            is_phantom = bool(causes_here)

            if is_phantom:
                cause_names = ", ".join(c.value for c in causes_here)
                note = f"Symptom von [{cause_names}] — wird nach Root-Cause-Reparatur neu bewertet"
            elif effects_here:
                effect_names = ", ".join(e.value for e in effects_here)
                note = f"Root Cause → verursacht [{effect_names}]"
            else:
                note = "Primär-Defekt (keine erkannten Abhängigkeiten)"

            nodes.append(
                CausalNode(
                    defect_score=score,
                    caused_by=causes_here,
                    causes=effects_here,
                    is_phantom=is_phantom,
                    causal_note=note,
                )
            )

        return nodes

    def get_phantom_defects(self, detected: list[DefectScore]) -> list[DefectType]:
        """Gibt Defekte zurück, die nur Symptome anderer erkannter Defekte sind."""
        return [n.defect_score.defect_type for n in self.build(detected) if n.is_phantom]

    def explain(self, detected: list[DefectScore]) -> str:
        """
        Prosaerklärung der kausalen Zusammenhänge — für Audit-Report.

        Returns:
            Mehrzeiliger String mit allen kausalen Beziehungen.
        """
        nodes = self.build(detected)
        ordered = self.resolve_causal_order(detected)
        ordered_types = [d.defect_type for d in ordered]

        lines = [
            "=== Kausale Defekt-Analyse (Aurik 10.0.0 CausalDefectGraph) ===",
            f"Erkannte Defekte: {len(detected)}",
            f"Phantomdefekte (reine Symptome): {sum(1 for n in nodes if n.is_phantom)}",
            "",
            "Reparaturreihenfolge (kausal → topologisch sortiert):",
        ]
        for i, dt in enumerate(ordered_types, 1):
            node = next(n for n in nodes if n.defect_score.defect_type == dt)
            prefix = "  PHANTOM" if node.is_phantom else "  ROOT   "
            lines.append(f"  {i:2d}. {prefix} [{dt.value}] sev={node.defect_score.severity:.2f} — {node.causal_note}")

        return "\n".join(lines)
