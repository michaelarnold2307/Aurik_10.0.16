"""§v10.15 Narrative Engine — Story-Driven Restoration Communication
=================================================================
Erzeugt eine fortlaufende, menschenähnliche Erzählung des
Restaurationsprozesses. Keine Checkliste — eine Geschichte.

Design-Prinzipien:
1. Kontext-bewusst: Material, Ära, Genre, Defekte fließen ein
2. Narrative Progression: Analyse → Reparatur → Vollendung
3. Sprach-Varianz: Kein Satz gleicht dem anderen
4. Personifiziert: Aurik als achtsamer Restaurator mit Charakter
5. Technisch präzise, poetisch ausgedrückt

Nutzung:
    engine = NarrativeEngine(material="cassette", era=1970, genre="Deutscher Schlager")
    text = engine.phase_narrative("phase_01_click_removal", defects_found=129)
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


class NarrativeEngine:
    """Erzeugt narrative Beschreibungen des Restaurationsprozesses."""

    # ── Charakter-Stimme (variiert pro Session) ─────────────────
    _PERSONAE = [
        {
            "name": "achtsam",
            "intro_style": ["Ich betrachte", "Lass mich sehen", "Ein Blick zeigt"],
            "action_style": ["behutsam", "mit ruhiger Hand", "konzentriert", "Schritt für Schritt"],
            "discovery": ["entdecke ich", "zeigt sich", "offenbart sich", "wird sichtbar"],
            "repair": ["löse ich", "bessere ich aus", "kümmere ich mich um", "restauriere ich"],
            "admire": ["Schön", "Gut", "Wunderbar", "Ausgezeichnet"],
        },
        {
            "name": "enthusiastisch",
            "intro_style": ["Ah,", "Interessant —", "Spannend:", "Oh, hier"],
            "action_style": ["entschlossen", "mit Feingefühl", "energiegeladen", "präzise"],
            "discovery": ["fällt mir auf", "springt ins Auge", "macht sich bemerkbar", "verrät sich"],
            "repair": ["bringe ich in Ordnung", "stelle ich wieder her", "repariere ich", "glätte ich"],
            "admire": ["Herrlich!", "Großartig!", "Perfekt!", "Sehr schön!"],
        },
        {
            "name": "erzählerisch",
            "intro_style": [
                "Die Aufnahme erzählt",
                "Man hört bereits",
                "Es wird deutlich",
                "Schon beim ersten Hinhören",
            ],
            "action_style": [
                "mit der Ruhe eines Archivars",
                "so wie ein Töpfer den Ton",
                "in der Tradition alter Meister",
                "ganz im Sinne des Originals",
            ],
            "discovery": ["dass hier", "wie sehr", "in welchem Maß", "auf welche Weise"],
            "repair": [
                "darf ich korrigieren",
                "kann ich heilen",
                "sollte behutsam eingegriffen werden",
                "möchte ich nachbessern",
            ],
            "admire": [
                "Das klingt schon viel besser.",
                "Der Charakter bleibt erhalten.",
                "Die Seele des Songs atmet auf.",
                "Man spürt den Unterschied.",
            ],
        },
    ]

    # ── Material-spezifische Metaphern ──────────────────────────
    _MATERIAL_VOCAB = {
        "cassette": {
            "nouns": ["das Magnetband", "die Kassette", "der Bandträger", "die Compact Cassette"],
            "verbs": ["spulen", "durchlaufen", "am Band entlang", "durch den Tapedeck"],
            "imagery": [
                "wie ein vertrautes Mixtape aus der Jugend",
                "mit dem warmen Bandklang der Siebziger",
                "dessen braune Hülle schon bessere Tage gesehen hat",
            ],
        },
        "vinyl": {
            "nouns": ["die Schallplatte", "die Rille", "der Vinylträger", "das schwarze Gold"],
            "verbs": ["drehen", "in der Rille gleiten", "unter der Nadel", "auf dem Plattenteller"],
            "imagery": [
                "wie eine gut gehütete Erstpressung",
                "mit dem unverkennbaren Vinyl-Knistern",
                "das so viele Wohnzimmerabende begleitet hat",
            ],
        },
        "reel_tape": {
            "nouns": ["das Tonband", "die Spule", "der Bandteller", "das Masterband"],
            "verbs": ["aufgezeichnet", "auf Band gebannt", "durch die Tonköpfe", "im Studio"],
            "imagery": [
                "wie eine Studio-Aufnahme aus bester Tradition",
                "mit den Spuren unzähliger Durchläufe",
                "das vielleicht nie für die Öffentlichkeit bestimmt war",
            ],
        },
        "unknown": {
            "nouns": ["die Aufnahme", "das Audiomaterial", "der Tonträger"],
            "verbs": ["erklingen", "hörbar werden", "im Klangbild erscheinen"],
            "imagery": ["mit seiner ganz eigenen Charakteristik"],
        },
    }

    # ── Ära-Atmosphäre ─────────────────────────────────────────
    _ERA_ATMOSPHERE = {
        1900: "aus den Kindertagen der Tonaufnahme",
        1920: "aus den wilden Zwanzigern",
        1930: "aus der Zeit der großen Orchester",
        1940: "aus den Kriegsjahren",
        1950: "aus der Rock'n'Roll-Ära",
        1960: "aus den Sechzigern — voller Experimentierfreude",
        1970: "aus den Siebzigern — dem Jahrzehnt des analogen Klangs",
        1980: "aus den Achtzigern — zwischen Analog und Digital",
        1990: "aus den Neunzigern",
        2000: "aus der Jahrtausendwende",
    }

    # ── Phasen-Erzählungen ──────────────────────────────────────
    _PHASE_NARRATIVES: dict[str, list[dict[str, Any]]] = {
        "phase_01_click_removal": [
            {
                "finding": "{n_found} winzige Klick-Schäden {discovery} — sie sitzen wie kleine Nadelstiche in der Zeit.",
                "action": "Jeden einzelnen {action} entfernt, ohne die musikalische Umgebung zu verletzen.",
                "result": "{admire} Die Transienten atmen jetzt frei.",
            },
            {
                "finding": "{discovery} {n_found} digitale Spitzen, die im Material stecken.",
                "action": "Wie ein Restaurator, der Staub von einem Gemälde pustet — {action} jeden Klick einzeln behandelt.",
                "result": "{n_found} Störungen sind verschwunden, der Klangfluss ist ungebrochen.",
            },
            {
                "finding": "{n_found} kleine Unreinheiten {discovery} im Klangbild.",
                "action": "{action}, wie ein Goldschmied, der jedes Korn einzeln fasst.",
                "result": "Die Oberfläche ist jetzt makellos.",
            },
        ],
        "phase_03_denoise": [
            {
                "finding": "Ein feiner Rauschschleier liegt über der Aufnahme.",
                "action": "Ich trenne behutsam das Rauschen vom Signal. {action}. Es ist, als würde man eine Fensterscheibe putzen — plötzlich sieht man hindurch.",
                "result": "Der Klang ist klarer, aber nicht steril. Die Wärme bleibt.",
            },
            {
                "finding": "Das Grundrauschen {discovery} — es gehört zur {imagery}, aber zu viel davon verdeckt die Details.",
                "action": "{action} ziehe ich einen transparenten Filter durch das Spektrum.",
                "result": "Die Musik steht jetzt im Vordergrund, nicht das Rauschen.",
            },
        ],
        "phase_04_eq_correction": [
            {
                "finding": "Die Klangbalance ist etwas aus dem Lot geraten.",
                "action": "Mit sanften EQ-Korrekturen {action} die ursprüngliche tonale Balance wiederhergestellt.",
                "result": "Die Stimme sitzt jetzt genau richtig im Mix.",
            },
        ],
        "phase_09_crackle_removal": [
            {
                "finding": "{n_found} Knisternester {discovery} in den leisen Passagen.",
                "action": "{action} jedes einzelne Nest aufgespürt und geglättet.",
                "result": "Die Stille zwischen den Noten ist wieder still.",
            },
        ],
        "phase_12_wow_flutter_fix": [
            {
                "finding": "Die Tonhöhe schwankt — ein typisches Zeichen für Bandlauf-Unruhe.",
                "action": "{action} den Gleichlauf stabilisiert. Wie ein Seiltänzer, der sein Gleichgewicht findet.",
                "result": "Die Musik schwingt jetzt gleichmäßig.",
            },
        ],
        "phase_19_de_esser": [
            {
                "finding": "Die Sibilanten — diese scharfen S-Laute — stechen etwas hervor.",
                "action": "Mit einem seidenweichen De-Esser {action} die Zischlaute gebändigt.",
                "result": "Die Stimme klingt geschmeidig, ohne an Präsenz zu verlieren.",
            },
        ],
        "phase_23_spectral_repair": [
            {
                "finding": "Im oberen Frequenzbereich fehlen Details — die Höhen sind stumpf.",
                "action": "{action} das Spektrum behutsam erweitert. Fehlende Obertöne rekonstruiert.",
                "result": "Der Glanz kehrt zurück. Man hört plötzlich Dinge, die vorher verborgen waren.",
            },
        ],
        "phase_29_tape_hiss_reduction": [
            {
                "finding": "Das charakteristische Bandrauschen — {imagery}.",
                "action": "{action} reduziere ich es, ohne die Höhen stumpf werden zu lassen.",
                "result": "Das Band atmet noch, aber es flüstert nur noch.",
            },
        ],
    }

    # ── Fallback-Narrative-Generator ────────────────────────────
    _FALLBACK_ACTIONS = [
        "analysiere ich das Spektrum",
        "untersuche ich die Struktur",
        "höre ich genau hin",
        "taste ich mich voran",
        "arbeite ich mich durch die Schichten",
        "folge ich den Spuren der Zeit",
        "reinige ich das Klangbild",
        "schärfe ich die Konturen",
        "glätte ich die Oberfläche",
        "bringe ich Ordnung ins Gefüge",
        "klopfe ich jede Frequenz behutsam ab",
        "dringe ich tiefer in die Aufnahme ein",
    ]

    def __init__(
        self,
        material: str = "unknown",
        era: int = 0,
        genre: str = "",
        defects: dict[str, Any] | None = None,
        song_title: str = "",
    ) -> None:
        self._material = str(material).lower().replace("-", "_").replace(" ", "_")
        self._era = era
        self._genre = genre
        self._defects = defects or {}
        self._title = song_title

        # Deterministic seed based on song — keeps persona consistent per song
        seed_int = int(
            hashlib.md5(f"{material}{era}{genre}{song_title}".encode(), usedforsecurity=False).hexdigest()[:8], 16
        )
        self._rng = random.Random(seed_int)

        # Pick persona
        self._persona = self._rng.choice(self._PERSONAE)
        self._phase_count: int = 0
        self._story_phase: str = "beginning"  # beginning → middle → end

        self._used_actions: list[str] = []
        self._used_images: list[str] = []

    # ── Public API ───────────────────────────────────────────────

    def opening(self, total_phases: int = 0) -> str:
        """Generate the opening narrative when restoration begins."""
        mat_vocab = self._MATERIAL_VOCAB.get(self._material, self._MATERIAL_VOCAB["unknown"])
        era_atmo = self._ERA_ATMOSPHERE.get((self._era // 10) * 10, "")

        parts: list[str] = []

        # First sentence: establish the scene
        if self._title:
            parts.append(f'"{self._title}" — eine Aufnahme{era_atmo and " " + era_atmo}.')
        elif era_atmo:
            parts.append(f"Eine Aufnahme {era_atmo}.")
        else:
            parts.append("Eine Aufnahme mit eigener Geschichte.")

        # Material context
        imagery = self._rng.choice(mat_vocab.get("imagery", ["mit Charakter"]))
        parts.append(imagery + ".")

        # What we notice
        intro = self._rng.choice(self._persona["intro_style"])
        parts.append(f"{intro} beginne ich mit der behutsamen Analyse.")

        if total_phases:
            parts.append(f"In {total_phases} Schritten werde ich die Zeit behutsam zurückdrehen.")

        self._story_phase = "beginning"
        return " ".join(parts)

    def phase_narrative(self, phase_id: str, **kwargs: Any) -> str:
        """Generate narrative for a specific phase."""
        self._phase_count += 1

        # Determine story arc position
        if self._phase_count <= 3:
            self._story_phase = "beginning"
        elif self._phase_count <= 10:
            self._story_phase = "middle"
        else:
            self._story_phase = "end"

        # Check for phase-specific narrative
        short_id = phase_id.replace("phase_", "")
        narratives = self._PHASE_NARRATIVES.get(phase_id, [])
        if not narratives:
            # Try partial match
            for key, val in self._PHASE_NARRATIVES.items():
                if key.endswith(short_id) or short_id in key:
                    narratives = val
                    break

        if narratives:
            template = self._rng.choice(narratives)
            return self._render_phase_template(template, phase_id, **kwargs)

        # Fallback: generate from scratch
        return self._generate_fallback(phase_id, **kwargs)

    def closing(self, quality_score: float = 0.0) -> str:
        """Generate the closing narrative."""
        closings = [
            "Die Reise durch die Zeit ist zu Ende. {result}",
            "Ich lege den Tonkopf behutsam zur Seite. {result}",
            "Die Aufnahme hat ihre Geschichte erzählt — ich habe nur zugehört und geholfen. {result}",
            "Fertig. Die Musik atmet wieder. {result}",
            "Was für eine Reise. {result}",
        ]

        results = [
            "Der Song klingt, als wäre die Zeit spurlos an ihm vorübergegangen.",
            "Die Seele des Originals ist unangetastet — nur der Staub der Jahrzehnte ist verschwunden.",
            "Man hört jetzt Dinge, die jahrzehntelang unter Rauschen und Knistern verborgen waren.",
            "Das ist kein perfekter Klang. Es ist der ECHTE Klang. Genau darum ging es.",
        ]

        template = self._rng.choice(closings)
        return template.format(result=self._rng.choice(results))

    # ── Internal ─────────────────────────────────────────────────

    def _render_phase_template(self, template: dict[str, str], phase_id: str, **kwargs: Any) -> str:
        """Render a phase narrative template with context."""
        parts: list[str] = []

        # Story arc connector
        connectors = {
            "beginning": [
                "Zunächst ",
                "Als Erstes ",
                "Zu Beginn ",
                "Am Anfang ",
            ],
            "middle": [
                "Nun ",
                "Als Nächstes ",
                "Danach ",
                "Im weiteren Verlauf ",
                "Jetzt, da die Grundlage sauber ist, ",
                "Mit ruhiger Hand ",
            ],
            "end": [
                "Zum Abschluss ",
                "Schließlich ",
                "Als letzten Feinschliff ",
                "Bevor ich die Arbeit beende, ",
                "Noch ein letzter Blick: ",
            ],
        }
        connector = self._pick_fresh(connectors[self._story_phase], self._used_actions)
        parts.append(connector)

        # Finding
        finding = template.get("finding", "untersuche ich die nächste Schicht.")
        finding = self._interpolate(finding, **kwargs)
        parts.append(finding)

        # Action
        action = template.get("action", "")
        if action:
            action = self._interpolate(action, **kwargs)
            parts.append(action)

        # Result
        result = template.get("result", "")
        if result:
            result = self._interpolate(result, **kwargs)
            parts.append(result)

        return " ".join(parts)

    def _generate_fallback(self, phase_id: str, **kwargs: Any) -> str:
        """Generate narrative for phases without specific templates."""
        mat_vocab = self._MATERIAL_VOCAB.get(self._material, self._MATERIAL_VOCAB["unknown"])

        connectors = {
            "beginning": ["Zunächst", "Am Anfang", "Als ersten Schritt"],
            "middle": ["Nun", "Weiter geht es", "Der nächste Arbeitsschritt"],
            "end": ["Zum Abschluss", "Schließlich", "Ein letzter Feinschliff"],
        }
        connector = self._rng.choice(connectors[self._story_phase])

        action = self._pick_fresh(self._FALLBACK_ACTIONS, self._used_actions)

        label = phase_id.replace("phase_", "").replace("_", " ")
        imagery = self._rng.choice(mat_vocab.get("imagery", ["mit Charakter"]))

        results = [
            "Das Klangbild wird runder.",
            "Die Musik atmet freier.",
            "Ein weiterer Schleier fällt.",
            "Der ursprüngliche Charakter tritt hervor.",
        ]

        return f"{connector} {action} — {label}. {imagery}. {self._rng.choice(results)}"

    def _interpolate(self, text: str, **kwargs: Any) -> str:
        """Fill template variables with context-aware values."""
        result = text

        # Replace template vars
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))

        # Replace persona vars
        result = result.replace("{discovery}", self._rng.choice(self._persona["discovery"]))
        result = result.replace("{action}", self._rng.choice(self._persona["action_style"]))
        result = result.replace("{admire}", self._rng.choice(self._persona["admire"]))

        # Material imagery
        mat_vocab = self._MATERIAL_VOCAB.get(self._material, self._MATERIAL_VOCAB["unknown"])
        result = result.replace("{imagery}", self._rng.choice(mat_vocab.get("imagery", [""])))

        # Remove any remaining unreplaced template vars
        import re

        result = re.sub(r"\{[^}]*\}", "", result).strip()

        # Clean double spaces
        while "  " in result:
            result = result.replace("  ", " ")

        return result

    def _pick_fresh(self, options: list[str], used: list[str]) -> str:
        """Pick an option, preferring unused ones."""
        available = [o for o in options if o not in used]
        if not available:
            available = options
            used.clear()
        choice = self._rng.choice(available)
        used.append(choice)
        if len(used) > 30:
            used.pop(0)
        return choice
