from __future__ import annotations

"""[RELEASE_MUST] §0a Restoration-Mode-Guard — normative CI-Gate (v10.0.0)

Dieser Test schließt strukturell die Klasse von Bugs, die am 2026-05-05 gefunden wurden:
Spec/Code-Drift bei §0a-verbotenen Phasen (phase_42 in forced_phases, phase_39 in
CAUSE_TO_PHASES, phase_21 fehlend im UV3-Guard).

Drei Prüfebenen:

  1. UV3-Guard-Vollständigkeit:
     ``_restoration_forbidden_stem_enhancement`` muss alle kanonisch §0a-verbotenen
     Phasen enthalten.  Neue §0a-verbotene Phasen → kanonische Liste hier erweitern.

  2. CAUSE_TO_PHASES-Reinheit:
     Kein Eintrag in CAUSE_TO_PHASES darf eine §0a-verbotene Phase enthalten.
     CausalDefectReasoner routet blind in den Restoration-Modus — ohne UV3-Guard wäre
     jede solche Phase in jeder Restoration-Session aktiv.

  3. Genre-Profil-Reinheit:
     Kein ``forced_phases``-Schlüssel in einem Genre-Profil (GermanSchlagerClassifier,
     GenreClassifier, EraClassifier, …) darf eine §0a-verbotene Phase enthalten.
     UV3 setzt ``forced_phases`` vor dem ``_restoration_forbidden_stem_enhancement``
     Gate — der Guard entfernt sie zwar, aber die falsche Spec-Grundlage führt zu
     Folge-Bugs in abgeleiteten Genre-Profilen.

  4. Phase-Familie-Scan (Anti-Regression):
     Alle Phasendateien mit ``_enhancement`` im Dateinamen werden geprüft:
     Jede solche Phase MUSS entweder in der kanonischen Verboten-Liste stehen ODER
     eine dokumentierte Ausnahme in _SECTION_0A_DOCUMENTED_EXCEPTIONS haben.
     → Fängt zukünftige neue Enhancement-Phasen, die vergessen wurden zu sperren.

Aufruf:
    pytest tests/normative/test_section_0a_restoration_guard.py -v --timeout=30

Spec-Referenz:
    copilot-instructions.md §0a (Restoration VERBOTEN: Harmonic Exciter, Stem-Enhancement)
    copilot-instructions.md §0a Crossfire-Modus-Invariante
    .github/specs/06_phases_system.md Phasenfamilien-Tabelle (ENHANCEMENT)
"""


import ast
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Kanonische §0a-Verbotsliste (Restoration-Modus)
# ---------------------------------------------------------------------------
# Erweiterungen hier vornehmen wenn neue Phasen §0a-Kategorien zugeordnet werden.
# Jeder Eintrag ist eine normative Entscheidung — kein undokumentierter Änderung erlaubt.
_SECTION_0A_FORBIDDEN_IN_RESTORATION: frozenset[str] = frozenset(
    {
        "phase_21_exciter",  # §0a: Harmonic Exciter VERBOTEN in Restoration
        "phase_35_multiband_compression",  # §0a: Multiband-Kompression = Stem-Enhancement
        "phase_42_vocal_enhancement",  # §0a: Vocal AI = Stem-Enhancement
        "phase_44_guitar_enhancement",  # §0a: Instrument-Stem-Enhancement nur Studio 2026
        "phase_45_brass_enhancement",  # §0a: Instrument-Stem-Enhancement nur Studio 2026
        "phase_51_drums_enhancement",  # §0a: Instrument-Stem-Enhancement nur Studio 2026
        # phase_56_reference_mastering: keine .py-Datei in backend/core/phases/;
        # UV3-Guard-Eintrag ist vorbeugend für zukünftige Phase. → documented exception.
    }
)

# Phasen mit _enhancement-Suffix, die im Restoration-Modus ERLAUBT sind (mit Begründung).
# Jede Ausnahme muss eine Begründung haben.
_SECTION_0A_DOCUMENTED_EXCEPTIONS: dict[str, str] = {
    # Phasen mit _enhancement-Suffix die KEIN Stem-Enhancement im §0a-Sinne sind
    "phase_13_stereo_enhancement": "Stereo-Enhancement = geometrische Korrektur (M/S-Balance), kein Stem-Enhancement; §0a nicht zutreffend",
    "phase_37_bass_enhancement": "Bass-Enhancement = passiver EQ-Boost für Fundamental-Wiedergabe; kein Stem-Enhancement. In Restoration nur aktiv wenn Organ/Bass-Evidenz",
    "phase_39_air_band_enhancement": "Air-Band: §0a ambig. Im Restoration-Modus durch BW-Ceiling-Guard (§6.2c) begrenzt; über Material-BW-Ceiling hinaus verboten (BUG-FIX v10.0.0, CAUSE_TO_PHASES bereinigt)",
    "phase_43_ml_deesser": "ML-De-Esser = passive Defektkorrektur (Sibilanten-Entfernung), kein Stem-Enhancement",
    # Phase-IDs die im UV3-Guard stehen aber (noch) keine .py-Datei haben
    "phase_56_reference_mastering": "KEIN .PY: UV3-Guard-Eintrag ist vorbeugend für eine zukünftige Phase. Wenn Phase implementiert wird: automatisch §0a-verboten.",
    # Eindeutig erlaubte Enhancement-Phasen
    "phase_46_spatial_enhancement": "Spatial-Enhancement: in Restoration nur aktiv wenn §2.49c Baseline-Evidenz vorhanden; UV3 prüft spatial_evidence_threshold",
    "phase_47_truepeak_limiter": "True-Peak-Limiter = Schutzfunktion, kein Stem-Enhancement",
    "phase_48_stereo_width_enhancer": "Stereo-Enhancer: in Restoration nur über phase_13/phase_15 routing (§2.51); nicht als eigenständiges Enhancement",
    "phase_52_piano_enhancement": "Piano-Enhancement: derzeit nur Studio-2026 via UV3 mode-check; TODO: explizit in _restoration_forbidden aufnehmen wenn aktiviert",
    "phase_53_strings_enhancement": "Strings-Enhancement: derzeit nur Studio-2026 via UV3 mode-check; TODO: explizit in _restoration_forbidden aufnehmen wenn aktiviert",
    "phase_54_transparent_dynamics": "Transparent Dynamics: Dynamik-Reparatur = passive Korrektur, kein Hinzufügen von Energie",
    "phase_57_print_through_reduction": "Print-Through-Reduktion = subtraktiv/reparativ, kein Enhancement",
    "phase_58_lyrics_guided_enhancement": "Lyrics-Guided: §2.36 PFLICHT auch in Restoration für phonem-bewusste NR-Steuerung; Enhancement-Suffix ist irreführend",
    "phase_59_modulation_noise_reduction": "Modulationsrauschen-Reduktion = subtraktiv, kein Enhancement",
    "phase_61_room_acoustic_eq": "Room-EQ = Korrektur (subtraktiv), kein Enhancement",
    "phase_62_vinyl_warmth_restoration": "Warmth-Restoration = Carrier-Signatur bewahren (§0a Restoration-Ziel), kein additives Enhancement",
    "phase_64_ai_upsampling": "AI-Upsampling: nur Studio-2026 via UV3 mode-check; Hard-Blocked in Restoration durch UV3 _STUDIO_2026_ONLY_PHASES",
}

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_UV3_PATH = _ROOT / "backend" / "core" / "unified_restorer_v3.py"
_CDR_PATH = _ROOT / "backend" / "core" / "causal_defect_reasoner.py"
_PHASES_DIR = _ROOT / "backend" / "core" / "phases"
_GENRE_CLASSIFIER_PATH = _ROOT / "backend" / "core" / "genre_classifier.py"


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _extract_uv3_forbidden_set(uv3_source: str) -> frozenset[str]:
    """Extrahiert `_restoration_forbidden_stem_enhancement` aus UV3-Quelltext per Regex.

    Sucht nach dem Set-Literal:
        _restoration_forbidden_stem_enhancement = {
            "phase_XX_...",
            ...
        }
    und gibt die enthaltenen Phase-Strings zurück.
    """
    pattern = re.compile(
        r"_restoration_forbidden_stem_enhancement\s*=\s*\{([^}]+)\}",
        re.DOTALL,
    )
    match = pattern.search(uv3_source)
    if not match:
        return frozenset()
    block = match.group(1)
    # Extrahiere alle "phase_XX_..."-Strings
    return frozenset(re.findall(r'"(phase_[^"]+)"', block))


def _extract_cause_to_phases(module_source: str) -> dict[str, list[str]]:
    """Parst CAUSE_TO_PHASES via AST aus dem Quelltext von causal_defect_reasoner.py.

    Unterst\u00fctzt sowohl normales Assignment (``CAUSE_TO_PHASES = {...}``)
    als auch type-annotiertes Assignment (``CAUSE_TO_PHASES: dict[...] = {...}``)
    da Python 3.6+ annotierte Assignments als ``ast.AnnAssign`` repräsentiert.
    """
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        # Normales Assignment: CAUSE_TO_PHASES = {...}
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "CAUSE_TO_PHASES"
            and isinstance(node.value, ast.Dict)
        ):
            return _parse_cause_to_phases_dict(node.value)
        # Annotiertes Assignment: CAUSE_TO_PHASES: dict[str, list[str]] = {...}
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "CAUSE_TO_PHASES"
            and node.value is not None
            and isinstance(node.value, ast.Dict)
        ):
            return _parse_cause_to_phases_dict(node.value)
    return {}


def _parse_cause_to_phases_dict(dict_node: ast.Dict) -> dict[str, list[str]]:
    """Extrahiert key→[phase-strings] aus einem ``ast.Dict``-Knoten."""
    result: dict[str, list[str]] = {}
    for key_node, val_node in zip(dict_node.keys, dict_node.values):
        if not isinstance(key_node, ast.Constant):
            continue
        cause_key = key_node.value
        phases: list[str] = []
        if isinstance(val_node, ast.List):
            for elt in val_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    phases.append(elt.value)
        result[cause_key] = phases  # type: ignore[index]
    return result


def _find_forced_phases_in_source(source: str, filepath: Path) -> list[tuple[str, str, int]]:
    """Findet alle `forced_phases`-Listen in einem Python-Quelltext.

    Gibt eine Liste von (dateiname, phase_string, zeilennummer) zurück.
    Parst per AST — robust gegenüber Formatierungsänderungen.
    """
    results: list[tuple[str, str, int]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return results

    for node in ast.walk(tree):
        # Suche nach Dict-Einträgen mit key="forced_phases"
        if isinstance(node, ast.Dict):
            for key_node, val_node in zip(node.keys, node.values):
                if (
                    isinstance(key_node, ast.Constant)
                    and key_node.value == "forced_phases"
                    and isinstance(val_node, ast.List)
                ):
                    for elt in val_node.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            results.append((str(filepath), elt.value, getattr(elt, "lineno", 0)))
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUV3ForbiddenSetComplete:
    """Ebene 1: UV3 _restoration_forbidden_stem_enhancement enthält alle §0a-verbotenen Phasen."""

    def test_uv3_file_exists(self) -> None:
        assert _UV3_PATH.exists(), f"UV3 nicht gefunden: {_UV3_PATH}"

    def test_forbidden_set_extracted(self) -> None:
        """Set muss extrahierbar und nicht leer sein."""
        source = _UV3_PATH.read_text(encoding="utf-8")
        extracted = _extract_uv3_forbidden_set(source)
        assert len(extracted) >= 4, (
            "_restoration_forbidden_stem_enhancement in UV3 ist leer oder nicht gefunden. "
            "Prüfe ob der Set-Literal noch vorhanden ist."
        )

    def test_canonical_phases_all_present(self) -> None:
        """Jede Phase in _SECTION_0A_FORBIDDEN_IN_RESTORATION muss im UV3-Guard sein."""
        source = _UV3_PATH.read_text(encoding="utf-8")
        actual = _extract_uv3_forbidden_set(source)
        missing = _SECTION_0A_FORBIDDEN_IN_RESTORATION - actual
        assert not missing, (
            "§0a-Verletzung: Folgende kanonisch verbotenen Phasen fehlen im UV3 "
            "_restoration_forbidden_stem_enhancement:\n"
            + "\n".join(f"  - {p}" for p in sorted(missing))
            + "\n\nFix: Phasen in UV3 backend/core/unified_restorer_v3.py zum Set hinzufügen."
        )


class TestCauseToPhasesPurity:
    """Ebene 2: CAUSE_TO_PHASES darf keine §0a-verbotenen Phasen enthalten."""

    def test_cdr_file_exists(self) -> None:
        assert _CDR_PATH.exists(), f"CausalDefectReasoner nicht gefunden: {_CDR_PATH}"

    def test_no_forbidden_phase_in_any_cause(self) -> None:
        """Kein CAUSE_TO_PHASES-Wert darf eine §0a-verbotene Phase enthalten."""
        source = _CDR_PATH.read_text(encoding="utf-8")
        c2p = _extract_cause_to_phases(source)
        assert c2p, "CAUSE_TO_PHASES konnte nicht geparst werden — AST-Parse fehlgeschlagen."

        violations: list[tuple[str, str]] = []
        for cause, phases in c2p.items():
            for phase in phases:
                if phase in _SECTION_0A_FORBIDDEN_IN_RESTORATION:
                    violations.append((cause, phase))

        assert not violations, (
            "§0a-Verletzung: CAUSE_TO_PHASES enthält §0a-verbotene Phasen:\n"
            + "\n".join(f"  CAUSE '{c}' → '{p}'" for c, p in violations)
            + "\n\nFix: Phasen aus CAUSE_TO_PHASES in causal_defect_reasoner.py entfernen."
        )

    def test_no_enhancement_suffix_phase_in_cause_to_phases(self) -> None:
        """Keine Phase mit '_enhancement'-Suffix darf in CAUSE_TO_PHASES stehen
        (alle Enhancement-Phasen sind entweder §0a-verboten oder Studio-2026-only).

        Ausnahmen: phase_43_ml_deesser ist kein echter Enhancement-Typ;
        phase_58_lyrics_guided_enhancement ist §2.36-Pflicht aber phonem-bewusst.
        """
        _ALLOWED_ENHANCEMENT_IN_C2P: frozenset[str] = frozenset(
            {
                # §2.36-Pflicht: phonem-bewusste NR-Steuerung, kein echtes Enhancement
                "phase_58_lyrics_guided_enhancement",
            }
        )
        source = _CDR_PATH.read_text(encoding="utf-8")
        c2p = _extract_cause_to_phases(source)
        assert c2p, "CAUSE_TO_PHASES konnte nicht geparst werden."

        violations: list[tuple[str, str]] = []
        for cause, phases in c2p.items():
            for phase in phases:
                if "_enhancement" in phase and phase not in _ALLOWED_ENHANCEMENT_IN_C2P:
                    violations.append((cause, phase))

        assert not violations, (
            "§0a-Verletzung (Muster-Scan): CAUSE_TO_PHASES enthält Phasen mit "
            "'_enhancement'-Suffix:\n"
            + "\n".join(f"  CAUSE '{c}' → '{p}'" for c, p in violations)
            + "\n\nEnhancement-Phasen gehören nicht in CAUSE_TO_PHASES — sie sind entweder "
            "§0a-verboten (Restoration) oder Studio-2026-only. "
            "Falls die Phase kein Stem-Enhancement ist, eine Ausnahme in "
            "_ALLOWED_ENHANCEMENT_IN_C2P in diesem Test dokumentieren."
        )


class TestGenreProfilePurity:
    """Ebene 3: Genre-Profile dürfen in forced_phases keine §0a-verbotenen Phasen haben."""

    def _collect_forced_phases_violations(self) -> list[tuple[str, str, int]]:
        """Scannt alle Python-Dateien in backend/core/ auf forced_phases-Verletzungen."""
        violations: list[tuple[str, str, int]] = []
        scan_dirs = [
            _ROOT / "backend" / "core",
            _ROOT / "backend",
        ]
        seen_files: set[Path] = set()
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for pyfile in scan_dir.glob("*.py"):
                if pyfile in seen_files:
                    continue
                seen_files.add(pyfile)
                source = pyfile.read_text(encoding="utf-8")
                entries = _find_forced_phases_in_source(source, pyfile)
                for filepath, phase, line in entries:
                    if phase in _SECTION_0A_FORBIDDEN_IN_RESTORATION:
                        violations.append((filepath, phase, line))
        return violations

    def test_no_forbidden_phase_in_forced_phases(self) -> None:
        """Kein Genre-Profil darf §0a-verbotene Phasen in forced_phases enthalten."""
        violations = self._collect_forced_phases_violations()
        assert not violations, (
            "§0a-Verletzung: forced_phases enthält §0a-verbotene Phasen:\n"
            + "\n".join(f"  {Path(fp).name}:{ln}  phase='{p}'" for fp, p, ln in violations)
            + "\n\nFix: Phasen aus dem jeweiligen forced_phases-Eintrag entfernen. "
            "UV3 _restoration_forbidden_stem_enhancement erzwingt §0a bereits universell — "
            "ein Eintrag in forced_phases ist daher überflüssig UND normativ falsch."
        )


class TestEnhancementPhaseFamilyScan:
    """Ebene 4 (Anti-Regression): Neue Enhancement-Phasen werden automatisch erkannt
    und müssen entweder in der kanonischen Verbotsliste oder in _SECTION_0A_DOCUMENTED_EXCEPTIONS
    stehen.

    Verhindert, dass eine neue phase_XX_enhancement.py-Datei still vergessen wird.
    """

    def _get_enhancement_phase_ids(self) -> list[str]:
        """Findet alle Phasendateien mit _enhancement im Namen."""
        if not _PHASES_DIR.exists():
            return []
        ids = []
        for f in _PHASES_DIR.glob("phase_*_enhancement*.py"):
            stem = f.stem  # z.B. "phase_44_guitar_enhancement"
            ids.append(stem)
        return sorted(ids)

    def test_all_enhancement_phases_classified(self) -> None:
        """Jede Enhancement-Phase muss entweder §0a-verboten ODER dokumentierte Ausnahme sein."""
        enhancement_phases = self._get_enhancement_phase_ids()
        if not enhancement_phases:
            pytest.skip("Keine *_enhancement*.py-Phasendateien in backend/core/phases/ gefunden.")

        known = _SECTION_0A_FORBIDDEN_IN_RESTORATION | frozenset(_SECTION_0A_DOCUMENTED_EXCEPTIONS)
        unclassified = [p for p in enhancement_phases if p not in known]

        assert not unclassified, (
            "§0a-Lücke: Neue Enhancement-Phase(n) ohne §0a-Klassifizierung gefunden:\n"
            + "\n".join(f"  - {p}" for p in unclassified)
            + "\n\nJede neue *_enhancement*.py-Phase MUSS eine der folgenden Aktionen auslösen:\n"
            "  A) Phase in _SECTION_0A_FORBIDDEN_IN_RESTORATION eintragen (dieser Test)\n"
            "     UND in UV3 _restoration_forbidden_stem_enhancement aufnehmen.\n"
            "  B) Phase in _SECTION_0A_DOCUMENTED_EXCEPTIONS eintragen (dieser Test)\n"
            "     MIT Begründung warum sie kein §0a-Stem-Enhancement ist."
        )

    def test_forbidden_phases_have_files(self) -> None:
        """Alle kanonisch verbotenen Phasen müssen als .py-Datei existieren.

        Verhindert, dass veraltete Phase-IDs in der Verbotsliste stehen und die
        Checks wertlos werden (false negatives).

        Ausnahme: Phasen in _SECTION_0A_DOCUMENTED_EXCEPTIONS mit 'KEIN .PY'
        in der Begründung sind dokumentierte Ausnahmen (z.B. vorbeugender Guard-Eintrag).
        """
        missing_files: list[str] = []
        for phase_id in _SECTION_0A_FORBIDDEN_IN_RESTORATION:
            exc = _SECTION_0A_DOCUMENTED_EXCEPTIONS.get(phase_id, "")
            if "KEIN .PY" in exc or "no .py" in exc.lower():
                continue
            phase_file = _PHASES_DIR / f"{phase_id}.py"
            if not phase_file.exists():
                missing_files.append(phase_id)

        assert not missing_files, (
            "Veraltete Einträge in _SECTION_0A_FORBIDDEN_IN_RESTORATION — "
            "keine entsprechende .py-Datei in backend/core/phases/:\n"
            + "\n".join(f"  - {p}" for p in missing_files)
            + "\n\nFix: Entweder Phase wieder anlegen ODER Eintrag aus der kanonischen Liste "
            "in diesem Test entfernen (Begründung in _SECTION_0A_DOCUMENTED_EXCEPTIONS "
            "mit 'KEIN .PY' im Text)."
        )


class TestCrossFireModusInvariant:
    """§0a Crossfire-Modus-Invariante: Studio-2026-only-Phasen dürfen nicht in
    Restoration-Konfigurationen auftauchen."""

    # Bekannte Studio-2026-only-Phasen (explizit in Spec 06 markiert)
    # Erweiterungen hier vornehmen wenn neue Studio-2026-only-Phasen hinzukommen.
    _STUDIO_2026_ONLY_PHASES: frozenset[str] = frozenset(
        {
            "phase_64_ai_upsampling",
        }
    )

    def test_studio_2026_only_not_in_cause_to_phases(self) -> None:
        """Studio-2026-only-Phasen dürfen nicht in CAUSE_TO_PHASES stehen."""
        if not _CDR_PATH.exists():
            pytest.skip("CausalDefectReasoner nicht gefunden.")
        source = _CDR_PATH.read_text(encoding="utf-8")
        c2p = _extract_cause_to_phases(source)
        violations: list[tuple[str, str]] = []
        for cause, phases in c2p.items():
            for phase in phases:
                if phase in self._STUDIO_2026_ONLY_PHASES:
                    violations.append((cause, phase))
        assert not violations, "§0a Crossfire-Verletzung: Studio-2026-only-Phase in CAUSE_TO_PHASES:\n" + "\n".join(
            f"  CAUSE '{c}' → '{p}'" for c, p in violations
        )

    def test_studio_2026_only_not_in_forced_phases(self) -> None:
        """Studio-2026-only-Phasen dürfen nicht in forced_phases eines Genre-Profils stehen."""
        scan_dir = _ROOT / "backend" / "core"
        if not scan_dir.exists():
            pytest.skip("backend/core nicht gefunden.")
        violations: list[tuple[str, str, int]] = []
        for pyfile in scan_dir.glob("*.py"):
            source = pyfile.read_text(encoding="utf-8")
            entries = _find_forced_phases_in_source(source, pyfile)
            for filepath, phase, line in entries:
                if phase in self._STUDIO_2026_ONLY_PHASES:
                    violations.append((filepath, phase, line))
        assert not violations, "§0a Crossfire-Verletzung: Studio-2026-only-Phase in forced_phases:\n" + "\n".join(
            f"  {Path(fp).name}:{ln}  phase='{p}'" for fp, p, ln in violations
        )
