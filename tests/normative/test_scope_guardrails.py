"""Scope Guardrails CI Enforcement — §v10.300.

Erzwingt die in policy/scope_guardrails.yaml definierten Obergrenzen:
- Keine neuen Phasen ohne Echt-Audio-Evidenz
- Keine neuen Musical Goals ohne gemessene HPI-Korrelation
- Keine neuen DefectTypes ohne manifestierte Fälle
- Keine neuen Materialien ohne Corpus

Zusätzlich wird die Phase-Count- und Goal-Count-Konsistenz mit PROJECT_STATUS.md
und der tatsächlichen Codebasis geprüft.

Nutzung:
  pytest tests/normative/test_scope_guardrails.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

GUARDRAILS_PATH = REPO_ROOT / "policy" / "scope_guardrails.yaml"
PROJECT_STATUS_PATH = REPO_ROOT / "docs" / "PROJECT_STATUS.md"
SPEC_PATH = REPO_ROOT / ".github" / "specs" / "06_phases_system.md"


def _load_guardrails() -> dict | None:
    """Lädt die Scope-Guardrails-YAML."""
    try:
        import yaml
    except ImportError:
        return None
    if not GUARDRAILS_PATH.exists():
        return None
    with open(GUARDRAILS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _count_phases_in_code() -> int:
    """Zählt die tatsächlichen Phasen-Dateien im Code."""
    phases_dir = REPO_ROOT / "backend" / "core" / "phases"
    if not phases_dir.exists():
        return 0
    phase_files = list(phases_dir.glob("phase_*.py"))
    return len(phase_files)


def _count_goals_in_code() -> int:
    """Zählt die Musical-Goal-Dateien."""
    goals_file = REPO_ROOT / "backend" / "core" / "musical_goals" / "musical_goals_metrics.py"
    if not goals_file.exists():
        return 0
    content = goals_file.read_text(encoding="utf-8")
    # Zähle class-Klassen die auf "Metric" enden
    import re

    return len(re.findall(r"^class (\w*Metric)\b", content, re.MULTILINE))


def _count_defect_types_in_code() -> int:
    """Zählt die DefectTypes im DefectScanner."""
    scanner_file = REPO_ROOT / "backend" / "core" / "defect_scanner.py"
    if not scanner_file.exists():
        return 0
    content = scanner_file.read_text(encoding="utf-8")
    # Zähle Enum-Werte oder Konstanten
    import re

    return len(re.findall(r"^\s+[A-Z_]+ = \"", content, re.MULTILINE))


def _count_materials_in_code() -> int:
    """Zählt die Material-Typen."""
    material_file = REPO_ROOT / "backend" / "core" / "forensics" / "medium_detector.py"
    if not material_file.exists():
        # Alternativ: unified_restorer_v3.py
        material_file = REPO_ROOT / "backend" / "core" / "unified_restorer_v3.py"
    if not material_file.exists():
        return 0
    content = material_file.read_text(encoding="utf-8")
    # Zähle Material-Konstanten
    import re

    return len(
        set(
            re.findall(
                r"\"(shellac|vinyl|tape|reel_tape|cassette|dat|"
                r"cd_digital|mp3_low|mp3_high|aac|minidisc|streaming|"
                r"lacquer_disc|wax_cylinder|wire_recording|unknown|"
                r"lp|kassette)\"",
                content,
            )
        )
    )


# ── CI Enforcement Tests ───────────────────────────────────────────────────


@pytest.mark.scope_guardrails
class TestScopeGuardrails:
    """Erzwingt Scope-Obergrenzen via CI."""

    @pytest.fixture(autouse=True)
    def guardrails(self) -> dict:
        data = _load_guardrails()
        if data is None:
            pytest.skip("policy/scope_guardrails.yaml nicht gefunden oder nicht parsebar.")
        return data

    def test_guardrails_file_exists(self):
        """Guardrails-Policy-Datei muss existieren."""
        assert GUARDRAILS_PATH.exists(), (
            "policy/scope_guardrails.yaml fehlt. Diese Datei definiert erzwungene Scope-Grenzen."
        )

    def test_phase_count_within_limit(self, guardrails: dict):
        """Phasen-Count darf Obergrenze nicht überschreiten."""
        caps = guardrails.get("caps", {})
        max_phases_limit = caps.get("max_phases", {}).get("limit", 68)
        current = _count_phases_in_code()
        assert current <= max_phases_limit, (
            f"Phase-Count {current} überschreitet Limit {max_phases_limit}. "
            f"Neue Phasen benötigen Echt-Audio-Evidenz: "
            f"policy/scope_guardrails.yaml → evidence_requirements.new_phase"
        )

    def test_musical_goals_within_limit(self, guardrails: dict):
        """Musical-Goal-Count darf Obergrenze nicht überschreiten."""
        caps = guardrails.get("caps", {})
        max_goals_limit = caps.get("max_musical_goals", {}).get("limit", 14)
        current = _count_goals_in_code()
        assert current <= max_goals_limit, (
            f"Musical-Goal-Count {current} überschreitet Limit {max_goals_limit}. "
            f"Neue Goals benötigen Echt-Audio-Evidenz: "
            f"policy/scope_guardrails.yaml → evidence_requirements.new_musical_goal"
        )

    def test_defect_types_within_limit(self, guardrails: dict):
        """DefectType-Count darf Obergrenze nicht überschreiten."""
        caps = guardrails.get("caps", {})
        max_defects_limit = caps.get("max_defect_types", {}).get("limit", 62)
        current = _count_defect_types_in_code()
        assert current <= max_defects_limit, (
            f"DefectType-Count {current} überschreitet Limit {max_defects_limit}. "
            f"Neue DefectTypes benötigen manifestierte Echt-Audio-Fälle."
        )

    def test_materials_within_limit(self, guardrails: dict):
        """Material-Count darf Obergrenze nicht überschreiten."""
        caps = guardrails.get("caps", {})
        max_materials_limit = caps.get("max_materials", {}).get("limit", 17)
        current = _count_materials_in_code()
        assert current <= max_materials_limit, (
            f"Material-Count {current} überschreitet Limit {max_materials_limit}. "
            f"Neue Materialien benötigen Corpus-Fälle."
        )

    def test_guardrails_enforced_in_ci(self, guardrails: dict):
        """Guardrails müssen in CI erzwungen sein."""
        enforced = guardrails.get("enforced_in_ci", False)
        assert enforced, "scope_guardrails.yaml: enforced_in_ci muss true sein."

    def test_project_status_consistent(self, guardrails: dict):
        """PROJECT_STATUS.md muss konsistent mit Guardrails sein."""
        if not PROJECT_STATUS_PATH.exists():
            pytest.skip("PROJECT_STATUS.md nicht gefunden")
        content = PROJECT_STATUS_PATH.read_text(encoding="utf-8")

        # Prüfe: Status sollte nicht "Produktionsbereit" sagen wenn Gate failed
        has_prod_ready = "Produktionsbereit" in content
        has_gate_failed = "Quality Gate NICHT bestanden" in content

        # Dies ist ein Soft-Check: Wenn der Gate failed, sollte der Status
        # NICHT "Produktionsbereit" sagen (Commitment C2)
        if has_prod_ready:
            # Prüfe ob der Status relativierend ist
            if "Produktionsbereit" in content and "✅" in content:
                # Akzeptiere nur wenn innerhalb von 200 Zeichen auch eine
                # Relativierung oder ein "nur für synthetische Tests" steht
                idx = content.index("Produktionsbereit")
                context = content[max(0, idx - 100) : idx + 300]
                if (
                    "synthetisch" not in context.lower()
                    and "real-audio" not in context.lower()
                    and "gate" not in context.lower()
                ):
                    pytest.fail(
                        "PROJECT_STATUS.md deklariert 'Produktionsbereit' ohne "
                        "Relativierung auf synthetische Tests. "
                        "Commitment C2: Status erst aktivieren wenn Quality Gate passed."
                    )
