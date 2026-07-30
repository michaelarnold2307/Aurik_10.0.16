"""
§2.59 Cross-Module Contract Validator (2026-07-09)

Validiert zur Laufzeit (bei restore()-Start), dass alle Defekt-Namen,
Phasen-IDs und Material-Schlüssel zwischen den Modulen konsistent sind.

Verhindert stille Mismatches wie:
  - DefectScanner produziert "clicks", PhasePruner erwartet "click"
  - UV3 ruft analyze_defects() auf, aber Methode heißt analyze_instances
  - MaterialType hat neue Werte, aber _MATERIAL_SKIP_PHASES kennt sie nicht

Wird einmal pro restore()-Aufruf ausgeführt. Logged WARNINGs,
niemals Exceptions — die Pipeline soll trotzdem laufen.
"""

from __future__ import annotations

import importlib
import logging
import os  # §v10.105 module-level (prevents UnboundLocalError)
from typing import Any

logger = logging.getLogger(__name__)


class ContractViolation:
    """Eine gefundene Cross-Module-Inkonsistenz."""

    def __init__(self, source: str, target: str, detail: str) -> None:
        self.source = source
        self.target = target
        self.detail = detail

    def __str__(self) -> str:
        return f"[{self.source} → {self.target}] {self.detail}"


def validate_defect_contracts(
    *,
    verbose: bool = False,
) -> list[ContractViolation]:
    """Validiert alle Cross-Module-Defekt-Contracts.

    Returns:
        Liste von ContractViolation (leer = alles OK).
    """
    violations: list[ContractViolation] = []

    # ── 1. PhasePruner: alle Requirements gegen DefectType ──────────────
    try:
        from backend.core.defect_scanner import DefectType
        from backend.core.phase_pruner import (
            _MATERIAL_SKIP_PHASES,
            _PHASE_DEFECT_REQUIREMENTS,
        )

        CANONICAL = {e.value for e in DefectType}

        for phase_id, reqs in _PHASE_DEFECT_REQUIREMENTS.items():
            for req in reqs:
                if req and req not in CANONICAL:
                    violations.append(
                        ContractViolation(
                            "PhasePruner._PHASE_DEFECT_REQUIREMENTS",
                            "DefectType",
                            f"Phase '{phase_id}' referenziert unbekannten Defekt '{req}'",
                        )
                    )

        # Check all MaterialType values are covered
        all_materials = {
            e.value for e in __import__("backend.core.defect_scanner", fromlist=["MaterialType"]).MaterialType
        }
        digital = {"aac", "cd_digital", "dat", "minidisc", "mp3_high", "mp3_low", "streaming"}
        for mat in digital:
            if mat not in _MATERIAL_SKIP_PHASES:
                violations.append(
                    ContractViolation(
                        "PhasePruner._MATERIAL_SKIP_PHASES",
                        "MaterialType",
                        f"Digital-Material '{mat}' hat keine Skip-Regeln",
                    )
                )
        # Analog materials should NOT have skip rules
        analog = all_materials - digital - {"unknown"}
        for mat in analog:
            if mat in _MATERIAL_SKIP_PHASES:
                violations.append(
                    ContractViolation(
                        "PhasePruner._MATERIAL_SKIP_PHASES",
                        "MaterialType",
                        f"Analog-Material '{mat}' hat unerwartete Skip-Regeln",
                    )
                )

    except ImportError as e:
        logger.warning("⚠️ SOTA ContractValidator: PhasePruner-Check skipped (%s)", e)

    # ── 2. DefectPrecisionEnhancer: analyze_defects existiert ─────────
    try:
        from backend.core.defect_precision_enhancer import DefectPrecisionEnhancer

        if not hasattr(DefectPrecisionEnhancer, "analyze_defects"):
            violations.append(
                ContractViolation(
                    "UV3 §AD",
                    "DefectPrecisionEnhancer",
                    "Methode 'analyze_defects' fehlt — §AD Precision ist inaktiv",
                )
            )
    except ImportError:
        logger.warning("⚠️ SOTA ContractValidator: DefectPrecisionEnhancer-Check skipped (not importable)")

    # ── 3. SongGoalImportance: Defekt-Keys prüfen ─────────────────────
    try:
        import inspect

        from backend.core.song_goal_importance import estimate_goal_importance as _egi

        sig = inspect.signature(_egi)
        param_names = list(sig.parameters.keys())
        if "defect_severities" not in param_names:
            violations.append(
                ContractViolation(
                    "UV3",
                    "SongGoalImportance",
                    "estimate_goal_importance akzeptiert 'defect_severities' nicht",
                )
            )
    except ImportError:
        logger.warning("⚠️ SOTA ContractValidator: SongGoalImportance-Check skipped")

    # ── 4. DefectManifest ↔ PhasePruner Synchronisation ─────────────
    try:
        from backend.core.defect_manifest import get_defect_manifest
        from backend.core.phase_pruner import _PHASE_DEFECT_REQUIREMENTS as _PPR

        _dm = get_defect_manifest()
        for defect_name, entry in _dm._entries.items():
            expected_phases = set(entry.phases)
            # Check: jede Phase im Manifest hat einen Pruner-Eintrag
            for phase_id in expected_phases:
                if phase_id not in _PPR:
                    violations.append(
                        ContractViolation(
                            "DefectManifest",
                            "PhasePruner",
                            f"Defekt '{defect_name}' → Phase '{phase_id}' fehlt im PhasePruner",
                        )
                    )
    except ImportError:
        logger.warning("⚠️ SOTA ContractValidator: DefectManifest-Check skipped")

    # ── 5. ML-Health-Check: Kritische Modelle prüfen ──────────────────
    try:
        _ml_models = {
            "DeepFilterNetV3": "plugins.deepfilternet_v3_ii_plugin",
            "PANNs": "plugins.panns_plugin",
            "FCPE": "plugins.fcpe_plugin",
        }
        for model_name, module_path in _ml_models.items():
            try:
                importlib.import_module(module_path)
            except ImportError:
                pass  # Plugin nicht installiert — kein Fehler
            except Exception as e:
                violations.append(
                    ContractViolation(
                        "ML-Model",
                        model_name,
                        f"Modul {module_path} konnte nicht geladen werden: {e}",
                    )
                )
    except ImportError:
        logger.warning("⚠️ SOTA ContractValidator: ML-Health-Check skipped")

    # ── 6. Keine toten Verzeichnisse mehr ──────────────────────────────
    # import os → module level (§v10.105)
    dead_paths = [
        "backend/adaptive_pipeline.py",
        "backend/defect_detection/",
        "backend/region_analysis.py",
    ]
    for dp in dead_paths:
        if os.path.exists(dp):
            violations.append(
                ContractViolation(
                    "Filesystem",
                    "DeadCode",
                    f"Tote Datei existiert noch: {dp}",
                )
            )

    return violations


def run_contract_validation(verbose: bool = False) -> dict[str, Any]:
    """Führt alle Contract-Validierungen aus und logged das Ergebnis.

    Sollte einmal pro restore()-Aufruf aufgerufen werden.

    Returns:
        {"ok": bool, "violations": int, "details": list[str]}
    """
    violations = validate_defect_contracts(verbose=verbose)

    if not violations:
        logger.info("✅ ContractValidator: Alle Cross-Module-Contracts OK")
        return {"ok": True, "violations": 0, "details": []}

    logger.warning(
        "ContractValidator: %d Cross-Module-Inkonsistenzen gefunden — "
        "Pipeline läuft weiter, aber Ergebnisse können suboptimal sein.",
        len(violations),
    )
    for v in violations:
        logger.warning("  %s", v)

    return {
        "ok": False,
        "violations": len(violations),
        "details": [str(v) for v in violations],
    }
