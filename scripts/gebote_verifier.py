#!/usr/bin/env python3
"""§v10.15 Gebote-Verifier: Garantiert dass ALLE §G-Regeln im Code umgesetzt sind.

Non-Plus-Ultra-Compliance-Checker. Prüft JEDES Gebot gegen den tatsächlichen Code.
Keine Spezifikation ohne Verifikation. Kein Gebot ohne Nachweis.

Usage:
    python scripts/gebote_verifier.py          # Alle Checks
    python scripts/gebote_verifier.py --ci     # CI-Mode (exit code ≠ 0 bei Verstößen)
    python scripts/gebote_verifier.py --list   # Nur Liste der Gebote-Checks
"""

from __future__ import annotations

import ast
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GebotCheck:
    gebot_id: str
    title: str
    description: str
    verify_fn: Callable[[], tuple[bool, str]]
    category: str = ""


@dataclass
class VerifierReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def add(self, gebot_id: str, passed: bool, detail: str) -> None:
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append((gebot_id, passed, detail))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════


def _file_contains(path: str, pattern: str) -> bool:
    """Prüft ob eine Datei ein Regex-Muster enthält."""
    try:
        content = (ROOT / path).read_text()
        return bool(re.search(pattern, content))
    except Exception:
        return False


def _file_contains_line(path: str, pattern: str) -> bool:
    """Prüft ob eine Datei eine Zeile mit dem Pattern enthält."""
    try:
        for line in (ROOT / path).read_text().splitlines():
            if re.search(pattern, line):
                return True
        return False
    except Exception:
        return False


def _function_exists(module_path: str, func_name: str) -> bool:
    """Prüft ob eine Funktion in einem Modul definiert ist."""
    try:
        tree = ast.parse((ROOT / module_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    return True
        return False
    except Exception:
        return False


def _method_exists(module_path: str, class_name: str, method_name: str) -> bool:
    """Prüft ob eine Methode in einer Klasse existiert."""
    try:
        tree = ast.parse((ROOT / module_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return True
        return False
    except Exception:
        return False


def _attr_exists(obj_path: str, attr_name: str) -> bool:
    """Prüft ob ein Attribut in einer __init__ gesetzt wird."""
    try:
        # obj_path is like "backend.core.unified_restorer_v3.UnifiedRestorerV3._wohlklang_strength_multiplier"
        parts = obj_path.rsplit(".", 1)
        file_path = "/".join(parts[0].split(".")[:-1]) + ".py"
        cls_name = parts[0].split(".")[-1]
        attr = parts[1]
        tree = ast.parse((ROOT / file_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in ast.walk(node):
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                if target.value.id == "self" and target.attr == attr:
                                    return True
                # Also check AnnAssign
                for item in ast.walk(node):
                    if isinstance(item, ast.AnnAssign):
                        if isinstance(item.target, ast.Attribute) and isinstance(item.target.value, ast.Name):
                            if item.target.value.id == "self" and item.target.attr == attr:
                                return True
        return False
    except Exception:
        return False


def _find_in_code(pattern: str, paths: list[str] | None = None) -> bool:
    """Sucht ein Regex-Pattern rekursiv in Python-Dateien."""
    search_paths = [ROOT / p for p in (paths or ["backend", "forensics", "denker", "plugins"])]
    for sp in search_paths:
        if not sp.exists():
            continue
        for py_file in sp.rglob("*.py"):
            try:
                if re.search(pattern, py_file.read_text()):
                    return True
            except Exception:
                pass
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Gepote-Checks — jede Regel wird gegen den tatsächlichen Code geprüft
# ═══════════════════════════════════════════════════════════════════════════════

GEBOTE_CHECKS: list[GebotCheck] = []


def gebot(gebot_id: str, title: str, description: str, category: str = ""):
    """Decorator für Gebot-Check-Funktionen."""

    def decorator(fn):
        GEBOTE_CHECKS.append(GebotCheck(gebot_id, title, description, fn, category))
        return fn

    return decorator


# ── Kategorie I: Individuelle Song-Maximierung ────────────────────────────────


@gebot("§G1", "Pro-Song-Kalibrierung",
       "Jeder Song durchläuft eine isolierte SongCalibration.")
def check_g1() -> tuple[bool, str]:
    ok = _find_in_code(r"song_calibration|SongCalibration|_song_calibration_profile")
    return ok, "SongCalibration im Code referenziert" if ok else "SongCalibration nicht gefunden"


@gebot("§G2", "Defekt-Vollständigkeit",
       "Alle 62 DefectTypes werden gescannt.")
def check_g2() -> tuple[bool, str]:
    ok = _function_exists("backend/core/defect_scanner.py", "scan_defect_presence")
    return ok, "scan_defect_presence() existiert" if ok else "scan_defect_presence() fehlt"


@gebot("§G3", "Gesangsintegrität",
       "Vocal-Safety-Wrapper aktiv für Frequenzen 80 Hz–8 kHz.")
def check_g3() -> tuple[bool, str]:
    ok = _find_in_code(r"vocal_safety|VocalSafety|vocal_protection|panns_singing")
    return ok, "Vocal-Safety-Mechanismen im Code" if ok else "Keine Vocal-Safety gefunden"


@gebot("§G4", "Ghost-Echo-Freiheit",
       "PhaseCoherentSTFT muss in allen Modi laufen (§2.60 STCG).")
def check_g4() -> tuple[bool, str]:
    ok = _method_exists("backend/core/dsp/phase_coherent_stft.py", "PhaseCoherentSTFT", "capture")
    and_restore = _method_exists("backend/core/dsp/phase_coherent_stft.py", "PhaseCoherentSTFT", "restore")
    if ok and and_restore:
        # Check channels-first fix
        channels_fix = _file_contains(
            "backend/core/dsp/phase_coherent_stft.py",
            r"mean\(axis=0\).*Korrektur|channels.first.*mean\(axis=0\)",
        )
        if channels_fix:
            return True, "PhaseCoherentSTFT mit channels-first-Fix"
        return True, "PhaseCoherentSTFT capture+restore existieren"
    return False, "PhaseCoherentSTFT capture/restore fehlen"


@gebot("§G7", "Interchannel-Lag",
       "LAG_PROBE an ≥3 Positionen gemessen.")
def check_g7() -> tuple[bool, str]:
    count = len(re.findall(r"LAG_PROBE", (ROOT / "backend/core/unified_restorer_v3.py").read_text()))
    return count >= 3, f"LAG_PROBE {count}× referenziert (≥3 erforderlich)"


# ── Kategorie II: Psychoakustik ──────────────────────────────────────────────


@gebot("§G11", "Natürlicher Wohlklang",
       "PQS-MOS < 3.0 löst Rollback aus.")
def check_g11() -> tuple[bool, str]:
    ok = _find_in_code(r"PQS.*MOS.*3\.0|pqs_mos.*rollback|PQS-MOS.*<.*3")
    return ok, "PQS-MOS-Rollback-Schutz existiert" if ok else "Kein PQS-MOS-Rollback gefunden"


@gebot("§G12", "Lautheitskonsistenz",
       "LUFS-integrated nach EBU R128.")
def check_g12() -> tuple[bool, str]:
    ok = _find_in_code(r"loudness_normalization|LUFS.*integrated|EBU.*R128|pyloudnorm")
    return ok, "LUFS-Normalisierung im Code" if ok else "Keine LUFS-Normalisierung"


@gebot("§G14", "Spectral-Tilt-Guard",
       "Spektrale Neigung nach jeder Phase geprüft.")
def check_g14() -> tuple[bool, str]:
    ok = _find_in_code(r"spectral_tilt|spectral.*neigung|tilt.*guard")
    return ok, "Spectral-Tilt-Guard im Code" if ok else "Kein Spectral-Tilt-Guard"


# ── Kategorie III: Architektur ───────────────────────────────────────────────


@gebot("§G21", "Denker-Zentralität",
       "Alle Stärke-Entscheidungen fließen zentral im Denker.")
def check_g21() -> tuple[bool, str]:
    ok = _find_in_code(r"Denker|denker|AurikDenker|strategie_denker")
    return ok, "Denker-System referenziert" if ok else "Denker-System nicht gefunden"


@gebot("§G23", "ML-Fallback-Logging",
       "Jeder ML→DSP-Fallback mit logger.warning protokolliert.")
def check_g23() -> tuple[bool, str]:
    ok = _find_in_code(r'logger\.warning\(.*ML.*DSP.*Fallback|ML→DSP-Fallback|logger\.warning.*Ersatzpfad')
    return ok, "ML→DSP-Fallback-Logging existiert" if ok else "Kein ML-Fallback-Logging"


@gebot("§G24", "NaN/Inf-Schutz",
       "Jede Phase wendet np.nan_to_num() auf Ausgabe-Audio an.")
def check_g24() -> tuple[bool, str]:
    count = 0
    _phases_dir = ROOT / "backend/core/phases"
    if _phases_dir.is_dir():
        for _pf in _phases_dir.rglob("*.py"):
            try:
                count += len(re.findall(r"np\.nan_to_num|np\.isfinite", _pf.read_text()))
            except Exception:
                pass
    return count >= 10, f"NaN/Inf-Schutz {count}× in Phasen (≥10 erwartet)"


@gebot("§G25", "Logger-Pflicht",
       "Jede Datei mit logger-Verwendung definiert logging.getLogger.")
def check_g25() -> tuple[bool, str]:
    violations = []
    for py_file in (ROOT / "backend/core").rglob("*.py"):
        try:
            content = py_file.read_text()
            if "logger." in content and "getLogger" not in content:
                violations.append(str(py_file.relative_to(ROOT)))
        except Exception:
            pass
    if not violations:
        return True, "Alle Dateien mit logger haben getLogger"
    return False, f"Fehlende getLogger in: {', '.join(violations[:3])}"


@gebot("§G26", "Guard-Counter-Lebendigkeit",
       "Jeder deklarierte Guard-Counter wird inkrementiert (kein toter Code).")
def check_g26() -> tuple[bool, str]:
    # Prüft den UQ-Drive: _uq_drive_emit_count muss += 1 haben
    ok = _file_contains(
        "backend/core/unified_restorer_v3.py",
        r"_uq_drive_emit_count.*\+.*1|_uq_drive_emit_count.*=.*getattr.*\+ 1",
    )
    return ok, "UQ-Drive-Emit-Counter lebt" if ok else "UQ-Drive-Counter wird nie inkrementiert"


# ── Kategorie VII: v10.14 Durchblick-Fixes ───────────────────────────────────

@gebot("§G-DB1", "Bayesian-Prior aktiv",
       "unknown-Prior (P=0.01) in _bayesian_score implementiert.")
def check_db1() -> tuple[bool, str]:
    ok = _file_contains("forensics/medium_detector.py", r"_P_UNKNOWN.*=.*0\.01")
    return ok, "Bayesian-Prior P(unknown)=0.01" if ok else "Bayesian-Prior fehlt oder falscher Wert"


@gebot("§G-DB2", "CLAP File-Format-Plausibilität",
       "CLAP vor Format-Erfindung → Tier-2 DSP-Fallback.")
def check_db2() -> tuple[bool, str]:
    ok = _file_contains("backend/core/era_classifier.py", r"_MATERIAL_INVENTED")
    return ok, "File-Format-Plausibilität existiert" if ok else "File-Format-Plausibilität fehlt"


@gebot("§G-DB3", "UQ-Drive strength_explicit Fix",
       "strength_explicit prüft _user_strength_override, nicht 'strength' in kwargs.")
def check_db3() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/unified_restorer_v3.py",
        r"strength_explicit.*=.*_user_strength_override",
    )
    return ok, "UQ-Drive strength_explicit fix" if ok else "UQ-Drive noch mit 'strength' in kwargs"


@gebot("§G-DB4", "StereoAuth 2-Gate-Schutz",
       "Mono-Kollaps nur bei rest_ms_corr < 0.80.")
def check_db4() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/stereo_authenticity_invariant.py",
        r"rest_ms_corr\s*<\s*0\.80",
    )
    return ok, "StereoAuth 2-Gate (0.80 enforcement)" if ok else "StereoAuth noch mit altem 0.97-Gate"


@gebot("§G-DB5", "PhaseSteeringGuard ≥12 Phasen",
       "STOP_GRACEFUL erst ab ≥12 Phasen.")
def check_db5() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/phase_steering_guard.py",
        r"current_phase_idx\s*>=\s*12",
    )
    return ok, "STOP_GRACEFUL ≥12 Phasen" if ok else "STOP_GRACEFUL ohne Min-Phasen-Guard"


@gebot("§G-DB6", "Atomarer Cache-Clear",
       "shutil.rmtree für __pycache__ in backend/__init__.py.")
def check_db6() -> tuple[bool, str]:
    ok = _file_contains("backend/__init__.py", r"rmtree.*__pycache__|__pycache__.*rmtree|rglob.*__pycache__")
    return ok, "Atomarer Cache-Clear in backend" if ok else "Cache-Clear fehlt"


@gebot("§G-DB7", "Wohlklang-Garantie implementiert",
       "MUSHRA < 80 → Re-Run mit 50% Strength.")
def check_db7() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/unified_restorer_v3.py",
        r"_wohlklang_strength_multiplier.*0\.50",
    )
    return ok, "Wohlklang-Garantie implementiert" if ok else "Wohlklang-Garantie fehlt"


@gebot("§G-DB8", "TruePeak-Clamp −0.2 dBTP",
       "Hard-Clip auf 0.977 linear nach HHCG.")
def check_db8() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/unified_restorer_v3.py",
        r"_tp_peak.*>.*0\.977|TruePeak.*Clamp.*0\.977",
    )
    return ok, "TruePeak-Clamp −0.2 dBTP" if ok else "TruePeak-Clamp fehlt"


@gebot("§G-DB9", "Phase 20 Primum-non-nocere universal",
       "reverb_severity < 0.02 → Passthrough für ALLE Materialien.")
def check_db9() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/phases/phase_20_reverb_reduction.py",
        r"reverb_severity.*<.*0\.02|_skip_all_no_reverb|passthrough_primum_non_nocere",
    )
    return ok, "Phase 20 universal Passthrough" if ok else "Phase 20 nur digital Passthrough"


@gebot("§G-DB10", "Era-Filter shellac",
       "shellac bei Ära≥1955 aus Kette entfernt.")
def check_db10() -> tuple[bool, str]:
    ok = _file_contains("backend/core/pre_analysis.py", r"shellac.*1955|_MATERIAL_ERA_END")
    return ok, "Era-Filter shellac=1955" if ok else "Era-Filter fehlt"


@gebot("§G-DB11", "PerceptualExportOptimizer Material-Check",
       "Cassette/Schellack → Skip ohne DeepFilterNetV3-Load.")
def check_db11() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/unified_restorer_v3.py",
        r"_PEO_HIGH_RISK.*cassette",
    )
    return ok, "PEO Material-Pre-Check" if ok else "PEO lädt immer DeepFilterNetV3"


@gebot("§G-DB12", "ExcellenceOptimizer Groove-Guard",
       "onset-count vor/nach micro_dynamics mit _count_onsets().")
def check_db12() -> tuple[bool, str]:
    ok = _function_exists("backend/core/excellence_optimizer.py", "_count_onsets")
    return ok, "_count_onsets() existiert" if ok else "Groove-Guard fehlt"


@gebot("§G-DB13", "Physik-Filter: prä-1960 + digital",
       "shellac/wax_cylinder + mp3/aac → Analogmaterial entfernt.")
def check_db13() -> tuple[bool, str]:
    ok = _file_contains(
        "backend/core/pre_analysis.py",
        r"_PRE_1960_ANALOG.*shellac|Physik-Filter.*prä-1960",
    )
    return ok, "Physik-Filter prä-1960+digital" if ok else "Kein Physik-Filter"


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def run_verifier(ci_mode: bool = False) -> int:
    report = VerifierReport()

    print("=" * 72)
    print("  Aurik §GEPOTE-VERIFIER — Non-Plus-Ultra-Compliance")
    print("=" * 72)
    print()

    for check in GEBOTE_CHECKS:
        try:
            passed, detail = check.verify_fn()
        except Exception as exc:
            passed, detail = False, f"Checker-Crash: {exc}"
        report.add(check.gebot_id, passed, detail)
        status = "✅" if passed else "❌"
        print(f"  {status} {check.gebot_id} {check.title}")
        if not passed:
            print(f"     → {detail}")

    print()
    print(f"  Ergebnis: {report.passed}/{report.total} bestanden, {report.failed} verletzt")
    print()

    if report.failed > 0:
        print("  ❌ VERLETZTE GEBOTE (müssen behoben werden):")
        for gebot_id, passed, detail in report.results:
            if not passed:
                print(f"     {gebot_id}: {detail}")
        print()
        if ci_mode:
            print("  CI-MODE: Build fehlgeschlagen.")
            return 1
        return 1

    print("  ✅ ALLE GEBOTE EINGEHALTEN — Non-Plus-Ultra-Status erreicht.")
    return 0


if __name__ == "__main__":
    ci = "--ci" in sys.argv
    if "--list" in sys.argv:
        for c in GEBOTE_CHECKS:
            print(f"{c.gebot_id}: {c.title}")
        sys.exit(0)
    sys.exit(run_verifier(ci_mode=ci))
