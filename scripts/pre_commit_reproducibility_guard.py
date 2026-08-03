#!/usr/bin/env python3
"""§v10.702/§v10.703 Reproduzierbarkeits- und Autopilot-Guard (§G131–§G141, §V40–§V44).

Prüft vor jedem Commit die architektonischen Garantien für vollständige SOTA-
Reproduzierbarkeit. Exit 0 wenn alle Garantien erfüllt, Exit 1 bei Verstößen.

Usage: python3 scripts/pre_commit_reproducibility_guard.py [file ...]
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# §v10.706: Feature-Lücken (Roadmap) — warnen, nicht blocken
SOFT_GAPS: list[str] = []
# Echte Architektur-Verstöße (z.B. R11 Aurik10-Import) — blocken weiterhin
HARD_VIOLATIONS: list[str] = []

# ── §G131: MQA muss musical_improvement als primäre Verbesserungs-Metrik nutzen ──
def check_b1_perceptual_improvement_metric(filepath: Path) -> None:
    """§G131: musical_improvement MUSS primäre Verbesserungs-Metrik sein (§V40/§G138)."""
    if "musical_quality_assurance.py" not in str(filepath):
        return
    content = filepath.read_text()
    # Positiv-Check: musical_improvement > 0.005 muss vorhanden sein
    if "musical_improvement > 0.005" not in content:
        SOFT_GAPS.append(f"{filepath}: §G131 — musical_improvement > 0.005 Check fehlt. (Feature-Lücke)")
        return
    # Negativ-Check: darf KEINEN reinen BlindQuality-Ratio-Check haben
    if "_minimal_improvement = (_output_score / _input_score)" in content:
        HARD_VIOLATIONS.append(f"{filepath}: §V40/§G138 — reiner BlindQuality-Ratio-Check gefunden.")
    # Check: quality_uncertain muss vorhanden sein
    if "quality_uncertain" not in content:
        SOFT_GAPS.append(f"{filepath}: §G138 — quality_uncertain Schutz fehlt. (Feature-Lücke)")

# ── §G133: defect_reduction_per_type muss in RestorationResult + _mqa_result stehen ──
def check_b2_defect_reduction(filepath: Path) -> None:
    """Prüft, ob defect_reduction_per_type in RestorationResult und _mqa_result gespeichert wird."""
    if "unified_restorer_v3.py" not in str(filepath):
        return
    content = filepath.read_text()
    # Prüfe: defect_reduction_per_type in _mqa_result
    if '"defect_reduction_per_type"' not in content:
        SOFT_GAPS.append(f"{filepath}: §G133 — defect_reduction_per_type fehlt in _mqa_result. (Feature-Lücke)")
    # Prüfe: defect_reduction_per_type in metadata
    if "'defect_reduction_per_type'" not in content and '"defect_reduction_per_type"' not in content:
        SOFT_GAPS.append(f"{filepath}: §G133 — defect_reduction_per_type fehlt in RestorationResult.metadata. (Feature-Lücke)")

# ── §G135: Chunked-Streaming muss State einfrieren ──
def check_b3_chunked_determinism(filepath: Path) -> None:
    """Prüft, ob _restore_chunked Pre-Analysis-State für Folge-Chunks einfriert."""
    if "unified_restorer_v3.py" not in str(filepath):
        return
    content = filepath.read_text()
    # Prüfe: _b3_frozen_calibration_profile muss in _restore_chunked gesetzt werden
    if "_b3_frozen_calibration_profile" not in content:
        SOFT_GAPS.append(f"{filepath}: §G135 — _b3_frozen_calibration_profile fehlt in _restore_chunked. (Feature-Lücke)")
    if "_b3_frozen_phase_plan" not in content:
        SOFT_GAPS.append(f"{filepath}: §G135 — _b3_frozen_phase_plan fehlt in _restore_chunked. (Feature-Lücke)")
    if "_b3_skip_pre_analysis" not in content:
        SOFT_GAPS.append(f"{filepath}: §G135 — _b3_skip_pre_analysis Bypass fehlt. (Feature-Lücke)")
    # Prüfe: Phase-Override am Loop-Start (selected_phases = list(_b3_frozen_phases))
    if "_b3_frozen_phases" in content and "selected_phases = list(_b3_frozen_phases)" not in content:
        SOFT_GAPS.append(f"{filepath}: §G135 — selected_phases wird nicht aus _b3_frozen_phases übernommen. (Feature-Lücke)")

# ── §V18 (R11): Kein Aurik10-Import aus backend ──
def check_r11_no_aurik10_import(filepath: Path) -> None:
    """Backend darf nicht aus Aurik10 importieren."""
    if "backend" not in str(filepath):
        return
    # bridge.py ist die EINZIGE Ausnahme — aber nur über backend.api.bridge_calibration_data
    content = filepath.read_text()
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        if re.match(r'\s*(from|import)\s+Aurik10', line):
            # bridge.py ist OK wenn es von bridge_calibration_data importiert
            if "bridge.py" in str(filepath) and "bridge_calibration_data" in line:
                continue
            HARD_VIOLATIONS.append(f"{filepath}:{i}: §V18 — from Aurik10.* import ist verboten. Nutze backend.api.bridge_calibration_data.")


# ── §G137 (B3-P2): Full-Song Defect-Presence Pre-Scan ──
def check_b3p2_full_song_defect_presence(filepath: Path) -> None:
    """§G137: scan_defect_presence() und Early Merge muessen existieren."""
    if "defect_scanner.py" in str(filepath):
        c = filepath.read_text()
        if "def scan_defect_presence" not in c:
            SOFT_GAPS.append(f"{filepath}: §G137 — scan_defect_presence() fehlt im DefectScanner. (Feature-Lücke)")
    if "unified_restorer_v3.py" in str(filepath):
        c = filepath.read_text()
        if "_b3_full_song_defect_types" not in c:
            SOFT_GAPS.append(f"{filepath}: §G137 — _b3_full_song_defect_types fehlt in _restore_chunked. (Feature-Lücke)")

# ── §V40/§G138: BlindQuality nicht als alleinige Ground Truth ──
def check_v40_no_blindquality_ground_truth(filepath: Path) -> None:
    """§V40/§G138: BlindQuality darf NICHT alleinige Verbesserungs-Metrik sein."""
    if "musical_quality_assurance.py" not in str(filepath):
        return
    c = filepath.read_text()
    if "_minimal_improvement = (_output_score / _input_score)" in c:
        HARD_VIOLATIONS.append(f"{filepath}: §V40/§G138 — reiner BlindQuality-Ratio-Check ohne musical_improvement.")
    if "_score_ratio >= 1" in c and "quality_uncertain" not in c:
        HARD_VIOLATIONS.append(f"{filepath}: §G138 — BlindQuality-Fallback ohne QUALITY_UNCERTAIN protection.")

# ── §G139: Defekt-Countdown ──
def check_g139_defect_countdown(filepath: Path) -> None:
    """§G139: defect_countdown muss berechnet und gespeichert werden."""
    if "unified_restorer_v3.py" not in str(filepath):
        return
    c = filepath.read_text()
    if "defect_countdown" not in c:
        SOFT_GAPS.append(f"{filepath}: §G139 — defect_countdown fehlt in RestorationResult/Backend. (Feature-Lücke)")

# ── §G140: export_gate ──
def check_g140_export_gate(filepath: Path) -> None:
    """§G140: export_gate() muss existieren."""
    if "musical_quality_assurance.py" not in str(filepath):
        return
    c = filepath.read_text()
    if "def export_gate" not in c:
        SOFT_GAPS.append(f"{filepath}: §G140 — export_gate() Methode fehlt. (Feature-Lücke)")
    if "zero_audible_defects" not in c:
        SOFT_GAPS.append(f"{filepath}: §G140 — zero_audible_defects Check fehlt in export_gate(). (Feature-Lücke)")

# ── §G142: Per-Band-MUSHRA ──
def check_g142_per_band_mushra(filepath: Path) -> None:
    """§G142: per_band_mushra Modul muss existieren mit 24 Bark-Bändern."""
    if "per_band_mushra.py" in str(filepath):
        c = filepath.read_text()
        if "class PerBandMUSHRA" not in c and "class PerBandMushraResult" not in c:
            SOFT_GAPS.append(f"{filepath}: §G142 — PerBandMUSHRA/PerBandMushraResult Klasse fehlt. (Feature-Lücke)")
        if "BARK_BAND_EDGES_HZ" not in c:
            SOFT_GAPS.append(f"{filepath}: §G142 — Bark-Band-Definitionen (BARK_BAND_EDGES_HZ) fehlen. (Feature-Lücke)")

# ── §G144/§G145: MUSHRA-Proxy + Rollback ──
def check_g144_g145_mushra_proxy_rollback(filepath: Path) -> None:
    """§G144/§G145: MUSHRA-Proxy muss in _safe_process integriert sein."""
    if "phase_interface.py" in str(filepath):
        c = filepath.read_text()
        if "mushra_proxy" not in c:
            SOFT_GAPS.append(f"{filepath}: §G144 — MUSHRA-Proxy fehlt in _safe_process(). (Feature-Lücke)")
        if "mushra_proxy_delta" not in c:
            SOFT_GAPS.append(f"{filepath}: §G144 — mushra_proxy_delta fehlt in result.metadata. (Feature-Lücke)")
        if "mushra_proxy_rollback" not in c:
            SOFT_GAPS.append(f"{filepath}: §G145 — Rollback-Logik (mushra_proxy_rollback) fehlt. (Feature-Lücke)")
    if "mushra_proxy.py" in str(filepath):
        c = filepath.read_text()
        if "class MushraProxy" not in c:
            SOFT_GAPS.append(f"{filepath}: §G144 — MUSHRAProxy Klasse fehlt. (Feature-Lücke)")

# ── §G141: wohlklang_garantie_check ──
def check_g141_wohlklang_garantie(filepath: Path) -> None:
    """§G141: wohlklang_garantie_check() muss existieren."""
    if "musical_quality_assurance.py" not in str(filepath):
        return
    c = filepath.read_text()
    if "def wohlklang_garantie_check" not in c:
        SOFT_GAPS.append(f"{filepath}: §G141 — wohlklang_garantie_check() fehlt. (Feature-Lücke)")

def _find_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_file() and pp.suffix == ".py":
            files.append(pp)
        elif pp.is_dir():
            files.extend(pp.rglob("*.py"))
    return files


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["backend"]
    files = _find_files(paths)

    # Nur relevante Dateien prüfen
    for f in files:
        try:
            check_b1_perceptual_improvement_metric(f)
            check_b2_defect_reduction(f)
            check_b3_chunked_determinism(f)
            check_b3p2_full_song_defect_presence(f)
            check_r11_no_aurik10_import(f)
            check_v40_no_blindquality_ground_truth(f)
            check_g139_defect_countdown(f)
            check_g140_export_gate(f)
            check_g141_wohlklang_garantie(f)
            check_g142_per_band_mushra(f)
            check_g144_g145_mushra_proxy_rollback(f)
        except SyntaxError:
            pass
        except Exception as e:
            print(f"WARN: {f}: {e}", file=sys.stderr)

    if SOFT_GAPS:
        print(f"\n⚠️  {len(SOFT_GAPS)} Feature-Lücken (Roadmap, kein Commit-Block §v10.706):\n")
        for v in SOFT_GAPS:
            print(f"  {v}")
        print("\n👉 Specs sind Zielarchitektur — Code-Implementierung folgt in späteren Sprints.")

    if HARD_VIOLATIONS:
        print(f"\n❌ {len(HARD_VIOLATIONS)} Architektur-Verstöße gegen §v10.702:\n")
        for v in HARD_VIOLATIONS:
            print(f"  {v}")
        print("\n👉 Siehe .github/specs/v10.702_critical_bugs_b1_b2_b3.md und GEBOTE.md Kategorie XIX")
        sys.exit(1)
    else:
        if SOFT_GAPS:
            print(f"\n✅ §v10.702: Keine Architektur-Verstöße ({len(SOFT_GAPS)} Feature-Lücken via §v10.706 toleriert)")
        else:
            print("✅ §v10.702 Reproduzierbarkeits-Guard: Alle B1/B2/B3-Garantien erfüllt")
        sys.exit(0)


if __name__ == "__main__":
    main()
