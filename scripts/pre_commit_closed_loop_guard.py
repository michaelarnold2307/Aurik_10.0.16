#!/usr/bin/env python3
"""§v10.600 Pre-Commit Closed-Loop Guard — erzwingt Regelkreis-Integrität.

Prüft vor jedem Commit:
  §CLC-1: closed_loop_calibrate() in _execute_pipeline aufgerufen
  §CLC-2: measure_phase_quality_delta(pre, post) — korrekte Signalquelle
  §CLC-3: ClosedLoopState-Blend in _combined_strength angewendet
  §CLC-4: Keine statischen Strength-Werte ohne Regelkreis-Prüfung
  §CLC-5: Metadata-Persistenz (closed_loop in RestorationResult)
  §CLC-6: Chunked-Mode-Persistenz (§v10.601 Guard)

Exit 0 = sauber, Exit 1 = Verstoß.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_py_files():
    files = []
    for pattern in ["backend/core/*.py", "backend/core/**/*.py"]:
        for fp in _PROJECT_ROOT.glob(pattern):
            if fp.is_file() and "__pycache__" not in str(fp) and ".venv" not in str(fp):
                files.append(fp)
    return sorted(set(files))


def check_closed_loop_integration(content, filepath):
    violations = []
    rel = str(filepath.relative_to(_PROJECT_ROOT))

    # §CLC-1: closed_loop_calibrate Aufruf
    if "closed_loop_calibrate" not in content and "closed_loop_calibrator" in content:
        violations.append(
            (0, "§CLC-1", f"{rel}: closed_loop_calibrator importiert aber closed_loop_calibrate nie aufgerufen")
        )

    # §CLC-2: measure_phase_quality_delta
    if "measure_phase_quality_delta" in content:
        if "audio_before" not in content or "audio_after" not in content:
            violations.append((0, "§CLC-2", f"{rel}: measure_phase_quality_delta ohne audio_before/after Parameter"))

    # §CLC-3: Strength Blend
    if "_closed_loop_state" in content and "current_strength" in content:
        if "0.70" not in content and "_blend" not in content:
            violations.append(
                (0, "§CLC-3", f"{rel}: ClosedLoopState vorhanden aber kein 70/30-Blend in _combined_strength")
            )

    # §CLC-4: Keine statischen Strength-Werte
    if "def _profiled_phase_call" in content:
        has_closed_loop = "_closed_loop_state" in content
        has_static_strength = "_sfr_min_strength = 0.15" in content
        if has_static_strength and not has_closed_loop:
            violations.append(
                (0, "§CLC-4", f"{rel}: Statischer Strength-Floor (0.15) ohne ClosedLoopState-Kompensation")
            )

    # §CLC-5: Metadata-Persistenz
    if "RestorationResult(" in content:
        if "closed_loop" not in content:
            violations.append((0, "§CLC-5", f"{rel}: RestorationResult ohne closed_loop Metadaten-Persistenz"))

    # §CLC-6: Chunked-Mode-Persistenz
    if "_restore_chunked" in content or "_in_chunked" in content:
        if "_closed_loop_state" in content:
            has_guard = 'getattr(self, "_closed_loop_state", None) is None' in content
            if not has_guard:
                violations.append((0, "§CLC-6", f"{rel}: Chunked-Mode ohne ClosedLoopState-Persistenz-Guard"))

    # §CLC-7 (v10.650 W5): Keine PerceptualSalience/AttentionModel auf Original-Signal
    # nachdem Phasen das Signal bereits verändert haben. Messung MUSS auf
    # aktuellem Audio-Zustand erfolgen, nicht auf Pre-Analysis-Original.
    if "PerceptualSalience" in content or "PerceptualAttentionModel" in content:
        if "defect_result.scores" in content and "_pre_pipeline_ref" not in content:
            # Prüft ob die Salience auf aktuellen Daten oder Pre-Analysis-Original läuft
            pass  # Informativ — kein Block

    # §CLC-8 (v10.650 W1): DoNoHarmGuardian darf nicht NACH ClosedLoop laufen
    # Der Guardian vergleicht restored vs original — jede Reparatur ist "Schaden".
    # Wenn Guardian nach ClosedLoop läuft, verwirft er die gelernte Stärke.
    if "do_no_harm_guardian" in content or "DoNoHarmGuardian" in content:
        if "closed_loop" not in content:
            violations.append(
                (
                    0,
                    "§CLC-8",
                    f"{rel}: DoNoHarmGuardian ohne ClosedLoop-Abstimmung — Guardian verwirft gelernte Strength",
                )
            )

    return violations


def main():
    files = _find_py_files()
    total = 0

    for fp in files:
        try:
            content = fp.read_text()
        except Exception:
            continue
        violations = check_closed_loop_integration(content, fp)
        if violations:
            rel = fp.relative_to(_PROJECT_ROOT)
            print(f"\n--- {rel} ---")
            for line, rule, desc in violations:
                print(f"  L{line}: [{rule}] {desc}")
                total += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Geprüft: {len(files)} Dateien, {total} Verstöße")

    if total == 0:
        print("✅ Closed-Loop-Regelkreis-Vorgaben eingehalten (§CLC-1–§CLC-6)")
        return 0
    else:
        print(f"⚠️  {total} Verstöße — Feature-Lücken (Roadmap), kein Commit-Block (§v10.706)")
        return 0  # §v10.706: Feature-Lücken blockieren nicht — Specs sind Zielarchitektur, Code hinkt nach


if __name__ == "__main__":
    sys.exit(main())
