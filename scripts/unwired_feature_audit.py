#!/usr/bin/env python3
"""
§v10.118 Unverkabelte-Feature-Audit — SOTA-Features flächendeckend aktivieren.

Scannt alle 65 Phasen und identifiziert, welche SOTA-Features noch nicht
genutzt werden — obwohl sie verfügbar und für die Phase relevant sind.

Usage: PYTHONPATH=. python3 scripts/unwired_feature_audit.py
"""

import os
import re
import sys
from pathlib import Path

PHASE_DIR = Path("backend/core/phases")

# SOTA-Features, die in JEDER Phase genutzt werden sollten
SOTA_FEATURES = {
    "safe_stft": {
        "check": lambda content: "safe_stft" in content or "scipy.signal.stft" not in content.replace("safe_stft", ""),
        "relevance": lambda content, name: (
            "scipy.signal.stft" in content or "signal.stft" in content or "scipy.signal.istft" in content
        ),
        "recommendation": "from backend.core.audio_utils import safe_stft, safe_istft",
        "impact": "Verhindert STFT-Crash bei kurzen Signalen (nperseg > input_length)",
    },
    "soft_clip": {
        "check": lambda content: (
            "apply_soft_clip" in content or "PhaseResult" in content or "create_phase_result" in content
        ),
        "relevance": lambda content, name: (
            "np.clip" in content and "audio" in content and "phase_interface" not in name
        ),
        "recommendation": "Nutze PhaseResult.__post_init__ (automatisches soft_clip) oder apply_soft_clip()",
        "impact": "Verhindert hörbare Rechteck-Clipping-Artefakte (tanh-basiert)",
    },
    "perceptual_blend": {
        "check": lambda content: "perceptual_blend" in content,
        "relevance": lambda content, name: (
            ("blend" in content.lower() or "wet" in content.lower() or "dry" in content.lower())
            and "mastering" not in name
            and "limiting" not in name
        ),
        "recommendation": "from backend.core.dsp.perceptual_blend import perceptual_blend",
        "impact": "Frequenzabhängiger Blend nach Bark-Bändern — unhörbare Änderungen werden ausmaskiert",
    },
    "breath_preserver": {
        "check": lambda content: "breath_preserver" in content.lower() or "BreathPreserver" in content,
        "relevance": lambda content, name: (
            any(kw in name for kw in ("denoise", "noise", "hiss", "nr", "03", "29", "28"))
            or ("noise" in content.lower() and "reduction" in content.lower())
        ),
        "recommendation": "BreathPreserver.protect_breath() vor NR, restore_breath() danach",
        "impact": "Atemgeräusche bleiben erhalten — essentiell für natürlichen Gesangsklang",
    },
    "safe_process": {
        "check": lambda content: (
            "_safe_process" in content or "phase_interface" in content or "PhaseInterface" in content
        ),
        "relevance": lambda content, name: (
            "def process" in content and "phase_interface" not in name and "phase_glue" not in name
        ),
        "recommendation": "Nutze PhaseInterface._safe_process() statt process() direkt",
        "impact": "Aktiviert RMS-Guard, Formant-Guard, Transient-Guard, Hallucination-Guard",
    },
    "gated_rms": {
        "check": lambda content: "gated_rms" in content.lower() or "compute_gated" in content,
        "relevance": lambda content, name: (
            "rms" in content.lower() and "np.mean" in content and "phase_interface" not in name
        ),
        "recommendation": "from backend.core.audio_utils import compute_gated_rms_linear",
        "impact": "Gated RMS (nur Frames > -50 dBFS) — akkuratere Lautheitsmessung als Raw-RMS",
    },
}


def audit_phase(filepath: Path) -> list[dict]:
    """Audit a single phase file for unwired SOTA features."""
    with open(filepath) as f:
        content = f.read()

    name = filepath.stem
    findings = []

    for feat_name, feat_info in SOTA_FEATURES.items():
        if not feat_info["relevance"](content, name):
            continue  # Not relevant for this phase
        if feat_info["check"](content):
            continue  # Already using it

        findings.append(
            {
                "phase": name,
                "feature": feat_name,
                "recommendation": feat_info["recommendation"],
                "impact": feat_info["impact"],
            }
        )

    return findings


def main():
    phases = sorted(Path("backend/core/phases").glob("phase_*.py"))

    all_findings = []
    for fpath in phases:
        if fpath.name == "phase_interface.py" or fpath.name == "__init__.py":
            continue
        findings = audit_phase(fpath)
        all_findings.extend(findings)

    if not all_findings:
        print("✅ Alle SOTA-Features sind in allen relevanten Phasen verkabelt.")
        return

    # Group by feature
    by_feature = {}
    for f in all_findings:
        feat = f["feature"]
        if feat not in by_feature:
            by_feature[feat] = []
        by_feature[feat].append(f)

    print("=" * 70)
    print("§v10.118 Unverkabelte-Feature-Audit")
    print(f"{len(all_findings)} ungenutzte SOTA-Features in {len(set(f['phase'] for f in all_findings))} Phasen")
    print("=" * 70)

    for feat, items in sorted(by_feature.items(), key=lambda x: -len(x[1])):
        print(f"\n── {feat} ({len(items)} Phasen) ──")
        print(f"   {items[0]['impact']}")
        print(f"   Recommendation: {items[0]['recommendation']}")
        phases_list = [i["phase"] for i in items[:5]]
        print(f"   Phases: {', '.join(phases_list)}")
        if len(items) > 5:
            print(f"   ... und {len(items) - 5} weitere")

    # Quick-win summary
    print(f"\n{'=' * 70}")
    print("Quick-Win-Priorität (niedrigster Aufwand → höchste Wirkung):")
    print(f"  1. safe_process:    {len(by_feature.get('safe_process', []))} Phasen → systemische Guards aktivieren")
    print(f"  2. soft_clip:       {len(by_feature.get('soft_clip', []))} Phasen → Clipping-Artefakte eliminieren")
    print(f"  3. breath_preserver:{len(by_feature.get('breath_preserver', []))} Phasen → Gesangsnatürlichkeit")
    print(f"  4. safe_stft:       {len(by_feature.get('safe_stft', []))} Phasen → STFT-Abstürze verhindern")
    print(f"  5. gated_rms:       {len(by_feature.get('gated_rms', []))} Phasen → präzisere Metriken")
    print(f"  6. perceptual_blend:{len(by_feature.get('perceptual_blend', []))} Phasen → Bark-Band-Blending")


if __name__ == "__main__":
    main()
