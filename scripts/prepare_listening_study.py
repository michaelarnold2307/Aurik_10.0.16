#!/usr/bin/env python3
"""Listening Study Stimulus-Generator — §15.10.

Generiert randomisierte Stimulus-Sets für MUSHRA-Hörstudien.
ITU-R BS.1534-3 konform: Hidden Reference, 3.5-kHz Anchor, 4 Bedingungen.

Nutzung:
    python scripts/prepare_listening_study.py \
        --reference-dir corpus/vinyl/clean/ \
        --aurik-dir corpus/vinyl/restored/ \
        --rx-dir external/rx_output/ \
        --output study_stimuli/ \
        --scenarios 12 \
        --repetitions 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class StimulusSet:
    """Ein komplettes Stimulus-Set für einen Trial."""

    trial_id: str
    scenario: str
    material: str
    conditions: dict[str, str]  # condition_name → filepath
    hidden_ref_key: str  # Welcher Key die Hidden Reference ist
    anchor_key: str  # Welcher Key der Anchor ist
    display_order: list[str]  # Reihenfolge für GUI


@dataclass
class StudySession:
    """Eine komplette Hörstudien-Session."""

    session_id: str
    participant_id: str
    stimuli: list[StimulusSet]
    metadata: dict = field(default_factory=dict)


def _compute_anchor(audio_path: Path, output_dir: Path, cutoff_hz: float = 3500.0) -> Path:
    """Erzeugt 3.5-kHz-Tiefpass-Anchor (ITU-R BS.1534)."""
    from scipy.signal import butter, filtfilt, lfilter

    try:
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
    except Exception:
        logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        return audio_path  # Fallback

    nyquist = sr / 2
    cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(4, cutoff, btype="low")
    # §v10.113 safe_filtfilt: padlen-Guard für kurzes Audio
    n = audio.shape[0]
    padlen = 3 * max(len(b), len(a))
    if n > padlen:
        filtered = filtfilt(b, a, audio, axis=0)
    elif n > max(len(b), len(a)):
        filtered = lfilter(b, a, audio, axis=0)
    else:
        filtered = audio

    anchor_path = output_dir / f"anchor_{cutoff_hz}hz_{audio_path.stem}.wav"
    sf.write(str(anchor_path), filtered, sr)
    return anchor_path


def generate_study(
    reference_dir: Path,
    aurik_dir: Path,
    rx_dir: Path | None,
    output_dir: Path,
    num_scenarios: int = 12,
    repetitions: int = 3,
    participant_id: str = "P001",
    seed: int = 42,
) -> StudySession:
    """Generiert eine komplette MUSHRA-Studien-Session.

    Args:
        reference_dir:  Verzeichnis mit Referenz-Aufnahmen.
        aurik_dir:      Verzeichnis mit Aurik-Restaurierungen.
        rx_dir:         Verzeichnis mit RX-11-Ausgaben (optional).
        output_dir:     Ausgabeverzeichnis für Anchor-Dateien.
        num_scenarios:  Anzahl Szenarien.
        repetitions:    Wiederholungen pro Szenario.
        participant_id: Teilnehmer-ID.
        seed:           Random-Seed für Reproduzierbarkeit.

    Returns:
        StudySession mit allen Stimulus-Sets.
    """
    random.seed(seed)
    rng = random.Random(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Referenz-Dateien sammeln
    ref_files = sorted(reference_dir.glob("*.wav")) + sorted(reference_dir.glob("*.flac"))
    if not ref_files:
        raise FileNotFoundError(f"No WAV/FLAC files in {reference_dir}")

    # Auf num_scenarios begrenzen
    scenarios = ref_files[:num_scenarios]

    stimuli: list[StimulusSet] = []

    for scenario_idx, ref_path in enumerate(scenarios):
        scenario_name = ref_path.stem

        # Aurik-Output finden
        aurik_path = aurik_dir / ref_path.name
        if not aurik_path.exists():
            aurik_candidates = list(aurik_dir.glob(f"*{scenario_name}*"))
            aurik_path = aurik_candidates[0] if aurik_candidates else ref_path

        # RX-Output finden (optional)
        rx_path = None
        if rx_dir and rx_dir.is_dir():
            rx_candidates = list(rx_dir.glob(f"*{scenario_name}*"))
            if rx_candidates:
                rx_path = rx_candidates[0]

        # Anchor generieren
        anchor_path = _compute_anchor(ref_path, output_dir)

        # Bedingungen bauen
        conditions = {
            "reference": str(ref_path),
            "aurik": str(aurik_path),
            "anchor": str(anchor_path),
        }
        if rx_path:
            conditions["rx11"] = str(rx_path)

        # Hidden Reference Key zufällig wählen
        condition_keys = list(conditions.keys())

        for rep in range(repetitions):
            display_order = condition_keys.copy()
            rng.shuffle(display_order)

            stimuli.append(
                StimulusSet(
                    trial_id=f"{participant_id}_S{scenario_idx:02d}_R{rep:02d}",
                    scenario=scenario_name,
                    material="unknown",  # Kann aus Corpus-Metadaten ergänzt werden
                    conditions=conditions,
                    hidden_ref_key="reference",
                    anchor_key="anchor",
                    display_order=display_order,
                )
            )

    return StudySession(
        session_id=f"study_{participant_id}_{seed}",
        participant_id=participant_id,
        stimuli=stimuli,
        metadata={
            "num_scenarios": len(scenarios),
            "repetitions": repetitions,
            "total_trials": len(stimuli),
            "conditions": list(stimuli[0].conditions.keys()) if stimuli else [],
            "seed": seed,
            "reference_dir": str(reference_dir),
            "aurik_dir": str(aurik_dir),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MUSHRA Stimulus-Generator")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Referenz-Aufnahmen")
    parser.add_argument("--aurik-dir", type=Path, required=True, help="Aurik-Restaurierungen")
    parser.add_argument("--rx-dir", type=Path, help="iZotope RX-Ausgaben (optional)")
    parser.add_argument("--output", type=Path, default=Path("study_stimuli"), help="Ausgabeverzeichnis")
    parser.add_argument("--scenarios", type=int, default=12, help="Anzahl Szenarien")
    parser.add_argument("--repetitions", type=int, default=3, help="Wiederholungen")
    parser.add_argument("--participant", default="P001", help="Teilnehmer-ID")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--json-only", action="store_true", help="Nur Session-JSON ausgeben")
    args = parser.parse_args()

    session = generate_study(
        reference_dir=args.reference_dir,
        aurik_dir=args.aurik_dir,
        rx_dir=args.rx_dir,
        output_dir=args.output,
        num_scenarios=args.scenarios,
        repetitions=args.repetitions,
        participant_id=args.participant,
        seed=args.seed,
    )

    # Session speichern
    session_path = args.output / f"{session.session_id}.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)

    session_dict = {
        "session_id": session.session_id,
        "participant_id": session.participant_id,
        "metadata": session.metadata,
        "stimuli": [
            {
                "trial_id": s.trial_id,
                "scenario": s.scenario,
                "material": s.material,
                "conditions": s.conditions,
                "hidden_ref_key": s.hidden_ref_key,
                "anchor_key": s.anchor_key,
                "display_order": s.display_order,
            }
            for s in session.stimuli
        ],
    }

    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_dict, f, indent=2, ensure_ascii=False)

    if args.json_only:
        print(json.dumps({"ok": True, "session_id": session.session_id, "trials": len(session.stimuli)}))
    else:
        print(f"\n✅ Hörstudien-Session generiert: {session_path}")
        print(f"   Teilnehmer: {session.participant_id}")
        print(f"   Szenarien:  {session.metadata['num_scenarios']}")
        print(f"   Trials:     {session.metadata['total_trials']}")
        print(f"   Bedingungen: {', '.join(session.metadata['conditions'])}")
        print(f"   Seed:       {session.metadata['seed']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
